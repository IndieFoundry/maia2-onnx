import { Chess } from 'chess.js';
import { cleanCastlingFen, mirrorUci } from './encode.js';
import { ALL_MOVES, NUM_MOVES } from './vocab.js';

export interface MoveProb {
  /** Move in UCI, in the real orientation of the position (e.g. e7e5 for Black). */
  uci: string;
  san: string;
  /** Probability, renormalised over the legal moves (sums to 1). */
  p: number;
}

/** UCI (side-to-move-is-White frame) -> index in the model's move logits. */
export const MOVE_INDEX: ReadonlyMap<string, number> = new Map(ALL_MOVES.map((m, i) => [m, i]));

export interface LegalMove {
  uci: string;
  san: string;
  /** Index into the model's logits. */
  index: number;
}

/** Legal moves of `fen` with their logit index (chess.js validates the FEN and throws if invalid). */
export function legalMoves(fen: string): LegalMove[] {
  const mirrored = fen.trim().split(/\s+/)[1] === 'b';
  const chess = new Chess(cleanCastlingFen(fen));
  return chess.moves({ verbose: true }).map((m) => {
    const uci = m.from + m.to + (m.promotion ?? '');
    const key = mirrored ? mirrorUci(uci) : uci;
    const index = MOVE_INDEX.get(key);
    if (index === undefined) throw new Error(`Move ${uci} (${key}) is not in the Maia-2 vocabulary`);
    return { uci, san: m.san, index };
  });
}

/**
 * Turn one row of move logits (length 1880) into a probability distribution over
 * the legal moves of `fen`, sorted by decreasing probability. Same maths as
 * maia2's masked softmax: softmax restricted to the legal moves.
 */
export function decodeMoves(logits: ArrayLike<number>, fen: string): MoveProb[] {
  if (logits.length !== NUM_MOVES) {
    throw new Error(`Expected ${NUM_MOVES} logits, got ${logits.length}`);
  }
  const legal = legalMoves(fen);
  if (legal.length === 0) throw new Error(`No legal moves in position: ${fen}`);
  let max = -Infinity;
  for (const m of legal) max = Math.max(max, logits[m.index]!);
  const exps = legal.map((m) => Math.exp(logits[m.index]! - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return legal
    .map((m, i) => ({ uci: m.uci, san: m.san, p: exps[i]! / sum }))
    .sort((a, b) => b.p - a.p);
}

/**
 * maia2's win probability: the value head predicts, for the side to move,
 * clamp(v / 2 + 0.5, 0, 1). maia2's inference helpers then flip it when Black is
 * to move, so `winProb` is from WHITE's point of view (parity with maia2).
 */
export function decodeValue(valueLogit: number, fen: string): { winProb: number; sideToMoveWinProb: number } {
  const stm = Math.min(1, Math.max(0, valueLogit / 2 + 0.5));
  const black = fen.trim().split(/\s+/)[1] === 'b';
  return { winProb: black ? 1 - stm : stm, sideToMoveWinProb: stm };
}
