/**
 * Maia-2 input encoding, a line-by-line port of maia2.inference.preprocessing()
 * and maia2.utils.board_to_tensor / map_to_category (maia2 0.11.0, MIT).
 *
 * Tensor layout: float32 [18, 8, 8], flat index = channel * 64 + rank * 8 + file
 * (rank 0 = rank "1", file 0 = file "a"), always from the point of view of the
 * side to move: when Black is to move the board is mirrored first (python-chess
 * Board.mirror(): flip ranks, swap colours, give the move to White).
 *
 *   0..5   White (= side to move) P, N, B, R, Q, K
 *   6..11  Black (= opponent)     P, N, B, R, Q, K
 *   12     side-to-move plane (all 1s: after mirroring White is always to move)
 *   13..16 castling K, Q, k, q (after mirroring)
 *   17     en-passant target square (one-hot)
 */

export const NUM_CHANNELS = 18;
export const BOARD_TENSOR_SIZE = NUM_CHANNELS * 64;

const PIECE_CHANNEL: Record<string, number> = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };

/** Maia-2 Elo buckets: <1100 -> 0, [1100,1200) -> 1, ..., [1900,2000) -> 9, >=2000 -> 10. */
export function eloCategory(elo: number): number {
  if (!Number.isFinite(elo)) throw new RangeError(`Invalid Elo: ${elo}`);
  if (elo < 1100) return 0;
  if (elo >= 2000) return 10;
  return Math.floor((elo - 1100) / 100) + 1;
}

export interface EncodedPosition {
  /** float32 [18*8*8], see module doc for the layout. */
  planes: Float32Array;
  /** True when Black is to move: the board was mirrored and moves must be mirrored back. */
  mirrored: boolean;
}

function squareIndex(name: string): number {
  const file = name.charCodeAt(0) - 97;
  const rank = name.charCodeAt(1) - 49;
  if (name.length !== 2 || file < 0 || file > 7 || rank < 0 || rank > 7) {
    throw new Error(`Invalid square: ${name}`);
  }
  return rank * 8 + file;
}

/** Mirror a UCI move vertically (a2a4 <-> a7a5), keeping any promotion suffix. */
export function mirrorUci(uci: string): string {
  const m = (sq: string) => sq[0] + String(9 - Number(sq[1]));
  return m(uci.slice(0, 2)) + m(uci.slice(2, 4)) + uci.slice(4);
}

/**
 * Drop castling flags whose king/rook are not on their original squares, as
 * python-chess does (Board.clean_castling_rights). chess.js 1.x keeps such flags
 * from a FEN and would then generate an illegal castling move.
 */
export function cleanCastlingFen(fen: string): string {
  const fields = fen.trim().split(/\s+/);
  const placement = fields[0] ?? '';
  const castling = fields[2];
  if (!castling || castling === '-') return fields.join(' ');
  // Leave malformed rights untouched so chess.js rejects the FEN (as python-chess would).
  if (!/^[KQkq]+$/.test(castling)) return fields.join(' ');
  const rows = placement.split('/');
  const at = (rank: number, file: number): string | null => {
    const row = rows[7 - rank];
    if (!row) return null;
    let f = 0;
    for (const ch of row) {
      if (ch >= '1' && ch <= '8') f += Number(ch);
      else {
        if (f === file) return ch;
        f += 1;
      }
      if (f > file) return null;
    }
    return null;
  };
  const keep = [...castling].filter((flag) => {
    const white = flag === flag.toUpperCase();
    const rank = white ? 0 : 7;
    const rookFile = flag.toLowerCase() === 'k' ? 7 : flag.toLowerCase() === 'q' ? 0 : -1;
    return rookFile >= 0 && at(rank, 4) === (white ? 'K' : 'k') && at(rank, rookFile) === (white ? 'R' : 'r');
  });
  fields[2] = keep.length ? keep.join('') : '-';
  return fields.join(' ');
}

export function encodePosition(fen: string): EncodedPosition {
  const fields = fen.trim().split(/\s+/);
  const [placement, turn, castling = '-', ep = '-'] = fields;
  if (!placement || (turn !== 'w' && turn !== 'b')) throw new Error(`Invalid fen: ${fen}`);
  if (castling !== '-' && !/^[KQkq]+$/.test(castling)) throw new Error(`Invalid fen castling: ${fen}`);
  const mirrored = turn === 'b';

  // board[sq] = piece letter (uppercase = White) in the ORIGINAL orientation.
  const board: (string | null)[] = new Array(64).fill(null);
  const rows = placement.split('/');
  if (rows.length !== 8) throw new Error(`Invalid fen placement: ${fen}`);
  rows.forEach((row, i) => {
    const rank = 7 - i;
    let file = 0;
    for (const ch of row) {
      if (ch >= '1' && ch <= '8') file += Number(ch);
      else {
        if (!(ch.toLowerCase() in PIECE_CHANNEL) || file > 7) throw new Error(`Invalid fen placement: ${fen}`);
        board[rank * 8 + file] = ch;
        file += 1;
      }
    }
    if (file !== 8) throw new Error(`Invalid fen placement: ${fen}`);
  });

  // python-chess keeps a castling right only if king and rook stand on their
  // original squares (Board.clean_castling_rights, standard chess).
  const right = (flag: string): boolean => {
    if (!castling.includes(flag)) return false;
    const white = flag === flag.toUpperCase();
    const king = white ? 'K' : 'k';
    const rook = white ? 'R' : 'r';
    const back = white ? 0 : 56;
    const rookSq = back + (flag.toLowerCase() === 'k' ? 7 : 0);
    return board[back + 4] === king && board[rookSq] === rook;
  };
  // Order in the tensor: K, Q, k, q of the (possibly mirrored) board.
  const rights = mirrored
    ? [right('k'), right('q'), right('K'), right('Q')]
    : [right('K'), right('Q'), right('k'), right('q')];

  const planes = new Float32Array(BOARD_TENSOR_SIZE);
  for (let sq = 0; sq < 64; sq++) {
    const piece = board[sq];
    if (!piece) continue;
    const isWhite = piece === piece.toUpperCase();
    const own = mirrored ? !isWhite : isWhite; // "White" after mirroring
    const target = mirrored ? sq ^ 56 : sq;
    const channel = PIECE_CHANNEL[piece.toLowerCase()]! + (own ? 0 : 6);
    planes[channel * 64 + target] = 1;
  }
  planes.fill(1, 12 * 64, 13 * 64); // White (after mirroring) to move
  rights.forEach((has, i) => {
    if (has) planes.fill(1, (13 + i) * 64, (14 + i) * 64);
  });
  if (ep !== '-') {
    const sq = squareIndex(ep);
    planes[17 * 64 + (mirrored ? sq ^ 56 : sq)] = 1;
  }
  return { planes, mirrored };
}
