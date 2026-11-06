import { describe, expect, it } from 'vitest';
import { cleanCastlingFen, createMaia, encodePosition, NUM_MOVES, sampleMove, type TensorConstructor } from '../src/index.js';

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

describe('sampleMove with a tiny positive temperature', () => {
  it('behaves like argmax instead of returning the least likely move', () => {
    const pred = { moves: [{ uci: 'a', san: 'a', p: 0.6 }, { uci: 'b', san: 'b', p: 0.3 }, { uci: 'c', san: 'c', p: 0.1 }] };
    for (const t of [1e-309, 1e-12]) {
      for (const r of [0, 0.5, 0.999]) expect(sampleMove(pred, { temperature: t, rng: () => r }).uci).toBe('a');
    }
  });
});

describe('malformed castling field', () => {
  it('is left untouched so chess.js rejects the FEN', () => {
    const bad = START.replace('KQkq', 'garbage');
    expect(cleanCastlingFen(bad)).toBe(bad);
    expect(() => encodePosition(bad)).toThrow();
  });
  it('still cleans valid but impossible rights', () => {
    expect(cleanCastlingFen('4k3/8/8/8/8/8/8/4K3 w KQkq - 0 1')).toBe('4k3/8/8/8/8/8/8/4K3 w - - 0 1');
  });
});

describe('createMaia output validation', () => {
  class FakeTensor {
    constructor(public type: string, public data: unknown, public dims: readonly number[]) {}
  }
  it('rejects a value output that does not hold one value per position', async () => {
    const session = {
      run: async () => ({
        logits_move: { data: new Float32Array(2 * NUM_MOVES) },
        logits_value: { data: new Float32Array(1) },
      }),
    };
    const maia = createMaia({ session: session as never, Tensor: FakeTensor as unknown as TensorConstructor });
    await expect(maia.predictBatch([{ fen: START, eloSelf: 1500, eloOppo: 1500 }, { fen: START, eloSelf: 1500, eloOppo: 1500 }]))
      .rejects.toThrow(/Unexpected model outputs/);
  });
});
