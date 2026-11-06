import type { MoveProb } from './decode.js';

export interface SampleOptions {
  /**
   * 1 = sample from Maia's distribution as is (most human-like).
   * <1 sharpens towards the top move, >1 flattens. 0 = argmax (deterministic).
   */
  temperature?: number;
  /** Only sample among the k most likely moves (default: all). */
  topK?: number;
  /** Drop moves whose probability is below this threshold (default 0). Top move is always kept. */
  minP?: number;
  /** Uniform [0,1) generator; inject a seeded one for reproducible games/tests. */
  rng?: () => number;
}

/** Pick a move "like a human": sample from the predicted distribution rather than taking the argmax. */
export function sampleMove(pred: { moves: readonly MoveProb[] }, opts: SampleOptions = {}): MoveProb {
  const { temperature = 1, topK, minP = 0, rng = Math.random } = opts;
  if (pred.moves.length === 0) throw new Error('No moves to sample from');
  if (!(temperature >= 0)) throw new RangeError(`Invalid temperature: ${temperature}`);
  const sorted = [...pred.moves].sort((a, b) => b.p - a.p);
  if (temperature === 0) return sorted[0]!;

  let pool = sorted.filter((m, i) => i === 0 || m.p >= minP);
  if (topK !== undefined) pool = pool.slice(0, Math.max(1, topK));
  // p^(1/T), computed in log space; subtract the max BEFORE dividing by T so a
  // tiny positive T gives 0 / -Infinity (argmax-like) instead of NaN.
  const logs = pool.map((m) => Math.log(Math.max(m.p, 1e-30)));
  const max = Math.max(...logs);
  const w = logs.map((l) => Math.exp((l - max) / temperature));
  const total = w.reduce((a, b) => a + b, 0);
  let r = rng() * total;
  for (let i = 0; i < pool.length; i++) {
    r -= w[i]!;
    if (r < 0) return pool[i]!;
  }
  return pool[pool.length - 1]!;
}

/** Small seeded PRNG (mulberry32) for reproducible sampling. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
