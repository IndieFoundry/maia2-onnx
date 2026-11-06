import { describe, expect, it } from 'vitest';
import { mulberry32, sampleMove } from '../src/index.js';

const pred = {
  moves: [
    { uci: 'e2e4', san: 'e4', p: 0.6 },
    { uci: 'd2d4', san: 'd4', p: 0.3 },
    { uci: 'g1f3', san: 'Nf3', p: 0.1 },
  ],
};

describe('sampleMove', () => {
  it('temperature 0 is argmax', () => {
    expect(sampleMove(pred, { temperature: 0, rng: () => 0.99 }).uci).toBe('e2e4');
  });
  it('uses the injected rng (inverse CDF)', () => {
    expect(sampleMove(pred, { rng: () => 0.0 }).uci).toBe('e2e4');
    expect(sampleMove(pred, { rng: () => 0.65 }).uci).toBe('d2d4');
    expect(sampleMove(pred, { rng: () => 0.95 }).uci).toBe('g1f3');
  });
  it('empirical frequencies follow the distribution at T=1', () => {
    const rng = mulberry32(42);
    const counts: Record<string, number> = {};
    const N = 20000;
    for (let i = 0; i < N; i++) {
      const m = sampleMove(pred, { rng }).uci;
      counts[m] = (counts[m] ?? 0) + 1;
    }
    expect(counts['e2e4']! / N).toBeCloseTo(0.6, 1);
    expect(counts['d2d4']! / N).toBeCloseTo(0.3, 1);
    expect(counts['g1f3']! / N).toBeCloseTo(0.1, 1);
  });
  it('low temperature sharpens, topK / minP truncate', () => {
    // T=0.5: weights 0.36, 0.09, 0.01 -> e4 = 0.36/0.46
    expect(sampleMove(pred, { temperature: 0.5, rng: () => 0.78 }).uci).toBe('e2e4');
    expect(sampleMove(pred, { topK: 1, rng: () => 0.99 }).uci).toBe('e2e4');
    expect(sampleMove(pred, { minP: 0.2, rng: () => 0.99 }).uci).toBe('d2d4');
  });
  it('is reproducible with a seed', () => {
    const a = mulberry32(7), b = mulberry32(7);
    for (let i = 0; i < 50; i++) expect(sampleMove(pred, { rng: a }).uci).toBe(sampleMove(pred, { rng: b }).uci);
  });
});
