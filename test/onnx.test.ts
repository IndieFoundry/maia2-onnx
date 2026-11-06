import { writeFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';
import { describe, expect, it } from 'vitest';
import { createMaia, type Prediction } from '../src/index.js';
import { exists, readJson, root, type Golden } from './helpers.js';

/**
 * End-to-end parity: TS encoding -> onnxruntime-node -> TS decoding, against the
 * PyTorch golden fixtures produced by maia2's own inference helpers.
 */
const KINDS = ['rapid', 'blitz'] as const;
const VARIANTS = [
  { suffix: '', label: 'fp32' },
  { suffix: '.fp16', label: 'fp16' },
  { suffix: '.int8', label: 'int8' },
  { suffix: '.int8-fp16', label: 'int8-fp16' },
] as const;

interface Stats {
  model: string;
  n: number;
  top1Agreement: number;
  top3SetAgreement: number;
  maxAbsDeltaTop10: number;
  meanAbsDeltaTop10: number;
  maxAbsDeltaWinProb: number;
  top1DisagreementsAtGapAbove5pct: number;
  byCategory: Record<string, { n: number; top1Agreement: number; maxAbsDeltaTop10: number }>;
}

function compare(golden: Golden[], preds: Prediction[], model: string): Stats {
  let top1 = 0, top3 = 0, maxD = 0, sumD = 0, cntD = 0, maxW = 0, bigMiss = 0;
  const cat: Record<string, { n: number; top1: number; maxD: number }> = {};
  golden.forEach((g, i) => {
    const p = preds[i]!;
    const byUci = new Map(p.moves.map((m) => [m.uci, m.p]));
    const c = (cat[g.category] ??= { n: 0, top1: 0, maxD: 0 });
    c.n++;
    const ok = p.moves[0]!.uci === g.top10[0]!.uci;
    if (ok) { top1++; c.top1++; }
    else {
      // gap between the golden winner and the golden probability of the move WE picked
      const goldenP = new Map(g.top10.map((m) => [m.uci, m.p]));
      if (g.top10[0]!.p - (goldenP.get(p.moves[0]!.uci) ?? 0) > 0.05) bigMiss++;
    }
    const g3 = new Set(g.top10.slice(0, 3).map((m) => m.uci));
    if (p.moves.slice(0, 3).every((m) => g3.has(m.uci))) top3++;
    for (const m of g.top10) {
      const d = Math.abs((byUci.get(m.uci) ?? 0) - m.p);
      maxD = Math.max(maxD, d); c.maxD = Math.max(c.maxD, d); sumD += d; cntD++;
    }
    maxW = Math.max(maxW, Math.abs(p.winProb - g.maia2.winProb));
  });
  const n = golden.length;
  return {
    model, n,
    top1Agreement: top1 / n,
    top3SetAgreement: top3 / n,
    maxAbsDeltaTop10: maxD,
    meanAbsDeltaTop10: sumD / cntD,
    maxAbsDeltaWinProb: maxW,
    top1DisagreementsAtGapAbove5pct: bigMiss,
    byCategory: Object.fromEntries(
      Object.entries(cat).map(([k, v]) => [k, { n: v.n, top1Agreement: v.top1 / v.n, maxAbsDeltaTop10: v.maxD }]),
    ),
  };
}

const report: (Stats & { maxBatchVsSingleDelta: number })[] = [];

for (const kind of KINDS) {
  const goldenPath = `fixtures/golden-${kind}.json`;
  for (const v of VARIANTS) {
    const modelPath = `models/maia2-${kind}${v.suffix}.onnx`;
    const ready = exists(goldenPath) && exists(modelPath);
    describe.skipIf(!ready)(`${kind} ${v.label} vs PyTorch golden`, () => {
      it('matches', async () => {
        const golden = readJson<Golden[]>(goldenPath);
        const session = await ort.InferenceSession.create(root + modelPath);
        const maia = createMaia({ session, Tensor: ort.Tensor });
        // Stats are computed on single-position runs (the realistic use). Dynamic int8
        // quantization computes activation scales over the whole batch tensor, so an
        // int8 prediction depends slightly on what else is in the batch.
        const preds: Prediction[] = [];
        for (const g of golden) preds.push(await maia.predict(g.fen, g.eloSelf, g.eloOppo));
        const batched = await maia.predictBatch(golden.map((g) => ({ fen: g.fen, eloSelf: g.eloSelf, eloOppo: g.eloOppo })));
        let batchDelta = 0;
        batched.forEach((b, i) => {
          const single = new Map(preds[i]!.moves.map((m) => [m.uci, m.p]));
          for (const m of b.moves) batchDelta = Math.max(batchDelta, Math.abs(m.p - single.get(m.uci)!));
        });
        console.log(`${kind}-${v.label}: max |batched - single| prob delta = ${batchDelta}`);
        expect(batchDelta).toBeLessThan(v.label.startsWith('int8') ? 0.1 : 1e-5);
        preds.forEach((p, i) => {
          expect(p.moves.length).toBe(golden[i]!.numLegal);
          expect(Math.abs(p.moves.reduce((a, m) => a + m.p, 0) - 1)).toBeLessThan(1e-5);
        });
        const s = { ...compare(golden, preds, `${kind}-${v.label}`), maxBatchVsSingleDelta: batchDelta };
        report.push(s);
        console.log(JSON.stringify(s));
        if (v.label === 'fp32') {
          expect(s.maxAbsDeltaTop10).toBeLessThan(1e-4);
          expect(s.maxAbsDeltaWinProb).toBeLessThanOrEqual(1e-4 + 5e-5); // maia2 rounds winProb to 4 dp
          // exact ranks when probabilities are not tied
          golden.forEach((g, i) => {
            g.top10.forEach((m, r) => {
              const next = g.top10[r + 1];
              if (!next || m.p - next.p > 1e-4) expect(preds[i]!.moves[r]!.uci).toBe(m.uci);
            });
          });
        } else if (v.label === 'fp16') {
          expect(s.top1Agreement).toBeGreaterThanOrEqual(0.99);
          expect(s.top1DisagreementsAtGapAbove5pct).toBe(0);
          expect(s.maxAbsDeltaTop10).toBeLessThan(5e-3);
        } else {
          // int8 acceptance thresholds (see README "int8 verdict")
          expect(s.top1Agreement).toBeGreaterThanOrEqual(0.965);
          expect(s.top1DisagreementsAtGapAbove5pct).toBe(0);
          expect(s.maxAbsDeltaTop10).toBeLessThan(0.05);
        }
        await (session as unknown as { release?: () => Promise<void> }).release?.();
      });
    });
  }
}

describe('report', () => {
  it('writes fixtures/parity-report.json', () => {
    // Only overwrite the committed report when every model ran (a partial run would drop rows).
    const expected = KINDS.length * VARIANTS.length;
    if (report.length === expected) writeFileSync(root + 'fixtures/parity-report.json', JSON.stringify(report, null, 2) + '\n');
    else console.log(`parity-report.json not rewritten: ${report.length}/${expected} models ran`);
  });
});
