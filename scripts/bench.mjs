// Latency benchmark in onnxruntime-node. Run `npm run build` first.
//   node scripts/bench.mjs                 -> spawns one fresh process per model/thread setting
//   node scripts/bench.mjs <model> <threads>  (internal: one measurement)
import { spawnSync } from 'node:child_process';
import { readFileSync, statSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const MODELS = ['maia2-rapid.onnx', 'maia2-rapid.fp16.onnx', 'maia2-rapid.int8.onnx', 'maia2-rapid.int8-fp16.onnx'];

async function one(model, threads) {
  const t0 = performance.now();
  const ort = await import('onnxruntime-node');
  const { createMaia, encodePosition, decodeMoves } = await import('../dist/index.js');
  const t1 = performance.now();
  const opts = threads > 0 ? { intraOpNumThreads: threads, interOpNumThreads: 1 } : {};
  const session = await ort.InferenceSession.create(root + 'models/' + model, opts);
  const t2 = performance.now();
  const maia = createMaia({ session, Tensor: ort.Tensor });
  const golden = JSON.parse(readFileSync(root + 'fixtures/golden-rapid.json', 'utf8'));
  await maia.predict(golden[0].fen, 1500, 1500);
  const t3 = performance.now();
  const lat = [];
  for (let r = 0; r < 2; r++) {
    for (const g of golden) {
      const s = performance.now();
      await maia.predict(g.fen, g.eloSelf, g.eloOppo);
      lat.push(performance.now() - s);
    }
  }
  lat.sort((a, b) => a - b);
  const q = (x) => lat[Math.floor(x * (lat.length - 1))];
  const inputs = golden.slice(0, 32).map((g) => ({ fen: g.fen, eloSelf: g.eloSelf, eloOppo: g.eloOppo }));
  const bs = performance.now();
  for (let i = 0; i < 5; i++) await maia.predictBatch(inputs);
  const batchPerPos = (performance.now() - bs) / (5 * inputs.length);
  // JS-side overhead (encode + decode with chess.js), no model
  const zeros = new Float32Array(1880);
  const js = performance.now();
  for (const g of golden) { encodePosition(g.fen); decodeMoves(zeros, g.fen); }
  const jsPerPos = (performance.now() - js) / golden.length;
  return {
    model, bytes: statSync(root + 'models/' + model).size, threads: threads || 'default',
    importMs: +(t1 - t0).toFixed(0), sessionCreateMs: +(t2 - t1).toFixed(0), firstRunMs: +(t3 - t2).toFixed(0),
    coldTotalMs: +(t3 - t0).toFixed(0),
    p50Ms: +q(0.5).toFixed(2), p95Ms: +q(0.95).toFixed(2), maxMs: +lat[lat.length - 1].toFixed(2),
    batch32PerPosMs: +batchPerPos.toFixed(2), jsEncodeDecodePerPosMs: +jsPerPos.toFixed(3),
    rssMB: +(process.memoryUsage().rss / 2 ** 20).toFixed(0),
  };
}

if (process.argv[2]) {
  console.log(JSON.stringify(await one(process.argv[2], Number(process.argv[3] ?? 0))));
} else {
  const rows = [];
  for (const m of MODELS) for (const th of [0, 1, 2]) {
    const r = spawnSync(process.execPath, [fileURLToPath(import.meta.url), m, String(th)], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] });
    const row = JSON.parse(r.stdout.trim().split('\n').pop());
    rows.push(row);
    console.log(JSON.stringify(row));
  }
  console.table(rows.map(({ model, threads, bytes, coldTotalMs, sessionCreateMs, p50Ms, p95Ms, batch32PerPosMs, rssMB }) =>
    ({ model, threads, MB: +(bytes / 1e6).toFixed(1), coldTotalMs, sessionCreateMs, p50Ms, p95Ms, batch32PerPosMs, rssMB })));
}
