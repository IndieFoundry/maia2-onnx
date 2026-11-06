// Can onnxruntime-web (wasm backend) load and run the models? Node-hosted check.
//   npm run build && node scripts/web-check.mjs [threads]
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-web';
import { createMaia } from '../dist/index.js';

const root = fileURLToPath(new URL('..', import.meta.url));
ort.env.wasm.numThreads = Number(process.argv[2] ?? 1);
const golden = JSON.parse(readFileSync(root + 'fixtures/golden-rapid.json', 'utf8'));
const MODELS = ['maia2-rapid.onnx', 'maia2-rapid.fp16.onnx', 'maia2-rapid.int8.onnx', 'maia2-rapid.int8-fp16.onnx', 'maia2-rapid.w8.onnx'];

for (const model of MODELS) {
  try {
    const buf = readFileSync(root + 'models/' + model); // browsers: fetch() -> ArrayBuffer
    const t0 = performance.now();
    const session = await ort.InferenceSession.create(new Uint8Array(buf), { executionProviders: ['wasm'] });
    const t1 = performance.now();
    const maia = createMaia({ session, Tensor: ort.Tensor });
    await maia.predict(golden[0].fen, 1500, 1500);
    const t2 = performance.now();
    const lat = [];
    let top1 = 0, maxD = 0;
    for (const g of golden) {
      const s = performance.now();
      const p = await maia.predict(g.fen, g.eloSelf, g.eloOppo);
      lat.push(performance.now() - s);
      if (p.moves[0].uci === g.top10[0].uci) top1++;
      const by = new Map(p.moves.map((m) => [m.uci, m.p]));
      for (const m of g.top10) maxD = Math.max(maxD, Math.abs(by.get(m.uci) - m.p));
    }
    lat.sort((a, b) => a - b);
    console.log(JSON.stringify({ model, wasmThreads: ort.env.wasm.numThreads, ok: true,
      sessionCreateMs: Math.round(t1 - t0), firstRunMs: Math.round(t2 - t1),
      p50Ms: +lat[lat.length >> 1].toFixed(1), p95Ms: +lat[Math.floor(lat.length * 0.95)].toFixed(1),
      top1Agreement: +(top1 / golden.length).toFixed(4), maxAbsDeltaTop10: +maxD.toExponential(2) }));
    await session.release();
  } catch (e) {
    console.log(JSON.stringify({ model, ok: false, error: String(e).slice(0, 300) }));
  }
}
