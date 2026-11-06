# maia2-onnx

Run [Maia-2](https://github.com/CSSLab/maia2), the human-like chess model from
CSSLab (University of Toronto, NeurIPS 2024), in JavaScript and TypeScript.

This repository exports Maia-2 to ONNX and ships a small TypeScript module
that encodes positions exactly like maia2 does, runs the model through any
onnxruntime flavour (Node, browser/WebAssembly, React Native) and decodes a
human-like move distribution for a given rating. No Python at runtime.

```ts
const pred = await maia.predict(fen, 1500, 1500);       // side to move rated 1500, opponent 1500
// pred.moves -> [{ uci, san, p }, ...]  every legal move with the probability a 1500 plays it
const move = sampleMove(pred, { minP: 0.01 });           // play like a human
```

Parity with the PyTorch reference is checked on 360 predictions per model:
100 % top-1 agreement for fp32, 99.7-100 % for fp16 (details below).

**In production:** it powers [Play against Maia](https://chessatlas.net/features/play-against-maia)
in [ChessAtlas](https://chessatlas.net), where a human-like opponent plays your
opening repertoire and shows the moment your opponent leaves your prep.

## Quick start

```bash
npm install @indiefoundry/maia2-onnx chess.js onnxruntime-node
```

The npm package holds the TypeScript module only. The ONNX files (27-85 MB
each) are generated from the official maia2 weights by the export script in
this repository:

```bash
git clone https://github.com/IndieFoundry/maia2-onnx && cd maia2-onnx
python3.12 -m venv .venv && .venv/bin/pip install maia2 onnx onnxruntime onnxscript
.venv/bin/python python/export_onnx.py rapid     # writes models/maia2-rapid.*.onnx
(cd models && shasum -a 256 -c SHA256SUMS --ignore-missing)
```

Then copy one file (e.g. `models/maia2-rapid.fp16.onnx`) and `NOTICE` into
your project.

```ts
import * as ort from 'onnxruntime-node';
import { createMaia, sampleMove } from '@indiefoundry/maia2-onnx';

const session = await ort.InferenceSession.create('models/maia2-rapid.fp16.onnx');
const maia = createMaia({ session, Tensor: ort.Tensor });
// Ruy Lopez, Chigorin, move 13 for White
const pred = await maia.predict('r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/5N1P/PPBN1PP1/R1BQR1K1 w - - 0 13', 1500, 1500);
console.log(pred.moves.slice(0, 3)); // Nf1 ~0.66, then b4/b3 and d5 around 0.06-0.07
console.log(sampleMove(pred, { minP: 0.01 }));
```

Maia-2 is one skill-conditioned network: input = position + Elo of the side to
move + Elo of the opponent; output = logits over a 1880-move vocabulary + a
value head. Models: `rapid` (maia2's default) and `blitz`, same architecture
(23.3 M parameters in the checkpoint, 21.2 M after dropping the auxiliary
side-info head that inference never uses).

## Layout

```
python/export_onnx.py     checkpoint -> models/maia2-<type>.{onnx,int8.onnx,fp16.onnx} + src/vocab.ts
python/make_fixtures.py   golden fixtures from the PyTorch model via maia2's own helpers
python/quant_variants.py  experimental variants (per-tensor int8, weight-only int8, int8+fp16) + eval
src/                      the TS module (ESM, strict, no framework)
test/                     vitest: encoding parity, ONNX end-to-end parity, sampling
scripts/bench.mjs         onnxruntime-node latency / cold load
scripts/web-check.mjs     onnxruntime-web (wasm) load + parity check, Node-hosted
fixtures/                 positions, encoding tensors, golden predictions, parity reports
models/                   ONNX files (gitignored, regenerate) + SHA256SUMS (committed)
```

## Regenerate everything

```bash
python3.12 -m venv .venv                       # maia2 0.11 requires Python >=3.10,<3.13
.venv/bin/pip install maia2 onnx onnxruntime onnxscript
.venv/bin/python python/export_onnx.py rapid blitz   # downloads weights to maia2_models/ (~280 MB each)
.venv/bin/python python/make_fixtures.py rapid
.venv/bin/python python/make_fixtures.py blitz
.venv/bin/python python/quant_variants.py rapid       # int8-fp16 + experimental variants, fixtures/variants-rapid.json
.venv/bin/python python/quant_variants.py blitz
npm install
npm test                   # 161 tests, ~2-3 min (runs 360 predictions x 8 model files)
npm run bench              # node latency
node scripts/web-check.mjs # wasm check (needs npm run build first)
```

Versions used: maia2 0.11.0, torch 2.8.0, onnx 1.23.1, onnxruntime 1.30.0
(python and node/web), chess.js 1.4.0. Export: legacy TorchScript exporter
(`dynamo=False`), opset 17, dynamic batch axis. `models/SHA256SUMS` lets you
check that a regeneration produced the same files.

## Model files

| file | bytes | gzip -9 | what it is |
|---|---:|---:|---|
| `maia2-<type>.onnx` | 84,958,999 | 78,978,652 | fp32 |
| `maia2-<type>.fp16.onnx` | 42,524,714 | 39,392,119 | fp16 **storage**: weights stored as fp16, `Cast` to fp32 folded at session creation; compute is fp32 |
| `maia2-<type>.int8.onnx` | 39,401,883 | 36,229,537 | dynamic int8 (MatMul/Gemm, **per-channel** weights); convs stay fp32 |
| `maia2-<type>.int8-fp16.onnx` | 27,391,018 | 25,033,925 | int8 above + fp16 storage of the conv weights |
| `maia2-<type>.int8pt.onnx` | 39,314,974 | 32,532,932 | experimental: dynamic int8, per-tensor (onnxruntime default) — **rejected** |
| `maia2-<type>.w8.onnx` | 47,089,027 | 44,128,088 | experimental: weight-only int8 (`com.microsoft.MatMulNBits`, block 32) |

Sizes are identical for rapid and blitz (same graph); the gzip column was
measured on rapid. Brotli -q11: fp16 37.5 MB, int8 35.9 MB — the weights barely
compress.

Graph I/O:

| name | type | shape |
|---|---|---|
| `boards` (in) | float32 | [batch, 18, 8, 8] |
| `elo_self` (in) | int64 | [batch] — Elo **category** 0..10 |
| `elo_oppo` (in) | int64 | [batch] |
| `logits_move` (out) | float32 | [batch, 1880] — raw logits, mask + softmax on the caller side |
| `logits_value` (out) | float32 | [batch] — raw value output |

## Encoding facts (read from maia2 0.11.0 source, verified by tests)

- **Mirroring.** If Black is to move, maia2 calls python-chess
  `Board.mirror()` (flip ranks, swap colours, give the move to White) before
  encoding. The network therefore always sees "White to move"; predicted moves
  are in the mirrored frame and are mirrored back (`e7e5` <-> `e2e4`, promotion
  suffix kept).
- **Planes** `[18, 8, 8]`, flat index `channel*64 + rank*8 + file` (a1 = 0):
  0-5 = side-to-move P N B R Q K, 6-11 = opponent P N B R Q K, 12 = turn plane
  (all ones: always White after mirroring), 13-16 = castling K Q k q (after
  mirroring), 17 = en-passant target square one-hot.
- **Castling** uses python-chess's cleaned rights: a flag counts only if king
  and rook are on their original squares. chess.js 1.4 does not clean them on
  load and would then generate illegal castling moves; `cleanCastlingFen()`
  fixes that before chess.js sees the FEN.
- **En passant** is taken verbatim from the FEN, like maia2's inference. The
  training data, however, came from python-chess `board.fen()`, which only
  writes the ep square when an ep capture is legal. chess.js 1.4 `fen()` does
  the same, so pass FENs produced by chess.js (not a raw Lichess FEN with a
  pseudo-ep square) to stay in the training distribution.
- **Elo buckets** (`map_to_category`): `<1100` -> 0, `[1100,1200)` -> 1, …,
  `[1900,2000)` -> 9, `>=2000` -> 10. 11 categories, so the model cannot
  distinguish 700 from 1000 or 2000 from 2600. Non-integer Elo is fine; NaN
  throws.
- **Move vocabulary** (1880 entries, `src/vocab.ts`, generated from
  `get_all_possible_moves()`): all queen-like and knight moves from every
  square on an empty board, in python-chess generation order, then the 88 White
  promotions from rank 7 (q, r, b, n; straight and both captures). Castling is
  `e1g1` / `e1c1`. There are no Black promotions in the vocabulary: Black
  positions are mirrored.
- **Decoding** = maia2's `_masked_softmax`: softmax restricted to the legal
  moves (the other 1880 - n logits are ignored), so probabilities sum to 1 over
  legal moves.
- **Value.** `v_stm = clamp(logits_value / 2 + 0.5, 0, 1)` for the side to
  move; maia2's `inference_each` returns `1 - that` when Black is to move, so
  its `win_prob` is from **White's** point of view. Despite the name it is an
  **expected score** (about P(win) + 0.5 x P(draw)), not a pure win
  probability: 0.5 means a balanced position, not a 50 % chance of winning.
  We keep maia2's naming and return both (`winProb` = White,
  `sideToMoveWinProb`).
- **Openings are out of distribution.** Training skips the first 10 plies of
  every game (`first_n_moves: 10`, `main.py:95`). From the start position at
  1500 the model gives Nf3 33 %, Nc3 24 %, Na3 8 % and 1.e4 barely shows up.
  Use an opening book for the first ~5 moves.

## TypeScript API (`src/index.ts`)

```ts
encodePosition(fen): { planes: Float32Array /* 1152 */, mirrored: boolean }
eloCategory(elo): number                       // 0..10
cleanCastlingFen(fen): string
legalMoves(fen): { uci, san, index }[]
decodeMoves(logits /* 1880 */, fen): { uci, san, p }[]   // legal only, sorted, sum = 1
decodeValue(valueLogit, fen): { winProb, sideToMoveWinProb }
createMaia({ session, Tensor }): {
  predict(fen, eloSelf, eloOppo): Promise<{ moves: {uci,san,p}[], winProb, sideToMoveWinProb }>
  predictBatch([{ fen, eloSelf, eloOppo }]): Promise<Prediction[]>   // one session.run
}
sampleMove(pred, { temperature = 1, topK?, minP = 0, rng = Math.random }): { uci, san, p }
mulberry32(seed): () => number                 // seeded rng for reproducible sampling
ALL_MOVES, NUM_MOVES (1880), NUM_ELO_CATEGORIES (11), MOVE_INDEX
```

`createMaia` takes the `Tensor` class alongside the session because
onnxruntime-common checks `instanceof Tensor`, and node/web each ship their
own copy; pass `ort.Tensor` from the same package as the session. The module
imports only `chess.js`.

"Play like a human": `sampleMove(pred)` samples from Maia's distribution
(temperature 1 = as predicted). `temperature: 0` is argmax (strongest, least
human). Keep `minP` around 0.01-0.02 to avoid the rare absurd blunder at
very low probability without flattening the human error profile.

## Parity results

Golden set: 60 positions (20 openings, 20 middlegames, 20 endgames incl.
promotion for both colours, KR/KQ/KBN vs K, Lucena, Vancura, opposition) x 6
Elo pairs (1100/1100, 1500/1500, 1900/1900, 1100/1900, 1900/1100, 800/2400) =
360 predictions per model, produced by `maia2.inference.inference_each` plus
the same computation kept unrounded.

**Encoding**: the TS tensor equals `maia2.inference.preprocessing()` exactly
(set of non-zero indices, values all 1.0) on all 64 positions (60 + 4
castling/ep edge cases); the legal-move index set matches maia2's mask on all
64; `eloCategory` matches `map_to_category` on 15 probes.

**End to end** (TS encode -> onnxruntime-node -> TS decode vs PyTorch), single
position runs, 360 predictions each (`fixtures/parity-report.json`):

| model | top-1 agree | top-3 set agree | max \|Δp\| top-10 | mean \|Δp\| | max \|Δ winProb\| | top-1 misses with gap > 5 pp |
|---|---:|---:|---:|---:|---:|---:|
| rapid fp32 | 100 % | 100 % | 1.9e-6 | 1.4e-7 | 5.0e-5 (maia2 rounds to 4 dp) | 0 |
| rapid fp16 | 100 % | 100 % | 8.4e-4 | 6.7e-5 | 8.7e-4 | 0 |
| rapid int8 | 98.6 % | 95.8 % | 0.027 | 1.8e-3 | 0.017 | 0 |
| rapid int8-fp16 | 99.2 % | 96.4 % | 0.029 | 1.8e-3 | 0.016 | 0 |
| blitz fp32 | 100 % | 100 % | 2.4e-6 | 1.4e-7 | 5.0e-5 | 0 |
| blitz fp16 | 99.7 % | 100 % | 9.5e-4 | 7.7e-5 | 5.2e-4 | 0 |
| blitz int8 | 98.3 % | 97.8 % | 0.030 | 1.8e-3 | 0.014 | 0 |
| blitz int8-fp16 | 97.2 % | 98.3 % | 0.032 | 1.9e-3 | 0.015 | 0 |

Experimental (python onnxruntime, rapid, `fixtures/variants-rapid.json`):
per-tensor int8 91.4 % top-1, max Δp 0.127; weight-only int8 (MatMulNBits)
99.2 %, max Δp 0.0057 but 47 MB and slower in wasm.

fp32 ranks match PyTorch exactly wherever the golden probabilities are not tied
within 1e-4. Every top-1 disagreement of fp16/int8 is between two moves whose
golden probabilities are within 5 percentage points of each other.

**Batching and int8**: dynamic quantization computes activation scales over the
whole input tensor, so an int8 prediction changes slightly with what else is in
the batch (max Δp 0.04-0.05 batched vs single). fp32/fp16 are batch-invariant
(1.5e-6).

### int8 verdict

- The onnxruntime default (per-tensor) int8 is **not acceptable**: 91 % top-1,
  up to 12.7 pp probability error.
- Per-channel int8 (`*.int8.onnx`) is **acceptable** for "play like a human"
  sampling and move hints (98-99 % top-1, misses only on near-ties, ≤ 3 pp
  error) but **not needed**: on the server it is no faster than fp32 on Apple
  silicon and only saves RAM (~160 MB vs ~220-260 MB RSS). Its only real
  benefit is download size for client-side inference, where `int8-fp16`
  (27.4 MB) is the smallest usable file.
- `fp16` storage gives half the file size of fp32 with no measurable accuracy
  loss: it is the default recommendation wherever file size matters.

## Performance

Apple M1 Max (10 cores), Node 24.2, onnxruntime-node 1.30.0, rapid, fresh
process per row (`npm run bench`). "cold" = import ort + module, create
session, first inference. p50/p95 over 720 single-position `predict()` calls
(encode + run + decode). JS encode+decode alone (chess.js move generation) is
~0.05 ms/position.

| model | threads | cold total | session create | p50 | p95 | batch 32, per position | RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| fp32 | default | 262 ms | 180 ms | 10.4 ms | 24.8 ms | 14.2 ms | 228 MB |
| fp32 | 1 | 208 ms | 150 ms | 16.4 ms | 21.2 ms | 15.2 ms | 219 MB |
| fp32 | 2 | 121 ms | 80 ms | 12.9 ms | 35.7 ms | 17.6 ms | 211 MB |
| fp16 | default | 305 ms | 184 ms | 9.7 ms | 36.6 ms | 7.5 ms | 262 MB |
| fp16 | 1 | 142 ms | 91 ms | 14.2 ms | 16.9 ms | 21.8 ms | 212 MB |
| fp16 | 2 | 206 ms | 143 ms | 12.1 ms | 26.0 ms | 9.0 ms | 262 MB |
| int8 | default | 92 ms | 52 ms | 10.7 ms | 26.3 ms | 9.9 ms | 162 MB |
| int8 | 1 | 106 ms | 56 ms | 14.4 ms | 18.1 ms | 12.4 ms | 161 MB |
| int8 | 2 | 93 ms | 52 ms | 9.3 ms | 13.3 ms | 10.6 ms | 163 MB |
| int8-fp16 | default | 105 ms | 64 ms | 12.1 ms | 37.5 ms | 14.9 ms | 176 MB |
| int8-fp16 | 1 | 108 ms | 60 ms | 12.8 ms | 15.9 ms | 11.1 ms | 166 MB |
| int8-fp16 | 2 | 113 ms | 69 ms | 8.3 ms | 11.0 ms | 8.1 ms | 203 MB |

The numbers are noisy (laptop, other processes) and ~10-15 ms per position
across the board: the model is small enough that variant and thread count
matter less than the machine. Expect 2-3x slower on a shared serverless
x86 vCPU (not measured).

**onnxruntime-web (wasm)**, Node-hosted (`scripts/web-check.mjs`, 1 wasm
thread, model passed as bytes). All files load and run, parity identical to
onnxruntime-node:

| model | session create | first run | p50 | p95 | top-1 vs golden |
|---|---:|---:|---:|---:|---:|
| fp32 | 351 ms | 50 ms | 28.4 ms | 30.8 ms | 100 % |
| fp16 | 111 ms | 30 ms | 30.1 ms | 41.8 ms | 100 % |
| int8 | 68 ms | 37 ms | 35.7 ms | 48.9 ms | 98.6 % |
| int8-fp16 | 57 ms | 42 ms | 34.1 ms | 57.8 ms | 99.2 % |
| w8 (MatMulNBits) | 68 ms | 66 ms | 55.4 ms | 74.8 ms | 99.2 % |

Not tested: a real browser, WebGPU, low-end phones. On top of the model the
browser downloads the ORT wasm binary (`ort-wasm-simd-threaded.wasm`, 14.2 MB,
3.7 MB gzipped).

## Where to run it: server or client

For an app, **server-side with the `fp16` file and onnxruntime-node** is
usually the better choice. The numbers:

- Client-side costs a 27-42 MB model download (25-39 MB gzipped; the weights
  do not compress) + 3.7 MB of wasm, per device, before the first move. That
  is acceptable on desktop, heavy on mobile web.
- Mobile Expo apps would need `onnxruntime-react-native` (a native module, so
  a new binary build, not an OTA) and would ship 27-42 MB more in each app
  bundle.
- Server-side, one position costs ~10-15 ms of CPU on M1 (maybe 30-50 ms on a
  small x86 vCPU), cold start ~0.1-0.3 s, ~220 MB RSS per process holding one
  model. A 2-vCPU box handles tens of positions per second, far above what
  human-speed games need. Load the session once per process, not per request.
- Keep fp32 compute; use the `fp16` file (half the disk/deploy size, same
  accuracy). Use `int8` only if RAM per process is the constraint.
- Client-side (browser worker) is a reasonable later option for an offline
  endgame trainer on desktop: use `int8-fp16` (27 MB) or `fp16` (42 MB),
  cached with the Cache API.

## How to integrate

Install the package and ship one model file. Keep `NOTICE` alongside the
model, and with any copy of `src/` too: the encoding is a port of maia2's
code. Depends on `chess.js` ^1.4.

### (a) NestJS + onnxruntime-node

```ts
import { Injectable, OnModuleInit } from '@nestjs/common';
import * as ort from 'onnxruntime-node';
import { createMaia, sampleMove, type Maia } from '@indiefoundry/maia2-onnx';

@Injectable()
export class MaiaService implements OnModuleInit {
  private maia!: Maia;
  async onModuleInit() {
    const session = await ort.InferenceSession.create('assets/maia2-rapid.fp16.onnx', {
      intraOpNumThreads: 2, // leave cores to the rest of the app
    });
    this.maia = createMaia({ session, Tensor: ort.Tensor });
  }
  async humanMove(fen: string, elo: number, oppoElo: number) {
    const pred = await this.maia.predict(fen, elo, oppoElo);
    return { move: sampleMove(pred, { minP: 0.01 }), pred };
  }
}
```

One session serves concurrent requests (checked: 360 concurrent `predict()` on one fp16 session, all top-1 identical to golden, 2.9 s total);
a burst can be grouped with `predictBatch` (fp32/fp16 are batch-invariant).

### (b) Vercel / Next.js route or Node serverless function

```ts
// app/api/maia/route.ts
export const runtime = 'nodejs'; // not edge: onnxruntime-node is a native addon
import * as ort from 'onnxruntime-node';
import path from 'node:path';
import { createMaia } from '@indiefoundry/maia2-onnx';

let maiaP: Promise<ReturnType<typeof createMaia>> | undefined;
const getMaia = () => (maiaP ??= ort.InferenceSession
  .create(path.join(process.cwd(), 'models/maia2-rapid.fp16.onnx'), { intraOpNumThreads: 1 })
  .then((session) => createMaia({ session, Tensor: ort.Tensor })));

export async function POST(req: Request) {
  const { fen, eloSelf, eloOppo } = await req.json();
  return Response.json(await (await getMaia()).predict(fen, eloSelf, eloOppo));
}
```

Caveats (not verified on Vercel): onnxruntime-node ships binaries for all
platforms (288 MB installed; linux x64 `libonnxruntime.so.1` alone is 46 MB);
the function bundle limit is 250 MB unzipped, so the model file (42 MB) must be
included (`outputFileTracingIncludes`) and the darwin/win32 binaries excluded
(`outputFileTracingExcludes`, `serverExternalPackages: ['onnxruntime-node']`).
Each cold start pays the session creation (~0.1-0.3 s + reading the file).
Given this, a long-running Node server is the better home for inference.

### (c) Browser, onnxruntime-web in a Web Worker

```ts
// maia.worker.ts
import * as ort from 'onnxruntime-web';
import { createMaia } from '@indiefoundry/maia2-onnx';

ort.env.wasm.numThreads = 1; // >1 needs cross-origin isolation (COOP/COEP) for SharedArrayBuffer
ort.env.wasm.wasmPaths = '/ort/'; // serve the matching ort-wasm-*.wasm AND ort-wasm-*.mjs from node_modules/onnxruntime-web/dist/ here
const maiaP = (async () => {
  const cache = await caches.open('maia2');
  const url = '/models/maia2-rapid.int8-fp16.onnx';
  const res = (await cache.match(url)) ?? (await fetch(url).then(async (r) => { await cache.put(url, r.clone()); return r; }));
  const session = await ort.InferenceSession.create(new Uint8Array(await res.arrayBuffer()), { executionProviders: ['wasm'] });
  return createMaia({ session, Tensor: ort.Tensor });
})();

self.onmessage = async ({ data: { id, fen, eloSelf, eloOppo } }) => {
  self.postMessage({ id, pred: await (await maiaP).predict(fen, eloSelf, eloOppo) });
};
```

## Licence due diligence (facts)

- **Code**: github.com/CSSLab/maia2 and the PyPI package `maia2` 0.11.0 are
  MIT, "Copyright (c) 2024 CSSLab" (`LICENSE` in the wheel's
  `dist-info/licenses/`, PyPI classifier "License :: OSI Approved :: MIT
  License"). README: "Maia-2 is released under the MIT License."
- **Weights download source** (`maia2/model.py`, `_MODEL_ASSETS`): Google
  Drive, via `gdown`, with a SHA-256 check:
  - rapid: `https://drive.google.com/uc?id=1gbC1-c7c0EQOPPAVpGWubezeEW8grVwc`
    (Drive file name `weights.v2.pt`, saved as `rapid_model.pt`, 279,704,570
    bytes, sha256 `65aae8465eed5e65df66a24ea7370715579f9e5435098d06fe18bdb1e267e997`)
  - blitz: `https://drive.google.com/uc?id=1X-Z4J3PX3MQFJoa8gRt3aL8CIH0PWoyt`
    (Drive file name `maia2_blitz.pt`, saved as `blitz_model.pt`, 279,704,570
    bytes, sha256 `5090d5d0d49dc29787c08d13febbed5b2da81a18c2ddcd8a90070cd3b43c44b2`)
  - The files sit in a personal Google Drive, not on a CSSLab host or
    Hugging Face.
- **No licence text is attached to the weights.** The checkpoint is a plain
  torch dict (`model_state_dict`, `optimizer_state_dict`,
  `accumulated_samples`, `accumulated_games`), no metadata; the Drive files
  carry no licence; there is no Maia-2 model card (the Hugging Face
  collection `UofTCSSLab/maia3` is Maia-3 only). The only licence in the
  distribution chain is the repository/package MIT licence, and the package
  that downloads the weights is itself MIT and documents the weights as "the
  released models".
- **Paper** (arXiv 2409.20553, NeurIPS 2024): the abstract page contains no
  statement on code, weights or licence.
- **Training data**: Lichess games May 2018 - Nov 2023 (`configs/maia2-training.yaml`);
  the Lichess database is published under CC0.
- Practical reading (not legal advice): the weights are distributed as part of
  an MIT project with no separate terms, so commercial use in closed apps is
  consistent with what is published; keep the MIT notice (`NOTICE`) with the
  model files and cite the paper. The one residual risk is that the weights'
  licence is implied by the repo rather than stated on the files themselves;
  a written confirmation from the authors would close it.

## Known limits

- Openings (first ~10 plies) are out of the training distribution (see above).
- Elo is bucketed into 11 categories between <1100 and >=2000.
- Only standard chess (no Chess960 castling).
- `predict` trusts the FEN; chess.js throws on an invalid FEN, and a position
  without legal moves (mate/stalemate) throws.

## Citation

If you use this in research, cite the Maia-2 paper:

```bibtex
@inproceedings{tang2024maia2,
  title     = {Maia-2: A Unified Model for Human-AI Alignment in Chess},
  author    = {Tang, Zhenwei and Jiao, Difan and McIlroy-Young, Reid and Kleinberg, Jon and Sen, Siddhartha and Anderson, Ashton},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

## License

The TypeScript code and scripts in this repository are MIT (`LICENSE`).
Maia-2 and its weights are by CSSLab, MIT (`NOTICE`); keep `NOTICE` with any
model file you redistribute and with any copy of the ported encoding code. This project is not affiliated with CSSLab.
