"""Build and compare compressed variants of an exported Maia-2 ONNX model.

Usage (from the repo root, after export_onnx.py and make_fixtures.py):
    .venv/bin/python python/quant_variants.py rapid

Experimental variants written to models/ (export_onnx.py already wrote fp32, int8, fp16):
    maia2-<type>.int8pt.onnx      dynamic int8, per-TENSOR weights (onnxruntime default)
    maia2-<type>.w8.onnx          weight-only int8 (com.microsoft MatMulNBits, bits=8, block 32),
                                  activations stay fp32
    maia2-<type>.int8-fp16.onnx   int8 (per-channel) + fp16 storage of the remaining float
                                  weights (conv)

Each variant is evaluated against fixtures/golden-<type>.json (single-position runs,
encoding by maia2.inference.preprocessing, python onnxruntime).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from maia2 import inference
from maia2.utils import mirror_move
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import fp16_storage  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def build(kind: str):
    fp32 = MODELS / f"maia2-{kind}.onnx"
    quantize_dynamic(str(fp32), str(MODELS / f"maia2-{kind}.int8pt.onnx"), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"], per_channel=False)
    q = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=8, block_size=32, is_symmetric=True)
    q.process()
    q.model.save_model_to_file(str(MODELS / f"maia2-{kind}.w8.onnx"), use_external_data_format=False)
    # int8 per-channel for MatMul/Gemm + fp16 storage for what stays float (conv weights)
    fp16_storage(MODELS / f"maia2-{kind}.int8.onnx", MODELS / f"maia2-{kind}.int8-fp16.onnx")


def evaluate(kind: str, path: Path):
    golden = json.load(open(ROOT / "fixtures" / f"golden-{kind}.json"))
    all_moves_dict, elo_dict, rev = inference.prepare()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    top1 = 0
    maxd = 0.0
    maxw = 0.0
    for g in golden:
        b, cs, co, legal = inference.preprocessing(g["fen"], g["eloSelf"], g["eloOppo"], elo_dict, all_moves_dict)
        lm, lv = sess.run(None, {
            "boards": b.numpy()[None],
            "elo_self": np.array([cs], dtype=np.int64),
            "elo_oppo": np.array([co], dtype=np.int64),
        })
        mask = legal.numpy().astype(bool)
        x = np.where(mask, lm[0], -np.inf)
        p = np.exp(x - x.max())
        p /= p.sum()
        black = g["fen"].split(" ")[1] == "b"
        probs = {(mirror_move(rev[i]) if black else rev[i]): float(p[i]) for i in np.nonzero(mask)[0]}
        best = max(probs, key=probs.get)
        top1 += best == g["top10"][0]["uci"]
        for m in g["top10"]:
            maxd = max(maxd, abs(probs[m["uci"]] - m["p"]))
        v = min(1, max(0, lv[0] / 2 + 0.5))
        if black:
            v = 1 - v
        maxw = max(maxw, abs(v - g["maia2"]["winProb"]))
    return {
        "file": path.name,
        "bytes": os.path.getsize(path),
        "top1Agreement": round(top1 / len(golden), 4),
        "maxAbsDeltaTop10": round(float(maxd), 5),
        "maxAbsDeltaWinProb": round(float(maxw), 5),
    }


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "rapid"
    build(kind)
    rows = [evaluate(kind, MODELS / f"maia2-{kind}{s}.onnx") for s in ["", ".fp16", ".int8", ".int8pt", ".w8", ".int8-fp16"]]
    for r in rows:
        print(json.dumps(r))
    (ROOT / "fixtures" / f"variants-{kind}.json").write_text(json.dumps(rows, indent=2) + "\n")
