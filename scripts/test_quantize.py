#!/usr/bin/env python3
"""量子化の自己テスト。小さなモデルを作って、実際に読めるかまで見る。

本物のモデルは数百 MB あり、CI で回すと数分かかる。壊れ方の多くは
グラフの型の食い違いなので、**同じ形をした小さなモデル**で再現できる。
本番の量子化を始める前にここで落とせば、10 分待ってから失敗するのを防げる。

守っている不具合:
  ・注意機構の `Cast(to=FLOAT) → Softmax` を fp16 変換すると、
    Cast の `to` が書き換わらないまま重みだけ float16 になり、
    後続の MatMul が float と float16 を混ぜた壊れたグラフになる。
    onnxruntime は読み込み時に弾く（実際に deploy #18 で発生）。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantize as q  # noqa: E402

EXCLUDE = ["LayerNormalization", "Softmax", "Sigmoid"]


def attention_like(path: Path, dim: int = 32) -> None:
    """注意機構と同じ形。数値安定性のための明示的な Cast を含む。"""
    rng = np.random.default_rng(0)
    W = numpy_helper.from_array(rng.standard_normal((dim, dim)).astype(np.float32), "W")
    V = numpy_helper.from_array(rng.standard_normal((dim, dim)).astype(np.float32), "V")
    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["scores"]),
        # ここが肝。fp16 変換はこの to 属性を書き換えない。
        helper.make_node("Cast", ["scores"], ["scores_f32"], to=TensorProto.FLOAT, name="attn/Cast_1"),
        helper.make_node("Softmax", ["scores_f32"], ["probs"], axis=-1),
        helper.make_node("Cast", ["probs"], ["probs_2"], to=TensorProto.FLOAT, name="attn/Cast_2"),
        helper.make_node("MatMul", ["probs_2", "V"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "attention_like",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, dim])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, dim])],
        [W, V],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(onnx.shape_inference.infer_shapes(model), str(path))


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK  ' if ok else 'NG  '} {name}{(' — ' + detail) if detail else ''}")
    return ok


def main() -> int:
    passed = True
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        src = d / "attn.onnx"
        attention_like(src)

        x = np.random.default_rng(1).standard_normal((1, 32)).astype(np.float32)
        ref = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]

        for mode, fn in [("q4f16", q.quantize_q4f16), ("uint8", q.quantize_uint8)]:
            dst = d / f"attn.{mode}.onnx"
            try:
                fn(src, dst, {"excludeOpTypes": EXCLUDE})
            except Exception as e:
                passed &= check(f"{mode}: 量子化", False, str(e)[:200])
                continue
            passed &= check(f"{mode}: 量子化", True)

            err = q.verify_loadable(dst)
            passed &= check(f"{mode}: onnxruntime で読める", err is None, err or "")
            if err:
                continue

            got = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
            finite = bool(np.isfinite(got).all())
            passed &= check(f"{mode}: 出力が有限", finite)
            if finite:
                corr = float(np.corrcoef(ref.ravel(), got.ravel())[0, 1])
                # 4bit 量子化を乱数の重みに掛けた最悪ケースでも 0.9 は下回らない
                passed &= check(f"{mode}: fp32 と相関 {corr:.4f}", corr > 0.9)

        # 壊れたファイルを見逃さないこと
        bad = d / "broken.onnx"
        bad.write_bytes(b"not an onnx file")
        passed &= check("壊れたファイルを検出できる", q.verify_loadable(bad) is not None)

    print("量子化の自己テスト:", "合格" if passed else "不合格")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
