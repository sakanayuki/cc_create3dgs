#!/usr/bin/env python3
"""PoC-2 のスレッド計測に使う合成モデルを生成する。

実モデル（27MB）を落とさずにスレッドのスケーリングだけ即座に測るためのもの。
畳み込みを積んで、単スレッドで数秒かかる程度の計算量にしてある。

決定的に生成されるので、出力をリポジトリにコミットしてよい。
`.gitignore` の `*.onnx` に対する例外として `public/threadbench.onnx` を登録している。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CHANNELS = 48
LAYERS = 10
KERNEL = 3
INPUT = 512


def build(seed: int = 0) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    inits: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []
    prev = "x"

    for i in range(LAYERS):
        cin = 3 if i == 0 else CHANNELS
        w = numpy_helper.from_array(
            (rng.standard_normal((CHANNELS, cin, KERNEL, KERNEL)) * 0.05).astype(np.float32),
            f"w{i}",
        )
        inits.append(w)
        nodes.append(
            helper.make_node("Conv", [prev, f"w{i}"], [f"c{i}"], name=f"conv{i}",
                             kernel_shape=[KERNEL, KERNEL], pads=[1, 1, 1, 1])
        )
        nodes.append(helper.make_node("Relu", [f"c{i}"], [f"h{i}"], name=f"relu{i}"))
        prev = f"h{i}"

    inits.append(
        numpy_helper.from_array(
            (rng.standard_normal((1, CHANNELS, 1, 1)) * 0.05).astype(np.float32), "wout"
        )
    )
    nodes.append(helper.make_node("Conv", [prev, "wout"], ["y"], name="convout", kernel_shape=[1, 1]))

    graph = helper.make_graph(
        nodes,
        "threadbench",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, INPUT, INPUT])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, INPUT, INPUT])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description="スレッド計測用の合成モデルを生成する")
    ap.add_argument("--out", type=Path, default=Path("public/threadbench.onnx"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build(), str(args.out))
    size = args.out.stat().st_size
    print(f"[threadbench] {args.out}: {size / 1e6:.2f} MB "
          f"（畳み込み {LAYERS} 層 × {CHANNELS}ch @{INPUT}²）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
