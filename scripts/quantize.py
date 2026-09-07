#!/usr/bin/env python3
"""registry.json の設定に従って ONNX モデルを量子化する。

このスクリプトが CI に載っていることの価値は、`excludeOpTypes` のような
「試して測るしかない」設定をコードとして残し、PR ごとに精度差分を出せる点にある
（docs/07 §7.3）。手元の手作業ではこの探索は回らない。

出力: <out>/<model-id>.<mode>.onnx と <out>/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import onnx

from _registry import Model, Registry

# 量子化から常に除外する演算。数値的に敏感で、量子化すると出力が大きく崩れる。
ALWAYS_EXCLUDE_OPS = {"LayerNormalization", "Softmax", "InstanceNormalization", "GroupNormalization"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def node_names_of_types(model: onnx.ModelProto, op_types: set[str]) -> list[str]:
    """指定した演算型のノード名を集める。ORT の nodes_to_exclude はノード名で指定するため。"""
    names: list[str] = []
    for node in model.graph.node:
        if node.op_type in op_types and node.name:
            names.append(node.name)
    return names


def head_node_names(model: onnx.ModelProto, tail_ratio: float = 0.15) -> list[str]:
    """グラフ末尾側（デコーダ／ヘッド）のノード名。

    深度推定では最終的な数値精度をヘッドが決めるので、ここを 4bit にすると
    AbsRel が跳ね上がる。トポロジカル順の末尾 tail_ratio を「ヘッド」とみなす
    ヒューリスティックで、閾値は CI の較正結果を見て調整する（docs/08 Q群）。
    """
    nodes = [n for n in model.graph.node if n.name]
    cut = max(1, int(len(nodes) * (1.0 - tail_ratio)))
    return [n.name for n in nodes[cut:]]


def load_model(path: Path) -> onnx.ModelProto:
    # 外部データ（model.onnx_data）を伴うモデルもここで読み込まれる
    return onnx.load(str(path), load_external_data=True)


def save_model(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # ブラウザ配信では単一ファイルのほうが扱いが簡単なので、外部データにしない。
    # 2GB を超える場合のみ外部データへ退避する。
    try:
        onnx.save(model, str(path), save_as_external_data=False)
    except ValueError:
        onnx.save(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + "_data",
        )


def to_fp16(model: onnx.ModelProto, block_ops: set[str]) -> onnx.ModelProto:
    from onnxconverter_common import float16

    return float16.convert_float_to_float16(
        model,
        keep_io_types=True,  # 入出力は fp32 のまま。呼び出し側の前後処理を単純に保つ
        disable_shape_infer=False,
        op_block_list=sorted(block_ops),
    )


def quantize_q4f16(src: Path, dst: Path, cfg: dict[str, Any]) -> None:
    """fp16 化したうえで MatMul の重みを 4bit にする（transformers.js の q4f16 相当）。"""
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer,
        RTNWeightOnlyQuantConfig,
    )

    model = load_model(src)
    block_ops = ALWAYS_EXCLUDE_OPS | set(cfg.get("excludeOpTypes", []))
    model = to_fp16(model, block_ops)

    exclude = node_names_of_types(model, block_ops)
    if cfg.get("keepHeadFp16", False):
        exclude += head_node_names(model, tail_ratio=float(cfg.get("headTailRatio", 0.15)))

    quantizer = MatMulNBitsQuantizer(
        model,
        block_size=int(cfg.get("blockSize", 32)),
        is_symmetric=bool(cfg.get("symmetric", True)),
        nodes_to_exclude=sorted(set(exclude)),
        algo_config=RTNWeightOnlyQuantConfig(),
    )
    quantizer.process()
    save_model(quantizer.model.model, dst)


def quantize_uint8(src: Path, dst: Path, cfg: dict[str, Any]) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = load_model(src)
    block_ops = ALWAYS_EXCLUDE_OPS | set(cfg.get("excludeOpTypes", []))
    exclude = node_names_of_types(model, block_ops)

    # 外部データ付きの元モデルは一度単一ファイルに正規化してから渡す
    tmp = dst.with_suffix(".tmp.onnx")
    save_model(model, tmp)
    try:
        quantize_dynamic(
            model_input=str(tmp),
            model_output=str(dst),
            weight_type=QuantType.QUInt8,
            per_channel=bool(cfg.get("perChannel", True)),
            reduce_range=bool(cfg.get("reduceRange", False)),
            nodes_to_exclude=sorted(set(exclude)),
            extra_options={"EnableSubgraph": True},
        )
    finally:
        tmp.unlink(missing_ok=True)
        Path(str(tmp) + "_data").unlink(missing_ok=True)


def quantize_fp16(src: Path, dst: Path, cfg: dict[str, Any]) -> None:
    model = load_model(src)
    block_ops = ALWAYS_EXCLUDE_OPS | set(cfg.get("excludeOpTypes", []))
    save_model(to_fp16(model, block_ops), dst)


QUANTIZERS = {"q4f16": quantize_q4f16, "uint8": quantize_uint8, "fp16": quantize_fp16}


def process_model(m: Model, raw_dir: Path, out_dir: Path, force: bool) -> dict[str, Any]:
    mode = m.quant_mode
    dst = out_dir / f"{m.id}.{mode}.onnx"
    entry: dict[str, Any] = {
        "id": m.id, "role": m.role, "mode": mode, "license": m.license,
        "commercialUse": m.commercial_use, "inputSize": list(m.input_size),
    }

    # 公開済みの量子化版があれば流用する（CI 時間の節約。registry の prequantized）
    pre_rel = m.prequantized(mode)
    pre_path = (raw_dir / m.id / pre_rel) if pre_rel else None
    src = raw_dir / m.id / m.raw["hf"]["file"]

    if not force and pre_path is not None and pre_path.exists():
        shutil.copyfile(pre_path, dst)
        entry["source"] = f"prequantized:{pre_rel}"
    elif src.exists():
        fn = QUANTIZERS.get(mode)
        if fn is None:
            shutil.copyfile(src, dst)
            entry["source"] = "copied (mode=none)"
        else:
            print(f"[quantize] {m.id}: {mode} …", flush=True)
            fn(src, dst, m.raw.get("quantization", {}))
            entry["source"] = "quantized"
    else:
        entry["error"] = f"元モデルが見つかりません: {src}"
        return entry

    size = dst.stat().st_size
    entry["bytes"] = size
    entry["sha256"] = sha256_of(dst)
    entry["file"] = dst.name

    expected = m.expected_bytes
    if expected:
        ratio = abs(size - expected) / expected
        entry["expectedBytes"] = expected
        entry["sizeDeviation"] = round(ratio, 3)
        if ratio > m.size_tolerance:
            entry["warning"] = (
                f"サイズが想定から {ratio:.0%} 乖離（実 {size/1e6:.1f}MB / 想定 {expected/1e6:.1f}MB）"
            )
    print(f"[quantize] {m.id}: {size/1e6:.1f} MB ({entry['source']})", flush=True)
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description="registry.json に従って ONNX を量子化する")
    ap.add_argument("--in", dest="raw", type=Path, required=True, help="fetch_models.py の出力先")
    ap.add_argument("--out", type=Path, required=True, help="量子化結果の出力先")
    ap.add_argument("--registry", type=Path, default=None, help="registry.json のパス（差分比較用）")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true", help="prequantized を無視して必ず自前量子化する")
    args = ap.parse_args()

    reg = Registry.load(args.registry) if args.registry else Registry.load()
    models = list(reg.models.values()) if args.all else reg.models_for_profile(args.profile)

    args.out.mkdir(parents=True, exist_ok=True)
    entries = [process_model(m, args.raw, args.out, args.force) for m in models]

    non_commercial = [e["id"] for e in entries if not e.get("commercialUse", True)]
    manifest = {
        "profile": args.profile or reg.default_profile,
        "models": entries,
        "totalBytes": sum(int(e.get("bytes", 0)) for e in entries),
        "warnings": [e["warning"] for e in entries if "warning" in e],
        "nonCommercialModels": non_commercial,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    errs = [e for e in entries if "error" in e]
    for e in errs:
        print(f"[quantize] エラー {e['id']}: {e['error']}", file=sys.stderr)
    for w in manifest["warnings"]:
        print(f"[quantize] 警告: {w}", file=sys.stderr)
    if non_commercial:
        print(
            f"[quantize] 警告: 非商用ライセンスのモデルが含まれます: {', '.join(non_commercial)}",
            file=sys.stderr,
        )

    print(f"[quantize] 合計 {manifest['totalBytes']/1e6:.1f} MB")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
