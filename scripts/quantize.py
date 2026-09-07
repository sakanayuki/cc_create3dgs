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


def _clear_value_info(graph: onnx.GraphProto) -> None:
    """中間テンソルの型宣言を、入れ子のサブグラフまで含めて捨てる。"""
    del graph.value_info[:]
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                _clear_value_info(attr.g)
            for sub in attr.graphs:
                _clear_value_info(sub)


def refresh_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    """中間テンソルの型宣言を、実際のノードから付け直す。

    fp16 変換のあとに必ず必要になる。onnxconverter_common は value_info を
    まとめて float16 に書き換えるが、`to=FLOAT` を持つ Cast ノードは
    変換後も float を出す。宣言と実体が食い違ったまま保存されるため、
    onnxruntime が読み込み時に弾く。

        Type Error: Type (tensor(float16)) of output arg (…/attn/Cast_2) of
        node (…/attn/Cast_2) does not match expected type (tensor(float)).

    グラフ自体は正しいので、宣言を捨てて推論し直せば直る。
    """
    _clear_value_info(model.graph)
    try:
        return onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception as e:  # 推論できなくても、宣言が空なら実行時に困らない
        print(f"[quantize] 形状推論をやり直せませんでした（続行します）: {e}", file=sys.stderr)
        return model


# 入力の型が揃っていなければならない演算。ONNX の型制約で、
# 浮動小数の入力すべてが同じ型パラメータに束縛される。
_SAME_FLOAT_INPUT_OPS = {
    "Add", "Sub", "Mul", "Div", "Pow", "Min", "Max", "Mean", "Sum",
    "MatMul", "Gemm", "Concat", "Where", "Equal", "Greater", "Less",
    "GreaterOrEqual", "LessOrEqual", "Mod", "PRelu", "BiasGelu",
}
_FLOAT_TYPES = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}


def _type_map(model: onnx.ModelProto) -> dict[str, int]:
    """テンソル名 → 要素型。推論できた範囲で集める。"""
    types: dict[str, int] = {}
    for init in model.graph.initializer:
        types[init.name] = init.data_type
    for group in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in group:
            if vi.type.HasField("tensor_type"):
                types[vi.name] = vi.type.tensor_type.elem_type
    return types


def repair_mixed_precision(model: onnx.ModelProto, block_ops: set[str]) -> onnx.ModelProto:
    """浮動小数の型が混ざったノードに Cast を挿して直す。

    onnxconverter_common の fp16 変換は、**モデルに元からある Cast ノードの
    `to` 属性を書き換えない**。注意機構では数値安定性のために
    `Cast(to=FLOAT) → Softmax` を明示的に置くのが定石なので、変換後に
    「float のテンソル」と「float16 になった重み」が同じ MatMul に入る、
    という壊れたグラフができる。onnxruntime は読み込み時にこう言って弾く。

        Type Error: Type parameter (T) of Optype (MatMul) bound to
        different types (tensor(float) and tensor(float16))

    ブロックした演算は fp32 で動かしたいので float に、それ以外は float16 に
    揃える。型推論のやり直しと交互に回し、変化しなくなるまで繰り返す。
    """
    for _ in range(4):
        model = refresh_value_info(model)
        types = _type_map(model)
        inserted: list[onnx.NodeProto] = []
        index: dict[str, int] = {}

        for pos, node in enumerate(model.graph.node):
            if node.op_type not in _SAME_FLOAT_INPUT_OPS:
                continue
            kinds = {types.get(i) for i in node.input if types.get(i) in _FLOAT_TYPES}
            if len(kinds) < 2:
                continue

            target = (
                onnx.TensorProto.FLOAT
                if node.op_type in block_ops
                else onnx.TensorProto.FLOAT16
            )
            for slot, name in enumerate(node.input):
                if types.get(name) not in _FLOAT_TYPES or types.get(name) == target:
                    continue
                cast_name = f"{name}_fixcast_{len(inserted)}"
                inserted.append(
                    onnx.helper.make_node(
                        "Cast", [name], [cast_name], to=target,
                        name=f"repair_cast_{len(inserted)}",
                    )
                )
                index[cast_name] = pos
                node.input[slot] = cast_name

        if not inserted:
            return model

        # 挿した Cast は、使う側のノードの直前に置く。
        nodes = list(model.graph.node)
        for cast in sorted(inserted, key=lambda n: index[n.output[0]], reverse=True):
            nodes.insert(index[cast.output[0]], cast)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        print(f"[quantize] 型の食い違いを {len(inserted)} 箇所直しました", flush=True)

    return refresh_value_info(model)


def to_fp16(model: onnx.ModelProto, block_ops: set[str]) -> onnx.ModelProto:
    from onnxconverter_common import float16

    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=True,  # 入出力は fp32 のまま。呼び出し側の前後処理を単純に保つ
        disable_shape_infer=False,
        op_block_list=sorted(block_ops),
    )
    return repair_mixed_precision(converted, block_ops)


def quantize_q4f16(src: Path, dst: Path, cfg: dict[str, Any]) -> None:
    """fp16 化したうえで MatMul の重みを 4bit にする（transformers.js の q4f16 相当）。"""
    from onnxruntime.quantization import matmul_nbits_quantizer as mnq

    model = load_model(src)
    block_ops = ALWAYS_EXCLUDE_OPS | set(cfg.get("excludeOpTypes", []))
    model = to_fp16(model, block_ops)

    exclude = node_names_of_types(model, block_ops)
    if cfg.get("keepHeadFp16", False):
        exclude += head_node_names(model, tail_ratio=float(cfg.get("headTailRatio", 0.15)))

    # DefaultWeightOnlyQuantConfig は onnxruntime だけで完結する。
    # RTNWeightOnlyQuantConfig は onnxruntime 1.22 系では neural-compressor
    # （torch を引き連れてくる）を要求し、CI がそれで落ちた。手法としては
    # どちらも RTN（round-to-nearest）なので、結果に実質的な差は無い。
    # 引数の受け口はバージョンで違う（CI は 1.22 系、手元は 1.29 系）。
    # 名前付きで渡せなければ既定値で作る。ブロックサイズと対称性は
    # MatMulNBitsQuantizer 側にも渡しているので、そちらが効く。
    try:
        algo_config = mnq.DefaultWeightOnlyQuantConfig(
            block_size=int(cfg.get("blockSize", 32)),
            is_symmetric=bool(cfg.get("symmetric", True)),
        )
    except TypeError:
        algo_config = mnq.DefaultWeightOnlyQuantConfig()

    quantizer = mnq.MatMulNBitsQuantizer(
        model,
        block_size=int(cfg.get("blockSize", 32)),
        is_symmetric=bool(cfg.get("symmetric", True)),
        nodes_to_exclude=sorted(set(exclude)),
        algo_config=algo_config,
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


def verify_loadable(path: Path) -> str | None:
    """作ったモデルを onnxruntime で実際に読み込んでみる。

    量子化は「ファイルはできたが読めない」形で壊れうる。実際、fp16 変換が
    Cast ノードを見落として型の食い違うグラフを作り、CI はそれを配布物に
    含めたまま先へ進んで、後の工程で初めて落ちた。作った直後にここで
    読んでおけば、どのモデルのどの方式が壊れたのかがその場で分かる。

    @return エラーメッセージ。読めれば None。
    """
    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        return None
    except Exception as e:  # 読めないこと自体が結果なので、種類は問わない
        return str(e).splitlines()[0][:300]


def build_one(
    m: Model, mode: str, raw_dir: Path, out_dir: Path, force: bool
) -> dict[str, Any]:
    """1つの量子化方式ぶんを作り、その結果を返す。"""
    dst = out_dir / f"{m.id}.{mode}.onnx"
    info: dict[str, Any] = {"mode": mode, "file": dst.name}

    # 公開済みの量子化版があれば流用する（CI 時間の節約。registry の prequantized）
    pre_rel = m.prequantized(mode)
    pre_path = (raw_dir / m.id / pre_rel) if pre_rel else None
    src = raw_dir / m.id / m.raw["hf"]["file"]

    if not force and pre_path is not None and pre_path.exists():
        shutil.copyfile(pre_path, dst)
        info["source"] = f"prequantized:{pre_rel}"
    elif dst.exists() and not force:
        # 同じ方式を2つのバックエンドが共有する場合、2度作らない。
        info["source"] = "reused"
    elif src.exists():
        fn = QUANTIZERS.get(mode)
        if fn is None:
            shutil.copyfile(src, dst)
            info["source"] = "copied (mode=none)"
        else:
            print(f"[quantize] {m.id}: {mode} …", flush=True)
            fn(src, dst, m.raw.get("quantization", {}))
            info["source"] = "quantized"
    else:
        info["error"] = f"元モデルが見つかりません: {src}"
        return info

    # 読めないファイルを配布物に混ぜない。ここで止めれば原因がその場で分かる。
    load_error = verify_loadable(dst)
    if load_error:
        info["error"] = f"{mode} のモデルを読み込めません: {load_error}"
        print(f"[quantize] {m.id}/{mode}: 読み込み失敗 — {load_error}", file=sys.stderr)
        return info

    size = dst.stat().st_size
    info["bytes"] = size
    info["sha256"] = sha256_of(dst)
    print(f"[quantize] {m.id}/{mode}: {size/1e6:.1f} MB ({info['source']}, 読み込み確認済み)", flush=True)
    return info


def process_model(
    m: Model, raw_dir: Path, out_dir: Path, force: bool, backend_defaults: dict[str, str]
) -> dict[str, Any]:
    """バックエンドごとに量子化して、1モデルぶんのマニフェスト項目を返す（決定 D22）。

    同じ方式を2つのバックエンドが指す場合はファイルを1つだけ作り、
    マニフェストの byBackend が同じファイルを指す。
    """
    modes = m.modes_by_backend(backend_defaults) or {"default": m.quant_mode}
    entry: dict[str, Any] = {
        "id": m.id, "role": m.role, "license": m.license,
        "commercialUse": m.commercial_use, "inputSize": list(m.input_size),
    }

    built: dict[str, dict[str, Any]] = {}
    by_backend: dict[str, str] = {}
    for backend, mode in modes.items():
        if mode not in built:
            built[mode] = build_one(m, mode, raw_dir, out_dir, force)
        info = built[mode]
        if "error" in info:
            entry["error"] = info["error"]
            return entry
        by_backend[backend] = str(info["file"])

    entry["byBackend"] = by_backend
    entry["variants"] = [
        {k: v for k, v in info.items() if k != "source"} | {"source": info.get("source", "")}
        for info in built.values()
    ]
    # 互換のため、代表として最初のバックエンドのものを従来の位置にも書く。
    first = built[next(iter(modes.values()))]
    entry["mode"] = first["mode"]
    entry["file"] = first["file"]
    entry["bytes"] = first.get("bytes", 0)
    entry["sha256"] = first.get("sha256", "")
    entry["source"] = first.get("source", "")

    expected = m.expected_bytes
    size = int(entry["bytes"])
    if expected and size:
        ratio = abs(size - expected) / expected
        entry["expectedBytes"] = expected
        entry["sizeDeviation"] = round(ratio, 3)
        if ratio > m.size_tolerance:
            entry["warning"] = (
                f"サイズが想定から {ratio:.0%} 乖離（実 {size/1e6:.1f}MB / 想定 {expected/1e6:.1f}MB）"
            )
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
    backend_defaults = reg.backend_defaults
    entries = [
        process_model(m, args.raw, args.out, args.force, backend_defaults) for m in models
    ]

    non_commercial = [e["id"] for e in entries if not e.get("commercialUse", True)]
    manifest = {
        "profile": args.profile or reg.default_profile,
        "backends": list(backend_defaults.keys()),
        "models": entries,
        # 総量はファイルの重複を除いて数える。同じ方式を2つのバックエンドが
        # 共有していると、単純に足すと二重に数えてしまう。
        "totalBytes": sum(
            int(v.get("bytes", 0))
            for e in entries
            for v in {x["file"]: x for x in e.get("variants", [])}.values()
        ),
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
