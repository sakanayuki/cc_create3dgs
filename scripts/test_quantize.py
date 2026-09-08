#!/usr/bin/env python3
"""量子化と較正の自己テスト。小さなモデルを作って、実際に読めるかまで見る。

本物のモデルは数百 MB あり、CI で回すと数分かかる。壊れ方の多くは
グラフの型の食い違いなので、**同じ形をした小さなモデル**で再現できる。
本番の量子化を始める前にここで落とせば、10 分待ってから失敗するのを防げる。

守っている不具合:
  ・注意機構の `Cast(to=FLOAT) → Softmax` を fp16 変換すると、
    Cast の `to` が書き換わらないまま重みだけ float16 になり、
    後続の MatMul が float と float16 を混ぜた壊れたグラフになる。
    onnxruntime は読み込み時に弾く（実際に deploy #18 で発生）。
  ・その修復で挿す Cast の名前が、周回ごとに 0 から振り直されて
    重複する（two nodes with same node name）。1ブロックのモデルでは
    1周で収まるので見逃した。ここではブロックを重ねて何周も回させる。
  ・Depth Anything 3 は多視点モデルで、入力が [batch, views, 3, H, W] の
    5階になる。NCHW を決め打ちで渡すと onnxruntime が弾く
    （Invalid rank for input: pixel_values Got: 4 Expected: 5）。
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


def attention_like(path: Path, dim: int = 32, blocks: int = 4) -> None:
    """注意機構と同じ形のブロックを重ねる。

    ブロックを複数にするのは、型の修復が**何周も回る**状況を作るため。
    1周で収まるモデルだけを見ていたせいで、周回ごとに名前を振り直す
    不具合（two nodes with same node name）を見逃した。
    """
    rng = np.random.default_rng(0)
    nodes = []
    inits = []
    cur = "x"
    for b in range(blocks):
        w, v = f"W{b}", f"V{b}"
        inits += [
            numpy_helper.from_array(rng.standard_normal((dim, dim)).astype(np.float32), w),
            numpy_helper.from_array(rng.standard_normal((dim, dim)).astype(np.float32), v),
        ]
        nodes += [
            helper.make_node("MatMul", [cur, w], [f"scores{b}"]),
            # ここが肝。fp16 変換はこの to 属性を書き換えない。
            helper.make_node("Cast", [f"scores{b}"], [f"scores_f32_{b}"],
                             to=TensorProto.FLOAT, name=f"attn{b}/Cast_1"),
            helper.make_node("Softmax", [f"scores_f32_{b}"], [f"probs{b}"], axis=-1),
            helper.make_node("Cast", [f"probs{b}"], [f"probs_2_{b}"],
                             to=TensorProto.FLOAT, name=f"attn{b}/Cast_2"),
            helper.make_node("MatMul", [f"probs_2_{b}", v], [f"blk{b}"]),
        ]
        cur = f"blk{b}"
    nodes.append(helper.make_node("Identity", [cur], ["y"]))

    graph = helper.make_graph(
        nodes,
        "attention_like",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, dim])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, dim])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(onnx.shape_inference.infer_shapes(model), str(path))


def da3_like(path: Path, size: int = 16) -> None:
    """Depth Anything 3 と同じ形。入力が 5 階、出力に視点軸が付く。"""
    axes = numpy_helper.from_array(np.array([2], dtype=np.int64), "axes")
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["pixel_values", "axes"], ["depth"], keepdims=0)],
        "da3_like",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 1, 3, size, size])],
        [helper.make_tensor_value_info("depth", TensorProto.FLOAT, [1, 1, size, size])],
        [axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, str(path))


def two_input_like(path: Path, size: int = 16) -> None:
    """MI-GAN と同じ形。uint8 の画像とマスクの2入力を取る。"""
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["mask"], ["m32"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["image"], ["i32"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["i32", "m32"], ["r32"]),
            helper.make_node("Cast", ["r32"], ["result"], to=TensorProto.UINT8),
        ],
        "two_input",
        [
            helper.make_tensor_value_info("image", TensorProto.UINT8, [1, 3, size, size]),
            helper.make_tensor_value_info("mask", TensorProto.UINT8, [1, 1, size, size]),
        ],
        [helper.make_tensor_value_info("result", TensorProto.UINT8, [1, 3, size, size])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


def check_multi_input(d: Path) -> bool:
    """2入力のモデル（MI-GAN）に、必要な入力を全部渡せることを見る。

    較正はここでも落ちた。最初の1つしか渡していなかったため。

        Required inputs (['mask']) are missing from input feed (['image'])
    """
    import calibrate as cal

    src = d / "twoinput.onnx"
    two_input_like(src)
    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    img = np.random.default_rng(3).random((1, 3, 16, 16)).astype(np.float32)

    ok = True
    try:
        sess.run(None, {"image": (img * 255).astype(np.uint8)})
        ok &= check("2入力モデルに1つだけ渡すと落ちる（前提の確認）", False, "落ちなかった")
    except Exception:
        ok &= check("2入力モデルに1つだけ渡すと落ちる（前提の確認）", True)

    try:
        feed = cal.make_feed(sess, img, "inpaint")
        names = sorted(feed)
        ok &= check("較正が2入力を全部埋める", names == ["image", "mask"], ", ".join(names))
        ok &= check(
            "uint8 の入力を uint8 で渡す",
            all(feed[n].dtype == np.uint8 for n in feed),
            ", ".join(f"{n}={feed[n].dtype}" for n in feed),
        )
        sess.run(None, feed)
        ok &= check("較正が2入力モデルを動かせる", True)
    except Exception as e:
        ok &= check("較正が2入力モデルを動かせる", False, str(e)[:200])
    return ok


def check_rank_handling(d: Path) -> bool:
    """多視点モデル（5階入力）に NCHW を渡しても通ることを見る。

    較正はここで落ちた。本物のモデルを 300 MB 落としてからでは遅い。
    """
    import calibrate as cal

    src = d / "da3like.onnx"
    da3_like(src)
    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(2).random((1, 3, 16, 16)).astype(np.float32)

    ok = True
    try:
        sess.run(None, {"pixel_values": x})
        ok &= check("5階モデルに NCHW を直接渡すと落ちる（前提の確認）", False, "落ちなかった")
    except Exception:
        ok &= check("5階モデルに NCHW を直接渡すと落ちる（前提の確認）", True)

    try:
        out = cal.run(sess, x, "depth")
        ok &= check("較正が5階モデルを扱える", out.ndim == 3, f"出力の階数 {out.ndim}")
    except Exception as e:
        ok &= check("較正が5階モデルを扱える", False, str(e)[:200])
    return ok


def check_raw_paths() -> bool:
    """registry の全モデルで、`raw_path` が fetch_models.py の置き場所と一致するか。

    deploy #? で落ちた: `raw_path` が HF のファイル名だけを取っていて、
    `onnx/model.onnx` の `onnx/` が消えていた。`hf_hub_download(local_dir=...)`
    はリポジトリ内の相対パスをそのまま再現するので、実際は `onnx/` の下に
    置かれる。DA3 と ISNet だけが「元モデルが見つかりません」で落ち、
    ディレクトリを持たない mi-gan と、量子化済みを流用する modnet は
    通ってしまうので、気づきにくい壊れ方をした。

    ここは registry を読むだけで済む検査なので、本物のモデルは要らない。
    """
    from _registry import Registry

    reg = Registry.load()
    root = Path("/tmp/raw")
    ok = True
    for m in reg.models.values():
        got = m.raw_path(root)
        if m.url:
            want = root / m.id / Path(m.url).name
        else:
            # fetch_models.py: hf_hub_download(filename=rel, local_dir=out/id)
            want = root / m.id / m.raw["hf"]["file"]
        ok &= check(f"raw_path {m.id}", got == want, "" if got == want else f"{got} ≠ {want}")
    return ok


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

        def correlation(path: Path) -> float:
            got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
                None, {"x": x}
            )[0]
            if not np.isfinite(got).all():
                return float("nan")
            return float(np.corrcoef(ref.ravel(), got.ravel())[0, 1])

        # --- fp16 変換だけを単独で見る（4bit 量子化を挟まない）
        #
        # ここが**壊れたグラフを見つける一番鋭い検査**になる。fp16 は
        # 丸めしか起きないので、正しく変換できていれば相関はほぼ 1 になる。
        # 逆に、結線が狂えば一気に落ちる。実際 op_block_list を渡していた頃は、
        # ブロックが2段になった時点で −0.14 まで落ちていた。
        fp16_path = d / "attn.fp16.onnx"
        try:
            q.save_model(q.to_fp16(q.load_model(src), set(EXCLUDE)), fp16_path)
            err = q.verify_loadable(fp16_path)
            passed &= check("fp16: onnxruntime で読める", err is None, err or "")
            if err is None:
                c = correlation(fp16_path)
                passed &= check(f"fp16: fp32 と相関 {c:.4f}", c > 0.99)
        except Exception as e:
            passed &= check("fp16: 変換", False, str(e)[:200])

        for mode, fn in [("q4f16", q.quantize_q4f16), ("uint8", q.quantize_uint8)]:
            dst = d / f"attn.{mode}.onnx"
            try:
                fn(src, dst, {"excludeOpTypes": EXCLUDE})
            except Exception as e:
                passed &= check(f"{mode}: 量子化", False, str(e)[:200])
                continue
            passed &= check(f"{mode}: 量子化", True)

            # 名前の一意性はグラフの妥当性そのもの。onnxruntime も
            # 「two nodes with same node name」で弾く。読めるかどうかとは
            # 別に、ここで直接見ておく（原因が分かりやすいため）。
            produced = onnx.load(str(dst))
            names = [n.name for n in produced.graph.node if n.name]
            dupes = {n for n in names if names.count(n) > 1}
            passed &= check(f"{mode}: ノード名が重複しない", not dupes, ", ".join(sorted(dupes))[:120])

            tensors = [o for n in produced.graph.node for o in n.output]
            tdupes = {t for t in tensors if tensors.count(t) > 1}
            passed &= check(f"{mode}: 出力テンソル名が重複しない", not tdupes, ", ".join(sorted(tdupes))[:120])

            err = q.verify_loadable(dst)
            passed &= check(f"{mode}: onnxruntime で読める", err is None, err or "")
            if err:
                continue

            corr = correlation(dst)
            passed &= check(f"{mode}: 出力が有限", not np.isnan(corr))
            if not np.isnan(corr):
                # 閾値は方式ごとに変える。乱数の重みに 4bit 量子化を掛けるのは
                # 最悪ケース（低ランク構造が無いので RTN の誤差が最大になり、
                # ブロックを重ねるほど積み上がる）で、学習済みの重みでは
                # ここまで落ちない。この検査の狙いは精度の保証ではなく、
                # **グラフが壊れていないこと**なので、4bit には緩い下限を置く。
                # 精度の鋭い検査は上の fp16 単独（> 0.99）が担う。
                floor = 0.5 if mode == "q4f16" else 0.99
                passed &= check(f"{mode}: fp32 と相関 {corr:.4f}", corr > floor)

        passed &= check_rank_handling(d)
        passed &= check_multi_input(d)
        passed &= check_raw_paths()

        # 壊れたファイルを見逃さないこと
        bad = d / "broken.onnx"
        bad.write_bytes(b"not an onnx file")
        passed &= check("壊れたファイルを検出できる", q.verify_loadable(bad) is not None)

    print("量子化と較正の自己テスト:", "合格" if passed else "不合格")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
