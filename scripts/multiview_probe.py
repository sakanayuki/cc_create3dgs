#!/usr/bin/env python3
"""複数枚モードの位置合わせを実素材で測るための、前段だけを回す道具（docs/12 §12.15 PoC-3A）。

位置合わせ（`src/pipeline/align/`）を実写で試すには、その手前の
⓪前処理・①背景除去・②深度推定を通した結果が要る。本番はブラウザで回すが、
測るためだけにブラウザを立てるのは重い。ここでは同じモデルを Python の
onnxruntime で回し、**位置合わせの入力だけ**を書き出す。

出力は view ごとに3つ。

    <out>/<slot>.json       … 幅・高さ・焦点距離・主点
    <out>/<slot>.alpha.u8   … α（uint8, grid×grid）
    <out>/<slot>.raw.f32    … 深度モデルの**生出力**（float32, grid×grid）

これを `scripts/align_probe.ts` が読んで位置合わせを回す。

③の較正は**ここではやらない**。生の深度を出して、TypeScript 側（align_probe.ts）が
本番と同じ `calibrate()` を呼ぶ。最初は較正なしで回したが、DA3 の実寸をそのまま
使うと被写体までの距離が被写体の高さより近いことになり、透視が実際より強く出た。
`calibrate()` は「奥行き ÷ 幅」を妥当な帯に収める処理を持っていて、そこが要る
（docs/12 §12.15.3、docs/03 §3.5.4）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from _registry import Registry

# ImageNet の正規化（src/pipeline/imageOps.ts の IMAGENET_MEAN / STD と同じ値）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 既定の作業グリッド（docs/03 §3.2、決定 D17）
WORKING_GRID = 1024


def fetch(model_id: str, cache: Path) -> tuple[Path, tuple[int, int]]:
    """registry のモデルを取ってきて、置いた場所と入力寸法を返す。

    **置き場所の知識は `_registry.py` の `Model` だけが持つ。** レジストリの
    生の辞書を読んで自分でパスを組み立ててはいけない
    （`scripts/test_quantize.py` の `check_registry_contract` が禁じている）。
    レジストリに 1 行足しただけで黙って壊れる前提を、散らさないための決まりである。
    最初この決まりを知らずに生の辞書を読んで、CI に叱られた。

    なお検出器は `#` コメントしか除いてくれないので、**この説明の中でも
    禁じられたキーを字面で書いてはいけない**。それも踏んだ。

    取り方は `fetch_models.py` と同じ。HF から取るモデルと、URL に直接
    置いてあるモデル（u2netp / face-mesh）の 2 通りがある。
    """
    from huggingface_hub import hf_hub_download

    m = Registry.load().models[model_id]
    dest = m.raw_path(cache)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if m.url:
        if not dest.exists():
            urllib.request.urlretrieve(m.url, dest)  # noqa: S310 - registry の固定 URL
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if m.sha256 and got != m.sha256:
            raise RuntimeError(f"{model_id} の sha256 が合いません: {got}")
        return dest, m.input_size

    for rel in m.hf_files():
        hf_hub_download(repo_id=m.hf_repo, filename=rel, local_dir=str(cache / m.id))
    return dest, m.input_size


def letterbox(path: Path, size: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """長辺を size に合わせ、正方形の中央に置く（src/pipeline/0-preprocess.ts と同じ考え）。"""
    im = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    s = size / max(im.size)
    w = max(1, round(im.width * s))
    h = max(1, round(im.height * s))
    im = im.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    ox = (size - w) // 2
    oy = (size - h) // 2
    canvas.paste(im, (ox, oy))
    return canvas, (ox, oy, w, h)


def to_nchw(img: Image.Image, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - mean) / std
    return np.transpose(a, (2, 0, 1))[None]


def run_matte(session, img: Image.Image, size: int, grid: int) -> np.ndarray:
    """MODNet で人物の α を出す。正規化は [-1,1]（imageOps.ts の既定と同じ）。"""
    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    x = to_nchw(img.resize((size, size), Image.BILINEAR), half, half)
    out = session.run(None, {session.get_inputs()[0].name: x})[0]
    matte = np.asarray(out).reshape(out.shape[-2], out.shape[-1])
    m = Image.fromarray(np.clip(matte * 255.0, 0, 255).astype(np.uint8))
    return np.array(m.resize((grid, grid), Image.BILINEAR), copy=True)


def shared_focal(session, imgs: list[Image.Image], size: int, grid: int) -> float:
    """3枚まとめて入れて、焦点距離だけを取る（InstantSplat の `--focal_avg` と同じ役）。

    **1枚ずつ回すと焦点距離が view 間で 18% ばらついた**（1057 / 983 / 1161 px）。
    同じカメラで撮った3枚なのだから、これは推定の誤差である。まとめて入れると
    1.5% 以内（828 / 822 / 816 px）に収まったので、こちらを使う。
    深度のほうはまとめて入れない（静止した場面を仮定するモデルなので、
    被写体が回っている素材とは前提が合わない）。docs/12 §12.15.3。
    """
    x = np.stack([to_nchw(im.resize((size, size), Image.BILINEAR), IMAGENET_MEAN, IMAGENET_STD)[0] for im in imgs])[None]
    names = [o.name for o in session.get_outputs()]
    out = dict(zip(names, session.run(None, {session.get_inputs()[0].name: x})))
    intr = np.asarray(out["intrinsics"]).reshape(-1, 3, 3)
    fx = float(np.mean(intr[:, 0, 0]))
    return fx * (grid / size)


def run_depth(session, img: Image.Image, size: int, grid: int) -> tuple[np.ndarray, float]:
    """DA3 で深度と焦点距離を出す。

    **1枚ずつ回す。** このモデルは複数枚を一度に受け取れる（入力が
    [batch, num_images, 3, H, W]）が、そちらは「静止した場面を別の位置から
    撮った」前提で解く。私たちの素材はカメラが固定で被写体が回っているので、
    前提が合わない。実際に3枚まとめて入れると extrinsics はほぼ単位行列に
    なった（＝カメラは動いていない）。docs/12 §12.15.3。
    """
    x = to_nchw(img.resize((size, size), Image.BILINEAR), IMAGENET_MEAN, IMAGENET_STD)[None]
    names = [o.name for o in session.get_outputs()]
    out = dict(zip(names, session.run(None, {session.get_inputs()[0].name: x})))

    depth = np.asarray(out["predicted_depth"]).reshape(size, size)
    d = np.asarray(Image.fromarray(depth).resize((grid, grid), Image.BILINEAR), dtype=np.float32)

    intr = np.asarray(out["intrinsics"]).reshape(3, 3)
    fx = float(intr[0, 0])
    return d, fx * (grid / size)


# MediaPipe Face Mesh の landmark 番号。顔の左右の端。
# 234 が被写体から見て右側、454 が左側（正面を向いていれば 234 が画像の左に写る）。
LM_RIGHT_SIDE = 234
LM_LEFT_SIDE = 454
LM_NOSE_TIP = 1


def head_box(alpha: np.ndarray) -> tuple[int, int, int] | None:
    """マットから頭のあたりの正方形を当てる（顔検出モデルは載せない、registry の方針）。"""
    ys, xs = np.nonzero(alpha >= 128)
    if len(ys) == 0:
        return None
    top, bottom = int(ys.min()), int(ys.max())
    h = bottom - top + 1
    band = (ys >= top) & (ys <= top + 0.20 * h)
    bx = xs[band]
    if len(bx) == 0:
        return None
    cx = int((bx.min() + bx.max()) / 2)
    side = int(max(bx.max() - bx.min() + 1, 0.18 * h) * 1.5)
    cy = int(top + side * 0.45)
    return cx - side // 2, cy - side // 2, side


def head_yaw(session, img: Image.Image, alpha: np.ndarray) -> dict | None:
    """顔の 3D landmark から頭のヨーを測る（docs/12 §12.15.5）。

    landmark は x/y/z が同じ尺度で返る（z は頭の中心を原点にした相対値、
    小さいほど手前）。顔の左右の端を結んだベクトルを x–z 平面で見れば、
    それがそのまま頭の向きになる。

        正面を向く … ベクトルは +x（画像の右）→ ヨー 0
        画面の右を向く … ベクトルは +z（奥）→ ヨー +90°

    符号の約束は `src/pipeline/align/rigid.ts` の ViewPose.yaw と同じにしてある。
    """
    box = head_box(alpha)
    if box is None:
        return None
    x0, y0, side = box
    patch = img.crop((x0, y0, x0 + side, y0 + side)).resize((256, 256), Image.BILINEAR)
    x = (np.asarray(patch, dtype=np.float32) / 255.0)[None]  # NHWC, 0..1（正規化しない）
    outs = session.run(None, {session.get_inputs()[0].name: x})
    lm = np.asarray(outs[0]).reshape(-1, 3)
    score = float(np.asarray(outs[1]).reshape(-1)[0])

    v = lm[LM_LEFT_SIDE] - lm[LM_RIGHT_SIDE]
    yaw = float(np.degrees(np.arctan2(v[2], v[0])))
    # 鼻がどれだけ前に出ているかも見る（顔らしさの目安）
    center = (lm[LM_LEFT_SIDE] + lm[LM_RIGHT_SIDE]) / 2
    nose = lm[LM_NOSE_TIP] - center
    return {
        "yawDeg": yaw,
        "score": score,
        "faceWidthPx": float(np.linalg.norm(v)),
        "noseForward": float(-nose[2]),
        "box": [x0, y0, side],
    }


SLOTS = ("front", "right", "left")


def main() -> int:
    ap = argparse.ArgumentParser(description="位置合わせの入力を実写から作る（PoC-3A）")
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--right", type=Path)
    ap.add_argument("--left", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path(".cache/models"))
    ap.add_argument("--grid", type=int, default=WORKING_GRID)
    args = ap.parse_args()

    import onnxruntime as ort

    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    print("モデルを用意します…", flush=True)
    depth_path, (depth_size, _) = fetch("depth-anything-v3-small", args.cache)
    matte_path, (matte_size, _) = fetch("modnet", args.cache)
    face_path, _ = fetch("face-mesh", args.cache)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    depth_sess = ort.InferenceSession(str(depth_path), opts, providers=["CPUExecutionProvider"])
    matte_sess = ort.InferenceSession(str(matte_path), opts, providers=["CPUExecutionProvider"])
    face_sess = ort.InferenceSession(str(face_path), opts, providers=["CPUExecutionProvider"])

    present = [(slot, getattr(args, slot)) for slot in SLOTS if getattr(args, slot) is not None]
    for slot, path in present:
        if not path.exists():
            print(f"  {slot}: {path} がありません", file=sys.stderr)
            return 1
    boxes = {slot: letterbox(path, args.grid) for slot, path in present}

    focal = shared_focal(depth_sess, [boxes[s][0] for s, _ in present], depth_size, args.grid)
    print(f"焦点距離（3枚まとめて推定・共有）= {focal:.1f}px", flush=True)

    manifest = []
    for slot, path in present:
        print(f"  {slot}: {path.name}", flush=True)
        img, box = boxes[slot]

        alpha = run_matte(matte_sess, img, matte_size, args.grid)
        depth, ownFocal = run_depth(depth_sess, img, depth_size, args.grid)

        # レターボックスの詰め物は被写体ではない（docs/03 §3.2.1）
        ox, oy, w, h = box
        outside = np.ones((args.grid, args.grid), dtype=bool)
        outside[oy : oy + h, ox : ox + w] = False
        alpha[outside] = 0
        depth = depth.astype(np.float32)

        # **生の深度をそのまま出す。** 較正（③）は TypeScript 側の calibrate() に
        # やらせる。ここで被写体の外を 0 で潰すと、較正が使う分位も外れ値の切り方も
        # 変わってしまう。位置合わせに渡す深度は align_probe.ts が作る。
        face = head_yaw(face_sess, img, alpha)

        (args.out / f"{slot}.alpha.u8").write_bytes(alpha.astype(np.uint8).tobytes())
        (args.out / f"{slot}.raw.f32").write_bytes(depth.tobytes())
        meta = {
            "slot": slot,
            "source": str(path),
            "width": args.grid,
            "height": args.grid,
            "focalPx": focal,
            "focalPxOwn": ownFocal,
            "cx": args.grid / 2,
            "cy": args.grid / 2,
            "subjectPixels": int((alpha >= 128).sum()),
            "rawRange": [float(depth[alpha >= 128].min()), float(depth[alpha >= 128].max())],
            "face": face,
        }
        (args.out / f"{slot}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        manifest.append(meta)
        print(
            f"    α={meta['subjectPixels']}px ({100 * meta['subjectPixels'] / args.grid**2:.1f}%) "
            f"焦点={focal:.1f}px 生の深度=[{meta['rawRange'][0]:.3f}, {meta['rawRange'][1]:.3f}]",
            flush=True,
        )
        if face:
            print(
                f"    顔: ヨー={face['yawDeg']:+.1f}° 確からしさ={face['score']:.2f} "
                f"顔幅={face['faceWidthPx']:.1f}px 鼻の出={face['noseForward']:+.1f}",
                flush=True,
            )
        else:
            print("    顔: 取れませんでした", flush=True)

    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"書き出しました: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
