#!/usr/bin/env python3
"""複数枚モードの位置合わせを実素材で測るための、前段だけを回す道具（docs/12 §12.15 PoC-3A）。

位置合わせ（`src/pipeline/align/`）を実写で試すには、その手前の
⓪前処理・①背景除去・②深度推定を通した結果が要る。本番はブラウザで回すが、
測るためだけにブラウザを立てるのは重い。ここでは同じモデルを Python の
onnxruntime で回し、**位置合わせの入力だけ**を書き出す。

出力は view ごとに3つ。

    <out>/<slot>.json       … 幅・高さ・焦点距離・主点
    <out>/<slot>.alpha.u8   … α（uint8, grid×grid）
    <out>/<slot>.depth.f32  … 深度（float32, grid×grid, 被写体の外は 0）

これを `scripts/align_probe.ts` が読んで位置合わせを回す。

**本番のパイプラインの再現ではない。** ③の較正（エッジ保存シャープ化・局所強調・
断面の丸み…）は入れていない。位置合わせが見ているのはシルエットと大づかみの
深度だけなので、そこまでは要らない。ここで出す数字は「位置合わせが実写で立つか」
だけを答える。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# ImageNet の正規化（src/pipeline/imageOps.ts の IMAGENET_MEAN / STD と同じ値）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 既定の作業グリッド（docs/03 §3.2、決定 D17）
WORKING_GRID = 1024


def load_registry() -> dict:
    return json.loads((Path(__file__).resolve().parent.parent / "models" / "registry.json").read_text())


def fetch(model_id: str, cache: Path) -> Path:
    """registry.json の hf 定義に従ってモデルを落とす（決定 D7 と同じ出どころ）。"""
    from huggingface_hub import hf_hub_download

    entry = load_registry()["models"][model_id]
    hf = entry["hf"]
    dest = cache / model_id
    for name in [hf["file"], *hf.get("extraFiles", [])]:
        hf_hub_download(repo_id=hf["repo"], filename=name, local_dir=str(dest))
    return dest / hf["file"]


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
    depth_path = fetch("depth-anything-v3-small", args.cache)
    matte_path = fetch("modnet", args.cache)
    reg = load_registry()["models"]
    depth_size = int(reg["depth-anything-v3-small"]["inputSize"][0])
    matte_size = int(reg["modnet"]["inputSize"][0])

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    depth_sess = ort.InferenceSession(str(depth_path), opts, providers=["CPUExecutionProvider"])
    matte_sess = ort.InferenceSession(str(matte_path), opts, providers=["CPUExecutionProvider"])

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
        depth[alpha < 128] = 0.0

        (args.out / f"{slot}.alpha.u8").write_bytes(alpha.astype(np.uint8).tobytes())
        (args.out / f"{slot}.depth.f32").write_bytes(depth.tobytes())
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
            "depthRange": [float(depth[alpha >= 128].min()), float(depth[alpha >= 128].max())],
        }
        (args.out / f"{slot}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        manifest.append(meta)
        print(
            f"    α={meta['subjectPixels']}px ({100 * meta['subjectPixels'] / args.grid**2:.1f}%) "
            f"焦点={focal:.1f}px 深度=[{meta['depthRange'][0]:.3f}, {meta['depthRange'][1]:.3f}]",
            flush=True,
        )

    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"書き出しました: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
