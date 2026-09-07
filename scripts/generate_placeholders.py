#!/usr/bin/env python3
"""較正用のプレースホルダ画像を手続き的に生成する。

実写ではないので精度評価には使えない（models/calibration/README.md）。
配管の疎通確認と、実写が用意されるまでの穴埋めが目的。
深度推定・マット・インペイントのいずれもが「前景と背景の分離」「なだらかな階調」
「高周波テクスチャ」を含む入力を必要とするので、それらを混ぜた図を作る。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make(seed: int, size: int = 1024) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size

    # 背景: なだらかなグラデーション + 低周波の斑
    bg = np.stack([
        0.35 + 0.30 * yy + 0.10 * np.sin(6.0 * xx + seed),
        0.40 + 0.22 * yy + 0.10 * np.cos(5.0 * xx + seed * 1.7),
        0.50 + 0.18 * (1.0 - yy),
    ], axis=-1)

    # 前景: 楕円の被写体。深度らしい陰影を付ける
    cx, cy = 0.5 + 0.06 * rng.standard_normal(), 0.55 + 0.05 * rng.standard_normal()
    rx, ry = 0.20 + 0.06 * rng.random(), 0.30 + 0.08 * rng.random()
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    inside = d < 1.0
    shade = np.sqrt(np.clip(1.0 - d, 0.0, 1.0))  # 中心ほど手前＝明るい

    base = rng.random(3) * 0.5 + 0.35
    fg = base[None, None, :] * (0.45 + 0.55 * shade[..., None])

    # 高周波テクスチャ（コーデックと深度シャープ化の効きを見るため）
    tex = 0.06 * np.sin(60.0 * xx + 4.0 * seed) * np.sin(48.0 * yy)
    fg = fg + tex[..., None]

    img = np.where(inside[..., None], fg, bg)
    img = np.clip(img + 0.012 * rng.standard_normal(img.shape), 0.0, 1.0)
    return Image.fromarray((img * 255).astype(np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("models/calibration"))
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=82)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        kind = "person" if i < args.count // 2 else "object"
        n = i % (args.count // 2) + 1
        p = args.out / f"placeholder-{kind}-{n:02d}.jpg"
        make(seed=i * 17 + 3, size=args.size).save(p, quality=args.quality, optimize=True)
        print(f"[placeholder] {p} ({p.stat().st_size/1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
