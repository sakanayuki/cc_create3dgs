#!/usr/bin/env python3
"""量子化前後の精度を測り、閾値を割ったら CI を失敗させる。

役割ごとに見る指標を変える（docs/07 §7.3）:
  depth   : AbsRel と δ<1.05 精度の両方。AbsRel だけだと「顔の中央だけへこむ」
            ような局所的破綻を見逃すため。
  matte   : IoU と境界5px帯の MAE。
  inpaint : バンド領域の PSNR。GAN は量子化で崩れやすい。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from _registry import Model, Registry

THRESHOLDS: dict[str, dict[str, float]] = {
    # key: (指標名, 合格条件) — lower_is_better は max_、higher_is_better は min_
    "depth": {"max_absRel": 0.02, "min_delta105": 0.97},
    "matte": {"min_iou": 0.985, "max_boundaryMae": 4 / 255},
    "inpaint": {"min_psnr": 32.0},
}


def load_images(dir_: Path, size: tuple[int, int], limit: int | None = None) -> list[np.ndarray]:
    paths = sorted(p for p in dir_.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if limit:
        paths = paths[:limit]
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize(size, Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        out.append(a.transpose(2, 0, 1)[None])  # NCHW
    return out


def run(sess: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    name = sess.get_inputs()[0].name
    inp = sess.get_inputs()[0]
    # 入力が fp16 のモデルにも対応する
    dtype = np.float16 if "float16" in str(inp.type) else np.float32
    out = sess.run(None, {name: x.astype(dtype)})[0]
    return np.asarray(out, dtype=np.float32)


def normalize01(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi - lo < 1e-8:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def metrics_depth(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    # 深度はスケール不定なので、正規化してから比較する
    r, g = normalize01(ref.ravel()), normalize01(got.ravel())
    denom = np.maximum(r, 1e-3)
    abs_rel = float(np.mean(np.abs(r - g) / denom))
    ratio = np.maximum(np.maximum(r, 1e-3) / np.maximum(g, 1e-3),
                       np.maximum(g, 1e-3) / np.maximum(r, 1e-3))
    delta = float(np.mean(ratio < 1.05))
    return {"absRel": abs_rel, "delta105": delta}


def metrics_matte(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    r, g = normalize01(ref.ravel()), normalize01(got.ravel())
    rb, gb = r > 0.5, g > 0.5
    union = float(np.sum(rb | gb))
    iou = float(np.sum(rb & gb) / union) if union > 0 else 1.0
    # 境界帯: 参照の値が中間（0.15〜0.85）の画素
    band = (r > 0.15) & (r < 0.85)
    mae = float(np.mean(np.abs(r[band] - g[band]))) if band.any() else 0.0
    return {"iou": iou, "boundaryMae": mae}


def metrics_inpaint(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    r, g = normalize01(ref.ravel()), normalize01(got.ravel())
    mse = float(np.mean((r - g) ** 2))
    psnr = 99.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))
    return {"psnr": psnr}


METRICS = {"depth": metrics_depth, "matte": metrics_matte, "inpaint": metrics_inpaint}


def evaluate(m: Model, ref_path: Path, test_path: Path, images: list[np.ndarray]) -> dict[str, Any]:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    ref_sess = ort.InferenceSession(str(ref_path), opts, providers=["CPUExecutionProvider"])
    test_sess = ort.InferenceSession(str(test_path), opts, providers=["CPUExecutionProvider"])

    fn = METRICS[m.role]
    acc: dict[str, list[float]] = {}
    for x in images:
        got = fn(run(ref_sess, x), run(test_sess, x))
        for k, v in got.items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def check(role: str, vals: dict[str, float]) -> list[str]:
    failures = []
    for rule, limit in THRESHOLDS.get(role, {}).items():
        kind, key = rule.split("_", 1)
        got = vals.get(key)
        if got is None:
            continue
        if kind == "max" and got > limit:
            failures.append(f"{key} = {got:.4f} > 上限 {limit}")
        if kind == "min" and got < limit:
            failures.append(f"{key} = {got:.4f} < 下限 {limit}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description="量子化前後の精度を測る")
    ap.add_argument("--raw", type=Path, help="元モデル（fp32）のディレクトリ")
    ap.add_argument("--models", type=Path, help="量子化済みモデルのディレクトリ")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "HEAD"),
                    help="2つの量子化結果ディレクトリを比較する（PR 差分レポート用）")
    ap.add_argument("--calibration", type=Path, default=Path("models/calibration"))
    ap.add_argument("--report", type=Path, help="Markdown レポートの出力先")
    ap.add_argument("--fail-under-threshold", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="使う較正画像の枚数上限")
    args = ap.parse_args()

    reg = Registry.load()
    rows: list[dict[str, Any]] = []
    failed = False

    if args.compare:
        base_dir, head_dir = args.compare
        for m in reg.models_for_profile():
            b = base_dir / f"{m.id}.{m.quant_mode}.onnx"
            h = head_dir / f"{m.id}.{m.quant_mode}.onnx"
            if not (b.exists() and h.exists()):
                continue
            imgs = load_images(args.calibration, m.input_size, args.limit)
            if not imgs:
                continue
            vals = evaluate(m, b, h, imgs)
            rows.append({"id": m.id, "role": m.role, "kind": "base→head", "metrics": vals,
                         "baseBytes": b.stat().st_size, "headBytes": h.stat().st_size})
    else:
        if not (args.raw and args.models):
            ap.error("--raw と --models、または --compare を指定してください")
        for m in reg.models_for_profile():
            ref = args.raw / m.id / m.raw["hf"]["file"]
            test = args.models / f"{m.id}.{m.quant_mode}.onnx"
            if not (ref.exists() and test.exists()):
                print(f"[calibrate] スキップ {m.id}（モデルが見つかりません）")
                continue
            imgs = load_images(args.calibration, m.input_size, args.limit)
            if not imgs:
                print(f"[calibrate] スキップ {m.id}（較正画像がありません）")
                continue
            vals = evaluate(m, ref, test, imgs)
            fails = check(m.role, vals)
            rows.append({"id": m.id, "role": m.role, "kind": f"fp32→{m.quant_mode}",
                         "metrics": vals, "failures": fails,
                         "bytes": test.stat().st_size})
            status = "OK" if not fails else "NG"
            print(f"[calibrate] {m.id} [{status}] " +
                  " ".join(f"{k}={v:.4f}" for k, v in vals.items()))
            for f in fails:
                print(f"            {f}", file=sys.stderr)
            failed = failed or bool(fails)

    if args.report:
        lines = ["## 量子化の精度レポート", ""]
        if not rows:
            lines.append("_比較対象がありませんでした（較正画像またはモデルが不足）。_")
        else:
            lines += ["| モデル | 役割 | 比較 | 指標 | サイズ |", "|---|---|---|---|---|"]
            for r in rows:
                mt = " / ".join(f"`{k}` {v:.4f}" for k, v in r["metrics"].items())
                if "bytes" in r:
                    size = f"{r['bytes']/1e6:.1f} MB"
                else:
                    d = (r["headBytes"] - r["baseBytes"]) / 1e6
                    size = f"{r['headBytes']/1e6:.1f} MB ({d:+.1f})"
                mark = "" if not r.get("failures") else " ⚠️"
                lines.append(f"| `{r['id']}`{mark} | {r['role']} | {r['kind']} | {mt} | {size} |")
            bad = [f"- `{r['id']}`: {f}" for r in rows for f in r.get("failures", [])]
            if bad:
                lines += ["", "### 閾値を割った項目", *bad]
        lines += ["", "_閾値は `scripts/calibrate.py` の `THRESHOLDS` に定義。_"]
        args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[calibrate] レポート: {args.report}")

    return 1 if (failed and args.fail_under_threshold) else 0


if __name__ == "__main__":
    sys.exit(main())
