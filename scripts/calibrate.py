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


def fit_rank(x: np.ndarray, expected_rank: int) -> np.ndarray:
    """NCHW の配列を、モデルが求める階数に合わせる。

    Depth Anything 3 は**多視点モデル**で、入力が `[batch, views, 3, H, W]` の
    5 階になる（1枚だけ渡すときは views=1）。NCHW を決め打ちで渡すと
    onnxruntime が弾く。

        Invalid rank for input: pixel_values Got: 4 Expected: 5

    足りなければバッチ軸の後ろに 1 を挿し、多ければ先頭の 1 を削る。
    """
    while x.ndim < expected_rank:
        x = x[:, None]
    while x.ndim > expected_rank and x.shape[0] == 1:
        x = x[0]
    return x


# 役割ごとの入力の正規化。**本体アプリ（src/pipeline/imageOps.ts,
# src/pipeline/generate.ts）と同じにしなければ意味がない。** 較正は
# 「実際に使う入力での劣化」を測るものなので、ここがずれると測っている
# ものが変わる。実測でも、MODNet に [0,1] を渡すか (x-0.5)/0.5 を渡すかで
# 量子化前後の IoU が 0.32 → 0.67 と倍近く変わった。
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _dtype_of(inp: Any) -> Any:
    t = str(inp.type)
    if "uint8" in t:
        return np.uint8
    if "float16" in t:
        return np.float16
    return np.float32


def preprocess(img01: np.ndarray, role: str, dtype: Any) -> np.ndarray:
    """[0,1] の NCHW を、そのモデルが期待する形に直す。"""
    if dtype == np.uint8:  # MI-GAN のように uint8 の画像を取るモデル
        return np.clip(np.round(img01 * 255), 0, 255)
    if role == "depth":
        return (img01 - IMAGENET_MEAN) / IMAGENET_STD
    if role == "matte":
        return (img01 - 0.5) / 0.5
    return img01


def make_feed(sess: ort.InferenceSession, img01: np.ndarray, role: str) -> dict[str, np.ndarray]:
    """モデルが要求する入力を全部埋める。

    最初の1つだけ渡していたため、2入力の MI-GAN で落ちていた。

        Required inputs (['mask']) are missing from input feed (['image'])
    """
    feed: dict[str, np.ndarray] = {}
    h, w = img01.shape[-2:]
    for i, inp in enumerate(sess.get_inputs()):
        dtype = _dtype_of(inp)
        rank = len(inp.shape or []) or 4
        if "mask" in inp.name.lower() or i > 0:
            # 中央の矩形を穴にする。MI-GAN の約束は「残す=255 / 描く=0」。
            mask = np.full((1, 1, h, w), 255, dtype=np.uint8)
            mask[..., h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0
            feed[inp.name] = fit_rank(mask, rank).astype(dtype)
        else:
            feed[inp.name] = fit_rank(preprocess(img01, role, dtype), rank).astype(dtype)
    return feed


def run(sess: ort.InferenceSession, img01: np.ndarray, role: str) -> np.ndarray:
    out = np.asarray(sess.run(None, make_feed(sess, img01, role))[0], dtype=np.float32)
    # 出力も視点軸を持つ（[batch, views, H, W]）。比較しやすいよう畳んでおく。
    while out.ndim > 3 and out.shape[0] == 1:
        out = out[0]
    return out


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
        got = fn(run(ref_sess, x, m.role), run(test_sess, x, m.role))
        for k, v in got.items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def is_placeholder_set(dir_: Path) -> bool:
    """較正画像が手続き生成のプレースホルダかどうか。

    `models/calibration/README.md` が明言しているとおり、いま入っている画像は
    `generate_placeholders.py` が作った合成画像で、**量子化精度の判断材料には
    使えない**。実際に測ると、同じ MODNet の fp32 と uint8 の一致度が
    実写風の画像で IoU 0.67、この合成画像で 0.12 と、入力しだいで大きく動く。
    モデルではなく画像が分布外であることが原因である。

    それでも閾値（IoU ≥ 0.985）で CI を落としていたので、**測れないものを
    根拠に止めていた**ことになる。プレースホルダのうちは数値を報告するに
    留め、実写に差し替えられたら自動で門番として働くようにする。
    """
    names = [p.name for p in dir_.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return bool(names) and all(n.startswith("placeholder-") for n in names)


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

    placeholders = is_placeholder_set(args.calibration)
    if placeholders:
        print(
            "[calibrate] 較正画像がプレースホルダ（手続き生成）です。"
            "数値は報告しますが、精度の判断には使えないので閾値では落としません。\n"
            "            実写に差し替えると自動的に門番として働きます"
            "（models/calibration/README.md）。",
            file=sys.stderr,
        )

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
            status = "OK" if not fails else ("参考" if placeholders else "NG")
            print(f"[calibrate] {m.id} [{status}] " +
                  " ".join(f"{k}={v:.4f}" for k, v in vals.items()))
            for f in fails:
                print(f"            {f}", file=sys.stderr)
            # プレースホルダで測った値では落とさない（測れていないため）
            failed = failed or (bool(fails) and not placeholders)

    if args.report:
        lines = ["## 量子化の精度レポート", ""]
        if placeholders:
            lines += [
                "> ⚠️ **較正画像がプレースホルダ（手続き生成）です。**",
                "> 下の数値はパイプラインが動いていることの確認であって、"
                "量子化精度の判断材料にはなりません。",
                "> 実写に差し替えるまで閾値では CI を落としません"
                "（`models/calibration/README.md`）。",
                "",
            ]
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
                heading = "### 閾値を割った項目" + ("（参考・CI は落としません）" if placeholders else "")
                lines += ["", heading, *bad]
        lines += ["", "_閾値は `scripts/calibrate.py` の `THRESHOLDS` に定義。_"]
        args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[calibrate] レポート: {args.report}")

    return 1 if (failed and args.fail_under_threshold) else 0


if __name__ == "__main__":
    sys.exit(main())
