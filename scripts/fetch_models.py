#!/usr/bin/env python3
"""registry.json に従って HuggingFace から元モデルを取得する。

決定 D7 により、モデルバイナリは git にコミットしない。CI がこれを実行して
取得し、量子化した成果物だけを Pages にデプロイする。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

from _registry import Registry


def main() -> int:
    ap = argparse.ArgumentParser(description="registry.json のモデルを HF から取得する")
    ap.add_argument("--out", type=Path, required=True, help="取得先ディレクトリ")
    ap.add_argument("--profile", default=None, help="プロファイル名（既定: registry の defaultProfile）")
    ap.add_argument("--all", action="store_true", help="プロファイルに関係なく全モデルを取得")
    ap.add_argument("--deployed-only", action="store_true",
                    help="Pages に置くモデルだけ取得する（registry の deploy）")
    args = ap.parse_args()

    reg = Registry.load()
    models = (
        list(reg.models.values())
        if args.all
        else reg.models_for_profile(args.profile, deployed_only=args.deployed_only)
    )

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for m in models:
        dest_dir = args.out / m.id
        dest_dir.mkdir(parents=True, exist_ok=True)

        # HF ではなく直接 URL から取るモデル（u2netp）
        if m.url:
            dest = m.raw_path(args.out)
            print(f"[fetch] {m.id}: {m.url}", flush=True)
            urllib.request.urlretrieve(m.url, dest)  # noqa: S310 - registry の固定 URL
            data = dest.read_bytes()
            got = hashlib.sha256(data).hexdigest()
            if m.sha256 and got != m.sha256:
                print(f"[fetch] {m.id}: SHA256 が違う\n  期待 {m.sha256}\n  実際 {got}", file=sys.stderr)
                return 1
            total += len(data)
            print(f"        -> {len(data) / 1e6:.1f} MB (sha256 一致)", flush=True)
            continue

        # 公開済みの量子化版があればそれも落としておく（CI での再量子化を省ける）
        wanted = list(m.hf_files())
        pre = m.prequantized(m.quant_mode)
        if pre:
            wanted.append(pre)

        for rel in wanted:
            print(f"[fetch] {m.id}: {m.hf_repo}/{rel}", flush=True)
            path = hf_hub_download(repo_id=m.hf_repo, filename=rel, local_dir=str(dest_dir))
            size = Path(path).stat().st_size
            total += size
            print(f"        -> {size / 1e6:.1f} MB", flush=True)

    print(f"[fetch] 完了: {len(models)} モデル, 合計 {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
