"""models/registry.json の読み込みとパス解決を共通化する。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "models" / "registry.json"


@dataclass
class Model:
    id: str
    raw: dict[str, Any]

    @property
    def role(self) -> str:
        return str(self.raw["role"])

    @property
    def title(self) -> str:
        return str(self.raw.get("title", self.id))

    @property
    def license(self) -> str:
        return str(self.raw.get("license", "unknown"))

    @property
    def commercial_use(self) -> bool:
        return bool(self.raw.get("commercialUse", False))

    @property
    def quant_mode(self) -> str:
        return str(self.raw.get("quantization", {}).get("mode", "none"))

    @property
    def expected_bytes(self) -> int:
        return int(self.raw.get("expectedBytes", 0))

    @property
    def size_tolerance(self) -> float:
        return float(self.raw.get("sizeTolerance", 0.25))

    @property
    def input_size(self) -> tuple[int, int]:
        w, h = self.raw.get("inputSize", [512, 512])
        return int(w), int(h)

    def hf_files(self) -> list[str]:
        """HF リポジトリから取得すべきファイル。主ファイルが先頭。"""
        hf = self.raw["hf"]
        return [hf["file"], *hf.get("extraFiles", [])]

    @property
    def hf_repo(self) -> str:
        return str(self.raw["hf"]["repo"])

    def modes_by_backend(self, defaults: dict[str, str]) -> dict[str, str]:
        """バックエンドごとの量子化方式（決定 D22）。

        PoC-1 の実測で、WASM は q4f16 が最遅・uint8 が最速だった
        （MatMulNBits に WASM の速い経路が無い）。WebGPU では逆に q4f16 が
        小さくて速い。したがって同じモデルでも配る形を変える。

        モデル固有の指定（quantization.byBackend）があればそれを、
        無ければレジストリ既定を使う。既定にも無ければ mode をそのまま。
        """
        by = self.raw.get("quantization", {}).get("byBackend", {})
        out: dict[str, str] = {}
        for backend, fallback in defaults.items():
            out[backend] = str(by.get(backend, fallback or self.quant_mode))
        return out

    def prequantized(self, mode: str) -> str | None:
        """公開済みの量子化版があればその HF パス。無ければ None。"""
        v = self.raw.get("prequantized", {}).get(mode)
        return str(v) if v else None

    # --- 配置先 ---
    def raw_path(self, out: Path) -> Path:
        return out / self.id / Path(self.raw["hf"]["file"]).name

    def out_path(self, out: Path) -> Path:
        return out / f"{self.id}.{self.quant_mode}.onnx"


@dataclass
class Registry:
    raw: dict[str, Any]
    models: dict[str, Model] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = REGISTRY_PATH) -> "Registry":
        raw = json.loads(path.read_text(encoding="utf-8"))
        reg = cls(raw=raw)
        reg.models = {k: Model(k, v) for k, v in raw["models"].items()}
        return reg

    @property
    def default_profile(self) -> str:
        return str(self.raw.get("defaultProfile", "permissive"))

    def profile(self, name: str | None = None) -> dict[str, str]:
        p = dict(self.raw["profiles"][name or self.default_profile])
        p.pop("_comment", None)
        return p

    @property
    def backend_defaults(self) -> dict[str, str]:
        """バックエンドごとの既定の量子化方式（決定 D22）。"""
        bq = self.raw.get("backendQuantization", {})
        if isinstance(bq, dict):
            backends = bq.get("backends", [])
            defaults = bq.get("defaults", {})
            return {str(b): str(defaults.get(b, "")) for b in backends}
        # 旧形式（バックエンド名の配列だけ）との互換。方式はモデル側に任せる。
        return {str(b): "" for b in bq}

    def models_for_profile(self, name: str | None = None) -> list[Model]:
        """プロファイルが参照するモデルを重複なく返す。"""
        seen: dict[str, Model] = {}
        for model_id in self.profile(name).values():
            if model_id in self.models:
                seen[model_id] = self.models[model_id]
        return list(seen.values())
