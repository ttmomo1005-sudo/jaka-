from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"

YOLO_FACE_URLS = {
    "n": "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face-lindevs.pt",
    "s": "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8s-face-lindevs.pt",
    "m": "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8m-face-lindevs.pt",
    "l": "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8l-face-lindevs.pt",
    "x": "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8x-face-lindevs.pt",
}


def resolve_device(cfg: dict[str, Any]) -> str:
    """config 里 device: auto → 有 NVIDIA GPU 用 CUDA，否则 CPU。"""
    raw = str((cfg.get("model") or {}).get("device", "cpu") or "cpu").strip().lower()
    want_cuda = raw in ("auto", "cuda", "cuda:0", "gpu", "0")
    if want_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        if raw == "auto":
            return "cpu"
        print("[config] CUDA 不可用，回退 CPU")
        return "cpu"
    return raw if raw else "cpu"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}

    # 环境变量可覆盖（方便真机/队友机器）
    if os.environ.get("FACE_Y8_CACHE"):
        cfg.setdefault("paths", {})["cache_dir"] = os.environ["FACE_Y8_CACHE"]
    if os.environ.get("FACE_Y8_GALLERY"):
        cfg.setdefault("paths", {})["gallery_dir"] = os.environ["FACE_Y8_GALLERY"]
    if os.environ.get("FACE_Y8_DEVICE"):
        cfg.setdefault("model", {})["device"] = os.environ["FACE_Y8_DEVICE"]
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> tuple[Path, Path]:
    cache = Path(cfg["paths"]["cache_dir"])
    gallery = Path(cfg["paths"]["gallery_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    gallery.mkdir(parents=True, exist_ok=True)
    (gallery / "people").mkdir(parents=True, exist_ok=True)
    return cache, gallery
