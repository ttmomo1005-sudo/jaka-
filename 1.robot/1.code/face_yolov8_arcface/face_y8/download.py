"""下载 lindevs YOLOv8-Face 权重到 ASCII 缓存目录。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .config import YOLO_FACE_URLS, ensure_dirs


def yolo_weight_path(cfg: dict[str, Any]) -> Path:
    cache, _ = ensure_dirs(cfg)
    size = str(cfg.get("model", {}).get("yolo_size", "l")).lower()
    if size not in YOLO_FACE_URLS:
        raise ValueError(f"未知 yolo_size={size}，可选: {list(YOLO_FACE_URLS)}")
    return cache / f"yolov8{size}-face-lindevs.pt"


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {url}")
    print(f"[download] -> {dest}")
    req = Request(url, headers={"User-Agent": "jaka-face-y8/1.0"})
    with urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        total = resp.headers.get("Content-Length")
        total_i = int(total) if total and total.isdigit() else None
        done = 0
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            done += len(buf)
            if total_i:
                pct = 100.0 * done / total_i
                print(f"\r[download] {pct:5.1f}% ({done}/{total_i})", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def ensure_yolo_weights(cfg: dict[str, Any]) -> Path:
    size = str(cfg.get("model", {}).get("yolo_size", "l")).lower()
    path = yolo_weight_path(cfg)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    return download_file(YOLO_FACE_URLS[size], path)
