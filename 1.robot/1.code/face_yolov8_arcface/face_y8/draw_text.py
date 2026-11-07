"""用 Pillow 在 OpenCV 画面上绘制中文（cv2.putText 不支持中文会变成 ???）。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]


@lru_cache(maxsize=1)
def _font_path() -> str:
    for p in _FONT_CANDIDATES:
        if p.exists():
            return str(p)
    raise FileNotFoundError("未找到中文字体（微软雅黑/黑体等）")


@lru_cache(maxsize=8)
def _pil_font(size: int):
    from PIL import ImageFont

    return ImageFont.truetype(_font_path(), size=size)


def put_text_cn(
    frame_bgr: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color_bgr: Tuple[int, int, int] = (240, 240, 240),
    font_size: int = 22,
) -> None:
    """在 frame 上绘制可含中文的文字。org 为左下角附近基准（与 cv2.putText 习惯接近）。"""
    if not text:
        return
    # 纯 ASCII 且无中文时仍可用更快路径，但统一走 PIL 更稳
    from PIL import Image, ImageDraw

    x, y = int(org[0]), int(org[1])
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _pil_font(font_size)
    # PIL 的 text 锚点用左上；把 y 上移一行高度
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        th = bbox[3] - bbox[1]
    except Exception:
        th = font_size
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text((x, max(0, y - th)), text, font=font, fill=color_rgb)
    frame_bgr[:, :, :] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
