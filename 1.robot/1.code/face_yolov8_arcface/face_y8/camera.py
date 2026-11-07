from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def open_camera(index: int = 0, width: int = 1280, height: int = 720, buffer_size: int = 1) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 index={index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffer_size)))
    except Exception:
        pass
    return cap


def read_frame(cap: cv2.VideoCapture, grab_latest: bool = True) -> Tuple[bool, Optional[np.ndarray]]:
    """grab_latest=True 时丢掉积压帧，只取最新，显著降低延迟。"""
    if grab_latest:
        # 尽量排空缓冲；不同后端对 grab 支持不一
        for _ in range(2):
            if not cap.grab():
                break
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            ok, frame = cap.read()
    else:
        ok, frame = cap.read()
    if not ok or frame is None:
        return False, None
    return True, frame


def resize_max_side(frame: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """等比缩放，返回 (新图, scale=新/旧)。"""
    if max_side <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return frame, 1.0
    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h * scale)
    out = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, scale
