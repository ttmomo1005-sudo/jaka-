"""单张录入：高质量特征 + 轻量图像增强（无需拍几十张）。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detector import FaceDet
from .embedder import ArcFaceEmbedder


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def crop_face(frame_bgr: np.ndarray, det: FaceDet, pad: float = 0.22) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    px, py = int(det.w * pad), int(det.h * pad)
    x1 = max(0, det.x1 - px)
    y1 = max(0, det.y1 - py)
    x2 = min(w, det.x2 + px)
    y2 = min(h, det.y2 + py)
    return frame_bgr[y1:y2, x1:x2].copy()


def _augment_crops(crop: np.ndarray) -> List[np.ndarray]:
    """由一张脸生成若干变体，模拟多角度/光照，提升单样本鲁棒性。"""
    out = [crop]
    out.append(cv2.flip(crop, 1))
    bright = cv2.convertScaleAbs(crop, alpha=1.15, beta=12)
    dark = cv2.convertScaleAbs(crop, alpha=0.88, beta=-8)
    out.append(bright)
    out.append(dark)
    out.append(cv2.flip(bright, 1))
    # 轻微缩放中心裁切
    h, w = crop.shape[:2]
    if h > 20 and w > 20:
        y0, x0 = int(h * 0.05), int(w * 0.05)
        zoom = crop[y0 : h - y0, x0 : w - x0]
        if zoom.size:
            out.append(cv2.resize(zoom, (w, h), interpolation=cv2.INTER_LINEAR))
    return out


def embeddings_from_one_shot(
    embedder: ArcFaceEmbedder,
    frame_bgr: np.ndarray,
    det: FaceDet,
    augment: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    返回 (embeddings列表, 缩略图)。
    主特征用对齐路径；增强图用快速路径，避免录入瞬间卡太久。
    """
    crop = crop_face(frame_bgr, det)
    if crop.size == 0:
        return [], crop

    embeddings: List[np.ndarray] = []
    main = embedder.embed_from_det(frame_bgr, det, force_align=True)
    if main is not None:
        embeddings.append(main)

    if augment:
        for var in _augment_crops(crop):
            # 增强图也走对齐，避免把未对齐特征混进库导致分数偏低、认错人
            emb = embedder.embed_crop(var, force_align=True)
            if emb is None:
                continue
            if main is not None and float(np.dot(main, emb)) < 0.35:
                continue
            embeddings.append(emb)

    # 去重：几乎相同的向量只留一个
    uniq: List[np.ndarray] = []
    for e in embeddings:
        if all(float(np.dot(e, u)) < 0.995 for u in uniq):
            uniq.append(e)
    if not uniq and main is not None:
        uniq = [main]
    return uniq, crop


def mean_embedding(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    if not embeddings:
        return None
    mat = np.stack(embeddings, axis=0).astype(np.float32)
    mean = mat.mean(axis=0)
    return _l2_normalize(mean)
