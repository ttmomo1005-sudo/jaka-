"""InsightFace 引擎：SCRFD 检测 + 关键点对齐 ArcFace。

网页实时不再使用 YOLO。检测和特征来自同一次对齐结果，避免
「框是一套、特征是另一套」导致分数掉到 0.3、一动就 unknown。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .config import resolve_device
from .detector import FaceDet


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


@dataclass
class FaceHit:
    det: FaceDet
    embedding: Optional[np.ndarray] = None


class InsightEngine:
    def __init__(self, cfg: dict[str, Any]):
        from insightface.app import FaceAnalysis

        mcfg = cfg.get("model", {})
        pack = str(mcfg.get("insightface_pack", "buffalo_l"))
        det_size = int(mcfg.get("insightface_det_size", 640))
        det_thresh = float(mcfg.get("insightface_det_thresh", 0.40))
        device = resolve_device(cfg)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        cache = cfg["paths"]["cache_dir"]
        self.app = FaceAnalysis(
            name=pack,
            root=str(cache),
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        ctx_id = 0 if device.startswith("cuda") else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        if hasattr(self.app, "det_thresh"):
            self.app.det_thresh = det_thresh
        self.rec_model = self.app.models.get("recognition")
        self._det_lock = threading.Lock()
        self._rec_lock = threading.Lock()
        print(
            f"[engine] SCRFD+ArcFace {pack} det={det_size} thresh={det_thresh} "
            f"device={device} providers={providers}"
        )

    def analyze(self, frame_bgr: np.ndarray, max_num: int = 2) -> List[FaceHit]:
        """一次调用同时给出框、关键点和对齐后的 ArcFace 向量。"""
        h, w = frame_bgr.shape[:2]
        with self._det_lock:
            with self._rec_lock:
                try:
                    faces = self.app.get(frame_bgr, max_num=max(0, max_num))
                except TypeError:
                    faces = self.app.get(frame_bgr)
        hits: List[FaceHit] = []
        for face in faces or []:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            kps = getattr(face, "kps", None)
            kps_arr = None if kps is None else np.asarray(kps, dtype=np.float32)
            det = FaceDet(x1, y1, x2, y2, float(getattr(face, "det_score", 0.0)), kps=kps_arr).clip(w, h)
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                raw = getattr(face, "embedding", None)
                emb = None if raw is None else _l2_normalize(np.asarray(raw, dtype=np.float32).reshape(-1))
            else:
                emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            if det.w >= 8 and det.h >= 8:
                hits.append(FaceHit(det=det, embedding=emb))
        hits.sort(key=lambda x: x.det.area, reverse=True)
        return hits

    def detect(self, frame_bgr: np.ndarray, max_num: int = 2) -> List[FaceDet]:
        """只跑 SCRFD，用于「只检测框」。"""
        h, w = frame_bgr.shape[:2]
        with self._det_lock:
            bboxes, kpss = self.app.det_model.detect(frame_bgr, max_num=max(0, max_num))
        out: List[FaceDet] = []
        if bboxes is None or len(bboxes) == 0:
            return out
        for i in range(len(bboxes)):
            x1, y1, x2, y2, score = [float(v) for v in bboxes[i][:5]]
            kps = None
            if kpss is not None and i < len(kpss):
                kps = np.asarray(kpss[i], dtype=np.float32)
            det = FaceDet(int(x1), int(y1), int(x2), int(y2), float(score), kps=kps).clip(w, h)
            if det.w >= 8 and det.h >= 8:
                out.append(det)
        out.sort(key=lambda d: d.area, reverse=True)
        return out

    def embed(self, frame_bgr: np.ndarray, det: FaceDet) -> Optional[np.ndarray]:
        """只用已有框和关键点跑 ArcFace，不再重新检测。"""
        from insightface.app.common import Face

        if self.rec_model is None or det.kps is None:
            return None
        bbox = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)
        face = Face(bbox=bbox, kps=np.asarray(det.kps, dtype=np.float32), det_score=float(det.conf))
        with self._rec_lock:
            self.rec_model.get(frame_bgr, face)
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            raw = getattr(face, "embedding", None)
            if raw is None:
                return None
            return _l2_normalize(np.asarray(raw, dtype=np.float32).reshape(-1))
        return np.asarray(emb, dtype=np.float32).reshape(-1)

    def enroll_embeddings(
        self, frame_bgr: np.ndarray, augment: bool = True
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        hits = self.analyze(frame_bgr, max_num=1)
        if not hits or hits[0].embedding is None:
            return [], np.zeros((1, 1, 3), dtype=np.uint8)
        main = hits[0].embedding
        det = hits[0].det
        embeddings = [main]
        if augment:
            variants = [
                cv2.flip(frame_bgr, 1),
                cv2.convertScaleAbs(frame_bgr, alpha=1.12, beta=10),
                cv2.convertScaleAbs(frame_bgr, alpha=0.90, beta=-8),
            ]
            for img in variants:
                extra = self.analyze(img, max_num=1)
                if not extra or extra[0].embedding is None:
                    continue
                emb = extra[0].embedding
                if float(np.dot(main, emb)) >= 0.35:
                    embeddings.append(emb)
        return embeddings, _crop(frame_bgr, det)


def _crop(frame_bgr: np.ndarray, det: FaceDet, pad: float = 0.18) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    px, py = int(det.w * pad), int(det.h * pad)
    x1 = max(0, det.x1 - px)
    y1 = max(0, det.y1 - py)
    x2 = min(w, det.x2 + px)
    y2 = min(h, det.y2 + py)
    crop = frame_bgr[y1:y2, x1:x2]
    return crop.copy() if crop.size else crop
