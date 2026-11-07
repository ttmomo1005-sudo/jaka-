"""InsightFace ArcFace 特征提取。"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from .config import resolve_device
from .detector import FaceDet


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


class ArcFaceEmbedder:
    """
    YOLO 给框后提特征。
    - fast_embed=True：crop 直接 112x112 过识别头（快，适合实时）
    - fast_embed=False：crop 内再跑检测对齐（慢，录入时可更稳）
    """

    def __init__(self, cfg: dict[str, Any]):
        from insightface.app import FaceAnalysis

        mcfg = cfg.get("model", {})
        pack = str(mcfg.get("insightface_pack", "buffalo_s"))
        det_size = int(mcfg.get("insightface_det_size", 320))
        device = resolve_device(cfg)
        self.fast_embed = bool(mcfg.get("fast_embed", True))
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        cache = cfg["paths"]["cache_dir"]
        # 只要 detection+recognition，少加载性别年龄等头，启动更快
        self.app = FaceAnalysis(
            name=pack,
            root=str(cache),
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        ctx_id = 0 if device.startswith("cuda") else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self.rec_model = self.app.models.get("recognition")
        print(
            f"[embedder] InsightFace {pack} ready fast_embed={self.fast_embed} providers={providers}"
        )

    def embed_crop(self, crop_bgr: np.ndarray, force_align: bool = False) -> Optional[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        use_fast = self.fast_embed and not force_align and self.rec_model is not None
        if use_fast:
            face = cv2.resize(crop_bgr, (112, 112), interpolation=cv2.INTER_AREA)
            feat = self.rec_model.get_feat(face)
            emb = np.asarray(feat, dtype=np.float32).reshape(-1)
            return _l2_normalize(emb)

        h, w = crop_bgr.shape[:2]
        if min(h, w) < 128:
            scale = 128.0 / max(1, min(h, w))
            crop_bgr = cv2.resize(
                crop_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
            )
        faces = self.app.get(crop_bgr)
        if not faces:
            # 对齐失败时退回快速路径，避免整帧白跑
            if self.rec_model is not None:
                face = cv2.resize(crop_bgr, (112, 112), interpolation=cv2.INTER_AREA)
                feat = self.rec_model.get_feat(face)
                return _l2_normalize(np.asarray(feat, dtype=np.float32).reshape(-1))
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = _l2_normalize(np.asarray(face.embedding, dtype=np.float32))
        else:
            emb = np.asarray(emb, dtype=np.float32)
        return emb

    def embed_from_det(
        self,
        frame_bgr: np.ndarray,
        det: FaceDet,
        pad: float = 0.18,
        force_align: bool = False,
    ) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        bw, bh = det.w, det.h
        px, py = int(bw * pad), int(bh * pad)
        x1 = max(0, det.x1 - px)
        y1 = max(0, det.y1 - py)
        x2 = min(w, det.x2 + px)
        y2 = min(h, det.y2 + py)
        crop = frame_bgr[y1:y2, x1:x2]
        return self.embed_crop(crop, force_align=force_align)
