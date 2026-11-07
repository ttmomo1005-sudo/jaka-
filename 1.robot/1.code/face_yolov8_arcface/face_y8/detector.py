"""YOLOv8-Face 人脸检测（lindevs 预训练权重）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from .config import resolve_device
from .download import ensure_yolo_weights


@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    kps: Optional[np.ndarray] = None

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.w * self.h

    def clip(self, frame_w: int, frame_h: int) -> "FaceDet":
        return FaceDet(
            x1=max(0, min(self.x1, frame_w - 1)),
            y1=max(0, min(self.y1, frame_h - 1)),
            x2=max(0, min(self.x2, frame_w)),
            y2=max(0, min(self.y2, frame_h)),
            conf=self.conf,
            kps=None if self.kps is None else np.asarray(self.kps).copy(),
        )

    def scaled(self, scale: float) -> "FaceDet":
        """把处理图坐标映射回原图/显示图。scale=处理图/原图。"""
        if abs(scale - 1.0) < 1e-6:
            return self
        inv = 1.0 / scale
        kps = None
        if self.kps is not None:
            kps = np.asarray(self.kps, dtype=np.float32) * inv
        return FaceDet(
            x1=int(self.x1 * inv),
            y1=int(self.y1 * inv),
            x2=int(self.x2 * inv),
            y2=int(self.y2 * inv),
            conf=self.conf,
            kps=kps,
        )


class YoloFaceDetector:
    def __init__(self, cfg: dict[str, Any]):
        from ultralytics import YOLO

        mcfg = cfg.get("model", {})
        weights = ensure_yolo_weights(cfg)
        self.model = YOLO(str(weights))
        self.conf = float(mcfg.get("yolo_conf", 0.45))
        self.iou = float(mcfg.get("yolo_iou", 0.5))
        self.imgsz = int(mcfg.get("imgsz", 640))
        self.device = resolve_device(cfg)
        self.half = self.device.startswith("cuda")
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(
            source=dummy,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
            max_det=5,
        )
        print(
            f"[detector] YOLOv8{str(mcfg.get('yolo_size', 'n')).lower()}-face "
            f"device={self.device} half={self.half} ready: {weights.name}"
        )

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDet]:
        h, w = frame_bgr.shape[:2]
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
            max_det=5,
        )
        out: List[FaceDet] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            det = FaceDet(int(x1), int(y1), int(x2), int(y2), float(conf)).clip(w, h)
            if det.w >= 8 and det.h >= 8:
                out.append(det)
        out.sort(key=lambda d: d.area, reverse=True)
        return out
