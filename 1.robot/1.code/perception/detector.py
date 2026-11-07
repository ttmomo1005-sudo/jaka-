"""YOLOv8 行人检测封装。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from azimuth import pick_primary_person
from schemas import BBox


class PersonDetector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.45,
        iou: float = 0.5,
        person_class_id: int = 0,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.person_class_id = person_class_id
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[BBox, float]]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False,
        )
        out: List[Tuple[BBox, float]] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            out.append(
                (
                    BBox(float(x1), float(y1), float(x2), float(y2)),
                    float(conf),
                )
            )
        return out

    def detect_primary(self, frame_bgr: np.ndarray) -> Optional[Tuple[BBox, float]]:
        return pick_primary_person(self.detect(frame_bgr))
