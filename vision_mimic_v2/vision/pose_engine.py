"""
全身骨骼估计

优先：MediaPipe Pose Landmarker Heavy（33 点 + world 3D，适合后续机器人 IK）
回退：Ultralytics YOLOv8l-pose（COCO 17 点，多人体稳、国内下载更容易）

参考开源：
- https://github.com/google-ai-edge/mediapipe
- https://github.com/ultralytics/ultralytics (YOLOv8-pose)
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# MediaPipe Pose 33 关键点官方命名
MP_LANDMARK_NAMES: list[str] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

MP_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# COCO-17（YOLOv8-pose）
COCO_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

DEFAULT_MP_MODEL = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_heavy.task"
MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
DEFAULT_YOLO_POSE = Path(__file__).resolve().parents[1] / "models" / "yolov8l-pose.pt"


def _try_download_mp(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        return True
    try:
        import urllib.request

        print(f"[pose] downloading MediaPipe Heavy -> {path}")
        urllib.request.urlretrieve(MP_MODEL_URL, str(path))
        return path.exists() and path.stat().st_size > 1_000_000
    except Exception as e:
        print(f"[pose] MediaPipe model download failed: {e}")
        return False


class _MediaPipeBackend:
    def __init__(self, model_path: Path, num_poses: int = 2) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        # Native MediaPipe file loading fails on non-ASCII Windows paths.
        base = mp_python.BaseOptions(model_asset_buffer=model_path.read_bytes())
        options = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._ts_ms = 0
        self.name = "mediapipe-pose-heavy"
        self.landmark_count = 33
        self.connections = MP_CONNECTIONS
        print(f"[pose] MediaPipe Heavy ready: {model_path.name}")

    def process(self, bgr: np.ndarray) -> dict[str, Any]:
        import mediapipe as mp

        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms = max(self._ts_ms + 1, int(time.monotonic() * 1000))
        result = self._landmarker.detect_for_video(mp_image, self._ts_ms)

        persons: list[dict[str, Any]] = []
        pose_list = list(result.pose_landmarks or [])
        world_list = list(result.pose_world_landmarks or [])
        for pi, lms in enumerate(pose_list):
            keypoints = []
            xs, ys = [], []
            for i, lm in enumerate(lms):
                x_px = float(lm.x) * w
                y_px = float(lm.y) * h
                vis = float(getattr(lm, "visibility", 0.0) or 0.0)
                presence = float(getattr(lm, "presence", vis) or 0.0)
                name = MP_LANDMARK_NAMES[i] if i < len(MP_LANDMARK_NAMES) else f"pt{i}"
                keypoints.append(
                    {
                        "id": i,
                        "name": name,
                        "x": round(x_px, 2),
                        "y": round(y_px, 2),
                        "z": round(float(getattr(lm, "z", 0.0) or 0.0), 5),
                        "visibility": round(vis, 3),
                        "presence": round(presence, 3),
                    }
                )
                if vis >= 0.3 or presence >= 0.3:
                    xs.append(x_px)
                    ys.append(y_px)

            world_keypoints: list[dict[str, Any]] = []
            if pi < len(world_list):
                for i, lm in enumerate(world_list[pi]):
                    name = MP_LANDMARK_NAMES[i] if i < len(MP_LANDMARK_NAMES) else f"pt{i}"
                    world_keypoints.append(
                        {
                            "id": i,
                            "name": name,
                            "x": round(float(lm.x), 5),
                            "y": round(float(lm.y), 5),
                            "z": round(float(lm.z), 5),
                            "visibility": round(float(getattr(lm, "visibility", 0.0) or 0.0), 3),
                        }
                    )

            bbox = _bbox_from_xy(xs, ys, w, h)
            persons.append(
                {
                    "id": pi + 1,
                    "bbox": bbox,
                    "keypoints": keypoints,
                    "world_keypoints": world_keypoints,
                    "bones": [{"a": a, "b": b} for a, b in MP_CONNECTIONS],
                    "gesture": _guess_gesture(keypoints),
                }
            )
        return {
            "persons": persons,
            "width": w,
            "height": h,
            "backend": self.name,
            "landmark_count": self.landmark_count,
            "connections": [{"a": a, "b": b} for a, b in MP_CONNECTIONS],
        }


class _YoloPoseBackend:
    def __init__(self, weights: Path | None = None) -> None:
        from ultralytics import YOLO

        weights = Path(weights) if weights else DEFAULT_YOLO_POSE
        weights.parent.mkdir(parents=True, exist_ok=True)
        # 若本地没有，ultralytics 会拉 yolov8l-pose.pt；再复制到 cache
        src = str(weights) if weights.exists() else "yolov8l-pose.pt"
        self._model = YOLO(src)
        if not weights.exists():
            try:
                # ultralytics 默认下载到 cwd / weights 名；尽量落到 cache
                import shutil

                for cand in (Path("yolov8l-pose.pt"), Path.home() / "yolov8l-pose.pt"):
                    if cand.exists():
                        shutil.copy2(cand, weights)
                        break
            except Exception:
                pass
        self.name = "yolov8l-pose"
        self.landmark_count = 17
        self.connections = COCO_CONNECTIONS
        print(f"[pose] YOLOv8l-pose ready")

    def process(self, bgr: np.ndarray) -> dict[str, Any]:
        h, w = bgr.shape[:2]
        # imgsz 640：速度/精度折中；conf 略低避免漏检全身
        results = self._model.predict(bgr, verbose=False, conf=0.35, iou=0.5, imgsz=640, max_det=3)
        persons: list[dict[str, Any]] = []
        if not results:
            return {
                "persons": [],
                "width": w,
                "height": h,
                "backend": self.name,
                "landmark_count": 17,
                "connections": [{"a": a, "b": b} for a, b in COCO_CONNECTIONS],
            }
        r0 = results[0]
        boxes = r0.boxes
        kpts = r0.keypoints
        n = 0 if boxes is None else len(boxes)
        for i in range(n):
            box = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
            keypoints = []
            xs, ys = [], []
            if kpts is not None and kpts.data is not None and i < len(kpts.data):
                arr = kpts.data[i].cpu().numpy()  # (17, 2|3)
                for j in range(min(17, arr.shape[0])):
                    x = float(arr[j, 0])
                    y = float(arr[j, 1])
                    vis = float(arr[j, 2]) if arr.shape[1] > 2 else 1.0
                    name = COCO_NAMES[j]
                    keypoints.append(
                        {
                            "id": j,
                            "name": name,
                            "x": round(x, 2),
                            "y": round(y, 2),
                            "z": 0.0,
                            "visibility": round(vis, 3),
                            "presence": round(vis, 3),
                        }
                    )
                    if vis >= 0.3 and x > 1 and y > 1:
                        xs.append(x)
                        ys.append(y)
            bbox = {
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
            }
            persons.append(
                {
                    "id": i + 1,
                    "score": round(conf, 3),
                    "bbox": bbox,
                    "keypoints": keypoints,
                    "world_keypoints": [],
                    "bones": [{"a": a, "b": b} for a, b in COCO_CONNECTIONS],
                    "gesture": _guess_gesture(keypoints),
                }
            )
        return {
            "persons": persons,
            "width": w,
            "height": h,
            "backend": self.name,
            "landmark_count": 17,
            "connections": [{"a": a, "b": b} for a, b in COCO_CONNECTIONS],
        }


def _bbox_from_xy(xs: list[float], ys: list[float], w: int, h: int) -> dict[str, int]:
    if not xs or not ys:
        return {"x1": 0, "y1": 0, "x2": w - 1, "y2": h - 1}
    pad = 12.0
    return {
        "x1": int(max(0.0, min(xs) - pad)),
        "y1": int(max(0.0, min(ys) - pad)),
        "x2": int(min(float(w - 1), max(xs) + pad)),
        "y2": int(min(float(h - 1), max(ys) + pad)),
    }


def _kp_map(keypoints: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {k["name"]: k for k in keypoints}


def _guess_gesture(keypoints: list[dict[str, Any]]) -> dict[str, Any]:
    m = _kp_map(keypoints)
    labels: list[str] = []

    def ok(name: str, thr: float = 0.35) -> bool:
        k = m.get(name)
        return bool(k and max(k.get("visibility", 0), k.get("presence", 0)) >= thr)

    if ok("left_wrist") and ok("left_shoulder") and m["left_wrist"]["y"] < m["left_shoulder"]["y"] - 20:
        labels.append("left_arm_up")
    if ok("right_wrist") and ok("right_shoulder") and m["right_wrist"]["y"] < m["right_shoulder"]["y"] - 20:
        labels.append("right_arm_up")
    if "left_arm_up" in labels and "right_arm_up" in labels:
        labels.append("both_arms_up")
    return {"labels": labels, "primary": labels[-1] if labels else "neutral"}


from jaka_map import filter_pose_result


class PoseEngine:
    """线程安全：优先 MediaPipe Heavy，否则 YOLOv8l-pose；输出 JAKA π 同名 27 关节。"""

    def __init__(self, prefer: str = "auto") -> None:
        self._lock = threading.Lock()
        self._backend: Any = None
        self.backend_name = ""
        self._init_backend(prefer)

    def _init_backend(self, prefer: str) -> None:
        want_mp = prefer in ("auto", "mediapipe", "mp")
        want_yolo = prefer in ("auto", "yolo", "yolov8")
        if want_mp and _try_download_mp(DEFAULT_MP_MODEL):
            try:
                self._backend = _MediaPipeBackend(DEFAULT_MP_MODEL, num_poses=1)
                self.backend_name = self._backend.name
                return
            except Exception as e:
                print(f"[pose] MediaPipe init failed, fallback YOLO: {e}")
        if want_yolo:
            if not DEFAULT_YOLO_POSE.is_file():
                raise RuntimeError("MediaPipe initialization failed and local YOLO fallback model is missing")
            self._backend = _YoloPoseBackend(DEFAULT_YOLO_POSE)
            self.backend_name = self._backend.name
            return
        raise RuntimeError("无法初始化任何姿态后端（MediaPipe / YOLOv8-pose）")

    def process(self, bgr: np.ndarray) -> dict[str, Any]:
        if bgr is None or bgr.size == 0:
            return {
                "persons": [],
                "width": 0,
                "height": 0,
                "backend": self.backend_name,
                "landmark_count": 0,
                "connections": [],
                "schema": "jaka-pi-27dof",
            }
        with self._lock:
            raw = self._backend.process(bgr)
        return filter_pose_result(raw)


_engine: Optional[PoseEngine] = None
_engine_lock = threading.Lock()


def get_pose_engine() -> PoseEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = PoseEngine(prefer="auto")
        return _engine
