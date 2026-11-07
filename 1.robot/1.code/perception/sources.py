"""图像源：webcam / 视频 / 图片；ROS 真机接口先占位。"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


class FrameSource:
    def frames(self) -> Generator[np.ndarray, None, None]:
        raise NotImplementedError

    def release(self) -> None:
        pass


class WebcamSource(FrameSource):
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 index={index}")

    def frames(self) -> Generator[np.ndarray, None, None]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        self.cap.release()


class VideoSource(FrameSource):
    def __init__(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"视频不存在: {path}")
        self.cap = cv2.VideoCapture(str(p))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")

    def frames(self) -> Generator[np.ndarray, None, None]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        self.cap.release()


class ImageSource(FrameSource):
    def __init__(self, path: str, repeat: int = 1):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"图片不存在: {path}")
        self.frame = cv2.imread(str(p))
        if self.frame is None:
            raise RuntimeError(f"无法读取图片: {path}")
        self.repeat = max(1, repeat)

    def frames(self) -> Generator[np.ndarray, None, None]:
        for _ in range(self.repeat):
            yield self.frame.copy()


class RosImageSource(FrameSource):
    """真机联调占位：需要 rclpy + cv_bridge 后再实现。"""

    def __init__(self, topic: str):
        raise NotImplementedError(
            f"ROS 源尚未实现。真机联调时订阅 {topic}。"
            "无真机请用 --source webcam|video|image。"
        )


def build_source(cfg: dict) -> FrameSource:
    src = cfg.get("source", {})
    kind = str(src.get("type", "webcam")).lower()
    if kind == "webcam":
        return WebcamSource(int(src.get("webcam_index", 0)))
    if kind == "video":
        return VideoSource(str(src.get("video_path")))
    if kind == "image":
        return ImageSource(str(src.get("image_path")), repeat=60)
    if kind == "ros":
        return RosImageSource(str(src.get("ros_rgb_topic", "/camera/color/image_raw")))
    raise ValueError(f"未知 source.type: {kind}")
