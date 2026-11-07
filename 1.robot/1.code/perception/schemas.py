"""视觉组对外事件结构。语音组 / 决策层只依赖这些字段。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional
import json
import time


Direction = Literal["left", "center", "right", "none"]


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class PersonEvent:
    """
    视觉 → 决策/语音 的最小约定。

    offset_x:
      画面水平方位，约 [-1, 1]
      负 = 人在画面左侧（机器人应左转）
      正 = 人在画面右侧（机器人应右转）
      接近 0 = 已基本正对

    trigger_hint:
      vision_enter  — 视野内新出现人（站岗主动打招呼可用）
      tracking      — 持续跟踪中
      lost          — 目标丢失
      none          — 无人
    """

    timestamp: float
    person_detected: bool
    confidence: float = 0.0
    offset_x: float = 0.0
    direction: Direction = "none"
    bbox: Optional[BBox] = None
    frame_width: int = 0
    frame_height: int = 0
    trigger_hint: str = "none"
    # 预留：深度相机联调后填米；无深度时为 None
    distance_m: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def empty(frame_w: int = 0, frame_h: int = 0, hint: str = "none") -> "PersonEvent":
        return PersonEvent(
            timestamp=time.time(),
            person_detected=False,
            frame_width=frame_w,
            frame_height=frame_h,
            trigger_hint=hint,
        )
