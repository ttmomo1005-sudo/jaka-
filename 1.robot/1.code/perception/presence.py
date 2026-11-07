"""确认帧滤波：抑制误检闪烁，生成 trigger_hint。"""

from __future__ import annotations

import time
from typing import Optional, Tuple

from azimuth import bbox_to_offset_x, offset_to_direction
from schemas import BBox, PersonEvent


class PresenceFilter:
    def __init__(self, confirm_frames: int = 3, lost_frames: int = 8, deadzone: float = 0.12):
        self.confirm_frames = confirm_frames
        self.lost_frames = lost_frames
        self.deadzone = deadzone
        self._hit = 0
        self._miss = 0
        self._active = False

    def update(
        self,
        primary: Optional[Tuple[BBox, float]],
        frame_w: int,
        frame_h: int,
    ) -> PersonEvent:
        if primary is None:
            self._hit = 0
            self._miss += 1
            if self._active and self._miss >= self.lost_frames:
                self._active = False
                return PersonEvent.empty(frame_w, frame_h, hint="lost")
            if not self._active:
                return PersonEvent.empty(frame_w, frame_h, hint="none")
            # 短暂丢帧：保持无人事件但 hint 仍 tracking 的“弱丢失”不发 lost
            return PersonEvent.empty(frame_w, frame_h, hint="tracking")

        bbox, conf = primary
        self._miss = 0
        self._hit += 1
        offset = bbox_to_offset_x(bbox, frame_w)
        direction = offset_to_direction(offset, self.deadzone)

        if not self._active:
            if self._hit < self.confirm_frames:
                return PersonEvent.empty(frame_w, frame_h, hint="none")
            self._active = True
            hint = "vision_enter"
        else:
            hint = "tracking"

        return PersonEvent(
            timestamp=time.time(),
            person_detected=True,
            confidence=conf,
            offset_x=offset,
            direction=direction,
            bbox=bbox,
            frame_width=frame_w,
            frame_height=frame_h,
            trigger_hint=hint,
        )
