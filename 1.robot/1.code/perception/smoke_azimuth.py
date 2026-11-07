#!/usr/bin/env python3
"""不依赖摄像头/YOLO 权重的方位与滤波冒烟测试。"""

from __future__ import annotations

from azimuth import bbox_to_offset_x, offset_to_direction
from presence import PresenceFilter
from schemas import BBox


def main() -> None:
    w = 640
    left = BBox(40, 100, 140, 400)
    center = BBox(270, 100, 370, 400)
    right = BBox(500, 100, 600, 400)

    assert offset_to_direction(bbox_to_offset_x(left, w), 0.12) == "left"
    assert offset_to_direction(bbox_to_offset_x(center, w), 0.12) == "center"
    assert offset_to_direction(bbox_to_offset_x(right, w), 0.12) == "right"

    filt = PresenceFilter(confirm_frames=3, lost_frames=3, deadzone=0.12)
    # 前两帧未确认
    e1 = filt.update((left, 0.9), w, 480)
    e2 = filt.update((left, 0.9), w, 480)
    assert e1.person_detected is False and e2.person_detected is False
    e3 = filt.update((left, 0.9), w, 480)
    assert e3.person_detected is True
    assert e3.trigger_hint == "vision_enter"
    assert e3.direction == "left"

    e4 = filt.update((right, 0.8), w, 480)
    assert e4.trigger_hint == "tracking"
    assert e4.direction == "right"

    # 连续丢失
    filt.update(None, w, 480)
    filt.update(None, w, 480)
    e_lost = filt.update(None, w, 480)
    assert e_lost.trigger_hint == "lost"
    assert e_lost.person_detected is False

    print("smoke_azimuth OK")
    print(e3.to_json())


if __name__ == "__main__":
    main()
