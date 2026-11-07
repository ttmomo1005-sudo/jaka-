"""由检测框计算站岗转向方位。"""

from __future__ import annotations

from typing import Optional, Tuple

from schemas import BBox, Direction


def bbox_to_offset_x(bbox: BBox, frame_width: int) -> float:
    """像素中心 → [-1, 1]，画面中心为 0。"""
    if frame_width <= 0:
        return 0.0
    return (bbox.cx / float(frame_width)) * 2.0 - 1.0


def offset_to_direction(offset_x: float, deadzone: float) -> Direction:
    if abs(offset_x) <= deadzone:
        return "center"
    return "left" if offset_x < 0 else "right"


def pick_primary_person(
    boxes: list[Tuple[BBox, float]],
) -> Optional[Tuple[BBox, float]]:
    """多人时优先选面积最大（通常最近/最显著）。"""
    if not boxes:
        return None
    return max(boxes, key=lambda item: item[0].area)
