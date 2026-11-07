# -*- coding: utf-8 -*-
"""站立平衡粗检：质心投影是否落在双足支撑多边形内。

用于网页姿态编辑提醒（不替代真实动力学稳定控制）。
限位来源：官方 Khan_mini.xml joint range（物理硬限位）。
"""
from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import numpy as np

# 官方 MuJoCo XML 物理硬限位（部署顺序 27 轴，踝为 pitch/roll 空间）
DOWN_LIMIT = np.array([
    -1.9198, -0.5236, -2.0071, -0.3490, -0.5236, -0.3490,
    -1.5707, -1.9189, -1.5707, -0.8726, -1.5707, 0.0000,
    -1.9198, -0.5236, -2.0071, -0.3490, -0.5236, -0.3490,
    -1.5707, -1.9189, -1.5707, -0.8726, -1.5707, 0.0000,
    -2.0071, -1.5707, -0.6108,
], dtype=np.float64)
UP_LIMIT = np.array([
    1.2217, 2.3562, 3.5779, 2.2689, 0.6108, 0.3490,
    1.5707, 1.9189, 1.5707, 2.2689, 1.5707, 1.5707,
    1.2217, 2.3562, 3.5779, 2.2689, 0.6108, 0.3490,
    1.5707, 1.9189, 1.5707, 2.2689, 1.5707, 1.5707,
    3.5779, 1.5707, 0.6108,
], dtype=np.float64)

JOINT_NAMES_MJ = [
    "Left_hip_pitch_joint", "Left_hip_roll_joint", "Left_hip_yaw_joint", "Left_knee_joint",
    "Left_drive1_joint", "Left_drive2_joint",
    "Left_shoulder_pitch_joint", "Left_shoulder_roll_joint", "Left_shoulder_yaw_joint",
    "Left_elbow_joint", "Left_wrist_roll_joint", "Left_wrist_yaw_joint",
    "Right_hip_pitch_joint", "Right_hip_roll_joint", "Right_hip_yaw_joint", "Right_knee_joint",
    "Right_drive1_joint", "Right_drive2_joint",
    "Right_shoulder_pitch_joint", "Right_shoulder_roll_joint", "Right_shoulder_yaw_joint",
    "Right_elbow_joint", "Right_wrist_roll_joint", "Right_wrist_yaw_joint",
    "waist_yaw_joint", "Neck_yaw_joint", "Neck_pitch_joint",
]

# 足底半尺寸近似（米）：前后 / 左右
FOOT_HX = 0.10
FOOT_HY = 0.045
# 单脚离地超过该高度视为几乎不支撑（米）
LIFT_EPS = 0.025
# 质心到支撑边界的安全裕度（米）
WARN_MARGIN = 0.025
OK_MARGIN = 0.045


class BalanceChecker:
    def __init__(self, deploy: Path):
        import sys
        import mujoco

        self.mujoco = mujoco
        self.deploy = Path(deploy)
        sys.path.insert(0, str(self.deploy))
        from Khan_mini_mujoco import pitch_roll_to_ab  # noqa: WPS433

        self.pitch_roll_to_ab = pitch_roll_to_ab
        self.m = mujoco.MjModel.from_xml_path(os.path.relpath(self.deploy / "Khan_mini.xml"))
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([
            self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES_MJ
        ])
        self.base_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.l_foot = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_base_Link")
        self.r_foot = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "Right_ankle_base_Link")
        if min(self.base_id, self.l_foot, self.r_foot) < 0:
            raise RuntimeError("balance body ids not found")

    def clip(self, q_pr: np.ndarray) -> np.ndarray:
        q = np.asarray(q_pr, dtype=np.float64).reshape(27)
        return np.minimum(UP_LIMIT, np.maximum(DOWN_LIMIT, q))

    def _apply_pose(self, q_pr: np.ndarray) -> None:
        q0 = self.clip(q_pr).copy()
        a, b = self.pitch_roll_to_ab(q0[4], q0[5])
        q0[4], q0[5] = a, b
        a, b = self.pitch_roll_to_ab(q0[16], q0[17])
        q0[16], q0[17] = -a, -b

        d, m = self.d, self.m
        d.qpos[:] = 0.0
        d.qpos[2] = 0.85
        d.qpos[3] = 1.0
        d.qpos[self.qadr] = q0
        d.qvel[:] = 0.0
        self.mujoco.mj_forward(m, d)

        # 把双足放到地面（最低足底 z=0），底座水平不倾斜
        lz = float(d.xpos[self.l_foot][2])
        rz = float(d.xpos[self.r_foot][2])
        d.qpos[2] -= min(lz, rz)
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qvel[:] = 0.0
        self.mujoco.mj_forward(m, d)

    @staticmethod
    def _foot_corners(cx: float, cy: float) -> np.ndarray:
        return np.array([
            [cx - FOOT_HX, cy - FOOT_HY],
            [cx + FOOT_HX, cy - FOOT_HY],
            [cx + FOOT_HX, cy + FOOT_HY],
            [cx - FOOT_HX, cy + FOOT_HY],
        ], dtype=np.float64)

    @staticmethod
    def _point_in_poly(px: float, py: float, poly: np.ndarray) -> bool:
        # ray casting
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            inter = ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
            )
            if inter:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _dist_to_poly_edge(px: float, py: float, poly: np.ndarray) -> float:
        """到多边形边界的有符号距离：内正外负（近似用最近边）。"""
        best = 1e9
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            ab = b - a
            t = np.clip(np.dot([px, py] - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
            p = a + t * ab
            best = min(best, float(np.hypot(px - p[0], py - p[1])))
        inside = BalanceChecker._point_in_poly(px, py, poly)
        return best if inside else -best

    def evaluate(self, q_pr) -> dict[str, Any]:
        self._apply_pose(np.asarray(q_pr, dtype=np.float64))
        d = self.d
        lf = d.xpos[self.l_foot].copy()
        rf = d.xpos[self.r_foot].copy()
        com = d.subtree_com[self.base_id].copy()

        lz, rz = float(lf[2]), float(rf[2])
        left_down = lz <= LIFT_EPS
        right_down = rz <= LIFT_EPS
        reasons: list[str] = []

        polys = []
        if left_down:
            polys.append(self._foot_corners(float(lf[0]), float(lf[1])))
        else:
            reasons.append("左脚离地较高，几乎不支撑")
        if right_down:
            polys.append(self._foot_corners(float(rf[0]), float(rf[1])))
        else:
            reasons.append("右脚离地较高，几乎不支撑")

        if not polys:
            return {
                "ok": False,
                "level": "danger",
                "message": "双脚都离地 — 站立会失衡摔倒",
                "margin_m": -1.0,
                "com_xy": [float(com[0]), float(com[1])],
                "reasons": reasons,
            }

        # 合并足角点为凸包（简易：对矩形角点做 Jarvis 或排序扫描）
        pts = np.vstack(polys)
        hull = _convex_hull(pts)
        margin = self._dist_to_poly_edge(float(com[0]), float(com[1]), hull)

        if margin < 0:
            level = "danger"
            message = "质心已偏出支撑面 — 该姿态站立会不平衡"
            reasons.append("CoM 投影在双足支撑多边形外")
        elif margin < WARN_MARGIN:
            level = "warn"
            message = "接近失衡边界 — 建议收腿/回正躯干"
            reasons.append("CoM 距支撑边界仅 %.1f cm" % (margin * 100))
        elif margin < OK_MARGIN:
            level = "warn"
            message = "平衡裕度偏小 — 模仿动作时注意重心"
            reasons.append("CoM 裕度 %.1f cm" % (margin * 100))
        else:
            level = "ok"
            message = "站立平衡裕度正常"
            reasons.append("CoM 裕度 %.1f cm" % (margin * 100))

        # 额外：大幅腰转 + 单脚 → 提示
        q = self.clip(np.asarray(q_pr, dtype=np.float64))
        if abs(q[24]) > 0.8 and not (left_down and right_down):
            reasons.append("腰大幅扭转且非双脚支撑，易扭倒")
            if level == "ok":
                level = "warn"
                message = "腰转过大，单脚支撑易失衡"

        return {
            "ok": level == "ok",
            "level": level,
            "message": message,
            "margin_m": float(margin),
            "com_xy": [float(com[0]), float(com[1])],
            "foot_z": {"left": lz, "right": rz},
            "reasons": reasons,
        }


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(points, 6), axis=0)
    if len(pts) <= 2:
        return pts
    # Andrew monotone chain
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull if len(hull) else pts
