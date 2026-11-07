"""
人体姿态 → JAKA π（π仔）27 关节一一对应。

关节名与顺序对齐官方开源仓 JAKA_PI_JOINT_NAMES_DEPLOY：
  左腿6 + 左臂6 + 右腿6 + 右臂6 + 腰1 + 颈2 = 27
每个关节输出：官方名、角度(rad/deg)、屏幕坐标、可见度，供网页显示与后续录制/反解。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# 与 https://github.com/JAKARobotics/JAKA_PI_Humanoid 中 jaka.py 完全一致
JAKA_JOINT_NAMES: list[str] = [
    "Left_hip_pitch_joint",
    "Left_hip_roll_joint",
    "Left_hip_yaw_joint",
    "Left_knee_joint",
    "Left_ankle_pitch_joint",
    "Left_ankle_roll_joint",
    "Left_shoulder_pitch_joint",
    "Left_shoulder_roll_joint",
    "Left_shoulder_yaw_joint",
    "Left_elbow_joint",
    "Left_wrist_roll_joint",
    "Left_wrist_yaw_joint",
    "Right_hip_pitch_joint",
    "Right_hip_roll_joint",
    "Right_hip_yaw_joint",
    "Right_knee_joint",
    "Right_ankle_pitch_joint",
    "Right_ankle_roll_joint",
    "Right_shoulder_pitch_joint",
    "Right_shoulder_roll_joint",
    "Right_shoulder_yaw_joint",
    "Right_elbow_joint",
    "Right_wrist_roll_joint",
    "Right_wrist_yaw_joint",
    "waist_yaw_joint",
    "Neck_yaw_joint",
    "Neck_pitch_joint",
]

assert len(JAKA_JOINT_NAMES) == 27

JAKA_DOF = {
    "robot": "JAKA-pi",
    "total_dof": 27,
    "groups": {
        "left_leg": 6,
        "left_arm": 6,
        "right_leg": 6,
        "right_arm": 6,
        "waist": 1,
        "head": 2,
    },
    "joint_names": JAKA_JOINT_NAMES,
}

# 绘制用骨骼：用「有代表性的」关节索引连成人形链（多自由度共位的关节画同一点）
# 索引对应 JAKA_JOINT_NAMES
JAKA_BONES: list[tuple[int, int]] = [
    (24, 25),  # waist -> neck
    (24, 0),   # waist -> left hip
    (24, 12),  # waist -> right hip
    (24, 6),   # waist -> left shoulder
    (24, 18),  # waist -> right shoulder
    (0, 3), (3, 5),      # left leg
    (12, 15), (15, 17),  # right leg
    (6, 9), (9, 11),     # left arm
    (18, 21), (21, 23),  # right arm
]

# 共位点：同一解剖位置上的多自由度关节，绘制时用同一像素，但角度各自独立
_SITE_OF: dict[str, str] = {
    "Left_hip_pitch_joint": "left_hip",
    "Left_hip_roll_joint": "left_hip",
    "Left_hip_yaw_joint": "left_hip",
    "Left_knee_joint": "left_knee",
    "Left_ankle_pitch_joint": "left_ankle",
    "Left_ankle_roll_joint": "left_ankle",
    "Left_shoulder_pitch_joint": "left_shoulder",
    "Left_shoulder_roll_joint": "left_shoulder",
    "Left_shoulder_yaw_joint": "left_shoulder",
    "Left_elbow_joint": "left_elbow",
    "Left_wrist_roll_joint": "left_wrist",
    "Left_wrist_yaw_joint": "left_wrist",
    "Right_hip_pitch_joint": "right_hip",
    "Right_hip_roll_joint": "right_hip",
    "Right_hip_yaw_joint": "right_hip",
    "Right_knee_joint": "right_knee",
    "Right_ankle_pitch_joint": "right_ankle",
    "Right_ankle_roll_joint": "right_ankle",
    "Right_shoulder_pitch_joint": "right_shoulder",
    "Right_shoulder_roll_joint": "right_shoulder",
    "Right_shoulder_yaw_joint": "right_shoulder",
    "Right_elbow_joint": "right_elbow",
    "Right_wrist_roll_joint": "right_wrist",
    "Right_wrist_yaw_joint": "right_wrist",
    "waist_yaw_joint": "mid_hip",
    "Neck_yaw_joint": "neck",
    "Neck_pitch_joint": "neck",
}


def _ok(p: dict[str, Any] | None, thr: float = 0.25) -> bool:
    if not p:
        return False
    return max(float(p.get("visibility", 0) or 0), float(p.get("presence", 0) or 0)) >= thr


def _xyz(p: dict[str, Any] | None) -> np.ndarray | None:
    if not p:
        return None
    return np.array([float(p["x"]), float(p["y"]), float(p.get("z", 0) or 0)], dtype=np.float64)


def _avg(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    if not _ok(a) or not _ok(b):
        return None
    va = max(float(a.get("visibility", 0) or 0), float(a.get("presence", 0) or 0))
    vb = max(float(b.get("visibility", 0) or 0), float(b.get("presence", 0) or 0))
    return {
        "x": (float(a["x"]) + float(b["x"])) * 0.5,
        "y": (float(a["y"]) + float(b["y"])) * 0.5,
        "z": (float(a.get("z", 0) or 0) + float(b.get("z", 0) or 0)) * 0.5,
        "visibility": min(va, vb),
        "presence": min(va, vb),
    }


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v * 0.0
    return v / n


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """B 处夹角，返回关节弯曲角（伸直≈0）。"""
    u = _norm(a - b)
    v = _norm(c - b)
    return float(math.acos(_clamp(float(np.dot(u, v)))))


def _project_angles_limb(
    parent: np.ndarray,
    joint: np.ndarray,
    child: np.ndarray,
    pelvis_x: np.ndarray,
    pelvis_y: np.ndarray,
    pelvis_z: np.ndarray,
    is_arm: bool,
    left: bool,
) -> tuple[float, float, float]:
    """
    把肢段方向分解为 pitch / roll / yaw（相对骨盆/躯干坐标系）。
    坐标系：x 右、y 上（world 里 MediaPipe 常为 y 下，下面会翻转）、z 前。
    """
    bone = _norm(child - joint)
    # MediaPipe world：大致 +y 向下；翻成 +y 向上便于 pitch 语义
    by = np.array([bone[0], -bone[1], bone[2]], dtype=np.float64)
    by = _norm(by)

    # pitch：前后抬起（相对竖直向下/向侧）
    # 腿：竖直向下为 0；臂：竖直向下为 0，向前抬为正 pitch
    down = np.array([0.0, -1.0, 0.0])  # 在翻后坐标系里「向下」
    if is_arm:
        # 臂默认朝下
        pitch = float(math.atan2(by[2], -by[1]))  # 前(+z)抬起
        roll = float(math.atan2(by[0] if left else -by[0], -by[1]))
    else:
        pitch = float(math.atan2(-by[2], -by[1]))  # 腿前踢
        roll = float(math.atan2(by[0] if left else -by[0], -by[1]))

    # yaw：绕肢段长轴扭转，用 parent→joint→child 平面相对躯干侧向估计
    side = pelvis_x if left else -pelvis_x
    plane_n = _norm(np.cross(parent - joint, child - joint))
    if np.linalg.norm(plane_n) < 1e-6:
        yaw = 0.0
    else:
        yaw = float(math.asin(_clamp(float(np.dot(plane_n, side)))))

    # 限制到合理范围，避免噪声爆炸
    pitch = float(np.clip(pitch, -2.5, 2.5))
    roll = float(np.clip(roll, -1.8, 1.8))
    yaw = float(np.clip(yaw, -1.5, 1.5))
    return pitch, roll, yaw


def _wrist_angles(
    elbow: np.ndarray,
    wrist: np.ndarray,
    hand: np.ndarray | None,
    left: bool,
) -> tuple[float, float]:
    if hand is None:
        return 0.0, 0.0
    forearm = _norm(wrist - elbow)
    hand_dir = _norm(hand - wrist)
    # roll：手相对前臂的侧倾；yaw：左右偏
    # 用简单投影
    axis = forearm
    # 构造前臂局部框
    up = np.array([0.0, -1.0, 0.0])
    x = _norm(np.cross(up, axis))
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    y = _norm(np.cross(axis, x))
    hx = float(np.dot(hand_dir, x))
    hy = float(np.dot(hand_dir, y))
    hz = float(np.dot(hand_dir, axis))
    roll = float(math.atan2(hy, max(1e-6, hz)))
    yaw = float(math.atan2(hx if left else -hx, max(1e-6, hz)))
    return float(np.clip(roll, -1.5, 1.5)), float(np.clip(yaw, -1.5, 1.5))


def _ankle_angles(
    knee: np.ndarray,
    ankle: np.ndarray,
    foot: np.ndarray | None,
    left: bool,
) -> tuple[float, float]:
    if foot is None:
        return 0.0, 0.0
    shin = _norm(ankle - knee)
    foot_dir = _norm(foot - ankle)
    # pitch：脚尖上下；roll：内外翻
    shin_f = np.array([shin[0], -shin[1], shin[2]])
    foot_f = np.array([foot_dir[0], -foot_dir[1], foot_dir[2]])
    shin_f, foot_f = _norm(shin_f), _norm(foot_f)
    pitch = float(math.atan2(foot_f[2], max(1e-6, -foot_f[1])) - math.atan2(shin_f[2], max(1e-6, -shin_f[1])))
    roll = float(math.atan2(foot_f[0] if left else -foot_f[0], max(1e-6, -foot_f[1])))
    return float(np.clip(pitch, -1.2, 1.2)), float(np.clip(roll, -0.8, 0.8))


def _estimate_angles(raw: dict[str, dict[str, Any]], world: dict[str, dict[str, Any]]) -> list[float]:
    """返回长度 27 的关节角（rad），顺序同 JAKA_JOINT_NAMES。缺测为 0。"""
    # 优先 world 3D；否则用像素点（z 用相对深度）
    use_world = len(world) >= 8
    src = world if use_world else {n: {**p, "z": 0.0} for n,p in raw.items()}

    def P(name: str) -> np.ndarray | None:
        p = src.get(name)
        if not _ok(p):
            return None
        return _xyz(p)

    def P_any(*names: str) -> np.ndarray | None:
        for n in names:
            v = P(n)
            if v is not None:
                return v
        return None

    ls, rs = P("left_shoulder"), P("right_shoulder")
    lh, rh = P("left_hip"), P("right_hip")
    lk, rk = P("left_knee"), P("right_knee")
    la, ra = P("left_ankle"), P("right_ankle")
    le, re = P("left_elbow"), P("right_elbow")
    lw, rw = P("left_wrist"), P("right_wrist")
    nose = P("nose")
    l_ear, r_ear = P("left_ear"), P("right_ear")
    l_idx = P_any("left_index", "left_pinky", "left_thumb")
    r_idx = P_any("right_index", "right_pinky", "right_thumb")
    l_foot = P_any("left_foot_index", "left_heel")
    r_foot = P_any("right_foot_index", "right_heel")

    q = [0.0] * 27

    # 骨盆坐标系
    if lh is not None and rh is not None:
        mid_hip = 0.5 * (lh + rh)
        pelvis_x = _norm(rh - lh)  # 右
    else:
        mid_hip = None
        pelvis_x = np.array([1.0, 0.0, 0.0])
    if ls is not None and rs is not None:
        mid_sh = 0.5 * (ls + rs)
        shoulder_x = _norm(rs - ls)
    else:
        mid_sh = None
        shoulder_x = pelvis_x
    pelvis_y = np.array([0.0, -1.0, 0.0])  # MediaPipe 向下
    if mid_hip is not None and mid_sh is not None:
        pelvis_z = _norm(np.cross(pelvis_x, mid_sh - mid_hip))
        pelvis_y = _norm(np.cross(pelvis_z, pelvis_x))
    else:
        pelvis_z = np.array([0.0, 0.0, 1.0])

    # —— 左腿 0-5 ——
    if lh is not None and lk is not None and la is not None and mid_hip is not None:
        pitch, roll, yaw = _project_angles_limb(mid_hip, lh, lk, pelvis_x, pelvis_y, pelvis_z, False, True)
        q[0], q[1], q[2] = pitch, roll, yaw
        q[3] = float(np.clip(math.pi - _angle_3pt(lh, lk, la), 0.0, 2.4))
        ap, ar = _ankle_angles(lk, la, l_foot, True)
        q[4], q[5] = ap, ar

    # —— 左臂 6-11 ——
    if ls is not None and le is not None and lw is not None and mid_sh is not None:
        pitch, roll, yaw = _project_angles_limb(mid_sh, ls, le, shoulder_x, pelvis_y, pelvis_z, True, True)
        q[6], q[7], q[8] = pitch, roll, yaw
        q[9] = float(np.clip(math.pi - _angle_3pt(ls, le, lw), 0.0, 2.6))
        wr, wy = _wrist_angles(le, lw, l_idx, True)
        q[10], q[11] = wr, wy

    # —— 右腿 12-17 ——
    if rh is not None and rk is not None and ra is not None and mid_hip is not None:
        pitch, roll, yaw = _project_angles_limb(mid_hip, rh, rk, pelvis_x, pelvis_y, pelvis_z, False, False)
        q[12], q[13], q[14] = pitch, roll, yaw
        q[15] = float(np.clip(math.pi - _angle_3pt(rh, rk, ra), 0.0, 2.4))
        ap, ar = _ankle_angles(rk, ra, r_foot, False)
        q[16], q[17] = ap, ar

    # —— 右臂 18-23 ——
    if rs is not None and re is not None and rw is not None and mid_sh is not None:
        pitch, roll, yaw = _project_angles_limb(mid_sh, rs, re, shoulder_x, pelvis_y, pelvis_z, True, False)
        q[18], q[19], q[20] = pitch, roll, yaw
        q[21] = float(np.clip(math.pi - _angle_3pt(rs, re, rw), 0.0, 2.6))
        wr, wy = _wrist_angles(re, rw, r_idx, False)
        q[22], q[23] = wr, wy

    # —— 腰 yaw 24：肩线相对髋线 ——
    if lh is not None and rh is not None and ls is not None and rs is not None:
        hip = (rh-lh)[[0,2]]
        shoulder = (rs-ls)[[0,2]]
        # Relative signed angle directly: independent atan2 values straddle +/-pi
        # in a front-facing pose and their raw subtraction jumps by a full turn.
        if min(np.linalg.norm(hip),np.linalg.norm(shoulder)) > 1e-8:
            turn = math.atan2(float(hip[0]*shoulder[1]-hip[1]*shoulder[0]), float(np.dot(hip,shoulder)))
            q[24] = float(np.clip(turn, -1.2, 1.2))

    # —— 颈 25-26 ——
    head = nose
    if head is None and l_ear is not None and r_ear is not None:
        head = 0.5 * (l_ear + r_ear)
    if head is not None and mid_sh is not None:
        neck_dir = head - mid_sh
        nd = np.array([neck_dir[0], -neck_dir[1], neck_dir[2]], dtype=np.float64)
        nd = _norm(nd)
        q[25] = float(np.clip(math.atan2(nd[0], max(1e-6, nd[1])), -1.2, 1.2))  # yaw
        q[26] = float(np.clip(math.atan2(nd[2], max(1e-6, nd[1])), -1.0, 1.0))  # pitch

    return q


def _build_sites(raw: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    mid_hip = _avg(raw.get("left_hip"), raw.get("right_hip"))
    mid_sh = _avg(raw.get("left_shoulder"), raw.get("right_shoulder"))
    neck = None
    if mid_sh and _ok(raw.get("nose")):
        nose = raw["nose"]
        va = max(float(mid_sh.get("visibility", 0) or 0), float(mid_sh.get("presence", 0) or 0))
        vb = max(float(nose.get("visibility", 0) or 0), float(nose.get("presence", 0) or 0))
        neck = {
            "x": float(mid_sh["x"]) * 0.35 + float(nose["x"]) * 0.65,
            "y": float(mid_sh["y"]) * 0.35 + float(nose["y"]) * 0.65,
            "z": (float(mid_sh.get("z", 0) or 0) + float(nose.get("z", 0) or 0)) * 0.5,
            "visibility": min(va, vb),
            "presence": min(va, vb),
        }
    elif mid_sh:
        neck = dict(mid_sh)

    return {
        "left_hip": raw.get("left_hip"),
        "left_knee": raw.get("left_knee"),
        "left_ankle": raw.get("left_ankle"),
        "left_shoulder": raw.get("left_shoulder"),
        "left_elbow": raw.get("left_elbow"),
        "left_wrist": raw.get("left_wrist"),
        "right_hip": raw.get("right_hip"),
        "right_knee": raw.get("right_knee"),
        "right_ankle": raw.get("right_ankle"),
        "right_shoulder": raw.get("right_shoulder"),
        "right_elbow": raw.get("right_elbow"),
        "right_wrist": raw.get("right_wrist"),
        "mid_hip": mid_hip,
        "neck": neck,
    }


def to_jaka_person(person: dict[str, Any]) -> dict[str, Any]:
    raw = {k["name"]: k for k in (person.get("keypoints") or []) if "name" in k}
    world = {k["name"]: k for k in (person.get("world_keypoints") or []) if "name" in k}
    sites = _build_sites(raw)
    angles = _estimate_angles(raw, world)

    joints: list[dict[str, Any]] = []
    for i, name in enumerate(JAKA_JOINT_NAMES):
        site_name = _SITE_OF[name]
        site = sites.get(site_name)
        if _ok(site):
            vis = max(float(site.get("visibility", 0) or 0), float(site.get("presence", 0) or 0))
            x, y, z = float(site["x"]), float(site["y"]), float(site.get("z", 0) or 0)
        else:
            vis, x, y, z = 0.0, 0.0, 0.0, 0.0
        ang = float(angles[i])
        joints.append(
            {
                "id": i,
                "name": name,
                "site": site_name,
                "angle": round(ang, 5),
                "angle_deg": round(ang * 180.0 / math.pi, 2),
                "x": round(x, 2),
                "y": round(y, 2),
                "z": round(z, 5),
                "visibility": round(vis, 3),
                "presence": round(vis, 3),
            }
        )

    # keypoints 与 joints 同序同长度=27，前端按关节画骨架
    keypoints = [
        {
            "id": j["id"],
            "name": j["name"],
            "x": j["x"],
            "y": j["y"],
            "z": j["z"],
            "visibility": j["visibility"],
            "presence": j["presence"],
            "angle": j["angle"],
            "angle_deg": j["angle_deg"],
            "site": j["site"],
        }
        for j in joints
    ]

    xs = [k["x"] for k in keypoints if k["visibility"] >= 0.3]
    ys = [k["y"] for k in keypoints if k["visibility"] >= 0.3]
    if xs and ys:
        pad = 16.0
        bbox = {
            "x1": int(max(0, min(xs) - pad)),
            "y1": int(max(0, min(ys) - pad)),
            "x2": int(max(xs) + pad),
            "y2": int(max(ys) + pad),
        }
    else:
        bbox = person.get("bbox") or {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

    visible = sum(1 for k in keypoints if k["visibility"] >= 0.3)

    mapped = {
        "recording_landmarks": {"id":person.get("id",1),"keypoints":person.get("keypoints") or [],
                                "world_keypoints":person.get("world_keypoints") or []},
        "id": person.get("id", 1),
        "score": person.get("score"),
        "bbox": bbox,
        "keypoints": keypoints,
        "joints": joints,
        "q": [j["angle"] for j in joints],
        "q_deg": [j["angle_deg"] for j in joints],
        "q_names": list(JAKA_JOINT_NAMES),
        "bones": [{"a": a, "b": b} for a, b in JAKA_BONES],
        "gesture": person.get("gesture") or {"labels": [], "primary": "neutral"},
        "jaka": {
            **JAKA_DOF,
            "visible_count": visible,
            "mapped": True,
        },
    }
    # robot_q：MuJoCo/URDF 空间；q 仍为 geom_q 兼容旧页
    try:
        from robot_q import attach_robot_q

        mapped = attach_robot_q(mapped, person, stand_legs=True)
    except Exception:
        mapped["geom_q"] = list(mapped["q"])
        from robot_q import INIT_POS
        mapped["robot_q"] = INIT_POS.tolist()
        mapped["robot_valid"] = [False] * 27
        mapped["q_space"] = "geom"
    return mapped


def filter_pose_result(out: dict[str, Any]) -> dict[str, Any]:
    persons = [to_jaka_person(p) for p in (out.get("persons") or [])]
    return {
        **out,
        "persons": persons,
        "landmark_count": 27,
        "connections": [{"a": a, "b": b} for a, b in JAKA_BONES],
        "jaka": JAKA_DOF,
        "joint_names": JAKA_JOINT_NAMES,
        "schema": "jaka-pi-27dof-joints",
    }
