# -*- coding: utf-8 -*-
"""派仔网页预览 + MuJoCo 跟随。

- 浏览器加载官方 Khan_mini URDF/STL（与仿真同一套外观）
- 网页拖关节 / 滑条 → POST /api/pose → MuJoCo 窗口 PD 跟随
- 注意：Windows 上 MuJoCo viewer 必须跑在主线程
"""
from __future__ import annotations

import json
import mimetypes
import os
import socketserver
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote,parse_qs,urlsplit

import numpy as np

from balance_check import DOWN_LIMIT, UP_LIMIT, BalanceChecker
from pose_filter import PoseFilter
import viewer_window
from session_recorder import SessionRecorder

HOST = "127.0.0.1"
PORT = int(os.environ.get("WEB_PORT", "8788"))
# 视觉骨架服务（face_yolov8_web）；mimic 页经本端口代理，避免 CORS
VISION_POSE_URL = "http://127.0.0.1:%s/api/pose" % os.environ.get("VISION_PORT", "8766")
ROOT = Path(__file__).resolve().parent
MESH_DIRS = [

    ROOT.parent / "deploy" / "meshes",
]
DEPLOY_CANDIDATES = [

    ROOT.parent / "deploy",
]

NUM_ACT = 27
SIM_POLICY = os.environ.get("SIM_POLICY", "official_stand").strip()
if SIM_POLICY not in ("official_stand", "legacy_loco"):
    raise ValueError("SIM_POLICY must be official_stand or legacy_loco")
DEFAULT_POSE = np.array([
    0, 0, 0, 0, 0, 0,
    0, -1.57, 0, 1.57, 0, 0.3,
    0, 0, 0, 0, 0, 0,
    0, -1.57, 0, 1.57, 0, 0.3,
    0, 0, 0,
], dtype=np.float64)

_pose_lock = threading.Lock()
_target = np.minimum(UP_LIMIT, np.maximum(DOWN_LIMIT, DEFAULT_POSE.copy()))
_state = {
    "version": "vision_mimic_v2",
    "mujoco": "starting",
    "viewer": "starting",
    "simulation_lifecycle_version": 1,
    "simulation_run": 0,
    "process_id": os.getpid(),
    "detail": "",
    "updated": 0.0,
    # Official standing policy owns the legs; camera supplies upper references.
    "control_mode": "upper",
    "balance_policy": SIM_POLICY,
    "full_body_allowed": SIM_POLICY == "legacy_loco",
    "bridge_lifecycle_version": 2,
    "waist_locked": True,
    "recording_schema_version": 1,
    "balance": {
        "ok": True,
        "level": "ok",
        "message": "平衡检测未就绪",
        "margin_m": 0.0,
        "reasons": [],
    },
}
_flags = {"quit": False}
_sim_start = threading.Event()
_sim_stop = threading.Event()
_viewer_show = threading.Event()
_vision_lock = threading.Lock()
_vision_filter = PoseFilter(DEFAULT_POSE)
_vision_session = 0
_vision_enabled = False
_bridge_frame = {"seq": 0, "q": None, "valid": [False]*27, "captured_at": 0.0}
_bridge_until = 0.0
_bridge_active = False
_balancer: BalanceChecker | None = None
_bal_lock = threading.Lock()
recorder=SessionRecorder(ROOT.parent/'recordings')


def expire_bridge_lease():
    global _bridge_active, _bridge_until, _vision_enabled, _vision_session
    with _pose_lock:
        if _bridge_active and time.perf_counter() >= _bridge_until:
            _bridge_active = False
            _bridge_until = 0.0
            _vision_enabled = False
            _vision_session += 1
            _bridge_frame.update(q=None,valid=[False]*27,captured_at=0.0)
            _state["bridge_state"] = "lease_expired_holding_requires_manual_restart"


def get_control_mode() -> str:
    with _pose_lock:
        return str(_state.get("control_mode") or "upper")


def request_simulation_start():
    """Request only; MuJoCo/viewer creation stays on the process main thread."""
    with _pose_lock:
        if resolve_deploy() is None or os.environ.get("WEB_ONLY") == "1":
            raise ValueError("当前为网页预览模式，请用 START_V2.cmd 重新启动完整服务")
        if os.environ.get("SIM_HEADLESS") == "1":
            raise ValueError("当前为无窗口测试进程，请重新启动完整服务")
        if _state["mujoco"] in ("starting", "running"):
            viewer_window.set_pinned(True)
            _viewer_show.set()
            return {"ok": True, "state": _state["mujoco"], "reused": True}
        _sim_stop.clear()
        viewer_window.set_pinned(True)
        _viewer_show.set()
        _state.update(mujoco="starting", viewer="starting", detail="正在重新打开 MuJoCo 窗口")
        _sim_start.set()
        return {"ok": True, "state": "starting", "reused": False}


def set_control_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if m in ("upper", "upper_only", "safe"):
        m = "upper"
    elif m in ("full", "all", "whole"):
        m = "full"
    else:
        raise ValueError("control_mode must be 'upper' or 'full'")
    if m == "full" and SIM_POLICY == "official_stand" and os.environ.get("WEB_ONLY") != "1":
        raise ValueError("站立平衡策略已启用：下半身由策略控制，仅接收上身模仿目标")
    with _pose_lock:
        _state["control_mode"] = m
        if m == "upper":
            _state["detail"] = "模式：仅上半身 · 下半身由平衡策略控制"
        else:
            _state["detail"] = "模式：全身跟随网页 · 易摔倒会自动复位"
    print("[Mode]", m, flush=True)
    return m


def resolve_meshes() -> Path:
    for p in MESH_DIRS:
        if p.is_dir() and (p / "base_link.STL").exists():
            return p
    raise SystemExit("未找到派仔 meshes（base_link.STL）。请确认 C:\\Users\\Public\\JAKA_Dev\\deploy\\meshes")


def resolve_deploy() -> Path | None:
    for p in DEPLOY_CANDIDATES:
        if (p / "Khan_mini.xml").is_file() and (p / "Khan_mini_config.yaml").is_file():
            return p
    return None


MESH_ROOT = resolve_meshes()


def clip_pose(arr) -> np.ndarray:
    q = np.asarray(arr, dtype=np.float64).reshape(-1)
    if q.size != NUM_ACT:
        raise ValueError("need 27 joint values")
    if not np.isfinite(q).all():
        raise ValueError("joint values must be finite")
    return np.minimum(UP_LIMIT, np.maximum(DOWN_LIMIT, q))

def mimic_pose(arr) -> np.ndarray:
    q = clip_pose(arr)
    # This iteration follows the user's fixed-waist policy in both mimic modes.
    q[24] = DEFAULT_POSE[24]
    return q


def _refresh_balance(q: np.ndarray) -> None:
    if _balancer is None:
        return
    try:
        with _bal_lock:
            bal = _balancer.evaluate(q)
        with _pose_lock:
            _state["balance"] = bal
    except Exception as e:
        with _pose_lock:
            _state["balance"] = {
                "ok": False,
                "level": "warn",
                "message": "平衡检测失败: %s" % e,
                "margin_m": 0.0,
                "reasons": [str(e)],
            }


def set_target(arr) -> None:
    q = mimic_pose(arr)
    with _pose_lock:
        _target[:] = q
        _state["updated"] = time.time()
    _refresh_balance(q)


def get_target() -> list[float]:
    with _pose_lock:
        return _target.tolist()


def get_limits_payload() -> dict:
    return {
        "source": "Khan_mini.xml physical joint range",
        "dof": NUM_ACT,
        "down": DOWN_LIMIT.tolist(),
        "up": UP_LIMIT.tolist(),
        "unit": "rad",
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        if path.startswith('/analysis/'):
            base=(ROOT.parent/'reports/recording_analysis').resolve()
            candidate=(base/path[len('/analysis/'):]).resolve()
            if not candidate.is_relative_to(base) or candidate.suffix!='.html' or not candidate.is_file():
                self.send_error(404);return
            body=candidate.read_bytes();self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body);return
        if path == '/api/recording/status':
            self.reply_json(200,{**recorder.status(),'files':recorder.files()})
            return
        if path == '/api/recording/file':
            try:
                name=parse_qs(urlsplit(self.path).query).get('id',[''])[0]
                body=recorder.file(name).read_bytes()
            except (ValueError,OSError) as error:
                self.reply_json(400,{'error':str(error)})
                return
            self.send_response(200);self.send_header('Content-Type','application/x-ndjson; charset=utf-8')
            self.send_header('Content-Disposition','attachment; filename="recording.jsonl"')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            return
        if path == "/api/bridge/frame":
            with _pose_lock:
                frame = dict(_bridge_frame)
                captured = frame.pop("captured_at")
                frame.update(enabled=_vision_enabled, session=_vision_session,
                    age_ms=max(0., (time.perf_counter()-captured)*1000) if captured else None)
            self.reply_json(200, frame)
            return
        if path == "/api/pose":
            with _pose_lock:
                bal = dict(_state.get("balance") or {})
            body = json.dumps({
                "q": get_target(),
                "mujoco": _state["mujoco"],
                "detail": _state["detail"],
                "updated": _state["updated"],
                "balance": bal,
                "limits": get_limits_payload(),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/limits":
            body = json.dumps(get_limits_payload()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/balance":
            with _pose_lock:
                bal = dict(_state.get("balance") or {})
                q = _target.tolist()
            body = json.dumps({"q": q, "balance": bal}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/mode":
            body = json.dumps({
                "control_mode": get_control_mode(),
                "modes": {
                    "upper": "上身目标跟随，下半身由站立平衡策略控制",
                    "full": "站立策略启用时禁止覆盖腿部" if SIM_POLICY == "official_stand" else "全身跟随网页（易摔倒）",
                },
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/status":
            with _pose_lock:
                status = dict(_state)
            if status.get("viewer") != "headless":
                status["window"] = viewer_window.status()
                if status["mujoco"] == "running" and status["window"]["supported"]:
                    status["viewer"] = "visible" if status["window"]["visible"] else "hidden"
            body = json.dumps(status).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/meshes/"):
            rel = path[len("/meshes/"):]
            target = (MESH_ROOT / rel).resolve()
            if not target.is_relative_to(MESH_ROOT.resolve()) or not target.is_file():
                self.send_error(404, "mesh not found")
                return
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if target.suffix.lower() == ".stl":
                ctype = "application/octet-stream"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def do_POST(self):
        global _vision_session, _vision_enabled, _vision_filter, _bridge_until, _bridge_active
        path = unquote(self.path.split("?", 1)[0])
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n > 0 else b"{}"
        if path == "/api/vision":
            self.handle_vision(raw)
            return
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception as e:
            msg = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        if path == '/api/recording/analyze':
            try:
                if recorder.status()['active']:raise ValueError('先停止记录，再生成分析')
                source=recorder.file(payload.get('id'))
                import sys
                sys.path.insert(0,str(ROOT.parent))
                from recording_tools.analyze import report
                output=report(source)
                self.reply_json(200,dict(ok=True,url='/analysis/'+output.name+'/回放与分析.html'))
            except (ValueError,OSError,KeyError) as error:self.reply_json(400,dict(ok=False,error=str(error)))
            return
        if path in ('/api/recording/start','/api/recording/stop'):
            try:
                if path.endswith('/start'):
                    result=recorder.start(dict(policy=SIM_POLICY,simulation_run=_state['simulation_run'],
                                               session=_vision_session,recording_version=1))
                else:result=recorder.stop()
                self.reply_json(200,dict(ok=True,**result))
            except (ValueError,OSError) as error:self.reply_json(409,dict(ok=False,error=str(error)))
            return
        if path == "/api/simulation/window":
            if type(payload.get("topmost")) is not bool:
                self.reply_json(400, {"ok":False,"error":"topmost must be boolean"})
                return
            viewer_window.set_pinned(payload["topmost"])
            _viewer_show.set()
            self.reply_json(200, {"ok":True,"topmost":payload["topmost"]})
            return
        if path == "/api/mimic":
            with _pose_lock:
                _vision_session += 1
                _vision_enabled = bool(payload.get("enabled"))
                session = _vision_session
                _vision_filter = PoseFilter(_target.copy())
                _bridge_frame.update(q=None, valid=[False]*27, captured_at=0.0)
                _bridge_until = 0.0
                _bridge_active = False
            self.reply_json(200, {"ok": True, "session": session})
            return
        if path == "/api/simulation/start":
            try:
                self.reply_json(200, request_simulation_start())
            except ValueError as e:
                self.reply_json(409, {"ok": False, "error": str(e)})
            return
        if path == "/api/simulation/stop":
            # Stops only the local simulation viewer, never a robot or vision service.
            _sim_stop.set()
            self.reply_json(200, {"ok": True})
            return
        if path == "/api/bridge/pose":
            # Local simulation hook only. The authenticated mock bridge owns its motion limits.
            try:
                if payload.get("release"):
                    with _pose_lock:
                        _bridge_until = 0.0
                        _bridge_active = False
                        _vision_filter = PoseFilter(_target.copy())
                        if payload.get("stop_vision"):
                            _vision_session += 1
                            _vision_enabled = False
                            _bridge_frame.update(q=None,valid=[False]*27,captured_at=0.0)
                else:
                    profile = json.loads((ROOT.parent / "robot_bridge/config/arms8_mock.json").read_text(encoding="utf-8"))
                    q = clip_pose(payload["q"])
                    lo, hi = np.array(profile["demo_lower"]), np.array(profile["demo_upper"])
                    if np.any(q < lo-1e-9) or np.any(q > hi+1e-9):
                        raise ValueError("Outside mock arms-only envelope")
                    with _pose_lock:
                        _target[:] = q
                        _bridge_until = time.perf_counter()+.15
                        _bridge_active = True
                        _state["updated"] = time.time()
                self.reply_json(200, {"ok": True, "simulation_only": True})
            except Exception as e:
                self.reply_json(400, {"ok": False, "error": str(e)})
            return
        if path == "/api/mode":
            try:
                mode = set_control_mode(payload.get("control_mode", payload.get("mode", "")))
                body = json.dumps({"ok": True, "control_mode": mode}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                msg = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            return

        if path != "/api/pose":
            self.send_error(404)
            return
        try:
            q = payload.get("q", payload)
            set_target(q)
            with _pose_lock:
                bal = dict(_state.get("balance") or {})
            body = json.dumps({
                "ok": True,
                "mujoco": _state["mujoco"],
                "q": get_target(),
                "clipped": True,
                "balance": bal,
                "control_mode": get_control_mode(),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            msg = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def reply_json(self, code, payload):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_vision(self, raw):
        expire_bridge_lease()
        if not _vision_lock.acquire(blocking=False):
            self.reply_json(429, {"ok": False, "error": "上一帧仍在处理"})
            return
        try:
            session = int(self.headers.get("X-Mimic-Session", "-1"))
            with _pose_lock:
                active = _vision_enabled and session == _vision_session
            if not active:
                self.reply_json(409, {"ok": False, "error": "模仿已停止或会话过期"})
                return
            if not raw or len(raw) > 5_000_000:
                self.reply_json(400, {"ok": False, "error": "无效图像大小"})
                return
            t0 = time.perf_counter()
            req = urllib.request.Request(VISION_POSE_URL, data=raw,
                headers={"Content-Type": "image/jpeg", "X-Paired-Depth":self.headers.get("X-Paired-Depth","1")}, method="POST")
            with urllib.request.urlopen(req, timeout=12) as response:
                data = json.loads(response.read())
            persons = data.get("persons") or []
            if persons:
                person = persons[0]
                with _pose_lock:
                    # Expiring a protected source never resumes direct camera targets.
                    if not _vision_enabled or session != _vision_session:
                        self.reply_json(409, {"ok": False, "error": "已丢弃停止前的旧帧"})
                        return
                    q = clip_pose(person["robot_q"])
                    person["robot_q_raw"] = q.tolist()
                    _bridge_frame.update(seq=_bridge_frame["seq"]+1, q=q.tolist(),
                        valid=person.get("robot_valid", [False]*27), captured_at=t0)
                    filtered = mimic_pose(_vision_filter.update(q, person.get("robot_valid", [False]*27), time.perf_counter()))
                    if time.perf_counter() >= _bridge_until:
                        _target[:] = filtered
                    _state["updated"] = time.time()
                    person["robot_q"] = filtered.tolist()
                    recorder.emit('vision',dict(source_seq=_bridge_frame['seq'],session=session,
                        raw_person=person.get('recording_landmarks'),geom_q=person.get('geom_q'),
                        mapped_q=q.tolist(),valid=person.get('robot_valid',[False]*27),filtered_q=filtered.tolist(),
                        arm_targets=person.get('arm_targets',{}),arm_fit_error=person.get('arm_fit_error',{}),
                        depth_consistency=person.get('depth_consistency',{}),arm_mapping_version=person.get('arm_mapping_version'),
                        paired_depth=self.headers.get('X-Paired-Depth','1')=='1',
                        proxy_receipt_monotonic=t0,mapping_complete_monotonic=time.perf_counter(),timing_ms=data.get('timing_ms',{})))
                _refresh_balance(filtered)
            else:
                with _pose_lock:
                    if _vision_enabled and session == _vision_session:
                        _bridge_frame.update(seq=_bridge_frame["seq"]+1, q=DEFAULT_POSE.tolist(),
                            valid=[False]*27, captured_at=t0)
                        recorder.emit('vision',dict(source_seq=_bridge_frame['seq'],session=session,raw_person=None,
                            valid=[False]*27,mapped_q=None,filtered_q=_target.tolist(),reason='person_missing'))
            data["timing_ms"]["proxy_total"] = round((time.perf_counter()-t0)*1000,1)
            data["applied"] = bool(persons)
            self.reply_json(200, data)
        except Exception as e:
            self.reply_json(502, {"ok": False, "error": "视觉服务异常: " + str(e)})
        finally:
            _vision_lock.release()

    def log_message(self, fmt, *args):
        if args and any(x in str(args[0]) for x in ("/api/pose", "/api/status", "/api/balance", "/api/mode", "/api/vision", "/api/bridge")):
            return
        print("[%s] %s" % (self.log_date_time_string(), fmt % args), flush=True)


def start_http_server():
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer((HOST, PORT), Handler)
    print("[HTTP] http://%s:%d/" % (HOST, PORT), flush=True)
    httpd.serve_forever()


def open_browser():
    # WEB_JOINT_NO_BROWSER=1 跳过；WEB_JOINT_START_PATH=/mimic.html 开模仿页
    if os.environ.get("WEB_JOINT_NO_BROWSER", "").strip() in ("1", "true", "yes"):
        print("[HTTP] browser open skipped (WEB_JOINT_NO_BROWSER)", flush=True)
        return
    time.sleep(1.2)
    path = os.environ.get("WEB_JOINT_START_PATH", "/").strip() or "/"
    if not path.startswith("/"):
        path = "/" + path
    url = "http://%s:%d%s" % (HOST, PORT, path)
    print("[HTTP] Opening browser:", url, flush=True)
    webbrowser.open(url)


def mujoco_follower(deploy: Path) -> None:
    """Free-base physics; official standing policy receives upper references.

    legacy_loco is retained only for reproducible before/after comparisons.
    """
    import sys
    import yaml
    import mujoco
    import mujoco.viewer

    _state["mujoco"] = "starting"
    _state["detail"] = str(deploy)
    print("[MuJoCo] loading", deploy, flush=True)

    with open(deploy / "Khan_mini_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    sim_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])
    syn_decimation = int(cfg.get("syn_decimation", 40))
    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)
    kps_pr = np.array(cfg["kps_pr"], dtype=np.float32)
    kds_pr = np.array(cfg["kds_pr"], dtype=np.float32)
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)
    init_pos = np.array(cfg["init_pos"], dtype=np.float64)
    ang_vel_scale = float(cfg["ang_vel_scale"])
    dof_pos_scale = float(cfg["dof_pos_scale"])
    dof_vel_scale = float(cfg["dof_vel_scale"])
    action_scale = float(cfg["action_scale"])
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
    num_actions = int(cfg["num_actions"])
    num_obs = int(cfg["num_obs"])
    num_frame = int(cfg["num_frame"])

    # 优先官方默认 locomotion；找不到再扫常见文件名
    policy_candidates = [
        deploy / Path(str(cfg.get("policy_path", "policy_loco.pt"))).name,
        deploy / "policy_loco.pt",
        deploy / "policy_loco_final.pt",
        deploy / "policy_loco_2398.pt",
    ]
    policy_path = next((p for p in policy_candidates if p.is_file()), None)
    if policy_path is None and SIM_POLICY == "legacy_loco":
        raise FileNotFoundError("未找到 locomotion 策略 policy_loco*.pt，无法保持站立平衡")

    set_target(init_pos)

    sys.path.insert(0, str(deploy))
    from Khan_mini_mujoco import pitch_roll_to_ab, ab_to_pitch_roll, pd_control

    m = mujoco.MjModel.from_xml_path(os.path.relpath(deploy / "Khan_mini.xml"))
    m.opt.timestep = sim_dt
    m.opt.gravity[:] = [0.0, 0.0, -9.81]
    d = mujoco.MjData(m)
    jn = [
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
    qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn])
    dadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn])
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    waist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "waist_yaw_Link")
    l_foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_base_Link")
    r_foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Right_ankle_base_Link")
    kps[4:6] = kps_pr[:2]
    kps[16:18] = kps_pr[2:]
    kds[4:6] = kds_pr[:2]
    kds[16:18] = kds_pr[2:]

    # 上半身（臂+颈）可由网页覆盖；腿+腰交给平衡策略
    UPPER_IDS = np.array([6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 25, 26], dtype=int)

    STAND_H = 0.70
    FALL_UPRIGHT = 0.45
    FALL_BASE_Z = 0.22
    FALL_XY = 2.5
    RESET_COOLDOWN = 1.2
    FALL_GRACE = 1.8
    last_reset_t = 0.0
    fall_count = 0

    standing = None
    if SIM_POLICY == "official_stand":
        from standing_policy import StandingPolicy, MODEL_SHA256
        standing = StandingPolicy(default_angles, DOWN_LIMIT, UP_LIMIT)
        _state.update(policy_model="policy_base_v1.onnx", policy_sha256=MODEL_SHA256,
                      policy_period_s=.02, policy_input_size=1650,
                      onnx_runtime=standing.runtime_version,
                      reference_speed_limit_rad_s=2.0, reference_acceleration_limit_rad_s2=4.0)
        print("[MuJoCo] official policy_base_v1 ONNX · 165 x 10 · 50 Hz", flush=True)
    else:
        import io
        import torch
        print("[MuJoCo] baseline balance policy:", policy_path, flush=True)
        policy = torch.jit.load(io.BytesIO(policy_path.read_bytes()), map_location="cpu")
        policy.eval()

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = init_pos.astype(np.float32).copy()
    obs_history = np.zeros((num_obs, num_frame), dtype=np.float32)
    obs_history[:3, 0] = 0
    obs_history[3:6, 0] = np.array([0, 0, -1], dtype=np.float32)
    obs_history[6:9, 0] = 0
    for i in range(num_frame):
        obs_history[:, i] = obs_history[:, 0]

    def _disable_base_weld_only():
        if m.neq <= 0:
            return
        d.eq_active[:] = 1
        d.eq_active[0] = 0

    def set_standing_pose(q_pr, reason: str = ""):
        nonlocal last_reset_t, action, target_dof_pos, obs_history
        q0 = np.asarray(q_pr, dtype=np.float64).copy()
        a, b = pitch_roll_to_ab(q0[4], q0[5])
        q0[4], q0[5] = a, b
        a, b = pitch_roll_to_ab(q0[16], q0[17])
        q0[16], q0[17] = -a, -b

        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        d.ctrl[:] = 0.0
        d.qpos[0:3] = [0.0, 0.0, STAND_H]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[qadr] = q0
        _disable_base_weld_only()
        mujoco.mj_forward(m, d)
        if l_foot >= 0 and r_foot >= 0:
            zmin = min(float(d.xpos[l_foot][2]), float(d.xpos[r_foot][2]))
            d.qpos[2] -= zmin
            d.qpos[2] += 0.002
        d.qvel[:] = 0.0
        _disable_base_weld_only()
        mujoco.mj_forward(m, d)

        action[:] = 0
        target_dof_pos[:] = init_pos.astype(np.float32)
        obs_history[:] = 0
        obs_history[3:6, :] = np.array([0, 0, -1], dtype=np.float32)[:, None]
        if standing is not None:
            standing.reset(q_pr)
        last_reset_t = float(d.time)
        if reason:
            print("[MuJoCo] reset standing:", reason, flush=True)

    def fall_reason() -> str:
        try:
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                return "nonfinite"
            upright = float(d.xmat[base_id].reshape(3, 3)[2, 2])
            bz = float(d.qpos[2])
            bx, by = float(d.qpos[0]), float(d.qpos[1])
            if upright < FALL_UPRIGHT:
                return "tilt"
            if bz < FALL_BASE_Z:
                return "height"
            if abs(bx) > FALL_XY or abs(by) > FALL_XY:
                return "drift"
        except Exception:
            return "state_error"
        return ""

    def _read_pr_state():
        qj = d.qpos[qadr].copy()
        dqj = d.qvel[dadr].copy()
        p, r, dp, dr, J_l = ab_to_pitch_roll(qj[4], qj[5], dqj[4], dqj[5])
        qj[4], qj[5], dqj[4], dqj[5] = p, r, dp, dr
        p, r, dp, dr, J_r = ab_to_pitch_roll(-qj[16], -qj[17], -dqj[16], -dqj[17])
        qj[16], qj[17], dqj[16], dqj[17] = p, r, dp, dr
        return qj, dqj, J_l, J_r

    set_standing_pose(init_pos, "initial")

    J_l = np.zeros((2, 2))
    J_r = np.zeros((2, 2))
    at = np.zeros(2, dtype=np.float32)
    cmd = np.zeros(3, dtype=np.float32)  # 站立：零速度指令
    counter = 0

    headless = os.environ.get("SIM_HEADLESS") == "1"
    realtime = not headless or os.environ.get("SIM_REALTIME", "1") != "0"
    test_frames = None
    if headless and os.environ.get("SIM_TEST_TRAJECTORY"):
        trajectory_path = Path(os.environ["SIM_TEST_TRAJECTORY"]).resolve()
        if not trajectory_path.is_relative_to(ROOT.parent.resolve()):
            raise ValueError("Test trajectory must be inside V2")
        test_frames = json.loads(trajectory_path.read_text(encoding="utf-8"))["frames"]
        test_times = np.array([frame["time"] for frame in test_frames])
        test_q = np.array([clip_pose(frame["q"]) for frame in test_frames])
    metrics = dict(max_displacement_m=0.0, min_upright=1.0, min_base_z=1.0,
                   max_actual_upper_speed_rad_s=0.0, step_errors=0, control_steps=0,
                   tracking_squared_error=0.0, reference_tracking_squared_error=0.0, base_weld_active=False,
                   gravity=[0.0, 0.0, -9.81], parallel_ankle_constraints_active=True)
    metrics["resets_by_reason"] = dict(tilt=0, height=0, drift=0, nonfinite=0, state_error=0)
    inference_ms = []
    samples = []
    if headless:
        class HeadlessViewer:
            def is_running(self):
                return d.time < float(os.environ.get("SIM_TEST_SECONDS", "3"))
            def sync(self): pass
            def close(self): pass
        v = HeadlessViewer()
    else:
        v = mujoco.viewer.launch_passive(m, d)
        _viewer_show.set()
    with _pose_lock:
        _state["mujoco"] = "running"
        _state["viewer"] = "headless" if headless else "visible"
        _state["detail"] = ("官方站立 ONNX · 上身目标参与平衡 · 腰目标固定 · 下半身自动支撑"
                            if standing is not None else "旧 loco + 上身覆盖（对照基线）")
    print("[MuJoCo] balance:", SIM_POLICY, "· free base · fall auto-reset", flush=True)
    try:
        while not _flags["quit"] and not _sim_stop.is_set() and v.is_running():
            t0 = time.time()
            if not headless and _viewer_show.is_set():
                window = viewer_window.show()
                if window.get("visible"):
                    _viewer_show.clear()
            try:
                expire_bridge_lease()
                with _pose_lock:
                    web_q = _target.copy().astype(np.float32)
                if test_frames is not None:
                    web_q = np.array([np.interp(d.time, test_times, test_q[:,i]) for i in range(27)], dtype=np.float32)
                    web_q[24] = 0
                _disable_base_weld_only()

                since_reset = float(d.time) - last_reset_t
                reason = fall_reason() if since_reset >= max(RESET_COOLDOWN, FALL_GRACE) else ""
                if reason:
                    fall_count += 1
                    metrics["resets_by_reason"][reason] += 1
                    set_target(init_pos)
                    set_standing_pose(init_pos, "%s #%d" % (reason, fall_count))
                    _state["detail"] = "摔倒已复位 · %s（累计 %d）" % (SIM_POLICY, fall_count)
                    counter = 0

                qj, dqj, J_l, J_r = _read_pr_state()
                tau = pd_control(target_dof_pos, qj, kps, np.zeros_like(kds), dqj, kds)
                at[0], at[1] = tau[4], tau[5]
                at = J_l.T.copy() @ at
                tau[4], tau[5] = at[0], at[1]
                at[0], at[1] = tau[16], tau[17]
                at = J_r.T.copy() @ at
                tau[16], tau[17] = -at[0], -at[1]
                if not np.isfinite(tau).all():
                    tau = np.zeros_like(tau)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                counter += 1

                # 50Hz：策略推理（cmd=0 站立平衡）
                if counter % control_decimation == 0:
                    mode = get_control_mode()
                    qj, dqj, _, _ = _read_pr_state()
                    R_w = d.xmat[waist_id].reshape(3, 3)
                    omega = d.cvel[waist_id][:3]
                    omega = R_w.T @ omega
                    gravity_orientation = R_w.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

                    inference_start = time.perf_counter()
                    if standing is not None:
                        # No hard upper-body overwrite after inference: the model
                        # sees the references and jointly controls all 27 joints.
                        target_dof_pos = standing.step(qj, dqj, omega, gravity_orientation, web_q)
                    else:
                        qj_obs = (qj - default_angles) * dof_pos_scale
                        dqj_obs = dqj * dof_vel_scale
                        obs_history[:, :-1] = obs_history[:, 1:]
                        obs_history[:3, -1] = omega * ang_vel_scale
                        obs_history[3:6, -1] = gravity_orientation
                        obs_history[6:9, -1] = cmd * cmd_scale
                        obs_history[9:9 + num_actions, -1] = qj_obs
                        obs_history[9 + num_actions:9 + 2 * num_actions, -1] = dqj_obs
                        obs_history[9 + 2 * num_actions:9 + 3 * num_actions, -1] = action
                        obs = obs_history.reshape(-1, order="F")
                        with torch.no_grad():
                            action[:] = policy(torch.from_numpy(obs).unsqueeze(0)).detach().cpu().numpy().squeeze()
                        if mode == "full":
                            target_dof_pos = web_q.copy()
                        else:
                            target_dof_pos = action * action_scale + default_angles
                            target_dof_pos[UPPER_IDS] = web_q[UPPER_IDS]
                    if headless:
                        inference_ms.append((time.perf_counter()-inference_start)*1000)
                        xy = float(np.linalg.norm(d.qpos[:2]))
                        upright = float(d.xmat[base_id].reshape(3,3)[2,2])
                        metrics["max_displacement_m"] = max(metrics["max_displacement_m"], xy)
                        metrics["min_upright"] = min(metrics["min_upright"], upright)
                        metrics["min_base_z"] = min(metrics["min_base_z"], float(d.qpos[2]))
                        metrics["max_actual_upper_speed_rad_s"] = max(metrics["max_actual_upper_speed_rad_s"], float(np.max(np.abs(dqj[UPPER_IDS]))))
                        metrics["control_steps"] += 1
                        metrics["tracking_squared_error"] += float(np.mean((qj[UPPER_IDS]-web_q[UPPER_IDS])**2))
                        bounded_reference = standing.reference.q if standing is not None else web_q
                        metrics["reference_tracking_squared_error"] += float(np.mean((qj[UPPER_IDS]-bounded_reference[UPPER_IDS])**2))
                        metrics["base_weld_active"] |= bool(d.eq_active[0])
                        metrics["parallel_ankle_constraints_active"] &= bool(np.all(d.eq_active[1:]))
                        if counter % 200 == 0:
                            samples.append(dict(time=round(float(d.time),3), xy_m=xy,
                                                upright=upright, falls=fall_count,
                                                arm_q=qj[UPPER_IDS].tolist()))
                    target_dof_pos[4] = np.clip(target_dof_pos[4], -0.52, 0.61)
                    target_dof_pos[5] = np.clip(target_dof_pos[5], -0.35, 0.35)
                    target_dof_pos[16] = np.clip(target_dof_pos[16], -0.52, 0.61)
                    target_dof_pos[17] = np.clip(target_dof_pos[17], -0.35, 0.35)
                    target_dof_pos[24] = init_pos[24]
                    recorder.emit('physics',dict(simulation_run=_state['simulation_run'],simulation_time=float(d.time),
                        source_seq=_bridge_frame['seq'],bridge_active=_bridge_active,mimic_enabled=_vision_enabled,
                        requested_q=web_q.tolist(),bounded_q=(standing.reference.q if standing else web_q).tolist(),
                        policy_target_q=target_dof_pos.tolist(),actual_q=qj.tolist(),actual_dq=dqj.tolist(),
                        fall_count=fall_count,upright=float(d.xmat[base_id].reshape(3,3)[2,2]),
                        base_displacement_m=float(np.linalg.norm(d.qpos[:2]))))

            except Exception as e:
                metrics["step_errors"] += 1
                print("[MuJoCo] step error, reset:", e, flush=True)
                try:
                    set_standing_pose(init_pos, "exception recover")
                    counter = 0
                except Exception as e2:
                    print("[MuJoCo] reset failed:", e2, flush=True)
                    time.sleep(0.05)

            if counter % syn_decimation == 0:
                try:
                    actual_q, _, _, _ = _read_pr_state()
                    _state["actual_q"] = actual_q.tolist()
                    _state["fall_count"] = fall_count
                    _state["base_z"] = float(d.qpos[2])
                    _state["upright"] = float(d.xmat[base_id].reshape(3,3)[2,2])
                    _state["simulation_time"] = float(d.time)
                    _state["base_displacement_m"] = float(np.linalg.norm(d.qpos[:2]))
                    _state["policy_reference_q"] = standing.reference.q.tolist() if standing is not None else web_q.tolist()
                    _state["policy_target_q"] = target_dof_pos.tolist()
                    v.sync()
                except Exception:
                    pass
            rest = sim_dt - (time.time() - t0)
            if realtime and rest > 0:
                time.sleep(rest)
    finally:
        try:
            v.close()
        except Exception:
            pass
        with _pose_lock:
            _state["mujoco"] = "offline"
            _state["viewer"] = "closed"
            _state["detail"] = "MuJoCo 窗口已关闭（再次点击启动快捷方式可重开）"
        print("[MuJoCo] viewer closed", flush=True)
        if headless:
            report = {**_state, "q_finite": bool(np.isfinite(d.qpos).all()), "simulation_time": float(d.time)}
            metrics["tracking_rms_rad"] = float(np.sqrt(metrics.pop("tracking_squared_error")/max(1,metrics["control_steps"])))
            metrics["reference_tracking_rms_rad"] = float(np.sqrt(metrics.pop("reference_tracking_squared_error")/max(1,metrics["control_steps"])))
            metrics["inference_p95_ms"] = float(np.percentile(inference_ms,95)) if inference_ms else None
            report.update(fall_count=fall_count, metrics=metrics, samples=samples)
            report_name = Path(os.environ.get("SIM_REPORT_FILE", "simulation_checks.json")).name
            (ROOT.parent / "reports" / report_name).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    # Keep native model paths ASCII-relative even when the project root is Chinese.
    os.chdir(ROOT.parent)
    if not (ROOT / "index.html").exists():
        raise SystemExit("missing index.html in %s" % ROOT)
    if not (ROOT / "robot.urdf").exists():
        raise SystemExit("missing robot.urdf in %s" % ROOT)

    global _balancer
    deploy = resolve_deploy()
    if deploy is not None:
        try:
            _balancer = BalanceChecker(deploy)
            _refresh_balance(_target.copy())
            print("[Balance] checker ready (CoM vs feet support)", flush=True)
        except Exception as e:
            _balancer = None
            print("[Balance] init failed:", e, flush=True)

    print("=" * 56, flush=True)
    print(" JAKA pi Web + MuJoCo Sync", flush=True)
    print(" URDF:   %s" % (ROOT / "robot.urdf"), flush=True)
    print(" Meshes: %s" % MESH_ROOT, flush=True)
    print(" Deploy: %s" % (deploy or "(none)"), flush=True)
    print(" Limits: Khan_mini.xml physical ranges (hard clip)", flush=True)
    print(" URL:    http://%s:%d/" % (HOST, PORT), flush=True)
    print(" Drag joints in browser → MuJoCo follows", flush=True)
    print(" Close MuJoCo window or Ctrl+C to stop.", flush=True)
    print("=" * 56, flush=True)

    http_th = threading.Thread(target=start_http_server, daemon=True)
    http_th.start()
    threading.Thread(target=open_browser, daemon=True).start()

    if deploy is None or os.environ.get("WEB_ONLY") == "1":
        _state["mujoco"] = "offline"
        _state["viewer"] = "disabled"
        _state["detail"] = "测试预览模式：未运行物理仿真" if os.environ.get("WEB_ONLY") == "1" else "未找到 deploy，仅网页预览"
        print("[WARN] deploy not found — web only; Ctrl+C to stop", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
        return

    _sim_start.set()
    try:
        while not _flags["quit"]:
            if not _sim_start.wait(.25):
                continue
            _sim_start.clear()
            with _pose_lock:
                _state.update(mujoco="starting", viewer="starting", simulation_time=0.0,
                    simulation_run=_state["simulation_run"]+1)
            try:
                mujoco_follower(deploy)
            except Exception as e:
                with _pose_lock:
                    _state.update(mujoco="error", viewer="closed", detail=str(e))
                print("[MuJoCo] ERROR:", e, flush=True)
                if os.environ.get("SIM_HEADLESS") == "1":
                    raise
            if os.environ.get("SIM_HEADLESS") == "1":
                return
            print("[HTTP] MuJoCo stopped; web remains available. Launch shortcut can reopen it.", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        _flags["quit"] = True


if __name__ == "__main__":
    main()
