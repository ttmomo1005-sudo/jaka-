from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from .camera import open_camera, read_frame, resize_max_side
from .config import load_config
from .draw_text import put_text_cn
from .gallery import FaceGallery
from .matcher import MatchResult
from .pipeline import FacePipeline


def _open_cam(cfg: dict, camera_index: int):
    cam = cfg.get("camera", {})
    return open_camera(
        camera_index,
        int(cam.get("width", 1280)),
        int(cam.get("height", 720)),
        buffer_size=int(cam.get("buffer_size", 1)),
    )


def _copy_result(r: MatchResult, det) -> MatchResult:
    return MatchResult(
        name=r.name,
        score=r.score,
        accepted=r.accepted,
        det=det,
        second_score=r.second_score,
        pending=r.pending,
        vote_count=r.vote_count,
        vote_need=r.vote_need,
        vote_window_filled=r.vote_window_filled,
        tid=r.tid,
    )


def _draw(frame, results: List[MatchResult], fps: float = 0.0, show_fps: bool = True) -> None:
    for r in results:
        d = r.det
        if r.accepted:
            color = (0, 200, 0)
        elif r.pending:
            color = (0, 180, 220)
        else:
            color = (0, 0, 220)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        if r.accepted:
            label = f"{r.name} {r.score:.2f}"
        elif r.pending and r.name != "unknown":
            label = f"确认中 {r.name} {r.vote_count}/{r.vote_need}"
        else:
            label = f"{r.name} {r.score:.2f}" if r.score > 0 else r.name
        put_text_cn(frame, label, (d.x1, max(24, d.y1 - 8)), color_bgr=color, font_size=22)
    if show_fps and fps > 0:
        cv2.putText(frame, f"{fps:.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)


def check_camera(camera_index: int = 0) -> int:
    cfg = load_config()
    cap = _open_cam(cfg, camera_index)
    ok, frame = read_frame(cap, grab_latest=False)
    cap.release()
    if not ok or frame is None:
        print("[face] camera FAIL")
        return 1
    h, w = frame.shape[:2]
    print(f"[face] OK: camera={camera_index} size={w}x{h}")
    return 0


def list_people() -> int:
    # 只读人脸库，不加载 YOLO/ArcFace，秒开
    gallery = FaceGallery(load_config())
    names = gallery.list_people()
    if not names:
        print("[face] gallery empty")
        return 0
    for n in names:
        rec = gallery.records[n]
        print(f"- {n}  samples={rec.samples}")
    return 0


def delete_person(name: str) -> int:
    gallery = FaceGallery(load_config())
    if gallery.delete(name):
        print(f"[face] deleted {name}")
        return 0
    print(f"[face] not found: {name}")
    return 1


def preview(camera_index: int = 0) -> int:
    cfg = load_config()
    cap = _open_cam(cfg, camera_index)
    print("[face] preview  Q/ESC quit")
    while True:
        ok, frame = read_frame(cap)
        if not ok:
            break
        cv2.imshow("face_preview", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def _scale_det(det, s: float):
    from .detector import FaceDet

    return FaceDet(int(det.x1 * s), int(det.y1 * s), int(det.x2 * s), int(det.y2 * s), det.conf)


def detect(camera_index: int = 0) -> int:
    cfg = load_config()
    perf = cfg.get("performance", {})
    process_max = int(perf.get("process_max_side", 640))
    display_max = int(perf.get("display_max_side", 960))
    detect_every = max(1, int(perf.get("detect_every", 2)))
    show_fps = bool(perf.get("show_fps", True))

    pipe = FacePipeline(cfg, load_detector=True, load_embedder=False)
    cap = _open_cam(cfg, camera_index)
    print("[face] detect (fast)  Q/ESC quit")

    last_dets = []
    frame_i = 0
    fps = 0.0
    fps_t = time.time()
    fps_n = 0
    while True:
        ok, frame = read_frame(cap, grab_latest=True)
        if not ok:
            break
        frame_i += 1
        proc, scale = resize_max_side(frame, process_max)
        if detect_every <= 1 or frame_i % detect_every == 1 or not last_dets:
            last_dets = [r.det for r in pipe.process(proc, identify=False)]

        vis, _ = resize_max_side(frame, display_max)
        s_disp = vis.shape[1] / float(frame.shape[1])
        for d in last_dets:
            dd = _scale_det(d.scaled(scale), s_disp)
            cv2.rectangle(vis, (dd.x1, dd.y1), (dd.x2, dd.y2), (0, 200, 255), 2)
            cv2.putText(vis, f"{d.conf:.2f}", (dd.x1, dd.y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        fps_n += 1
        now = time.time()
        if now - fps_t >= 0.5:
            fps = fps_n / (now - fps_t)
            fps_t = now
            fps_n = 0
        if show_fps:
            cv2.putText(vis, f"{fps:.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)

        cv2.imshow("face_detect_y8", vis)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def enroll(name: str, camera_index: int = 0, samples: Optional[int] = None, image: Optional[str] = None) -> int:
    """
    单张/一键录入（站岗认识新人）：
    - 预览阶段只跑轻量 YOLO，不跑 ArcFace → 流畅
    - 空格 / 脸稳定自动拍 1 张 → 再提特征 + 增强
    - 也可 --image 直接用一张照片录入
    """
    if not name.strip():
        print("[face] --name 必填")
        return 1

    cfg = load_config()
    if image:
        return _enroll_from_image(name, image, cfg)

    # samples 参数保留兼容：>1 时仍只拍 1 次，内部增强
    _ = samples
    enroll_cfg = cfg.get("enroll", {})
    perf = cfg.get("performance", {})
    process_max = int(perf.get("process_max_side", 640))
    display_max = int(perf.get("display_max_side", 960))
    detect_every = max(1, int(perf.get("detect_every", 2)))
    auto_sec = float(enroll_cfg.get("auto_capture_sec", 1.0))
    min_conf = float(enroll_cfg.get("min_det_conf", 0.45))
    min_face = int(enroll_cfg.get("min_face_px", 90))
    do_aug = bool(enroll_cfg.get("augment", True))

    # 预览不加载 embedder；拍下后再加载，启动更快、预览不卡
    pipe = FacePipeline(cfg, load_detector=True, load_embedder=False)
    cap = _open_cam(cfg, camera_index)
    print("[face] 一键录入：正对镜头，按 【空格】拍照；脸稳定也可自动拍")
    print("[face] Q/ESC 取消")

    last_dets = []
    frame_i = 0
    stable_t0: Optional[float] = None
    fps = 0.0
    fps_t = time.time()
    fps_n = 0
    capture_frame = None
    capture_det = None

    while True:
        ok, frame = read_frame(cap, grab_latest=True)
        if not ok:
            break
        frame_i += 1
        proc, scale = resize_max_side(frame, process_max)
        if detect_every <= 1 or frame_i % detect_every == 1 or not last_dets:
            last_dets = [
                d
                for d in pipe.detect(proc)
                if d.conf >= min_conf and min(d.w, d.h) >= max(1, int(min_face * scale))
            ]

        face_ok = bool(last_dets)
        now = time.time()
        if face_ok:
            if stable_t0 is None:
                stable_t0 = now
        else:
            stable_t0 = None

        auto_ready = face_ok and stable_t0 is not None and (now - stable_t0) >= auto_sec
        hint = "SPACE=拍照"
        if face_ok:
            left = max(0.0, auto_sec - (now - (stable_t0 or now)))
            hint = f"SPACE=拍照  自动还有 {left:.1f}s" if not auto_ready else "即将自动拍照..."
        else:
            hint = "请把脸放入画面中央"

        vis, _ = resize_max_side(frame, display_max)
        s_disp = vis.shape[1] / float(frame.shape[1])
        for d in last_dets:
            dd = _scale_det(d.scaled(scale), s_disp)
            color = (0, 220, 0) if face_ok else (0, 165, 255)
            cv2.rectangle(vis, (dd.x1, dd.y1), (dd.x2, dd.y2), color, 2)
        cv2.putText(vis, f"enroll: ", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
        put_text_cn(vis, name, (110, 28), color_bgr=(240, 240, 240), font_size=24)
        put_text_cn(vis, hint, (12, 58), color_bgr=(240, 240, 240), font_size=20)

        fps_n += 1
        if now - fps_t >= 0.5:
            fps = fps_n / (now - fps_t)
            fps_t = now
            fps_n = 0
        cv2.putText(vis, f"{fps:.1f} FPS", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)

        cv2.imshow("face_enroll_y8", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            cap.release()
            cv2.destroyAllWindows()
            print("[face] enroll cancelled")
            return 1

        do_capture = False
        if key == ord(" "):
            if not face_ok:
                print("[face] 未检测到合格人脸，无法拍照")
            else:
                do_capture = True
        elif auto_ready:
            do_capture = True

        if do_capture:
            capture_frame = frame.copy()
            capture_det = last_dets[0].scaled(scale)
            break

    cap.release()
    cv2.destroyAllWindows()
    if capture_frame is None or capture_det is None:
        print("[face] 未拍到人脸")
        return 1

    print("[face] 正在提取特征（仅这一次，稍等）...")
    rec = pipe.enroll_from_frame(capture_frame, name, augment=do_aug)
    if rec is None:
        print("[face] 特征提取失败，请正对镜头重试")
        return 1
    print(f"[face] enrolled {rec.name}  features={rec.samples}")
    return 0


def _enroll_from_image(name: str, image_path: str, cfg: dict) -> int:
    from pathlib import Path

    p = Path(image_path)
    if not p.exists():
        print(f"[face] 图片不存在: {image_path}")
        return 1
    frame = cv2.imread(str(p))
    if frame is None:
        print(f"[face] 无法读取图片: {image_path}")
        return 1

    pipe = FacePipeline(cfg, load_detector=True, load_embedder=True)
    do_aug = bool(cfg.get("enroll", {}).get("augment", True))
    rec = pipe.enroll_from_frame(frame, name, augment=do_aug)
    if rec is None:
        print("[face] 特征提取失败")
        return 1
    print(f"[face] enrolled {rec.name} from image, features={rec.samples}")
    return 0


def recognize(camera_index: int = 0, threshold: Optional[float] = None) -> int:
    cfg = load_config()
    if threshold is not None:
        cfg.setdefault("match", {})["cosine_threshold"] = float(threshold)
    perf = cfg.get("performance", {})
    process_max = int(perf.get("process_max_side", 640))
    display_max = int(perf.get("display_max_side", 960))
    detect_every = max(1, int(perf.get("detect_every", 2)))
    recognize_every = max(1, int(perf.get("recognize_every", 5)))
    show_fps = bool(perf.get("show_fps", True))

    pipe = FacePipeline(cfg, load_detector=True, load_embedder=True)
    if not pipe.gallery.list_people():
        print("[face] gallery empty — 先 enroll")
        return 1
    cap = _open_cam(cfg, camera_index)
    print("[face] recognize (smooth)  Q/ESC quit")
    print(f"[face] detect_every={detect_every} recognize_every={recognize_every} yolo={cfg['model'].get('yolo_size')} pack={cfg['model'].get('insightface_pack')}")

    last_results: List[MatchResult] = []
    frame_i = 0
    fps = 0.0
    fps_t = time.time()
    fps_n = 0

    while True:
        ok, frame = read_frame(cap, grab_latest=True)
        if not ok:
            break
        frame_i += 1
        proc, scale = resize_max_side(frame, process_max)

        do_det = detect_every <= 1 or (frame_i % detect_every == 1) or not last_results
        if do_det:
            last_results = pipe.process(proc, identify=True)
        else:
            last_results = pipe.results_now() or last_results

        vis, _ = resize_max_side(frame, display_max)
        s_disp = vis.shape[1] / float(frame.shape[1])
        draw_list = []
        for r in last_results:
            det_orig = r.det.scaled(scale)
            det_disp = _scale_det(det_orig, s_disp)
            draw_list.append(_copy_result(r, det_disp))

        fps_n += 1
        now = time.time()
        if now - fps_t >= 0.5:
            fps = fps_n / (now - fps_t)
            fps_t = now
            fps_n = 0
        _draw(vis, draw_list, fps=fps, show_fps=show_fps)
        cv2.imshow("face_recognize_y8", vis)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0
