#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
站岗视觉 · 无真机调试入口

用法（建议 WSL）:
  cd <repository>/1.robot/1.code/perception
  source ~/jaka_venv/bin/activate   # 或本目录独立 venv
  pip install -r requirements.txt
  python run_detect.py --source webcam
  python run_detect.py --source video --video samples/guard_demo.mp4
  python run_detect.py --source image --image samples/person.jpg --no-window

输出:
  - 控制台 JSON（PersonEvent）
  - runtime/person_event.json（给语音/决策组先读文件联调）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

from detector import PersonDetector
from presence import PresenceFilter
from sources import build_source


ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_event(frame, event) -> None:
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
    if event.person_detected and event.bbox is not None:
        b = event.bbox
        color = (0, 200, 0) if event.direction == "center" else (0, 165, 255)
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        cv2.circle(frame, (int(b.cx), int(b.cy)), 4, color, -1)
    label = (
        f"det={event.person_detected} dir={event.direction} "
        f"ox={event.offset_x:+.2f} hint={event.trigger_hint} "
        f"conf={event.confidence:.2f}"
    )
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)


def write_event_file(path: Path, event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(event.to_json() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JAKA guard YOLO person azimuth demo")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--source", choices=["webcam", "video", "image", "ros"], default=None)
    p.add_argument("--video", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--webcam-index", type=int, default=None)
    p.add_argument("--weights", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help=">0 时跑完即退，便于冒烟")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))

    if args.source:
        cfg.setdefault("source", {})["type"] = args.source
    if args.video:
        cfg.setdefault("source", {})["video_path"] = args.video
    if args.image:
        cfg.setdefault("source", {})["image_path"] = args.image
    if args.webcam_index is not None:
        cfg.setdefault("source", {})["webcam_index"] = args.webcam_index
    if args.weights:
        cfg.setdefault("model", {})["weights"] = args.weights
    if args.device:
        cfg.setdefault("model", {})["device"] = args.device
    if args.no_window:
        cfg.setdefault("output", {})["show_window"] = False

    model_cfg = cfg.get("model", {})
    az_cfg = cfg.get("azimuth", {})
    out_cfg = cfg.get("output", {})

    detector = PersonDetector(
        weights=str(model_cfg.get("weights", "yolov8n.pt")),
        conf=float(model_cfg.get("conf", 0.45)),
        iou=float(model_cfg.get("iou", 0.5)),
        person_class_id=int(model_cfg.get("person_class_id", 0)),
        device=str(model_cfg.get("device", "cpu")),
    )
    filt = PresenceFilter(
        confirm_frames=int(az_cfg.get("confirm_frames", 3)),
        lost_frames=int(az_cfg.get("lost_frames", 8)),
        deadzone=float(az_cfg.get("deadzone", 0.12)),
    )

    source = build_source(cfg)
    event_path = ROOT / str(out_cfg.get("event_file", "runtime/person_event.json"))
    show = bool(out_cfg.get("show_window", True))
    print_json = bool(out_cfg.get("print_json", True))

    print("[perception] source =", cfg.get("source", {}).get("type"))
    print("[perception] event_file =", event_path)
    print("[perception] q / ESC 退出窗口")

    last_hint = None
    n = 0
    try:
        for frame in source.frames():
            n += 1
            h, w = frame.shape[:2]
            primary = detector.detect_primary(frame)
            event = filt.update(primary, w, h)

            if print_json and event.trigger_hint != last_hint:
                print(event.to_json())
                last_hint = event.trigger_hint
            elif print_json and event.trigger_hint == "tracking" and n % 15 == 0:
                print(event.to_json())

            write_event_file(event_path, event)

            if show:
                vis = frame.copy()
                draw_event(vis, event)
                cv2.imshow("guard_yolo", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if args.max_frames > 0 and n >= args.max_frames:
                break
    finally:
        source.release()
        cv2.destroyAllWindows()

    print(f"[perception] done frames={n}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NotImplementedError as e:
        print("[perception]", e, file=sys.stderr)
        raise SystemExit(2)
    except FileNotFoundError as e:
        print("[perception]", e, file=sys.stderr)
        raise SystemExit(2)
