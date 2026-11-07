#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸识别入口（站岗版）
- 识别：流畅实时
- 录入：一键拍 1 张（或一张照片），无需拍几十张

示例:
  python face_app.py --check
  python face_app.py --mode enroll --name 示例用户
  python face_app.py --mode enroll --name 访客 --image face.jpg
  python face_app.py --mode recognize
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("ORT_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_y8 import app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8-Face + ArcFace 人脸系统")
    p.add_argument("--mode", choices=["detect", "enroll", "recognize", "preview", "check"], default="detect")
    p.add_argument("--name", default="", help="录入姓名")
    p.add_argument("--samples", type=int, default=None, help="兼容旧参数，一键录入可忽略")
    p.add_argument("--image", default="", help="用一张图片直接录入（机器人/证件照）")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=None, help="余弦相似度阈值，越大越严")
    p.add_argument("--list", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--delete", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        return app.list_people()
    if args.delete:
        return app.delete_person(args.delete)
    if args.check or args.mode == "check":
        return app.check_camera(args.camera)
    if args.mode == "preview":
        return app.preview(args.camera)
    if args.mode == "detect":
        return app.detect(args.camera)
    if args.mode == "enroll":
        return app.enroll(
            args.name,
            camera_index=args.camera,
            samples=args.samples,
            image=args.image or None,
        )
    return app.recognize(args.camera, threshold=args.threshold)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[face] {exc}", file=sys.stderr)
        raise SystemExit(1)
