"""网页实时：SCRFD + 对齐 ArcFace + 身份槽。不加载 YOLO。"""

from __future__ import annotations

import threading
from typing import Any, List, Optional

import numpy as np

from .detector import FaceDet
from .engine import InsightEngine
from .gallery import FaceGallery
from .matcher import FaceMatcher, MatchResult
from .tracker import FaceTrack, IoUTracker


class FacePipeline:
    def __init__(self, cfg: dict[str, Any], load_detector: bool = True, load_embedder: bool = True):
        self.cfg = cfg
        self._engine: Optional[InsightEngine] = None
        self.gallery = FaceGallery(cfg)
        m = cfg.get("match", {})
        self.min_face_px = int(m.get("detect_min_face_px", m.get("min_face_px", 28)))
        self.max_faces = int(m.get("max_faces", 2))
        self.switch_margin = float(m.get("switch_margin", 0.10))
        vote_window = int(m.get("vote_window", 2))
        vote_min = int(m.get("vote_min", 1))
        self.matcher = FaceMatcher(
            self.gallery,
            cosine_threshold=float(m.get("cosine_threshold", 0.45)),
            margin=float(m.get("margin", 0.06)),
            vote_window=vote_window,
            vote_min=vote_min,
            lock_score=float(m.get("lock_score", 0.45)),
        )
        tcfg = cfg.get("track", {})
        self.tracker = IoUTracker(
            iou_min=float(tcfg.get("iou_min", 0.12)),
            max_missed=int(tcfg.get("max_missed", 40)),
            smooth=float(tcfg.get("smooth", 0.70)),
            vote_window=vote_window,
            vote_min=vote_min,
            nms_iou=float(tcfg.get("nms_iou", 0.40)),
        )
        self._lock = threading.RLock()
        self._id_latest: Optional[np.ndarray] = None
        self._id_busy = False
        self._id_submit = threading.Lock()
        if load_detector or load_embedder:
            self._ensure_engine()

    def _ensure_engine(self) -> InsightEngine:
        if self._engine is None:
            print("[pipeline] loading InsightFace SCRFD+ArcFace...")
            self._engine = InsightEngine(self.cfg)
        return self._engine

    def _ensure_detector(self) -> InsightEngine:
        return self._ensure_engine()

    def _ensure_embedder(self) -> InsightEngine:
        return self._ensure_engine()

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDet]:
        dets = self._ensure_engine().detect(frame_bgr, max_num=max(1, self.max_faces))
        dets = [d for d in dets if min(d.w, d.h) >= self.min_face_px]
        dets.sort(key=lambda d: d.area, reverse=True)
        return dets

    def process(self, frame_bgr: np.ndarray, identify: bool = True) -> List[MatchResult]:
        """框始终只跑 SCRFD。认人在后台用已有关键点提特征，不挡下一帧的框。"""
        from .engine import FaceHit

        engine = self._ensure_engine()
        hits = [FaceHit(det=d, embedding=None) for d in engine.detect(frame_bgr, max_num=max(1, self.max_faces))]
        hits = [h for h in hits if min(h.det.w, h.det.h) >= self.min_face_px]
        with self._lock:
            self.tracker.update_hits(hits, max_faces=self.max_faces)
            results = self.tracker.snapshot_results()
        if identify:
            self.submit_identify(frame_bgr)
        return results

    def track_faces(self, frame_bgr: np.ndarray) -> List[FaceTrack]:
        self.process(frame_bgr, identify=False)
        with self._lock:
            return [t for t in self.tracker.tracks if t.missed == 0]

    def results_now(self) -> List[MatchResult]:
        with self._lock:
            return self.tracker.snapshot_results()

    def submit_identify(self, frame_bgr: np.ndarray) -> None:
        self._ensure_engine()
        with self._lock:
            jobs = [(t.tid, t.det) for t in self.tracker.tracks if t.missed == 0][: max(1, self.max_faces)]
        if not jobs:
            return
        with self._id_submit:
            self._id_latest = (frame_bgr.copy(), jobs)
            if self._id_busy:
                return
            self._id_busy = True
        threading.Thread(target=self._id_worker, name="face-id", daemon=True).start()

    def _id_worker(self) -> None:
        try:
            while True:
                with self._id_submit:
                    payload = self._id_latest
                    self._id_latest = None
                    if payload is None:
                        self._id_busy = False
                        return
                try:
                    self._identify_tracks(payload[0], payload[1])
                except Exception:
                    import traceback
                    traceback.print_exc()
        except Exception:
            with self._id_submit:
                self._id_busy = False
            raise

    def _identify_tracks(self, frame_bgr: np.ndarray, jobs) -> None:
        engine = self._ensure_engine()
        for tid, det in jobs:
            if det.kps is None:
                continue
            emb = engine.embed(frame_bgr, det)
            if emb is None:
                continue
            matched = self.matcher.match_embedding(emb, det)
            with self._lock:
                live = self.tracker.get(tid)
                if live is None:
                    continue
                live.last_emb = emb
                self._apply_sticky(live, matched)

    def recognize(
        self,
        frame_bgr: np.ndarray,
        dets: Optional[List[FaceDet]] = None,
        use_vote: bool = True,
    ) -> List[MatchResult]:
        _ = dets, use_vote
        return self.process(frame_bgr, identify=True)

    def _apply_sticky(self, live: FaceTrack, matched: MatchResult) -> None:
        """认出之后，晃动/侧脸分数掉下来也不改回 unknown。离开画面才由 tracker 清槽。"""
        if live.accepted:
            if matched.accepted and matched.name != live.name and matched.score >= live.score + self.switch_margin:
                live.apply_result(matched)
                return
            live.pending = False
            live.accepted = True
            if matched.score > 0:
                live.score = float(matched.score)
                live.second_score = matched.second_score
            if matched.accepted and matched.name == live.name:
                live.vote_count = max(live.vote_count, 1)
            return
        if matched.accepted:
            live.apply_result(matched)
            return
        live.apply_result(matched)

    def enroll_from_frame(self, frame_bgr: np.ndarray, name: str, augment: bool = True):
        embeddings, thumb = self._ensure_engine().enroll_embeddings(frame_bgr, augment=augment)
        if not embeddings:
            return None
        rec = self.gallery.save_person(name, embeddings, [thumb])
        self.gallery.reload()
        self.matcher.gallery = self.gallery
        self.matcher.reset()
        self.tracker.reset()
        return rec

    def embed_one(
        self,
        frame_bgr: np.ndarray,
        det: FaceDet,
        force_align: bool = False,
    ) -> Optional[np.ndarray]:
        _ = force_align
        hits = self._ensure_engine().analyze(frame_bgr, max_num=max(1, self.max_faces))
        if not hits:
            return None
        return hits[0].embedding
