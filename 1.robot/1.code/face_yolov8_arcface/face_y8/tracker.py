"""人脸身份槽：框可以抖，名字不能一抖就丢。

用 IOU / 中心距 / 外观向量把当前脸接到已有槽上。
接不上时优先复活最近的旧槽，而不是新建一个 unknown。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from .detector import FaceDet
from .engine import FaceHit
from .matcher import MatchResult, cosine


def _iou(a: FaceDet, b: FaceDet) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter + 1e-6
    return inter / union


def _center(d: FaceDet) -> Tuple[float, float]:
    return (d.x1 + d.x2) / 2.0, (d.y1 + d.y2) / 2.0


def _blend(d0: FaceDet, d1: FaceDet, a: float) -> FaceDet:
    a = min(1.0, max(0.0, a))
    b = 1.0 - a
    kps = d1.kps if d1.kps is not None else d0.kps
    return FaceDet(
        x1=int(b * d0.x1 + a * d1.x1),
        y1=int(b * d0.y1 + a * d1.y1),
        x2=int(b * d0.x2 + a * d1.x2),
        y2=int(b * d0.y2 + a * d1.y2),
        conf=float(d1.conf),
        kps=None if kps is None else np.asarray(kps).copy(),
    )


@dataclass
class FaceTrack:
    tid: int
    det: FaceDet
    name: str = "unknown"
    score: float = 0.0
    second_score: float = 0.0
    accepted: bool = False
    pending: bool = False
    vote_count: int = 0
    vote_need: int = 1
    missed: int = 0
    last_emb: Optional[np.ndarray] = None
    votes: Deque[Tuple[str, float]] = field(default_factory=lambda: deque(maxlen=3))

    def to_result(self) -> MatchResult:
        return MatchResult(
            name=self.name,
            score=self.score,
            accepted=self.accepted,
            det=self.det,
            second_score=self.second_score,
            pending=self.pending,
            vote_count=self.vote_count,
            vote_need=self.vote_need,
            vote_window_filled=len(self.votes),
            tid=self.tid,
        )

    def apply_result(self, r: MatchResult) -> None:
        self.name = r.name
        self.score = r.score
        self.second_score = r.second_score
        self.accepted = r.accepted
        self.pending = r.pending
        self.vote_count = r.vote_count
        self.vote_need = r.vote_need

    def attach_hit(self, hit: FaceHit, smooth: float) -> None:
        self.det = _blend(self.det, hit.det, smooth)
        if hit.det.kps is not None:
            self.det.kps = np.asarray(hit.det.kps).copy()
        if hit.embedding is not None:
            self.last_emb = hit.embedding
        self.missed = 0


class IoUTracker:
    def __init__(
        self,
        iou_min: float = 0.12,
        max_missed: int = 40,
        smooth: float = 0.7,
        vote_window: int = 3,
        vote_min: int = 1,
        nms_iou: float = 0.4,
    ):
        self.iou_min = iou_min
        self.max_missed = max_missed
        self.smooth = smooth
        self.nms_iou = nms_iou
        self.vote_window = max(1, vote_window)
        self.vote_min = max(1, vote_min)
        self.tracks: List[FaceTrack] = []
        self._next = 1

    def reset(self) -> None:
        self.tracks.clear()

    def get(self, tid: int) -> Optional[FaceTrack]:
        for t in self.tracks:
            if t.tid == tid:
                return t
        return None

    def snapshot_results(self) -> List[MatchResult]:
        return [t.to_result() for t in self.tracks if t.missed == 0]

    def _assoc_score(self, tr: FaceTrack, hit: FaceHit) -> float:
        iou = _iou(tr.det, hit.det)
        emb_s = 0.0
        if tr.last_emb is not None and hit.embedding is not None:
            emb_s = cosine(tr.last_emb, hit.embedding)
        if emb_s >= 0.42:
            return 2.0 + emb_s
        if iou >= self.iou_min:
            return 1.0 + iou + 0.4 * emb_s
        cx0, cy0 = _center(tr.det)
        cx1, cy1 = _center(hit.det)
        dist = ((cx0 - cx1) ** 2 + (cy0 - cy1) ** 2) ** 0.5
        scale = max(tr.det.w, tr.det.h, hit.det.w, hit.det.h, 1)
        if dist < 1.15 * scale:
            return 0.55 + 0.3 * emb_s - dist / (scale * 4.0)
        if emb_s >= 0.32:
            return 0.35 + emb_s
        return -1.0

    def update(self, dets: List[FaceDet], max_faces: int = 0) -> List[FaceTrack]:
        hits = [FaceHit(det=d, embedding=None) for d in dets]
        return self.update_hits(hits, max_faces=max_faces)

    def update_hits(self, hits: List[FaceHit], max_faces: int = 0) -> List[FaceTrack]:
        if max_faces > 0:
            hits = hits[:max_faces]
        unused = set(range(len(hits)))
        assigned: set[int] = set()

        pairs: List[Tuple[float, int, int]] = []
        for ti, tr in enumerate(self.tracks):
            for hi in unused:
                s = self._assoc_score(tr, hits[hi])
                if s > 0:
                    pairs.append((s, ti, hi))
        pairs.sort(key=lambda x: x[0], reverse=True)
        for _, ti, hi in pairs:
            if ti in assigned or hi not in unused:
                continue
            self.tracks[ti].attach_hit(hits[hi], self.smooth)
            assigned.add(ti)
            unused.discard(hi)

        for ti, tr in enumerate(self.tracks):
            if ti not in assigned:
                tr.missed += 1

        for hi in sorted(unused):
            hit = hits[hi]
            revived = self._revive_nearest(hit)
            if revived is not None:
                revived.attach_hit(hit, self.smooth)
                continue
            self.tracks.append(
                FaceTrack(
                    tid=self._next,
                    det=hit.det,
                    last_emb=None if hit.embedding is None else hit.embedding.copy(),
                    vote_need=self.vote_min,
                    votes=deque(maxlen=self.vote_window),
                )
            )
            self._next += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        vis = [t for t in self.tracks if t.missed == 0]
        vis.sort(key=lambda t: (t.accepted, t.det.area), reverse=True)
        kept_vis: List[FaceTrack] = []
        for t in vis:
            if any(_iou(t.det, k.det) >= self.nms_iou for k in kept_vis):
                continue
            kept_vis.append(t)
        if max_faces > 0:
            kept_vis = kept_vis[:max_faces]
        keep_ids = {t.tid for t in kept_vis}
        for t in vis:
            if t.tid not in keep_ids:
                t.missed = max(t.missed, 1)
        mem = [t for t in self.tracks if t.missed > 0]
        self.tracks = kept_vis + mem
        return kept_vis

    def _revive_nearest(self, hit: FaceHit) -> Optional[FaceTrack]:
        """晃动导致 IOU 对不上时，把新框接回已有身份槽，避免变成新的 unknown。"""
        best: Optional[FaceTrack] = None
        best_d = 1e9
        cx1, cy1 = _center(hit.det)
        scale = max(hit.det.w, hit.det.h, 1)
        for tr in self.tracks:
            if tr.missed <= 0:
                continue
            cx0, cy0 = _center(tr.det)
            dist = ((cx0 - cx1) ** 2 + (cy0 - cy1) ** 2) ** 0.5
            if dist < best_d and dist < 1.6 * scale:
                best_d = dist
                best = tr
        return best
