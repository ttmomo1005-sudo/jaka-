"""余弦相似度匹配 + 时序投票，降低认错人概率。"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np

from .detector import FaceDet
from .gallery import FaceGallery, PersonRecord


@dataclass
class MatchResult:
    name: str
    score: float
    accepted: bool
    det: FaceDet
    second_score: float = 0.0
    pending: bool = False
    vote_count: int = 0
    vote_need: int = 0
    vote_window_filled: int = 0
    tid: int = 0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class FaceMatcher:
    def __init__(
        self,
        gallery: FaceGallery,
        cosine_threshold: float = 0.45,
        margin: float = 0.06,
        vote_window: int = 2,
        vote_min: int = 1,
        lock_score: float = 0.45,
    ):
        self.gallery = gallery
        self.cosine_threshold = cosine_threshold
        self.margin = margin
        self.vote_window = max(1, vote_window)
        self.vote_min = max(1, min(vote_min, self.vote_window))
        self.lock_score = float(lock_score)
        self._votes: Deque[Tuple[str, float]] = deque(maxlen=self.vote_window)

    def reset(self) -> None:
        self._votes.clear()

    def vote_on(self, result: MatchResult, votes: Deque[Tuple[str, float]]) -> MatchResult:
        """对某一条轨迹投票。过阈值的当帧直接锁定，不再等 3/5、7/10。"""
        if result.accepted:
            votes.append((result.name, float(result.score)))
            return MatchResult(
                name=result.name,
                score=result.score,
                accepted=True,
                det=result.det,
                second_score=result.second_score,
                pending=False,
                vote_count=max(1, self.vote_min),
                vote_need=self.vote_min,
                vote_window_filled=len(votes),
            )
        label = result.name if result.accepted else "unknown"
        votes.append((label, float(result.score)))
        filled = len(votes)
        counts = Counter(n for n, _ in votes)
        name, count = counts.most_common(1)[0]
        mean_score = result.score
        if name != "unknown":
            named_scores = [s for n, s in votes if n == name]
            if named_scores:
                mean_score = float(sum(named_scores) / len(named_scores))
        locked = name != "unknown" and count >= self.vote_min
        pending = (not locked) and name != "unknown" and count > 0
        return MatchResult(
            name=name if (locked or pending) else "unknown",
            score=mean_score if locked else result.score,
            accepted=locked,
            det=result.det,
            second_score=result.second_score,
            pending=pending,
            vote_count=count if name != "unknown" else 0,
            vote_need=self.vote_min,
            vote_window_filled=filled,
        )

    def match_embedding(self, emb: np.ndarray, det: FaceDet) -> MatchResult:
        records: List[PersonRecord] = list(self.gallery.records.values())
        if not records:
            return MatchResult(name="unknown", score=0.0, accepted=False, det=det)

        scored: List[Tuple[str, float]] = [(r.name, cosine(emb, r.embedding)) for r in records]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_name, best = scored[0]
        second = scored[1][1] if len(scored) > 1 else 0.0
        accepted = best >= self.cosine_threshold and (best - second) >= self.margin
        name = best_name if accepted else "unknown"
        return MatchResult(name=name, score=best, accepted=accepted, det=det, second_score=second)

    def vote(self, result: MatchResult) -> MatchResult:
        return self.vote_on(result, self._votes)
