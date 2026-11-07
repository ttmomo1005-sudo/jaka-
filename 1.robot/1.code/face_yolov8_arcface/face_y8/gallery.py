"""人脸库：每人多张样本 → 平均 L2 归一化 embedding。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import ensure_dirs


def _safe_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("姓名不能为空")
    # 目录名尽量可读；非法字符替换
    return re.sub(r'[<>:"/\\|?*]', "_", name)


@dataclass
class PersonRecord:
    name: str
    embedding: np.ndarray
    samples: int


class FaceGallery:
    def __init__(self, cfg: dict[str, Any]):
        _, gallery = ensure_dirs(cfg)
        self.root = gallery
        self.people_dir = gallery / "people"
        self.index_path = gallery / "index.json"
        self.records: Dict[str, PersonRecord] = {}
        self.reload()

    def reload(self) -> None:
        self.records.clear()
        if not self.index_path.exists():
            self._rebuild_index()
            return
        data = json.loads(self.index_path.read_text(encoding="utf-8-sig"))
        for item in data.get("people", []):
            name = item["name"]
            emb_path = self.root / item["embedding"]
            if not emb_path.exists():
                continue
            emb = np.load(str(emb_path)).astype(np.float32)
            self.records[name] = PersonRecord(name=name, embedding=emb, samples=int(item.get("samples", 0)))

    def _rebuild_index(self) -> None:
        people = []
        for person_dir in sorted(self.people_dir.glob("*")):
            if not person_dir.is_dir():
                continue
            emb_path = person_dir / "mean_embedding.npy"
            if not emb_path.exists():
                continue
            meta_path = person_dir / "meta.json"
            samples = 0
            if meta_path.exists():
                samples = int(json.loads(meta_path.read_text(encoding="utf-8-sig")).get("samples", 0))
            people.append(
                {
                    "name": person_dir.name,
                    "embedding": str(emb_path.relative_to(self.root)).replace("\\", "/"),
                    "samples": samples,
                }
            )
        self.index_path.write_text(json.dumps({"people": people}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.reload()

    def list_people(self) -> List[str]:
        return sorted(self.records.keys())

    def delete(self, name: str) -> bool:
        import shutil

        name = _safe_name(name)
        person_dir = self.people_dir / name
        if not person_dir.exists():
            return False
        shutil.rmtree(person_dir, ignore_errors=True)
        self._rebuild_index()
        return True

    def person_dir(self, name: str) -> Path:
        return self.people_dir / _safe_name(name)

    def save_person(self, name: str, embeddings: List[np.ndarray], thumbs: Optional[List[np.ndarray]] = None) -> PersonRecord:
        name = _safe_name(name)
        if not embeddings:
            raise ValueError("没有可用 embedding")
        mat = np.stack(embeddings, axis=0).astype(np.float32)
        mean = mat.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-12)
        person_dir = self.person_dir(name)
        person_dir.mkdir(parents=True, exist_ok=True)
        samples_dir = person_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        np.save(str(person_dir / "mean_embedding.npy"), mean.astype(np.float32))
        np.save(str(person_dir / "embeddings.npy"), mat.astype(np.float32))
        meta = {"name": name, "samples": len(embeddings)}
        (person_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if thumbs:
            for i, img in enumerate(thumbs[: len(embeddings)]):
                cv2.imwrite(str(samples_dir / f"{i:03d}.jpg"), img)
        self._rebuild_index()
        return self.records[name]
