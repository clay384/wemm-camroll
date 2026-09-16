"""Immutable, versioned photo library backed by WeMM and Chinese FTS5.

Each Library instance pins CURRENT on construction. Publishing never changes a
reader's records or matrices; construct a new instance to observe a new version.
"""
from __future__ import annotations

import copy
import contextlib
import datetime as dt
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from camroll_agent.identity import expanded_person_box, resolve_person
from camroll_agent.metadata import place_matches

CONFIGS = {"M1": ("image", False), "M2": ("image", True),
           "M3": ("dual", False), "M4": ("dual", True)}
KINDS = {"both", "images", "captions", "events"}
RRF_K = 60
BRANCH_TOP_K = 50


def chinese_tokens(text: str) -> list[str]:
    """The same pinned dictionary and segmentation for ingestion and queries."""
    import jieba
    if jieba.__version__ != "0.42.1":
        raise RuntimeError("This index requires jieba==0.42.1")
    return [t.casefold() for t in jieba.cut(text, HMM=False)
            if t.strip() and any(ch.isalnum() for ch in t)]


def validate_date(value: str | None) -> str | None:
    if value in (None, "", "unknown"):
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("date must be YYYY-MM-DD")
    dt.date.fromisoformat(value)
    return value


def _date_filter(record: dict, lower: str | None, upper: str | None) -> bool:
    if not lower and not upper:
        return True
    if record.get("kind") == "event":
        start, end = record.get("date_start"), record.get("date_end")
        return bool(start and end and (not lower or end >= lower)
                    and (not upper or start <= upper))
    # A deduplicated photo can have several independently supplied source dates.
    dates = {s.get("date") for s in record.get("sources", [])}
    dates.add(record.get("date"))
    return any(d and d != "unknown" and (not lower or d >= lower)
               and (not upper or d <= upper) for d in dates)


class Library:
    def __init__(self, memory_dir: str | Path, embedding_client: Any = None,
                 *, config: str = "M1"):
        if config not in CONFIGS:
            raise ValueError("config must be M1, M2, M3, or M4")
        self.memory_dir = Path(memory_dir).expanduser().resolve()
        self.config = config
        self.embedding = embedding_client
        self.version = (self.memory_dir / "CURRENT").read_text().strip()
        if not re.fullmatch(r"[a-f0-9]{32}", self.version):
            raise ValueError("Invalid CURRENT index version")
        self.snapshot_dir = self.memory_dir / "snapshots" / self.version
        payload = json.loads((self.snapshot_dir / "records.json").read_text())
        self.images = {r["id"]: r for r in payload["images"]}
        self.events = {r["id"]: r for r in payload["events"]}
        self.metadata = payload["metadata"]
        self.fingerprint = self.metadata["embedding_fingerprint"]
        self._vector_ids = payload["vector_ids"]
        with np.load(self.snapshot_dir / "vectors.npz", allow_pickle=False) as data:
            self._vectors = {name: data[name].copy() for name in data.files}
        for matrix in self._vectors.values():
            matrix.flags.writeable = False

    @staticmethod
    def _validate(kind, date_from, date_to):
        if kind not in KINDS:
            raise ValueError("kind must be both, images, captions, or events")
        lo, hi = validate_date(date_from), validate_date(date_to)
        if lo and hi and lo > hi:
            raise ValueError("date_from must not be after date_to")
        return lo, hi

    @staticmethod
    def _positive(value, name, maximum=1000):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer between 1 and {maximum}")

    def _person_ids(self, person=None, persons=None):
        requested = ([person] if person else []) + list(persons or [])
        registry = self.metadata.get("people_registry", [])
        return [(name, resolve_person(name, registry)) for name in requested]

    def _metadata_match(self, row, location=None, person=None, persons=None):
        if location:
            if row.get("metadata_version"):
                trusted = row.get("trusted_location") or ""
                if not (location.casefold() in trusted.casefold() or place_matches(location, row.get("place") or {})):
                    return False
            elif location.casefold() not in row.get("location", "").casefold():
                return False
        for name, pid in self._person_ids(person, persons):
            if row.get("metadata_version"):
                known = {f.get("person_id") for f in row.get("face_instances", []) if f.get("identity_status") == "matched"}
                trusted = {p.casefold() for p in row.get("trusted_people", [])}
                if not ((pid and pid in known) or name.casefold() in trusted):
                    return False
            elif not any(name.casefold() in p.casefold() for p in row.get("people", [])):
                return False
        return True

    def _eligible(self, lo, hi, location=None, person=None, persons=None):
        images = {i for i, r in self.images.items() if _date_filter(r, lo, hi)
                  and self._metadata_match(r, location, person, persons)}
        # Events with an eligible member: date restrictions apply before ranking.
        events = {self.images[i].get("event_id") for i in images}
        return images, {i for i in events if i in self.events}

    def _rank(self, matrix_name: str, query: np.ndarray, eligible: set[str]):
        ids = self._vector_ids.get(matrix_name, [])
        indices = [n for n, item in enumerate(ids) if item in eligible]
        if not indices:
            return []
        matrix = self._vectors[matrix_name]
        if matrix.shape[1] != query.shape[0]:
            raise ValueError("Embedding dimensions differ from the published index")
        scores = matrix[indices] @ query
        by_id: dict[str, float] = {}
        for n, score in zip(indices, scores):
            by_id[ids[n]] = max(by_id.get(ids[n], float("-inf")), float(score))
        return sorted(by_id.items(), key=lambda p: (-p[1], p[0]))

    @staticmethod
    def _fuse(first, second):
        scores: dict[str, float] = {}
        ranks: dict[str, dict] = {}
        for name, branch in (("image", first), ("text", second)):
            for rank, (item, score) in enumerate(branch[:BRANCH_TOP_K], 1):
                scores[item] = scores.get(item, 0) + 1 / (RRF_K + rank)
                ranks.setdefault(item, {})[name] = {"rank": rank, "score": score}
        return [(i, s, ranks[i]) for i, s in sorted(scores.items(), key=lambda p: (-p[1], p[0]))]

    def _decorate(self, ranked, records):
        result = []
        for rank, item in enumerate(ranked, 1):
            id_, score = item[:2]
            row = copy.deepcopy(records[id_])
            row.update(score=score, rank=rank)
            row["branch_ranks"] = (item[2] if len(item) > 2
                                   else {"image": {"rank": rank, "score": score}})
            result.append(row)
        return result

    @staticmethod
    def _combine(images, events, kind):
        if kind in {"images", "captions"}:
            return images
        if kind == "events":
            return events
        combined = []
        for n in range(max(len(images), len(events))):
            if n < len(images):
                combined.append(images[n])
            if n < len(events):
                combined.append(events[n])
        return combined

    def search(self, query: str, top_k: int = 10, kind: str = "both",
               date_from=None, date_to=None, location=None, person=None, persons=None, *, query_vector=None) -> dict:
        self._positive(top_k, "top_k")
        lo, hi = self._validate(kind, date_from, date_to)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if self.embedding is None:
            from camroll_agent.wemm import EmbeddingClient
            self.embedding = EmbeddingClient()
        if self.embedding.fingerprint != self.fingerprint:
            raise ValueError("WeMM fingerprint differs from the published index; rebuild required")
        # The embedding endpoint enforces the query token limit; never split queries.
        q = np.asarray(self.embedding.encode(text=query) if query_vector is None
                       else query_vector, dtype=np.float32).reshape(-1)
        if not np.isfinite(q).all() or np.linalg.norm(q) == 0:
            raise ValueError("Invalid query embedding")
        q = q / np.linalg.norm(q)
        image_ids, event_ids = self._eligible(lo, hi, location, person, persons)
        image_rank = self._rank("image", q, image_ids)
        event_rank = []
        if kind in {"both", "events"}:
            aggregate: dict[str, float] = {}
            for image_id, score in image_rank:
                event_id = self.images[image_id].get("event_id")
                if event_id in event_ids:
                    aggregate[event_id] = max(aggregate.get(event_id, float("-inf")), score)
            event_rank = sorted(aggregate.items(), key=lambda p: (-p[1], p[0]))
            if CONFIGS[self.config][0] == "dual":
                event_rank = self._fuse(event_rank, self._rank("event", q, event_ids))
        if CONFIGS[self.config][0] == "dual" and kind != "events":
            image_rank = self._fuse(image_rank, self._rank("caption", q, image_ids))
        combined = self._combine(image_rank, event_rank, kind)
        selected = combined[:top_k]
        # The mixed prefix contains a prefix of each branch, so _decorate keeps
        # the original within-branch ranks while copying only returned records.
        selected_images = [item for item in selected if item[0] in self.images]
        selected_events = [item for item in selected if item[0] in self.events]
        result = self._combine(self._decorate(selected_images, self.images),
                               self._decorate(selected_events, self.events), kind)
        for row in result:
            if row.get("kind") == "event":
                row["matched_image_ids"] = [i for i in row.get("image_ids", []) if i in image_ids]
        return {"results": result, "count": len(combined), "query": query,
                "count_scope": "all_ranked_candidates_before_top_k",
                "index_version": self.version, "config": self.config,
                "text_coverage": self.metadata.get("text_coverage", 0),
                "filter_coverage": self._filter_coverage(person, persons)}

    def grep(self, query: str, kind: str = "both", top_k: int = 10,
             date_from=None, date_to=None) -> dict:
        self._positive(top_k, "top_k")
        lo, hi = self._validate(kind, date_from, date_to)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        tokens = chinese_tokens(query)
        image_ids, event_ids = self._eligible(lo, hi)
        matches = []
        if tokens:
            # Quoted token conjunction prevents untrusted FTS expressions.
            expression = " AND ".join('"' + t.replace('"', '""') + '"' for t in tokens)
            with contextlib.closing(sqlite3.connect(
                    f"file:{self.snapshot_dir / 'search.sqlite'}?mode=ro", uri=True)) as db:
                matches = db.execute("SELECT id, kind, bm25(records_fts) FROM records_fts "
                                     "WHERE records_fts MATCH ? ORDER BY bm25(records_fts), id",
                                     (expression,)).fetchall()
        images, events = [], []
        for id_, item_kind, score in matches:
            if item_kind == "image" and id_ in image_ids:
                images.append(dict(copy.deepcopy(self.images[id_]), score=-score))
            elif item_kind == "event" and id_ in event_ids:
                events.append(dict(copy.deepcopy(self.events[id_]), score=-score))
        result = self._combine(images, events, kind)
        return {"results": result[:top_k], "count": len(result), "index_version": self.version,
                "notice": "零结果仅代表索引文字没有匹配，不能证明相册里不存在该内容。"}

    def list_by_date(self, date_from=None, date_to=None, kind="both", location=None,
                     person=None, persons=None, limit=50, offset=0) -> dict:
        self._positive(limit, "limit")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        lo, hi = self._validate(kind, date_from, date_to)
        image_ids, event_ids = self._eligible(lo, hi, location, person, persons)
        def ordered(ids, records):
            rows = [copy.deepcopy(records[i]) for i in ids]
            for row in rows:
                if row.get("kind") == "event":
                    row["matched_image_ids"] = [i for i in row.get("image_ids", []) if i in image_ids]
            return sorted(rows, key=lambda r: (r.get("date") or r.get("date_start") or "9999", r["id"]))
        result = self._combine(ordered(image_ids, self.images), ordered(event_ids, self.events), kind)
        return {"results": result[offset:offset + limit], "count": len(result), "offset": offset,
                "index_version": self.version, "filter_coverage": self._filter_coverage(person, persons)}

    def _filter_coverage(self, person=None, persons=None):
        return {"identity_processed_images": sum(r.get("identity_status") == "processed" for r in self.images.values()),
                "album_image_count": len(self.images),
                "unregistered_person_filters": [name for name, pid in self._person_ids(person, persons) if not pid],
                "notice": "身份匹配可能漏检或不确定；零匹配不能证明该人物没有出现。地点条件只使用可信元数据，不能证明具体景点。"}

    def query_photo_people(self, image_ids, person=None):
        self.image_paths(image_ids)  # Enforce snapshot/album boundary before returning data.
        registry = self.metadata.get("people_registry", [])
        pid = resolve_person(person, registry) if person else None
        images = []
        for iid in image_ids:
            row = self.images[iid]
            faces = copy.deepcopy(row.get("face_instances", []))
            if person:
                faces = [f for f in faces if pid and f.get("person_id") == pid]
            images.append({"id": iid, "image_id": iid, "kind": "image",
                           "identity_status": row.get("identity_status", "not_processed"),
                           "face_instances": faces,
                           "unknown_faces": sum(f.get("identity_status") == "unknown" for f in row.get("face_instances", [])),
                           "all_face_count": len(row.get("face_instances", []))})
        return {"images": images, "people_registry": copy.deepcopy(registry),
                "relationships": copy.deepcopy(self.metadata.get("relationships", [])),
                "requested_person_status": "not_registered" if person and not pid else "registered" if person else None,
                "index_version": self.version, "identity_version": self.metadata.get("identity_version"),
                "notice": "matched 表示命名参考照的算法匹配，未人工确认；分数是相似度，不是准确率。unknown 不代表人物不存在。"}

    def target_regions(self, image_ids, target_person=None, face_id=None):
        self.image_paths(image_ids)
        registry = self.metadata.get("people_registry", [])
        pid = resolve_person(target_person, registry) if target_person else None
        if target_person and not pid:
            raise ValueError("Target person is not registered; no target crop is available")
        regions = []
        for iid in image_ids:
            found = [f for f in self.images[iid].get("face_instances", [])
                     if (not pid or f.get("person_id") == pid and f.get("identity_status") == "matched")
                     and (not face_id or f.get("face_id") == face_id)]
            if not found:
                raise ValueError("Target face not located in every requested photo; inspect full images or query_photo_people first")
            if len(found) != 1:
                raise ValueError("Target face is ambiguous; select an explicit face_id")
            face = found[0]
            regions.append({"image_id": iid, "face_id": face["face_id"], "person_id": face.get("person_id"),
                            "name": face.get("name"), "face_bbox": face["bbox"],
                            "crop_bbox": expanded_person_box(face["bbox"], face["image_size"]),
                            "coordinate_space": face.get("coordinate_space"),
                            "limitation": "扩展区域可能包含其他人，需结合原图核对；不等于人体分割。"})
        return regions

    def get(self, id: str) -> dict:
        record = self.images.get(id) or self.events.get(id)
        if record is None:
            return {"error": {"code": "unknown_id", "field": "id", "message": f"Unknown asset ID: {id}"}}
        return copy.deepcopy(record)

    def image_paths(self, image_ids: list[str]) -> list[str]:
        if not isinstance(image_ids, list) or not image_ids:
            raise ValueError("image_ids must be a nonempty list")
        paths = []
        for image_id in image_ids:
            if not isinstance(image_id, str) or image_id not in self.images:
                raise ValueError(f"Unknown image_id: {image_id}")
            path = Path(self.images[image_id]["path"]).resolve()
            if not path.is_relative_to(self.memory_dir / "assets") or not path.is_file():
                raise ValueError(f"Managed image is unavailable: {image_id}")
            paths.append(str(path))
        return paths

    def stats(self) -> dict:
        return {"image_count": len(self.images), "event_count": len(self.events),
                "index_version": self.version, "config": self.config, **copy.deepcopy(self.metadata)}
