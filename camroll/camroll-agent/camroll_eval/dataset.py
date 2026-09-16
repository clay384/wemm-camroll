"""Read PhotoBench inputs; export only originals and identity references to Agent."""
from __future__ import annotations

from pathlib import Path
import json

from .common import read_json, sha256, safe_child, object_hash
from .compat import _normalize_media_refs

FORMAL_QA_SHA256 = "f625ae9e26b4666c277b424bd472b6790baf0d6773e9020b20eff985d4cc2eb0"
FORMAL_MANIFEST_SHA256 = "9b0edad9682bbddfe1a84a787f7e0bc56e1c141dd42a11f9295227e1556bae1c"


class Dataset:
    def __init__(self, root, qa_set="100qa-full", *, strict=True):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        self.qa_set = qa_set
        relative = (self.manifest.get("qa_sets") or {}).get(qa_set)
        if not relative:
            raise ValueError("unknown QA set")
        self.qa_path = safe_child(self.root, relative)
        self.rows = [json.loads(line) for line in self.qa_path.read_text().splitlines() if line.strip()]
        self.photos = [safe_child(self.root, path) for path in self.manifest.get("photos", [])]
        if len({p.name.casefold() for p in self.photos}) != len(self.photos):
            raise ValueError("ambiguous duplicate photo filename")
        ids = [row.get("qa_id") for row in self.rows]
        if len(ids) != len(set(ids)) or any(not i for i in ids):
            raise ValueError("missing or duplicate qa_id")
        if strict and (self.manifest.get("album_id") != "album3-max" or
                       qa_set != "100qa-full" or len(self.photos) != 355 or len(self.rows) != 100):
            raise ValueError("formal protocol requires album3-max / 355 photos / 100qa-full")
        self.faces = self.manifest.get("faces") or []
        self.relationships = self.manifest.get("family_relationships") or []
        if strict:
            if len(self.faces) != 8 or len(self.relationships) != 6:
                raise ValueError("formal protocol requires 8 identity references and 6 relationships")
            if sum(len(self.turns(row)) for row in self.rows) != 111 or sum(
                row.get("answerability") != "unanswerable" for row in self.rows) != 81:
                raise ValueError("formal protocol requires 111 turns and 81 retrieval-eligible questions")
            if sha256(self.qa_path) != FORMAL_QA_SHA256 or sha256(self.root / "manifest.json") != FORMAL_MANIFEST_SHA256:
                raise ValueError("formal QA or manifest differs from the frozen PhotoBench source")
        inputs = [self.root / "manifest.json", self.qa_path, *self.photos,
                  *[safe_child(self.root, f["ref_image"]) for f in self.faces]]
        self.files = sorted(set(inputs), key=lambda p: str(p.relative_to(self.root)))
        missing = [str(path.relative_to(self.root)) for path in self.files if not path.is_file()]
        if missing:
            raise ValueError("dataset is missing " + str(len(missing)) + " files")
        available = {path.name.casefold() for path in self.photos}
        self.unresolved_gt = [{"qa_id": row["qa_id"], "media_id": ref["media_id"]}
                              for row in self.rows for ref in _normalize_media_refs(row)
                              if ref["media_type"] != "image" or
                              Path(ref["media_id"]).name.casefold() not in available]
        if self.unresolved_gt:
            raise ValueError("QA GT cannot be resolved against the complete original photo set")

    def integrity(self):
        import hashlib
        rows = [{"path": str(p.relative_to(self.root)), "sha256": sha256(p), "bytes": p.stat().st_size}
                for p in self.files]
        combined = hashlib.sha256("\n".join(f"{r['path']}:{r['sha256']}" for r in rows).encode()).hexdigest()
        policy = Path(__file__).with_name("task_actions.json")
        return {"algorithm": "sha256", "manifest_sha256": sha256(self.root / "manifest.json"),
                "qa_sha256": sha256(self.qa_path), "dataset_sha256": combined,
                "task_action_policy_sha256": sha256(policy), "files_checked": len(rows),
                "missing_files": [], "files": rows}

    def build_spec(self, album_id):
        persons = []
        lookup = {}
        for face in self.faces:
            pid = "person_" + str(face["face_id"])
            name = str(face["canonical_name"])
            if name in lookup:
                raise ValueError("duplicate canonical identity")
            lookup[name] = pid
            persons.append({"person_id": pid, "name": name, "aliases": face.get("aliases") or [],
                            "family_role": face.get("family_role") or "",
                            "reference_image": face["ref_image"]})
        relations = []
        for relation in self.relationships:
            try:
                relations.append({"subject_id": lookup[relation["subject"]],
                                  "predicate": relation["predicate"],
                                  "object_id": lookup[relation["object"]], "source": "manifest"})
            except KeyError as exc:
                raise ValueError("relationship references an unregistered person") from None
        return {"album_id": album_id, "root_folder": str(self.root),
                "input_policy": "originals_and_identity_references",
                "images": [{"path": str(p.relative_to(self.root))} for p in self.photos],
                "identity": {"persons": persons, "relationships": relations}}

    @staticmethod
    def turns(row):
        turns = row.get("conversation")
        if turns:
            return [{"message": str(t["message"]), "expected_action": t.get("expected_action", "answer"),
                     "reference_answer": t.get("reference_answer") or row.get("answer", "")}
                    for t in turns]
        return [{"message": row["question"], "expected_action": row.get("expected_action", "answer"),
                 "reference_answer": row.get("answer", "")}]


def snapshot_media(memory_dir, version):
    root = Path(memory_dir).resolve()
    records = read_json(safe_child(root, f"snapshots/{version}/records.json"))
    mapping = {}
    for row in records.get("images", []):
        sources = row.get("sources") or []
        filename = row.get("filename") or (sources[0].get("filename") if sources else None)
        source_path = row.get("source_path") or (sources[0].get("source_path") if sources else None)
        if not filename and source_path:
            filename = Path(source_path).name
        mapping[row["id"]] = {"asset_id": row["id"], "media_id": filename,
                              "file_name": filename, "media_type": "image",
                              "sha256": row.get("asset_id") or str(row["id"]).removeprefix("img_")}
    return mapping, records.get("metadata") or {}
