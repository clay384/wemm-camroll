"""Album-local named reference matching. No anonymous clustering or VLM names.

The only identity input is a named reference. Matching remains an algorithmic
association with an unknown outcome, never a claim of human verification.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IDENTITY_VERSION = "buffalo-l-cpu-reference-match-v1"
POLICY = {"detection_score_min": 0.60, "min_face_pixels": 24,
          "cosine_min": 0.40, "top_two_margin_min": 0.08,
          "det_size": [640, 640], "execution_provider": "CPUExecutionProvider",
          "reference_padding_fraction": 0.25, "reference_padding_rgb": [127, 127, 127]}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_identity_spec(spec_path):
    spec_path = Path(spec_path).resolve()
    raw = json.loads(spec_path.read_text())
    block = raw.get("identity")
    if not block:
        return {"persons": [], "relationships": []}
    if not isinstance(block, dict) or set(block) - {"persons", "relationships"}:
        raise ValueError("identity must contain only persons and relationships")
    root = Path(raw.get("root_folder") or ".").expanduser()
    root = (root if root.is_absolute() else spec_path.parent / root).resolve()
    people, ids = [], set()
    for person in block.get("persons", []):
        pid, name = person.get("person_id"), person.get("name")
        if not isinstance(pid, str) or not re.fullmatch(r"person_[A-Za-z0-9_-]+", pid) or pid in ids:
            raise ValueError("Unique stable person_id values are required")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each reference identity needs a name")
        aliases = person.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
            raise ValueError("Identity aliases must be strings")
        reference = person.get("reference_image")
        if not isinstance(reference, str) or not reference:
            raise ValueError("Named identity requires reference_image")
        reference = Path(reference).expanduser()
        reference = (reference if reference.is_absolute() else root / reference).resolve()
        if not reference.is_file():
            raise FileNotFoundError("Named reference image is unavailable")
        ids.add(pid)
        people.append({"person_id": pid, "name": name.strip(),
                       "aliases": sorted(set(a.strip() for a in aliases if a.strip())),
                       "family_role": str(person.get("family_role") or ""),
                       "reference_image": str(reference), "reference_sha256": _sha(reference),
                       "source": "named_reference_manifest"})
    if len(people) > 100:
        raise ValueError("At most 100 named references are supported")
    relationships = []
    for relationship in block.get("relationships", []):
        subject, obj = relationship.get("subject_id"), relationship.get("object_id")
        predicate = relationship.get("predicate")
        if subject not in ids or obj not in ids or not isinstance(predicate, str) or not predicate.strip():
            raise ValueError("Relationship must name registered subjects/objects and a predicate")
        relationships.append({"subject_id": subject, "object_id": obj, "predicate": predicate.strip(),
                              "source": "named_reference_manifest"})
    return {"persons": people, "relationships": relationships}


def public_registry(spec):
    return [{k: v for k, v in p.items() if k != "reference_image"} for p in spec["persons"]]


def resolve_person(query, persons):
    if not query:
        return None
    query = str(query).casefold().strip()
    matches = [p["person_id"] for p in persons
               if query in {str(v).casefold() for v in [p["person_id"], p["name"], p.get("family_role", ""), *p.get("aliases", [])]}]
    if len(matches) > 1:
        raise ValueError("Ambiguous identity alias; use an explicit person_id")
    return matches[0] if matches else None


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not len(vector) or not np.isfinite(vector).all() or norm <= 0:
        raise ValueError("Invalid face embedding")
    return vector / norm


def match_embedding(embedding, prototypes):
    """Frozen open-set threshold and top-two ambiguity margin, no QA tuning."""
    if not prototypes:
        return {"identity_status": "unknown", "reason": "no_named_references", "match_score": None}
    vector = normalize(embedding)
    scores = sorted(((pid, float(vector @ normalize(value))) for pid, value in prototypes.items()),
                    key=lambda item: (-item[1], item[0]))
    pid, score = scores[0]
    margin = score - scores[1][1] if len(scores) > 1 else None
    matched = score >= POLICY["cosine_min"] and (margin is None or margin >= POLICY["top_two_margin_min"])
    return {"identity_status": "matched" if matched else "unknown",
            "person_id": pid if matched else None,
            "match_score": round(score, 6), "match_margin": round(margin, 6) if margin is not None else None,
            "reason": "reference_match" if matched else "below_threshold_or_ambiguous",
            "score_semantics": "cosine_similarity_not_probability"}


class FaceEngine:
    """Explicit local ONNX paths and CPU-only sessions; never downloads weights."""
    def __init__(self, model_dir=None):
        model_dir = Path(model_dir or os.getenv("CAMROLL_FACE_MODEL_DIR", "~/.insightface/models/buffalo_l")).expanduser()
        paths = [model_dir / name for name in ("det_10g.onnx", "w600k_r50.onnx")]
        if not all(p.is_file() for p in paths):
            raise FileNotFoundError("Existing buffalo_l detector and recognition weights are required; automatic download disabled")
        import onnxruntime as ort
        from insightface.model_zoo import get_model
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        kwargs = {"providers": ["CPUExecutionProvider"], "sess_options": options, "download": False}
        self.detector = get_model(str(paths[0]), **kwargs)
        self.recognizer = get_model(str(paths[1]), **kwargs)
        self.detector.prepare(ctx_id=-1, input_size=tuple(POLICY["det_size"]), det_thresh=POLICY["detection_score_min"])
        self.recognizer.prepare(ctx_id=-1)
        self.provenance = {"version": IDENTITY_VERSION, "policy": dict(POLICY),
                           "model_sha256": {p.name: _sha(p) for p in paths},
                           "detector_class": type(self.detector).__name__,
                           "recognizer_class": type(self.recognizer).__name__,
                           "coordinate_space": "exif_transposed_rgb_pixels"}

    def detect(self, path):
        return self._detect(path, reference=False)

    def detect_reference(self, path):
        # Named inputs are often tight face crops. Fixed padding restores detector
        # context without changing thresholds or accessing evaluation answers.
        return self._detect(path, reference=True)

    def _detect(self, path, *, reference=False):
        import cv2
        from insightface.app.common import Face
        with Image.open(path) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        found = []
        height, width = rgb.shape[:2]
        padding = int(max(height, width) * POLICY["reference_padding_fraction"]) if reference else 0
        model_image = (cv2.copyMakeBorder(image, padding, padding, padding, padding,
                       cv2.BORDER_CONSTANT, value=tuple(POLICY["reference_padding_rgb"])) if padding else image)
        boxes, points = self.detector.detect(model_image, max_num=0, metric="default")
        for index, raw_box in enumerate(boxes):
            x1, y1, x2, y2, score = map(float, raw_box)
            x1, y1, x2, y2 = (v - padding for v in (x1, y1, x2, y2))
            bbox = [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]
            valid = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) >= POLICY["min_face_pixels"] and score >= POLICY["detection_score_min"] and points is not None
            row = {"bbox": [round(x, 3) for x in bbox], "detection_score": round(score, 6),
                   "image_size": [width, height], "eligible": bool(valid),
                   "reference_padding_pixels": padding}
            if valid:
                face = Face(bbox=np.asarray(raw_box[:4]), kps=points[index], det_score=score)
                self.recognizer.get(model_image, face)
                row["embedding"] = normalize(face.embedding).tolist()
            found.append(row)
        return sorted(found, key=lambda item: tuple(item["bbox"]))


def seed_references(spec, engine):
    prototypes, references = {}, []
    for person in spec["persons"]:
        detect = getattr(engine, "detect_reference", engine.detect)
        found = [face for face in detect(person["reference_image"]) if face.get("eligible")]
        if len(found) != 1:
            raise ValueError(f"Reference {person['person_id']} must contain exactly one eligible face; found {len(found)}")
        prototypes[person["person_id"]] = normalize(found[0]["embedding"]).tolist()
        references.append({"person_id": person["person_id"], "reference_sha256": person["reference_sha256"],
                           "bbox": found[0]["bbox"], "status": "registered"})
    return {"prototypes": prototypes, "references": references, "provenance": engine.provenance}


def photo_faces(image_id, path, engine, seeded, registry):
    people = {p["person_id"]: p for p in registry}
    result = []
    for detected in engine.detect(path):
        signature = json.dumps([image_id, detected["bbox"], IDENTITY_VERSION], sort_keys=True).encode()
        row = {k: v for k, v in detected.items() if k != "embedding"}
        row.update(face_id="face_" + hashlib.sha256(signature).hexdigest()[:24], image_id=image_id,
                   source="named_reference_match", identity_version=IDENTITY_VERSION,
                   coordinate_space="exif_transposed_rgb_pixels")
        if detected.get("eligible"):
            row.update(match_embedding(detected["embedding"], seeded["prototypes"]))
        else:
            row.update(identity_status="unknown", person_id=None, match_score=None, reason="face_quality_insufficient")
        if row.get("person_id"):
            row["name"] = people[row["person_id"]]["name"]
        result.append(row)
    return result


def expanded_person_box(bbox, image_size):
    """Context crop includes torso; it is a region, not body segmentation."""
    width, height = image_size
    x1, y1, x2, y2 = bbox
    fw, fh = x2 - x1, y2 - y1
    return [max(0, int(x1 - fw * .8)), max(0, int(y1 - fh * .3)),
            min(width, int(x2 + fw * .8)), min(height, int(y2 + fh * 3.5))]
