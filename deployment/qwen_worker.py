#!/usr/bin/env python3
"""CamRoll's isolated, serialized, offline Qwen retrieval worker."""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional

GPU_UUID = "GPU-4ef9200c-d8f3-638a-3eeb-26fe4fa3d7b0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(os.environ.get("CAMROLL_QWEN_RUNTIME", "/home/realmagic/camroll-wemm/runtime/qwen-retrieval")).resolve()
SOURCE_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
MODEL_ROOT = Path("/home/realmagic/weknora-album-lab/models")
os.environ["HF_HOME"] = str(ROOT / "hf-cache")
os.environ["HF_MODULES_CACHE"] = str(ROOT / "hf-modules")
os.environ["TORCH_HOME"] = str(ROOT / "torch-cache")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

LOGGER = logging.getLogger("camroll.qwen_worker")
TEXT_LIMIT = 2048
SOFT_CHUNK_LIMIT = 2048
MIN_PIXELS = 4096
MAX_PIXELS = 1843200
TOTAL_LIMITS = {"qwen": 8192, "reranker": 10240}
DIMENSIONS = {"qwen": 2048}
PATHS = {
    "qwen": MODEL_ROOT / "Qwen3-VL-Embedding-2B",
    "reranker": MODEL_ROOT / "Qwen3-VL-Reranker-2B",
}
ALLOWED_IMAGE_ROOTS = tuple(
    Path(value).resolve() for value in json.loads(os.environ.get(
        "CAMROLL_QWEN_IMAGE_ROOTS", json.dumps([str(ROOT / "media"), str(ROOT / "fixtures")])
    ))
)
QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."
DOCUMENT_INSTRUCTION = "Represent the user's input."
RERANK_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
RERANK_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)
REVISION = "camroll-qwen-worker-v1"
MANIFEST_SHA256 = "f6ef1a9b61c70cfe836e836e510ace33f40bc9def99e5e8e4417630e53fc68ba"


class WorkerError(Exception):
    def __init__(self, category: str, message: str, status: int = 400, **details):
        self.category, self.message, self.status, self.details = category, message, status, details
        super().__init__(message)


class BaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EncodeRequest(BaseRequest):
    model: Literal["qwen"] = "qwen"
    texts: Optional[list[str]] = None
    images: Optional[list[str]] = None
    role: Literal["query", "document"]
    cache_namespace: str
    dimension: Optional[int] = None
    trace_input_ids: bool = False


class Candidate(BaseRequest):
    id: str = Field(min_length=1)
    type: Literal["image", "text"]
    image_path: Optional[str] = None
    caption: str = ""
    visible_text: str = ""
    text: str = ""

    @model_validator(mode="after")
    def check_modality(self):
        if not self.id.strip():
            raise ValueError("Candidate id must not be blank")
        if self.type == "image" and (not self.image_path or self.text):
            raise ValueError("Image candidates require image_path and use caption/visible_text")
        if self.type == "text" and (self.image_path or self.caption or self.visible_text or not self.text.strip()):
            raise ValueError("Text candidates require text exclusively")
        return self


class RerankRequest(BaseRequest):
    query: str
    candidates: list[Candidate] = Field(min_length=1, max_length=30)
    cache_namespace: str
    trace_input_ids: bool = False


class TokenizeRequest(BaseRequest):
    model: Literal["qwen", "reranker"] = "qwen"
    texts: list[str] = Field(min_length=1, max_length=32)
    role: Literal["query", "document"] = "document"
    trace_input_ids: bool = False
    cache_namespace: str


class SplitRequest(BaseRequest):
    model: Literal["qwen", "reranker"] = "qwen"
    text: str
    cache_namespace: str


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def namespace_checked(namespace: str, trace: bool = False) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", namespace):
        raise WorkerError("invalid_namespace", "cache_namespace must contain only letters, digits, '_' or '-'.")
    if trace and not namespace.startswith("acceptance"):
        raise WorkerError("trace_requires_acceptance_namespace", "Full input_ids are limited to synthetic acceptance requests.")
    return namespace


def request_items(request: EncodeRequest) -> tuple[str, list[str]]:
    namespace_checked(request.cache_namespace, request.trace_input_ids)
    if (request.texts is None) == (request.images is None):
        raise WorkerError("invalid_input", "Provide exactly one of texts or images.")
    modality, values = ("text", request.texts) if request.texts is not None else ("image", request.images)
    if not values or len(values) > 32:
        raise WorkerError("invalid_batch", "Batch size must be between 1 and 32.")
    if request.dimension is not None and request.dimension != DIMENSIONS[request.model]:
        raise WorkerError("dimension_mismatch", "Worker returns the complete native embedding; truncate and renormalize in caller.",
                          expected=DIMENSIONS[request.model])
    return modality, values


def validate_image_path(value: str) -> Path:
    try:
        path = Path(value).resolve(strict=True)
    except (OSError, RuntimeError):
        raise WorkerError("image_not_found", "Image is missing or cannot be resolved.")
    if not any(path.is_relative_to(root) for root in ALLOWED_IMAGE_ROOTS):
        raise WorkerError("image_path_forbidden", "Image must be inside an authorized project media directory.", 403)
    if not path.is_file():
        raise WorkerError("invalid_image", "Image reference must resolve to a regular file.")
    return path


def read_image(value: str):
    from PIL import Image, ImageOps, UnidentifiedImageError
    path = validate_image_path(value)
    try:
        raw = path.read_bytes()
        with Image.open(path) as verify_image:
            verify_image.verify()
        with Image.open(path) as original:
            normalized = ImageOps.exif_transpose(original).convert("RGB").copy()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise WorkerError("invalid_image", f"Image decoding failed ({type(exc).__name__}); no text-only fallback.")
    return normalized, {
        "input_sha256": sha_bytes(raw),
        "oriented_rgb_sha256": sha_bytes(
            f"{normalized.width}x{normalized.height}:RGB:".encode() + normalized.tobytes()
        ),
        "source_width": normalized.width,
        "source_height": normalized.height,
    }


def check_content(processor, text: str) -> int:
    if not isinstance(text, str) or not text:
        raise WorkerError("invalid_text", "Text must be a non-empty string.")
    if len(text) > 1_000_000:
        raise WorkerError("text_too_large", "Text exceeds the pre-tokenization character safety limit.", 413)
    count = len(processor.tokenizer(text, add_special_tokens=False)["input_ids"])
    if count > TEXT_LIMIT:
        raise WorkerError("content_length_exceeded", "Text must be explicitly chunked by caller; silent truncation is disabled.",
                          413, content_tokens=count, content_hard_limit=TEXT_LIMIT)
    return count


def split_lossless(processor, text: str, limit: int = TEXT_LIMIT) -> list[str]:
    if not isinstance(text, str) or len(text) > 1_000_000:
        raise WorkerError("invalid_text", "Text must be a string of at most 1000000 characters.")
    chunks = []
    remaining = text
    while remaining:
        encoded = processor.tokenizer(remaining, add_special_tokens=False, return_offsets_mapping=True)
        if len(encoded["input_ids"]) <= limit:
            chunks.append(remaining)
            break
        offsets = encoded["offset_mapping"]
        boundary = max(1, int(offsets[limit - 1][1]))
        while boundary and len(processor.tokenizer(remaining[:boundary], add_special_tokens=False)["input_ids"]) > limit:
            boundary -= 1
        if boundary == 0:
            raise WorkerError("unsplittable_text", "A single Unicode character exceeds the token budget.", 413)
        chunks.append(remaining[:boundary])
        remaining = remaining[boundary:]
    if "".join(chunks) != text:
        raise WorkerError("split_integrity_failure", "Tokenizer split did not preserve input.", 500)
    return chunks


def image_content(image):
    return {"type": "image", "image": image, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}


def messages_for(model: str, text: str | None, image=None, role="document"):
    content = []
    if image is not None:
        content.append(image_content(image))
    if text is not None:
        content.append({"type": "text", "text": text})
    instruction = QUERY_INSTRUCTION if role == "query" else DOCUMENT_INSTRUCTION
    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]


def rerank_messages(query: str, document: str, image=None):
    contents = [
        {"type": "text", "text": "<Instruct>: " + RERANK_INSTRUCTION},
        {"type": "text", "text": "<Query>:"},
        {"type": "text", "text": query},
        {"type": "text", "text": "\n<Document>:"},
    ]
    if image is not None:
        contents.append(image_content(image))
    if document:
        contents.append({"type": "text", "text": document})
    return [
        {"role": "system", "content": [{"type": "text", "text": RERANK_SYSTEM}]},
        {"role": "user", "content": contents},
    ]


class Runtime:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = threading.BoundedSemaphore(8)
        self.processors: dict[str, Any] = {}
        self.fingerprints: dict[str, str] = {}
        self.loaded_name: str | None = None
        self.loaded: Any = None
        self.gpu_enabled = os.environ.get("CAMROLL_QWEN_GPU_ENABLED") == "1"
        self.cache_root = ROOT / "vectors"
        self.telemetry_path = ROOT / "telemetry.jsonl"
        self.verified_files: dict[str, dict] = {}

    @contextlib.contextmanager
    def serialized(self):
        start = time.monotonic()
        if not self.pending.acquire(blocking=False):
            raise WorkerError("queue_full", "Model queue is full.", 429)
        acquired = False
        try:
            acquired = self.lock.acquire(timeout=900)
            if not acquired:
                raise WorkerError("queue_timeout", "Timed out waiting for the model queue.", 503)
            yield (time.monotonic() - start) * 1000
        finally:
            if acquired:
                self.lock.release()
            self.pending.release()

    def processor(self, model: str):
        if model not in self.processors:
            path = PATHS[model]
            if not (path / "config.json").is_file():
                raise WorkerError("model_unavailable", f"Local {model} model snapshot is incomplete.", 503)
            from transformers import AutoProcessor
            self.processors[model] = AutoProcessor.from_pretrained(
                str(path), trust_remote_code=False, local_files_only=True, use_fast=True
            )
            self.processors[model].tokenizer.padding_side = "left" if model == "reranker" else "right"
        return self.processors[model]

    def fingerprint(self, model: str) -> str:
        if model not in self.fingerprints:
            manifest = {"model_files": self.verify_model(model)}
            manifest["worker_source_sha256"] = sha_bytes(Path(__file__).read_bytes())
            vendor = SOURCE_ROOT / "vendor/qwen_vl/SOURCE_LOCK.json"
            if vendor.is_file():
                manifest["vendor_lock"] = sha_bytes(vendor.read_bytes())
                manifest["vendor_runtime_files"] = {p.name: sha_bytes(p.read_bytes()) for p in sorted(vendor.parent.glob("*.py"))}
                compatibility = vendor.parent / "COMPATIBILITY_PATCH.json"
                if compatibility.is_file():
                    manifest["compatibility_patch"] = sha_bytes(compatibility.read_bytes())
            self.fingerprints[model] = sha_json({
                "model": model, "files": manifest, "dtype": "bfloat16", "attention": "sdpa",
                "text_limit": TEXT_LIMIT, "total_limit": TOTAL_LIMITS[model],
                "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "processor_use_fast": True,
                "query_instruction": QUERY_INSTRUCTION, "document_instruction": DOCUMENT_INSTRUCTION,
                "rerank_instruction": RERANK_INSTRUCTION, "rerank_system": RERANK_SYSTEM,
                "chunking": "lossless_unicode_offsets_2048", "role_separation": True,
            })
        return self.fingerprints[model]

    def verify_model(self, model: str) -> dict:
        if model in self.verified_files:
            return self.verified_files[model]
        try:
            raw = (MODEL_ROOT / "MODEL_WORKER_DOWNLOAD_MANIFEST.json").read_bytes()
            if sha_bytes(raw) != MANIFEST_SHA256:
                raise ValueError("download manifest hash changed")
            entry = json.loads(raw)["models"]["Qwen/" + PATHS[model].name]
            if not entry["complete"] or not entry["files"]:
                raise ValueError("incomplete download manifest")
            verified = {}
            for item in entry["files"]:
                path = (PATHS[model] / item["path"]).resolve(strict=True)
                if not path.is_relative_to(PATHS[model].resolve()) or path.stat().st_size != item["size"]:
                    raise ValueError("model file path or size mismatch")
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError("model checksum mismatch: " + item["path"])
                verified[item["path"]] = {"sha256": digest.hexdigest(), "size": item["size"]}
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise WorkerError("model_unverified", "Locked model verification failed: " + str(error), 503) from None
        self.verified_files[model] = verified
        self.log({"operation": "model_sha256_audit", "model": model, "files": verified,
                  "download_manifest_sha256": MANIFEST_SHA256})
        return verified

    def prepare(self, model: str, messages, content_tokens: int, expected_images: int, trace=False):
        import torch
        from qwen_vl_utils import process_vision_info
        processor = self.processor(model)
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            images, videos, video_kwargs = process_vision_info(
                messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
            )
        except Exception as exc:
            raise WorkerError("vision_preprocessing_failed", f"Visual preprocessing failed ({type(exc).__name__}); no fallback.")
        if videos:
            raise WorkerError("video_not_supported", "This photo worker accepts still images only.")
        inputs = processor(
            text=[prompt], images=images, videos=None, truncation=False,
            padding=True, do_resize=False, return_tensors="pt", **video_kwargs
        )
        ids = inputs["input_ids"][0]
        mask = inputs["attention_mask"][0]
        valid = ids[mask.bool()].tolist()
        if len(valid) > TOTAL_LIMITS[model]:
            raise WorkerError("total_length_exceeded", "Template plus visual/text tokens exceed the frozen total budget.",
                              413, effective_input_tokens=len(valid), model_total_limit=TOTAL_LIMITS[model])
        grid = inputs.get("image_grid_thw")
        actual_images = int(grid.shape[0]) if grid is not None else 0
        if actual_images != expected_images or (expected_images and inputs.get("pixel_values") is None):
            raise WorkerError("image_tensor_mismatch", "Requested images do not match model image tensors.", 500)
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        pooling_position = int(positions[-1].item())
        telemetry = {
            "content_tokens": content_tokens, "soft_chunk_limit": SOFT_CHUNK_LIMIT,
            "exceeds_soft_chunk_limit": content_tokens > SOFT_CHUNK_LIMIT,
            "template_total_tokens": len(valid), "effective_input_tokens": int(mask.sum().item()),
            "input_ids_sha256": sha_json(valid), "input_ids_tail": valid[-12:],
            "pooling_position": pooling_position, "pooling_token_id": valid[-1],
            "image_count": actual_images,
            "image_grid_thw": grid.tolist() if grid is not None else [],
            "processed_pixels": int(grid[:, 0].mul(grid[:, 1]).mul(grid[:, 2]).sum().item() * 16 * 16)
                if grid is not None else 0,
            "truncated": False, "padding_side": processor.tokenizer.padding_side,
        }
        if expected_images:
            pixels = inputs["pixel_values"].detach().cpu().contiguous()
            telemetry["pixel_values_sha256"] = sha_bytes(pixels.numpy().tobytes())
            telemetry["pixel_values_shape"] = list(pixels.shape)
            telemetry["pixel_values_dtype"] = str(pixels.dtype)
        if trace:
            telemetry["input_ids"] = valid
            telemetry["attention_mask"] = mask.tolist()
        return inputs, telemetry

    def _unload(self) -> dict:
        old = self.loaded_name
        self.loaded = None
        self.loaded_name = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return {"unloaded": old, "pid": os.getpid(), "scope": "this_process_only"}

    def load(self, model: str) -> float:
        if not self.gpu_enabled:
            raise WorkerError("GPU_WINDOW_CLOSED", "GPU loading is disabled; root must open the authorized GPU1 window.", 503)
        if self.loaded_name == model:
            return 0.0
        self.verify_model(model)
        info = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        if not any(row.strip() == "1, " + GPU_UUID for row in info):
            raise WorkerError("gpu_identity_mismatch", "Physical GPU1 UUID does not match the authorized GPU.", 503)
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        for row in processes:
            fields = [field.strip() for field in row.split(",")]
            if len(fields) == 2 and fields[0] == GPU_UUID and fields[1] != str(os.getpid()):
                raise WorkerError("gpu_in_use", "Authorized GPU1 is occupied by another process; no model was loaded.", 503)
        self._unload()
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise WorkerError("gpu_unavailable", "Expected one visible CUDA device mapped to the authorized GPU1.", 503)
        started = time.monotonic()
        kwargs = dict(local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        try:
            if model == "qwen":
                from vendor.qwen_vl.qwen3_vl_embedding import Qwen3VLEmbedder
                loaded = Qwen3VLEmbedder(
                    str(PATHS[model]), max_length=TOTAL_LIMITS[model],
                    min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, **kwargs
                )
            else:
                from vendor.qwen_vl.qwen3_vl_reranker import Qwen3VLReranker
                loaded = Qwen3VLReranker(
                    str(PATHS[model]), max_length=TOTAL_LIMITS[model],
                    min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, **kwargs
                )
            self.loaded = loaded
            self.loaded_name = model
            torch.cuda.synchronize()
        except Exception:
            self._unload()
            raise
        return (time.monotonic() - started) * 1000

    def reset_independent_state(self, model: str):
        # Each request/item is a fresh sequence, never generation continuation.
        # Transformers 5.2 can reuse a previous image's rope_deltas for text-only
        # Qwen3-VL calls. Reset only our own loaded module; official vendor code
        # and the read-only legacy WeMM snapshot remain unchanged.
        if model == "qwen":
            self.loaded.model.model.rope_deltas = None
        elif model == "reranker":
            self.loaded.model.rope_deltas = None

    def run_embedding(self, model: str, inputs):
        import torch
        import torch.nn.functional as functional
        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            self.reset_independent_state(model)
            outputs = self.loaded.forward(inputs)
            result = self.loaded._pooling_last(
                outputs["last_hidden_state"], outputs["attention_mask"]
            ).float()
            result = functional.normalize(result, p=2, dim=-1)
        torch.cuda.synchronize()
        values = result[0].cpu().tolist()
        if len(values) != DIMENSIONS[model] or not all(math.isfinite(x) for x in values):
            raise WorkerError("invalid_embedding", "Model returned wrong dimensions or non-finite values.", 500)
        norm = math.sqrt(sum(x*x for x in values))
        if not 0.999 <= norm <= 1.001:
            raise WorkerError("invalid_embedding_norm", "Model returned a non-unit embedding.", 500)
        return values, {
            "inference_ms": (time.monotonic() - started) * 1000,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "dimension": len(values), "norm": norm,
        }

    def log(self, data: dict):
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.telemetry_path.open("a") as fh:
            fh.write(json.dumps({"time": time.time(), "pid": os.getpid(), **data}, ensure_ascii=False) + "\n")

    def encode(self, request: EncodeRequest):
        started = time.monotonic()
        modality, values = request_items(request)
        with self.serialized() as queue_ms:
            processor = self.processor(request.model)
            fp = self.fingerprint(request.model)
            # Preflight every input before any model inference or cache publication.
            prepared = []
            for value in values:
                metadata = {}
                if modality == "text":
                    count = check_content(processor, value)
                    metadata["input_sha256"] = sha_bytes(value.encode())
                    messages = messages_for(request.model, value, role=request.role)
                else:
                    image, metadata = read_image(value)
                    count = 0
                    messages = messages_for(request.model, None, image=image, role=request.role)
                inputs, token_info = self.prepare(
                    request.model, messages, count, int(modality == "image"), request.trace_input_ids
                )
                prepared.append((inputs, {**metadata, **token_info}))
            embeddings, rows = [], []
            total_load = 0.0
            for index, (inputs, metadata) in enumerate(prepared):
                key = sha_json({
                    "fp": fp, "input": metadata["input_sha256"], "role": request.role,
                    "modality": modality, "input_ids": metadata["input_ids_sha256"],
                    "rgb": metadata.get("oriented_rgb_sha256"),
                })
                cache_path = self.cache_root / request.cache_namespace / (key + ".json")
                cached = None
                if cache_path.is_file():
                    try:
                        cached = json.loads(cache_path.read_text())
                        vector = cached["embedding"]
                        if (len(vector) != DIMENSIONS[request.model]
                                or not all(math.isfinite(x) for x in vector)
                                or cached.get("fingerprint") != fp
                                or not 0.999 <= math.sqrt(sum(x*x for x in vector)) <= 1.001):
                            cached = None
                    except (OSError, ValueError, KeyError, TypeError):
                        cached = None
                if cached is not None:
                    vector = cached["embedding"]
                    timing = {"inference_ms": 0.0, "model_load_ms": 0.0, "cache_hit": True,
                              "dimension": len(vector), "norm": math.sqrt(sum(x*x for x in vector))}
                else:
                    load_ms = self.load(request.model)
                    total_load += load_ms
                    vector, timing = self.run_embedding(request.model, inputs)
                    timing.update(model_load_ms=load_ms, cache_hit=False)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps({"embedding": vector, "fingerprint": fp}))
                    os.replace(temporary, cache_path)
                row = {"index": index, "model": request.model, "role": request.role, "modality": modality,
                       "fingerprint": fp, **metadata, **timing}
                embeddings.append(vector)
                rows.append(row)
            telemetry = {"items": rows, "queue_ms": queue_ms, "model_load_ms": total_load,
                         "inference_ms": sum(x["inference_ms"] for x in rows),
                         "end_to_end_ms": (time.monotonic()-started)*1000,
                         "gpu_uuid": GPU_UUID, "dtype": "bfloat16", "batch_execution": "serial_singleton"}
            self.log({"operation": "encode", "model": request.model, "telemetry": telemetry})
            return {"embeddings": embeddings, "dimension": DIMENSIONS[request.model],
                    "fingerprint": fp, "telemetry": telemetry}

    def tokenize(self, request: TokenizeRequest):
        namespace_checked(request.cache_namespace, request.trace_input_ids)
        with self.serialized() as queue_ms:
            processor = self.processor(request.model)
            results = []
            for text in request.texts:
                count = check_content(processor, text)
                # For reranker, report text-token counts without claiming a pair was validated.
                if request.model == "reranker":
                    ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
                    row = {"content_tokens": count, "input_ids_sha256": sha_json(ids),
                           "soft_chunk_limit": SOFT_CHUNK_LIMIT, "pair_validation": "use /rerank"}
                    if request.trace_input_ids:
                        row["content_input_ids"] = ids
                else:
                    _, row = self.prepare(request.model, messages_for(request.model, text, role=request.role),
                                          count, 0, request.trace_input_ids)
                results.append(row)
            return {"model": request.model, "items": results, "body_hard_limit": TEXT_LIMIT,
                    "soft_chunk_limit": SOFT_CHUNK_LIMIT, "queue_ms": queue_ms,
                    "fingerprint": self.fingerprint(request.model)}

    def rerank(self, request: RerankRequest):
        namespace_checked(request.cache_namespace, request.trace_input_ids)
        if len({candidate.id for candidate in request.candidates}) != len(request.candidates):
            raise WorkerError("duplicate_candidate_id", "Candidate IDs must be unique.")
        started = time.monotonic()
        with self.serialized() as queue_ms:
            processor = self.processor("reranker")
            query_tokens = check_content(processor, request.query)
            prepared = []
            for candidate in request.candidates:
                text = ("\n\n".join(part for part in (candidate.caption, candidate.visible_text) if part)
                        if candidate.type == "image" else candidate.text)
                chunks = split_lossless(processor, text) or [""]
                image, image_meta = read_image(candidate.image_path) if candidate.type == "image" else (None, {})
                prepared.append((candidate, chunks, image, image_meta))
            fingerprint = self.fingerprint("reranker")
            load_ms = 0.0
            scores, rows = [], []
            for index, (candidate, chunks, image, image_meta) in enumerate(prepared):
                chunk_scores = []
                for chunk_index, chunk in enumerate(chunks):
                    document_tokens = check_content(processor, chunk) if chunk else 0
                    inputs, token_info = self.prepare(
                        "reranker", rerank_messages(request.query, chunk, image),
                        query_tokens + document_tokens, int(image is not None), request.trace_input_ids,
                    )
                    load_ms += self.load("reranker")
                    value, timing = self.run_rerank(inputs)
                    chunk_scores.append(value)
                    rows.append({"index": index, "id": candidate.id, "type": candidate.type,
                                 "chunk_index": chunk_index, "chunk_count": len(chunks),
                                 "chunk_sha256": sha_bytes(chunk.encode()), "score": value,
                                 "query_content_tokens": query_tokens, "document_content_tokens": document_tokens,
                                 **image_meta, **token_info, **timing})
                scores.append({"index": index, "id": candidate.id, "score": max(chunk_scores),
                               "relevance_score": max(chunk_scores), "chunk_count": len(chunks),
                               "chunk_scores": chunk_scores})
            telemetry = {"items": rows, "queue_ms": queue_ms, "model_load_ms": load_ms,
                         "inference_ms": sum(x["inference_ms"] for x in rows),
                         "end_to_end_ms": (time.monotonic()-started)*1000,
                         "gpu_uuid": GPU_UUID, "dtype": "bfloat16",
                         "fingerprint": fingerprint, "cache_hit": False,
                         "scoring": "official_sigmoid_yes_minus_no", "batch_execution": "serial_singleton"}
            self.log({"operation": "rerank", "telemetry": telemetry})
            return {"scores": scores, "fingerprint": fingerprint, "telemetry": telemetry}

    def run_rerank(self, inputs):
        import torch
        inputs = {key: value.to("cuda:0") if hasattr(value, "to") else value for key, value in inputs.items()}
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        with torch.inference_mode():
            self.reset_independent_state("reranker")
            values = self.loaded.compute_scores(inputs)
        torch.cuda.synchronize()
        if len(values) != 1 or not math.isfinite(values[0]) or not 0 <= values[0] <= 1:
            raise WorkerError("invalid_rerank_score", "Reranker returned invalid scores.", 500)
        return float(values[0]), {"inference_ms": (time.monotonic() - tick) * 1000,
                                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                 "peak_reserved_bytes": torch.cuda.max_memory_reserved()}

    def split_text(self, request: SplitRequest):
        namespace_checked(request.cache_namespace)
        with self.serialized() as queue_ms:
            processor = self.processor(request.model)
            chunks = split_lossless(processor, request.text)
            counts = [len(processor.tokenizer(chunk, add_special_tokens=False)["input_ids"]) for chunk in chunks]
            return {"chunks": chunks, "token_counts": counts, "limit": TEXT_LIMIT,
                    "fingerprint": self.fingerprint(request.model), "queue_ms": queue_ms}

    def health(self):
        return {"status": "ok", "worker_revision": REVISION, "pid": os.getpid(),
                "gpu_uuid": GPU_UUID, "gpu_loading_enabled": self.gpu_enabled,
                "loaded_model": self.loaded_name, "offline": True,
                "snapshots": {name: (path / "config.json").is_file() for name, path in PATHS.items()},
                "text_hard_limit": TEXT_LIMIT, "soft_chunk_limit": SOFT_CHUNK_LIMIT,
                "max_pixels": MAX_PIXELS}


runtime = Runtime()
app = FastAPI(title="CamRoll Qwen retrieval", docs_url=None, redoc_url=None)


@app.exception_handler(WorkerError)
async def worker_error_handler(request: Request, error: WorkerError):
    return JSONResponse(status_code=error.status, content={
        "error": error.category, "message": error.message, **error.details
    })


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, error: Exception):
    LOGGER.exception("Model operation failed: %s", type(error).__name__)
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(error, torch.cuda.OutOfMemoryError):
        with runtime.serialized():
            runtime._unload()
        return JSONResponse(status_code=507, content={"error": "gpu_out_of_memory", "message": "GPU inference failed; only this worker model was unloaded."})
    return JSONResponse(status_code=500, content={
        "error": "model_operation_failed", "message": type(error).__name__
    })


@app.get("/health")
def health():
    return runtime.health()


@app.post("/encode")
def encode(request: EncodeRequest):
    return runtime.encode(request)


@app.post("/tokenize")
def tokenize(request: TokenizeRequest):
    return runtime.tokenize(request)


@app.post("/split_text")
def split_text(request: SplitRequest):
    return runtime.split_text(request)


@app.get("/fingerprints")
def fingerprints():
    with runtime.serialized():
        return {"fingerprint": runtime.fingerprint("qwen"),
                "reranker_fingerprint": runtime.fingerprint("reranker"),
                "dimension": 2048, "worker_revision": REVISION}


@app.post("/rerank")
def rerank(request: RerankRequest):
    return runtime.rerank(request)


@app.post("/unload")
def unload():
    with runtime.serialized() as queue_ms:
        return {**runtime._unload(), "queue_ms": queue_ms}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18778, workers=1, access_log=False)
