#!/usr/bin/env python3
"""Frozen PhotoBench image retrieval comparison; never calls a cloud LLM.

Gold labels are used only by this evaluator, after album indexing. Each original
question is encoded once and the identical vector is used by M1 and M3.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
import platform
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "camroll-agent"))
from camroll_agent.library import BRANCH_TOP_K, RRF_K, Library
from camroll_agent.wemm import EmbeddingClient

PROTOCOL = "camroll-photobench-images-m1-m3-v1"
EXPECTED_QUESTIONS = 800
EXPECTED_IMAGES = 334
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260914
APPROVED_HASHES = {
    "freeze.json": "e88634697179b24517e3b7859e59734479fbebc14cebb7bad5df6d31d5d7acad",
    "media.json": "1f8253eb0bed177a98d05e8581d1b7aa875b9c6bc61e87d2c71f212b895aebec",
    "eligible.jsonl": "0a31c31ccda29ff3656de4dee1d3e0664166005bdb4f7f61cfb832e00914f549",
    "results.jsonl": "927a3795d6c2c9029b82fd8fb62331e1e409f3d5216982aa45f349cf1abb43fa",
}
METRICS = ("Hit@1", "Hit@5", "Hit@10", "Hit@20", "Recall@10", "Recall@20",
           "AllCoverage@10", "AllCoverage@20", "MRR@50")
ARMS = ("M1", "M3")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def object_hash(value):
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with open(temporary, "xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def read_jsonl(path):
    with open(path) as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError("Blank row in {}:{}".format(Path(path).name, number))
            yield json.loads(line)


def score_ranking(ranked_assets, gold_assets, *, failed=False):
    """Score unique asset identities, retaining all distinct required targets."""
    gold = set(gold_assets)
    if not gold:
        raise ValueError("Gold target set must not be empty")
    ranking = list(dict.fromkeys(ranked_assets))[:50] if not failed else []
    ranks = {asset: index + 1 for index, asset in enumerate(ranking)}
    result = {}
    for k in (1, 5, 10, 20):
        count = len(gold.intersection(ranking[:k]))
        result["Hit@{}".format(k)] = float(count > 0)
        if k in (10, 20):
            result["Recall@{}".format(k)] = count / len(gold)
            result["AllCoverage@{}".format(k)] = float(count == len(gold))
    first = min((ranks[asset] for asset in gold if asset in ranks), default=None)
    result["MRR@50"] = 1 / first if first else 0.0
    return {key: result[key] for key in METRICS}, {asset: ranks.get(asset) for asset in sorted(gold)}


def load_frozen(source):
    source = Path(source).resolve()
    actual = {name: sha256(source / name) for name in APPROVED_HASHES}
    if actual != APPROVED_HASHES:
        changed = [name for name in actual if actual[name] != APPROVED_HASHES[name]]
        raise ValueError("Approved frozen source hash mismatch: " + ", ".join(changed))
    freeze = json.loads((source / "freeze.json").read_text())
    for name in ("eligible.jsonl", "media.json"):
        if freeze["generated_files"].get(name) != actual[name]:
            raise ValueError("Freeze manifest does not match " + name)
    media = json.loads((source / "media.json").read_text())
    images = {row["sha256"] for row in media if row["kind"] == "image"}
    if len(images) != EXPECTED_IMAGES:
        raise ValueError("Frozen candidate set must contain exactly 334 images")
    baseline = {}
    for row in read_jsonl(source / "results.jsonl"):
        if row.get("track") == "images" and row.get("arm") == "A":
            if row["qa_id"] in baseline:
                raise ValueError("Duplicate frozen baseline QA ID")
            baseline[row["qa_id"]] = row
    questions = []
    for row in read_jsonl(source / "eligible.jsonl"):
        if row.get("cohort") != "main" or row.get("composition") != "image":
            continue
        original = baseline.get(row["qa_id"])
        gold = sorted(set(row["gold_assets"]))
        if (original is None or not gold or not set(gold) <= images
                or set(original["gold_assets"]) != set(gold)
                or original["question"] != row["question"]
                or original["query"] != row["question"]):
            raise ValueError("Frozen question, targets or raw-query baseline mismatch")
        metrics, _ = score_ranking([hit["asset"] for hit in original["top50"]], gold,
                                   failed=original["status"] != "OK")
        for name, value in metrics.items():
            old_name = name.replace("AllCoverage@", "All@")
            if not np.isclose(value, original["metrics"][old_name]):
                raise ValueError("Frozen scorer parity failed: " + old_name)
        questions.append({"qa_id": row["qa_id"], "question": row["question"],
                          "event_id": row["event_id"], "gold_assets": gold,
                          "target_count": len(gold), "task_type": row.get("task_type"),
                          "frozen_eligibility": {"cohort": row["cohort"], "composition": row["composition"],
                              "track": "images", "source_arm": "A", "human_verified": row.get("human_verified", False)},
                          "historical_raw_status": original["status"], "historical_raw_metrics": metrics})
    ids = [row["qa_id"] for row in questions]
    if len(ids) != EXPECTED_QUESTIONS or len(set(ids)) != EXPECTED_QUESTIONS or set(ids) != set(baseline):
        raise ValueError("Main image evaluation must contain the exact 800 frozen QA IDs")
    return sorted(questions, key=lambda row: row["qa_id"]), images, {
        "run_id": freeze["run_id"], "hashes": actual,
        "source_lock_sha256": freeze["source_lock_sha256"], "image_count": len(images),
        "question_count": len(questions), "event_groups": len({q["event_id"] for q in questions}),
    }


def make_lock(source_info, questions, library, source, memory):
    if RRF_K != 60 or BRANCH_TOP_K != 50:
        raise ValueError("Evaluation requires fixed RRF=60 and branch Top-50")
    code = [Path(__file__).resolve(), REPO / "camroll-agent/camroll_agent/library.py",
            REPO / "camroll-agent/camroll_agent/wemm.py"]
    return {"protocol": PROTOCOL, "source": source_info, "source_directory": str(source),
            "memory_directory": str(memory), "question_set_sha256": object_hash(questions),
            "index_version": library.version, "embedding_fingerprint": library.fingerprint,
            "index_hashes": {name: sha256(library.snapshot_dir / name) for name in ("records.json", "vectors.npz")},
            "index_metadata": library.metadata, "code_hashes": {str(p.relative_to(REPO)): sha256(p) for p in code},
            "arms": {"M1": "WeMM image", "M3": "WeMM image + caption, equal RRF"},
            "branch_top_k": BRANCH_TOP_K, "rrf_k": RRF_K, "retrieval_top_k": 50,
            "kind": "images", "query": "original frozen question, no rewrite or date filter",
            "failure_policy": "fixed denominator, failed query/arm contributes zero, never retried by resume",
            "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED,
                          "unit": "frozen event_id cluster", "paired": True, "confidence": 0.95},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "cloud_llm_calls": 0}


def load_resume(output, questions, lock_hash):
    path = output / "results.jsonl"
    if not path.exists():
        return {}
    expected = {q["qa_id"]: q for q in questions}
    complete = {}
    data = path.read_bytes()
    offset = 0
    for part in data.splitlines(keepends=True):
        # Interrupted append: save the incomplete bytes, then continue from the
        # last fully committed newline. Existing complete rows never change.
        if not part.endswith(b"\n"):
            tail = output / ("incomplete-tail-" + uuid.uuid4().hex + ".bin")
            atomic_bytes(tail, part)
            with path.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
            break
        row = json.loads(part)
        qid = row.get("qa_id")
        if qid not in expected or qid in complete or row.get("lock_sha256") != lock_hash:
            raise ValueError("Existing results do not belong to this unique frozen run")
        question = expected[qid]
        if any(row.get(k) != question[k] for k in ("question", "gold_assets", "event_id", "frozen_eligibility")):
            raise ValueError("Existing result changed question eligibility or targets")
        for arm in ARMS:
            value = row["outcomes"][arm]
            if value["status"] not in {"OK", "ERROR"}:
                raise ValueError("Unknown outcome status")
            calculated, ranks = score_ranking([v["asset"] for v in value["top50"]], question["gold_assets"], failed=value["status"] != "OK")
            if value["metrics"] != calculated or value["gold_ranks"] != ranks:
                raise ValueError("Existing result metrics do not match its ranking")
        vector = row.get("query_vector")
        if vector:
            vector_path = output / "query_vectors" / (vector["key"] + ".npy")
            if sha256(vector_path) != vector["sha256"]:
                raise ValueError("Previously scored query vector was changed")
        complete[qid] = row
        offset += len(part)
    return complete


def query_vector(query, embedding, output, dimension):
    key = object_hash({"query": query, "fingerprint": embedding.fingerprint})
    directory = output / "query_vectors"
    directory.mkdir(exist_ok=True)
    path, meta_path = directory / (key + ".npy"), directory / (key + ".json")
    started = time.perf_counter()
    cached = path.exists() and meta_path.exists()
    if cached:
        meta = json.loads(meta_path.read_text())
        if meta["sha256"] != sha256(path) or meta["fingerprint"] != embedding.fingerprint:
            raise ValueError("Query vector cache failed its hash or fingerprint check")
        vector = np.load(path, allow_pickle=False)
    else:
        vector = np.asarray(embedding.encode(text=query), dtype=np.float32)
        elapsed = (time.perf_counter() - started) * 1000
        if vector.shape != (dimension,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("Query vector has invalid dimensions or values")
        buffer = io.BytesIO()
        np.save(buffer, vector, allow_pickle=False)
        atomic_bytes(path, buffer.getvalue())
        timings = {k: v for k, v in getattr(embedding, "last_timings", {}).items()
                   if isinstance(v, (int, float, bool)) or v is None}
        meta = {"key": key, "sha256": sha256(path), "fingerprint": embedding.fingerprint,
                "dimension": dimension, "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "encoding_ms": elapsed, "server_timings": timings}
        atomic_json(meta_path, meta)
    if vector.shape != (dimension,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError("Cached query vector is invalid")
    return vector, {**meta, "cache_hit": cached, "current_load_or_encode_ms": (time.perf_counter() - started) * 1000}


def error_outcome(gold, exc, stage, elapsed_ms=0.0):
    metrics, ranks = score_ranking([], gold, failed=True)
    # Exception details can contain server data; stable types/codes suffice.
    return {"status": "ERROR", "error": {"stage": stage, "type": type(exc).__name__,
               "code": getattr(exc, "code", "evaluation_error")},
            "retrieval_ms": elapsed_ms, "top50": [], "metrics": metrics, "gold_ranks": ranks}


def evaluate_question(question, libraries, embedding, output, lock_hash):
    row = dict(question, lock_sha256=lock_hash, outcomes={})
    started = time.perf_counter()
    try:
        vector, meta = query_vector(question["question"], embedding, output,
                                    libraries["M1"].metadata["dimensions"])
        row["query_vector"] = meta
    except Exception as exc:
        row["query_vector"] = None
        row["query_encoding_ms"] = (time.perf_counter() - started) * 1000
        row["outcomes"] = {arm: error_outcome(question["gold_assets"], exc, "query_encoding") for arm in ARMS}
        return row
    row["query_encoding_ms"] = meta["encoding_ms"]
    for arm in ARMS:
        started = time.perf_counter()
        try:
            result = libraries[arm].search(question["question"], kind="images", top_k=50, query_vector=vector)
            elapsed = (time.perf_counter() - started) * 1000
            hits = []
            for item in result["results"]:
                asset = item["asset_id"]
                if not re.fullmatch(r"[0-9a-f]{64}", asset) or item["kind"] != "image":
                    raise ValueError("Retrieval returned invalid image identity")
                hits.append({"id": item["id"], "asset": asset, "rank": item["rank"],
                             "score": item["score"], "branch_ranks": item.get("branch_ranks", {})})
            if len({hit["asset"] for hit in hits}) != len(hits):
                raise ValueError("Photo ranking contains duplicate assets")
            metrics, ranks = score_ranking([hit["asset"] for hit in hits], question["gold_assets"])
            row["outcomes"][arm] = {"status": "OK", "error": None, "retrieval_ms": elapsed,
                                    "top50": hits, "metrics": metrics, "gold_ranks": ranks}
        except Exception as exc:
            row["outcomes"][arm] = error_outcome(question["gold_assets"], exc, "retrieval", (time.perf_counter() - started) * 1000)
    return row


def paired_bootstrap(rows, *, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    groups = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)
    metrics = list(METRICS) + ["failure_rate"]
    sums, counts = [], []
    for event in sorted(groups):
        group = groups[event]
        values = []
        for row in group:
            delta = [row["outcomes"]["M3"]["metrics"][m] - row["outcomes"]["M1"]["metrics"][m] for m in METRICS]
            delta.append(float(row["outcomes"]["M3"]["status"] != "OK") - float(row["outcomes"]["M1"]["status"] != "OK"))
            values.append(delta)
        sums.append(np.sum(values, axis=0))
        counts.append(len(group))
    sums, counts = np.asarray(sums), np.asarray(counts)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(samples, len(groups)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    point = sums.sum(axis=0) / counts.sum()
    return {metric: {"delta_M3_minus_M1": float(point[n]),
                     "ci95": [float(v) for v in np.quantile(boot[:, n], [0.025, 0.975])]}
            for n, metric in enumerate(metrics)}


def summarize(rows, lock):
    count = len(rows)
    arms = {}
    for arm in ARMS:
        values = [row["outcomes"][arm] for row in rows]
        times = [v["retrieval_ms"] for v in values if v["status"] == "OK"]
        arms[arm] = {"count": count, "failed": sum(v["status"] != "OK" for v in values),
                     "failure_rate": sum(v["status"] != "OK" for v in values) / count,
                     "metrics": {name: sum(v["metrics"][name] for v in values) / count for name in METRICS},
                     "retrieval_ms_success_only": {"p50": float(np.percentile(times, 50)) if times else None,
                                                    "p95": float(np.percentile(times, 95)) if times else None}}
    encode_times = [row["query_encoding_ms"] for row in rows if row.get("query_vector")]
    reference = {name: sum(row["historical_raw_metrics"][name] for row in rows) / count for name in METRICS}
    return {"run_id": lock["run_id"], "complete": count == EXPECTED_QUESTIONS,
            "expected_denominator": EXPECTED_QUESTIONS, "scored_questions": count,
            "event_groups": len({row["event_id"] for row in rows}), "arms": arms,
            "historical_WeMM_raw_reference": reference, "paired_event_bootstrap": paired_bootstrap(rows),
            "query_encoding_ms": {"p50": float(np.percentile(encode_times, 50)) if encode_times else None,
                                    "p95": float(np.percentile(encode_times, 95)) if encode_times else None},
            "common_success_count": sum(all(row["outcomes"][a]["status"] == "OK" for a in ARMS) for row in rows),
            "cloud_llm_calls": 0, "protocol_lock": lock,
            "limitations": ["仅衡量冻结标注证据召回，不评分最终答案；未标注的合理照片不被视为正确目标。",
                            "置信区间按冻结人工事件聚类，不使用Qwen生成的事件分组。",
                            "两路共用原问题向量；M1/M3共享caption与同一索引快照。",
                            "所有已记录失败固定计零，续跑不挑选或覆盖失败；参数未使用这800题调优。"]}


def write_report(output, rows, summary):
    atomic_json(output / "summary.json", summary)
    fields = ["qa_id", "event_id", "target_count", "query_encoding_ms"]
    for arm in ARMS:
        fields += [arm + "_status", arm + "_retrieval_ms"] + [arm + "_" + m for m in METRICS]
    fields += ["delta_" + m for m in METRICS]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        flat = {key: row[key] for key in fields[:4]}
        for arm in ARMS:
            value = row["outcomes"][arm]
            flat[arm + "_status"], flat[arm + "_retrieval_ms"] = value["status"], value["retrieval_ms"]
            flat.update({arm + "_" + m: value["metrics"][m] for m in METRICS})
        flat.update({"delta_" + m: row["outcomes"]["M3"]["metrics"][m] - row["outcomes"]["M1"]["metrics"][m] for m in METRICS})
        writer.writerow(flat)
    atomic_bytes(output / "per_question.csv", buffer.getvalue().encode("utf-8-sig"))
    lock = summary["protocol_lock"]
    lines = ["# CamRoll M1 / M3 冻结图片检索对照", "", "观测时间：" + dt.datetime.now(dt.timezone.utc).isoformat(), "",
             "- 本次 run_id：`" + summary["run_id"] + "`", "- 冻结源：`" + lock["source"]["run_id"] + "`，334张图片、800个主测问题。",
             "- 完成：{}/800；固定失败计零；共同成功：{}。".format(summary["scored_questions"], summary["common_success_count"]),
             "- 索引：`" + lock["index_version"] + "`；WeMM 指纹：`" + lock["embedding_fingerprint"] + "`。",
             "- M1：原图向量；M3：原图＋中文caption文本，等权 RRF=60，每路Top-50。",
             "- 输入：原始中文问题，共用同一个WeMM查询向量，无改写、无日期过滤、无Qwen评测调用。", "",
             "## 结果", "", "| 指标 | M1 | M3 | M3−M1 | 95% 配对事件bootstrap区间 |",
             "|---|---:|---:|---:|---:|"]
    for metric in list(METRICS) + ["failure_rate"]:
        a = summary["arms"]["M1"]
        b = summary["arms"]["M3"]
        av, bv = (a[metric], b[metric]) if metric == "failure_rate" else (a["metrics"][metric], b["metrics"][metric])
        interval = summary["paired_event_bootstrap"][metric]
        lo, hi = interval["ci95"]
        lines.append(f"| {metric} | {av:.6f} | {bv:.6f} | {bv-av:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ["", "Hit：至少召回一个标注目标；Recall：召回的不同目标占全部不同目标的比例；AllCoverage：全部目标均召回；MRR@50：首个目标倒数排名。",
              "", "## 时间与可复核性", "",
              "- 查询编码耗时P50/P95（毫秒）：`" + json_text(summary["query_encoding_ms"]) + "`。"]
    for arm in ARMS:
        lines.append("- " + arm + " 检索耗时P50/P95（仅成功项，毫秒）：`" + json_text(summary["arms"][arm]["retrieval_ms_success_only"]) + "`。")
    lines += ["- bootstrap按冻结event_id整组有放回抽样2000次，seed=20260914，保留事件内相关性；每次仍按问题数加权。",
              "- `protocol_lock.json` 保存四份冻结源、索引、实现源码哈希和模型/预处理/提示词版本；`results.jsonl` 逐题保存目标SHA、Top-50排名、状态和耗时。",
              "- `query_vectors/` 保存各原问题向量与哈希；断点续跑校验后仅执行缺失问题，已记录失败不会重跑。",
              "", "## 解释边界", ""]
    lines += ["- " + text for text in summary["limitations"]]
    lines += ["- 历史原问题WeMM结果仅作评分器与迁移参照，不能把caption/最终回答模式差异解释为同一条件下的LLM质量改善。", ""]
    atomic_bytes(output / "REPORT.md", "\n".join(lines).encode("utf-8"))


@contextlib.contextmanager
def run_exclusive(output):
    with (output / ".running.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This evaluation output is already in use") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(memory, source_report, output):
    memory, source, output = (Path(p).expanduser().resolve() for p in (memory, source_report, output))
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Evaluation output must be separate from the frozen source")
    if output == memory or output in memory.parents or memory in output.parents:
        raise ValueError("Evaluation output must be separate from album storage")
    questions, source_images, source_info = load_frozen(source)
    embedding = EmbeddingClient()
    libraries = {arm: Library(memory, embedding, config=arm) for arm in ARMS}
    library = libraries["M1"]
    if library.version != libraries["M3"].version:
        raise ValueError("Index publication changed while opening both arms")
    if ({row["asset_id"] for row in library.images.values()} != source_images
            or len(library.images) != EXPECTED_IMAGES
            or library.metadata.get("image_embedding_count") != EXPECTED_IMAGES
            or len(set(library._vector_ids["image"])) != EXPECTED_IMAGES):
        raise ValueError("Album must have all 334 exact frozen image identities and image vectors")
    if embedding.fingerprint != library.fingerprint:
        raise ValueError("Live WeMM and frozen album fingerprints differ")
    signature = make_lock(source_info, questions, library, source, memory)
    output.mkdir(parents=True, exist_ok=True)
    with run_exclusive(output):
        lock_path = output / "protocol_lock.json"
        if lock_path.exists():
            lock = json.loads(lock_path.read_text())
            prior = {k: v for k, v in lock.items() if k not in {"run_id", "created_utc"}}
            if prior != signature:
                raise ValueError("Resume refused: source, model, index, code or protocol lock changed; use a new run directory")
        else:
            if any(p.name != ".running.lock" for p in output.iterdir()):
                raise ValueError("New evaluation output must be empty")
            lock = dict(signature, run_id=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8],
                        created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            atomic_json(lock_path, lock)
        lock_hash = object_hash(lock)
        complete = load_resume(output, questions, lock_hash)
        with (output / "results.jsonl").open("ab") as result_file:
            for question in questions:
                if question["qa_id"] in complete:
                    continue
                row = evaluate_question(question, libraries, embedding, output, lock_hash)
                result_file.write((json_text(row) + "\n").encode("utf-8"))
                result_file.flush()
                os.fsync(result_file.fileno())
                complete[question["qa_id"]] = row
                if len(complete) % 25 == 0:
                    print(json_text({"completed": len(complete), "total": EXPECTED_QUESTIONS,
                                     "failed": {arm: sum(v["outcomes"][arm]["status"] != "OK" for v in complete.values()) for arm in ARMS}}), flush=True)
        rows = [complete[q["qa_id"]] for q in questions]
        summary = summarize(rows, lock)
        write_report(output, rows, summary)
        atomic_json(output / "complete.json", {"run_id": lock["run_id"], "questions": len(rows),
                    "results_sha256": sha256(output / "results.jsonl"), "lock_sha256": lock_hash,
                    "summary_sha256": sha256(output / "summary.json"), "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run(args.memory, args.source_report, args.output)
    print(json_text({"run_id": summary["run_id"], "complete": summary["complete"], "arms": summary["arms"]}))


if __name__ == "__main__":
    main()
