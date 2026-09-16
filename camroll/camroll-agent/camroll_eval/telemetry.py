"""Read-only sampling of the authorized GPU and this evaluator process."""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path
import subprocess
import threading
import time

from .common import append_jsonl, now_iso

GPU_UUID = "GPU-4ef9200c-d8f3-638a-3eeb-26fe4fa3d7b0"


def verified_wemm_pids(processes, proc_root=Path("/proc"), service_path=None):
    """Match a live service ancestry and cwd; never infer ownership from GPU ID."""
    import json
    project = Path(os.getenv("CAMROLL_EVAL_WEMM_ROOT", "/home/realmagic/wemm-embedding-lab")).resolve()
    service_path = Path(service_path or project / "run/service.json")
    try:
        service = json.loads(service_path.read_text())
        service_pid = int(service["pid"])
        if (proc_root / str(service_pid) / "cwd").resolve() != project:
            return []
        matched = []
        for row in processes:
            pid = int(row["pid"])
            if (proc_root / str(pid) / "cwd").resolve() != project:
                continue
            current, seen = pid, set()
            while current > 1 and current not in seen:
                if current == service_pid:
                    matched.append(pid)
                    break
                seen.add(current)
                # /proc comm may contain spaces/parentheses; parent PID follows
                # the closing comm parenthesis and the process state field.
                raw = (proc_root / str(current) / "stat").read_text()
                current = int(raw[raw.rfind(")") + 2:].split()[1])
        return matched
    except (OSError, ValueError, KeyError, IndexError):
        return []


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sample_resources():
    result = {"observed_at": now_iso(), "epoch": time.time(), "gpu_uuid": GPU_UUID,
              "cloud_gpu_memory_gib": None, "evaluator_pid": os.getpid()}
    fields = ["memory.used", "utilization.gpu", "temperature.gpu", "power.draw", "clocks.sm"]
    try:
        raw = subprocess.run(["nvidia-smi", "--id=" + GPU_UUID, "--query-gpu=" + ",".join(fields),
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4, check=True).stdout
        values = next(csv.reader(io.StringIO(raw)))
        keys = ["gpu_memory_mib", "gpu_utilization_percent", "gpu_temperature_c", "gpu_power_w", "gpu_sm_clock_mhz"]
        result.update({key: number(value.strip()) for key, value in zip(keys, values)})
        raw = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4, check=True).stdout
        processes = []
        for row in csv.reader(io.StringIO(raw)):
            if len(row) == 3 and row[0].strip() == GPU_UUID:
                processes.append({"pid": int(row[1]), "gpu_memory_mib": number(row[2].strip())})
        result["gpu_processes"] = processes
        verified = verified_wemm_pids(processes)
        result["wemm_verified_pids"] = verified
        owned_memory = [r["gpu_memory_mib"] for r in processes if r["pid"] in verified and r["gpu_memory_mib"] is not None]
        result["wemm_process_memory_mib"] = sum(owned_memory) if owned_memory else None
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration):
        result["gpu_status"] = "unavailable"
    try:
        mem = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
               if line.split(":")[0] in {"MemTotal", "MemAvailable"}}
        result["host_ram_used_mib"] = (mem["MemTotal"] - mem["MemAvailable"]) / 1024
        process = {line.split(":")[0]: line.split(":")[1].strip()
                   for line in Path("/proc/self/status").read_text().splitlines()}
        result["evaluator_rss_mib"] = int(process["VmRSS"].split()[0]) / 1024
    except (OSError, KeyError, ValueError):
        result["host_ram_status"] = "unavailable"
    return result


class ResourceSampler:
    def __init__(self, path, phase, interval=2):
        self.path, self.phase, self.interval = path, phase, interval
        self._stop = threading.Event()
        self._thread = None
        self._rows = []

    def start(self):
        def loop():
            while not self._stop.is_set():
                row = sample_resources()
                row["phase"] = self.phase()
                self._rows.append(row)
                append_jsonl(self.path, row)
                self._stop.wait(self.interval)
        self._thread = threading.Thread(target=loop, name="camroll-eval-telemetry", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def summary(self):
        def maximum(field):
            values = [row[field] for row in self._rows if isinstance(row.get(field), (int, float))]
            return max(values) if values else None
        return {"samples": len(self._rows), "gpu_uuid": GPU_UUID, "cloud_gpu_memory_gib": None,
                "gpu_memory_mib_peak": maximum("gpu_memory_mib"),
                "wemm_process_memory_mib_peak": maximum("wemm_process_memory_mib"),
                "host_ram_used_mib_peak": maximum("host_ram_used_mib"),
                "evaluator_rss_mib_peak": maximum("evaluator_rss_mib"),
                "gpu_utilization_percent_peak": maximum("gpu_utilization_percent")}
