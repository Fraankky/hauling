"""Sample CPU, RAM, GPU, and VRAM while a command runs."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _proc_sample(pid: int) -> tuple[float, float] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return None
    try:
        rss_kib = next(float(line.split()[1]) for line in status if line.startswith("VmRSS:"))
    except StopIteration:
        return None
    cpu_seconds = (float(fields[13]) + float(fields[14])) / os.sysconf("SC_CLK_TCK")
    return cpu_seconds, rss_kib / 1024


def _memory_sample() -> tuple[float, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value, *_ = line.replace(":", "").split()
        values[name] = float(value)
    total = values["MemTotal"] / 1024
    return total, (values["MemTotal"] - values["MemAvailable"]) / 1024


def _gpu_sample(pid: int) -> dict[str, Any] | None:
    try:
        row = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        name, utilization, used, total = (part.strip() for part in row.split(","))
        process_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        process_used = sum(
            float(parts[1].strip())
            for line in process_rows
            if len(parts := line.split(",")) == 2 and parts[0].strip() == str(pid)
        )
        return {
            "name": name,
            "utilization_percent": float(utilization),
            "device_memory_used_mib": float(used),
            "device_memory_total_mib": float(total),
            "process_memory_used_mib": process_used,
        }
    except (FileNotFoundError, IndexError, subprocess.CalledProcessError, ValueError):
        return None


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"average": round(sum(values) / len(values), 2), "peak": round(max(values), 2)}


def _summary(
    samples: list[dict[str, Any]], duration: float, logical_cores: int, interval_seconds: float
) -> dict[str, Any]:
    cpu = [sample["cpu_percent"] for sample in samples if sample.get("cpu_percent") is not None]
    ram = [sample["process_ram_mib"] for sample in samples if sample.get("process_ram_mib") is not None]
    system_ram = [sample["system_ram_used_mib"] for sample in samples]
    gpu = [sample["gpu"] for sample in samples if sample.get("gpu") is not None]
    raw_cpu = _stats(cpu)
    normalized_cpu = _stats([value / logical_cores for value in cpu])
    first_gpu = gpu[0] if gpu else None
    return {
        "sampling_interval_seconds": interval_seconds,
        "sample_count": len(samples),
        "duration_seconds": round(duration, 3),
        "cpu": {
            "logical_cores": logical_cores,
            "process_percent": raw_cpu,
            "normalized_process_percent": normalized_cpu,
            "note": "Process percent may exceed 100; 200 percent means about two logical cores.",
        },
        "ram": {
            "system_total_mib": round(samples[0]["system_ram_total_mib"], 2) if samples else None,
            "process_mib": _stats(ram),
            "system_used_mib": _stats(system_ram),
        },
        "gpu": {
            "available": first_gpu is not None,
            "name": first_gpu["name"] if first_gpu else None,
            "utilization_percent": _stats([sample["utilization_percent"] for sample in gpu]),
            "process_vram_mib": _stats([sample["process_memory_used_mib"] for sample in gpu]),
            "device_vram_used_mib": _stats([sample["device_memory_used_mib"] for sample in gpu]),
            "device_vram_total_mib": first_gpu["device_memory_total_mib"] if first_gpu else None,
            "note": "GPU utilization and device VRAM are device-wide nvidia-smi samples.",
        },
    }


def run_monitored(
    command: list[str], *, cwd: Path, output_dir: Path, interval_seconds: float = 1.0
) -> subprocess.CompletedProcess:
    """Run a command and persist sampled resource usage beside its output."""
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd)
    logical_cores = os.cpu_count() or 1
    samples: list[dict[str, Any]] = []
    previous_cpu: tuple[float, float] | None = None
    try:
        while process.poll() is None:
            sampled_at = time.monotonic()
            proc = _proc_sample(process.pid)
            total_ram, used_ram = _memory_sample()
            cpu_percent = None
            if proc is not None and previous_cpu is not None:
                elapsed = sampled_at - previous_cpu[0]
                cpu_percent = 100 * (proc[0] - previous_cpu[1]) / elapsed if elapsed > 0 else None
            if proc is not None:
                previous_cpu = sampled_at, proc[0]
            samples.append(
                {
                    "elapsed_seconds": round(sampled_at - started, 3),
                    "cpu_percent": round(cpu_percent, 2) if cpu_percent is not None else None,
                    "process_ram_mib": round(proc[1], 2) if proc is not None else None,
                    "system_ram_total_mib": round(total_ram, 2),
                    "system_ram_used_mib": round(used_ram, 2),
                    "gpu": _gpu_sample(process.pid),
                }
            )
            time.sleep(interval_seconds)
    except BaseException:
        process.terminate()
        process.wait()
        raise
    returncode = process.wait()
    usage = _summary(samples, time.monotonic() - started, logical_cores, interval_seconds)
    usage["samples"] = samples
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resource_usage.json").write_text(json.dumps(usage, indent=2) + "\n", encoding="utf-8")
    summary_path = output_dir / "run_summary.json"
    if summary_path.is_file():
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_summary["resource_usage"] = {key: value for key, value in usage.items() if key != "samples"}
        summary_path.write_text(json.dumps(run_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return subprocess.CompletedProcess(command, returncode)
