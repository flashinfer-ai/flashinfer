"""Continuous allocation telemetry, independent of pytest and CUDA imports.

CSV streams are flushed after every sample so killed jobs retain observations.
An atomic GPU process snapshot lets all pytest workers share one driver query.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any


# The shell helper executes this file directly, outside package import mode.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.test_sharding.host_cpu import (
    HostCpuTimes,
    host_cpu_percentages,
    read_host_cpu_times,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _number(value: str) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except ValueError:
        return None


def host_memory(proc: Path = Path("/proc")) -> dict[str, Any]:
    values = {}
    for line in _read(proc / "meminfo").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    total, available = values["MemTotal"], values.get("MemAvailable")
    used = None if available is None else max(0, total - available)
    return {
        "memory_total_bytes": total,
        "memory_available_bytes": available,
        "memory_used_bytes": used,
        "memory_used_percent": 100 * used / total
        if used is not None and total
        else None,
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_used_bytes": values.get("SwapTotal", 0) - values.get("SwapFree", 0),
    }


def cgroup_paths(proc: Path = Path("/proc")) -> list[tuple[int, Path]]:
    """Resolve v1/v2 memory membership against mounts, including namespaces."""
    memberships = {}
    for line in _read(proc / "self/cgroup").splitlines():
        _, controllers, path = line.split(":", 2)
        if controllers == "" or "memory" in controllers.split(","):
            memberships[2 if controllers == "" else 1] = PurePosixPath(path)
    for line in _read(proc / "self/mountinfo").splitlines():
        left, right = line.split(" - ", 1)
        fields, filesystem = left.split(), right.split()
        version = 2 if filesystem[0] == "cgroup2" else 1
        if version not in memberships or (
            version == 1
            and (filesystem[0] != "cgroup" or "memory" not in filesystem[2].split(","))
        ):
            continue
        root = PurePosixPath(fields[3].replace("\\040", " "))
        mount = Path(fields[4].replace("\\040", " "))
        member = memberships[version]
        namespace_relative = False
        try:
            relative = member.relative_to(root)
        except ValueError:
            relative = member.relative_to("/")
            namespace_relative = True
        if ".." in relative.parts:
            continue
        current = mount / relative
        if namespace_relative:
            # Namespaced membership can be relative to a bind mount's root.
            # Require direct evidence before attributing that subtree to us.
            try:
                if str(os.getpid()) not in _read(current / "cgroup.procs").split():
                    continue
            except OSError:
                continue
        memory_file = "memory.current" if version == 2 else "memory.usage_in_bytes"
        if not current.is_dir():
            continue
        result = []
        while current.is_relative_to(mount):
            if (current / memory_file).exists():
                result.append((version, current))
            if current == mount:
                break
            current = current.parent
        if result:
            return result
    return []


def cgroup_memory(version: int, path: Path) -> dict[str, Any]:
    def read_number(name: str) -> int | None:
        try:
            value = _read(path / name)
            return None if value == "max" else int(value)
        except (OSError, ValueError):
            return None

    if version == 2:
        used, limit = read_number("memory.current"), read_number("memory.max")
        peak = read_number("memory.peak")
        swap = read_number("memory.swap.current")
        try:
            events = dict(
                line.split() for line in _read(path / "memory.events").splitlines()
            )
        except (OSError, ValueError):
            events = {}
        oom = int(events["oom"]) if "oom" in events else None
        kills = int(events["oom_kill"]) if "oom_kill" in events else None
        limit_hits = int(events["max"]) if "max" in events else None
    else:
        used, limit = (
            read_number("memory.usage_in_bytes"),
            read_number("memory.limit_in_bytes"),
        )
        if limit is not None and limit >= 1 << 60:
            limit = None  # v1's page-aligned LONG_MAX means unlimited.
        peak = read_number("memory.max_usage_in_bytes")
        memsw = read_number("memory.memsw.usage_in_bytes")
        swap = max(0, memsw - used) if memsw is not None and used is not None else None
        limit_hits, oom, kills = read_number("memory.failcnt"), None, None
    return {
        "version": version,
        "path": str(path),
        "memory_used_bytes": used,
        "memory_limit_bytes": limit,
        "memory_used_percent": 100 * used / limit
        if used is not None and limit
        else None,
        "memory_peak_bytes": peak,
        "swap_used_bytes": swap,
        "limit_hits": limit_hits,
        "oom_events": oom,
        "oom_kills": kills,
    }


def _query_gpu(query: str) -> str:
    command = ["nvidia-smi", query, "--format=csv,noheader,nounits"]
    # nvidia-smi does not honor CUDA_VISIBLE_DEVICES itself. Never query
    # unallocated devices when the caller supplies an explicit allocation.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        if not visible.strip() or visible.strip() == "-1":
            return ""
        command.append(f"--id={visible}")
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=5, check=False
    )
    if result.returncode:
        raise OSError(
            f"nvidia-smi exited {result.returncode}: {result.stderr.strip()[:300]}"
        )
    return result.stdout


def gpu_samples() -> list[dict[str, Any]]:
    result = []
    output = _query_gpu(
        "--query-gpu=uuid,index,utilization.gpu,utilization.memory,memory.used,memory.total"
    )
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        uuid, index, utilization, bandwidth, used, total = row
        used_mib, total_mib = _number(used), _number(total)
        result.append(
            {
                "uuid": uuid.strip(),
                "index": index.strip(),
                "gpu_utilization_percent": _number(utilization),
                "memory_bandwidth_utilization_percent": _number(bandwidth),
                "memory_used_mib": used_mib,
                "memory_total_mib": total_mib,
                "memory_used_percent": 100 * used_mib / total_mib
                if used_mib is not None and total_mib
                else None,
            }
        )
    return result


def gpu_process_memory() -> dict[str, float | None]:
    memory: dict[str, float | None] = {}
    for row in csv.reader(
        _query_gpu("--query-compute-apps=pid,used_memory").splitlines(),
        skipinitialspace=True,
    ):
        pid, used = row
        value = _number(used)
        pid = pid.strip()
        previous = memory.get(pid, 0.0)
        # A process may appear on multiple GPUs; any unknown reading makes
        # its total unknown, regardless of the order of observations.
        memory[pid] = None if value is None or previous is None else previous + value
    return memory


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class _CsvStreams(ExitStack):
    def __init__(self, output: Path):
        super().__init__()
        self.output = output
        self.streams: dict[str, Any] = {}
        self.writers: dict[str, Any] = {}

    def write(self, name: str, row: dict[str, Any]) -> None:
        if name not in self.writers:
            stream = self.enter_context(
                (self.output / f"{name}.csv").open("w", newline="", encoding="utf-8")
            )
            self.streams[name] = stream
            self.writers[name] = csv.DictWriter(stream, fieldnames=list(row))
            self.writers[name].writeheader()
        self.writers[name].writerow(row)
        self.streams[name].flush()


def _update_summary(
    summary: dict[str, Any],
    initial_events: dict[str, int],
    name: str,
    row: dict[str, Any],
) -> None:
    identity = str(row.get("uuid", row.get("path", "host")))
    for key, value in row.items():
        if key == "timestamp" or not isinstance(value, (int, float)):
            continue
        metric = f"{name}/{identity}/{key}"
        summary["maxima"][metric] = max(value, summary["maxima"].get(metric, value))
        if key in {"oom_events", "oom_kills", "limit_hits"}:
            initial_events.setdefault(metric, int(value))
            summary.setdefault("event_deltas", {})[metric] = max(
                0, value - initial_events[metric]
            )


class ResourceMonitor:
    """One sampler per wrapper/runner invocation; nested runners reuse it."""

    def __init__(self, output: Path, interval: float = 5.0):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("RESOURCE_MONITOR_INTERVAL must be positive and finite")
        self.output = output
        self.interval = interval
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.previous_cpu: HostCpuTimes | None = None

    def sample_once(self) -> Iterator[tuple[str, dict[str, Any]]]:
        tick = time.monotonic()
        timestamp = time.time()
        host: dict[str, Any] = {"timestamp": timestamp}
        # Missing counters stay blank, with diagnostics in summary.json.
        try:
            host.update(host_memory())
        except (OSError, ValueError, KeyError) as error:
            self.errors[f"host memory: {error}"] += 1
            host.update(
                dict.fromkeys(
                    [
                        "memory_total_bytes",
                        "memory_available_bytes",
                        "memory_used_bytes",
                        "memory_used_percent",
                        "swap_total_bytes",
                        "swap_used_bytes",
                    ]
                )
            )
        current_cpu = read_host_cpu_times()
        if current_cpu is None:
            self.errors["host CPU: counters unavailable"] += 1
        busy, wait = host_cpu_percentages(self.previous_cpu, current_cpu)
        host.update(cpu_busy_percent=busy, cpu_iowait_percent=wait)
        self.previous_cpu = current_cpu
        try:
            loads = os.getloadavg()
        except OSError:
            loads = (None, None, None)
        host.update(zip(["load1", "load5", "load15"], loads, strict=True))
        # Flush observations before a slow driver query can delay or lose them.
        yield "host", host
        for version, path in self.groups:
            try:
                yield "cgroup", {"timestamp": timestamp, **cgroup_memory(version, path)}
            except (OSError, ValueError) as error:
                self.errors[f"cgroup {path}: {error}"] += 1
        try:
            for gpu in gpu_samples():
                yield "gpu", {"timestamp": timestamp, **gpu}
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            self.errors[f"GPU: {error}"] += 1
        try:
            memory = gpu_process_memory()
            snapshot = {
                "timestamp": timestamp,
                "interval_seconds": self.interval,
                "process_memory_mib": memory,
                "error": "",
            }
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            self.errors[f"GPU processes: {error}"] += 1
            snapshot = {
                "timestamp": timestamp,
                "interval_seconds": self.interval,
                "process_memory_mib": None,
                "error": str(error),
            }
        _atomic_json(self.output / "gpu-processes.json", snapshot)
        # Thread time excludes pytest workers even for direct Python calls.
        # Driver subprocess cost is included in sample_wall_seconds.
        yield (
            "sampler",
            {
                "timestamp": timestamp,
                "sample_wall_seconds": time.monotonic() - tick,
                "sampler_cpu_seconds": time.thread_time() - self.started_cpu,
            },
        )

    def run(self, parent_pid: int | None = None) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        started = time.time()
        self.started_cpu = time.thread_time()
        summary: dict[str, Any] = {
            "schema_version": 1,
            "started_at": started,
            "samples": 0,
            "complete": False,
            "interval_seconds": self.interval,
            "maxima": {},
        }
        self.errors: Counter[str] = Counter()
        initial_events: dict[str, int] = {}
        self.previous_cpu = read_host_cpu_times()
        self.groups: list[tuple[int, Path]] = []
        try:
            self.groups = cgroup_paths()
        except (OSError, ValueError, IndexError) as error:
            self.errors[f"initialization: {error}"] += 1
        metadata = {
            "schema_version": 1,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "parent_pid": parent_pid,
            "started_at": started,
            "interval_seconds": self.interval,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cgroups": [str(p) for _, p in self.groups],
        }
        _atomic_json(self.output / "metadata.json", metadata)
        with _CsvStreams(self.output) as streams:
            while True:
                tick = time.monotonic()
                timestamp = time.time()
                for name, row in self.sample_once():
                    streams.write(name, row)
                    _update_summary(summary, initial_events, name, row)
                summary["samples"] += 1
                summary["last_sample_at"] = timestamp
                summary["errors"] = dict(self.errors)
                _atomic_json(self.output / "summary.json", summary)
                if self.stop.is_set():
                    break
                if parent_pid is not None:
                    if os.getppid() != parent_pid:
                        break
                    try:
                        os.kill(parent_pid, 0)
                    except ProcessLookupError:
                        break
                self.stop.wait(max(0, self.interval - (time.monotonic() - tick)))
            summary["complete"] = True
            summary["ended_at"] = time.time()
            _atomic_json(self.output / "summary.json", summary)

    def __enter__(self) -> "ResourceMonitor":
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=12)


def monitor_from_env(junit_dir: Path) -> ResourceMonitor | None:
    if os.environ.get("MONITOR_TEST_MEMORY", "true").lower() in {"0", "false", "no"}:
        return None
    if os.environ.get("FLASHINFER_RESOURCE_SNAPSHOT"):
        try:
            owner_pid = int(os.environ.get("FLASHINFER_RESOURCE_MONITOR_PID", "0"))
            if owner_pid > 0:
                os.kill(owner_pid, 0)
                return None
        except (OSError, ValueError):
            pass  # Recover from an inherited sampler that exited during setup.
    interval = float(os.environ.get("RESOURCE_MONITOR_INTERVAL", "5"))
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("RESOURCE_MONITOR_INTERVAL must be positive and finite")
    # A unique directory preserves concurrent shards, retries and resumed runs.
    root = junit_dir.resolve() / "resource-telemetry"
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    monitor = ResourceMonitor(output, interval)
    os.environ["FLASHINFER_RESOURCE_SNAPSHOT"] = str(output / "gpu-processes.json")
    # Direct Python runs own a sampler thread rather than a child process.
    os.environ["FLASHINFER_RESOURCE_MONITOR_PID"] = str(os.getpid())
    return monitor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    monitor = ResourceMonitor(args.output_dir, args.interval)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: monitor.stop.set())
    monitor.run(args.parent_pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
