#!/usr/bin/env python3
"""
Monitor memory usage for AOT build/import CI steps.

The shell script owns command launch and exit-code handling. This utility owns
the Linux process/cgroup sampling, CSV report writing, and readable summaries.
"""

import argparse
import re
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


CGROUP_FILES = [
    Path("/sys/fs/cgroup/memory.events"),
    Path("/sys/fs/cgroup/memory.current"),
    Path("/sys/fs/cgroup/memory.peak"),
    Path("/sys/fs/cgroup/memory/memory.oom_control"),
    Path("/sys/fs/cgroup/memory/memory.failcnt"),
    Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
]


@dataclass
class MemoryStats:
    peak_rss_kib: int = 0
    peak_cgroup_current_kib: int = 0
    max_job_cgroup_peak_kib: int = 0
    min_system_mem_available_kib: int = 0
    peak_process_count: int = 0
    sample_count: int = 0
    duration_seconds: int = 0


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", label)


def validate_positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(
            f"ERROR: Invalid {name}={value}; expected a positive number"
        ) from None
    if parsed <= 0:
        raise ValueError(f"ERROR: Invalid {name}={value}; expected a positive number")
    return parsed


def validate_positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(
            f"ERROR: Invalid {name}={value}; expected a positive integer"
        ) from None
    if parsed <= 0:
        raise ValueError(f"ERROR: Invalid {name}={value}; expected a positive integer")
    return parsed


def process_exists(pid: int) -> bool:
    if Path("/proc").exists():
        return Path(f"/proc/{pid}").exists()
    return (
        subprocess.run(
            ["kill", "-0", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def read_proc_ppid(stat_file: Path) -> int | None:
    try:
        stat = stat_file.read_text()
    except OSError:
        return None

    end_comm = stat.rfind(")")
    if end_comm == -1:
        return None
    fields_after_comm = stat[end_comm + 2 :].split()
    if len(fields_after_comm) < 2:
        return None
    try:
        return int(fields_after_comm[1])
    except ValueError:
        return None


def list_process_tree_pids_from_proc(root_pid: int) -> list[int]:
    proc = Path("/proc")
    if not proc.exists():
        return []

    alive: set[int] = set()
    children: dict[int, list[int]] = defaultdict(list)
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        ppid = read_proc_ppid(entry / "stat")
        if ppid is None:
            continue
        alive.add(pid)
        children[ppid].append(pid)

    if root_pid not in alive:
        return []

    tree: list[int] = []
    queue: deque[int] = deque([root_pid])
    seen = {root_pid}
    while queue:
        pid = queue.popleft()
        tree.append(pid)
        for child in children.get(pid, []):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return tree


def list_process_tree_pids_from_ps(root_pid: int) -> list[int]:
    result = subprocess.run(
        ["ps", "-e", "-o", "pid=", "-o", "ppid="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return []

    alive: set[int] = set()
    children: dict[int, list[int]] = defaultdict(list)
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        alive.add(pid)
        children[ppid].append(pid)

    if root_pid not in alive:
        return []

    tree: list[int] = []
    queue: deque[int] = deque([root_pid])
    seen = {root_pid}
    while queue:
        pid = queue.popleft()
        tree.append(pid)
        for child in children.get(pid, []):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return tree


def list_process_tree_pids(root_pid: int) -> list[int]:
    pids = list_process_tree_pids_from_proc(root_pid)
    if pids:
        return pids
    return list_process_tree_pids_from_ps(root_pid)


def read_proc_rss_kib(pid: int) -> int:
    status_file = Path(f"/proc/{pid}/status")
    try:
        for line in status_file.read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except (OSError, ValueError):
        return 0
    return 0


def read_ps_rss_kib(pid: int) -> int:
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return 0
    try:
        return sum(
            int(line.strip()) for line in result.stdout.splitlines() if line.strip()
        )
    except ValueError:
        return 0


def sum_rss_kib_for_pids(pids: list[int]) -> int:
    if Path("/proc").exists():
        return sum(read_proc_rss_kib(pid) for pid in pids)
    return sum(read_ps_rss_kib(pid) for pid in pids)


def read_system_mem_available_kib() -> int:
    meminfo = Path("/proc/meminfo")
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except (OSError, ValueError):
        return 0
    return 0


def read_cgroup_memory_kib(path: Path) -> int:
    try:
        value = path.read_text().strip()
    except OSError:
        return 0
    if not value.isdigit():
        return 0
    return (int(value) + 1023) // 1024


def read_cgroup_current_kib() -> int:
    if Path("/sys/fs/cgroup/memory.current").is_file():
        return read_cgroup_memory_kib(Path("/sys/fs/cgroup/memory.current"))
    return read_cgroup_memory_kib(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))


def read_job_cgroup_peak_kib() -> int:
    if Path("/sys/fs/cgroup/memory.peak").is_file():
        return read_cgroup_memory_kib(Path("/sys/fs/cgroup/memory.peak"))
    return read_cgroup_memory_kib(
        Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes")
    )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def kib_to_mib(kib: int) -> int:
    return (kib + 1023) // 1024


def update_stats(stats: MemoryStats, sample: dict[str, int]) -> None:
    stats.peak_rss_kib = max(stats.peak_rss_kib, sample["rss_kib"])
    stats.peak_cgroup_current_kib = max(
        stats.peak_cgroup_current_kib, sample["cgroup_current_kib"]
    )
    stats.max_job_cgroup_peak_kib = max(
        stats.max_job_cgroup_peak_kib, sample["job_cgroup_peak_kib"]
    )
    stats.peak_process_count = max(stats.peak_process_count, sample["process_count"])
    if sample["system_mem_available_kib"] > 0 and (
        stats.min_system_mem_available_kib == 0
        or sample["system_mem_available_kib"] < stats.min_system_mem_available_kib
    ):
        stats.min_system_mem_available_kib = sample["system_mem_available_kib"]
    stats.sample_count += 1


def print_memory_sample(label: str, stats: MemoryStats, sample: dict[str, int]) -> None:
    print(
        "MEMORY sample: "
        f"{label}: "
        f"RSS {kib_to_mib(sample['rss_kib'])} MiB, "
        f"cgroup current {kib_to_mib(sample['cgroup_current_kib'])} MiB, "
        f"job cgroup peak {kib_to_mib(sample['job_cgroup_peak_kib'])} MiB, "
        f"MemAvailable {kib_to_mib(sample['system_mem_available_kib'])} MiB, "
        f"processes {sample['process_count']}, "
        f"samples {stats.sample_count}",
        flush=True,
    )


def monitor(args: argparse.Namespace) -> int:
    interval = validate_positive_float(args.interval, "AOT_MEMORY_MONITOR_INTERVAL")
    log_interval = validate_positive_int(args.log_interval, "AOT_MEMORY_LOG_INTERVAL")
    root_pid = int(args.pid)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)

    stats = MemoryStats()
    start_epoch = int(time.time())
    last_log_epoch = 0

    with report.open("w", buffering=1) as f:
        f.write(f"# label={args.label}\n")
        f.write(f"# root_pid={root_pid}\n")
        f.write(f"# sample_interval_seconds={interval:g}\n")
        f.write(
            "timestamp_utc,rss_kib,system_mem_available_kib,"
            "cgroup_current_kib,job_cgroup_peak_kib,process_count\n"
        )

        while process_exists(root_pid):
            tree_pids = list_process_tree_pids(root_pid)
            if not tree_pids:
                time.sleep(interval)
                continue

            sample = {
                "rss_kib": sum_rss_kib_for_pids(tree_pids),
                "system_mem_available_kib": read_system_mem_available_kib(),
                "cgroup_current_kib": read_cgroup_current_kib(),
                "job_cgroup_peak_kib": read_job_cgroup_peak_kib(),
                "process_count": len(tree_pids),
            }
            f.write(
                f"{utc_timestamp()},"
                f"{sample['rss_kib']},"
                f"{sample['system_mem_available_kib']},"
                f"{sample['cgroup_current_kib']},"
                f"{sample['job_cgroup_peak_kib']},"
                f"{sample['process_count']}\n"
            )

            update_stats(stats, sample)

            now_epoch = int(time.time())
            if now_epoch - last_log_epoch >= log_interval:
                print_memory_sample(args.label, stats, sample)
                last_log_epoch = now_epoch

            time.sleep(interval)

        stats.duration_seconds = int(time.time()) - start_epoch
        f.write("# summary\n")
        f.write(f"peak_rss_kib={stats.peak_rss_kib}\n")
        f.write(f"peak_cgroup_current_kib={stats.peak_cgroup_current_kib}\n")
        f.write(f"max_job_cgroup_peak_kib={stats.max_job_cgroup_peak_kib}\n")
        f.write(f"min_system_mem_available_kib={stats.min_system_mem_available_kib}\n")
        f.write(f"peak_process_count={stats.peak_process_count}\n")
        f.write(f"sample_count={stats.sample_count}\n")
        f.write(f"duration_seconds={stats.duration_seconds}\n")

    return 0


def parse_summary(report: Path) -> MemoryStats:
    values: dict[str, int] = {}
    try:
        lines = report.read_text().splitlines()
    except OSError:
        return MemoryStats()

    for line in lines:
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key in {
            "peak_rss_kib",
            "peak_cgroup_current_kib",
            "max_job_cgroup_peak_kib",
            "min_system_mem_available_kib",
            "peak_process_count",
            "sample_count",
            "duration_seconds",
        }:
            try:
                values[key] = int(value)
            except ValueError:
                values[key] = 0

    return MemoryStats(
        peak_rss_kib=values.get("peak_rss_kib", 0),
        peak_cgroup_current_kib=values.get("peak_cgroup_current_kib", 0),
        max_job_cgroup_peak_kib=values.get("max_job_cgroup_peak_kib", 0),
        min_system_mem_available_kib=values.get("min_system_mem_available_kib", 0),
        peak_process_count=values.get("peak_process_count", 0),
        sample_count=values.get("sample_count", 0),
        duration_seconds=values.get("duration_seconds", 0),
    )


def summary(args: argparse.Namespace) -> int:
    report = Path(args.report)
    if not report.exists():
        return 0

    stats = parse_summary(report)
    print(
        "MEMORY: "
        f"{args.label}: "
        f"peak RSS {kib_to_mib(stats.peak_rss_kib)} MiB, "
        f"peak cgroup current {kib_to_mib(stats.peak_cgroup_current_kib)} MiB, "
        f"job cgroup peak {kib_to_mib(stats.max_job_cgroup_peak_kib)} MiB, "
        f"min MemAvailable {kib_to_mib(stats.min_system_mem_available_kib)} MiB, "
        f"peak processes {stats.peak_process_count}, "
        f"samples {stats.sample_count}, "
        f"duration {stats.duration_seconds}s"
    )
    print(f"Memory report: {report}")
    return 0


def diagnostics(_: argparse.Namespace) -> int:
    print()
    print("AOT memory diagnostics:")
    for path in CGROUP_FILES:
        if path.is_file():
            print(f"--- {path} ---")
            try:
                print(path.read_text().rstrip())
            except OSError as exc:
                print(f"unavailable: {exc}")

    print("--- recent kernel OOM events ---")
    result = subprocess.run(
        ["dmesg", "-T"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        print("dmesg unavailable or no OOM events found")
        return 0

    matches = [
        line
        for line in result.stdout.splitlines()
        if re.search(r"oom|killed process|out of memory", line, re.IGNORECASE)
    ]
    if matches:
        for line in matches[-20:]:
            print(line)
    else:
        print("dmesg unavailable or no OOM events found")
    return 0


def validate_config(args: argparse.Namespace) -> int:
    try:
        validate_positive_float(args.interval, "AOT_MEMORY_MONITOR_INTERVAL")
        validate_positive_int(args.log_interval, "AOT_MEMORY_LOG_INTERVAL")
    except ValueError as exc:
        print(exc)
        return 1
    return 0


def safe_label(args: argparse.Namespace) -> int:
    print(sanitize_label(args.label))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    safe_label_parser = subparsers.add_parser("safe-label")
    safe_label_parser.add_argument("label")
    safe_label_parser.set_defaults(func=safe_label)

    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("--interval", required=True)
    validate_parser.add_argument("--log-interval", required=True)
    validate_parser.set_defaults(func=validate_config)

    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--pid", required=True, type=int)
    monitor_parser.add_argument("--label", required=True)
    monitor_parser.add_argument("--report", required=True)
    monitor_parser.add_argument("--interval", required=True)
    monitor_parser.add_argument("--log-interval", required=True)
    monitor_parser.set_defaults(func=monitor)

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--label", required=True)
    summary_parser.add_argument("--report", required=True)
    summary_parser.set_defaults(func=summary)

    diagnostics_parser = subparsers.add_parser("diagnostics")
    diagnostics_parser.set_defaults(func=diagnostics)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
