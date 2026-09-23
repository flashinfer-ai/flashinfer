#!/usr/bin/env python3
"""Collect and summarize Ninja timing state from a JIT-cache provider build."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NinjaLogEntry:
    start_ms: int
    end_ms: int
    output_mtime: int
    output: str
    command_hash: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def parse_ninja_log(path: Path) -> tuple[str, list[NinjaLogEntry]]:
    lines = path.read_text(errors="replace").splitlines()
    version = lines[0].removeprefix("# ninja log ").strip() if lines else "unknown"
    entries = []
    for line in lines[1:]:
        fields = line.split("\t", 4)
        if len(fields) != 5:
            continue
        try:
            start_ms, end_ms, output_mtime = map(int, fields[:3])
        except ValueError:
            continue
        entries.append(
            NinjaLogEntry(
                start_ms=start_ms,
                end_ms=end_ms,
                output_mtime=output_mtime,
                output=fields[3],
                command_hash=fields[4],
            )
        )
    return version, entries


def latest_invocation(entries: list[NinjaLogEntry]) -> list[NinjaLogEntry]:
    """Return the final monotonic completion-time segment from an appended log."""
    start = 0
    previous_end = -1
    for index, entry in enumerate(entries):
        if entry.end_ms < previous_end:
            start = index
        previous_end = entry.end_ms
    return entries[start:]


def percentile(values: list[int], percent: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, (len(ordered) * percent + 99) // 100 - 1)
    return ordered[index]


def max_completed_parallelism(entries: list[NinjaLogEntry]) -> int:
    events = []
    for entry in entries:
        events.append((entry.start_ms, 1))
        events.append((entry.end_ms, -1))
    running = 0
    maximum = 0
    # An edge ending when another starts does not overlap with the new edge.
    for _, change in sorted(events, key=lambda event: (event[0], event[1])):
        running += change
        maximum = max(maximum, running)
    return maximum


def summarize_log(path: Path, build_root: Path) -> dict[str, Any]:
    version, all_entries = parse_ninja_log(path)
    entries = latest_invocation(all_entries)
    by_completion = sorted(entries, key=lambda entry: entry.end_ms)
    durations = [entry.duration_ms for entry in entries]
    completion_gaps = [by_completion[0].end_ms] if by_completion else []
    completion_gaps.extend(
        current.end_ms - previous.end_ms
        for previous, current in zip(by_completion, by_completion[1:], strict=False)
    )

    def render_entry(entry: NinjaLogEntry) -> dict[str, Any]:
        return {
            **asdict(entry),
            "duration_ms": entry.duration_ms,
        }

    return {
        "path": str(path.relative_to(build_root)),
        "version": version,
        "records_in_file": len(all_entries),
        "records_in_latest_invocation": len(entries),
        "latest_completion_ms": max((entry.end_ms for entry in entries), default=None),
        "maximum_gap_between_completions_ms": max(completion_gaps, default=None),
        "maximum_parallel_completed_edges": max_completed_parallelism(entries),
        "duration_ms": {
            "maximum": max(durations, default=None),
            "p50": percentile(durations, 50),
            "p90": percentile(durations, 90),
            "p95": percentile(durations, 95),
            "p99": percentile(durations, 99),
        },
        "longest_completed_edges": [
            render_entry(entry)
            for entry in sorted(
                entries, key=lambda entry: entry.duration_ms, reverse=True
            )[:100]
        ],
        "last_completed_edges": [render_entry(entry) for entry in by_completion[-100:]],
    }


def render_text_report(report: dict[str, Any]) -> str:
    lines = [
        "Ninja completed-edge diagnostics",
        "",
        "The Ninja log contains successful completed edges only. Combine missing",
        "outputs with the watchdog process snapshot to identify in-flight commands.",
        "",
        f"build_root: {report['build_root']}",
        f"ninja_version: {report['ninja_version']}",
        f"ninja_status: {report['ninja_status']}",
        f"ninja_logs_found: {len(report['logs'])}",
    ]
    for log in report["logs"]:
        duration = log["duration_ms"]
        lines.extend(
            [
                "",
                f"log: {log['path']}",
                f"version: {log['version']}",
                f"records_in_file: {log['records_in_file']}",
                f"records_in_latest_invocation: {log['records_in_latest_invocation']}",
                f"latest_completion_ms: {log['latest_completion_ms']}",
                "maximum_gap_between_completions_ms: "
                f"{log['maximum_gap_between_completions_ms']}",
                "maximum_parallel_completed_edges: "
                f"{log['maximum_parallel_completed_edges']}",
                "duration_ms: "
                f"max={duration['maximum']} p50={duration['p50']} "
                f"p90={duration['p90']} p95={duration['p95']} p99={duration['p99']}",
                "",
                "Longest completed edges:",
            ]
        )
        for entry in log["longest_completed_edges"]:
            lines.append(
                f"{entry['duration_ms']:>10} ms  "
                f"start={entry['start_ms']:>10} end={entry['end_ms']:>10}  "
                f"{entry['output']}"
            )
        lines.extend(("", "Last completed edges:"))
        for entry in log["last_completed_edges"]:
            lines.append(
                f"end={entry['end_ms']:>10} ms duration={entry['duration_ms']:>10} ms  "
                f"{entry['output']}"
            )
    return "\n".join(lines) + "\n"


def archive_ninja_state(build_root: Path, output_path: Path) -> int:
    paths = sorted(
        path
        for path in build_root.rglob("*")
        if path.is_file() and (path.name == ".ninja_log" or path.suffix == ".ninja")
    )
    if not paths:
        return 0
    with tarfile.open(output_path, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(build_root), recursive=False)
    return len(paths)


def ninja_version() -> str | None:
    try:
        result = subprocess.run(
            ["ninja", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def collect(build_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = sorted(build_root.rglob(".ninja_log")) if build_root.is_dir() else []
    report: dict[str, Any] = {
        "schema_version": 1,
        "build_root": str(build_root),
        "ninja_version": ninja_version(),
        "ninja_status": os.environ.get("NINJA_STATUS"),
        "logs": [summarize_log(path, build_root) for path in logs],
    }

    if len(logs) == 1:
        shutil.copy2(logs[0], output_dir / "ninja-log.tsv")
    elif logs:
        raw_dir = output_dir / "raw-logs"
        for path in logs:
            destination = raw_dir / path.relative_to(build_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    report["archived_state_files"] = archive_ninja_state(
        build_root, output_dir / "ninja-state.tar.gz"
    )
    (output_dir / "ninja-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "ninja-summary.txt").write_text(render_text_report(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect(args.build_root.resolve(), args.output_dir.resolve())
    print(
        "Collected Ninja diagnostics: "
        f"{len(report['logs'])} log(s), "
        f"{report['archived_state_files']} raw state file(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
