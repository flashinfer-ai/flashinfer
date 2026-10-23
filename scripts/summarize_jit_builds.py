#!/usr/bin/env python3
"""
Summarize the JIT build timing log written via FLASHINFER_JIT_BUILD_LOG.

Each line of the log is one JitSpec build (see ``_record_jit_build`` in
flashinfer/jit/core.py). Builds shorter than ``--min-seconds`` are counted as
up to date (an nvcc module whose ninja build had nothing to do, or a trivial
compile) and left out of the totals.

Usage:
    python scripts/summarize_jit_builds.py junit/jit-builds.jsonl [--markdown]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_records(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a process killed mid-write leaves a partial line
    return records


def summarize(records: list, min_seconds: float, top: int, markdown: bool) -> str:
    totals: dict = defaultdict(lambda: [0, 0, 0.0])  # builds, up to date, seconds
    for record in records:
        entry = totals[record.get("toolchain", "unknown")]
        if record["seconds"] < min_seconds:
            entry[1] += 1
        else:
            entry[0] += 1
            entry[2] += record["seconds"]
    slowest = sorted(records, key=lambda r: r["seconds"], reverse=True)[:top]

    if markdown:
        lines = [
            "### JIT builds by toolchain",
            "",
            f"Builds of at least {min_seconds:g} s; summed over processes, "
            "so parallel builds add up past wall time.",
            "",
            "| Toolchain | Builds | Total | Up to date |",
            "|---|---:|---:|---:|",
        ]
        for name, (builds, fresh, seconds) in sorted(
            totals.items(), key=lambda item: -item[1][2]
        ):
            lines.append(f"| {name} | {builds} | {seconds / 60:.1f} min | {fresh} |")
        lines += ["", f"Slowest {len(slowest)}:", "", "| Seconds | Toolchain | Name |"]
        lines.append("|---:|---|---|")
        for r in slowest:
            lines.append(f"| {r['seconds']:.1f} | {r['toolchain']} | `{r['name']}` |")
        return "\n".join(lines) + "\n"

    lines = [f"JIT BUILD SUMMARY (builds >= {min_seconds:g} s, summed over processes)"]
    for name, (builds, fresh, seconds) in sorted(
        totals.items(), key=lambda item: -item[1][2]
    ):
        lines.append(
            f"  toolchain={name} builds={builds} total_seconds={seconds:.1f} "
            f"up_to_date={fresh}"
        )
    lines.append(f"SLOWEST {len(slowest)} JIT BUILDS")
    for r in slowest:
        lines.append(f"  {r['seconds']:8.1f}s {r['toolchain']:<10} {r['name']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log", type=Path)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"No JIT build log at {args.log}; nothing was built or logging was off.")
        return 0
    records = load_records(args.log)
    if not records:
        print(f"JIT build log {args.log} has no records.")
        return 0
    sys.stdout.write(summarize(records, args.min_seconds, args.top, args.markdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
