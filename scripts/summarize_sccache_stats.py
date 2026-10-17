#!/usr/bin/env python3
"""Summarize an sccache JSON stats dump.

`sccache --show-stats` leads with a hit rate computed only over cacheable work,
so a build whose compiles are never eligible for the cache can report a ~100%
hit rate while spending all of its time compiling. This prints the full
accounting - hits, misses, and non-cacheable compilations - along with the
reasons sccache recorded for declining to cache, and the wall-clock split
implied by sccache's own average durations.

Usage:
    python scripts/summarize_sccache_stats.py sccache-stats.json [--warn-threshold 0.05]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


def _duration_seconds(value: Dict[str, int]) -> float:
    return value.get("secs", 0) + value.get("nanos", 0) / 1e9


def _total(counts: Dict[str, int]) -> int:
    return sum(counts.values())


def _format_counts(counts: Dict[str, int], indent: str = "    ") -> str:
    if not counts:
        return f"{indent}(none)"
    width = max(len(name) for name in counts)
    return "\n".join(
        f"{indent}{name:<{width}}  {count}"
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1])
    )


def summarize(stats: Dict, warn_threshold: float) -> int:
    hit_counts = stats["cache_hits"]["counts"]
    miss_counts = stats["cache_misses"]["counts"]
    hits = _total(hit_counts)
    misses = _total(miss_counts)
    non_cacheable = stats["non_cacheable_compilations"]

    # Every compile lands in exactly one of these three buckets, but the
    # headline hit rate only divides by the first two.
    accounted = hits + misses + non_cacheable

    print("sccache accounting")
    print(f"  compile requests            {stats['compile_requests']}")
    print(f"  requests executed           {stats['requests_executed']}")
    print(f"  cache hits                  {hits}")
    print(f"  cache misses                {misses}")
    print(f"  non-cacheable compilations  {non_cacheable}")
    print(f"  requests not cacheable      {stats['requests_not_cacheable']}")
    print(f"  compilations                {stats['compilations']}")
    print(f"  compile failures            {stats['compile_fails']}")

    if hits + misses:
        print(f"  hit rate (cacheable only)   {100 * hits / (hits + misses):.2f} %")
    if accounted:
        print(f"  hit rate (all compiles)     {100 * hits / accounted:.2f} %")
        print(f"  non-cacheable share         {100 * non_cacheable / accounted:.2f} %")

    print("  cache hits by kind:")
    print(_format_counts(hit_counts))
    print("  cache misses by kind:")
    print(_format_counts(miss_counts))

    # The most direct evidence for *why* work bypassed the cache.
    print("  non-cacheable reasons:")
    print(_format_counts(stats.get("not_cached") or {}))

    # Reproduce the cost split from sccache's own averages, so a slow build with
    # a healthy-looking hit rate is visible directly in the job log.
    hit_seconds = _duration_seconds(stats["cache_read_hit_duration"])
    compile_seconds = _duration_seconds(stats["compiler_write_duration"])
    compilations = stats["compilations"]
    avg_hit = hit_seconds / hits if hits else 0.0
    avg_compile = compile_seconds / compilations if compilations else 0.0
    estimated_total = hit_seconds + compile_seconds

    print("estimated wall-clock split (count x sccache average)")
    print(f"  average cache read hit      {avg_hit:.3f} s")
    print(f"  average compiler            {avg_compile:.3f} s")
    if estimated_total > 0:
        print(
            f"  cache reads                 {hit_seconds:9.1f} s "
            f"({100 * hit_seconds / estimated_total:.1f} %)"
        )
        print(
            f"  compilations                {compile_seconds:9.1f} s "
            f"({100 * compile_seconds / estimated_total:.1f} %)"
        )

    if accounted and non_cacheable / accounted > warn_threshold:
        share = 100 * non_cacheable / accounted
        print(
            f"::warning::{non_cacheable} of {accounted} compiles ({share:.1f} %) were "
            "not eligible for the sccache cache; the headline hit rate excludes them"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats_file", type=Path, help="Path to sccache-stats.json")
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.05,
        help="Emit a GitHub warning when the non-cacheable share exceeds this fraction",
    )
    args = parser.parse_args()

    if not args.stats_file.is_file():
        print(f"No sccache stats file at {args.stats_file}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(args.stats_file.read_text())
    except json.JSONDecodeError as e:
        print(f"Could not parse {args.stats_file}: {e}", file=sys.stderr)
        return 1

    stats = payload.get("stats")
    if not isinstance(stats, dict):
        print(f"{args.stats_file} has no 'stats' object", file=sys.stderr)
        return 1

    return summarize(stats, args.warn_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
