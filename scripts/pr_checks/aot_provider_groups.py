#!/usr/bin/env python3
"""Split the PR AOT provider architectures into balanced build groups.

The PR ``AOT Build Import`` check builds one provider wheel per architecture
listed in ``ci/cuda-versions.json``. Building them one after another on one
runner takes the sum of their build times, so the workflow builds each group on
its own runner and assembles the shim from the resulting wheels afterwards.

Prints a JSON array with one space-separated architecture string per group.
Architectures are assigned longest first to the least loaded group (LPT), so a
new architecture needs no edit here: it is costed at ``DEFAULT_COST``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Minutes of a 100%-sccache-hit provider build on a 64-core x64 runner, cu130
# (PR run 40929394275). Only the ratios matter. Unlisted architectures cost
# DEFAULT_COST; an out-of-date table gives a less balanced split, never a wrong one.
PROVIDER_BUILD_COST = {
    "8.0": 3.3,
    "8.9": 3.3,
    "9.0a": 11.0,
    "10.0a": 11.0,
    "10.3a": 10.8,
    "12.0f": 7.7,
}
DEFAULT_COST = 8.0
DEFAULT_GROUP_COUNT = 2
CPU_ARCHITECTURES = ("x86_64", "aarch64")


def provider_architectures(config: dict, cuda_label: str, cpu_arch: str) -> list[str]:
    entry = next(item for item in config["jit_cache"] if item["label"] == cuda_label)
    return list(entry[f"{cpu_arch}_provider_architectures"])


def split_groups(architectures: list[str], group_count: int) -> list[list[str]]:
    """Balance ``architectures`` over at most ``group_count`` non-empty groups."""
    if group_count < 1:
        raise ValueError("group_count must be at least 1")
    group_count = min(group_count, len(architectures))
    if group_count == 0:
        return []
    groups: list[list[str]] = [[] for _ in range(group_count)]
    loads = [0.0] * group_count
    # sorted() is stable, so equal costs keep their configured order.
    for arch in sorted(
        architectures, key=lambda a: -PROVIDER_BUILD_COST.get(a, DEFAULT_COST)
    ):
        target = loads.index(min(loads))
        groups[target].append(arch)
        loads[target] += PROVIDER_BUILD_COST.get(arch, DEFAULT_COST)
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cuda", required=True, help="CUDA label, e.g. cu130")
    parser.add_argument("--cpu-architecture", required=True, choices=CPU_ARCHITECTURES)
    parser.add_argument("--groups", type=int, default=DEFAULT_GROUP_COUNT)
    parser.add_argument("--config", type=Path, default=Path("ci/cuda-versions.json"))
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text())
    archs = provider_architectures(config, args.cuda, args.cpu_architecture)
    groups = split_groups(archs, args.groups)
    print(json.dumps([" ".join(group) for group in groups]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
