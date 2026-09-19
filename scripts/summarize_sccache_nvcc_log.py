#!/usr/bin/env python3
"""Aggregate the nvcc decomposition markers from an sccache server log.

Diagnostic scaffolding for investigating why CUDA compiles bypass the sccache
cache. sccache never caches the outer `nvcc` invocation; it decomposes it with
`nvcc --dryrun` and caches the constituent subcommands, of which `fatbinary`
and `nvlink` are never cacheable. Separately, a depfile flag
(`--generate-dependencies-with-compile`) makes sccache run an extra
dependency-only pass.

Both of those produce roughly one event per translation unit, so a raw count of
non-cacheable compilations cannot tell them apart. This counts each marker so
they can be compared against the stats buckets.

Requires the server log to include the nvcc backend at trace level, e.g.
SCCACHE_LOG=sccache=debug,sccache::compiler::nvcc=trace

Usage:
    python scripts/summarize_sccache_nvcc_log.py sccache-server.log
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# trace!("[{}]: dependencies command: {:?}", output_file_name, cmd)
DEPENDENCIES_RE = re.compile(r"dependencies command:")
# trace!("[{}]: nvcc dryrun command: {:?}", output_file_name, cmd)
DRYRUN_RE = re.compile(r"nvcc dryrun command:")
# trace!("[{}]: transformed nvcc command: \"cd <dir> && <exe> <args>\"", ...)
TRANSFORMED_RE = re.compile(r'transformed nvcc command: "(?:cd [^&]*&& )?(\S+)')
# Each marker is prefixed with the object file it belongs to.
OUTPUT_RE = re.compile(r"\[([^\]]+\.o)\]:")

# sccache marks these subcommands non-cacheable regardless of any compiler flag.
ALWAYS_NON_CACHEABLE = {"fatbinary", "nvlink"}


def summarize(log_path: Path) -> int:
    dependency_passes = 0
    dryrun_invocations = 0
    subcommands: Counter = Counter()
    translation_units = set()

    # The log can reach hundreds of MB, so stream it.
    with log_path.open(errors="replace") as handle:
        for line in handle:
            if DEPENDENCIES_RE.search(line):
                dependency_passes += 1
            if DRYRUN_RE.search(line):
                dryrun_invocations += 1
            match = TRANSFORMED_RE.search(line)
            if match:
                subcommands[match.group(1).rsplit("/", 1)[-1]] += 1
            output = OUTPUT_RE.search(line)
            if output:
                translation_units.add(output.group(1))

    units = len(translation_units)
    print(f"nvcc decomposition markers in {log_path}")
    print(f"  translation units seen      {units}")
    print(f"  dependency-only passes      {dependency_passes}")
    print(f"  --dryrun invocations        {dryrun_invocations}")
    print("  decomposed subcommands:")
    if subcommands:
        width = max(len(name) for name in subcommands)
        for name, count in subcommands.most_common():
            flag = "  (never cacheable)" if name in ALWAYS_NON_CACHEABLE else ""
            print(f"    {name:<{width}}  {count}{flag}")
    else:
        print("    (none)")

    if units:
        print("per translation unit")
        print(f"  dependency-only passes      {dependency_passes / units:.2f}")
        never_cacheable = sum(
            count for name, count in subcommands.items() if name in ALWAYS_NON_CACHEABLE
        )
        print(f"  never-cacheable subcommands {never_cacheable / units:.2f}")

    if not subcommands and not dependency_passes:
        print(
            "::warning::No nvcc trace markers found; the server log likely does not "
            "include sccache::compiler::nvcc at trace level"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path, help="Path to the sccache server log")
    args = parser.parse_args()

    if not args.log_file.is_file():
        print(f"No sccache server log at {args.log_file}", file=sys.stderr)
        return 1

    return summarize(args.log_file)


if __name__ == "__main__":
    raise SystemExit(main())
