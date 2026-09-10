#!/usr/bin/env python3
"""Validate one architecture-specific JIT-cache provider wheel."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from jit_cache_provider_validation import (
    Wheel,
    canonicalize_distribution,
    inspect_cuda_architectures,
    validate_provider,
    write_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--provider-platform-tag", required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument(
        "--cuda-architecture-policy",
        choices=("strict", "report"),
        default="strict",
        help="Fail on non-provider cubins, or retain them in the inventory report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wheel_paths = sorted(args.artifact_dir.glob("*.whl"))
    if len(wheel_paths) != 1:
        raise ValueError(f"Expected one provider wheel, found {len(wheel_paths)}")

    wheel = Wheel.open(wheel_paths[0])
    manifest, module_paths = validate_provider(
        wheel,
        args.provider,
        args.version,
        args.provider_platform_tag,
    )
    module_architectures, ptx_modules = inspect_cuda_architectures(
        wheel,
        module_paths,
        args.provider,
        args.cuobjdump,
        strict=args.cuda_architecture_policy == "strict",
    )
    write_validation_report(
        args.artifact_dir,
        {canonicalize_distribution(wheel.distribution): wheel},
        args.provider,
        manifest,
        module_architectures,
        ptx_modules,
        "provider-validation.json",
    )
    print(
        f"Validated {args.provider}: {len(module_paths)} modules, "
        f"{len(ptx_modules)} PTX modules"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
