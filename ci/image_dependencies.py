#!/usr/bin/env python3
"""Resolve exact Python dependencies installed in FlashInfer CI images."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


CONFIG_FILE = Path(__file__).with_name("cuda-versions.json")


def get_ci_image_dependency_requirements(
    cuda_major: str, config: dict[str, Any]
) -> list[str]:
    """Return exact CI image requirements for a CUDA major version."""
    requirements = []
    for package, dependency in config["ci_image_dependencies"].items():
        extras = dependency.get("cuda_major_extras", {}).get(cuda_major, [])
        package_spec = package
        if extras:
            package_spec += f"[{','.join(extras)}]"
        requirements.append(f"{package_spec}=={dependency['version']}")
    return requirements


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cuda_major", help="CUDA major version, such as 12 or 13")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print one requirement per line for shell consumers."""
    args = _parse_args(argv)
    if not args.cuda_major.isdigit():
        raise SystemExit(f"ERROR: invalid CUDA major version: {args.cuda_major!r}")

    config = json.loads(CONFIG_FILE.read_text())
    print(*get_ci_image_dependency_requirements(args.cuda_major, config), sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
