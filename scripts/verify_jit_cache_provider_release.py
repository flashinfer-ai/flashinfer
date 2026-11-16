#!/usr/bin/env python3
"""Validate an assembled shim and architecture-provider wheel set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from jit_cache_provider_validation import (
    Wheel,
    canonicalize_distribution,
    require,
    validate_provider,
    validate_shim,
)


def _installed_provider_ids(target: Path) -> list[str]:
    code = """
import json
from flashinfer_jit_cache import get_jit_cache_providers

print(json.dumps([provider.provider_id for provider in get_jit_cache_providers()]))
"""
    process = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(target)},
    )
    return json.loads(process.stdout.strip().splitlines()[-1])


def install_smoke(
    wheelhouse: Path,
    shim: Wheel,
    providers: dict[str, Wheel],
) -> None:
    with tempfile.TemporaryDirectory(prefix="flashinfer-provider-all-") as temp_dir:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--find-links",
                str(wheelhouse),
                "--target",
                temp_dir,
                str(shim.path),
            ],
            check=True,
        )
        require(
            _installed_provider_ids(Path(temp_dir)) == sorted(providers),
            "Default shim installation did not discover every provider",
        )

    selected_provider = sorted(providers)[0]
    with tempfile.TemporaryDirectory(prefix="flashinfer-provider-minimal-") as temp_dir:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-index",
                "--target",
                temp_dir,
                str(shim.path),
                str(providers[selected_provider].path),
            ],
            check=True,
        )
        require(
            _installed_provider_ids(Path(temp_dir)) == [selected_provider],
            "Minimal installation discovered an unexpected provider set",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--providers", nargs="+", required=True)
    parser.add_argument("--provider-platform-tag", required=True)
    parser.add_argument("--install-smoke", action="store_true")
    return parser.parse_args()


def load_wheels(wheelhouse: Path) -> dict[str, Wheel]:
    """Load one wheel for each canonical distribution in a wheelhouse."""
    wheels = {}
    for path in sorted(wheelhouse.glob("*.whl")):
        wheel = Wheel.open(path)
        distribution = canonicalize_distribution(wheel.distribution)
        require(
            distribution not in wheels,
            f"Duplicate wheel distribution: {distribution}",
        )
        wheels[distribution] = wheel
    return wheels


def main() -> int:
    args = parse_args()
    expected_providers = set(args.providers)
    require(
        len(expected_providers) == len(args.providers),
        "Expected provider list contains duplicates",
    )

    wheels = load_wheels(args.wheelhouse)
    expected_distributions = {
        "flashinfer-jit-cache",
        *(f"flashinfer-jit-cache-{provider}" for provider in expected_providers),
    }
    require(
        set(wheels) == expected_distributions,
        f"Wheel distributions {sorted(wheels)} do not match {sorted(expected_distributions)}",
    )

    shim = wheels["flashinfer-jit-cache"]
    validate_shim(
        shim,
        args.version,
        expected_providers,
        args.provider_platform_tag,
    )
    providers = {}
    for provider in expected_providers:
        wheel = wheels[f"flashinfer-jit-cache-{provider}"]
        validate_provider(
            wheel,
            provider,
            args.version,
            args.provider_platform_tag,
        )
        providers[provider] = wheel

    if args.install_smoke:
        install_smoke(args.wheelhouse, shim, providers)

    print(
        f"Validated {len(providers)} providers and shim for {args.version} "
        f"on {args.provider_platform_tag}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
