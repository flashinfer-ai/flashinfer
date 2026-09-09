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

from verify_jit_cache_provider_wheelhouse import (
    Wheel,
    canonicalize_distribution,
    normalize_requirement,
    require,
    validate_provider,
)


def validate_shim(
    wheel: Wheel, expected_version: str, expected_providers: set[str]
) -> None:
    require(
        canonicalize_distribution(wheel.distribution) == "flashinfer-jit-cache",
        f"Unexpected shim distribution: {wheel.distribution}",
    )
    require(
        wheel.version == expected_version,
        f"Shim version {wheel.version} does not match {expected_version}",
    )
    require(
        not any(path.endswith(".so") for path in wheel.contents),
        "Shim wheel must not contain shared libraries",
    )
    requirements = dict(map(normalize_requirement, wheel.requirements))
    expected_requirements = {
        f"flashinfer-jit-cache-{provider}": expected_version
        for provider in expected_providers
    }
    require(
        requirements == expected_requirements,
        f"Shim requirements {requirements} do not match {expected_requirements}",
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


def main() -> int:
    args = parse_args()
    expected_providers = set(args.providers)
    require(
        len(expected_providers) == len(args.providers),
        "Expected provider list contains duplicates",
    )

    wheels = {
        canonicalize_distribution(wheel.distribution): wheel
        for wheel in (
            Wheel.open(path) for path in sorted(args.wheelhouse.glob("*.whl"))
        )
    }
    expected_distributions = {
        "flashinfer-jit-cache",
        *(f"flashinfer-jit-cache-{provider}" for provider in expected_providers),
    }
    require(
        set(wheels) == expected_distributions,
        f"Wheel distributions {sorted(wheels)} do not match {sorted(expected_distributions)}",
    )

    shim = wheels["flashinfer-jit-cache"]
    validate_shim(shim, args.version, expected_providers)
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
