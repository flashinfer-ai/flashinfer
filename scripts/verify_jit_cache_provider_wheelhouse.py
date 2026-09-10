#!/usr/bin/env python3
"""Validate a local FlashInfer jit-cache provider wheelhouse."""

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
    inspect_cuda_architectures,
    require,
    validate_provider,
    validate_shim,
    write_validation_report,
)


def validate_flashinfer_python(wheel: Wheel, expected_version: str) -> None:
    require(
        canonicalize_distribution(wheel.distribution) == "flashinfer-python",
        f"Unexpected FlashInfer distribution: {wheel.distribution}",
    )
    require(
        wheel.version == expected_version,
        f"FlashInfer version {wheel.version} does not match {expected_version}",
    )


def run_install_smoke(shim: Wheel, provider: Wheel, provider_id: str) -> None:
    with tempfile.TemporaryDirectory(
        prefix="flashinfer-wheelhouse-install-"
    ) as temp_dir:
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
                str(provider.path),
                str(shim.path),
            ],
            check=True,
        )
        smoke_code = """
import json
from flashinfer_jit_cache import get_jit_cache_providers

providers = get_jit_cache_providers()
assert len(providers) == 1, providers
provider = providers[0]
print(json.dumps({
    "provider_id": provider.provider_id,
    "cuda_architectures": sorted(provider.cuda_architectures),
    "module_count": len(provider.modules),
}))
"""
        process = subprocess.run(
            [sys.executable, "-c", smoke_code],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": temp_dir},
        )
        result = json.loads(process.stdout.strip().splitlines()[-1])
        require(result["provider_id"] == provider_id, "Installed provider ID mismatch")
        require(
            result["cuda_architectures"] == [provider_id],
            "Installed provider architecture mismatch",
        )
        require(result["module_count"] > 0, "Installed provider has no modules")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--provider-platform-tag", default="")
    parser.add_argument("--cuobjdump", type=Path)
    parser.add_argument(
        "--cuda-architecture-policy",
        choices=("strict", "report"),
        default="strict",
        help="Fail on non-provider cubins, or retain them in the inventory report",
    )
    parser.add_argument("--install-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wheel_paths = sorted(args.wheelhouse.glob("*.whl"))
    require(len(wheel_paths) == 3, f"Expected three wheels, found {len(wheel_paths)}")
    opened_wheels = [Wheel.open(path) for path in wheel_paths]
    wheels = {
        canonicalize_distribution(wheel.distribution): wheel for wheel in opened_wheels
    }
    expected_distributions = {
        "flashinfer-python",
        "flashinfer-jit-cache",
        f"flashinfer-jit-cache-{args.provider}",
    }
    require(
        set(wheels) == expected_distributions,
        f"Unexpected wheel distributions: {sorted(wheels)}",
    )

    flashinfer_python = wheels["flashinfer-python"]
    shim = wheels["flashinfer-jit-cache"]
    provider_wheel = wheels[f"flashinfer-jit-cache-{args.provider}"]
    validate_flashinfer_python(flashinfer_python, args.version)
    manifest, module_paths = validate_provider(
        provider_wheel,
        args.provider,
        args.version,
        args.provider_platform_tag or None,
    )
    validate_shim(shim, args.version, {args.provider})

    module_architectures: dict[str, list[str]] = {}
    ptx_modules: list[str] = []
    if args.cuobjdump is not None:
        module_architectures, ptx_modules = inspect_cuda_architectures(
            provider_wheel,
            module_paths,
            args.provider,
            args.cuobjdump,
            strict=args.cuda_architecture_policy == "strict",
        )
    if args.install_smoke:
        run_install_smoke(shim, provider_wheel, args.provider)

    write_validation_report(
        args.wheelhouse,
        wheels,
        args.provider,
        manifest,
        module_architectures,
        ptx_modules,
        "wheelhouse.json",
    )
    print(
        f"Validated {args.provider}: {len(module_paths)} modules, "
        f"{len(ptx_modules)} PTX modules, {len(wheel_paths)} wheels"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
