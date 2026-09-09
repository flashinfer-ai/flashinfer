#!/usr/bin/env python3
"""Validate a local FlashInfer jit-cache provider wheelhouse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jit_cache_provider_validation import (
    Wheel,
    canonicalize_distribution,
    inspect_cuda_architectures,
    require,
    validate_provider,
    validate_shim,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_report(
    wheelhouse: Path,
    wheels: dict[str, Wheel],
    provider: str,
    manifest: dict[str, Any],
    module_architectures: dict[str, list[str]],
    ptx_modules: list[str],
) -> None:
    architecture_summary = {
        "provider_only": sum(
            targets == [provider] for targets in module_architectures.values()
        ),
        "mixed": sum(
            provider in targets and targets != [provider]
            for targets in module_architectures.values()
        ),
        "foreign_only": sum(
            bool(targets) and provider not in targets
            for targets in module_architectures.values()
        ),
        "no_cubin": sum(not targets for targets in module_architectures.values()),
    }
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider_id": provider,
        "version": manifest["version"],
        "cuda_architectures": manifest["cuda_architectures"],
        "module_count": len(manifest["modules"]),
        "modules": sorted(manifest["modules"]),
        "module_cuda_architectures": module_architectures,
        "module_cuda_architecture_summary": architecture_summary,
        "ptx_module_count": len(ptx_modules),
        "ptx_modules": ptx_modules,
        "wheels": {
            distribution: {
                "filename": wheel.path.name,
                "size_bytes": wheel.path.stat().st_size,
                "sha256": sha256(wheel.path),
            }
            for distribution, wheel in sorted(wheels.items())
        },
    }
    (wheelhouse / "wheelhouse.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    checksum_lines = [
        f"{details['sha256']}  {details['filename']}"
        for details in report["wheels"].values()
    ]
    (wheelhouse / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")


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
    parser.add_argument(
        "--provider-only",
        action="store_true",
        help="Validate a single provider artifact without a shim or Python wheel",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wheel_paths = sorted(args.wheelhouse.glob("*.whl"))
    if args.provider_only:
        require(
            not args.install_smoke,
            "--install-smoke requires a shim and cannot be used with --provider-only",
        )
        require(len(wheel_paths) == 1, f"Expected one wheel, found {len(wheel_paths)}")
        provider_wheel = Wheel.open(wheel_paths[0])
        wheels = {
            canonicalize_distribution(provider_wheel.distribution): provider_wheel
        }
        manifest, module_paths = validate_provider(
            provider_wheel,
            args.provider,
            args.version,
            args.provider_platform_tag or None,
        )
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
        write_report(
            args.wheelhouse,
            wheels,
            args.provider,
            manifest,
            module_architectures,
            ptx_modules,
        )
        print(
            f"Validated {args.provider}: {len(module_paths)} modules, "
            f"{len(ptx_modules)} PTX modules, one provider wheel"
        )
        return 0

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

    write_report(
        args.wheelhouse,
        wheels,
        args.provider,
        manifest,
        module_architectures,
        ptx_modules,
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
