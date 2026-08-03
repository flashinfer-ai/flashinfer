#!/usr/bin/env python3
"""Validate a local FlashInfer jit-cache provider wheelhouse."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path
from typing import Any


def canonicalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Wheel:
    path: Path
    distribution: str
    version: str
    requirements: tuple[str, ...]
    contents: tuple[str, ...]
    metadata_path: str

    @classmethod
    def open(cls, path: Path) -> "Wheel":
        with zipfile.ZipFile(path) as archive:
            contents = tuple(sorted(archive.namelist()))
            metadata_paths = [
                name for name in contents if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_paths) != 1:
                raise ValueError(
                    f"{path.name}: expected one METADATA file, found "
                    f"{len(metadata_paths)}"
                )
            metadata_path = metadata_paths[0]
            metadata = BytesParser().parsebytes(archive.read(metadata_path))
        return cls(
            path=path,
            distribution=metadata["Name"],
            version=metadata["Version"],
            requirements=tuple(metadata.get_all("Requires-Dist", [])),
            contents=contents,
            metadata_path=metadata_path,
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize_requirement(requirement: str) -> tuple[str, str]:
    match = re.fullmatch(r"\s*([A-Za-z0-9_.-]+)\s*==\s*([^\s;]+)\s*", requirement)
    if match is None:
        raise ValueError(f"Expected an exact provider pin, got {requirement!r}")
    return canonicalize_distribution(match.group(1)), match.group(2)


def read_provider_manifest(wheel: Wheel, provider: str) -> tuple[dict[str, Any], str]:
    package_suffix = f"flashinfer_jit_cache/providers/{provider}/manifest.json"
    manifest_paths = [
        path
        for path in wheel.contents
        if path == package_suffix or path.endswith(f".data/purelib/{package_suffix}")
    ]
    require(
        len(manifest_paths) == 1,
        f"{wheel.path.name}: expected one {package_suffix}, found {manifest_paths}",
    )
    manifest_path = manifest_paths[0]
    package_prefix = manifest_path.removesuffix("/manifest.json")
    with zipfile.ZipFile(wheel.path) as archive:
        manifest = json.loads(archive.read(manifest_path))
    return manifest, package_prefix


def validate_provider(
    wheel: Wheel, provider: str, expected_version: str
) -> tuple[dict[str, Any], dict[str, str]]:
    expected_distribution = f"flashinfer-jit-cache-{provider}"
    require(
        canonicalize_distribution(wheel.distribution) == expected_distribution,
        f"Unexpected provider distribution: {wheel.distribution}",
    )
    require(
        wheel.version == expected_version,
        f"Provider version {wheel.version} does not match {expected_version}",
    )
    require(not wheel.requirements, "Provider wheel must not depend on other wheels")

    manifest, package_prefix = read_provider_manifest(wheel, provider)
    require(manifest.get("schema_version") == 1, "Unsupported provider manifest")
    require(manifest.get("provider_id") == provider, "Provider ID mismatch")
    require(
        canonicalize_distribution(str(manifest.get("distribution", "")))
        == expected_distribution,
        "Provider manifest distribution mismatch",
    )
    require(manifest.get("version") == expected_version, "Manifest version mismatch")
    require(
        manifest.get("cuda_architectures") == [provider],
        "Provider manifest must declare exactly its own architecture",
    )

    module_paths: dict[str, str] = {}
    so_prefix = f"{package_prefix}/jit_cache/"
    for path in wheel.contents:
        if not path.startswith(so_prefix) or not path.endswith(".so"):
            continue
        relative = path.removeprefix(so_prefix)
        parts = relative.split("/")
        require(
            len(parts) == 2 and parts[1] == f"{parts[0]}.so",
            f"Unexpected provider shared-library path: {path}",
        )
        module_paths[parts[0]] = path

    require(module_paths, "Provider wheel contains no shared libraries")
    manifest_modules = set(manifest.get("modules", []))
    require(
        manifest_modules == set(module_paths),
        "Provider manifest modules do not match packaged shared libraries",
    )

    entry_points_path = wheel.metadata_path.replace("METADATA", "entry_points.txt")
    require(
        entry_points_path in wheel.contents,
        f"{wheel.path.name}: missing provider entry point",
    )
    parser = configparser.ConfigParser()
    with zipfile.ZipFile(wheel.path) as archive:
        parser.read_string(archive.read(entry_points_path).decode())
    group = "flashinfer.jit_cache.providers"
    require(parser.has_section(group), f"Missing {group} entry-point group")
    expected_entry_point = f"flashinfer_jit_cache.providers.{provider}:get_provider"
    require(
        parser.get(group, provider, fallback="").strip() == expected_entry_point,
        f"Provider entry point does not match {expected_entry_point}",
    )
    return manifest, module_paths


def validate_shim(wheel: Wheel, provider: str, expected_version: str) -> None:
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
    require(
        len(wheel.requirements) == 1,
        f"Shim must have one provider requirement, found {wheel.requirements}",
    )
    requirement_name, requirement_version = normalize_requirement(wheel.requirements[0])
    require(
        requirement_name == f"flashinfer-jit-cache-{provider}",
        f"Shim requires unexpected provider {requirement_name}",
    )
    require(
        requirement_version == expected_version,
        f"Shim provider pin {requirement_version} does not match {expected_version}",
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


def inspect_cuda_architectures(
    wheel: Wheel,
    module_paths: dict[str, str],
    provider: str,
    cuobjdump: Path,
) -> dict[str, list[str]]:
    require(cuobjdump.is_file(), f"cuobjdump not found: {cuobjdump}")
    result: dict[str, list[str]] = {}
    architecture_pattern = re.compile(r"\bsm[_-]?([0-9]{2,3}[af]?)\b", re.IGNORECASE)

    with tempfile.TemporaryDirectory(prefix="flashinfer-cuobjdump-") as temp_dir:
        temp_root = Path(temp_dir)
        with zipfile.ZipFile(wheel.path) as archive:
            for module, archive_path in sorted(module_paths.items()):
                extracted_path = Path(archive.extract(archive_path, temp_root))
                process = subprocess.run(
                    [str(cuobjdump), "--list-elf", str(extracted_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                require(
                    process.returncode == 0,
                    f"cuobjdump failed for {module}: {process.stderr.strip()}",
                )
                targets = sorted(
                    {
                        f"sm{match.lower()}"
                        for match in architecture_pattern.findall(process.stdout)
                    }
                )
                require(targets, f"cuobjdump found no cubin targets in {module}")
                require(
                    targets == [provider],
                    f"{module} contains {targets}; expected only {provider}",
                )
                result[module] = targets
    return result


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
) -> None:
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider_id": provider,
        "version": manifest["version"],
        "cuda_architectures": manifest["cuda_architectures"],
        "module_count": len(manifest["modules"]),
        "modules": sorted(manifest["modules"]),
        "module_cuda_architectures": module_architectures,
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
    parser.add_argument("--cuobjdump", type=Path)
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
        provider_wheel, args.provider, args.version
    )
    validate_shim(shim, args.provider, args.version)

    module_architectures: dict[str, list[str]] = {}
    if args.cuobjdump is not None:
        module_architectures = inspect_cuda_architectures(
            provider_wheel,
            module_paths,
            args.provider,
            args.cuobjdump,
        )
    if args.install_smoke:
        run_install_smoke(shim, provider_wheel, args.provider)

    write_report(
        args.wheelhouse,
        wheels,
        args.provider,
        manifest,
        module_architectures,
    )
    print(
        f"Validated {args.provider}: {len(module_paths)} modules, "
        f"{len(wheel_paths)} wheels"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
