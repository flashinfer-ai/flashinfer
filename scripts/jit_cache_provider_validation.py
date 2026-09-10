"""Shared validation primitives for FlashInfer JIT-cache provider artifacts."""

from __future__ import annotations

import configparser
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Mapping


def canonicalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


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
    wheel: Wheel,
    provider: str,
    expected_version: str,
    expected_platform_tag: str | None = None,
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
    if expected_platform_tag:
        require(
            wheel.path.name.endswith(f"-{expected_platform_tag}.whl"),
            f"Provider wheel {wheel.path.name} does not use {expected_platform_tag}",
        )

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
    parser = configparser.ConfigParser(interpolation=None)
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


def validate_shim(
    wheel: Wheel,
    expected_version: str,
    expected_providers: set[str],
    expected_platform_tag: str | None = None,
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
    if expected_platform_tag:
        require(
            wheel.path.name.endswith(f"-{expected_platform_tag}.whl"),
            f"Shim wheel {wheel.path.name} does not use {expected_platform_tag}",
        )

    normalized_requirements = [
        normalize_requirement(requirement) for requirement in wheel.requirements
    ]
    requirements = dict(normalized_requirements)
    require(
        len(requirements) == len(normalized_requirements),
        "Shim contains duplicate provider requirements",
    )
    expected_requirements = {
        f"flashinfer-jit-cache-{provider}": expected_version
        for provider in expected_providers
    }
    require(
        requirements == expected_requirements,
        f"Shim requirements {requirements} do not match {expected_requirements}",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_validation_report(
    output_dir: Path,
    wheels: Mapping[str, Wheel],
    provider: str,
    manifest: Mapping[str, Any],
    module_architectures: Mapping[str, list[str]],
    ptx_modules: list[str],
    report_filename: str,
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
    (output_dir / report_filename).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    checksum_lines = [
        f"{details['sha256']}  {details['filename']}"
        for details in report["wheels"].values()
    ]
    (output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")


def inspect_cuda_architectures(
    wheel: Wheel,
    module_paths: dict[str, str],
    provider: str,
    cuobjdump: Path,
    strict: bool,
) -> tuple[dict[str, list[str]], list[str]]:
    require(cuobjdump.is_file(), f"cuobjdump not found: {cuobjdump}")
    result: dict[str, list[str]] = {}
    ptx_modules: list[str] = []
    architecture_pattern = re.compile(r"\bsm[_-]?([0-9]{2,3}[af]?)\b", re.IGNORECASE)

    with tempfile.TemporaryDirectory(prefix="flashinfer-cuobjdump-") as temp_dir:
        temp_root = Path(temp_dir)
        with zipfile.ZipFile(wheel.path) as archive:
            for index, (module, archive_path) in enumerate(
                sorted(module_paths.items())
            ):
                extracted_path = temp_root / f"{index}.so"
                extracted_path.write_bytes(archive.read(archive_path))
                process = subprocess.run(
                    [str(cuobjdump), "--list-elf", str(extracted_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if (
                    process.returncode != 0
                    and "does not contain device code" in process.stderr
                ):
                    result[module] = []
                    continue
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
                result[module] = targets
                ptx_process = subprocess.run(
                    [str(cuobjdump), "--list-ptx", str(extracted_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                ptx_output = "\n".join(
                    output.strip()
                    for output in (ptx_process.stdout, ptx_process.stderr)
                    if output.strip()
                )
                require(
                    ptx_process.returncode == 0,
                    f"cuobjdump PTX inspection failed for {module}: {ptx_output}",
                )
                if "No PTX file found" not in ptx_output:
                    ptx_modules.append(module)

    mismatches = {
        module: targets
        for module, targets in result.items()
        if targets and targets != [provider]
    }
    if strict and mismatches:
        examples = ", ".join(
            f"{module}={targets}"
            for module, targets in list(sorted(mismatches.items()))[:10]
        )
        require(
            False,
            f"{len(mismatches)} modules do not contain only {provider}: {examples}",
        )
    if strict and ptx_modules:
        examples = ", ".join(sorted(ptx_modules)[:10])
        require(
            False,
            f"{len(ptx_modules)} native-provider modules contain PTX: {examples}",
        )
    return result, sorted(ptx_modules)
