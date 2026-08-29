#!/usr/bin/env python3
"""Validate the shared CUDA matrix and its checked-in consumers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


LABEL_PATTERN = re.compile(r"^cu[0-9]+$")
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+$")
PYTORCH_INDEX_PATTERN = re.compile(r"^(?:nightly/)?cu[0-9]+$")
IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*:[A-Za-z0-9_][A-Za-z0-9._-]*$")
CUDNN_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+){3}$")
ARCH_LIST_PATTERN = re.compile(r"^[0-9]+\.[0-9]+[a-z]?(?: [0-9]+\.[0-9]+[a-z]?)*$")
DEPENDENCY_PACKAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEPENDENCY_VERSION_PATTERN = re.compile(
    r"^(?P<release>[0-9]+(?:\.[0-9]+)*)(?:(?P<phase>a|b|rc)(?P<serial>[0-9]+))?$"
)
DEPENDENCY_SPECIFIER_PATTERN = re.compile(
    r"^(?P<operator>==|>=)(?P<version>[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?)$"
)
DEPENDENCY_EXTRA_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CUDA_MAJOR_PATTERN = re.compile(r"^[0-9]+$")


class ConfigError(ValueError):
    """Raised when the shared CUDA configuration is invalid or inconsistent."""


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be an object")
    return value


def _string(
    entry: dict[str, Any], field: str, context: str, pattern: re.Pattern[str]
) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ConfigError(f"{context}.{field} has an invalid value: {value!r}")
    return value


def _entries(config: dict[str, Any], section: str) -> list[dict[str, Any]]:
    entries = config.get(section)
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"{section} must be a non-empty array")
    return [
        _mapping(entry, f"{section}[{index}]") for index, entry in enumerate(entries)
    ]


def _version_key(version: str) -> tuple[int, ...]:
    match = DEPENDENCY_VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ConfigError(f"invalid dependency version: {version!r}")
    release = tuple(int(part) for part in match.group("release").split("."))
    release = (*release, *(0 for _ in range(4 - len(release))))
    phase = match.group("phase")
    phase_rank = {"a": 0, "b": 1, "rc": 2, None: 3}[phase]
    serial = int(match.group("serial") or 0)
    return (*release, phase_rank, serial)


def _toml_string_arrays(path: Path, section: str) -> dict[str, list[str]]:
    """Read the simple string arrays used by a pyproject table."""
    values: dict[str, list[str]] = {}
    in_section = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            if in_section:
                break
            in_section = stripped == f"[{section}]"
            continue
        if not in_section or not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_-]+)\s*=\s*(\[.*\])", stripped)
        if match is None:
            raise ConfigError(f"cannot parse {section} entry in {path}: {stripped!r}")
        try:
            parsed = json.loads(match.group(2))
        except json.JSONDecodeError as error:
            raise ConfigError(
                f"cannot parse {section}.{match.group(1)}: {error}"
            ) from error
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) for item in parsed
        ):
            raise ConfigError(f"{section}.{match.group(1)} must be a string array")
        values[match.group(1)] = parsed
    if not in_section and not values:
        raise ConfigError(f"missing [{section}] in {path}")
    return values


def _dependency_requirement(
    package: str,
    dependency: dict[str, Any],
    specifier_field: str,
    cuda_major: str | None = None,
) -> str:
    extras = dependency.get("cuda_major_extras", {}).get(cuda_major, [])
    package_spec = package
    if extras:
        package_spec += f"[{','.join(extras)}]"
    return f"{package_spec}{dependency[specifier_field]}"


def _validate_dependency_policy(
    config: dict[str, Any], repo_root: Path, cuda_majors: set[str]
) -> None:
    policy = _mapping(config.get("dependency_policy"), "dependency_policy")
    if not policy:
        raise ConfigError("dependency_policy must not be empty")

    requirements_path = repo_root / "requirements.txt"
    requirements = {
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    optional_dependencies = _toml_string_arrays(
        repo_root / "pyproject.toml", "project.optional-dependencies"
    )

    for package, raw_dependency in policy.items():
        if DEPENDENCY_PACKAGE_PATTERN.fullmatch(package) is None:
            raise ConfigError(f"invalid dependency package name: {package!r}")
        context = f"dependency_policy.{package}"
        dependency = _mapping(raw_dependency, context)
        expected_fields = {
            "provider_build_specifier",
            "cuda_extra_specifier",
            "ci_image_specifier",
            "cuda_major_extras",
        }
        if set(dependency) != expected_fields:
            raise ConfigError(
                f"{context} fields must be {sorted(expected_fields)}, "
                f"got {sorted(dependency)}"
            )

        versions = {}
        expected_operators = {
            "provider_build_specifier": ">=",
            "cuda_extra_specifier": ">=",
            "ci_image_specifier": "==",
        }
        for field, expected_operator in expected_operators.items():
            specifier = _string(
                dependency, field, context, DEPENDENCY_SPECIFIER_PATTERN
            )
            match = DEPENDENCY_SPECIFIER_PATTERN.fullmatch(specifier)
            assert match is not None
            if match.group("operator") != expected_operator:
                raise ConfigError(
                    f"{context}.{field} must use {expected_operator!r}, "
                    f"got {specifier!r}"
                )
            versions[field] = match.group("version")
        if _version_key(versions["provider_build_specifier"]) > _version_key(
            versions["cuda_extra_specifier"]
        ):
            raise ConfigError(
                f"{context}.provider_build_specifier must not exceed "
                f"{context}.cuda_extra_specifier"
            )
        if _version_key(versions["ci_image_specifier"]) < _version_key(
            versions["cuda_extra_specifier"]
        ):
            raise ConfigError(
                f"{context}.ci_image_specifier must satisfy the CUDA-extra specifier"
            )

        extras_by_major = _mapping(
            dependency["cuda_major_extras"], f"{context}.cuda_major_extras"
        )
        for cuda_major, extras in extras_by_major.items():
            if CUDA_MAJOR_PATTERN.fullmatch(cuda_major) is None:
                raise ConfigError(f"{context} has invalid CUDA major: {cuda_major!r}")
            if not isinstance(extras, list) or not extras:
                raise ConfigError(f"{context} extras for CUDA {cuda_major} are empty")
            if len(extras) != len(set(extras)) or any(
                not isinstance(extra, str)
                or DEPENDENCY_EXTRA_PATTERN.fullmatch(extra) is None
                for extra in extras
            ):
                raise ConfigError(
                    f"{context} extras for CUDA {cuda_major} are invalid: {extras!r}"
                )

        base_requirement = _dependency_requirement(
            package, dependency, "provider_build_specifier"
        )
        if base_requirement not in requirements:
            raise ConfigError(
                f"requirements.txt must contain the provider-build floor "
                f"{base_requirement!r}"
            )

        for cuda_major in cuda_majors:
            extra_name = f"cu{cuda_major}"
            cuda_requirement = _dependency_requirement(
                package,
                dependency,
                "cuda_extra_specifier",
                cuda_major,
            )
            if cuda_requirement not in optional_dependencies.get(extra_name, []):
                raise ConfigError(
                    f"pyproject.toml {extra_name} extra must contain "
                    f"{cuda_requirement!r}"
                )


def _validate_identity(entry: dict[str, Any], context: str) -> tuple[str, str, str]:
    label = _string(entry, "label", context, LABEL_PATTERN)
    version = _string(entry, "version", context, VERSION_PATTERN)
    pytorch_index = _string(entry, "pytorch_index", context, PYTORCH_INDEX_PATTERN)
    expected_label = f"cu{version.replace('.', '')}"
    if label != expected_label:
        raise ConfigError(
            f"{context}.label must be {expected_label!r} for CUDA {version}, got {label!r}"
        )
    if pytorch_index.rsplit("/", 1)[-1] != label:
        raise ConfigError(
            f"{context}.pytorch_index must select {label!r}, got {pytorch_index!r}"
        )
    return label, version, pytorch_index


def _validate_devcontainer(
    repo_root: Path, runtime: dict[str, Any], context: str
) -> None:
    label = runtime["label"]
    path = repo_root / ".devcontainer" / label / "devcontainer.json"
    if not path.is_file():
        raise ConfigError(f"missing development container for {label}: {path}")

    try:
        devcontainer = _mapping(json.loads(path.read_text()), str(path))
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON in {path}: {error}") from error

    build = _mapping(devcontainer.get("build"), f"{path}.build")
    expected_build = {
        "dockerfile": "../../docker/Dockerfile.ci",
        "context": "../../",
        "target": "dev",
    }
    for name, value in expected_build.items():
        if build.get(name) != value:
            raise ConfigError(
                f"{path}.build.{name} must be {value!r}, got {build.get(name)!r}"
            )

    args = _mapping(build.get("args"), f"{path}.build.args")
    expected = {
        "CUDA_IMAGE": runtime["image"],
        "PYTORCH_INDEX": runtime["pytorch_index"],
        "CUDNN_VERSION": runtime["cudnn_version"],
    }
    for name, value in expected.items():
        if args.get(name) != value:
            raise ConfigError(
                f"{path}.build.args.{name} must match {context}: "
                f"expected {value!r}, got {args.get(name)!r}"
            )


def validate_cuda_config(config: Any, repo_root: Path) -> None:
    """Validate matrix syntax, safe values, and cross-file consistency."""
    config = _mapping(config, "CUDA configuration")
    runtime_entries = _entries(config, "runtime")
    jit_entries = _entries(config, "jit_cache")

    runtime_by_label: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(runtime_entries):
        context = f"runtime[{index}]"
        label, _, _ = _validate_identity(entry, context)
        if label in runtime_by_label:
            raise ConfigError(f"duplicate runtime label: {label}")
        _string(entry, "image", context, IMAGE_PATTERN)
        _string(entry, "cudnn_version", context, CUDNN_PATTERN)
        runtime_by_label[label] = entry
        _validate_devcontainer(repo_root, entry, context)

    jit_by_label: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(jit_entries):
        context = f"jit_cache[{index}]"
        label, _, _ = _validate_identity(entry, context)
        if label in jit_by_label:
            raise ConfigError(f"duplicate JIT-cache label: {label}")
        _string(entry, "x86_64_arch_list", context, ARCH_LIST_PATTERN)
        _string(entry, "aarch64_arch_list", context, ARCH_LIST_PATTERN)
        jit_by_label[label] = entry

    missing_jit = sorted(runtime_by_label.keys() - jit_by_label.keys())
    if missing_jit:
        raise ConfigError(
            "runtime entries are missing matching JIT-cache entries: "
            + ", ".join(missing_jit)
        )
    for label, runtime in runtime_by_label.items():
        jit = jit_by_label[label]
        for field in ("version", "pytorch_index"):
            if runtime[field] != jit[field]:
                raise ConfigError(
                    f"runtime and JIT-cache {label} {field} values differ: "
                    f"{runtime[field]!r} != {jit[field]!r}"
                )

    cuda_majors = {
        entry["version"].split(".", 1)[0] for entry in [*runtime_entries, *jit_entries]
    }
    _validate_dependency_policy(config, repo_root, cuda_majors)


def main() -> int:
    """Validate the configured file and print a concise success message."""
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=default_root / "ci" / "cuda-versions.json",
    )
    parser.add_argument("--repo-root", type=Path, default=default_root)
    args = parser.parse_args()

    try:
        config = json.loads(args.config.read_text())
        validate_cuda_config(config, args.repo_root)
    except (ConfigError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        "Validated CUDA configuration: "
        f"{len(config['runtime'])} runtime images, "
        f"{len(config['jit_cache'])} JIT-cache targets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
