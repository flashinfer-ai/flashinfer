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
    _mapping(config.get("build_dependencies"), "build_dependencies")
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
