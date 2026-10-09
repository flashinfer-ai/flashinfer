#!/usr/bin/env python3
"""Validate that nightly tests use the kernel wheels built in this run."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path

from packaging.utils import canonicalize_name


def _required_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"Required nightly package {distribution!r} is not installed"
        ) from error


def _provider_versions() -> dict[str, str]:
    providers = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        normalized = canonicalize_name(name)
        if normalized.startswith("flashinfer-jit-cache-"):
            providers[normalized] = distribution.version
    return providers


def _validation_errors() -> list[str]:
    flashinfer_version = _required_version("flashinfer-python")
    cubin_version = _required_version("flashinfer-cubin")
    shim_version = _required_version("flashinfer-jit-cache")
    errors = []

    if cubin_version != flashinfer_version:
        errors.append(
            "flashinfer-cubin version "
            f"{cubin_version!r} does not match flashinfer-python "
            f"{flashinfer_version!r}"
        )

    if shim_version != flashinfer_version and not shim_version.startswith(
        f"{flashinfer_version}+"
    ):
        errors.append(
            "flashinfer-jit-cache version "
            f"{shim_version!r} does not match flashinfer-python "
            f"{flashinfer_version!r}"
        )

    for distribution, version in sorted(_provider_versions().items()):
        if version != shim_version:
            errors.append(
                f"{distribution} version {version!r} does not match "
                f"flashinfer-jit-cache {shim_version!r}"
            )

    flashinfer_cubin = importlib.import_module("flashinfer_cubin")
    jit_env = importlib.import_module("flashinfer.jit.env")
    package_cubin_dir = Path(flashinfer_cubin.get_cubin_dir()).resolve()
    selected_cubin_dir = Path(jit_env.FLASHINFER_CUBIN_DIR).resolve()
    if selected_cubin_dir != package_cubin_dir:
        errors.append(
            "FlashInfer selected cubin directory "
            f"{selected_cubin_dir}, not the installed flashinfer-cubin directory "
            f"{package_cubin_dir}"
        )

    return errors


def main() -> None:
    errors = _validation_errors()
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"Nightly package validation failed:\n{detail}")
    print("Nightly kernel package versions and cubin directory are consistent.")


if __name__ == "__main__":
    main()
