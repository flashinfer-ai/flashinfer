# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared source-checkout and device helpers for the sweep tool."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


def available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def repo_root() -> Path:
    root = Path(__file__).resolve().parents[5]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"cannot locate FlashInfer source root from {__file__}")
    return root


def git_output(
    *args: str, repository: str | os.PathLike[str] | None = None
) -> str | None:
    try:
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(repo_root() if repository is None else repository),
                *args,
            ),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.rstrip("\n")


def nvml_driver_version() -> str | None:
    """Return the NVIDIA driver version reported by NVML, if available."""

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            value = pynvml.nvmlSystemGetDriverVersion()
            return value.decode() if isinstance(value, bytes) else str(value)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def resolve_visible_gpu(selector: str, visible_spec: str | None = None) -> str:
    """Resolve and validate a selector against inherited CUDA visibility."""

    if visible_spec is None:
        visible_spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_spec:
        return selector
    devices = [value.strip() for value in visible_spec.split(",") if value.strip()]
    if selector in devices:
        return selector
    try:
        index = int(selector)
    except ValueError:
        index = -1
    if index < 0 or index >= len(devices):
        raise ValueError(
            f"GPU {selector!r} is not in inherited "
            f"CUDA_VISIBLE_DEVICES={visible_spec!r}"
        )
    return devices[index]


def nvml_handle_for_uuid(nvml, uuid: object, *, index: int | None = None):
    """Resolve an NVML device, tolerating str/bytes API differences."""

    values = (
        (uuid, uuid.decode("utf-8"))
        if isinstance(uuid, bytes)
        else (uuid, str(uuid).encode("utf-8"))
    )
    getter = getattr(nvml, "nvmlDeviceGetHandleByUUID", None)
    if getter is not None:
        for value in values:
            try:
                return getter(value)
            except Exception:
                continue
    if index is not None:
        return nvml.nvmlDeviceGetHandleByIndex(index)
    raise RuntimeError(f"NVML could not resolve GPU UUID {uuid}")
