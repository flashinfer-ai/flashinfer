# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Which per-arch source tree of the FROST GEMM engine serves this process."""

from __future__ import annotations

import functools
import os
from pathlib import Path

FROST_DIR = Path(__file__).parent

FAMILIES: tuple[str, ...] = ("sm100",)  # [vendor.py P6] the vendored tree carries the sm100 family only

DEFAULT_FAMILY = "sm100"

FAMILY_ENV = "CUDNN_FRONTEND_GEMM_ARCH_FAMILY"


def current_arch(device=None) -> int | None:
    """SM version of ``device`` (default: the current one) as ``major*10+minor``, or ``None`` when no GPU is visible (render-only / CI)."""
    try:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import compute_capability, is_available, resolve_device

        if is_available():
            major, minor = compute_capability(resolve_device(device))
            return major * 10 + minor
    except Exception:  # noqa: BLE001 — render path must work without a GPU
        pass
    return None


def family_for_arch(arch: int | None) -> str:
    """[vendor.py P6] The vendored tree carries one per-arch tree, :data:`DEFAULT_FAMILY`:
    it serves every SM ``arch`` (``None`` included); a device outside its range is
    declined, by name, by the kernels' own device gates."""
    return DEFAULT_FAMILY


def select_family(forced: str | None, arch: int | None) -> str:
    """Select the source family from an override or GPU architecture."""
    forced = (forced or "").strip()
    if forced:
        if forced not in FAMILIES:
            raise ValueError(f"{FAMILY_ENV}={forced!r}: expected one of {', '.join(FAMILIES)}")
        return forced
    return family_for_arch(arch)


def serving_family(arch: int | None) -> str:
    """Resolve the source family for the requested GPU architecture."""
    return select_family(os.environ.get(FAMILY_ENV), arch)


@functools.lru_cache(maxsize=None)
def active_family() -> str:
    """The family this process compiles with -- decided once, on first call."""
    return serving_family(current_arch())


def family_dir(family: str) -> Path:
    """``cudnn/gemm/frost/<family>/``."""
    return FROST_DIR / family


def template_dir(family: str) -> Path:
    """``cudnn/gemm/frost/<family>/kernel_templates/``."""
    return FROST_DIR / family / "kernel_templates"


def template_files() -> list[Path]:
    """Every kernel template that ships, across all families -- the ``sm*.py``
    inventory the source-level tests sweep."""
    return sorted(p for family in FAMILIES for p in template_dir(family).glob("sm*.py"))
