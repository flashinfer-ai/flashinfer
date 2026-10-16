# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Which per-arch source tree of the FROST GEMM engine serves this process.

The arch-SPECIFIC half of the engine -- ``compiler.py``, ``epilogue_codegen.py``
and the ``kernel_templates/`` they render -- exists once per ARCH FAMILY, in a
package of its own under ``cudnn/gemm/frost/``::

    sm100/   the tcgen05 line, SM 10.x (sm100, sm103, sm107 ...)
    sm120/   consumer Blackwell, SM 12.x: warp-scoped MMA, STG epilogue

Everything arch-NEUTRAL (graph analysis, recipe, tile configs, the kernel
registry, knobs, the engine contract) stays at ``cudnn/gemm/frost/`` level and
is shared by both trees. A module the TEMPLATES of both families import
(``kernel_templates/split_k_reduction_epilogue_fusion.py``) likewise sits above
them, so the directory a file is in always names its only owner.

``cudnn.gemm.frost.compiler`` and ``cudnn.gemm.frost.epilogue_codegen`` are
facades: on first import they resolve :func:`active_family` and BECOME the
chosen family's module (see the facades for the mechanism). The choice is made
once per process, from the GPU current at that moment -- one family per
process, which is also what the baked ``--gpu-arch`` and the plan-device check
already assume of a plan. Set ``CUDNN_FRONTEND_GEMM_ARCH_FAMILY=sm100|sm120``
to pin the family regardless of the GPU (render-only work, CI without the part).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

FROST_DIR = Path(__file__).parent

#: The per-arch packages, best first (``kernel_registry._AUTO_PIPELINE_ORDER``
#: is the pipeline-level twin).
FAMILIES: tuple[str, ...] = ("sm100", "sm120")

#: Serves a GPU no other family claims -- and no GPU at all (render-only / CI).
#: sm100 is the flagship line and the one the auto path ranks first.
DEFAULT_FAMILY = "sm100"

FAMILY_ENV = "CUDNN_FRONTEND_GEMM_ARCH_FAMILY"


def current_arch(device=None) -> int | None:
    """SM version of ``device`` (default: the current one) as ``major*10+minor``,
    or ``None`` when no GPU is visible (render-only / CI). The one arch probe the
    compiler copies (``_current_arch``) and the facades share."""
    try:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import compute_capability, is_available, resolve_device

        if is_available():
            major, minor = compute_capability(resolve_device(device))
            return major * 10 + minor
    except Exception:  # noqa: BLE001 — render path must work without a GPU
        pass
    return None


def family_for_arch(arch: int | None) -> str:
    """The per-arch tree that serves SM ``arch``: ``sm120`` for SM 12.x, the
    :data:`DEFAULT_FAMILY` for everything else, ``None`` included."""
    if arch is not None and 120 <= arch < 130:
        return "sm120"
    return DEFAULT_FAMILY


def select_family(forced: str | None, arch: int | None) -> str:
    """:func:`active_family` without the environment and GPU reads: ``forced``
    (the ``CUDNN_FRONTEND_GEMM_ARCH_FAMILY`` value) wins when non-empty, else
    ``arch`` decides via :func:`family_for_arch`."""
    forced = (forced or "").strip()
    if forced:
        if forced not in FAMILIES:
            raise ValueError(f"{FAMILY_ENV}={forced!r}: expected one of {', '.join(FAMILIES)}")
        return forced
    return family_for_arch(arch)


def serving_family(arch: int | None) -> str:
    """The family a process on a GPU of SM ``arch`` runs: the
    ``CUDNN_FRONTEND_GEMM_ARCH_FAMILY`` override when set, else
    :func:`family_for_arch`. :func:`active_family` is this, evaluated once for
    the real GPU; the registry evaluates it live against the compiler's arch
    probe so a test that pins ``_current_arch`` reasons about the family that
    GPU would be served by."""
    return select_family(os.environ.get(FAMILY_ENV), arch)


@functools.lru_cache(maxsize=None)
def active_family() -> str:
    """The family this process compiles with -- decided once, on first call.

    ``CUDNN_FRONTEND_GEMM_ARCH_FAMILY`` wins when set; otherwise the GPU current
    at that moment (a ``build_device`` scope included) picks. Cached because the
    facades turn the answer into module identity: ``cudnn.gemm.frost.compiler``
    cannot change family after it has been imported."""
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
