# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy, native Rubin compiler capability checks."""

from __future__ import annotations

import os


def require_sm107_dsl() -> None:
    configured = os.environ.get("CUTE_DSL_ARCH")
    if configured not in (None, "sm_107", "sm_107a"):
        raise RuntimeError(
            f"SM107 MegaMoE requires CUTE_DSL_ARCH=sm_107a before Python starts; got {configured!r}."
        )
    os.environ["CUTE_DSL_ARCH"] = "sm_107a"
    try:
        from cutlass.base_dsl.arch import Arch
        from cutlass.utils import rubin_helpers

        Arch["sm_107a"]
        if not callable(
            getattr(rubin_helpers, "make_blockscaled_trivial_tiled_mma", None)
        ):
            raise AttributeError("missing Rubin block-scaled MMA helper")
    except (ImportError, KeyError, AttributeError) as exc:
        raise RuntimeError(
            "SM107 MegaMoE requires a CuTe DSL build with native sm_107a and "
            "cutlass.utils.rubin_helpers support (public 4.8.0.dev0 or a compatible "
            "newer build). The general FlashInfer DSL minimum does not provide "
            "this backend. See the SM107 qualification runbook."
        ) from exc

    from flashinfer.cute_dsl.availability import _dsl_captured_arch

    captured = _dsl_captured_arch()
    if captured not in ("sm_107", "sm_107a"):
        raise RuntimeError(
            f"CuTe DSL already captured target {captured!r}; restart Python with "
            "CUTE_DSL_ARCH=sm_107a exported before importing FlashInfer or cutlass."
        )
