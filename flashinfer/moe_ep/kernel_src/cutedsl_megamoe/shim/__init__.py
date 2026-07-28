# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Thin adapters over the raw CuTeDSL kernel sources under ``../src``.

``comm`` holds dist / symmetric-heap / compile helpers; ``nvfp4`` and ``mxfp8``
each hold their dtype's lazy-compile frontend plus the symmetric-buffer +
fused-launch wrappers.  The parent :mod:`..api` re-exports the curated subset
that FlashInfer ``moe_ep`` consumes.
"""

from __future__ import annotations

# The per-dtype modules import the raw kernel packages (moe_nvfp4_swapab,
# common, ...) at module load, so ``src/`` must be on sys.path first.  The
# bootstrap lives here in shim/ (not in src/, which is a verbatim kernel drop).
from ._paths import bootstrap_paths

bootstrap_paths()


def _check_dsl_perf_floor() -> None:
    """Warn once if the installed CuTe-DSL predates the 4.6.1 perf floor.

    The MegaMoE kernels compile 34-54% slower on nvidia-cutlass-dsl 4.5.x
    (measured 2026-07-15, TUNING.md "CuTe-DSL runtime sensitivity"; vllm_e2e
    RUNS.md run 14). Results are CORRECT on 4.5.x — only slower — so this
    warns instead of raising. Silence with FLASHINFER_MOE_EP_SKIP_DSL_CHECK=1.
    """
    import os as _os

    if _os.environ.get("FLASHINFER_MOE_EP_SKIP_DSL_CHECK") == "1":
        return
    try:
        from importlib.metadata import version as _version

        ver = _version("nvidia-cutlass-dsl")
    except Exception:
        return  # unknown packaging (source checkout etc.) — nothing to claim
    import re as _re

    parts = tuple(int(p) for p in _re.findall(r"\d+", ver)[:3])
    if parts and parts < (4, 6, 1):
        import warnings as _warnings

        _warnings.warn(
            f"nvidia-cutlass-dsl {ver} detected: the CuTeDSL MegaMoE kernels "
            "compile 34-54% slower on <4.6.1 (perf floor; results stay "
            "correct). Install nvidia-cutlass-dsl[cu13]>=4.6.1 — see "
            "kernel_src/cutedsl_megamoe/TUNING.md 'CuTe-DSL runtime "
            "sensitivity'. Silence with FLASHINFER_MOE_EP_SKIP_DSL_CHECK=1.",
            UserWarning,
            stacklevel=2,
        )


_check_dsl_perf_floor()

from .comm import (
    bootstrap_dist,
    finalize_dist,
    free_sym_tensor,
    reset_compiled_mega_workspaces,
    resolve_gate_up_clamp,
    sym_zeros,
)

# Light kernel helpers/constants the FI backend + tests need (drop-audit point).
# The heavy mega_runner/mega_reference helpers stay behind kernel_helpers'
# module ``__getattr__`` and are surfaced lazily by the parent ``__init__``.
from .kernel_helpers import (
    Mxfp8BlockSize,
    Mxfp8ScaleDtype,
    Nvfp4BlockSize,
    _stack_byte_reinterpretable_tensors,
    ceil_div,
    kind_data_dtype,
    mxfp8_quantize_per_block_32,
    nvfp4_quantize_per_block_16,
    round_up,
    to_blocked,
)
from .nvfp4 import (
    MegaMoENvfp4Config,
    MegaMoENvfp4Frontend,
    MegaMoENvfp4Inputs,
    MegaMoESymmBuffer,
    TransformedWeights,
    create_dummy_inputs as create_dummy_nvfp4_inputs,
    get_symm_buffer_for_mega_moe,
    init_dist,
    make_dummy_epilogue_params,
    nvfp4_mega_launch_thunk,
    nvfp4_mega_moe,
)
from .mxfp8 import (
    MegaMoEMxfp8Config,
    MegaMoEMxfp8Frontend,
    MegaMoEMxfp8Inputs,
    MegaMoEMxfp8SymmBuffer,
    create_dummy_inputs as create_dummy_mxfp8_inputs,
    get_symm_buffer_for_mxfp8_mega_moe,
    mxfp8_mega_launch_thunk,
    mxfp8_mega_moe,
)

# Kernel tuning knobs (tactic enumeration + config application).
from . import tuner
from .tuner import (
    CORRECTNESS_KNOBS,
    PERF_KNOBS,
    iter_candidates,
    with_knobs,
)

# Online (warmup-time) collective knob autotuning.
from .autotune import (
    autotune_knobs,
    autotune_mxfp8_mega_moe,
    autotune_nvfp4_mega_moe,
)

# Fused bf16 -> quant + routing staging (single-launch DataPreprocess).
from .quant_stage import (
    forget_staged_tokens,
    fused_quant_stage,
    fused_quant_stage_supported,
    note_staged_tokens,
    staged_tokens,
)

# Persistent offline-tuning knob cache (pure-lookup hot path).
from .knob_cache import lookup_knobs, record_knobs, resolve_knobs

__all__ = [
    # paths
    "bootstrap_paths",
    # quant_stage
    "forget_staged_tokens",
    "fused_quant_stage",
    "fused_quant_stage_supported",
    "note_staged_tokens",
    "staged_tokens",
    # knob_cache
    "lookup_knobs",
    "record_knobs",
    "resolve_knobs",
    # comm
    "bootstrap_dist",
    "finalize_dist",
    "free_sym_tensor",
    "reset_compiled_mega_workspaces",
    "resolve_gate_up_clamp",
    "sym_zeros",
    # kernel_helpers (light)
    "Mxfp8BlockSize",
    "Mxfp8ScaleDtype",
    "Nvfp4BlockSize",
    "_stack_byte_reinterpretable_tensors",
    "ceil_div",
    "kind_data_dtype",
    "mxfp8_quantize_per_block_32",
    "nvfp4_quantize_per_block_16",
    "round_up",
    "to_blocked",
    # nvfp4
    "MegaMoENvfp4Config",
    "MegaMoENvfp4Frontend",
    "MegaMoENvfp4Inputs",
    "MegaMoESymmBuffer",
    "TransformedWeights",
    "create_dummy_nvfp4_inputs",
    "get_symm_buffer_for_mega_moe",
    "init_dist",
    "make_dummy_epilogue_params",
    "nvfp4_mega_launch_thunk",
    "nvfp4_mega_moe",
    # mxfp8
    "MegaMoEMxfp8Config",
    "MegaMoEMxfp8Frontend",
    "MegaMoEMxfp8Inputs",
    "MegaMoEMxfp8SymmBuffer",
    "create_dummy_mxfp8_inputs",
    "get_symm_buffer_for_mxfp8_mega_moe",
    "mxfp8_mega_launch_thunk",
    "mxfp8_mega_moe",
    # tuner
    "tuner",
    "CORRECTNESS_KNOBS",
    "PERF_KNOBS",
    "iter_candidates",
    "with_knobs",
    # autotune
    "autotune_knobs",
    "autotune_mxfp8_mega_moe",
    "autotune_nvfp4_mega_moe",
]
