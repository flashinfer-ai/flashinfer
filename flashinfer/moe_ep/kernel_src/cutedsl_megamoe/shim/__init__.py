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

from .comm import (
    bootstrap_dist,
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

__all__ = [
    # paths
    "bootstrap_paths",
    # comm
    "bootstrap_dist",
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
