# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Vendored ``next/`` greenfield CuTe-DSL MegaMoE drop (SM107 / Rubin, inference).

Layout (see VENDOR.md for provenance and the recorded local diffs):

- ``src/`` — the kernel team's drop, verbatim.  Its single top-level package
  (``sources``) resolves via ``shim/_paths.py``.
- ``shim/`` — the only code that imports ``src/``.

moe_ep backends import from this ``__init__`` only (never shim submodules) —
see ``kernel_src/README.md`` for the layering rules.  Shim modules keep their
cutlass / cuda imports lazy, so importing this package stays CPU-safe.
"""

from .shim import (
    Sm107BlockScaledMoeConfig,
    Sm107BlockScaledSymmBuffer,
    Sm107QuantKind,
    TransformedBlockScaledWeights,
    bootstrap_paths,
    ensure_not_capturing,
    free_sym_tensor,
    get_symm_buffer_for_sm107_block_scaled_mega_moe,
    sm107_block_scaled_mega_launch_thunk,
    sm107_block_scaled_mega_moe,
    sym_zeros,
)

# Torch-side kernel helpers (quant / SF swizzle / oracle reference) resolve
# lazily so this package import never pulls anything CUDA-adjacent.
_LAZY_HELPERS = (
    "GateUpInterleave",
    "Mxfp8BlockSize",
    "Nvfp4BlockSize",
    "ceil_div",
    "compute_megamoe_reference_sm107_block_scaled",
    "e8m0_to_f32",
    "interleave_gate_up_16",
    "pack_f32_to_fp4",
    "quantize_mxfp8_block32",
    "quantize_nvfp4_block16",
    "round_up",
    "scale_to_f32",
    "swizzled_flat_sf_size",
    "to_blocked",
    "unpack_fp4_to_f32",
)


def __getattr__(name):  # PEP 562
    if name in _LAZY_HELPERS:
        from .shim import kernel_helpers

        return getattr(kernel_helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *_LAZY_HELPERS,
    "Sm107BlockScaledMoeConfig",
    "Sm107BlockScaledSymmBuffer",
    "Sm107QuantKind",
    "TransformedBlockScaledWeights",
    "bootstrap_paths",
    "ensure_not_capturing",
    "free_sym_tensor",
    "get_symm_buffer_for_sm107_block_scaled_mega_moe",
    "sm107_block_scaled_mega_launch_thunk",
    "sm107_block_scaled_mega_moe",
    "sym_zeros",
]
