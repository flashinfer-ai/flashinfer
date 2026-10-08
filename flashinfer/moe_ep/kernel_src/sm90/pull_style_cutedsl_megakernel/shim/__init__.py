# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Thin adapters over the SM90 (Hopper) ``src/`` kernel drop.

Mirrors the SM100 package's shim layer
(``kernel_src/sm100/cutedsl_megamoe/shim``): all adaptation over the verbatim
``src/`` drop lives here, and the package ``__init__`` re-exports only from
this layer.  ``comm`` holds dist / symmetric-heap / compile helpers;
``hopper_fp8`` holds the SM90 FP8 lazy-compile frontend plus the
symmetric-buffer + fused-launch wrappers, ``hopper_bf16`` the BF16 twin
(``tuner_bf16`` / ``autotune_bf16`` are its knob surface; the knob cache is
shared).
"""

from __future__ import annotations

# hopper_fp8 imports the raw kernel packages (moe_hopper_fp8, common, ...)
# lazily, but src/ must be on sys.path before any of them resolve.  The
# bootstrap lives here in shim/ (not in src/, which is a verbatim kernel drop)
# and also guards against the sibling SM100 tree owning this process.
from ._paths import bootstrap_paths

bootstrap_paths()

from .comm import (
    bootstrap_dist,
    finalize_dist,
    free_sym_tensor,
    reset_compiled_mega_workspaces,
    resolve_gate_up_clamp,
    sym_zeros,
)
from .autotune import (
    autotune_hopper_fp8_mega_moe,
    autotune_knobs,
    hopper_fp8_candidates,
)
from .knob_cache import (
    knob_cache_path,
    lookup_knobs,
    record_knobs,
    resolve_knobs,
)
from .tuner import (
    default_knobs,
    is_valid,
    iter_candidates,
    with_knobs,
)
from .hopper_fp8 import (
    MegaMoEHopperFp8Config,
    MegaMoEHopperFp8Frontend,
    MegaMoEHopperFp8Inputs,
    MegaMoEHopperFp8SymmBuffer,
    TransformedFp8Weights,
    create_dummy_inputs as create_dummy_hopper_fp8_inputs,
    get_symm_buffer_for_hopper_fp8_mega_moe,
    hopper_fp8_mega_launch_thunk,
    hopper_fp8_mega_moe,
    init_dist,
)
from .autotune_bf16 import (
    autotune_hopper_bf16_mega_moe,
    hopper_bf16_candidates,
)
from .tuner_bf16 import (
    default_knobs as bf16_default_knobs,
    is_valid as bf16_is_valid,
    iter_candidates as bf16_iter_candidates,
    lookup_knobs as bf16_lookup_knobs,
    record_knobs as bf16_record_knobs,
    resolve_knobs as bf16_resolve_knobs,
)
from .hopper_bf16 import (
    MegaMoEHopperBf16Config,
    MegaMoEHopperBf16Frontend,
    MegaMoEHopperBf16Inputs,
    MegaMoEHopperBf16SymmBuffer,
    TransformedBf16Weights,
    create_dummy_inputs as create_dummy_hopper_bf16_inputs,
    get_symm_buffer_for_hopper_bf16_mega_moe,
    hopper_bf16_mega_launch_thunk,
    hopper_bf16_mega_moe,
)

__all__ = [
    # paths
    "bootstrap_paths",
    # comm
    "bootstrap_dist",
    "finalize_dist",
    "free_sym_tensor",
    "reset_compiled_mega_workspaces",
    "resolve_gate_up_clamp",
    "sym_zeros",
    # tuner / knob cache / autotune
    "autotune_hopper_fp8_mega_moe",
    "autotune_knobs",
    "default_knobs",
    "hopper_fp8_candidates",
    "is_valid",
    "iter_candidates",
    "knob_cache_path",
    "lookup_knobs",
    "record_knobs",
    "resolve_knobs",
    "with_knobs",
    # hopper_fp8
    "MegaMoEHopperFp8Config",
    "MegaMoEHopperFp8Frontend",
    "MegaMoEHopperFp8Inputs",
    "MegaMoEHopperFp8SymmBuffer",
    "TransformedFp8Weights",
    "create_dummy_hopper_fp8_inputs",
    "get_symm_buffer_for_hopper_fp8_mega_moe",
    "hopper_fp8_mega_launch_thunk",
    "hopper_fp8_mega_moe",
    "init_dist",
    # bf16 tuner / autotune (knob cache shared with fp8)
    "autotune_hopper_bf16_mega_moe",
    "bf16_default_knobs",
    "bf16_is_valid",
    "bf16_iter_candidates",
    "bf16_lookup_knobs",
    "bf16_record_knobs",
    "bf16_resolve_knobs",
    "hopper_bf16_candidates",
    # hopper_bf16
    "MegaMoEHopperBf16Config",
    "MegaMoEHopperBf16Frontend",
    "MegaMoEHopperBf16Inputs",
    "MegaMoEHopperBf16SymmBuffer",
    "TransformedBf16Weights",
    "create_dummy_hopper_bf16_inputs",
    "get_symm_buffer_for_hopper_bf16_mega_moe",
    "hopper_bf16_mega_launch_thunk",
    "hopper_bf16_mega_moe",
]
