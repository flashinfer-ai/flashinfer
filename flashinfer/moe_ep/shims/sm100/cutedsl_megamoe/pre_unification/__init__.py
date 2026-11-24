# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Legacy Blackwell MegaMoE adapters. Compiler and device imports stay lazy."""

from __future__ import annotations

from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification._paths import (
    bootstrap_paths,
)

bootstrap_paths()


def _check_dsl_perf_floor() -> None:
    """Warn once if the installed CuTe-DSL predates the 4.5.2 perf floor.

    Every published version >=4.5.2 measures at full parity (TUNING.md
    "CuTe-DSL runtime sensitivity" + 2026-07-22 follow-up): 4.5.2 via the
    the mainloop workaround in kernel_fc12.py (const_expr-gated on ==4.5.2 —
    the codegen regression existed only in 4.5.2, fixed upstream in
    4.5.3), 4.5.3/4.6.0/4.6.1 verified natively fast; mxfp8 never
    regressed. Below 4.5.2: 4.5.0 fails outright at cute.compile, 4.5.1
    unmeasured/presumed slow. Results are CORRECT wherever compilation
    succeeds — only slower — so this warns instead of raising. Silence
    with FLASHINFER_MOE_EP_SKIP_DSL_CHECK=1.
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
    if parts and parts < (4, 5, 2):
        import warnings as _warnings

        _warnings.warn(
            f"nvidia-cutlass-dsl {ver} detected: the CuTeDSL MegaMoE kernels "
            "compile 34-54% slower on <4.5.2 (perf floor; results stay "
            "correct where compilation succeeds; 4.5.0 fails to compile). "
            "Install nvidia-cutlass-dsl[cu13]>=4.5.2 — see "
            "shims/sm100/cutedsl_megamoe/TUNING.md 'CuTe-DSL runtime "
            "sensitivity'. Silence with FLASHINFER_MOE_EP_SKIP_DSL_CHECK=1.",
            UserWarning,
            stacklevel=2,
        )


_check_dsl_perf_floor()

from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.comm import (
    bootstrap_dist,
    ensure_not_capturing,
    finalize_dist,
    free_sym_tensor,
    reset_compiled_mega_workspaces,
    resolve_gate_up_clamp,
    sym_zeros,
)

# Light kernel helpers/constants the FI backend + tests need (drop-audit point).
# The heavy mega_runner/mega_reference helpers stay behind kernel_helpers'
# module ``__getattr__`` and are surfaced lazily by the parent ``__init__``.
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.kernel_helpers import (
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
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.nvfp4 import (
    COMBINE_FORMAT_NAMES,
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
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.bf16_mxfp8 import (
    MegaMoEBf16Mxfp8Config,
    MegaMoEBf16Mxfp8Frontend,
    MegaMoEBf16Mxfp8Inputs,
    MegaMoEBf16Mxfp8SymmBuffer,
    create_dummy_inputs as create_dummy_bf16_mxfp8_inputs,
    get_symm_buffer_for_bf16_mxfp8_mega_moe,
    bf16_mxfp8_mega_launch_thunk,
    bf16_mxfp8_mega_moe,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.bf16 import (
    MegaMoEBf16Config,
    MegaMoEBf16Frontend,
    MegaMoEBf16Inputs,
    MegaMoEBf16SymmBuffer,
    bf16_mega_launch_thunk,
    bf16_mega_moe,
    create_dummy_inputs as create_dummy_bf16_inputs,
    get_symm_buffer_for_bf16_mega_moe,
)

# Kernel tuning knobs (tactic enumeration + config application).
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification import tuner
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.tuner import (
    CORRECTNESS_KNOBS,
    PERF_KNOBS,
    default_knobs,
    iter_candidates,
    with_knobs,
)

# Online (warmup-time) collective knob autotuning.
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.autotune import (
    autotune_bf16_mega_moe,
    autotune_bf16_mxfp8_mega_moe,
    autotune_knobs,
    bf16_candidates,
    bf16_mxfp8_candidates,
    autotune_nvfp4_mega_moe,
    nvfp4_candidates,
)

from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.bf16_stage import (
    fused_bf16_stage,
)

# Fused bf16 -> quant + routing staging (single-launch DataPreprocess).
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.quant_stage import (
    forget_staged_tokens,
    fused_quant_stage,
    fused_quant_stage_supported,
    note_staged_tokens,
    staged_tokens,
)

# Persistent offline-tuning knob cache (pure-lookup hot path).
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.knob_cache import (
    knob_cache_path,
    lookup_knobs,
    record_knobs,
    resolve_knobs,
)


from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification import (
    kernel_helpers as _kernel_helpers,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.autotune import (
    _session_candidates,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.comm import (
    _CompiledMega,
    _compute_peer_offsets,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.nvfp4 import (
    _resolve_per_expert_epilogue,
)

_LAZY_HELPERS = (
    *_kernel_helpers.__all__,
    "CombineFormat",
    "combine_roundtrip_to_fp32",
    "compute_megamoe_reference_bf16",
    "compute_megamoe_reference_bf16_mxfp8",
)


def __getattr__(name):
    if name in _LAZY_HELPERS:
        return getattr(_kernel_helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Backward-compatible name: ``create_dummy_inputs`` historically meant NVFP4.
create_dummy_inputs = create_dummy_nvfp4_inputs

__all__ = [
    "_CompiledMega",
    "_compute_peer_offsets",
    "_resolve_per_expert_epilogue",
    "_session_candidates",
    "free_sym_tensor",
    "resolve_gate_up_clamp",
    "sym_zeros",
    "COMBINE_FORMAT_NAMES",
    "CombineFormat",
    "MegaMoEBf16Inputs",
    "MegaMoEBf16SymmBuffer",
    "MegaMoEBf16Mxfp8Config",
    "MegaMoEBf16Mxfp8Inputs",
    "MegaMoEBf16Mxfp8SymmBuffer",
    "MegaMoENvfp4Config",
    "MegaMoENvfp4Inputs",
    "MegaMoESymmBuffer",
    "Mxfp8BlockSize",
    "Mxfp8ScaleDtype",
    "Nvfp4BlockSize",
    "TransformedWeights",
    "autotune_knobs",
    "autotune_bf16_mega_moe",
    "autotune_bf16_mxfp8_mega_moe",
    "autotune_nvfp4_mega_moe",
    "bootstrap_paths",
    "ceil_div",
    "combine_roundtrip_to_fp32",
    "compute_megamoe_reference_bf16",
    "compute_megamoe_reference_bf16_mxfp8",
    "create_dummy_bf16_inputs",
    "create_dummy_bf16_mxfp8_inputs",
    "create_dummy_inputs",
    "create_dummy_nvfp4_inputs",
    "default_knobs",
    "ensure_not_capturing",
    "finalize_dist",
    "forget_staged_tokens",
    "fused_quant_stage",
    "fused_bf16_stage",
    "fused_quant_stage_supported",
    "bf16_mxfp8_candidates",
    "get_symm_buffer_for_mega_moe",
    "get_symm_buffer_for_bf16_mxfp8_mega_moe",
    "get_symm_buffer_for_bf16_mega_moe",
    "init_dist",
    "kind_data_dtype",
    "knob_cache_path",
    "lookup_knobs",
    "make_dummy_epilogue_params",
    "note_staged_tokens",
    "staged_tokens",
    "bf16_mxfp8_mega_launch_thunk",
    "bf16_mxfp8_mega_moe",
    "bf16_mega_launch_thunk",
    "bf16_mega_moe",
    "mxfp8_quantize_per_block_32",
    "nvfp4_candidates",
    "nvfp4_mega_launch_thunk",
    "nvfp4_mega_moe",
    "nvfp4_quantize_per_block_16",
    "record_knobs",
    "resolve_knobs",
    "round_up",
    "to_blocked",
    "tuner",
    "with_knobs",
    "iter_candidates",
    "CORRECTNESS_KNOBS",
    "PERF_KNOBS",
    "_stack_byte_reinterpretable_tensors",
]
__all__ += [name for name in _kernel_helpers.__all__ if name not in __all__]
