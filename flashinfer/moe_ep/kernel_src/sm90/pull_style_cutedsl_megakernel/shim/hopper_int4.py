# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Signed INT4/K128-BF16 weights on the SM90 FP8 MegaMoE pipeline.

Weights are converted in registers as RN_E4M3_satfinite(RN_BF16(q * scale)). The
activation, communication, SwiGLU and accumulation contracts are inherited
from the packed-weight hybrid kernel. The entry point supports tactic
selection and the common fused launcher.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Callable, ClassVar, Optional, Tuple

import torch

from .comm import resolve_gate_up_clamp
from .hopper_fp8 import MegaMoEHopperFp8Frontend
from .hopper_mxfp4 import (
    MegaMoEHopperMxfp4Config,
    MegaMoEHopperMxfp4Frontend,
    MegaMoEHopperMxfp4Inputs,
    MegaMoEHopperMxfp4SymmBuffer,
    _allocate_fp4_symm_buffer,
    _build_fp4_inputs,
)

TransformedInt4Weights = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
_INT4_TUNING_DTYPE_ID = (
    "sm90_w_int4_signed_k128_bf16_xor_reglut_linear_m64_a_fp8_hybrid_fused_v1"
)


def _validate_complete_int4_tactic(knobs, *, source):
    from .int4_tuner import normalize_hopper_int4_tactic

    try:
        return normalize_hopper_int4_tactic(knobs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class MegaMoEHopperInt4Config(MegaMoEHopperMxfp4Config):
    """Packed INT4 tactic; group size and scale type are fixed."""

    _weight_format: ClassVar[str] = "INT4"
    mma_tiler_mnk: Tuple[int, int, int] = (256, 64, 128)
    compact_pull_buffer: bool = True

    def _validate_conversion_layout(self) -> None:
        if self.fc1_ready_mode != "tile" or self.fc2_tail_n8:
            raise ValueError(
                "INT4 currently requires fc1_ready_mode='tile', fc2_tail_n8=False"
            )


@dataclasses.dataclass
class MegaMoEHopperInt4Inputs(MegaMoEHopperMxfp4Inputs):
    _weight_format: ClassVar[str] = "INT4"


class MegaMoEHopperInt4Frontend(MegaMoEHopperMxfp4Frontend):
    """Reuse the fused launcher with INT4 payload and FP8 register tables."""

    _weight_format = "INT4"
    _config_type = MegaMoEHopperInt4Config
    _inputs_type = MegaMoEHopperInt4Inputs
    _folded_scale_rows = 8
    _folded_scale_cols = 8
    _folded_scale_dtype = torch.uint64
    _validate_tactic = staticmethod(_validate_complete_int4_tactic)

    def _mega_compile_key(self) -> tuple:
        c = self.config
        return (
            "sm90_int4_fp8_megamoe_v2",
            "signed_int4_bf16_k128_xor_reglut_linear_m64",
            ("optimization_policy", c.optimizations.identity()),
            ("compact_pull_buffer", c.compact_pull_buffer),
            ("dispatch_pull", c.dispatch_pull_batch, c.dispatch_pull_rotate),
            *MegaMoEHopperFp8Frontend._mega_compile_key(self),
        )

    @staticmethod
    def _assert_mirrored_constants() -> None:
        MegaMoEHopperFp8Frontend._assert_mirrored_constants()

    def _kernel_class(self):
        from .int4_kernels import Sm90MegaMoESwapABInt4Fp8Kernel

        return Sm90MegaMoESwapABInt4Fp8Kernel


@dataclasses.dataclass
class MegaMoEHopperInt4SymmBuffer(MegaMoEHopperMxfp4SymmBuffer):
    _frontend: MegaMoEHopperInt4Frontend


def resolve_hopper_int4_knobs(
    knobs: Optional[Any],
    *,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_total_experts: int,
    num_topk: int,
    num_max_tokens: int,
    gate_up_clamp: Optional[float] = 10.0,
    routing_profile: str = "block_permutation_v1",
    num_tokens: Optional[int] = None,
) -> dict:
    """Read the INT4 cache, falling back to the first legal search tactic."""
    if isinstance(knobs, dict):
        return _validate_complete_int4_tactic(knobs, source="explicit INT4 knobs")
    if knobs is not None:
        raise ValueError(
            "knobs must be None or an INT4 tactic dict; use "
            "autotune_hopper_int4_mega_moe for collective tuning"
        )
    from .int4_tuner import (
        hopper_int4_cache_provenance_sha256,
        hopper_int4_candidates,
    )
    from .knob_cache import lookup_knobs
    from .mxfp4_tuner import (
        is_hopper_mxfp4_tactic_shape_compatible,
        require_hopper_mxfp4_fused_tuning_device,
    )

    require_hopper_mxfp4_fused_tuning_device()
    domain: dict[str, Any] = dict(
        max_tokens=num_max_tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_total_experts,
        world_size=world_size,
        routing_profile=routing_profile,
    )
    cached = lookup_knobs(
        dtype=_INT4_TUNING_DTYPE_ID,
        fp8_scale_mode="mxfp4_hybrid",
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_total_experts,
        topk=num_topk,
        max_tokens=num_max_tokens,
        live_tokens=num_tokens,
        gate_up_clamp=gate_up_clamp,
        routing_profile=routing_profile,
        tuning_provenance_sha256=hopper_int4_cache_provenance_sha256(**domain),
    )
    if cached is not None:
        tactic = _validate_complete_int4_tactic(
            cached, source="INT4 tuning-cache entry"
        )
        if not is_hopper_mxfp4_tactic_shape_compatible(
            tactic, hidden=hidden, intermediate=intermediate
        ):
            raise ValueError(
                "INT4 tuning-cache tactic is incompatible with the model shape"
            )
        return tactic
    return hopper_int4_candidates(**domain)[0]


def _resolve_hopper_int4_mega_moe_config(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    gate_up_clamp: Optional[float] = 10.0,
    activation_clamp: Optional[float] = None,
    knobs: Optional[Any] = None,
    **config_kwargs,
) -> MegaMoEHopperInt4Config:
    manual_fields = {
        "swap_ab",
        "pingpong",
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "group_hint",
        "num_sched_stages",
        "load_balance_mode",
        "dedup_dispatch",
        "active_dispatch_warps",
        "fc1_store_offload",
        "fc1_early_done_publish",
        "fold_producer_warps",
    }
    config_kwargs = {
        key: value for key, value in config_kwargs.items() if value is not None
    }
    manual = bool(manual_fields.intersection(config_kwargs))
    if manual and knobs is not None:
        raise ValueError(
            "pass either explicit INT4 geometry/execution axes or knobs=, not both"
        )
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp, activation_clamp=activation_clamp
    )
    config = MegaMoEHopperInt4Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        gate_up_clamp=clamp,
        **config_kwargs,
    )
    if manual:
        return config
    from .mxfp4_optimization import MXFP4_PULL_DEFAULTS

    tactic = resolve_hopper_int4_knobs(
        knobs,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_total_experts=num_total_experts,
        num_topk=num_topk,
        num_max_tokens=num_max_tokens,
        gate_up_clamp=clamp,
        routing_profile=config.routing_profile,
    )
    return dataclasses.replace(
        config, **{**MXFP4_PULL_DEFAULTS, **tactic, **config_kwargs}
    )


def get_symm_buffer_for_hopper_int4_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    gate_up_clamp: Optional[float] = 10.0,
    activation_clamp: Optional[float] = None,
    knobs: Optional[Any] = None,
    **config_kwargs,
) -> MegaMoEHopperInt4SymmBuffer:
    """Allocate using cached/default tactics or cache-free explicit geometry.

    Stage inputs with ``stage_mega_moe_inputs`` (hybrid per-token FP8). All
    ranks allocate and launch in the same order. Collective tuning is exposed
    by ``autotune_hopper_int4_mega_moe``.
    """
    config = _resolve_hopper_int4_mega_moe_config(
        num_total_experts,
        num_max_tokens,
        num_topk,
        hidden,
        intermediate,
        rank,
        world_size,
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
        knobs=knobs,
        **config_kwargs,
    )
    return _get_symm_buffer_for_hopper_int4_mega_moe_from_resolved_config(config)


def _get_symm_buffer_for_hopper_int4_mega_moe_from_resolved_config(
    config: MegaMoEHopperInt4Config,
) -> MegaMoEHopperInt4SymmBuffer:
    return _allocate_fp4_symm_buffer(
        config, MegaMoEHopperInt4Frontend, MegaMoEHopperInt4SymmBuffer
    )


def _resolve_transformed_int4_weights(transformed, leg):
    if not isinstance(transformed, (tuple, list)) or len(transformed) != 4:
        raise ValueError(
            f"transformed_{leg} must be (packed_weight, folded_scale, "
            "unit_activation_scale, unit_expert_scale)"
        )
    if any(not isinstance(t, torch.Tensor) for t in transformed):
        raise ValueError(f"transformed_{leg} must contain four tensors")
    return tuple(transformed)


def _build_int4_inputs(symm_buffer, transformed_l1, transformed_l2):
    return _build_fp4_inputs(
        symm_buffer,
        transformed_l1,
        transformed_l2,
        MegaMoEHopperInt4Inputs,
        _resolve_transformed_int4_weights,
    )


def hopper_int4_mega_moe(
    y: Optional[torch.Tensor],
    transformed_l1: TransformedInt4Weights,
    transformed_l2: TransformedInt4Weights,
    symm_buffer: MegaMoEHopperInt4SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    sync: bool = False,
) -> Optional[torch.Tensor]:
    """Launch fused dispatch, signed INT4 FC1/FC2 and BF16 combine.

    Inactive capacity rows must have topk_idx=-1; use the common input staging
    helper to reset them between launches. Returned output aliases workspace.
    """
    if not fast_math:
        warnings.warn(
            "fast_math=False has no effect in the CuTeDSL SM90 INT4 path.", stacklevel=2
        )
    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")
    n = symm_buffer.num_max_tokens if num_tokens is None else num_tokens
    if not 0 <= n <= symm_buffer.num_max_tokens:
        raise ValueError("num_tokens is outside the symmetric buffer capacity")
    if y is not None and (
        tuple(y.shape) != (n, symm_buffer.hidden)
        or y.dtype != torch.bfloat16
        or y.device != symm_buffer.output_activation.device
    ):
        raise ValueError("y must be BF16 [num_tokens, hidden] on the workspace device")
    if n == 0 and symm_buffer._frontend.config.in_kernel_fc2_reduce:
        return symm_buffer.output_activation[:0] if y is None else None
    if gate_up_clamp is not None or activation_clamp is not None:
        symm_buffer._frontend.set_gate_up_clamp(
            resolve_gate_up_clamp(
                gate_up_clamp=gate_up_clamp, activation_clamp=activation_clamp
            )
        )
    inputs = _build_int4_inputs(symm_buffer, transformed_l1, transformed_l2)
    out = symm_buffer._frontend.run(inputs, num_tokens=None, sync=False)
    if y is not None:
        if out is not None:
            y.copy_(out[:n])
        result = None
    else:
        result = out[:n] if out is not None else symm_buffer.output_activation[:0]
    if sync and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()
    return result


def hopper_int4_mega_launch_thunk(
    transformed_l1: TransformedInt4Weights,
    transformed_l2: TransformedInt4Weights,
    symm_buffer: MegaMoEHopperInt4SymmBuffer,
) -> Callable[[], None]:
    """Compile/validate once and bind a launch to the current CUDA stream.

    Construct the thunk on the intended capture stream before graph capture.
    Rebuild after changing the configuration, pointers, or CUDA stream.
    """
    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")
    return symm_buffer._frontend.make_launch_thunk(
        _build_int4_inputs(symm_buffer, transformed_l1, transformed_l2)
    )


__all__ = [
    "MegaMoEHopperInt4Config",
    "MegaMoEHopperInt4Frontend",
    "MegaMoEHopperInt4Inputs",
    "MegaMoEHopperInt4SymmBuffer",
    "TransformedInt4Weights",
    "resolve_hopper_int4_knobs",
    "get_symm_buffer_for_hopper_int4_mega_moe",
    "hopper_int4_mega_moe",
    "hopper_int4_mega_launch_thunk",
]
