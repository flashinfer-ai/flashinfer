# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Signed INT4/K128-BF16 weights on the SM90 FP8 MegaMoE pipeline.

Weights are converted in registers as RN_E4M3_satfinite(RN_BF16(q * scale)). The
activation, communication, SwiGLU and accumulation contracts are inherited
from the packed-weight hybrid kernel. The entry point uses an explicit
configuration and the common fused launcher.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, ClassVar, Optional, Tuple

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


@dataclasses.dataclass(frozen=True)
class MegaMoEHopperInt4Config(MegaMoEHopperMxfp4Config):
    """Explicit packed INT4 tactic; group size and scale type are fixed."""

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
    **config_kwargs,
) -> MegaMoEHopperInt4SymmBuffer:
    """Collectively allocate INT4 workspace using explicit config fields.

    Stage inputs with ``stage_mega_moe_inputs`` (hybrid per-token FP8). All
    ranks must select the same configuration and launch in the same order.
    """
    config = MegaMoEHopperInt4Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        gate_up_clamp=gate_up_clamp,
        **config_kwargs,
    )
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
    sync: bool = False,
) -> Optional[torch.Tensor]:
    """Launch fused dispatch, signed INT4 FC1/FC2 and BF16 combine.

    Inactive capacity rows must have topk_idx=-1; use the common input staging
    helper to reset them between launches. Returned output aliases workspace.
    """
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
    if gate_up_clamp is not None:
        symm_buffer._frontend.set_gate_up_clamp(
            resolve_gate_up_clamp(gate_up_clamp=gate_up_clamp, activation_clamp=None)
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
    "get_symm_buffer_for_hopper_int4_mega_moe",
    "hopper_int4_mega_moe",
    "hopper_int4_mega_launch_thunk",
]
