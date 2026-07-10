# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Thin CuTeDSL NVFP4 MegaMoE API (vLLM-oriented).

NVFP4 activations + NVFP4 expert weights, with fp8 plain scale-factor planes.
Output is bf16 after top-k combine.  Call layout mirrors DeepGEMM's
``deep_gemm.fp8_fp4_mega_moe`` (symm buffer + weight tuples + one launch), but
this path is **not** FP8-activation MegaMoE — see ``dummy_fp8_fp4_mega_moe.py``
for the DeepGEMM FP8+FP4 reference.

  * Activations: NVFP4 data + fp8 plain SF (not FP8 UE8M0 activations).
  * Weights: NVFP4 + atom-swizzled SF (kernel-ready layout from ``mega_runner``).
  * fc1 layout: ``(E, hidden, intermediate)`` swap-AB gate+up.

Workspace vs per-launch inputs (mirrors DeepGEMM):

  * :func:`get_symm_buffer_for_mega_moe` owns activations, routing, combine
    staging, and per-local-expert epilogue scalars (``fc1_alpha``, ``fc2_alpha``,
    ``fc1_norm_const``).
  * Expert weights are **not** in the symm buffer — pass kernel-ready
    ``(weight, scale)`` tuples to :func:`nvfp4_mega_moe` on every launch.
    Use ``mega_runner`` weight assembly or a host preprocessor (e.g.
    ``flashinfer`` ``preprocess_mega_weights``); :func:`create_dummy_inputs`
    only generates random weights for smoke/benchmark scripts.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch

from .megamoe_frontend.api_nvfp4 import (
    MegaMoENvfp4Config,
    MegaMoENvfp4Frontend,
    MegaMoENvfp4Inputs,
)
from .megamoe_frontend.common import bootstrap_dist, free_sym_tensor, sym_zeros
from ._util import resolve_gate_up_clamp
from common.megamoe_constants import Nvfp4BlockSize
from moe_nvfp4_swapab.runner_common import (
    _DataDtype,
    _ScaleDtype,
    ceil_div,
    round_up,
)

TransformedWeights = Tuple[torch.Tensor, torch.Tensor]


PerExpertEpilogue = Union[torch.Tensor, int, float]


def _resolve_per_expert_epilogue(
    name: str,
    value: Optional[PerExpertEpilogue],
    num_experts_per_rank: int,
) -> torch.Tensor:
    """Build a per-local-expert fp32 CUDA vector (default 1.0)."""
    out = torch.ones(
        (num_experts_per_rank,),
        dtype=torch.float32,
        device="cuda",
    )
    if value is None:
        return out
    if isinstance(value, (int, float)):
        out.fill_(float(value))
        return out
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got device {value.device}.")
    if value.shape != (num_experts_per_rank,):
        raise ValueError(
            f"{name} must have shape ({num_experts_per_rank},), "
            f"got {tuple(value.shape)}."
        )
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {value.dtype}.")
    out.copy_(value)
    return out


def _sym_zeros_byte_view(
    logical_shape: Tuple[int, ...],
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 / fp8 symmetric heap via uint8 reinterpret (matches mega_runner).

    Returns ``(view, root_uint8_buffer)``; free the root via :func:`free_sym_tensor`.
    """
    if target_dtype == _DataDtype:
        if not logical_shape or logical_shape[-1] % 2 != 0:
            raise ValueError(
                f"NVFP4 sym view needs non-empty logical_shape with even last dim, "
                f"got {logical_shape}."
            )
        storage_shape = (*logical_shape[:-1], logical_shape[-1] // 2)
    elif target_dtype == _ScaleDtype:
        storage_shape = tuple(logical_shape)
    else:
        raise ValueError(
            f"_sym_zeros_byte_view: dtype must be {_DataDtype} or {_ScaleDtype}, "
            f"got {target_dtype}."
        )
    total_bytes = 1
    for dim_size in storage_shape:
        total_bytes *= dim_size
    root = sym_zeros((total_bytes,), torch.uint8)
    view = root.view(target_dtype).reshape(storage_shape)
    return view, root


def init_dist() -> Tuple[int, int]:
    """Initialize torch.distributed + NVSHMEM (or single-rank when ``MEGA_NO_DIST=1``).

    Returns ``(rank, world_size)``.
    """
    _, rank, world_size, _ = bootstrap_dist()
    return rank, world_size


@dataclass
class MegaMoESymmBuffer:
    """Symmetric-heap staging buffers for one MegaMoE session.

    Mirrors DeepGEMM's symm-buffer object: exposes ``x``, ``x_sf``,
    ``topk_idx``, and ``topk_weights`` views sized for ``num_max_tokens``.
    Internal combine / epilogue tensors are owned here but not part of the
    public DeepGEMM surface.

    Expert weights are **not** stored here — pass ``transformed_l1`` /
    ``transformed_l2`` to :func:`nvfp4_mega_moe` each launch.
    """

    num_total_experts: int
    num_max_tokens: int
    num_topk: int
    hidden: int
    intermediate: int
    rank: int
    world_size: int

    x: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    combine_output: torch.Tensor
    combine_reduced_output: torch.Tensor
    fc1_alpha: torch.Tensor
    fc2_alpha: torch.Tensor
    fc1_norm_const: torch.Tensor

    _frontend: MegaMoENvfp4Frontend
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    _destroyed: bool = False

    def destroy(self) -> None:
        """Release symmetric-heap allocations and compiled kernel workspaces."""
        if self._destroyed:
            return
        self._frontend.release()
        for root in self._sym_roots:
            free_sym_tensor(root)
        self._sym_roots.clear()
        self._destroyed = True

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size


def get_symm_buffer_for_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    apply_topk_in_fc1: bool = True,
    fc1_alpha: Optional[PerExpertEpilogue] = None,
    fc2_alpha: Optional[PerExpertEpilogue] = None,
    fc1_norm_const: Optional[PerExpertEpilogue] = None,
) -> MegaMoESymmBuffer:
    """Allocate symmetric-heap inputs + combine staging for one MegaMoE session.

    Argument order follows ``deep_gemm.get_symm_buffer_for_mega_moe`` (problem
    sizes first).  Pass ``rank`` / ``world_size`` from :func:`init_dist` instead
    of a ``ProcessGroup`` — NVSHMEM bootstrap is handled internally.

    ``gate_up_clamp`` sets the kernel gate-up clamp.  ``activation_clamp`` is a
    deprecated alias for ``gate_up_clamp``.

    ``apply_topk_in_fc1`` mirrors ``mega_runner``'s
    ``ref_compute_graph == "deepgemm"`` behaviour when ``True`` (default).

    ``fc1_alpha``, ``fc2_alpha``, and ``fc1_norm_const`` are per-local-expert
    fp32 epilogue scalars with shape ``(num_total_experts // world_size,)``.
    Pass a scalar to broadcast one value to all local experts, or pass a CUDA
    float32 tensor with that shape.  When omitted, each defaults to ``1.0``.

    Expert weights are not allocated here; supply kernel-ready
    ``(weight, scale)`` tuples to :func:`nvfp4_mega_moe` instead.
    """
    if hidden % 128 != 0 or intermediate % 128 != 0:
        raise ValueError(
            "MegaMoE requires hidden and intermediate to be multiples of 128."
        )
    if num_total_experts % world_size != 0:
        raise ValueError("num_total_experts must be divisible by world_size.")

    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )
    num_experts_per_rank = num_total_experts // world_size

    cfg = MegaMoENvfp4Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        gate_up_clamp=clamp,
        apply_topk_in_fc1=apply_topk_in_fc1,
    )
    frontend = MegaMoENvfp4Frontend(cfg)

    hidden_sf_cols = ceil_div(hidden, Nvfp4BlockSize)
    hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)

    sym_roots: list[torch.Tensor] = []
    x, x_root = _sym_zeros_byte_view((num_max_tokens, hidden), _DataDtype)
    sym_roots.append(x_root)
    x_sf, x_sf_root = _sym_zeros_byte_view(
        (num_max_tokens, hidden_sf_cols_padded),
        _ScaleDtype,
    )
    sym_roots.append(x_sf_root)
    topk_idx = sym_zeros((num_max_tokens, num_topk), torch.int64)
    sym_roots.append(topk_idx)
    topk_weights = sym_zeros((num_max_tokens, num_topk), torch.float32)
    sym_roots.append(topk_weights)
    combine_output = sym_zeros(
        (num_max_tokens, num_topk, hidden),
        torch.bfloat16,
    )
    sym_roots.append(combine_output)
    combine_reduced_output = torch.empty(
        (num_max_tokens, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    fc1_alpha = _resolve_per_expert_epilogue(
        "fc1_alpha",
        fc1_alpha,
        num_experts_per_rank,
    )
    fc2_alpha = _resolve_per_expert_epilogue(
        "fc2_alpha",
        fc2_alpha,
        num_experts_per_rank,
    )
    fc1_norm_const = _resolve_per_expert_epilogue(
        "fc1_norm_const",
        fc1_norm_const,
        num_experts_per_rank,
    )

    return MegaMoESymmBuffer(
        num_total_experts=num_total_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate=intermediate,
        rank=rank,
        world_size=world_size,
        x=x,
        x_sf=x_sf,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        combine_output=combine_output,
        combine_reduced_output=combine_reduced_output,
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
        _frontend=frontend,
        _sym_roots=sym_roots,
    )


def nvfp4_mega_moe(
    y: torch.Tensor,
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoESymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
) -> None:
    """Launch the fused CuTeDSL NVFP4 MegaMoE kernel (dispatch + fc1 + fc2 + combine).

    Caller must stage ``symm_buffer.x`` / routing slices before calling.

    ``transformed_l1`` / ``transformed_l2`` are ``(weight, scale)`` tuples in
    the **kernel-ready** NVFP4 + swizzled-SF layout (see ``mega_runner`` weight
    assembly, not ``deep_gemm.transform_weights_for_mega_moe``).  Weights are
    always caller-supplied here — unlike epilogue scalars in
    :func:`get_symm_buffer_for_mega_moe`, they are not owned by the symm buffer.

    ``y`` receives the top-k-reduced bf16 output for ``[:num_tokens]``.
    ``gate_up_clamp`` updates the kernel clamp for this session when set.
    ``activation_clamp`` is a deprecated alias for ``gate_up_clamp``.
    ``fast_math`` is accepted for DeepGEMM API parity and has no effect here.
    """
    if not fast_math:
        warnings.warn(
            "fast_math=False has no effect in the CuTeDSL NVFP4 MegaMoE path.",
            UserWarning,
            stacklevel=2,
        )

    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")

    n = num_tokens if num_tokens is not None else symm_buffer.num_max_tokens
    if n < 0 or n > symm_buffer.num_max_tokens:
        raise ValueError(
            f"num_tokens must be in [0, {symm_buffer.num_max_tokens}], got {n}."
        )
    if n == 0 and symm_buffer._frontend.config.fc2_reduces_topk:
        return
    if y.shape != (n, symm_buffer.hidden):
        raise ValueError(
            f"y must be ({n}, {symm_buffer.hidden}), got {tuple(y.shape)}."
        )
    if y.dtype != torch.bfloat16:
        raise ValueError(f"y must be bfloat16, got {y.dtype}.")

    fc1_weight, fc1_weight_sf = transformed_l1
    fc2_weight, fc2_weight_sf = transformed_l2

    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )
    if clamp is not None:
        symm_buffer._frontend.set_gate_up_clamp(clamp)

    inputs = MegaMoENvfp4Inputs(
        activation=symm_buffer.x,
        activation_sf=symm_buffer.x_sf,
        topk_idx=symm_buffer.topk_idx,
        topk_weights=symm_buffer.topk_weights,
        fc1_weight=fc1_weight,
        fc1_weight_sf=fc1_weight_sf,
        fc2_weight=fc2_weight,
        fc2_weight_sf=fc2_weight_sf,
        fc1_alpha=symm_buffer.fc1_alpha,
        fc2_alpha=symm_buffer.fc2_alpha,
        fc1_norm_const=symm_buffer.fc1_norm_const,
        combine_output=symm_buffer.combine_output,
        combine_reduced_output=symm_buffer.combine_reduced_output,
    )

    if symm_buffer._frontend.config.fc2_reduces_topk:
        out = symm_buffer._frontend.run(inputs, num_tokens=n)
    else:
        out = symm_buffer._frontend.run(
            inputs,
            num_tokens=None,
            reduce_topk=False,
        )
    if out is not None:
        if symm_buffer._frontend.config.fc2_reduces_topk:
            y.copy_(out[:n])
        else:
            active_form_a = out[:n]
            active_form_a_fp32 = active_form_a.to(torch.float32)
            if symm_buffer._frontend.config.apply_topk_in_fc1:
                reduced = active_form_a_fp32.sum(dim=1).to(y.dtype)
            else:
                reduced = (
                    (
                        active_form_a_fp32
                        * symm_buffer.topk_weights[:n, :, None].to(torch.float32)
                    )
                    .sum(dim=1)
                    .to(y.dtype)
                )
            y.copy_(reduced)


def make_dummy_epilogue_params(
    num_local_experts: int,
    *,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random per-local-expert epilogue scalars (matches ``mega_runner``)."""
    fc1_alpha = (
        torch.randint(
            1,
            5,
            (num_local_experts,),
            generator=generator,
            device="cuda",
        ).to(torch.float32)
        * 0.5
    )
    fc2_alpha = (
        torch.randint(
            1,
            5,
            (num_local_experts,),
            generator=generator,
            device="cuda",
        ).to(torch.float32)
        * 0.5
    )
    fc1_norm_const = (
        torch.randint(
            2,
            5,
            (num_local_experts,),
            generator=generator,
            device="cuda",
        ).to(torch.float32)
        * 0.5
    )
    return fc1_alpha, fc2_alpha, fc1_norm_const


def _create_dummy_weights(
    num_local_experts: int,
    hidden: int,
    intermediate: int,
    generator: torch.Generator,
) -> Tuple[TransformedWeights, TransformedWeights]:
    """Random NVFP4 weights + swizzled SF for local smoke scripts."""
    from moe_nvfp4_swapab.mega_runner import (
        _stack_byte_reinterpretable_tensors,
    )
    from moe_nvfp4_swapab.runner_common import (
        make_nvfp4_tensor_from_torch_rng,
        make_raw_scale_tensor_from_torch_rng,
        to_blocked,
    )

    intermediate_down = intermediate // 2
    hidden_sf_cols = ceil_div(hidden, Nvfp4BlockSize)
    intermediate_down_sf_cols = ceil_div(intermediate_down, Nvfp4BlockSize)

    fc1_weight = make_nvfp4_tensor_from_torch_rng(
        generator,
        (num_local_experts, hidden, intermediate),
        packed_dim=1,
        perf_run=True,
    )
    fc1_weight_sf_plain = make_raw_scale_tensor_from_torch_rng(
        generator,
        num_local_experts * intermediate,
        hidden,
        blocksize=Nvfp4BlockSize,
        strict=True,
    ).reshape(num_local_experts, intermediate, hidden_sf_cols)
    fc1_sf_swizzled = [
        to_blocked(fc1_weight_sf_plain[e]) for e in range(num_local_experts)
    ]
    fc1_flat_sf_size = fc1_sf_swizzled[0].numel()
    fc1_weight_sf = _stack_byte_reinterpretable_tensors(fc1_sf_swizzled, dim=0).view(
        num_local_experts, fc1_flat_sf_size
    )

    fc2_weight = make_nvfp4_tensor_from_torch_rng(
        generator,
        (num_local_experts, intermediate_down, hidden),
        packed_dim=1,
        perf_run=True,
    )
    fc2_weight_sf_plain = make_raw_scale_tensor_from_torch_rng(
        generator,
        num_local_experts * hidden,
        intermediate_down,
        blocksize=Nvfp4BlockSize,
        strict=True,
    ).reshape(num_local_experts, hidden, intermediate_down_sf_cols)
    fc2_sf_swizzled = [
        to_blocked(fc2_weight_sf_plain[e]) for e in range(num_local_experts)
    ]
    fc2_flat_sf_size = fc2_sf_swizzled[0].numel()
    fc2_weight_sf = _stack_byte_reinterpretable_tensors(fc2_sf_swizzled, dim=0).view(
        num_local_experts, fc2_flat_sf_size
    )

    return (fc1_weight, fc1_weight_sf), (fc2_weight, fc2_weight_sf)


def create_dummy_inputs(
    rank: int,
    world_size: int,
    num_total_experts: int,
    num_max_tokens: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    *,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fc1_alpha: Optional[PerExpertEpilogue] = None,
    fc2_alpha: Optional[PerExpertEpilogue] = None,
    fc1_norm_const: Optional[PerExpertEpilogue] = None,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    TransformedWeights,
    TransformedWeights,
    MegaMoESymmBuffer,
]:
    """Allocate symm buffer, NVFP4 weights, and stage activations + routing.

    Mirrors ``dummy_fp8_fp4_mega_moe.create_dummy_inputs`` for the NVFP4 path.
    When ``fc1_alpha`` / ``fc2_alpha`` / ``fc1_norm_const`` are omitted, random
    per-local-expert values are generated from ``seed`` (see
    :func:`make_dummy_epilogue_params`).
    """
    if num_tokens < 0 or num_tokens > num_max_tokens:
        raise ValueError(
            f"num_tokens must be in [0, {num_max_tokens}], got {num_tokens}."
        )

    num_local_experts = num_total_experts // world_size
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed + rank)
    if fc1_alpha is None and fc2_alpha is None and fc1_norm_const is None:
        fc1_alpha, fc2_alpha, fc1_norm_const = make_dummy_epilogue_params(
            num_local_experts,
            generator=gen,
        )

    symm_buffer = get_symm_buffer_for_mega_moe(
        num_total_experts,
        num_max_tokens,
        num_topk,
        hidden,
        intermediate,
        rank,
        world_size,
        gate_up_clamp=clamp,
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
    )

    transformed_l1, transformed_l2 = _create_dummy_weights(
        num_local_experts,
        hidden,
        intermediate,
        gen,
    )

    from moe_nvfp4_swapab.runner_common import (
        make_nvfp4_tensor_from_torch_rng,
        make_raw_scale_tensor_from_torch_rng,
    )

    activation = make_nvfp4_tensor_from_torch_rng(
        gen,
        (num_tokens, hidden),
        packed_dim=-1,
        perf_run=True,
    )
    activation_sf = make_raw_scale_tensor_from_torch_rng(
        gen,
        num_tokens,
        hidden,
        blocksize=Nvfp4BlockSize,
        strict=True,
    ).reshape(num_tokens, ceil_div(hidden, Nvfp4BlockSize))

    scores = torch.randn(
        num_tokens,
        num_total_experts,
        device="cuda",
        dtype=torch.float32,
    )
    topk_weights, topk_idx = torch.topk(
        scores,
        num_topk,
        dim=-1,
        largest=True,
        sorted=False,
    )

    symm_buffer.x[:num_tokens].copy_(activation)
    hidden_sf_cols = ceil_div(hidden, Nvfp4BlockSize)
    symm_buffer.x_sf[:num_tokens, :hidden_sf_cols].copy_(activation_sf)
    symm_buffer.topk_idx[:num_tokens].copy_(topk_idx.to(torch.int64))
    symm_buffer.topk_weights[:num_tokens].copy_(topk_weights.to(torch.float32))

    y = torch.empty(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    return y, transformed_l1, transformed_l2, symm_buffer


def _main() -> None:
    """Minimal torchrun smoke for the NVFP4 MegaMoE thin API."""
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1 or not bool(int(os.environ.get("MEGA_NO_DIST", "0"))):
        torch.cuda.set_device(local_rank)

    HIDDEN = 2048
    INTERMEDIATE = 1024
    NUM_TOKENS = 128
    NUM_MAX_TOKENS = 128
    NUM_TOPK = 4
    NUM_EXPERTS = 32
    GATE_UP_CLAMP = 10.0

    rank, world_size = init_dist()
    symm_buffer = None
    num_local_experts = NUM_EXPERTS // world_size

    try:
        epilogue_gen = torch.Generator(device="cuda")
        epilogue_gen.manual_seed(0 + rank)
        fc1_alpha, fc2_alpha, fc1_norm_const = make_dummy_epilogue_params(
            num_local_experts,
            generator=epilogue_gen,
        )

        y, transformed_l1, transformed_l2, symm_buffer = create_dummy_inputs(
            rank,
            world_size,
            NUM_EXPERTS,
            NUM_MAX_TOKENS,
            NUM_TOKENS,
            NUM_TOPK,
            HIDDEN,
            INTERMEDIATE,
            gate_up_clamp=GATE_UP_CLAMP,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
            seed=0,
        )

        nvfp4_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=NUM_TOKENS,
            gate_up_clamp=GATE_UP_CLAMP,
        )
        torch.cuda.synchronize()

        if rank == 0:
            print("ok")
            print("y:", y.shape, y.dtype)
    finally:
        if symm_buffer is not None:
            symm_buffer.destroy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        no_dist = bool(int(os.environ.get("MEGA_NO_DIST", "0")))
        if not no_dist and dist.is_initialized():
            from src.bootstrap import finalize_dist_and_nvshmem

            finalize_dist_and_nvshmem()
