# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Thin CuTeDSL MXFP8 MegaMoE API (vLLM-oriented).

MXFP8 activations + MXFP8 expert weights, with E8M0 block scale-factor planes.
Output is bf16 after top-k combine.  Call layout mirrors the NVFP4 frontend
(:func:`get_symm_buffer_for_mxfp8_mega_moe` + weight tuples + one launch).

  * Activations: fp8 e4m3 / e5m2 data + E8M0 block SF (``sf_vec_size = 32``).
  * Weights: fp8 + atom-swizzled SF (kernel-ready layout from ``mega_runner``).
  * fc1 layout: ``(E, hidden, 2 * intermediate)`` with hidden K-major
    (stride-1), where ``intermediate`` is the post-SwiGLU width.

Workspace vs per-launch inputs (mirrors the NVFP4 frontend):

  * :func:`get_symm_buffer_for_mxfp8_mega_moe` owns activations, routing, and
    combine staging.
  * Expert weights are **not** in the symm buffer — pass kernel-ready
    ``(weight, scale)`` tuples to :func:`mxfp8_mega_moe` on every launch.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import torch

from .megamoe_frontend.common import bootstrap_dist, free_sym_tensor, sym_zeros
from .megamoe_frontend.api_mxfp8 import (
    MegaMoEMxfp8Config,
    MegaMoEMxfp8Frontend,
    MegaMoEMxfp8Inputs,
    _KIND_TO_TORCH_DTYPE,
)
from ._util import resolve_gate_up_clamp
from common.megamoe_constants import Mxfp8BlockSize
from moe_nvfp4_swapab.runner_common import ceil_div, round_up

TransformedWeights = Tuple[torch.Tensor, torch.Tensor]

Mxfp8Kind = Literal["mxfp8_e4m3", "mxfp8_e5m2"]


def _sym_zeros_byte_view_1b(
    logical_shape: Tuple[int, ...],
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp8 / E8M0 symmetric heap via uint8 reinterpret (matches mega_runner).

    Returns ``(view, root_uint8_buffer)``; free the root via :func:`free_sym_tensor`.
    """
    total_bytes = 1
    for dim_size in logical_shape:
        total_bytes *= dim_size
    root = sym_zeros((total_bytes,), torch.uint8)
    view = root.view(target_dtype).reshape(logical_shape)
    return view, root


def init_dist() -> Tuple[int, int]:
    """Initialize torch.distributed + NVSHMEM (or single-rank when ``MEGA_NO_DIST=1``).

    Returns ``(rank, world_size)``.
    """
    _, rank, world_size, _ = bootstrap_dist()
    return rank, world_size


@dataclass
class MegaMoEMxfp8SymmBuffer:
    """Symmetric-heap staging buffers for one MXFP8 MegaMoE session.

    Mirrors the NVFP4 :class:`MegaMoESymmBuffer`: exposes ``x``, ``x_sf``,
    ``topk_idx``, and ``topk_weights`` views sized for ``num_max_tokens``.

    Expert weights are **not** stored here — pass ``transformed_l1`` /
    ``transformed_l2`` to :func:`mxfp8_mega_moe` each launch.
    """

    num_total_experts: int
    num_max_tokens: int
    num_topk: int
    hidden: int
    intermediate: int
    rank: int
    world_size: int
    kind: Mxfp8Kind

    x: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    combine_output: torch.Tensor

    _frontend: MegaMoEMxfp8Frontend
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


def get_symm_buffer_for_mxfp8_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    kind: Mxfp8Kind = "mxfp8_e4m3",
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    in_kernel_fc2_reduce: bool = False,
    token_back_by_dispatch: bool = False,
) -> MegaMoEMxfp8SymmBuffer:
    """Allocate symmetric-heap inputs + combine staging for one MXFP8 session.

    Argument order follows the NVFP4 frontend (problem sizes first).  Pass
    ``rank`` / ``world_size`` from :func:`init_dist`.

    ``kind`` selects the fp8 element format (``mxfp8_e4m3`` or ``mxfp8_e5m2``).
    ``gate_up_clamp`` sets the kernel gate-up clamp.  ``activation_clamp`` is a
    deprecated alias for ``gate_up_clamp``.
    ``intermediate`` is the post-SwiGLU width, matching NVFP4 and SGLang.

    Expert weights are not allocated here; supply kernel-ready ``(weight, scale)``
    tuples to :func:`mxfp8_mega_moe` instead.
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

    cfg = MegaMoEMxfp8Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        kind=kind,
        gate_up_clamp=clamp,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_by_dispatch=token_back_by_dispatch,
    )
    frontend = MegaMoEMxfp8Frontend(cfg)

    hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
    hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)
    data_dtype = cfg.torch_ab_dtype

    from moe_nvfp4_swapab.runner_common import Mxfp8ScaleDtype

    sym_roots: list[torch.Tensor] = []
    x, x_root = _sym_zeros_byte_view_1b((num_max_tokens, hidden), data_dtype)
    sym_roots.append(x_root)
    x_sf, x_sf_root = _sym_zeros_byte_view_1b(
        (num_max_tokens, hidden_sf_cols_padded),
        Mxfp8ScaleDtype,
    )
    sym_roots.append(x_sf_root)
    topk_idx = sym_zeros((num_max_tokens, num_topk), torch.int64)
    sym_roots.append(topk_idx)
    topk_weights = sym_zeros((num_max_tokens, num_topk), torch.float32)
    sym_roots.append(topk_weights)
    combine_k = 1 if in_kernel_fc2_reduce else num_topk
    combine_output = sym_zeros(
        (num_max_tokens, combine_k, hidden),
        torch.bfloat16,
    )
    sym_roots.append(combine_output)

    return MegaMoEMxfp8SymmBuffer(
        num_total_experts=num_total_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate=intermediate,
        rank=rank,
        world_size=world_size,
        kind=kind,
        x=x,
        x_sf=x_sf,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        combine_output=combine_output,
        _frontend=frontend,
        _sym_roots=sym_roots,
    )


def mxfp8_mega_moe(
    y: torch.Tensor,
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoEMxfp8SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
) -> None:
    """Launch the fused CuTeDSL MXFP8 MegaMoE kernel (dispatch + fc1 + fc2 + combine).

    Caller must stage ``symm_buffer.x`` / routing slices before calling.

    ``transformed_l1`` / ``transformed_l2`` are ``(weight, scale)`` tuples in
    the **kernel-ready** fp8 + swizzled-SF layout (see ``mega_runner`` weight
    assembly).  Weights are always caller-supplied here — they are not owned by
    the symm buffer.

    ``y`` receives the top-k-reduced bf16 output for ``[:num_tokens]``.
    ``gate_up_clamp`` updates the kernel clamp for this session when set.
    ``activation_clamp`` is a deprecated alias for ``gate_up_clamp``.
    ``fast_math`` is accepted for DeepGEMM API parity and has no effect here.
    """
    if not fast_math:
        warnings.warn(
            "fast_math=False has no effect in the CuTeDSL MXFP8 MegaMoE path.",
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
    if n == 0 and symm_buffer._frontend.config.in_kernel_fc2_reduce:
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

    inputs = MegaMoEMxfp8Inputs(
        activation=symm_buffer.x,
        activation_sf=symm_buffer.x_sf,
        topk_idx=symm_buffer.topk_idx,
        topk_weights=symm_buffer.topk_weights,
        fc1_weight=fc1_weight,
        fc1_weight_sf=fc1_weight_sf,
        fc2_weight=fc2_weight,
        fc2_weight_sf=fc2_weight_sf,
        combine_output=symm_buffer.combine_output,
    )

    if symm_buffer._frontend.config.in_kernel_fc2_reduce:
        out = symm_buffer._frontend.run(inputs, num_tokens=n)
        if out is not None:
            y.copy_(out[:n])
    else:
        out = symm_buffer._frontend.run(
            inputs,
            num_tokens=None,
            reduce_topk=False,
        )
        if out is not None:
            reduced = (
                (
                    out[:n].to(torch.float32)
                    * symm_buffer.topk_weights[:n, :, None].to(torch.float32)
                )
                .sum(dim=1)
                .to(y.dtype)
            )
            y.copy_(reduced)


def _create_dummy_weights(
    num_local_experts: int,
    hidden: int,
    intermediate: int,
    generator: torch.Generator,
    *,
    kind: Mxfp8Kind,
) -> Tuple[TransformedWeights, TransformedWeights]:
    """Random MXFP8 weights + swizzled SF for local smoke scripts."""
    from moe_mxfp8_glu.mega_runner import (
        _make_e8m0_scale_tensor,
        _make_fp8_tensor,
    )
    from moe_nvfp4_swapab.mega_runner import _stack_byte_reinterpretable_tensors
    from moe_nvfp4_swapab.runner_common import to_blocked

    data_dtype = _KIND_TO_TORCH_DTYPE[kind]

    fc1_out = 2 * intermediate
    hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
    intermediate_sf_cols = ceil_div(intermediate, Mxfp8BlockSize)

    fc1_weight = _make_fp8_tensor(
        generator,
        (num_local_experts, hidden, fc1_out),
        data_dtype,
        perf_run=True,
    )
    fc1_weight_sf_plain = _make_e8m0_scale_tensor(
        generator,
        num_local_experts * fc1_out,
        hidden,
        blocksize=Mxfp8BlockSize,
    ).reshape(num_local_experts, fc1_out, hidden_sf_cols)
    fc1_sf_swizzled = [
        to_blocked(fc1_weight_sf_plain[e]) for e in range(num_local_experts)
    ]
    fc1_flat_sf_size = fc1_sf_swizzled[0].numel()
    fc1_weight_sf = _stack_byte_reinterpretable_tensors(fc1_sf_swizzled, dim=0).view(
        num_local_experts, fc1_flat_sf_size
    )

    fc2_weight = _make_fp8_tensor(
        generator,
        (num_local_experts, intermediate, hidden),
        data_dtype,
        perf_run=True,
    )
    fc2_weight_sf_plain = _make_e8m0_scale_tensor(
        generator,
        num_local_experts * hidden,
        intermediate,
        blocksize=Mxfp8BlockSize,
    ).reshape(num_local_experts, hidden, intermediate_sf_cols)
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
    kind: Mxfp8Kind = "mxfp8_e4m3",
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    TransformedWeights,
    TransformedWeights,
    MegaMoEMxfp8SymmBuffer,
]:
    """Allocate symm buffer, MXFP8 weights, and stage activations + routing."""
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

    symm_buffer = get_symm_buffer_for_mxfp8_mega_moe(
        num_total_experts,
        num_max_tokens,
        num_topk,
        hidden,
        intermediate,
        rank,
        world_size,
        kind=kind,
        gate_up_clamp=clamp,
    )

    transformed_l1, transformed_l2 = _create_dummy_weights(
        num_local_experts,
        hidden,
        intermediate,
        gen,
        kind=kind,
    )

    from moe_mxfp8_glu.mega_runner import _make_e8m0_scale_tensor, _make_fp8_tensor

    data_dtype = symm_buffer._frontend.config.torch_ab_dtype
    activation = _make_fp8_tensor(
        gen,
        (num_tokens, hidden),
        data_dtype,
        perf_run=True,
    )
    activation_sf = _make_e8m0_scale_tensor(
        gen,
        num_tokens,
        hidden,
        blocksize=Mxfp8BlockSize,
    ).reshape(num_tokens, ceil_div(hidden, Mxfp8BlockSize))

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

    symm_buffer.x[:num_tokens].view(torch.uint8).copy_(
        activation.view(torch.uint8),
    )
    hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
    symm_buffer.x_sf[:num_tokens, :hidden_sf_cols].view(torch.uint8).copy_(
        activation_sf.view(torch.uint8),
    )
    symm_buffer.topk_idx[:num_tokens].copy_(topk_idx.to(torch.int64))
    symm_buffer.topk_weights[:num_tokens].copy_(topk_weights.to(torch.float32))

    y = torch.empty(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    return y, transformed_l1, transformed_l2, symm_buffer


def _main() -> None:
    """Minimal torchrun smoke for the MXFP8 MegaMoE thin API."""
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

    try:
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
            seed=0,
        )

        mxfp8_mega_moe(
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


if __name__ == "__main__":
    _main()
