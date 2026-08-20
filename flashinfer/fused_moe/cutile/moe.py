# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BF16 cuTile kernels for the unified MoE backend.

The implementation uses the align-block representation also used by LightLM:
routing assignments are stably grouped by expert and padded to a fixed M tile.
Both expert GEMMs consume that representation and leave their results in the
original assignment order, allowing a deterministic weighted combine.

This module deliberately contains no FlashInfer dispatch or configuration
logic. The unified runner owns validation, workspace lifetime, and tactic
selection; functions here only launch kernels into caller-owned buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TypeAlias

import cuda.tile as ct
import torch

from ...cutile.cutile_common import cached_replace_hints
from ...tllm_enums import ActivationType
from ...utils import next_positive_power_of_2
from .activation import launch_activation

ConstInt: TypeAlias = ct.Constant[int]

_PERMUTE_TILE_CAP = 16384


@dataclass(frozen=True)
class GemmConfig:
    """Compile-time grouped-GEMM tile and occupancy."""

    tile_n: int
    tile_k: int
    occupancy: int


@dataclass
class Workspace:
    """Caller-owned buffers needed by one MoE invocation shape."""

    sorted_slots: torch.Tensor
    block_expert: torch.Tensor
    num_post_pad: torch.Tensor
    ranks: torch.Tensor
    hist: torch.Tensor
    base: torch.Tensor
    pad_off: torch.Tensor
    slab_tot: torch.Tensor
    gemm1_out: torch.Tensor
    activation_out: torch.Tensor
    gemm2_out: torch.Tensor


def _permute_shape(num_assignments: int, num_experts: int) -> tuple[int, int, int]:
    epow2 = next_positive_power_of_2(num_experts)
    chunk = min(
        max(8, _PERMUTE_TILE_CAP // epow2),
        max(8, next_positive_power_of_2(num_assignments)),
    )
    num_chunks = max(1, (num_assignments + chunk - 1) // chunk)
    ncp = next_positive_power_of_2(num_chunks)
    return epow2, chunk, ncp


def allocate_workspace(
    *,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    is_gated: bool,
    block_sizes: Sequence[int],
    device: torch.device,
) -> Workspace:
    """Allocate graph-stable buffers for one exact token shape and tactic set."""
    if not block_sizes or any(block_size <= 0 for block_size in block_sizes):
        raise ValueError("block_sizes must contain positive integers.")
    num_assignments = num_tokens * top_k
    max_em = max(
        num_assignments + num_experts * (block_size - 1) for block_size in block_sizes
    )
    max_blocks = max(
        (num_assignments + num_experts * (block_size - 1) + block_size - 1)
        // block_size
        for block_size in block_sizes
    )
    epow2, _, ncp = _permute_shape(num_assignments, num_experts)
    ncp_slab = max(1, min(ncp, _PERMUTE_TILE_CAP // epow2))
    num_slabs = ncp // ncp_slab
    int32 = torch.int32
    bf16 = torch.bfloat16
    return Workspace(
        sorted_slots=torch.empty(max_em, dtype=int32, device=device),
        block_expert=torch.empty(max_blocks, dtype=int32, device=device),
        num_post_pad=torch.empty(1, dtype=int32, device=device),
        ranks=torch.empty(num_assignments, dtype=int32, device=device),
        hist=torch.empty(ncp * epow2, dtype=int32, device=device),
        base=torch.empty(ncp * epow2, dtype=int32, device=device),
        pad_off=torch.empty(num_experts + 1, dtype=int32, device=device),
        slab_tot=torch.empty(num_slabs * epow2, dtype=int32, device=device),
        gemm1_out=torch.empty(
            num_assignments,
            intermediate_size * (2 if is_gated else 1),
            dtype=bf16,
            device=device,
        ),
        activation_out=torch.empty(
            num_assignments, intermediate_size, dtype=bf16, device=device
        ),
        gemm2_out=torch.empty(num_assignments, hidden_size, dtype=bf16, device=device),
    )


def _lane_ids(topk_ids, numel, chunk_size):
    chunk_id = ct.bid(0)
    offsets = ct.arange(chunk_size, dtype=ct.int32)
    assignment = chunk_id * chunk_size + offsets
    valid = assignment < numel
    ids = ct.gather(
        topk_ids,
        (ct.minimum(assignment, numel - 1),),
        padding_value=0,
    )
    ids = ct.where(valid, ids, ct.full((chunk_size,), -1, dtype=ct.int32))
    return chunk_id, assignment, valid, ids


@ct.kernel
def _permute_chunk_rank(
    TOPK_IDS,
    RANKS,
    HIST,
    numel,
    EPOW2: ConstInt,
    CHUNK_SIZE: ConstInt,
):
    chunk_id, assignment, valid, ids = _lane_ids(TOPK_IDS, numel, CHUNK_SIZE)
    expert_offsets = ct.arange(EPOW2, dtype=ct.int32)
    one_hot = ct.astype(
        ct.reshape(ids, (CHUNK_SIZE, 1)) == ct.reshape(expert_offsets, (1, EPOW2)),
        ct.int32,
    )
    inclusive = ct.cumsum(one_hot, axis=0)
    rank = ct.sum(inclusive * one_hot, axis=1) - 1
    ct.scatter(RANKS, (assignment,), rank, mask=valid)
    ct.scatter(
        HIST,
        (chunk_id * EPOW2 + expert_offsets,),
        ct.sum(one_hot, axis=0),
    )


@ct.kernel
def _permute_scan_partials(
    HIST,
    SLAB_TOTALS,
    EPOW2: ConstInt,
    CHUNKS_PER_SLAB: ConstInt,
):
    slab = ct.bid(0)
    expert_offsets = ct.arange(EPOW2, dtype=ct.int32)
    slab_offsets = ct.arange(CHUNKS_PER_SLAB, dtype=ct.int32)
    rows = slab * CHUNKS_PER_SLAB + slab_offsets
    indices = (
        ct.reshape(rows, (CHUNKS_PER_SLAB, 1)) * EPOW2
        + ct.reshape(expert_offsets, (1, EPOW2)),
    )
    total = ct.sum(ct.gather(HIST, indices, padding_value=0), axis=0)
    ct.scatter(SLAB_TOTALS, (slab * EPOW2 + expert_offsets,), total)


@ct.kernel
def _permute_scan_combine(
    SLAB_TOTALS,
    PAD_OFFSETS,
    NUM_POST_PAD,
    E: ConstInt,
    EPOW2: ConstInt,
    NUM_SLABS: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    expert_offsets = ct.arange(EPOW2, dtype=ct.int32)
    counts = ct.zeros((EPOW2,), dtype=ct.int32)
    for slab in range(NUM_SLABS):
        counts = counts + ct.gather(
            SLAB_TOTALS,
            (slab * EPOW2 + expert_offsets,),
            padding_value=0,
        )
    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    exclusive = ct.cumsum(padded, axis=0) - padded
    ct.scatter(PAD_OFFSETS, (expert_offsets,), exclusive, mask=expert_offsets < E)
    total = ct.reshape(ct.sum(padded), (1,))
    ct.scatter(PAD_OFFSETS, (ct.full((1,), E, dtype=ct.int32),), total)
    ct.scatter(NUM_POST_PAD, (ct.zeros((1,), dtype=ct.int32),), total)
    carry = exclusive
    for slab in range(NUM_SLABS):
        indices = (slab * EPOW2 + expert_offsets,)
        slab_total = ct.gather(SLAB_TOTALS, indices, padding_value=0)
        ct.scatter(SLAB_TOTALS, indices, carry)
        carry = carry + slab_total


@ct.kernel
def _permute_scan_bases(
    HIST,
    SLAB_TOTALS,
    BASE,
    EPOW2: ConstInt,
    CHUNKS_PER_SLAB: ConstInt,
):
    slab = ct.bid(0)
    expert_offsets = ct.arange(EPOW2, dtype=ct.int32)
    slab_offsets = ct.arange(CHUNKS_PER_SLAB, dtype=ct.int32)
    rows = slab * CHUNKS_PER_SLAB + slab_offsets
    indices = (
        ct.reshape(rows, (CHUNKS_PER_SLAB, 1)) * EPOW2
        + ct.reshape(expert_offsets, (1, EPOW2)),
    )
    hist = ct.gather(HIST, indices, padding_value=0)
    exclusive = ct.cumsum(hist, axis=0) - hist
    carry = ct.reshape(
        ct.gather(
            SLAB_TOTALS,
            (slab * EPOW2 + expert_offsets,),
            padding_value=0,
        ),
        (1, EPOW2),
    )
    ct.scatter(BASE, indices, exclusive + carry)


@ct.kernel
def _permute_scan(
    HIST,
    PAD_OFFSETS,
    BASE,
    NUM_POST_PAD,
    BLOCK_EXPERT,
    num_blocks,
    E: ConstInt,
    EPOW2: ConstInt,
    CHUNKS_PER_SLAB: ConstInt,
    NUM_SLABS: ConstInt,
    BLOCK_SIZE: ConstInt,
    NBPOW2: ConstInt,
):
    expert_offsets = ct.arange(EPOW2, dtype=ct.int32)
    slab_offsets = ct.arange(CHUNKS_PER_SLAB, dtype=ct.int32)
    counts = ct.zeros((EPOW2,), dtype=ct.int32)
    for slab in range(NUM_SLABS):
        rows = slab * CHUNKS_PER_SLAB + slab_offsets
        indices = (
            ct.reshape(rows, (CHUNKS_PER_SLAB, 1)) * EPOW2
            + ct.reshape(expert_offsets, (1, EPOW2)),
        )
        counts = counts + ct.sum(ct.gather(HIST, indices, padding_value=0), axis=0)
    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    exclusive = ct.cumsum(padded, axis=0) - padded
    ct.scatter(PAD_OFFSETS, (expert_offsets,), exclusive, mask=expert_offsets < E)
    total = ct.reshape(ct.sum(padded), (1,))
    ct.scatter(PAD_OFFSETS, (ct.full((1,), E, dtype=ct.int32),), total)
    ct.scatter(NUM_POST_PAD, (ct.zeros((1,), dtype=ct.int32),), total)
    block_offsets = ct.arange(NBPOW2, dtype=ct.int32)
    ct.scatter(
        BLOCK_EXPERT,
        (block_offsets,),
        ct.zeros((NBPOW2,), dtype=ct.int32),
        mask=block_offsets < num_blocks,
    )
    carry = ct.reshape(exclusive, (1, EPOW2))
    for slab in range(NUM_SLABS):
        rows = slab * CHUNKS_PER_SLAB + slab_offsets
        indices = (
            ct.reshape(rows, (CHUNKS_PER_SLAB, 1)) * EPOW2
            + ct.reshape(expert_offsets, (1, EPOW2)),
        )
        hist = ct.gather(HIST, indices, padding_value=0)
        exclusive_chunks = ct.cumsum(hist, axis=0) - hist
        ct.scatter(BASE, indices, exclusive_chunks + carry)
        carry = carry + ct.reshape(ct.sum(hist, axis=0), (1, EPOW2))


@ct.kernel
def _permute_scatter(
    TOPK_IDS,
    RANKS,
    BASE,
    SORTED_SLOTS,
    numel,
    EPOW2: ConstInt,
    CHUNK_SIZE: ConstInt,
):
    chunk_id, assignment, valid, ids = _lane_ids(TOPK_IDS, numel, CHUNK_SIZE)
    rank = ct.gather(RANKS, (ct.minimum(assignment, numel - 1),), padding_value=0)
    base = ct.gather(
        BASE,
        (chunk_id * EPOW2 + ct.maximum(ids, 0),),
        padding_value=0,
    )
    ct.scatter(SORTED_SLOTS, (base + rank,), assignment, mask=valid)


@ct.kernel
def _permute_block_expert(
    PAD_OFFSETS,
    BLOCK_EXPERT,
    BLOCK_SIZE: ConstInt,
):
    expert = ct.bid(0)
    start = ct.gather(
        PAD_OFFSETS,
        ct.full((1,), expert, dtype=ct.int32),
        padding_value=0,
    )
    end = ct.gather(
        PAD_OFFSETS,
        ct.full((1,), expert + 1, dtype=ct.int32),
        padding_value=0,
    )
    first_block = (start // BLOCK_SIZE).item()
    num_blocks = ((end - start) // BLOCK_SIZE).item()
    for index in range(num_blocks):
        ct.scatter(
            BLOCK_EXPERT,
            ct.full((1,), first_block + index, dtype=ct.int32),
            ct.full((1,), expert, dtype=ct.int32),
        )


@ct.kernel
def _permute_small(
    TOPK_IDS,
    SORTED_SLOTS,
    BLOCK_EXPERT,
    NUM_POST_PAD,
    numel,
    em,
    num_blocks,
    E: ConstInt,
    EPOW2: ConstInt,
    CHUNK_SIZE: ConstInt,
    NBPOW2: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    offsets = ct.arange(CHUNK_SIZE, dtype=ct.int32)
    valid = offsets < numel
    ids = ct.gather(TOPK_IDS, (ct.minimum(offsets, numel - 1),), padding_value=0)
    ids = ct.where(valid, ids, ct.full((CHUNK_SIZE,), -1, dtype=ct.int32))
    same = ct.astype(
        ct.reshape(ids, (CHUNK_SIZE, 1)) == ct.reshape(ids, (1, CHUNK_SIZE)),
        ct.int32,
    ) * ct.astype(
        ct.reshape(offsets, (1, CHUNK_SIZE)) < ct.reshape(offsets, (CHUNK_SIZE, 1)),
        ct.int32,
    )
    rank = ct.sum(same, axis=1)
    expert_slab = min(EPOW2, max(128, _PERMUTE_TILE_CAP // CHUNK_SIZE))
    total = ct.zeros((), dtype=ct.int32)
    position_base = ct.zeros((CHUNK_SIZE,), dtype=ct.int32)
    for slab in range(EPOW2 // expert_slab):
        experts = slab * expert_slab + ct.arange(expert_slab, dtype=ct.int32)
        one_hot = ct.astype(
            ct.reshape(ids, (CHUNK_SIZE, 1)) == ct.reshape(experts, (1, expert_slab)),
            ct.int32,
        )
        counts = ct.sum(one_hot, axis=0)
        padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        exclusive = ct.cumsum(padded, axis=0) - padded + total
        position_base = position_base + ct.sum(
            one_hot * ct.reshape(exclusive, (1, expert_slab)), axis=1
        )
        total = total + ct.sum(padded)
    ct.scatter(
        NUM_POST_PAD,
        (ct.zeros((1,), dtype=ct.int32),),
        ct.reshape(total, (1,)),
    )
    for fill_round in range(ct.cdiv(em, 2048)):
        slot_offsets = fill_round * 2048 + ct.arange(2048, dtype=ct.int32)
        ct.scatter(
            SORTED_SLOTS,
            (slot_offsets,),
            ct.zeros((2048,), dtype=ct.int32) + numel,
            mask=slot_offsets < em,
        )
    ct.scatter(SORTED_SLOTS, (position_base + rank,), offsets, mask=valid)
    block_offsets = ct.arange(NBPOW2, dtype=ct.int32)
    ct.scatter(
        BLOCK_EXPERT,
        (block_offsets,),
        ct.zeros((NBPOW2,), dtype=ct.int32),
        mask=block_offsets < num_blocks,
    )
    ct.scatter(
        BLOCK_EXPERT,
        ((position_base + rank) // BLOCK_SIZE,),
        ct.maximum(ids, 0),
        mask=valid,
    )


@ct.kernel
def _grouped_gemm_bf16(
    X,
    W,
    SORTED_SLOTS,
    BLOCK_EXPERT,
    NUM_POST_PAD,
    OUT,
    top_k,
    grid_m,
    K_IN: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
):
    initial_m_block = ct.bid(0)
    n_block = ct.bid(1)
    num_post_pad = ct.gather(
        NUM_POST_PAD, ct.zeros((1,), dtype=ct.int32), padding_value=0
    ).item()
    num_live_blocks = (num_post_pad + TILE_M - 1) // TILE_M
    num_iterations = (num_live_blocks - initial_m_block + grid_m - 1) // grid_m
    k_offsets = ct.arange(TILE_K, dtype=ct.int32)
    n_offsets = n_block * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
    num_k_tiles = (K_IN + TILE_K - 1) // TILE_K
    for iteration in range(num_iterations):
        m_block = initial_m_block + iteration * grid_m
        if m_block * TILE_M < num_post_pad:
            m_offsets = m_block * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
            slots = ct.gather(SORTED_SLOTS, (m_offsets,), padding_value=0)
            expert = ct.gather(
                BLOCK_EXPERT,
                ct.full((1,), m_block, dtype=ct.int32),
                padding_value=0,
            ).item()
            rows = slots // top_k
            accumulator = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
            for k_tile in range(num_k_tiles):
                a_indices = ct.reshape(rows, (TILE_M, 1)) * K_IN + ct.reshape(
                    k_tile * TILE_K + k_offsets, (1, TILE_K)
                )
                a = ct.gather(X, (a_indices,), padding_value=0, latency=3)
                b = ct.reshape(
                    ct.load(
                        W,
                        index=(expert, k_tile, n_block),
                        shape=(1, TILE_K, TILE_N),
                        order=(0, 1, 2),
                        allow_tma=True,
                        latency=3,
                        padding_mode=ct.PaddingMode.ZERO,
                    ),
                    (TILE_K, TILE_N),
                )
                accumulator = ct.mma(a, b, accumulator)
            ct.scatter(
                OUT,
                (
                    ct.reshape(slots, (TILE_M, 1)),
                    ct.reshape(n_offsets, (1, TILE_N)),
                ),
                ct.astype(accumulator, OUT.dtype),
                check_bounds=True,
            )


@ct.kernel
def _combine(
    Y,
    ROUTING_WEIGHTS,
    OUT,
    top_k,
    H: ConstInt,
    TILE_H: ConstInt,
):
    token = ct.bid(0)
    h_tile = ct.bid(1)
    h_offsets = h_tile * TILE_H + ct.arange(TILE_H, dtype=ct.int32)
    valid = h_offsets < H
    accumulator = ct.zeros((TILE_H,), dtype=ct.float32)
    for expert_slot in range(top_k):
        weight = ct.gather(
            ROUTING_WEIGHTS,
            (ct.full((1,), token * top_k + expert_slot, dtype=ct.int32),),
            padding_value=0.0,
        ).item()
        values = ct.gather(
            Y,
            ((token * top_k + expert_slot) * H + ct.minimum(h_offsets, H - 1),),
            padding_value=0,
        )
        accumulator = accumulator + ct.astype(values, ct.float32) * weight
    ct.scatter(
        OUT,
        (token * H + h_offsets,),
        ct.astype(accumulator, OUT.dtype),
        mask=valid,
    )


def _permute(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    workspace: Workspace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_assignments = topk_ids.numel()
    em = num_assignments + num_experts * (block_size - 1)
    num_blocks = (em + block_size - 1) // block_size
    sorted_slots = workspace.sorted_slots[:em]
    block_expert = workspace.block_expert[:num_blocks]
    if num_assignments <= 16:
        epow2 = next_positive_power_of_2(num_experts)
        ct.launch(
            torch.cuda.current_stream(topk_ids.device),
            (1,),
            _permute_small,
            (
                topk_ids.reshape(-1),
                sorted_slots,
                block_expert,
                workspace.num_post_pad,
                num_assignments,
                em,
                num_blocks,
                num_experts,
                epow2,
                max(8, next_positive_power_of_2(num_assignments)),
                next_positive_power_of_2(num_blocks),
                block_size,
            ),
        )
        return sorted_slots, block_expert, workspace.num_post_pad

    epow2, chunk_size, ncp = _permute_shape(num_assignments, num_experts)
    num_chunks = max(1, (num_assignments + chunk_size - 1) // chunk_size)
    chunks_per_slab = max(1, min(ncp, _PERMUTE_TILE_CAP // epow2))
    num_slabs = ncp // chunks_per_slab
    hist_size = (num_chunks if num_slabs <= 2 else ncp) * epow2
    hist = workspace.hist[:hist_size]
    base = workspace.base[: ncp * epow2]
    pad_off = workspace.pad_off[: num_experts + 1]
    stream = torch.cuda.current_stream(topk_ids.device)
    sorted_slots.fill_(num_assignments)
    if num_slabs > 2:
        hist.zero_()
        block_expert.zero_()
    ct.launch(
        stream,
        (num_chunks,),
        _permute_chunk_rank,
        (
            topk_ids.reshape(-1),
            workspace.ranks[:num_assignments],
            hist,
            num_assignments,
            epow2,
            chunk_size,
        ),
    )
    if num_slabs <= 2:
        ct.launch(
            stream,
            (1,),
            _permute_scan,
            (
                hist,
                pad_off,
                base,
                workspace.num_post_pad,
                block_expert,
                num_blocks,
                num_experts,
                epow2,
                chunks_per_slab,
                num_slabs,
                block_size,
                next_positive_power_of_2(num_blocks),
            ),
        )
    else:
        slab_totals = workspace.slab_tot[: num_slabs * epow2]
        ct.launch(
            stream,
            (num_slabs,),
            _permute_scan_partials,
            (hist, slab_totals, epow2, chunks_per_slab),
        )
        ct.launch(
            stream,
            (1,),
            _permute_scan_combine,
            (
                slab_totals,
                pad_off,
                workspace.num_post_pad,
                num_experts,
                epow2,
                num_slabs,
                block_size,
            ),
        )
        ct.launch(
            stream,
            (num_slabs,),
            _permute_scan_bases,
            (hist, slab_totals, base, epow2, chunks_per_slab),
        )
    ct.launch(
        stream,
        (num_chunks,),
        _permute_scatter,
        (
            topk_ids.reshape(-1),
            workspace.ranks[:num_assignments],
            base,
            sorted_slots,
            num_assignments,
            epow2,
            chunk_size,
        ),
    )
    ct.launch(
        stream,
        (num_experts,),
        _permute_block_expert,
        (pad_off, block_expert, block_size),
    )
    return sorted_slots, block_expert, workspace.num_post_pad


def _grouped_gemm(
    x: torch.Tensor,
    weights: torch.Tensor,
    sorted_slots: torch.Tensor,
    block_expert: torch.Tensor,
    num_post_pad: torch.Tensor,
    output: torch.Tensor,
    *,
    top_k: int,
    block_size: int,
    config: GemmConfig,
) -> None:
    num_assignment_rows = output.shape[0]
    n = output.shape[1]
    m_blocks = (sorted_slots.shape[0] + block_size - 1) // block_size
    grid_m = max(1, min(m_blocks, num_assignment_rows))
    kernel = cached_replace_hints(_grouped_gemm_bf16, occupancy=config.occupancy)
    ct.launch(
        torch.cuda.current_stream(x.device),
        (grid_m, (n + config.tile_n - 1) // config.tile_n),
        kernel,
        (
            x.reshape(-1),
            weights,
            sorted_slots,
            block_expert,
            num_post_pad,
            output,
            top_k,
            grid_m,
            x.shape[1],
            block_size,
            config.tile_n,
            config.tile_k,
        ),
    )


def run_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    output: torch.Tensor,
    workspace: Workspace,
    *,
    activation_type: ActivationType,
    block_size: int,
    gemm1_config: GemmConfig,
    gemm2_config: GemmConfig,
) -> torch.Tensor:
    """Run the complete pre-routed BF16 MoE pipeline."""
    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    num_assignments = num_tokens * top_k
    if num_tokens == 0:
        return output
    sorted_slots, block_expert, num_post_pad = _permute(
        topk_ids, w1.shape[0], block_size, workspace
    )
    gemm1_out = workspace.gemm1_out[:num_assignments]
    _grouped_gemm(
        hidden_states,
        w1,
        sorted_slots,
        block_expert,
        num_post_pad,
        gemm1_out,
        top_k=top_k,
        block_size=block_size,
        config=gemm1_config,
    )
    activation_out = workspace.activation_out[:num_assignments]
    launch_activation(gemm1_out, activation_out, activation_type)
    gemm2_out = workspace.gemm2_out[:num_assignments]
    _grouped_gemm(
        activation_out,
        w2,
        sorted_slots,
        block_expert,
        num_post_pad,
        gemm2_out,
        top_k=1,
        block_size=block_size,
        config=gemm2_config,
    )
    tile_h = min(next_positive_power_of_2(hidden_size), 1024)
    ct.launch(
        torch.cuda.current_stream(hidden_states.device),
        (num_tokens, (hidden_size + tile_h - 1) // tile_h),
        _combine,
        (
            gemm2_out.reshape(-1),
            topk_weights.reshape(-1),
            output.reshape(-1),
            top_k,
            hidden_size,
            tile_h,
        ),
    )
    return output


__all__ = ["GemmConfig", "Workspace", "allocate_workspace", "run_moe"]
