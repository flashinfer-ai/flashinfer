# Copyright (c) 2025 by FlashInfer team.
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
"""Shared expert-route metadata kernels for Q0-to-route implementations."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def count_expert_kernel(
    topk_ids, counts, total_pairs: tl.constexpr, block_n: tl.constexpr
):
    expert_idx = tl.program_id(0)
    offs = tl.arange(0, block_n)
    acc = tl.zeros((block_n,), dtype=tl.int32)
    for pair_begin in tl.range(0, total_pairs, block_n):
        pair_idx = pair_begin + offs
        expert = tl.load(topk_ids + pair_idx, mask=pair_idx < total_pairs, other=-1)
        acc += tl.where(expert == expert_idx, 1, 0)
    tl.store(counts + expert_idx, tl.sum(acc))


@triton.jit
def count_routes_kernel(
    topk_ids, counts, total_pairs: tl.constexpr, block_n: tl.constexpr
):
    pair_idx = tl.program_id(0) * block_n + tl.arange(0, block_n)
    valid = pair_idx < total_pairs
    expert = tl.load(topk_ids + pair_idx, mask=valid, other=0)
    tl.atomic_add(counts + expert, 1, mask=valid, sem="relaxed")


@triton.jit
def prefix_cursor_kernel(
    counts,
    offsets,
    cursor,
    num_experts: tl.constexpr,
    block_e: tl.constexpr,
):
    expert = tl.arange(0, block_e)
    valid = expert < num_experts
    count = tl.load(counts + expert, mask=valid, other=0)
    inclusive = tl.cumsum(count, axis=0)
    exclusive = inclusive - count
    tl.store(offsets + expert, exclusive, mask=valid)
    tl.store(cursor + expert, exclusive, mask=valid)
    tl.store(offsets + expert + 1, inclusive, mask=expert == num_experts - 1)


@triton.jit
def route_assign_kernel(
    topk_ids,
    topk_weights,
    offsets,
    expert_cursor,
    token_map,
    token_weights,
    dst_rows,
    scale_dst_rows,
    total_pairs: tl.constexpr,
    top_k: tl.constexpr,
    scale_align: tl.constexpr,
    block_n: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, block_n)
    pair_idx = pid * block_n + offs
    valid = pair_idx < total_pairs
    expert = tl.load(topk_ids + pair_idx, mask=valid, other=0)
    routed_row = tl.atomic_add(expert_cursor + expert, 1, mask=valid, sem="relaxed")
    token_idx = pair_idx // top_k
    expert_begin = tl.load(offsets + expert, mask=valid, other=0)
    scale_begin = (expert_begin + expert * (scale_align - 1)) & -scale_align
    tl.store(token_map + routed_row, token_idx, mask=valid)
    tl.store(
        token_weights + routed_row,
        tl.load(topk_weights + pair_idx, mask=valid, other=0.0),
        mask=valid,
    )
    tl.store(dst_rows + pair_idx, routed_row, mask=valid)
    tl.store(
        scale_dst_rows + pair_idx, scale_begin + routed_row - expert_begin, mask=valid
    )


@triton.jit
def route_assign_decode_kernel(
    topk_ids,
    topk_weights,
    offsets,
    token_map,
    token_weights,
    dst_rows,
    scale_dst_rows,
    scale_out,
    total_pairs: tl.constexpr,
    top_k: tl.constexpr,
    num_experts: tl.constexpr,
    scale_align: tl.constexpr,
    block_n: tl.constexpr,
    total_scale: tl.constexpr,
    padded_rows: tl.constexpr,
    s_som: tl.constexpr,
    s_sok: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, block_n)
    valid = offs < total_pairs
    experts = tl.load(topk_ids + offs, mask=valid, other=num_experts)
    flat = pid * block_n + offs
    k_block = flat // padded_rows
    scale_row = flat - k_block * padded_rows
    tl.store(
        scale_out + k_block * s_som + scale_row * s_sok, 0, mask=flat < total_scale
    )
    expert_prefix = tl.sum(tl.where(valid & (experts < pid), 1, 0), 0)
    tl.store(offsets + pid, expert_prefix, mask=pid < num_experts)
    tl.store(offsets + num_experts, total_pairs, mask=pid == 0)
    pair_expert = tl.load(topk_ids + pid, mask=pid < total_pairs, other=0)
    expert_begin = tl.sum(tl.where(valid & (experts < pair_expert), 1, 0), 0)
    rank = tl.sum(tl.where(valid & (experts == pair_expert) & (offs < pid), 1, 0), 0)
    routed_row = expert_begin + rank
    scale_begin = (expert_begin + pair_expert * (scale_align - 1)) & -scale_align
    token_idx = pid // top_k
    tl.store(token_map + routed_row, token_idx, mask=pid < total_pairs)
    tl.store(
        token_weights + routed_row,
        tl.load(topk_weights + pid, mask=pid < total_pairs, other=0.0),
        mask=pid < total_pairs,
    )
    tl.store(dst_rows + pid, routed_row, mask=pid < total_pairs)
    tl.store(scale_dst_rows + pid, scale_begin + rank, mask=pid < total_pairs)


@triton.jit
def moe_tile_select_kernel(
    m_indptr,
    tile_mn,
    plain_tactic,
    selected_tactic,
    hidden_size: tl.constexpr,
    fc1_inter_size: tl.constexpr,
    num_experts,
    num_sms,
    block_e: tl.constexpr,
    block_c: tl.constexpr,
):
    expert = tl.arange(0, block_e)
    mask = expert < num_experts
    rows = tl.load(m_indptr + expert + 1, mask=mask, other=0) - tl.load(
        m_indptr + expert, mask=mask, other=0
    )
    candidate = tl.arange(0, block_c)
    for op in tl.static_range(0, 2):
        offset = op * block_c * 2 + candidate * 2
        bm, bn = tl.load(tile_mn + offset), tl.load(tile_mn + offset + 1)
        valid = (bm > 0) & (bn > 0)
        safe_bm, safe_bn = tl.where(valid, bm, 1), tl.where(valid, bn, 1)
        m_tiles = tl.sum(
            (rows[:, None] + safe_bm[None, :] - 1) // safe_bm[None, :], axis=0
        ).to(tl.int64)
        scheduled_rows = m_tiles * safe_bm.to(tl.int64)
        n = fc1_inter_size // 2 if op == 0 else hidden_size
        n_tiles = (n + safe_bn - 1) // safe_bn
        work_tiles = m_tiles * n_tiles.to(tl.int64)
        waves = (work_tiles + num_sms - 1) // num_sms
        work = scheduled_rows * n_tiles.to(tl.int64) * safe_bn.to(tl.int64)
        plain = tl.load(plain_tactic + op)
        plain_mask = candidate == plain
        waves_plain = tl.sum(tl.where(plain_mask, waves, 0), axis=0)
        work_plain = tl.sum(tl.where(plain_mask, work, 0), axis=0)
        score = waves * work_plain + work * waves_plain
        score = tl.where(valid, score, 0x7FFFFFFFFFFFFFFF)
        best_score = tl.min(score, axis=0)
        first_best = tl.argmin(score, axis=0, tie_break_left=True)
        plain_best = (
            tl.sum(tl.where(plain_mask & (score == best_score), 1, 0), axis=0) > 0
        )
        chosen = tl.where(plain_best, plain, first_best)
        tl.store(selected_tactic + op, chosen)


def moe_tile_selector(
    m_indptr: torch.Tensor,
    tile_mn: torch.Tensor,
    plain_tactic: torch.Tensor,
    selected_tactic: torch.Tensor,
    *,
    hidden_size: int,
    fc1_inter_size: int,
    num_sms: int,
) -> None:
    num_experts = m_indptr.numel() - 1
    if m_indptr.dtype is not torch.int32 or m_indptr.ndim != 1:
        raise ValueError("m_indptr must be a rank-1 int32 tensor")
    if tile_mn.dtype is not torch.int32 or tile_mn.shape != (2, 4, 2):
        raise ValueError("tile_mn must have shape [2, 4, 2] and dtype int32")
    if plain_tactic.dtype is not torch.int32 or plain_tactic.shape != (2,):
        raise ValueError("plain_tactic must have shape [2] and dtype int32")
    if selected_tactic.dtype is not torch.int32 or selected_tactic.shape != (2,):
        raise ValueError("selected_tactic must have shape [2] and dtype int32")
    if not (
        m_indptr.device
        == tile_mn.device
        == plain_tactic.device
        == selected_tactic.device
    ):
        raise ValueError("selector tensors must share one device")
    if num_experts <= 0 or num_experts > 256:
        raise ValueError("selector requires 1..256 experts")
    if hidden_size <= 0 or fc1_inter_size <= 0 or fc1_inter_size % 2 != 0:
        raise ValueError("hidden_size and an even fc1_inter_size must be positive")
    if num_sms <= 0:
        raise ValueError("num_sms must be positive")
    moe_tile_select_kernel[(1,)](
        m_indptr,
        tile_mn,
        plain_tactic,
        selected_tactic,
        hidden_size=hidden_size,
        fc1_inter_size=fc1_inter_size,
        num_experts=num_experts,
        num_sms=num_sms,
        block_e=triton.next_power_of_2(num_experts),
        block_c=4,
    )


class Mxfp8Mxfp4TileSelectorRuntime:
    TILE_MN = (
        ((8, 128), (32, 128), (64, 128), (0, 0)),
        ((32, 128), (64, 128), (128, 128), (0, 0)),
    )

    def __init__(
        self,
        device: torch.device,
        hidden_size: int,
        intermediate: int,
        plain_bms: tuple[int, int],
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("tile selector runtime requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.hidden_size = hidden_size
        self.intermediate = intermediate
        indices = tuple(
            next(index for index, tile in enumerate(self.TILE_MN[op]) if tile[0] == bm)
            for op, bm in enumerate(plain_bms)
        )
        self.plain_bms = plain_bms
        self.tile_mn = torch.tensor(self.TILE_MN, dtype=torch.int32, device=self.device)
        self.plain_tactic = torch.tensor(indices, dtype=torch.int32, device=self.device)
        self.selected_cpu = torch.empty(2, dtype=torch.int32, pin_memory=True)
        with torch.cuda.device(self.device):
            self.selector_done = torch.cuda.Event()
            self.copy_done = torch.cuda.Event()
            self.copy_stream = torch.cuda.Stream(device=self.device)

    def matches(
        self,
        device: torch.device,
        hidden_size: int,
        intermediate: int,
        plain_bms: tuple[int, int],
    ) -> bool:
        return (
            self.device == torch.device(device)
            and self.hidden_size == hidden_size
            and self.intermediate == intermediate
            and self.plain_bms == plain_bms
        )

    def launch(
        self, m_indptr: torch.Tensor, selected_tactic: torch.Tensor, num_sms: int
    ) -> None:
        with torch.cuda.device(self.device):
            moe_tile_selector(
                m_indptr,
                self.tile_mn,
                self.plain_tactic,
                selected_tactic,
                hidden_size=self.hidden_size,
                fc1_inter_size=2 * self.intermediate,
                num_sms=num_sms,
            )
            self.selector_done.record()
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(self.selector_done)
                self.selected_cpu.copy_(selected_tactic, non_blocking=True)
                self.copy_done.record()

    def result(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        self.copy_done.synchronize()
        indices = tuple(int(value) for value in self.selected_cpu.tolist())
        if any(
            index < 0 or index >= len(self.TILE_MN[op])
            for op, index in enumerate(indices)
        ):
            raise RuntimeError(f"tile selector returned invalid indices {indices}")
        return (
            (*self.TILE_MN[0][indices[0]], 128),
            (*self.TILE_MN[1][indices[1]], 128),
        )


__all__ = [
    "Mxfp8Mxfp4TileSelectorRuntime",
    "count_expert_kernel",
    "count_routes_kernel",
    "moe_tile_select_kernel",
    "moe_tile_selector",
    "prefix_cursor_kernel",
    "route_assign_decode_kernel",
    "route_assign_kernel",
]
