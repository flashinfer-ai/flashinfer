# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0

"""Schedule selection for the SM12x mega gate; pure Python, no CUDA required.

As in DeepGEMM, a token block is computed by ``L = num_expert_groups *
num_split_k`` logical CTAs (128 experts and one K slice each) and the grid holds
``num_sms // L`` worker groups of L CTAs that stride over the token blocks. The
token block grows with the token count so that a call has about four blocks
(``select_block_tokens``); the split-K factor is then the argmin of a cost model
of one CTA's critical path (``estimated_us``). Both were derived from every
candidate schedule measured on the target board, the 110-SM RTX PRO 5000
(K=5120, E=384, top-6): over 21 token counts from 1 to 16384 the pick is 0.7%
slower than the measured best on average and 2.9% at worst. Deterministic
requests never split K, which keeps every token's logits independent of the
token count (batch invariance).
"""

from __future__ import annotations

from dataclasses import dataclass

BLOCK_K = 64
"""K extent of one pipeline stage: one 128 B swizzle row of BF16."""

EXPERTS_PER_CTA = 128
"""Experts per CTA: the MMA M extent, 32 experts per MMA warp."""

TOKEN_ALIGNMENT = 16
MAX_BLOCK_TOKENS = 64
BLOCK_TOKENS_CHOICES = tuple(
    range(TOKEN_ALIGNMENT, MAX_BLOCK_TOKENS + 1, TOKEN_ALIGNMENT)
)
MAX_SPLIT_K = 16
"""Bounds the split-K partials a gate lane loads per wave (one float4 each)."""
MAX_STAGES = 8
MAX_EXPERTS = 1024
"""Top-k keeps ``ceil(E / 128) * 4`` candidates per lane in registers."""

SMEM_CAPACITY_BYTES = 101376
"""Opt-in dynamic shared memory per CTA on SM120/SM121 (99 KiB)."""

SMEM_RESERVED_BYTES = 2048
"""Pipeline barriers and the exit flag, plus the 1 KiB alignment of the operand buffers."""

GATE_WARP_CHOICES = (4, 8)
"""Gate warps per CTA; every logical CTA of a block ranks a share of its tokens."""
MAX_TOKENS_PER_CTA_FOR_4_GATE_WARPS = 4
"""Above this share of a block per CTA, 8 gate warps shorten the serial per-warp token chain
(measured 1.03x-1.09x at 512-2048 tokens, 0.97x-0.99x below on a 110-SM board)."""


def select_gate_warps(block_tokens: int, num_logical_ctas: int) -> int:
    tokens_per_cta = ceil_div(block_tokens, num_logical_ctas)
    return 4 if tokens_per_cta <= MAX_TOKENS_PER_CTA_FOR_4_GATE_WARPS else 8


TARGET_TOKEN_BLOCKS = 4
"""Token blocks per call before the block grows (16, 32, 48 then 64 tokens)."""

# Split-K cost model of one CTA's critical path, in microseconds, fitted
# (non-negative least squares, relative error) on the measured schedules whose
# block size ``select_block_tokens`` picks, on the 110-SM RTX PRO 5000.
COST_PER_TILE_ROW = 0.00125
"""Per K block and staged tile row (weight rows + token rows): the load side."""
COST_PER_MMA_TOKEN = 0.00501
"""Per K block and token of the block: the tensor-core side."""
COST_PER_WAVE = 3.252
"""Per token block a worker group runs: partial stores, arrival, pipeline turnaround."""
COST_PER_TAIL_TOKEN = 1.574
"""Per token a gate warp ranks after the last block, when nothing overlaps it."""
COST_PER_TAIL_TOKEN_SPLIT = 0.403
"""Per split-K partial such a tail token reduces."""


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def stage_bytes(block_tokens: int) -> int:
    """Shared memory of one pipeline stage: the weight and token tiles of one K block."""
    return (EXPERTS_PER_CTA + block_tokens) * BLOCK_K * 2


def max_stages(block_tokens: int) -> int:
    return min(
        MAX_STAGES,
        (SMEM_CAPACITY_BYTES - SMEM_RESERVED_BYTES) // stage_bytes(block_tokens),
    )


@dataclass(frozen=True)
class MegaGateConfig:
    """One compiled schedule; the token and worker-group counts stay runtime arguments."""

    block_tokens: int
    num_split_k: int
    num_expert_groups: int
    num_stages: int
    num_gate_warps: int = 4

    @property
    def num_logical_ctas(self) -> int:
        """CTAs that contribute partial logits to one token block."""
        return self.num_expert_groups * self.num_split_k

    def num_token_blocks(self, num_tokens: int) -> int:
        return ceil_div(num_tokens, self.block_tokens)

    def num_worker_groups(self, num_tokens: int, num_sms: int) -> int:
        """Persistent groups of ``num_logical_ctas`` CTAs, at most one CTA per SM."""
        return max(
            1,
            min(num_sms // self.num_logical_ctas, self.num_token_blocks(num_tokens)),
        )

    def num_ctas(self, num_tokens: int, num_sms: int) -> int:
        return self.num_worker_groups(num_tokens, num_sms) * self.num_logical_ctas

    def scratch_shape(self, num_tokens: int, num_experts: int) -> tuple:
        """FP32 partial logits ``[token blocks, split-K, block tokens, aligned experts]``."""
        return (
            self.num_token_blocks(num_tokens),
            self.num_split_k,
            self.block_tokens,
            align(num_experts, EXPERTS_PER_CTA),
        )


def select_block_tokens(num_tokens: int) -> int:
    """16-aligned block of about ``num_tokens / TARGET_TOKEN_BLOCKS`` tokens, 16 to 64."""
    block_tokens = align(ceil_div(num_tokens, TARGET_TOKEN_BLOCKS), TOKEN_ALIGNMENT)
    return min(MAX_BLOCK_TOKENS, max(TOKEN_ALIGNMENT, block_tokens))


def estimated_us(
    num_tokens: int,
    num_k_blocks: int,
    num_groups: int,
    num_sms: int,
    block_tokens: int,
    split: int,
) -> float:
    """Modelled critical path of one CTA for a schedule (up to a constant)."""
    num_logical_ctas = num_groups * split
    num_worker_groups = num_sms // num_logical_ctas
    num_waves = ceil_div(ceil_div(num_tokens, block_tokens), num_worker_groups)
    k_blocks = num_waves * (num_k_blocks // split)
    tail_tokens = ceil_div(
        min(block_tokens, num_tokens),
        num_logical_ctas * select_gate_warps(block_tokens, num_logical_ctas),
    )
    return (
        COST_PER_TILE_ROW * k_blocks * (EXPERTS_PER_CTA + block_tokens)
        + COST_PER_MMA_TOKEN * k_blocks * block_tokens
        + COST_PER_WAVE * num_waves
        + tail_tokens * (COST_PER_TAIL_TOKEN + COST_PER_TAIL_TOKEN_SPLIT * split)
    )


def select_config(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_sms: int,
    deterministic: bool = False,
) -> MegaGateConfig:
    """Pick the schedule for ``num_tokens`` tokens on a ``num_sms``-SM device."""
    if hidden <= 0 or hidden % BLOCK_K:
        raise ValueError(f"hidden must be a positive multiple of {BLOCK_K}")
    if not 0 < num_experts <= MAX_EXPERTS:
        raise ValueError(f"num_experts must be in [1, {MAX_EXPERTS}]")
    num_groups = align(num_experts, EXPERTS_PER_CTA) // EXPERTS_PER_CTA
    if num_groups > num_sms:
        raise ValueError("every expert group of a token block needs its own SM")
    num_k_blocks = hidden // BLOCK_K
    num_tokens = max(num_tokens, 1)

    block_tokens = select_block_tokens(num_tokens)
    best = None
    for split in (1,) if deterministic else range(1, MAX_SPLIT_K + 1):
        if num_k_blocks % split or num_groups * split > num_sms:
            continue
        cost = estimated_us(
            num_tokens, num_k_blocks, num_groups, num_sms, block_tokens, split
        )
        # Ties: fewer partials to reduce.
        if best is None or (cost, split) < best[0]:
            best = ((cost, split), split)
    _, split = best
    # The pipeline runs across the token blocks of a worker group, so the stage
    # count is bounded by shared memory only.
    return MegaGateConfig(
        block_tokens=block_tokens,
        num_split_k=split,
        num_expert_groups=num_groups,
        num_stages=max_stages(block_tokens),
        num_gate_warps=select_gate_warps(block_tokens, num_groups * split),
    )
