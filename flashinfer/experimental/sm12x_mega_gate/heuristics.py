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
"""Largest block of the size rule: 16, 32, 48 or 64 tokens."""
BLOCK_TOKENS_CHOICES = (16, 32, 48, 64, 96, 128)
"""Blocks above 64 tokens split the tokens over two MMA warps per 32 experts (a 32x64
accumulator tile per warp) and read each weight tile for twice the tokens."""
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


MAX_COOPERATIVE_TOKENS = 3
"""Up to this many tokens per CTA and block, every gate warp ranks them (cooperative gate):
a (token, 128-expert wave) item per warp instead of a whole token per warp."""
MIN_COOPERATIVE_STAGES = 3
"""The cooperative gate's exchange takes shared memory from the pipeline; keep this many
stages or fall back to one warp per token."""


def cooperative_slots(block_tokens: int, num_logical_ctas: int) -> int:
    """Tokens of a block a CTA ranks (at most)."""
    return ceil_div(block_tokens, num_logical_ctas)


def is_cooperative(block_tokens: int, num_logical_ctas: int, num_groups: int) -> bool:
    """Whether a schedule runs the cooperative gate: few tokens per CTA, 4 MMA warps (8
    gate warps next to 8 MMA warps would spill), and enough stages left by the exchange."""
    slots = cooperative_slots(block_tokens, num_logical_ctas)
    return (
        select_mma_warps_n(block_tokens) == 1
        and slots <= MAX_COOPERATIVE_TOKENS
        and max_stages(
            block_tokens, slots * exchange_bytes(num_groups * EXPERTS_PER_CTA)
        )
        >= MIN_COOPERATIVE_STAGES
    )


def select_gate_warps(
    block_tokens: int,
    num_logical_ctas: int,
    mma_warps_n: int = 1,
    cooperative: bool = False,
) -> int:
    # With 8 MMA warps, 8 gate warps would cap a thread at 120 registers and spill the
    # mainloop (measured 2.5x slower at 128-token blocks).
    if mma_warps_n > 1:
        return 4
    tokens_per_cta = ceil_div(block_tokens, num_logical_ctas)
    if cooperative:
        # Warp 0 polls; one token's waves (3 for 384 experts) take warps 1-3.
        return 4 if tokens_per_cta == 1 else 8
    return 4 if tokens_per_cta <= MAX_TOKENS_PER_CTA_FOR_4_GATE_WARPS else 8


def select_mma_warps_n(block_tokens: int) -> int:
    """MMA warps along the tokens: 1 up to 64 block tokens, else 2 (each warp keeps a 32x64
    accumulator tile; at 64 tokens, 2 measured 0.93x-1.02x of 1)."""
    return 1 if block_tokens <= 64 else 2


TARGET_TOKEN_BLOCKS = 4
"""Token blocks per call before the block grows (16, 32, 48 then 64 tokens)."""
WIDE_BLOCK_MIN_TOKENS = 513
"""From this many tokens the schedule is chosen among 96- and 128-token blocks (each weight
tile serves more tokens; 5-11% faster than 64-token blocks from 768 tokens, slower at 512)."""
WIDE_BLOCK_TOKENS = (96, 128)

# Cost model of one CTA's critical path, in microseconds, fitted (non-negative least
# squares, relative error) on the schedules measured within 15% of the best of their token
# count (16-128 token blocks, every split, 1-16384 tokens) on the 110-SM RTX PRO 5000.
# With the cooperative gate of up to 3 tokens per CTA: 0.7% slower than the measured best
# on average over 26 token counts, 3.9% at worst (320 tokens, the block rule).
COST_PER_TILE_ROW = 0.0017247
"""Per K block and staged tile row (weight rows + token rows): the load side."""
COST_PER_MMA_TOKEN = 0.0040045
"""Per K block and token of the block: the tensor-core side (4x1 MMA warps)."""
COST_PER_WIDE_MMA_TOKEN = 0.0040944
"""The same with 4x2 MMA warps (blocks above 64 tokens)."""
COST_PER_WAVE = 1.6644
"""Per token block a worker group runs: arrival and pipeline turnaround."""
COST_PER_WAVE_TOKEN = 0.016512
"""Per token block and token: the partial-logit stores."""
COST_PER_TAIL_TOKEN = 2.7365
"""Per token a gate warp ranks after the last block, when nothing overlaps it."""
COST_PER_TAIL_TOKEN_SPLIT = 0.19731
"""Per split-K partial such a tail token reduces."""
COST_PER_COOPERATIVE_TAIL = 2.0183
"""The last block's tokens when every gate warp of the CTA ranks them."""
COST_PER_COOPERATIVE_TAIL_SPLIT = 0.15841
"""Per split-K partial it reduces."""


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def stage_bytes(block_tokens: int) -> int:
    """Shared memory of one pipeline stage: the weight and token tiles of one K block."""
    return (EXPERTS_PER_CTA + block_tokens) * BLOCK_K * 2


def exchange_bytes(num_experts: int) -> int:
    """Shared memory of a cooperative gate's exchange: a sorted key, an expert id, a value
    and a physical expert (4 B each) per aligned expert."""
    return 4 * 4 * align(num_experts, EXPERTS_PER_CTA)


def max_stages(block_tokens: int, reserved: int = 0) -> int:
    return min(
        MAX_STAGES,
        (SMEM_CAPACITY_BYTES - SMEM_RESERVED_BYTES - reserved)
        // stage_bytes(block_tokens),
    )


@dataclass(frozen=True)
class MegaGateConfig:
    """One compiled schedule; the token and worker-group counts stay runtime arguments."""

    block_tokens: int
    num_split_k: int
    num_expert_groups: int
    num_stages: int
    num_gate_warps: int = 4
    mma_warps_n: int = 1
    """MMA warps along the tokens (4 along the experts); see ``select_mma_warps_n``."""
    cooperative_gate: bool = False
    """At most ``MAX_COOPERATIVE_TOKENS`` tokens per CTA and block: every gate warp ranks
    them (the split-K loads and transcendentals of one token's 128-expert wave per item),
    instead of one warp per token."""

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
    mma_warps_n = select_mma_warps_n(block_tokens)
    cost = (
        COST_PER_TILE_ROW * k_blocks * (EXPERTS_PER_CTA + block_tokens)
        + (COST_PER_MMA_TOKEN if mma_warps_n == 1 else COST_PER_WIDE_MMA_TOKEN)
        * k_blocks
        * block_tokens
        + (COST_PER_WAVE + COST_PER_WAVE_TOKEN * block_tokens) * num_waves
    )
    if is_cooperative(block_tokens, num_logical_ctas, num_groups):
        # Up to 3 tokens per CTA, 8 gate warps rank them in about the time 4 take for one.
        return (
            cost + COST_PER_COOPERATIVE_TAIL + COST_PER_COOPERATIVE_TAIL_SPLIT * split
        )
    tail_tokens = ceil_div(
        min(block_tokens, num_tokens),
        num_logical_ctas
        * select_gate_warps(block_tokens, num_logical_ctas, mma_warps_n),
    )
    return cost + tail_tokens * (
        COST_PER_TAIL_TOKEN + COST_PER_TAIL_TOKEN_SPLIT * split
    )


def select_config(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_sms: int,
    deterministic: bool = False,
) -> MegaGateConfig:
    """Pick the schedule for ``num_tokens`` tokens on a ``num_sms``-SM device: the block
    size rule and the modelled best split up to 512 tokens, the modelled best wide block
    and split above."""
    if hidden <= 0 or hidden % BLOCK_K:
        raise ValueError(f"hidden must be a positive multiple of {BLOCK_K}")
    if not 0 < num_experts <= MAX_EXPERTS:
        raise ValueError(f"num_experts must be in [1, {MAX_EXPERTS}]")
    num_groups = align(num_experts, EXPERTS_PER_CTA) // EXPERTS_PER_CTA
    if num_groups > num_sms:
        raise ValueError("every expert group of a token block needs its own SM")
    num_k_blocks = hidden // BLOCK_K
    num_tokens = max(num_tokens, 1)

    block_choices = (
        WIDE_BLOCK_TOKENS
        if num_tokens >= WIDE_BLOCK_MIN_TOKENS
        else (select_block_tokens(num_tokens),)
    )
    best = None
    for block_tokens in block_choices:
        for split in (1,) if deterministic else range(1, MAX_SPLIT_K + 1):
            if num_k_blocks % split or num_groups * split > num_sms:
                continue
            cost = estimated_us(
                num_tokens, num_k_blocks, num_groups, num_sms, block_tokens, split
            )
            # Ties: fewer partials to reduce, then the smaller block.
            key = (cost, split, block_tokens)
            if best is None or key < best:
                best = key
    _, split, block_tokens = best
    return make_config(block_tokens, split, num_experts)


def make_config(
    block_tokens: int, num_split_k: int, num_experts: int
) -> MegaGateConfig:
    """The schedule of a block size and split-K: gate warps, gate mode and stage count."""
    num_groups = align(num_experts, EXPERTS_PER_CTA) // EXPERTS_PER_CTA
    num_logical_ctas = num_groups * num_split_k
    cooperative = is_cooperative(block_tokens, num_logical_ctas, num_groups)
    mma_warps_n = select_mma_warps_n(block_tokens)
    exchange = (
        cooperative_slots(block_tokens, num_logical_ctas) * exchange_bytes(num_experts)
        if cooperative
        else 0
    )
    # The pipeline runs across the token blocks of a worker group, so the stage
    # count is bounded by shared memory only.
    return MegaGateConfig(
        block_tokens=block_tokens,
        num_split_k=num_split_k,
        num_expert_groups=num_groups,
        num_stages=max_stages(block_tokens, exchange),
        num_gate_warps=select_gate_warps(
            block_tokens, num_logical_ctas, mma_warps_n, cooperative
        ),
        mma_warps_n=mma_warps_n,
        cooperative_gate=cooperative,
    )
