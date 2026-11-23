# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Fixed geometry of the GLM-5 TP8 decode operators (no CUDA imports)."""

WORLD_SIZE = 8
GRID_SIZE = 148
HIDDEN_SIZE = 6144
EXPERTS = 256
EXPERT_BANKS = EXPERTS + 1
MOE_INTERMEDIATE_SIZE = 2048
LOCAL_INTERMEDIATE = MOE_INTERMEDIATE_SIZE // WORLD_SIZE
TOP_K_EXPERTS = 8
QA_SIZE = 2048
KV_SIZE = 512
ROPE_SIZE = 64
NUM_HEADS = 64
QK_NOPE_HEAD_DIM = 192
V_HEAD_DIM = 256
HEADS_PER_MLA_RANK = 10
INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
SELECTED_TOKENS = 2048
DEFAULT_MAX_SEQ_LEN = 202752
MAX_SEQ_LEN = 1048576 + 4096
CACHE_PAD = 8  # cache rows required beyond max_seq_len

MOE_GATE_UP_SHAPE = (EXPERT_BANKS, 16, 6, 32896)
MOE_DOWN_SHAPE = (EXPERT_BANKS, HIDDEN_SIZE, LOCAL_INTERMEDIATE)
MOE_DOWN_SCALE_SHAPE = (EXPERT_BANKS, 1024, 2)
MOE_STATE_BYTES = 0x500
Q1_STATE_BYTES = 1024
Q4_RECORD_BYTES = 0xE0000
Q1_RECORD_BYTES = 0x38000
ATTENTION_ARENA_BYTES = 16 << 20

# Byte offsets that the kernels address by fixed constants; the csrc name of
# each is given in its comment.
# Per-CTA launch counters inside the per-device state buffers.
MOE_Q1_COUNTER_OFFSET = 256  # csrc/moe_api.h kMoeQ1CounterOffset
MOE_Q4_COUNTER_OFFSET = 0x200  # csrc/moe_api.h kMoeQ4CounterOffset
ATTENTION_Q1_COUNTER_OFFSET = 256  # csrc/attention/q1_launch.cuh kQ1Counters
# Hidden-state partial records of the attention ranks in the q4 attention arena
# (csrc/attention/mla_q4_types.cuh and indexer_q4_types.cuh kHiddenRx and
# kHiddenRxBytes).
ATTENTION_Q4_HIDDEN_RX_OFFSET = 5309952
ATTENTION_Q4_HIDDEN_RX_BYTES = 917504


def communication_layout(operators):
    """Logical bytes and offsets; the allocator may round to a larger granularity."""
    operators = tuple(sorted(set(operators)))
    if not operators or any(x not in ("attention", "moe") for x in operators):
        raise ValueError("operators must contain 'attention' and/or 'moe'")
    offsets = {}
    cursor = 0
    if "attention" in operators:
        offsets["attention"] = cursor
        cursor += ATTENTION_ARENA_BYTES
        offsets["attention_q1"] = cursor
        cursor += Q1_RECORD_BYTES
        offsets["ll"] = cursor
        cursor += 128 << 10
    if "moe" in operators:
        offsets["moe_q4"] = cursor
        cursor += Q4_RECORD_BYTES
        offsets["moe_q1"] = cursor
        cursor += Q1_RECORD_BYTES
    return cursor, offsets


def check_q_len(q_len: int) -> int:
    """Validate the only two supported query lengths."""
    if isinstance(q_len, bool) or not isinstance(q_len, int) or q_len not in (1, 4):
        raise ValueError("GLM-5 TP8 decode supports q_len=1 or q_len=4")
    return int(q_len)


def check_rank(rank: int) -> int:
    """Validate logical rank independently of the local CUDA device index."""
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 0 <= rank < WORLD_SIZE
    ):
        raise ValueError("rank must be an integer in [0, 8)")
    return rank


def check_capacity(max_seq_len: int) -> int:
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
        raise ValueError("max_seq_len must be an integer")
    if not 1 <= max_seq_len <= MAX_SEQ_LEN:
        raise ValueError(f"max_seq_len must be in [1, {MAX_SEQ_LEN}]")
    return max_seq_len
