"""Pure-Python reference oracle for the dispatch kernel.

Simulates the entire dispatch protocol on the CPU and returns the expected
state of the nine workspace buffers, per receiving rank, after the kernel
finishes. Result is the byte-level ground truth used to validate the GPU
kernel output.

The simulation mirrors the receiver-driven advertise + pull dispatch protocol
implemented in
``DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh``
(lines 432-766). Each receiving rank's pool is laid out by iterating the
local experts in order, and within each expert the tokens are placed in the
exact round-robin (min-peeling) order produced by the device-side scheduler.

Token metadata 12-byte packing convention (matches ``TokenSrcMetadata`` in
``DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh``):
``[rank_idx: u32 LE][token_idx: u32 LE][topk_idx: u32 LE]``.
Padding entries are filled with ``0xFF`` per byte (sentinel).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import (
    DSV4Config,
    DSV4,
    MAX_SLOT,
    TOKEN_METADATA_BYTES,
    transform_sf_token_idx_numpy,
)


_METADATA_PAD_BYTE: int = 0xFF


@dataclass
class ExpectedBuffers:
    """Expected state of the nine workspace buffers for one receiving rank."""

    expert_send_count: (
        np.ndarray
    )  # (num_total_experts,)              uint64 (bit-packed)
    expert_recv_count: np.ndarray  # (num_ranks, num_experts_per_rank) uint64
    expert_recv_count_sum: (
        np.ndarray
    )  # (num_experts_per_rank,)           uint64 (bit-packed)
    src_token_topk_idx: np.ndarray  # (num_experts_per_rank, num_ranks, MAX_SLOT) uint32
    token_src_metadata: np.ndarray  # (num_max_pool_tokens, 8)          uint8
    l1_arrival_count: np.ndarray  # (num_max_task_tiles,)             uint32
    l1_token_buffer: np.ndarray  # (num_max_pool_tokens, hidden_bytes) uint8
    l1_sf_buffer: np.ndarray  # (sf_uint32_per_token, num_padded_sf_pool_tokens) uint32
    l1_topk_weights_buf: np.ndarray  # (num_max_pool_tokens,)            float32


def _pack_metadata(rank_idx: int, token_idx: int, topk_idx: int) -> np.ndarray:
    """Pack {rank, token, topk} into the i64 record written by
    ``TokenSrcMetadata.store``: low 32b = token, high 32b = (rank << 16) | topk.
    """
    rec = np.zeros(TOKEN_METADATA_BYTES, dtype=np.uint8)
    hi = ((int(rank_idx) << 16) | int(topk_idx)) & 0xFFFFFFFF
    rec[0:4] = np.frombuffer(np.uint32(token_idx).tobytes(), dtype=np.uint8)
    rec[4:8] = np.frombuffer(np.uint32(hi).tobytes(), dtype=np.uint8)
    return rec


def _round_robin_pool_order(rank_counts: np.ndarray) -> List[int]:
    """Return the source-rank for each pool slot inside one expert.

    Mirrors the device-side round-robin min-peeling loop: each round contributes
    ``length * num_active_ranks`` tokens, where ``length`` is the smallest
    non-zero per-rank remaining count. Within a round the source ranks are
    visited in ascending rank-id order (lane order on the device).
    """
    remaining = rank_counts.astype(np.int64).copy()
    order: List[int] = []
    while True:
        active = [r for r, n in enumerate(remaining) if n > 0]
        if not active:
            break
        length = min(remaining[r] for r in active)
        for _ in range(int(length)):
            for r in active:
                order.append(r)
        for r in active:
            remaining[r] -= length
    return order


def dispatch_oracle(
    input_token_buffer: np.ndarray,
    input_sf_buffer: np.ndarray,
    input_topk_idx_buffer: np.ndarray,
    input_topk_weights_buffer: np.ndarray,
    config: DSV4Config = DSV4,
) -> List[ExpectedBuffers]:
    """Compute the expected per-rank workspace state after dispatch.

    Args:
        input_token_buffer: shape ``(num_ranks, num_tokens_per_rank,
            hidden_bytes)`` uint8 tensor holding every rank's token bodies
            (FP8 = 7168 B/token, NVFP4 = 3584 B/token).
        input_sf_buffer: shape ``(num_ranks, num_tokens_per_rank,
            sf_uint32_per_token)`` uint32 tensor of per-token scaling-factor
            columns.
        input_topk_idx_buffer: shape ``(num_ranks, num_tokens_per_rank,
            num_topk)`` int64 tensor of routing destinations. Negative values
            mean masked routing (the entry is skipped, matching DEC-4).
        input_topk_weights_buffer: shape ``(num_ranks, num_tokens_per_rank,
            num_topk)`` float32 tensor of routing weights.
        config: locked test configuration (DSV4 by default).

    Returns:
        A list of length ``num_ranks``; entry ``r`` describes the expected
        workspace state on the rank with rank-id ``r``.
    """
    R = config.num_ranks
    T = config.num_tokens_per_rank
    K = config.num_topk
    H_BYTES = config.hidden_bytes
    E_local = config.num_experts_per_rank
    E_total = config.num_total_experts
    P_TOK = config.num_max_pool_tokens
    P_SF = config.num_padded_sf_pool_tokens
    P_TT = config.num_max_task_tiles
    SFU = config.sf_uint32_per_token
    BM = config.block_m
    CTM = config.effective_cluster_tile_m
    SFBM = config.sf_block_m
    NSM = config.num_sms

    assert input_token_buffer.shape == (R, T, H_BYTES)
    assert input_sf_buffer.shape == (R, T, SFU)
    assert input_topk_idx_buffer.shape == (R, T, K)
    assert input_topk_weights_buffer.shape == (R, T, K)
    assert E_total == R * E_local

    # Per-(src_rank, global_expert) token counts.
    per_src_send = np.zeros((R, E_total), dtype=np.int64)
    for src_rank in range(R):
        for src_token in range(T):
            for src_topk in range(K):
                expert_id = int(input_topk_idx_buffer[src_rank, src_token, src_topk])
                if expert_id >= 0:
                    per_src_send[src_rank, expert_id] += 1

    # Build initial expected buffers (zero-fill matches kernel post-conditions
    # for unused regions, except token_src_metadata which uses 0xFF sentinel).
    expected: List[ExpectedBuffers] = []
    for _ in range(R):
        expected.append(
            ExpectedBuffers(
                expert_send_count=np.zeros((E_total,), dtype=np.uint64),
                expert_recv_count=np.zeros((R, E_local), dtype=np.uint64),
                expert_recv_count_sum=np.zeros((E_local,), dtype=np.uint64),
                src_token_topk_idx=np.zeros((E_local, R, MAX_SLOT), dtype=np.uint32),
                token_src_metadata=np.full(
                    (P_TOK, TOKEN_METADATA_BYTES), _METADATA_PAD_BYTE, dtype=np.uint8
                ),
                l1_arrival_count=np.zeros((P_TT,), dtype=np.uint32),
                l1_token_buffer=np.zeros((P_TOK, H_BYTES), dtype=np.uint8),
                l1_sf_buffer=np.zeros((SFU, P_SF), dtype=np.uint32),
                l1_topk_weights_buf=np.zeros((P_TOK,), dtype=np.float32),
            )
        )

    # expert_send_count on rank ``src_rank``: low 32 bits = local token count,
    # high 32 bits = publisher count (one per dispatch SM).
    publisher_per_rank = np.uint64(NSM)
    for src_rank in range(R):
        for expert_id in range(E_total):
            tokens = np.uint64(int(per_src_send[src_rank, expert_id]))
            expected[src_rank].expert_send_count[expert_id] = (
                publisher_per_rank << np.uint64(32)
            ) | tokens

    # expert_recv_count[recv_rank][src_rank, local_expert]: low 32 = token count,
    # high 32 = 0 (high half is delivered to expert_recv_count_sum instead).
    for recv_rank in range(R):
        for src_rank in range(R):
            for local_expert in range(E_local):
                global_expert = recv_rank * E_local + local_expert
                tokens = np.uint64(int(per_src_send[src_rank, global_expert]))
                expected[recv_rank].expert_recv_count[src_rank, local_expert] = tokens

    # expert_recv_count_sum[recv_rank][local_expert]: low 32 = sum_over_src tokens,
    # high 32 = num_ranks * num_sms = sum of publisher counts from every rank's SMs.
    publishers_global = np.uint64(R * NSM)
    for recv_rank in range(R):
        for local_expert in range(E_local):
            global_expert = recv_rank * E_local + local_expert
            tokens = np.uint64(int(per_src_send[:, global_expert].sum()))
            expected[recv_rank].expert_recv_count_sum[local_expert] = (
                publishers_global << np.uint64(32)
            ) | tokens

    # Populate src_token_topk_idx by replaying each source rank's topk scan.
    # Slot is per (local_expert, src_rank): the advertise-table layout already
    # separates src_rank into its own dimension, so each (local_expert, src_rank)
    # series starts at slot 0.
    for src_rank in range(R):
        cursor = np.zeros(E_total, dtype=np.int64)
        for src_token in range(T):
            for src_topk in range(K):
                expert_id = int(input_topk_idx_buffer[src_rank, src_token, src_topk])
                if expert_id < 0:
                    continue
                dst_rank = expert_id // E_local
                local_expert = expert_id % E_local
                slot = int(cursor[expert_id])
                cursor[expert_id] += 1
                token_topk_word = np.uint32(src_token * K + src_topk)
                expected[dst_rank].src_token_topk_idx[local_expert, src_rank, slot] = (
                    token_topk_word
                )

    # Pool layout per receiving rank: experts in order; within each expert the
    # per-rank counts decide a round-robin order matching the kernel scheduler.
    for recv_rank in range(R):
        eb = expected[recv_rank]
        pool_block_offset = 0
        task_tile_offset = 0
        for local_expert in range(E_local):
            global_expert = recv_rank * E_local + local_expert
            rank_counts = per_src_send[:, global_expert].astype(np.int64)
            T_e = int(rank_counts.sum())
            if T_e == 0:
                continue

            order = _round_robin_pool_order(rank_counts)
            assert len(order) == T_e

            per_rank_cursor = np.zeros((R,), dtype=np.int64)
            for token_idx_in_expert, src_rank in enumerate(order):
                token_idx_in_rank = int(per_rank_cursor[src_rank])
                per_rank_cursor[src_rank] += 1

                src_token_topk = int(
                    eb.src_token_topk_idx[local_expert, src_rank, token_idx_in_rank]
                )
                src_token = src_token_topk // K
                src_topk = src_token_topk % K

                pool_token_idx = pool_block_offset * BM + token_idx_in_expert

                eb.l1_token_buffer[pool_token_idx, :] = input_token_buffer[
                    src_rank, src_token, :
                ]
                eb.l1_topk_weights_buf[pool_token_idx] = input_topk_weights_buffer[
                    src_rank, src_token, src_topk
                ]

                sf_pool_token_idx = pool_block_offset * SFBM + int(
                    transform_sf_token_idx_numpy(token_idx_in_expert)
                )
                eb.l1_sf_buffer[:, sf_pool_token_idx] = input_sf_buffer[
                    src_rank, src_token, :
                ]

                eb.token_src_metadata[pool_token_idx, :] = _pack_metadata(
                    src_rank, src_token, src_topk
                )

                task_tile = task_tile_offset + token_idx_in_expert // CTM
                eb.l1_arrival_count[task_tile] += np.uint32(1)

            pool_block_offset += (T_e + BM - 1) // BM
            task_tile_offset += (T_e + CTM - 1) // CTM

    return expected


def assert_buffer_equal(
    actual: np.ndarray,
    expected: np.ndarray,
    name: str,
    mode: str = "byte_exact",
) -> None:
    """Compare ``actual`` against ``expected`` and raise on mismatch.

    Args:
        actual: device-side result, already moved to host as a NumPy array.
        expected: oracle result.
        name: human-readable label printed on mismatch.
        mode: ``"byte_exact"`` requires ``np.array_equal``; ``"logical"``
            tolerates floating-point round-off via ``np.allclose`` for float
            arrays and exact equality otherwise.

    Raises:
        AssertionError: if any element differs (according to the chosen mode),
            with a message that pinpoints the first differing flat index and
            the actual / expected scalar values there.
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"[{name}] shape mismatch: actual={actual.shape} expected={expected.shape}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"[{name}] dtype mismatch: actual={actual.dtype} expected={expected.dtype}"
        )

    if mode == "byte_exact":
        if np.array_equal(actual, expected):
            return
        diff_mask = actual != expected
    elif mode == "logical":
        if np.issubdtype(actual.dtype, np.floating):
            if np.allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True):
                return
            diff_mask = ~np.isclose(
                actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True
            )
        else:
            if np.array_equal(actual, expected):
                return
            diff_mask = actual != expected
    else:
        raise ValueError(f"Unknown comparison mode: {mode!r}")

    flat_idx = int(np.argmax(diff_mask.reshape(-1)))
    multi_idx = np.unravel_index(flat_idx, actual.shape)
    raise AssertionError(
        f"[{name}] mismatch at index {multi_idx} (flat {flat_idx}): "
        f"actual={actual[multi_idx]!r} expected={expected[multi_idx]!r} "
        f"(total mismatches={int(diff_mask.sum())})"
    )
