# Copyright (c) 2026 by NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Strict-overlap workloads with a fully visible, fixed KV history.

seq_len counts measured queries, following HISTORY_TOKENS unmeasured tokens.
Sliding top-k windows wrap around the fixed history, whose capacity is at
least the largest eight-query union. Wrapping therefore preserves the exact
intersection, including across query-stage boundaries, at every query length.
"""

import math
import functools

import numpy as np

BLOCK_SIZE = 64
Q_STAGE_TOKENS = 8
HISTORY_TOKENS = 8192


def _stride(sharing_ratio, topk_blocks):
    if not 1 <= topk_blocks <= 16:
        raise ValueError("topk_blocks must be in [1, 16]")
    if not 0 <= sharing_ratio <= 1:
        raise ValueError("sharing_ratio must be in [0, 1]")
    stride = topk_blocks * (1 - sharing_ratio)
    if not math.isclose(stride, round(stride), abs_tol=1e-9):
        raise ValueError("sharing_ratio must be exactly representable at this top-k")
    return round(stride)


def make_history_indices(seq_len, sharing_ratio, topk_blocks=16):
    """Return sorted strict-overlap indices for queries after a fixed history."""
    if seq_len <= 0 or seq_len % Q_STAGE_TOKENS:
        raise ValueError("seq_len must be positive and divisible by 8")
    stride = _stride(sharing_ratio, topk_blocks)
    pool = HISTORY_TOKENS // BLOCK_SIZE
    rows = (
        np.arange(seq_len, dtype=np.int64)[:, None] * stride + np.arange(topk_blocks)
    ) % pool
    indices = np.sort(rows, axis=-1).astype(np.int32)
    validate_history_indices(indices, sharing_ratio, topk_blocks)
    return indices


def validate_history_indices(indices, sharing_ratio, topk_blocks=16):
    """Fail before GPU timing if any row, adjacency or stage union is wrong."""
    stride = _stride(sharing_ratio, topk_blocks)
    if indices.ndim != 2 or indices.shape[1] != topk_blocks:
        raise ValueError("indices must have shape [measured_queries, topk_blocks]")
    if len(indices) == 0 or len(indices) % Q_STAGE_TOKENS:
        raise ValueError("measured_queries must be positive and divisible by 8")
    if indices.min() < 0 or indices.max() >= HISTORY_TOKENS // BLOCK_SIZE:
        raise ValueError("selected blocks must lie entirely in the visible KV history")
    if not np.all(np.diff(indices, axis=-1) > 0):
        raise ValueError("rows must be sorted and unique")
    intersections = (indices[1:, :, None] == indices[:-1, None, :]).any(-1).sum(-1)
    if not np.all(intersections == topk_blocks - stride):
        raise ValueError("actual adjacent overlap differs from requested sharing")
    stages = np.sort(indices.reshape(-1, Q_STAGE_TOKENS * topk_blocks), axis=-1)
    union_counts = 1 + (np.diff(stages, axis=-1) != 0).sum(-1)
    expected_union = topk_blocks + (Q_STAGE_TOKENS - 1) * stride
    if not np.all(union_counts == expected_union):
        raise ValueError("stage union differs from the requested workload")
    return {
        "actual_sharing_ratio": (topk_blocks - stride) / topk_blocks,
        "max_union_blocks": expected_union,
        "effective_pairs_per_head": len(indices) * topk_blocks * BLOCK_SIZE,
    }


@functools.cache
def _benchmark_module():
    from flashinfer.experimental.union_gqa.jit import gen_union_gqa_module

    return gen_union_gqa_module(benchmark_q_prefix=HISTORY_TOKENS).build_and_load()


def make_union_runner(q, k, v, indices, out):
    """Build a benchmark-only launch that omits all history-query CTAs.

    Full Q/K/V and output storage retain the production self-attention layout.
    Only the active query suffix is scheduled; math and sparse masks are the
    production kernel's. Baselines receive the same suffix and complete K/V.
    """
    import torch

    module = _benchmark_module()
    full_len = q.shape[0]
    if full_len != HISTORY_TOKENS + len(indices):
        raise ValueError("Q storage must include the history prefix")
    pattern = torch.full(
        (full_len, k.shape[1], indices.shape[1]),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    active = torch.from_numpy(indices).to(device=q.device)
    pattern[HISTORY_TOKENS:] = active[:, None, :]
    cu = torch.tensor([0, full_len], dtype=torch.int32, device=q.device)

    def run():
        module.run(q, k, v, pattern, cu, out, 128**-0.5, full_len)

    return run
