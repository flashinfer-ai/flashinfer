# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""CPU coverage for vectorized Prims-TS launch-I/O transformations."""

import pytest
import torch

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    SfLayout,
    shuffled_matrix_row_indices,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
    _expand_bf16_activations,
    _expand_fp4_activations,
    _make_token_layout,
    _permute_mma_sf_rows,
    _sf_offsets,
    reorder_rows_for_gated_act,
    shuffle_matrix,
)


def _loop_expand(compact: torch.Tensor, layout) -> torch.Tensor:
    expanded = torch.zeros(
        (layout.total_padded_tokens, compact.shape[1]), dtype=compact.dtype
    )
    for expanded_idx, expert_idx in enumerate(layout.expanded_to_expert):
        if expert_idx >= 0:
            expanded[expanded_idx] = compact[layout.expanded_to_token[expanded_idx]]
    return expanded


def _loop_gated_reorder(tensor: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(tensor)
    half = tensor.shape[-2] // 2
    for row in range(half):
        out[..., 2 * row, :] = tensor[..., row, :]
        out[..., 2 * row + 1, :] = tensor[..., row + half, :]
    return out


def _loop_shuffle(tensor: torch.Tensor, epilogue_tile_m: int) -> torch.Tensor:
    indices = shuffled_matrix_row_indices(epilogue_tile_m)
    block_size = len(indices)
    out = torch.empty_like(tensor)
    for row in range(tensor.shape[-2]):
        block_idx = row // block_size
        destination = block_idx * block_size + indices[row % block_size]
        out[..., destination, :] = tensor[..., row, :]
    return out


def _loop_permute_sf(
    sf: torch.Tensor,
    *,
    row_source_indices: torch.Tensor,
    logical_cols: int,
    layout: int,
    num_experts: int,
    sf_vec_size: int,
) -> torch.Tensor:
    out = sf.clone()
    sf_cols = (logical_cols + sf_vec_size - 1) // sf_vec_size
    rows_per_expert = row_source_indices.numel()
    for expert_idx in range(num_experts):
        for dst_row, src_row in enumerate(row_source_indices.tolist()):
            src_offsets = _sf_offsets(
                row_idx=expert_idx * rows_per_expert + src_row,
                sf_cols=sf_cols,
                layout=layout,
                device=sf.device,
            )
            dst_offsets = _sf_offsets(
                row_idx=expert_idx * rows_per_expert + dst_row,
                sf_cols=sf_cols,
                layout=layout,
                device=sf.device,
            )
            out[dst_offsets] = sf[src_offsets]
    return out


def test_vectorized_activation_expansion_matches_loop() -> None:
    layout = _make_token_layout(
        num_tokens=7,
        num_experts=4,
        top_k=2,
        tile_size=8,
        cluster_dim_in_token=2,
    )
    bf16 = torch.arange(7 * 12, dtype=torch.bfloat16).reshape(7, 12)
    fp4 = torch.arange(7 * 6, dtype=torch.uint8).reshape(7, 6)

    assert torch.equal(
        _expand_bf16_activations(bf16, layout), _loop_expand(bf16, layout)
    )
    assert torch.equal(_expand_fp4_activations(fp4, layout), _loop_expand(fp4, layout))


def test_vectorized_gated_reorder_matches_loop() -> None:
    tensor = torch.arange(2 * 32 * 5).reshape(2, 32, 5)
    assert torch.equal(reorder_rows_for_gated_act(tensor), _loop_gated_reorder(tensor))


@pytest.mark.parametrize("epilogue_tile_m", [64, 128])
def test_vectorized_shuffle_matches_loop(epilogue_tile_m: int) -> None:
    tensor = torch.arange(2 * 64 * 5).reshape(2, 64, 5)
    assert torch.equal(
        shuffle_matrix(tensor, epilogue_tile_m),
        _loop_shuffle(tensor, epilogue_tile_m),
    )


@pytest.mark.parametrize(
    "layout",
    [int(SfLayout.LINEAR), int(SfLayout.R8c4), int(SfLayout.R128c4)],
)
@pytest.mark.parametrize("gated", [False, True])
def test_vectorized_sf_row_permutation_matches_loop(layout: int, gated: bool) -> None:
    num_experts = 3
    rows_per_expert = 128
    logical_cols = 256
    sf_vec_size = 16
    sf_cols = logical_cols // sf_vec_size
    all_rows = torch.arange(num_experts * rows_per_expert)
    total = (
        int(
            _sf_offsets(
                row_idx=all_rows,
                sf_cols=sf_cols,
                layout=layout,
                device="cpu",
            ).max()
        )
        + 1
    )
    sf = torch.arange(total, dtype=torch.int64).remainder(251).to(torch.uint8)
    row_sources = torch.arange(rows_per_expert)
    if gated:
        row_sources = reorder_rows_for_gated_act(row_sources.reshape(-1, 1)).flatten()
    row_sources = shuffle_matrix(row_sources.reshape(-1, 1), 128).flatten()

    expected = _loop_permute_sf(
        sf,
        row_source_indices=row_sources,
        logical_cols=logical_cols,
        layout=layout,
        num_experts=num_experts,
        sf_vec_size=sf_vec_size,
    )
    actual = _permute_mma_sf_rows(
        sf,
        row_source_indices=row_sources,
        logical_cols=logical_cols,
        layout=layout,
        num_experts=num_experts,
        sf_vec_size=sf_vec_size,
    )

    assert torch.equal(actual, expected)
