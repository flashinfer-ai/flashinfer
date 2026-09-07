"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for the SM90 MoE GEMM2 (grouped GEMM + fused/deterministic finalize).
"""

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_available
from flashinfer.utils import get_compute_capability

cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="Requires cute-dsl (nvidia-cutlass-dsl)"
)


def is_sm90():
    if not torch.cuda.is_available():
        return False
    return get_compute_capability(torch.device("cuda"))[0] == 9


sm90_required = pytest.mark.skipif(not is_sm90(), reason="Requires SM90 (Hopper) GPU")


def make_random_topk_ids(num_experts, num_tokens, top_k, device="cuda"):
    return torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).int()


def _build_case(num_experts, top_k, num_tokens, hidden, inter, tile_m, dtype):
    """Route tokens with moe_sort and build a contiguous-grouped intermediate.

    Invalid rows (tile padding) are poisoned with a large constant so any
    scatter of a padded row shows up as a gross output error.
    """
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort

    ids = make_random_topk_ids(num_experts, num_tokens, top_k)
    scales = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)
    # Zero-scale routes must contribute exactly nothing in fused mode.
    scales[:, 0] = 0.0
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _total_padded,
        num_non_exiting_tiles,
    ) = moe_sort(
        ids, scales, num_experts=num_experts, top_k=top_k, tile_tokens_dim=tile_m
    )
    permuted_m = tile_idx_to_expert_idx.numel() * tile_m

    intermediate = torch.full((permuted_m, inter), 1000.0, device="cuda", dtype=dtype)
    n_tiles = int(num_non_exiting_tiles.item())
    rows = torch.arange(permuted_m, device="cuda")
    row_tile = rows // tile_m
    row_valid = (row_tile < n_tiles) & (rows < tile_idx_to_mn_limit.long()[row_tile])
    intermediate[row_valid] = torch.randn(
        int(row_valid.sum()), inter, device="cuda", dtype=dtype
    ) / (inter**0.25)

    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    maps = (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
    )
    return intermediate, w2, scales, maps, expanded_idx_to_permuted_idx, row_valid


def _ref_expanded_gemm(intermediate, w2, maps, row_valid, tile_m):
    """Per-valid-row expert GEMM in fp32: ref[r] = intermediate[r] @ w2[e].T."""
    tile_idx_to_expert_idx, _, _, num_non_exiting_tiles = maps
    permuted_m = intermediate.shape[0]
    hidden = w2.shape[1]
    rows = torch.arange(permuted_m, device="cuda")
    row_tile = rows // tile_m
    expert_of_tile = tile_idx_to_expert_idx.long()
    n_tiles = int(num_non_exiting_tiles.item())
    ref = torch.zeros(permuted_m, hidden, device="cuda", dtype=torch.float32)
    for e in torch.unique(expert_of_tile[:n_tiles]).tolist():
        rows_e = rows[row_valid & (expert_of_tile[row_tile] == e)]
        if rows_e.numel():
            ref[rows_e] = intermediate[rows_e].float() @ w2[e].float().T
    return ref


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize(
    "hidden,inter,tile_shape_mn,tile_k,cluster_shape_mn,raster_along_m",
    [
        # Qwen3-30B-A3B GEMM2: hidden=2048, I=768/tp
        (2048, 768, (128, 128), 64, (1, 1), False),  # tp=1
        (2048, 768, (128, 256), 32, (1, 1), False),  # tp=1, 2-WG tile, k32
        (2048, 768, (128, 128), 64, (1, 2), False),  # tp=1, N-cluster pair
        (2048, 768, (128, 128), 64, (1, 1), True),  # tp=1, M-major raster
        (2048, 192, (64, 128), 32, (1, 1), False),  # tp=4, tile_m=64
        (2048, 96, (128, 64), 32, (1, 1), False),  # tp=8, I%64!=0 (k32 only)
    ],
)
@pytest.mark.parametrize("num_tokens", [3, 777])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cute_dsl_grouped_gemm_finalize_fused(
    hidden,
    inter,
    tile_shape_mn,
    tile_k,
    cluster_shape_mn,
    raster_along_m,
    num_tokens,
    dtype,
):
    """Fused mode: router-scaled scatter-reduce into the token output."""
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_grouped_gemm_finalize_fusion import (
        sm90_contiguous_grouped_gemm_finalize_fusion,
    )

    torch.manual_seed(7)
    num_experts, top_k = 128, 8
    tile_m = tile_shape_mn[0]
    intermediate, w2, scales, maps, expanded_to_perm, row_valid = _build_case(
        num_experts, top_k, num_tokens, hidden, inter, tile_m, dtype
    )
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
    ) = maps

    out = torch.zeros(num_tokens, hidden, device="cuda", dtype=dtype)
    sm90_contiguous_grouped_gemm_finalize_fusion(
        intermediate,
        w2,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        scales,
        out,
        topk=top_k,
        use_fused_finalize=True,
        tile_shape_mn=tile_shape_mn,
        tile_k=tile_k,
        cluster_shape_mn=cluster_shape_mn,
        raster_along_m=raster_along_m,
    )

    ref_rows = _ref_expanded_gemm(intermediate, w2, maps, row_valid, tile_m)
    perm_rows = expanded_to_perm.flatten().long()
    ref = (
        (ref_rows[perm_rows] * scales.flatten().unsqueeze(1))
        .view(num_tokens, top_k, hidden)
        .sum(dim=1)
    )
    torch.testing.assert_close(out.float(), ref, atol=2e-1, rtol=3e-2)


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize(
    "hidden,inter,tile_shape_mn,tile_k",
    [
        (2048, 768, (128, 128), 64),
        (2048, 96, (128, 64), 32),
    ],
)
@pytest.mark.parametrize("num_tokens", [3, 777])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cute_dsl_grouped_gemm_finalize_deterministic(
    hidden, inter, tile_shape_mn, tile_k, num_tokens, dtype
):
    """Deterministic mode: unscaled rows scattered to expanded (token, slot)
    order; every valid route lands exactly once and padding rows never do."""
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_grouped_gemm_finalize_fusion import (
        sm90_contiguous_grouped_gemm_finalize_fusion,
    )

    torch.manual_seed(13)
    num_experts, top_k = 128, 8
    tile_m = tile_shape_mn[0]
    intermediate, w2, scales, maps, expanded_to_perm, row_valid = _build_case(
        num_experts, top_k, num_tokens, hidden, inter, tile_m, dtype
    )
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
    ) = maps

    out = torch.zeros(num_tokens * top_k, hidden, device="cuda", dtype=dtype)
    sm90_contiguous_grouped_gemm_finalize_fusion(
        intermediate,
        w2,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        scales,
        out,
        topk=top_k,
        use_fused_finalize=False,
        tile_shape_mn=tile_shape_mn,
        tile_k=tile_k,
        cluster_shape_mn=(1, 1),
        raster_along_m=False,
    )

    ref_rows = _ref_expanded_gemm(intermediate, w2, maps, row_valid, tile_m)
    perm_rows = expanded_to_perm.flatten().long()
    ref = ref_rows[perm_rows]
    torch.testing.assert_close(out.float(), ref, atol=2e-1, rtol=3e-2)
