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

Tests for the SM90 MoE GEMM1 (gather + grouped GEMM + SiLU gating).
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


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize(
    "n2,k,tile_shape_mn",
    [
        # Qwen3-30B-A3B GEMM1: N2 = 2*768/tp interleaved up/gate, K = 2048
        (1536, 2048, (128, 128)),  # tp=1
        (1536, 2048, (128, 256)),  # tp=1, 2 MMA warpgroups (reg-budget fixed)
        (768, 2048, (64, 128)),  # tp=2, tile_m=64
        (192, 2048, (128, 192)),  # tp=8, 2-WG small-N tile
        (192, 2048, (128, 64)),  # tp=8 (tile N=64)
        (384, 2048, (128, 128)),  # tp=4
    ],
)
@pytest.mark.parametrize("num_tokens", [3, 777])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cute_dsl_bf16_gather_grouped_gemm(n2, k, tile_shape_mn, num_tokens, dtype):
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
        sm90_contiguous_gather_grouped_gemm_act_fusion,
    )

    torch.manual_seed(0)
    num_experts, top_k = 128, 8
    tile_m = tile_shape_mn[0]
    inter = n2 // 2

    ids = make_random_topk_ids(num_experts, num_tokens, top_k)
    scales = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)
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

    x = torch.randn(num_tokens, k, device="cuda", dtype=dtype) / (k**0.25)
    # vLLM-style [gate; up] concatenated weights -> 32-col interleave.
    w_gate_up = torch.randn(num_experts, n2, k, device="cuda", dtype=dtype) / (k**0.25)
    w1 = interleave_up_gate_sm90(w_gate_up)

    out = sm90_contiguous_gather_grouped_gemm_act_fusion(
        x,
        w1,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        topk=top_k,
        permuted_m=permuted_m,
        tile_shape_mn=tile_shape_mn,
    )
    assert out.shape == (permuted_m, inter)

    # Reference on valid rows only: silu(x @ gate.T) * (x @ up.T).
    n_tiles = int(num_non_exiting_tiles.item())
    rows = torch.arange(permuted_m, device="cuda")
    row_tile = rows // tile_m
    row_valid = (row_tile < n_tiles) & (rows < tile_idx_to_mn_limit.long()[row_tile])

    w_gate = w_gate_up[:, :inter].float()
    w_up = w_gate_up[:, inter:].float()
    expert_of_tile = tile_idx_to_expert_idx.long()
    token_of_expanded = torch.arange(num_tokens, device="cuda").repeat_interleave(top_k)
    token_of_row = torch.zeros(permuted_m, dtype=torch.long, device="cuda")
    perm_rows = expanded_idx_to_permuted_idx.flatten().long()
    token_of_row[perm_rows] = token_of_expanded

    ref = torch.zeros(permuted_m, inter, device="cuda", dtype=torch.float32)
    for e in torch.unique(expert_of_tile[:n_tiles]).tolist():
        rows_e = rows[row_valid & (expert_of_tile[row_tile] == e)]
        if rows_e.numel():
            xe = x[token_of_row[rows_e]].float()
            gate = xe @ w_gate[e].T
            up = xe @ w_up[e].T
            ref[rows_e] = torch.nn.functional.silu(gate) * up

    torch.testing.assert_close(
        out[row_valid].float(), ref[row_valid], atol=2e-1, rtol=3e-2
    )
