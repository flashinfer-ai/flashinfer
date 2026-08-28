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
"""

"""Fused CuTe routing coverage matching the moe_sort route-window contracts."""

import pytest
import torch

from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_permuted_tokens
from flashinfer.fused_moe.cute_dsl.moe_utils import moe_routing
from flashinfer.utils import is_sm100a_supported


CONTIGUOUS_WINDOW_MIN_TOKENS = 65536
NUM_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = NUM_EXPERTS
LOCAL_EXPERT_OFFSET = 0
HIDDEN_SIZE = 6144

sm100_required = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_sm100a_supported(torch.device("cuda"))),
    reason="Requires an SM100-family GPU with CUDA 12.8+",
)


def _run_moe_sort(num_tokens: int, tile_size: int, seed: int):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand(
        num_tokens,
        NUM_EXPERTS,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    padded_m = get_max_num_permuted_tokens(num_tokens, TOP_K, LOCAL_EXPERTS, tile_size)
    routing = moe_routing(
        scores,
        top_k=TOP_K,
        padded_m=padded_m,
        tile_size=tile_size,
        use_pdl=False,
        emit_expanded_to_permuted=True,
    )
    weights = routing["token_final_scales"]
    selected = routing["token_selected_experts"]
    torch.cuda.synchronize()
    tile_expert = routing["tile_idx_to_expert_idx"]
    tile_limit = routing["tile_idx_to_mn_limit"]
    permuted_to_expanded = routing["permuted_idx_to_expanded_idx"]
    num_tiles = routing["num_non_exiting_tiles"]
    expanded_to_permuted = routing["expanded_idx_to_permuted_idx"]
    outputs = (
        tile_expert,
        tile_limit,
        expanded_to_permuted,
        permuted_to_expanded,
        num_tiles * tile_size,
        num_tiles,
    )
    return scores, weights, selected, outputs


def _live_mask(tile_limit, num_tiles, tile_size, device):
    starts = torch.arange(num_tiles, device=device, dtype=torch.int64) * tile_size
    col = torch.arange(tile_size, device=device, dtype=torch.int64)
    return (starts[:, None] + col[None, :]) < tile_limit[:num_tiles].to(torch.int64)[
        :, None
    ]


@sm100_required
@pytest.mark.parametrize("hot_expert", [False, True], ids=["uniform", "hot"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "num_tokens,num_experts,top_k,tile_size",
    [
        pytest.param(1, 8, 1, 8, id="single-token"),
        pytest.param(8, 16, 2, 16, id="single-cta"),
        pytest.param(32, 256, 8, 64, id="single-cta-boundary"),
        pytest.param(33, 384, 2, 128, id="multi-cta-boundary"),
        pytest.param(64, 384, 2, 192, id="small-multi-cta"),
        pytest.param(128, 512, 8, 256, id="cluster"),
        pytest.param(513, 1024, 8, 128, id="noncluster-multi-cta"),
    ],
)
def test_topk_and_tiled_metadata(
    num_tokens, num_experts, top_k, tile_size, dtype, hot_expert
):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(7)
    scores = torch.randn(
        num_tokens,
        num_experts,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if hot_expert:
        scores[:, 0] += 32
    padded_m = get_max_num_permuted_tokens(num_tokens, top_k, num_experts, tile_size)
    routing = moe_routing(
        scores,
        top_k=top_k,
        padded_m=padded_m,
        tile_size=tile_size,
        use_pdl=False,
        emit_expanded_to_permuted=True,
    )
    weights = routing["token_final_scales"]
    selected = routing["token_selected_experts"]
    torch.cuda.synchronize()
    tile_expert = routing["tile_idx_to_expert_idx"]
    tile_limit = routing["tile_idx_to_mn_limit"]
    permuted_to_expanded = routing["permuted_idx_to_expanded_idx"]
    num_tiles_t = routing["num_non_exiting_tiles"]
    inverse = routing["expanded_idx_to_permuted_idx"]
    sorted_logits, sorted_ids = torch.sort(
        scores.float(), dim=-1, descending=True, stable=True
    )
    topk_logits = sorted_logits[:, :top_k]
    reference_ids = sorted_ids[:, :top_k]
    torch.testing.assert_close(selected, reference_ids.to(torch.int32))
    torch.testing.assert_close(
        weights, torch.softmax(topk_logits, dim=-1), atol=2e-4, rtol=2e-4
    )

    num_tiles = int(num_tiles_t.item())
    live = _live_mask(tile_limit, num_tiles, tile_size, device)
    rows = permuted_to_expanded[: num_tiles * tile_size].view(num_tiles, tile_size)
    live_rows = rows[live]
    num_pairs = num_tokens * top_k
    expanded = torch.arange(num_pairs, dtype=torch.int32, device=device)
    assert torch.equal(live_rows.sort().values, expanded)
    assert torch.equal(
        permuted_to_expanded[inverse.reshape(-1).long()],
        expanded,
    )
    assert torch.equal(
        selected.reshape(-1)[live_rows.long()],
        tile_expert[:num_tiles, None].expand(-1, tile_size)[live],
    )


@sm100_required
@pytest.mark.parametrize(
    "num_tokens,tile_size",
    [
        pytest.param(num_tokens, tile_size, id=f"{window}-tile{tile_size}")
        for num_tokens, window in (
            (32768, "below-threshold"),
            (CONTIGUOUS_WINDOW_MIN_TOKENS, "at-threshold"),
            (2 * CONTIGUOUS_WINDOW_MIN_TOKENS, "above-threshold"),
        )
        for tile_size in (128, 256)
    ]
    + [
        pytest.param(8192, tile_size, id=f"multi-cta-tile{tile_size}")
        for tile_size in (8, 16, 32, 64, 128)
    ],
)
def test_permutation_is_a_bijection_over_local_routes(num_tokens, tile_size):
    device = torch.device("cuda")
    (
        scores,
        weights,
        selected,
        (
            tile_expert,
            tile_limit,
            expanded_to_permuted,
            permuted_to_expanded,
            _total,
            num_tiles_t,
        ),
    ) = _run_moe_sort(num_tokens, tile_size, seed=7)
    num_tiles = int(num_tiles_t.item())

    if num_tokens == 8192:
        scores_f32 = scores.float()
        topk_logits, reference_ids = torch.topk(scores_f32, TOP_K, dim=-1, sorted=True)
        torch.testing.assert_close(weights, torch.softmax(topk_logits, dim=-1))
        torch.testing.assert_close(scores_f32.gather(1, selected.long()), topk_logits)
        sorted_selected = selected.sort(dim=-1).values
        assert torch.all(sorted_selected[:, 1:] != sorted_selected[:, :-1])
        torch.testing.assert_close(
            sorted_selected,
            reference_ids.to(torch.int32).sort(dim=-1).values,
        )

    flat_selected = selected.reshape(-1)
    is_local = (flat_selected >= LOCAL_EXPERT_OFFSET) & (
        flat_selected < LOCAL_EXPERT_OFFSET + LOCAL_EXPERTS
    )
    e2p = expanded_to_permuted.reshape(-1)
    assert torch.all(e2p[~is_local] == -1)
    local_perm = e2p[is_local]
    assert torch.all(local_perm >= 0)
    assert local_perm.unique().numel() == int(is_local.sum())
    assert torch.equal(
        permuted_to_expanded[local_perm.long()],
        torch.nonzero(is_local, as_tuple=True)[0].to(torch.int32),
    )

    live = _live_mask(tile_limit, num_tiles, tile_size, device)
    assert int(live.sum()) == int(is_local.sum())
    hist = torch.bincount(
        flat_selected[is_local].long() - LOCAL_EXPERT_OFFSET,
        minlength=LOCAL_EXPERTS,
    )
    per_expert = torch.zeros(LOCAL_EXPERTS, dtype=torch.int64, device=device)
    per_expert.index_add_(0, tile_expert[:num_tiles].long(), live.sum(dim=1))
    assert torch.equal(per_expert, hist)

    rows = permuted_to_expanded[: num_tiles * tile_size].view(num_tiles, tile_size)
    safe_rows = rows.long().clamp(0, flat_selected.numel() - 1)
    row_expert = torch.where(live, flat_selected[safe_rows].long(), -1)
    want = tile_expert[:num_tiles].long()[:, None].expand_as(row_expert)
    assert torch.all((row_expert == want) | ~live)


@sm100_required
def test_contiguous_windows_bound_the_gather_footprint():
    """A tile's gather footprint must stop growing with the batch size."""
    device = torch.device("cuda")
    tile_size = 256
    tokens_per_page = (2 << 20) // HIDDEN_SIZE

    def mean_pages_per_tile(num_tokens: int) -> float:
        (
            _scores,
            _weights,
            _selected,
            (
                _tile_expert,
                tile_limit,
                _e2p,
                permuted_to_expanded,
                _total,
                num_tiles_t,
            ),
        ) = _run_moe_sort(num_tokens, tile_size, seed=7)
        num_tiles = int(num_tiles_t.item())
        live = _live_mask(tile_limit, num_tiles, tile_size, device)
        rows = permuted_to_expanded[: num_tiles * tile_size].view(num_tiles, tile_size)
        tokens = torch.div(rows.long().clamp(min=0), TOP_K, rounding_mode="floor")
        pages = torch.div(tokens, tokens_per_page, rounding_mode="floor")
        sample = min(num_tiles, 128)
        counts = [
            int(torch.unique(pages[i][live[i]]).numel())
            for i in range(sample)
            if bool(live[i].any())
        ]
        return sum(counts) / max(len(counts), 1)

    small = mean_pages_per_tile(CONTIGUOUS_WINDOW_MIN_TOKENS)
    large = mean_pages_per_tile(4 * CONTIGUOUS_WINDOW_MIN_TOKENS)
    assert large <= small * 1.5, (
        f"gather footprint grew with batch size: {small:.1f} -> {large:.1f} pages/tile"
    )


if __name__ == "__main__":
    pytest.main([__file__])
