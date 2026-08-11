"""
Copyright (c) 2025 by FlashInfer team.

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

"""Contracts for moe_sort's contiguous per-CTA route windows.

Above ``kContiguousRouteWindowMinTokens`` (65536) the cooperative routing
kernel gives each CTA one contiguous span of expanded indices instead of a grid
stride, so a grouped-GEMM tile gathers from a few narrow token windows. These
tests pin the permutation invariants that the ordering must preserve, plus the
locality property that motivates it.

The row-accounting assertions matter specifically: an ordering bug that lets a
CTA run past its own span duplicates routes while still producing a
self-consistent expanded<->permuted map and single-expert tiles, so the
bijection check alone does not catch it.
"""

import pytest
import torch

from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort
from flashinfer.utils import is_sm100a_supported

# Must match kContiguousRouteWindowMinTokens in csrc/moe_utils_binding.cu.
CONTIGUOUS_WINDOW_MIN_TOKENS = 65536

NUM_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
LOCAL_EXPERT_OFFSET = 0
HIDDEN_SIZE = 6144

# The marker is evaluated at import time, so short-circuit on CPU-only hosts:
# is_sm100a_supported queries the current device.
sm100_required = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_sm100a_supported(torch.device("cuda"))),
    reason="Requires an SM100-family GPU with CUDA 12.8+",
)


def _run_moe_sort(num_tokens: int, tile_size: int, seed: int):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    selected = (
        torch.rand(num_tokens, NUM_EXPERTS, device=device, generator=generator)
        .topk(TOP_K, dim=1)
        .indices.to(torch.int32)
        .contiguous()
    )
    final_scales = torch.softmax(
        torch.randn(num_tokens, TOP_K, device=device, generator=generator), dim=-1
    ).contiguous()
    outputs = moe_sort(
        token_selected_experts=selected,
        token_final_scales=final_scales,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=LOCAL_EXPERT_OFFSET,
        num_local_experts=LOCAL_EXPERTS,
        tile_tokens_dim=tile_size,
        enable_pdl=False,
    )
    torch.cuda.synchronize()
    return selected, outputs


def _live_mask(tile_limit, num_tiles, tile_size, device):
    starts = torch.arange(num_tiles, device=device, dtype=torch.int64) * tile_size
    col = torch.arange(tile_size, device=device, dtype=torch.int64)
    return (starts[:, None] + col[None, :]) < tile_limit[:num_tiles].to(torch.int64)[
        :, None
    ]


@sm100_required
@pytest.mark.parametrize(
    "num_tokens",
    [
        pytest.param(32768, id="below-threshold"),
        pytest.param(CONTIGUOUS_WINDOW_MIN_TOKENS, id="at-threshold"),
        pytest.param(2 * CONTIGUOUS_WINDOW_MIN_TOKENS, id="above-threshold"),
    ],
)
@pytest.mark.parametrize("tile_size", [128, 256])
def test_permutation_is_a_bijection_over_local_routes(num_tokens, tile_size):
    device = torch.device("cuda")
    (
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
    # Every local route occupies exactly one live slot, and the per-expert
    # counts must reproduce the routing histogram.
    assert int(live.sum()) == int(is_local.sum())
    hist = torch.bincount(
        flat_selected[is_local].long() - LOCAL_EXPERT_OFFSET, minlength=LOCAL_EXPERTS
    )
    per_expert = torch.zeros(LOCAL_EXPERTS, dtype=torch.int64, device=device)
    per_expert.index_add_(0, tile_expert[:num_tiles].long(), live.sum(dim=1))
    assert torch.equal(per_expert, hist)

    # Each tile holds rows of a single expert.
    rows = permuted_to_expanded[: num_tiles * tile_size].view(num_tiles, tile_size)
    safe_rows = rows.long().clamp(0, flat_selected.numel() - 1)
    row_expert = torch.where(live, flat_selected[safe_rows].long(), -1)
    want = tile_expert[:num_tiles].long()[:, None].expand_as(row_expert)
    assert torch.all((row_expert == want) | ~live)


@sm100_required
def test_contiguous_windows_bound_the_gather_footprint():
    """A tile's gather footprint must stop growing with the batch size.

    Under the grid-stride ordering the distinct pages touched by one tile grow
    with the token count; contiguous windows keep it bounded, which is the
    whole point of the ordering.
    """
    device = torch.device("cuda")
    tile_size = 256
    tokens_per_page = (2 << 20) // HIDDEN_SIZE

    def mean_pages_per_tile(num_tokens: int) -> float:
        (
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
    # A 4x larger batch must not widen the per-tile footprint. Compared with a
    # generous margin so the check tracks the ordering, not routing jitter.
    assert large <= small * 1.5, (
        f"gather footprint grew with batch size: {small:.1f} -> {large:.1f} pages/tile"
    )


if __name__ == "__main__":
    pytest.main([__file__])
