"""Compiler-visible storage dependencies for prepared FP8 index selection."""

import torch

from b12x.preparation.types import plan_from_handle, require_prepared


@torch.library.custom_op(
    "b12x::dsa_indexer_fp8", mutates_args=("scratch", "indices", "scores")
)
def select(
    query: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    active_width: torch.Tensor | None,
    schedule: torch.Tensor | None,
    scratch: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor | None,
    handle: int,
) -> None:
    state = require_prepared(
        plan_from_handle(handle), "attention.dsa_indexer", query.device
    )
    caps = state.layout.caps
    runtime = state.bind(
        scratch=scratch,
        real_page_table=page_table,
        cache_seqlens_int32=lengths,
        active_width=active_width,
        schedule_metadata=schedule,
        expected_num_q_heads=caps.num_q_heads,
        shared_page_table=caps.shared_page_table,
        output_physical_slots=caps.output_physical_slots,
        _initialize=False,
    )
    state.run(
        runtime,
        q_fp8=query,
        query_weights=weights,
        index_k_cache=cache,
        output_indices=indices,
        output_scores=scores,
    )


@select.register_fake
def _fake_select(
    query,
    weights,
    cache,
    page_table,
    lengths,
    active_width,
    schedule,
    scratch,
    indices,
    scores,
    handle,
):
    pass
