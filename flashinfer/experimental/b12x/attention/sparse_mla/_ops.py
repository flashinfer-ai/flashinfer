"""Compiler-visible mutations for prepared sparse attention and latent writes."""

import torch

from b12x.preparation.types import plan_from_handle


@torch.library.custom_op("b12x::sparse_mla_bound", mutates_args=("scratch",))
def attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    selected: torch.Tensor,
    cache_lengths: torch.Tensor,
    selected_lengths: torch.Tensor,
    attention_sink: torch.Tensor | None,
    scratch: torch.Tensor,
    plan_handle: int,
) -> None:
    from ._preparation import state

    prepared = state(plan_from_handle(plan_handle), device=q.device)
    runtime = prepared.bind(
        scratch=scratch,
        q=q,
        selected_indices=selected,
        cache_seqlens_int32=cache_lengths,
        nsa_cache_seqlens_int32=selected_lengths,
        kv_cache=kv_cache,
    )
    prepared.run(runtime, kv_cache=kv_cache, attention_sink=attention_sink)


@attention.register_fake
def _fake_attention(
    q,
    kv_cache,
    selected,
    cache_lengths,
    selected_lengths,
    attention_sink,
    scratch,
    plan_handle,
):
    pass


@torch.library.custom_op("b12x::glm_mla_cache_write", mutates_args=("kv_cache",))
def cache_write(
    latent: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    plan_handle: int,
) -> None:
    from ._preparation import writer_state

    writer_state(plan_from_handle(plan_handle), device=kv_cache.device).run(
        latent, kv_cache, slots
    )


@cache_write.register_fake
def _fake_cache_write(latent, kv_cache, slots, plan_handle):
    pass


@torch.library.custom_op(
    "b12x::sparse_mla_pooled_selection", mutates_args=("output", "active_counts")
)
def pooled_selection(
    pools: torch.Tensor,
    positions: torch.Tensor,
    requests: torch.Tensor,
    block_table: torch.Tensor,
    output: torch.Tensor,
    active_counts: torch.Tensor,
    pool_size: int,
    block_size: int,
    block_stride_rows: int,
    num_cache_blocks: int,
    plan_handle: int,
) -> None:
    from b12x.preparation.types import require_prepared
    from .pooled_selection import _PooledSelectionState

    state = require_prepared(
        plan_from_handle(plan_handle), "attention.sparse_mla", pools.device
    )
    if not isinstance(state, _PooledSelectionState):
        raise ValueError("pooled selection requires a pooled-selection plan")
    state.run(
        pools,
        positions,
        requests,
        block_table,
        output,
        active_counts,
        pool_size=pool_size,
        block_size=block_size,
        block_stride_rows=block_stride_rows,
        num_cache_blocks=num_cache_blocks,
    )


@pooled_selection.register_fake
def _fake_pooled_selection(
    pools,
    positions,
    requests,
    block_table,
    output,
    active_counts,
    pool_size,
    block_size,
    block_stride_rows,
    num_cache_blocks,
    plan_handle,
):
    pass
