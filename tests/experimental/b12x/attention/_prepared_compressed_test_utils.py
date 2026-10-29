"""Preparation helper for tests migrated from raw compressed-MLA workspaces."""

from dataclasses import replace

import torch

from b12x.attention import compressed_sparse_mla
from b12x.attention.compressed_sparse_mla._tuning import TUNING
from b12x.attention._shared.mla.compressed_config import (
    compressed_sparse_mla_split_config_for_contract,
)
from b12x.preparation import PreparationSession, PreparedCall


def prepare_compressed_binding(
    *,
    caps,
    q,
    swa_k_cache,
    swa_indices,
    swa_lengths,
    sm_scale,
    attn_sink=None,
    indexed_k_cache=None,
    indexed_indices=None,
    indexed_lengths=None,
    swa_page_size=None,
    indexed_page_size=None,
    out=None,
):
    """Prepare one compressed MLA binding while retaining split capacity and replay owners."""
    # Retain numerical inputs and declared capacities while describing their ABI.
    if swa_page_size is None:
        swa_page_size = caps.swa_page_size
    caps = replace(
        caps,
        device=q.device,
        swa_width=int(swa_indices.shape[1]),
        indexed_width=0 if indexed_indices is None else int(indexed_indices.shape[1]),
        swa_page_size=swa_page_size,
        indexed_page_size=indexed_page_size or swa_page_size,
        shared_width_capacity=False,
    )
    invocation = compressed_sparse_mla.invocation_from_tensors(
        q=q,
        swa_k_cache=swa_k_cache,
        attn_sink=attn_sink,
        indexed_k_cache=indexed_k_cache,
        out=out,
    )
    session = PreparationSession(device=q.device, autotune=False, compile_workers=2)
    declaration = compressed_sparse_mla.plan(caps, invocation=invocation)
    config = TUNING.configure(declaration.query, device=session.device.identity).default
    if caps.max_chunks_per_row is not None and not config.single_pass:
        split = compressed_sparse_mla_split_config_for_contract(
            rows=caps.max_q_rows,
            width=caps.max_width,
            max_chunks=caps.max_chunks_per_row,
            decode_row_capacity=caps.decode_row_capacity,
        )
        config = replace(
            config,
            max_chunks_per_row=caps.max_chunks_per_row,
            split_chunk_size=split.chunk_size,
        )
        declaration = compressed_sparse_mla.plan(
            caps,
            invocation=invocation,
            override=config,
        )
    owned = {}
    bind_args = dict(
        q=q,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
    )
    run_args = dict(
        swa_k_cache=swa_k_cache,
        attn_sink=attn_sink,
        sm_scale=sm_scale,
        indexed_k_cache=indexed_k_cache,
        out=out,
    )

    def prepare_call(state):
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind_for_preparation(scratch=scratch, **bind_args)
        owned["scratch"] = scratch
        return PreparedCall(
            run=lambda: state.run(binding, **run_args), owners=(scratch, binding)
        )

    result = session.prepare(
        (
            declaration.request(
                name="compressed-mla-migrated-test",
                prepare_call=prepare_call,
            ),
        )
    )
    binding = compressed_sparse_mla.bind(
        declaration,
        scratch=owned["scratch"],
        **bind_args,
    )
    # Preparation may execute the route; subsequent assertions must observe
    # an actual public run, not a previously populated output buffer.
    binding.scratch.output_buffer.fill_(float("nan"))
    if out is not None:
        out.fill_(float("nan"))
    # Keep session/result alive for CUDA-graph replay in the calling test.
    return session, result, declaration, binding
