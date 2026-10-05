"""Experimental paged FP8 MQA lightning-indexer logits (DeepGEMM paged signatures)."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def get_paged_mqa_logits_metadata(
    context_lens, block_kv, num_sms, indices=None, *, out=None
):
    """DeepGEMM's ``get_paged_mqa_logits_metadata`` signature, as a placeholder (no launch).

    ``context_lens`` int32 ``[B, next_n]`` (two-dimensional), ``block_kv`` an
    exported page size, ``num_sms`` the CTA budget of the logits call,
    ``indices`` must be ``None``. The exported paged logits programs derive
    their schedule in-kernel, so this validates the call and returns (or
    zero-fills ``out``) an int32 ``[num_sms + 1, 2]`` placeholder that
    :func:`fp8_paged_mqa_logits` accepts for signature parity and never reads.
    Not interchangeable with DeepGEMM's buffer.
    """
    from .experimental.deepgemm_dense_mqa.paged_mqa import (
        get_paged_mqa_logits_metadata as _get_paged_mqa_logits_metadata,
    )

    return _get_paged_mqa_logits_metadata(
        context_lens, block_kv, num_sms, indices=indices, out=out
    )


@flashinfer_experimental_api
def fp8_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_meta,
    max_context_len,
    clean_logits=False,
    indices=None,
):
    """Paged FP8 MQA lightning-indexer logits with DeepGEMM's ``fp8_paged_mqa_logits`` signature.

    ``q`` E4M3 ``[B, next_n, H, 128]`` (``H`` in the catalog's head counts),
    fused uint8 ``kv_cache [pages, block_kv, 1, 132]`` read in place, FP32
    ``weights [B * next_n, H]``, int32 ``context_lens [B, next_n]``, int32
    ``block_table [B, S]`` with unit column stride, ``schedule_meta`` the
    placeholder from :func:`get_paged_mqa_logits_metadata` (fixes the CTA
    budget through its first dimension; ``None`` = the device's SM count; not
    read otherwise). ONE kernel launch. Returns the FP32 ``[B * next_n,
    max_context_len]`` view of a row-padded buffer with DeepGEMM's
    ``clean_logits=False`` semantics; ``clean_logits=True`` and a non-``None``
    ``indices`` are rejected. Any batch size; the per-architecture admission
    rules (``paged_route_admitted``) decide which (batch, max_context_len)
    calls this entry serves. ``paged_route_available(H, block_kv, next_n)`` in
    ``flashinfer.experimental.deepgemm_dense_mqa.paged_mqa`` reports the
    shipped routes; no fallback is selected for an unsupported configuration.
    """
    from .experimental.deepgemm_dense_mqa.paged_mqa import (
        fp8_paged_mqa_logits as _fp8_paged_mqa_logits,
    )

    return _fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits=clean_logits,
        indices=indices,
    )


@flashinfer_experimental_api
def prepare_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    max_context_len,
    *,
    schedule_meta=None,
    output=None,
    sm_count=None,
):
    """Prepare repeated paged MQA logits on caller-provided operands.

    Returns a :class:`PagedMqaPlan`; ``plan.run()`` submits the logits program
    (one launch) on the current PyTorch stream without allocating (CUDA Graph
    replay with changed tensor contents is supported); ``plan.logical_output``
    is the ``[B * next_n, max_context_len]`` view. ``schedule_meta`` is
    accepted for signature parity and not read; ``sm_count`` overrides the CTA
    budget.
    """
    from .experimental.deepgemm_dense_mqa.paged_mqa import PagedMqaPlan

    return PagedMqaPlan(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
        schedule_meta=schedule_meta,
        output=output,
        sm_count=sm_count,
    )
