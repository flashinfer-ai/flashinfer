"""Experimental SM120 paged FP8 MQA lightning-indexer logits (DeepGEMM signatures)."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def get_paged_mqa_logits_metadata(
    context_lens, page_kv, num_sms, indices=None, *, out=None
):
    """DeepGEMM's ``get_paged_mqa_logits_metadata`` on SM120a: a real launch.

    ``context_lens`` int32 ``[B, next_n]`` (two-dimensional), ``page_kv`` an
    exported page size, ``num_sms`` the CTA budget of the logits call,
    ``indices`` must be ``None``. Submits the single-warp scheduler and returns
    (or fills ``out`` with) the int32 ``[num_sms + 1, 2]`` schedule the exported
    logits programs read: per-CTA ``(q_atom_idx, kv_split_idx)`` walk
    boundaries over 128-row KV segments. Not interchangeable with DeepGEMM's
    buffer; pass it to :func:`fp8_paged_mqa_logits`, which rebuilds it in the
    same sequence before the logits program reads it.
    """
    from .experimental.deepgemm_sm120_paged_mqa_logits.sm120_paged_mqa import (
        get_paged_mqa_logits_metadata as _get_paged_mqa_logits_metadata,
    )

    return _get_paged_mqa_logits_metadata(
        context_lens, page_kv, num_sms, indices=indices, out=out
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
    *,
    page_kv=None,
):
    """Paged FP8 MQA lightning-indexer logits with DeepGEMM's signature (SM120a).

    ``q`` E4M3 ``[B, next_n, H, 128]`` (``H`` in the catalog's head counts),
    fused uint8 ``kv_cache [pages, page_kv, 1, 132]`` read in place (or its
    padded 2-D ``[pages, block_stride_bytes]`` view with ``page_kv`` given),
    FP32 ``weights [B * next_n, H]``, int32 ``context_lens [B, next_n]``, int32
    ``block_table [B, S]`` with unit column stride, ``schedule_meta`` the
    buffer from :func:`get_paged_mqa_logits_metadata` (its first dimension
    fixes the CTA budget; ``None`` = the device's SM count). ONE FFI
    submission, TWO kernel launches: the scheduler is rebuilt into
    ``schedule_meta`` and the persistent logits program consumes it.

    Returns the FP32 ``[B * next_n, max_context_len]`` view of a row-padded
    buffer with DeepGEMM's ``clean_logits=False`` semantics (positions at or
    past a token's own context length are unspecified); ``clean_logits=True``
    and a non-``None`` ``indices`` are rejected.
    ``route_available(H, page_kv, next_n)`` in
    ``flashinfer.experimental.deepgemm_sm120_paged_mqa_logits.sm120_paged_mqa``
    reports the shipped routes; no fallback is selected for an unsupported
    configuration.
    """
    from .experimental.deepgemm_sm120_paged_mqa_logits.sm120_paged_mqa import (
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
        page_kv=page_kv,
    )


@flashinfer_experimental_api
def prepare_sm120_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    max_context_len,
    *,
    page_kv=None,
    schedule_meta=None,
    output=None,
    sm_count=None,
):
    """Prepare repeated paged MQA logits on caller-provided operands (SM120a).

    Returns an ``Sm120PagedIndexerPlan``; ``plan.run()`` submits the
    ``(metadata, logits)`` prepared sequence on the current PyTorch stream
    without allocating (CUDA Graph replay with changed tensor contents is
    supported) and returns ``plan.logical_output``, the ``[B * next_n,
    max_context_len]`` view. ``schedule_meta`` is rebuilt by every run;
    ``sm_count`` overrides the CTA budget.
    """
    from .experimental.deepgemm_sm120_paged_mqa_logits.sm120_paged_mqa import (
        Sm120PagedIndexerPlan,
    )

    return Sm120PagedIndexerPlan(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
        page_kv=page_kv,
        schedule_meta=schedule_meta,
        output=output,
        sm_count=sm_count,
    )
