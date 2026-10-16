"""Experimental paged FP8 MQA lightning-indexer logits (DeepGEMM paged signatures)."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def get_paged_mqa_logits_metadata(
    context_lens, block_kv, num_sms, indices=None, *, out=None
):
    """DeepGEMM's ``get_paged_mqa_logits_metadata`` signature, as a placeholder (no launch).

    ``context_lens`` int32 ``[B, next_n]`` (two-dimensional), ``block_kv`` an
    exported page size, ``num_sms`` the CTA budget of the logits call,
    ``indices`` must be ``None``. The default paged logits programs derive
    their schedule in-kernel, so this validates the call and returns (or
    zero-fills ``out``) an int32 ``[num_sms + 1, 2]`` placeholder that
    :func:`fp8_paged_mqa_logits` accepts for signature parity and never reads.
    Not interchangeable with DeepGEMM's buffer. The produced table of the
    optional table path comes from :func:`prepare_paged_mqa_schedule_meta`.
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
    read otherwise: the default path) or the :class:`PagedScheduleMetaPlan` of
    :func:`prepare_paged_mqa_schedule_meta`, already run on the current
    ``context_lens`` (the optional table path: inside the catalog's
    ``sched_meta_windows`` the table-consuming static program, bit-identical
    logits; the default program elsewhere). ONE logits kernel launch on both
    paths. Returns the FP32 ``[B * next_n,
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
    is the ``[B * next_n, max_context_len]`` view. ``schedule_meta``: ``None``
    or the DeepGEMM placeholder tensor keeps the default path (not read); a
    :class:`PagedScheduleMetaPlan` from :func:`prepare_paged_mqa_schedule_meta`
    opts into the table path -- served inside the catalog's
    ``sched_meta_windows`` on the plain static route (``plan.schedule_meta_used``
    True: the table-consuming program, bit-identical logits, one launch), the
    default plan elsewhere; run the meta plan before every logits launch whose
    ``context_lens`` contents changed. ``sm_count`` overrides the CTA budget (a
    meta plan fixes it to its own).
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


@flashinfer_experimental_api
def prepare_paged_mqa_schedule_meta(context_lens, *, num_sms=None, out=None):
    """Prepare the optional per-forward schedule-meta producer launch.

    ``context_lens`` int32 ``[B, next_n]`` (two-dimensional, on CUDA; the same
    tensor the logits plans bind), ``num_sms`` the CTA budget of those plans
    (the device's SM count by default), ``out`` an optional contiguous int32
    ``[num_sms + 1, 2]`` table to write into. Returns a
    :class:`PagedScheduleMetaPlan` (nothing is launched here); ``plan.run()``
    submits the catalogued producer program (ONE launch, CUDA-graph
    capturable, no allocation) on the current PyTorch stream and returns
    ``plan.table``. Pass the plan as ``schedule_meta`` to
    :func:`prepare_paged_mqa_logits` / :func:`fp8_paged_mqa_logits`; those
    serve the table-consuming program where the catalog admits the call and the
    default program elsewhere (``logits_plan.schedule_meta_used``).
    """
    from .experimental.deepgemm_dense_mqa.paged_mqa import PagedScheduleMetaPlan

    return PagedScheduleMetaPlan(context_lens, num_sms=num_sms, out=out)
