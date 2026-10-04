"""Experimental prepared dense FP4/FP8 MQA lightning-indexer logits."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_dense_mqa_logits(
    precision,
    q,
    kv,
    weights,
    starts,
    ends,
    *,
    q_scales=None,
    kv_scales=None,
    output=None,
    metadata=None,
):
    """Prepare repeated dense MQA logits on caller-provided quantized operands.

    Returns a plan with ``run()``, ``output`` and ``logical_output``. ``run()``
    enqueues metadata generation and the fused logits/cleanup kernel on the
    current PyTorch stream. See :class:`DenseMqaPlan` for packed formats,
    padding and storage ownership. Any query count up to ``max_queries()`` and
    any KV length that is a multiple of 256 are accepted on the catalogued
    architectures (SM100a, SM103a) with any SM count; no fallback is selected
    for an unsupported configuration.
    """
    from .experimental.deepgemm_dense_mqa.dense_mqa import DenseMqaPlan

    return DenseMqaPlan(
        precision,
        q,
        kv,
        weights,
        starts,
        ends,
        q_scales=q_scales,
        kv_scales=kv_scales,
        output=output,
        metadata=metadata,
    )


@flashinfer_experimental_api
def fp8_mqa_logits(
    q,
    kv,
    weights,
    ks,
    ke,
    clean_logits=False,
    max_seqlen_k=0,
    *,
    sm_count=None,
):
    """FP8 dense MQA lightning-indexer logits with DeepGEMM's ``fp8_mqa_logits`` signature.

    ``q`` E4M3 ``[Q, H, 128]`` (``H`` in the catalog's head counts), ``kv`` the
    pair ``(E4M3 [K, 128], FP32 scales [K])``, FP32 ``weights [Q, H]``, int32
    ``ks`` / ``ke [Q]`` with ``0 <= ks <= ke <= K``. Returns the FP32 ``[Q, K]``
    view of a row-padded buffer whose every cell is written (``-inf`` outside
    each row's window), so ``clean_logits`` does not change the result;
    ``max_seqlen_k`` must be 0. ``sm_count`` overrides the CTA budget (default:
    the device's SM count). Which ``(H, Q, K)`` points are served is a catalog
    property: ``dense_route_available(H, Q, K)`` in
    ``flashinfer.experimental.deepgemm_dense_mqa.dense_mqa``; no fallback is
    selected for an unsupported configuration.
    """
    from .experimental.deepgemm_dense_mqa.dense_mqa import (
        fp8_mqa_logits as _fp8_mqa_logits,
    )

    return _fp8_mqa_logits(
        q,
        kv,
        weights,
        ks,
        ke,
        clean_logits=clean_logits,
        max_seqlen_k=max_seqlen_k,
        sm_count=sm_count,
    )
