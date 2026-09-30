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
    padding and storage ownership. Only the exported physical schedules of the
    present architecture and SM count are accepted; no fallback is selected for
    an unsupported configuration.
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
