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


def dense_route_available(num_heads, queries, keys, precision="fp8", *, arch=None):
    """Host-only admission of one ``(precision, H, Q, K)`` point on ``arch``: True when the shipped catalog
    carries its route and the route is admitted there.  This is the table the engine consults before taking
    the Cake route; a 64-head tier the producer withheld on an architecture (``dense_admission(arch)``) is not
    served there, so the engine keeps its stock kernel (the 64-head ``:short`` one-split routes are the exception: a
    withheld ``:short`` call is served by its tier route, so this stays True).  ``arch`` (:func:`device_arch` of the device, e.g.
    ``"sm_100a"``) may be omitted only where the architectures agree on the route; otherwise ``ValueError``."""
    from .experimental.deepgemm_dense_mqa.dense_mqa import (
        dense_route_available as _available,
    )

    return _available(num_heads, queries, keys, precision, arch=arch)


def device_arch(device):
    """Generated-program architecture of a CUDA device (``"sm_100a"`` / ``"sm_103a"``); the ``arch`` argument
    of :func:`dense_route_available` and :func:`dense_admission`."""
    from .experimental.deepgemm_dense_mqa.dense_mqa import device_arch as _device_arch

    return _device_arch(device)


def dense_admission(arch=None):
    """The 64-head dense family's per-tier, per-architecture admission, host-only, from the shipped catalog.

    ``dense_admission(arch)`` returns ``{"admitted_routes": [...], "withheld_routes": [...], "reason": str | None}``
    for one architecture: ``admitted_routes`` are the ``fp8:h64:*`` route names served there (every query count
    of those tiers), ``withheld_routes`` the tier names the producer measured and did not admit there (the engine
    keeps stock DeepGEMM; ``reason`` names why), disjoint lists; all empty when the catalog has no 64-head
    family.  ``dense_admission()`` returns that record for every catalogued architecture, keyed by arch.  Head
    counts other than 64 are admitted by the ``routes`` table alone (``dense_route_available``).
    """
    from .experimental.deepgemm_dense_mqa.dense_mqa import _catalog, h64_admission

    if arch is None:
        return {name: h64_admission(name) for name in sorted(_catalog()["arches"])}
    return h64_admission(arch)
