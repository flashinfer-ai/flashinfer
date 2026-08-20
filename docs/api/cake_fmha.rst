.. _apicake_fmha:

flashinfer.cake_fmha
====================

``cake_fmha`` is the versioned Cake implementation of FlashInfer's conventional
TensorRT-LLM paged FMHA decode and context contracts.  It is an explicit
Blackwell backend: importing FlashInfer or calling the existing APIs without a
backend continues to select the existing FlashInfer implementation.

The public functions support B200/GB200 (SM100a, CUDA 12.8+) and B300/GB300
(SM103a, CUDA 12.9+).  They accept the same arguments and return the same values
as :func:`flashinfer.decode.trtllm_batch_decode_with_kv_cache` and
:func:`flashinfer.prefill.trtllm_batch_context_with_kv_cache`.

The checked-in source product contains the optimized Cake route portfolio, a
complete-domain compatibility component, and the DCP speculative-decode
add-on.  One content-addressed manifest pins all source files, public C ABIs,
the base capability matrix, and the FlashInfer revision against which the
matrix was audited.  FlashInfer authenticates every standalone artifact before
JIT or AOT compilation.  :func:`cake_fmha_manifest` returns a defensive copy
of that product record.

All 1,798 optimized cells have authenticated high-level adapters for their
complete component chains.  The selector accepts the pinned matrix's normalized
NHD context views and device scalar FP8/NVFP4 scales.  Its numerically inert
``1e-30`` skip-softmax probe is canonicalized to ordinary softmax only after an
exact optimized match; other nonzero thresholds remain compatibility routes.
Selector misses, insufficient route workspace, and NVFP4 adapter load failures
fail closed to ``compat_v1``.

Optimized routes are fail-closed.  In particular, optimized FP8 decode is
qualified for HND pages, a shared K/V page table, and GQA group size eight;
other valid FP8 decode shapes remain Cake-owned and use the authenticated
complete-domain component.

The manifest's per-route counts are inventory metadata, not a proof that a
particular high-level selector revision reproduces the pinned matrix.  The
checked-in allocation-free replay therefore enumerates the independent pinned
capability corpus (80,768 raw cells, 57,280 valid cells), calls the actual
high-level selectors, and authenticates every case/route pair with canonical
SHA-256 ``d47bf01c2d27409c6a39759d02e30bb9df65e98c353f53d7335081dd26b3f3a8``.
It requires exactly 1,798 optimized cells and 55,482 ``compat_v1`` cells, with
the same per-route accounting as the manifest.  Both a fresh source checkout
and the installed wheel must execute this replay and then exercise selected
routes on SM100a and SM103a; representative family tests and manifest-count
accounting do not replace those gates.

The distributed-context-parallel feature remains additive.  Supplying
``causal_seqlens_kv_global`` to :func:`cake_batch_decode_with_kv_cache` selects
the authenticated ``cake_fmha_dcp_spec`` profile; ordinary calls continue to
select conventional FMHA.  The DCP JIT cache key includes the same root
manifest digest plus an authenticated FlashInfer-adapter digest and uses exact
SM100a or SM103a targets, so the add-on cannot silently drift from the base
package.

.. currentmodule:: flashinfer.cake_fmha

.. autosummary::
    :toctree: ../generated

    cake_batch_decode_with_kv_cache
    cake_batch_context_with_kv_cache
    cake_fmha_manifest
    get_cake_fmha_module

The same implementation can also be selected on the existing APIs with
``backend="cake"``::

    output = flashinfer.trtllm_batch_decode_with_kv_cache(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_kv_len,
        backend="cake",
    )
