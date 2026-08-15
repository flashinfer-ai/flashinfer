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

The checked-in source product contains the optimized Cake route portfolio plus
a complete-domain compatibility component.  A content-addressed manifest pins its source
files, public C ABI, capability matrix, and the FlashInfer revision against
which the matrix was audited.  FlashInfer authenticates every source artifact
before JIT or AOT compilation.  :func:`cake_fmha_manifest` returns a defensive
copy of that product record.

The distributed-context-parallel feature is an additive profile.  It does not
change the conventional entrypoints or their default behavior.

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
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        batch_size,
        backend="cake",
    )
