.. _apicute_dsl:

flashinfer.cute_dsl
===================

CuTe-DSL implementations of selected FlashInfer kernels. These symbols are
available only when the ``nvidia-cutlass-dsl`` package is installed and the
host has a supported NVIDIA GPU; the module guards its imports with
``is_cute_dsl_available()``.

.. note::

    A handful of GEMM symbols (``grouped_gemm_nt_masked``,
    ``Sm100BlockScaledPersistentDenseGemmKernel``,
    ``create_scale_factor_tensor``) used to live in ``flashinfer.cute_dsl`` and
    are still re-exported for backwards compatibility, but their canonical
    home is :doc:`gemm`. New code should import from ``flashinfer.gemm``.

.. currentmodule:: flashinfer.cute_dsl

Availability
------------

.. autosummary::
    :toctree: ../generated

    is_cute_dsl_available

RMSNorm + FP4 Quantization
--------------------------

.. autosummary::
    :toctree: ../generated

    rmsnorm_fp4quant
    add_rmsnorm_fp4quant

.. autoclass:: RMSNormFP4QuantKernel
    :members:

    .. automethod:: __init__

.. autoclass:: AddRMSNormFP4QuantKernel
    :members:

    .. automethod:: __init__

Attention Wrappers
------------------

CuTe-DSL implementations of the batch attention wrappers.

.. currentmodule:: flashinfer.cute_dsl.attention.wrappers.batch_mla

.. autoclass:: BatchMLADecodeCuteDSLWrapper
    :members:

    .. automethod:: __init__

.. currentmodule:: flashinfer.cute_dsl.attention.wrappers.batch_prefill

.. autoclass:: BatchPrefillCuteDSLWrapper
    :members:

    .. automethod:: __init__

.. currentmodule:: flashinfer.cute_dsl.attention.wrappers.batch_decode

.. autoclass:: BatchDecodeCuteDSLWrapper
    :members:

    .. automethod:: __init__

.. autoclass:: BatchDecodePagedCuteDSLWrapper
    :members:

    .. automethod:: __init__

Block Sparse Attention
----------------------

CuTe-DSL block-sparse attention forward kernels.

.. currentmodule:: flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk128

.. autosummary::
    :toctree: ../generated

    bsa_attn_sm100_blk128_fwd

.. currentmodule:: flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64

.. autosummary::
    :toctree: ../generated

    bsa_attn_sm100_blk64_fwd

.. currentmodule:: flashinfer.cute_dsl.sparse.bsa_attn_sm120

.. autosummary::
    :toctree: ../generated

    bsa_attn_sm120_blk64_fwd
    bsa_attn_sm120_blk64_sage_fwd

HCA Decode
----------

.. currentmodule:: flashinfer.cute_dsl.attention.wrappers.batch_hca

.. autosummary::
    :toctree: ../generated

    cute_dsl_hca_decode

The recommended public entry point is
``flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4`` with
``backend="cute-dsl"``. ``cute_dsl_hca_decode`` is the lower-level wrapper for
callers that already use the explicit HCA metadata ABI. The sliding-window
cache is flattened into token rows and selected by an ``[B * Q, 128]`` INT32
``window_indices`` tensor of absolute row indices; ring rotation and wraparound
are supported. The compressed cache remains paged and uses an ``[B * Q,
max_pages]`` INT32 block table. Masked window padding must still contain a
legal row index because gather4 reads every coordinate before masking.
``hca_seq_lens`` describes the backing HCA footprint scheduled by TMA, not the
effective top-k. Each compressed block-table row must therefore contain legal
page IDs for that footprint rounded up to 128-slot tiles, including slots later
masked by a shorter ``sparse_topk_lens``.

Callers whose existing ``sparse_indices`` have a canonical compressed-page
expansion may set
``hca_sparse_indices_format="compressed-page-aligned"`` to generate SWA gather
indices, the compressed block table, and HCA lengths. SWA entries remain
arbitrary absolute rows; only the compressed segment must be the canonical
page expansion. This one-shot compatibility path validates values, allocates
metadata, synchronizes the device, immediately launches the decode, and is not
CUDA Graph capture safe. It is not a hot-loop path.
Latency-sensitive callers must precompute with
``convert_compressed_page_aligned_sparse_indices_to_hca_metadata`` and reuse the
returned metadata through the explicit HCA arguments. Arbitrary TRTLLM-GEN
token-row selections in the compressed segment cannot be represented by an HCA
page table without repacking the compressed KV pool.

Raw-index small-head DSv4 decode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``flashinfer.mla.try_batch_decode_sparse_mla_dsv4_raw`` is a pre-packing
integration helper for B300 ordinary FP8 decode. It accepts logical SWA IDs,
request-local compressed IDs, request IDs, a physical-page table and token
validity. Address translation is fused into the existing attention gather,
removing the separate index-packing launch. H16 supports 1/2/4 query rows;
H32 supports 1/2 rows. Head dimension is 512, SWA width is 128 and compressed
width is 512. This interface does not accept packed NVFP4/HCA cache formats,
prefill, speculative verification or mixed decode/prefill batches.

The caller provides normalized HND FP8 pools, an output tensor and serialized
uint8 scratch. ``get_dsv4_raw_decode_workspace_size(rows, heads)`` returns
scratch bytes without CUDA work. Warm each geometry outside CUDA Graph capture.
The helper returns ``False`` before any GPU work for unsupported metadata or
an unprepared capture. Keep the original packing and attention call on that
branch. Compilation and execution failures propagate.

.. code-block:: python

    from flashinfer.mla import (
        get_dsv4_raw_decode_workspace_size,
        try_batch_decode_sparse_mla_dsv4_raw,
    )

    # Allocate once outside capture, or reuse sufficiently large native scratch.
    scratch = torch.empty(
        get_dsv4_raw_decode_workspace_size(rows, heads),
        dtype=torch.uint8, device=query.device,
    )
    selected = try_batch_decode_sparse_mla_dsv4_raw(
        query=query, swa_kv_cache=swa_pool, compressed_kv_cache=compressed_pool,
        swa_indices=swa_ids, compressed_indices=compressed_ids,
        seq_lens=seq_lens, request_ids=request_ids,
        compressed_block_table=block_table, token_valid=token_valid,
        sinks=sinks, workspace_buffer=scratch, out=output,
        bmm1_scale=qk_scale, bmm2_scale=value_scale, enable_pdl=True,
    )
    if not selected:
        original_pack_and_decode(...)  # Existing framework fallback.

The compressed length is the count of nonnegative IDs, interpreted as a
prefix rather than a compacted list. Interior holes and negative page entries
are masked. Invalid-token rows suppress only compressed attention; SWA validity
still follows its indices and original sequence length. Request IDs and
nonnegative compressed IDs/page entries must remain in bounds even on masked
rows. The API documents these caller-owned value invariants; execution checks
tensor metadata only and performs no device-to-host validation.

This path needs a framework call before index packing. Upgrading FlashInfer
alone preserves the existing automatic packed-index optimization but does not
remove a framework's index-packing kernel. The two paths share the same QK/PV
and merge arithmetic; the raw path changes address resolution and length
calculation. Performance should be compared over packing plus attention, with
matching geometry, cache conditions and downstream consumers.

Fused MoE
---------

.. currentmodule:: flashinfer.fused_moe.cute_dsl

.. autosummary::
    :toctree: ../generated

    cute_dsl_fused_moe_bf16
    cute_dsl_fused_moe_nvfp4
    cute_dsl_fused_moe_mxfp8_mxfp4
    b12x_fused_moe
