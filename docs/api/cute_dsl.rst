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

.. currentmodule:: flashinfer.cute_dsl.sparse

.. autosummary::
    :toctree: ../generated

    bsa_attn_fwd
    bsa_attn_blk64_fwd

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

Callers whose existing ``sparse_indices`` are a canonical page-aligned HCA
expansion may set ``hca_sparse_indices_format="page-aligned"`` to generate SWA
gather indices, the compressed block table, and HCA lengths. Active SWA entries
may be arbitrary absolute rows; only the compressed segment must be a canonical
page expansion. This one-shot compatibility path validates values, allocates
metadata, synchronizes the device, immediately launches the decode, and is not
CUDA Graph capture safe. It is not a hot-loop path.
Latency-sensitive callers must precompute with
``convert_page_aligned_sparse_indices_to_hca_metadata`` and reuse the returned
metadata through the explicit HCA arguments. Arbitrary TRTLLM-GEN token-row
selections in the compressed segment cannot be represented by an HCA page table
without repacking the compressed KV pool.
