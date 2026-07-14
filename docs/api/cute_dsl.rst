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

.. currentmodule:: flashinfer.cute_dsl.attention.wrappers.batch_hca

.. autosummary::
    :toctree: ../generated

    cute_dsl_hca_decode

The recommended public entry point is
``flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4`` with
``backend="cute-dsl"``. ``cute_dsl_hca_decode`` is the lower-level wrapper for
callers that already use the HCA page-table metadata ABI.

Callers whose existing ``sparse_indices`` are a canonical page-aligned HCA
expansion may set ``hca_sparse_indices_format="page-aligned"`` to generate the
block tables and HCA lengths. This one-shot compatibility path validates
values, allocates metadata, synchronizes the device, immediately launches the
decode, and is not CUDA Graph capture safe. It is not a hot-loop path.
Latency-sensitive callers must precompute with
``convert_page_aligned_sparse_indices_to_hca_metadata`` and reuse the returned
metadata through the explicit HCA arguments. Arbitrary TRTLLM-GEN token-row
selections cannot be represented by HCA page tables without repacking the KV
pools.
