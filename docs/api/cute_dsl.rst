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
