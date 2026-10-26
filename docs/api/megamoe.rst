.. _apimegamoe:

flashinfer.megamoe
==================

.. currentmodule:: flashinfer.megamoe

SM90 W4A8 expert-parallel MegaMoE with NVFP4 fused/split and MXFP4 fused kernels.
All paths use FP8 E4M3 activations and return BF16 outputs. Execution requires
NVLink-connected Hopper GPUs and a symmetric buffer allocated collectively
before CUDA graph capture. NVFP4 supports four or eight ranks; MXFP4 supports
power-of-two rank counts from one to 32.

Buffer and execution
--------------------

.. autosummary::
    :toctree: ../generated

    MegaMoESymmBuffer
    nvfp4_mega_moe
    mxfp4_mega_moe

Weight preparation
------------------

.. autosummary::
    :toctree: ../generated

    transform_nvfp4_weights_for_mega_moe
    transform_mxfp4_weights_for_mega_moe
