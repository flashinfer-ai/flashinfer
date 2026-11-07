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

Replica plan (MXFP4)
--------------------

Redundant expert slots per rank, the plan tables that route each logical
expert's token blocks onto its copies, and the weight copy of the slots whose
expert changed, either as a separate kernel or inside the MXFP4 fused kernel.

.. autosummary::
    :toctree: ../generated

    MegaMoEExpertSlots

Weight preparation
------------------

.. autosummary::
    :toctree: ../generated

    transform_nvfp4_weights_for_mega_moe
    transform_mxfp4_weights_for_mega_moe
