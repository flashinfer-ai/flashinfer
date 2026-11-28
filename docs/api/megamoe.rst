.. _apimegamoe:

flashinfer.megamoe
==================

.. currentmodule:: flashinfer.megamoe

SM90 W4A8 expert-parallel MegaMoE with NVFP4 fused/split and MXFP4 fused kernels.
All paths use FP8 E4M3 activations and return BF16 outputs. Execution requires
NVLink-connected Hopper GPUs within each node and a symmetric buffer allocated
collectively before CUDA graph capture. NVFP4 supports four or eight ranks on
one node. MXFP4 supports power-of-two rank counts from one to 32, including
multiple nodes through :class:`MegaMoENcclTransport`. The inter-node path needs
NCCL device API headers and a GIN-capable NCCL installation; set ``NCCL_HOME``
when these are not provided by the ``nvidia-nccl`` wheel. Set ``NCCL_GIN_TYPE=2``
before initializing the process group to select the supported GIN proxy backend.

Buffer and execution
--------------------

.. autosummary::
    :toctree: ../generated

    MegaMoESymmBuffer
    MegaMoENcclTransport
    nvfp4_mega_moe
    mxfp4_mega_moe

Replica plan (MXFP4)
--------------------

Redundant expert slots per rank, the plan tables that route each logical
expert's token blocks onto its copies, and the weight copy of the slots whose
expert changed, either as a separate kernel or inside the MXFP4 fused kernel.
Replica planning and fused weight copies currently require one node and the
default MXFP4 weight storage.

.. autosummary::
    :toctree: ../generated

    MegaMoEExpertSlots

Weight preparation
------------------

.. autosummary::
    :toctree: ../generated

    transform_nvfp4_weights_for_mega_moe
    transform_mxfp4_weights_for_mega_moe
