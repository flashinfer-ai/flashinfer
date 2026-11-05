.. _apimegamoe:

flashinfer.megamoe
==================

.. currentmodule:: flashinfer.megamoe

W4A8 expert-parallel MegaMoE: SM90 NVFP4 fused/split and MXFP4 fused kernels,
and an SM120 MXFP4 fused kernel. All paths use FP8 E4M3 activations and return
BF16 outputs, with a symmetric buffer allocated collectively before CUDA graph
capture. On SM90 execution requires NVLink-connected Hopper GPUs; NVFP4 supports
four or eight ranks and MXFP4 power-of-two rank counts from one to 32.

On SM120 (RTX PRO 6000 class) ``mxfp4_mega_moe`` runs one host's ranks (one to
eight) over PCIe peer-to-peer, for decode-size batches. The kernel requantises the
staged activations to MXFP8 (a UE8M0 scale per 32 channels), multiplies them with
the MXFP4 weights on the block-scaled tensor-core MMA, quantises the clamped SwiGLU
output the same way and applies the routing weight in the top-k combine.
``transform_mxfp4_weights_for_mega_moe`` produces the SM120 layout when its inputs
are on an SM120 device. ``intermediate_hidden`` must be a multiple of 256 and
``hidden`` at most 7936, and ``num_max_tokens_per_rank`` at most 256: above
that, use the ``flashinfer.moe_ep`` SM120 MXFP4 x MXFP8 backends.

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
