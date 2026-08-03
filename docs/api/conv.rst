.. _apiconv:

flashinfer.conv
===============

.. currentmodule:: flashinfer.conv

This module provides 3D convolution operations.

NVFP4 Conv3d (SM120)
--------------------

The SM120 NVFP4 path uses BF16 input and output tensors with NVFP4 weights
and dynamically quantized NVFP4 activations. Prepare weights once when loading
the model, then reuse the packed weights and scales for inference.

The current implementation requires an SM120 GPU and CUDA 13 or newer. It
supports batch size one, 3x3x3 filters, input and output channel counts that
are multiples of 128, unit stride and dilation, one group, and padding
``(0, 0, 0)`` or ``(0, 1, 1)``.

The activation quantizer is included in SM120 AOT builds. The CuTe DSL Conv3d
kernel is compiled for each shape on first use and persisted in FlashInfer's
kernel cache for subsequent processes.

.. autosummary::
    :toctree: ../generated

    prepare_nvfp4_conv3d_weight
    conv3d_nvfp4
