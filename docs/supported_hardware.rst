.. _supported_hardware:

Supported Hardware
==================

FlashInfer leverages NVIDIA Tensor Core instructions that require specific GPU
architectures. This page summarizes the minimum compute capability required for
each quantization data type supported by FlashInfer.

.. tip::

   Run ``flashinfer show-config`` to check your GPU's compute capability.

Quantization Data Types
-----------------------

The table below shows the minimum compute capability at which *any* backend
supports the given data type. Some backends may require a higher capability;
consult the per-function API documentation for exact requirements.

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Data Type
     - Min Compute Capability
     - GPU Architecture
     - Notes
   * - FP8 (e4m3 / e5m2)
     - sm_89 (8.9)
     - Ada / Hopper+
     - cuDNN and cuBLAS backends. CUTLASS requires sm_100+.
   * - MXFP8
     - sm_100 (10.0)
     - Blackwell+
     -
   * - NVFP4 KV dequantize
     - sm_80 (8.0)
     - Ampere+
     -
   * - NVFP4 / MXFP4 (quantize & GEMM)
     - sm_100 (10.0)
     - Blackwell+
     -

.. note::

   This table reflects FlashInfer's current support. The per-function API
   documentation (e.g., :doc:`/api/fp4_quantization`) is the authoritative
   source for each operation's hardware requirements.

   For the official NVIDIA hardware specifications, see the
   `PTX ISA Reference <https://docs.nvidia.com/cuda/parallel-thread-execution/>`_
   and the
   `CUDA Compute Capability list <https://developer.nvidia.com/cuda-gpus>`_.
