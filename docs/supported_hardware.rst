.. _supported_hardware:

Supported Hardware
==================

FlashInfer leverages NVIDIA Tensor Core instructions that require specific GPU
architectures. This page provides a high-level, manually maintained summary of
the minimum compute capability required for each quantization data type
supported by FlashInfer.

.. tip::

   Run ``flashinfer show-config`` to check your GPU's compute capability.

Quantization Data Types
-----------------------

The table below shows the minimum compute capability at which *any* backend
supports the given data type. Backend coverage can still vary by operation
(``mm`` vs ``bmm``), scale layout, and CUDA/cuDNN version; consult the
per-function API documentation for exact requirements.

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
      - Support is backend-specific. MXFP8 support starts at sm_100 for GEMM on
        Blackwell, but BMM uses cuDNN on sm_100 / sm_103 and CUTLASS on sm_120 /
        sm_121. See :doc:`/api/gemm` for per-operation backend details.
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
   documentation (e.g., :doc:`/api/fp4_quantization` and :doc:`/api/gemm`) is
   the authoritative source for each operation's hardware requirements.

   Compute capability is not the only requirement. FlashInfer currently
   supports CUDA 12.6, 12.8, 13.0, and 13.1 (see :doc:`/installation`), and
   Blackwell FP4 / MXFP4 features require CUDA 12.8+.

   For the official NVIDIA hardware specifications, see the
   `PTX ISA Reference <https://docs.nvidia.com/cuda/parallel-thread-execution/>`_
   and the
   `CUDA Compute Capability list <https://developer.nvidia.com/cuda-gpus>`_.
