.. _apitesting:

flashinfer.testing
==================

.. currentmodule:: flashinfer.testing

This module provides comprehensive testing utilities for benchmarking, performance analysis in FlashInfer.

Test Environment Setup
----------------------

.. autosummary::
    :toctree: ../generated

    set_seed
    sleep_after_kernel_run

Performance Analysis
--------------------

FLOPS Calculation
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    attention_flops
    attention_flops_with_actual_seq_lens
    attention_tflops_per_sec
    attention_tflops_per_sec_with_actual_seq_lens

Throughput Analysis
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    attention_tb_per_sec
    attention_tb_per_sec_with_actual_seq_lens

GPU Benchmarking
----------------

.. autosummary::
    :toctree: ../generated

    bench_gpu_time
    bench_gpu_time_with_cuda_event
    bench_gpu_time_with_cudagraph
    bench_gpu_time_with_cupti

FP8 Quantization Helpers
------------------------

Reference quantize/dequantize helpers used by FP8 tests and benchmarks. Import
from the ``flashinfer.testing.fp8`` submodule:

.. code-block:: python

    from flashinfer.testing.fp8 import quantize_fp8, dequantize_fp8
    from flashinfer.testing.fp8 import per_token_cast_to_fp8, per_block_cast_to_fp8

.. currentmodule:: flashinfer.testing.fp8

.. autosummary::
    :toctree: ../generated

    per_token_cast_to_fp8
    per_block_cast_to_fp8
    quantize_fp8
    dequantize_fp8
