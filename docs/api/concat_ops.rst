.. _apiconcat_ops:

flashinfer.concat_ops
=====================

.. currentmodule:: flashinfer.concat_ops

Helpers that concatenate / pack tensors for MLA-style attention layouts.
``flashinfer.dsv3_ops`` re-exports :func:`concat_mla_k` for DeepSeek-V3 call
sites.

.. autosummary::
    :toctree: ../generated

    concat_mla_k

flashinfer.mla_kv_pack
======================

.. currentmodule:: flashinfer.mla_kv_pack

Fused MLA context K/V pack with fp8 e4m3 quantization (``key = [k_nope ‖ k_pe]``,
``value = v``) for ragged MLA prefill kernels that take separate fp8 Q/K/V.

.. autosummary::
    :toctree: ../generated

    concat_mla_kv_quant_fp8

The fused kernel (a generated Cake program per head group, built for the
exact compute capabilities 10.0 and 10.3) serves bf16 inputs with head
geometry nope 128 | rope 64 | v 128 within the package-data allowlist
``flashinfer/mla_kv_pack_fp8_workloads.json``; every other call takes a
composable torch path with the same saturating e4m3 numerics
(``torch >= 2.13`` ``Tensor.to(float8_e4m3fn)``). Setting
``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` (read at call time) forces the
composable path; ``mla_kv_pack._concat_mla_kv_quant_fp8_stats()`` reports
the dispatch counters.
