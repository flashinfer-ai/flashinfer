.. _apifp4_quantization:

flashinfer.fp4_quantization
===========================

.. note::

    Starting in FlashInfer 0.6.12, the canonical home for FP4 quantization
    APIs is :ref:`apiquantization`.

    ``flashinfer.fp4_quantization`` remains as a backwards-compatibility
    shim that re-exports the same symbols, so existing code such as
    ``from flashinfer.fp4_quantization import fp4_quantize`` keeps
    working. New code should import from
    ``flashinfer.quantization.fp4_quantization`` (or its canonical
    re-export at ``flashinfer.quantization``).

This page intentionally does not re-document the FP4 symbols, because
each symbol is the same Python object as the one rendered on
:ref:`apiquantization` — duplicating the autosummary entries here would
make Sphinx emit "duplicate object description" warnings under
``sphinx -W``.

See Also
--------

* :ref:`apiquantization` — canonical FP4 / FP8 / packbits API reference,
  including all of the following symbols that ``flashinfer.fp4_quantization``
  used to host:

  - :func:`flashinfer.quantization.fp4_quantize`
  - :func:`flashinfer.quantization.nvfp4_quantize`
  - :func:`flashinfer.quantization.nvfp4_batched_quantize`
  - :func:`flashinfer.quantization.block_scale_interleave`
    (alias: ``nvfp4_block_scale_interleave``)
  - :func:`flashinfer.quantization.e2m1_and_ufp8sf_scale_to_float`
  - :func:`flashinfer.quantization.scaled_fp4_grouped_quantize`
  - :func:`flashinfer.quantization.silu_and_mul_nvfp4_quantize`
  - :func:`flashinfer.quantization.shuffle_matrix_a`
  - :func:`flashinfer.quantization.shuffle_matrix_sf_a`
  - :func:`flashinfer.quantization.nvfp4_kv_quantize`
  - :func:`flashinfer.quantization.nvfp4_kv_dequantize`
  - :func:`flashinfer.quantization.nvfp4_quantize_paged_kv_cache`
  - :class:`flashinfer.quantization.SfLayout`
