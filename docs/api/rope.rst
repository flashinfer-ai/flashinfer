.. _apirope:

flashinfer.rope
===============

Kernels for applying rotary embeddings.

.. currentmodule:: flashinfer.rope

.. autosummary::
    :toctree: ../generated

    apply_rope_inplace
    apply_llama31_rope_inplace
    apply_rope
    apply_llama31_rope
    apply_rope_pos_ids
    apply_rope_pos_ids_inplace
    apply_llama31_rope_pos_ids
    apply_llama31_rope_pos_ids_inplace
    apply_rope_with_cos_sin_cache
    apply_rope_with_cos_sin_cache_inplace
    apply_mrope_with_cos_sin_cache_inplace
    rope_quantize_fp8
    rope_quantize_fp8_append_paged_kv_cache
    mla_rope_quantize_fp8

Cached multimodal positions
--------------------------

``apply_mrope_with_cos_sin_cache_inplace`` consumes temporal, height and width
positions directly as ``[3, nnz]``. It accepts flattened Q/K views into packed
QKV rows, avoiding a separate per-token cache-gather buffer. For example,
Qwen3-VL-style interleaving with rotary dimension 128 uses section counts
``(24, 20, 20)``; the flag changes frequency-axis selection, not NeoX pairing.
Use the model's own section metadata, positions and cos/sin cache in an adapter.

The standalone example ``examples/pytorch/mrope.py`` constructs a small video
patch grid, preserves V in shared QKV storage, and checks the result against
explicit three-axis cache selection::

    python examples/pytorch/mrope.py

FP16/BF16 Q/K and either matching-dtype or FP32 caches are supported. Cache dtype
controls arithmetic rounding; results need not be bitwise equal to ordinary
RoPE applied to a materialized FP32 composite cache. The operation applies no
normalization, scaling policy or KV-cache append. See the API docstring for
shape, stride, aliasing and position-value preconditions.
