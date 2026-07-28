"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Tests for flashinfer.fi_trace: definition JSON generation."""

import json
from contextlib import suppress
from pathlib import Path

import pytest
import torch

from flashinfer.fi_trace import fi_trace


# ---------------------------------------------------------------------------
# Helper: validate common fields of a definition dict
# ---------------------------------------------------------------------------


def _check_defn(defn, op_type, fi_api_substr):
    assert isinstance(defn, dict), "fi_trace must return a dict"
    assert defn["op_type"] == op_type, f"op_type mismatch: {defn['op_type']!r}"
    assert "name" in defn and isinstance(defn["name"], str) and defn["name"]
    assert "axes" in defn and isinstance(defn["axes"], dict)
    assert "inputs" in defn and isinstance(defn["inputs"], dict)
    assert "outputs" in defn and isinstance(defn["outputs"], dict)
    assert any(fi_api_substr in t for t in defn["tags"]), (
        f"Expected fi_api tag containing {fi_api_substr!r}, got {defn['tags']}"
    )
    # Must be round-trippable through JSON
    json.dumps(defn)


def test_trace_default_check():
    from flashinfer.trace import default_check, default_tolerances, standard_check

    assert default_tolerances(torch.bfloat16) == (1e-2, 1e-2)

    ref = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    actual = ref + 1e-6
    assert default_check([ref], [actual])
    assert default_check({"out": ref}, {"out": actual})

    opposite = -ref
    assert not default_check(
        [ref],
        [opposite],
        rtol=10.0,
        atol=10.0,
        max_mismatch_pct=100.0,
        min_cos_sim=0.99,
    )

    ref_int = torch.tensor([1, 2, 3], dtype=torch.int32)
    actual_int = torch.tensor([1, 2, 0], dtype=torch.int32)
    assert default_check([ref_int], [actual_int], max_mismatch_pct=34.0)
    assert standard_check([ref], [actual])


def test_all_registered_trace_templates_have_check():
    from flashinfer.api_logging import _TRACE_REGISTRY

    import flashinfer.activation  # noqa: F401
    import flashinfer.cascade  # noqa: F401
    import flashinfer.decode  # noqa: F401
    import flashinfer.fused_moe  # noqa: F401
    import flashinfer.gdn_decode  # noqa: F401
    import flashinfer.gdn_prefill  # noqa: F401
    import flashinfer.gemm  # noqa: F401
    import flashinfer.norm  # noqa: F401
    import flashinfer.page  # noqa: F401
    import flashinfer.prefill  # noqa: F401
    import flashinfer.quantization  # noqa: F401
    import flashinfer.rope  # noqa: F401
    import flashinfer.sampling  # noqa: F401

    with suppress(Exception):
        import flashinfer.cudnn  # noqa: F401

    missing = [
        getattr(template, "name_prefix", None) or template.op_type
        for _, template, _ in _TRACE_REGISTRY
        if template.check is None
    ]
    assert not missing


def test_norm_trace_check_tolerances_match_unit_tests():
    from flashinfer.trace.templates.norm import (
        fused_add_rmsnorm_quant_trace,
        layernorm_trace,
        rmsnorm_quant_trace,
        rmsnorm_trace,
    )

    ref = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    assert rmsnorm_trace.check([ref], [ref + 5e-4])
    assert not rmsnorm_trace.check([ref], [ref + 5e-3])

    assert layernorm_trace.check([ref], [ref + 5e-3])
    assert not layernorm_trace.check([ref], [ref + 5e-2])

    assert rmsnorm_quant_trace.check([ref], [ref + 0.5])
    assert not rmsnorm_quant_trace.check([ref], [ref + 4.0])

    residual = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    assert fused_add_rmsnorm_quant_trace.check(
        [ref, residual],
        [ref + 0.5, residual + 5e-4],
    )
    assert not fused_add_rmsnorm_quant_trace.check(
        [ref, residual],
        [ref + 0.5, residual + 5e-3],
    )


def test_gemm_trace_check_tolerances_match_unit_tests():
    from flashinfer.trace.templates.gemm import (
        bmm_mxfp8_trace,
        mm_bf16_trace,
        mm_fp4_trace,
        mm_mxfp8_trace,
    )

    ref = torch.tensor([1.0, 0.0], dtype=torch.float32)
    assert mm_bf16_trace.check([ref], [torch.tensor([1.0, 0.05])])
    assert not mm_bf16_trace.check([ref], [torch.tensor([0.0, 1.0])])

    assert mm_mxfp8_trace.check([ref], [torch.tensor([1.0, 0.6])])
    assert not mm_mxfp8_trace.check([ref], [torch.tensor([1.0, 0.7])])

    assert mm_fp4_trace.check([ref], [torch.tensor([1.0, 0.2])])
    assert not mm_fp4_trace.check([ref], [torch.tensor([1.0, 0.3])])

    assert bmm_mxfp8_trace.check([ref], [torch.tensor([1.0, 0.45])])
    assert not bmm_mxfp8_trace.check([ref], [torch.tensor([1.0, 0.6])])


def test_attention_trace_check_tolerances_match_unit_tests():
    from flashinfer.trace.templates.attention import (
        single_decode_with_kv_cache_trace,
        single_prefill_with_kv_cache_trace,
    )

    ref = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    assert single_decode_with_kv_cache_trace.check([ref], [ref + 5e-4])
    assert not single_decode_with_kv_cache_trace.check([ref], [ref + 5e-3])

    assert single_prefill_with_kv_cache_trace.check([ref], [ref + 5e-4])
    assert not single_prefill_with_kv_cache_trace.check([ref], [ref + 5e-3])


def test_recurrent_kda_fi_trace():
    import flashinfer.kda_decode

    batch_size, num_q_heads, num_v_heads, head_dim = 4, 8, 16, 128
    q = torch.empty(batch_size, 1, num_q_heads, head_dim, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty(batch_size, 1, num_v_heads, head_dim, dtype=torch.bfloat16)
    g = torch.empty_like(v)
    beta = torch.empty(batch_size, 1, num_v_heads, dtype=torch.bfloat16)
    state = torch.empty(
        batch_size, num_v_heads, head_dim, head_dim, dtype=torch.bfloat16
    )
    source = torch.empty(
        batch_size + 2, num_v_heads, head_dim, head_dim, dtype=torch.bfloat16
    )
    source_indices = torch.arange(batch_size, dtype=torch.int32)

    defn = flashinfer.kda_decode.recurrent_kda.fi_trace(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        initial_state_source=source,
        initial_state_indices=source_indices,
        beta_is_logit=True,
    )

    _check_defn(defn, "kda", "flashinfer.kda_decode.recurrent_kda")
    assert defn["inputs"]["initial_state_source"]["shape"] == [
        "source_pool_size",
        "num_v_heads",
        "head_dim",
        "head_dim",
    ]
    assert defn["inputs"]["initial_state_indices"]["shape"] == ["num_sequences"]
    assert defn["axes"]["head_dim"]["value"] == head_dim


# ---------------------------------------------------------------------------
# rmsnorm
# ---------------------------------------------------------------------------


def test_rmsnorm_fi_trace():
    import flashinfer.norm

    hidden = torch.randn(32, 4096, dtype=torch.bfloat16)
    weight = torch.ones(4096, dtype=torch.bfloat16)

    # Access via the function attribute
    defn = flashinfer.norm.rmsnorm.fi_trace(input=hidden, weight=weight)
    _check_defn(defn, "rmsnorm", "flashinfer.norm.rmsnorm")

    axes = defn["axes"]
    assert axes["batch_size"]["type"] == "var"
    assert axes["hidden_size"]["type"] == "const"
    assert axes["hidden_size"]["value"] == 4096

    assert defn["inputs"]["hidden_states"]["shape"] == ["batch_size", "hidden_size"]
    assert defn["inputs"]["weight"]["shape"] == ["hidden_size"]
    assert defn["outputs"]["output"]["shape"] == ["batch_size", "hidden_size"]
    assert defn["outputs"]["output"]["dtype"] == "bfloat16"
    assert "check" in defn
    assert "def _norm_check" in defn["check"]


def test_rmsnorm_fi_trace_via_helper():
    import flashinfer.norm

    hidden = torch.randn(16, 7168, dtype=torch.bfloat16)
    weight = torch.ones(7168, dtype=torch.bfloat16)

    defn = fi_trace(flashinfer.norm.rmsnorm, input=hidden, weight=weight)
    _check_defn(defn, "rmsnorm", "flashinfer.norm.rmsnorm")
    assert defn["axes"]["hidden_size"]["value"] == 7168


def test_fused_add_rmsnorm_fi_trace():
    import flashinfer.norm

    x = torch.randn(8, 5120, dtype=torch.bfloat16)
    res = torch.randn(8, 5120, dtype=torch.bfloat16)
    weight = torch.ones(5120, dtype=torch.bfloat16)

    defn = flashinfer.norm.fused_add_rmsnorm.fi_trace(
        input=x, residual=res, weight=weight
    )
    _check_defn(defn, "rmsnorm", "flashinfer.norm.fused_add_rmsnorm")
    assert defn["axes"]["hidden_size"]["value"] == 5120
    assert "residual" in defn["inputs"]
    assert "residual" in defn["outputs"]


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def test_top_k_sampling_fi_trace():
    import flashinfer.sampling

    probs = torch.rand(64, 128256, dtype=torch.float32)
    top_k = torch.full((64,), 50, dtype=torch.int32)

    defn = flashinfer.sampling.top_k_sampling_from_probs.fi_trace(
        probs=probs, top_k=top_k
    )
    _check_defn(defn, "sampling", "top_k_sampling_from_probs")
    assert defn["axes"]["vocab_size"]["value"] == 128256
    assert defn["inputs"]["probs"]["shape"] == ["batch_size", "vocab_size"]
    assert defn["outputs"]["samples"]["dtype"] == "int64"


def test_top_p_sampling_fi_trace():
    import flashinfer.sampling

    probs = torch.rand(32, 151936, dtype=torch.float32)
    top_p = torch.full((32,), 0.9, dtype=torch.float32)

    defn = flashinfer.sampling.top_p_sampling_from_probs.fi_trace(
        probs=probs, top_p=top_p
    )
    _check_defn(defn, "sampling", "top_p_sampling_from_probs")
    assert defn["axes"]["vocab_size"]["value"] == 151936


def test_top_k_top_p_sampling_fi_trace():
    import flashinfer.sampling

    probs = torch.rand(16, 129280, dtype=torch.float32)
    top_k = torch.full((16,), 100, dtype=torch.int32)
    top_p = torch.full((16,), 0.9, dtype=torch.float32)

    defn = flashinfer.sampling.top_k_top_p_sampling_from_probs.fi_trace(
        probs=probs, top_k=top_k, top_p=top_p
    )
    _check_defn(defn, "sampling", "top_k_top_p_sampling_from_probs")
    assert defn["axes"]["vocab_size"]["value"] == 129280
    assert "top_k" in defn["inputs"]
    assert "top_p" in defn["inputs"]


# ---------------------------------------------------------------------------
# gemm
# ---------------------------------------------------------------------------


def test_mm_bf16_fi_trace():
    import flashinfer.gemm

    a = torch.randn(128, 4096, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, dtype=torch.bfloat16)

    defn = flashinfer.gemm.mm_bf16.fi_trace(a=a, b=b)
    _check_defn(defn, "gemm_bf16", "mm_bf16")
    assert defn["axes"]["N"]["value"] == 4096
    assert defn["axes"]["K"]["value"] == 4096
    assert defn["axes"]["M"]["type"] == "var"
    assert defn["inputs"]["A"]["shape"] == ["M", "K"]
    assert defn["inputs"]["B"]["shape"] == ["K", "N"]
    assert defn["outputs"]["C"]["shape"] == ["M", "N"]


# ---------------------------------------------------------------------------
# quantization
# ---------------------------------------------------------------------------


def test_nvfp4_kv_dequantize_paged_fi_trace():
    import flashinfer

    batch_size = 2
    max_seq_len = 7
    num_pages = 8
    page_size = 4
    num_heads = 2
    k_head_dim = 64
    v_head_dim = 128

    for kv_layout in ("NHD", "HND"):
        if kv_layout == "NHD":
            k_cache_shape = (num_pages, page_size, num_heads, k_head_dim // 2)
            v_cache_shape = (num_pages, page_size, num_heads, v_head_dim // 2)
            k_scale_shape = (num_pages, page_size, num_heads, k_head_dim // 16)
            v_scale_shape = (num_pages, page_size, num_heads, v_head_dim // 16)
            expected_cache_shape = [
                "num_pages",
                "page_size",
                "num_heads",
                "k_packed_dim",
            ]
            expected_name = "nvfp4_kv_dequantize_paged"
            expected_init_name = "_nvfp4_kv_dequantize_paged_nhd_init"
        else:
            k_cache_shape = (num_pages, num_heads, page_size, k_head_dim // 2)
            v_cache_shape = (num_pages, num_heads, page_size, v_head_dim // 2)
            k_scale_shape = (num_pages, num_heads, page_size, k_head_dim // 16)
            v_scale_shape = (num_pages, num_heads, page_size, v_head_dim // 16)
            expected_cache_shape = [
                "num_pages",
                "num_heads",
                "page_size",
                "k_packed_dim",
            ]
            expected_name = "nvfp4_kv_dequantize_paged_hnd"
            expected_init_name = "_nvfp4_kv_dequantize_paged_hnd_init"

        k_cache = torch.empty(k_cache_shape, dtype=torch.uint8)
        v_cache = torch.empty(v_cache_shape, dtype=torch.uint8)
        k_scales = torch.empty(k_scale_shape, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        v_scales = torch.empty(v_scale_shape, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        block_tables = torch.zeros(batch_size, 2, dtype=torch.int32)
        seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32)
        k_scale = torch.ones(1, dtype=torch.float32)
        v_scale = torch.ones(1, dtype=torch.float32)
        output_k = torch.empty(
            batch_size, max_seq_len, num_heads, k_head_dim, dtype=torch.bfloat16
        )
        output_v = torch.empty(
            batch_size, max_seq_len, num_heads, v_head_dim, dtype=torch.bfloat16
        )

        defn = flashinfer.nvfp4_kv_dequantize_paged.fi_trace(
            paged_kv_cache=(k_cache, v_cache),
            kv_cache_sf=(k_scales, v_scales),
            block_tables=block_tables,
            seq_lens=seq_lens,
            k_scale=k_scale,
            v_scale=v_scale,
            output_k=output_k,
            output_v=output_v,
            kv_layout=kv_layout,
        )
        _check_defn(defn, "dequantize_fp4", "nvfp4_kv_dequantize_paged")
        assert defn["name"].startswith(expected_name)
        axes = defn["axes"]
        assert axes["num_heads"]["value"] == num_heads
        assert axes["k_head_dim"]["value"] == k_head_dim
        assert axes["v_head_dim"]["value"] == v_head_dim
        assert axes["page_size"]["value"] == page_size
        assert axes["batch_size"]["type"] == "var"

        assert defn["inputs"]["paged_k_cache"]["shape"] == expected_cache_shape
        assert defn["outputs"]["output_k"]["dtype"] == "bfloat16"
        assert "block_table_stride * page_size >= max_seq_len" in defn["constraints"]
        assert "init" in defn

        init_namespace = {}
        exec(defn["init"], init_namespace)
        dumped_init = init_namespace[expected_init_name]
        dumped_init_inputs = dumped_init(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            k_head_dim=k_head_dim,
            v_head_dim=v_head_dim,
            num_pages=num_pages,
            page_size=page_size,
            device="cpu",
        )
        dumped_init_k_cache, dumped_init_v_cache = dumped_init_inputs["paged_kv_cache"]
        dumped_init_k_scales, dumped_init_v_scales = dumped_init_inputs["kv_cache_sf"]
        assert dumped_init_inputs["kv_layout"] == kv_layout
        assert dumped_init_k_cache.shape == k_cache_shape
        assert dumped_init_v_cache.shape == v_cache_shape
        assert dumped_init_k_scales.shape == k_scale_shape
        assert dumped_init_v_scales.shape == v_scale_shape

        init_inputs = flashinfer.nvfp4_kv_dequantize_paged.fi_init(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            k_head_dim=k_head_dim,
            v_head_dim=v_head_dim,
            num_pages=num_pages,
            page_size=page_size,
            kv_layout=kv_layout,
            device="cpu",
        )
        init_k_cache, init_v_cache = init_inputs["paged_kv_cache"]
        init_k_scales, init_v_scales = init_inputs["kv_cache_sf"]
        assert init_inputs["kv_layout"] == kv_layout
        assert init_k_cache.shape == k_cache_shape
        assert init_v_cache.shape == v_cache_shape
        assert init_k_scales.shape == k_scale_shape
        assert init_v_scales.shape == v_scale_shape


# ---------------------------------------------------------------------------
# GQA paged decode
# ---------------------------------------------------------------------------


def test_gqa_paged_decode_fi_trace():
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

    batch_size = 32
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    num_pages = 512
    page_size = 16

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.bfloat16)
    k_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16
    )
    v_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16
    )

    defn = BatchDecodeWithPagedKVCacheWrapper.run.fi_trace(
        q=q, paged_kv_cache=(k_cache, v_cache)
    )
    _check_defn(defn, "gqa_paged", "BatchDecodeWithPagedKVCacheWrapper")
    axes = defn["axes"]
    assert axes["num_qo_heads"]["value"] == num_qo_heads
    assert axes["num_kv_heads"]["value"] == num_kv_heads
    assert axes["head_dim"]["value"] == head_dim
    assert axes["page_size"]["value"] == page_size
    assert axes["batch_size"]["type"] == "var"
    assert axes["num_pages"]["type"] == "var"

    assert "k_cache" in defn["inputs"]
    assert "v_cache" in defn["inputs"]
    assert defn["inputs"]["k_cache"]["shape"] == [
        "num_pages",
        "page_size",
        "num_kv_heads",
        "head_dim",
    ]


# ---------------------------------------------------------------------------
# GQA ragged prefill
# ---------------------------------------------------------------------------


def test_gqa_ragged_prefill_fi_trace():
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

    total_q = 256
    total_kv = 512
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128

    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(total_kv, num_kv_heads, head_dim, dtype=torch.bfloat16)
    v = torch.randn(total_kv, num_kv_heads, head_dim, dtype=torch.bfloat16)

    defn = BatchPrefillWithRaggedKVCacheWrapper.run.fi_trace(q=q, k=k, v=v)
    _check_defn(defn, "gqa_ragged", "BatchPrefillWithRaggedKVCacheWrapper")
    axes = defn["axes"]
    assert axes["num_qo_heads"]["value"] == num_qo_heads
    assert axes["num_kv_heads"]["value"] == num_kv_heads
    assert axes["head_dim"]["value"] == head_dim
    assert axes["total_q"]["type"] == "var"
    assert axes["total_kv"]["type"] == "var"

    assert "constraints" in defn


# ---------------------------------------------------------------------------
# MLA paged
# ---------------------------------------------------------------------------


def test_mla_paged_fi_trace():
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    batch_size = 16
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_pages = 256
    page_size = 64

    q_nope = torch.randn(batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16)
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16)
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16)

    defn = BatchMLAPagedAttentionWrapper.run.fi_trace(
        q_nope=q_nope, q_pe=q_pe, ckv_cache=ckv_cache, kpe_cache=kpe_cache
    )
    _check_defn(defn, "mla_paged", "BatchMLAPagedAttentionWrapper")
    axes = defn["axes"]
    assert axes["num_qo_heads"]["value"] == num_qo_heads
    assert axes["head_dim_ckv"]["value"] == head_dim_ckv
    assert axes["head_dim_kpe"]["value"] == head_dim_kpe
    assert axes["page_size"]["value"] == page_size


# ---------------------------------------------------------------------------
# GDN decode
# ---------------------------------------------------------------------------


def test_gdn_decode_fi_trace():
    import flashinfer.gdn_decode

    B, H, HV, K = 4, 8, 16, 128

    q = torch.randn(B, 1, H, K, dtype=torch.bfloat16)
    k = torch.randn(B, 1, H, K, dtype=torch.bfloat16)
    v = torch.randn(B, 1, HV, K, dtype=torch.bfloat16)
    state = torch.zeros(B, HV, K, K, dtype=torch.float32)
    A_log = torch.zeros(HV, dtype=torch.float32)
    a = torch.zeros(B, 1, HV, dtype=torch.bfloat16)
    dt_bias = torch.zeros(HV, dtype=torch.float32)
    b = torch.zeros(B, 1, HV, dtype=torch.bfloat16)

    defn = flashinfer.gdn_decode.gated_delta_rule_decode.fi_trace(
        q=q, k=k, v=v, state=state, A_log=A_log, a=a, dt_bias=dt_bias, b=b
    )
    _check_defn(defn, "gdn", "gated_delta_rule_decode")
    axes = defn["axes"]
    assert axes["seq_len"]["value"] == 1
    assert axes["num_q_heads"]["value"] == H
    assert axes["num_v_heads"]["value"] == HV
    assert axes["head_size"]["value"] == K
    assert axes["batch_size"]["type"] == "var"


# ---------------------------------------------------------------------------
# Named tensor layer: verify refine_names is applied
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Module-level fi_trace helper: bound method support
# ---------------------------------------------------------------------------


def test_fi_trace_helper_bound_method():
    """fi_trace() helper must work with a bound method via __func__ unwrapping."""
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

    q = torch.randn(64, 32, 128, dtype=torch.bfloat16)
    k = torch.randn(128, 8, 128, dtype=torch.bfloat16)
    v = torch.randn(128, 8, 128, dtype=torch.bfloat16)

    # Create a dummy instance — we don't call run(), only fi_trace()
    class _FakeWrapper:
        run = BatchPrefillWithRaggedKVCacheWrapper.run

    instance = _FakeWrapper()
    # Accessing instance.run gives a bound method; fi_trace() must handle it
    defn = fi_trace(instance.run, q=q, k=k, v=v)
    _check_defn(defn, "gqa_ragged", "BatchPrefillWithRaggedKVCacheWrapper")


# ---------------------------------------------------------------------------
# End-to-end use case: simulate a Llama-3.1-8B decode step and produce a
# complete flashinfer-bench definition file ready to save to disk.
# ---------------------------------------------------------------------------


def test_usecase_llama31_decode_step(tmp_path):
    """
    Use case: profiling a Llama-3.1-8B decode step.

    A developer wants to benchmark their model's attention kernel. They run a
    forward pass with representative tensors, call fi_trace on the wrapper's
    .run method, and get back a JSON definition they can pass directly to
    flashinfer-bench -- without manually figuring out axis names or shapes.

    Model config (TP=1):
      num_qo_heads=32, num_kv_heads=8, head_dim=128, page_size=16
    Runtime:
      batch_size=64, num_pages=8192 (across all sequences in the batch)
    """
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

    # ── Shapes matching a Llama-3.1-8B decode at batch_size=64 ──────────────
    batch_size = 64
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    num_pages = 8192
    page_size = 16

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.bfloat16)
    k_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16
    )
    v_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16
    )

    # ── Generate the definition and write it to disk in one call ─────────────
    traces_dir = tmp_path / "benchmark_traces"
    defn = BatchDecodeWithPagedKVCacheWrapper.run.fi_trace(
        save_dir=traces_dir,
        q=q,
        paged_kv_cache=(k_cache, v_cache),
    )

    # ── Validate the definition matches the flashinfer-bench schema ──────────
    _check_defn(defn, "gqa_paged", "BatchDecodeWithPagedKVCacheWrapper")

    # Variable axes have no "value"; const axes carry the model config.
    assert defn["axes"]["batch_size"]["type"] == "var"
    assert defn["axes"]["num_pages"]["type"] == "var"
    assert defn["axes"]["num_qo_heads"] == {"type": "const", "value": num_qo_heads}
    assert defn["axes"]["num_kv_heads"] == {"type": "const", "value": num_kv_heads}
    assert defn["axes"]["head_dim"] == {"type": "const", "value": head_dim}
    assert defn["axes"]["page_size"] == {"type": "const", "value": page_size}

    # Input shapes use axis names, not raw integers.
    assert defn["inputs"]["q"]["shape"] == ["batch_size", "num_qo_heads", "head_dim"]
    assert defn["inputs"]["k_cache"]["shape"] == [
        "num_pages",
        "page_size",
        "num_kv_heads",
        "head_dim",
    ]
    assert defn["inputs"]["k_cache"]["dtype"] == "bfloat16"

    # Output mirrors the query shape.
    assert defn["outputs"]["output"]["shape"] == [
        "batch_size",
        "num_qo_heads",
        "head_dim",
    ]
    assert defn["outputs"]["output"]["dtype"] == "bfloat16"
    assert defn["outputs"]["lse"]["shape"] == ["batch_size", "num_qo_heads"]
    assert defn["outputs"]["lse"]["dtype"] == "float32"

    # ── The JSON file was written to disk ────────────────────────────────────
    json_file = traces_dir / f"{defn['name']}.json"
    assert json_file.exists(), f"Expected definition file at {json_file}"
    on_disk = json.loads(json_file.read_text())
    assert on_disk["axes"]["num_qo_heads"]["value"] == num_qo_heads

    assert json.loads(json_file.read_text())["axes"]["num_qo_heads"]["value"] == 32


def test_usecase_deepseek_mla_decode():
    """
    Use case: profiling a DeepSeek-V3 MLA decode step (TP=8).

    Model config (TP=8):
      num_qo_heads=16, head_dim_ckv=512, head_dim_kpe=64, page_size=64
    """
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    batch_size = 128  # tokens in the decode batch
    num_qo_heads = 16  # after TP=8 split
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_pages = 4096
    page_size = 64

    q_nope = torch.randn(batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16)
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16)
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16)

    defn = BatchMLAPagedAttentionWrapper.run.fi_trace(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
    )

    _check_defn(defn, "mla_paged", "BatchMLAPagedAttentionWrapper")

    assert defn["axes"]["num_qo_heads"]["value"] == num_qo_heads
    assert defn["axes"]["head_dim_ckv"]["value"] == head_dim_ckv
    assert defn["axes"]["head_dim_kpe"]["value"] == head_dim_kpe
    assert defn["axes"]["page_size"]["value"] == page_size
    assert defn["axes"]["batch_size"]["type"] == "var"

    # The output uses the CKV head dimension (not KPE).
    assert defn["outputs"]["output"]["shape"] == [
        "batch_size",
        "num_qo_heads",
        "head_dim_ckv",
    ]

    # Enrich with model metadata, then round-trip through JSON.
    defn["tags"] += ["model:deepseek-v3", "model:deepseek-r1", "tp:8", "stage:decode"]
    assert json.loads(json.dumps(defn))["axes"]["head_dim_ckv"]["value"] == 512


def test_usecase_sampling_vocab_discovery():
    """
    Use case: automatically discover the vocabulary size from live tensors.
    """
    import flashinfer.sampling

    # Qwen3 vocabulary size
    vocab_size = 151936
    batch_size = 32

    probs = torch.rand(batch_size, vocab_size, dtype=torch.float32)
    top_k = torch.full((batch_size,), 40, dtype=torch.int32)
    top_p = torch.full((batch_size,), 0.95, dtype=torch.float32)

    defn = flashinfer.sampling.top_k_top_p_sampling_from_probs.fi_trace(
        probs=probs, top_k=top_k, top_p=top_p
    )

    # vocab_size is automatically discovered from the probs tensor shape.
    assert defn["axes"]["vocab_size"]["type"] == "const"
    assert defn["axes"]["vocab_size"]["value"] == vocab_size

    # The definition name embeds the const axes values.
    assert str(vocab_size) in defn["name"]

    # Confirm the JSON is ready for flashinfer-bench.
    parsed = json.loads(json.dumps(defn))
    assert parsed["inputs"]["probs"]["dtype"] == "float32"
    assert parsed["outputs"]["samples"]["dtype"] == "int64"


def test_trtllm_batch_decode_mla_fi_trace_dense_and_ragged():
    import flashinfer.mla

    common = {
        "kv_cache": torch.empty(4, 64, 576, dtype=torch.bfloat16),
        "workspace_buffer": torch.empty(1024, dtype=torch.int8),
        "qk_nope_head_dim": 512,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "block_tables": torch.zeros(2, 1, dtype=torch.int32),
        "seq_lens": torch.full((2,), 64, dtype=torch.int32),
        "max_seq_len": 64,
    }

    dense = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla.fi_trace(
        query=torch.empty(2, 3, 128, 576, dtype=torch.bfloat16),
        **common,
    )
    _check_defn(
        dense,
        "mla_paged",
        "flashinfer.mla._core.trtllm_batch_decode_with_kv_cache_mla",
    )
    assert dense["name"].startswith("trtllm_batch_decode_mla_dense")
    assert dense["inputs"]["query"]["shape"] == [
        "batch_size",
        "q_len_per_request",
        "num_heads",
        "head_dim_qk",
    ]
    assert dense["outputs"]["output"]["shape"] == [
        "batch_size",
        "q_len_per_request",
        "num_heads",
        "kv_lora_rank",
    ]

    ragged = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla.fi_trace(
        query=torch.empty(5, 128, 576, dtype=torch.bfloat16),
        cum_seq_lens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        max_q_len=3,
        **common,
    )
    _check_defn(
        ragged,
        "mla_paged",
        "flashinfer.mla._core.trtllm_batch_decode_with_kv_cache_mla",
    )
    assert ragged["name"].startswith("trtllm_batch_decode_mla_ragged")
    assert ragged["inputs"]["query"]["shape"] == [
        "num_tokens",
        "num_heads",
        "head_dim_qk",
    ]
    assert ragged["outputs"]["output"]["shape"] == [
        "num_tokens",
        "num_heads",
        "kv_lora_rank",
    ]
    assert ragged["inputs"]["max_q_len"]["shape"] is None


# ---------------------------------------------------------------------------
# JSON file output
# ---------------------------------------------------------------------------


def test_fi_trace_writes_json_file(tmp_path):
    """fi_trace writes a <name>.json file when save_dir is given."""
    import flashinfer.norm

    hidden = torch.randn(16, 4096, dtype=torch.bfloat16)
    weight = torch.ones(4096, dtype=torch.bfloat16)

    defn = flashinfer.norm.rmsnorm.fi_trace(
        save_dir=tmp_path, input=hidden, weight=weight
    )

    expected_file = tmp_path / f"{defn['name']}.json"
    assert expected_file.exists(), f"Expected JSON file at {expected_file}"

    on_disk = json.loads(expected_file.read_text())
    assert on_disk == defn


def test_fi_trace_helper_writes_json_file(tmp_path):
    """The module-level fi_trace() helper threads save_dir through correctly."""
    import flashinfer.norm

    hidden = torch.randn(8, 7168, dtype=torch.bfloat16)
    weight = torch.ones(7168, dtype=torch.bfloat16)

    defn = fi_trace(
        flashinfer.norm.rmsnorm,
        save_dir=tmp_path,
        input=hidden,
        weight=weight,
    )

    expected_file = tmp_path / f"{defn['name']}.json"
    assert expected_file.exists()
    on_disk = json.loads(expected_file.read_text())
    assert on_disk["axes"]["hidden_size"]["value"] == 7168


def test_fi_trace_env_var_writes_json_file(tmp_path, monkeypatch):
    """FLASHINFER_TRACE_DUMP_DIR env-var (shared with logging) triggers file writing without save_dir."""
    import flashinfer.sampling

    # Use the real env-var; the template reads os.environ at call time.
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP_DIR", str(tmp_path))

    probs = torch.rand(4, 128256, dtype=torch.float32)
    top_k = torch.full((4,), 50, dtype=torch.int32)

    defn = flashinfer.sampling.top_k_sampling_from_probs.fi_trace(
        probs=probs, top_k=top_k
    )

    expected_file = tmp_path / f"{defn['name']}.json"
    assert expected_file.exists(), f"Expected file {expected_file}"
    assert json.loads(expected_file.read_text())["op_type"] == "sampling"


def test_fi_trace_creates_nested_save_dir(tmp_path):
    """save_dir is created automatically even if it doesn't exist yet."""
    import flashinfer.norm

    nested = tmp_path / "traces" / "rmsnorm"
    assert not nested.exists()

    hidden = torch.randn(4, 2048, dtype=torch.bfloat16)
    weight = torch.ones(2048, dtype=torch.bfloat16)

    defn = flashinfer.norm.rmsnorm.fi_trace(
        save_dir=nested, input=hidden, weight=weight
    )

    assert nested.exists()
    files = list(nested.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["name"] == defn["name"]


def test_fi_trace_filename_matches_definition_name(tmp_path):
    """The written filename is exactly '<definition_name>.json'."""
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

    q = torch.randn(4, 32, 128, dtype=torch.bfloat16)
    k_cache = torch.randn(64, 16, 8, 128, dtype=torch.bfloat16)
    v_cache = torch.randn(64, 16, 8, 128, dtype=torch.bfloat16)

    defn = BatchDecodeWithPagedKVCacheWrapper.run.fi_trace(
        save_dir=tmp_path,
        q=q,
        paged_kv_cache=(k_cache, v_cache),
    )

    expected_name = defn["name"]
    expected_file = tmp_path / f"{expected_name}.json"
    assert expected_file.exists()
    assert json.loads(expected_file.read_text())["name"] == expected_name


def test_nvfp4_append_trace_json_init_is_self_contained():
    trace_dir = Path(__file__).parent / "fi_trace_out"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cases = [
        (
            "nvfp4_quantize_append_paged_kv_cache_kv2_d64_pd32_sd4_ps4.json",
            "_nvfp4_quantize_append_paged_kv_cache_init",
        ),
        (
            "nvfp4_quantize_append_paged_kv_cache_with_slot_mapping_kv2_d64_pd32_sd4_ps4.json",
            "_nvfp4_quantize_append_paged_kv_cache_with_slot_mapping_init",
        ),
    ]
    for filename, init_name in cases:
        source = json.loads((trace_dir / filename).read_text())["init"]
        namespace = {}
        exec(source, namespace)
        assert init_name in namespace
        try:
            result = namespace[init_name](nnz_kv=1, device=device)
        except (RuntimeError, NotImplementedError, ValueError, ImportError) as exc:
            if device == "cpu":
                pytest.skip(f"{filename} init unsupported on CPU: {exc}")
            raise
        assert isinstance(result, dict)
