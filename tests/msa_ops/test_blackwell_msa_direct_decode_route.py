"""CPU-only policy tests for the direct persistent M16 decode route."""

import inspect

import torch

from flashinfer.msa_ops._blackwell_sm100 import (
    _decode_variant,
    _prefill_variant,
    _resolve_fp8_q1_schedule,
    _run_decode_module,
    _should_use_long_prefill,
    _uniform_fp8_decode_grid,
)


def test_decode_dtype_and_layout_select_only_direct_m16_variants() -> None:
    assert _decode_variant(
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        paged=False,
    ) == "decode_m16_bf16_flat"
    assert _decode_variant(
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        paged=True,
    ) == "decode_m16_bf16_paged"
    assert _decode_variant(
        q_dtype=torch.float16,
        k_dtype=torch.float16,
        paged=False,
    ) == "decode_m16_fp16_flat"
    assert _decode_variant(
        q_dtype=torch.float16,
        k_dtype=torch.float16,
        paged=True,
    ) == "decode_m16_fp16_paged"
    assert _decode_variant(
        q_dtype=torch.bfloat16,
        k_dtype=torch.float8_e4m3fn,
        paged=False,
    ) == "decode_m16_bf16_query_fp8_kv_flat"
    assert _decode_variant(
        q_dtype=torch.bfloat16,
        k_dtype=torch.float8_e4m3fn,
        paged=True,
    ) == "decode_m16_bf16_query_fp8_kv_paged"


def test_direct_decode_launcher_has_no_mixed_or_split_route_controls() -> None:
    signature = inspect.signature(_run_decode_module)
    source = inspect.getsource(_run_decode_module)

    assert "force_fused" not in signature.parameters
    assert "persistent_unsplit" not in signature.parameters
    assert "_MODE_MIXED" not in source
    assert "_SPLIT_FORCED" not in source
    assert "decode_partial_o" not in source
    assert "decode_split_completion" not in source
    assert "physical_ctas = min(total_tasks, num_sms)" in source
    assert "max_splits = 1" in source


def test_prefill_variant_selects_exact_gqa16_paged_mask_family() -> None:
    common = dict(
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        paged=True,
        folded_gqa_group=16,
    )
    assert _prefill_variant(**common, causal=True, max_pages=64).endswith(
        "causal_mask64"
    )
    assert _prefill_variant(**common, causal=True, max_pages=65).endswith(
        "causal_large"
    )
    assert _prefill_variant(**common, causal=False, max_pages=64).endswith(
        "noncausal"
    )


def test_default_fp8_q1_promotion_is_shape_exact() -> None:
    common = dict(
        requested="",
        capturing=False,
        force_fused=True,
        causal=True,
        q_offset_is_none=True,
        q_dtype=torch.bfloat16,
        k_dtype=torch.float8_e4m3fn,
        total_q=128,
        seqlen_q=1,
        num_q_heads=64,
        num_kv_heads=4,
        topk=16,
    )
    assert _resolve_fp8_q1_schedule(
        **common,
        paged=True,
        batch_size=128,
        k_outer_dim=4096,
        max_pages=32,
    ) == "q1_paged_xform2"
    assert _resolve_fp8_q1_schedule(
        **{**common, "total_q": 32},
        paged=False,
        batch_size=32,
        k_outer_dim=262144,
        max_pages=0,
    ) == "q1_flat_xform2"
    assert _resolve_fp8_q1_schedule(
        **common,
        paged=True,
        batch_size=128,
        k_outer_dim=4095,
        max_pages=32,
    ) == ""


def test_long_prefill_predicate_covers_flat_and_paged_boundaries() -> None:
    common = dict(
        requested_schedule="",
        batch_size=1,
        total_q=8192,
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        v_dtype=torch.bfloat16,
        causal=True,
        q_offset_is_none=True,
        return_temperature_lse=False,
        lse_temperature_scale=1.0,
    )
    assert _should_use_long_prefill(
        **common, paged=False, group_size=16, max_pages=0, k_outer_dim=8192
    )
    assert _should_use_long_prefill(
        **common, paged=True, group_size=8, max_pages=64, k_outer_dim=64
    )
    assert not _should_use_long_prefill(
        **common, paged=True, group_size=16, max_pages=64, k_outer_dim=64
    )


def test_uniform_fp8_grid_uses_full_even_wave_when_available() -> None:
    assert _uniform_fp8_decode_grid(
        total_work_items=256, num_sms=148, seqlen_q=4
    ) == 128
    assert _uniform_fp8_decode_grid(
        total_work_items=256, num_sms=148, seqlen_q=1
    ) == 148
