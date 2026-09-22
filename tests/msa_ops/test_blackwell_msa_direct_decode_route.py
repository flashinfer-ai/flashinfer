"""CPU-only policy tests for the direct persistent M16 decode route."""

import inspect

import torch

from flashinfer.msa_ops._blackwell_sm100 import (
    _decode_variant,
    _exact_non16_decode_variant,
    _is_exact_bf16_topk4_qload4_prefill,
    _is_exact_fp8_topk8_qagg_prefill,
    _prefill_variant,
    _resolve_fp8_q1_schedule,
    _run_decode_module,
    _should_use_long_prefill,
    _uniform_fp8_decode_grid,
)


def _meta_tensor(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


def test_exact_non16_decode_routes_and_neighbors_fail_closed() -> None:
    common = dict(
        requested_schedule="",
        capturing=False,
        paged=True,
        force_fused=True,
        causal=True,
        q_offset_is_none=True,
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
    )
    assert (
        _exact_non16_decode_variant(
            **common,
            batch_size=64,
            total_q=512,
            seqlen_q=8,
            num_q_heads=64,
            num_kv_heads=4,
            topk=32,
            k_outer_dim=32768,
            max_pages=512,
        )
        == "decode_m16_bf16_paged_topk32"
    )
    assert (
        _exact_non16_decode_variant(
            **common,
            batch_size=2,
            total_q=2,
            seqlen_q=1,
            num_q_heads=8,
            num_kv_heads=1,
            topk=4,
            k_outer_dim=6,
            max_pages=3,
        )
        == "decode_m16_bf16_paged_topk4_exact512"
    )
    assert (
        _exact_non16_decode_variant(
            **common,
            batch_size=64,
            total_q=512,
            seqlen_q=8,
            num_q_heads=64,
            num_kv_heads=4,
            topk=32,
            k_outer_dim=32767,
            max_pages=512,
        )
        is None
    )
    assert (
        _exact_non16_decode_variant(
            **{**common, "capturing": True},
            batch_size=2,
            total_q=2,
            seqlen_q=1,
            num_q_heads=8,
            num_kv_heads=1,
            topk=4,
            k_outer_dim=6,
            max_pages=3,
        )
        is None
    )


def test_exact_non16_prefill_routes_and_neighbors_fail_closed() -> None:
    qagg = dict(
        q=_meta_tensor((3072, 32, 128), torch.bfloat16),
        k=_meta_tensor((24576, 2, 128), torch.float8_e4m3fn),
        v=_meta_tensor((24576, 2, 128), torch.float8_e4m3fn),
        q2k_indices=_meta_tensor((2, 3072, 8), torch.int32),
        cu_q=_meta_tensor((4,), torch.int32),
        cu_k=_meta_tensor((4,), torch.int32),
        paged=False,
        batch_size=3,
        causal=True,
        q_offset_is_none=True,
        softmax_scale=None,
        return_temperature_lse=True,
        lse_temperature_scale=1.0,
        requested_schedule="",
        capturing=False,
    )
    assert _is_exact_fp8_topk8_qagg_prefill(**qagg)
    assert not _is_exact_fp8_topk8_qagg_prefill(**{**qagg, "capturing": True})
    assert not _is_exact_fp8_topk8_qagg_prefill(
        **{
            **qagg,
            "q2k_indices": _meta_tensor((2, 3072, 16), torch.int32),
        }
    )

    qload4 = dict(
        q=_meta_tensor((12288, 8, 128), torch.bfloat16),
        k=_meta_tensor((192, 2, 128, 128), torch.bfloat16),
        v=_meta_tensor((192, 2, 128, 128), torch.bfloat16),
        q2k_indices=_meta_tensor((2, 12288, 4), torch.int32),
        cu_q=_meta_tensor((4,), torch.int32),
        cu_k=_meta_tensor((4,), torch.int32),
        page_table=_meta_tensor((3, 64), torch.int32),
        kv_lens=_meta_tensor((3,), torch.int32),
        paged=True,
        batch_size=3,
        causal=True,
        q_offset_is_none=True,
        softmax_scale=None,
        return_temperature_lse=False,
        lse_temperature_scale=1.0,
        requested_schedule="",
    )
    assert _is_exact_bf16_topk4_qload4_prefill(**qload4)
    assert not _is_exact_bf16_topk4_qload4_prefill(
        **{**qload4, "requested_schedule": "m64"}
    )
    assert not _is_exact_bf16_topk4_qload4_prefill(
        **{
            **qload4,
            "page_table": _meta_tensor((3, 63), torch.int32),
        }
    )


def test_decode_dtype_and_layout_select_only_direct_m16_variants() -> None:
    assert (
        _decode_variant(
            q_dtype=torch.bfloat16,
            k_dtype=torch.bfloat16,
            paged=False,
        )
        == "decode_m16_bf16_flat"
    )
    assert (
        _decode_variant(
            q_dtype=torch.bfloat16,
            k_dtype=torch.bfloat16,
            paged=True,
        )
        == "decode_m16_bf16_paged"
    )
    assert (
        _decode_variant(
            q_dtype=torch.float16,
            k_dtype=torch.float16,
            paged=False,
        )
        == "decode_m16_fp16_flat"
    )
    assert (
        _decode_variant(
            q_dtype=torch.float16,
            k_dtype=torch.float16,
            paged=True,
        )
        == "decode_m16_fp16_paged"
    )
    assert (
        _decode_variant(
            q_dtype=torch.bfloat16,
            k_dtype=torch.float8_e4m3fn,
            paged=False,
        )
        == "decode_m16_bf16_query_fp8_kv_flat"
    )
    assert (
        _decode_variant(
            q_dtype=torch.bfloat16,
            k_dtype=torch.float8_e4m3fn,
            paged=True,
        )
        == "decode_m16_bf16_query_fp8_kv_paged"
    )


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
    assert _prefill_variant(**common, causal=False, max_pages=64).endswith("noncausal")


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
    assert (
        _resolve_fp8_q1_schedule(
            **common,
            paged=True,
            batch_size=128,
            k_outer_dim=4096,
            max_pages=32,
        )
        == "q1_paged_xform2"
    )
    assert (
        _resolve_fp8_q1_schedule(
            **{**common, "total_q": 32},
            paged=False,
            batch_size=32,
            k_outer_dim=262144,
            max_pages=0,
        )
        == "q1_flat_xform2"
    )
    assert (
        _resolve_fp8_q1_schedule(
            **common,
            paged=True,
            batch_size=128,
            k_outer_dim=4095,
            max_pages=32,
        )
        == ""
    )


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


def test_long_prefill_temperature_lse_requires_unit_scale() -> None:
    logits = torch.tensor([-2.5, -0.25, 1.5], dtype=torch.float32)
    ordinary_lse = torch.logsumexp(logits, dim=0)
    unit_temperature_lse = torch.logsumexp(logits * 1.0, dim=0)
    torch.testing.assert_close(unit_temperature_lse, ordinary_lse, rtol=0, atol=0)

    common = dict(
        requested_schedule="",
        batch_size=1,
        total_q=8192,
        paged=False,
        group_size=16,
        max_pages=0,
        k_outer_dim=8192,
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        v_dtype=torch.bfloat16,
        causal=True,
        q_offset_is_none=True,
        return_temperature_lse=True,
    )
    assert _should_use_long_prefill(**common, lse_temperature_scale=1.0)
    assert not _should_use_long_prefill(**common, lse_temperature_scale=0.7)


def test_uniform_fp8_grid_uses_full_even_wave_when_available() -> None:
    assert (
        _uniform_fp8_decode_grid(total_work_items=256, num_sms=148, seqlen_q=4) == 128
    )
    assert (
        _uniform_fp8_decode_grid(total_work_items=256, num_sms=148, seqlen_q=1) == 148
    )
