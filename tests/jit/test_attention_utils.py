from pathlib import Path

import torch

from flashinfer.jit import env as jit_env
from flashinfer.jit.attention.modules import gen_customize_batch_prefill_module
from flashinfer.jit.attention.utils import generate_additional_params


def test_generate_additional_params_emits_nvfp4_kv_sf_strides():
    decl, func_params, setter = generate_additional_params(
        ["maybe_k_cache_sf", "maybe_v_cache_sf"],
        ["uint8_t", "uint8_t"],
        [],
        [],
    )

    assert "uint32_t maybe_k_cache_sf_stride_page;" in decl
    assert "uint32_t maybe_k_cache_sf_stride_h;" in decl
    assert "uint32_t maybe_k_cache_sf_stride_n;" in decl
    assert "uint32_t maybe_v_cache_sf_stride_page;" in decl
    assert "uint32_t maybe_v_cache_sf_stride_h;" in decl
    assert "uint32_t maybe_v_cache_sf_stride_n;" in decl
    assert "Optional<ffi::Tensor> maybe_k_cache_sf" in func_params
    assert "Optional<ffi::Tensor> maybe_v_cache_sf" in func_params
    assert "params.maybe_k_cache_sf_stride_page" in setter
    assert "params.maybe_v_cache_sf_stride_page" in setter
    assert "kv_layout == QKVLayout::kNHD" in setter


def test_generate_additional_params_does_not_emit_unrelated_strides():
    decl, _, setter = generate_additional_params(
        ["maybe_alibi_slopes"],
        ["float"],
        ["sm_scale"],
        ["double"],
    )

    assert "stride_page" not in decl
    assert "stride_page" not in setter


def test_batch_prefill_nvfp4_swa_paged_params_declares_sf_strides(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "generated")
    monkeypatch.setattr(jit_env, "FLASHINFER_CSRC_DIR", repo_root / "csrc")

    uri = "test_batch_prefill_nvfp4_swa"
    gen_customize_batch_prefill_module(
        "fa2",
        uri,
        torch.bfloat16,
        torch.uint8,
        torch.bfloat16,
        torch.int32,
        128,
        128,
        ["maybe_k_cache_sf", "maybe_v_cache_sf"],
        ["uint8_t", "uint8_t"],
        [],
        [],
        "DefaultAttention",
        "struct DefaultAttention {};",
        use_sliding_window=True,
    )

    generated = (tmp_path / "generated" / uri / "batch_prefill_config.inc").read_text()
    assert "constexpr bool REQUIRE_FP4_KV_CACHE = true;" in generated
    assert "constexpr auto USE_SLIDING_WINDOW = true;" in generated
    for field in (
        "maybe_k_cache_sf",
        "maybe_v_cache_sf",
    ):
        assert f"uint8_t* {field};" in generated
        assert f"uint32_t {field}_stride_page;" in generated
        assert f"uint32_t {field}_stride_h;" in generated
        assert f"uint32_t {field}_stride_n;" in generated


def test_batch_prefill_nvfp4_requires_sf_tensors():
    try:
        gen_customize_batch_prefill_module(
            "fa2",
            "test_batch_prefill_nvfp4_missing_sf",
            torch.bfloat16,
            torch.uint8,
            torch.bfloat16,
            torch.int32,
            128,
            128,
            [],
            [],
            [],
            [],
            "DefaultAttention",
            "struct DefaultAttention {};",
        )
    except ValueError as exc:
        assert "maybe_k_cache_sf" in str(exc)
        assert "maybe_v_cache_sf" in str(exc)
    else:
        raise AssertionError("expected NVFP4 KV prefill without SF tensors to fail")
