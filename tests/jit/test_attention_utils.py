from pathlib import Path

import torch

from flashinfer.jit import core
from flashinfer.jit import env as jit_env
from flashinfer.jit.attention import modules as attention_modules
from flashinfer.jit.attention.modules import gen_customize_batch_prefill_module


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
    # The FP4 KV gate is emitted as compile-time checks: an #error if the
    # FP4 enable flag is missing plus a static_assert pinning DTypeKV to the
    # packed FP4 container type.
    assert (
        '#error "NVFP4 KV paged prefill compiled without FLASHINFER_ENABLE_FP4_E2M1"'
        in generated
    )
    assert "static_assert(std::is_same_v<DTypeKV, __nv_fp4x2_e2m1>," in generated
    assert "constexpr auto USE_SLIDING_WINDOW = true;" in generated
    for field in (
        "maybe_k_cache_sf",
        "maybe_v_cache_sf",
    ):
        assert f"uint8_t* {field};" in generated
    # SF strides ride on the upstream static param fields (set from the actual
    # SF tensors via GetFP4ScaleStrides in the generated params setter).
    for field in ("k_sf", "v_sf"):
        assert f"uint32_t {field}_stride_page;" in generated
        assert f"uint32_t {field}_stride_h;" in generated
        assert f"uint32_t {field}_stride_n;" in generated


def test_batch_prefill_generates_equal_and_independent_stride_specializations(
    tmp_path, monkeypatch
):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(
        attention_modules.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, 3)},
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "generated")
    monkeypatch.setattr(jit_env, "FLASHINFER_CSRC_DIR", repo_root / "csrc")

    uri = "test_batch_prefill_stride_specializations"
    spec = gen_customize_batch_prefill_module(
        "fa2",
        uri,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        64,
        64,
        [],
        [],
        [],
        [],
        "DefaultAttention",
        "struct DefaultAttention {};",
    )

    assert spec.name == uri
    paged = tmp_path / "generated" / uri / "batch_prefill_paged_kernel_mask_0.cu"
    ragged = tmp_path / "generated" / uri / "batch_prefill_ragged_kernel_mask_0.cu"
    binding = tmp_path / "generated" / uri / "batch_prefill.cu"
    paged_text = paged.read_text()
    assert (
        paged_text.count("template cudaError_t BatchPrefillWithPagedKVCacheDispatched<")
        == 6
    )
    assert paged_text.count("/*SAME_KV_STRIDES=*/true") == 3
    assert paged_text.count("/*SAME_KV_STRIDES=*/false") == 3
    assert (
        ragged.read_text().count(
            "template cudaError_t BatchPrefillWithRaggedKVCacheDispatched<"
        )
        == 3
    )
    assert (
        "DISPATCH_BOOL(kv_strides_are_identical, SAME_KV_STRIDES" in binding.read_text()
    )


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
