import inspect
from pathlib import Path

import pytest
import torch

from flashinfer import prefill
from flashinfer.jit import core
from flashinfer.jit import env as jit_env
from flashinfer.jit.attention import modules as attention_modules
from flashinfer.jit.attention.modules import gen_customize_batch_prefill_module


@pytest.mark.parametrize(
    "backend,mode",
    [("fa2", None), ("fa2", "independent"), ("fa3", None), ("fa3", "runtime")],
)
def test_custom_prefill_adapter_preserves_stride_specialization(
    tmp_path, monkeypatch, backend, mode
):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(
        attention_modules.current_compilation_context, "TARGET_CUDA_ARCHS", {(9, 0)}
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "generated")
    monkeypatch.setattr(jit_env, "FLASHINFER_CSRC_DIR", repo_root / "csrc")
    # Exercise the cached adapter and real generator; only compilation is stubbed.
    monkeypatch.setattr(core.JitSpec, "build_and_load", lambda self: self)
    uri = f"test_custom_prefill_adapter_{backend}_{mode}_{tmp_path.name}"
    kwargs = (
        {} if mode is None else {"paged_kv_stride_mode": mode, "module_surface": "full"}
    )
    spec = prefill.get_customize_batch_prefill_module(
        backend,
        uri,
        torch.bfloat16,
        torch.bfloat16,
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
        **kwargs,
    )
    assert spec.name == uri
    assert len(spec.sources) == 10
    generated_dir = tmp_path / "generated" / uri
    suffix = "_sm90" if backend == "fa3" else ""
    binding = (generated_dir / f"batch_prefill{suffix}_jit_binding.cu").read_text()
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run," in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged_run," in binding
    if backend == "fa2":
        expected_mode = "INDEPENDENT" if mode == "independent" else "RUNTIME"
        config = (generated_dir / "batch_prefill_config.inc").read_text()
        assert (
            f"#define PAGED_KV_STRIDE_MODE PAGED_KV_STRIDE_MODE_{expected_mode}"
            in config
        )
        for mask in range(4):
            source = (
                generated_dir / f"batch_prefill_paged_kernel_mask_{mask}.cu"
            ).read_text()
            assert ("/*SAME_KV_STRIDES=*/true" in source) == (mode is None)
            assert "/*SAME_KV_STRIDES=*/false" in source


def test_batch_prefill_nvfp4_swa_paged_params_declares_sf_strides(
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

    public_parameters = inspect.signature(
        attention_modules.gen_batch_prefill_module
    ).parameters
    assert list(public_parameters) == [
        "backend",
        "dtype_q",
        "dtype_kv",
        "dtype_o",
        "dtype_idx",
        "head_dim_qk",
        "head_dim_vo",
        "pos_encoding_mode",
        "use_sliding_window",
        "use_logits_soft_cap",
        "use_fp16_qk_reduction",
    ]
    assert all(
        parameter.default is inspect.Parameter.empty
        and parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in public_parameters.values()
    )

    args = (
        "fa2",
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        64,
        64,
        0,
        False,
        False,
        False,
    )
    base_uri = attention_modules.get_batch_prefill_uri(*args)
    specs = {
        "runtime_full": attention_modules.gen_batch_prefill_module(*args),
        "equal_full": attention_modules._gen_batch_prefill_primary_module(*args),
        "independent_full": (
            attention_modules._gen_batch_prefill_independent_full_module(*args)
        ),
        "independent_paged": (
            attention_modules._gen_batch_prefill_independent_paged_module(*args)
        ),
    }

    assert {variant: spec.name for variant, spec in specs.items()} == {
        "runtime_full": base_uri,
        "equal_full": f"{base_uri}_kv_stride_equal",
        "independent_full": f"{base_uri}_kv_stride_independent",
        "independent_paged": f"{base_uri}_paged_kv_stride_independent",
    }
    assert len({spec.name for spec in specs.values()}) == 4

    full_sources = sorted(
        [f"batch_prefill_paged_kernel_mask_{mask_mode}.cu" for mask_mode in range(4)]
        + [f"batch_prefill_ragged_kernel_mask_{mask_mode}.cu" for mask_mode in range(4)]
        + ["batch_prefill.cu", "batch_prefill_jit_binding.cu"]
    )
    paged_sources = sorted(
        [f"batch_prefill_paged_kernel_mask_{mask_mode}.cu" for mask_mode in range(4)]
        + ["batch_prefill_paged.cu", "batch_prefill_paged_jit_binding.cu"]
    )
    expected_sources = {
        "runtime_full": full_sources,
        "equal_full": full_sources,
        "independent_full": full_sources,
        "independent_paged": paged_sources,
    }
    expected_instantiations = {
        "runtime_full": {"true": 3, "false": 3},
        "equal_full": {"true": 3, "false": 0},
        "independent_full": {"true": 0, "false": 3},
        "independent_paged": {"true": 0, "false": 3},
    }
    expected_mode_macros = {
        "runtime_full": "PAGED_KV_STRIDE_MODE_RUNTIME",
        "equal_full": "PAGED_KV_STRIDE_MODE_EQUAL",
        "independent_full": "PAGED_KV_STRIDE_MODE_INDEPENDENT",
        "independent_paged": "PAGED_KV_STRIDE_MODE_INDEPENDENT",
    }

    for variant, spec in specs.items():
        generated_dir = tmp_path / "generated" / spec.name
        assert {source.parent for source in spec.sources} == {generated_dir}
        assert (
            sorted(source.name for source in spec.sources) == expected_sources[variant]
        )
        assert len(spec.sources) == (6 if variant == "independent_paged" else 10)

        config = (generated_dir / "batch_prefill_config.inc").read_text()
        assert "#define PAGED_KV_STRIDE_MODE " + expected_mode_macros[variant] in config
        paged_header = generated_dir / "batch_prefill_paged.cuh"
        assert paged_header.exists()
        assert paged_header not in spec.sources
        paged_host_impl = paged_header.read_text()
        assert "paged_k_cache.ndim()" in paged_host_impl
        assert "route unequal layouts" in paged_host_impl
        assert "through the independent paged module" in paged_host_impl

        is_paged_only = variant == "independent_paged"
        host_name = "batch_prefill_paged.cu" if is_paged_only else "batch_prefill.cu"
        binding_name = (
            "batch_prefill_paged_jit_binding.cu"
            if is_paged_only
            else "batch_prefill_jit_binding.cu"
        )
        host = (generated_dir / host_name).read_text()
        assert '#include "batch_prefill_paged.cuh"' in host
        binding = (generated_dir / binding_name).read_text()
        assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run" in binding
        assert binding.count("TVM_FFI_DLL_EXPORT_TYPED_FUNC(") == (
            1 if is_paged_only else 4
        )
        if is_paged_only:
            for excluded_export in ("plan", "workspace_size", "ragged_run"):
                assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({excluded_export}" not in binding

        for mask_mode in range(4):
            paged_text = (
                generated_dir / f"batch_prefill_paged_kernel_mask_{mask_mode}.cu"
            ).read_text()
            assert paged_text.count(
                "template cudaError_t BatchPrefillWithPagedKVCacheDispatched<"
            ) == sum(expected_instantiations[variant].values())
            for value, count in expected_instantiations[variant].items():
                assert paged_text.count(f"/*SAME_KV_STRIDES=*/{value}") == count

            ragged_path = (
                generated_dir / f"batch_prefill_ragged_kernel_mask_{mask_mode}.cu"
            )
            if is_paged_only:
                assert not ragged_path.exists()
            else:
                assert (
                    ragged_path.read_text().count(
                        "template cudaError_t BatchPrefillWithRaggedKVCacheDispatched<"
                    )
                    == 3
                )

    for invalid_mode in ("runtime", "equal"):
        invalid_uri = f"invalid_{invalid_mode}_paged_surface"
        with pytest.raises(
            ValueError, match="Unsupported batch-prefill mode/surface combination"
        ):
            gen_customize_batch_prefill_module(
                "fa2",
                invalid_uri,
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
                paged_kv_stride_mode=invalid_mode,
                module_surface="paged",
            )
        assert not (tmp_path / "generated" / invalid_uri).exists()


def test_gen_fa2_aot_uses_equal_primary_only(monkeypatch):
    import flashinfer.aot as flashinfer_aot

    sentinel = object()
    calls = []

    def fake_primary(**kwargs):
        calls.append(kwargs)
        return sentinel

    def unexpected_public_full(**kwargs):
        raise AssertionError("gen_fa2 must not put the runtime/full module in AOT")

    monkeypatch.setattr(
        flashinfer_aot, "_gen_batch_prefill_primary_module", fake_primary
    )
    monkeypatch.setattr(
        flashinfer_aot, "gen_batch_prefill_module", unexpected_public_full
    )

    specs = list(
        flashinfer_aot.gen_fa2(
            dtype_qo=torch.bfloat16,
            dtype_kv=torch.bfloat16,
            head_dim_qk=128,
            head_dim_vo=128,
            use_sliding_window=False,
            use_logits_soft_cap=False,
            prefill_only=True,
        )
    )

    assert specs == [sentinel]
    assert len(calls) == 1
    assert calls[0]["backend"] == "fa2"


def test_gen_attention_aot_matrix_contains_equal_primaries_only(tmp_path, monkeypatch):
    import flashinfer.aot as flashinfer_aot

    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(
        attention_modules.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, 3)},
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "generated")
    monkeypatch.setattr(jit_env, "FLASHINFER_CSRC_DIR", repo_root / "csrc")

    specs = list(
        flashinfer_aot.gen_attention(
            f16_dtype_=[torch.bfloat16],
            f8_dtype_=[],
            fa2_head_dim_=[(64, 64), (128, 128)],
            fa3_head_dim_=[],
            use_sliding_window_=[False, True],
            use_logits_soft_cap_=[False],
            has_sm90=False,
            has_sm100=False,
            add_gemma=False,
            add_oai_oss=False,
        )
    )
    standard_prefill_specs = [
        spec for spec in specs if spec.name.startswith("batch_prefill_with_kv_cache_")
    ]

    assert len(standard_prefill_specs) == 4
    assert all(
        spec.name.endswith("_kv_stride_equal") for spec in standard_prefill_specs
    )
    assert all(len(spec.sources) == 10 for spec in standard_prefill_specs)
    assert not any(
        "_paged_kv_stride_independent" in spec.name
        or spec.name.endswith("_kv_stride_independent")
        for spec in specs
    )


def test_attention_sink_stays_independent_full(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(
        attention_modules.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, 3)},
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "generated")
    monkeypatch.setattr(jit_env, "FLASHINFER_CSRC_DIR", repo_root / "csrc")

    args = (
        "fa2",
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        64,
        64,
        0,
        False,
    )
    spec = attention_modules.gen_batch_prefill_attention_sink_module(*args)
    assert spec.name == attention_modules.get_batch_prefill_attention_sink_uri(*args)
    assert len(spec.sources) == 10

    generated_dir = tmp_path / "generated" / spec.name
    config = (generated_dir / "batch_prefill_config.inc").read_text()
    assert "#define PAGED_KV_STRIDE_MODE PAGED_KV_STRIDE_MODE_INDEPENDENT" in config
    binding = (generated_dir / "batch_prefill_jit_binding.cu").read_text()
    assert binding.count("TVM_FFI_DLL_EXPORT_TYPED_FUNC(") == 4
    for mask_mode in range(4):
        paged_text = (
            generated_dir / f"batch_prefill_paged_kernel_mask_{mask_mode}.cu"
        ).read_text()
        assert paged_text.count("/*SAME_KV_STRIDES=*/true") == 0
        assert paged_text.count("/*SAME_KV_STRIDES=*/false") == 3


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
