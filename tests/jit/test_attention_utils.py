from pathlib import Path
import re

import pytest
import torch

from flashinfer import prefill
from flashinfer.jit import core
from flashinfer.jit import env as jit_env
from flashinfer.jit.attention import modules as attention_modules
from flashinfer.jit.attention.modules import gen_customize_batch_prefill_module


@pytest.fixture
def generated_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(
        attention_modules.current_compilation_context, "TARGET_CUDA_ARCHS", {(9, 0)}
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    monkeypatch.setattr(
        jit_env, "FLASHINFER_CSRC_DIR", Path(__file__).resolve().parents[2] / "csrc"
    )
    return tmp_path


@pytest.mark.parametrize(
    "backend,mode",
    [("fa2", None), ("fa2", "independent"), ("fa3", None), ("fa3", "runtime")],
)
def test_custom_prefill_adapter_preserves_stride_specialization(
    generated_sources, monkeypatch, backend, mode
):
    # Exercise the cached adapter and real generator; only compilation is stubbed.
    monkeypatch.setattr(core.JitSpec, "build_and_load", lambda self: self)
    uri = f"test_custom_prefill_adapter_{backend}_{mode}_{generated_sources.name}"
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
    generated_dir = generated_sources / uri
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


@pytest.mark.parametrize(
    "factory,suffix,source_count,variants",
    [
        ("gen_batch_prefill_module", "", 10, {"true", "false"}),
        ("_gen_batch_prefill_primary_module", "_kv_stride_equal", 10, {"true"}),
        (
            "_gen_batch_prefill_independent_paged_module",
            "_paged_kv_stride_independent",
            6,
            {"false"},
        ),
    ],
)
def test_batch_prefill_stride_module_surface(
    generated_sources, factory, suffix, source_count, variants
):
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
    spec = getattr(attention_modules, factory)(*args)
    assert spec.name == attention_modules.get_batch_prefill_uri(*args) + suffix
    assert len(spec.sources) == source_count
    for mask in range(4):
        source = (
            generated_sources / spec.name / f"batch_prefill_paged_kernel_mask_{mask}.cu"
        ).read_text()
        for variant in ("true", "false"):
            assert (f"/*SAME_KV_STRIDES=*/{variant}" in source) == (variant in variants)
    binding_name = (
        "batch_prefill_paged_jit_binding.cu"
        if source_count == 6
        else "batch_prefill_jit_binding.cu"
    )
    binding = (generated_sources / spec.name / binding_name).read_text()
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run," in binding
    for entrypoint in ("plan", "workspace_size", "ragged_run"):
        assert (f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({entrypoint}," in binding) == (
            source_count == 10
        )


def test_gen_attention_aot_matrix_contains_equal_primaries_only(generated_sources):
    import flashinfer.aot as flashinfer_aot

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
    persistent_specs = [
        spec for spec in specs if spec.name.startswith("batch_attention_with_kv_cache_")
    ]
    # The sliding-window cross product duplicates URIs; the shipped portfolio
    # contains only one equal-stride primary per actual configuration.
    assert len(persistent_specs) == 4
    assert len({spec.name for spec in persistent_specs}) == 2
    assert all(spec.name.endswith("_kv_stride_equal") for spec in persistent_specs)
    assert all(len(spec.sources) == 6 for spec in persistent_specs)
    assert not any(
        "_paged_kv_stride_independent" in spec.name
        or spec.name.endswith("_kv_stride_independent")
        for spec in specs
    )


def test_attention_sink_stays_independent_full(generated_sources):
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

    generated_dir = generated_sources / spec.name
    config = (generated_dir / "batch_prefill_config.inc").read_text()
    assert "#define PAGED_KV_STRIDE_MODE PAGED_KV_STRIDE_MODE_INDEPENDENT" in config
    binding = (generated_dir / "batch_prefill_jit_binding.cu").read_text()
    for entrypoint in ("plan", "workspace_size", "ragged_run", "paged_run"):
        assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({entrypoint}," in binding
    for mask_mode in range(4):
        paged_text = (
            generated_dir / f"batch_prefill_paged_kernel_mask_{mask_mode}.cu"
        ).read_text()
        assert "/*SAME_KV_STRIDES=*/true" not in paged_text
        assert "/*SAME_KV_STRIDES=*/false" in paged_text


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


def _persistent_instantiation_arguments(source):
    # Ignore comments and whitespace; inspect the actual explicit template args.
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
    return [
        tuple(re.sub(r"\s+", "", argument) for argument in arguments.split(","))
        for arguments in re.findall(
            r"template\s+cudaError_t\s+BatchPagedAttentionPersistent\s*<(.*?)>\s*\(",
            source,
            flags=re.DOTALL,
        )
    ]


def _assert_persistent_module_surface(directory, variants):
    for mask in range(4):
        source = (
            directory / f"batch_attention_paged_kernel_mask_{mask}.cu"
        ).read_text()
        assert sorted(_persistent_instantiation_arguments(source)) == sorted(
            (
                "128",
                "16",
                "64",
                "64",
                (
                    "MaskMode::kNone",
                    "MaskMode::kCausal",
                    "MaskMode::kCustom",
                    "MaskMode::kMultiItemScoring",
                )[mask],
                variant,
                "StandardAttention<false>",
                "PersistentParams",
            )
            for variant in variants
        )
    binding = (directory / "batch_attention_jit_binding.cu").read_text()
    for entrypoint in ("plan", "run"):
        assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({entrypoint}," in binding


@pytest.mark.parametrize(
    "factory,suffix,mode,variants",
    [
        ("gen_batch_attention_module", "", "RUNTIME", {"true", "false"}),
        ("_gen_batch_attention_primary_module", "_kv_stride_equal", "EQUAL", {"true"}),
        (
            "_gen_batch_attention_independent_module",
            "_kv_stride_independent",
            "INDEPENDENT",
            {"false"},
        ),
    ],
)
@pytest.mark.parametrize("profiler", [False, True])
def test_batch_attention_stride_module_surface(
    generated_sources, factory, suffix, mode, variants, profiler
):
    args = (
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        64,
        64,
        0,
        False,
        profiler,
    )
    spec = getattr(attention_modules, factory)(*args)
    assert spec.name == attention_modules.get_batch_attention_uri(*args) + suffix
    assert len(spec.sources) == 6
    directory = generated_sources / spec.name
    config = (directory / "batch_attention_config.inc").read_text()
    assert (
        f"#define BATCH_ATTENTION_KV_STRIDE_MODE BATCH_ATTENTION_KV_STRIDE_MODE_{mode}"
        in config
    )
    assert ("-DFLASHINFER_ENABLE_PROFILER" in spec.extra_cuda_cflags) == profiler
    _assert_persistent_module_surface(directory, variants)


def test_custom_batch_attention_preserves_caller_uri_and_dual_surface(
    generated_sources,
):
    uri = "test_custom_batch_attention_runtime_compatibility"
    spec = attention_modules.gen_customize_batch_attention_module(
        uri,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        64,
        64,
        ["maybe_k_cache_sf", "maybe_v_cache_sf"],
        ["uint8_t", "uint8_t"],
        [],
        [],
        "StandardAttention<false>",
        "#include <flashinfer/attention/variants.cuh>",
    )
    assert spec.name == uri and len(spec.sources) == 6
    directory = generated_sources / uri
    config = (directory / "batch_attention_config.inc").read_text()
    assert (
        "#define BATCH_ATTENTION_KV_STRIDE_MODE BATCH_ATTENTION_KV_STRIDE_MODE_RUNTIME"
        in config
    )
    _assert_persistent_module_surface(directory, ("true", "false"))


def test_persistent_warmup_contains_equal_primaries_only(generated_sources):
    from tests.test_helpers.jit_utils import gen_persistent_batch_attention_modules

    specs = gen_persistent_batch_attention_modules(
        [torch.float16, torch.bfloat16],
        [torch.float16, torch.bfloat16],
        [64, 128, 256],
        [False, True],
    )
    assert len(specs) == 12
    assert len({spec.name for spec in specs}) == 12
    assert all(spec.name.endswith("_kv_stride_equal") for spec in specs)
