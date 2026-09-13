import subprocess
from types import SimpleNamespace

import pytest
import torch

import flashinfer
from flashinfer.jit import core, cpp_ext
from flashinfer.jit.attention import modules as attention_modules
from flashinfer.utils import (
    PosEncodingMode,
    determine_attention_backend,
    is_fa3_prefill_head_dim_supported,
)
from tests.test_helpers import jit_utils


def test_nvcc_parallelism_flags_use_flashinfer_nvcc_threads(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "4")

    assert cpp_ext.get_nvcc_parallelism_flags() == ["--threads=4"]


def test_nvcc_parallelism_flags_ignore_sccache_launcher(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "4")
    monkeypatch.setenv("FLASHINFER_NVCC_LAUNCHER", "sccache")

    assert cpp_ext.get_nvcc_parallelism_flags() == ["--threads=4"]


def test_jit_uses_size_optimized_fatbin_compression(monkeypatch):
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(name="test_module", sources=[])

    assert "-Xfatbin=-compress-all" in spec.extra_cuda_cflags
    assert "--compress-mode=size" in spec.extra_cuda_cflags


def test_generate_ninja_uses_sccache_compatible_nvcc_depfile_flag(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: "/usr/local/cuda")
    monkeypatch.setattr(cpp_ext.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "7.5")

    ninja = cpp_ext.generate_ninja_build_for_op(
        name="test_module",
        sources=[tmp_path / "generated" / "kernel.cu"],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )

    assert "--generate-dependencies-with-compile -MF $out.d" in ninja
    assert "--dependency-output" not in ninja


def test_debug_jit_uses_sccache_compatible_nvcc_device_debug_flag(monkeypatch):
    monkeypatch.setenv("FLASHINFER_JIT_DEBUG", "1")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
    )

    assert "--device-debug" in spec.extra_cuda_cflags
    assert "-G" not in spec.extra_cuda_cflags


def test_release_jit_propagates_ndebug_to_host_cflags(monkeypatch):
    monkeypatch.delenv("FLASHINFER_JIT_DEBUG", raising=False)
    monkeypatch.delenv("FLASHINFER_JIT_VERBOSE", raising=False)
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
    )

    assert "-DNDEBUG" in spec.extra_cflags
    assert "-DNDEBUG" in spec.extra_cuda_cflags


def test_debug_jit_does_not_propagate_ndebug(monkeypatch):
    monkeypatch.setenv("FLASHINFER_JIT_DEBUG", "1")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
    )

    assert "-DNDEBUG" not in spec.extra_cflags
    assert "-DNDEBUG" not in spec.extra_cuda_cflags


def test_run_ninja_uses_max_jobs(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("MAX_JOBS", "8")
    monkeypatch.setattr(cpp_ext.subprocess, "run", fake_run)

    cpp_ext.run_ninja(tmp_path, tmp_path / "build.ninja", verbose=False)

    assert commands == [
        [
            "ninja",
            "-v",
            "-C",
            str(tmp_path.resolve()),
            "-f",
            str((tmp_path / "build.ninja").resolve()),
            "-j",
            "8",
        ]
    ]


@pytest.fixture
def jit_spec_nvcc(monkeypatch, tmp_path):
    monkeypatch.setattr(core.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    return core.JitSpecNvcc(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )


def test_jit_spec_build_rewrites_ninja_before_build(monkeypatch, jit_spec_nvcc):
    writes = []
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)

    spec = jit_spec_nvcc
    monkeypatch.setattr(spec, "write_ninja", lambda: writes.append(True))
    monkeypatch.setattr(core, "run_ninja", lambda *_args, **_kwargs: None)

    spec.build(verbose=False, need_lock=False)

    assert writes == [True]


@pytest.mark.parametrize("cached", [False, True])
def test_jit_spec_build_logs_only_for_cold_cache(monkeypatch, jit_spec_nvcc, cached):
    events = []
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)

    spec = jit_spec_nvcc
    if cached:
        spec.jit_library_path.parent.mkdir(parents=True)
        spec.jit_library_path.touch()
    monkeypatch.setattr(spec, "write_ninja", lambda: None)
    monkeypatch.setattr(
        core, "run_ninja", lambda *_args, **_kwargs: events.append("run_ninja")
    )
    monkeypatch.setattr(
        core.logger,
        "info_once",
        lambda message, *args: events.append(message % args),
    )

    spec.build(verbose=False, need_lock=False)

    expected = ["run_ninja"]
    if not cached:
        expected.insert(
            0,
            "Building JIT module test_module; this can take several minutes on "
            "first use.",
        )
    assert events == expected


def test_jit_spec_aot_cache_hit_does_not_log_jit_build(
    monkeypatch, tmp_path, jit_spec_nvcc
):
    logs = []
    cached_module = object()
    monkeypatch.setattr(core.jit_env, "FLASHINFER_AOT_DIR", tmp_path)

    spec = jit_spec_nvcc
    spec.aot_path.parent.mkdir(parents=True)
    spec.aot_path.touch()

    monkeypatch.setattr(spec, "load", lambda _path=None: cached_module)
    monkeypatch.setattr(core.logger, "info_once", logs.append)

    assert spec.build_and_load() is cached_module
    assert logs == []


@pytest.mark.parametrize("is_aot", [False, True])
def test_jit_spec_post_load_adapter_applies_to_jit_and_aot_load_paths(
    monkeypatch, tmp_path, is_aot
):
    raw_module = object()
    seen = []
    spec = core.JitSpecNvcc(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
        post_load_adapter=lambda module: seen.append(module) or ("wrapped", module),
    )
    monkeypatch.setattr(core.tvm_ffi, "load_module", lambda _path: raw_module)
    path = tmp_path / ("aot.so" if is_aot else "jit.so")

    assert spec.load(path if is_aot else None) == ("wrapped", raw_module)
    assert seen == [raw_module]


def test_customize_batch_prefill_nvfp4_large_head_uses_prefill_flags(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(
        attention_modules.current_compilation_context, "TARGET_CUDA_ARCHS", {(8, 6)}
    )
    monkeypatch.setattr(
        attention_modules.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "gen"
    )

    spec = attention_modules.gen_customize_batch_prefill_module(
        "fa2",
        "test_batch_prefill_nvfp4_large_head",
        torch.float16,
        torch.uint8,
        torch.float16,
        torch.int32,
        512,
        512,
        # NVFP4 (uint8) KV paged prefill now requires the scale-factor tensors as
        # additional inputs (maybe_k_cache_sf / maybe_v_cache_sf), matching the
        # generator contract; pass them so generation reaches the flag assertions.
        ["maybe_k_cache_sf", "maybe_v_cache_sf"],
        ["uint8_t", "uint8_t"],
        ["sm_scale"],
        ["double"],
        "DefaultAttention<false, false, false, false>",
        "#include <flashinfer/attention/variants.cuh>",
    )

    assert any("sm_86" in flag for flag in spec.extra_cuda_cflags)
    with pytest.raises(RuntimeError, match="No supported CUDA architectures"):
        attention_modules._fa2_head_dim_nvcc_flags(512, 512, torch.uint8)


@pytest.mark.parametrize(
    ("head_dim_qk", "head_dim_vo", "supported"),
    [
        (64, 64, True),
        (128, 128, True),
        (256, 256, True),
        (192, 128, True),
        (512, 512, False),
        (256, 128, False),
        (128, 192, False),
    ],
)
def test_fa3_prefill_head_dim_supported(head_dim_qk, head_dim_vo, supported):
    assert is_fa3_prefill_head_dim_supported(head_dim_qk, head_dim_vo) is supported


@pytest.mark.parametrize(
    ("head_dim_qk", "head_dim_vo", "expected_backend"),
    [
        (256, 256, "fa3"),
        (192, 128, "fa3"),
        (512, 512, "fa2"),
    ],
)
def test_determine_attention_backend_respects_fa3_prefill_head_dim(
    monkeypatch, head_dim_qk, head_dim_vo, expected_backend
):
    monkeypatch.setattr(flashinfer.utils, "is_sm90a_supported", lambda device: True)

    backend = determine_attention_backend(
        torch.device("cuda"),
        PosEncodingMode.NONE.value,
        use_fp16_qk_reductions=False,
        use_custom_mask=False,
        dtype_q=torch.float16,
        dtype_kv=torch.float16,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
    )

    assert backend == expected_backend


def test_prefill_jit_helper_skips_fa3_unsupported_large_head(monkeypatch):
    calls = []

    def fake_single_prefill_module(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim_qk,
        head_dim_vo,
        *_args,
    ):
        calls.append(("single", backend, head_dim_qk, head_dim_vo))
        return SimpleNamespace(name=f"{backend}_single_{head_dim_qk}_{head_dim_vo}")

    def fake_batch_prefill_module(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        idtype,
        head_dim_qk,
        head_dim_vo,
        *_args,
    ):
        calls.append(("batch", backend, head_dim_qk, head_dim_vo))
        return SimpleNamespace(name=f"{backend}_batch_{head_dim_qk}_{head_dim_vo}")

    monkeypatch.setattr(jit_utils, "is_sm90a_supported", lambda device: True)
    monkeypatch.setattr(
        flashinfer.prefill, "gen_single_prefill_module", fake_single_prefill_module
    )
    monkeypatch.setattr(
        flashinfer.prefill, "gen_batch_prefill_module", fake_batch_prefill_module
    )
    monkeypatch.setattr(
        flashinfer.quantization,
        "gen_quantization_module",
        lambda: SimpleNamespace(name="quantization"),
    )
    monkeypatch.setattr(
        flashinfer.page,
        "gen_page_module",
        lambda: SimpleNamespace(name="page"),
    )

    jit_utils.gen_prefill_attention_modules(
        q_dtypes=[torch.float16],
        kv_dtypes=[torch.float16],
        head_dims=[512],
        pos_encoding_modes=[PosEncodingMode.NONE.value],
        use_sliding_window_options=[False],
        use_logits_soft_cap_options=[False],
        use_fp16_qk_reduction_options=[False],
    )

    assert ("single", "fa3", 512, 512) not in calls
    assert ("batch", "fa3", 512, 512) not in calls
    assert ("single", "fa2", 512, 512) in calls
    assert ("batch", "fa2", 512, 512) in calls
