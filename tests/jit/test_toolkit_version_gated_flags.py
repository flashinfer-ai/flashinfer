"""CPU-only tests for CUDA-toolkit-version-gated JIT compilation flags.

Flags such as -DENABLE_FP4 and -DENABLE_FP8_BLOCK_SCALE are dropped from JIT
nvcc flag lists when the local CUDA toolkit is older than 12.8. Issue #3951:
this used to happen silently, and the resulting module later failed deep in
kernel-runner construction with a misleading dtype-combination error. These
tests pin the guarded behavior: the drop must warn loudly at flag-generation
time, naming the toolkit version found and the required one, unless a prebuilt
AOT module will serve the kernels anyway (then the drop is info-level), and
capability probes must report the gap. The toolkit-version probe is
monkeypatched so no nvcc or GPU is needed.
"""

import logging
from types import SimpleNamespace

import pytest
from packaging.version import Version

from flashinfer.jit import core, cpp_ext
from flashinfer.jit import env as jit_env
from flashinfer.jit import fp4_quantization as jit_fp4_quantization
from flashinfer.jit import fused_moe as jit_fused_moe
from flashinfer.jit.gemm import fp8_blockscale as jit_fp8_blockscale
from flashinfer.quantization import fp4_quantization as quantization_fp4_quantization

CPP_EXT_LOGGER = "flashinfer.jit.cpp_ext"


@pytest.fixture(autouse=True)
def _isolated_capability_state(monkeypatch, tmp_path):
    # Isolate from the host's flashinfer-jit-cache install (AOT presence
    # changes warning behavior) and from the cached capability probe.
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_DIR", tmp_path / "aot-isolated")
    jit_fused_moe.cutlass_fused_moe_fp8_block_scale_supported.cache_clear()
    jit_fp4_quantization.has_fp4_support.cache_clear()
    yield
    jit_fused_moe.cutlass_fused_moe_fp8_block_scale_supported.cache_clear()
    jit_fp4_quantization.has_fp4_support.cache_clear()


def _set_cuda_version(monkeypatch, version_str: str) -> None:
    monkeypatch.setattr(cpp_ext, "get_cuda_version", lambda: Version(version_str))


def _make_aot_module(monkeypatch, tmp_path, module_name: str) -> None:
    aot_dir = tmp_path / "aot"
    (aot_dir / module_name).mkdir(parents=True)
    (aot_dir / module_name / f"{module_name}.so").touch()
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_DIR", aot_dir)


def _cpp_ext_records(caplog):
    return [r for r in caplog.records if r.name == CPP_EXT_LOGGER]


def test_version_gated_nvcc_flag_kept_when_toolkit_new_enough(monkeypatch, caplog):
    _set_cuda_version(monkeypatch, "12.8")
    with caplog.at_level(logging.WARNING, logger=CPP_EXT_LOGGER):
        flag = cpp_ext.version_gated_nvcc_flag("-DENABLE_FP4", "12.8", "some_module")

    assert flag == "-DENABLE_FP4"
    assert not _cpp_ext_records(caplog)


def test_version_gated_nvcc_flag_dropped_with_loud_warning(monkeypatch, caplog):
    _set_cuda_version(monkeypatch, "12.4")
    with caplog.at_level(logging.WARNING, logger=CPP_EXT_LOGGER):
        flag = cpp_ext.version_gated_nvcc_flag("-DENABLE_FP4", "12.8", "some_module")

    assert flag == ""
    warnings = [r for r in _cpp_ext_records(caplog) if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "some_module" in message
    assert "-DENABLE_FP4" in message
    assert "12.4" in message
    assert "12.8" in message


def test_version_gated_nvcc_flag_demoted_to_info_with_aot_module(
    monkeypatch, caplog, tmp_path
):
    _make_aot_module(monkeypatch, tmp_path, "some_module")
    _set_cuda_version(monkeypatch, "12.4")

    with caplog.at_level(logging.INFO, logger=CPP_EXT_LOGGER):
        flag = cpp_ext.version_gated_nvcc_flag("-DENABLE_FP4", "12.8", "some_module")

    # The flag is still dropped from the (never-compiled) JIT flag list, but
    # the module will be served from the AOT cache, so no warning fires.
    assert flag == ""
    records = _cpp_ext_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    message = records[0].getMessage()
    assert "some_module" in message
    assert "-DENABLE_FP4" in message
    assert "AOT" in message


def _capture_cutlass_fused_moe_flags(monkeypatch):
    captured = {}

    def fake_gen(nvcc_flags, device_arch, use_fast_build=False):
        captured["flags"] = nvcc_flags
        captured["arch"] = device_arch
        return SimpleNamespace(name=f"fused_moe_{device_arch}")

    monkeypatch.setattr(jit_fused_moe, "gen_cutlass_fused_moe_module", fake_gen)
    return captured


def test_fused_moe_sm90_drops_fp8_block_scale_with_warning_below_12_8(
    monkeypatch, caplog
):
    _set_cuda_version(monkeypatch, "12.4")
    captured = _capture_cutlass_fused_moe_flags(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=CPP_EXT_LOGGER):
        jit_fused_moe.gen_cutlass_fused_moe_sm90_module()

    assert "-DENABLE_FP8_BLOCK_SCALE" not in captured["flags"]
    # FP4 on SM90 goes through cutlass::float_e2m1_t (fp4_compat.h) and must
    # stay enabled regardless of the toolkit version (issue #3951 regression).
    assert "-DENABLE_FP4" in captured["flags"]
    warnings = [r.getMessage() for r in _cpp_ext_records(caplog)]
    assert any(
        "fused_moe_90" in m and "-DENABLE_FP8_BLOCK_SCALE" in m and "12.8" in m
        for m in warnings
    )


def test_fused_moe_sm90_keeps_fp8_block_scale_at_12_8(monkeypatch, caplog):
    _set_cuda_version(monkeypatch, "12.8")
    captured = _capture_cutlass_fused_moe_flags(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=CPP_EXT_LOGGER):
        jit_fused_moe.gen_cutlass_fused_moe_sm90_module()

    assert "-DENABLE_FP8_BLOCK_SCALE" in captured["flags"]
    assert "-DENABLE_FP4" in captured["flags"]
    assert not _cpp_ext_records(caplog)


def test_fused_moe_sm90_no_warning_with_aot_module_below_12_8(
    monkeypatch, caplog, tmp_path
):
    # The recommended install: flashinfer-jit-cache wheel plus an old local
    # toolkit. The gated kernels come from the AOT module, so generating the
    # (unused) JIT flags must not warn.
    _make_aot_module(monkeypatch, tmp_path, "fused_moe_90")
    _set_cuda_version(monkeypatch, "12.4")
    captured = _capture_cutlass_fused_moe_flags(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=CPP_EXT_LOGGER):
        jit_fused_moe.gen_cutlass_fused_moe_sm90_module()

    assert "-DENABLE_FP8_BLOCK_SCALE" not in captured["flags"]
    assert not [r for r in _cpp_ext_records(caplog) if r.levelno >= logging.WARNING]


def test_fp8_blockscale_gemm_sm90_drops_flag_with_warning_below_12_8(
    monkeypatch, caplog
):
    _set_cuda_version(monkeypatch, "12.4")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)

    with caplog.at_level(logging.WARNING, logger=CPP_EXT_LOGGER):
        spec = jit_fp8_blockscale.gen_fp8_blockscale_gemm_sm90_module()

    assert "-DENABLE_FP8_BLOCK_SCALE" not in spec.extra_cuda_cflags
    warnings = [r.getMessage() for r in _cpp_ext_records(caplog)]
    assert len(warnings) == 1
    assert "fp8_blockscale_gemm_90" in warnings[0]
    assert "-DENABLE_FP8_BLOCK_SCALE" in warnings[0]
    assert "12.4" in warnings[0]
    assert "12.8" in warnings[0]


def test_fp8_blockscale_gemm_sm90_keeps_flag_at_12_8(monkeypatch, caplog):
    _set_cuda_version(monkeypatch, "12.8")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)

    with caplog.at_level(logging.WARNING, logger=CPP_EXT_LOGGER):
        spec = jit_fp8_blockscale.gen_fp8_blockscale_gemm_sm90_module()

    assert "-DENABLE_FP8_BLOCK_SCALE" in spec.extra_cuda_cflags
    assert not _cpp_ext_records(caplog)


@pytest.mark.parametrize(
    "gen_module",
    [
        jit_fp4_quantization.gen_fp4_quantization_module,
        quantization_fp4_quantization.gen_fp4_quantization_module,
    ],
    ids=["jit", "quantization"],
)
def test_fp4_quantization_drops_fp4_with_single_warning_below_12_8(
    monkeypatch, caplog, gen_module
):
    _set_cuda_version(monkeypatch, "12.4")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)

    with caplog.at_level(logging.WARNING, logger=CPP_EXT_LOGGER):
        spec = gen_module([], "100")

    assert "-DENABLE_FP4" not in spec.extra_cuda_cflags
    assert "-DENABLE_FP4" not in spec.extra_cflags
    warnings = [
        r.getMessage()
        for r in _cpp_ext_records(caplog)
        if "fp4_quantization_100" in r.getMessage()
    ]
    # One warning per generated module, not one per flag list.
    assert len(warnings) == 1
    assert "-DENABLE_FP4" in warnings[0]
    assert "12.4" in warnings[0]
    assert "12.8" in warnings[0]


@pytest.mark.parametrize(
    "gen_module",
    [
        jit_fp4_quantization.gen_fp4_quantization_module,
        quantization_fp4_quantization.gen_fp4_quantization_module,
    ],
    ids=["jit", "quantization"],
)
def test_fp4_quantization_keeps_fp4_at_12_8(monkeypatch, caplog, gen_module):
    _set_cuda_version(monkeypatch, "12.8")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)

    with caplog.at_level(logging.WARNING, logger=CPP_EXT_LOGGER):
        spec = gen_module([], "100")

    assert "-DENABLE_FP4" in spec.extra_cuda_cflags
    assert "-DENABLE_FP4" in spec.extra_cflags
    assert not _cpp_ext_records(caplog)


def test_fp8_block_scale_supported_follows_toolkit_version(monkeypatch, tmp_path):
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_DIR", tmp_path / "aot")

    _set_cuda_version(monkeypatch, "12.4")
    assert jit_fused_moe.cutlass_fused_moe_fp8_block_scale_supported("90") is False

    jit_fused_moe.cutlass_fused_moe_fp8_block_scale_supported.cache_clear()
    _set_cuda_version(monkeypatch, "12.8")
    assert jit_fused_moe.cutlass_fused_moe_fp8_block_scale_supported("90") is True


def test_fp8_block_scale_supported_with_aot_module_and_old_toolkit(
    monkeypatch, tmp_path
):
    _make_aot_module(monkeypatch, tmp_path, "fused_moe_90")

    _set_cuda_version(monkeypatch, "12.4")
    assert jit_fused_moe.cutlass_fused_moe_fp8_block_scale_supported("90") is True


def test_has_fp4_support_follows_toolkit_version(monkeypatch, tmp_path):
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_DIR", tmp_path / "aot")

    _set_cuda_version(monkeypatch, "12.4")
    assert jit_fp4_quantization.has_fp4_support("100") is False

    jit_fp4_quantization.has_fp4_support.cache_clear()
    _set_cuda_version(monkeypatch, "12.8")
    # No-argument form requested in issue #3951; defaults to arch "100".
    assert jit_fp4_quantization.has_fp4_support() is True


def test_has_fp4_support_with_aot_module_and_old_toolkit(monkeypatch, tmp_path):
    _make_aot_module(monkeypatch, tmp_path, "fp4_quantization_100")

    _set_cuda_version(monkeypatch, "12.4")
    assert jit_fp4_quantization.has_fp4_support("100") is True
    # AOT presence is per-module: other arches still fall back to the
    # toolkit-version check.
    assert jit_fp4_quantization.has_fp4_support("90") is False


def test_has_fp4_support_is_exported_from_flashinfer_jit():
    # Issue #3951 asks for flashinfer.jit.has_fp4_support().
    import flashinfer.jit

    assert flashinfer.jit.has_fp4_support is jit_fp4_quantization.has_fp4_support
