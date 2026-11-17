from types import SimpleNamespace

import pytest

import flashinfer.jit.core as jit_core
from flashinfer.compilation_context import CompilationContext


@pytest.mark.cpu_only
def test_check_cuda_arch_refreshes_stale_global_context(monkeypatch):
    stale_context = SimpleNamespace(TARGET_CUDA_ARCHS=set())
    fresh_context = SimpleNamespace(TARGET_CUDA_ARCHS={(12, "0f")})

    monkeypatch.setattr(jit_core, "current_compilation_context", stale_context)
    monkeypatch.setattr(jit_core, "CompilationContext", lambda: fresh_context)

    jit_core.check_cuda_arch()

    assert {(12, "0f")} == jit_core.current_compilation_context.TARGET_CUDA_ARCHS


@pytest.mark.cpu_only
def test_check_cuda_arch_reuses_populated_context(monkeypatch):
    cached_context = SimpleNamespace(TARGET_CUDA_ARCHS={(12, "0f")})

    def unexpected_probe():
        raise AssertionError("A populated context must not re-probe CUDA")

    monkeypatch.setattr(jit_core, "current_compilation_context", cached_context)
    monkeypatch.setattr(jit_core, "CompilationContext", unexpected_probe)

    jit_core.check_cuda_arch()

    assert jit_core.current_compilation_context is cached_context
    assert {(12, "0f")} == cached_context.TARGET_CUDA_ARCHS


@pytest.mark.cpu_only
def test_check_cuda_arch_error_reports_detected_archs(monkeypatch):
    stale_context = SimpleNamespace(TARGET_CUDA_ARCHS=set())
    empty_context = SimpleNamespace(TARGET_CUDA_ARCHS=set())

    monkeypatch.setattr(jit_core, "current_compilation_context", stale_context)
    monkeypatch.setattr(jit_core, "CompilationContext", lambda: empty_context)
    monkeypatch.delenv("FLASHINFER_CUDA_ARCH_LIST", raising=False)

    with pytest.raises(RuntimeError, match=r"Detected TARGET_CUDA_ARCHS=\[\]"):
        jit_core.check_cuda_arch()


@pytest.mark.cpu_only
def test_normalize_sm120_uses_cuda_12_8_compatible_suffix(monkeypatch):
    monkeypatch.setattr(
        "flashinfer.jit.cpp_ext.is_cuda_version_at_least",
        lambda version: version == "12.8",
    )

    assert CompilationContext._normalize_cuda_arch(12, 0) == (12, "0a")


@pytest.mark.cpu_only
def test_normalize_sm120_prefers_f_suffix_on_cuda_12_9(monkeypatch):
    monkeypatch.setattr(
        "flashinfer.jit.cpp_ext.is_cuda_version_at_least", lambda _version: True
    )

    assert CompilationContext._normalize_cuda_arch(12, 0) == (12, "0f")


@pytest.mark.cpu_only
def test_normalize_sm121_requires_cuda_12_9(monkeypatch):
    monkeypatch.setattr(
        "flashinfer.jit.cpp_ext.is_cuda_version_at_least",
        lambda version: version == "12.8",
    )

    with pytest.raises(RuntimeError, match=r"SM 12.1\+ requires CUDA >= 12.9"):
        CompilationContext._normalize_cuda_arch(12, 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "arch_list",
    ["8.0 9.0a 12.0f", " 8.0 9.0a 12.0f ", "8.0  9.0a\t12.0f\n"],
)
def test_parse_cuda_arch_list_ignores_extra_whitespace(arch_list):
    assert CompilationContext._parse_cuda_arch_list(arch_list) == {
        (8, "0"),
        (9, "0a"),
        (12, "0f"),
    }


@pytest.mark.cpu_only
def test_parse_cuda_arch_list_normalizes_entries_without_suffix():
    assert CompilationContext._parse_cuda_arch_list("9.0 10.3") == {
        (9, "0a"),
        (10, "3a"),
    }


@pytest.mark.cpu_only
@pytest.mark.parametrize("entry", ["8.0,9.0", "90", "sm90", "9.", ".0", "9.0ab", "9.a"])
def test_parse_cuda_arch_list_rejects_malformed_entry(entry):
    with pytest.raises(ValueError, match="FLASHINFER_CUDA_ARCH_LIST") as excinfo:
        CompilationContext._parse_cuda_arch_list(f"8.0 {entry}")
    assert repr(entry) in str(excinfo.value)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch_list", ["", "   "])
def test_parse_cuda_arch_list_rejects_empty_value(arch_list):
    with pytest.raises(ValueError, match="FLASHINFER_CUDA_ARCH_LIST is set but empty"):
        CompilationContext._parse_cuda_arch_list(arch_list)


@pytest.mark.cpu_only
def test_compilation_context_reads_arch_list_with_trailing_space(monkeypatch):
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "8.0 9.0a ")

    assert {(8, "0"), (9, "0a")} == CompilationContext().TARGET_CUDA_ARCHS
