from types import SimpleNamespace

import pytest

import flashinfer.jit.core as jit_core
from flashinfer.compilation_context import CompilationContext


def test_check_cuda_arch_refreshes_stale_global_context(monkeypatch):
    stale_context = SimpleNamespace(TARGET_CUDA_ARCHS=set())
    fresh_context = SimpleNamespace(TARGET_CUDA_ARCHS={(12, "0f")})

    monkeypatch.setattr(jit_core, "current_compilation_context", stale_context)
    monkeypatch.setattr(jit_core, "CompilationContext", lambda: fresh_context)

    jit_core.check_cuda_arch()

    assert jit_core.current_compilation_context.TARGET_CUDA_ARCHS == {(12, "0f")}


def test_check_cuda_arch_error_reports_detected_archs(monkeypatch):
    stale_context = SimpleNamespace(TARGET_CUDA_ARCHS={(12, "0f")})
    empty_context = SimpleNamespace(TARGET_CUDA_ARCHS=set())

    monkeypatch.setattr(jit_core, "current_compilation_context", stale_context)
    monkeypatch.setattr(jit_core, "CompilationContext", lambda: empty_context)
    monkeypatch.delenv("FLASHINFER_CUDA_ARCH_LIST", raising=False)

    with pytest.raises(RuntimeError, match=r"Detected TARGET_CUDA_ARCHS=\[\]"):
        jit_core.check_cuda_arch()


def test_normalize_sm120_uses_cuda_12_8_compatible_suffix(monkeypatch):
    monkeypatch.setattr(
        "flashinfer.jit.cpp_ext.is_cuda_version_at_least",
        lambda version: version == "12.8",
    )

    assert CompilationContext._normalize_cuda_arch(12, 0) == (12, "0a")


def test_normalize_sm120_prefers_f_suffix_on_cuda_12_9(monkeypatch):
    monkeypatch.setattr("flashinfer.jit.cpp_ext.is_cuda_version_at_least", lambda _version: True)

    assert CompilationContext._normalize_cuda_arch(12, 0) == (12, "0f")


def test_normalize_sm121_requires_cuda_12_9(monkeypatch):
    monkeypatch.setattr(
        "flashinfer.jit.cpp_ext.is_cuda_version_at_least",
        lambda version: version == "12.8",
    )

    with pytest.raises(RuntimeError, match=r"SM 12.1\+ requires CUDA >= 12.9"):
        CompilationContext._normalize_cuda_arch(12, 1)
