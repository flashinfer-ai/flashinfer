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


def test_compilation_context_logs_probe_state_on_detection_failure(monkeypatch, caplog):
    monkeypatch.delenv("FLASHINFER_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr("torch.cuda.device_count", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    with caplog.at_level("WARNING"):
        context = CompilationContext()

    assert context.TARGET_CUDA_ARCHS == set()
    assert "Failed to get device capability" in caplog.text
    assert "torch.cuda.is_available()=True" in caplog.text
    assert "torch.cuda.device_count()=<error:" in caplog.text
