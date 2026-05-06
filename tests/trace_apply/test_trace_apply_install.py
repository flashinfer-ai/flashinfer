"""Integration test for trace_apply.install() dispatch (no GPU required).

Builds a tiny on-disk FlashInfer Trace (one rmsnorm definition + a pure-Python
solution + one PASSED record), points FLASHINFER_TRACE_PATH at it, installs, and
asserts that calling the *real* public ``flashinfer.norm.rmsnorm`` is routed to
the trace solution (a hit) rather than the GPU kernel. ``current_sm`` is
monkeypatched so the test runs on CPU-only machines.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import sys

import flashinfer
import flashinfer.trace_apply as ta
import flashinfer.trace_apply.install  # noqa: F401 — ensure submodule in sys.modules
import flashinfer.trace_apply.runtime  # noqa: F401

# The package re-exports a function named ``install``, shadowing the submodule
# attribute; fetch the real module objects from sys.modules to monkeypatch them.
install_mod = sys.modules["flashinfer.trace_apply.install"]
runtime_mod = sys.modules["flashinfer.trace_apply.runtime"]

FI_API = "flashinfer.norm.rmsnorm"
SENTINEL = 4242.0


def _write_trace(root: Path) -> None:
    (root / "definitions" / "rmsnorm").mkdir(parents=True, exist_ok=True)
    (root / "solutions" / "rmsnorm").mkdir(parents=True, exist_ok=True)
    (root / "traces" / "rmsnorm").mkdir(parents=True, exist_ok=True)

    (root / "definitions" / "rmsnorm" / "rmsnorm_h16.json").write_text(
        json.dumps(
            {
                "name": "rmsnorm_h16",
                "op_type": "rmsnorm",
                "tags": [f"fi_api:{FI_API}", "status:verified"],
                "axes": {
                    "batch_size": {"type": "var"},
                    "hidden_size": {"type": "const", "value": 16},
                },
                "inputs": {
                    "hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
                    "weight": {"shape": ["hidden_size"], "dtype": "float32"},
                },
                "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}},
            }
        )
    )
    # Candidate ignores the math and returns a sentinel-filled tensor shaped like
    # the input, so the test can prove the candidate (not the kernel) ran.
    # Candidate is invoked with Definition-input names (rmsnorm: hidden_states, weight).
    src = (
        "import torch\n"
        "def run(hidden_states, weight, **kwargs):\n"
        f"    return torch.full_like(hidden_states, {SENTINEL})\n"
    )
    (root / "solutions" / "rmsnorm" / "py_sentinel.json").write_text(
        json.dumps(
            {
                "name": "py_sentinel",
                "definition": "rmsnorm_h16",
                "author": "tester",
                "spec": {
                    "language": "python",
                    "target_hardware": ["NVIDIA B200"],
                    "entry_point": "main.py::run",
                },
                "sources": [{"path": "main.py", "content": src}],
            }
        )
    )
    (root / "traces" / "rmsnorm" / "records.jsonl").write_text(
        json.dumps(
            {
                "definition": "rmsnorm_h16",
                "solution": "py_sentinel",
                "workload": {"axes": {"batch_size": 8}, "uuid": "w0"},
                "evaluation": {
                    "status": "PASSED",
                    "environment": {"hardware": "NVIDIA B200", "libs": {"torch": "2.11.0"}},
                    "performance": {"latency_ms": 0.05, "reference_latency_ms": 1.0},
                },
            }
        )
        + "\n"
    )


def test_install_dispatches_to_trace_solution(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # hermetic solution cache
    root = tmp_path / "trace"
    _write_trace(root)

    # Pretend we are on a B200 (sm100) regardless of the test host.
    monkeypatch.setattr(install_mod, "current_sm", lambda *a, **k: "sm100")
    monkeypatch.setattr(runtime_mod, "current_sm", lambda *a, **k: "sm100")

    try:
        n = ta.install(path=str(root))
        assert ta.is_installed()
        assert n >= 1  # rmsnorm (and any module-level aliases) wrapped

        # The full concrete axes for this call are {hidden_size:16, batch_size:8},
        # which match the indexed record → candidate should run.
        inp = torch.ones(8, 16)
        w = torch.ones(16)
        out = flashinfer.norm.rmsnorm(inp, w)
        assert torch.allclose(out, torch.full_like(inp, SENTINEL)), "candidate did not run"

        # A top-level alias (flashinfer.rmsnorm) should also be patched.
        out2 = flashinfer.rmsnorm(inp, w)
        assert torch.allclose(out2, torch.full_like(inp, SENTINEL))

        stats = ta.stats()
        assert stats.get(FI_API, {}).get("hit", 0) >= 1

        # explain() traces the selection back to the record.
        ex = ta.explain(FI_API, {"hidden_size": 16, "batch_size": 8}, "sm100")
        assert ex["selected"]["solution"] == "py_sentinel"
        assert ex["selected"]["author"] == "tester"
        # An unknown shape resolves to no candidate (miss path; not executed
        # here to avoid invoking the real CUDA kernel on a CPU test host).
        ex_miss = ta.explain(FI_API, {"hidden_size": 16, "batch_size": 99}, "sm100")
        assert ex_miss["selected"] is None
    finally:
        ta.disable()
        assert not ta.is_installed()
        # rmsnorm restored to the real implementation.
        assert getattr(flashinfer.norm.rmsnorm, "_trace_apply", False) is False
