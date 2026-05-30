"""Integration test for trace_apply enable_apply() dispatch (no GPU required).

Builds a tiny on-disk solution folder (one rmsnorm definition + a pure-Python
solution), points FLASHINFER_TRACE_PATH at it, enables apply, and asserts that
calling the *real* public ``flashinfer.norm.rmsnorm`` is routed to the solution
(a hit) rather than the GPU kernel. ``current_sm`` is monkeypatched so the test
runs on CPU-only machines.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import flashinfer
import flashinfer.trace_apply as ta
import flashinfer.trace_apply.apply as apply_mod

FI_API = "flashinfer.norm.rmsnorm"
SENTINEL = 4242.0
# rmsnorm inputs are 'hidden_states' (param 'input') and 'weight', both float32.
DT_F32 = frozenset({("hidden_states", "float32"), ("weight", "float32")})


def _write_trace(root: Path) -> None:
    (root / "definitions" / "rmsnorm").mkdir(parents=True, exist_ok=True)
    (root / "solutions" / "rmsnorm").mkdir(parents=True, exist_ok=True)

    (root / "definitions" / "rmsnorm" / "rmsnorm_h16.json").write_text(
        json.dumps(
            {
                "name": "rmsnorm_h16",
                "op_type": "rmsnorm",
                "tags": [f"fi_api:{FI_API}"],
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
    # the input, so the test proves the candidate (not the kernel) ran. Invoked
    # with Definition-input names (rmsnorm: hidden_states, weight).
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
                    "target_hardware": ["sm100"],
                    "entry_point": "main.py::run",
                },
                "sources": [{"path": "main.py", "content": src}],
            }
        )
    )


def test_enable_apply_dispatches_to_solution(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # hermetic solution cache
    root = tmp_path / "trace"
    _write_trace(root)

    # Pretend we are on a B200 (sm100) regardless of the test host.
    monkeypatch.setattr(apply_mod, "current_sm", lambda *a, **k: "sm100")

    try:
        n = ta.enable_apply(path=str(root))
        assert ta.is_enabled()
        assert n >= 1

        w = torch.ones(16)
        out = flashinfer.norm.rmsnorm(torch.ones(8, 16), w)
        assert torch.allclose(out, torch.full_like(out, SENTINEL)), "candidate did not run"
        # different var (batch) still routes to the candidate — one solution.
        out_b = flashinfer.norm.rmsnorm(torch.ones(3, 16), w)
        assert torch.allclose(out_b, torch.full_like(out_b, SENTINEL))

        # A top-level alias (flashinfer.rmsnorm) should also be patched.
        out2 = flashinfer.rmsnorm(torch.ones(8, 16), w)
        assert torch.allclose(out2, torch.full_like(out2, SENTINEL))

        stats = ta.stats()
        assert stats.get(FI_API, {}).get("hit", 0) >= 1

        ex = ta.explain(FI_API, {"hidden_size": 16}, input_dtypes=DT_F32, sm_arch="sm100")
        assert ex["selected"]["solution"] == "py_sentinel"
        assert ex["selected"]["author"] == "tester"
        # an unregistered const-shape → miss
        ex_miss = ta.explain(FI_API, {"hidden_size": 32}, input_dtypes=DT_F32, sm_arch="sm100")
        assert ex_miss["selected"] is None
    finally:
        ta.disable_apply()
        assert not ta.is_enabled()
        assert getattr(flashinfer.norm.rmsnorm, "_trace_apply", False) is False


def test_strict_mode_matched_solution_error_raises(tmp_path, monkeypatch):
    """Strict mode: a matched solution that raises must propagate (not silently
    fall back to the original kernel)."""
    import pytest

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "trace"
    (root / "definitions" / "rmsnorm").mkdir(parents=True)
    (root / "solutions" / "rmsnorm").mkdir(parents=True)
    (root / "definitions" / "rmsnorm" / "rmsnorm_h16.json").write_text(
        json.dumps(
            {
                "name": "rmsnorm_h16",
                "op_type": "rmsnorm",
                "tags": [f"fi_api:{FI_API}"],
                "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 16}},
                "inputs": {
                    "hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
                    "weight": {"shape": ["hidden_size"], "dtype": "float32"},
                },
                "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}},
            }
        )
    )
    (root / "solutions" / "rmsnorm" / "boom.json").write_text(
        json.dumps(
            {
                "name": "boom",
                "definition": "rmsnorm_h16",
                "author": "tester",
                "spec": {"language": "python", "target_hardware": ["sm100"], "entry_point": "main.py::run"},
                "sources": [{"path": "main.py", "content": "def run(hidden_states, weight, **k):\n    raise ValueError('boom')\n"}],
            }
        )
    )
    monkeypatch.setattr(apply_mod, "current_sm", lambda *a, **k: "sm100")
    try:
        ta.enable_apply(path=str(root))
        with pytest.raises(ValueError, match="boom"):
            flashinfer.norm.rmsnorm(torch.ones(8, 16), torch.ones(16))
        assert ta.stats().get(FI_API, {}).get("error", 0) >= 1
    finally:
        ta.disable_apply()
