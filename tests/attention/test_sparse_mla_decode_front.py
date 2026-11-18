"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Dispatch / structure tests for the generalized sparse-MLA decode front.
Everything here is GPU-free: the underlying backends are stubbed and only
the front's routing, refusal, and pass-through contracts are exercised.
The per-backend numerics live with each backend's own test suite
(test_sparse_mla_sm90.py and the SM120 suites) and need silicon.
"""

import pytest
import torch

import flashinfer.mla as mla
from flashinfer.mla import SPARSE_MLA_DECODE_BACKENDS, sparse_mla_decode
from flashinfer.mla import _sparse_decode as front


@pytest.fixture(autouse=True)
def _clean():
    front._reset_sparse_mla_decode_stats()
    yield
    front._reset_sparse_mla_decode_stats()


def _cpu_call_args():
    q = torch.zeros(2, 1, 8, 576, dtype=torch.bfloat16)
    kv = torch.zeros(4, 64, 656, dtype=torch.uint8)
    idx = torch.full((2, 1, 2048), -1, dtype=torch.int32)
    return q, kv, idx


# --------------------------------------------------------------------------
# Surface structure
# --------------------------------------------------------------------------


def test_backends_tuple_is_plain_strings():
    assert SPARSE_MLA_DECODE_BACKENDS == (
        "auto",
        "specialized",
        "sm120",
    )
    assert all(isinstance(b, str) for b in SPARSE_MLA_DECODE_BACKENDS)


def test_lazy_exports_present():
    assert "sparse_mla_decode" in dir(mla)
    assert "SPARSE_MLA_DECODE_BACKENDS" in dir(mla)
    assert "is_sparse_mla_decode_supported" in dir(mla)
    assert mla.sparse_mla_decode is sparse_mla_decode


def test_unknown_backend_raises_and_never_swaps():
    q, kv, idx = _cpu_call_args()
    with pytest.raises(ValueError, match="never silently swapped"):
        sparse_mla_decode(q, kv, idx, backend="fa2")


def test_dsv4_is_not_a_backend():
    # The DeepSeek-V4 windowed dual-pool contract is a different op (the
    # kernel derives SWA validity from seq_lens and partitions the index
    # table); it must not be reachable through this front.
    q, kv, idx = _cpu_call_args()
    with pytest.raises(ValueError, match="never silently swapped"):
        sparse_mla_decode(q, kv, idx, backend="trtllm_dsv4")
    assert "trtllm_dsv4" not in SPARSE_MLA_DECODE_BACKENDS


def test_supported_false_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CPU-only check")
    assert front.is_sparse_mla_decode_supported(torch.device("cpu")) is False
    assert front.is_sparse_mla_decode_supported(torch.device("cuda")) is False


def test_auto_on_cpu_tensor_raises():
    q, kv, idx = _cpu_call_args()
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        sparse_mla_decode(q, kv, idx, backend="auto")


# --------------------------------------------------------------------------
# Auto routing by compute capability (capability probe stubbed)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cc,expected",
    [
        ((9, 0), "specialized"),
        ((12, 0), "sm120"),
        ((12, 1), "sm120"),
    ],
)
def test_auto_routes_by_capability(monkeypatch, cc, expected):
    monkeypatch.setattr(front, "_capability", lambda device: cc)
    assert front._resolve_backend("auto", torch.device("cpu")) == expected
    stats = front._sparse_mla_decode_stats()
    assert stats["auto_routes"] == {f"{cc[0]}.{cc[1]}->{expected}": 1}


@pytest.mark.parametrize("cc", [(8, 0), (10, 0), (10, 3), (10, 7)])
def test_auto_refuses_unrouted_capability(monkeypatch, cc):
    # SM100/SM103/SM107 deliberately have NO route: their in-tree
    # sparse-MLA decode entry is the DeepSeek-V4 op, not this one.
    monkeypatch.setattr(front, "_capability", lambda device: cc)
    with pytest.raises(ValueError, match="no selection-only"):
        front._resolve_backend("auto", torch.device("cpu"))


def test_explicit_backend_skips_capability_probe(monkeypatch):
    def _boom(device):
        raise AssertionError("capability probed for an explicit backend")

    monkeypatch.setattr(front, "_capability", _boom)
    for name in ("specialized", "sm120"):
        assert front._resolve_backend(name, torch.device("cpu")) == name


# --------------------------------------------------------------------------
# Per-backend adapter dispatch (adapters stubbed; explicit never swapped)
# --------------------------------------------------------------------------


def test_dispatch_reaches_selected_adapter_only(monkeypatch):
    q, kv, idx = _cpu_call_args()
    calls = []
    monkeypatch.setattr(
        front, "_run_specialized", lambda *a, **k: calls.append("specialized")
    )
    monkeypatch.setattr(front, "_run_sm120", lambda *a, **k: calls.append("sm120"))
    sparse_mla_decode(q, kv, idx, backend="specialized")
    sparse_mla_decode(q, kv, idx, backend="sm120")
    assert calls == ["specialized", "sm120"]
    stats = front._sparse_mla_decode_stats()
    assert stats["calls"] == 2
    assert stats["backend_dispatches"] == {
        "specialized": 1,
        "sm120": 1,
    }


def test_specialized_adapter_pins_inner_explicit_backend(monkeypatch):
    q, kv, idx = _cpu_call_args()
    seen = {}

    def fake_sm90(query, kv_cache, indices, seq_lens, workspace, **kw):
        seen.update(kw)
        return query

    import flashinfer.mla._sparse_mla_sm90 as sm90

    monkeypatch.setattr(sm90, "sparse_mla_decode_sm90", fake_sm90)
    seq = torch.zeros(2, dtype=torch.int32)
    ws = torch.zeros(8, dtype=torch.uint8)
    sparse_mla_decode(q, kv, idx, seq, ws, sm_scale=0.0625, backend="specialized")
    assert seen["backend"] == "specialized"
    seen.clear()
    # The front's auto defers to the SM90 entry's own allowlist.
    front._run_specialized(q, kv, idx, seq, ws, 0.0625, None, "auto", {})
    assert seen["backend"] == "auto"


# --------------------------------------------------------------------------
# Refusals: the front never papers over contract differences
# --------------------------------------------------------------------------


def test_specialized_requires_routing_core():
    q, kv, idx = _cpu_call_args()
    seq = torch.zeros(2, dtype=torch.int32)
    ws = torch.zeros(8, dtype=torch.uint8)
    with pytest.raises(ValueError, match="requires seq_lens"):
        sparse_mla_decode(q, kv, idx, None, ws, sm_scale=1.0, backend="specialized")
    with pytest.raises(ValueError, match="requires\\s+workspace_buffer"):
        sparse_mla_decode(q, kv, idx, seq, None, sm_scale=1.0, backend="specialized")
    with pytest.raises(ValueError, match="requires sm_scale"):
        sparse_mla_decode(q, kv, idx, seq, ws, backend="specialized")


def test_specialized_rejects_backend_args():
    q, kv, idx = _cpu_call_args()
    with pytest.raises(ValueError, match="does not accept"):
        sparse_mla_decode(
            q, kv, idx, backend="specialized", backend_args={"kv_layout": "HND"}
        )


def test_sm120_refuses_seq_lens_workspace_and_requires_sm_scale():
    q, kv, idx = _cpu_call_args()
    with pytest.raises(ValueError, match="topk_length"):
        sparse_mla_decode(
            q, kv, idx, torch.zeros(2, dtype=torch.int32), backend="sm120"
        )
    with pytest.raises(ValueError, match="does not take a\\s+workspace_buffer"):
        sparse_mla_decode(
            q,
            kv,
            idx,
            workspace_buffer=torch.zeros(8, dtype=torch.uint8),
            backend="sm120",
        )
    with pytest.raises(ValueError, match="requires sm_scale"):
        sparse_mla_decode(q, kv, idx, backend="sm120")
    with pytest.raises(ValueError, match="does not accept"):
        sparse_mla_decode(
            q,
            kv,
            idx,
            sm_scale=1.0,
            backend="sm120",
            backend_args={"sparse_topk_lens": torch.zeros(2)},
        )


# --------------------------------------------------------------------------
# Pass-through fidelity (underlying entries stubbed)
# --------------------------------------------------------------------------


def test_sm120_passthrough_caches_runner_and_allocates_out(monkeypatch):
    q = torch.zeros(3, 8, 576, dtype=torch.bfloat16)
    kv = torch.zeros(4, 584, dtype=torch.uint8)
    idx = torch.full((3, 2048), -1, dtype=torch.int32)
    made = []

    class FakeRunner:
        def __init__(self, **kw):
            made.append(kw)
            self.runs = []

        def run(self, q, kv_cache, indices, output, sm_scale, **kw):
            self.runs.append((q, kv_cache, indices, output, sm_scale, kw))
            return None

    import flashinfer.mla._sparse_mla_sm120 as sm120mod

    monkeypatch.setattr(sm120mod, "SparseMLASm120Wrapper", FakeRunner)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0, raising=False)
    front._sm120_runners.clear()
    topk = torch.full((3,), 16, dtype=torch.int32)
    out1 = sparse_mla_decode(
        q,
        kv,
        idx,
        sm_scale=0.0625,
        backend="sm120",
        backend_args={"topk_length": topk, "kv_cache_format": "fp8"},
    )
    out2 = sparse_mla_decode(
        q,
        kv,
        idx,
        sm_scale=0.0625,
        backend="sm120",
        backend_args={"topk_length": topk, "kv_cache_format": "fp8"},
    )
    # One construction, two runs: the front caches per (device, ctor knobs).
    assert len(made) == 1
    assert made[0]["kv_cache_format"] == "fp8"
    runner = next(iter(front._sm120_runners.values()))
    assert len(runner.runs) == 2
    assert runner.runs[0][5]["topk_length"] is topk
    # Allocated latent-space output: bf16, d_v=512 by default.
    assert out1.shape == (3, 8, 512) and out1.dtype == torch.bfloat16
    assert out2.shape == (3, 8, 512)
    front._sm120_runners.clear()
