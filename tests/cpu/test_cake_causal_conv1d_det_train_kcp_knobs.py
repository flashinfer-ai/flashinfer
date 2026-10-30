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

CPU tests of the conv KCP module's private test knobs and of the bookkeeping of its CUDA-graph replay cache
(the graph replay of the KCP route): the knob is refused without ``CAKE_KCP_TEST_KNOBS=1``, validated against
``GRAPH_MODES`` and off by default; the validity key tells operands apart by data pointer, shape, stride and
dtype, keeps one entry per key, and the first call of a key runs eagerly.  No kernel runs here.
"""

from __future__ import annotations

import types

import pytest
import torch

from flashinfer.cake_causal_conv1d_det_train_backend import kcp as conv_kcp


@pytest.mark.cpu_only
def test_graph_knob_is_gated_and_validated(monkeypatch) -> None:
    monkeypatch.delenv(conv_kcp.TEST_KNOBS_ENV, raising=False)
    assert conv_kcp.GRAPH_MODES == ("off", "replay")
    assert conv_kcp._set_test_knobs(None) == {"graph": "off", "host_trim": 1}
    with pytest.raises(ValueError, match="CAKE_KCP_TEST_KNOBS"):
        conv_kcp._set_test_knobs({"graph": "replay"})
    with pytest.raises(ValueError, match="CAKE_KCP_TEST_KNOBS"):
        conv_kcp._set_test_knobs({"host_trim": 1})
    assert conv_kcp._TEST_KNOBS == {"graph": "off", "host_trim": 1}
    monkeypatch.setenv(conv_kcp.TEST_KNOBS_ENV, "1")
    try:
        with pytest.raises(ValueError, match="unknown"):
            conv_kcp._set_test_knobs({"grap": "replay"})
        with pytest.raises(ValueError):
            conv_kcp._set_test_knobs({"graph": "capture"})
        with pytest.raises(TypeError):
            conv_kcp._set_test_knobs(["replay"])
        assert conv_kcp._set_test_knobs({"graph": "replay"}) == {
            "graph": "replay",
            "host_trim": 1,
        }
        assert conv_kcp._set_test_knobs({"graph": "off"}) == {
            "graph": "off",
            "host_trim": 1,
        }
    finally:
        conv_kcp._set_test_knobs(None)


@pytest.mark.cpu_only
def test_host_trim_knob_is_gated_and_validated(monkeypatch) -> None:
    """``host_trim`` (the trimmed host path): refused without ``CAKE_KCP_TEST_KNOBS=1``,
    validated against ``HOST_TRIMS`` (1 shipped = the trimmed host path of the eager halo call,
    0 = the previous host path), independent of ``graph``; a dict naming one knob keeps the other at its shipped default."""
    monkeypatch.delenv(conv_kcp.TEST_KNOBS_ENV, raising=False)
    assert conv_kcp.HOST_TRIMS == (0, 1) and conv_kcp.HOST_TRIM_DEFAULT == 1
    with pytest.raises(ValueError, match="CAKE_KCP_TEST_KNOBS"):
        conv_kcp._set_test_knobs({"host_trim": 0})
    assert conv_kcp._TEST_KNOBS["host_trim"] == 1
    monkeypatch.setenv(conv_kcp.TEST_KNOBS_ENV, "1")
    try:
        assert conv_kcp._set_test_knobs({"host_trim": 1}) == {
            "graph": "off",
            "host_trim": 1,
        }
        assert conv_kcp._graph_mode(None) == "off"
        assert conv_kcp._set_test_knobs({"host_trim": 1, "graph": "replay"}) == {
            "graph": "replay",
            "host_trim": 1,
        }
        assert conv_kcp._set_test_knobs({"graph": "replay"}) == {
            "graph": "replay",
            "host_trim": 1,
        }
        assert conv_kcp._set_test_knobs({}) == {"graph": "off", "host_trim": 1}
        for bad in (
            {"host_trim": 2},
            {"host_trim": "1"},
            {"host_trim": True},
            {"host_tri": 1},
        ):
            with pytest.raises(ValueError):
                conv_kcp._set_test_knobs(bad)
        assert conv_kcp._set_test_knobs({"host_trim": 0}) == {
            "graph": "off",
            "host_trim": 0,
        }
    finally:
        conv_kcp._set_test_knobs(None)
    assert conv_kcp._TEST_KNOBS == {"graph": "off", "host_trim": 1}


@pytest.mark.cpu_only
def test_graph_cache_key_and_eager_first_call(monkeypatch) -> None:
    monkeypatch.setenv(conv_kcp.TEST_KNOBS_ENV, "1")
    x = torch.empty(8, 16, dtype=torch.bfloat16)
    w = torch.empty(16, 4, dtype=torch.float32)
    ctx = types.SimpleNamespace(T_r=8, n_frag=1, world_size=1)
    assert (
        conv_kcp._graph_entry("fwd", ctx, (x, w, None)) is None
    )  # knob off: never captured
    conv_kcp._set_test_knobs({"graph": "replay"})
    try:
        e = conv_kcp._graph_entry("fwd", ctx, (x, w, None))
        assert e is not None and e.seen == 0 and e.graph is None
        assert e.digest.startswith("fwd:")
        assert conv_kcp._graph_entry("fwd", ctx, (x, w, None)) is e
        assert conv_kcp._graph_entry("bwd", ctx, (x, w, None)) is not e
        assert conv_kcp._graph_entry("fwd", ctx, (x.clone(), w, None)) is not e
        assert (
            conv_kcp._graph_entry("fwd", ctx, (x.t(), w, None)) is not e
        )  # same pointer, other stride
        assert (
            conv_kcp._graph_entry("fwd", ctx, (x.view(torch.int16), w, None)) is not e
        )
        assert conv_kcp._graph_entry("fwd", ctx, (x[:4], w, None)) is not e
        other = types.SimpleNamespace(T_r=8, n_frag=1, world_size=1)
        assert conv_kcp._graph_entry("fwd", other, (x, w, None)) is not e
        empty = types.SimpleNamespace(T_r=0, n_frag=0, world_size=1)
        assert (
            conv_kcp._graph_entry("fwd", empty, (x, w, None)) is None
        )  # a rank without tokens
        n0 = conv_kcp.GRAPH_STATS["eager"]
        assert conv_kcp._graph_run("fwd", e, lambda: ("y", "hist")) == ("y", "hist")
        assert e.seen == 1 and e.graph is None
        assert conv_kcp.GRAPH_STATS["eager"] == n0 + 1
        assert conv_kcp.LAST_CALL["fwd"] == {
            "graph": "replay",
            "graph_event": "eager",
            "graph_key": e.digest,
            "graph_fallback": None,
        }
        assert conv_kcp._graph_run("fwd", None, lambda: ("y", "hist")) == ("y", "hist")
        assert conv_kcp.LAST_CALL["fwd"]["graph_key"] is None
        # graph_cache_clear(): drops every entry (to be called before destroy_process_group), reports the count
        n_entries = len(conv_kcp._GRAPHS)
        assert n_entries >= 2 and "graph_cache_clear" in conv_kcp.__all__
        assert conv_kcp.graph_cache_clear() == n_entries
        assert conv_kcp._GRAPHS == {} and conv_kcp.graph_cache_clear() == 0
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp._GRAPHS.clear()
        conv_kcp.LAST_CALL.clear()


@pytest.mark.cpu_only
def test_graph_public_keyword_mode_and_key(monkeypatch) -> None:
    """``graph=`` is a public keyword of the conv KCP calls (no environment gate): validated against
    ``GRAPH_MODES``, ``None`` = the module's shipped default (``GRAPH_DEFAULT == "off"``), keyed and recorded like the
    knob; a keyword contradicting an active knob is a ``ValueError``.  No kernel runs here."""
    monkeypatch.delenv(conv_kcp.TEST_KNOBS_ENV, raising=False)
    conv_kcp._set_test_knobs(None)
    assert conv_kcp.GRAPH_DEFAULT == "off" and "GRAPH_DEFAULT" in conv_kcp.__all__
    assert conv_kcp._graph_mode(None) == "off"
    assert (
        conv_kcp._graph_mode("off") == "off"
        and conv_kcp._graph_mode("replay") == "replay"
    )
    with pytest.raises(ValueError, match="graph"):
        conv_kcp._graph_mode("capture")
    x = torch.empty(8, 16, dtype=torch.bfloat16)
    w = torch.empty(16, 4, dtype=torch.float32)
    ctx = types.SimpleNamespace(T_r=8, n_frag=1, world_size=1)
    for call in (  # validated on the caller's frame, before any device work
        lambda: conv_kcp.cake_causal_conv1d_det_kcp(x, w, ctx, graph="capture"),
        lambda: conv_kcp.cake_causal_conv1d_det_forward_kcp(x, w, ctx, graph="capture"),
        lambda: conv_kcp.cake_causal_conv1d_det_backward_kcp(
            x, w, ctx, x, x, graph="capture"
        ),
    ):
        with pytest.raises(ValueError, match="graph"):
            call()
    assert conv_kcp._graph_entry("fwd", ctx, (x, w, None), "off") is None
    try:
        e = conv_kcp._graph_entry(
            "fwd", ctx, (x, w, None), "replay"
        )  # the keyword keys the call with the knob off
        assert e is not None and e.seen == 0 and e.graph is None
        assert conv_kcp._graph_entry("fwd", ctx, (x, w, None), "replay") is e
        n0 = conv_kcp.GRAPH_STATS["eager"]
        assert conv_kcp._graph_run("fwd", e, lambda: ("y", "hist"), "replay") == (
            "y",
            "hist",
        )
        assert e.seen == 1 and conv_kcp.GRAPH_STATS["eager"] == n0 + 1
        assert conv_kcp.LAST_CALL["fwd"] == {
            "graph": "replay",
            "graph_event": "eager",
            "graph_key": e.digest,
            "graph_fallback": None,
        }
        monkeypatch.setenv(conv_kcp.TEST_KNOBS_ENV, "1")
        conv_kcp._set_test_knobs({"graph": "replay"})
        assert (
            conv_kcp._graph_mode(None) == "replay"
            and conv_kcp._graph_mode("replay") == "replay"
        )
        with pytest.raises(ValueError, match="twice"):
            conv_kcp._graph_mode("off")
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp._GRAPHS.clear()
        conv_kcp.LAST_CALL.clear()


@pytest.mark.cpu_only
def test_host_trim_cache_is_per_context_and_bounded(monkeypatch) -> None:
    """The trimmed host path keeps one entry set per context object (by identity, the context pinned), bounded
    by ``HOST_CACHE_MAX`` (the oldest context dropped) and cleared by ``host_cache_clear()``."""
    conv_kcp.host_cache_clear()
    a, b = types.SimpleNamespace(), types.SimpleNamespace()
    key = ("fwd", 8, 16, "cpu", (0, 8))
    ea = conv_kcp._host_cache_entry(a, key)
    assert ea == {} and conv_kcp._host_cache_entry(a, key) is ea
    ea["x"] = 1
    eb = conv_kcp._host_cache_entry(b, key)
    assert eb == {} and eb is not ea
    assert conv_kcp._host_cache_entry(a, ("bwd", 8, 16, "cpu", (0, 8))) is not ea
    assert conv_kcp._host_cache_entry(a, key) is ea
    monkeypatch.setattr(conv_kcp, "HOST_CACHE_MAX", 2)
    conv_kcp._host_cache_entry(
        types.SimpleNamespace(), key
    )  # a third context evicts the oldest (a)
    assert id(a) not in conv_kcp._HOST_CACHE and id(b) in conv_kcp._HOST_CACHE
    assert conv_kcp.host_cache_clear() == 2 and conv_kcp._HOST_CACHE == {}


@pytest.mark.cpu_only
def test_host_trim_ffi_device_and_cpu_refusal(monkeypatch) -> None:
    """Trimmed host path: the FFI device object is built once per index; the trimmed forward / backward are selected by the knob
    and refuse CPU tensors like today's path (the operand checks first, then the architecture check)."""
    dev = conv_kcp._ffi_device(0)
    assert (
        conv_kcp._ffi_device(0) is dev
        and int(dev.index) == 0
        and int(dev.dlpack_device_type()) == 2
    )
    x = torch.zeros(4, 16, dtype=torch.bfloat16)
    w = torch.zeros(16, 4, dtype=torch.float32)
    hist = torch.zeros(1, 16, dtype=torch.bfloat16)
    ctx = types.SimpleNamespace(cu_seqlens_local_cpu=(0, 4))
    monkeypatch.setenv(conv_kcp.TEST_KNOBS_ENV, "1")
    try:
        conv_kcp._set_test_knobs({"host_trim": 1})
        with pytest.raises(ValueError, match="CUDA"):
            conv_kcp._kcp_forward(x, w, ctx)
        with pytest.raises(ValueError, match="CUDA"):
            conv_kcp._kcp_backward(x, w, ctx, x, hist)
    finally:
        conv_kcp._set_test_knobs(None)
    with pytest.raises(ValueError, match="CUDA"):
        conv_kcp._kcp_forward(x, w, ctx)
    conv_kcp.host_cache_clear()


# ----------------------------------------------------------------------------------------------------------------
# Recompute handle (issues #7815 / #7818): the handle object, the phase helper and the entry rules (no kernel runs).


@pytest.mark.cpu_only
def test_recompute_handle_object() -> None:
    h = conv_kcp.RecomputeHandleDetKcp()
    assert conv_kcp.RecomputeHandleDetKcp.__slots__ == ("entries",)
    assert h.entries == {} and h.nbytes() == 0
    with pytest.raises(AttributeError):
        h.other = 1
    a = torch.zeros(3, 16, dtype=torch.bfloat16)
    h.entries["conv"] = {"hist": a, "shape": (3, 16)}
    h.entries["gdn"] = {"S_in": torch.zeros(2, dtype=torch.float32), "pair": (a, 1)}
    assert h.nbytes() == 3 * 16 * 2 + 2 * 4  # distinct tensors only
    h.clear()
    assert h.entries == {} and h.nbytes() == 0
    assert conv_kcp.HANDLE_KEY == "conv"
    assert {"RecomputeHandleDetKcp", "HANDLE_KEY"} <= set(conv_kcp.__all__)


@pytest.mark.cpu_only
def test_recompute_handle_phase_and_entry_rules() -> None:
    """Outside the backward: no entry -> original forward (None), an entry -> raise.  Inside the backward: an entry
    -> the recompute (the entry, not popped), no entry -> an original forward too (nested regions)."""
    assert conv_kcp._kcp_in_backward() is False
    seen = {}

    class Probe(torch.autograd.Function):
        @staticmethod
        def forward(fctx, t):
            return t.clone()

        @staticmethod
        def backward(fctx, g):
            seen["in_backward"] = conv_kcp._kcp_in_backward()
            seen["empty"] = conv_kcp._handle_lookup(conv_kcp.RecomputeHandleDetKcp())
            seen["entry"] = conv_kcp._handle_lookup(full)
            return g

    full = conv_kcp.RecomputeHandleDetKcp()
    full.entries["conv"] = {"hist": torch.zeros(1, 8, dtype=torch.bfloat16)}
    t = torch.zeros(4, requires_grad=True)
    Probe.apply(t).sum().backward()
    assert seen["in_backward"] is True
    assert seen["empty"] is None
    assert seen["entry"] is full.entries["conv"] and "conv" in full.entries
    assert conv_kcp._handle_lookup(conv_kcp.RecomputeHandleDetKcp()) is None
    with pytest.raises(RuntimeError, match="micro-step"):
        conv_kcp._handle_lookup(full)
    with pytest.raises(TypeError, match="entries"):
        conv_kcp._handle_lookup(object())
    with pytest.raises(TypeError, match="entries"):
        conv_kcp._handle_lookup(types.SimpleNamespace(entries=[]))


@pytest.mark.cpu_only
def test_recompute_handle_needs_the_graph_task_binding(monkeypatch) -> None:
    monkeypatch.setattr(conv_kcp, "_GRAPH_TASK_ID_BINDING", None)
    monkeypatch.setattr(conv_kcp, "_GRAPH_TASK_ID_USABLE", None)
    with pytest.raises(RuntimeError, match="_current_graph_task_id"):
        conv_kcp._handle_lookup(conv_kcp.RecomputeHandleDetKcp())
    with pytest.raises(RuntimeError, match="_current_graph_task_id"):
        conv_kcp._kcp_in_backward()


@pytest.mark.cpu_only
def test_recompute_handle_hist_validation() -> None:
    ctx = types.SimpleNamespace()
    x = torch.zeros(5, 16, dtype=torch.bfloat16)
    hist = torch.zeros(3, 16, dtype=torch.bfloat16)
    entry = {
        "hist": hist,
        "shape": (3, 16),
        "dtype": torch.bfloat16,
        "device": hist.device,
        "ctx": ctx,
        "key": (5, 16),
    }
    assert conv_kcp._handle_hist(entry, x, ctx) is hist
    assert conv_kcp._handle_hist(entry, x.view(1, 5, 16), ctx) is hist
    for change in (
        {"ctx": types.SimpleNamespace()},
        {"key": (4, 16)},
        {"shape": (2, 16)},
        {"dtype": torch.float32},
        {"hist": hist.float()},
        {"hist": None},
    ):
        with pytest.raises(ValueError, match="recompute_handle"):
            conv_kcp._handle_hist({**entry, **change}, x, ctx)
    with pytest.raises(ValueError, match="recompute_handle"):
        conv_kcp._handle_hist(entry, torch.zeros(5, 8, dtype=torch.bfloat16), ctx)


@pytest.mark.cpu_only
def test_facade_reexports_the_handle_lazily() -> None:
    """``flashinfer.cake_causal_conv1d_det_train_kcp.RecomputeHandleDetKcp`` is the backend's class; importing the KCP
    facade (or the det facade, which keeps the previous KCP names) does not import the KCP host modules (``kcp`` imports
    torch.distributed; ``kcp_jit``) -- checked in a fresh interpreter; the facade's KCP entry carries the keyword-only
    ``recompute_handle=None``, and the previous names are the same objects."""
    import subprocess
    import sys

    code = (
        "import inspect, sys\n"
        "import flashinfer.cake_causal_conv1d_det_train_kcp as f\n"
        "import flashinfer.cake_causal_conv1d_det_train as d\n"
        "kcp = 'flashinfer.cake_causal_conv1d_det_train_backend.kcp'\n"
        "jit = 'flashinfer.cake_causal_conv1d_det_train_backend.kcp_jit'\n"
        "assert kcp not in sys.modules and jit not in sys.modules, sorted(m for m in sys.modules if 'kcp' in m)\n"
        "p = inspect.signature(f.cake_causal_conv1d_det_kcp).parameters['recompute_handle']\n"
        "assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None\n"
        "assert d.cake_causal_conv1d_kcp is f.cake_causal_conv1d_det_kcp\n"
        "assert kcp not in sys.modules\n"
        "from flashinfer.cake_causal_conv1d_det_train_kcp import RecomputeHandleDetKcp\n"
        "from flashinfer.cake_causal_conv1d_det_train_backend.kcp import RecomputeHandleDetKcp as B\n"
        "from flashinfer.cake_causal_conv1d_det_train_backend.kcp import KcpRecomputeHandle as C\n"
        "assert RecomputeHandleDetKcp is B is C is d.KcpRecomputeHandle and 'RecomputeHandleDetKcp' in f.__all__\n"
        "print('FACADE_IMPORT_OK')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0 and "FACADE_IMPORT_OK" in r.stdout, (r.stdout, r.stderr)
