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

from flashinfer.cake_causal_conv1d_train_backend import kcp as conv_kcp


def test_graph_knob_is_gated_and_validated(monkeypatch) -> None:
    monkeypatch.delenv(conv_kcp.TEST_KNOBS_ENV, raising=False)
    assert conv_kcp.GRAPH_MODES == ("off", "replay")
    assert conv_kcp._set_test_knobs(None) == {"graph": "off"}
    with pytest.raises(ValueError, match="CAKE_KCP_TEST_KNOBS"):
        conv_kcp._set_test_knobs({"graph": "replay"})
    assert conv_kcp._TEST_KNOBS == {"graph": "off"}
    monkeypatch.setenv(conv_kcp.TEST_KNOBS_ENV, "1")
    try:
        with pytest.raises(ValueError, match="unknown"):
            conv_kcp._set_test_knobs({"grap": "replay"})
        with pytest.raises(ValueError):
            conv_kcp._set_test_knobs({"graph": "capture"})
        with pytest.raises(TypeError):
            conv_kcp._set_test_knobs(["replay"])
        assert conv_kcp._set_test_knobs({"graph": "replay"}) == {"graph": "replay"}
        assert conv_kcp._set_test_knobs({"graph": "off"}) == {"graph": "off"}
    finally:
        conv_kcp._set_test_knobs(None)


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
        lambda: conv_kcp.cake_causal_conv1d_kcp(x, w, ctx, graph="capture"),
        lambda: conv_kcp.cake_causal_conv1d_kcp_forward(x, w, ctx, graph="capture"),
        lambda: conv_kcp.cake_causal_conv1d_kcp_backward(
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
