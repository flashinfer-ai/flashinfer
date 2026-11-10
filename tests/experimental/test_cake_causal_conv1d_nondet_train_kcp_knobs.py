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

CPU tests of the non-deterministic conv KCP module's private test knobs, the bookkeeping of its CUDA-graph replay
cache, its trimmed host path, its recompute handle and its public names: the knob is a private hook with no
environment gate (the backend reads no environment variable), validated against ``GRAPH_MODES`` and off by default;
the validity key tells operands
apart by data pointer, shape, stride and dtype, keeps one entry per key, and the first call of a key runs eagerly;
no public entry takes a ``deterministic`` keyword (the ordinary ``TypeError``) and every public name carries
``nondet``; the non-deterministic KCP package ``flashinfer.cake_causal_conv1d_nondet_train_kcp`` imports no KCP host
module until a call needs one.  No kernel runs here.
"""

from __future__ import annotations

import types

import pytest
import torch

from flashinfer.experimental.cake.cake_causal_conv1d_nondet_train import (
    kcp as conv_kcp,
)
from tests.cake.test_helpers.cake_package import PACKAGES


@pytest.mark.cpu_only
def test_backend_reads_no_environment_variable() -> None:
    """The KCP route has no environment gate: no ``TEST_KNOBS_ENV`` constant, and neither ``kcp.py`` nor ``kcp_jit.py``
    reads the environment (``os.environ`` / ``os.getenv`` / ``getenv``)."""
    import inspect

    from flashinfer.experimental.cake.cake_causal_conv1d_nondet_train import (
        kcp_jit,
    )

    assert not hasattr(conv_kcp, "TEST_KNOBS_ENV")
    for module in (conv_kcp, kcp_jit):
        source = inspect.getsource(module)
        for token in ("os.environ", "getenv", "environ.get", "import os"):
            assert token not in source, (module.__name__, token)


@pytest.mark.cpu_only
def test_graph_knob_is_validated() -> None:
    assert conv_kcp.GRAPH_MODES == ("off", "replay")
    assert conv_kcp._set_test_knobs(None) == {
        "graph": "off",
        "host_trim": 1,
        "bwd_route": "overlap",
        "fwd_main": None,
        "fwd_overlap": None,
    }
    assert conv_kcp._TEST_KNOBS == {
        "graph": "off",
        "host_trim": 1,
        "bwd_route": "overlap",
        "fwd_main": None,
        "fwd_overlap": None,
    }
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
            "bwd_route": "overlap",
            "fwd_main": None,
            "fwd_overlap": None,
        }
        assert conv_kcp._set_test_knobs({"graph": "off"}) == {
            "graph": "off",
            "host_trim": 1,
            "bwd_route": "overlap",
            "fwd_main": None,
            "fwd_overlap": None,
        }
    finally:
        conv_kcp._set_test_knobs(None)


@pytest.mark.cpu_only
def test_host_trim_knob_is_validated() -> None:
    """``host_trim`` (the trimmed host path): validated against ``HOST_TRIMS`` (1 shipped = the trimmed host path of the eager halo call,
    0 = the previous host path), independent of ``graph``; a dict naming one knob keeps the other at its shipped default."""
    assert conv_kcp.HOST_TRIMS == (0, 1) and conv_kcp.HOST_TRIM_DEFAULT == 1
    assert conv_kcp._TEST_KNOBS["host_trim"] == 1
    try:
        assert conv_kcp._set_test_knobs({"host_trim": 1}) == {
            "graph": "off",
            "host_trim": 1,
            "bwd_route": "overlap",
            "fwd_main": None,
            "fwd_overlap": None,
        }
        assert conv_kcp._graph_mode(None) == "off"
        assert conv_kcp._set_test_knobs({"host_trim": 1, "graph": "replay"}) == {
            "graph": "replay",
            "host_trim": 1,
            "bwd_route": "overlap",
            "fwd_main": None,
            "fwd_overlap": None,
        }
        assert conv_kcp._set_test_knobs({"graph": "replay"}) == {
            "graph": "replay",
            "host_trim": 1,
            "bwd_route": "overlap",
            "fwd_main": None,
            "fwd_overlap": None,
        }
        assert conv_kcp._set_test_knobs({}) == {
            "graph": "off",
            "host_trim": 1,
            "bwd_route": "overlap",
            "fwd_main": None,
            "fwd_overlap": None,
        }
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
            "bwd_route": "overlap",
            "fwd_main": None,
            "fwd_overlap": None,
        }
    finally:
        conv_kcp._set_test_knobs(None)
    assert conv_kcp._TEST_KNOBS == {
        "graph": "off",
        "host_trim": 1,
        "bwd_route": "overlap",
        "fwd_main": None,
        "fwd_overlap": None,
    }


@pytest.mark.cpu_only
def test_bwd_route_knob_is_validated() -> None:
    """``bwd_route``: validated against ``BWD_ROUTES`` (``"overlap"`` shipped = the overlapped pair of programs around
    the exchange's wait, ``"single"`` = the one backward program after it), independent of the other knobs; a
    backward graph key holds it when not the default, a forward key never does."""
    assert conv_kcp.BWD_ROUTES == ("overlap", "single")
    assert conv_kcp.BWD_ROUTE_DEFAULT == "overlap"
    assert conv_kcp._TEST_KNOBS["bwd_route"] == "overlap"
    ctx = types.SimpleNamespace(T_r=8, n_frag=1, world_size=1)
    try:
        assert conv_kcp._set_test_knobs({"bwd_route": "single"}) == {
            "graph": "off",
            "host_trim": 1,
            "bwd_route": "single",
            "fwd_main": None,
            "fwd_overlap": None,
        }
        assert conv_kcp._overlapped(1024, 8, 0) is False
        for bad in ({"bwd_route": "dyn"}, {"bwd_route": None}, {"bwd_rout": "single"}):
            with pytest.raises(ValueError):
                conv_kcp._set_test_knobs(bad)
        conv_kcp._set_test_knobs({"bwd_route": "single", "graph": "replay"})
        bwd = conv_kcp._graph_entry("bwd", ctx, ())
        fwd = conv_kcp._graph_entry("fwd", ctx, ())
        assert bwd.key[-1] == ("bwd_route", "single")
        assert ("bwd_route", "single") not in fwd.key
        conv_kcp._set_test_knobs({"graph": "replay"})
        assert ("bwd_route", "overlap") not in conv_kcp._graph_entry("bwd", ctx, ()).key
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()
    assert conv_kcp._TEST_KNOBS["bwd_route"] == "overlap"


ARCHES = ("sm_100a", "sm_103a", "sm_107a")
FORMS = ((False, False), (True, False), (False, True), (True, True))
# The shipped ``_fwd_form(arch, multi=...)`` (single-rank main program, overlapped) of the single call and the multi
# entry per architecture: the deterministic KCP package's (FlashInfer #9225)
SHIPPED_FWD_FORMS = {
    "sm_100a": ((True, False), (True, False)),
    "sm_103a": ((True, False), (True, False)),
    "sm_107a": ((True, False), (True, True)),
}
FWD_SETS = (
    "FWD_MAIN_ARCHS",
    "FWD_OVERLAP_ARCHS",
    "MULTI_FWD_MAIN_ARCHS",
    "MULTI_FWD_OVERLAP_ARCHS",
)


@pytest.mark.cpu_only
def test_fwd_form_knobs_are_validated(monkeypatch) -> None:
    """``fwd_main`` / ``fwd_overlap`` (the two-launch forward): private setters validated against ``FWD_MAINS`` /
    ``FWD_OVERLAPS`` (``None`` shipped = the architecture sets, ``0`` = off, ``1`` = on), independent of each other
    and of the other knobs.  With the knobs ``None``, ``_fwd_form`` is the shipped form of ``SHIPPED_FWD_FORMS`` (the
    sets hold exactly its architectures, the deterministic KCP package's sets); a ``0`` / ``1`` knob forces its form
    on every architecture and entry; with every set emptied, an architecture added to one set takes that form in that
    set's entry only."""
    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as det_kcp,
    )

    assert conv_kcp.FWD_MAINS == (None, 0, 1) and conv_kcp.FWD_MAIN_DEFAULT is None
    assert conv_kcp.FWD_OVERLAPS == (None, 0, 1)
    assert conv_kcp.FWD_OVERLAP_DEFAULT is None
    for i, name in enumerate(FWD_SETS):
        # FWD_SETS[i] holds the architectures whose entry (i // 2: single, multi) ships form field i % 2 on
        multi, field = divmod(i, 2)
        want = {a for a in ARCHES if SHIPPED_FWD_FORMS[a][multi][field]}
        assert getattr(conv_kcp, name) == frozenset(want), name
        assert getattr(conv_kcp, name) == getattr(det_kcp, name), name
    try:
        for arch in ARCHES:
            for multi in (False, True):
                assert (
                    conv_kcp._fwd_form(arch, multi=multi)
                    == SHIPPED_FWD_FORMS[arch][multi]
                )
        for main in (0, 1):
            for overlap in (0, 1):
                knobs = {"fwd_main": main, "fwd_overlap": overlap}
                assert conv_kcp._set_test_knobs(knobs) == {
                    "graph": "off",
                    "host_trim": 1,
                    "bwd_route": "overlap",
                    **knobs,
                }
                for arch in ARCHES:
                    for multi in (False, True):
                        assert conv_kcp._fwd_form(arch, multi=multi) == (
                            bool(main),
                            bool(overlap),
                        )
        assert conv_kcp._set_test_knobs({"fwd_overlap": 1})["fwd_main"] is None
        assert conv_kcp._fwd_form("sm_100a") == (
            SHIPPED_FWD_FORMS["sm_100a"][0][0],
            True,
        )
        for bad in (
            {"fwd_main": 2},
            {"fwd_main": True},
            {"fwd_main": "1"},
            {"fwd_overlap": -1},
            {"fwd_overlap": False},
            {"fwd_overlap": "0"},
            {"fwd_mian": 1},
        ):
            with pytest.raises(ValueError):
                conv_kcp._set_test_knobs(bad)
        conv_kcp._set_test_knobs(None)
        # every set emptied and an architecture added to one: that form in that entry only, and a knob of 0 / 1
        # still forces its form
        for name, form, multi in (
            ("FWD_MAIN_ARCHS", (True, False), False),
            ("FWD_OVERLAP_ARCHS", (False, True), False),
            ("MULTI_FWD_MAIN_ARCHS", (True, False), True),
            ("MULTI_FWD_OVERLAP_ARCHS", (False, True), True),
        ):
            for other in FWD_SETS:
                monkeypatch.setattr(conv_kcp, other, frozenset())
            monkeypatch.setattr(conv_kcp, name, frozenset({"sm_103a"}))
            assert conv_kcp._fwd_form("sm_103a", multi=multi) == form, name
            assert conv_kcp._fwd_form("sm_103a", multi=not multi) == (False, False)
            assert conv_kcp._fwd_form("sm_100a", multi=multi) == (False, False)
            conv_kcp._set_test_knobs({"fwd_main": 0, "fwd_overlap": 0})
            assert conv_kcp._fwd_form("sm_103a", multi=multi) == (False, False)
            conv_kcp._set_test_knobs(None)
            monkeypatch.undo()
    finally:
        monkeypatch.undo()
        conv_kcp._set_test_knobs(None)
    assert conv_kcp._TEST_KNOBS["fwd_main"] is None
    assert conv_kcp._TEST_KNOBS["fwd_overlap"] is None


@pytest.mark.cpu_only
def test_graph_key_and_record_of_the_forward_form() -> None:
    """The forward's graph-key value is its effective form (``("fwd_form", _fwd_variant(x))`` with ``_fwd_variant`` =
    ``_fwd_form`` of the device: the four forms are four keys, the same form one key), and ``LAST_CALL[kind]`` holds
    the launch form the body records (``_fwd_record``) beside the graph event: afresh on an eager call, the captured
    call's on a replay.  No kernel runs here (a stand-in graph replays)."""
    ctx = types.SimpleNamespace(T_r=8, n_frag=1, world_size=1)
    x = torch.zeros(8, 16, dtype=torch.bfloat16)
    try:
        keys = {
            form: conv_kcp._graph_entry(
                "fwd", ctx, (x, None), "replay", extra=(("fwd_form", form),)
            )
            for form in FORMS
        }
        assert len({id(e) for e in keys.values()}) == len(FORMS)
        for form in FORMS:
            again = conv_kcp._graph_entry(
                "fwd", ctx, (x, None), "replay", extra=(("fwd_form", form),)
            )
            assert again is keys[form], form
        e = keys[(True, True)]

        def body():
            conv_kcp._fwd_record("fwd", True, True, True)
            return ("y", "hist")

        conv_kcp.LAST_CALL["fwd"] = {"stale": 1}
        assert conv_kcp._graph_run("fwd", e, body, "replay") == ("y", "hist")
        assert conv_kcp.LAST_CALL["fwd"] == {
            "main": "sr",
            "form": "pair",
            "repair": True,
            "graph": "replay",
            "graph_event": "eager",
            "graph_key": e.digest,
            "graph_fallback": None,
        }
        e.graph = types.SimpleNamespace(replay=lambda: None)
        e.result = ("y", "hist")
        e.record = {"main": "f", "form": "pair", "repair": False}
        conv_kcp.LAST_CALL["fwd"] = {"stale": 1}
        assert conv_kcp._graph_run("fwd", e, body, "replay") == ("y", "hist")
        assert conv_kcp.LAST_CALL["fwd"] == {
            "main": "f",
            "form": "pair",
            "repair": False,
            "graph": "replay",
            "graph_event": "replay",
            "graph_key": e.digest,
            "graph_fallback": None,
        }
        # a rank without tokens launches nothing
        conv_kcp._fwd_record("fwd_multi", None, False, False)
        assert conv_kcp.LAST_CALL["fwd_multi"] == {
            "main": None,
            "form": "wait",
            "repair": False,
        }
    finally:
        conv_kcp._GRAPHS.clear()
        conv_kcp.LAST_CALL.clear()


@pytest.mark.cpu_only
def test_overlapped_route_rule() -> None:
    """The overlapped pair runs every backward call with tokens and at most ``DYN_MAX_CHANNELS`` channels and no SM
    reserve; a reserve, a wider ``C`` or no tokens run the single program."""
    limit = conv_kcp._loader.DYN_MAX_CHANNELS
    assert limit == 65536
    assert conv_kcp._overlapped(8, 1, 0) and conv_kcp._overlapped(limit, 1 << 20, 0)
    assert not conv_kcp._overlapped(limit + 8, 4096, 0)
    assert not conv_kcp._overlapped(1024, 0, 0)
    assert not conv_kcp._overlapped(1024, 4096, 1)


@pytest.mark.cpu_only
def test_graph_cache_key_and_eager_first_call(monkeypatch) -> None:
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
        # cake_causal_conv1d_nondet_train_kcp_graph_cache_clear(): drops every entry (to be called before destroy_process_group), reports the count
        n_entries = len(conv_kcp._GRAPHS)
        assert (
            n_entries >= 2
            and "cake_causal_conv1d_nondet_train_kcp_graph_cache_clear"
            in conv_kcp.__all__
        )
        assert (
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()
            == n_entries
        )
        assert (
            conv_kcp._GRAPHS == {}
            and conv_kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear() == 0
        )
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp._GRAPHS.clear()
        conv_kcp.LAST_CALL.clear()


@pytest.mark.cpu_only
def test_graph_public_keyword_mode_and_key(monkeypatch) -> None:
    """``graph=`` is a public keyword of the conv KCP calls (no environment gate): validated against
    ``GRAPH_MODES``, ``None`` = the module's shipped default (``GRAPH_DEFAULT == "off"``), keyed and recorded like the
    knob; a keyword contradicting an active knob is a ``ValueError``.  No kernel runs here."""
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
        lambda: conv_kcp.cake_causal_conv1d_nondet_kcp(x, w, ctx, graph="capture"),
        lambda: conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
            x, w, ctx, graph="capture"
        ),
        lambda: conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
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
    by ``HOST_CACHE_MAX`` (the oldest context dropped) and cleared by ``cake_causal_conv1d_nondet_train_kcp_host_cache_clear()``."""
    conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
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
    assert (
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear() == 2
        and conv_kcp._HOST_CACHE == {}
    )


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
    conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()


# ----------------------------------------------------------------------------------------------------------------
# Recompute handle (issues #7815 / #7818): the handle object, the phase helper and the entry rules (no kernel runs).


@pytest.mark.cpu_only
def test_recompute_handle_object() -> None:
    h = conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
    assert conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle.__slots__ == ("entries",)
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
    assert {"CakeCausalConv1dNondetKcpRecomputeHandle", "HANDLE_KEY"} <= set(
        conv_kcp.__all__
    )


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
            seen["empty"] = conv_kcp._handle_lookup(
                conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
            )
            seen["entry"] = conv_kcp._handle_lookup(full)
            return g

    full = conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
    full.entries["conv"] = {"hist": torch.zeros(1, 8, dtype=torch.bfloat16)}
    t = torch.zeros(4, requires_grad=True)
    Probe.apply(t).sum().backward()
    assert seen["in_backward"] is True
    assert seen["empty"] is None
    assert seen["entry"] is full.entries["conv"] and "conv" in full.entries
    assert (
        conv_kcp._handle_lookup(conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle())
        is None
    )
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
        conv_kcp._handle_lookup(conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle())
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
def test_kcp_entries_take_no_deterministic_keyword() -> None:
    """No public entry of the non-deterministic KCP backend takes ``deterministic``: passing it is the ordinary
    ``TypeError`` (raised before the body runs, so no device is needed); every public name of the module carries
    ``nondet`` (the constants and the cache helper aside) and the previous names are gone."""
    import inspect

    x = torch.empty(8, 16, dtype=torch.bfloat16)
    w = torch.empty(16, 4, dtype=torch.float32)
    ctx = types.SimpleNamespace(T_r=8, n_frag=1, world_size=1)
    calls = {
        conv_kcp.cake_causal_conv1d_nondet_kcp: lambda fn, d: fn(x, w, ctx, **d),
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward: lambda fn, d: fn(x, w, ctx, **d),
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward: lambda fn, d: fn(x, w, ctx, x, x, **d),
        conv_kcp.cake_causal_conv1d_nondet_kcp_multi: lambda fn, d: fn((x,), (w,), ctx, **d),
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward: lambda fn, d: fn((x,), (w,), ctx, **d),
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward: lambda fn, d: fn((x,), (w,), ctx, (x,), (x,), **d),
    }  # fmt: skip
    for fn, call in calls.items():
        assert "deterministic" not in inspect.signature(fn).parameters, fn.__name__
        for value in (False, True):
            with pytest.raises(TypeError, match="deterministic"):
                call(fn, {"deterministic": value})
    for name in conv_kcp.__all__:
        if name.isupper() or name in (
            "cake_causal_conv1d_nondet_train_kcp_graph_cache_clear",
        ):
            continue
        assert "nondet" in name.lower(), name
        assert "det" not in name.lower().replace("nondet", ""), name
    for old in (
        "KcpRecomputeHandle",
        "RecomputeHandleKcp",
        "CakeCausalConv1dFunctionKcp",
        "CakeCausalConv1dMultiFunctionKcp",
        "cake_causal_conv1d_kcp",
        "cake_causal_conv1d_multi_kcp",
        "cake_causal_conv1d_forward_kcp",
        "cake_causal_conv1d_backward_kcp",
        "cake_causal_conv1d_multi_forward_kcp",
        "cake_causal_conv1d_multi_backward_kcp",
    ):
        assert not hasattr(conv_kcp, old), old


@pytest.mark.cpu_only
def test_facade_reexports_the_handle_lazily() -> None:
    """``flashinfer.cake_causal_conv1d_nondet_train_kcp.CakeCausalConv1dNondetKcpRecomputeHandle`` is the
    non-deterministic backend's class; importing the facade imports neither the non-deterministic backend package
    nor a KCP host module (``kcp`` imports torch.distributed; ``kcp_jit``) -- checked in a fresh interpreter; both
    facade entries carry the keyword-only ``recompute_handle=None`` and no ``deterministic`` keyword; the
    deterministic names are unchanged."""
    import subprocess
    import sys

    code = (
        "import inspect, sys\n"
        "import flashinfer.cake_causal_conv1d_nondet_train_kcp as f\n"
        "pkg = 'flashinfer.experimental.cake.cake_causal_conv1d_nondet_train'\n"
        f"det = {PACKAGES['cake_causal_conv1d_det_train']!r}\n"
        "loaded = sorted(m for m in sys.modules if m.startswith((pkg, det + '.kcp')))\n"
        "assert not loaded, loaded\n"
        "for fn in (f.cake_causal_conv1d_nondet_kcp, f.cake_causal_conv1d_nondet_kcp_multi):\n"
        "    ps = inspect.signature(fn).parameters\n"
        "    assert ps['recompute_handle'].kind is inspect.Parameter.KEYWORD_ONLY and ps['recompute_handle'].default is None\n"
        "    assert 'deterministic' not in ps\n"
        "assert pkg not in sys.modules\n"
        "H = 'CakeCausalConv1dNondetKcpRecomputeHandle'\n"
        "from flashinfer.cake_causal_conv1d_nondet_train_kcp import CakeCausalConv1dNondetKcpRecomputeHandle as A\n"
        "from flashinfer.experimental.cake.cake_causal_conv1d_nondet_train import kcp\n"
        "assert A is kcp.CakeCausalConv1dNondetKcpRecomputeHandle and H in f.__all__ and H in dir(f)\n"
        "import flashinfer.cake_causal_conv1d_det_train_kcp as d\n"
        "from flashinfer.cake_causal_conv1d_det_train_kcp import CakeCausalConv1dDetKcpRecomputeHandle\n"
        "assert CakeCausalConv1dDetKcpRecomputeHandle is not A\n"
        "assert d.cake_causal_conv1d_det_kcp is not f.cake_causal_conv1d_nondet_kcp\n"
        "print('FACADE_IMPORT_OK')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0 and "FACADE_IMPORT_OK" in r.stdout, (r.stdout, r.stderr)
