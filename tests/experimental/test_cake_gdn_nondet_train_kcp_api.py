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

CPU tests of the public surface of the NON-DETERMINISTIC context-parallel packed GDN training package
``flashinfer.experimental.cake.cake_gdn_nondet_train_kcp`` and of its facade ``flashinfer.cake_gdn_nondet_train_kcp`` (this
package only; the deterministic counterpart is ``flashinfer.cake_gdn_det_train_kcp``): the options validator
(``cake_options``), the keyword-only signatures, the facade's and the package's exact public names (each with the
word ``nondet``; the former names and module paths fail at import), the facade as a stable API (``@flashinfer_api``
entries, not marked experimental, a first call warns of nothing), the numerical mode (in the name only: no entry
takes ``deterministic``, a stray keyword is a ``TypeError``; every launch being fixed-order, nothing alerts, reroutes
or warns under ``torch.use_deterministic_algorithms``; no context, partition or options field holds the mode), the
facade's hand-off of ``use_qk_l2norm_in_kernel`` to the package, the package hygiene rules (no environment variable,
no KDA package import, standalone: no import of another FlashInfer package, its deterministic counterpart included,
and no library name of that package) and the host tables built on the CPU for the correctness rows' partitions.  No
kernel runs here: the compute entries run on the fake backend (the CPU emulation), directly or through
``KernelBackend`` with a stub launch glue (the architecture the kernel backend passes to the #7 walks).
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import importlib
import importlib.util
import inspect
import json
import pathlib
import re
import subprocess
import sys
import types
import typing
import warnings

import pytest
import torch

PACKAGE = "flashinfer.experimental.cake.cake_gdn_nondet_train_kcp"
FACADE = "flashinfer.cake_gdn_nondet_train_kcp"
DET_FACADE = "flashinfer.cake_gdn_det_train_kcp"


@pytest.fixture(autouse=True, scope="module")
def _few_cpu_threads():
    """The CPU emulation runs many small matmuls and elementwise ops.  On a host with hundreds of CPUs, torch's default
    intra-op pool and the oneDNN matmul path (its own pool, which torch.set_num_threads does not resize) spend far more
    time synchronizing threads than computing (one test of this suite on a 352-CPU host: 272 s by default, 6 s with at
    most 8 threads and oneDNN off), so this module runs that way."""
    threads, onednn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(min(threads, 8))
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = onednn
        torch.set_num_threads(threads)


def _mod(sub: str | None = None):
    """The package (or one of its modules) through the test-helper registry when it lists the package, else directly."""
    try:
        from tests.cake.test_helpers.cake_package import cake_package

        return cake_package("cake_gdn_nondet_train_kcp", sub)
    except (ImportError, KeyError):
        return importlib.import_module(PACKAGE if sub is None else f"{PACKAGE}.{sub}")


opt = _mod("cake_options")
CakeGDNNondetKcpOptions, make_options = opt.CakeGDNNondetKcpOptions, opt.make_options


# ----------------------------------------------------------------------------------------------------------------
# options validator (the contract's option vocabulary)


def test_options_defaults_are_the_shipped_behaviour() -> None:
    o = make_options()
    assert o.exchange == "a2a"
    assert o.save_transfer is True
    # without the in-kernel norm; with it the default is "light_akk"
    assert o.save_mode == "light"
    assert make_options(use_qk_l2norm_in_kernel=True).save_mode == "light_akk"
    assert o.boundary_mode == "saved"
    assert o.precision_mode == 0
    assert o.transfer_walk == "tf32_hilo"
    assert o.offset_class == "fla"
    assert o.layout == "tma"
    assert o.group_reduce == "fused"
    assert o.graph == "off"
    assert o.reuse_recompute_tapes is opt.REUSE_RECOMPUTE_TAPES_DEFAULT
    assert o.recompute_output == "auto"
    assert o.backend == "kernels"
    assert o.segments is None
    assert o.test_knobs is None
    assert o.knob("schedule") == "wait_first" and o.knob("fused_walks") == 1
    assert o.knob("cols_rule") == "fixed" and o.knob("debug_sequence") is False
    assert o.knob("p_rule") is None


def test_every_option_field_is_keyword_only() -> None:
    for f in dataclasses.fields(CakeGDNNondetKcpOptions):
        assert f.kw_only, f.name
    with pytest.raises(TypeError):
        CakeGDNNondetKcpOptions("ring")  # type: ignore[misc]


@pytest.mark.parametrize("mode", [1, 2, 3, -1, True, False, 1.0, "0"])
def test_precision_mode_rejects_unknown_values_and_bare_test_values(mode) -> None:
    """Template numbering 0 / 1 / 2; 1 and 2 are test-only and refused unless ``test_knobs``
    names the same value -- the public API alone cannot select a test program."""
    with pytest.raises((ValueError, TypeError)):
        make_options(precision_mode=mode)


def test_precision_mode_test_values_need_test_knobs() -> None:
    assert opt.PRECISION_MODES == (0, 1, 2) and opt.PRECISION_MODE_DEFAULT == 0
    assert opt.PRECISION_MODES_TEST_ONLY == (1, 2)
    assert make_options(precision_mode=0).precision_mode == 0
    for mode in (1, 2):
        o = make_options(precision_mode=mode, test_knobs={"precision_mode": mode})
        assert o.precision_mode == mode and o.knob("precision_mode") == mode
    with pytest.raises(ValueError):  # the named value must be the selected one
        make_options(precision_mode=1, test_knobs={"precision_mode": 2})
    with pytest.raises(ValueError):  # naming the shipped record is not a test knob
        make_options(test_knobs={"precision_mode": 0})
    assert make_options(test_knobs={"precision_mode": 2}).precision_mode == 0


def test_transfer_walk_bf16x3_is_in_the_vocabulary_but_not_built() -> None:
    """No GDN BF16x3 transfer walk is built in this release -- the name stays in the vocabulary, every attempt to select it
    is refused (with or without ``test_knobs``)."""
    assert opt.TRANSFER_WALKS == ("tf32_hilo", "bf16x3")
    assert opt.TRANSFER_WALK_DEFAULT == "tf32_hilo"
    assert opt.TRANSFER_WALKS_UNBUILT == ("bf16x3",)
    assert make_options(transfer_walk="tf32_hilo").transfer_walk == "tf32_hilo"
    with pytest.raises(ValueError, match="not built"):
        make_options(transfer_walk="bf16x3")
    with pytest.raises(ValueError):
        make_options(transfer_walk="tf32x3")
    with pytest.raises(ValueError):  # not a test knob either
        make_options(transfer_walk="bf16x3", test_knobs={"transfer_walk": "bf16x3"})


def test_ring_requires_the_recompute_form() -> None:
    with pytest.raises(ValueError):
        make_options(exchange="ring")  # save_transfer defaults to True: no ring form
    with pytest.raises(ValueError):
        make_options(exchange="ring", save_transfer=True)
    assert make_options(exchange="ring", save_transfer=False).exchange == "ring"
    with pytest.raises(ValueError):
        make_options(exchange="auto")


def test_ring_hop_limit() -> None:
    assert opt.RING_MAX_HOPS == 2
    opt.validate_ring(1)
    opt.validate_ring(3)  # n_d_max = 3 -> 2 hops
    with pytest.raises(ValueError):
        opt.validate_ring(4)


@pytest.mark.parametrize(
    "pair,r",
    [((16, 48), 3), ((8, 24), 3), ((4, 12), 3), ((2, 6), 3), ((1, 3), 3), ((4, 4), 1)],
)
def test_head_pairs_of_the_goal_are_legal(pair, r) -> None:
    assert opt.validate_heads(*pair) == r


@pytest.mark.parametrize("pair", [(3, 4), (4, 6), (16, 40), (0, 3), (2, 0), (5, 1)])
def test_head_rule_rejects_non_multiples(pair) -> None:
    with pytest.raises(ValueError):
        opt.validate_heads(*pair)


@pytest.mark.parametrize(
    "field,values",
    [
        ("save_mode", ("light", "full")),
        ("boundary_mode", ("saved", "replay")),
        ("graph", ("off", "replay")),
        ("layout", ("tma", "copy")),
        ("group_reduce", ("fused", "host")),
        ("backend", ("kernels", "fake")),
        ("offset_class", ("fla", "fp32")),
    ],
)
def test_enumerated_fields(field, values) -> None:
    for v in values:
        assert getattr(make_options(**{field: v}), field) == v
    for bad in ("tape", "auto", "", None, 1):
        # None = unset: the default ("light" without the norm)
        if field == "save_mode" and bad is None:
            assert make_options(save_mode=None).save_mode == "light"
            continue
        with pytest.raises((ValueError, TypeError)):
            make_options(**{field: bad})


def test_reuse_recompute_tapes_defaults_to_auto_in_both_graph_modes() -> None:
    """The checkpoint-aware light backward is a keyword-only switch -- ``"auto"`` (the shipped default: the tapes are
    kept only when the node closes the checkpointed region), ``True`` (always) or ``False`` (the replay form) -- part
    of the options key (so of the CUDA-graph key) and typed; ``graph="replay"`` has the same default and accepts every
    value (the kept tapes are then the workspace arena's own: ``cake_kcp._kept_stale``)."""
    assert "reuse_recompute_tapes" in opt.OPTION_NAMES
    # the shipped default is this one constant (and this line) -- every other test names the state it exercises
    assert opt.REUSE_RECOMPUTE_TAPES_DEFAULT == "auto"
    assert make_options().reuse_recompute_tapes == opt.REUSE_RECOMPUTE_TAPES_DEFAULT
    # None = the shipped default, under graph="replay" too
    assert make_options(reuse_recompute_tapes=None).reuse_recompute_tapes == "auto"
    assert make_options(graph="replay").reuse_recompute_tapes == "auto"
    both = make_options(reuse_recompute_tapes=None, graph="replay")
    assert both.reuse_recompute_tapes == "auto"
    assert CakeGDNNondetKcpOptions(graph="replay").reuse_recompute_tapes == "auto"
    assert make_options(graph="replay").key() != make_options().key()
    on = make_options(reuse_recompute_tapes=True)
    off = make_options(reuse_recompute_tapes=False)
    assert on.reuse_recompute_tapes is True and off.reuse_recompute_tapes is False
    assert on.key() != off.key()
    from_instance = make_options(CakeGDNNondetKcpOptions(reuse_recompute_tapes=True))
    assert from_instance.reuse_recompute_tapes is True
    with pytest.raises(TypeError):
        make_options(reuse_recompute_tapes=1)
    for value in (True, False, "auto"):
        o = make_options(reuse_recompute_tapes=value, graph="replay")
        assert o.reuse_recompute_tapes == value and o.graph == "replay"
    # the other modes accept the switch (it has no effect there: documented as a no-op)
    for extra in (
        {"boundary_mode": "replay"},
        {"save_mode": "full"},
        {"save_transfer": False},
    ):
        o = make_options(reuse_recompute_tapes=True, **extra)
        assert o.reuse_recompute_tapes is True


def test_save_transfer_must_be_a_bool() -> None:
    with pytest.raises(TypeError):
        make_options(save_transfer=1)
    with pytest.raises(TypeError):
        make_options(save_transfer="yes")


def test_recompute_output_is_a_two_value_keyword_switch_defaulting_to_auto() -> None:
    """``recompute_output`` -- whether the recompute of a checkpointed forward launches its dead output program -- is a
    keyword-only field of the options (so part of the CUDA-graph key), ``"auto"`` by default (the frame-proof skip),
    ``"always"`` the unconditional launch; anything else is refused.  It is independent of the kept-tape switch and
    of the graph mode (under ``graph="replay"`` the recompute's decision is part of its CUDA-graph key)."""
    assert "recompute_output" in opt.OPTION_NAMES
    assert opt.RECOMPUTE_OUTPUTS == ("auto", "always")
    assert opt.RECOMPUTE_OUTPUT_DEFAULT == "auto"
    assert make_options().recompute_output == "auto"
    assert make_options(recompute_output="always").recompute_output == "always"
    assert (
        make_options(
            CakeGDNNondetKcpOptions(recompute_output="always")
        ).recompute_output
        == "always"
    )
    for bad in ("never", "", None, True, 1):
        with pytest.raises((ValueError, TypeError)):
            make_options(recompute_output=bad)
    o = make_options(reuse_recompute_tapes=False, recompute_output="auto")
    assert o.recompute_output == "auto" and o.reuse_recompute_tapes is False
    assert make_options(graph="replay").recompute_output == "auto"


def test_unknown_option_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        make_options(foo=1)
    with pytest.raises(TypeError):
        make_options({"walk_precision": "bf16x3"})
    with pytest.raises(TypeError):
        make_options(_kcp_test_knobs={"precision_mode": 0})


def test_segments_range() -> None:
    assert make_options(segments=1).segments == 1
    assert make_options(segments=64).segments == 64
    for bad in (0, 65, -1):
        with pytest.raises(ValueError):
            make_options(segments=bad)
    for bad in (True, 2.0, "2"):
        with pytest.raises(TypeError):
            make_options(segments=bad)


def test_test_knobs_are_a_keyword_dict_with_known_keys() -> None:
    o = make_options(
        test_knobs={
            "schedule": "overlap",
            "fused_walks": 0,
            "cols_rule": "by_chains",
            "debug_sequence": True,
            "p_rule": "occ",
        }
    )
    assert o.knob("schedule") == "overlap" and o.knob("fused_walks") == 0
    assert o.knob("cols_rule") == "by_chains" and o.knob("debug_sequence") is True
    assert o.knob("p_rule") == "occ"
    with pytest.raises(ValueError):
        make_options(test_knobs={"precision_mode": 0})
    assert make_options(test_knobs={"fwdh_vh": 64}).knob("fwdh_vh") == 64
    assert make_options().knob("fwdh_vh") is None
    with pytest.raises(ValueError):
        make_options(test_knobs={"fwdh_vh": 32})
    assert make_options(test_knobs={"dhu_vh": 128}).knob("dhu_vh") == 128
    assert make_options().knob("dhu_vh") is None
    with pytest.raises(ValueError):
        make_options(test_knobs={"dhu_vh": 96})
    with pytest.raises(ValueError):
        make_options(test_knobs={"schedule": "eager"})
    with pytest.raises(ValueError):
        make_options(test_knobs={"fused_walks": True})
    with pytest.raises(ValueError):
        make_options(test_knobs={"p_rule": 3})
    with pytest.raises(TypeError):
        make_options(test_knobs=["schedule"])
    with pytest.raises(KeyError):
        o.knob("walk_geometry")


def test_walk_pair_knob_defaults_to_auto() -> None:
    assert opt.TEST_KNOBS["walk_pair"] == (None, 0, 1)
    assert opt.TEST_KNOB_DEFAULTS["walk_pair"] is None
    assert make_options().knob("walk_pair") is None
    assert make_options(test_knobs={"walk_pair": None}).knob("walk_pair") is None
    assert make_options(test_knobs={"walk_pair": 0}).knob("walk_pair") == 0
    assert make_options(test_knobs={"walk_pair": 1}).knob("walk_pair") == 1
    for bad in (2, -1, True, False, "1"):
        with pytest.raises(ValueError):
            make_options(test_knobs={"walk_pair": bad})
    keys = {make_options(test_knobs={"walk_pair": v}).key() for v in (None, 0, 1)}
    assert len(keys) == 3
    with pytest.raises(TypeError):
        make_options(walk_pair=1)  # a test knob, never a public option


@pytest.mark.parametrize("knob", ("fused_o", "fused_o_tape"))
def test_output_walk_knobs_default_to_the_shipped_walks(knob: str) -> None:
    """``fused_o`` (the light forward's output walk) and ``fused_o_tape`` (the ``save_mode="full"`` forward's
    tape-and-output walk) are private levers: 1 (shipped) or 0 (the split tape walk and #5), part of the options key
    (a captured graph of one form never replays for the other), never public options."""
    assert opt.TEST_KNOBS[knob] == (0, 1)
    assert opt.TEST_KNOB_DEFAULTS[knob] == 1
    assert make_options().knob(knob) == 1
    assert make_options(test_knobs={knob: 0}).knob(knob) == 0
    for bad in (2, -1, True, False, None, "1"):
        with pytest.raises(ValueError):
            make_options(test_knobs={knob: bad})
    keys = {make_options(test_knobs={knob: v}).key() for v in (0, 1)}
    assert len(keys) == 2
    with pytest.raises(TypeError):
        make_options(**{knob: 0})  # a test knob, never a public option


@pytest.mark.parametrize(
    "knob, layout, world, capability, on",
    [
        (None, "zigzag", 2, (10, 3), True),
        (None, "zigzag", 4, (10, 0), True),
        (None, "zigzag_per_document", 4, (10, 3), True),
        (None, "zigzag_per_document", 8, (10, 0), True),
        (None, "contiguous", 4, (10, 3), False),
        (None, "explicit", 4, (10, 0), False),
        (None, "zigzag", 16, (10, 3), False),
        (None, "zigzag_per_document", 16, (10, 0), False),
        (None, "zigzag", 4, (10, 7), False),
        (None, "zigzag_per_document", 8, (10, 7), False),
        (None, "zigzag", 4, (9, 0), False),
        (None, "zigzag", 4, (12, 0), False),
        (0, "zigzag", 4, (10, 3), False),
        (0, "zigzag_per_document", 8, (10, 0), False),
        (1, "zigzag", 4, (10, 7), True),
        (1, "zigzag", 4, (10, 3), True),
        (1, "zigzag", 16, (10, 3), False),
        (1, "contiguous", 4, (10, 0), False),
    ],
)
def test_walk_pair_auto_is_on_for_zigzag_up_to_8_ranks_on_sm100_and_sm103(
    monkeypatch, knob, layout, world, capability, on
) -> None:
    """The default (``None``) pairs the walks on a zigzag / zigzag-per-document partition of at most 8 ranks on
    ``sm_100a`` / ``sm_103a`` only (``sm_107a`` keeps one stream); an explicit 0 / 1 forces either path there."""
    kcp = _mod("cake_kcp")
    assert kcp.WALK_PAIR_AUTO_CAPABILITIES == ((10, 0), (10, 3))
    queried: list = []

    def capability_of(index):
        queried.append(index)
        return capability

    monkeypatch.setattr(kcp, "_CAPABILITIES", {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability_of)
    ctx = types.SimpleNamespace(
        partition=types.SimpleNamespace(layout=layout, world_size=world)
    )
    device = torch.device("cuda", 1)
    assert kcp._walk_pair_enabled(knob, ctx, device) is on
    assert kcp._walk_pair_enabled(knob, ctx, device) is on
    assert len(queried) <= 1  # the capability is read once per device
    assert all(index == 1 for index in queried)


def test_walk_pair_auto_is_off_without_a_cuda_device(monkeypatch) -> None:
    kcp = _mod("cake_kcp")

    def no_query(index):
        raise AssertionError("no capability query off CUDA")

    monkeypatch.setattr(kcp, "_CAPABILITIES", {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", no_query)
    ctx = types.SimpleNamespace(
        partition=types.SimpleNamespace(layout="zigzag", world_size=4)
    )
    assert kcp._walk_pair_enabled(None, ctx, torch.device("cpu")) is False
    assert kcp._walk_pair_enabled(None, ctx) is False
    assert kcp._walk_pair_enabled(1, ctx, torch.device("cpu")) is True


@pytest.mark.parametrize(
    "knob, layout, world, on",
    [
        (1, "zigzag", 2, True),
        (1, "zigzag", 8, True),
        (1, "zigzag_per_document", 8, True),
        (1, "zigzag", 16, False),
        (1, "zigzag_per_document", 16, False),
        (1, "contiguous", 8, False),
        (1, "explicit", 8, False),
        (0, "zigzag", 8, False),
        (0, "zigzag_per_document", 4, False),
    ],
)
def test_walk_pair_is_on_for_zigzag_partitions_up_to_8_ranks(
    knob, layout, world, on
) -> None:
    kcp = _mod("cake_kcp")
    assert kcp.WALK_PAIR_LAYOUTS == ("zigzag", "zigzag_per_document")
    assert kcp.WALK_PAIR_MAX_WORLD == 8
    ctx = types.SimpleNamespace(
        partition=types.SimpleNamespace(layout=layout, world_size=world)
    )
    assert kcp._walk_pair_enabled(knob, ctx) is on
    assert kcp._walk_pair_enabled(knob, types.SimpleNamespace()) is False


class _Stream:
    def __init__(self, name: str, log: list) -> None:
        self.name, self.log = name, log

    def wait_event(self, event) -> None:
        self.log.append(("wait", self.name, event.name))


def _walk_pair_stub(monkeypatch, *, walk_pair: bool, device: str):
    """A stand-in call for ``cake_kcp._walk_pair`` with CPU stand-ins for the CUDA stream / event API: every
    stream operation and every backend launch (with the stream it was issued on) goes to the returned log; the call's
    ``kernel`` is the host module's own ``CakeGDNNondetKcpCall.kernel`` (the deferral under test)."""
    kcp = _mod("cake_kcp")
    log: list = []
    current, side = _Stream("current", log), _Stream("side", log)
    active = [current]
    names = iter(f"ev{i}" for i in range(100))

    class _Event:
        def __init__(self) -> None:
            self.name = next(names)

        def record(self, stream) -> None:
            log.append(("record", self.name, stream.name))

    class _StreamContext:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self) -> None:
            active.append(self.stream)

        def __exit__(self, *exc) -> None:
            active.pop()

    class _Backend:
        def run(self, name: str, args: dict) -> None:
            log.append(("run", name, active[-1].name))

    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current)
    monkeypatch.setattr(torch.cuda, "stream", _StreamContext)
    monkeypatch.setattr(kcp, "_side_stream", lambda device: side)
    call = types.SimpleNamespace(
        walk_pair=walk_pair,
        device=torch.device(device),
        _deferred=None,
        backend=_Backend(),
        counters={"launches": 0},
        sequence=[],
    )

    def launch(name: str):
        return lambda: kcp.CakeGDNNondetKcpCall.kernel(call, name, 1)

    return kcp, call, log, launch


def test_walk_pair_forks_and_joins_around_the_side_launches(monkeypatch) -> None:
    """``walk_pair=1``: the offset walk's launch is resolved first and issued on the side stream after the side stream
    waits on a fork event recorded on the current stream; the transfer walk follows on the current stream, which then
    waits on the side stream's join event.  The call's launch sequence is the one-stream order."""
    kcp, call, log, launch = _walk_pair_stub(monkeypatch, walk_pair=True, device="cuda")
    paired = kcp.WALK_PAIR_STATS["paired"]
    kcp._walk_pair(call, launch("kcp_fwdh_offset"), launch("kcp_transfer_walk"))
    assert call.sequence == ["kcp_fwdh_offset", "kcp_transfer_walk"]
    assert call._deferred is None
    assert log == [
        ("record", "ev0", "current"),
        ("wait", "side", "ev0"),
        ("run", "kcp_fwdh_offset", "side"),
        ("record", "ev1", "side"),
        ("run", "kcp_transfer_walk", "current"),
        ("wait", "current", "ev1"),
    ]
    assert kcp.WALK_PAIR_STATS["paired"] == paired + 1


@pytest.mark.parametrize("walk_pair, device", [(False, "cuda"), (True, "cpu")])
def test_walk_pair_off_or_on_the_cpu_is_one_stream(
    monkeypatch, walk_pair, device
) -> None:
    kcp, call, log, launch = _walk_pair_stub(
        monkeypatch, walk_pair=walk_pair, device=device
    )
    kcp._walk_pair(call, launch("kcp_fwdh_offset"), launch("kcp_transfer_walk"))
    assert log == [
        ("run", "kcp_fwdh_offset", "current"),
        ("run", "kcp_transfer_walk", "current"),
    ]
    assert call.sequence == ["kcp_fwdh_offset", "kcp_transfer_walk"]


def test_walk_pair_with_an_empty_side_walk_issues_on_the_current_stream(
    monkeypatch,
) -> None:
    kcp, call, log, launch = _walk_pair_stub(monkeypatch, walk_pair=True, device="cuda")
    kcp._walk_pair(call, lambda: None, launch("kcp_transfer_walk"))
    assert log == [("run", "kcp_transfer_walk", "current")]
    assert call._deferred is None


def test_options_from_instance_mapping_and_keywords_agree() -> None:
    a = make_options(CakeGDNNondetKcpOptions(save_mode="full"), boundary_mode="replay")
    b = make_options({"save_mode": "full", "boundary_mode": "replay"})
    c = make_options(save_mode="full", boundary_mode="replay")
    assert a == b == c
    assert a.key() == b.key() == c.key()
    assert hash(a.key()) == hash(c.key())
    assert (
        make_options(test_knobs={"schedule": "overlap"}).key() != make_options().key()
    )
    with pytest.raises(TypeError):
        make_options(["save_mode"])  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------------------------------
# signatures: every knob keyword-only


def _after(sig: inspect.Signature, anchor: str) -> list[inspect.Parameter]:
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    return params[names.index(anchor) + 1 :]


def test_entry_points_take_every_knob_keyword_only() -> None:
    kcp = _mod("cake_kcp")
    for fn, anchor in (
        (kcp.cake_gdn_nondet_kcp, "ctx"),
        (kcp.cake_gdn_nondet_train_kcp_forward, "ctx"),
        (kcp.cake_gdn_nondet_train_kcp_backward, "dout"),
    ):
        sig = inspect.signature(fn)
        for p in _after(sig, anchor):
            assert p.kind is p.KEYWORD_ONLY, (fn.__name__, p.name)
    sig = inspect.signature(kcp.cake_gdn_nondet_kcp)
    # unset: "light_akk" with use_qk_l2norm_in_kernel=True, "light" without it (CakeGDNNondetKcpOptions)
    assert sig.parameters["save_mode"].default is None
    assert sig.parameters["boundary_mode"].default == "saved"
    assert sig.parameters["graph"].default == "off"
    assert sig.parameters["scale"].default is None


# The public surface of the facade: the functions and the types it resolves on first use (the exact ``__all__``).
FACADE_FUNCTIONS = (
    "cake_gdn_nondet_kcp",
    "cake_gdn_nondet_train_kcp_forward",
    "cake_gdn_nondet_train_kcp_backward",
    "cake_gdn_nondet_train_kcp_partition",
    "cake_gdn_nondet_train_kcp_build_context",
    "cake_gdn_nondet_train_kcp_packed_qkv_views",
    "cake_gdn_nondet_train_kcp_workspace_bytes",
    "cake_gdn_nondet_train_kcp_saved_bytes",
    "cake_gdn_nondet_train_kcp_comm_bytes",
    "is_cake_gdn_nondet_train_kcp_supported",
)
FACADE_TYPES = (
    "CakeGDNNondetKcpContext",
    "CakeGDNNondetKcpOptions",
    "CakeGDNNondetKcpPartition",
    "CakeGDNNondetKcpRecomputeHandle",
    "CakeGDNNondetKcpSaved",
    "CakeGDNNondetKcpWorkspace",
    "CakeGDNNondetKcpFunction",
)
# The package root exports the facade's names except the support query, plus three package-level objects.
PACKAGE_NAMES = (set(FACADE_FUNCTIONS) - {"is_cake_gdn_nondet_train_kcp_supported"}) | {
    *FACADE_TYPES,
    "CakeGDNNondetKcpFragment",
    "CakeGDNNondetKcpCall",
    "CakeGDNNondetKcpPackedQKVSplitFunction",
}


def _det(sub: str | None = None):
    """The deterministic package (or one of its modules) through the test-helper registry: a stale entry fails here
    and never skips."""
    from tests.cake.test_helpers.cake_package import cake_package

    return cake_package("cake_gdn_det_train_kcp", sub)


def test_facade_exposes_the_contract_names_keyword_only() -> None:
    facade = importlib.import_module(FACADE)
    assert set(facade.__all__) == set(FACADE_FUNCTIONS) | set(FACADE_TYPES)
    assert len(facade.__all__) == len(set(facade.__all__))
    for n in FACADE_FUNCTIONS:
        assert callable(getattr(facade, n)), n
    assert set(FACADE_TYPES) <= set(dir(facade))
    for fn, anchor in (
        (facade.cake_gdn_nondet_kcp, "ctx"),
        (facade.cake_gdn_nondet_train_kcp_forward, "ctx"),
        (facade.cake_gdn_nondet_train_kcp_backward, "dout"),
    ):
        for p in _after(inspect.signature(fn), anchor):
            assert p.kind is p.KEYWORD_ONLY, (fn.__name__, p.name)
    for fn in (
        facade.cake_gdn_nondet_train_kcp_workspace_bytes,
        facade.cake_gdn_nondet_train_kcp_saved_bytes,
        facade.cake_gdn_nondet_train_kcp_comm_bytes,
    ):
        for p in _after(inspect.signature(fn), "ctx"):
            assert p.kind is p.KEYWORD_ONLY, (fn.__name__, p.name)
    for p in _after(
        inspect.signature(facade.cake_gdn_nondet_train_kcp_packed_qkv_views), "qkv"
    ):
        assert p.kind == p.KEYWORD_ONLY, p.name
    pkg = _mod()
    for n in FACADE_TYPES:
        assert getattr(facade, n) is getattr(pkg, n), n
    # the default scale keeps its name (an ALL_CAPS constant, not in ``__all__``): the package's SCALE
    assert "DEFAULT_SCALE" not in facade.__all__
    assert facade.DEFAULT_SCALE == 128**-0.5 == _mod("cake_kcp").SCALE
    with pytest.raises(AttributeError):
        facade.no_such_name  # noqa: B018


def test_package_init_exports_exactly_the_mode_named_names() -> None:
    """The package root exports the public names only (every one names the mode); the internals (``SCALE``,
    ``LAST_CALL``, the backend controls, the option helpers, the plan internals) stay in their submodules."""
    pkg = _mod()
    assert set(pkg.__all__) == PACKAGE_NAMES
    assert len(pkg.__all__) == len(set(pkg.__all__))
    for n in PACKAGE_NAMES:
        assert hasattr(pkg, n), n
    kcp, opt_mod = _mod("cake_kcp"), _mod("cake_options")
    for n in (
        "SCALE",
        "LAST_CALL",
        "PROGRAMS",
        "KCP_SAVED_VERSION",
        "Backend",
        "KernelBackend",
        "FakeBackend",
        "get_backend",
        "set_backend",
        "graph_cache_clear",
    ):
        assert hasattr(kcp, n), n
        assert not hasattr(pkg, n), n
    for n in ("make_options", "validate_heads", "validate_ring"):
        assert hasattr(opt_mod, n), n
        assert not hasattr(pkg, n), n
    assert pkg.cake_gdn_nondet_kcp is kcp.cake_gdn_nondet_kcp
    assert pkg.CakeGDNNondetKcpFunction is kcp.CakeGDNNondetKcpFunction
    assert pkg.CakeGDNNondetKcpFunction.__name__ == "CakeGDNNondetKcpFunction"
    assert pkg.CakeGDNNondetKcpFunction.__qualname__ == "CakeGDNNondetKcpFunction"


# One word of a name part: an acronym run before a capitalized word, a capitalized or lower-case word, the rest of an
# acronym run, digits.
_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _name_tokens(name: str) -> list[str]:
    """The lower-cased words of ``name``, split on ``_`` and on CamelCase boundaries, an acronym run as one word
    (``CakeGDNNondetKcpFunction``: ``cake``, ``gdn``, ``nondet``, ``kcp``, ``function``)."""
    return [word.lower() for part in name.split("_") for word in _WORD.findall(part)]


def test_every_public_name_names_the_nondet_mode() -> None:
    """Every name of the facade's and the package's ``__all__`` has the mode word ``nondet`` as a word of its own and
    no word ``det`` (see ``_name_tokens``: ``det`` is a substring of ``nondet``, so a substring check cannot tell the
    modes apart)."""
    # the split itself: an acronym run is one word, and ``det`` is a word of a det name only
    for name, words in (
        ("CakeGDNNondetKcpFunction", "cake gdn nondet kcp function"),
        (
            "CakeGDNNondetKcpPackedQKVSplitFunction",
            "cake gdn nondet kcp packed qkv split function",
        ),
        (
            "is_cake_gdn_nondet_train_kcp_supported",
            "is cake gdn nondet train kcp supported",
        ),
        ("CakeGDNDetKcpFunction", "cake gdn det kcp function"),
        ("cake_gdn_det_train_kcp_forward", "cake gdn det train kcp forward"),
    ):
        assert _name_tokens(name) == words.split(), name
    facade = importlib.import_module(FACADE)
    names = set(facade.__all__) | set(_mod().__all__)
    assert names == set(FACADE_FUNCTIONS) | PACKAGE_NAMES
    for name in sorted(names):
        tokens = _name_tokens(name)
        assert "nondet" in tokens and "det" not in tokens, (name, tokens)


def test_importing_the_facade_does_not_import_the_package() -> None:
    """A fresh interpreter imports the facade without importing the backend package; the package is imported on the
    first use of a facade name, or on the first read of the parameters of an entry that hands its option keywords to
    the package (``inspect.signature`` alone imports nothing)."""
    code = (
        "import inspect\n"
        "import sys\n"
        f"import {FACADE} as facade\n"
        f"assert {PACKAGE!r} not in sys.modules, 'imported at facade import'\n"
        "sig = inspect.signature(facade.cake_gdn_nondet_kcp)\n"
        f"assert {PACKAGE!r} not in sys.modules, 'imported by inspect.signature'\n"
        "assert 'save_transfer' in sig.parameters\n"
        f"assert {PACKAGE!r} in sys.modules, 'not imported on the first read of the parameters'\n"
    )
    root = pathlib.Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]


def test_the_facade_is_a_stable_api() -> None:
    # real ``@flashinfer_api`` defs of the facade module, not marked by the experimental-API decorator (it sets
    # ``is_experimental`` and ``experimental_feature`` and puts a ``.. warning::`` banner first in the docstring)
    facade = importlib.import_module(FACADE)
    for name in FACADE_FUNCTIONS:
        entry = getattr(facade, name)
        assert inspect.unwrap(entry).__module__ == FACADE, name
        assert not getattr(entry, "is_experimental", False), name
        assert not hasattr(entry, "experimental_feature"), name
        assert not entry.__doc__.lstrip().startswith(".. warning::"), name
    # the source: every entry is a top-level def with the one decorator ``@flashinfer_api`` (at the default log level
    # the decorator returns the function itself, so only the source shows it), the only name taken from
    # ``api_logging``; under it the entries of OPTION_KEYWORD_ENTRIES carry the facade's ``_forwards_keywords``, which
    # declares the keywords they hand to the package (``test_option_keyword_entries_declare_every_option_field``)
    tree = ast.parse(_facade_source().read_text())
    decorators = {
        node.name: [ast.unparse(d) for d in node.decorator_list]
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in FACADE_FUNCTIONS:
        want = ["flashinfer_api"]
        if name in OPTION_KEYWORD_ENTRIES:
            want.append("_forwards_keywords")
        assert decorators.get(name) == want, (name, decorators.get(name))
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "api_logging"
        for alias in node.names
    ]
    assert imported == ["flashinfer_api"], imported
    # a fresh interpreter, since the experimental-API decorator warns on the first call of a process only: the first
    # call of every facade entry, with a keyword the entry does not take, fails with a TypeError on host operands
    # (at the signature, or in the package before any work for the entries that take ``**knobs`` / ``**kw``) and
    # warns of nothing.  The package is imported before the calls, so only the entries' own first calls are recorded.
    code = (
        "import warnings\n"
        "import torch\n"
        f"import {FACADE} as facade\n"
        "pkg = facade._pkg()\n"
        f"part = pkg.cake_gdn_nondet_train_kcp_partition({MODE_CU!r}, 1, 'contiguous')\n"
        "ctx = pkg.cake_gdn_nondet_train_kcp_build_context(part, None, 0, 'cpu', sm_count=148)\n"
        f"T = {MODE_CU[-1]}\n"
        "q = torch.zeros(T, 2, 128, dtype=torch.bfloat16)\n"
        "v = torch.zeros(T, 6, 128, dtype=torch.bfloat16)\n"
        "g, beta = torch.zeros(T, 6), torch.zeros(T, 6)\n"
        "bad = {'deterministic': False}\n"
        "calls = {\n"
        "    'cake_gdn_nondet_train_kcp_partition': lambda: facade.cake_gdn_nondet_train_kcp_partition(\n"
        f"        {MODE_CU!r}, 1, **bad),\n"
        "    'cake_gdn_nondet_train_kcp_build_context': lambda: facade.cake_gdn_nondet_train_kcp_build_context(\n"
        "        part, None, 0, 'cpu', **bad),\n"
        "    'cake_gdn_nondet_kcp': lambda: facade.cake_gdn_nondet_kcp(\n"
        "        q, q, v, g, beta, ctx, backend='fake', **bad),\n"
        "    'cake_gdn_nondet_train_kcp_forward': lambda: facade.cake_gdn_nondet_train_kcp_forward(\n"
        "        q, q, v, g, beta, ctx, backend='fake', **bad),\n"
        "    'cake_gdn_nondet_train_kcp_backward': lambda: facade.cake_gdn_nondet_train_kcp_backward(\n"
        "        q, q, v, g, beta, ctx, v, backend='fake', **bad),\n"
        "    'cake_gdn_nondet_train_kcp_packed_qkv_views': lambda: facade.cake_gdn_nondet_train_kcp_packed_qkv_views(\n"
        "        torch.zeros(4, 10 * 128, dtype=torch.bfloat16), H_qk=2, HV=6, **bad),\n"
        "    'cake_gdn_nondet_train_kcp_workspace_bytes': lambda: facade.cake_gdn_nondet_train_kcp_workspace_bytes(\n"
        "        ctx, H_qk=2, HV=6, **bad),\n"
        "    'cake_gdn_nondet_train_kcp_saved_bytes': lambda: facade.cake_gdn_nondet_train_kcp_saved_bytes(\n"
        "        ctx, H_qk=2, HV=6, **bad),\n"
        "    'cake_gdn_nondet_train_kcp_comm_bytes': lambda: facade.cake_gdn_nondet_train_kcp_comm_bytes(\n"
        "        ctx, H_qk=2, HV=6, **bad),\n"
        "    'is_cake_gdn_nondet_train_kcp_supported': lambda: facade.is_cake_gdn_nondet_train_kcp_supported(\n"
        "        'cpu', **bad),\n"
        "}\n"
        f"assert sorted(calls) == {sorted(FACADE_FUNCTIONS)!r}, sorted(calls)\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    for name, call in calls.items():\n"
        "        try:\n"
        "            call()\n"
        "        except TypeError as exc:\n"
        "            assert 'deterministic' in str(exc), (name, exc)\n"
        "        else:\n"
        "            raise AssertionError(name + ' took deterministic=')\n"
        "assert not caught, [(w.category.__name__, str(w.message)) for w in caught]\n"
        "print('ok', len(calls))\n"
    )
    root = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().splitlines()[-1] == f"ok {len(FACADE_FUNCTIONS)}"


# The former names of this route (#8928) and the deterministic package's pre-rename spellings: no module of the
# package, nor the facade, binds or exports one.
OLD_NAMES = frozenset(
    {
        # this route's former names
        "chunk_gdn_kcp",
        "ChunkGDNFunctionKcp",
        "chunk_gdn_fwd_kcp",
        "chunk_gdn_bwd_kcp",
        "partition_kcp",
        "build_context_kcp",
        "packed_qkv_views_kcp",
        "gdn_workspace_bytes_kcp",
        "gdn_saved_bytes_kcp",
        "gdn_comm_bytes_kcp",
        "gdn_saved_layout_kcp",
        "gdn_workspace_layout_kcp",
        "is_gdn_train_kcp_supported",
        "ContextKcp",
        "PartitionKcp",
        "FragmentKcp",
        "OptionsKcp",
        "RecomputeHandleKcp",
        "SavedKcp",
        "WorkspaceKcp",
        "CallKcp",
        "PackedQKVSplitKcp",
        "MODE",
        "check_mode",
        "ND_ONLY_SAVE_MODES",
        # the deterministic package's pre-rename spellings
        "kcp_partition",
        "build_kcp_context",
        "chunk_gdn_kcp_fwd",
        "chunk_gdn_kcp_bwd",
        "gdn_kcp_workspace_bytes",
        "gdn_kcp_saved_bytes",
        "gdn_kcp_comm_bytes",
        "gdn_kcp_saved_layout",
        "gdn_kcp_workspace_layout",
        "is_gdn_train_supported",
        "ChunkGDNKcpFunction",
        "KcpCall",
        "KcpContext",
        "KcpFragment",
        "KcpOptions",
        "KcpPartition",
        "KcpRecomputeHandle",
        "KcpSaved",
        "KcpWorkspace",
    }
)
# The former module paths of this route: they fail at import (no alias, no shim).
OLD_MODULES = (
    "flashinfer.cake_gdn_train_kcp",
    "flashinfer.experimental.cake_gdn_train_kcp",
)
SUBMODULES = (
    "cake_backend",
    "cake_build",
    "cake_bytes",
    "cake_fake_backend",
    "cake_jit",
    "cake_kcp",
    "cake_kcp_plan",
    "cake_l2n",
    "cake_options",
    "cake_tables",
)


def test_no_module_of_the_package_or_the_facade_binds_an_old_name() -> None:
    """The package, each of its modules and the facade neither bind nor export a former name of this route or a
    pre-rename spelling of the deterministic package."""
    facade = importlib.import_module(FACADE)
    pkg = _mod()
    for module in [pkg, facade] + [_mod(sub) for sub in SUBMODULES]:
        exported = set(getattr(module, "__all__", ()))
        for name in sorted(OLD_NAMES):
            assert not hasattr(module, name), (module.__name__, name)
            assert name not in exported, (module.__name__, name)
    entry = _mod("cake_kcp").cake_gdn_nondet_kcp
    assert pkg.cake_gdn_nondet_kcp is entry
    assert entry.__module__ == f"{PACKAGE}.cake_kcp"
    assert entry.__name__ == "cake_gdn_nondet_kcp"
    for name in ("chunk_gdn_kcp", "partition_kcp", "OptionsKcp"):
        with pytest.raises(ImportError):
            exec(f"from {FACADE} import {name}", {})
        with pytest.raises(ImportError):
            exec(f"from {PACKAGE} import {name}", {})


def test_the_former_modules_fail_at_import() -> None:
    """``flashinfer.cake_gdn_train_kcp`` and ``flashinfer.experimental.cake_gdn_train_kcp`` raise
    ``ModuleNotFoundError``.  A checkout may keep an untracked leftover of the former package directory (``aot/``,
    ``__pycache__``), which Python imports as an empty namespace package: such a leftover holds no module and none of
    the former names."""
    for old in OLD_MODULES:
        spec = importlib.util.find_spec(old)
        if (
            spec is not None
            and spec.origin in (None, "namespace")
            and spec.submodule_search_locations
        ):
            mod = importlib.import_module(old)
            for name in sorted(OLD_NAMES):
                assert not hasattr(mod, name), (old, name)
            for sub in SUBMODULES:
                with pytest.raises(ModuleNotFoundError):
                    importlib.import_module(f"{old}.{sub}")
            continue
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old)


def test_the_package_shares_no_entry_or_type_with_the_deterministic_package() -> None:
    """This package's entries (also through the facade) and types are never the deterministic package's objects."""
    det_pkg = _det()
    facade = importlib.import_module(FACADE)
    for nondet_entry in (
        _mod().cake_gdn_nondet_kcp,
        _mod("cake_kcp").cake_gdn_nondet_kcp,
        facade.cake_gdn_nondet_kcp,
    ):
        assert nondet_entry is not det_pkg.cake_gdn_det_kcp
    for name in (
        "CakeGDNNondetKcpOptions",
        "CakeGDNNondetKcpContext",
        "CakeGDNNondetKcpPartition",
        "CakeGDNNondetKcpSaved",
        "CakeGDNNondetKcpWorkspace",
        "CakeGDNNondetKcpFunction",
    ):
        assert getattr(_mod(), name) is not getattr(
            det_pkg, name.replace("Nondet", "Det")
        ), name


# ----------------------------------------------------------------------------------------------------------------
# the numerical mode: in the name only (no ``deterministic`` keyword; fixed-order launches)

ENTRIES = (
    "cake_gdn_nondet_kcp",
    "cake_gdn_nondet_train_kcp_forward",
    "cake_gdn_nondet_train_kcp_backward",
)
MODE_CU = [0, 100, 230, 307]  # three documents, boundaries inside chunks
MODE_HEADS = (2, 6)


@pytest.fixture
def torch_deterministic():
    """Restore torch's deterministic-algorithms setting (and its warn-only flag) after the test."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _mode_ops(
    cu=MODE_CU, heads=MODE_HEADS, seed: int = 8063
) -> dict[str, torch.Tensor]:
    """Seeded CPU operands without the batch axis: q / k BF16 L2-normalised rows, v / dout BF16, g < 0 FP32 (the
    natural-log decay), beta in (0, 1) FP32."""
    H_qk, HV = heads
    T = cu[-1]
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def l2(x):
        return (x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        )

    return {
        "q": l2(torch.randn(T, H_qk, 128, generator=gen)),
        "k": l2(torch.randn(T, H_qk, 128, generator=gen)),
        "v": torch.randn(T, HV, 128, generator=gen).to(torch.bfloat16),
        "g": (-torch.nn.functional.softplus(torch.randn(T, HV, generator=gen))).float(),
        "beta": torch.sigmoid(torch.randn(T, HV, generator=gen)).float(),
        "dout": torch.randn(T, HV, 128, generator=gen).to(torch.bfloat16),
    }


def _mode_case():
    """A one-rank CPU context of :data:`MODE_CU` and its operands."""
    plan = _mod("cake_kcp_plan")
    part = plan.cake_gdn_nondet_train_kcp_partition(MODE_CU, 1, "contiguous")
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(
        part, None, 0, "cpu", sm_count=148
    )
    return ctx, _mode_ops()


def _call(entry: str, ctx, ops, module=None, **kw):
    """One compute entry of the package (or of ``module``, the facade) on the fake backend
    (``cake_gdn_nondet_train_kcp_backward`` without a saved set: the replay)."""
    mod = _mod("cake_kcp") if module is None else module
    args = [ops[n] for n in ("q", "k", "v", "g", "beta")]
    if entry == "cake_gdn_nondet_train_kcp_backward":
        return mod.cake_gdn_nondet_train_kcp_backward(
            *args, ctx, ops["dout"], backend="fake", **kw
        )
    return getattr(mod, entry)(*args, ctx, backend="fake", **kw)


def _callables(module) -> dict:
    """The callables of ``module.__all__``: a function itself, a class by its constructor, an autograd Function by
    its ``forward`` and ``backward`` (and by its ``apply`` when that signature names its parameters)."""
    var_kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    found = {}
    for name in module.__all__:
        obj = getattr(module, name)
        assert callable(obj), name
        if isinstance(obj, type) and issubclass(obj, torch.autograd.Function):
            found[f"{name}.forward"] = obj.forward
            found[f"{name}.backward"] = obj.backward
            # ``torch.autograd.Function.apply`` takes ``*args, **kwargs``: its signature binds any keyword, so the
            # check reads ``forward``, the parameters ``apply`` passes on, and ``apply`` only if it names them.  Some
            # torch builds expose ``apply`` as a builtin without a signature (``ValueError``): it names nothing then
            try:
                params = list(inspect.signature(obj.apply).parameters.values())
            except ValueError:
                params = None
            if params is not None and all(p.kind not in var_kinds for p in params):
                found[f"{name}.apply"] = obj.apply
        else:
            found[name] = obj
    return found


# The entries that take the option keywords as ``**knobs`` / ``**kw``.  Their signatures declare the call's keywords,
# with no ``**`` parameter (``test_option_keyword_entries_declare_every_option_field``), so a stray ``deterministic=``
# fails ``inspect.signature(...).bind_partial`` as it fails the call, which refuses it with ``TypeError`` before any
# work (``test_a_deterministic_keyword_is_a_type_error``, ``test_the_other_entries_refuse_a_deterministic_keyword``).
OPTION_KEYWORD_ENTRIES = frozenset(
    {
        "cake_gdn_nondet_kcp",
        "cake_gdn_nondet_train_kcp_forward",
        "cake_gdn_nondet_train_kcp_backward",
        "cake_gdn_nondet_train_kcp_workspace_bytes",
        "cake_gdn_nondet_train_kcp_saved_bytes",
        "cake_gdn_nondet_train_kcp_comm_bytes",
    }
)


def test_no_public_callable_takes_a_deterministic_parameter() -> None:
    """Every callable of the facade's and the package's ``__all__`` (the types by their constructors, the autograd
    Functions by ``forward`` and ``backward``) refuses ``deterministic=`` at its signature:
    ``inspect.signature(...).bind_partial(deterministic=False)`` raises ``TypeError``.  No signature has a ``**``
    parameter: the entries that take the option keywords through ``**knobs`` / ``**kw``
    (:data:`OPTION_KEYWORD_ENTRIES`) declare those keywords, and they refuse ``deterministic=`` at the call too."""
    facade = importlib.import_module(FACADE)
    for module, expected in (
        (facade, set(FACADE_FUNCTIONS) | set(FACADE_TYPES)),
        (_mod(), PACKAGE_NAMES),
    ):
        callables = _callables(module)
        assert {
            "CakeGDNNondetKcpFunction.forward",
            "CakeGDNNondetKcpFunction.backward",
            "cake_gdn_nondet_train_kcp_partition",
            "cake_gdn_nondet_train_kcp_build_context",
            "CakeGDNNondetKcpOptions",
        } <= set(callables), module.__name__
        assert {n.split(".")[0] for n in callables} == expected, module.__name__
        assert set(callables) >= OPTION_KEYWORD_ENTRIES, module.__name__
        for name, fn in callables.items():
            sig = inspect.signature(fn)
            assert "deterministic" not in sig.parameters, (module.__name__, name)
            assert all(p.kind is not p.VAR_KEYWORD for p in sig.parameters.values()), (
                module.__name__,
                name,
            )
            try:
                sig.bind_partial(deterministic=False)
            except TypeError as exc:
                assert "deterministic" in str(exc), (module.__name__, name, exc)
            else:
                pytest.fail(f"{module.__name__}.{name} binds deterministic=")
    plan = _mod("cake_kcp_plan")
    for fn in (
        plan.cake_gdn_nondet_train_kcp_partition,
        plan.cake_gdn_nondet_train_kcp_build_context,
    ):
        assert "deterministic" not in fn.__code__.co_varnames, fn.__name__


def _own_keyword_only(fn, name: str) -> set[str]:
    """The keyword-only parameters that the def of ``fn`` names itself, read from its code object (through a
    ``functools.wraps`` wrapper of ``@flashinfer_api`` at a raised log level), not from its declared signature."""
    raw = inspect.unwrap(
        fn,
        stop=lambda f: getattr(getattr(f, "__code__", None), "co_name", None) == name,
    )
    code = raw.__code__
    assert code.co_name == name, (name, code.co_name)
    first = code.co_argcount
    return set(code.co_varnames[first : first + code.co_kwonlyargcount])


def test_option_keyword_entries_declare_every_option_field() -> None:
    """The entries of :data:`OPTION_KEYWORD_ENTRIES` take their option keywords through ``**knobs`` / ``**kw``, and
    their signatures (facade, package root, ``cake_kcp``) declare the call's keywords, with no ``**`` parameter and
    no ``deterministic``.  A compute entry declares the keywords its def names plus one keyword-only parameter per
    :class:`CakeGDNNondetKcpOptions` field (the field's default where the def does not name it); the facade declares
    the package's keywords.  A byte query of the package is the byte module's function of its name, and the facade
    declares the keyword-only parameters of that function (its own ``C`` default: 10240 channels)."""
    facade = importlib.import_module(FACADE)
    pkg, kcp, nbytes = _mod(), _mod("cake_kcp"), _mod("cake_bytes")
    fields = {f.name: f.default for f in dataclasses.fields(CakeGDNNondetKcpOptions)}
    assert all(d is not dataclasses.MISSING for d in fields.values()), fields
    for name in sorted(OPTION_KEYWORD_ENTRIES):
        byte_query = name.endswith("_bytes")
        if byte_query:
            reference = getattr(nbytes, name)
            assert (
                getattr(kcp, name) is reference and getattr(pkg, name) is reference
            ), name
        else:
            reference = getattr(kcp, name)
            assert getattr(pkg, name) is reference, name
        ref_params = inspect.signature(reference).parameters
        ref_kw = {n for n, p in ref_params.items() if p.kind is p.KEYWORD_ONLY}
        for module in (facade, pkg, kcp):
            fn = getattr(module, name)
            params = inspect.signature(fn).parameters
            where = (module.__name__, name)
            assert all(p.kind is not p.VAR_KEYWORD for p in params.values()), where
            assert "deterministic" not in params, where
            kw_only = {n for n, p in params.items() if p.kind is p.KEYWORD_ONLY}
            assert kw_only == ref_kw, (*where, kw_only ^ ref_kw)
            own = _own_keyword_only(fn, name)
            if byte_query:
                assert own <= kw_only, (*where, own - kw_only)
                for n in kw_only - own:
                    assert params[n].default == ref_params[n].default, (*where, n)
            else:
                assert kw_only == own | set(fields), (
                    *where,
                    kw_only ^ (own | set(fields)),
                )
                for n in set(fields) - own:
                    assert params[n].default == fields[n], (
                        *where,
                        n,
                        params[n].default,
                    )
            try:
                inspect.signature(fn).bind_partial(deterministic=False)
            except TypeError as exc:
                assert "deterministic" in str(exc), (*where, exc)
            else:
                pytest.fail(f"{module.__name__}.{name} binds deterministic=")
    comm = inspect.signature(facade.cake_gdn_nondet_train_kcp_comm_bytes).parameters
    assert (
        comm["C"].kind is inspect.Parameter.KEYWORD_ONLY and comm["C"].default == 10240
    )
    # the facade entries stay plain functions: the declared signature is their ``__signature__`` (no ``__wrapped__``
    # stand-in), so their source and type hints are their own
    for name in sorted(OPTION_KEYWORD_ENTRIES):
        fn = getattr(facade, name)
        sig = inspect.signature(fn)
        assert isinstance(sig, inspect.Signature), name
        assert str(sig).startswith("("), name
        assert sig.replace(return_annotation="int").return_annotation == "int", name
        raw = inspect.unwrap(fn)
        assert raw.__name__ == name and raw.__module__ == FACADE, name
        assert f"def {name}(" in inspect.getsource(fn), name
        assert "return" in typing.get_type_hints(fn), name


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("entry", ENTRIES)
def test_a_deterministic_keyword_is_a_type_error(entry, value) -> None:
    """A stray ``deterministic=`` keyword (``True`` or ``False``) is refused with ``TypeError`` before any work, on
    the package's compute entries and on the facade's: it never lands silently in the options."""
    ctx, ops = _mode_case()
    facade = importlib.import_module(FACADE)
    for module in (None, facade):
        with pytest.raises(TypeError, match="deterministic"):
            _call(entry, ctx, ops, module=module, deterministic=value)


def test_the_other_entries_refuse_a_deterministic_keyword() -> None:
    """The plan, byte-query, view and support entries of the facade, and the plan, byte-query and view entries of the
    package root, refuse ``deterministic=`` with ``TypeError``."""
    facade = importlib.import_module(FACADE)
    ctx, _ops = _mode_case()
    kw = {"deterministic": False}
    for module in (facade, _mod()):
        calls = [
            lambda m=module: m.cake_gdn_nondet_train_kcp_partition(MODE_CU, 1, **kw),
            lambda m=module: m.cake_gdn_nondet_train_kcp_build_context(
                ctx.partition, None, 0, "cpu", **kw
            ),
            lambda m=module: m.cake_gdn_nondet_train_kcp_workspace_bytes(
                ctx, H_qk=2, HV=6, **kw
            ),
            lambda m=module: m.cake_gdn_nondet_train_kcp_saved_bytes(
                ctx, H_qk=2, HV=6, **kw
            ),
            lambda m=module: m.cake_gdn_nondet_train_kcp_comm_bytes(
                ctx, H_qk=2, HV=6, **kw
            ),
            lambda m=module: m.cake_gdn_nondet_train_kcp_packed_qkv_views(
                torch.zeros(4, 10 * 128, dtype=torch.bfloat16), H_qk=2, HV=6, **kw
            ),
        ]
        if module is facade:
            calls.append(
                lambda: facade.is_cake_gdn_nondet_train_kcp_supported("cpu", **kw)
            )
        for call in calls:
            with pytest.raises(TypeError, match="deterministic"):
                call()


def test_the_support_query_returns_a_bool() -> None:
    facade = importlib.import_module(FACADE)
    assert isinstance(facade.is_cake_gdn_nondet_train_kcp_supported(), bool)
    assert facade.is_cake_gdn_nondet_train_kcp_supported("cpu") is False
    assert isinstance(facade.is_cake_gdn_nondet_train_kcp_supported(0), bool)


def test_package_entries_run_on_the_fake_backend() -> None:
    """The forward, the saved-set backward, the replay backward and the autograd node run on the fake backend
    (finite outputs of the contract's shapes and dtypes; the accuracy is the fake-backend suite's)."""
    kcp = _mod("cake_kcp")
    ctx, ops = _mode_case()
    args = [ops[n] for n in ("q", "k", "v", "g", "beta")]
    out, saved = kcp.cake_gdn_nondet_train_kcp_forward(*args, ctx, backend="fake")
    assert isinstance(saved, kcp.CakeGDNNondetKcpSaved)
    assert out.shape == ops["v"].shape and out.dtype == torch.bfloat16
    for saved_set in (saved, None):
        grads = kcp.cake_gdn_nondet_train_kcp_backward(
            *args, ctx, ops["dout"], saved=saved_set, backend="fake"
        )
        for t, like in zip(grads, ("q", "k", "v", "beta", "g"), strict=True):
            assert t.shape == ops[like].shape and t.dtype == ops[like].dtype, like
            assert bool(torch.isfinite(t.float()).all()), like
    leaves = [t.detach().clone().requires_grad_(True) for t in args]
    y = kcp.cake_gdn_nondet_kcp(*leaves, ctx, backend="fake")
    assert y.grad_fn is not None and y.grad_fn.kcp_mode == kcp._MODE
    y.backward(ops["dout"])
    assert all(
        lf.grad is not None and bool(torch.isfinite(lf.grad.float()).all())
        for lf in leaves
    )


@pytest.mark.parametrize("warn_only", [False, True])
@pytest.mark.parametrize("entry", ENTRIES)
def test_no_alert_under_torch_deterministic_algorithms(
    entry, warn_only, torch_deterministic
) -> None:
    """Every launch of this version is fixed-order, so with ``torch.use_deterministic_algorithms(True)`` (and with
    ``warn_only=True``) a call of the package runs: no ``RuntimeError``, no warning, no fallback."""
    ctx, ops = _mode_case()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _call(entry, ctx, ops)


def test_autograd_backward_runs_under_torch_deterministic_mode(
    torch_deterministic,
) -> None:
    """A forward run with torch's deterministic algorithms off and a backward with them on (and the reverse): the
    node's backward runs without an alert or a warning, and both give the same gradients (fixed-order launches)."""
    kcp = _mod("cake_kcp")
    ctx, ops = _mode_case()
    grads = []
    for fwd_det, bwd_det in ((False, True), (True, False), (False, False)):
        leaves = [
            ops[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]
        torch.use_deterministic_algorithms(fwd_det)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y = kcp.cake_gdn_nondet_kcp(*leaves, ctx, backend="fake")
            torch.use_deterministic_algorithms(bwd_det)
            y.backward(ops["dout"])
        grads.append([y.detach()] + [t.grad for t in leaves])
    for other in grads[1:]:
        for x, y in zip(grads[0], other, strict=True):
            assert torch.equal(x, y)


def test_autograd_function_refuses_a_node_of_another_package() -> None:
    """The node keeps this package's private tag (``fctx.kcp_mode = _MODE``); a backward whose node carries another
    tag (or none) is refused before any work -- on a real node and on a stand-in context."""
    kcp = _mod("cake_kcp")
    ctx, ops = _mode_case()
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    y = kcp.cake_gdn_nondet_kcp(*leaves, ctx, backend="fake")
    assert y.grad_fn.kcp_mode == "nondeterministic" == kcp._MODE
    y.grad_fn.kcp_mode = "deterministic"
    with pytest.raises(ValueError, match="forward ran in mode 'deterministic'"):
        y.backward(ops["dout"])

    class StandIn:
        saved_tensors = tuple(torch.zeros(1) for _ in range(5))
        kcp = (None, None, None, None, None)

    for mode in ("deterministic", None):
        fctx = StandIn()
        if mode is not None:
            fctx.kcp_mode = mode
        with pytest.raises(
            ValueError,
            match="^CakeGDNNondetKcpFunction.backward: the forward ran in mode",
        ):
            kcp.CakeGDNNondetKcpFunction.backward(fctx, torch.zeros(1))


def _saved_kwargs(mod) -> dict:
    """The fields of a minimal saved set (no tensor is read before the type check)."""
    return dict(
        version=mod.KCP_SAVED_VERSION,
        S_in=torch.zeros(4),
        c1_recv=None,
        M_seg=None,
        tapes=None,
        T_r=10,
        H_qk=2,
        HV=6,
        n_frag=1,
        P=1,
        Hg=3,
        scale=128**-0.5,
        save_mode="light",
        save_transfer=False,
        offset_class="fla",
        exchange="a2a",
        precision_mode=0,
        transfer_walk="tf32_hilo",
        ctx=None,
    )


def test_a_saved_set_is_read_by_its_own_packages_backward_only() -> None:
    """The package backward refuses a deterministic package's ``CakeGDNDetKcpSaved`` (and any other object) with
    ``ValueError``; so does the facade, which hands the saved set to the package."""
    kcp = _mod("cake_kcp")
    det_kcp = _det("cake_kcp")
    det_saved = det_kcp.CakeGDNDetKcpSaved(**_saved_kwargs(det_kcp))
    ctx, ops = _mode_case()
    args = [ops[n] for n in ("q", "k", "v", "g", "beta")]
    facade = importlib.import_module(FACADE)
    for bad in (det_saved, object()):
        for module in (kcp, facade):
            with pytest.raises(
                ValueError,
                match=f"not a saved set of {PACKAGE}",
            ):
                module.cake_gdn_nondet_train_kcp_backward(
                    *args, ctx, ops["dout"], saved=bad, backend="fake"
                )


# The facade entries that call one entry of the same name in this package.
FACADE_ENTRIES = (
    "cake_gdn_nondet_train_kcp_partition",
    "cake_gdn_nondet_train_kcp_build_context",
    "cake_gdn_nondet_kcp",
    "cake_gdn_nondet_train_kcp_forward",
    "cake_gdn_nondet_train_kcp_backward",
    "cake_gdn_nondet_train_kcp_packed_qkv_views",
    "cake_gdn_nondet_train_kcp_workspace_bytes",
    "cake_gdn_nondet_train_kcp_saved_bytes",
    "cake_gdn_nondet_train_kcp_comm_bytes",
)


def _route_recorders(monkeypatch) -> list:
    """Replace every entry of this package that the facade calls by a recorder returning its tag (``nondet:<name>``);
    the calls ``(tag, kwargs)`` land in the returned list."""
    pkg = _mod()
    seen: list = []

    def recorder(tag):
        def fn(*args, **kwargs):
            seen.append((tag, kwargs))
            return tag

        return fn

    for entry in FACADE_ENTRIES:
        monkeypatch.setattr(pkg, entry, recorder(f"nondet:{entry}"))
    return seen


def test_facade_never_imports_the_deterministic_package() -> None:
    """A fresh interpreter calls every facade entry (this package's entries replaced by stubs) and the support query:
    no module of the deterministic package (``cake_gdn_det``) is imported on the way."""
    code = (
        "import sys\n"
        "import torch\n"
        f"import {FACADE} as facade\n"
        "import flashinfer.experimental.cake\n"
        "before = {m for m in sys.modules if 'cake_gdn_det' in m}\n"
        "pkg = facade._pkg()\n"
        f"for name in {FACADE_ENTRIES!r}:\n"
        "    setattr(pkg, name, lambda *a, _n=name, **k: _n)\n"
        "t, ctx = torch.zeros(1), object()\n"
        "facade.cake_gdn_nondet_train_kcp_partition([0, 4], 1)\n"
        "facade.cake_gdn_nondet_train_kcp_build_context(object(), None, 0, 'cpu')\n"
        "facade.cake_gdn_nondet_kcp(t, t, t, t, t, ctx)\n"
        "facade.cake_gdn_nondet_train_kcp_forward(t, t, t, t, t, ctx)\n"
        "facade.cake_gdn_nondet_train_kcp_backward(t, t, t, t, t, ctx, t)\n"
        "facade.cake_gdn_nondet_train_kcp_packed_qkv_views(t, H_qk=2, HV=6)\n"
        "facade.cake_gdn_nondet_train_kcp_workspace_bytes(ctx, H_qk=2, HV=6)\n"
        "facade.cake_gdn_nondet_train_kcp_saved_bytes(ctx, H_qk=2, HV=6)\n"
        "facade.cake_gdn_nondet_train_kcp_comm_bytes(ctx, H_qk=2, HV=6)\n"
        "facade.is_cake_gdn_nondet_train_kcp_supported('cpu')\n"
        "facade.is_cake_gdn_nondet_train_kcp_supported()\n"
        "new = sorted({m for m in sys.modules if 'cake_gdn_det' in m} - before)\n"
        "assert not new, new\n"
    )
    root = pathlib.Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]


def _facade_calls(facade) -> dict:
    t = torch.zeros(1)
    ctx = object()  # the recorders read nothing
    return {
        "cake_gdn_nondet_train_kcp_partition": lambda **m: (
            facade.cake_gdn_nondet_train_kcp_partition([0, 4], 1, **m)
        ),
        "cake_gdn_nondet_train_kcp_build_context": lambda **m: (
            facade.cake_gdn_nondet_train_kcp_build_context(
                object(), None, 0, "cpu", **m
            )
        ),
        "cake_gdn_nondet_kcp": lambda **m: facade.cake_gdn_nondet_kcp(
            t, t, t, t, t, ctx, **m
        ),
        "cake_gdn_nondet_train_kcp_forward": lambda **m: (
            facade.cake_gdn_nondet_train_kcp_forward(t, t, t, t, t, ctx, **m)
        ),
        "cake_gdn_nondet_train_kcp_backward": lambda **m: (
            facade.cake_gdn_nondet_train_kcp_backward(t, t, t, t, t, ctx, t, **m)
        ),
        "cake_gdn_nondet_train_kcp_packed_qkv_views": lambda **m: (
            facade.cake_gdn_nondet_train_kcp_packed_qkv_views(t, H_qk=2, HV=6, **m)
        ),
        "cake_gdn_nondet_train_kcp_workspace_bytes": lambda **m: (
            facade.cake_gdn_nondet_train_kcp_workspace_bytes(ctx, H_qk=2, HV=6, **m)
        ),
        "cake_gdn_nondet_train_kcp_saved_bytes": lambda **m: (
            facade.cake_gdn_nondet_train_kcp_saved_bytes(ctx, H_qk=2, HV=6, **m)
        ),
        "cake_gdn_nondet_train_kcp_comm_bytes": lambda **m: (
            facade.cake_gdn_nondet_train_kcp_comm_bytes(ctx, H_qk=2, HV=6, **m)
        ),
    }


# torch's deterministic-algorithms settings: off, strict, warn-only
TORCH_MODES = ((False, False), (True, False), (True, True))


@pytest.mark.parametrize("torch_mode", TORCH_MODES, ids=("off", "strict", "warn_only"))
def test_facade_runs_this_package_only(
    monkeypatch, torch_deterministic, torch_mode
) -> None:
    """Every facade entry runs this package's object (the compute entries with the contract's default scale), the
    support query included, with torch's deterministic algorithms off, on and on with ``warn_only``: no reroute to
    the deterministic package, no fallback and no warning."""
    facade = importlib.import_module(FACADE)
    facade.is_cake_gdn_nondet_train_kcp_supported()  # its first call imports the package's loader
    seen = _route_recorders(monkeypatch)
    torch.use_deterministic_algorithms(torch_mode[0], warn_only=torch_mode[1])
    calls = _facade_calls(facade)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            for entry, call in calls.items():
                seen.clear()
                assert call() == f"nondet:{entry}", entry
                assert [tag for tag, _ in seen] == [f"nondet:{entry}"], entry
                if entry in ENTRIES:
                    assert seen[-1][1]["scale"] == 128**-0.5, entry
        seen.clear()
        assert isinstance(facade.is_cake_gdn_nondet_train_kcp_supported("cpu"), bool)
        assert isinstance(facade.is_cake_gdn_nondet_train_kcp_supported(), bool)
    assert not seen
    for name in (
        "_det",
        "_deterministic",
        "_DET_FALLBACK_WARNED",
        "_refuse_nd_only",
        "_require_nondeterministic",
        "_DET_ENTRIES",
        "_DET_MODULE",
    ):
        assert not hasattr(facade, name), name


def test_facade_passes_the_light_akk_save_mode_on(
    monkeypatch, torch_deterministic
) -> None:
    """``"light_akk"`` is the save_mode value of this package that the deterministic package does not take; the
    facade passes it on unchanged (keyword, knob, options mapping or options instance), on every compute entry and in
    the saved / workspace byte queries, also under torch's deterministic algorithms (strict and ``warn_only``)."""
    facade = importlib.import_module(FACADE)
    det_opt = _det("cake_options")
    nondet_only = ("light_akk",)
    assert set(nondet_only) == set(opt.SAVE_MODES) - set(det_opt.SAVE_MODES)
    seen = _route_recorders(monkeypatch)
    calls = _facade_calls(facade)
    entries = (
        "cake_gdn_nondet_kcp",
        "cake_gdn_nondet_train_kcp_forward",
        "cake_gdn_nondet_train_kcp_backward",
        "cake_gdn_nondet_train_kcp_workspace_bytes",
        "cake_gdn_nondet_train_kcp_saved_bytes",
    )
    for mode in nondet_only:
        forms = [
            {"save_mode": mode},
            {"options": {"save_mode": mode}},
            {
                "options": CakeGDNNondetKcpOptions(
                    save_mode=mode, use_qk_l2norm_in_kernel=True
                )
            },
        ]
        for torch_mode in TORCH_MODES:
            torch.use_deterministic_algorithms(torch_mode[0], warn_only=torch_mode[1])
            for entry in entries:
                for form in forms:
                    seen.clear()
                    assert calls[entry](**form) == f"nondet:{entry}"
                    sent = seen[-1][1]
                    got = sent.get("options") if "options" in form else sent
                    got = got if isinstance(got, dict) else dataclasses.asdict(got)
                    assert got["save_mode"] == mode, (entry, form, sent, torch_mode)


def test_facade_imports_the_package_once(monkeypatch) -> None:
    """The facade imports the package on first use only: later calls run no import and reach the same module object.
    It imports nothing of the deterministic package."""
    facade = importlib.import_module(FACADE)
    _route_recorders(monkeypatch)
    calls = _facade_calls(facade).values()
    for call in calls:  # first use
        call()
    imports: list = []
    real_import = builtins.__import__

    def counting_import(name, *args, **kwargs):
        imports.append(name)
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", counting_import)
        for call in calls:
            call()
    assert imports == [], imports
    assert facade._pkg() is importlib.import_module(PACKAGE)
    tree = ast.parse(pathlib.Path(facade.__file__).read_text())
    imported = [
        a.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for a in node.names
    ] + [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not [n for n in imported if "cake_gdn_det" in n], imported


def test_facade_context_graph_replay_hint(monkeypatch) -> None:
    """``cake_gdn_nondet_train_kcp_build_context(..., graph_replay=...)``: keyword-only, default ``False``, a non-bool
    is a ``TypeError``; the package receives the hint."""
    facade = importlib.import_module(FACADE)
    p = inspect.signature(facade.cake_gdn_nondet_train_kcp_build_context).parameters[
        "graph_replay"
    ]
    assert p.kind is p.KEYWORD_ONLY and p.default is False
    seen = _route_recorders(monkeypatch)
    for hint in (False, True):
        seen.clear()
        assert (
            facade.cake_gdn_nondet_train_kcp_build_context(
                object(), None, 0, "cpu", graph_replay=hint
            )
            == "nondet:cake_gdn_nondet_train_kcp_build_context"
        )
        assert seen[-1][1]["graph_replay"] is hint
    for bad in (1, None, "True"):
        with pytest.raises(
            TypeError,
            match="^cake_gdn_nondet_train_kcp_build_context: graph_replay must be a bool",
        ):
            facade.cake_gdn_nondet_train_kcp_build_context(
                object(), None, 0, "cpu", graph_replay=bad
            )


def test_facade_forwards_the_l2n_option_to_the_package(monkeypatch) -> None:
    """#7817 / #8063: ``use_qk_l2norm_in_kernel`` reaches the package unchanged (keyword, mapping and this package's
    options instance) on every compute entry and byte query."""
    facade = importlib.import_module(FACADE)
    seen = _route_recorders(monkeypatch)
    calls = _facade_calls(facade)
    for entry in (
        "cake_gdn_nondet_kcp",
        "cake_gdn_nondet_train_kcp_forward",
        "cake_gdn_nondet_train_kcp_backward",
        "cake_gdn_nondet_train_kcp_workspace_bytes",
        "cake_gdn_nondet_train_kcp_saved_bytes",
    ):
        for flag in (False, True):
            seen.clear()
            assert calls[entry](use_qk_l2norm_in_kernel=flag) == f"nondet:{entry}"
            assert seen[-1][1]["use_qk_l2norm_in_kernel"] is flag, entry
    t = torch.zeros(1)
    mapping = {"use_qk_l2norm_in_kernel": True}
    facade.cake_gdn_nondet_train_kcp_forward(t, t, t, t, t, object(), options=mapping)
    assert seen[-1][0] == "nondet:cake_gdn_nondet_train_kcp_forward"
    assert seen[-1][1]["options"] == mapping
    nd_o = CakeGDNNondetKcpOptions(use_qk_l2norm_in_kernel=True)
    assert nd_o.use_qk_l2norm_in_kernel is True
    facade.cake_gdn_nondet_train_kcp_backward(t, t, t, t, t, object(), t, options=nd_o)
    assert seen[-1][0] == "nondet:cake_gdn_nondet_train_kcp_backward"
    assert seen[-1][1]["options"] is nd_o


# The one-rank CPU pack of the in-kernel L2 norm suite (``test_cake_gdn_nondet_train_kcp_l2n.py``: its ``CU``).
L2N_CU = [0, 100, 740, 817, 1000]
L2N_TENSORS = ("out", "dq", "dk", "dv", "dg", "dbeta")


def _l2n_case():
    """The norm suite's one-rank CPU context of :data:`L2N_CU` (built as that suite builds it) and seeded operands of
    heads ``(2, 6)``."""
    plan = _mod("cake_kcp_plan")
    part = plan.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(L2N_CU, dtype=torch.int64), 1, "contiguous"
    )
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(
        part, None, 0, torch.device("cpu")
    )
    return ctx, _mode_ops(cu=L2N_CU)


def test_facade_runs_the_norm() -> None:
    """The facade hands ``use_qk_l2norm_in_kernel`` (keyword, mapping or this package's options instance) to this
    package, which runs the norm: bitwise the package's own call; ``False`` is the norm-off route; the autograd node
    runs the norm in its forward and its backward.  The compute entries take the option through their keyword
    arguments, with no rstd keyword.  An options instance of the deterministic package is a ``TypeError``."""
    facade = importlib.import_module(FACADE)
    kcp = _mod("cake_kcp")
    det_options = _det("cake_options")
    for fn in (
        facade.cake_gdn_nondet_kcp,
        facade.cake_gdn_nondet_train_kcp_forward,
        facade.cake_gdn_nondet_train_kcp_backward,
    ):
        params = inspect.signature(fn).parameters
        assert (
            params["use_qk_l2norm_in_kernel"].kind is inspect.Parameter.KEYWORD_ONLY
        ), fn
        assert params["use_qk_l2norm_in_kernel"].default is False, fn
        assert not any("rstd" in n for n in params), fn
    ctx, ops = _l2n_case()
    args = [ops[n] for n in ("q", "k", "v", "g", "beta")]
    on = {"backend": "fake", "use_qk_l2norm_in_kernel": True}
    out, saved = kcp.cake_gdn_nondet_train_kcp_forward(*args, ctx, **on)
    dq, dk, dv, dbeta, dg = kcp.cake_gdn_nondet_train_kcp_backward(
        *args, ctx, ops["dout"], saved=saved, **on
    )
    want = {"out": out, "dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta}
    for how in (
        {"use_qk_l2norm_in_kernel": True},
        {"options": {"backend": "fake", "use_qk_l2norm_in_kernel": True}},
        {"options": kcp.make_options(backend="fake", use_qk_l2norm_in_kernel=True)},
    ):
        kw = {} if "options" in how else {"backend": "fake"}
        out, saved = facade.cake_gdn_nondet_train_kcp_forward(*args, ctx, **kw, **how)
        assert kcp.LAST_CALL["fwd"]["qk_l2norm"] is True and saved.qk_l2norm, how
        dq, dk, dv, dbeta, dg = facade.cake_gdn_nondet_train_kcp_backward(
            *args, ctx, ops["dout"], saved=saved, **kw, **how
        )
        got = {"out": out, "dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta}
        for n in L2N_TENSORS:
            assert torch.equal(got[n], want[n]), (how, n)
    off = facade.cake_gdn_nondet_train_kcp_forward(
        *args, ctx, backend="fake", use_qk_l2norm_in_kernel=False
    )[1]
    assert off.qk_l2norm is False and off.qk_rstd is None
    assert kcp.LAST_CALL["fwd"]["qk_l2norm"] is False
    leaves = [x.detach().clone().requires_grad_(True) for x in args]
    facade.cake_gdn_nondet_kcp(*leaves, ctx, **on).backward(ops["dout"])
    assert (
        kcp.LAST_CALL["fwd"]["qk_l2norm"] is True
        and kcp.LAST_CALL["bwd"]["qk_l2norm"] is True
    )
    det_o = det_options.CakeGDNDetKcpOptions(
        backend="fake", use_qk_l2norm_in_kernel=True
    )
    with pytest.raises(TypeError, match="^options must be a CakeGDNNondetKcpOptions"):
        facade.cake_gdn_nondet_train_kcp_forward(*args, ctx, options=det_o)


def test_facade_byte_queries_take_the_option() -> None:
    """The facade's byte queries forward ``use_qk_l2norm_in_kernel`` to this package; its norm terms equal those of the
    deterministic facade's own queries (``flashinfer.cake_gdn_det_train_kcp``; same plan inputs, same formulas) at
    ``save_mode="light"``, and the saved set grows by the FP32 ``r`` of q and k (``8 T_r H_qk`` bytes).  With the norm
    the default save mode is ``"light_akk"``: the default saved set also holds the ``Akk`` tape (``128 R HV`` bytes)."""
    facade = importlib.import_module(FACADE)
    det_facade = importlib.import_module(DET_FACADE)
    gbytes = _mod("cake_bytes")
    ctx, _ops = _l2n_case()
    H_qk, HV = 2, 6
    for query in ("saved_bytes", "workspace_bytes"):
        q = getattr(facade, f"cake_gdn_nondet_train_kcp_{query}")
        det_q = getattr(det_facade, f"cake_gdn_det_train_kcp_{query}")
        on = q(ctx, H_qk=H_qk, HV=HV, save_mode="light", use_qk_l2norm_in_kernel=True)
        off = q(ctx, H_qk=H_qk, HV=HV, use_qk_l2norm_in_kernel=False)
        det_on = det_q(ctx, H_qk=H_qk, HV=HV, use_qk_l2norm_in_kernel=True)
        det_off = det_q(ctx, H_qk=H_qk, HV=HV, use_qk_l2norm_in_kernel=False)
        assert on > off and det_on > det_off, query
        assert on - off == det_on - det_off, query
        assert on == getattr(gbytes, f"cake_gdn_nondet_train_kcp_{query}")(
            ctx, H_qk=H_qk, HV=HV, save_mode="light", use_qk_l2norm_in_kernel=True
        ), query
        assert q(ctx, H_qk=H_qk, HV=HV, use_qk_l2norm_in_kernel=True) == q(
            ctx, H_qk=H_qk, HV=HV, save_mode="light_akk", use_qk_l2norm_in_kernel=True
        ), query
    light = facade.cake_gdn_nondet_train_kcp_saved_bytes(
        ctx, H_qk=H_qk, HV=HV, save_mode="light", use_qk_l2norm_in_kernel=True
    )
    assert (
        light - facade.cake_gdn_nondet_train_kcp_saved_bytes(ctx, H_qk=H_qk, HV=HV)
        == 8 * int(ctx.T_r) * H_qk
    )
    assert (
        facade.cake_gdn_nondet_train_kcp_saved_bytes(
            ctx, H_qk=H_qk, HV=HV, use_qk_l2norm_in_kernel=True
        )
        - light
        == 128 * int(ctx.R) * HV
    )


def test_facade_refuses_an_options_instance_of_another_package(monkeypatch) -> None:
    """An options instance of another package (the deterministic one's, or any other object) is a ``TypeError``
    before the package is called; this package's own instance, a mapping and ``None`` pass through unchanged."""
    facade = importlib.import_module(FACADE)
    det_opt = _det("cake_options")
    seen = _route_recorders(monkeypatch)
    calls = _facade_calls(facade)
    det_o = det_opt.CakeGDNDetKcpOptions(save_mode="full")
    for entry in ENTRIES:
        for bad in (det_o, object(), ["save_mode"]):
            seen.clear()
            with pytest.raises(
                TypeError, match="^options must be a CakeGDNNondetKcpOptions"
            ):
                calls[entry](options=bad)
            assert seen == [], (entry, bad)
        own_o = CakeGDNNondetKcpOptions(save_mode="full")
        for passthrough in (own_o, CakeGDNNondetKcpOptions(graph="replay"), None):
            seen.clear()
            calls[entry](options=passthrough)
            assert seen[-1][1]["options"] is passthrough, entry
        mapping = {"save_mode": "full"}
        calls[entry](options=mapping)
        assert seen[-1][1]["options"] == mapping, entry


@pytest.mark.parametrize("warn_only", [False, True])
def test_facade_compute_runs_the_package_under_torch_deterministic_mode(
    torch_deterministic, warn_only
) -> None:
    """Without the recorders, on the fake backend: under ``torch.use_deterministic_algorithms(True)`` (strict and
    ``warn_only``) the facade's forward, backward and autograd node run this package (its saved set, its node tag)
    without an alert or a warning.  The explicit forward's output and backward's gradients, and the autograd node's
    output and leaf gradients (``q``, ``k``, ``v``, ``g``, ``beta``), equal the package's own forward and explicit
    backward called with torch's deterministic algorithms off (every launch of the package is fixed-order)."""
    facade = importlib.import_module(FACADE)
    kcp = _mod("cake_kcp")
    ctx, ops = _mode_case()
    args = [ops[n] for n in ("q", "k", "v", "g", "beta")]
    nd = dict(scale=facade.DEFAULT_SCALE, backend="fake")
    want_out, want_saved = kcp.cake_gdn_nondet_train_kcp_forward(*args, ctx, **nd)
    want_grads = kcp.cake_gdn_nondet_train_kcp_backward(
        *args, ctx, ops["dout"], saved=want_saved, **nd
    )
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, saved = facade.cake_gdn_nondet_train_kcp_forward(
            *args, ctx, backend="fake"
        )
        assert isinstance(saved, kcp.CakeGDNNondetKcpSaved)
        grads = facade.cake_gdn_nondet_train_kcp_backward(
            *args, ctx, ops["dout"], saved=saved, backend="fake"
        )
        leaves = [t.detach().clone().requires_grad_(True) for t in args]
        y = facade.cake_gdn_nondet_kcp(*leaves, ctx, backend="fake")
        assert y.grad_fn is not None and y.grad_fn.kcp_mode == kcp._MODE
        y.backward(ops["dout"])
    assert torch.equal(out, want_out)
    for got, want in zip(grads, want_grads, strict=True):
        assert torch.equal(got, want)
    # the explicit backward returns (dq, dk, dv, dbeta, dg); the leaves are (q, k, v, g, beta)
    dq, dk, dv, dbeta, dg = want_grads
    assert torch.equal(y.detach(), want_out)
    for name, leaf, want in zip(
        ("q", "k", "v", "g", "beta"), leaves, (dq, dk, dv, dg, dbeta), strict=True
    ):
        assert leaf.grad is not None and torch.equal(leaf.grad, want), name


def test_facade_and_the_deterministic_facade_agree_on_this_small_plan() -> None:
    """Without the recorders: the facade builds this package's partition / context (``CakeGDNNondetKcpPartition`` /
    ``CakeGDNNondetKcpContext``), the deterministic facade its own (``CakeGDNDetKcpPartition`` /
    ``CakeGDNDetKcpContext``).  The partition algorithm is the same, so the partitions are equal; the head plans are
    each package's own (their ``P`` rules and table layouts differ in general), and on this small two-rank row they
    give the same ``P`` and segment count.  On it both describe the same plan (the comm byte query, the saved byte
    query at the default options, under the replay boundary and at ``P = 2`` with either ``save_mode`` and
    ``save_transfer``) and the same workspace terms for every keyword of the workspace query (``graph``,
    ``save_for_backward``, ``fused_walks``, ``P``) except the b1 pools, whose layouts differ: the twin's b1 send
    carries the re-sent maps of the fragments crossing in and out (``n_io``), this package's maps are paired with the
    crossing-in slots.  Under ``graph="replay"`` only this package's query takes ``reuse_recompute_tapes=True`` (the
    arena hand-off) and ``"auto"``; the twin's refuses both, as its options do.  The default (``None``) gives the
    replay form's terms on both packages."""
    facade = importlib.import_module(FACADE)
    det_facade = importlib.import_module(DET_FACADE)
    det_pkg = _det()
    pkg = _mod()
    cu = [0, 100, 740, 817, 1817, 1946]
    det_part = det_facade.cake_gdn_det_train_kcp_partition(cu, 2, "zigzag")
    nondet_part = facade.cake_gdn_nondet_train_kcp_partition(cu, 2, "zigzag")
    assert isinstance(det_part, det_pkg.CakeGDNDetKcpPartition)
    assert isinstance(nondet_part, pkg.CakeGDNNondetKcpPartition)
    assert det_part.to_dict() == nondet_part.to_dict()
    det_queries = {
        "cake_gdn_nondet_train_kcp_comm_bytes": det_facade.cake_gdn_det_train_kcp_comm_bytes,
        "cake_gdn_nondet_train_kcp_saved_bytes": det_facade.cake_gdn_det_train_kcp_saved_bytes,
    }
    for rank in range(2):
        det_ctx = det_facade.cake_gdn_det_train_kcp_build_context(
            det_part, None, rank, "cpu", sm_count=148
        )
        ctx = facade.cake_gdn_nondet_train_kcp_build_context(
            nondet_part, None, rank, "cpu", sm_count=148
        )
        assert isinstance(det_ctx, det_pkg.CakeGDNDetKcpContext)
        assert isinstance(ctx, pkg.CakeGDNNondetKcpContext)
        det_hp, hp = det_ctx.heads(6), ctx.heads(6)
        assert (det_hp.P, det_hp.NSEG) == (hp.P, hp.NSEG), rank
        for query, det_query in det_queries.items():
            det_value = det_query(det_ctx, H_qk=2, HV=6)
            value = getattr(facade, query)(ctx, H_qk=2, HV=6)
            assert det_value == value, (rank, query)
        # a replay boundary saves nothing in either package (save_mode="full" included)
        replay_full = dict(H_qk=2, HV=6, save_mode="full", boundary_mode="replay")
        assert det_facade.cake_gdn_det_train_kcp_saved_bytes(
            det_ctx, **replay_full
        ) == facade.cake_gdn_nondet_train_kcp_saved_bytes(ctx, **replay_full), rank
        # the M_seg rule at P > 1 is shared (P n_frag s with save_transfer or save_mode="full")
        assert ctx.n_frag > 0
        for kw in (
            {},
            {"save_transfer": False},
            {"save_mode": "full"},
            {"save_mode": "full", "save_transfer": False},
        ):
            det_value = det_facade.cake_gdn_det_train_kcp_saved_bytes(
                det_ctx, H_qk=2, HV=6, P=2, **kw
            )
            value = facade.cake_gdn_nondet_train_kcp_saved_bytes(
                ctx, H_qk=2, HV=6, P=2, **kw
            )
            assert det_value == value, (rank, kw)
        # the workspace queries take the same keywords on both packages and count the same terms, except the b1
        # pools: the twin's dense b1 send carries the re-sent maps of the n_io fragments crossing in and out (its d_f
        # plane + n_io map slots), this package's the maps paired with the n_in crossing-in slots
        state = 6 * _mod("cake_bytes").STATE_TILE_BYTES
        det_gb = _det("cake_bytes")
        b1_keys = ("bwd.b1_send", "bwd.b1_recv")
        for kw in (
            {},
            {"P": 2},
            {"P": 2, "save_transfer": False},
            {"graph": "replay"},
            {"save_for_backward": False},
            {"fused_walks": 0, "P": 2, "backward": False},
        ):
            det_lay = det_gb.cake_gdn_det_train_kcp_workspace_layout(
                det_ctx, H_qk=2, HV=6, **kw
            )
            lay = _mod("cake_bytes").cake_gdn_nondet_train_kcp_workspace_layout(
                ctx, H_qk=2, HV=6, **kw
            )
            assert set(det_lay) == set(lay), (rank, kw)
            fwd_keys = [
                k
                for k in lay
                if k.startswith("fwd.") or k in ("forward", "forward_split")
            ]
            assert {k: det_lay[k] for k in fwd_keys} == {k: lay[k] for k in fwd_keys}, (
                rank,
                kw,
            )
            assert det_facade.cake_gdn_det_train_kcp_workspace_bytes(
                det_ctx, H_qk=2, HV=6, **kw
            ) == det_gb.cake_gdn_det_train_kcp_workspace_bytes(
                det_ctx, H_qk=2, HV=6, **kw
            ), (
                rank,
                kw,
            )
            assert facade.cake_gdn_nondet_train_kcp_workspace_bytes(
                ctx, H_qk=2, HV=6, **kw
            ) == pkg.cake_gdn_nondet_train_kcp_workspace_bytes(
                ctx, H_qk=2, HV=6, **kw
            ), (rank, kw)
            if kw.get("backward", True):
                hp = det_ctx.heads(6)
                n_io, N_io = hp.n_io_by_rank[rank], sum(hp.n_io_by_rank)
                assert (
                    lay["bwd.b1_send"] - det_lay["bwd.b1_send"]
                    == (det_ctx.n_in - n_io) * state
                ), (rank, kw)
                # an owner receives the b1 of every crossing-in fragment of the group (and the maps of the N_io
                # fragments on the twin's side, of the N_in crossing-in slots on this package's)
                s_owner = hp.Hg_r * _mod("cake_bytes").STATE_TILE_BYTES
                assert (
                    lay["bwd.b1_recv"] - det_lay["bwd.b1_recv"]
                    == (det_ctx.N_in - N_io) * s_owner
                ), (rank, kw)
                # the .phase. keys are moment sums, some of which hold the b1 pools: they carry the b1 deltas above,
                # so they are left out of this term-by-term comparison
                rest = {
                    k: v
                    for k, v in lay.items()
                    if k.startswith("bwd.") and k not in b1_keys and ".phase." not in k
                }
                assert rest == {k: det_lay[k] for k in rest}, (rank, kw)
        # reuse_recompute_tapes=True (the arena hand-off) and "auto" under graph="replay": this package's query takes
        # them, the twin's refuses both, as its options do
        for reuse in (True, "auto"):
            graph_kw = dict(H_qk=2, HV=6, graph="replay", reuse_recompute_tapes=reuse)
            assert facade.cake_gdn_nondet_train_kcp_workspace_bytes(
                ctx, **graph_kw
            ) == pkg.cake_gdn_nondet_train_kcp_workspace_bytes(ctx, **graph_kw), (
                rank,
                reuse,
            )
            with pytest.raises(ValueError, match="needs graph='off'"):
                det_facade.cake_gdn_det_train_kcp_workspace_bytes(det_ctx, **graph_kw)
        # save_for_backward (a forward without a backward): each package's own query
        no_saved_set = dict(H_qk=2, HV=6, save_for_backward=False, backward=False)
        assert det_facade.cake_gdn_det_train_kcp_workspace_bytes(
            det_ctx, **no_saved_set
        ) == det_gb.cake_gdn_det_train_kcp_workspace_bytes(det_ctx, **no_saved_set), (
            rank
        )
        assert facade.cake_gdn_nondet_train_kcp_workspace_bytes(
            ctx, **no_saved_set
        ) == pkg.cake_gdn_nondet_train_kcp_workspace_bytes(ctx, **no_saved_set), rank
        # fused_walks=0 passes on: at P = 2 the forward has no carry slots for the crossing fragments
        assert max(ctx.n_in, ctx.n_out) > 0
        for c, query in (
            (ctx, facade.cake_gdn_nondet_train_kcp_workspace_bytes),
            (det_ctx, det_facade.cake_gdn_det_train_kcp_workspace_bytes),
        ):
            split_walks = query(c, H_qk=2, HV=6, fused_walks=0, P=2, backward=False)
            fused = query(c, H_qk=2, HV=6, P=2, backward=False)
            assert fused - split_walks == max(c.n_in, c.n_out) * state, (rank, c)


def test_the_context_holds_no_mode() -> None:
    """No field of ``CakeGDNNondetKcpContext`` / ``CakeGDNNondetKcpPartition`` / ``CakeGDNNondetKcpFragment`` /
    ``CakeGDNNondetKcpOptions`` names a mode, nor any attribute of a built context, its partition record or its
    description: the mode is in the package name only."""
    plan = _mod("cake_kcp_plan")
    fields = list(plan.CakeGDNNondetKcpContext._FIELDS)
    fields += [f.name for f in dataclasses.fields(plan.CakeGDNNondetKcpPartition)]
    fields += [f.name for f in dataclasses.fields(plan.CakeGDNNondetKcpFragment)]
    fields += [f.name for f in dataclasses.fields(CakeGDNNondetKcpOptions)]
    fields += list(opt.OPTION_NAMES)
    ctx, _ops = _mode_case()
    fields += [n for n in dir(ctx) if not n.startswith("__")]
    fields += list(ctx.partition.to_dict())
    fields += list(ctx.describe(6))
    bad = sorted({f for f in fields if "determin" in f.lower() or f.lower() == "mode"})
    assert not bad, bad
    assert (
        "deterministic"
        not in plan.cake_gdn_nondet_train_kcp_build_context.__code__.co_varnames
    )
    assert (
        "deterministic"
        not in plan.cake_gdn_nondet_train_kcp_partition.__code__.co_varnames
    )


# ----------------------------------------------------------------------------------------------------------------
# standalone (flashinfer #6553): the package imports itself, torch, torch.distributed, tvm_ffi and the standard
# library only, and names no library of its deterministic twin


def test_the_package_imports_no_other_flashinfer_package() -> None:
    """No import of ``flashinfer`` outside the package (relative imports one level deep only) and none of the twin;
    the one exception is the pinned loader's JIT route (``cake_build.py``: ``flashinfer.jit.*`` inside a function).
    Every other absolute import is the standard library, ``torch``, ``torch.distributed`` or ``tvm_ffi``."""
    stdlib = set(sys.stdlib_module_names)
    for path in _package_sources():
        tree = ast.parse(path.read_text())
        top_level = {id(node) for node in tree.body}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.level == 1, (path.name, node.lineno, node.level)
                continue
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert "cake_gdn_det" not in name, (path.name, node.lineno, name)
                head = name.split(".")[0]
                if head == "flashinfer":
                    assert (
                        path.name == "cake_build.py"
                        and name.startswith("flashinfer.jit.")
                        and id(node) not in top_level
                    ), (path.name, node.lineno, name)
                    continue
                assert head in stdlib or name in (
                    "torch",
                    "torch.distributed",
                    "tvm_ffi",
                ), (
                    path.name,
                    node.lineno,
                    name,
                )


def test_the_registry_names_no_library_of_the_deterministic_twin() -> None:
    """Every record of this package's registry (shipped and test-only) names a ``cake_gdn_train_`` library and a
    kernel symbol of its own; none equals a name of the twin's registry or rendered manifests."""
    jit = _mod("cake_jit")
    det_jit = _det("cake_jit")
    assert jit.PACKAGE == PACKAGE and jit.LIBRARY_PREFIX == "cake_gdn_train"
    records = {**jit.MODULES, **jit.TEST_MODULES}
    det_records = {**det_jit.MODULES, **det_jit.TEST_MODULES}
    ours = {rec["library"] for rec in records.values()}
    theirs = {rec["library"] for rec in det_records.values()}
    assert ours and all(n.startswith("cake_gdn_train_") for n in ours), sorted(ours)
    assert not ours & theirs
    assert not {rec["kernel_symbol"] for rec in records.values()} & {
        rec["kernel_symbol"] for rec in det_records.values()
    }
    root = pathlib.Path(_mod().__file__).resolve().parent
    det_root = pathlib.Path(det_jit.__file__).resolve().parent
    for manifest in ("build.json", "build.test.json"):
        path = root / manifest
        if not path.is_file():
            continue  # rendered by the export; checked once present
        document = json.loads(path.read_text())
        assert document["package"] == PACKAGE, manifest
        names = {entry["name"] for entry in document["libraries"]}
        assert all(n.startswith("cake_gdn_train_") for n in names), manifest
        det_path = det_root / manifest
        if det_path.is_file():
            det_names = {
                e["name"] for e in json.loads(det_path.read_text())["libraries"]
            }
            assert not names & det_names, manifest
        for entry in document["libraries"]:
            for source in entry["sources"]:
                assert "cake_gdn_det" not in source, (manifest, source)


# ----------------------------------------------------------------------------------------------------------------
# package hygiene (no KDA package import; no os.environ; the host module's import allow-list)


def _package_sources() -> list[pathlib.Path]:
    root = pathlib.Path(_mod().__file__).resolve().parent
    files = sorted(root.glob("*.py"))
    assert files
    return files


def _facade_source() -> pathlib.Path:
    return pathlib.Path(importlib.import_module(FACADE).__file__).resolve()


def test_package_never_imports_a_kda_package() -> None:
    """Neither the package nor the facade imports a KDA package: no imported module path or name contains ``cake_kda``
    (the deterministic and the non-deterministic KDA packages)."""
    for path in _package_sources() + [_facade_source()]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert "cake_kda" not in a.name, (path.name, a.name)
            elif isinstance(node, ast.ImportFrom):
                assert "cake_kda" not in (node.module or ""), (
                    path.name,
                    node.module,
                )
                for a in node.names:
                    assert "cake_kda" not in a.name, (path.name, a.name)


def test_package_reads_no_environment_variable() -> None:
    """D6 / G10: no ``os.environ`` / ``os.getenv`` / ``os.putenv`` anywhere in the package or the facade; the pinned
    export loader is the only module allowed to import ``os`` (for paths), and it reads the environment nowhere."""
    for path in _package_sources() + [_facade_source()]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in (
                "environ",
                "getenv",
                "putenv",
                "unsetenv",
            ):
                raise AssertionError(
                    f"{path.name}: reads the environment ({node.attr})"
                )
            if isinstance(node, ast.ImportFrom) and node.module == "os":
                for a in node.names:
                    assert a.name not in ("environ", "getenv", "putenv"), (
                        path.name,
                        a.name,
                    )
        if path.name in (
            "cake_kcp.py",
            "cake_options.py",
            "cake_kcp_plan.py",
            "cake_tables.py",
            "cake_bytes.py",
            "__init__.py",
        ):
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(a.name != "os" for a in node.names), (
                        f"{path.name}: imports os"
                    )


def test_host_modules_import_only_torch_and_their_package() -> None:
    """The host module and the options module import the standard library's typing helpers (``inspect`` builds the
    entries' declared signatures), torch, torch.distributed and their own package only (relative imports); the
    registry / emulation are imported lazily inside functions."""
    allowed = {
        "__future__",
        "dataclasses",
        "inspect",
        "math",
        "types",
        "typing",
        "collections",
        "collections.abc",
        "torch",
        "torch.distributed",
    }
    for name in ("cake_kcp", "cake_options"):
        src = pathlib.Path(_mod(name).__file__).read_text()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name in allowed, (name, a.name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module in allowed, (name, node.module)


# ----------------------------------------------------------------------------------------------------------------
# frozen vocabulary of the dispatch table


def test_program_vocabulary_is_the_design_list() -> None:
    kcp = _mod("cake_kcp")
    assert kcp.PROGRAMS == (
        "fwd_prep",
        "kcp_fwdh_offset",
        "kcp_fwdh_tape",
        "kcp_fwdh_out",
        "kcp_fwdh_out_tape",
        "kcp_transfer_walk",
        "kcp_affine_merge",
        "fwd_o",
        "bwd_prep",
        "kcp_dhu_offset",
        "kcp_dhu_tape",
        "kcp_dhu_cut_start",
        "bwd_dqkg",
        "gate_epilogue",
        "fwd_prep_l2n",
        "bwd_dqkg_l2n",
        "qk_apply",
        "fwd_prep_replay_l2n",
    )
    # the in-kernel Q / K L2 norm (use_qk_l2norm_in_kernel): #1 / #8 siblings and the backward's qk_apply
    assert kcp.L2N_PROGRAMS == ("fwd_prep_l2n", "bwd_dqkg_l2n", "qk_apply")
    assert set(kcp.L2N_PROGRAMS) <= set(kcp.LOCAL_PROGRAMS)
    assert kcp.L2N_SIBLING_OF == {
        "fwd_prep": "fwd_prep_l2n",
        "bwd_dqkg": "bwd_dqkg_l2n",
    }
    assert kcp.QK_TAPES == ("qn", "kn") and kcp.QK_RSTD == "qk_rstd"
    assert set(kcp.PROGRAM_ARGS) == set(kcp.PROGRAMS)
    assert (
        kcp.TART_MODULE_OF["kcp_affine_merge"] == "kda_chunk_kcp_affine_merge_fp32_v1"
    )  # reused verbatim
    # the output walks are this package's own programs: det-neutral modules that no deterministic twin ships
    assert kcp.TART_MODULE_OF["kcp_fwdh_out"] == "gdn_chunk_fwdh_out_v2_kcp"
    assert kcp.TART_MODULE_OF["kcp_fwdh_out_tape"] == "gdn_chunk_fwdh_out_tape_v2_kcp"
    # save_mode="light_akk": the solve-free light replay of #1_l2n (this package's own module; a local program,
    # launched only in place of fwd_prep_l2n when the saved set holds the Akk tape)
    assert (
        kcp.TART_MODULE_OF["fwd_prep_replay_l2n"]
        == "gdn_chunk_train_fwd_prep_replay_l2n_v1"
    )
    assert kcp.AKK_REPLAY_OF == {"fwd_prep_l2n": "fwd_prep_replay_l2n"}
    assert kcp.AKK_REPLAY_PROGRAMS == ("fwd_prep_replay_l2n",)
    assert set(kcp.AKK_REPLAY_PROGRAMS) <= set(kcp.LOCAL_PROGRAMS)
    assert kcp.PROGRAM_ARGS["fwd_prep_replay_l2n"] == kcp.PROGRAM_ARGS["fwd_prep_l2n"]
    own = (
        "kcp_affine_merge",
        "kcp_fwdh_out",
        "kcp_fwdh_out_tape",
        "fwd_prep_replay_l2n",
        "kcp_dhu_offset",  # the cut walk (gdn_chunk_dhu_cut_v1_kcp)
        "kcp_dhu_cut_start",
    )
    for name, module in kcp.TART_MODULE_OF.items():
        if name not in own:
            # the other programs' kernel modules are, so far, the deterministic family's (provenance stems; a lever
            # that replaces a program updates this line)
            assert module.startswith("gdn_chunk_det_"), (name, module)
        if name != "kcp_affine_merge":
            # the context-parallel-only programs carry the ``_kcp`` suffix
            assert module.endswith("_kcp") == name.startswith("kcp_"), (name, module)
    assert kcp.FWD_TAPES == (
        "w",
        "u",
        "aqk",
        "akk",
        "g_cum",
        "rowscale",
        "qscale",
        "gamma",
        "h_tape",
        "v_new",
    )
    assert kcp.BWD_TAPES == (
        "do_g",
        "dv_intra",
        "dh_tape",
        "dv_new",
        "dg_part",
        "gamma_dot",
    )
    for prog in (
        "kcp_fwdh_offset",
        "kcp_fwdh_out",
        "kcp_fwdh_out_tape",
        "kcp_transfer_walk",
        "kcp_dhu_offset",
    ):
        args = kcp.PROGRAM_ARGS[prog]
        assert (
            "num_heads" in args
            and "num_qk_heads" in args
            and "walk_seg" in args
            and "tile_stride" in args
        )
        assert "write_tapes" not in args  # §1.4: no such argument on #2 / #7 (A9)
        assert "program" not in args  # static build knob (record name), never a value
        assert {"seg_chunk_start", "chunk_bos", "chunk_len"} <= set(args)
        assert (
            "seq_chunk_start" not in args
        )  # the walks' segment table is the logical seg_chunk_start
    # precision_mode / transfer_walk are static build knobs (record selection), never kernel arguments
    assert "precision_mode" not in kcp.PROGRAM_ARGS["kcp_transfer_walk"]
    assert {"init_mode", "use_u", "write_transposed"} <= set(
        kcp.PROGRAM_ARGS["kcp_transfer_walk"]
    )
    assert "group_reduce" in kcp.PROGRAM_ARGS["bwd_dqkg"]
    assert set(kcp.HOST_ONLY_ARGS) <= set(kcp.PROGRAM_ARGS["bwd_dqkg"])
    assert "o" in kcp.PROGRAM_ARGS["fwd_o"] and "out" not in kcp.PROGRAM_ARGS["fwd_o"]
    # the output walk: the tape walk's inputs and #5's operands, writes o and neither tape nor final-state pool
    fused = kcp.PROGRAM_ARGS["kcp_fwdh_out"]
    assert {"q", "aqk", "qscale", "o", "h0", "h0_index"} <= set(fused)
    assert not {"h_tape", "v_new", "out", "out_index", "ticket"} & set(fused)
    assert set(fused) - {"q", "aqk", "qscale", "o"} <= set(
        kcp.PROGRAM_ARGS["kcp_fwdh_tape"]
    )
    # the tape-and-output walk: the output walk's arguments and the tape walk's h / v_new tapes, no final-state pool
    fused_tape = kcp.PROGRAM_ARGS["kcp_fwdh_out_tape"]
    assert set(fused_tape) == set(fused) | {"h_tape", "v_new"}
    assert not {"out", "out_index", "ticket"} & set(fused_tape)
    assert set(fused_tape) - {"q", "aqk", "qscale", "o"} <= set(
        kcp.PROGRAM_ARGS["kcp_fwdh_tape"]
    )
    assert (
        "do" in kcp.PROGRAM_ARGS["bwd_prep"]
        and "dout" not in kcp.PROGRAM_ARGS["bwd_prep"]
    )
    assert "dbeta" not in kcp.PROGRAM_ARGS["gate_epilogue"]
    assert kcp.LIGHT_TAPES == ()  # the light backward re-runs #1: no saved tape
    assert kcp.saved_tapes("full") == kcp.FWD_TAPES
    assert kcp.saved_tapes("light") == kcp.LIGHT_TAPES
    assert kcp.test_record("kcp_transfer_walk", 0, "tf32_hilo") is None
    assert kcp.test_record("kcp_fwdh_offset", 2, "bf16x3") is None
    assert (
        kcp.test_record("kcp_transfer_walk", 1, "tf32_hilo")
        == "kcp_transfer_walk_three_mma"
    )
    assert (
        kcp.test_record("kcp_transfer_walk", 2, "tf32_hilo")
        == "kcp_transfer_walk_plain_tf32"
    )
    assert (
        "kcp_transfer_walk",
        "bf16x3",
    ) not in kcp.TEST_RECORDS  # no bf16x3 module
    with pytest.raises(ValueError):
        kcp.test_record("kcp_transfer_walk", 0, "bf16x3")


def test_program_args_are_the_registry_logical_names() -> None:
    """Every value the host passes is a logical name of the registry, and every logical argument of every registry
    record is provided by the host."""
    kcp = _mod("cake_kcp")
    jit = _mod("cake_jit")
    vocab = set(jit.LOGICAL_TENSORS) | set(jit.LOGICAL_SCALARS)
    host_only = set(kcp.HOST_ONLY_ARGS) | set(kcp.PROVISIONAL_ARGS)
    for prog, names in kcp.PROGRAM_ARGS.items():
        extra = set(names) - host_only - vocab
        assert not extra, (prog, sorted(extra))
    for rec_name, rec in {**jit.MODULES, **jit.TEST_MODULES}.items():
        needed = set(rec["logical"].values())
        missing = needed - set(kcp.PROGRAM_ARGS[rec["program"]])
        assert not missing, (rec_name, sorted(missing))
    # record resolution through the registry: shipped records, the merge variants by ncols, the test twins (the
    # in-kernel L2 norm programs once their records ship; refused with a clear message before)
    for prog in kcp.PROGRAMS:
        if prog == "kcp_affine_merge":
            continue
        if prog in kcp.L2N_PROGRAMS and not jit.records_of_program(prog):
            with pytest.raises(ValueError, match="no shipped record"):
                jit.select_record(prog)
            continue
        assert jit.select_record(prog) == prog
    assert jit.select_record("kcp_affine_merge", ncols=128) == "kcp_affine_merge_c8"
    assert jit.select_record("kcp_affine_merge", ncols=256) == "kcp_affine_merge_c16"
    for mode, record in kcp.TEST_RECORDS.items():
        assert jit.select_record("kcp_transfer_walk", precision_mode=mode[1]) == record
    with pytest.raises(ValueError):
        jit.select_record("kcp_transfer_walk", transfer_walk="bf16x3")


def test_fwdh_form_rule_follows_the_chain_clause() -> None:
    """#2 runs the vh = 64 form when ``n_walk * HV <= sm_count`` or when the launch's two-CTA grid finishes within
    its longest chain (``walk_vh64_fits``: LPT of ``2 * HV`` CTAs per segment over ``2 * sm_count`` slots), else
    vh = 128; the test knob ``fwdh_vh`` forces a form (the two forms are within the same accuracy budget; the
    fake-backend suite checks it)."""
    kcp = _mod("cake_kcp")
    assert kcp.FWDH_PROGRAMS == (
        "kcp_fwdh_offset",
        "kcp_fwdh_tape",
        "kcp_fwdh_out",
        "kcp_fwdh_out_tape",
    )
    assert kcp.fwdh_vh(3, 48, 152) == 64  # 144 head-CTAs <= 152 SMs (GB300)
    assert kcp.fwdh_vh(4, 48, 152) == 128  # 192 > 152, no chain clause
    assert kcp.fwdh_vh(4, 48, 152, fits=True) == 64
    assert kcp.fwdh_vh(1, 48, 148) == 64
    assert kcp.fwdh_vh(6, 48, 152, forced=64) == 64
    assert kcp.fwdh_vh(1, 48, 152, forced=128, fits=True) == 128
    # four segments of 1536 / 1024 / 896 / 640 chunks at HV 48 (the K3E CP1 rank): 384 two-CTA items on 296 slots
    # (148 SMs) end at 1536 = the longest chain; on 128 slots they do not; one balanced segment kind fits only
    # while its items fill one round
    k3e = ((1536, 1), (1024, 1), (896, 1), (640, 1))
    assert kcp.walk_vh64_fits(k3e, 48, 148)
    assert not kcp.walk_vh64_fits(k3e, 48, 64)
    assert kcp.walk_vh64_fits(((100, 3),), 48, 148)  # 288 items <= 296 slots
    # 384 items: a second round of 100
    assert not kcp.walk_vh64_fits(((100, 4),), 48, 148)
    assert not kcp.walk_vh64_fits((), 48, 148)
    assert not kcp.walk_vh64_fits(k3e, 48, 0)
    assert kcp.VH64_SUFFIX == "_vh64"
    jit = _mod("cake_jit")
    if "kcp_fwdh_offset_vh64" in jit.MODULES:  # the registry of record ships both forms
        for prog in kcp.FWDH_PROGRAMS:
            assert jit.select_record(prog) == prog
            assert jit.select_record(prog, ctas_per_head=2) == prog + kcp.VH64_SUFFIX
            assert jit.launch_grid(prog, n_walk=3, num_heads=48) == (144, 1, 1)
            assert jit.launch_grid(
                prog + kcp.VH64_SUFFIX, n_walk=3, num_heads=48, ctas_per_head=2
            ) == (288, 1, 1)


def test_forward_walk_form_follows_the_plan_chains() -> None:
    """Plan test of the #2 chain clause: one contiguous rank of four documents of 24 / 16 / 14 / 10 chunks at HV 6,
    one segment per document.  ``n_walk * HV = 24`` exceeds both SM counts below, so the one-wave clause alone takes
    vh = 128; at 20 SMs the 48 two-CTA items on 40 slots end at the longest chain (LPT makespan 24), so the forward's
    tape / output walk launches the ``_vh64`` record; at 12 SMs (makespan 34) it stays vh = 128.  The fake backend
    runs the forward the plan launches (the record list of ``LAST_CALL``)."""
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    cu = _cu((24 * 64, 16 * 64, 14 * 64, 10 * 64))
    ops = _mode_ops(cu, (2, 6))
    args = [ops[n] for n in ("q", "k", "v", "g", "beta")]
    for sm, want in ((20, True), (12, False)):
        ctx = plan.cake_gdn_nondet_train_kcp_build_context(
            plan.cake_gdn_nondet_train_kcp_partition(cu, 1, "contiguous"),
            None,
            0,
            "cpu",
            sm_count=sm,
        )
        tb = kcp._Tables(ctx, 6, P=1)
        assert len(tb.walk_live) == 4 and tb.walk_fits(tb.walk_live, sm) is want
        assert kcp.fwdh_vh(4, 6, sm) == 128
        kcp.cake_gdn_nondet_train_kcp_forward(*args, ctx, backend="fake", segments=1)
        walks = [
            r
            for r, _ in kcp.LAST_CALL["fwd"]["records"]
            if r.startswith(("kcp_fwdh_tape", "kcp_fwdh_out"))
        ]
        assert walks and all(r.endswith(kcp.VH64_SUFFIX) is want for r in walks), (
            sm,
            walks,
        )


@pytest.mark.parametrize(("sm", "want"), [(20, True), (12, False)])
def test_walk_forms_agree_across_forward_recompute_and_handle(
    monkeypatch, sm, want
) -> None:
    """The width rules read host plan integers only, so the forward, torch's checkpoint recompute and the recompute
    from a :class:`CakeGDNNondetKcpRecomputeHandle` launch the same form of every #2 and #7 walk program.  The rank of
    :func:`test_forward_walk_form_follows_the_plan_chains` (HV 6, one segment per document): at 20 SMs the chain
    clause takes the ``_vh64`` record of the forward walks, at 12 SMs vh = 128; the fake backend has no
    architecture, so #7 takes vh = 64 (the #7 forms per architecture across the three paths:
    :func:`test_dhu_forms_agree_across_forward_recompute_and_handle_on_each_architecture`).  Each path logs the
    records of every recorded call (``_record``)."""
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    cu = _cu((24 * 64, 16 * 64, 14 * 64, 10 * 64))
    ops = _mode_ops(cu, (2, 6))
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(
        plan.cake_gdn_nondet_train_kcp_partition(cu, 1, "contiguous"),
        None,
        0,
        "cpu",
        sm_count=sm,
    )
    log: list = []
    record = kcp._record

    def recording(kind, call):
        record(kind, call)
        log.append((kind, [r for r, _ in kcp.LAST_CALL[kind]["records"]]))

    monkeypatch.setattr(kcp, "_record", recording)

    def run(ckpt: bool, handle: bool) -> dict:
        del log[:]
        leaves = [
            ops[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]
        hd = kcp.CakeGDNNondetKcpRecomputeHandle() if handle else None

        def fn(q, k, v, g, beta, h):
            return kcp.cake_gdn_nondet_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                recompute_handle=h,
                backend="fake",
                segments=1,
            )

        if ckpt:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(fn, *leaves, hd, use_reentrant=False)
        else:
            out = fn(*leaves, hd)
        out.backward(ops["dout"])
        forms: dict = {}
        for _kind, records in log:
            for r in records:
                base = r.removesuffix(kcp.VH64_SUFFIX)
                if base in kcp.FWDH_PROGRAMS + kcp.DHU_PROGRAMS:
                    forms.setdefault(base, set()).add(r.endswith(kcp.VH64_SUFFIX))
        return {"kinds": [k for k, _ in log], "forms": forms}

    paths = {
        "forward": run(False, False),
        "recompute": run(True, False),
        "handle": run(True, True),
    }
    for name in ("recompute", "handle"):
        # a checkpointed path records the forward and torch's recompute of it
        assert paths[name]["kinds"].count("fwd") == 2, (name, paths[name]["kinds"])
    for name, p in paths.items():
        forms = p["forms"]
        fwdh = [b for b in forms if b in kcp.FWDH_PROGRAMS]
        dhu = [b for b in forms if b in kcp.DHU_PROGRAMS]
        assert fwdh and dhu, (name, forms)
        assert all(forms[b] == {want} for b in fwdh), (name, sm, forms)
        assert all(forms[b] == {True} for b in dhu), (name, forms)
    assert paths["recompute"]["forms"] == paths["forward"]["forms"]
    assert paths["handle"]["forms"] == paths["forward"]["forms"]


def test_tape_shapes_follow_the_frozen_set() -> None:
    kcp = _mod("cake_kcp")
    R, NC_pad, HV, T = 256, 4, 6, 200
    bf, f32 = torch.bfloat16, torch.float32
    assert kcp.tape_spec("w", R=R, NC_pad=NC_pad, HV=HV, T=T) == ((R, HV, 128), bf)
    assert kcp.tape_spec("aqk", R=R, NC_pad=NC_pad, HV=HV, T=T) == ((R, HV, 64), bf)
    assert kcp.tape_spec("g_cum", R=R, NC_pad=NC_pad, HV=HV, T=T) == ((R, HV), f32)
    assert kcp.tape_spec("gamma", R=R, NC_pad=NC_pad, HV=HV, T=T) == ((NC_pad, HV), f32)
    assert kcp.tape_spec("h_tape", R=R, NC_pad=NC_pad, HV=HV, T=T) == (
        (NC_pad, HV, 128, 128),
        bf,
    )
    assert kcp.tape_spec("dq_part", R=R, NC_pad=NC_pad, HV=HV, T=T) == (
        (T, HV, 128),
        f32,
    )
    with pytest.raises(KeyError):
        kcp.tape_spec("nope", R=R, NC_pad=NC_pad, HV=HV, T=T)


def test_contract_dtypes() -> None:
    kcp = _mod("cake_kcp")
    dt = kcp.CONTRACT_DTYPES
    assert dt["g"] == dt["beta"] == dt["dg"] == dt["dbeta"] == torch.float32
    assert (
        dt["q"]
        == dt["k"]
        == dt["v"]
        == dt["out"]
        == dt["dout"]
        == dt["dq"]
        == dt["dk"]
        == dt["dv"]
        == torch.bfloat16
    )
    assert dt["state"] == torch.float32


# ----------------------------------------------------------------------------------------------------------------
# host tables on the CPU for the correctness rows' partitions (no kernel; the exchange tables, chains and pools)

ROWS = [
    # (lengths, (H_qk, HV), W, layout)  -- the C1..C5 / C9 rows plus the ragged tables
    ((100, 640, 77, 1000, 129), (2, 6), 2, "contiguous"),
    ((100, 640, 77, 1000, 129), (2, 6), 4, "contiguous"),
    ((3000,), (4, 12), 4, "contiguous"),
    ((3000,), (4, 12), 4, "zigzag"),
    ((3000,), (4, 12), 4, "zigzag_per_document"),
    ((2048, 5, 3000, 1), (1, 3), 4, "zigzag_per_document"),
    ((700 + 1 + 1300 + 999,), (8, 24), 4, "ragged:700,1,1300,999"),
    ((1200 + 900 + 900,), (8, 24), 4, "ragged:1200,0,900,900"),
    ((1,), (2, 6), 2, "contiguous"),
    ((0,), (2, 6), 2, "contiguous"),
    ((65553,), (16, 48), 8, "zigzag"),
]


def _cu(lengths):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return cu


def _ragged(cu, t_r):
    nested = [[] for _ in t_r]
    pos = 0
    for r, n in enumerate(t_r):
        end = pos + n
        d = 0
        while d < len(cu) - 1 and cu[d + 1] <= pos:
            d += 1
        p = pos
        while p < end and d < len(cu) - 1:
            e = min(end, cu[d + 1])
            if e > p:
                nested[r].append((d, p - cu[d], e - p))
            p = e
            d += 1
        pos = end
    return nested


def _partition(lengths, W, layout):
    plan = _mod("cake_kcp_plan")
    cu = _cu(lengths)
    if layout.startswith("ragged:"):
        t_r = [int(x) for x in layout.split(":")[1].split(",")]
        assert len(t_r) == W and sum(t_r) == cu[-1]
        return plan.cake_gdn_nondet_train_kcp_partition(
            cu, W, fragments=_ragged(cu, t_r)
        ), cu
    return plan.cake_gdn_nondet_train_kcp_partition(cu, W, layout=layout), cu


@pytest.mark.parametrize(
    "row",
    ROWS,
    ids=lambda r: (
        "s"
        + "_".join(map(str, r[0]))
        + f"-h{r[1][0]}x{r[1][1]}-w{r[2]}-{r[3].split(':')[0]}"
    ),
)
def test_host_tables_build_on_the_cpu_for_every_rank(row) -> None:
    lengths, (H_qk, HV), W, layout = row
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    part, cu = _partition(lengths, W, layout)
    for rank in range(W):
        ctx = plan.cake_gdn_nondet_train_kcp_build_context(
            part, None, rank, "cpu", sm_count=148
        )
        ws = kcp.CakeGDNNondetKcpWorkspace()
        tb = ws.ensure(ctx, H_qk, HV, device="cpu", backward=True)
        nseg = ctx.n_frag * tb.P
        assert tb.H == HV and tb.P >= 1 and nseg == tb.NSEG
        s = HV * kcp.TILE
        assert (
            tb.sizes["S_in"] == tb.numel["c2_recv"] >= ctx.n_in * s - (s - 1)
        )  # padded owners may add tiles
        assert tb.sizes["b2_recv"] >= ctx.n_out * s - (s - 1)
        assert ws.pool("eye").numel() == kcp.TILE and torch.equal(
            ws.pool("eye").view(128, 128), torch.eye(128)
        )
        for name in ("c1", "c2", "b1_d", "b1_dm", "b2"):
            send, recv = tb.split_elems(name)
            assert len(send) == W and len(recv) == W
        assert sum(tb.split_elems("c2")[1]) == tb.sizes["S_in"]
        # every chain family addresses its pools within bounds (checked at construction) and the walk lists cover
        # the live segments exactly once
        live = sorted(tb.walk_live)
        assert live == sorted(set(live))
        if tb.P == 1:
            assert sorted(tb.walk_offset) == sorted(
                s
                for s in range(tb.NSEG)
                if ctx.fragments[s].crossing_out and ctx.fragments[s].length > 0
            )
        desc = tb.describe()
        assert desc["n_in"] == ctx.n_in and desc["n_out"] == ctx.n_out
        # the tape geometry is bound and every tape of the frozen set has a shape
        assert ws.geometry == (ctx.R, ctx.NC_pad, HV, ctx.T_r)
        for name in kcp.FWD_TAPES + kcp.BWD_TAPES:
            kcp.tape_spec(name, R=ctx.R, NC_pad=ctx.NC_pad, HV=HV, T=ctx.T_r)
        # dv_new is stored in place of dv_intra: one buffer under both names, counted once
        assert kcp.TAPE_ALIASES == {"dv_new": "dv_intra"}
        before = ws.tape_bytes
        dv = ws.tape("dv_intra")
        assert ws.tape("dv_new") is dv and tuple(dv.shape) == (ctx.R, HV, 128)
        assert dv.dtype == torch.bfloat16
        assert ws.tape_bytes == before + ctx.R * HV * 128 * 2
        # the forward drops these arena tapes before it allocates out; #5 never reads them
        assert set(kcp.FWD_DEAD_BEFORE_O) <= set(kcp.FWD_TAPES)
        assert not set(kcp.FWD_DEAD_BEFORE_O) & set(kcp.PROGRAM_ARGS["fwd_o"])
        # #1's backward-only outputs: no forward program after fwd_prep names them (dropped right after #1)
        assert set(kcp.FWD_DEAD_AFTER_PREP) <= set(kcp.FWD_DEAD_BEFORE_O)
        for prog in (
            "kcp_fwdh_offset",
            "kcp_fwdh_tape",
            "kcp_fwdh_out",
            "kcp_fwdh_out_tape",
            "kcp_transfer_walk",
            "kcp_affine_merge",
            "fwd_o",
        ):
            assert not set(kcp.FWD_DEAD_AFTER_PREP) & set(kcp.PROGRAM_ARGS[prog]), prog
        assert set(kcp.FWD_DEAD_AFTER_PREP) <= set(kcp.PROGRAM_ARGS["fwd_prep"])
        w = ws.tape("w")
        held = ws.tape_bytes
        assert ws.release_tape("w") and not ws.release_tape("w")
        assert ws.tape_bytes == held - w.numel() * w.element_size()
        assert ws.tape("w") is not w and ws.tape_bytes == held
        # the forward drops every pool but the identity tile before out; the backward keeps carry / b2_recv for
        # the adjoint tape walk and drops them before #8; a dropped pool is sized again by ensure()
        assert set(tb.pool_sizes(False)) - set(kcp.FWD_DEAD_POOLS) == {"eye"}
        assert set(tb.pool_sizes(True)) - set(kcp.BWD_DEAD_BEFORE_TAPES) - set(
            kcp.BWD_DEAD_BEFORE_DQKG
        ) == {"eye"}
        assert not set(kcp.BWD_DEAD_BEFORE_TAPES) & set(kcp.BWD_DEAD_BEFORE_DQKG)
        # the pools dropped after the adjoint merge are a subset of the before-tape-walk set and leave the b2
        # exchange, the carry plane and the identity tile alone
        assert set(kcp.BWD_DEAD_AFTER_ADJ_MERGE) < set(kcp.BWD_DEAD_BEFORE_TAPES)
        assert set(kcp.BWD_DEAD_BEFORE_TAPES) - set(kcp.BWD_DEAD_AFTER_ADJ_MERGE) == {
            "b2_send",
            "stage_b",
            "stage_m",
        }  # b2 is in flight; the dcarry merges still read the staging planes
        assert not set(kcp.BWD_DEAD_AFTER_ADJ_MERGE) & {"b2_recv", "carry", "eye"}
        for name in ("dcarry_end", "dcarry_cross", "carry_start", "carry_in"):
            spec = tb.merge_spec(name) if hasattr(tb, "merge_spec") else None
            if spec is not None:
                assert not {spec["m"], spec["c"]} & set(kcp.BWD_DEAD_AFTER_ADJ_MERGE)
        assert ws.has("stage_b") and ws.release_pool("stage_b")
        assert not ws.has("stage_b") and not ws.release_pool("stage_b")
        ws.ensure(ctx, H_qk, HV, device="cpu", backward=True)
        assert ws.has("stage_b")
        # the launch tables are plan data cached on the context: a second (per-call) workspace reuses them
        assert kcp.CakeGDNNondetKcpWorkspace().tables(ctx, HV) is tb
        assert ("kcp_tables", HV, 1, None) in ctx._caches
        ws2 = kcp.CakeGDNNondetKcpWorkspace()
        ws2.ensure(ctx, H_qk, HV, device="cpu", backward=False)
        assert ws2.pool("eye") is ws.pool(
            "eye"
        )  # the identity tile is built once per context


def test_workspace_rejects_a_bad_head_pair_and_device_change() -> None:
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    part, _ = _partition((300, 200), 2, "contiguous")
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(
        part, None, 0, "cpu", sm_count=148
    )
    ws = kcp.CakeGDNNondetKcpWorkspace()
    with pytest.raises(ValueError):
        ws.ensure(ctx, 4, 6, device="cpu")
    ws.ensure(ctx, 2, 6, device="cpu")
    with pytest.raises(ValueError):
        ws.ensure(ctx, 2, 6, device="meta")


def test_exchange_scheme_ring_is_refused_beyond_two_hops() -> None:
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    part, _ = _partition((3000,), 4, "zigzag")  # one document over 8 pieces: 7 hops
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(
        part, None, 0, "cpu", sm_count=148
    )
    tb = kcp.CakeGDNNondetKcpWorkspace().ensure(ctx, 4, 12, device="cpu")
    with pytest.raises(ValueError):
        kcp._exchange_scheme(tb, make_options(exchange="ring", save_transfer=False))
    assert kcp._exchange_scheme(tb, make_options()) == "a2a"
    part2, _ = _partition(
        (100, 640, 77, 1000, 129), 2, "contiguous"
    )  # CP2 contiguous: at most 1 hop
    ctx2 = plan.cake_gdn_nondet_train_kcp_build_context(
        part2, None, 0, "cpu", sm_count=148
    )
    tb2 = kcp.CakeGDNNondetKcpWorkspace().ensure(ctx2, 2, 6, device="cpu")
    assert (
        kcp._exchange_scheme(tb2, make_options(exchange="ring", save_transfer=False))
        == "ring"
    )


def _assert_same_saved(back, saved) -> None:
    """Field-by-field identity of two saved sets: tensors (and tensor tuples) by object identity, every other field by
    ``==``.  (The dataclass ``__eq__`` of Python >= 3.13 compares the fields one by one without the tuple identity
    shortcut, so a tensor field would be compared element-wise.)"""
    for f in dataclasses.fields(saved):
        a, b = getattr(back, f.name), getattr(saved, f.name)
        if isinstance(b, torch.Tensor) or (
            isinstance(b, tuple) and any(isinstance(x, torch.Tensor) for x in b)
        ):
            if isinstance(b, tuple):
                assert len(a) == len(b) and all(
                    x is y for x, y in zip(a, b, strict=True)
                ), f.name
            else:
                assert a is b, f.name
        else:
            assert a == b, (f.name, a, b)


def test_saved_set_round_trips_through_tensors_and_meta() -> None:
    kcp = _mod("cake_kcp")
    S_in = torch.zeros(4)
    tapes = tuple(torch.zeros(1) for _ in kcp.FWD_TAPES)
    saved = kcp.CakeGDNNondetKcpSaved(
        version=kcp.KCP_SAVED_VERSION,
        S_in=S_in,
        c1_recv=None,
        M_seg=None,
        tapes=tapes,
        T_r=10,
        H_qk=2,
        HV=6,
        n_frag=1,
        P=1,
        Hg=3,
        scale=0.5,
        save_mode="full",
        save_transfer=False,
        offset_class="fla",
        exchange="a2a",
        precision_mode=0,
        transfer_walk="tf32_hilo",
        ctx=None,
    )
    back = kcp.CakeGDNNondetKcpSaved.from_tensors(
        saved.tensors(), saved.flags(), saved.meta(), None
    )
    _assert_same_saved(back, saved)
    assert back.tape("h_tape") is tapes[kcp.FWD_TAPES.index("h_tape")]
    assert saved.bytes == 4 * 4 + 4 * len(kcp.FWD_TAPES)
    light_tapes = tuple(torch.zeros(2) for _ in kcp.LIGHT_TAPES)  # empty
    light = dataclasses.replace(saved, tapes=light_tapes, save_mode="light")
    assert light.flags() == (False, False, True)
    assert len(light.tensors()) == 1 + len(kcp.LIGHT_TAPES) == 1
    back_light = kcp.CakeGDNNondetKcpSaved.from_tensors(
        light.tensors(), light.flags(), light.meta(), None
    )
    _assert_same_saved(back_light, light)
    assert back_light.tapes == ()
    with pytest.raises(KeyError):
        light.tape("akk")
    with pytest.raises(KeyError):
        light.tape("w")
    empty = dataclasses.replace(
        saved, tapes=None, save_mode="light"
    )  # T_r = 0: no tapes at all
    assert empty.flags() == (False, False, False) and len(empty.tensors()) == 1
    with pytest.raises(KeyError):
        empty.tape("gamma")


def test_graph_entry_key_ignores_the_arena_and_the_capture_pins_its_signature() -> None:
    """The replay key names the call (context, workspace, backend, options, operands), not the arena's pointers:
    an eager call drops its dead pools and tapes and the next ``ensure()`` re-allocates them, which must not start a
    new entry (one eager call, one capture, then replays).  The capture records the arena signature instead, and a
    replay needs that exact arena: a dropped or re-allocated buffer resets the entry, which then runs eagerly once
    and is captured again."""
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    part, _cu = _partition([100, 640, 77], 2, "contiguous")
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(
        part, None, 0, "cpu", sm_count=148
    )
    H_qk, HV = 2, 6
    ws = kcp.CakeGDNNondetKcpWorkspace()
    ws.ensure(ctx, H_qk, HV, device="cpu", backward=True)
    for name in kcp.FWD_TAPES:
        ws.tape(name)
    ops = tuple(torch.zeros(4, 2, 8) for _ in range(5))  # pointer-stable operands
    opts = make_options(None, graph="replay")
    backend = object()
    kcp.graph_cache_clear()
    e1 = kcp._graph_entry("fwd", ctx, ws, backend, opts, None, ops, (True,))
    assert e1 is not None and e1.graph is None and e1.arena is None
    for name in kcp.FWD_DEAD_BEFORE_O:  # what an eager forward drops before `out`
        ws.release_tape(name)
    for name in kcp.FWD_DEAD_POOLS:
        ws.release_pool(name)
    assert kcp._graph_entry("fwd", ctx, ws, backend, opts, None, ops, (True,)) is e1
    ws.ensure(
        ctx, H_qk, HV, device="cpu", backward=True
    )  # the next call re-grows the arena
    for name in kcp.FWD_TAPES:
        ws.tape(name)
    assert kcp._graph_entry("fwd", ctx, ws, backend, opts, None, ops, (True,)) is e1
    # another workspace, context or operand pointer is another key
    assert (
        kcp._graph_entry(
            "fwd",
            ctx,
            kcp.CakeGDNNondetKcpWorkspace(),
            backend,
            opts,
            None,
            ops,
            (True,),
        )
        is not e1
    )
    assert (
        kcp._graph_entry(
            "fwd",
            ctx,
            ws,
            backend,
            opts,
            None,
            ops[:4] + (torch.zeros(4, 2, 8),),
            (True,),
        )
        is not e1
    )
    # a captured entry replays only on the arena it baked
    sig = kcp._arena_sig(ws)
    assert sig == kcp._arena_sig(ws) and len(sig[0]) == len(ws._pools)
    e1.graph, e1.arena, e1.seen = object(), sig, 2
    assert e1.replayable(ws)
    before = kcp.GRAPH_STATS["reset"]
    ws.release_pool("stage_b")
    assert not e1.replayable(ws)
    assert (
        e1.graph is None and e1.arena is None and e1.seen == 0 and e1.reset == "arena"
    )
    assert kcp.GRAPH_STATS["reset"] == before + 1
    assert not e1.replayable(ws)  # no graph: eager, and no second reset
    assert kcp.GRAPH_STATS["reset"] == before + 1
    assert kcp.graph_cache_clear() == 3


def test_dhu_form_rule_follows_the_architecture_recommendation() -> None:
    """#7 runs the vh = 64 two-CTA form for the offset walk always and for the tape walk on ``sm_100a`` / ``sm_103a``
    always (``arch`` ``None``, the fake backend, counts as one of them); on ``sm_107a`` the tape walk runs vh = 64 when
    its grid ``n_walk * HV * 2`` fits two waves at two CTAs per SM (``<= 4 * sm_count``), else vh = 128; the test
    knob ``dhu_vh`` forces a form (both forms within the same accuracy budget).  ``resolve_launch`` names the record
    the registry provides: the ``_vh64`` record where the registry ships it, the full-head record with the grid
    ``n_walk * HV`` where it does not (no error either way)."""
    kcp = _mod("cake_kcp")
    assert kcp.DHU_PROGRAMS == ("kcp_dhu_offset", "kcp_dhu_tape")
    assert kcp.DHU_TAPE_VH64_ARCHES == ("sm_100a", "sm_103a")
    for arch in (None, "sm_100a", "sm_103a"):
        assert kcp.dhu_vh(96, 48, 152, arch=arch) == 64
        assert kcp.dhu_vh(96, 48, 152, tape=False, arch=arch) == 64
    r = "sm_107a"
    # the offset walk: vh 64 everywhere
    assert kcp.dhu_vh(96, 48, 212, tape=False, arch=r) == 64
    assert (
        kcp.dhu_vh(8, 38, 152, arch=r) == 64
    )  # 8 * 38 * 2 = 608 == 4 * 152: the boundary is vh 64
    assert kcp.dhu_vh(8, 38, 151, arch=r) == 128  # 608 > 604
    assert kcp.dhu_vh(8, 48, 212, arch=r) == 64  # 768 <= 848
    assert kcp.dhu_vh(9, 48, 212, arch=r) == 128  # 864 > 848
    assert kcp.dhu_vh(9, 48, 212, forced=64, arch=r) == 64
    assert kcp.dhu_vh(1, 48, 152, forced=128) == 128
    jit = _mod("cake_jit")
    ships_vh64 = all((p + kcp.VH64_SUFFIX) in jit.MODULES for p in kcp.DHU_PROGRAMS)
    for prog in kcp.DHU_PROGRAMS:
        assert (
            jit.select_record(prog) == prog
        )  # the default selection stays the full-head record
        for forced, n_walk, arch in (
            (None, 3, None),
            (64, 9, "sm_107a"),
            (128, 3, None),
            (None, 7, None),
            (None, 7, "sm_107a"),
        ):
            args = {
                "_grid_args": {"n_walk": n_walk, "num_heads": 48},
                "_sm_count": 152,
                "_dhu_vh": forced,
                "_arch": arch,
                "_grid": n_walk * 48,
            }
            record, values, grid, is_test = kcp.resolve_launch(prog, dict(args))
            assert values == {} and not is_test
            vh = kcp.dhu_vh(
                n_walk, 48, 152, forced, tape=prog == "kcp_dhu_tape", arch=arch
            )
            if (prog, n_walk, arch) == ("kcp_dhu_tape", 7, "sm_107a"):
                assert vh == 128  # 672 > 608: the sm_107a two-wave rule
            if vh == 64 and ships_vh64:
                assert record == prog + kcp.VH64_SUFFIX, (prog, forced, n_walk)
                assert grid == (n_walk * 48 * 2, 1, 1)
                assert jit.launch_grid(record, n_walk=n_walk, num_heads=48) == grid
            else:
                assert record == prog and grid == (n_walk * 48, 1, 1), (
                    prog,
                    forced,
                    n_walk,
                )
    if ships_vh64:
        for prog in kcp.DHU_PROGRAMS:
            assert jit.select_record(prog, ctas_per_head=2) == prog + kcp.VH64_SUFFIX
            assert jit.MODULES[prog + kcp.VH64_SUFFIX]["program"] == prog


def test_dhu_long_chain_walk_takes_the_two_wave_rule_on_sm_100a() -> None:
    """On ``sm_100a`` (:data:`DHU_LONG_CHAIN_ARCHES`) a #7 walk whose longest segment has at least
    ``DHU_LONG_CHAIN_MIN_CHUNKS`` = 400 chunks and whose two-CTA grid ends later than its one-CTA grid (LPT over
    ``2 * sm_count`` slots: :func:`walk_dhu_long_chain`) takes the two-wave rule (vh = 64 iff ``2 * n_walk * HV <= 4 *
    sm_count``) for the offset and the tape walk.  ``sm_103a``, ``sm_107a`` and the fake backend (``arch`` ``None``)
    keep their rules for the same walk."""
    kcp = _mod("cake_kcp")
    assert kcp.DHU_LONG_CHAIN_ARCHES == ("sm_100a",)
    assert kcp.DHU_LONG_CHAIN_MIN_CHUNKS == 400
    # one segment of L0 chunks and five of 140 at HV 48 on 148 SMs (296 slots): the one-CTA grid ends at L0 (48 long
    # CTAs, 240 short ones one per slot); the two-CTA grid puts 480 short CTAs on the 200 other slots, three deep on
    # 80 of them: 420 > L0 for L0 = 399 / 400, so only the 400-chunk segment reaches the clause
    assert kcp.walk_dhu_long_chain(((400, 1), (140, 5)), 48, 148)
    assert not kcp.walk_dhu_long_chain(((399, 1), (140, 5)), 48, 148)
    # a long segment whose two-CTA grid still ends at it (three short CTAs deep = 300 < 500): not a long-chain walk
    assert not kcp.walk_dhu_long_chain(((500, 1), (100, 5)), 48, 148)
    assert not kcp.walk_dhu_long_chain((), 48, 148)
    assert not kcp.walk_dhu_long_chain(((400, 1), (140, 5)), 48, 0)
    # the twelve-segment #7 walk of a P1-shaped CP2 rank on 148 SMs: one-CTA LPT 536, two-CTA LPT 640
    p1 = (
        (536, 1),
        (451, 1),
        (147, 1),
        (138, 1),
        (126, 2),
        (96, 1),
        (90, 1),
        (61, 1),
        (59, 1),
        (44, 1),
        (43, 1),
    )
    assert kcp.walk_dhu_long_chain(p1, 48, 148)
    for tape in (True, False):
        # sm_100a: 2 * 12 * 48 = 1152 > 4 * 148 -> vh 128 for both forms; without the clause both are vh 64
        assert (
            kcp.dhu_vh(12, 48, 148, tape=tape, arch="sm_100a", long_chain=True) == 128
        )
        assert kcp.dhu_vh(12, 48, 148, tape=tape, arch="sm_100a") == 64
        # the two-wave rule still takes vh 64 where the grid fits: 2 * 6 * 48 = 576 <= 592
        assert kcp.dhu_vh(6, 48, 148, tape=tape, arch="sm_100a", long_chain=True) == 64
        # forced knob wins
        assert (
            kcp.dhu_vh(12, 48, 148, 64, tape=tape, arch="sm_100a", long_chain=True)
            == 64
        )
        # not on sm_103a or the fake backend
        for arch in ("sm_103a", None):
            assert kcp.dhu_vh(12, 48, 152, tape=tape, arch=arch, long_chain=True) == 64
    # sm_107a keeps its rules: the offset walk stays vh 64; the tape walk keeps the two-wave rule it already has
    assert kcp.dhu_vh(12, 48, 208, tape=False, arch="sm_107a", long_chain=True) == 64
    assert kcp.dhu_vh(8, 48, 212, arch="sm_107a", long_chain=True) == 64  # 768 <= 848
    assert kcp.dhu_vh(9, 48, 212, arch="sm_107a", long_chain=True) == 128  # 864 > 848
    # resolve_launch reads the clause flag and the launch cache key carries it
    for prog in kcp.DHU_PROGRAMS:
        base = {
            "_grid_args": {"n_walk": 12, "num_heads": 48},
            "_sm_count": 148,
            "_dhu_vh": None,
            "_arch": "sm_100a",
            "_grid": 12 * 48,
        }
        long_args = {**base, "_dhu_long_chain": True}
        record, _values, grid, _is_test = kcp.resolve_launch(prog, dict(long_args))
        assert record == prog and grid == (12 * 48, 1, 1), (prog, record, grid)
        assert kcp._launch_key(prog, long_args, 8) != kcp._launch_key(prog, base, 8)


@pytest.mark.parametrize("longest", [24, 23])
def test_dhu_long_chain_flag_follows_the_plan_chains(monkeypatch, longest) -> None:
    """Plan test of the #7 long-chain flag (the 400-chunk threshold lowered to 24 so the rank stays small): one
    contiguous rank of six documents of ``longest`` / 10 / 10 / 10 / 10 / 10 chunks at HV 6 on 18 SMs (36 slots),
    one segment per document.  The one-CTA grid ends at the longest segment (6 long CTAs, 30 short ones one per
    slot); the two-CTA grid puts 60 short CTAs on the 24 other slots, three deep on 12 of them (30 > ``longest``).
    So ``_Tables.walk_long_chain`` holds for 24 and not for 23, and the backward's #7 launches carry that flag to
    :func:`dhu_vh` (the fake backend has no architecture, so the forms themselves stay vh 64)."""
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    monkeypatch.setattr(kcp, "DHU_LONG_CHAIN_MIN_CHUNKS", 24)
    cu = _cu((longest * 64,) + (10 * 64,) * 5)
    ops = _mode_ops(cu, (2, 6))
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(
        plan.cake_gdn_nondet_train_kcp_partition(cu, 1, "contiguous"),
        None,
        0,
        "cpu",
        sm_count=18,
    )
    tb = kcp._Tables(ctx, 6, P=1)
    assert sorted(tb._walk_chains(tb.walk_live)) == [(10, 5), (longest, 1)]
    assert tb.walk_long_chain(tb.walk_live, 18) is (longest == 24)
    seen: list = []
    real = kcp.dhu_vh

    def recording(*args, **kw):
        seen.append(kw.get("long_chain"))
        return real(*args, **kw)

    monkeypatch.setattr(kcp, "dhu_vh", recording)
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    out = kcp.cake_gdn_nondet_kcp(*leaves, ctx, backend="fake", segments=1)
    out.backward(ops["dout"])
    assert seen and all(isinstance(x, bool) for x in seen)
    assert any(seen) is (longest == 24), seen


# ----------------------------------------------------------------------------------------------------------------
# the architecture the kernel backend passes to the #7 walks (KernelBackend.run on a stub launch glue, no kernel)

# One contiguous rank of six documents of 8 / 4 / 4 / 4 / 4 / 4 chunks at HV 6 on 18 SMs (36 slots), run with
# ``segments=2``: twelve segments, two of 4 chunks and ten of 2.  The #7 walks of this rank visit all twelve, so
# their two-wave rule takes vh = 128 (2 * 12 * 6 = 144 > 4 * 18); with ``DHU_LONG_CHAIN_MIN_CHUNKS`` lowered to 4
# they are long-chain walks (LPT over the 36 slots: the one-CTA grid ends at 6, the two-CTA grid at 10).
ARCH_CHUNKS = (8, 4, 4, 4, 4, 4)
ARCH_HEADS = (2, 6)
ARCH_SM = 18
ARCH_LONG_CHAIN_MIN = 4


class _ArchGlue:
    """A stub launch glue of :class:`KernelBackend` (its ``launcher``): ``device_arch`` answers ``arch`` for every
    device index; ``run_module`` logs each launch's ``(record, grid)`` and runs the record on the fake backend's
    emulation (the same logical arguments), so a real plan runs through ``KernelBackend.run`` on the CPU."""

    def __init__(self, arch: str) -> None:
        self.arch = arch
        self.launches: list[tuple[str, tuple[int, ...]]] = []
        self._fake = _mod("cake_fake_backend")

    def device_arch(self, device_index: int) -> str:
        return self.arch

    def run_module(self, record, values, *, grid, device_index, module=None) -> None:
        self.launches.append((str(record), tuple(int(x) for x in grid)))
        self._fake.run_module(record, values, grid=grid, device_index=None)


def _arch_backend(monkeypatch, arch: str) -> tuple[_ArchGlue, list]:
    """Install ``KernelBackend(launcher=_ArchGlue(arch))`` as the ``"kernels"`` backend for one test and log, in
    launch order (one entry per ``run_module`` call), the program and the ``_arch`` / ``_dhu_long_chain`` /
    ``_grid_args`` that ``KernelBackend.run`` resolves each launch with.  A CPU context has no device index, so
    ``torch.cuda.current_device`` answers 0 (the glue ignores the index)."""
    kcp = _mod("cake_kcp")
    glue = _ArchGlue(arch)
    monkeypatch.setitem(kcp._BACKENDS, "kernels", kcp.KernelBackend(launcher=glue))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    resolved: list = []
    real = kcp._resolve_cached

    def resolving(name, args, **kw):
        resolved.append(
            (
                name,
                args.get("_arch"),
                args.get("_dhu_long_chain"),
                dict(args.get("_grid_args") or {}),
            )
        )
        return real(name, args, **kw)

    monkeypatch.setattr(kcp, "_resolve_cached", resolving)
    return glue, resolved


def _dhu_launches(glue: _ArchGlue, resolved: list) -> list[dict]:
    """The #7 launches of a run in launch order: program, record, grid and the scalars ``KernelBackend.run``
    resolved them with (``_arch``, ``_dhu_long_chain``, ``n_walk``)."""
    kcp = _mod("cake_kcp")
    assert len(glue.launches) == len(resolved) > 0
    return [
        {
            "program": name,
            "record": record,
            "grid": grid,
            "arch": arch,
            "long_chain": long_chain,
            "n_walk": int(grid_args["n_walk"]),
        }
        for (record, grid), (name, arch, long_chain, grid_args) in zip(
            glue.launches, resolved, strict=True
        )
        if name in kcp.DHU_PROGRAMS
    ]


def _dhu_rule_vh(
    arch: str, tape: bool, n_walk: int, num_heads: int, sm_count: int, long_chain: bool
) -> int:
    """The #7 form of one launch as the rule text states it (restated here, not read from :func:`dhu_vh`): the
    two-wave rule (vh = 64 iff ``2 * n_walk * HV <= 4 * sm_count``) for the tape walk on ``sm_107a`` and for a
    long-chain walk of either form on ``sm_100a``; vh = 64 for every other launch."""
    two_wave = 64 if 2 * n_walk * num_heads <= 4 * sm_count else 128
    if (tape and arch == "sm_107a") or (long_chain and arch == "sm_100a"):
        return two_wave
    return 64


def _arch_case(monkeypatch, long_chain: bool):
    """The context and operands of the :data:`ARCH_CHUNKS` rank (``DHU_LONG_CHAIN_MIN_CHUNKS`` lowered to
    :data:`ARCH_LONG_CHAIN_MIN` when ``long_chain``), after checking the plan's #7 chains."""
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    if long_chain:
        monkeypatch.setattr(kcp, "DHU_LONG_CHAIN_MIN_CHUNKS", ARCH_LONG_CHAIN_MIN)
    cu = _cu(tuple(n * 64 for n in ARCH_CHUNKS))
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(
        plan.cake_gdn_nondet_train_kcp_partition(cu, 1, "contiguous"),
        None,
        0,
        "cpu",
        sm_count=ARCH_SM,
    )
    HV = ARCH_HEADS[1]
    tb = kcp._Tables(ctx, HV, P=2)
    assert sorted(tb._walk_chains(tb.walk_all)) == [(2, 10), (4, 2)]
    assert tb.walk_long_chain(tb.walk_all, ARCH_SM) is long_chain
    return ctx, _mode_ops(cu, ARCH_HEADS)


@pytest.mark.parametrize("long_chain", [False, True])
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_kernel_backend_passes_the_device_architecture_to_the_dhu_walks(
    monkeypatch, arch, long_chain
) -> None:
    """``KernelBackend.run`` is the only producer of the ``_arch`` the #7 width rule reads (:func:`dhu_vh`; the fake
    backend has none).  A real plan (the :data:`ARCH_CHUNKS` rank, forward and backward with ``segments=2``) runs
    through it with a stub launch glue whose ``device_arch`` answers ``arch`` (:func:`_arch_backend`).  Every #7
    launch receives that architecture, and its record is the form the rule text gives (:func:`_dhu_rule_vh`) on
    the grid of that form.  On this plan: ``sm_100a`` and ``sm_103a`` launch vh = 64 everywhere; ``sm_107a`` keeps
    the two-wave rule for the tape walk (vh = 128 here) and vh = 64 for the offset walk; with the long-chain
    threshold lowered, ``sm_100a`` takes the two-wave rule (vh = 128) for the offset and the tape walk while
    ``sm_103a`` stays vh = 64 and ``sm_107a`` keeps its forms.  The persistent launches receive the same
    architecture in their grid arguments."""
    kcp = _mod("cake_kcp")
    ctx, ops = _arch_case(monkeypatch, long_chain)
    glue, resolved = _arch_backend(monkeypatch, arch)
    HV = ARCH_HEADS[1]
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    out = kcp.cake_gdn_nondet_kcp(*leaves, ctx, backend="kernels", segments=2)
    out.backward(ops["dout"])
    launches = _dhu_launches(glue, resolved)
    assert {x["program"] for x in launches} == set(kcp.DHU_PROGRAMS), launches
    for x in launches:
        tape = x["program"] == "kcp_dhu_tape"
        vh = 64 if x["record"].endswith(kcp.VH64_SUFFIX) else 128
        assert x["arch"] == arch, x
        assert x["grid"] == (x["n_walk"] * HV * (2 if vh == 64 else 1), 1, 1), x
        assert vh == _dhu_rule_vh(
            arch, tape, x["n_walk"], HV, ARCH_SM, x["long_chain"]
        ), (arch, x)
    # the plan reaches every clause: each #7 program has launches whose two-wave rule is vh = 128, and a long-chain
    # one among them exactly where the threshold is lowered
    for prog in kcp.DHU_PROGRAMS:
        wide = [
            x
            for x in launches
            if x["program"] == prog and 2 * x["n_walk"] * HV > 4 * ARCH_SM
        ]
        assert wide and any(x["long_chain"] for x in wide) is long_chain, (
            prog,
            launches,
        )
    full = {x["program"] for x in launches if not x["record"].endswith(kcp.VH64_SUFFIX)}
    want = {
        "sm_100a": set(kcp.DHU_PROGRAMS) if long_chain else set(),
        "sm_103a": set(),
        "sm_107a": {"kcp_dhu_tape"},
    }[arch]
    assert full == want, (arch, long_chain, launches)
    persistent = [ga for _n, _a, _l, ga in resolved if "sm_count" in ga]
    assert persistent and all(ga["arch"] == arch for ga in persistent)


@pytest.mark.parametrize(
    ("arch", "want"),
    [
        ("sm_100a", {"kcp_dhu_offset", "kcp_dhu_tape"}),
        ("sm_103a", set()),
        ("sm_107a", {"kcp_dhu_tape"}),
    ],
)
def test_dhu_forms_agree_across_forward_recompute_and_handle_on_each_architecture(
    monkeypatch, arch, want
) -> None:
    """The #7 form of every launch reads host plan integers and the architecture ``KernelBackend.run`` passes, so
    the backward after the forward, after torch's checkpoint recompute and after the recompute from a
    :class:`CakeGDNNondetKcpRecomputeHandle` launch the same #7 records on the same grids.  The :data:`ARCH_CHUNKS` rank with
    ``segments=2`` and the long-chain threshold lowered, through the stub launch glue of
    :func:`test_kernel_backend_passes_the_device_architecture_to_the_dhu_walks`: on ``sm_100a`` the long-chain
    clause fires (the offset and the tape walk launch vh = 128); on ``sm_103a`` it does not (vh = 64), for the
    same long-chain walks; ``sm_107a`` keeps the two-wave rule of the tape walk (vh = 128) and vh = 64 for the
    offset walk.  ``want``: the #7 programs that launch vh = 128.  Each path logs the kinds of its recorded calls
    (``_record``)."""
    kcp = _mod("cake_kcp")
    ctx, ops = _arch_case(monkeypatch, True)
    glue, resolved = _arch_backend(monkeypatch, arch)
    kinds: list = []
    record = kcp._record

    def recording(kind, call):
        record(kind, call)
        kinds.append(kind)

    monkeypatch.setattr(kcp, "_record", recording)

    def run(ckpt: bool, handle: bool) -> dict:
        del kinds[:], glue.launches[:], resolved[:]
        leaves = [
            ops[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]
        hd = kcp.CakeGDNNondetKcpRecomputeHandle() if handle else None

        def fn(q, k, v, g, beta, h):
            return kcp.cake_gdn_nondet_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                recompute_handle=h,
                backend="kernels",
                segments=2,
            )

        if ckpt:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(fn, *leaves, hd, use_reentrant=False)
        else:
            out = fn(*leaves, hd)
        out.backward(ops["dout"])
        return {"kinds": list(kinds), "dhu": _dhu_launches(glue, resolved)}

    paths = {
        "forward": run(False, False),
        "recompute": run(True, False),
        "handle": run(True, True),
    }
    for name in ("recompute", "handle"):
        # a checkpointed path records the forward and torch's recompute of it
        assert paths[name]["kinds"].count("fwd") == 2, (name, paths[name]["kinds"])
    forms = {}
    for name, p in paths.items():
        dhu = p["dhu"]
        assert {x["program"] for x in dhu} == set(kcp.DHU_PROGRAMS), (name, dhu)
        assert all(x["arch"] == arch for x in dhu), (name, dhu)
        assert any(x["long_chain"] for x in dhu), (name, dhu)
        full = {x["program"] for x in dhu if not x["record"].endswith(kcp.VH64_SUFFIX)}
        assert full == want, (name, dhu)
        forms[name] = [(x["record"], x["grid"]) for x in dhu]
    assert forms["recompute"] == forms["forward"], forms
    assert forms["handle"] == forms["forward"], forms


def test_workspace_holds_one_zeroed_ticket_slot_per_persistent_site() -> None:
    """The persistent programs' dynamic-ticket words are workspace slots: one 128-byte u32 slot per launch site
    (``fwd_o``, ``bwd_prep``, ``bwd_dqkg``), allocated and zeroed once by ``ensure()`` (never per call), kept across
    pool / tape releases and re-sizing, counted by ``bytes`` and by the public workspace query in both lifetimes,
    and part of the arena signature a capture pins."""
    kcp, plan, gb = _mod("cake_kcp"), _mod("cake_kcp_plan"), _mod("cake_bytes")
    assert kcp.TICKET_SITES == gb.TICKET_SITES == ("fwd_o", "bwd_prep", "bwd_dqkg")
    assert kcp.TICKET_SLOT_BYTES == gb.TICKET_SLOT_BYTES == 128
    assert kcp.TICKET_SLOT_WORDS == 32 and gb.TICKET_BYTES == 3 * 128
    assert set(kcp.TICKET_SITES) <= set(kcp.LOCAL_PROGRAMS)
    for site in kcp.TICKET_SITES:
        assert kcp.PROGRAM_ARGS[site][-1] == "ticket"
    assert "ticket" in kcp.PROVISIONAL_ARGS
    part, _cu = _partition([100, 640, 77], 2, "contiguous")
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(
        part, None, 0, "cpu", sm_count=148
    )
    H_qk, HV = 2, 6
    ws = kcp.CakeGDNNondetKcpWorkspace()
    assert ws.ticket_bytes == 0 and ws.bytes == 0
    with pytest.raises(KeyError, match="ensure"):
        ws.ticket("fwd_o")
    ws.ensure(ctx, H_qk, HV, device="cpu", backward=True)
    slots = {site: ws.ticket(site) for site in kcp.TICKET_SITES}
    for site, t in slots.items():
        assert t.dtype == torch.uint32 and t.numel() == 32, site
        assert t.data_ptr() % 128 == 0, site
        assert int(t.view(torch.int32).abs().sum()) == 0, site
    assert len({t.data_ptr() for t in slots.values()}) == 3
    with pytest.raises(KeyError, match="persistent launch site"):
        ws.ticket("fwd_prep")
    assert ws.ticket_bytes == gb.TICKET_BYTES
    assert ws.bytes == sum(ws.pool_bytes.values()) + ws.tape_bytes + gb.TICKET_BYTES
    # one allocation per workspace: releases and a re-size keep the slots
    for name in kcp.FWD_TAPES:
        ws.tape(name)
    for name in kcp.FWD_DEAD_POOLS:
        ws.release_pool(name)
    for name in kcp.FWD_DEAD_BEFORE_O:
        ws.release_tape(name)
    ws.ensure(ctx, H_qk, HV, device="cpu", backward=True)
    assert all(ws.ticket(s).data_ptr() == slots[s].data_ptr() for s in kcp.TICKET_SITES)
    sig = kcp._arena_sig(ws)
    assert sig[2] == (("tickets", ws._tickets.data_ptr(), 3 * 32),)
    other = kcp.CakeGDNNondetKcpWorkspace()
    other.ensure(ctx, H_qk, HV, device="cpu", backward=True)
    assert other.ticket("fwd_o").data_ptr() != ws.ticket("fwd_o").data_ptr()
    # the public query counts every site's slot in both lifetimes
    layout = gb.cake_gdn_nondet_train_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV)
    assert layout["fwd.tickets"] == layout["bwd.tickets"] == gb.TICKET_BYTES
    assert (
        gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, backward=False
        )["fwd.tickets"]
        == gb.TICKET_BYTES
    )
