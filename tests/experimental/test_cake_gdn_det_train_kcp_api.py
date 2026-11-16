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

CPU tests of the public surface of the context-parallel packed GDN training route: the options validator
(``cake_gdn_det_train_kcp.cake_options``), the keyword-only signatures, the facade ``flashinfer.cake_gdn_det_train_kcp`` (the contract names),
the package hygiene rules (no environment variable, no KDA package import, the host module's import allow-list)
and the host tables built on the CPU for the correctness rows' partitions.  No kernel runs here.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import inspect
import pathlib
import subprocess
import sys
import types
import typing

import pytest
import torch

PACKAGE = "flashinfer.experimental.cake.cake_gdn_det_train_kcp"


@pytest.fixture(autouse=True, scope="module")
def _few_cpu_threads():
    """The CPU tests run many small matmuls and elementwise ops.  On a host with hundreds of CPUs, torch's default
    intra-op pool and the oneDNN matmul path (its own pool, which torch.set_num_threads does not resize) spend far more
    time synchronizing threads than computing (the forward-form suite took about two hours on a shared CI host, against
    11 s on a GPU node), so this module runs with at most 8 threads and oneDNN off, as the non-deterministic KCP suite
    does."""
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

        return cake_package("cake_gdn_det_train_kcp", sub)
    except (ImportError, KeyError):
        return importlib.import_module(PACKAGE if sub is None else f"{PACKAGE}.{sub}")


opt = _mod("cake_options")
CakeGDNDetKcpOptions, make_options = opt.CakeGDNDetKcpOptions, opt.make_options


# ----------------------------------------------------------------------------------------------------------------
# options validator (the contract's option vocabulary)


def test_options_defaults_are_the_shipped_behaviour() -> None:
    o = make_options()
    assert o.exchange == "a2a"
    assert o.save_transfer is True
    assert o.save_mode == "light"
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
    for f in dataclasses.fields(CakeGDNDetKcpOptions):
        assert f.kw_only, f.name
    with pytest.raises(TypeError):
        CakeGDNDetKcpOptions("ring")  # type: ignore[misc]


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
        with pytest.raises((ValueError, TypeError)):
            make_options(**{field: bad})


def test_save_mode_refuses_the_non_deterministic_light_akk() -> None:
    # ``light_akk`` is a save mode of the non-deterministic package only; the deterministic options refuse it
    assert "light_akk" not in _mod("cake_options").SAVE_MODES
    with pytest.raises(ValueError, match="save_mode must be one of"):
        make_options(save_mode="light_akk")
    with pytest.raises(ValueError, match="save_mode must be one of"):
        make_options({"save_mode": "light_akk"})


def test_reuse_recompute_tapes_defaults_to_auto_never_captured() -> None:
    """The checkpoint-aware light backward is a keyword-only switch -- ``"auto"`` (the shipped default: the tapes are
    kept only when the node closes the checkpointed region), ``True`` (always) or ``False`` (the replay form); under
    ``graph="replay"`` the default resolves to ``False`` -- part of the options key (so of the CUDA-graph key), typed,
    and refused with ``graph="replay"`` (a captured graph bakes the arena addresses of its tapes; the kept tapes are
    fresh tensors handed from the recompute forward to the backward)."""
    assert "reuse_recompute_tapes" in opt.OPTION_NAMES
    # the shipped default is this one constant (and this line) -- every other test names the state it exercises
    assert opt.REUSE_RECOMPUTE_TAPES_DEFAULT == "auto"
    assert make_options().reuse_recompute_tapes == opt.REUSE_RECOMPUTE_TAPES_DEFAULT
    # None = the shipped default; under graph="replay" (the one mode that excludes the kept tapes) it is False
    assert make_options(reuse_recompute_tapes=None).reuse_recompute_tapes == "auto"
    assert make_options(graph="replay").reuse_recompute_tapes is False
    both = make_options(reuse_recompute_tapes=None, graph="replay")
    assert both.reuse_recompute_tapes is False
    assert CakeGDNDetKcpOptions(graph="replay").reuse_recompute_tapes is False
    assert make_options(graph="replay").key() != make_options().key()
    on = make_options(reuse_recompute_tapes=True)
    off = make_options(reuse_recompute_tapes=False)
    assert on.reuse_recompute_tapes is True and off.reuse_recompute_tapes is False
    assert on.key() != off.key()
    from_instance = make_options(CakeGDNDetKcpOptions(reuse_recompute_tapes=True))
    assert from_instance.reuse_recompute_tapes is True
    with pytest.raises(TypeError):
        make_options(reuse_recompute_tapes=1)
    with pytest.raises(ValueError, match="graph"):
        make_options(reuse_recompute_tapes=True, graph="replay")
    with pytest.raises(ValueError, match="graph"):
        make_options(reuse_recompute_tapes="auto", graph="replay")
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
    of the graph mode (a captured graph never skips, the option value stays ``"auto"``)."""
    assert "recompute_output" in opt.OPTION_NAMES
    assert opt.RECOMPUTE_OUTPUTS == ("auto", "always")
    assert opt.RECOMPUTE_OUTPUT_DEFAULT == "auto"
    assert make_options().recompute_output == "auto"
    assert make_options(recompute_output="always").recompute_output == "always"
    assert (
        make_options(CakeGDNDetKcpOptions(recompute_output="always")).recompute_output
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
    ``kernel`` is the host module's own ``CakeGDNDetKcpCall.kernel`` (the deferral under test)."""
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
        return lambda: kcp.CakeGDNDetKcpCall.kernel(call, name, 1)

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
    a = make_options(CakeGDNDetKcpOptions(save_mode="full"), boundary_mode="replay")
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
        (kcp.cake_gdn_det_kcp, "ctx"),
        (kcp.cake_gdn_det_train_kcp_forward, "ctx"),
        (kcp.cake_gdn_det_train_kcp_backward, "dout"),
    ):
        sig = inspect.signature(fn)
        for p in _after(sig, anchor):
            assert p.kind is p.KEYWORD_ONLY, (fn.__name__, p.name)
    sig = inspect.signature(kcp.cake_gdn_det_kcp)
    assert sig.parameters["save_mode"].default == "light"
    assert sig.parameters["boundary_mode"].default == "saved"
    assert sig.parameters["graph"].default == "off"
    assert sig.parameters["scale"].default is None


def test_facade_exposes_the_contract_names_keyword_only() -> None:
    facade = importlib.import_module("flashinfer.cake_gdn_det_train_kcp")
    names = (
        "cake_gdn_det_train_kcp_partition",
        "cake_gdn_det_train_kcp_build_context",
        "cake_gdn_det_kcp",
        "cake_gdn_det_train_kcp_forward",
        "cake_gdn_det_train_kcp_backward",
        "cake_gdn_det_train_kcp_workspace_bytes",
        "cake_gdn_det_train_kcp_saved_bytes",
        "cake_gdn_det_train_kcp_comm_bytes",
        "is_cake_gdn_det_train_kcp_supported",
        "cake_gdn_det_train_kcp_packed_qkv_views",
    )
    for n in names:
        assert callable(getattr(facade, n)), n
        assert n in facade.__all__
    # exactly the contract names plus the public types (the saved set and workspace the forward returns and the
    # autograd Function included); the constant DEFAULT_SCALE is bound but not exported
    types = (
        "CakeGDNDetKcpOptions",
        "CakeGDNDetKcpContext",
        "CakeGDNDetKcpPartition",
        "CakeGDNDetKcpRecomputeHandle",
        "CakeGDNDetKcpSaved",
        "CakeGDNDetKcpWorkspace",
        "CakeGDNDetKcpFunction",
    )
    assert set(facade.__all__) == set(names) | set(types)
    assert "DEFAULT_SCALE" not in facade.__all__
    assert set(types) <= set(dir(facade))
    for fn, anchor in (
        (facade.cake_gdn_det_kcp, "ctx"),
        (facade.cake_gdn_det_train_kcp_forward, "ctx"),
        (facade.cake_gdn_det_train_kcp_backward, "dout"),
    ):
        for p in _after(inspect.signature(fn), anchor):
            assert p.kind is p.KEYWORD_ONLY, (fn.__name__, p.name)
    for fn in (
        facade.cake_gdn_det_train_kcp_workspace_bytes,
        facade.cake_gdn_det_train_kcp_saved_bytes,
        facade.cake_gdn_det_train_kcp_comm_bytes,
    ):
        for p in _after(inspect.signature(fn), "ctx"):
            assert p.kind is p.KEYWORD_ONLY, (fn.__name__, p.name)
    # every public function of the facade module is a real def decorated with ``@flashinfer_api`` (the support query
    # too); the decorator returns the function unchanged while logging is off, so the check reads the source
    tree = ast.parse(pathlib.Path(facade.__file__).read_text())
    decorators = {
        node.name: [ast.unparse(d) for d in node.decorator_list]
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }
    assert sorted(decorators) == sorted(names)
    # under it the entries of OPTION_KEYWORD_ENTRIES carry the facade's ``_forwards_keywords``, which declares the
    # keywords they hand to the package (``test_option_keyword_entries_declare_every_option_field``)
    for name, found in decorators.items():
        want = ["flashinfer_api"]
        if name in OPTION_KEYWORD_ENTRIES:
            want.append("_forwards_keywords")
        assert found == want, (name, found)
    # a stable API: no entry is marked experimental
    for n in names:
        assert getattr(getattr(facade, n), "is_experimental", False) is False, n
        assert not hasattr(getattr(facade, n), "experimental_feature"), n
    assert not hasattr(facade, "_FEATURE")
    pkg = _mod()
    assert facade.CakeGDNDetKcpOptions is pkg.CakeGDNDetKcpOptions
    assert facade.CakeGDNDetKcpContext is pkg.CakeGDNDetKcpContext
    assert facade.CakeGDNDetKcpPartition is pkg.CakeGDNDetKcpPartition
    assert facade.CakeGDNDetKcpRecomputeHandle is pkg.CakeGDNDetKcpRecomputeHandle
    assert facade.CakeGDNDetKcpSaved is pkg.CakeGDNDetKcpSaved
    assert facade.CakeGDNDetKcpWorkspace is pkg.CakeGDNDetKcpWorkspace
    assert facade.CakeGDNDetKcpFunction is pkg.CakeGDNDetKcpFunction
    assert facade.DEFAULT_SCALE == 128**-0.5 == _mod("cake_kcp").SCALE
    with pytest.raises(AttributeError):
        facade.no_such_name  # noqa: B018


def test_facade_entries_raise_no_experimental_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warnings  # noqa: PLC0415

    from flashinfer.api_logging import ExperimentalWarning  # noqa: PLC0415

    facade = importlib.import_module("flashinfer.cake_gdn_det_train_kcp")
    pkg = _mod()
    monkeypatch.setattr(
        pkg, "cake_gdn_det_train_kcp_comm_bytes", lambda ctx, **kw: {"ran": ctx}
    )
    monkeypatch.setattr(
        pkg, "cake_gdn_det_train_kcp_packed_qkv_views", lambda qkv, **kw: ("ran", qkv)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        ctx = object()
        assert facade.cake_gdn_det_train_kcp_comm_bytes(ctx, H_qk=16, HV=48) == {
            "ran": ctx
        }
        assert facade.cake_gdn_det_train_kcp_packed_qkv_views(
            "qkv", H_qk=16, HV=48
        ) == ("ran", "qkv")


def test_package_init_re_exports_the_route() -> None:
    pkg = _mod()
    for n in (
        "cake_gdn_det_train_kcp_partition",
        "cake_gdn_det_train_kcp_build_context",
        "cake_gdn_det_kcp",
        "cake_gdn_det_train_kcp_forward",
        "cake_gdn_det_train_kcp_backward",
        "cake_gdn_det_train_kcp_comm_bytes",
        "cake_gdn_det_train_kcp_saved_bytes",
        "cake_gdn_det_train_kcp_workspace_bytes",
        "cake_gdn_det_train_kcp_packed_qkv_views",
        "CakeGDNDetKcpOptions",
        "CakeGDNDetKcpContext",
        "CakeGDNDetKcpSaved",
        "CakeGDNDetKcpWorkspace",
    ):
        assert hasattr(pkg, n), n
        assert n in pkg.__all__


# The pre-rename names, removed without aliases: the legacy modules and every legacy spelling of a det object.
LEGACY_SUBMODULES = (
    "cake_backend",
    "cake_build",
    "cake_bytes",
    "cake_fake_backend",
    "cake_jit",
    "cake_kcp",
    "cake_kcp_plan",
    "cake_options",
    "cake_tables",
)
LEGACY_MODULES = (
    "flashinfer.gdn_train",
    "flashinfer.experimental.cake_gdn_train",
    *(f"flashinfer.experimental.cake_gdn_train.{sub}" for sub in LEGACY_SUBMODULES),
)
LEGACY_NAMES = (
    "kcp_partition",
    "build_kcp_context",
    "chunk_gdn_kcp",
    "chunk_gdn_kcp_fwd",
    "chunk_gdn_kcp_bwd",
    "gdn_kcp_workspace_bytes",
    "gdn_kcp_saved_bytes",
    "gdn_kcp_comm_bytes",
    "gdn_kcp_saved_layout",
    "gdn_kcp_workspace_layout",
    "is_gdn_train_supported",
    "KcpOptions",
    "KcpContext",
    "KcpPartition",
    "KcpRecomputeHandle",
    "KcpSaved",
    "KcpWorkspace",
    "KcpCall",
    "KcpFragment",
    "ChunkGDNKcpFunction",
)
# The package-level names without ``det``: they stay in their modules and are not exported by the package.
DET_FREE_NAMES = (
    "KCP_SAVED_VERSION",
    "LAST_CALL",
    "PROGRAMS",
    "SCALE",
    "Backend",
    "FakeBackend",
    "KernelBackend",
    "get_backend",
    "graph_cache_clear",
    "make_options",
    "set_backend",
    "validate_heads",
    "validate_ring",
)


# The entries that take the option keywords as ``**knobs`` / ``**kw``.  Their signatures declare the call's keywords,
# with no ``**`` parameter (``test_option_keyword_entries_declare_every_option_field``), so a stray ``deterministic=``
# fails ``inspect.signature(...).bind_partial`` as it fails the call, which refuses it with ``TypeError``.
OPTION_KEYWORD_ENTRIES = frozenset(
    {
        "cake_gdn_det_kcp",
        "cake_gdn_det_train_kcp_forward",
        "cake_gdn_det_train_kcp_backward",
        "cake_gdn_det_train_kcp_workspace_bytes",
        "cake_gdn_det_train_kcp_saved_bytes",
        "cake_gdn_det_train_kcp_comm_bytes",
    }
)


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
    :class:`CakeGDNDetKcpOptions` field (the field's default where the def does not name it); the facade declares
    the package's keywords.  A byte query of the package is the byte module's function of its name, and the facade
    declares the keyword-only parameters of that function (its own ``C`` default: 10240 channels)."""
    facade = importlib.import_module("flashinfer.cake_gdn_det_train_kcp")
    pkg, kcp, nbytes = _mod(), _mod("cake_kcp"), _mod("cake_bytes")
    fields = {f.name: f.default for f in dataclasses.fields(CakeGDNDetKcpOptions)}
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
    comm = inspect.signature(facade.cake_gdn_det_train_kcp_comm_bytes).parameters
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
        assert raw.__name__ == name and raw.__module__ == facade.__name__, name
        assert f"def {name}(" in inspect.getsource(fn), name
        assert "return" in typing.get_type_hints(fn), name


def test_importing_the_facade_does_not_import_the_package() -> None:
    """A fresh interpreter imports the facade without importing the backend package; the package is imported on the
    first read of the parameters of an entry that hands its option keywords to the package (``inspect.signature``
    alone imports nothing)."""
    code = (
        "import inspect\n"
        "import sys\n"
        "import flashinfer.cake_gdn_det_train_kcp as facade\n"
        f"assert {PACKAGE!r} not in sys.modules, 'imported at facade import'\n"
        "sig = inspect.signature(facade.cake_gdn_det_kcp)\n"
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


def test_legacy_names_are_removed() -> None:
    """``flashinfer.gdn_train`` and ``flashinfer.experimental.cake_gdn_train`` (and its modules) do not import, and no
    module of the deterministic route binds a pre-rename name.  A checkout may keep an untracked leftover of a former
    package directory (``aot/``, ``__pycache__``), which Python imports as an empty namespace package: such a leftover
    holds no module and none of the former names."""
    for name in LEGACY_MODULES:
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:  # a parent path does not import
            spec = None
        if (
            spec is not None
            and spec.origin in (None, "namespace")
            and spec.submodule_search_locations
        ):
            leftover = importlib.import_module(name)
            for old in LEGACY_NAMES:
                assert not hasattr(leftover, old), (name, old)
            for sub in LEGACY_SUBMODULES:
                with pytest.raises(ModuleNotFoundError):
                    importlib.import_module(f"{name}.{sub}")
            continue
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)
    facade = importlib.import_module("flashinfer.cake_gdn_det_train_kcp")
    modules = [facade, _mod()] + [_mod(sub) for sub in (*LEGACY_SUBMODULES, "cake_l2n")]
    for module in modules:
        for old in LEGACY_NAMES:
            assert not hasattr(module, old), (module.__name__, old)


def test_package_exports_the_det_names_only() -> None:
    """The package's ``__all__`` holds the names that carry ``det`` only, each bound; the constants and backend
    controls without ``det`` are neither exported nor bound by the package, and are read from their modules."""
    from tests.cake.test_helpers.cake_mode_names import assert_mode_names  # noqa: PLC0415

    pkg = _mod()
    assert pkg.__all__
    assert_mode_names(pkg, "det")
    assert all(hasattr(pkg, n) for n in pkg.__all__)
    holders = (_mod("cake_kcp"), _mod("cake_options"))
    for n in DET_FREE_NAMES:
        assert n not in pkg.__all__ and not hasattr(pkg, n), n
        assert any(hasattr(m, n) for m in holders), n


def test_det_entries_take_no_deterministic_keyword() -> None:
    """The det entries of the facade and the package take no ``deterministic`` keyword: any value is a
    ``TypeError``; the same call without it runs (fake backend, CPU)."""
    facade = importlib.import_module("flashinfer.cake_gdn_det_train_kcp")
    pkg = _mod()
    cpu = torch.device("cpu")
    cu = torch.tensor([0, 300], dtype=torch.int64)
    part = pkg.cake_gdn_det_train_kcp_partition(cu, 1)
    ctx = pkg.cake_gdn_det_train_kcp_build_context(part, None, 0, cpu)
    T, H_qk, HV = 300, 1, 1
    q = torch.zeros(T, H_qk, 128, dtype=torch.bfloat16)
    v = torch.zeros(T, HV, 128, dtype=torch.bfloat16)
    g = torch.zeros(T, HV)
    beta = torch.full((T, HV), 0.5)
    qkv = torch.zeros(T, (2 * H_qk + HV) * 128, dtype=torch.bfloat16)
    calls = {
        "cake_gdn_det_train_kcp_partition": lambda f, **kw: f(cu, 1, **kw),
        "cake_gdn_det_train_kcp_build_context": lambda f, **kw: f(
            part, None, 0, cpu, **kw
        ),
        "cake_gdn_det_kcp": lambda f, **kw: f(
            q, q, v, g, beta, ctx, backend="fake", **kw
        ),
        "cake_gdn_det_train_kcp_forward": lambda f, **kw: f(
            q, q, v, g, beta, ctx, backend="fake", **kw
        ),
        "cake_gdn_det_train_kcp_backward": lambda f, **kw: f(
            q, q, v, g, beta, ctx, v, backend="fake", **kw
        ),
        "cake_gdn_det_train_kcp_workspace_bytes": lambda f, **kw: f(
            ctx, H_qk=H_qk, HV=HV, **kw
        ),
        "cake_gdn_det_train_kcp_saved_bytes": lambda f, **kw: f(
            ctx, H_qk=H_qk, HV=HV, **kw
        ),
        "cake_gdn_det_train_kcp_comm_bytes": lambda f, **kw: f(
            ctx, H_qk=H_qk, HV=HV, **kw
        ),
        "cake_gdn_det_train_kcp_packed_qkv_views": lambda f, **kw: f(
            qkv, H_qk=H_qk, HV=HV, **kw
        ),
    }
    for name, call in calls.items():
        for fn in (getattr(facade, name), getattr(pkg, name)):
            assert "deterministic" not in inspect.signature(fn).parameters, name
            call(fn)
            for value in (True, False, None):
                with pytest.raises(TypeError, match="deterministic"):
                    call(fn, deterministic=value)
    fn = facade.is_cake_gdn_det_train_kcp_supported
    assert "deterministic" not in inspect.signature(fn).parameters
    with pytest.raises(TypeError, match="deterministic"):
        fn(None, deterministic=True)


# ----------------------------------------------------------------------------------------------------------------
# package hygiene (no KDA package import; no os.environ; the host module's import allow-list)


def _package_sources() -> list[pathlib.Path]:
    root = pathlib.Path(_mod().__file__).resolve().parent
    files = sorted(root.glob("*.py"))
    assert files
    return files


def _facade_source() -> pathlib.Path:
    return pathlib.Path(
        importlib.import_module("flashinfer.cake_gdn_det_train_kcp").__file__
    ).resolve()


def test_package_never_imports_a_kda_package() -> None:
    sources = _package_sources() + [_facade_source()]
    # the package modules and the facade (not vacuous)
    assert len(sources) > 10, sources
    seen = 0
    for path in sources:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                seen += 1
                for a in node.names:
                    assert "cake_kda" not in a.name, (path.name, a.name)
            elif isinstance(node, ast.ImportFrom):
                seen += 1
                assert "cake_kda" not in (node.module or ""), (
                    path.name,
                    node.module,
                )
                for a in node.names:
                    assert "cake_kda" not in a.name, (path.name, a.name)
    assert seen > 50, seen  # the scan sees the modules' import statements


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
    """The host module and the options module import the standard library's typing and introspection helpers
    (``inspect``: the declared signatures of the option-keyword entries), torch, torch.distributed and their own
    package only (relative imports); the registry / emulation are imported lazily inside functions."""
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
        "kcp_transfer_walk",
        "kcp_affine_merge",
        "fwd_o",
        "bwd_prep",
        "kcp_dhu_offset",
        "kcp_dhu_tape",
        "bwd_dqkg",
        "gate_epilogue",
        "fwd_prep_l2n",
        "bwd_dqkg_l2n",
        "qk_apply",
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
    # the output walk: a det-neutral module (the non-deterministic package ships the same program)
    assert kcp.TART_MODULE_OF["kcp_fwdh_out"] == "gdn_chunk_fwdh_out_v2_kcp"
    for name, module in kcp.TART_MODULE_OF.items():
        if name not in ("kcp_affine_merge", "kcp_fwdh_out"):
            # the deterministic family's stems
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
    record is provided by the host (skipped until ``cake_jit`` is in the package)."""
    kcp = _mod("cake_kcp")
    try:
        jit = _mod("cake_jit")
    except ImportError:
        pytest.skip("cake_jit not in the package yet")
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
    vh = 128; the test knob ``fwdh_vh`` forces a form (both forms are bitwise twins)."""
    kcp = _mod("cake_kcp")
    assert kcp.FWDH_PROGRAMS == ("kcp_fwdh_offset", "kcp_fwdh_tape", "kcp_fwdh_out")
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
    try:
        jit = _mod("cake_jit")
    except ImportError:
        return
    if "kcp_fwdh_offset_vh64" in jit.MODULES:  # the registry of record ships both forms
        for prog in kcp.FWDH_PROGRAMS:
            assert jit.select_record(prog) == prog
            assert jit.select_record(prog, ctas_per_head=2) == prog + kcp.VH64_SUFFIX
            assert jit.launch_grid(prog, n_walk=3, num_heads=48) == (144, 1, 1)
            assert jit.launch_grid(
                prog + kcp.VH64_SUFFIX, n_walk=3, num_heads=48, ctas_per_head=2
            ) == (288, 1, 1)
            for fits, vh in ((False, 128), (True, 64)):
                args = {
                    "_grid_args": {"n_walk": 4, "num_heads": 48},
                    "_sm_count": 152,
                    "_fwdh_vh": None,
                    "_vh64_fits": fits,
                    "_grid": 4 * 48,
                }
                record, _, grid, _ = kcp.resolve_launch(prog, dict(args))
                assert record == (prog + kcp.VH64_SUFFIX if vh == 64 else prog)
                assert grid == (4 * 48 * (2 if vh == 64 else 1), 1, 1)


# A pinned subset of the walk-width test vectors (schema r7c-walk-width-test-vectors-v1, sha256 08954080...: every
# walk list of the plan census with the decisions of the rule): per SM count (148 sm_100a, 152 sm_103a, 208 / 212 /
# 216 sm_107a) two lists of every class (one wave or not, chain clause fits or not, form changed by the rule or
# not).  Columns: vector id, SM count, architecture, HV, n_walk, chains (``chunks:count``), fits, #2 vh, #7 offset vh,
# #7 tape vh.
WALK_WIDTH_VECTORS = (
    "0 148 sm_100a 3 1 1:1 0 64 64 64",
    "1 148 sm_100a 3 1 2:1 0 64 64 64",
    "3774 148 sm_100a 24 8 244:4,113:4 1 64 64 64",
    "3896 148 sm_100a 24 18 158:1,157:2,42:4,41:2,29:3,19:3,14:3 1 64 64 64",
    "4466 148 sm_100a 48 4 89:4 0 128 64 64",
    "4470 148 sm_100a 48 4 96:4 0 128 64 64",
    "4475 148 sm_100a 48 4 118:2,28:2 1 64 64 64",
    "4681 148 sm_100a 48 7 286:1,140:1,119:1,98:1,78:1,57:1,23:1 1 64 64 64",
    "4723 148 sm_100a 48 8 188:8 0 128 64 64",
    "4726 148 sm_100a 48 8 210:8 0 128 64 64",
    "4857 152 sm_103a 3 1 1:1 0 64 64 64",
    "4858 152 sm_103a 3 1 2:1 0 64 64 64",
    "8638 152 sm_103a 24 8 244:4,113:4 1 64 64 64",
    "8756 152 sm_103a 24 18 158:1,157:2,42:4,41:2,29:3,19:3,14:3 1 64 64 64",
    "9334 152 sm_103a 48 4 89:4 0 128 64 64",
    "9342 152 sm_103a 48 4 118:2,28:2 1 64 64 64",
    "9347 152 sm_103a 48 4 128:4 0 128 64 64",
    "9541 152 sm_103a 48 7 286:1,140:1,119:1,98:1,78:1,57:1,23:1 1 64 64 64",
    "9599 152 sm_103a 48 8 256:8 0 128 64 64",
    "9615 152 sm_103a 48 10 90:10 0 128 64 64",
    "9730 208 sm_107a 3 1 1:1 0 64 64 64",
    "9731 208 sm_107a 3 1 2:1 0 64 64 64",
    "14480 208 sm_107a 48 6 128:3,43:2,42:1 1 64 64 64",
    "14506 208 sm_107a 48 6 176:6 0 128 64 64",
    "14517 208 sm_107a 48 6 249:6 0 128 64 64",
    "14518 208 sm_107a 48 6 253:3,6:2,5:1 1 64 64 64",
    "14647 208 sm_107a 48 12 2:9,1:3 0 128 64 128",
    "14652 208 sm_107a 48 12 40:4,39:8 0 128 64 128",
    "14656 208 sm_107a 48 12 63:1,58:1,18:2,17:2,12:2,8:2,6:2 1 64 64 128",
    "14657 208 sm_107a 48 12 63:1,61:1,17:1,16:3,12:2,8:2,6:2 1 64 64 128",
    "14768 212 sm_107a 3 1 1:1 0 64 64 64",
    "14769 212 sm_107a 3 1 2:1 0 64 64 64",
    "19515 212 sm_107a 48 6 128:3,43:2,42:1 1 64 64 64",
    "19538 212 sm_107a 48 6 176:6 0 128 64 64",
    "19549 212 sm_107a 48 6 256:2,160:2,96:2 1 64 64 64",
    "19598 212 sm_107a 48 8 64:8 0 128 64 64",
    "19681 212 sm_107a 48 10 90:10 0 128 64 128",
    "19683 212 sm_107a 48 10 92:10 0 128 64 128",
    "19707 212 sm_107a 48 12 63:1,58:1,18:2,17:2,12:2,8:2,6:2 1 64 64 128",
    "19708 212 sm_107a 48 12 63:1,61:1,17:1,16:3,12:2,8:2,6:2 1 64 64 128",
    "19826 216 sm_107a 3 1 1:1 0 64 64 64",
    "19827 216 sm_107a 3 1 2:1 0 64 64 64",
    "24527 216 sm_107a 48 5 149:5 0 128 64 64",
    "24549 216 sm_107a 48 6 64:6 0 128 64 64",
    "24578 216 sm_107a 48 6 128:3,43:2,42:1 1 64 64 64",
    "24615 216 sm_107a 48 6 236:3,87:1,86:2 1 64 64 64",
    "24790 216 sm_107a 48 12 63:1,58:1,18:2,17:2,12:2,8:2,6:2 1 64 64 128",
    "24791 216 sm_107a 48 12 63:1,61:1,17:1,16:3,12:2,8:2,6:2 1 64 64 128",
    "24839 216 sm_107a 48 16 55:2,54:14 0 128 64 128",
    "24840 216 sm_107a 48 16 56:4,55:12 0 128 64 128",
)


def _walk_width_rows():
    for line in WALK_WIDTH_VECTORS:
        vid, sm, arch, hv, n_walk, chains, fits, fwdh, off, tape = line.split()
        yield (
            int(vid),
            int(sm),
            arch,
            int(hv),
            int(n_walk),
            tuple(tuple(int(x) for x in c.split(":")) for c in chains.split(",")),
            fits == "1",
            int(fwdh),
            int(off),
            int(tape),
        )


def test_walk_width_rules_follow_the_pinned_vectors() -> None:
    """The #2 / #7 width decisions of the pinned vector lists: ``fits`` = ``n_walk * HV > SM`` and
    :func:`walk_vh64_fits` of the chains (as ``CakeGDNDetKcpCall.walk`` evaluates it), ``fwdh_vh`` with that clause and
    ``dhu_vh`` of the offset and tape walks under the list's architecture; ``_Tables.walk_fits`` of a plan whose
    segments carry the same chains agrees."""
    kcp = _mod("cake_kcp")
    rows = list(_walk_width_rows())
    assert len(rows) == len(WALK_WIDTH_VECTORS) >= 40
    assert {r[2] for r in rows} == {"sm_100a", "sm_103a", "sm_107a"}
    for vid, sm, arch, hv, n_walk, chains, fits, fwdh, off, tape in rows:
        assert sum(n for _, n in chains) == n_walk, vid
        got_fits = n_walk * hv > sm and kcp.walk_vh64_fits(chains, hv, sm)
        assert got_fits is fits, vid
        assert kcp.fwdh_vh(n_walk, hv, sm, None, got_fits) == fwdh, vid
        assert kcp.dhu_vh(n_walk, hv, sm, tape=False, arch=arch) == off, vid
        assert kcp.dhu_vh(n_walk, hv, sm, tape=True, arch=arch) == tape, vid
        seg = [0]
        for c, n in chains:
            for _ in range(n):
                seg.append(seg[-1] + c)
        tb = types.SimpleNamespace(
            memo={}, hp=types.SimpleNamespace(seg_chunk_start=tuple(seg)), H=hv
        )
        tb._memo_put = types.MethodType(kcp._Tables._memo_put, tb)
        tb._walk_chains = types.MethodType(kcp._Tables._walk_chains, tb)
        want = kcp.walk_vh64_fits(chains, hv, sm)
        assert kcp._Tables.walk_fits(tb, list(range(n_walk)), sm) is want, vid


def _walk_form_ops(cu, H_qk: int, HV: int) -> dict[str, torch.Tensor]:
    """Seeded CPU operands (q / k BF16 L2-normalised rows, v / dout BF16, g FP32 log-gates, beta FP32)."""
    gen = torch.Generator(device="cpu").manual_seed(20261103)
    T = cu[-1]

    def l2(x):
        return (x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        )

    return {
        "q": l2(torch.randn(T, H_qk, 128, generator=gen)),
        "k": l2(torch.randn(T, H_qk, 128, generator=gen)),
        "v": torch.randn(T, HV, 128, generator=gen).to(torch.bfloat16),
        "g": (-torch.nn.functional.softplus(torch.randn(T, HV, generator=gen)))
        .float()
        .contiguous(),
        "beta": torch.sigmoid(torch.randn(T, HV, generator=gen)).float().contiguous(),
        "dout": torch.randn(T, HV, 128, generator=gen).to(torch.bfloat16),
    }


def test_forward_walk_form_follows_the_plan_chains() -> None:
    """Plan test of the #2 chain clause: one contiguous rank of four documents of 24 / 16 / 14 / 10 chunks at HV 6,
    one segment per document.  ``n_walk * HV = 24`` exceeds both SM counts below, so the one-wave clause alone takes
    vh = 128; at 20 SMs the 48 two-CTA items on 40 slots end at the longest chain (LPT makespan 24), so the forward's
    walk launches the ``_vh64`` record; at 12 SMs (makespan 34) it stays vh = 128.  The rule is the same for the output
    walk ``kcp_fwdh_out`` of the shipped forward-only call and for the tape walk of the split forward (test knob
    ``fused_o=0``).  The fake backend runs the forward the plan launches (the record list of ``LAST_CALL``)."""
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    cu = _cu((24 * 64, 16 * 64, 14 * 64, 10 * 64))
    ops = _walk_form_ops(cu, 2, 6)
    args = [ops[n] for n in ("q", "k", "v", "g", "beta")]
    for sm, want in ((20, True), (12, False)):
        ctx = plan.cake_gdn_det_train_kcp_build_context(
            plan.cake_gdn_det_train_kcp_partition(cu, 1, "contiguous"),
            None,
            0,
            "cpu",
            sm_count=sm,
        )
        tb = kcp._Tables(ctx, 6, P=1)
        assert len(tb.walk_live) == 4 and tb.walk_fits(tb.walk_live, sm) is want
        assert kcp.fwdh_vh(4, 6, sm) == 128
        for knobs, prog in (
            ({}, "kcp_fwdh_out"),  # the shipped forward-only call: the output walk
            ({"test_knobs": {"fused_o": 0}}, "kcp_fwdh_tape"),  # the split forward
        ):
            kcp.cake_gdn_det_train_kcp_forward(
                *args, ctx, backend="fake", segments=1, **knobs
            )
            walks = [
                r
                for r, _ in kcp.LAST_CALL["fwd"]["records"]
                if r.removesuffix(kcp.VH64_SUFFIX) == prog
            ]
            forms = {r.endswith(kcp.VH64_SUFFIX) for r in walks}
            assert walks and forms == {want}, (sm, prog, walks)


@pytest.mark.parametrize(("sm", "want"), [(20, True), (12, False)])
def test_walk_forms_agree_across_forward_recompute_and_handle(
    monkeypatch, sm, want
) -> None:
    """The width rules read host plan integers only, so the forward, torch's checkpoint recompute and the recompute
    from a :class:`CakeGDNDetKcpRecomputeHandle` launch the same form of every #2 and #7 walk program.  The rank of
    :func:`test_forward_walk_form_follows_the_plan_chains` (HV 6, one segment per document): at 20 SMs the chain
    clause takes the ``_vh64`` record of the forward walks, at 12 SMs vh = 128; the fake backend has no
    architecture, so #7 takes vh = 64.  Each path logs the records of every recorded call (``_record``)."""
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    cu = _cu((24 * 64, 16 * 64, 14 * 64, 10 * 64))
    ops = _walk_form_ops(cu, 2, 6)
    ctx = plan.cake_gdn_det_train_kcp_build_context(
        plan.cake_gdn_det_train_kcp_partition(cu, 1, "contiguous"),
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
        hd = kcp.CakeGDNDetKcpRecomputeHandle() if handle else None

        def fn(q, k, v, g, beta, h):
            return kcp.cake_gdn_det_kcp(
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


def test_long_chain_dhu_forms_agree_across_forward_recompute_and_handle_on_sm_100a(
    monkeypatch,
) -> None:
    """The three paths of :func:`test_walk_forms_agree_across_forward_recompute_and_handle` through the kernel
    backend's host path on an ``sm_100a`` device: a :class:`KernelBackend` whose stub launcher reports ``sm_100a``
    and runs every record through the torch emulation (no GPU).  With the long-chain threshold lowered to 24, the
    rank of :func:`test_dhu_long_chain_flag_follows_the_plan_chains` (24 / 10 x 5 chunks, HV 6) on 17 SMs is a
    long-chain #7 walk whose two-CTA grid ``2 * 6 * 6 = 72`` exceeds ``4 * 17 = 68``, so ``sm_100a``'s two-wave
    rule takes vh = 128 for every #7 walk of the forward's backward, of torch's checkpoint recompute and of a
    :class:`CakeGDNDetKcpRecomputeHandle` recompute (the fake backend, which has no architecture, takes vh = 64)."""
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    fake = _mod("cake_fake_backend")
    monkeypatch.setattr(kcp, "DHU_LONG_CHAIN_MIN_CHUNKS", 24)
    sm = 17
    cu = _cu((24 * 64,) + (10 * 64,) * 5)
    ops = _walk_form_ops(cu, 2, 6)
    ctx = plan.cake_gdn_det_train_kcp_build_context(
        plan.cake_gdn_det_train_kcp_partition(cu, 1, "contiguous"),
        None,
        0,
        "cpu",
        sm_count=sm,
    )
    tb = kcp._Tables(ctx, 6, P=1)
    n_walk = len(tb.walk_live)
    assert n_walk == 6 and tb.walk_long_chain(tb.walk_live, sm)
    assert kcp.dhu_vh(n_walk, 6, sm, arch="sm_100a", long_chain=True) == 128
    assert kcp.dhu_vh(n_walk, 6, sm, arch=None, long_chain=True) == 64

    class Launcher:
        @staticmethod
        def device_arch(index):
            return "sm_100a"

        @staticmethod
        def run_module(record, values, *, grid, device_index, module=None):
            fake.run_module(record, values, grid=grid, device_index=None)

    monkeypatch.setitem(kcp._BACKENDS, "fake", kcp.KernelBackend(launcher=Launcher()))
    # a CPU call's launch device index (KernelBackend.run asks torch for it; no CUDA here)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
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
        hd = kcp.CakeGDNDetKcpRecomputeHandle() if handle else None

        def fn(q, k, v, g, beta, h):
            return kcp.cake_gdn_det_kcp(
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
        assert paths[name]["kinds"].count("fwd") == 2, (name, paths[name]["kinds"])
    for name, p in paths.items():
        forms = p["forms"]
        dhu = [b for b in forms if b in kcp.DHU_PROGRAMS]
        assert "kcp_dhu_tape" in dhu, (name, forms)
        assert all(forms[b] == {False} for b in dhu), (name, forms)
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
        return plan.cake_gdn_det_train_kcp_partition(
            cu, W, fragments=_ragged(cu, t_r)
        ), cu
    return plan.cake_gdn_det_train_kcp_partition(cu, W, layout=layout), cu


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
        ctx = plan.cake_gdn_det_train_kcp_build_context(
            part, None, rank, "cpu", sm_count=148
        )
        ws = kcp.CakeGDNDetKcpWorkspace()
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
        assert kcp.CakeGDNDetKcpWorkspace().tables(ctx, HV) is tb
        assert ("kcp_tables", HV, 1, None) in ctx._caches
        ws2 = kcp.CakeGDNDetKcpWorkspace()
        ws2.ensure(ctx, H_qk, HV, device="cpu", backward=False)
        assert ws2.pool("eye") is ws.pool(
            "eye"
        )  # the identity tile is built once per context


def test_workspace_rejects_a_bad_head_pair_and_device_change() -> None:
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    part, _ = _partition((300, 200), 2, "contiguous")
    ctx = plan.cake_gdn_det_train_kcp_build_context(part, None, 0, "cpu", sm_count=148)
    ws = kcp.CakeGDNDetKcpWorkspace()
    with pytest.raises(ValueError):
        ws.ensure(ctx, 4, 6, device="cpu")
    ws.ensure(ctx, 2, 6, device="cpu")
    with pytest.raises(ValueError):
        ws.ensure(ctx, 2, 6, device="meta")


def test_exchange_scheme_ring_is_refused_beyond_two_hops() -> None:
    kcp, plan = _mod("cake_kcp"), _mod("cake_kcp_plan")
    part, _ = _partition((3000,), 4, "zigzag")  # one document over 8 pieces: 7 hops
    ctx = plan.cake_gdn_det_train_kcp_build_context(part, None, 0, "cpu", sm_count=148)
    tb = kcp.CakeGDNDetKcpWorkspace().ensure(ctx, 4, 12, device="cpu")
    with pytest.raises(ValueError):
        kcp._exchange_scheme(tb, make_options(exchange="ring", save_transfer=False))
    assert kcp._exchange_scheme(tb, make_options()) == "a2a"
    part2, _ = _partition(
        (100, 640, 77, 1000, 129), 2, "contiguous"
    )  # CP2 contiguous: at most 1 hop
    ctx2 = plan.cake_gdn_det_train_kcp_build_context(
        part2, None, 0, "cpu", sm_count=148
    )
    tb2 = kcp.CakeGDNDetKcpWorkspace().ensure(ctx2, 2, 6, device="cpu")
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
                assert len(a) == len(b), f.name
                assert all(x is y for x, y in zip(a, b, strict=True)), f.name
            else:
                assert a is b, f.name
        else:
            assert a == b, (f.name, a, b)


def test_saved_set_round_trips_through_tensors_and_meta() -> None:
    kcp = _mod("cake_kcp")
    S_in = torch.zeros(4)
    tapes = tuple(torch.zeros(1) for _ in kcp.FWD_TAPES)
    saved = kcp.CakeGDNDetKcpSaved(
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
    back = kcp.CakeGDNDetKcpSaved.from_tensors(
        saved.tensors(), saved.flags(), saved.meta(), None
    )
    _assert_same_saved(back, saved)
    assert back.tape("h_tape") is tapes[kcp.FWD_TAPES.index("h_tape")]
    assert saved.bytes == 4 * 4 + 4 * len(kcp.FWD_TAPES)
    light_tapes = tuple(torch.zeros(2) for _ in kcp.LIGHT_TAPES)  # empty
    light = dataclasses.replace(saved, tapes=light_tapes, save_mode="light")
    assert light.flags() == (False, False, True)
    assert len(light.tensors()) == 1 + len(kcp.LIGHT_TAPES) == 1
    back_light = kcp.CakeGDNDetKcpSaved.from_tensors(
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
    ctx = plan.cake_gdn_det_train_kcp_build_context(part, None, 0, "cpu", sm_count=148)
    H_qk, HV = 2, 6
    ws = kcp.CakeGDNDetKcpWorkspace()
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
            "fwd", ctx, kcp.CakeGDNDetKcpWorkspace(), backend, opts, None, ops, (True,)
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
    knob ``dhu_vh`` forces a form (both forms are bitwise twins).  ``resolve_launch`` names the record the registry
    provides: the ``_vh64`` record where the registry ships it, the full-head record with the grid ``n_walk * HV``
    where it does not (no error either way)."""
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
    try:
        jit = _mod("cake_jit")
    except ImportError:
        return
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
    keep their rules for the same walk.  (The det package has no launch cache, so resolve_launch alone carries the
    flag.)"""
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
    # resolve_launch reads the clause flag
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
    ops = _walk_form_ops(cu, 2, 6)
    ctx = plan.cake_gdn_det_train_kcp_build_context(
        plan.cake_gdn_det_train_kcp_partition(cu, 1, "contiguous"),
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
    out = kcp.cake_gdn_det_kcp(*leaves, ctx, backend="fake", segments=1)
    out.backward(ops["dout"])
    assert seen and all(isinstance(x, bool) for x in seen)
    assert any(seen) is (longest == 24), seen


@pytest.mark.parametrize(
    ("arch", "prog", "n_walk", "sm", "long_chain", "want"),
    [
        # sm_107a: the tape walk's two-wave rule, 2 * 12 * 48 = 1152 > 4 * 208 -> vh 128
        ("sm_107a", "kcp_dhu_tape", 12, 208, False, 128),
        ("sm_107a", "kcp_dhu_offset", 12, 208, False, 64),
        # sm_100a: a long-chain walk takes the two-wave rule, 1152 > 4 * 148 -> vh 128
        ("sm_100a", "kcp_dhu_tape", 12, 148, True, 128),
        ("sm_100a", "kcp_dhu_offset", 12, 148, True, 128),
        ("sm_100a", "kcp_dhu_tape", 12, 148, False, 64),
        # sm_103a: the tape walk runs vh 64 at every load, long chain or not
        ("sm_103a", "kcp_dhu_tape", 12, 152, True, 64),
    ],
)
def test_kernel_backend_dhu_launch_reads_the_device_architecture(
    arch, prog, n_walk, sm, long_chain, want
) -> None:
    """:meth:`KernelBackend.run` reads the architecture of the launch's device (``launcher().device_arch``) for a #7
    launch, so the record it runs is the form :func:`dhu_vh` picks for that architecture.  A stub launcher stands in
    for the launch glue (no GPU): its ``device_arch`` reports ``arch`` and its ``run_module`` records the record name
    and grid of the launch."""
    kcp, jit = _mod("cake_kcp"), _mod("cake_jit")
    assert all((p + kcp.VH64_SUFFIX) in jit.MODULES for p in kcp.DHU_PROGRAMS)
    asked: list = []
    runs: list = []

    class Launcher:
        @staticmethod
        def device_arch(index):
            asked.append(index)
            return arch

        @staticmethod
        def run_module(record, values, *, grid, device_index, module=None):
            runs.append((record, grid, device_index, module))

    tape = prog == "kcp_dhu_tape"
    assert (
        kcp.dhu_vh(n_walk, 48, sm, tape=tape, arch=arch, long_chain=long_chain) == want
    )
    call = types.SimpleNamespace(device=torch.device("cuda", 0), records=[])
    args = {
        "_call": call,
        "_grid_args": {"n_walk": n_walk, "num_heads": 48},
        "_sm_count": sm,
        "_dhu_vh": None,
        "_dhu_long_chain": long_chain,
        "_grid": n_walk * 48,
    }
    kcp.KernelBackend(launcher=Launcher()).run(prog, args)
    record = prog + kcp.VH64_SUFFIX if want == 64 else prog
    grid = (n_walk * 48 * (2 if want == 64 else 1), 1, 1)
    assert asked == [0]
    assert runs == [(record, grid, 0, None)], (arch, prog, runs)
    assert call.records == [(record, grid)]


# The #7 long-chain clause rows of test_vectors_v3.json (R7C, schema r7c-walk-width-test-vectors-v2): every row with
# dhu_long_chain set plus rows with a segment of at least 400 chunks that are not long-chain walks, one or two per SM
# count.  Fields: id, SM, arch, HV, n_walk, chains (chunks:count, descending), dhu_long_chain, dhu_offset_vh, dhu_tape_vh.
DHU_LONG_CHAIN_VECTORS = (
    "4305 148 sm_100a 48 2 2048:1,2003:1 0 64 64",
    "4435 148 sm_100a 48 3 434:1,342:1,250:1 0 64 64",
    "4832 148 sm_100a 48 12 521:1,472:1,131:1,125:2,124:1,88:1,87:1,60:1,57:1,42:1,40:1 1 128 128",
    "4833 148 sm_100a 48 12 536:1,451:1,147:1,138:1,126:2,96:1,90:1,61:1,59:1,44:1,43:1 1 128 128",
    "8899 152 sm_103a 48 1 2003:1 0 64 64",
    "8900 152 sm_103a 48 1 2025:1 0 64 64",
    "9679 152 sm_103a 48 12 521:1,472:1,131:1,125:2,124:1,88:1,87:1,60:1,57:1,42:1,40:1 1 64 64",
    "9680 152 sm_103a 48 12 536:1,451:1,147:1,138:1,126:2,96:1,90:1,61:1,59:1,44:1,43:1 1 64 64",
    "14082 208 sm_107a 48 2 2048:1,2003:1 0 64 64",
    "14654 208 sm_107a 48 8 497:1,450:1,247:1,235:1,159:1,150:1,117:2 1 64 64",
    "19133 212 sm_107a 48 2 2048:1,2003:1 0 64 64",
    "24232 216 sm_107a 48 2 2048:1,2003:1 0 64 64",
)


def test_dhu_long_chain_rule_follows_the_pinned_vectors() -> None:
    """The #7 long-chain flag (:func:`walk_dhu_long_chain`) and the #7 forms of the pinned v3 vector rows: the flag
    of the chains, ``_Tables.walk_long_chain`` of a plan whose segments carry the same chains, and ``dhu_vh`` of the
    offset and tape walks with that flag under the row's architecture."""
    kcp = _mod("cake_kcp")
    rows = [line.split() for line in DHU_LONG_CHAIN_VECTORS]
    assert len(rows) >= 10
    assert sum(r[6] == "1" for r in rows) == 5
    for vid, sm, arch, hv, n_walk, chains, long_chain, off, tape in rows:
        sm, hv, n_walk = int(sm), int(hv), int(n_walk)
        ch = tuple(tuple(int(x) for x in c.split(":")) for c in chains.split(","))
        assert sum(n for _, n in ch) == n_walk, vid
        flag = kcp.walk_dhu_long_chain(ch, hv, sm)
        assert flag is (long_chain == "1"), vid
        seg = [0]
        for c, n in ch:
            for _ in range(n):
                seg.append(seg[-1] + c)
        tb = types.SimpleNamespace(
            memo={}, hp=types.SimpleNamespace(seg_chunk_start=tuple(seg)), H=hv
        )
        tb._memo_put = types.MethodType(kcp._Tables._memo_put, tb)
        tb._walk_chains = types.MethodType(kcp._Tables._walk_chains, tb)
        assert kcp._Tables.walk_long_chain(tb, list(range(n_walk)), sm) is flag, vid
        got_off = kcp.dhu_vh(n_walk, hv, sm, tape=False, arch=arch, long_chain=flag)
        got_tape = kcp.dhu_vh(n_walk, hv, sm, tape=True, arch=arch, long_chain=flag)
        assert (got_off, got_tape) == (int(off), int(tape)), vid


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
    ctx = plan.cake_gdn_det_train_kcp_build_context(part, None, 0, "cpu", sm_count=148)
    H_qk, HV = 2, 6
    ws = kcp.CakeGDNDetKcpWorkspace()
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
    other = kcp.CakeGDNDetKcpWorkspace()
    other.ensure(ctx, H_qk, HV, device="cpu", backward=True)
    assert other.ticket("fwd_o").data_ptr() != ws.ticket("fwd_o").data_ptr()
    # the public query counts every site's slot in both lifetimes
    layout = gb.cake_gdn_det_train_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV)
    assert layout["fwd.tickets"] == layout["bwd.tickets"] == gb.TICKET_BYTES
    assert (
        gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, backward=False
        )["fwd.tickets"]
        == gb.TICKET_BYTES
    )
