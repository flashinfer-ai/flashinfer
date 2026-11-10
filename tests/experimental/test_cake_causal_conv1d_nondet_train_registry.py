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

The single-rank registry (``cake_jit``) of the non-deterministic causal convolution training package against the
deterministic package's: the forward records are the deterministic forward programs (only their names differ),
every backward route keeps the deterministic route's programs and selection (but where a channel-band program takes
the lowest channel counts of its twin, the program of the route it shares a body with), each backward program is ONE
ordinary launch (no reduce program) that owns an ``accumulators`` workspace (FP32, or FP64 on the ``_fp32g``
routes; ``[C, 4]`` taps then ``[C]`` bias words; uint32 ticket counters per channel tile), and the workspace byte
layout and size depend on ``C`` and the route only.  The public names (each carries the mode word ``nondet``; no
entry point takes a mode keyword), and the package's wiring (test-helper key, package data, the docstring-only
``flashinfer.experimental.cake``).
No device is needed; nothing is built or launched.  The ``build.json`` checks skip until the exporter renders the
package.
"""

from __future__ import annotations

import ast
import filecmp
import hashlib
import importlib.util
import inspect
import os
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

KEY = "cake_causal_conv1d_nondet_train"
pkg = cake_package(KEY)
cake_build = cake_package(KEY, "cake_build")
cake_jit = cake_package(KEY, "cake_jit")
kcp_jit = cake_package(KEY, "kcp_jit")
det_build = cake_package("cake_causal_conv1d_det_train", "cake_build")
det_jit = cake_package("cake_causal_conv1d_det_train", "cake_jit")

PACKAGE_DIR = Path(cake_jit.__file__).parent
REPO = Path(__file__).resolve().parents[2]
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
NAME_FIELDS = ("kernel", "kernel_symbol", "library")
# A placeholder name until the exporter renders the package, then the identity rule's prefix + 20 hex digits.
PROGRAM_NAME = re.compile(
    r"cake_causal_conv1d_train_(?:TODOEXPORT_[a-z0-9_]+|[0-9a-f]{20})"
)
CHANNELS = (
    8,
    128,
    136,
    256,
    264,
    384,
    504,
    512,
    768,
    1024,
    1032,
    1160,
    1280,
    1344,
    1536,
    1600,
    1920,
    2048,
    2560,
    4096,
    8192,
)
# 262143 / 262144: both sides of the sm_107a token-floored backward program's floor.
TOKENS = (1, 1000, 65536, 98304, 147456, 250340, 262143, 262144)
# The deterministic package's programs this package does not have: the row-group backward program of route ``bwd``
# on sm_107a (264 <= C < 512) and the SM-wide row-group programs of route ``bwd`` on sm_100a, sm_103a and sm_107a
# (channel bands of 264 <= C < 512, from a token floor), and the copy of route ``bwd``'s fallback program on sm_107a
# (the channel band of 520 <= C < 1024).  Every other record pairs with this package's at the same route position.
DET_ONLY = frozenset(
    {
        "cake_causal_conv1d_det_train_50aa818e2b2338fb34a4",
        "cake_causal_conv1d_det_train_e057357c655024d4ff1d",
        "cake_causal_conv1d_det_train_baca4c969c783d07d81f",
        "cake_causal_conv1d_det_train_5ca7515dd2118dd96241",
        "cake_causal_conv1d_det_train_0d99ef0e0ed10b11dfd5",
    }
)


@contextmanager
def warnings_recorded():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _strip(record: dict, *fields: str) -> dict:
    return {key: value for key, value in record.items() if key not in fields}


def _args(record: dict) -> set[str]:
    return {name for kind, name in record["arg_plan"] if kind != "grid"}


def _det_route(arch: str, kind: str) -> list[str]:
    """The deterministic route of ``(arch, kind)`` without its :data:`DET_ONLY` programs."""

    return [name for name in det_jit.ROUTES[arch][kind] if name not in DET_ONLY]


def _pairs(role: str):
    """``(arch, kind, new record name, det record name)`` of every route of ``role``, in route order: the
    deterministic route's programs, position for position (the route's channel-band programs follow them, see
    :func:`_bands`)."""

    for arch in ARCHES:
        for kind in cake_jit.KINDS:
            if cake_jit.route_role(kind) != role:
                continue
            new, det = cake_jit.ROUTES[arch][kind], _det_route(arch, kind)
            assert len(new) >= len(det), (arch, kind)
            for name, det_name in zip(new[: len(det)], det, strict=True):
                yield arch, kind, name, det_name


def _channels(registry, name: str, arch: str) -> int | None:
    """The lowest channel threshold of a program's rule on ``arch``: a deterministic program the det-only programs
    split into several channel bands runs here from the first of them."""

    rule = registry.select_rule(registry.MODULES[name])
    value = None if rule is None else rule["channels"].get(arch)
    return value[0] if isinstance(value, list) else value


def _bands():
    """``(arch, kind, band name, twin name, det twin name)`` of every channel-band program: a program a route lists
    after the deterministic route's programs.  Its twin is the program of the route whose channel threshold on
    ``arch`` is no longer the deterministic one: the band program owns the twin's lowest channel counts there, from
    the deterministic threshold up to the twin's new one."""

    for arch in ARCHES:
        for kind in cake_jit.KINDS:
            new, det = cake_jit.ROUTES[arch][kind], _det_route(arch, kind)
            moved = [
                (name, det_name)
                for name, det_name in zip(new[: len(det)], det, strict=True)
                if _channels(cake_jit, name, arch) != _channels(det_jit, det_name, arch)
            ]
            for band in new[len(det) :]:
                (twin,) = [
                    pair
                    for pair in moved
                    if _channels(det_jit, pair[1], arch)
                    == _channels(cake_jit, band, arch)
                ]
                moved.remove(twin)
                yield arch, kind, band, *twin
            assert not moved, (arch, kind, moved)


@pytest.mark.cpu_only
def test_loader_is_the_det_loader() -> None:
    """``cake_build.py`` is byte for byte the deterministic package's (one manifest-driven loader)."""

    assert filecmp.cmp(
        Path(cake_build.__file__), Path(det_build.__file__), shallow=False
    )


@pytest.mark.cpu_only
def test_routes_mirror_the_deterministic_package() -> None:
    assert cake_jit.ARCHES == det_jit.ARCHES == ARCHES
    assert cake_jit.KINDS == det_jit.KINDS
    assert cake_jit.REQUIRED_KINDS == det_jit.REQUIRED_KINDS
    assert cake_jit.ROLES == ("fwd", "bwd")
    assert cake_jit.GRID_KINDS == ("co_resident", "occupancy")
    for arch in ARCHES:
        assert set(cake_jit.ROUTES[arch]) == set(det_jit.ROUTES[arch]), arch
    routed = {
        name
        for by_kind in cake_jit.ROUTES.values()
        for names in by_kind.values()
        for name in names
    }
    # No reduce program: every record is a routed forward or backward program.
    assert set(cake_jit.MODULES) == routed
    assert not set(cake_jit.MODULES) & set(det_jit.MODULES)
    for name, record in cake_jit.MODULES.items():
        assert PROGRAM_NAME.fullmatch(name), name
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["role"] in cake_jit.ROLES and "reduce" not in record, name
        assert record["grid_rule"]["kind"] in cake_jit.GRID_KINDS, name
        assert record["grid_rule"]["cooperative"] is False, name
        assert record["launch"]["cooperative"] is False, name


@pytest.mark.cpu_only
def test_forward_records_are_the_deterministic_forward_records() -> None:
    """The forward programs are the deterministic ones (same module, build keywords, plan, launch): ``y`` keeps
    its bits.  Only the three name fields differ."""

    for arch, kind, name, det_name in _pairs("fwd"):
        record, det = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        assert _strip(record, *NAME_FIELDS) == _strip(det, *NAME_FIELDS), (
            arch,
            kind,
            name,
        )


@pytest.mark.cpu_only
def test_backward_records_are_one_launch_with_accumulators() -> None:
    """Every backward program replaces the deterministic program + reduce pair of the same route position: the
    deterministic arguments without ``partial``, plus ``dw_acc`` / ``work`` and the gradient outs it now stores
    (``dw``, and ``db`` on the bias routes); the same build geometry, block, cluster, architectures and selection
    rule; an ordinary occupancy-sized launch without PDL; an ``accumulators`` workspace of the route's dtype and
    tap count."""

    vocabulary = set(cake_jit.LOGICAL_TENSORS) | set(cake_jit.LOGICAL_SCALARS)
    twins = {twin: band_arch for band_arch, _, _, twin, _ in _bands()}
    for arch, kind, name, det_name in _pairs("bwd"):
        record, det = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        bias = "bias" in kind
        label = (arch, kind, name)
        assert record["role"] == "bwd" and record["arches"] == det["arches"], label
        assert record["ffi_entry"] == det["ffi_entry"], label
        expected = (
            (_args(det) - {"partial"})
            | {"dw_acc", "work", "dw"}
            | ({"db"} if bias else set())
        )
        assert _args(record) == expected, label
        assert [k for k, _ in record["arg_plan"][-3:]] == ["grid", "grid", "grid"], (
            label
        )
        assert set(record["logical"]) == _args(record), label
        assert set(record["logical"].values()) <= vocabulary, label
        assert (
            record["logical"].get("x_map", "x") == "x"
            and record["logical"].get("dy_map", "dy") == "dy"
        )
        # The global accumulators: kernel parameter ``dw_acc`` (the det fragments' ``acc`` is a register), logical ``acc``.
        assert (
            record["logical"]["dw_acc"] == "acc" and record["logical"]["work"] == "work"
        ), label
        assert ("flags" in _args(record)) is (kind != "bwd"), label
        grid, det_grid = record["grid_rule"], det["grid_rule"]
        assert grid["kind"] == "occupancy" and grid["cooperative"] is False, label
        assert grid["geometry"] == det_grid["geometry"], label
        assert set(grid["grid_mult"]) == set(record["arches"]), label
        # ``ctas_per_sm`` is the rendered program's occupancy (the exporter's leaf).
        assert isinstance(grid["ctas_per_sm"], int) and grid["ctas_per_sm"] >= 1, label
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, label
        assert launch["block"] == det["launch"]["block"], label
        assert launch["cluster"] == det["launch"]["cluster"], label
        assert (
            isinstance(launch["dynamic_smem_bytes"], int)
            and launch["dynamic_smem_bytes"] >= 0
        )
        # A twin keeps the deterministic rule but on the architecture of its channel-band program.
        det_rule = det_jit.select_rule(det)
        if det_rule is not None:
            channels = {
                a: _channels(det_jit, det_name, a) for a in det_rule["channels"]
            }
            det_rule = {**det_rule, "channels": channels}
        if name in twins:
            det_rule["channels"][twins[name]] = _channels(cake_jit, name, twins[name])
        assert cake_jit.select_rule(record) == det_rule, label
        rule = dict(record["workspace_rule"])
        assert "slots" in rule, label  # required: the package never defaults it
        slots = rule.pop("slots")
        assert rule == {
            "acc": "float64" if kind.endswith("fp32g") else "float32",
            "kind": "accumulators",
            "taps": 5 if bias else 4,
            "work": "uint32",
        }, label
        # Accumulator copies, as the render names them: the fp32g programs may keep several, the FP32 routes one.
        assert isinstance(slots, int) and slots >= 1, label
        if not kind.endswith("fp32g"):
            assert slots == 1, label


@pytest.mark.cpu_only
def test_channel_band_programs_take_their_twins_lowest_channels() -> None:
    """A channel-band program ships for one architecture: its record is its twin's under its own name, architecture
    and selection rule (the rendered occupancy and shared memory aside).  It owns the calls from the deterministic
    program's channel threshold, and the twin those from a higher one, so the route still runs one program per
    ``(S, C)``."""

    bands = list(_bands())
    assert {(arch, kind) for arch, kind, *_ in bands} == {
        ("sm_107a", "bwd"),
        ("sm_107a", "bwd_bias"),
        ("sm_107a", "bwd_fp32g"),
    }
    for arch, kind, band, twin, det_twin in bands:
        record, twin_record = cake_jit.MODULES[band], cake_jit.MODULES[twin]
        label = (arch, kind, band)
        assert band not in det_jit.MODULES and record["arches"] == [arch], label
        assert record["library"] == record["kernel"] == band, label
        assert record["kernel_symbol"] == f"kernel_{band}", label
        threshold = _channels(cake_jit, band, arch)
        assert threshold == _channels(det_jit, det_twin, arch), label
        assert threshold < _channels(cake_jit, twin, arch), label
        assert cake_jit.select_rule(record) == {
            "channels": {arch: threshold},
            "kind": "channels_at_least",
        }, label
        grid, twin_grid = record["grid_rule"], twin_record["grid_rule"]
        assert grid["grid_mult"] == {arch: twin_grid["grid_mult"][arch]}, label
        derived = ("grid_mult", "ctas_per_sm")
        assert _strip(grid, *derived) == _strip(twin_grid, *derived), label
        launch = _strip(record["launch"], "dynamic_smem_bytes")
        assert launch == _strip(twin_record["launch"], "dynamic_smem_bytes"), label
        own = ("arches", "select_rule", "grid_rule", "launch", *NAME_FIELDS)
        assert _strip(record, *own) == _strip(twin_record, *own), label


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_selection_matches_the_deterministic_package(arch, monkeypatch) -> None:
    """A call selects the program at the same route position as the deterministic package for every ``(S, C)``,
    and plans the same scalars (``y`` / ``dx`` keep their per-element arithmetic).  Where the deterministic package
    selects a det-only program, this package selects what the deterministic package selects without its det-only
    records."""

    twin_of = {
        band: twin for band_arch, _, band, twin, _ in _bands() if band_arch == arch
    }
    det_only_hits = set()
    for kind in cake_jit.KINDS:
        new, det = cake_jit.ROUTES[arch][kind], _det_route(arch, kind)
        for C in CHANNELS:
            for S in TOKENS:
                (name,) = cake_jit.selected_modules(arch, kind, C=C, S=S)
                (det_name,) = det_jit.selected_modules(arch, kind, C=C, S=S)
                if det_name in DET_ONLY:
                    det_only_hits.add(det_name)
                    with monkeypatch.context() as m:
                        m.setitem(det_jit.ROUTES[arch], kind, det)
                        (det_name,) = det_jit.selected_modules(arch, kind, C=C, S=S)
                # A channel-band program serves its twin's route position.
                position = new.index(twin_of.get(name, name))
                assert position == det.index(det_name), (arch, kind, C, S)
                record, det_record = cake_jit.MODULES[name], det_jit.MODULES[det_name]
                geometry = record["grid_rule"]["geometry"]
                assert cake_jit.plan(
                    S, C, 3, geometry=geometry, role=record["role"]
                ) == det_jit.plan(
                    S,
                    C,
                    3,
                    geometry=det_record["grid_rule"]["geometry"],
                    role=det_record["role"],
                ), (arch, kind, C, S)
    det_only_routed = {
        name for name in DET_ONLY if arch in det_jit.MODULES[name]["arches"]
    }
    assert det_only_hits == det_only_routed, arch


def _align16(n: int) -> int:
    return -(-n // 16) * 16


@pytest.mark.cpu_only
def test_accumulator_layout() -> None:
    assert cake_jit.WORKSPACE_ALIGN == 16
    assert cake_jit.ACCUMULATOR_DTYPES == {"float32": 4, "float64": 8}
    assert cake_jit.COUNTER_DTYPES == {"uint32": 4}
    for arch, kind, name, _ in _pairs("bwd"):
        record = cake_jit.MODULES[name]
        rule = record["workspace_rule"]
        size = cake_jit.ACCUMULATOR_DTYPES[rule["acc"]]
        for C in CHANNELS:
            _, _, n_ch_tiles = cake_jit.channel_split(
                C, geometry=record["grid_rule"]["geometry"], role="bwd"
            )
            layout = cake_jit.accumulator_layout(record, C, n_ch_tiles)
            words = rule["slots"] * rule["taps"]
            assert cake_jit.accumulator_words_per_channel(record) == words
            acc_bytes = words * C * size
            assert layout == {
                "acc_dtype": rule["acc"],
                "acc_elems": words * C,
                "acc_bytes": acc_bytes,
                "work_dtype": "uint32",
                "work_offset": _align16(acc_bytes),
                "work_words": n_ch_tiles,
                "total_bytes": _align16(_align16(acc_bytes) + 4 * n_ch_tiles),
            }, (arch, kind, C)
            # The views the host takes stay inside the buffer and do not overlap.
            assert layout["acc_bytes"] <= layout["work_offset"]
            assert layout["work_offset"] + 4 * n_ch_tiles <= layout["total_bytes"]
            assert layout["work_offset"] % size == 0
        with pytest.raises(ValueError):
            cake_jit.accumulator_layout(record, 0, 1)
        with pytest.raises(ValueError):
            cake_jit.accumulator_layout(record, 256, 0)
    forward = cake_jit.MODULES[cake_jit.ROUTES["sm_100a"]["fwd"][0]]
    with pytest.raises(ValueError, match="no backward accumulator workspace"):
        cake_jit.accumulator_layout(forward, 256, 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ("bwd_fp32g", "bwd_bias_fp32g"))
@pytest.mark.parametrize("C", (512, 8192))
def test_accumulators_are_sized_from_the_rule(kind, C) -> None:
    """The accumulator size comes from the rendered rule, never a fixed ``C x taps``: an fp32g program whose rule
    names ``slots`` FP64 copies gets ``slots x taps x C`` words; a rule without ``slots``, or with a malformed one,
    is refused."""

    import copy

    for arch in ARCHES:
        if kind not in cake_jit.ROUTES[arch]:
            continue
        name = cake_jit.route_modules(arch, kind)[0]
        record = cake_jit.MODULES[name]
        rule = record["workspace_rule"]
        assert rule["acc"] == "float64"
        _, _, n_ch_tiles = cake_jit.channel_split(
            C, geometry=record["grid_rule"]["geometry"], role="bwd"
        )
        rendered = cake_jit.accumulator_layout(record, C, n_ch_tiles)
        assert rendered["acc_elems"] == rule["slots"] * rule["taps"] * C
        for slots in (1, 4):
            variant = copy.deepcopy(record)
            variant["workspace_rule"]["slots"] = slots
            layout = cake_jit.accumulator_layout(variant, C, n_ch_tiles)
            assert layout["acc_elems"] == slots * rule["taps"] * C, (arch, slots)
            assert layout["acc_bytes"] == 8 * slots * rule["taps"] * C
            assert layout["work_offset"] == _align16(layout["acc_bytes"])
            assert layout["total_bytes"] >= layout["work_offset"] + 4 * n_ch_tiles
        for slots in (0, -1, 1.5, True, "4", None):
            variant = copy.deepcopy(record)
            variant["workspace_rule"]["slots"] = slots
            with pytest.raises(ValueError, match="slots"):
                cake_jit.accumulator_layout(variant, C, n_ch_tiles)
        # A rule without the field is refused (no default could size dw_acc for every program).
        variant = copy.deepcopy(record)
        del variant["workspace_rule"]["slots"]
        with pytest.raises(ValueError, match="slots"):
            cake_jit.accumulator_layout(variant, C, n_ch_tiles)
        with pytest.raises(ValueError, match="slots"):
            cake_jit.accumulator_words_per_channel(variant)
        with pytest.raises(ValueError, match="slots"):
            pkg._accumulator_layout(variant, C, n_ch_tiles)


def _owner_tiles(record: dict, arch: str, C: int) -> int:
    """The largest channel tile count a backward ``record`` launches with at ``C`` on ``arch``, whatever ``S``:
    the split's, or its route's :data:`cake_jit.TILE_OVERRIDES` or :data:`cake_jit.NARROW_TILES` row's when that
    has more."""

    geometry = record["grid_rule"]["geometry"]
    route = cake_jit.record_route(record, arch)
    _, _, n_ch_tiles = cake_jit.channel_split(C, geometry=geometry, role="bwd")
    entry = cake_jit.TILE_OVERRIDES.get(arch, {}).get((route, C))
    if entry is not None:
        n_ch_tiles = max(n_ch_tiles, -(-(C // geometry["vec"]) // entry[0]))
    elif (
        cake_jit.narrow_tile_range(arch, route, C) is not None
        and geometry["threads"] == 128
    ):
        n_ch_tiles = max(n_ch_tiles, -(-(C // geometry["vec"]) // 32))
    return n_ch_tiles


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_backward_workspace_serves_every_backward_program(arch) -> None:
    """The public backward workspace is the largest layout over every backward program of the architecture, each
    at the largest tile count it may launch with (its split, or its route's ``TILE_OVERRIDES`` or ``NARROW_TILES``
    row), so one zeroed buffer serves whichever program, plan and route a call selects; the capacity owner is an
    FP64 bias program (8 bytes x slots x 5 taps per channel, 40 with one copy).  A narrow row's three tickets fit
    the 16-byte alignment of the single tile's, so its size is the split's."""

    owner = cake_jit.workspace_record(arch)
    assert (
        owner["workspace_rule"]["acc"] == "float64"
        and owner["workspace_rule"]["taps"] == 5
    )
    owners = cake_jit.workspace_owners(arch)
    assert len(owners) == sum(
        len(cake_jit.route_modules(arch, kind))
        for kind in cake_jit.backward_kinds(arch)
    )
    overridden = {C for _, C in cake_jit.TILE_OVERRIDES.get(arch, {})}
    narrow = {
        C for _, c_lo, c_hi in cake_jit.NARROW_TILES.get(arch, {}) for C in (c_lo, c_hi)
    }
    for C in sorted(set(CHANNELS) | overridden | narrow):
        nbytes = cake_jit.backward_workspace_bytes(arch, C)
        layouts = [
            cake_jit.accumulator_layout(record, C, _owner_tiles(record, arch, C))
            for record in owners
        ]
        assert nbytes == max(layout["total_bytes"] for layout in layouts), (arch, C)
        if C not in overridden:
            per_channel = 8 * cake_jit.accumulator_words_per_channel(owner)
            assert nbytes == _align16(_align16(per_channel * C) + 4 * cake_jit.channel_split(
                C, geometry=owner["grid_rule"]["geometry"], role="bwd")[2])  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------
# The narrower channel tiles of the backward (``TILE_OVERRIDES``).


@pytest.mark.cpu_only
def test_tile_overrides_are_backward_rows() -> None:
    """Every row of the rendered table names an exported architecture, one of its backward routes, a channel count
    of the package and a ``ct_vec`` that is a whole number of warps no wider than a CTA, over a closed ``S`` range."""

    assert set(cake_jit.TILE_OVERRIDES) <= set(ARCHES)
    for arch, rows in cake_jit.TILE_OVERRIDES.items():
        for (route, C), (ct_vec, s_min, s_max) in rows.items():
            label = (arch, route, C)
            assert route in cake_jit.backward_kinds(arch), label
            assert C > 0 and C % cake_jit.CHANNEL_MULTIPLE == 0, label
            assert 0 < s_min <= s_max, label
            for name in cake_jit.route_modules(arch, route):
                threads = cake_jit.MODULES[name]["grid_rule"]["geometry"]["threads"]
                assert ct_vec % 32 == 0 and 0 < ct_vec <= threads, label


@pytest.mark.cpu_only
def test_tile_override_bounds_and_plan() -> None:
    """A row's ``ct_vec`` applies from ``S_min`` through ``S_max`` (both included) and to no other token count,
    route or architecture; ``plan`` of a backward record given its architecture and route then plans ``(ct_vec,
    threads // ct_vec, ceil(C / vec / ct_vec))``, and :func:`cake_jit.channel_split`'s split outside the range, for
    the forward role, under ``legacy`` and without the architecture or the route."""

    for arch, rows in cake_jit.TILE_OVERRIDES.items():
        for (route, C), (ct_vec, s_min, s_max) in rows.items():
            label = (arch, route, C)
            assert cake_jit.tile_override(arch, route, C, s_min - 1) is None, label
            assert cake_jit.tile_override(arch, route, C, s_min) == ct_vec, label
            assert cake_jit.tile_override(arch, route, C, s_max) == ct_vec, label
            assert cake_jit.tile_override(arch, route, C, s_max + 1) is None, label
            for other in ARCHES:
                if (route, C) not in cake_jit.TILE_OVERRIDES.get(other, {}):
                    assert cake_jit.tile_override(other, route, C, s_min) is None
            for name in cake_jit.route_modules(arch, route):
                geometry = cake_jit.MODULES[name]["grid_rule"]["geometry"]
                threads, vec = geometry["threads"], geometry["vec"]
                split = dict(
                    zip(
                        ("ct_vec", "n_sub", "n_ch_tiles"),
                        cake_jit.channel_split(C, geometry=geometry, role="bwd"),
                        strict=True,
                    )
                )
                wanted = {
                    "ct_vec": ct_vec,
                    "n_sub": threads // ct_vec,
                    "n_ch_tiles": -(-(C // vec) // ct_vec),
                }
                for S in (s_min, s_max):
                    got = cake_jit.plan(
                        S, C, 3, geometry=geometry, role="bwd", arch=arch, route=route
                    )
                    assert {k: got[k] for k in wanted} == wanted, (label, S)
                for S in (s_min - 1, s_max + 1):
                    got = cake_jit.plan(
                        S, C, 3, geometry=geometry, role="bwd", arch=arch, route=route
                    )
                    assert {k: got[k] for k in split} == split, (label, S)
                for kwargs in (
                    {"role": "bwd"},
                    {"role": "bwd", "arch": arch},
                    {"role": "bwd", "route": route},
                    {"role": "fwd", "arch": arch, "route": route},
                ):
                    got = cake_jit.plan(s_min, C, 3, geometry=geometry, **kwargs)
                    want = cake_jit.channel_split(
                        C, geometry=geometry, role=kwargs["role"]
                    )
                    assert (got["ct_vec"], got["n_sub"], got["n_ch_tiles"]) == want, (
                        label,
                        kwargs,
                    )
                got = cake_jit.plan(
                    s_min, C, 3, geometry=geometry, legacy=True, arch=arch, route=route
                )
                want = cake_jit.channel_split(C, geometry=geometry, legacy=True)
                assert (got["ct_vec"], got["n_sub"], got["n_ch_tiles"]) == want, label


@pytest.mark.cpu_only
def test_backward_workspace_covers_every_override_row() -> None:
    """The public workspace of a row's ``C`` holds the layout of every program of the row's route at the row's tile
    count (one ticket per tile), whatever ``S``."""

    for arch, rows in cake_jit.TILE_OVERRIDES.items():
        for (route, C), (_, s_min, _) in rows.items():
            nbytes = cake_jit.backward_workspace_bytes(arch, C)
            for name in cake_jit.route_modules(arch, route):
                record = cake_jit.MODULES[name]
                geometry = record["grid_rule"]["geometry"]
                tiles = cake_jit.plan(
                    s_min, C, 3, geometry=geometry, role="bwd", arch=arch, route=route
                )["n_ch_tiles"]
                assert tiles == cake_jit.override_tiles(arch, route, C, geometry)
                layout = cake_jit.accumulator_layout(record, C, tiles)
                assert layout["total_bytes"] <= nbytes, (arch, route, C, name)


# ----------------------------------------------------------------------------------------------------------------
# The 32-vector channel tiles of the narrow single-tile backward rows (``NARROW_TILES``).

# A table that exercises the narrow-row plan on every architecture whatever the rendered table holds: the sm_100a
# rows of the shipped rule, plus rows on the other architectures (one of them spans a ``TILE_OVERRIDES`` row of its
# route, C = 768 on sm_103a, which keeps its override).
SYNTHETIC_NARROW_TILES = {
    "sm_100a": {
        ("bwd", 224, 224): (31292, 500680),
        ("bwd", 520, 624): (31292, 500680),
        ("bwd", 632, 760): (31292, 500680),
    },
    "sm_103a": {
        ("bwd", 632, 768): (62585, 250340),
        ("bwd_bias", 224, 224): (31292, 125170),
    },
    "sm_107a": {("bwd_bias", 520, 760): (31292, 62585)},
}
NARROW_TABLES = ("rendered", "synthetic")


@pytest.fixture
def narrow_table(request, monkeypatch):
    """The ``NARROW_TILES`` table of a test: the rendered one, or :data:`SYNTHETIC_NARROW_TILES` in its place."""

    if request.param == "synthetic":
        monkeypatch.setattr(cake_jit, "NARROW_TILES", SYNTHETIC_NARROW_TILES)
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    yield cake_jit.NARROW_TILES
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()


def _covering(table: dict, arch: str, route: str, C: int):
    """The ``(S_min, S_max)`` of the first key of ``table[arch]`` on ``route`` covering ``C`` (``None``: none, or
    a ``TILE_OVERRIDES`` row of ``(route, C)``)."""

    if C % 8 or (route, C) in cake_jit.TILE_OVERRIDES.get(arch, {}):
        return None
    for (name, c_lo, c_hi), bounds in table.get(arch, {}).items():
        if name == route and c_lo <= C <= c_hi:
            return bounds
    return None


def _split3(plan: dict) -> tuple[int, int, int]:
    return plan["ct_vec"], plan["n_sub"], plan["n_ch_tiles"]


@pytest.mark.cpu_only
def test_narrow_tiles_are_backward_rows() -> None:
    """Every key of the rendered table names an exported architecture, one of its backward routes, a closed range of
    channel counts (multiples of 8) and a closed ``S`` range; no key of an architecture and route overlaps another,
    and no ``(route, C)`` of the architecture's ``TILE_OVERRIDES`` lies inside a key."""

    assert set(cake_jit.NARROW_TILES) <= set(ARCHES)
    for arch, rows in cake_jit.NARROW_TILES.items():
        spans: dict[str, list[tuple[int, int]]] = {}
        for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
            label = (arch, route, c_lo, c_hi)
            assert route in cake_jit.backward_kinds(arch), label
            assert 0 < c_lo <= c_hi and c_lo % 8 == 0 and c_hi % 8 == 0, label
            assert 0 < s_min <= s_max, label
            for name in cake_jit.route_modules(arch, route):
                geometry = cake_jit.MODULES[name]["grid_rule"]["geometry"]
                assert geometry["threads"] % 32 == 0, (label, name)
            for override_route, C in cake_jit.TILE_OVERRIDES.get(arch, {}):
                assert not (override_route == route and c_lo <= C <= c_hi), label
            spans.setdefault(route, []).append((c_lo, c_hi))
        for route, ranges in spans.items():
            ranges.sort()
            for (_, hi), (lo, _) in zip(ranges, ranges[1:], strict=False):
                assert hi < lo, (arch, route, ranges)


def _check_narrow_plan(table: dict, arch: str, route: str, C: int, S: int) -> int:
    """One ``(route, C, S)`` call of ``arch`` against ``table``: the narrow-row reading on every architecture, and
    ``plan`` of every program of the route with and without the architecture / route, under ``legacy`` and at
    another thread count.  Returns the number of programs checked."""

    for other in ARCHES:
        bounds = _covering(table, other, route, C)
        inside = bounds is not None and bounds[0] <= S <= bounds[1]
        assert cake_jit.narrow_tile_range(other, route, C) == bounds, (other, route, C)
        assert cake_jit.narrow_tile_row(other, route, C, S) == inside, (other, route, C)
    bounds = _covering(table, arch, route, C)
    inside = bounds is not None and bounds[0] <= S <= bounds[1]
    override = cake_jit.tile_override(arch, route, C, S)
    on_row = {"arch": arch, "route": route}
    for name in cake_jit.route_modules(arch, route):
        geometry = cake_jit.MODULES[name]["grid_rule"]["geometry"]
        threads, vec = geometry["threads"], geometry["vec"]
        label = (arch, route, C, S, name)
        split = cake_jit.channel_split(C, geometry=geometry, role="bwd")
        if override is not None:
            want = (override, threads // override, -(-(C // vec) // override))
        elif inside and threads == 128:
            want = (32, threads // 32, -(-(C // vec) // 32))
        else:
            want = split
        got = cake_jit.plan(S, C, 3, geometry=geometry, role="bwd", **on_row)
        assert _split3(got) == want, label
        for kwargs in ({}, {"arch": arch}, {"route": route}):
            got = cake_jit.plan(S, C, 3, geometry=geometry, role="bwd", **kwargs)
            assert _split3(got) == split, (label, kwargs)
        got = cake_jit.plan(S, C, 3, geometry=geometry, legacy=True, **on_row)
        legacy = cake_jit.channel_split(C, geometry=geometry, legacy=True)
        assert _split3(got) == legacy, label
        wide = {**geometry, "threads": 256}
        got = cake_jit.plan(S, C, 3, geometry=wide, role="bwd", **on_row)
        if override is None:
            wide_split = cake_jit.channel_split(C, geometry=wide, role="bwd")
            assert _split3(got) == wide_split, label
        tiles = cake_jit.narrow_tiles(arch, route, C, geometry)
        narrow = bounds is not None and threads == 128
        assert tiles == (-(-(C // vec) // 32) if narrow else 0), label
    return len(cake_jit.route_modules(arch, route))


@pytest.mark.cpu_only
@pytest.mark.parametrize("narrow_table", NARROW_TABLES, indirect=True)
def test_narrow_tile_bounds_and_plan(narrow_table) -> None:
    """A key covers its channel counts from ``C_lo`` through ``C_hi`` and its token counts from ``S_min`` through
    ``S_max`` (both ends included; never a ``TILE_OVERRIDES`` row) on its architecture and route only.  ``plan`` of a
    backward record given its architecture and route then plans 32-vector tiles at 128 threads, ``(32, threads //
    32, ceil(C / vec / 32))``; :func:`cake_jit.channel_split`'s split (or the override's) everywhere else, at
    another thread count, under ``legacy`` and without the architecture or the route.  ``narrow_tiles`` (the
    workspace bound) holds the key's tile count whatever ``S``."""

    checked = 0
    for arch, rows in narrow_table.items():
        for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
            channels = {c_lo - 8, c_lo, (c_lo + c_hi) // 16 * 8, c_hi, c_hi + 8}
            for C in sorted(C for C in channels if C > 0):
                for S in (s_min - 1, s_min, s_max, s_max + 1):
                    checked += _check_narrow_plan(narrow_table, arch, route, C, S)
    assert checked or not narrow_table


@pytest.mark.cpu_only
@pytest.mark.parametrize("narrow_table", NARROW_TABLES, indirect=True)
def test_backward_workspace_covers_every_narrow_row(narrow_table) -> None:
    """The public workspace of a key's channel counts holds the layout of every program of the key's route at the
    key's tile count (one ticket per tile), whatever ``S``, and is the largest layout over every backward program
    at the largest tile count it may launch with."""

    for arch, rows in narrow_table.items():
        owners = cake_jit.workspace_owners(arch)
        for (route, c_lo, c_hi), (s_min, _) in rows.items():
            for C in sorted({c_lo, (c_lo + c_hi) // 16 * 8, c_hi}):
                nbytes = cake_jit.backward_workspace_bytes(arch, C)
                layouts = [
                    cake_jit.accumulator_layout(
                        record, C, _owner_tiles(record, arch, C)
                    )
                    for record in owners
                ]
                assert nbytes == max(layout["total_bytes"] for layout in layouts), (
                    arch,
                    route,
                    C,
                )
                for name in cake_jit.route_modules(arch, route):
                    record = cake_jit.MODULES[name]
                    geometry = record["grid_rule"]["geometry"]
                    tiles = cake_jit.plan(
                        s_min,
                        C,
                        3,
                        geometry=geometry,
                        role="bwd",
                        arch=arch,
                        route=route,
                    )["n_ch_tiles"]
                    layout = cake_jit.accumulator_layout(record, C, tiles)
                    assert layout["total_bytes"] <= nbytes, (arch, route, C, name)


# ----------------------------------------------------------------------------------------------------------------
# The host plan of a call row (``_ROW_PLANS``): planned by the row's first call, the same values as a per-call plan.

# The SM counts of the fake devices of the sweep, per architecture (B200 / GB300 / R200 boards first: the counts the
# grid-trim rules were measured at).  The default sweep takes the first; the full sweep every one.
PLAN_SWEEP_SMS = {"sm_100a": (148, 160), "sm_103a": (152, 148), "sm_107a": (212, 216)}
# A device index no real device has: the rows the sweep plans never serve a launch.
FAKE_DEVICE = 1 << 20
# CAKE_CONV1D_FULL_PLAN_SWEEP=1 runs the full sweep: every SM count, five document counts and three token orders.
FULL_PLAN_SWEEP = os.environ.get("CAKE_CONV1D_FULL_PLAN_SWEEP") == "1"


def _plan_sweep_channels() -> tuple[int, ...]:
    """``CHANNELS``, rows off the 32-channel grid, and every selection threshold and ``TILE_OVERRIDES`` channel count
    of the registry with the multiple of 8 below it, and both ends of every ``NARROW_TILES`` key with their
    neighbours outside it."""

    channels = {*CHANNELS, 16, 24, 248, 520, 640, 1048, 1152, 3072}
    for record in cake_jit.MODULES.values():
        rule = cake_jit.select_rule(record)
        for C in (rule or {}).get("channels", {}).values():
            channels |= {int(C) - 8, int(C)}
    for rows in cake_jit.TILE_OVERRIDES.values():
        for _, C in rows:
            channels |= {C - 8, C, C + 8}
    for rows in cake_jit.NARROW_TILES.values():
        for _, c_lo, c_hi in rows:
            channels |= {c_lo - 8, c_lo, c_hi, c_hi + 8}
    return tuple(sorted(C for C in channels if C > 0))


def _plan_sweep_tokens() -> tuple[int, ...]:
    """Token counts on both sides of every bound where ``S`` enters a plan: the ``TILE_OVERRIDES`` and
    ``NARROW_TILES`` ranges, the grid-trim minimums and the selection rules' token floors, plus a few others."""

    tokens = {1, 2, 63, 64, 1000, 250340, 1 << 20}
    for table in (cake_jit.TILE_OVERRIDES, cake_jit.NARROW_TILES):
        for rows in table.values():
            for value in rows.values():
                s_min, s_max = value[-2:]
                tokens |= {s_min - 1, s_min, s_max, s_max + 1}
    tokens |= {cake_jit.TILE_ALIGNED_MIN_S - 1, cake_jit.TILE_ALIGNED_MIN_S}
    for classes in cake_jit.TILE_ALIGNED_CLASSES.values():
        for s_min, _ in classes.values():
            tokens |= {s_min - 1, s_min}
    for record in cake_jit.MODULES.values():
        rule = cake_jit.select_rule(record)
        for S in (rule or {}).get("tokens", {}).values():
            tokens |= {int(S) - 1, int(S)}
    return tuple(sorted(S for S in tokens if S > 0))


@pytest.fixture
def fake_device(monkeypatch):
    """Plan on :data:`FAKE_DEVICE` of a given architecture and SM count: ``use(arch, sms)`` points ``cake_jit``'s
    device queries at it and empties ``_ROW_PLANS``; the rows planned are dropped again after the test."""

    def use(arch: str, sms: int) -> None:
        monkeypatch.setattr(cake_jit, "device_arch", lambda device_index: arch)
        monkeypatch.setattr(cake_jit, "sm_count", lambda device_index: sms)
        pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()

    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    yield use
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()


def _per_call(arch: str, kind: str, S: int, C: int, num_docs: int, gm: int, policy):
    """What a call plans without ``_ROW_PLANS`` (``_route`` then ``_launch``), or the error it raises."""

    try:
        names = pkg._route(arch, kind, needs=pkg._route_needs(policy), C=C, S=S)
        scalars, grid = pkg._launch(
            cake_jit.MODULES[names[0]],
            S,
            C,
            num_docs,
            device_index=FAKE_DEVICE,
            arch=arch,
            bwd_grid_mult=gm,
        )
        return tuple(names), pkg._scalars(scalars), tuple(grid)
    except (ValueError, RuntimeError) as exc:
        return type(exc).__name__, str(exc)


def _row_planned(arch: str, kind: str, S: int, C: int, num_docs: int, gm: int, policy):
    """What a call plans from its ``_ROW_PLANS`` row (``_call_plan``), or the error it raises."""

    try:
        names, record, scalars, grid, _ = pkg._call_plan(
            arch,
            kind,
            S,
            C,
            num_docs,
            device_index=FAKE_DEVICE,
            policy=policy,
            bwd_grid_mult=gm,
        )
        assert record is cake_jit.MODULES[names[0]]
        assert tuple(scalars) == cake_jit.PLAN_SCALARS
        assert all(type(value) is int for value in scalars.values())
        return tuple(names), dict(scalars), tuple(grid)
    except (ValueError, RuntimeError) as exc:
        return type(exc).__name__, str(exc)


def _sweep_row_plans(arch: str) -> int:
    """Every route of ``arch`` x the sweep's channel counts, token counts (in each order of the sweep, the row
    planned by the order's first call), document counts and backward grid multiples: the row plan equals the per-call
    plan, errors included.  Returns the number of calls checked."""

    channels, tokens = _plan_sweep_channels(), _plan_sweep_tokens()
    orders = [tokens]
    if FULL_PLAN_SWEEP:
        half = len(tokens) // 2
        orders += [tokens[::-1], tokens[half:] + tokens[:half]]
    docs = (1, 2, 4, 5, 1000) if FULL_PLAN_SWEEP else (1, 1000)
    checked = 0
    for order in orders:
        pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
        for kind in cake_jit.ROUTES[arch]:
            backward = cake_jit.route_role(kind) == "bwd"
            policy = "fp32" if kind.endswith("fp32g") else "bf16" if backward else None
            for gm in pkg.BWD_GRID_MULTS if backward else (1,):
                for C in channels:
                    for S in order:
                        for num_docs in docs:
                            want = _per_call(arch, kind, S, C, num_docs, gm, policy)
                            got = _row_planned(arch, kind, S, C, num_docs, gm, policy)
                            assert got == want, (arch, kind, C, S, num_docs, gm)
                            checked += 1
        assert len(pkg._ROW_PLANS) <= pkg._ROW_PLANS_MAX
    return checked


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_row_plans_match_the_per_call_plan(arch, fake_device, record_property) -> None:
    """A call planned from its ``_ROW_PLANS`` row launches the programs, plan scalars and grid a per-call plan
    (``_route`` + ``_launch``) gives, and raises the same errors, on fake devices of the architecture: every route,
    channel counts at every selection threshold and ``C % 32`` rows (whose grid ``launch_grid`` may trim by ``S``),
    token counts at every ``TILE_OVERRIDES`` / ``NARROW_TILES`` bound, grid-trim minimum and token floor, document
    counts and every ``bwd_grid_mult``.  ``CAKE_CONV1D_FULL_PLAN_SWEEP=1`` runs the full sweep."""

    checked = 0
    for sms in PLAN_SWEEP_SMS[arch][: None if FULL_PLAN_SWEEP else 1]:
        fake_device(arch, sms)
        checked += _sweep_row_plans(arch)
    record_property("row_plan_calls_checked", checked)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_row_plans_follow_a_replaced_override_table(
    arch, fake_device, monkeypatch
) -> None:
    """A row planned before ``TILE_OVERRIDES`` changes keeps its plan until
    :func:`cake_causal_conv1d_nondet_train_plan_cache_clear` empties the cache; after it, with the rows of every architecture on ``arch``, every row's calls on both
    sides of its token range plan as a per-call plan does, the override's tiles inside the range."""

    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    shipped = dict(cake_jit.TILE_OVERRIDES.get(arch, {}))
    rows = {
        key: value
        for table in cake_jit.TILE_OVERRIDES.values()
        for key, value in table.items()
    }

    def policy_of(route: str) -> str:
        return "fp32" if route.endswith("fp32g") else "bf16"

    planned = {
        key: _row_planned(arch, key[0], value[1], key[1], 3, 1, policy_of(key[0]))
        for key, value in rows.items()
        if key[0] in cake_jit.ROUTES[arch]
    }
    monkeypatch.setitem(cake_jit.TILE_OVERRIDES, arch, rows)
    changed = 0
    for (route, C), before in planned.items():
        _, s_min, _ = rows[(route, C)]
        now = _per_call(arch, route, s_min, C, 3, 1, policy_of(route))
        assert _row_planned(arch, route, s_min, C, 3, 1, policy_of(route)) == before
        changed += now != before
    assert changed or rows.items() <= shipped.items(), arch
    assert pkg.cake_causal_conv1d_nondet_train_plan_cache_clear() >= 1
    checked = 0
    for (route, C), (ct_vec, s_min, s_max) in rows.items():
        if route not in cake_jit.ROUTES[arch]:
            continue
        for S in (s_min - 1, s_min, s_max, s_max + 1):
            for gm in pkg.BWD_GRID_MULTS:
                want = _per_call(arch, route, S, C, 3, gm, policy_of(route))
                got = _row_planned(arch, route, S, C, 3, gm, policy_of(route))
                assert got == want, (route, C, S, gm)
                if s_min <= S <= s_max and isinstance(got[1], dict):
                    assert got[1]["ct_vec"] == ct_vec, (route, C, S, gm)
                checked += 1
    assert checked


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_row_plans_follow_a_replaced_narrow_table(
    arch, fake_device, monkeypatch
) -> None:
    """A row planned before ``NARROW_TILES`` changes keeps its plan until
    :func:`cake_causal_conv1d_nondet_train_plan_cache_clear` empties the cache; after it, with :data:`SYNTHETIC_NARROW_TILES` in place, every key's calls at its channel ends and an
    inner channel count, on both sides of its token range and at every ``bwd_grid_mult``, plan as a per-call plan
    does (the 32-vector tiles inside the range, the side-stream and whole-rank grid trims included), on fake
    devices of the architecture's SM counts."""

    rows = SYNTHETIC_NARROW_TILES[arch]

    def policy_of(route: str) -> str:
        return "fp32" if route.endswith("fp32g") else "bf16"

    def planned(route: str, S: int, C: int, gm: int = 1):
        return _row_planned(arch, route, S, C, 3, gm, policy_of(route))

    checked = 0
    for sms in PLAN_SWEEP_SMS[arch]:
        fake_device(arch, sms)
        before = {key: planned(key[0], value[0], key[1]) for key, value in rows.items()}
        with monkeypatch.context() as m:
            m.setattr(cake_jit, "NARROW_TILES", SYNTHETIC_NARROW_TILES)
            for key, value in rows.items():
                assert planned(key[0], value[0], key[1]) == before[key], (sms, key)
            assert pkg.cake_causal_conv1d_nondet_train_plan_cache_clear() >= 1
            for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
                for C in sorted({c_lo, (c_lo + c_hi) // 16 * 8, c_hi}):
                    for S in (s_min - 1, s_min, s_max, s_max + 1):
                        ct_vec = cake_jit.tile_override(arch, route, C, S) or 32
                        for gm in pkg.BWD_GRID_MULTS:
                            want = _per_call(arch, route, S, C, 3, gm, policy_of(route))
                            got = planned(route, S, C, gm)
                            assert got == want, (arch, sms, route, C, S, gm)
                            if s_min <= S <= s_max and isinstance(got[1], dict):
                                assert got[1]["ct_vec"] == ct_vec, (route, C, S, gm)
                            checked += 1
            pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    assert checked


# ----------------------------------------------------------------------------------------------------------------
# The public names: the mode word in every name, no mode keyword (FlashInfer issue #8928).

PUBLIC = (
    "cake_causal_conv1d_nondet_train_forward",
    "cake_causal_conv1d_nondet_train_backward",
    "cake_causal_conv1d_nondet",
    "cake_causal_conv1d_nondet_train_forward_workspace",
    "cake_causal_conv1d_nondet_train_forward_workspace_bytes",
    "cake_causal_conv1d_nondet_train_backward_workspace",
    "cake_causal_conv1d_nondet_train_backward_workspace_bytes",
)
# The names before the mode word joined them: removed, with no alias.
OLD_NAMES = (
    "plan_cache_clear",
    "cake_causal_conv1d",
    "CakeCausalConv1dFunction",
    "cake_causal_conv1d_train_forward",
    "cake_causal_conv1d_train_backward",
    "cake_causal_conv1d_train_forward_workspace",
    "cake_causal_conv1d_train_forward_workspace_bytes",
    "cake_causal_conv1d_train_backward_workspace",
    "cake_causal_conv1d_train_backward_workspace_bytes",
    "is_cake_causal_conv1d_train_supported",
)


@pytest.mark.cpu_only
def test_public_names_carry_the_mode_and_take_no_mode_keyword() -> None:
    """Every public name of ``__all__`` but the constants carries the mode word ``nondet``; no
    entry point and no parameter of the autograd function's forward is a ``deterministic`` keyword; the names before
    the rename are gone (no alias)."""

    for name in PUBLIC:
        assert name in pkg.__all__, name
        assert (
            "deterministic" not in inspect.signature(getattr(pkg, name)).parameters
        ), name
    assert "CakeCausalConv1dNondetFunction" in pkg.__all__
    assert (
        "deterministic"
        not in inspect.signature(pkg.CakeCausalConv1dNondetFunction.forward).parameters
    )
    for name in pkg.__all__:
        assert name.isupper() or "nondet" in name.lower(), name
    for name in OLD_NAMES:
        assert name not in pkg.__all__ and not hasattr(pkg, name), name


@pytest.mark.cpu_only
def test_a_mode_keyword_is_an_ordinary_type_error() -> None:
    """``deterministic=`` (either value) is Python's ordinary ``TypeError`` for an unexpected keyword at every entry
    point, raised before any argument, device or rendering check, so it needs neither a device nor ``build.json``;
    the autograd function takes no positional mode argument after ``bwd_grid_mult``."""

    x = torch.empty(8, 16, dtype=torch.bfloat16)
    w = torch.empty(16, 4, dtype=torch.float32)
    cu = torch.tensor([0, 8], dtype=torch.int32)
    cpu = torch.device("cpu")
    calls = {
        "forward": lambda **d: pkg.cake_causal_conv1d_nondet_train_forward(x, w, cu, **d),
        "backward": lambda **d: pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x, **d),
        "autograd": lambda **d: pkg.cake_causal_conv1d_nondet(x, w, cu, **d),
        "fwd_ws": lambda **d: pkg.cake_causal_conv1d_nondet_train_forward_workspace(8, 16, cpu, **d),
        "fwd_ws_bytes": lambda **d: pkg.cake_causal_conv1d_nondet_train_forward_workspace_bytes(8, 16, cpu, **d),
        "bwd_ws": lambda **d: pkg.cake_causal_conv1d_nondet_train_backward_workspace(8, 16, cpu, **d),
        "bwd_ws_bytes": lambda **d: pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(8, 16, cpu, **d),
    }  # fmt: skip
    for name, call in calls.items():
        for value in (True, False):
            with pytest.raises(TypeError) as info:
                call(deterministic=value)
            assert "unexpected keyword argument 'deterministic'" in str(info.value), (
                name
            )
    with pytest.raises(TypeError):
        pkg.CakeCausalConv1dNondetFunction.apply(x, w, cu, None, "bf16", 1, False)


@pytest.mark.cpu_only
def test_alert_helper_in_both_torch_modes(monkeypatch) -> None:
    """The backend's own non-determinism alert (no private PyTorch hook: ``torch._C._alert_not_deterministic`` is
    absent from some PyTorch builds, so it is removed here): silent with torch's deterministic algorithms off, a
    ``RuntimeError`` naming the entry and the deterministic package when on, a ``UserWarning`` with the same naming
    under ``warn_only=True``.  The KCP module uses the same helper."""

    conv_kcp = cake_package(KEY, "kcp")

    assert conv_kcp._alert_nondeterministic is pkg._alert_nondeterministic
    monkeypatch.delattr(torch._C, "_alert_not_deterministic", raising=False)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(False)
        with warnings_recorded() as caught:
            pkg._alert_nondeterministic("probe_entry")
        assert caught == []
        torch.use_deterministic_algorithms(True)
        with pytest.raises(
            RuntimeError,
            match="probe_entry does not have a deterministic implementation",
        ) as info:
            pkg._alert_nondeterministic("probe_entry")
        assert "flashinfer.cake_causal_conv1d_det_train" in str(info.value)
        assert "warn_only=True" in str(info.value)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(
            UserWarning,
            match="probe_entry does not have a deterministic implementation",
        ) as record:
            pkg._alert_nondeterministic("probe_entry")
        assert any(
            "flashinfer.cake_causal_conv1d_det_train" in str(w.message) for w in record
        )
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


@pytest.mark.cpu_only
def test_backward_entries_alert_before_any_device_work(monkeypatch) -> None:
    """Under torch's deterministic mode the single-rank backward alerts after argument validation and before any
    allocation or launch, and the forward does not alert.  Checked without a device: zero-token CPU tensors with the
    device / placement checks stubbed (the zero-token path returns before any device query)."""

    monkeypatch.setattr(pkg, "_device_arch", lambda device: "sm_100a")
    monkeypatch.setattr(pkg, "_check_placement", lambda *a, **k: None)
    launched = []
    monkeypatch.setattr(pkg, "run_module", lambda *a, **k: launched.append(a))
    x = torch.empty(0, 16, dtype=torch.bfloat16)
    w = torch.zeros(16, 4, dtype=torch.float32)
    cu = torch.tensor([0, 0], dtype=torch.int32)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(
            RuntimeError, match="cake_causal_conv1d_nondet_train_backward does not have"
        ):
            pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x)
        # The forward runs the deterministic forward programs: no alert.
        y = pkg.cake_causal_conv1d_nondet_train_forward(x, w, cu)
        assert tuple(y.shape) == (0, 16)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(
            UserWarning, match="cake_causal_conv1d_nondet_train_backward does not have"
        ):
            dx, dw = pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x)
        assert tuple(dx.shape) == (0, 16) and torch.equal(dw, torch.zeros_like(dw))
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
    assert launched == []


@pytest.mark.cpu_only
def test_unrendered_package_reports_unsupported() -> None:
    assert cake_jit.is_rendered() is (PACKAGE_DIR / "build.json").is_file()
    if not cake_jit.is_rendered():
        assert (
            pkg.is_cake_causal_conv1d_nondet_train_supported(torch.device("cpu"))
            is False
        )
        with pytest.raises(RuntimeError):
            cake_build.manifest()


# ----------------------------------------------------------------------------------------------------------------
# Wiring.


@pytest.mark.cpu_only
def test_package_wiring() -> None:
    assert (
        pkg.__name__ == "flashinfer.experimental.cake.cake_causal_conv1d_nondet_train"
    )
    tree = ast.parse((REPO / "flashinfer/experimental/cake/__init__.py").read_text())
    assert ast.get_docstring(tree)
    assert not [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert len(tree.body) == 1, "flashinfer.experimental.cake holds a docstring only"
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    package_data = data["tool"]["setuptools"]["package-data"]
    assert package_data[pkg.__name__] == ["csrc/**", "build.json"]


@pytest.mark.cpu_only
def test_build_manifest_lists_every_program() -> None:
    """Once rendered: ``build.json`` names this package, carries exactly one entry per (program, architecture) of
    the single-rank and KCP registries, each with its kernel and binding sources under ``csrc/`` and their digests,
    IEEE division / square root and no fast math; no record is a placeholder any more."""

    if not cake_jit.is_rendered():
        pytest.skip("build.json is not rendered yet (TODO-EXPORT)")
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    referenced = set()
    for registry in (cake_jit, kcp_jit):
        for name, record in registry.MODULES.items():
            assert "TODOEXPORT" not in name, name
            assert set(record["arches"]) <= set(manifest["arch_flags"]), name
            for arch in ARCHES:
                library = f"{name}_{arch}"
                if arch not in record["arches"]:
                    assert library not in libraries, library
                    continue
                entry = libraries[registry.library_name(name, arch)]
                assert entry["arch"] == arch and entry["entry"] == "run", library
                assert entry["sources"] == [
                    f"csrc/{name}_kernel.cu",
                    f"csrc/{name}_binding.cu",
                ]
                assert "--use_fast_math" not in entry["nvcc_flags"], library
                assert {"--prec-div=true", "--prec-sqrt=true", "--ftz=false"} <= set(
                    entry["nvcc_flags"]
                )
                for source in entry["sources"]:
                    path = PACKAGE_DIR / source
                    assert path.is_file(), source
                    assert (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        == entry["sha256"][source]
                    ), source
                referenced.add(library)
    assert referenced == set(libraries)


def _manifest_checker():
    """``scripts/check_cake_build_manifests.py`` of this checkout, loaded as a module (it never imports flashinfer)."""

    path = REPO / "scripts" / "check_cake_build_manifests.py"
    if not path.is_file():
        pytest.skip(
            "scripts/check_cake_build_manifests.py is not beside this test (installed package)"
        )
    spec = importlib.util.spec_from_file_location("check_cake_build_manifests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("check_cake_build_manifests", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu_only
def test_manifest_checker_lists_the_package() -> None:
    """The build-manifest checker lists this package (its ``cake_jit.py`` and the KCP registry ``kcp_jit.py``) and
    its vendored gate finds the package's suite key; once rendered, the in-tree checks report nothing."""

    checker = _manifest_checker()
    try:
        path = str(PACKAGE_DIR.relative_to(REPO))
    except ValueError:
        pytest.skip(
            f"the package is imported from {PACKAGE_DIR}, outside this checkout"
        )
    entries = [p for p in checker.PACKAGES if p.name == pkg.__name__]
    assert len(entries) == 1, entries
    (entry,) = entries
    assert (entry.path, entry.loader) == (path, "cake_jit.py")
    assert entry.refs is checker.refs_with_kcp
    assert checker.suite_key(REPO, entry) == KEY
    if not cake_jit.is_rendered():
        pytest.skip("build.json is not rendered yet (TODO-EXPORT)")
    assert checker.run_in_tree(REPO, [entry]) == []
