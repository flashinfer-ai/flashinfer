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

The KCP registry (``kcp_jit``) of the non-deterministic causal convolution training package against the
deterministic package's: the forward and head programs are the deterministic ones (only their names differ); the
main backward programs keep the deterministic route positions, selection rule, build geometry and arguments, with
the ``partial`` rows replaced by zeroed FP32 accumulators (kernel parameter ``dw_acc``, logical ``acc``) and uint32
``work`` ticket counters, an ordinary occupancy-sized
launch (no cooperative attribute, no grid barrier) and the workspace size of the legacy channel-tile rule the KCP
path plans with.  No device is needed; nothing is built or launched.
"""

from __future__ import annotations

import re

import pytest

from tests.test_helpers.cake_package import cake_package

KEY = "cake_causal_conv1d_nondet_train"
cake_jit = cake_package(KEY, "cake_jit")
kcp_jit = cake_package(KEY, "kcp_jit")
det_kcp_jit = cake_package("cake_causal_conv1d_det_train", "kcp_jit")

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
NAME_FIELDS = ("kernel", "kernel_symbol", "library")
PROGRAM_NAME = re.compile(
    r"cake_causal_conv1d_train_(?:TODOEXPORT_[a-z0-9_]+|[0-9a-f]{20})_kcp"
)
CHANNELS = (
    8,
    128,
    256,
    504,
    512,
    768,
    1016,
    1024,
    1160,
    1280,
    1344,
    1536,
    2048,
    4096,
    8192,
)
ACCUMULATORS = {"acc": "float32", "kind": "accumulators", "taps": 4, "work": "uint32"}


def _accumulator_rule(rule: dict) -> dict:
    """``rule`` without its ``slots``, which must name one copy (the KCP programs keep one FP32 accumulator)."""

    rule = dict(rule)
    assert rule.pop("slots", None) == 1, "a KCP accumulators rule names slots 1"
    return rule


def _strip(record: dict, *fields: str) -> dict:
    return {key: value for key, value in record.items() if key not in fields}


def _args(record: dict) -> set[str]:
    return {name for kind, name in record["arg_plan"] if kind != "grid"}


def _pairs():
    for arch in ARCHES:
        for kind in kcp_jit.KINDS:
            new, det = kcp_jit.ROUTES[arch][kind], det_kcp_jit.ROUTES[arch][kind]
            assert len(new) == len(det), (arch, kind)
            for name, det_name in zip(new, det, strict=True):
                yield arch, kind, name, det_name


@pytest.mark.cpu_only
def test_routes_mirror_the_deterministic_kcp_registry() -> None:
    assert kcp_jit.KINDS == det_kcp_jit.KINDS == ("kcp_fwd", "kcp_bwd", "kcp_bwd_head")
    assert kcp_jit.ARCHES == det_kcp_jit.ARCHES == ARCHES
    routed = {
        name
        for by_kind in kcp_jit.ROUTES.values()
        for names in by_kind.values()
        for name in names
    }
    assert set(kcp_jit.MODULES) == routed
    assert not set(kcp_jit.MODULES) & set(det_kcp_jit.MODULES)
    assert not set(kcp_jit.MODULES) & set(cake_jit.MODULES), (
        "the KCP programs are not the zero-history ones"
    )
    for name, record in kcp_jit.MODULES.items():
        assert PROGRAM_NAME.fullmatch(name), name
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["grid_rule"]["cooperative"] is False, name
        assert record["launch"]["cooperative"] is False, name
    assert (*kcp_jit.KINDS, "kcp_bwd_dyn", "kcp_bwd_halo") == kcp_jit.ROUTE_KINDS
    for arch in ARCHES:
        assert set(kcp_jit.ROUTES[arch]) == set(kcp_jit.ROUTE_KINDS), arch


@pytest.mark.cpu_only
def test_forward_and_head_records_are_the_deterministic_ones() -> None:
    for arch, kind, name, det_name in _pairs():
        if kind == "kcp_bwd":
            continue
        assert _strip(kcp_jit.MODULES[name], *NAME_FIELDS) == _strip(
            det_kcp_jit.MODULES[det_name], *NAME_FIELDS
        ), (arch, kind, name)


@pytest.mark.cpu_only
def test_main_backward_records_are_ordinary_launches_with_accumulators() -> None:
    vocabulary = set(kcp_jit.LOGICAL_TENSORS) | set(kcp_jit.LOGICAL_SCALARS)
    for arch, kind, name, det_name in _pairs():
        if kind != "kcp_bwd":
            continue
        record, det = kcp_jit.MODULES[name], det_kcp_jit.MODULES[det_name]
        label = (arch, name)
        assert record["role"] == "kcp_bwd" and record["arches"] == det["arches"], label
        assert record["ffi_entry"] == det["ffi_entry"] == "run", label
        assert _args(record) == (_args(det) - {"partial"}) | {"dw_acc", "work"}, label
        assert (
            record["logical"]["dw_acc"] == "acc" and record["logical"]["work"] == "work"
        ), label
        assert [k for k, _ in record["arg_plan"][-3:]] == ["grid", "grid", "grid"], (
            label
        )
        assert set(record["logical"]) == _args(record), label
        assert set(record["logical"].values()) <= vocabulary, label
        grid = record["grid_rule"]
        assert grid["kind"] == "occupancy" and grid["cooperative"] is False, label
        assert grid["geometry"] == det["grid_rule"]["geometry"], label
        # ``grid_mult`` (a rendered leaf): one positive factor per architecture of the record.
        assert set(grid["grid_mult"]) == set(record["arches"]), label
        assert all(isinstance(f, int) and f >= 1 for f in grid["grid_mult"].values()), (
            label
        )
        assert isinstance(grid["ctas_per_sm"], int) and grid["ctas_per_sm"] >= 1, label
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, label
        assert launch["block"] == det["launch"]["block"], label
        assert (
            isinstance(launch["dynamic_smem_bytes"], int)
            and launch["dynamic_smem_bytes"] >= 0
        )
        assert kcp_jit.select_rule(record) == det_kcp_jit.select_rule(det), label
        assert _accumulator_rule(record["workspace_rule"]) == ACCUMULATORS, label


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_selection_matches_the_deterministic_kcp_registry(arch) -> None:
    for kind in kcp_jit.KINDS:
        new, det = kcp_jit.ROUTES[arch][kind], det_kcp_jit.ROUTES[arch][kind]
        for C in CHANNELS:
            (name,) = kcp_jit.selected_modules(arch, kind, C=C)
            (det_name,) = det_kcp_jit.selected_modules(arch, kind, C=C)
            assert new.index(name) == det.index(det_name), (arch, kind, C)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_kcp_backward_workspace(arch) -> None:
    """The KCP backward's ``acc`` / ``work`` bytes follow the zero-history layout under the legacy channel-tile rule
    (``ct_vec = min(C / 8, 128)``): ``align16(align16(16 C) + 4 n_ch_tiles)``; one FP32 owner per route."""

    owners = kcp_jit.workspace_owners(arch)
    assert [o["kernel"] for o in owners] == list(kcp_jit.route_modules(arch, "kcp_bwd"))
    assert (
        _accumulator_rule(kcp_jit.workspace_record(arch)["workspace_rule"])
        == ACCUMULATORS
    )
    for C in CHANNELS:
        n_ch_tiles = -(-(C // 8) // min(C // 8, 128))
        layout = kcp_jit.accumulator_layout(owners[0], C, n_ch_tiles)
        assert layout["acc_dtype"] == "float32" and layout["acc_elems"] == 4 * C
        expected = -(-(-(-16 * C // 16) * 16 + 4 * n_ch_tiles) // 16) * 16
        assert kcp_jit.workspace_bytes(arch, C) == layout["total_bytes"] == expected, (
            arch,
            C,
        )


@pytest.mark.cpu_only
def test_head_grid_is_the_deterministic_rule() -> None:
    for arch in ARCHES:
        (head,) = kcp_jit.ROUTES[arch]["kcp_bwd_head"]
        (det_head,) = det_kcp_jit.ROUTES[arch]["kcp_bwd_head"]
        for n_frag in (0, 1, 3, 17):
            for C in CHANNELS:
                assert kcp_jit.head_grid(
                    kcp_jit.MODULES[head], n_frag=n_frag, C=C
                ) == det_kcp_jit.head_grid(
                    det_kcp_jit.MODULES[det_head], n_frag=n_frag, C=C
                )


@pytest.mark.cpu_only
def test_overlapped_backward_records() -> None:
    """The overlapped backward's two routes: one record each on every architecture.  The main program is an
    ordinary ``co_resident`` launch of one CTA per SM (the host passes the grid) that owns an ``accumulators``
    workspace of ``slots`` FP32 copies of four taps, its logical map names ``acc``, ``tile_tk``, ``claims`` and the
    two TMA tensors; the tail-row program has the head program's geometry and no workspace."""

    vocabulary = set(kcp_jit.LOGICAL_TENSORS) | set(kcp_jit.LOGICAL_SCALARS)
    for arch in ARCHES:
        (main,) = kcp_jit.route_modules(arch, kcp_jit.DYN_KIND)
        (halo,) = kcp_jit.route_modules(arch, kcp_jit.HALO_KIND)
        (head,) = kcp_jit.route_modules(arch, "kcp_bwd_head")
        for name, role in ((main, "kcp_bwd_dyn"), (halo, "kcp_bwd_halo")):
            record = kcp_jit.MODULES[name]
            label = (arch, name)
            assert record["role"] == role and arch in record["arches"], label
            assert record["ffi_entry"] == "run", label
            assert set(record["logical"]) == _args(record), label
            assert set(record["logical"].values()) <= vocabulary, label
            assert [k for k, _ in record["arg_plan"][-3:]] == ["grid"] * 3, label
            assert record["grid_rule"]["kind"] == "co_resident", label
            assert record["launch"]["cooperative"] is False, label
        record = kcp_jit.MODULES[main]
        logical = record["logical"]
        assert logical["dw_acc"] == "acc" and logical["tile_tk"] == "tile_tk"
        assert logical["claims"] == "claims"
        assert logical["x_map"] == "x" and logical["dy_map"] == "dy"
        assert {logical[k] for k in ("M", "NL", "P", "L", "n_ctas")} == {
            "M",
            "NL",
            "P",
            "L",
            "n_ctas",
        }
        assert record["grid_rule"]["ctas_per_sm"] == 1
        rule = record["workspace_rule"]
        assert rule["kind"] == "accumulators" and rule["acc"] == "float32"
        assert rule["taps"] == 4 and rule["work"] == "uint32"
        assert isinstance(rule["slots"], int) and rule["slots"] >= 1
        record = kcp_jit.MODULES[halo]
        assert record["workspace_rule"] is None
        assert (
            record["grid_rule"]["geometry"]
            == kcp_jit.MODULES[head]["grid_rule"]["geometry"]
        )
        assert "dpre_tail" in record["logical"] and "tail_row_off" in record["logical"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("sms", [132, 148, 160, 188])
def test_dyn_plan_meets_the_kernel_requirements(sms) -> None:
    """:func:`kcp_jit.dyn_plan` gives every work plan the main program's ``kernel_requires`` admit: ``L`` lanes of
    ``P`` per tile of 2048 channels, at most 256; one CTA per SM and at least one per lane; units of ``M`` tokens (a
    multiple of 8, at least 24); ``NL`` units per lane covering the rank's tokens; the claim words' 16-bit counts."""

    for C in (8, 1024, 2048, 2056, 8192, 10240, 32768, 65536):
        n_tiles = -(-C // kcp_jit.DYN_TILE)
        for S in (1, 7, 64, 1003, 16384, 32768, 1 << 20, 1 << 23):
            p = kcp_jit.dyn_plan(S, C, sms)
            label = (sms, C, S, p)
            assert p["n_tiles"] == n_tiles and p["L"] == n_tiles * p["P"], label
            assert 1 <= p["P"] <= 512 and p["L"] <= kcp_jit.DYN_MAX_LANES, label
            assert p["n_ctas"] == max(p["L"], sms), label
            assert p["M"] % 8 == 0 and p["M"] >= 24, label
            assert 1 <= p["NL"] <= 32768 and p["P"] * p["NL"] * p["M"] >= S, label
            assert (p["L"] + 1) * p["NL"] + p["n_ctas"] <= 65535, label
            if sms * kcp_jit.DYN_UNIT <= S and n_tiles <= sms:
                assert p["L"] >= sms - n_tiles + 1, label
    for S, C in ((0, 1024), (16, 0), (16, kcp_jit.DYN_MAX_CHANNELS + 8)):
        with pytest.raises(ValueError):
            kcp_jit.dyn_plan(S, C, 148)
    assert kcp_jit.dyn_applies(kcp_jit.DYN_MAX_CHANNELS, 1)
    assert not kcp_jit.dyn_applies(kcp_jit.DYN_MAX_CHANNELS + 8, 1)
    assert not kcp_jit.dyn_applies(1024, 0)


@pytest.mark.cpu_only
def test_dyn_workspace_layout() -> None:
    """The main program's workspace: ``slots`` copies of ``[C, 4]`` FP32 accumulators, then eight words per tile and
    ``L + 1`` claim words."""

    for arch in ARCHES:
        (main,) = kcp_jit.route_modules(arch, kcp_jit.DYN_KIND)
        record = kcp_jit.MODULES[main]
        slots = record["workspace_rule"]["slots"]
        for C in (8, 2048, 10240, 65536):
            p = kcp_jit.dyn_plan(32768, C, 148)
            layout = kcp_jit.dyn_workspace_layout(record, C, p["n_tiles"], p["L"])
            assert layout["acc_dtype"] == "float32"
            assert layout["acc_elems"] == slots * 4 * C
            assert layout["work_dtype"] == "uint32"
            assert layout["work_words"] == 8 * p["n_tiles"] + p["L"] + 1
            assert layout["work_offset"] % 16 == 0
            assert (
                layout["total_bytes"]
                >= layout["work_offset"] + 4 * layout["work_words"]
            )
