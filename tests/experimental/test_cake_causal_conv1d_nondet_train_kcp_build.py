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
deterministic package's: the forward, history-row and head programs are the deterministic ones (only their names
differ; the history-row program, route ``kcp_fwd_halo`` of the two-launch forward, is a route of its own outside
``KINDS``, and the forward's single-rank main program is the zero-history registry's ``fwd`` route); the main backward
programs keep the deterministic route positions, selection rule, build geometry and arguments, with
the ``partial`` rows replaced by zeroed FP32 accumulators (kernel parameter ``dw_acc``, logical ``acc``) and uint32
``work`` ticket counters, an ordinary occupancy-sized
launch (no cooperative attribute, no grid barrier) and the workspace size of the legacy channel-tile rule the KCP
path plans with.  No device is needed; nothing is built or launched.
"""

from __future__ import annotations

import re

import pytest

from tests.cake.test_helpers.cake_package import cake_package

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
    assert (
        *kcp_jit.KINDS,
        "kcp_bwd_dyn",
        "kcp_bwd_halo",
        "kcp_fwd_halo",
    ) == kcp_jit.ROUTE_KINDS
    assert kcp_jit.FWD_HALO_KIND == "kcp_fwd_halo"
    assert kcp_jit.FWD_HALO_KIND not in kcp_jit.KINDS
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
def test_history_row_record_is_the_deterministic_one() -> None:
    """The two-launch forward's history-row program (route ``kcp_fwd_halo``, one record on every architecture): the
    det package's record of that route under this package's prefix (the same twenty hex digits), with the head
    program's grid rule."""

    for arch in ARCHES:
        (name,) = kcp_jit.route_modules(arch, kcp_jit.FWD_HALO_KIND)
        (det_name,) = det_kcp_jit.route_modules(arch, "kcp_fwd_halo")
        assert name == det_name.replace(
            "cake_causal_conv1d_det_train_", "cake_causal_conv1d_train_"
        ), (arch, name, det_name)
        assert _strip(kcp_jit.MODULES[name], *NAME_FIELDS) == _strip(
            det_kcp_jit.MODULES[det_name], *NAME_FIELDS
        ), (arch, name)
        assert kcp_jit.selected_modules(arch, kcp_jit.FWD_HALO_KIND, C=8192) == (name,)
        (head,) = kcp_jit.route_modules(arch, "kcp_bwd_head")
        for n_frag in (0, 1, 3, 17):
            for C in CHANNELS:
                assert kcp_jit.head_grid(
                    kcp_jit.MODULES[name], n_frag=n_frag, C=C
                ) == kcp_jit.head_grid(kcp_jit.MODULES[head], n_frag=n_frag, C=C)


@pytest.mark.cpu_only
def test_two_launch_forward_programs() -> None:
    """The two-launch forward's programs (``kcp`` module docstring): the history-row program is the ``kcp_fwd_halo``
    route's one record and takes the arguments the host passes (``kcp._fwd_repair``); the single-rank main program is
    a record of this package's zero-history registry's ``fwd`` route (``kcp.FWD_MAIN_ROUTE``) on every architecture,
    the det package's zero-history record of the same route position, each record of that route takes only ``x``,
    ``w``, ``cu``, ``y``, its ``work`` counters (``tile_counters``) and the plan scalars (``kcp._fwd_main``), and
    launches the co-resident grid (no reserved SM), so the host can launch whichever record
    ``cake_jit.selected_modules`` picks for the call's ``C`` and token count."""

    kcp = cake_package(KEY, "kcp")
    det_cake_jit = cake_package("cake_causal_conv1d_det_train", "cake_jit")
    assert kcp.KCP_FWD_HALO == "kcp_fwd_halo" and kcp.FWD_MAIN_ROUTE == "fwd"
    passed = {"x", "w", "cu", "y", "hist", "hist_row_off", "C", "num_docs"}
    for arch in ARCHES:
        (repair,) = kcp_jit.route_modules(arch, kcp.KCP_FWD_HALO)
        assert set(kcp_jit.MODULES[repair]["logical"].values()) == passed
    main_values = {"x", "w", "cu", "y", "work", *cake_jit.PLAN_SCALARS}
    for arch in ARCHES:
        names = cake_jit.route_modules(arch, kcp.FWD_MAIN_ROUTE)
        assert names, arch
        for name in names:
            record = cake_jit.MODULES[name]
            assert record["role"] == "fwd" and arch in record["arches"], (arch, name)
            assert set(record["logical"].values()) <= main_values, (arch, name)
            assert record["workspace_rule"]["kind"] == "tile_counters", (arch, name)
            assert record["grid_rule"]["kind"] == "co_resident", (arch, name)
            assert "reserve_sms" not in record["grid_rule"], (arch, name)
            assert not record["grid_rule"]["cooperative"], (arch, name)
        for C, S in ((256, 1), (256, 65536), (2048, 1), (8192, 32768), (10240, 32768)):
            (name,) = cake_jit.selected_modules(arch, kcp.FWD_MAIN_ROUTE, C=C, S=S)
            assert name in names, (arch, C, S)
            # the det package's single-rank main program for the same call: the same program (twenty hex digits)
            (det_name,) = det_cake_jit.selected_modules(
                arch, kcp.FWD_MAIN_ROUTE, C=C, S=S
            )
            assert name.rsplit("_", 1)[-1] == det_name.rsplit("_", 1)[-1], (
                arch,
                C,
                S,
                name,
                det_name,
            )


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
    workspace of ``slots`` FP32 copies of four taps, its logical map names ``acc``, ``tile_tk``, ``claims``,
    ``departs``, the two TMA tensors and its two tail tables (``tail_row_off`` and ``skip_row_off``, the rows it
    leaves to the tail-row program); the tail-row program has the head program's geometry and no workspace."""

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
        assert logical["claims"] == "claims" and logical["departs"] == "departs"
        assert logical["x_map"] == "x" and logical["dy_map"] == "dy"
        assert logical["tail_row_off"] == "tail_row_off"
        assert logical["skip_row_off"] == "skip_row_off"
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
            if sms * p["M"] <= S and n_tiles <= sms:
                assert p["L"] >= sms - n_tiles + 1, label
    for S, C in ((0, 1024), (16, 0), (16, kcp_jit.DYN_MAX_CHANNELS + 8)):
        with pytest.raises(ValueError):
            kcp_jit.dyn_plan(S, C, 148)
    assert kcp_jit.dyn_applies(kcp_jit.DYN_MAX_CHANNELS, 1)
    assert not kcp_jit.dyn_applies(kcp_jit.DYN_MAX_CHANNELS + 8, 1)
    assert not kcp_jit.dyn_applies(1024, 0)


@pytest.mark.cpu_only
def test_dyn_plan_unit_size() -> None:
    """:func:`kcp_jit.dyn_plan` takes the 64-token plan when it leaves the last lane of each tile without tokens,
    or the 56-token plan (``DYN_IDLE_UNIT``) when that one leaves more idle CTAs (``n_ctas`` less the lanes with
    tokens; the 64-token plan on a tie); otherwise the unit size ``M`` (a multiple of 8 from 64 to 128) whose fitting plan has the least lane cost
    ``NL * M`` tokens plus ``NL`` fixed unit costs plus a tail cost per token of a unit, the smaller ``M`` on a tie;
    above that range it doubles ``M`` from 256 until the claim counts fit."""

    def lane_cost(p):
        unit = 10 * p["M"] + kcp_jit.DYN_UNIT_COST_TENTHS
        return p["NL"] * unit + kcp_jit.DYN_TAIL_COST_TENTHS * p["M"]

    # the customer per-rank shapes.  148-SM B200 (every lane busy; costs in tenths of a token): one tensor of 10240
    # channels, 32768 tokens (P 29: 11 units of 104 tokens, 11 * 1072 + 3 * 104 = 12104, against 13 x 88: 12120
    # and 18 x 64: 12288) and 8192 channels, 16384 tokens (P 37: 7 x 64, 7 * 672 + 192 = 4896, against 4 x 112:
    # 4944).  152 SMs (GB300, GB200) and 212 SMs (VR200): the 64-token lanes of 1152, 448, 832 and 320 tokens fill
    # 29 of 30, 37 of 38, 40 of 42 and 52 of 53 lanes, so M is 64 unless 56-token lanes leave more CTAs idle: they
    # leave 2, 4 (a tie), 2 and 16 idle against 7, 4, 12 and 4, so only the last (VR200, 8192 channels) takes
    # 6 units of 56 tokens (49 of 53 lanes busy).
    for (S, C, sms), want in (
        ((32768, 10240, 148), {"M": 104, "NL": 11, "P": 29, "L": 145, "n_ctas": 148}),
        ((16384, 8192, 148), {"M": 64, "NL": 7, "P": 37, "L": 148, "n_ctas": 148}),
        ((32768, 10240, 152), {"M": 64, "NL": 18, "P": 30, "L": 150, "n_ctas": 152}),
        ((16384, 8192, 152), {"M": 64, "NL": 7, "P": 38, "L": 152, "n_ctas": 152}),
        ((32768, 10240, 212), {"M": 64, "NL": 13, "P": 42, "L": 210, "n_ctas": 212}),
        ((16384, 8192, 212), {"M": 56, "NL": 6, "P": 53, "L": 212, "n_ctas": 212}),
    ):
        p = kcp_jit.dyn_plan(S, C, sms)
        assert {k: p[k] for k in want} == want, (S, C, sms, p)
    branches = {"idle": 0, "idle_unit": 0, "cost": 0, "doubling": 0}
    for sms in (132, 148, 152, 188, 212):
        for C in (8, 2048, 8192, 10240, 65536):
            n_tiles = -(-C // kcp_jit.DYN_TILE)
            for S in (1, 7, 1003, 4096, 16384, 32768, 100000, 1 << 20, 1 << 23):

                def plan_for(M):
                    lanes = min(sms // n_tiles, -(-S // M), 256 // n_tiles)
                    P = max(1, lanes)
                    per_lane = -(-S // P)
                    NL = -(-per_lane // M)
                    L = n_tiles * P
                    if (L + 1) * NL + max(L, sms) > 65535 or NL > 32768:
                        return None
                    n_ctas = max(L, sms)
                    return {"M": M, "NL": NL, "P": P, "L": L, "n_ctas": n_ctas}

                p = kcp_jit.dyn_plan(S, C, sms)
                p = {k: p[k] for k in ("M", "NL", "P", "L", "n_ctas")}
                fits = [q for q in map(plan_for, range(64, 129, 8)) if q is not None]
                label = (sms, C, S, p)

                def idle(q):
                    busy = min(q["P"], -(-S // (q["NL"] * q["M"])))
                    return q["n_ctas"] - n_tiles * busy

                base = plan_for(64)
                if base is not None and -(-S // (base["NL"] * 64)) < base["P"]:
                    small = plan_for(56)
                    if small is not None and idle(small) > idle(base):
                        branches["idle_unit"] += 1
                        assert p == small, label
                    else:
                        branches["idle"] += 1
                        assert p == base, label
                elif fits:
                    branches["cost"] += 1
                    best = min(lane_cost(q) for q in fits)
                    assert lane_cost(p) == best, label
                    tied = [q["M"] for q in fits if lane_cost(q) == best]
                    assert p["M"] == min(tied), label
                else:
                    # the first power of two from 256 whose plan fits (the previous doubling rule)
                    branches["doubling"] += 1
                    assert p["M"] >= 256 and p["M"] & (p["M"] - 1) == 0, label
                    assert p == plan_for(p["M"]), label
                    assert p["M"] == 256 or plan_for(p["M"] // 2) is None, label
    assert all(n > 0 for n in branches.values()), branches
    # a rank too long for any unit up to 128 tokens: the doubling from 256 (32 tiles, 4 lanes each, 256 units)
    assert kcp_jit.dyn_plan(1 << 23, 65536, 148)["M"] == 8192


@pytest.mark.cpu_only
def test_dyn_workspace_layout() -> None:
    """The main program's workspace: ``slots`` copies of ``[C, 4]`` FP32 accumulators, then eight words per tile,
    ``L`` claim words and one departure word."""

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
