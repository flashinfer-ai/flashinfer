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

The KCP registry (``kcp_jit``) of the Cake causal convolution training package: the records against the frozen
kernel ABIs and the identity rule, the routes and the selection by channel count, and the ``build.json`` entries of
the KCP programs.  Each backward program ships twice: the cooperative build (route ``kcp_bwd``) and its ``split``
build (route ``kcp_bwd_split``, FlashInfer issue #8365, built with ``pair=True``: the token arithmetic on packed
``f32x2`` channel pairs), an ordinary launch one SM short of co-resident (``reserve_sms``) whose ``dweight`` partial
rows the zero-history package's fixed-order reduce program sums.  The tail-row program (route ``kcp_bwd_halo``) of the
overlapped backward and the history-row program (route ``kcp_fwd_halo``) of the two-launch forward are routes of their
own, outside ``KINDS``; the two-launch forward's single-rank main program is the zero-history registry's ``fwd`` route.
No device is needed; nothing is built or launched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.cake.test_helpers.cake_package import cake_package

cake_build = cake_package("cake_causal_conv1d_det_train", "cake_build")
cake_jit = cake_package("cake_causal_conv1d_det_train", "cake_jit")
kcp_jit = cake_package("cake_causal_conv1d_det_train", "kcp_jit")

PACKAGE_DIR = Path(cake_build.__file__).parent
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
GRID = [["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]
_W = "sm100_sm103_sm107"
FWD = f"tart.causal_conv1d_kcp_fwd_bf16_silu.{_W}.causal_conv1d_kcp_fwd"
BWD = f"tart.causal_conv1d_kcp_bwd_bf16_silu.{_W}.causal_conv1d_kcp_bwd"
HEAD = f"tart.causal_conv1d_kcp_bwd_bf16_silu.{_W}.causal_conv1d_kcp_bwd_head"
# the tail-row program's identity: its kernel's key, with no build keyword
HALO = "causal_conv1d_kcp_bwd_bf16_silu/causal_conv1d_kcp_bwd_halo"
# the history-row program's identity, likewise
FWD_HALO = "causal_conv1d_kcp_fwd_bf16_silu/causal_conv1d_kcp_fwd_halo"

FWD_PLAN = [
    ["buffer", "x"], ["buffer", "w"], ["buffer", "cu"], ["buffer", "y"], ["buffer", "hist"], ["buffer", "hist_row_off"],
    ["parameter", "S"], ["parameter", "C"], ["parameter", "num_docs"], ["parameter", "search_steps"],
    ["parameter", "ct_vec"], ["parameter", "n_sub"], ["parameter", "n_ch_tiles"], *GRID,
]  # fmt: skip
BWD_SCALARS = [
    ["parameter", "S"], ["parameter", "C"], ["parameter", "num_docs"], ["parameter", "search_steps"],
    ["parameter", "ct_vec"], ["parameter", "n_sub"], ["parameter", "n_ch_tiles"],
]  # fmt: skip
BWD_BUFFERS = [
    ["buffer", "x"], ["buffer", "w"], ["buffer", "cu"], ["buffer", "dy"], ["buffer", "dx"], ["buffer", "dw"],
    ["buffer", "partial"], ["buffer", "hist"], ["buffer", "hist_row_off"], ["buffer", "dpre_tail"],
    ["buffer", "tail_row_off"],
]  # fmt: skip
BWD_PLAN = [*BWD_BUFFERS, *BWD_SCALARS, *GRID]
BWD_TMA_PLAN = [
    *BWD_BUFFERS,
    *BWD_SCALARS,
    ["buffer", "x_map"],
    ["buffer", "dy_map"],
    *GRID,
]
HEAD_PLAN = [
    ["buffer", "x"], ["buffer", "w"], ["buffer", "cu"], ["buffer", "dy"], ["buffer", "hist"], ["buffer", "hist_row_off"],
    ["buffer", "dpre_head"], ["buffer", "head_row_off"], ["parameter", "S"], ["parameter", "C"],
    ["parameter", "num_docs"], *GRID,
]  # fmt: skip
HALO_PLAN = [
    ["buffer", "x"], ["buffer", "w"], ["buffer", "cu"], ["buffer", "dy"], ["buffer", "dx"], ["buffer", "hist"],
    ["buffer", "hist_row_off"], ["buffer", "dpre_tail"], ["buffer", "tail_row_off"], ["parameter", "C"],
    ["parameter", "num_docs"], *GRID,
]  # fmt: skip
FWD_HALO_PLAN = [
    ["buffer", "x"], ["buffer", "w"], ["buffer", "cu"], ["buffer", "y"], ["buffer", "hist"], ["buffer", "hist_row_off"],
    ["parameter", "C"], ["parameter", "num_docs"], *GRID,
]  # fmt: skip

REDUCE = "cake_causal_conv1d_det_train_86b5ecfbd90aca4b52e5"
FALLBACK = "cake_causal_conv1d_det_train_5288d132a5cc927ae729_kcp"
R200_FALLBACK = "cake_causal_conv1d_det_train_6ea6d5fc9e7fa3f2d039_kcp"
BIG = "cake_causal_conv1d_det_train_7485529a1755e5bd3036_kcp"
SPLIT_FALLBACK = "cake_causal_conv1d_det_train_83900e631708e690fbb0_kcp"
SPLIT_R200_FALLBACK = "cake_causal_conv1d_det_train_2fc855aa2942ba7bb841_kcp"
SPLIT_BIG = "cake_causal_conv1d_det_train_7403791c52dfad8697ca_kcp"
# sm_107a's big-channel split program: SPLIT_BIG's build with its stage's tokens in two rolled passes
SPLIT_BIG_ROLL = "cake_causal_conv1d_det_train_8225ee9b6f8caea425da_kcp"
TAIL = "cake_causal_conv1d_det_train_ae0c093e1e0ac46aec3f_kcp"
REPAIR = "cake_causal_conv1d_det_train_138311f76132280b1e94_kcp"
BWD_ROLES = ("kcp_bwd", "kcp_bwd_split")

# program name -> (module, build keywords, arches, role, plan, ctas_per_sm)
PROGRAMS = {
    "cake_causal_conv1d_det_train_675940a7a4fc0dba4b53_kcp": (
        FWD,
        {},
        ARCHES,
        "kcp_fwd",
        FWD_PLAN,
        4,
    ),
    "cake_causal_conv1d_det_train_5288d132a5cc927ae729_kcp": (
        BWD,
        {"vec": 8, "copy": "tma2d"},
        ("sm_100a", "sm_103a"),
        "kcp_bwd",
        BWD_TMA_PLAN,
        3,
    ),
    "cake_causal_conv1d_det_train_6ea6d5fc9e7fa3f2d039_kcp": (
        BWD,
        {"vec": 8, "min_blocks": 2},
        ("sm_107a",),
        "kcp_bwd",
        BWD_PLAN,
        2,
    ),
    "cake_causal_conv1d_det_train_7485529a1755e5bd3036_kcp": (
        BWD,
        {"vec": 8, "copy": "tma2d", "stage_tokens": 8, "min_blocks": 2},
        ARCHES,
        "kcp_bwd",
        BWD_TMA_PLAN,
        2,
    ),
    SPLIT_FALLBACK: (
        BWD,
        {"vec": 8, "copy": "tma2d", "split": True, "pair": True},
        ("sm_100a", "sm_103a"),
        "kcp_bwd_split",
        BWD_TMA_PLAN,
        3,
    ),
    SPLIT_R200_FALLBACK: (
        BWD,
        {"vec": 8, "min_blocks": 2, "split": True, "pair": True},
        ("sm_107a",),
        "kcp_bwd_split",
        BWD_PLAN,
        2,
    ),
    SPLIT_BIG: (
        BWD,
        {
            "vec": 8,
            "copy": "tma2d",
            "stage_tokens": 8,
            "min_blocks": 2,
            "split": True,
            "pair": True,
        },
        ("sm_100a", "sm_103a"),
        "kcp_bwd_split",
        BWD_TMA_PLAN,
        2,
    ),
    SPLIT_BIG_ROLL: (
        BWD,
        {
            "vec": 8,
            "copy": "tma2d",
            "stage_tokens": 8,
            "min_blocks": 2,
            "split": True,
            "pair": True,
            "fast_roll": True,
        },
        ("sm_107a",),
        "kcp_bwd_split",
        BWD_TMA_PLAN,
        2,
    ),
    "cake_causal_conv1d_det_train_17cd42fb05079bbbf840_kcp": (
        HEAD,
        {},
        ARCHES,
        "kcp_bwd_head",
        HEAD_PLAN,
        5,
    ),
    TAIL: (HALO, {}, ARCHES, "kcp_bwd_halo", HALO_PLAN, 5),
    REPAIR: (FWD_HALO, {}, ARCHES, "kcp_fwd_halo", FWD_HALO_PLAN, 5),
}
BIG_C_MIN = {"sm_100a": 512, "sm_103a": 512, "sm_107a": 1024}


def _identity(module: str, kwargs: dict) -> str:
    identity = {"build": {key: kwargs[key] for key in sorted(kwargs)}, "module": module}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"cake_causal_conv1d_det_train_{digest[:20]}_kcp"


@pytest.mark.cpu_only
def test_program_names_follow_the_identity_rule() -> None:
    assert set(kcp_jit.MODULES) == set(PROGRAMS)
    for name, (module, kwargs, *_rest) in PROGRAMS.items():
        assert _identity(module, kwargs) == name, name
    assert not set(kcp_jit.MODULES) & set(cake_jit.MODULES), (
        "the KCP programs are new builds, not the zero-history ones"
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_record_matches_the_frozen_abi(name: str) -> None:
    _module, _kwargs, arches, role, plan, ctas_per_sm = PROGRAMS[name]
    record = kcp_jit.MODULES[name]
    assert record["library"] == record["kernel"] == name
    assert record["kernel_symbol"] == f"kernel_{name}"
    assert record["arches"] == list(arches)
    assert record["role"] == role
    assert record["arg_plan"] == plan
    assert record["ffi_entry"] == "run"
    assert record["grid_rule"]["kind"] == "co_resident"
    assert record["grid_rule"]["ctas_per_sm"] == ctas_per_sm
    assert record["grid_rule"]["geometry"] == {"threads": 128, "vec": 8}
    assert record["grid_rule"]["cooperative"] is (role == "kcp_bwd")
    assert record["launch"]["cooperative"] is (role == "kcp_bwd")
    assert record["grid_rule"].get("reserve_sms", 0) == (
        1 if role == "kcp_bwd_split" else 0
    )
    assert (
        record["launch"]["block"] == (128, 1, 1)
        and record["launch"]["use_pdl"] is False
    )
    kept = {parameter for kind, parameter in record["arg_plan"] if kind != "grid"}
    assert set(record["logical"]) == kept
    vocabulary = set(kcp_jit.LOGICAL_TENSORS) | set(kcp_jit.LOGICAL_SCALARS)
    assert set(record["logical"].values()) <= vocabulary
    if role in BWD_ROLES:
        assert record["workspace_rule"] == {
            "kind": "partial_rows",
            "taps": kcp_jit.WIDTH,
        }
    else:
        assert record["workspace_rule"] is None
    if role == "kcp_bwd_split":
        # the reduce program the host launches after the main program: the zero-history package's default reduce
        assert record["reduce"] == REDUCE and REDUCE in cake_jit.MODULES
    else:
        assert "reduce" not in record


@pytest.mark.cpu_only
def test_routes_and_selection() -> None:
    assert kcp_jit.KINDS == ("kcp_fwd", "kcp_bwd", "kcp_bwd_head")
    assert kcp_jit.ROUTE_KINDS == (
        "kcp_fwd",
        "kcp_bwd",
        "kcp_bwd_split",
        "kcp_bwd_head",
        "kcp_bwd_halo",
        "kcp_fwd_halo",
    )
    assert kcp_jit.HALO_KIND == "kcp_bwd_halo"
    assert kcp_jit.HALO_KIND not in kcp_jit.KINDS
    assert "kcp_fwd_halo" not in kcp_jit.KINDS
    assert kcp_jit.BWD_KINDS == BWD_ROLES
    for arch in ARCHES:
        routes = kcp_jit.ROUTES[arch]
        assert set(routes) == set(kcp_jit.ROUTE_KINDS)
        assert routes["kcp_fwd"] == [
            "cake_causal_conv1d_det_train_675940a7a4fc0dba4b53_kcp"
        ]
        assert routes["kcp_bwd_head"] == [
            "cake_causal_conv1d_det_train_17cd42fb05079bbbf840_kcp"
        ]
        assert routes["kcp_bwd_halo"] == [TAIL]
        assert kcp_jit.selected_modules(arch, "kcp_bwd_halo", C=8192) == (TAIL,)
        assert routes["kcp_fwd_halo"] == [REPAIR]
        assert kcp_jit.selected_modules(arch, "kcp_fwd_halo", C=8192) == (REPAIR,)
        for kind, fallback, big in (
            ("kcp_bwd", R200_FALLBACK if arch == "sm_107a" else FALLBACK, BIG),
            (
                "kcp_bwd_split",
                SPLIT_R200_FALLBACK if arch == "sm_107a" else SPLIT_FALLBACK,
                SPLIT_BIG_ROLL if arch == "sm_107a" else SPLIT_BIG,
            ),
        ):
            assert routes[kind] == [fallback, big]
            assert kcp_jit.route_modules(arch, kind) == (fallback, big)
            assert kcp_jit.select_rule(kcp_jit.MODULES[big]) == {
                "channels": {a: BIG_C_MIN[a] for a in kcp_jit.MODULES[big]["arches"]},
                "kind": "channels_at_least",
            }
            assert kcp_jit.select_rule(kcp_jit.MODULES[fallback]) is None
            assert kcp_jit.selected_modules(arch, kind, C=BIG_C_MIN[arch] - 8) == (
                fallback,
            )
            assert kcp_jit.selected_modules(arch, kind, C=BIG_C_MIN[arch]) == (big,)
            assert kcp_jit.selected_modules(arch, kind, C=8192) == (big,)
        assert kcp_jit.selected_modules(arch, "kcp_fwd", C=256) == (
            routes["kcp_fwd"][0],
        )
        assert kcp_jit.workspace_record(arch)["role"] == "kcp_bwd"
        assert {owner["role"] for owner in kcp_jit.workspace_owners(arch)} == set(
            BWD_ROLES
        )
    head = kcp_jit.MODULES["cake_causal_conv1d_det_train_17cd42fb05079bbbf840_kcp"]
    assert kcp_jit.head_grid(head, n_frag=3, C=4096) == (3 * 4, 1, 1)
    assert kcp_jit.head_grid(head, n_frag=0, C=4096) == (1, 1, 1)
    # the tail-row program: the head program's geometry and grid rule (one CTA per fragment and 1024 channels)
    tail = kcp_jit.MODULES[TAIL]
    assert tail["grid_rule"]["geometry"] == head["grid_rule"]["geometry"]
    assert kcp_jit.head_grid(tail, n_frag=3, C=10240) == (3 * 10, 1, 1)
    # the history-row program: the same rule
    repair = kcp_jit.MODULES[REPAIR]
    assert repair["grid_rule"]["geometry"] == head["grid_rule"]["geometry"]
    assert kcp_jit.head_grid(repair, n_frag=3, C=10240) == (3 * 10, 1, 1)


@pytest.mark.cpu_only
def test_build_manifest_lists_the_kcp_programs() -> None:
    libraries = cake_build.libraries_by_name()
    for name, record in kcp_jit.MODULES.items():
        for arch in record["arches"]:
            entry = libraries[kcp_jit.library_name(name, arch)]
            assert entry["arch"] == arch and entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{name}_kernel.cu",
                f"csrc/{name}_binding.cu",
            ]
            assert "--use_fast_math" not in entry["nvcc_flags"]
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
        for arch in set(ARCHES) - set(record["arches"]):
            assert f"{name}_{arch}" not in libraries, (name, arch)
            with pytest.raises(RuntimeError, match="is not delivered for"):
                kcp_jit.library_name(name, arch)


@pytest.mark.cpu_only
def test_backward_launch_form_selects_the_route(monkeypatch) -> None:
    """The host's ``cooperative`` argument (FlashInfer issue #8365): ``None`` is the package default, a bool picks the
    ``kcp_bwd`` (cooperative) or ``kcp_bwd_split`` route, anything else is refused; the split records launch
    ``ctas_per_sm x (SMs - reserve_sms)`` CTAs (:func:`kcp._launch_grid`), the cooperative ones the co-resident count."""

    kcp = cake_package("cake_causal_conv1d_det_train", "kcp")
    assert kcp.COOPERATIVE_DEFAULT is False
    assert kcp._cooperative(None) is kcp.COOPERATIVE_DEFAULT
    assert kcp._cooperative(True) is True and kcp._cooperative(False) is False
    for bad in (0, 1, "split", 1.0):
        with pytest.raises(TypeError, match="cooperative must be None or a bool"):
            kcp._cooperative(bad)
    assert kcp._bwd_kind(True) == "kcp_bwd"
    assert kcp._bwd_kind(False) == "kcp_bwd_split"
    for name, (_m, _k, _a, role, _p, ctas_per_sm) in PROGRAMS.items():
        if role not in BWD_ROLES:
            continue
        record = kcp_jit.MODULES[name]
        full = (ctas_per_sm * 148, 1, 1)  # the co-resident grid of a 148-SM device
        reserve = 1 if role == "kcp_bwd_split" else 0
        monkeypatch.setattr(kcp, "_record_launch_grid", lambda record, **_kw: full)
        assert kcp._launch_grid(record, device_index=0) == (
            ctas_per_sm * (148 - reserve),
            1,
            1,
        )


@pytest.mark.cpu_only
def test_forward_launches_one_sm_short(monkeypatch) -> None:
    """The forward record carries no ``reserve_sms`` (the non-deterministic package runs the same record); the host
    launches it ``ctas_per_sm x (SMs - FWD_RESERVE_SMS)`` CTAs wide (:func:`kcp._launch_grid`), one SM short of the
    co-resident count, while that grid holds every channel tile of ``C``, else the full co-resident grid
    (:func:`kcp._forward_grid`), also on a device too small to give up an SM."""

    kcp = cake_package("cake_causal_conv1d_det_train", "kcp")
    assert kcp.FWD_RESERVE_SMS == 1
    seen = set()
    for name, (_m, _k, _a, role, _p, ctas_per_sm) in PROGRAMS.items():
        if role != kcp.KCP_FWD:
            continue
        record = kcp_jit.MODULES[name]
        assert "reserve_sms" not in record["grid_rule"]
        full = (ctas_per_sm * 148, 1, 1)  # the co-resident grid of a 148-SM device
        monkeypatch.setattr(kcp, "_record_launch_grid", lambda record, **_kw: full)
        assert kcp._launch_grid(record, device_index=0) == (
            ctas_per_sm * (148 - kcp.FWD_RESERVE_SMS),
            1,
            1,
        )
        # kcp._forward_grid: the shorter grid while it holds every channel tile, the full one when a tile needs the
        # reserved SM (C launched with the full grid before the reservation), a ValueError past the full grid
        short = (ctas_per_sm * (148 - kcp.FWD_RESERVE_SMS), 1, 1)
        for n_ch_tiles, grid in (
            (1, short),
            (short[0], short),
            (short[0] + 1, full),
            (full[0], full),
        ):
            assert (
                kcp._forward_grid(record, device_index=0, C=0, n_ch_tiles=n_ch_tiles)
                == grid
            ), (name, n_ch_tiles)
        with pytest.raises(ValueError, match="spans more channel tiles"):
            kcp._forward_grid(record, device_index=0, C=0, n_ch_tiles=full[0] + 1)
        # a one-SM device: no SM to give up, so every C the full grid holds launches it
        one_sm = (ctas_per_sm, 1, 1)
        monkeypatch.setattr(kcp, "_record_launch_grid", lambda record, **_kw: one_sm)
        for n_ch_tiles in (1, one_sm[0]):
            assert (
                kcp._forward_grid(record, device_index=0, C=0, n_ch_tiles=n_ch_tiles)
                == one_sm
            ), (name, n_ch_tiles)
        with pytest.raises(ValueError, match="spans more channel tiles"):
            kcp._forward_grid(record, device_index=0, C=0, n_ch_tiles=one_sm[0] + 1)
        seen.add(name)
    assert seen, "no kcp_fwd program in the table"


@pytest.mark.cpu_only
def test_two_launch_forward_programs() -> None:
    """The two-launch forward's programs (``kcp`` module docstring): the history-row program is the ``kcp_fwd_halo``
    route's one record and takes the arguments the host passes (``kcp._fwd_repair``); the single-rank main program is
    a record of the zero-history registry's ``fwd`` route (``kcp.FWD_MAIN_ROUTE``) on every architecture, each record
    of that route takes only ``x``, ``w``, ``cu``, ``y``, its ``work`` counters (``tile_counters``) and the plan
    scalars (``kcp._fwd_main``), and launches the co-resident grid (no reserved SM), so the host can launch whichever
    record ``cake_jit.selected_modules`` picks for the call's ``C`` and token count."""

    kcp = cake_package("cake_causal_conv1d_det_train", "kcp")
    assert kcp.KCP_FWD_HALO == "kcp_fwd_halo" and kcp.FWD_MAIN_ROUTE == "fwd"
    passed = {"x", "w", "cu", "y", "hist", "hist_row_off", "C", "num_docs"}
    assert set(kcp_jit.MODULES[REPAIR]["logical"].values()) == passed
    main_values = {"x", "w", "cu", "y", "work", *cake_jit.PLAN_SCALARS}
    for arch in ARCHES:
        assert kcp_jit.route_modules(arch, kcp.KCP_FWD_HALO) == (REPAIR,)
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
