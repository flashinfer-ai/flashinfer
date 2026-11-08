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
build (route ``kcp_bwd_split``, FlashInfer issue #8365), an ordinary launch one SM short of co-resident
(``reserve_sms``) whose ``dweight`` partial rows the zero-history package's fixed-order reduce program sums.  No device
is needed; nothing is built or launched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.test_helpers.cake_package import cake_package

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

REDUCE = "cake_causal_conv1d_det_train_86b5ecfbd90aca4b52e5"
FALLBACK = "cake_causal_conv1d_det_train_5288d132a5cc927ae729_kcp"
R200_FALLBACK = "cake_causal_conv1d_det_train_6ea6d5fc9e7fa3f2d039_kcp"
BIG = "cake_causal_conv1d_det_train_7485529a1755e5bd3036_kcp"
SPLIT_FALLBACK = "cake_causal_conv1d_det_train_facdd7ca702cfb7d821a_kcp"
SPLIT_R200_FALLBACK = "cake_causal_conv1d_det_train_f3bd6b02735ab7bbb8b5_kcp"
SPLIT_BIG = "cake_causal_conv1d_det_train_4e81b876cb5eb25e60e5_kcp"
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
        {"vec": 8, "copy": "tma2d", "split": True},
        ("sm_100a", "sm_103a"),
        "kcp_bwd_split",
        BWD_TMA_PLAN,
        3,
    ),
    SPLIT_R200_FALLBACK: (
        BWD,
        {"vec": 8, "min_blocks": 2, "split": True},
        ("sm_107a",),
        "kcp_bwd_split",
        BWD_PLAN,
        2,
    ),
    SPLIT_BIG: (
        BWD,
        {"vec": 8, "copy": "tma2d", "stage_tokens": 8, "min_blocks": 2, "split": True},
        ARCHES,
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
    )
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
        for kind, fallback, big in (
            ("kcp_bwd", R200_FALLBACK if arch == "sm_107a" else FALLBACK, BIG),
            (
                "kcp_bwd_split",
                SPLIT_R200_FALLBACK if arch == "sm_107a" else SPLIT_FALLBACK,
                SPLIT_BIG,
            ),
        ):
            assert routes[kind] == [fallback, big]
            assert kcp_jit.route_modules(arch, kind) == (fallback, big)
            assert kcp_jit.select_rule(kcp_jit.MODULES[big]) == {
                "channels": BIG_C_MIN,
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
    co-resident count, and the head program's grid is not touched."""

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
        seen.add(name)
    assert seen, "no kcp_fwd program in the table"
