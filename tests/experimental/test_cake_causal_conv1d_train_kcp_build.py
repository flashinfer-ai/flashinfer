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

KEY = "cake.cake_causal_conv1d_train"
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
    """``rule`` without an explicit one-copy ``slots`` (the KCP programs keep one FP32 copy; a rule may name it)."""

    rule = dict(rule)
    if rule.get("slots") == 1:
        del rule["slots"]
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
    for arch in ARCHES:
        assert set(kcp_jit.ROUTES[arch]) == set(kcp_jit.KINDS), arch


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
    assert _accumulator_rule(kcp_jit.workspace_record(arch)["workspace_rule"]) == ACCUMULATORS
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
