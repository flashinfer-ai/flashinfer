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

CPU-only tests for the Cake MoE all-reduce union export (SM100 and SM103, world sizes 2, 4, 8)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flashinfer.comm import trtllm_ar
from flashinfer.jit import cake_trtllm_moe_allreduce_union as union

_WORLD_SIZES = (2, 4, 8)
_DTYPES = ("bfloat16", "float16")
_PDL = (False, True)
_SM100 = (10, 0)
_SM103 = (10, 3)
_SM120 = (12, 0)
# Exported architectures and the compute capability each one serves.
_ARCHES = {"sm_100a": _SM100, "sm_103a": _SM103}

_WIDE = union.SPECIALIZATION_WIDE_MLP
# World size 4 on SM100 keeps exactly the reviewed route set: the bfloat16
# classes keep a generic program (their T=1 / T=64 rows run wide_mlp; the
# no-PDL T=1 row additionally runs the reviewed single-CTA geometry), the
# float16 classes run wide_mlp for every token count (the resident and
# owner-forward shapes compose with it).
_SM100_WS4_ROUTE_KEYS = {
    ("sm_100a", 4, "bfloat16", False, union.SPECIALIZATION_GENERIC),
    ("sm_100a", 4, "bfloat16", True, union.SPECIALIZATION_GENERIC),
    (
        "sm_100a",
        4,
        "bfloat16",
        False,
        f"{_WIDE}_{union.SPECIALIZATION_T1_E8_SERIAL_CLEAR}_{union.SPECIALIZATION_CTA1}",
    ),
    ("sm_100a", 4, "bfloat16", True, _WIDE),
    (
        "sm_100a",
        4,
        "float16",
        False,
        f"{_WIDE}_{union.SPECIALIZATION_T64_E12_RESIDENT}",
    ),
    ("sm_100a", 4, "float16", False, _WIDE),
    (
        "sm_100a",
        4,
        "float16",
        True,
        f"{_WIDE}_{union.SPECIALIZATION_T128_E16_OWNER_FORWARD}",
    ),
    ("sm_100a", 4, "float16", True, _WIDE),
    # Convergence round: the three T=2048 rows run the pipelined publish / poll
    # variant, the bfloat16 no-PDL T=128 e16 row the clear-first wide body.
    ("sm_100a", 4, "bfloat16", True, "pipe2_u4_b5"),
    ("sm_100a", 4, "float16", True, "pipe2_u4_b5"),
    ("sm_100a", 4, "float16", False, "pipe2_u4_b5"),
    ("sm_100a", 4, "bfloat16", False, "clrfirst"),
}
# Cooperative (resident-grid) shape specializations of the union (alone or
# composed with the wide_mlp schedule).
_COOPERATIVE_SPECIALIZATIONS = {
    union.SPECIALIZATION_T64_E12_RESIDENT,
    union.SPECIALIZATION_T128_E16_OWNER_FORWARD,
    f"{_WIDE}_{union.SPECIALIZATION_T64_E12_RESIDENT}",
    f"{_WIDE}_{union.SPECIALIZATION_T128_E16_OWNER_FORWARD}",
}
# Every (arch, world size) carries one generic route per (dtype, launch_with_pdl)
# pair plus reviewed specializations drawn from this allow-list.  Extend a set
# when a new specialization is reviewed for that scope.  On SM103 the
# ``sm103_t1`` schedule variant applies at T=1 and composes with the
# world-size-4 shape specializations (serial clear at BF16/T1/E8/no-PDL).
_EXTRA_SPECIALIZATIONS = {
    ("sm_100a", 2): frozenset({_WIDE, f"{_WIDE}_{union.SPECIALIZATION_CTA1}", "pipe2_u4", "pipe2_u4_b5"}),
    ("sm_100a", 4): {key[4] for key in _SM100_WS4_ROUTE_KEYS}
    - {union.SPECIALIZATION_GENERIC},
    ("sm_100a", 8): frozenset({union.SPECIALIZATION_SM100_WS8_MID, "pipe1_u4_b5"}),
    ("sm_103a", 2): frozenset({union.SPECIALIZATION_SM103_T1, _WIDE, "pipe1_u4_b5"}),
    ("sm_103a", 4): frozenset(
        {
            f"{union.SPECIALIZATION_SM103_T1}_{union.SPECIALIZATION_T1_E8_SERIAL_CLEAR}",
            _WIDE,
            f"{_WIDE}_{union.SPECIALIZATION_T64_E12_RESIDENT}",
            f"{_WIDE}_{union.SPECIALIZATION_T128_E16_OWNER_FORWARD}",
            "pipe2_u4_b5",
        }
    ),
    ("sm_103a", 8): frozenset(
        {union.SPECIALIZATION_SM103_T1, union.SPECIALIZATION_SM103_WS8_MID, "pipe1_u4_b5", "push_g", "pipe1"}
    ),
}


def _raw_pointers(world_size: int) -> set[str]:
    return {
        "workspace_control",
        *(f"workspace_payload_{peer}" for peer in range(world_size)),
    }


def _module_scopes() -> dict[str, tuple[str, int]]:
    """Map every routed module to the (arch, world size) of the route that lists it."""

    scopes: dict[str, tuple[str, int]] = {}
    for (
        arch,
        world_size,
        _dtype,
        _pdl,
        _specialization,
    ), names in union.ROUTES.items():
        for name in names:
            assert scopes.setdefault(name, (arch, world_size)) == (arch, world_size)
    return scopes


def _route_keys(arch: str, world_size: int) -> set[tuple[str, int, str, bool, str]]:
    return {key for key in union.ROUTES if key[0] == arch and key[1] == world_size}


def test_module_inventory_is_verified_source_only() -> None:
    assert union.MODULES
    scopes = _module_scopes()
    # Every module is reachable through a route and every route names a module.
    assert set(scopes) == set(union.MODULES)
    for name, record in union.MODULES.items():
        arch, world_size = scopes[name]
        assert record["arch"] == arch and arch in _ARCHES
        assert name.startswith("cake_") and record["cache_name"].startswith("cake_")
        assert record["cache_name"].endswith(f"_{arch}")
        assert record["kernel_symbol"].startswith("kernel_cake_")
        assert record["ffi_entry"] == "run"
        assert record["compile_flags"] == ["--use_fast_math"]
        for relative in record["sources"]:
            assert relative.startswith(f"csrc/{union.SOURCE_PACKAGE}/{arch}/")
        paths = union.verified_sources(name)
        assert len(paths) == 2 and all(path.suffix == ".cu" for path in paths)
        kinds = [kind for kind, _key in record["arg_plan"]]
        assert set(kinds) <= {"buffer", "raw_pointer", "parameter", "grid"}
        assert record["arg_plan"][-3:] == [
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ]
        assert {
            key for kind, key in record["arg_plan"] if kind == "raw_pointer"
        } == _raw_pointers(world_size)
        assert ("buffer", "workspace_tensor") in record["arg_plan"]
        launch = record["launch"]
        block = tuple(launch["block"])
        cluster = tuple(launch["cluster"])
        # One 7168-wide token is covered by 896 threads (16 B per thread): four
        # 224-thread CTAs in a cluster, or one 896-thread CTA where the T=1
        # geometry probe measured the single-CTA build faster.
        assert cluster[1:] == (1, 1) and cluster[0] in {1, 4}
        assert block[1:] == (1, 1) and block[0] * cluster[0] == 896
        residency = launch["persistent_ctas_per_sm"]
        assert isinstance(residency, int) and residency >= 1
        cap = launch["max_persistent_clusters"]
        assert cap is None or (isinstance(cap, int) and cap >= 1)
        if launch["cooperative"]:
            assert residency == 1 and cap is None
        if cluster[0] == 1:
            assert record["arch"] == "sm_100a" and world_size in {2, 4}
            assert not launch["cooperative"] and launch["persistent_ctas_per_sm"] == 1
            assert cap is None


def test_exported_architectures_are_sm100_and_sm103() -> None:
    assert tuple(union.ARCHES) == tuple(_ARCHES)
    capability_arches = {capability: arch for arch, capability in _ARCHES.items()}
    assert capability_arches == union.ARCH_BY_CAPABILITY
    assert set(union.exported_arches()) == set(_ARCHES)
    for arch, capability in _ARCHES.items():
        assert union.arch_for_capability(capability) == arch
    assert union.arch_for_capability(_SM120) is None


def test_routes_cover_exactly_the_reviewed_specializations() -> None:
    assert tuple(union.WORLD_SIZES) == _WORLD_SIZES
    assert {(key[0], key[1]) for key in union.ROUTES} == {
        (arch, world_size) for arch in _ARCHES for world_size in _WORLD_SIZES
    }
    assert {key[2] for key in union.ROUTES} <= set(_DTYPES)
    assert {key[3] for key in union.ROUTES} <= set(_PDL)
    assert _route_keys("sm_100a", 4) == _SM100_WS4_ROUTE_KEYS
    classes = set(union._WIDE_MLP_CLASSES)
    assert all(len(key) == 4 for key in classes)
    for arch in _ARCHES:
        for world_size in _WORLD_SIZES:
            keys = _route_keys(arch, world_size)
            for dtype in _DTYPES:
                for pdl in _PDL:
                    # Every (dtype, PDL) class has one persistent program for
                    # unreviewed token counts: generic, or wide_mlp for a
                    # promoted class (which then exports no generic program).
                    if (arch, world_size, dtype, pdl) in classes:
                        assert (arch, world_size, dtype, pdl, _WIDE) in keys
                        assert (
                            arch,
                            world_size,
                            dtype,
                            pdl,
                            union.SPECIALIZATION_GENERIC,
                        ) not in keys
                    else:
                        assert (
                            arch,
                            world_size,
                            dtype,
                            pdl,
                            union.SPECIALIZATION_GENERIC,
                        ) in keys
            extra = {key[4] for key in keys} - {union.SPECIALIZATION_GENERIC}
            assert extra <= _EXTRA_SPECIALIZATIONS[(arch, world_size)]
    for (_arch, world_size, _dtype, pdl, specialization), names in union.ROUTES.items():
        assert len(names) == world_size
        launches = [union.MODULES[name]["launch"] for name in names]
        for launch in launches:
            assert launch["use_pdl"] is pdl
            # One route launches every rank the same way.
            assert launch["cooperative"] is launches[0]["cooperative"]
            assert tuple(launch["block"]) == tuple(launches[0]["block"])
            assert (
                launch["persistent_ctas_per_sm"]
                == launches[0]["persistent_ctas_per_sm"]
            )
            assert (
                launch["max_persistent_clusters"]
                == launches[0]["max_persistent_clusters"]
            )
        if specialization in {union.SPECIALIZATION_GENERIC, _WIDE}:
            # The persistent programs serve up to 2048 tokens; a one-cluster-per-token
            # cooperative grid cannot be co-resident at that size.
            assert launches[0]["cooperative"] is False
        if world_size == 4:
            assert launches[0]["cooperative"] is (
                specialization in _COOPERATIVE_SPECIALIZATIONS
            )
        if specialization.endswith(union.SPECIALIZATION_T128_E16_OWNER_FORWARD):
            # Owner forwarding is rank-specialized: one physical module per rank.
            assert len(set(names)) == world_size
        # The single-CTA geometry is a named specialization: exactly the routes
        # whose name carries the suffix launch one-CTA clusters.
        single_cta = specialization.endswith(union.SPECIALIZATION_CTA1)
        for launch in launches:
            assert (tuple(launch["cluster"])[0] == 1) is single_cta


@pytest.mark.parametrize(
    "arch,world_size,dtype_name,pdl,tokens,experts,expected",
    [
        # SM100 world size 4: bfloat16 keeps generic except its reviewed T=1 (serial
        # clear) and T=64 rows; float16 is a wide_mlp class for every token count.
        (
            "sm_100a",
            4,
            "bfloat16",
            False,
            1,
            8,
            f"{_WIDE}_{union.SPECIALIZATION_T1_E8_SERIAL_CLEAR}_{union.SPECIALIZATION_CTA1}",
        ),
        ("sm_100a", 4, "bfloat16", True, 1, 8, union.SPECIALIZATION_GENERIC),
        ("sm_100a", 4, "bfloat16", True, 64, 8, _WIDE),
        ("sm_100a", 4, "bfloat16", True, 64, 12, union.SPECIALIZATION_GENERIC),
        ("sm_100a", 4, "float16", False, 1, 8, _WIDE),
        (
            "sm_100a",
            4,
            "float16",
            False,
            64,
            12,
            f"{_WIDE}_{union.SPECIALIZATION_T64_E12_RESIDENT}",
        ),
        ("sm_100a", 4, "float16", True, 64, 12, _WIDE),
        ("sm_100a", 4, "float16", False, 512, 8, _WIDE),
        (
            "sm_100a",
            4,
            "float16",
            True,
            128,
            16,
            f"{_WIDE}_{union.SPECIALIZATION_T128_E16_OWNER_FORWARD}",
        ),
        ("sm_100a", 4, "bfloat16", False, 128, 16, "clrfirst"),
        ("sm_100a", 4, "bfloat16", False, 128, 8, union.SPECIALIZATION_GENERIC),
        ("sm_100a", 4, "bfloat16", False, 512, 8, union.SPECIALIZATION_GENERIC),
        ("sm_100a", 4, "bfloat16", True, 2048, 12, union.SPECIALIZATION_GENERIC),
        # The world-size-4 reviewed shapes do not leak into other world sizes;
        # every SM100 two-rank class runs wide_mlp (no generic program) and the
        # bfloat16 no-PDL T=1 row runs its single-CTA wide_mlp build.
        ("sm_100a", 2, "bfloat16", False, 1, 8, f"{_WIDE}_{union.SPECIALIZATION_CTA1}"),
        ("sm_100a", 2, "float16", False, 64, 12, _WIDE),
        ("sm_100a", 2, "float16", True, 128, 16, _WIDE),
        ("sm_100a", 2, "bfloat16", True, 2048, 12, _WIDE),
        ("sm_100a", 2, "bfloat16", True, 512, 8, _WIDE),
        ("sm_100a", 8, "bfloat16", False, 1, 8, union.SPECIALIZATION_GENERIC),
        ("sm_100a", 8, "float16", True, 1, 8, union.SPECIALIZATION_GENERIC),
        ("sm_100a", 8, "float16", False, 2048, 12, union.SPECIALIZATION_GENERIC),
        ("sm_100a", 8, "bfloat16", True, 2048, 16, union.SPECIALIZATION_GENERIC),
        ("sm_100a", 8, "bfloat16", True, 2048, 12, "pipe1_u4_b5"),
        ("sm_100a", 8, "float16", False, 2048, 8, "pipe1_u4_b5"),
        # SM103: the T=1 schedule variant is reviewed at every world size and
        # composes with the world-size-4 serial clear; the SM100 world-size-4
        # shape specializations apply on SM103 too.
        ("sm_103a", 2, "bfloat16", False, 1, 8, union.SPECIALIZATION_SM103_T1),
        (
            "sm_103a",
            4,
            "bfloat16",
            False,
            1,
            8,
            f"{union.SPECIALIZATION_SM103_T1}_{union.SPECIALIZATION_T1_E8_SERIAL_CLEAR}",
        ),
        ("sm_103a", 8, "bfloat16", False, 1, 8, union.SPECIALIZATION_SM103_T1),
        # SM103 wide_mlp classes (bfloat16 without PDL at two and four ranks, float16 at
        # four ranks) and reviewed rows inside mixed classes.
        (
            "sm_103a",
            4,
            "float16",
            False,
            64,
            12,
            f"{_WIDE}_{union.SPECIALIZATION_T64_E12_RESIDENT}",
        ),
        (
            "sm_103a",
            4,
            "float16",
            True,
            128,
            16,
            f"{_WIDE}_{union.SPECIALIZATION_T128_E16_OWNER_FORWARD}",
        ),
        ("sm_103a", 4, "float16", True, 1024, 16, _WIDE),
        ("sm_103a", 4, "bfloat16", False, 256, 8, _WIDE),
        ("sm_103a", 4, "bfloat16", True, 256, 12, _WIDE),
        ("sm_103a", 4, "bfloat16", True, 2048, 12, "pipe2_u4_b5"),
        ("sm_103a", 2, "bfloat16", False, 64, 8, _WIDE),
        ("sm_103a", 2, "bfloat16", True, 64, 8, _WIDE),
        ("sm_103a", 2, "bfloat16", True, 128, 16, union.SPECIALIZATION_GENERIC),
        ("sm_103a", 2, "float16", True, 128, 16, _WIDE),
        ("sm_103a", 2, "float16", True, 2048, 16, "pipe1_u4_b5"),
        ("sm_103a", 2, "float16", True, 1024, 16, union.SPECIALIZATION_GENERIC),
        ("sm_103a", 8, "float16", False, 2048, 8, "pipe1_u4_b5"),
        ("sm_103a", 8, "float16", False, 256, 8, union.SPECIALIZATION_GENERIC),
        ("sm_103a", 8, "bfloat16", True, 64, 8, "push_g"),
        ("sm_103a", 8, "bfloat16", True, 256, 12, "push_g"),
        ("sm_103a", 8, "bfloat16", False, 256, 8, "pipe1"),
        # Reviewed SM103 shapes do not leak into SM100 and vice versa.
        ("sm_100a", 2, "bfloat16", False, 1, 8, f"{_WIDE}_{union.SPECIALIZATION_CTA1}"),
    ],
)
def test_select_specialization_rules(
    arch, world_size, dtype_name, pdl, tokens, experts, expected
) -> None:
    assert (
        union.select_specialization(arch, world_size, dtype_name, pdl, tokens, experts)
        == expected
    )
    names = union.route_module_names(
        arch=arch,
        world_size=world_size,
        dtype_name=dtype_name,
        launch_with_pdl=pdl,
        token_num=tokens,
        active_experts=experts,
    )
    assert names == union.ROUTES[(arch, world_size, dtype_name, pdl, expected)]
    assert len(names) == world_size
    assert (
        union.route_module_name(
            arch=arch,
            world_size=world_size,
            dtype_name=dtype_name,
            launch_with_pdl=pdl,
            token_num=tokens,
            active_experts=experts,
            world_rank=world_size - 1,
        )
        == names[world_size - 1]
    )
    with pytest.raises(ValueError):
        union.route_module_name(
            arch=arch,
            world_size=world_size,
            dtype_name=dtype_name,
            launch_with_pdl=pdl,
            token_num=tokens,
            active_experts=experts,
            world_rank=world_size,
        )


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_sm100_ws8_mid_rows_are_exactly_the_reviewed_ones(world_size: int) -> None:
    mid = {
        key
        for key, kind in union._REVIEWED_SPECIALIZATIONS.items()
        if kind == union.SPECIALIZATION_SM100_WS8_MID
    }
    if world_size != 8:
        assert not {key for key in mid if key[1] == world_size}
        return
    assert mid == {
        ("sm_100a", 8, "bfloat16", True, 64, 8),
        ("sm_100a", 8, "float16", False, 64, 12),
        ("sm_100a", 8, "bfloat16", False, 128, 16),
        ("sm_100a", 8, "float16", True, 128, 16),
    }


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("capability", sorted(_ARCHES.values()))
def test_route_scope_is_sm100_sm103_with_allreduce_output(
    world_size: int, capability: tuple[int, int]
) -> None:
    assert union.route_applies(
        world_size=world_size, device_capability=capability, emit_moe_allreduce=True
    )
    assert not union.route_applies(
        world_size=world_size, device_capability=capability, emit_moe_allreduce=False
    )
    assert not union.route_applies(
        world_size=world_size, device_capability=_SM120, emit_moe_allreduce=True
    )


@pytest.mark.parametrize("world_size", (1, 3, 16))
def test_route_scope_rejects_unexported_world_sizes(world_size: int) -> None:
    for arch, capability in _ARCHES.items():
        assert not union.route_applies(
            world_size=world_size, device_capability=capability, emit_moe_allreduce=True
        )
        with pytest.raises(ValueError):
            union.route_module_names(
                arch=arch,
                world_size=world_size,
                dtype_name="float16",
                launch_with_pdl=False,
                token_num=1,
                active_experts=8,
            )


def test_launch_grid_rule() -> None:
    # SM-bounded persistent grid at one CTA per SM, then at k resident CTAs per SM
    # (four-CTA clusters).
    assert union.launch_grid_x(1, False, 148, 1, 4) == 4
    assert union.launch_grid_x(2048, False, 148, 1, 4) == 148
    assert union.launch_grid_x(1, False, 148, 4, 4) == 4
    assert union.launch_grid_x(64, False, 148, 4, 4) == 256
    assert union.launch_grid_x(2048, False, 148, 4, 4) == 592
    assert union.launch_grid_x(2048, False, 148, 5, 4) == 740
    # A recorded cluster cap bounds the persistent grid at that many co-resident
    # clusters (five CTAs per SM on 148 SMs would be 185 clusters against 175).
    assert union.launch_grid_x(2048, False, 148, 5, 4, 175) == 700
    assert union.launch_grid_x(64, False, 148, 5, 4, 175) == 256
    assert union.launch_grid_x(2048, False, 148, 4, 4, 175) == 592
    # Single-CTA modules (the reviewed T=1 rows) launch one CTA per token.
    assert union.launch_grid_x(1, False, 148, 1, 1) == 1
    assert union.launch_grid_x(64, False, 148, 1, 1) == 64
    # Resident-grid (cooperative) modules launch one cluster per token.
    assert union.launch_grid_x(64, True, 148, 1, 4) == 256
    assert union.launch_grid_x(128, True, 148, 1, 4) == 512
    with pytest.raises(ValueError):
        union.launch_grid_x(0, False, 148, 1, 4)
    with pytest.raises(ValueError):
        union.launch_grid_x(64, True, 148, 2, 4)
    with pytest.raises(ValueError):
        union.launch_grid_x(64, False, 148, 0, 4)
    with pytest.raises(ValueError):
        union.launch_grid_x(64, False, 148, 1, 0)
    with pytest.raises(ValueError):
        union.launch_grid_x(64, False, 148, 5, 4, 0)
    with pytest.raises(ValueError):
        union.launch_grid_x(64, True, 148, 1, 4, 175)
    for record in union.MODULES.values():
        launch = record["launch"]
        cluster_ctas = int(launch["cluster"][0])
        grid = union.launch_grid_x(
            2048,
            launch["cooperative"],
            148,
            launch["persistent_ctas_per_sm"],
            cluster_ctas,
            launch["max_persistent_clusters"],
        )
        assert grid % cluster_ctas == 0 and grid > 0
        if launch["max_persistent_clusters"] is not None:
            assert grid <= launch["max_persistent_clusters"] * cluster_ctas


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_workspace_pointer_registry_round_trip(world_size: int) -> None:
    entries = 3 * world_size + 1
    table = torch.arange(1, entries + 1, dtype=torch.int64)
    pointers = [int(value) * 4096 for value in range(1, entries + 1)]
    union.register_workspace_pointers(table, pointers)
    assert union.workspace_pointers(table, world_size) == tuple(pointers)
    unregistered = torch.arange(101, 101 + entries, dtype=torch.int64)
    assert union.workspace_pointers(unregistered, world_size) == tuple(
        range(101, 101 + entries)
    )
    with pytest.raises(ValueError):
        union.workspace_pointers(
            torch.zeros(entries - 1, dtype=torch.int64), world_size
        )
    with pytest.raises(ValueError):
        union.register_workspace_pointers(table, pointers[:-1])


def _arguments(world_size: int, *, emit_allreduce: bool) -> dict:
    tokens, hidden, experts = 1, 8, 2
    dtype = torch.float16
    return {
        "world_size": world_size,
        "world_rank": 1,
        "token_num": tokens,
        "hidden_dim": hidden,
        "workspace_ptrs": torch.zeros(3 * world_size + 1, dtype=torch.int64),
        "launch_with_pdl": True,
        "residual_in": torch.zeros(tokens, hidden, dtype=dtype),
        "rms_gamma": torch.ones(hidden, dtype=dtype),
        "rms_eps": 1e-5,
        "scale_factor": 1.0,
        "moe_reduction_device_num_experts": experts,
        "moe_reduction_scale_input": torch.ones(experts, tokens, dtype=torch.float32),
        "moe_reduction_active_experts_token_input": torch.zeros(
            experts, tokens, hidden, dtype=dtype
        ),
        "moe_reduction_token_input": torch.zeros(tokens, hidden, dtype=dtype),
        "layout_code": None,
        "moe_allreduce_out": torch.empty(tokens, hidden, dtype=dtype)
        if emit_allreduce
        else None,
        "residual_out": torch.empty(tokens, hidden, dtype=dtype),
        "norm_out": torch.empty(tokens, hidden, dtype=dtype),
        "quant_out": None,
        "scale_out": None,
        "weight_bias": 1.0,
    }


def _union_arguments(world_size: int) -> dict:
    """The public arguments narrowed to ``run_cake_moe_allreduce_union``'s signature."""

    arguments = _arguments(world_size, emit_allreduce=True)
    for public_only in ("layout_code", "quant_out", "scale_out"):
        del arguments[public_only]
    return arguments


def test_run_rejects_unexported_world_sizes_before_touching_the_device() -> None:
    with pytest.raises(ValueError):
        union.run_cake_moe_allreduce_union(backend="cake", **_union_arguments(16))
    with pytest.raises(ValueError):
        union.run_cake_moe_allreduce_union(backend="trtllm", **_union_arguments(4))


def _isolate_backends(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]
) -> tuple[list, list]:
    union_calls: list[dict] = []
    legacy_calls: list[tuple] = []
    monkeypatch.setattr(trtllm_ar, "_validate_cake_moe_allreduce", lambda **kwargs: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(
        union,
        "run_cake_moe_allreduce_union",
        lambda **kwargs: union_calls.append(kwargs),
    )
    monkeypatch.setattr(
        trtllm_ar,
        "get_cake_moe_allreduce_module",
        lambda device_index: SimpleNamespace(
            run_reduction=lambda *args: legacy_calls.append(args)
        ),
    )
    monkeypatch.setattr(
        trtllm_ar,
        "get_trtllm_comm_module",
        lambda: pytest.fail("TRT-LLM module must not load for backend='cake'"),
    )
    return union_calls, legacy_calls


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("capability", sorted(_ARCHES.values()))
def test_cake_backend_routes_sm100_sm103_with_allreduce_output_to_the_union(
    monkeypatch: pytest.MonkeyPatch, world_size: int, capability: tuple[int, int]
) -> None:
    union_calls, legacy_calls = _isolate_backends(monkeypatch, capability)
    arguments = _arguments(world_size, emit_allreduce=True)

    trtllm_ar.trtllm_moe_allreduce_fusion(**arguments, backend="cake")

    assert legacy_calls == []
    assert len(union_calls) == 1
    call = union_calls[0]
    assert call["backend"] == "cake"
    assert call["world_size"] == world_size and call["world_rank"] == 1
    assert call["workspace_ptrs"] is arguments["workspace_ptrs"]
    assert call["moe_allreduce_out"] is arguments["moe_allreduce_out"]
    assert call["residual_out"] is arguments["residual_out"]
    assert call["norm_out"] is arguments["norm_out"]
    assert call["launch_with_pdl"] is True and call["weight_bias"] == 1.0


_LEGACY_SCOPE_CASES = [
    pytest.param(
        world_size, capability, False, id=f"tp{world_size}-{arch}-no-allreduce-output"
    )
    for world_size in _WORLD_SIZES
    for arch, capability in _ARCHES.items()
]


@pytest.mark.parametrize("world_size,capability,emit_allreduce", _LEGACY_SCOPE_CASES)
def test_cake_backend_keeps_the_legacy_bundle_outside_the_union_scope(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    capability: tuple[int, int],
    emit_allreduce: bool,
) -> None:
    union_calls, legacy_calls = _isolate_backends(monkeypatch, capability)
    arguments = _arguments(world_size, emit_allreduce=emit_allreduce)

    trtllm_ar.trtllm_moe_allreduce_fusion(**arguments, backend="cake")

    assert union_calls == []
    assert len(legacy_calls) == 1 and len(legacy_calls[0]) == 18
    assert legacy_calls[0][0] == world_size


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
def test_workspace_creation_registers_the_pointer_table(
    monkeypatch: pytest.MonkeyPatch, world_size: int
) -> None:
    registered: list[tuple] = []
    monkeypatch.setattr(
        union,
        "register_workspace_pointers",
        lambda tensor, pointers: registered.append((tensor, tuple(pointers))),
    )
    entries = 3 * world_size + 1
    table = torch.arange(entries, dtype=torch.int64)
    trtllm_ar.register_cake_moe_allreduce_workspace_pointers(
        table, list(range(entries))
    )
    assert registered == [(table, tuple(range(entries)))]
