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

CPU-only routing tests for the Cake MoE all-reduce union (SM100 and SM103, world sizes 2, 4, 8)."""

from __future__ import annotations

import re
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
_ARCHES = {"sm_100a": _SM100, "sm_103a": _SM103}
_WIDE = union.SPECIALIZATION_WIDE_MLP
_GENERIC = union.SPECIALIZATION_GENERIC


def _route_keys(arch: str, world_size: int) -> set[tuple[str, int, str, bool, str]]:
    return {key for key in union.ROUTES if key[0] == arch and key[1] == world_size}


def _kernel_names(route: union.Route) -> tuple[str, ...]:
    return (route.kernel,) if isinstance(route.kernel, str) else tuple(route.kernel)


def test_every_route_names_a_kernel_of_its_world_size_and_dtype() -> None:
    assert tuple(union.WORLD_SIZES) == _WORLD_SIZES
    assert {(key[0], key[1]) for key in union.ROUTES} == {
        (arch, world_size) for arch in _ARCHES for world_size in _WORLD_SIZES
    }
    used: set[str] = set()
    for (arch, world_size, dtype, pdl, _spec), route in union.ROUTES.items():
        assert arch in _ARCHES and dtype in _DTYPES and pdl in _PDL
        names = _kernel_names(route)
        if isinstance(route.kernel, tuple):
            # Rank-specialised routes carry one kernel per rank.
            assert len(names) == world_size and len(set(names)) == world_size
        for name in names:
            kernel = union.KERNELS[name]
            assert kernel.dtype == dtype and kernel.world_size == world_size
            # One 7168-wide token is covered by 896 threads: four 224-thread
            # CTAs in a cluster, or one 896-thread CTA.
            assert kernel.block * kernel.cluster == 896 and kernel.cluster in {1, 4}
        used |= set(names)
        assert route.persistent_ctas_per_sm >= 1
        if route.cooperative:
            assert route.persistent_ctas_per_sm == 1
            assert route.max_persistent_clusters is None
        if _spec in {_GENERIC, _WIDE}:
            # The persistent programs serve up to 2048 tokens; a cooperative
            # one-cluster-per-token grid cannot be co-resident at that size.
            assert route.cooperative is False
    # Every kernel is reachable through a route.
    assert used == set(union.KERNELS)


def test_kernel_sources_exist_and_define_their_symbol() -> None:
    launcher = union._source_dir() / union.LAUNCHER_SOURCE
    assert launcher.is_file()
    for name in union.KERNELS:
        path = union.kernel_source(name)
        assert path.is_file(), path
        text = path.read_text()
        symbol = union.kernel_symbol(name)
        assert re.search(r"\bvoid\s+" + re.escape(symbol) + r"\(", text), symbol
        # The device text takes the rank at runtime and reads every workspace
        # address from the device pointer table.
        assert "world_rank" in text


def _class_program(arch: str, world_size: int, dtype: str, pdl: bool) -> str:
    return (
        _WIDE
        if (arch, world_size, dtype, pdl) in set(union._WIDE_MLP_CLASSES)
        else _GENERIC
    )


def test_every_class_has_one_program_for_unreviewed_token_counts() -> None:
    classes = set(union._WIDE_MLP_CLASSES)
    for arch in _ARCHES:
        for world_size in _WORLD_SIZES:
            keys = _route_keys(arch, world_size)
            for dtype in _DTYPES:
                for pdl in _PDL:
                    if (arch, world_size, dtype, pdl) in classes:
                        assert (arch, world_size, dtype, pdl, _WIDE) in keys
                        assert (arch, world_size, dtype, pdl, _GENERIC) not in keys
                    else:
                        assert (arch, world_size, dtype, pdl, _GENERIC) in keys


def test_specialization_rules_are_sorted_disjoint_token_ranges_naming_exported_routes() -> (
    None
):
    for (
        arch,
        world_size,
        dtype,
        pdl,
        experts,
    ), ranges in union._SPECIALIZATION_RULES.items():
        assert (
            arch in _ARCHES
            and world_size in _WORLD_SIZES
            and dtype in _DTYPES
            and pdl in _PDL
        )
        assert isinstance(experts, int) and experts >= 1
        assert len(ranges) >= 1
        previous_hi = 0
        for token_lo, token_hi, spec in ranges:
            # Inclusive, non-empty, sorted and disjoint ranges.
            assert 1 <= token_lo <= token_hi
            assert token_lo > previous_hi
            previous_hi = token_hi
            # A rule names a specialization other than its class program, and that
            # specialization has an exported route.
            assert spec != _class_program(arch, world_size, dtype, pdl)
            assert (arch, world_size, dtype, pdl, spec) in union.ROUTES


def test_every_route_is_a_class_program_or_named_by_a_rule() -> None:
    named: set[tuple[str, int, str, bool, str]] = set()
    for (
        arch,
        world_size,
        dtype,
        pdl,
        _experts,
    ), ranges in union._SPECIALIZATION_RULES.items():
        named |= {(arch, world_size, dtype, pdl, spec) for _lo, _hi, spec in ranges}
    for arch, world_size, dtype, pdl, spec in union.ROUTES:
        if spec == _class_program(arch, world_size, dtype, pdl):
            continue
        assert (arch, world_size, dtype, pdl, spec) in named, (
            arch,
            world_size,
            dtype,
            pdl,
            spec,
        )


def test_exported_architectures_are_sm100_and_sm103() -> None:
    assert tuple(union.ARCHES) == tuple(_ARCHES)
    assert {
        capability: arch for arch, capability in _ARCHES.items()
    } == union.ARCH_BY_CAPABILITY
    assert set(union.exported_arches()) == set(_ARCHES)
    for arch, capability in _ARCHES.items():
        assert union.arch_for_capability(capability) == arch
    assert union.arch_for_capability(_SM120) is None


@pytest.mark.parametrize(
    "arch,world_size,dtype_name,pdl,tokens,experts,expected",
    [
        # Rendered by the exporter: both ends of every token-range rule, one
        # token count per class outside its rules, and the reviewed token counts
        # of the S7 exact-shape decisions (kept rows at their specialization,
        # dropped rows at their class program).
    ],
)
def test_select_specialization_rules(
    arch, world_size, dtype_name, pdl, tokens, experts, expected
) -> None:
    assert (
        union.select_specialization(arch, world_size, dtype_name, pdl, tokens, experts)
        == expected
    )
    key, route = union.route_for(
        arch=arch,
        world_size=world_size,
        dtype_name=dtype_name,
        launch_with_pdl=pdl,
        token_num=tokens,
        active_experts=experts,
    )
    assert key == (arch, world_size, dtype_name, pdl, expected)
    assert route is union.ROUTES[key]
    for rank in range(world_size):
        name = union.route_kernel(route, rank, world_size)
        assert union.KERNELS[name].world_size == world_size
        assert union.KERNELS[name].dtype == dtype_name
    with pytest.raises(ValueError):
        union.route_kernel(route, world_size, world_size)


def test_pdl_is_a_launch_flag_not_a_kernel() -> None:
    """The same kernel serves both launch_with_pdl values of a route."""
    for (arch, world_size, dtype, pdl, spec), route in union.ROUTES.items():
        twin = union.ROUTES.get((arch, world_size, dtype, not pdl, spec))
        if twin is not None:
            assert _kernel_names(twin) == _kernel_names(route)


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("capability", sorted(_ARCHES.values()))
def test_route_scope_is_sm100_sm103_at_world_sizes_2_4_8(
    world_size: int, capability: tuple[int, int]
) -> None:
    # The all-reduce output is runtime-optional in every union kernel, so the
    # scope is the exported (architecture, world size) set alone.
    assert union.route_applies(world_size=world_size, device_capability=capability)
    assert not union.route_applies(world_size=world_size, device_capability=_SM120)


@pytest.mark.parametrize("world_size", (1, 3, 16))
def test_route_scope_rejects_unexported_world_sizes(world_size: int) -> None:
    for arch, capability in _ARCHES.items():
        assert not union.route_applies(
            world_size=world_size, device_capability=capability
        )
        with pytest.raises(ValueError):
            union.route_for(
                arch=arch,
                world_size=world_size,
                dtype_name="float16",
                launch_with_pdl=False,
                token_num=64,
                active_experts=8,
            )


def test_launch_grid_rule() -> None:
    assert union.launch_grid_x(1, False, 148, 1, 4) == 4
    assert union.launch_grid_x(2048, False, 148, 1, 4) == 148
    assert union.launch_grid_x(1, False, 148, 4, 4) == 4
    assert union.launch_grid_x(64, False, 148, 4, 4) == 256
    assert union.launch_grid_x(2048, False, 148, 4, 4) == 592
    assert union.launch_grid_x(2048, False, 148, 5, 4) == 740
    assert union.launch_grid_x(2048, False, 148, 5, 4, 175) == 700
    assert union.launch_grid_x(64, False, 148, 5, 4, 175) == 256
    assert union.launch_grid_x(2048, False, 148, 4, 4, 175) == 592
    assert union.launch_grid_x(1, False, 148, 1, 1) == 1
    assert union.launch_grid_x(64, False, 148, 1, 1) == 64
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
    for route in union.ROUTES.values():
        cluster = union.KERNELS[_kernel_names(route)[0]].cluster
        grid = union.launch_grid_x(
            2048,
            route.cooperative,
            148,
            route.persistent_ctas_per_sm,
            cluster,
            route.max_persistent_clusters,
        )
        assert grid > 0 and grid % cluster == 0


def test_compile_flags_bind_the_launcher_to_one_kernel_per_architecture() -> None:
    for name, kernel in union.KERNELS.items():
        defines = union.kernel_defines(name)
        assert f"-DCAKE_UNION_KERNEL={union.kernel_symbol(name)}" in defines
        assert f"-DCAKE_UNION_WORLD_SIZE={kernel.world_size}" in defines
        assert f"-DCAKE_UNION_BLOCK={kernel.block}" in defines
        assert f"-DCAKE_UNION_CLUSTER={kernel.cluster}" in defines
        ctype = "__half" if kernel.dtype == "float16" else "__nv_bfloat16"
        assert f"-DCAKE_UNION_DTYPE={ctype}" in defines
    name = next(iter(union.KERNELS))
    for arch, gencode in (
        ("sm_100a", "-gencode=arch=compute_100a,code=sm_100a"),
        ("sm_103a", "-gencode=arch=compute_103a,code=sm_103a"),
    ):
        flags = union.compile_flags(name, arch)
        assert gencode in flags and "--use_fast_math" in flags
        assert set(union.kernel_defines(name)) <= set(flags)
    with pytest.raises(ValueError):
        union.compile_flags(name, "sm_120a")


def _union_arguments(world_size: int) -> dict:
    tokens = 4
    hidden = union.HIDDEN_DIM
    activation = torch.empty(2, tokens, hidden, dtype=torch.float16)
    return dict(
        world_size=world_size,
        world_rank=0,
        token_num=tokens,
        hidden_dim=hidden,
        workspace_ptrs=torch.zeros(3 * world_size + 1, dtype=torch.int64),
        launch_with_pdl=False,
        residual_in=activation[0],
        rms_gamma=activation[0, 0],
        rms_eps=1e-6,
        scale_factor=1.0,
        moe_reduction_device_num_experts=2,
        moe_reduction_scale_input=torch.empty(2, tokens),
        moe_reduction_active_experts_token_input=activation,
        moe_reduction_token_input=activation[1],
        moe_allreduce_out=torch.empty_like(activation[0]),
        residual_out=torch.empty_like(activation[0]),
        norm_out=torch.empty_like(activation[0]),
        weight_bias=None,
    )


def test_run_rejects_unexported_world_sizes_before_touching_the_device() -> None:
    with pytest.raises(ValueError):
        union.run_cake_moe_allreduce_union(backend="cake", **_union_arguments(16))
    with pytest.raises(ValueError):
        union.run_cake_moe_allreduce_union(backend="trtllm", **_union_arguments(4))
    arguments = _union_arguments(4)
    arguments["workspace_ptrs"] = torch.zeros(3, dtype=torch.int64)
    with pytest.raises(ValueError):
        union.run_cake_moe_allreduce_union(backend="cake", **arguments)


def test_run_passes_an_absent_allreduce_output_to_the_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``moe_allreduce_out=None`` reaches the launcher as a tvm-ffi ``Optional`` (the kernel's
    runtime ``emit_moe_allreduce_out`` flag is then cleared); the public ``scale_factor`` is not a
    launcher argument (the union kernels emit no quant output)."""

    (arch, world_size, dtype_name, pdl, _spec), route = sorted(union.ROUTES.items())[0]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    runs: list[tuple] = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(union, "_device_arch", lambda index: arch)
    monkeypatch.setattr(union, "_sm_count", lambda index: 148)
    monkeypatch.setattr(
        union,
        "route_for",
        lambda **kwargs: ((arch, world_size, dtype_name, pdl, _spec), route),
    )
    monkeypatch.setattr(
        union,
        "load",
        lambda name, arch: SimpleNamespace(run=lambda *args: runs.append(args)),
    )
    tokens = 4
    activation = torch.empty(2, tokens, union.HIDDEN_DIM, dtype=dtype)
    arguments = dict(
        backend="cake",
        world_size=world_size,
        world_rank=0,
        token_num=tokens,
        hidden_dim=union.HIDDEN_DIM,
        workspace_ptrs=torch.zeros(3 * world_size + 1, dtype=torch.int64),
        launch_with_pdl=pdl,
        residual_in=activation[0],
        rms_gamma=activation[0, 0],
        rms_eps=1e-6,
        scale_factor=1.0,
        moe_reduction_device_num_experts=2,
        moe_reduction_scale_input=torch.empty(2, tokens),
        moe_reduction_active_experts_token_input=activation,
        moe_reduction_token_input=activation[1],
        moe_allreduce_out=None,
        residual_out=torch.empty_like(activation[0]),
        norm_out=torch.empty_like(activation[0]),
        weight_bias=None,
    )

    union.run_cake_moe_allreduce_union(**arguments)
    allreduce_out = torch.empty_like(activation[0])
    union.run_cake_moe_allreduce_union(
        **{**arguments, "moe_allreduce_out": allreduce_out}
    )

    assert len(runs) == 2
    absent, present = runs
    assert absent[5] is None and present[5] is allreduce_out
    assert absent[6] is arguments["residual_out"] and absent[7] is arguments["norm_out"]
    assert absent[8] is arguments["workspace_ptrs"]
    assert len(absent) == 17 and 1.0 not in absent[9:]


def _arguments(world_size: int, *, emit_allreduce: bool) -> dict:
    tokens = 4
    hidden = union.HIDDEN_DIM
    activation = torch.empty(2, tokens, hidden, dtype=torch.float16)
    return dict(
        world_size=world_size,
        world_rank=1,
        token_num=tokens,
        hidden_dim=hidden,
        workspace_ptrs=torch.zeros(3 * world_size + 1, dtype=torch.int64),
        launch_with_pdl=True,
        residual_in=activation[0],
        rms_gamma=activation[0, 0],
        rms_eps=1e-6,
        scale_factor=1.0,
        moe_reduction_device_num_experts=2,
        moe_reduction_scale_input=torch.empty(2, tokens),
        moe_reduction_active_experts_token_input=activation,
        moe_reduction_token_input=activation[1],
        layout_code=None,
        moe_allreduce_out=torch.empty_like(activation[0]) if emit_allreduce else None,
        residual_out=torch.empty_like(activation[0]),
        norm_out=torch.empty_like(activation[0]),
        quant_out=None,
        scale_out=None,
        weight_bias=1.0,
    )


def _isolate_backends(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]
) -> list[dict]:
    union_calls: list[dict] = []
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
        "get_trtllm_comm_module",
        lambda: pytest.fail("TRT-LLM module must not load for backend='cake'"),
    )
    return union_calls


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("capability", sorted(_ARCHES.values()))
def test_cake_backend_routes_sm100_sm103_with_allreduce_output_to_the_union(
    monkeypatch: pytest.MonkeyPatch, world_size: int, capability: tuple[int, int]
) -> None:
    union_calls = _isolate_backends(monkeypatch, capability)
    arguments = _arguments(world_size, emit_allreduce=True)

    trtllm_ar.trtllm_moe_allreduce_fusion(**arguments, backend="cake")

    assert len(union_calls) == 1
    call = union_calls[0]
    assert call["backend"] == "cake"
    assert call["world_size"] == world_size and call["world_rank"] == 1
    assert call["workspace_ptrs"] is arguments["workspace_ptrs"]
    assert call["moe_allreduce_out"] is arguments["moe_allreduce_out"]
    assert call["residual_out"] is arguments["residual_out"]
    assert call["norm_out"] is arguments["norm_out"]
    assert call["launch_with_pdl"] is True and call["weight_bias"] == 1.0


@pytest.mark.parametrize("world_size", _WORLD_SIZES)
@pytest.mark.parametrize("capability", sorted(_ARCHES.values()))
def test_cake_backend_routes_calls_without_allreduce_output_to_the_union(
    monkeypatch: pytest.MonkeyPatch, world_size: int, capability: tuple[int, int]
) -> None:
    # A call without ``moe_allreduce_out`` runs the same union kernels with the
    # all-reduce store disabled at launch; no scratch tensor is involved.
    union_calls = _isolate_backends(monkeypatch, capability)
    arguments = _arguments(world_size, emit_allreduce=False)

    trtllm_ar.trtllm_moe_allreduce_fusion(**arguments, backend="cake")

    assert len(union_calls) == 1
    call = union_calls[0]
    assert call["world_size"] == world_size
    assert call["moe_allreduce_out"] is None
    assert call["residual_out"] is arguments["residual_out"]
    assert call["norm_out"] is arguments["norm_out"]
    assert not hasattr(union, "scratch_allreduce_output")


def test_workspace_creation_has_no_pointer_registry() -> None:
    assert not hasattr(trtllm_ar, "register_cake_moe_allreduce_workspace_pointers")
    assert not hasattr(union, "register_workspace_pointers")
    assert not hasattr(union, "workspace_pointers")
