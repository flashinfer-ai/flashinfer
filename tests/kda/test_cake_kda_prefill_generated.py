"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""Generated BF16 KDA prefill programs in the shared KDA loader.

``_RENDERED_MODULES`` registers the generated programs (each compiled from its
own device unit and host binding under ``csrc/kda/prefill/``) and
``_RENDERED_ROUTES`` points the shipped modules of each architecture at them;
``_RENDERED_WORK_BIN_ROUTES`` names the programs built with work bins that
launches asking for work bins take instead, ``_RENDERED_OWNER_HELPER_ROUTES``
the owner/helper program a call takes when the device holds its grid, and
``_RENDERED_OWNER_HELPER_POOL_ROUTES`` its pooled program;
``_RENDERED_MAP_FOLD_ROUTES`` names the chain main bodies that also write the
window maps, which the chain composite launches instead of the map pass, and
``_RENDERED_PAIR_SHARE_ROUTES`` the M64 value-split programs that run a task's
two value halves as a 2-CTA cluster, which long launches take instead, and
``_RENDERED_WORK_BIN_TURN_ROUTES`` the work-binned programs that also turn a
bin's state over in the background, which work-binned launches of short tasks
take instead of the work-binned program, and
``_RENDERED_M64_OWNER_HELPER_ROUTES`` the M64 value-split programs whose helper
CTAs prepare a task's chunks for its two value-half owners, which long
cooperative launches take instead of the pair-share or routed program, and
``_RENDERED_WORK_BIN_MIX_ROUTES`` the bin-mix builds of the turnover programs,
which turnover launches whose bin table the bin-mix rule admits take
cooperatively instead of the turnover program on the ``WORK_BIN_MIX_ARCHES``
parts.

The CPU checks hold both tables to the units, to the shipped argument plans and
to the factory lookup. The GPU checks run the prepared BF16 prefill on the
serving shapes (FP32 in-place state pool, reversed int32 state indices, a
strided BF16 beta-logit slice, checkpoint rows every 64 tokens) with the routes
in place and with the routes cleared, each against an FP32 token-by-token
reference, and a launch with work bins bit for bit against the one-task-per-CTA
program and against the selection without work bins; hold a work-binned launch
of short tasks to the turnover build (bit for bit the work-binned launch, with
and without checkpoint rows, on both sides of the part's chunk limit); replay
the routed launch from a CUDA graph (a turnover launch among the rows); and
rebind it through a plan-cache hit.  An admitted bin-mix launch matches its
turnover launch bit for bit, a bin-mix launch the driver refuses runs the
turnover build instead, and on another part the same launches keep the
turnover build.

A program that takes tensor maps encodes them once per preparation into a host
buffer (``("workspace", "tensor_maps")`` in its plan); the checks hold every
stored map equal to a fresh encode of the bound tensors after preparation and
after a plan-cache rebind to new tensors.
"""

import re
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import flashinfer.cake_kda_tf32_runtime as runtime
import flashinfer.jit.cake_kda_tf32 as registry
from flashinfer import (
    KDAPrefillPlanCache,
    kda_prefill_supports_fp32_checkpoints,
    prepare_bf16_kda_prefill,
)
from flashinfer.cake_kda_tf32_runtime import (
    _MAP_FIRST_MODULES,
    _affine_pair_map_fold_enabled,
    _chain_map_fold_enabled,
)

ROOT = Path(registry.__file__).resolve().parents[2]
UNIT_DIR = "csrc/kda/prefill"
NAME = re.compile(r"cake_kda_prefill_[0-9a-f]{20}")
LEAN_KEYS = {"sources", "ffi_entry", "compile_flags", "arg_plan", "arches"}
GRID = [["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]
PLAN_KINDS = {"buffer", "tma_buffer", "parameter", "workspace", "grid"}
MAPS = list(registry.TENSOR_MAP_PLAN_ENTRY)
TENSOR_KINDS = {"buffer", "tma_buffer"}

HEAD_DIM = 128
CHECKPOINT_EVERY = 64
ATOL = RTOL = 1e-2

GENERATED = sorted(registry._RENDERED_MODULES)

requires_programs = pytest.mark.skipif(
    not GENERATED, reason="no generated KDA prefill program is registered"
)
requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="KDA export requires SM100a or SM103a",
)


# --------------------------------------------------------------------- tables


@requires_programs
def test_routes_point_shipped_modules_at_generated_programs():
    assert set(registry._RENDERED_ROUTES) == set(registry.ARCHES)
    routed = set()
    for arch, routes in registry._RENDERED_ROUTES.items():
        factory_modules = {
            module
            for variants in registry.FACTORIES[arch].values()
            for module in variants.values()
        }
        for module, program in routes.items():
            assert module in registry._MODULE_ROWS, module
            assert arch in registry._MODULE_ROWS[module][2], (arch, module)
            assert module in factory_modules, (arch, module)
            assert program in registry._RENDERED_MODULES, (arch, program)
            assert arch in registry._RENDERED_MODULES[program]["arches"]
            routed.add(program)
    assert set(registry._RENDERED_HEAD_ROUTES) == set(registry.ARCHES)
    for arch, modules in registry._RENDERED_HEAD_ROUTES.items():
        for module, by_heads in modules.items():
            # A head count the table lacks falls back to the module's route.
            assert module in registry._RENDERED_ROUTES[arch], (arch, module)
            for heads, (program, ctas_per_head) in by_heads.items():
                assert type(heads) is int and heads > 0, (arch, module, heads)
                assert type(ctas_per_head) is int and ctas_per_head > 0
                assert program in registry._RENDERED_MODULES, (arch, program)
                assert arch in registry._RENDERED_MODULES[program]["arches"]
                routed.add(program)
    assert set(registry._RENDERED_MERGE_ROUTES) == set(registry.ARCHES)
    for arch, modules in registry._RENDERED_MERGE_ROUTES.items():
        for module, program in modules.items():
            # Every other launch of the module keeps its route.
            assert module in registry._RENDERED_ROUTES[arch], (arch, module)
            assert program in registry._RENDERED_MODULES, (arch, program)
            assert arch in registry._RENDERED_MODULES[program]["arches"]
            routed.add(program)
    assert set(registry._RENDERED_WORK_BIN_ROUTES) == set(registry.ARCHES)
    for arch, modules in registry._RENDERED_WORK_BIN_ROUTES.items():
        for module, program in modules.items():
            # A launch that asks for no work bins keeps the module's route.
            assert module in registry._RENDERED_ROUTES[arch], (arch, module)
            assert program in registry._RENDERED_MODULES, (arch, program)
            assert arch in registry._RENDERED_MODULES[program]["arches"]
            routed.add(program)
    assert set(registry._RENDERED_OWNER_HELPER_ROUTES) == set(registry.ARCHES)
    for arch, modules in registry._RENDERED_OWNER_HELPER_ROUTES.items():
        for module, program in modules.items():
            # A call the owner/helper launch does not fit takes the module's route.
            assert module in registry._RENDERED_ROUTES[arch], (arch, module)
            assert program in registry._RENDERED_MODULES, (arch, program)
            assert arch in registry._RENDERED_MODULES[program]["arches"]
            routed.add(program)
    assert set(registry._RENDERED_OWNER_HELPER_POOL_ROUTES) == set(registry.ARCHES)
    for arch, modules in registry._RENDERED_OWNER_HELPER_POOL_ROUTES.items():
        for module, program in modules.items():
            # A call the pool does not serve takes the owner/helper program.
            assert module in registry._RENDERED_OWNER_HELPER_ROUTES[arch], (
                arch,
                module,
            )
            assert program != registry._RENDERED_OWNER_HELPER_ROUTES[arch][module]
            assert program in registry._RENDERED_MODULES, (arch, program)
            assert arch in registry._RENDERED_MODULES[program]["arches"]
            routed.add(program)
    assert set(registry._RENDERED_FOLD_ROUTES) == set(registry.ARCHES)
    for arch, modules in registry._RENDERED_FOLD_ROUTES.items():
        for module, program in modules.items():
            # Every other launch of the module keeps its route.
            assert module in registry._RENDERED_ROUTES[arch], (arch, module)
            assert program in registry._RENDERED_MODULES, (arch, program)
            assert arch in registry._RENDERED_MODULES[program]["arches"]
            routed.add(program)
    assert set(registry._RENDERED_MAP_FOLD_ROUTES) == set(registry.ARCHES)
    for arch, modules in registry._RENDERED_MAP_FOLD_ROUTES.items():
        for module, program in modules.items():
            # Every other launch of the module keeps its route.
            assert module in registry._RENDERED_ROUTES[arch], (arch, module)
            assert program in registry._RENDERED_MODULES, (arch, program)
            assert arch in registry._RENDERED_MODULES[program]["arches"]
            routed.add(program)
    assert set(registry._RENDERED_PAIR_SHARE_ROUTES) == set(registry.ARCHES)
    for arch, modules in registry._RENDERED_PAIR_SHARE_ROUTES.items():
        for module, program in modules.items():
            # Short launches of the module keep its route.
            assert module in registry._RENDERED_ROUTES[arch], (arch, module)
            assert program != registry._RENDERED_ROUTES[arch][module]
            assert program in registry._RENDERED_MODULES, (arch, program)
            assert arch in registry._RENDERED_MODULES[program]["arches"]
            routed.add(program)
    assert set(registry._RENDERED_WORK_BIN_TURN_ROUTES) == set(registry.ARCHES)
    for arch, modules in registry._RENDERED_WORK_BIN_TURN_ROUTES.items():
        for module, program in modules.items():
            # Every other work-binned launch of the module keeps its work-binned
            # program, a launch without work bins its route.
            assert module in registry._RENDERED_WORK_BIN_ROUTES[arch], (arch, module)
            assert program != registry._RENDERED_WORK_BIN_ROUTES[arch][module]
            assert program in registry._RENDERED_MODULES, (arch, program)
            assert arch in registry._RENDERED_MODULES[program]["arches"]
            routed.add(program)
    assert set(registry._RENDERED_M64_OWNER_HELPER_ROUTES) == set(registry.ARCHES)
    for arch, modules in registry._RENDERED_M64_OWNER_HELPER_ROUTES.items():
        for module, program in modules.items():
            # Every other launch of the module keeps its pair-share or routed
            # program.
            assert module in registry._RENDERED_PAIR_SHARE_ROUTES[arch], (arch, module)
            assert program != registry._RENDERED_PAIR_SHARE_ROUTES[arch][module]
            assert program != registry._RENDERED_ROUTES[arch][module]
            assert program in registry._RENDERED_MODULES, (arch, program)
            assert arch in registry._RENDERED_MODULES[program]["arches"]
            routed.add(program)
    assert set(registry._RENDERED_WORK_BIN_MIX_ROUTES) == set(registry.ARCHES)
    for arch, modules in registry._RENDERED_WORK_BIN_MIX_ROUTES.items():
        # A bin-mix build ships only where the runtime launches it.
        assert not modules or arch in runtime.WORK_BIN_MIX_ARCHES, arch
        for module, program in modules.items():
            # Every turnover launch the bin-mix rule does not admit keeps the
            # module's turnover program.
            assert module in registry._RENDERED_WORK_BIN_TURN_ROUTES[arch], (
                arch,
                module,
            )
            assert program != registry._RENDERED_WORK_BIN_TURN_ROUTES[arch][module]
            assert program in registry._RENDERED_MODULES, (arch, program)
            assert arch in registry._RENDERED_MODULES[program]["arches"]
            routed.add(program)
    assert routed == set(GENERATED)


@requires_programs
@pytest.mark.parametrize("name", GENERATED)
def test_generated_record_is_lean_and_names_its_units(name):
    record = registry._RENDERED_MODULES[name]
    assert NAME.fullmatch(name)
    assert set(record) == LEAN_KEYS
    assert registry.MODULES[name] is record
    assert name not in registry._MODULE_ROWS
    assert record["ffi_entry"] == "run"
    assert record["sources"] == [
        f"{UNIT_DIR}/{name}_kernel.cu",
        f"{UNIT_DIR}/{name}_binding.cu",
    ]
    assert record["compile_flags"][0] == "--use_fast_math"
    assert record["arches"] and set(record["arches"]) <= set(registry.ARCHES)
    plan = record["arg_plan"]
    assert [list(slot) for slot in plan[-3:]] == GRID
    assert {kind for kind, _ in plan} <= PLAN_KINDS
    assert [list(slot) for slot in plan if slot[0] == "workspace"] in ([], [MAPS])
    kernel = (ROOT / record["sources"][0]).read_text()
    binding = (ROOT / record["sources"][1]).read_text()
    symbol = re.compile(rf"\bkernel_{name}\s*\(")
    assert symbol.search(kernel), "the device unit does not define the kernel"
    assert symbol.search(binding), "the binding does not declare the kernel"
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run," in binding


@requires_programs
def test_generated_plans_bind_from_the_shipped_plans():
    """The launch classes bind by the shipped plan's names: a generated plan
    must name a subset of them, tensors as tensors and scalars as scalars."""
    for arch in registry.ARCHES:
        routes = [
            *registry._RENDERED_ROUTES[arch].items(),
            *registry._RENDERED_MERGE_ROUTES[arch].items(),
            *registry._RENDERED_WORK_BIN_ROUTES[arch].items(),
            *registry._RENDERED_OWNER_HELPER_ROUTES[arch].items(),
            *registry._RENDERED_OWNER_HELPER_POOL_ROUTES[arch].items(),
            *registry._RENDERED_FOLD_ROUTES[arch].items(),
            *registry._RENDERED_MAP_FOLD_ROUTES[arch].items(),
            *registry._RENDERED_PAIR_SHARE_ROUTES[arch].items(),
            *registry._RENDERED_WORK_BIN_TURN_ROUTES[arch].items(),
            *registry._RENDERED_M64_OWNER_HELPER_ROUTES[arch].items(),
            *registry._RENDERED_WORK_BIN_MIX_ROUTES[arch].items(),
        ]
        for module, program in routes:
            shipped = {
                name: kind for kind, name in registry.MODULES[module]["arg_plan"]
            }
            for kind, name in registry._RENDERED_MODULES[program]["arg_plan"]:
                if kind in ("grid", "workspace"):
                    continue
                where = (arch, module, program, name)
                assert name in shipped, where
                assert (kind in TENSOR_KINDS) == (shipped[name] in TENSOR_KINDS), where
                assert (kind == "parameter") == (shipped[name] == "parameter"), where


@requires_programs
@pytest.mark.parametrize("arch", sorted(registry.ARCHES))
def test_factory_resolves_a_head_count_through_the_head_routes(monkeypatch, arch):
    monkeypatch.setattr(registry, "device_arch", lambda device=None: arch)
    monkeypatch.setattr(
        registry,
        "PreparedKernel",
        lambda name, target: SimpleNamespace(name=name, arch=target),
    )
    by_module = registry._RENDERED_HEAD_ROUTES[arch]
    seen = set()
    for family, variants in registry.FACTORIES[arch].items():
        for (args, kwargs), module in variants.items():
            if module not in by_module:
                continue
            seen.add(module)
            for heads, (program, ctas_per_head) in by_module[module].items():
                kernel = registry._factory(
                    family, *args, route_heads=heads, **dict(kwargs)
                )
                assert (kernel.name, kernel.arch) == (program, arch)
                assert kernel.ctas_per_head == ctas_per_head
            other = registry._factory(family, *args, route_heads=7, **dict(kwargs))
            assert other.name == registry._RENDERED_ROUTES[arch][module]
            assert getattr(other, "ctas_per_head", None) is None
    assert seen == set(by_module)


@requires_programs
@pytest.mark.parametrize("arch", sorted(registry.ARCHES))
def test_factory_resolves_work_bins_through_the_work_bin_routes(monkeypatch, arch):
    monkeypatch.setattr(registry, "device_arch", lambda device=None: arch)
    monkeypatch.setattr(registry, "PreparedKernel", lambda name, target: (name, target))
    by_module = registry._RENDERED_WORK_BIN_ROUTES[arch]
    seen = set()
    for family, variants in registry.FACTORIES[arch].items():
        for (args, kwargs), module in variants.items():
            if module not in by_module:
                continue
            seen.add(module)
            binned = registry._factory(
                family, *args, route_work_bins=True, **dict(kwargs)
            )
            assert binned == (by_module[module], arch)
            plain = registry._factory(family, *args, **dict(kwargs))
            assert plain == (registry._RENDERED_ROUTES[arch][module], arch)
    assert seen == set(by_module)


@requires_programs
@pytest.mark.parametrize("arch", sorted(registry.ARCHES))
def test_factory_resolves_through_the_routes(monkeypatch, arch):
    monkeypatch.setattr(registry, "device_arch", lambda device=None: arch)
    monkeypatch.setattr(registry, "PreparedKernel", lambda name, target: (name, target))
    routes = registry._RENDERED_ROUTES[arch]
    seen = set()
    for family, variants in registry.FACTORIES[arch].items():
        for (args, kwargs), module in variants.items():
            name, target = registry._factory(family, *args, **dict(kwargs))
            assert target == arch
            assert name == routes.get(module, module)
            seen.add(module)
    assert set(routes) <= seen


@requires_programs
@pytest.mark.parametrize("arch", sorted(registry.ARCHES))
def test_factory_resolves_the_owner_helper_routes(monkeypatch, arch):
    monkeypatch.setattr(registry, "device_arch", lambda device=None: arch)
    monkeypatch.setattr(
        registry,
        "PreparedKernel",
        lambda name, target: SimpleNamespace(name=name, arch=target),
    )
    by_module = registry._RENDERED_OWNER_HELPER_ROUTES[arch]
    routes = registry._RENDERED_ROUTES[arch]
    seen = set()
    for family, variants in registry.FACTORIES[arch].items():
        for (args, kwargs), module in variants.items():
            kernel = registry._factory(family, *args, owner_helper=True, **dict(kwargs))
            if module in by_module:
                seen.add(module)
                assert (kernel.name, kernel.arch) == (by_module[module], arch)
                assert kernel.owner_helper is True
            else:
                # A module without an owner/helper program keeps its route.
                assert kernel.name == routes.get(module, module)
                assert getattr(kernel, "owner_helper", False) is False
    assert seen == set(by_module)


@requires_programs
@pytest.mark.parametrize("arch", sorted(registry.ARCHES))
def test_factory_resolves_the_owner_helper_pool_routes(monkeypatch, arch):
    monkeypatch.setattr(registry, "device_arch", lambda device=None: arch)
    monkeypatch.setattr(
        registry,
        "PreparedKernel",
        lambda name, target: SimpleNamespace(name=name, arch=target),
    )
    by_module = registry._RENDERED_OWNER_HELPER_POOL_ROUTES[arch]
    owner_helper = registry._RENDERED_OWNER_HELPER_ROUTES[arch]
    routes = registry._RENDERED_ROUTES[arch]
    seen = set()
    for family, variants in registry.FACTORIES[arch].items():
        for (args, kwargs), module in variants.items():
            kernel = registry._factory(
                family, *args, owner_helper=True, owner_helper_pool=True, **dict(kwargs)
            )
            if module in by_module:
                seen.add(module)
                assert (kernel.name, kernel.arch) == (by_module[module], arch)
                assert kernel.owner_helper is True and kernel.owner_helper_pool is True
            elif module in owner_helper:
                # A module without a pooled program keeps its owner/helper program.
                assert kernel.name == owner_helper[module]
                assert kernel.owner_helper is True
                assert getattr(kernel, "owner_helper_pool", False) is False
            else:
                assert kernel.name == routes.get(module, module)
                assert getattr(kernel, "owner_helper", False) is False
            # The pool alone asks for nothing: the call keeps its route.
            alone = registry._factory(
                family, *args, owner_helper_pool=True, **dict(kwargs)
            )
            assert alone.name == routes.get(module, module)
            assert getattr(alone, "owner_helper_pool", False) is False
    assert seen == set(by_module)


@requires_programs
def test_spec_compiles_the_generated_units(monkeypatch):
    name = GENERATED[0]
    record = registry._RENDERED_MODULES[name]
    arch = record["arches"][0]
    captured = {}
    monkeypatch.setattr(
        registry, "gen_jit_spec", lambda **kwargs: captured.update(kwargs) or kwargs
    )
    registry.spec.cache_clear()
    try:
        registry.spec(name, arch)
    finally:
        registry.spec.cache_clear()
    assert captured["name"] == f"{name}_{arch}"
    assert [Path(path).name for path in captured["sources"]] == [
        Path(source).name for source in record["sources"]
    ]
    flags = list(record["compile_flags"])
    assert list(captured["extra_cuda_cflags"][-len(flags) :]) == flags
    assert Path(captured["extra_include_paths"][0]).name == "prefill"


def _host_encoded(name):
    plan = registry._RENDERED_MODULES[name]["arg_plan"]
    return MAPS in [list(slot) for slot in plan]


ENCODED = [name for name in GENERATED if _host_encoded(name)]
PLAIN = [name for name in GENERATED if not _host_encoded(name)]


def _function_body(text, signature):
    start = text.index(signature)
    return text[start : text.index("\n}\n", start)]


@requires_programs
@pytest.mark.parametrize("name", GENERATED)
def test_a_program_with_tensor_maps_takes_them_from_the_loader(name):
    """The binding of a program that takes tensor maps is the host-encoded form:
    ``run`` copies the maps from the loader's buffer and encodes nothing, and
    ``encode_tensor_maps`` encodes each map from its own ``*_tma`` tensor and
    nothing else, so the stored bytes depend on that tensor's address, extents,
    strides and dtype alone -- what a plan-cache rebind tracks."""
    record = registry._RENDERED_MODULES[name]
    plan = [list(slot) for slot in record["arg_plan"]]
    binding = (ROOT / record["sources"][1]).read_text()
    sources = [slot_name for kind, slot_name in plan if kind == "tma_buffer"]
    assert all(source.endswith("_tma") for source in sources), sources
    encoders = re.findall(r"^CUtensorMap EncodeTma_(\w+)\(", binding, re.M)
    if not encoders:
        assert name in PLAIN and "encode_tensor_maps" not in binding
        assert not sources
        return
    assert name in ENCODED
    # The buffer sits after the kernel's arguments, before the launch argument
    # and the grid.
    tail = [kind for kind, _ in plan[plan.index(MAPS) :]]
    assert tail in (
        ["workspace", "grid", "grid", "grid"],
        ["workspace", "launch", "grid", "grid", "grid"],
    )
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(encode_tensor_maps," in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(tensor_map_count," in binding
    run = _function_body(binding, "void Run(")
    assert "EncodeTma_" not in run and "cuTensorMapEncodeTiled" not in run
    encode = _function_body(binding, "void EncodeTensorMaps(")
    stores = re.findall(
        r"^  maps\[(\d+)\] = EncodeTma_(\w+)\(([^)]*)\);$", encode, re.M
    )
    assert [int(index) for index, _, _ in stores] == list(range(len(stores)))
    assert sorted(map_name for _, map_name, _ in stores) == sorted(encoders)
    count = re.search(r"int64_t TensorMapCount\(\) \{ return (\d+); \}", binding)
    assert count and int(count.group(1)) == len(stores)
    for _, map_name, arguments in stores:
        assert re.fullmatch(r"a_\w+_tma", arguments), (map_name, arguments)
        assert arguments[2:] in sources, (map_name, arguments)


@requires_programs
def test_prepared_kernel_calls_run_without_a_descriptor_workspace(monkeypatch):
    assert PLAIN, "every generated program takes tensor maps"
    name = PLAIN[0]
    record = registry._RENDERED_MODULES[name]
    calls = []
    native = SimpleNamespace(run=lambda *args: calls.append(args))
    monkeypatch.setattr(registry, "load", lambda module, arch: native)
    kernel = registry.PreparedKernel(name, record["arches"][0])
    assert kernel._prepare is None and not kernel.encodes_on_host
    assert kernel.descriptor_storage is None
    names = [slot_name for kind, slot_name in record["arg_plan"] if kind != "grid"]
    bindings = {slot_name: object() for slot_name in names}
    bindings["unused_by_this_plan"] = object()
    kernel.prepare(grid=(3, 2), **bindings)
    assert calls == []
    kernel.launch(grid=(3, 2), **bindings)
    assert calls == [(*(bindings[slot_name] for slot_name in names), 3, 2, 1)]


@requires_programs
def test_prepared_kernel_encodes_the_maps_into_its_host_buffer(monkeypatch):
    assert ENCODED, "no generated program takes tensor maps"
    name = ENCODED[0]
    record = registry._RENDERED_MODULES[name]
    plan = [tuple(slot) for slot in record["arg_plan"]]
    encodes, runs = [], []

    def encode(*args):
        encodes.append(args)
        args[-1].fill_(len(encodes))

    native = SimpleNamespace(
        run=lambda *args: runs.append(args),
        encode_tensor_maps=encode,
        tensor_map_count=lambda: 3,
    )
    monkeypatch.setattr(registry, "load", lambda module, arch: native)
    kernel = registry.PreparedKernel(name, record["arches"][0])
    storage = kernel.descriptor_storage
    assert kernel.encodes_on_host
    assert storage.device.type == "cpu" and storage.dtype == torch.uint8
    assert storage.numel() == 3 * 128 and storage.data_ptr() % 128 == 0
    assert storage.is_contiguous()
    names = [n for kind, n in plan if kind not in ("grid", "workspace")]
    bindings = {n: object() for n in names}
    with pytest.raises(RuntimeError, match="must be initialized during preparation"):
        kernel.launch(grid=(5,), **bindings)
    kernel.prepare(grid=(5,), **bindings)
    # The encode entry takes run's arguments without the launch argument and
    # the grid; the buffer is the plan's workspace slot.
    expected = [
        storage if kind == "workspace" else bindings[n]
        for kind, n in plan
        if kind not in ("grid", "launch")
    ]
    assert len(encodes) == 1 and len(encodes[0]) == len(expected)
    assert all(a is b for a, b in zip(encodes[0], expected, strict=True))
    assert bool((storage == 1).all())
    kernel.launch(grid=(5,), **bindings)
    launched = [
        storage if kind == "workspace" else bindings[n]
        for kind, n in plan
        if kind != "grid"
    ]
    assert len(runs) == 1 and runs[0][-3:] == (5, 1, 1)
    assert all(a is b for a, b in zip(runs[0][:-3], launched, strict=True))
    # A fresh encode goes to a buffer of its own.
    fresh = kernel.encoded_tensor_maps(grid=(5,), **bindings)
    assert fresh.data_ptr() != storage.data_ptr() and bool((fresh == 2).all())
    assert bool((storage == 1).all())


@requires_programs
def test_map_first_programs_wait_before_touching_memory():
    """The chain route may launch a map program ahead of the index gather and
    the main kernel (``_MAP_FIRST_MODULES``).  Its generated replacement keeps
    that safe: every thread runs the grid-dependency wait before any memory
    argument is used, and the main kernel's final state, bound as
    ``initial_state_f32``, is never read."""
    shipped = (
        "cake_kda_bf16_f74a288bb51b335a1ccf",
        "cake_kda_bf16_5639519f5c942ed86e4f",
    )
    assert set(shipped) <= _MAP_FIRST_MODULES
    rendered = _MAP_FIRST_MODULES - set(shipped)
    routed = {
        routes[module]
        for routes in registry._RENDERED_ROUTES.values()
        for module in shipped
        if module in routes
    }
    # Every generated program a map module routes to, and nothing else.
    assert rendered == routed and rendered, (sorted(rendered), sorted(routed))
    assert rendered <= set(GENERATED)
    for name in sorted(rendered):
        record = registry._RENDERED_MODULES[name]
        text = (ROOT / record["sources"][0]).read_text()
        match = re.search(rf"\bvoid\s+kernel_{name}\s*\(([^)]*)\)\s*\{{", text)
        assert match, name
        signature, body = match.group(1), text[match.end() :]
        memory = {
            parameter.split()[-1]
            for parameter in signature.split(",")
            if "*" in parameter or "CUtensorMap" in parameter
        }
        assert "initial_state_f32" in memory
        wait = '\n    asm volatile("griddepcontrol.wait;" ::: "memory");\n'
        assert body.count("griddepcontrol.wait") == 1 and wait in body, name
        prologue = body[: body.index(wait)]
        used = set(re.findall(r"\b[A-Za-z_]\w*\b", prologue)) & memory
        assert not used, (name, sorted(used))
        assert not re.search(
            r"\b(?:ld|st)\.global|\bcp\.(?:async|reduce)|\b(?:atom|red|prefetch|multimem)\.",
            prologue,
        ), name
        assert not re.search(r"\binitial_state_f32\b", body), name


CORRECTION_MODULES = (
    "cake_kda_bf16_2b5f2c6b7cfc10ffa244",
    "cake_kda_bf16_f66adfecf2022fc198b6",
)
OUTPUT_REDUCE = re.compile(r"\bcp\.reduce\.async\.bulk\.tensor\.[\w:.]*\.add\b")
TAIL_REDUCE = re.compile(r"\bred\.[\w:.]*global\.add\.noftz\.bf16x2\b")


@requires_programs
def test_merging_correction_builds_reduce_into_the_composite():
    """The chain route's correction modules have a merging build per
    architecture (``_RENDERED_MERGE_ROUTES``): a generated program of its own,
    never the module's routed program, that reduce-adds its output tiles into
    the caller's output (a bulk tensor reduce, a packed BF16 add for a tail
    chunk).  The routed program, which every other launch keeps, stores."""
    assert set(registry._RENDERED_MERGE_ROUTES) == set(registry.ARCHES)
    for arch, merging in registry._RENDERED_MERGE_ROUTES.items():
        assert sorted(merging) == list(CORRECTION_MODULES), arch
        for module, name in merging.items():
            routed = registry._RENDERED_ROUTES[arch][module]
            assert name in GENERATED and name != routed, (arch, module)
            assert arch in registry._RENDERED_MODULES[name]["arches"]
            record = registry._RENDERED_MODULES[name]
            assert record["arg_plan"] == registry._RENDERED_MODULES[routed]["arg_plan"]
            text = (ROOT / record["sources"][0]).read_text()
            assert OUTPUT_REDUCE.search(text) and TAIL_REDUCE.search(text), name
            text = (ROOT / registry._RENDERED_MODULES[routed]["sources"][0]).read_text()
            assert not OUTPUT_REDUCE.search(text) and not TAIL_REDUCE.search(text), (
                routed
            )


FOLD_MODULES = ("cake_kda_bf16_15be1db8a50ee4d9fd16",)
FOLD_BARRIER = re.compile(r"\bfold_p_ready_addr\b")


@requires_programs
def test_pair_map_fold_builds_take_the_main_bodys_plan():
    """The affine apply route's main module has a pair-map fold build
    (``_RENDERED_FOLD_ROUTES``): a generated program of its own, never the
    module's routed program, with the routed program's argument plan (the
    composite binds the pair maps to the plan's ``tape_e`` / ``norm_inv_out``)."""
    assert set(registry._RENDERED_FOLD_ROUTES) == set(registry.ARCHES)
    for arch, folding in registry._RENDERED_FOLD_ROUTES.items():
        assert sorted(folding) == list(FOLD_MODULES), arch
        for module, name in folding.items():
            routed = registry._RENDERED_ROUTES[arch][module]
            assert name in GENERATED and name != routed, (arch, module)
            record = registry._RENDERED_MODULES[name]
            assert arch in record["arches"]
            assert record["arg_plan"] == registry._RENDERED_MODULES[routed]["arg_plan"]
            plan = {slot for _, slot in record["arg_plan"]}
            assert {"tape_e", "norm_inv_out"} <= plan, name
            text = (ROOT / record["sources"][0]).read_text()
            assert FOLD_BARRIER.search(text), name
            text = (ROOT / registry._RENDERED_MODULES[routed]["sources"][0]).read_text()
            assert not FOLD_BARRIER.search(text), routed


MAP_FOLD_MODULES = (
    "cake_kda_bf16_4ab394a09c63f264d7bd",
    "cake_kda_bf16_50c0f288f4c8d7ca03a8",
)
MAP_FOLD_BARRIER = re.compile(r"\bmap_fold_inp_ready_addr\b")


@requires_programs
def test_chain_map_fold_builds_take_the_main_bodys_plan():
    """The chain route's main modules have map-fold builds
    (``_RENDERED_MAP_FOLD_ROUTES``): generated programs of their own, never the
    module's routed program, with the routed program's argument plan (the
    composite binds the map states to the plan's ``tape_e``)."""
    assert set(registry._RENDERED_MAP_FOLD_ROUTES) == set(registry.ARCHES)
    for arch, folding in registry._RENDERED_MAP_FOLD_ROUTES.items():
        assert sorted(folding) == list(MAP_FOLD_MODULES), arch
        for module, name in folding.items():
            routed = registry._RENDERED_ROUTES[arch][module]
            assert name in GENERATED and name != routed, (arch, module)
            record = registry._RENDERED_MODULES[name]
            assert arch in record["arches"]
            assert record["arg_plan"] == registry._RENDERED_MODULES[routed]["arg_plan"]
            assert "tape_e" in {slot for _, slot in record["arg_plan"]}, name
            text = (ROOT / record["sources"][0]).read_text()
            assert MAP_FOLD_BARRIER.search(text), name
            text = (ROOT / registry._RENDERED_MODULES[routed]["sources"][0]).read_text()
            assert not MAP_FOLD_BARRIER.search(text), routed


# ---------------------------------------------------------------- GPU numerics

LAYER_CASES = [
    [8192],
    [16384],
    [8192, 8192],
    [3000, 13384],
    [4096] * 4,
    [8128, 64, 8128, 64],
    [1024, 64] * 8,
    [32768],
]
WRAPPER_CASES = [
    [64] * 4,
    [64] * 16,
    [64] * 64,
    [128] * 16,
    [256] * 16,
    [17, 64, 65, 127, 128, 200, 255, 96] * 2,
]
ROWS = [
    pytest.param(
        lengths,
        heads,
        lower_bound,
        id=f"{'+'.join(map(str, lengths[:4]))}{'..' if len(lengths) > 4 else ''}"
        f"x{len(lengths)}-h{heads}-{'unbounded' if lower_bound is None else 'lb5'}",
    )
    for lengths in LAYER_CASES + WRAPPER_CASES
    for heads in (12, 16)
    for lower_bound in (None, -5.0)
]


def _checkpoint_starts(lengths):
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + (n + CHECKPOINT_EVERY - 1) // CHECKPOINT_EVERY)
    return starts


def _inputs(lengths, heads, *, seed=0):
    """Serving-layer inputs: reversed int32 state indices into an FP32 pool with
    spare slots, and a strided [1, T, heads] slice of a 40-column beta carrier."""
    torch.manual_seed(seed)
    count = len(lengths)
    tokens = sum(lengths)
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    pool = count + 5

    def operand():
        return torch.randn(
            1, tokens, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )

    d = dict(
        q=operand(),
        k=operand(),
        v=operand(),
        g=operand(),
        beta=torch.randn(1, tokens, 40, device="cuda", dtype=torch.bfloat16)[
            :, :, 8 : 8 + heads
        ],
        A_log=torch.randn(1, 1, heads, 1, device="cuda") * 0.2,
        dt_bias=torch.randn(heads * HEAD_DIM, device="cuda") * 0.1,
        state=torch.randn(pool, heads, HEAD_DIM, HEAD_DIM, device="cuda") * 0.01,
        state_indices=torch.arange(
            pool - 1, pool - count - 1, -1, device="cuda", dtype=torch.int32
        ),
        cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int64),
        checkpoint_cu_starts=torch.tensor(
            _checkpoint_starts(lengths), device="cuda", dtype=torch.int64
        ),
        lengths=tuple(lengths),
        heads=heads,
    )
    d["initial_pool"] = d["state"].clone()
    return d


def _checkpoint_dtype(lower_bound):
    probe = None if lower_bound is None else -1.0
    supported = kda_prefill_supports_fp32_checkpoints(
        torch.device("cuda", torch.cuda.current_device()), lower_bound=probe
    )
    return torch.float32 if supported else torch.bfloat16


def _poison_free_cuda_memory(reserve=2 << 30, small_reserve=16 << 20):
    """Fill every free block of the CUDA caching allocator (this device and
    stream, outside graph pools) with 0xFF bytes, NaN in BF16 and FP32, after
    growing the free pools to ``reserve`` / ``small_reserve`` bytes.  A launch
    that skips a producer then reads poison, not what another arm or an
    earlier call left in memory the allocator hands out again."""
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    sizes, free = [], {"large": 0, "small": 0}
    for segment in torch.cuda.memory_snapshot():
        if (
            segment["device"] != device
            or segment["stream"] != stream
            or tuple(segment.get("segment_pool_id", (0, 0))) != (0, 0)
        ):
            continue
        kind = segment["segment_type"]
        for block in segment["blocks"]:
            if block["state"] != "inactive":
                continue
            size = block["size"]
            free[kind] += size
            # The small pool serves requests up to 1 MiB.
            step = 1 << 20 if kind == "small" else size
            sizes += [min(step, size - offset) for offset in range(0, size, step)]
    sizes.sort(reverse=True)
    grow = max(0, reserve - free["large"])
    sizes += [256 << 20] * -(-grow // (256 << 20))
    sizes += [1 << 20] * -(-max(0, small_reserve - free["small"]) // (1 << 20))
    held = [torch.empty(n, dtype=torch.uint8, device="cuda") for n in sizes]
    for tensor in held:
        tensor.fill_(0xFF)
    torch.cuda.synchronize()


class _Call:
    """One prepared launch with its own output and checkpoint rows.

    By default the serving call: the FP32 pool in place through the state
    indices, checkpoint rows every 64 tokens.  ``rows=False`` drops the
    checkpoint rows; ``indexed=False`` passes one dense FP32 state per
    sequence (the indexed call's slots, in sequence order) in place, and
    ``initial=False`` with it starts from the zero state into that tensor.
    Free CUDA memory is poisoned before the call is prepared and before every
    launch, and its outputs before every launch (``run``).
    """

    def __init__(
        self,
        d,
        lower_bound,
        *,
        cu_seqlens=None,
        plan_cache=None,
        rows=True,
        indexed=True,
        initial=True,
        checkpoint_dtype=None,
    ):
        heads = d["heads"]
        count = _checkpoint_starts(d["lengths"])[-1]
        self.d = d
        self.indexed = indexed
        self.initial = initial
        self.out = torch.full_like(d["q"], float("nan"))
        self.checkpoints = (
            torch.full(
                (count, heads, HEAD_DIM, HEAD_DIM),
                float("nan"),
                device="cuda",
                dtype=checkpoint_dtype or _checkpoint_dtype(lower_bound),
            )
            if rows
            else None
        )
        if indexed:
            assert initial, "the indexed call reads its initial states from the pool"
            initial_state = final_state = d["state"]
            state_indices = d["state_indices"]
        else:
            self.dense = d["initial_pool"][d["state_indices"].long()].clone()
            initial_state = self.dense if initial else None
            final_state = self.dense
            state_indices = None
        _poison_free_cuda_memory()
        self.prepared = prepare_bf16_kda_prefill(
            d["q"],
            d["k"],
            d["v"],
            d["g"],
            d["beta"],
            A_log=d["A_log"].reshape(heads),
            dt_bias=d["dt_bias"].reshape(heads, HEAD_DIM),
            out=self.out,
            initial_state=initial_state,
            final_state=final_state,
            scale=None,
            lower_bound=lower_bound,
            cu_seqlens=d["cu_seqlens"] if cu_seqlens is None else cu_seqlens,
            sequence_lengths=d["lengths"],
            state_indices=state_indices,
            state_checkpoints=self.checkpoints,
            checkpoint_cu_starts=d["checkpoint_cu_starts"] if rows else None,
            checkpoint_every_n_tokens=CHECKPOINT_EVERY if rows else 0,
            beta_is_logit=True,
            plan_cache=plan_cache,
        )

    def poison(self):
        if self.indexed:
            self.d["state"].copy_(self.d["initial_pool"])
        elif self.initial:
            self.dense.copy_(self.d["initial_pool"][self.d["state_indices"].long()])
        else:
            self.dense.fill_(float("nan"))
        self.out.fill_(float("nan"))
        if self.checkpoints is not None:
            self.checkpoints.fill_(float("nan"))

    def run(self):
        _poison_free_cuda_memory()
        self.poison()
        self.prepared.launch()
        torch.cuda.synchronize()
        return self.snapshot()

    def snapshot(self):
        d = self.d
        return (
            self.out.clone(),
            d["state"][d["state_indices"].long()].clone()
            if self.indexed
            else self.dense.clone(),
            None if self.checkpoints is None else self.checkpoints.clone(),
        )


def _reference(d, lower_bound):
    """FP32 token-by-token recurrence: l2-normalized q/k, per-key decay
    exp(-exp(A_log) softplus(g + dt_bias)) or exp(lb sigmoid(exp(A_log)(g + dt_bias))),
    S = S * decay; S += sigmoid(beta) (v - S k) k^T; o = S q / sqrt(128).
    Checkpoint row r of a sequence is the state entering its token 64 r."""
    heads = d["heads"]
    lengths = list(d["lengths"])
    with torch.no_grad():
        q = F.normalize(d["q"][0].float(), dim=-1)
        k = F.normalize(d["k"][0].float(), dim=-1)
        v = d["v"][0].float()
        rate = d["A_log"].reshape(heads).float().exp()[:, None]
        pre = d["g"][0].float() + d["dt_bias"].reshape(heads, HEAD_DIM).float()
        if lower_bound is None:
            decay = torch.exp(-rate * F.softplus(pre))
        else:
            decay = torch.exp(lower_bound * torch.sigmoid(rate * pre))
        del pre
        beta = torch.sigmoid(d["beta"][0].float())
        state = d["initial_pool"][d["state_indices"].long()].clone()
        count = len(lengths)
        lens = torch.tensor(lengths, device="cuda")
        first = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()[:-1]], device="cuda"
        )
        starts = _checkpoint_starts(lengths)
        row_base = torch.tensor(starts[:-1], device="cuda")
        checkpoints = torch.empty(
            (starts[-1], heads, HEAD_DIM, HEAD_DIM), device="cuda", dtype=torch.float32
        )
        out = torch.empty(sum(lengths), heads, HEAD_DIM, device="cuda")
        bounds = sorted(set(lengths))
        active = torch.arange(count, device="cuda")
        full = True
        for t in range(max(lengths)):
            if bounds and t == bounds[0]:
                while bounds and t >= bounds[0]:
                    bounds.pop(0)
                active = (lens > t).nonzero().squeeze(1)
                full = active.numel() == count
            token = first[active] + t
            s = state if full else state[active]
            if t % CHECKPOINT_EVERY == 0:
                checkpoints[row_base[active] + t // CHECKPOINT_EVERY] = s
            kt = k[token][:, :, None, :]
            s = s * decay[token][:, :, None, :]
            r = beta[token][:, :, None] * (v[token] - (s * kt).sum(-1))
            s = s + r[..., None] * kt
            out[token] = HEAD_DIM**-0.5 * (s * q[token][:, :, None, :]).sum(-1)
            if full:
                state = s
            else:
                state[active] = s
    return out[None], state, checkpoints


def _assert_close(got, want, label):
    for name, actual, expected in zip(
        ("out", "final state", "checkpoint rows"), got, want, strict=True
    ):
        a = actual.float()
        b = expected.float()
        assert bool(torch.isfinite(a).all()), f"{label}: {name} is not finite"
        bad = (a - b).abs() > ATOL + RTOL * b.abs()
        assert not bool(bad.any()), (
            f"{label}: {name}: {int(bad.sum())} of {a.numel()} elements outside "
            f"atol=rtol={ATOL}, max abs diff {float((a - b).abs().max()):.3e}"
        )


def _assert_bitwise(got, want, label):
    for name, a, b in zip(
        ("out", "final state", "checkpoint rows"), got, want, strict=True
    ):
        if a is None or b is None:
            assert a is None and b is None, f"{label}: {name} on one side only"
            continue
        bits_a = a.contiguous().view(torch.uint8)
        bits_b = b.contiguous().view(torch.uint8)
        if not torch.equal(bits_a, bits_b):
            same = (bits_a == bits_b).view(a.shape + (a.element_size(),)).all(-1)
            first = tuple(int(i) for i in (~same).nonzero()[0])
            diff = (a.float() - b.float()).abs()
            raise AssertionError(
                f"{label}: {name} differs: {int((~same).sum())} of {a.numel()} "
                f"elements, first at {first}, max abs diff {float(diff.max()):.3e}"
            )


def _modules(prepared):
    """Names of the loader modules a prepared launch holds."""
    found = set()
    seen = set()

    def visit(obj, depth):
        if depth > 7 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, registry.PreparedKernel):
            found.add(obj.name)
        elif isinstance(obj, dict):
            for value in obj.values():
                visit(value, depth + 1)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for value in obj:
                visit(value, depth + 1)
        elif type(obj).__module__.startswith("flashinfer") and hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                visit(value, depth + 1)

    visit(prepared, 0)
    return found


@pytest.fixture
def unrouted(monkeypatch):
    """Clear the routes: every factory row builds its shipped module."""

    def clear():
        for table in (
            "_RENDERED_ROUTES",
            "_RENDERED_HEAD_ROUTES",
            "_RENDERED_MERGE_ROUTES",
            "_RENDERED_WORK_BIN_ROUTES",
            "_RENDERED_OWNER_HELPER_ROUTES",
            "_RENDERED_OWNER_HELPER_POOL_ROUTES",
            "_RENDERED_FOLD_ROUTES",
            "_RENDERED_MAP_FOLD_ROUTES",
            "_RENDERED_PAIR_SHARE_ROUTES",
            "_RENDERED_WORK_BIN_TURN_ROUTES",
            "_RENDERED_M64_OWNER_HELPER_ROUTES",
            "_RENDERED_WORK_BIN_MIX_ROUTES",
        ):
            monkeypatch.setattr(registry, table, {arch: {} for arch in registry.ARCHES})

    return clear


@requires_blackwell
@pytest.mark.parametrize("lengths,heads,lower_bound", ROWS)
def test_serving_rows_match_the_fp32_reference(
    lengths, heads, lower_bound, unrouted, monkeypatch
):
    arch = registry.device_arch()
    routes = registry._RENDERED_ROUTES[arch]
    d = _inputs(lengths, heads)
    untouched = torch.ones(d["state"].shape[0], dtype=torch.bool, device="cuda")
    untouched[d["state_indices"].long()] = False
    want = _reference(d, lower_bound)

    routed = _Call(d, lower_bound)
    routed_result = routed.run()
    _assert_close(routed_result, want, "routed")
    assert torch.equal(d["state"][untouched], d["initial_pool"][untouched])
    loaded = _modules(routed.prepared)
    assert not loaded & set(routes), "a routed shipped module was loaded"
    _assert_bitwise(routed.run(), routed_result, "routed repeat")

    # Work bins only change which CTA runs a task: without them the launch takes
    # the module's one-task-per-CTA program and matches bit for bit.  The
    # turnover and bin-mix programs are built with work bins too (and a launch
    # takes one only where the module has a work-binned program).
    work_bin_programs = (
        set(registry._RENDERED_WORK_BIN_ROUTES[arch].values())
        | set(registry._RENDERED_WORK_BIN_TURN_ROUTES[arch].values())
        | set(registry._RENDERED_WORK_BIN_MIX_ROUTES[arch].values())
    )
    if loaded & work_bin_programs:
        with monkeypatch.context() as patch:
            patch.setattr(
                registry,
                "_RENDERED_WORK_BIN_ROUTES",
                {name: {} for name in registry.ARCHES},
            )
            one_task_per_cta = _Call(d, lower_bound)
            assert not _modules(one_task_per_cta.prepared) & work_bin_programs
            _assert_bitwise(one_task_per_cta.run(), routed_result, "one task per CTA")
        # Nor does the selection without work bins (where it takes the H12
        # early-pack body, the binned launch takes the binned N32 body): every
        # rule that asks for bins is off.
        with monkeypatch.context() as patch:
            for rule in (
                "_use_direct_work_bins",
                "_use_turnover_work_bins",
                "_use_bounded_work_bins",
            ):
                patch.setattr(runtime, rule, lambda *args, **kwargs: False)
            unbinned = _Call(d, lower_bound)
            assert not _modules(unbinned.prepared) & work_bin_programs
            _assert_bitwise(unbinned.run(), routed_result, "selection without bins")

    # The prepared path takes int64 sequence offsets (callers cast int32
    # offsets first); the routes leave that contract as it is.
    narrow = d["cu_seqlens"].to(torch.int32)
    with pytest.raises(TypeError, match="cu_seqlens must have dtype torch.int64"):
        _Call(d, lower_bound, cu_seqlens=narrow)
    widened = _Call(d, lower_bound, cu_seqlens=narrow.to(torch.int64))
    _assert_bitwise(widened.run(), routed_result, "int32 offsets cast to int64")

    unrouted()
    with pytest.raises(TypeError, match="cu_seqlens must have dtype torch.int64"):
        _Call(d, lower_bound, cu_seqlens=narrow)
    shipped = _Call(d, lower_bound)
    _assert_close(shipped.run(), want, "shipped")
    assert not _modules(shipped.prepared) & set(GENERATED)


MERGE_ROWS = [
    pytest.param([8192], 12, id="8192-h12"),
    pytest.param([3000, 13384], 16, id="3000+13384-h16"),
    pytest.param([8128, 64, 8128, 64], 12, id="8128+64x4-h12"),
]


@requires_blackwell
@pytest.mark.parametrize("lengths,heads", MERGE_ROWS)
def test_merging_correction_is_bitwise_the_epilogue_composite(
    lengths, heads, monkeypatch
):
    """A chain composite whose correction windows merge themselves equals,
    bit for bit, the composite that runs the epilogue kernel after them (the
    loader without merging builds): output, pool slots and checkpoint rows,
    every element written over NaN, the other pool slots untouched."""
    arch = registry.device_arch()
    merging = registry._RENDERED_MERGE_ROUTES[arch]
    routed = {registry._RENDERED_ROUTES[arch][module] for module in merging}
    d = _inputs(lengths, heads)
    untouched = torch.ones(d["state"].shape[0], dtype=torch.bool, device="cuda")
    untouched[d["state_indices"].long()] = False

    merged = _Call(d, -5.0)
    merged_result = merged.run()
    loaded = _modules(merged.prepared)
    assert not loaded & routed, "a chain correction kept its epilogue"
    chain = bool(loaded & set(merging.values()))
    if lengths in ([8192], [3000, 13384]):
        assert chain, "the chain route's correction did not merge"
    assert torch.equal(d["state"][untouched], d["initial_pool"][untouched])
    _assert_bitwise(merged.run(), merged_result, "merged repeat")

    monkeypatch.setattr(
        registry,
        "_RENDERED_MERGE_ROUTES",
        {name: {} for name in registry.ARCHES},
    )
    epilogue = _Call(d, -5.0)
    epilogue_result = epilogue.run()
    assert not _modules(epilogue.prepared) & set(merging.values())
    if chain:
        assert _modules(epilogue.prepared) & routed
    _assert_bitwise(merged_result, epilogue_result, "merged vs epilogue composite")
    assert torch.equal(d["state"][untouched], d["initial_pool"][untouched])


FOLD_ROWS = [
    pytest.param([8192], 12, id="8192-h12"),
    pytest.param([3000, 13384], 16, id="3000+13384-h16"),
    pytest.param([8128, 64, 8128, 64], 12, id="8128+64x4-h12"),
    pytest.param([100, 33, 200, 17, 65], 16, id="short-h16"),
]


@requires_blackwell
@pytest.mark.parametrize("lengths,heads", FOLD_ROWS)
def test_pair_map_fold_is_bitwise_the_pair_map_producer(lengths, heads, monkeypatch):
    """An apply composite whose main body writes the block pair maps equals,
    bit for bit, the composite that launches the pair-map producer (the loader
    without fold builds): output, pool slots and checkpoint rows, every
    element written over NaN, the other pool slots untouched."""
    arch = registry.device_arch()
    folding = registry._RENDERED_FOLD_ROUTES[arch]
    d = _inputs(lengths, heads)
    untouched = torch.ones(d["state"].shape[0], dtype=torch.bool, device="cuda")
    untouched[d["state_indices"].long()] = False

    folded = _Call(d, None)
    folded_result = folded.run()
    fold = bool(_modules(folded.prepared) & set(folding.values()))
    assert fold == ("_pairfold" in str(folded.prepared.schedule))
    if (
        _affine_pair_map_fold_enabled()
        and _checkpoint_dtype(None) == torch.float32
        and "_apply_leanmap" in str(folded.prepared.schedule)
    ):
        assert fold, "the lean-map apply composite launched the pair-map producer"
    assert torch.equal(d["state"][untouched], d["initial_pool"][untouched])
    _assert_bitwise(folded.run(), folded_result, "folded repeat")

    monkeypatch.setattr(
        registry, "_RENDERED_FOLD_ROUTES", {name: {} for name in registry.ARCHES}
    )
    producer = _Call(d, None)
    producer_result = producer.run()
    assert not _modules(producer.prepared) & set(folding.values())
    assert "_pairfold" not in str(producer.prepared.schedule)
    _assert_bitwise(folded_result, producer_result, "folded vs pair-map producer")
    assert torch.equal(d["state"][untouched], d["initial_pool"][untouched])


def _sm_count():
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


MAP_FOLD_ROWS = [
    pytest.param([8192], 12, None, id="8192-h12"),
    pytest.param([3000, 13384], 16, None, id="3000+13384-h16"),
    # The cost model takes the chain composite for this row on a 152-SM
    # sm_103a part only (44-chunk windows, 426 us against the sequential
    # body's 436 us); a 148-SM sm_100a part keeps the sequential M64 program
    # (471 against 461 us), which has no map pass.
    pytest.param([8128, 64, 8128, 64], 12, ("sm_103a", 152), id="8128+64x4-h12"),
    pytest.param([16384], 16, None, id="16384-h16"),
]


@requires_blackwell
@pytest.mark.parametrize("lengths,heads,part", MAP_FOLD_ROWS)
def test_chain_map_fold_is_bitwise_the_map_pass(lengths, heads, part, monkeypatch):
    """A chain composite whose main body writes the window maps equals, bit for
    bit, the composite that launches the map pass (the loader without map-fold
    builds): output, pool slots and checkpoint rows, every element written over
    NaN, the other pool slots untouched.  Every row takes the chain composite
    (on the part it names), whose main module has a map-fold build."""
    arch = registry.device_arch()
    if part is not None and part != (arch, _sm_count()):
        pytest.skip(f"only a {part[1]}-SM {part[0]} part takes the chain composite")
    folding = registry._RENDERED_MAP_FOLD_ROUTES[arch]
    d = _inputs(lengths, heads)
    untouched = torch.ones(d["state"].shape[0], dtype=torch.bool, device="cuda")
    untouched[d["state_indices"].long()] = False

    folded = _Call(d, -5.0)
    schedule = str(folded.prepared.schedule)
    assert schedule.startswith("split_seq_affine_prefix"), schedule
    folded_result = folded.run()
    fold = bool(_modules(folded.prepared) & set(folding.values()))
    assert fold == ("_mapfold" in schedule), schedule
    if _chain_map_fold_enabled():
        assert fold, "the chain composite launched the map pass"
    assert torch.equal(d["state"][untouched], d["initial_pool"][untouched])
    _assert_bitwise(folded.run(), folded_result, "folded repeat")

    monkeypatch.setattr(
        registry, "_RENDERED_MAP_FOLD_ROUTES", {name: {} for name in registry.ARCHES}
    )
    mapped = _Call(d, -5.0)
    mapped_result = mapped.run()
    assert not _modules(mapped.prepared) & set(folding.values())
    assert "_mapfold" not in str(mapped.prepared.schedule)
    _assert_bitwise(folded_result, mapped_result, "folded vs map pass")
    assert torch.equal(d["state"][untouched], d["initial_pool"][untouched])


def _turnover_chunk_limit():
    """The part's turnover chunk limit, as ``_work_bin_turnover_route`` reads it."""
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if registry.device_arch() == "sm_103a" and sm_count == 152:
        return runtime.BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS_GB300
    return runtime.BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS


def _turn_programs():
    return set(registry._RENDERED_WORK_BIN_TURN_ROUTES[registry.device_arch()].values())


def _turn_of():
    """Work-binned program -> its turnover build, for the turn table's modules."""
    arch = registry.device_arch()
    return {
        registry._RENDERED_WORK_BIN_ROUTES[arch][module]: program
        for module, program in registry._RENDERED_WORK_BIN_TURN_ROUTES[arch].items()
    }


def _without_turnover(patch):
    patch.setattr(
        registry,
        "_RENDERED_WORK_BIN_TURN_ROUTES",
        {arch: {} for arch in registry.ARCHES},
    )


def _assert_turnover_schedule(turn, plain, expect):
    """``turn`` (the loader as shipped) against ``plain`` (the same call without
    the turn table).  With ``expect`` ``plain`` takes a work-binned program the
    turn table names and ``turn`` the same programs with that one replaced by
    its turnover build, its schedule ``plain``'s with ``_turnover`` added;
    without it both hold the same programs and the same schedule."""
    schedule = str(turn.prepared.schedule)
    plain_schedule = str(plain.prepared.schedule)
    plain_programs = _modules(plain.prepared)
    if expect:
        turn_of = _turn_of()
        assert plain_programs & set(turn_of), plain_schedule
        assert schedule.count("_turnover") == 1, schedule
        assert schedule.replace("_turnover", "", 1) == plain_schedule
        assert _modules(turn.prepared) == {turn_of.get(p, p) for p in plain_programs}
    else:
        assert schedule == plain_schedule
        assert _modules(turn.prepared) == plain_programs
        assert not plain_programs & _turn_programs()


# Work-binned short launches on a 148-SM sm_100a and a 152-SM sm_103a part
# (``test_turnover_rows_take_the_route_on_both_parts``).
# Turnover launches the bin-mix rule does not admit (those take the bin-mix
# build: BIN_MIX_ROWS): 16 x 32 tokens at 12 heads pairs a two-chunk bin with
# a one-chunk bin, d = 1 gives g = 0, so the launch keeps the turnover build.
TURNOVER_ROWS = [
    pytest.param([32] * 16, 12, id="bs16x32-h12"),
    pytest.param([64] * 64, 16, id="bs64x64-h16"),
    pytest.param([17, 64, 65, 127, 128, 200, 255, 96] * 4, 12, id="mixed32-h12"),
    pytest.param([1, 33, 64, 96] * 12, 16, id="short-mixed-h16"),
]


def _assert_route(route, prepared):
    """The launch a graph / plan-cache row is there for, before it runs."""
    if route is None:
        return
    arch = registry.device_arch()
    schedule = str(prepared.schedule)
    loaded = _modules(prepared)
    owner_helper = set(registry._RENDERED_M64_OWNER_HELPER_ROUTES[arch].values())
    if route == "m64-owner-helper" and not owner_helper:
        route = "pair-share"  # a loader without the build keeps the pair-share one
    if route == "m64-owner-helper":
        assert schedule.startswith("fused_m64_independent_dvsplit"), schedule
        assert schedule.endswith("_owner2helper"), schedule
        assert loaded & owner_helper
    elif route == "pair-share":
        assert schedule.startswith("fused_m64_independent_dvsplit"), schedule
        assert schedule.endswith("_pair_share"), schedule
        assert loaded & set(registry._RENDERED_PAIR_SHARE_ROUTES[arch].values())
    elif route == "pool":
        assert re.search(r"_owner1helper_pool\d+$", schedule), schedule
        assert loaded & set(registry._RENDERED_OWNER_HELPER_POOL_ROUTES[arch].values())
    else:
        assert route == "map-fold", route
        assert schedule.startswith("split_seq_affine_prefix"), schedule
        if _chain_map_fold_enabled():
            assert "_mapfold" in schedule, schedule
            assert loaded & set(registry._RENDERED_MAP_FOLD_ROUTES[arch].values())


@requires_blackwell
@pytest.mark.parametrize("rows", [True, False], ids=["rows", "no-rows"])
@pytest.mark.parametrize("lengths,heads", TURNOVER_ROWS)
def test_turnover_launch_is_bitwise_the_work_binned_launch(
    lengths, heads, rows, monkeypatch
):
    """A work-binned launch of short tasks takes the turnover build: the same
    bins, bits and argument plan as the work-binned program, the next task's
    initial state loaded and the previous task's final state stored in the
    background.  Bit for bit the work-binned launch, with and without
    checkpoint rows, free memory and outputs poisoned before every launch, in
    either order."""
    d = _inputs(lengths, heads)
    turn = _Call(d, None, rows=rows)
    with monkeypatch.context() as patch:
        _without_turnover(patch)
        plain = _Call(d, None, rows=rows)
    _assert_turnover_schedule(turn, plain, True)
    got = turn.run()
    want = plain.run()
    _assert_bitwise(got, want, "turnover vs work-binned")
    _assert_bitwise(turn.run(), want, "turnover after the work-binned launch")
    _assert_bitwise(plain.run(), got, "work-binned repeat")


# Unbounded work-binned short launches whose turnover bin table the bin-mix
# rule admits on a 148-SM sm_100a and a 152-SM sm_103a part: bins of one and
# two tasks, every two-task bin paired with a one-task bin it gives chunks to.
# They take the bin-mix build on the WORK_BIN_MIX_ARCHES parts and keep the
# turnover build on the others.
BIN_MIX_ROWS = [
    pytest.param([64] * 16, 12, id="bs16x64-h12"),
    pytest.param([128] * 16, 12, id="bs16x128-h12"),
    pytest.param([256] * 16, 12, id="bs16x256-h12"),
    pytest.param([256, 224, 192, 160] * 4, 12, id="mixed16-h12"),
]


def _bin_mix_programs():
    arch = registry.device_arch()
    if arch not in runtime.WORK_BIN_MIX_ARCHES:
        pytest.skip(
            f"{arch} launches keep the turnover build "
            "(test_bin_mix_rows_keep_the_turnover_build_elsewhere)"
        )
    programs = set(registry._RENDERED_WORK_BIN_MIX_ROUTES[arch].values())
    assert programs, f"the loader names no bin-mix build on {arch}"
    return programs


def _without_bin_mix(patch):
    patch.setattr(
        registry,
        "_RENDERED_WORK_BIN_MIX_ROUTES",
        {arch: {} for arch in registry.ARCHES},
    )


@requires_blackwell
@pytest.mark.parametrize("rows", [True, False], ids=["rows", "no-rows"])
@pytest.mark.parametrize("lengths,heads", BIN_MIX_ROWS)
def test_bin_mix_launch_is_bitwise_the_turnover_launch(
    lengths, heads, rows, monkeypatch
):
    """A turnover launch whose bin table the bin-mix rule admits takes the
    bin-mix build: a one-task bin prepares the last chunks of its paired
    two-task bin after its own task and hands them over through the ring.  Its
    schedule is the turnover launch's with ``_bin_mix`` added, and that one the
    work-binned launch's with ``_turnover`` added.  Bit for bit the turnover
    launch and the work-binned launch, with and without checkpoint rows,
    launch after launch (the owners leave every ready word zero), in either
    order."""
    mix = _bin_mix_programs()
    d = _inputs(lengths, heads)
    mixed = _Call(d, None, rows=rows)
    schedule = str(mixed.prepared.schedule)
    assert schedule.endswith("_turnover_bin_mix"), schedule
    assert _modules(mixed.prepared) & mix
    with monkeypatch.context() as patch:
        _without_bin_mix(patch)
        turned = _Call(d, None, rows=rows)
    with monkeypatch.context() as patch:
        _without_turnover(patch)
        plain = _Call(d, None, rows=rows)
    assert str(turned.prepared.schedule) + "_bin_mix" == schedule
    _assert_turnover_schedule(turned, plain, True)
    assert not _modules(turned.prepared) & mix
    assert not _modules(plain.prepared) & mix
    got = mixed.run()
    want = turned.run()
    _assert_bitwise(got, want, "bin-mix vs turnover")
    _assert_bitwise(got, plain.run(), "bin-mix vs work-binned")
    _assert_bitwise(mixed.run(), want, "bin-mix after the other launches")
    _assert_bitwise(turned.run(), got, "turnover repeat")


@requires_blackwell
@pytest.mark.parametrize("rows", [True, False], ids=["rows", "no-rows"])
@pytest.mark.parametrize("lengths,heads", BIN_MIX_ROWS)
def test_bin_mix_rows_keep_the_turnover_build_elsewhere(
    lengths, heads, rows, monkeypatch
):
    """On a part outside ``WORK_BIN_MIX_ARCHES`` a turnover launch whose bin
    table the bin-mix rule admits keeps the turnover build: no bin-mix
    program, the work-binned launch's schedule with ``_turnover`` added, and
    bit for bit the work-binned launch."""
    arch = registry.device_arch()
    if arch in runtime.WORK_BIN_MIX_ARCHES:
        pytest.skip(
            f"{arch} takes the bin-mix build "
            "(test_bin_mix_launch_is_bitwise_the_turnover_launch)"
        )
    d = _inputs(lengths, heads)
    turned = _Call(d, None, rows=rows)
    assert str(turned.prepared.schedule).endswith("_turnover"), str(
        turned.prepared.schedule
    )
    assert not _modules(turned.prepared) & set(
        registry._RENDERED_WORK_BIN_MIX_ROUTES[arch].values()
    )
    with monkeypatch.context() as patch:
        _without_turnover(patch)
        plain = _Call(d, None, rows=rows)
    _assert_turnover_schedule(turned, plain, True)
    got = turned.run()
    _assert_bitwise(got, plain.run(), "turnover vs work-binned")
    _assert_bitwise(turned.run(), got, "turnover repeat")


@requires_blackwell
def test_refused_bin_mix_launch_runs_its_turnover_build(monkeypatch):
    """A cooperative grid the device cannot hold (the work-bin table built for
    twice the part's SMs, one bin per CTA): the driver refuses the bin-mix
    launch, the plan switches for good to its turnover build with that build's
    own bindings and matches it bit for bit, and later CUDA work is
    unaffected."""
    mix = _bin_mix_programs()
    lengths, heads = [128] * 32, 12
    sm_count = 2 * _sm_count()
    monkeypatch.setattr(runtime, "_device_sm_count", lambda device: sm_count)
    d = _inputs(lengths, heads)
    refused = _Call(d, None)
    schedule = str(refused.prepared.schedule)
    assert schedule.endswith("_turnover_bin_mix"), schedule
    assert _modules(refused.prepared) & mix
    result = refused.run()
    assert not _modules(refused.prepared) & mix
    probe = torch.zeros(1024, device="cuda")
    probe.add_(1)
    torch.cuda.synchronize()
    assert float(probe.sum()) == 1024.0
    _assert_bitwise(refused.run(), result, "fallback repeat")
    with monkeypatch.context() as patch:
        _without_bin_mix(patch)
        turned = _Call(d, None)
    assert str(turned.prepared.schedule) == schedule.removesuffix("_bin_mix")
    _assert_bitwise(result, turned.run(), "refused bin-mix vs its turnover build")


def _bounded_turnover_expected(lengths, heads):
    """Whether this device's part routes a work-binned bounded launch of these
    lengths to its module's turnover build (``_bounded_turnover_route``)."""
    sm = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    return runtime._bounded_turnover_route(
        bounded_gate=True,
        seq_lens=tuple(lengths),
        num_heads=heads,
        gpu_arch=registry.device_arch(),
        sm_count=sm,
    )


# Work-binned bounded launches whose bins all average more than the part's
# bounded chunks per task: the bounded N32 direct modules' work-binned
# programs have turnover builds too.
BOUNDED_TURNOVER_ROWS = [
    pytest.param([128] * 16, 16, id="bs16x128-h16"),
    pytest.param([256] * 16, 16, id="bs16x256-h16"),
    pytest.param([128] * 16, 12, id="bs16x128-h12"),
    pytest.param([256] * 16, 12, id="bs16x256-h12"),
    pytest.param([128, 160, 192, 100] * 12, 16, id="mixed-h16"),
]


@requires_blackwell
@pytest.mark.parametrize("rows", [True, False], ids=["rows", "no-rows"])
@pytest.mark.parametrize("lengths,heads", BOUNDED_TURNOVER_ROWS)
def test_bounded_turnover_launch_is_bitwise_the_work_binned_launch(
    lengths, heads, rows, monkeypatch
):
    """The bounded rows' turnover build against the same call without the turn
    table, bit for bit, with and without checkpoint rows, in either order."""
    if not _bounded_turnover_expected(lengths, heads):
        pytest.skip("this part keeps the bounded launch on its work-binned build")
    d = _inputs(lengths, heads)
    turn = _Call(d, -5.0, rows=rows)
    with monkeypatch.context() as patch:
        _without_turnover(patch)
        plain = _Call(d, -5.0, rows=rows)
    _assert_turnover_schedule(turn, plain, True)
    got = turn.run()
    want = plain.run()
    _assert_bitwise(got, want, "turnover vs work-binned")
    _assert_bitwise(turn.run(), want, "turnover after the work-binned launch")
    _assert_bitwise(plain.run(), got, "work-binned repeat")


TURNOVER_PARTS = [
    pytest.param("sm_100a", None, 8, id="sm_100a"),
    pytest.param("sm_103a", 152, 16, id="sm_103a-152"),
]


@requires_blackwell
@pytest.mark.parametrize("side", ["at", "past"])
@pytest.mark.parametrize("shape", ["uniform", "mixed"])
@pytest.mark.parametrize("arch,sm_count,chunks", TURNOVER_PARTS)
def test_turnover_route_takes_the_part_chunk_limit(
    arch, sm_count, chunks, shape, side, monkeypatch
):
    """The turnover route's chunk limit on either side: a work-binned H16
    launch whose longest sequence has the part's limit of 32-token chunks takes
    the turnover build, one token more keeps the work-binned build (still work
    binned); bit for bit either way.  Runs on the part the limit is for."""
    sm = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if registry.device_arch() != arch or sm_count not in (None, sm):
        pytest.skip(f"the {arch} limit; this device is {registry.device_arch()}")
    assert _turnover_chunk_limit() == chunks
    longest = runtime.BF16_M128_CHUNK * chunks + (side == "past")
    # Enough work that bins hold several tasks (``_use_direct_work_bins``).
    lengths = (
        [longest] * 16 if shape == "uniform" else [longest] * 4 + [longest // 2] * 28
    )
    d = _inputs(lengths, 16)
    turn = _Call(d, None)
    with monkeypatch.context() as patch:
        _without_turnover(patch)
        plain = _Call(d, None)
    # Past the limit the launch is still work binned: the limit, not the
    # bins, keeps the work-binned build.
    assert _modules(plain.prepared) & set(_turn_of()), str(plain.prepared.schedule)
    _assert_turnover_schedule(turn, plain, side == "at")
    _assert_bitwise(turn.run(), plain.run(), f"{side} the limit")


TURNOVER_ROUTE_PARTS = [
    pytest.param("sm_100a", 148, id="sm_100a-148"),
    pytest.param("sm_103a", 152, id="sm_103a-152"),
]


@pytest.mark.parametrize("arch,sm_count", TURNOVER_ROUTE_PARTS)
def test_turnover_rows_take_the_route_on_both_parts(arch, sm_count):
    """The rows' turnover expectation on each part, from the route rule alone:
    every turnover row asks for work bins (``_use_direct_work_bins``, else
    ``_use_turnover_work_bins``) and has at most the part's chunk limit; of the
    unbounded graph rows only those marked turnover do."""

    def route(lengths, heads):
        return runtime._work_bin_turnover_route(
            route_work_bins=runtime._use_direct_work_bins(
                tuple(lengths),
                num_heads=heads,
                sm_count=sm_count,
                chunk_tokens=runtime.BF16_M128_CHUNK,
            )
            or runtime._use_turnover_work_bins(
                tuple(lengths), num_heads=heads, sm_count=sm_count, gpu_arch=arch
            ),
            direct_n16=False,
            n32_ft_slab=False,
            max_seq_len=max(lengths),
            gpu_arch=arch,
            sm_count=sm_count,
        )

    for row in TURNOVER_ROWS:
        assert route(*row.values), row.id
    for row in GRAPH_ROWS:
        lengths, heads, lower_bound, launch, turnover = row.values
        if lower_bound is None and launch is None:
            assert route(lengths, heads) == turnover, row.id


@pytest.mark.parametrize("arch,sm_count", TURNOVER_ROUTE_PARTS)
def test_bounded_turnover_floor_on_both_parts(arch, sm_count):
    """The bounded predicate on each part: a bounded launch keeps the turnover
    only when every bin of its work-bin table averages more than the part's
    count of 32-token chunks per task; an unbounded launch is not held back,
    and a part without a count keeps every bounded launch on its work-binned
    build.  Uniform tasks of exactly the count stay below and one chunk more
    is above; a few long tasks do not lift the bins of short ones.  At most
    one task in the part's ratio may have two chunks or fewer."""
    per_task = (
        runtime.BF16_BOUNDED_TURNOVER_CHUNKS_PER_TASK_GB300
        if arch == "sm_103a" and sm_count == 152
        else runtime.BF16_BOUNDED_TURNOVER_CHUNKS_PER_TASK
    )

    def route(lengths, bounded=True):
        return runtime._bounded_turnover_route(
            bounded_gate=bounded,
            seq_lens=tuple(lengths),
            num_heads=16,
            gpu_arch=arch,
            sm_count=sm_count,
        )

    assert route([64] * 16, bounded=False) and route([96] * 48, bounded=False)
    if per_task is None:
        assert not any(route(lengths) for lengths in ([128] * 16, [256] * 64))
        return
    edge = runtime.BF16_M128_CHUNK * per_task
    assert not route([edge] * 48) and route([edge + runtime.BF16_M128_CHUNK] * 48)
    assert not route([64] * 16) and not route([64] * 64)
    assert not route([64] * 28 + [256] * 4) and not route([edge] * 28 + [256] * 4)
    assert route([256] * 16) and route([128] * 16) == (per_task < 4)
    ratio = (
        runtime.BF16_BOUNDED_TURNOVER_SHORT_TASK_RATIO_GB300
        if arch == "sm_103a" and sm_count == 152
        else runtime.BF16_BOUNDED_TURNOVER_SHORT_TASK_RATIO
    )
    quarter = 16 // ratio
    assert route([256] * (16 - quarter) + [64] * quarter)
    assert not route([256] * (15 - quarter) + [64] * (quarter + 1))
    assert route([256] * (15 - quarter) + [64] * (quarter + 1), bounded=False)


def _spy_work_bins(monkeypatch):
    """Record every ``_make_direct_work_bins`` call: caller, key, table."""
    calls = []
    original = runtime._make_direct_work_bins

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        caller = sys._getframe(1).f_code.co_name
        calls.append((caller, (args, tuple(kwargs.items())), result))
        return result

    monkeypatch.setattr(runtime, "_make_direct_work_bins", spy)
    return calls


@pytest.mark.parametrize("arch,sm_count", TURNOVER_ROUTE_PARTS)
def test_bounded_turnover_reads_the_work_bin_rule_table(arch, sm_count, monkeypatch):
    """The bounded predicate reads the work-bin rule's bin table on each part: it
    calls ``_make_direct_work_bins`` with the work-bin rule's key (the host
    lengths, the head count, the part's SM count, 32-token chunks) and gets the
    same cached table object, with at most one cache miss per launch shape.
    The lengths the prepared path derives from its offsets (packed host
    ``sequence_lengths`` or a fixed [B, T] shape) route as the direct call with
    the caller's list does."""
    cached = runtime._make_direct_work_bins
    calls = _spy_work_bins(monkeypatch)
    per_task = (
        runtime.BF16_BOUNDED_TURNOVER_CHUNKS_PER_TASK_GB300
        if arch == "sm_103a" and sm_count == 152
        else runtime.BF16_BOUNDED_TURNOVER_CHUNKS_PER_TASK
    )
    chunk = runtime.BF16_M128_CHUNK
    reads = 0
    for lengths, heads in (
        ([128] * 16, 16),
        ([256] * 16, 12),
        ([64] * 64, 16),
        ([96] * 48, 16),
        ([96] * 28 + [256] * 4, 16),
        ([128, 160, 192, 100] * 12, 16),
        ([17, 64, 65, 127, 128, 200, 255, 96] * 6, 12),
    ):
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        seq_lens = tuple(offsets[i + 1] - offsets[i] for i in range(len(lengths)))
        key = (
            (seq_lens,),
            (("num_heads", heads), ("worker_count", sm_count), ("chunk_tokens", chunk)),
        )
        calls.clear()
        info = cached.cache_info()
        binned = runtime._use_direct_work_bins(
            seq_lens, num_heads=heads, sm_count=sm_count, chunk_tokens=chunk
        )
        routed = runtime._bounded_turnover_route(
            bounded_gate=True,
            seq_lens=seq_lens,
            num_heads=heads,
            gpu_arch=arch,
            sm_count=sm_count,
        )
        assert routed == runtime._bounded_turnover_route(
            bounded_gate=True,
            seq_lens=tuple(lengths),
            num_heads=heads,
            gpu_arch=arch,
            sm_count=sm_count,
        ), lengths
        if len(set(lengths)) == 1:
            batch, tokens = len(lengths), lengths[0]
            fixed = [index * tokens for index in range(batch + 1)]
            assert routed == runtime._bounded_turnover_route(
                bounded_gate=True,
                seq_lens=tuple(fixed[i + 1] - fixed[i] for i in range(batch)),
                num_heads=heads,
                gpu_arch=arch,
                sm_count=sm_count,
            ), lengths
        assert binned, lengths  # every case asks for work bins on both parts
        callers = [caller for caller, _, _ in calls]
        assert callers[0] == "_use_direct_work_bins", callers
        assert set(callers[1:]) <= {"_bounded_turnover_route"}, callers
        assert per_task is not None or len(callers) == 1, callers
        reads += len(callers) > 1
        assert all(k == key for _, k, _ in calls), (calls[0][1], key)
        assert all(type(n) is int for n in calls[0][1][0][0]), calls[0][1]
        assert all(table is calls[0][2] for _, _, table in calls), lengths
        after = cached.cache_info()
        assert after.misses - info.misses <= 1, (info, after)
    assert per_task is None or reads, "the predicate never read a table"


# Short N32 launches the strict bin test keeps one task per CTA that take work
# bins for their module's turnover build (``_use_turnover_work_bins``) on both
# parts: the mixed-length H16 row and a bimodal H16 row.
TURNOVER_BIN_ROWS = [
    pytest.param([17, 64, 65, 127, 128, 200, 255, 96] * 2, 16, id="mixed-h16"),
    pytest.param([256, 17] * 8, 16, id="8x(256+17)-h16"),
]


def _work_bin_tables(prepared):
    """Host bin tables of the launch parts of a prepared launch."""
    tables = []
    seen = set()

    def visit(obj, depth):
        if depth > 7 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            for value in obj.values():
                visit(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                visit(value, depth + 1)
        elif type(obj).__module__.startswith("flashinfer") and hasattr(obj, "__dict__"):
            host = vars(obj).get("_work_bin_host")
            if host is not None:
                tables.append(tuple(host.tolist()))
            for value in vars(obj).values():
                visit(value, depth + 1)

    visit(prepared, 0)
    return tables


@requires_blackwell
@pytest.mark.parametrize("rows", [True, False], ids=["rows", "no-rows"])
@pytest.mark.parametrize("lengths,heads", TURNOVER_BIN_ROWS)
def test_turnover_bins_launch_is_bitwise_the_one_task_per_cta_launch(
    lengths, heads, rows, monkeypatch
):
    """A short launch the strict test keeps one task per CTA takes work bins
    charged the task cost and its module's turnover build; bit for bit the
    one-task-per-CTA launch it replaces, with and without checkpoint rows, free
    memory and outputs poisoned before every launch, in either order."""
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    assert not runtime._use_direct_work_bins(
        tuple(lengths),
        num_heads=heads,
        sm_count=sm_count,
        chunk_tokens=runtime.BF16_M128_CHUNK,
    )
    d = _inputs(lengths, heads)
    turn = _Call(d, None, rows=rows)
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "_use_turnover_work_bins", lambda *args, **kwargs: False)
        direct = _Call(d, None, rows=rows)
    arch = registry.device_arch()
    work_bin_programs = set(registry._RENDERED_WORK_BIN_ROUTES[arch].values())
    schedule = str(turn.prepared.schedule)
    assert schedule.count("_turnover") == 1, schedule
    assert _modules(turn.prepared) & _turn_programs(), schedule
    assert not _modules(direct.prepared) & (_turn_programs() | work_bin_programs)
    assert "_turnover" not in str(direct.prepared.schedule)
    assert _work_bin_tables(turn.prepared) == [
        runtime._make_direct_work_bins(
            tuple(lengths),
            num_heads=heads,
            worker_count=sm_count,
            chunk_tokens=runtime.BF16_M128_CHUNK,
            task_cost=runtime.BF16_WORK_BIN_TURNOVER_TASK_COST,
        )[0]
    ]
    assert _work_bin_tables(direct.prepared) == []
    got = turn.run()
    want = direct.run()
    _assert_bitwise(got, want, "turnover bins vs one task per CTA")
    _assert_bitwise(turn.run(), want, "turnover bins after one task per CTA")
    _assert_bitwise(direct.run(), got, "one task per CTA repeat")


@requires_blackwell
@pytest.mark.parametrize("lengths,heads", TURNOVER_BIN_ROWS)
def test_turnover_bins_keep_one_task_per_cta_without_a_turnover_build(
    lengths, heads, monkeypatch
):
    """A bounded launch never asks for turnover bins (its module's turnover
    build takes strict work-binned launches only; the bounded rule may give it
    plain bins instead), and a loader without the turn table has no turnover
    build: the launch keeps the programs, bin tables and schedule it takes
    without the turnover-bin rule, bit for bit."""
    for lower_bound, drop_turn in ((-5.0, False), (None, True)):
        d = _inputs(lengths, heads)
        with monkeypatch.context() as patch:
            if drop_turn:
                _without_turnover(patch)
            ruled = _Call(d, lower_bound)
            patch.setattr(
                runtime, "_use_turnover_work_bins", lambda *args, **kwargs: False
            )
            plain = _Call(d, lower_bound)
        assert str(ruled.prepared.schedule) == str(plain.prepared.schedule)
        assert _modules(ruled.prepared) == _modules(plain.prepared)
        assert _work_bin_tables(ruled.prepared) == _work_bin_tables(plain.prepared)
        assert not _modules(ruled.prepared) & _turn_programs()
        _assert_bitwise(ruled.run(), plain.run(), f"lb={lower_bound} turn={drop_turn}")


# Bounded launches the strict test keeps one task per CTA where, on both parts,
# the turnover-bin rule holds (``_use_turnover_work_bins``), the bounded
# predicate holds on the strict table (``_bounded_turnover_route``) and the
# module has a turnover build: the gate keeps them off the turnover bins, and
# the bounded rule (``_use_bounded_work_bins``) gives them plain bins.
BOUNDED_ONE_TASK_ROWS = [
    pytest.param([256] * 2 + [128] * 12, 16, id="2x256+12x128-h16"),
    pytest.param([256] * 3 + [128] * 14, 12, id="3x256+14x128-h12"),
    pytest.param([256] * 6 + [128] * 4, 16, id="6x256+4x128-h16"),
]


def _bounded_bin_table(lengths, heads, sm_count):
    """The bin table of a launch the bounded rule bins (its task cost)."""
    return runtime._make_direct_work_bins(
        tuple(lengths),
        num_heads=heads,
        worker_count=sm_count,
        chunk_tokens=runtime.BF16_M128_CHUNK,
        task_cost=runtime.BF16_BOUNDED_WORK_BIN_TASK_COST,
    )[0]


@requires_blackwell
@pytest.mark.parametrize("lengths,heads", BOUNDED_ONE_TASK_ROWS)
def test_bounded_launches_never_take_turnover_bins(lengths, heads, monkeypatch):
    """A bounded launch the strict test keeps one task per CTA never takes the
    turnover bins or the turnover build, although on this part the
    turnover-bin rule admits it, the bounded predicate holds on its strict bin
    table and its module has a turnover build: it takes the plain work-binned
    program with the bounded rule's table, the same programs, table and
    schedule as with the turnover-bin rule off, and bit for bit the
    one-task-per-CTA launch it takes with the bounded rule off."""
    sm = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    arch = registry.device_arch()
    seq_lens = tuple(lengths)
    assert not runtime._use_direct_work_bins(
        seq_lens, num_heads=heads, sm_count=sm, chunk_tokens=runtime.BF16_M128_CHUNK
    )
    assert runtime._use_turnover_work_bins(
        seq_lens, num_heads=heads, sm_count=sm, gpu_arch=arch
    )
    assert runtime._bounded_turnover_route(
        bounded_gate=True,
        seq_lens=seq_lens,
        num_heads=heads,
        gpu_arch=arch,
        sm_count=sm,
    )
    assert runtime._use_bounded_work_bins(seq_lens, num_heads=heads, sm_count=sm)
    builds = []
    original = runtime.FlashKDABlackwellBF16FusedLaunch._build_body

    def spy(self, factory, *args, **kwargs):
        if getattr(factory, "func", None) is runtime._factory:
            builds.append((factory, args, kwargs))
        return original(self, factory, *args, **kwargs)

    d = _inputs(lengths, heads)
    with monkeypatch.context() as patch:
        patch.setattr(runtime.FlashKDABlackwellBF16FusedLaunch, "_build_body", spy)
        ruled = _Call(d, -5.0)
    turnover_builds = [
        runtime._work_bin_turnover_program(
            *factory.args,
            *args,
            **{
                **factory.keywords,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("work_bin_turnover", "turnover_bins_only")
                },
                "route_work_bins": True,
            },
        )
        for factory, args, kwargs in builds
    ]
    assert any(program is not None for program in turnover_builds), builds
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "_use_turnover_work_bins", lambda *args, **kwargs: False)
        plain = _Call(d, -5.0)
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "_use_bounded_work_bins", lambda *args, **kwargs: False)
        one_task = _Call(d, -5.0)
    work_bin_programs = set(registry._RENDERED_WORK_BIN_ROUTES[arch].values())
    assert str(ruled.prepared.schedule) == str(plain.prepared.schedule)
    assert _modules(ruled.prepared) == _modules(plain.prepared)
    assert _modules(ruled.prepared) & work_bin_programs
    assert not _modules(ruled.prepared) & _turn_programs()
    assert "_turnover" not in str(ruled.prepared.schedule)
    assert (
        _work_bin_tables(ruled.prepared)
        == _work_bin_tables(plain.prepared)
        == [_bounded_bin_table(lengths, heads, sm)]
    )
    assert not _modules(one_task.prepared) & (work_bin_programs | _turn_programs())
    assert _work_bin_tables(one_task.prepared) == []
    got = ruled.run()
    want = one_task.run()
    _assert_bitwise(got, want, "bounded plain bins vs one task per CTA")
    _assert_bitwise(plain.run(), got, "turnover-bin rule off")
    _assert_bitwise(one_task.run(), got, "one task per CTA repeat")


# Short bounded N32 launches the strict bin test keeps one task per CTA that
# take plain work bins (``_use_bounded_work_bins``) on both parts: the mixed
# H16 row and a long-and-short H16 row.
BOUNDED_BIN_ROWS = [
    pytest.param([17, 64, 65, 127, 128, 200, 255, 96] * 2, 16, id="mixed-h16"),
    pytest.param([256] * 8 + [128] * 2, 16, id="8x256+2x128-h16"),
]


@requires_blackwell
@pytest.mark.parametrize("rows", [True, False], ids=["rows", "no-rows"])
@pytest.mark.parametrize("lengths,heads", BOUNDED_BIN_ROWS)
def test_bounded_bins_launch_is_bitwise_the_one_task_per_cta_launch(
    lengths, heads, rows, monkeypatch
):
    """A short bounded launch the strict test keeps one task per CTA takes its
    module's plain work-binned program with bins charged the bounded task cost,
    never the turnover build; bit for bit the one-task-per-CTA launch it
    replaces, with and without checkpoint rows, free memory and outputs
    poisoned before every launch, in either order."""
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    seq_lens = tuple(lengths)
    assert not runtime._use_direct_work_bins(
        seq_lens,
        num_heads=heads,
        sm_count=sm_count,
        chunk_tokens=runtime.BF16_M128_CHUNK,
    )
    assert runtime._use_bounded_work_bins(seq_lens, num_heads=heads, sm_count=sm_count)
    d = _inputs(lengths, heads)
    binned = _Call(d, -5.0, rows=rows)
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "_use_bounded_work_bins", lambda *args, **kwargs: False)
        direct = _Call(d, -5.0, rows=rows)
    arch = registry.device_arch()
    work_bin_programs = set(registry._RENDERED_WORK_BIN_ROUTES[arch].values())
    assert _modules(binned.prepared) & work_bin_programs
    assert not _modules(binned.prepared) & _turn_programs()
    assert "_turnover" not in str(binned.prepared.schedule)
    assert not _modules(direct.prepared) & (_turn_programs() | work_bin_programs)
    assert _work_bin_tables(binned.prepared) == [
        _bounded_bin_table(lengths, heads, sm_count)
    ]
    assert _work_bin_tables(direct.prepared) == []
    got = binned.run()
    want = direct.run()
    _assert_bitwise(got, want, "bounded plain bins vs one task per CTA")
    _assert_bitwise(binned.run(), want, "bounded plain bins after one task per CTA")
    _assert_bitwise(direct.run(), got, "one task per CTA repeat")


@requires_blackwell
@pytest.mark.parametrize("lower_bound", [-5.0, None], ids=["lb5", "unbounded"])
def test_bounded_and_long_launches_keep_their_build(lower_bound, monkeypatch):
    """Launches the turnover route leaves alone: bounded launches with a work
    bin of at most the part's chunks per task or more short tasks than the
    part's ratio allows, composite rows and a short launch without work bins
    keep the same programs and schedule with and without the turn table."""
    for lengths, heads in (
        ([64] * 16, 16),
        ([64] * 64, 16),
        ([64] * 28 + [256] * 4, 16),
        ([96] * 28 + [256] * 4, 16),
        ([96] * 48, 16),
        ([256] * 11 + [64] * 5, 16),
        ([8192], 12),
        ([17, 64, 65, 127, 128, 200, 255, 96] * 2, 12),
    ):
        if lower_bound is None and lengths in (
            [64] * 16,
            [64] * 64,
            [64] * 28 + [256] * 4,
            [96] * 28 + [256] * 4,
            [96] * 48,
            [256] * 11 + [64] * 5,
        ):
            continue
        d = _inputs(lengths, heads)
        turn = _Call(d, lower_bound)
        with monkeypatch.context() as patch:
            _without_turnover(patch)
            plain = _Call(d, lower_bound)
        _assert_turnover_schedule(turn, plain, False)
        _assert_bitwise(turn.run(), plain.run(), f"{lengths[:2]} h{heads}")


# The fourth field names the launch a row is there for: the last four rows hold
# the launches that differ most from the others in a graph (the cooperative
# launch of the M64 owner/helper program, the 2-CTA cluster launch of the
# pair-share program, the cooperative launch of the pooled owner/helper
# program, and the chain composite whose main body writes the window maps, no
# map pass).  The last field: whether the launch takes a
# turnover build (work-binned unbounded N32 launches of short tasks on both
# parts; None: the part's bounded predicate decides; the mixed H12 row is not
# work binned, its longest task alone sets the makespan).
GRAPH_ROWS = [
    pytest.param(
        [17, 64, 65, 127, 128, 200, 255, 96] * 2, 12, None, None, False, id="mixed-h12"
    ),
    pytest.param([64] * 16, 16, -5.0, None, False, id="bs16x64-h16-lb5"),
    pytest.param([128] * 16, 16, -5.0, None, None, id="bs16x128-h16-lb5"),
    pytest.param([8128, 64, 8128, 64], 16, None, None, False, id="long-short-h16"),
    pytest.param([64] * 16, 12, None, None, True, id="bs16x64-h12-turnover"),
    pytest.param(
        [8192] * 2, 16, -5.0, "m64-owner-helper", False, id="2x8192-h16-lb5-m64oh"
    ),
    pytest.param(
        [8128, 64] * 2,
        16,
        -5.0,
        "pair-share",
        False,
        id="2x(8128+64)-h16-lb5-pair-share",
    ),
    pytest.param([4096] * 4, 16, None, "pool", False, id="4x4096-h16-pool"),
    pytest.param([8192], 12, -5.0, "map-fold", False, id="8192-h12-lb5-map-fold"),
]


@requires_blackwell
@pytest.mark.parametrize("lengths,heads,lower_bound,route,turnover", GRAPH_ROWS)
def test_routed_launch_replays_from_a_cuda_graph(
    lengths, heads, lower_bound, route, turnover
):
    if turnover is None:
        turnover = _bounded_turnover_expected(lengths, heads)
    d = _inputs(lengths, heads)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        call = _Call(d, lower_bound)
        _assert_route(route, call.prepared)
        schedule = str(call.prepared.schedule)
        assert ("_turnover" in schedule) == turnover, schedule
        assert bool(_modules(call.prepared) & _turn_programs()) == turnover
        eager = call.run()
    torch.cuda.current_stream().wait_stream(side)
    call.poison()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        call.prepared.launch()
    for _ in range(2):
        call.poison()
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(call.snapshot(), eager, "graph replay")


@requires_blackwell
@pytest.mark.parametrize("lengths,heads,lower_bound,route,turnover", GRAPH_ROWS)
def test_plan_cache_hit_matches_a_fresh_preparation(
    lengths, heads, lower_bound, route, turnover
):
    if turnover is None:
        turnover = _bounded_turnover_expected(lengths, heads)
    cache = KDAPrefillPlanCache(8)
    first = _inputs(lengths, heads, seed=1)
    second = _inputs(lengths, heads, seed=2)
    miss = _Call(first, lower_bound, plan_cache=cache)
    _assert_route(route, miss.prepared)
    miss.run()
    hit = _Call(second, lower_bound, plan_cache=cache)
    assert (cache.misses, cache.hits) == (1, 1)
    _assert_route(route, hit.prepared)
    assert ("_turnover" in str(hit.prepared.schedule)) == turnover
    got = hit.run()
    fresh = _Call(second, lower_bound)
    _assert_bitwise(got, fresh.run(), "plan-cache hit")


@requires_blackwell
@pytest.mark.parametrize(
    "lengths,heads",
    [
        pytest.param([128] * 16, 16, id="bs16x128-h16"),
        pytest.param([64] * 16, 16, id="bs16x64-h16"),
        pytest.param([256] * 16, 12, id="bs16x256-h12"),
        pytest.param([128, 160, 192, 100] * 12, 16, id="mixed-h16"),
    ],
)
def test_bounded_turnover_launch_reads_its_own_bin_table(lengths, heads, monkeypatch):
    """A bounded launch's route reads the table the launch itself uploads: every
    ``_make_direct_work_bins`` call of the preparation (the work-bin rule, the
    bounded predicate where the part has a count, the launch's table) has one
    key and returns one cached table object, from host lengths; the loaded
    turnover program matches the direct call of the predicate, and a plan-cache
    hit makes no new table call and keeps the route."""
    sm = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    expected = _bounded_turnover_expected(lengths, heads)
    calls = _spy_work_bins(monkeypatch)
    cache = KDAPrefillPlanCache(8)
    miss = _Call(_inputs(lengths, heads, seed=1), -5.0, plan_cache=cache)
    callers = {caller for caller, _, _ in calls}
    key = (
        (tuple(lengths),),
        (("num_heads", heads), ("worker_count", sm), ("chunk_tokens", 32)),
    )
    assert {k for _, k, _ in calls} == {key}, [k for _, k, _ in calls]
    assert all(table is calls[0][2] for _, _, table in calls)
    assert {"_use_direct_work_bins", "__init__"} <= callers, callers
    assert callers <= {
        "_use_direct_work_bins",
        "_bounded_turnover_route",
        "__init__",
    }, callers
    turn = _turn_programs()
    assert bool(_modules(miss.prepared) & turn) == expected
    assert ("_turnover" in str(miss.prepared.schedule)) == expected
    miss.run()
    calls.clear()
    hit = _Call(_inputs(lengths, heads, seed=2), -5.0, plan_cache=cache)
    assert (cache.misses, cache.hits) == (1, 1) and not calls, calls
    assert bool(_modules(hit.prepared) & turn) == expected
    assert ("_turnover" in str(hit.prepared.schedule)) == expected


def _host_encoded_parts(prepared):
    """(owner, module, grid, bindings) of every launch part whose module encodes
    its tensor maps on the host: the part launches, their prepare kernels and,
    on the apply route, the pair-map (none when the main body folds it),
    prefix and fused apply kernels (owner ``None``: the composite re-prepares
    them itself)."""
    parts = []
    seen = set()

    def add(owner, module, grid, bindings):
        if isinstance(module, registry.PreparedKernel) and module.encodes_on_host:
            parts.append((owner, module, grid, bindings))

    def visit(obj, depth):
        if depth > 7 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            for value in obj.values():
                visit(value, depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for value in obj:
                visit(value, depth + 1)
            return
        if not (
            type(obj).__module__.startswith("flashinfer") and hasattr(obj, "__dict__")
        ):
            return
        attrs = vars(obj)
        for module, grid, bindings in (
            ("module", "grid", "args"),
            ("prepare_module", "prepare_grid", "prepare_args"),
            ("_projection_module", "_projection_grid", "_projection_args"),
        ):
            if attrs.get(module) is not None:
                add(obj, attrs[module], attrs[grid], attrs[bindings])
        if attrs.get("_apply_route"):
            for part in ("pairmap", "prefix", "apply"):
                if attrs[f"_{part}_module"] is None:
                    continue
                add(
                    None,
                    attrs[f"_{part}_module"],
                    attrs[f"_{part}_grid"],
                    getattr(obj, f"_{part}_bindings")(),
                )
        for value in attrs.values():
            visit(value, depth + 1)

    visit(prepared, 0)
    return parts


def _assert_stored_maps_are_current(prepared, label):
    """Every part holds the maps of the tensors it is bound to, or is marked
    stale and re-encodes before its next kernel: a part the route does not
    launch (the window-map launch of the lean apply route, which the pair-map
    producer and the prefix chain replace) keeps the maps of its last
    preparation after a rebind."""
    parts = _host_encoded_parts(prepared)
    assert parts, f"{label}: no part encodes its tensor maps on the host"
    for owner, module, grid, bindings in parts:
        assert module._ready, (label, module.name)
        fresh = module.encoded_tensor_maps(grid=grid, **bindings)
        if torch.equal(fresh, module.descriptor_storage):
            continue
        assert owner is not None and owner._descriptors_stale, (
            f"{label}: {module.name} launches with maps of other tensors"
        )
    return {id(module): module.descriptor_storage.clone() for _, module, _, _ in parts}


STORED_MAP_ROWS = [
    pytest.param([1024] * 4, 16, None, id="direct-h16"),
    pytest.param([8192], 12, -5.0, id="affine-chain-h12-lb5"),
    pytest.param([8392], 16, None, id="affine-tail-h16"),
    pytest.param([17, 64, 65, 127, 128, 200, 255, 96] * 2, 12, None, id="mixed-h12"),
]


@requires_blackwell
@pytest.mark.parametrize("lengths,heads,lower_bound", STORED_MAP_ROWS)
def test_stored_tensor_maps_equal_a_fresh_encode_after_prepare_and_rebind(
    lengths, heads, lower_bound
):
    """Every map a host-encoding module launches with is the map of the
    tensors it is bound to: after preparation, after a plan-cache rebind to a
    complete new tensor set (applied by the next launch, deferred parts
    included) and after a hit that moves the output alone."""
    cache = KDAPrefillPlanCache(8)
    first = _inputs(lengths, heads, seed=3)
    second = _inputs(lengths, heads, seed=4)
    miss = _Call(first, lower_bound, plan_cache=cache)
    before = _assert_stored_maps_are_current(miss.prepared, "prepared")
    miss.run()
    _assert_stored_maps_are_current(miss.prepared, "launched")
    hit = _Call(second, lower_bound, plan_cache=cache)
    assert (cache.misses, cache.hits) == (1, 1)
    got = hit.run()
    after = _assert_stored_maps_are_current(hit.prepared, "rebound")
    assert any(
        not torch.equal(before[key], after[key]) for key in before if key in after
    ), "the rebind to new tensors re-encoded no map"
    _assert_bitwise(got, _Call(second, lower_bound).run(), "plan-cache hit")
    # Serving allocates the output per call: a hit that moves the output and
    # the checkpoint rows (a _Call allocates both) and keeps every input.
    again = _Call(second, lower_bound, plan_cache=cache)
    assert (cache.misses, cache.hits) == (1, 2)
    _assert_bitwise(again.run(), got, "output moved")
    _assert_stored_maps_are_current(again.prepared, "output moved")


# ------------------------------------------------- host routes: PS, MM, fallbacks

PAIR_SHARE_ROWS = [
    pytest.param([8192, 17], 12, id="8192+17-h12"),
    pytest.param([3000, 13384], 16, id="3000+13384-h16"),
    pytest.param([8192], 6, id="8192-h6"),
]


def _pair_share_cases():
    """Without state indices (dense states, zero initial states) the long rows
    above take the M64 value split; with them they take the chain composite
    (test below), so the indexed pool runs on serving rows the pair-share
    route takes on both arches.  Without checkpoint rows the program skips
    its checkpoint handshakes: the long rows on dense states, and two
    4096-token sequences at H16 (the route-export pin's shape) on dense
    states and on the indexed pool."""
    rows = [([8192, 17], 12, "8192+17-h12"), ([3000, 13384], 16, "3000+13384-h16")]
    rows += [([8192], 6, "8192-h6")]
    states = [(dict(indexed=False), "dense-state")]
    states += [(dict(indexed=False, initial=False), "zero-initial-state")]
    states += [(dict(indexed=False, rows=False), "dense-state-no-rows")]
    cases = [
        pytest.param(lengths, heads, state, id=f"{row}-{name}")
        for lengths, heads, row in rows
        for state, name in states
    ]
    for lengths, row in (([8192, 8192], "2x8192"), ([8128, 64] * 2, "2x(8128+64)")):
        cases.append(
            pytest.param(lengths, 16, dict(indexed=True), id=f"{row}-h16-indexed-pool")
        )
    for state, name in (
        (dict(indexed=False, rows=False), "dense-state-no-rows"),
        (dict(indexed=True, rows=False), "indexed-pool-no-rows"),
    ):
        cases.append(pytest.param([4096] * 2, 16, state, id=f"2x4096-h16-{name}"))
    return cases


def _own_pool(d):
    """The same inputs with this arm's own copy of the initial pool and indices."""
    return dict(
        d, state=d["initial_pool"].clone(), state_indices=d["state_indices"].clone()
    )


@requires_blackwell
@pytest.mark.parametrize("lengths,heads,state", _pair_share_cases())
def test_pair_share_is_bitwise_the_routed_m64_program(
    lengths, heads, state, monkeypatch
):
    """A long M64 value-split launch that takes the pair-share build equals, bit
    for bit, the routed M64 program (the loader without pair-share builds):
    output, final states and checkpoint rows, with and without state indices,
    initial states and checkpoint rows; each arm has its own copy of the
    initial pool.  The owner/helper rule is held off: it would take the rows
    of up to 36 tasks whose sequences all reach its length."""
    arch = registry.device_arch()
    pair_share = set(registry._RENDERED_PAIR_SHARE_ROUTES[arch].values())
    _m64_owner_helper_rule(monkeypatch, 0, 0)
    d = _inputs(lengths, heads)

    shared = _Call(_own_pool(d), -5.0, **state)
    schedule = str(shared.prepared.schedule)
    assert schedule.startswith("fused_m64_independent_dvsplit"), schedule
    assert schedule.endswith("_pair_share"), schedule
    assert _modules(shared.prepared) & pair_share
    shared_result = shared.run()
    _assert_bitwise(shared.run(), shared_result, "pair-share repeat")

    monkeypatch.setattr(
        registry,
        "_RENDERED_PAIR_SHARE_ROUTES",
        {name: {} for name in registry.ARCHES},
    )
    routed = _Call(_own_pool(d), -5.0, **state)
    assert str(routed.prepared.schedule) == schedule.removesuffix("_pair_share")
    assert not _modules(routed.prepared) & pair_share
    _assert_bitwise(shared_result, routed.run(), "pair-share vs routed M64")


@requires_blackwell
@pytest.mark.parametrize("lengths,heads", PAIR_SHARE_ROWS)
def test_indexed_long_rows_take_the_chain_not_the_pair_share(
    lengths, heads, monkeypatch
):
    """With state indices the long rows above take the chain composite (the
    affine split needs indexed states), not the M64 value split: no pair-share
    program loads, and clearing the pair-share table changes no bit."""
    arch = registry.device_arch()
    pair_share = set(registry._RENDERED_PAIR_SHARE_ROUTES[arch].values())
    d = _inputs(lengths, heads)
    chain = _Call(_own_pool(d), -5.0)
    schedule = str(chain.prepared.schedule)
    assert schedule.startswith("split_seq_affine_prefix"), schedule
    assert not _modules(chain.prepared) & pair_share
    want = chain.run()
    monkeypatch.setattr(
        registry,
        "_RENDERED_PAIR_SHARE_ROUTES",
        {name: {} for name in registry.ARCHES},
    )
    cleared = _Call(_own_pool(d), -5.0)
    assert str(cleared.prepared.schedule) == schedule
    _assert_bitwise(cleared.run(), want, "pair-share table cleared")


# The pair-share decision of a part: M64_PAIR_SHARE_MIN_SEQ_LEN on a 148-SM
# sm_100a part, rows (32, 1024) and (48, 2048), and
# M64_PAIR_SHARE_MIN_SEQ_LEN_GB300 on a 152-SM sm_103a part, rows (32, 1024),
# (64, 2048) and (72, 4096).  The first row whose task count holds the
# launch's long-task load (heads x tokens / longest sequence) gives the
# shortest longest sequence that takes the build; a load beyond the last row
# keeps the routed program, and so does a launch with more tasks than the
# device holds 2-CTA clusters.  Expected: (148-SM sm_100a, 152-SM sm_103a).
PAIR_SHARE_PARTS = (("sm_100a", 148), ("sm_103a", 152))
PAIR_SHARE_THRESHOLD_ROWS = [
    # load 16: the 1024-token row of both parts
    pytest.param([1023], 16, (False, False), id="1023-h16"),
    pytest.param([1024], 16, (True, True), id="1024-h16"),
    pytest.param([1025], 16, (True, True), id="1025-h16"),
    # load 48: the 2048-token row of both parts
    pytest.param([2047] * 3, 16, (False, False), id="3x2047-h16"),
    pytest.param([2048] * 3, 16, (True, True), id="3x2048-h16"),
    # load 64, and 56 with a short sequence: beyond the sm_100a rows, the
    # 2048-token row of sm_103a
    pytest.param([2047] * 4, 16, (False, False), id="4x2047-h16"),
    pytest.param([2048] * 4, 16, (False, True), id="4x2048-h16"),
    pytest.param([2048] * 3 + [1024], 16, (False, True), id="3x2048+1024-h16"),
    # load 72: beyond the sm_100a rows, the 4096-token row of sm_103a
    pytest.param([4095] * 6, 12, (False, False), id="6x4095-h12"),
    pytest.param([4096] * 6, 12, (False, True), id="6x4096-h12"),
]


@requires_blackwell
@pytest.mark.parametrize("lengths,heads,expected", PAIR_SHARE_THRESHOLD_ROWS)
def test_pair_share_threshold_of_the_part(lengths, heads, expected, monkeypatch):
    """The side of the part's pair-share threshold an M64 value-split launch
    (dense FP32 states, checkpoint rows) lands on: the exact schedule of the
    routed or of the pair-share program.  Where the build is taken it equals
    the routed program bit for bit, and a device that holds one 2-CTA cluster
    fewer than the launch has tasks keeps the routed program.  The owner/helper
    rule is held off (it would take the launches of up to 36 tasks)."""
    _m64_owner_helper_rule(monkeypatch, 0, 0)
    part = (registry.device_arch(), _sm_count())
    if part not in PAIR_SHARE_PARTS:
        pytest.skip(
            f"{part}: the decisions are pinned for 148-SM sm_100a and 152-SM "
            "sm_103a parts"
        )
    shared = expected[PAIR_SHARE_PARTS.index(part)]
    tasks = len(lengths) * heads
    device = torch.device("cuda", torch.cuda.current_device())
    capacity = runtime._probe_m64_pair_cluster_capacity(device)
    assert capacity, "the cluster capacity probe failed"
    if shared and capacity < tasks:
        pytest.skip(f"{part}: {capacity} 2-CTA clusters for {tasks} tasks")
    pair_share = set(registry._RENDERED_PAIR_SHARE_ROUTES[part[0]].values())
    routed_schedule = "fused_m64_independent_dvsplit_fp32_state"
    d = _inputs(lengths, heads)
    call = _Call(_own_pool(d), -5.0, indexed=False)
    schedule = str(call.prepared.schedule)
    assert schedule == routed_schedule + ("_pair_share" if shared else ""), schedule
    assert bool(_modules(call.prepared) & pair_share) == shared
    if not shared:
        return
    result = call.run()
    monkeypatch.setattr(runtime, "_m64_pair_cluster_capacity", lambda _: tasks - 1)
    routed = _Call(_own_pool(d), -5.0, indexed=False)
    assert str(routed.prepared.schedule) == routed_schedule
    assert not _modules(routed.prepared) & pair_share
    _assert_bitwise(result, routed.run(), "pair-share vs routed M64")
    monkeypatch.setattr(runtime, "_m64_pair_cluster_capacity", lambda _: tasks)
    at_capacity = _Call(_own_pool(d), -5.0, indexed=False)
    assert str(at_capacity.prepared.schedule) == schedule


# The M64 owner/helper build: long M64 value-split launches (dense FP32
# states) whose owner and helper CTAs the device holds at once.  Rule rows
# that take exactly the launch's task count (``_m64_owner_helper_rule``) stand
# in for the measured rule.
M64_OWNER_HELPER_ROWS = [
    pytest.param([8192], 6, id="8192-h6"),
    pytest.param([1024], 16, id="1024-h16"),
    pytest.param([4096] * 2, 16, id="2x4096-h16"),
    pytest.param([8192, 17], 12, id="8192+17-h12"),
    pytest.param([3000, 13384], 16, id="3000+13384-h16"),
]
M64_OWNER_HELPER_STATES = [
    pytest.param(dict(), id="dense-state"),
    pytest.param(dict(initial=False), id="zero-initial-state"),
    pytest.param(dict(rows=False), id="dense-state-no-rows"),
]


def _m64_owner_helper_rule(monkeypatch, tasks, helpers):
    """Rule rows of both parts that take launches of ``tasks`` tasks with
    ``helpers`` helpers per task (no rows for 0)."""
    rows = ((tasks, 1, helpers),) if helpers else ()
    monkeypatch.setattr(runtime, "M64_OWNER_HELPER_MIN_SEQ_LEN", rows)
    monkeypatch.setattr(runtime, "M64_OWNER_HELPER_MIN_SEQ_LEN_GB300", rows)


def _m64_owner_helper_programs():
    arch = registry.device_arch()
    programs = set(registry._RENDERED_M64_OWNER_HELPER_ROUTES[arch].values())
    if not programs:
        pytest.skip(f"the loader names no M64 owner/helper build on {arch}")
    return programs


@requires_blackwell
@pytest.mark.parametrize("helpers", [2, 3])
@pytest.mark.parametrize("state", M64_OWNER_HELPER_STATES)
@pytest.mark.parametrize("lengths,heads", M64_OWNER_HELPER_ROWS)
def test_m64_owner_helper_is_bitwise_the_pair_share_and_routed_programs(
    lengths, heads, state, helpers, monkeypatch
):
    """A long M64 value-split launch that takes the owner/helper build equals,
    bit for bit, the program it replaces (pair-share where that rule takes the
    launch, else routed) and the routed program: output, final states and
    checkpoint rows, launch after launch (the program leaves its words zero),
    with and without initial states and checkpoint rows."""
    owner_helper = _m64_owner_helper_programs()
    tasks = len(lengths) * heads
    device = torch.device("cuda", torch.cuda.current_device())
    capacity = runtime._probe_m64_cooperative_capacity(device)
    assert capacity, "the cooperative capacity probe failed"
    if tasks * (2 + helpers) > capacity:
        pytest.skip(f"{capacity} resident CTAs for {tasks} x (2 + {helpers})")
    d = _inputs(lengths, heads)
    _m64_owner_helper_rule(monkeypatch, tasks, helpers)
    taken = _Call(_own_pool(d), -5.0, indexed=False, **state)
    schedule = str(taken.prepared.schedule)
    suffix = f"_owner{helpers}helper"
    assert schedule.startswith("fused_m64_independent_dvsplit_fp32_state"), schedule
    assert schedule.endswith(suffix), schedule
    assert _modules(taken.prepared) & owner_helper
    result = taken.run()
    _assert_bitwise(taken.run(), result, "owner/helper repeat")

    monkeypatch.undo()
    _m64_owner_helper_rule(monkeypatch, tasks, 0)
    base = schedule.removesuffix(suffix)
    kept = _Call(_own_pool(d), -5.0, indexed=False, **state)
    assert str(kept.prepared.schedule) in (base, base + "_pair_share")
    assert not _modules(kept.prepared) & owner_helper
    _assert_bitwise(result, kept.run(), "owner/helper vs the program it replaces")
    monkeypatch.setattr(
        registry,
        "_RENDERED_PAIR_SHARE_ROUTES",
        {name: {} for name in registry.ARCHES},
    )
    routed = _Call(_own_pool(d), -5.0, indexed=False, **state)
    assert str(routed.prepared.schedule) == base
    _assert_bitwise(result, routed.run(), "owner/helper vs routed M64")


@requires_blackwell
def test_refused_m64_owner_helper_launch_runs_its_fallback(monkeypatch):
    """A cooperative grid the device cannot hold (the capacity probe
    overridden): the driver refuses the owner/helper launch, the plan switches
    for good to the pair-share or routed program it would otherwise take and
    matches that program bit for bit, and later CUDA work is unaffected."""
    owner_helper = _m64_owner_helper_programs()
    lengths, heads = [4096] * 6, 12
    tasks = len(lengths) * heads
    assert tasks * 4 > _sm_count()
    d = _inputs(lengths, heads)
    monkeypatch.setattr(runtime, "_m64_cooperative_capacity", lambda device: 1 << 20)
    _m64_owner_helper_rule(monkeypatch, tasks, 2)
    refused = _Call(_own_pool(d), -5.0, indexed=False)
    schedule = str(refused.prepared.schedule)
    assert schedule.endswith("_owner2helper"), schedule
    assert _modules(refused.prepared) & owner_helper
    result = refused.run()
    assert not _modules(refused.prepared) & owner_helper
    probe = torch.zeros(1024, device="cuda")
    probe.add_(1)
    torch.cuda.synchronize()
    assert float(probe.sum()) == 1024.0
    _assert_bitwise(refused.run(), result, "fallback repeat")

    monkeypatch.undo()
    _m64_owner_helper_rule(monkeypatch, tasks, 0)
    kept = _Call(_own_pool(d), -5.0, indexed=False)
    assert not _modules(kept.prepared) & owner_helper
    assert str(kept.prepared.schedule) in (
        schedule.removesuffix("_owner2helper"),
        schedule.removesuffix("_owner2helper") + "_pair_share",
    )
    _assert_bitwise(result, kept.run(), "refused owner/helper vs its fallback")


CHAIN_ROWS = [
    pytest.param([8192], 12, id="8192-h12"),
    pytest.param([3000, 13384], 16, id="3000+13384-h16"),
]


@requires_blackwell
@pytest.mark.parametrize(
    "env,lower_bound,checkpoint_dtype",
    [
        pytest.param(
            "CAKE_KDA_AFFINE_APPLY", None, torch.bfloat16, id="apply-off-bf16-rows"
        ),
        pytest.param("CAKE_KDA_AFFINE_APPLY", None, None, id="apply-off-default-rows"),
        pytest.param(
            "CAKE_KDA_AFFINE_ROWS_IN_PLACE",
            None,
            torch.float32,
            id="fp32-rows-not-in-place",
        ),
    ],
)
@pytest.mark.parametrize("lengths,heads", CHAIN_ROWS)
def test_chain_composites_without_a_map_fold_build_keep_the_map_pass(
    lengths, heads, env, lower_bound, checkpoint_dtype, monkeypatch
):
    """Chain composites whose main module has no map-fold build launch the map
    pass: no ``_mapfold`` schedule, no map-fold program.  The unbounded gate
    takes the chain with the apply route off, or with FP32 rows that are not
    written in place (the apply route needs them in place)."""
    if checkpoint_dtype == torch.float32 and _checkpoint_dtype(None) != torch.float32:
        pytest.skip("no FP32 checkpoint rows for the unbounded gate here")
    monkeypatch.setenv(env, "0")
    arch = registry.device_arch()
    folding = set(registry._RENDERED_MAP_FOLD_ROUTES[arch].values())
    d = _inputs(lengths, heads)
    call = _Call(d, lower_bound, checkpoint_dtype=checkpoint_dtype)
    schedule = str(call.prepared.schedule)
    assert schedule.startswith("split_seq_affine_prefix"), schedule
    assert "_apply" not in schedule, schedule
    assert "_mapfold" not in schedule, schedule
    assert not _modules(call.prepared) & folding
    _assert_close(call.run(), _reference(d, lower_bound), env)


@requires_blackwell
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            dict(env="FLASHINFER_KDA_AFFINE_FUSED_EPILOGUE"), id="torch-epilogue"
        ),
        pytest.param(dict(rows=False), id="no-checkpoint-rows"),
    ],
)
@pytest.mark.parametrize("lengths,heads", CHAIN_ROWS)
def test_chain_map_fold_is_bitwise_the_map_pass_off_the_serving_path(
    lengths, heads, variant, monkeypatch
):
    """The chain composite with the torch epilogue, and without checkpoint rows,
    equals the composite that launches the map pass, bit for bit; where the
    main module has a map-fold build the composite takes it (``_mapfold``)."""
    if "env" in variant:
        monkeypatch.setenv(variant["env"], "0")
    rows = variant.get("rows", True)
    arch = registry.device_arch()
    folding = set(registry._RENDERED_MAP_FOLD_ROUTES[arch].values())
    d = _inputs(lengths, heads)
    untouched = torch.ones(d["state"].shape[0], dtype=torch.bool, device="cuda")
    untouched[d["state_indices"].long()] = False

    folded = _Call(d, -5.0, rows=rows)
    schedule = str(folded.prepared.schedule)
    assert schedule.startswith("split_seq_affine_prefix"), schedule
    fold = bool(_modules(folded.prepared) & folding)
    assert fold == ("_mapfold" in schedule), schedule
    if _chain_map_fold_enabled():
        assert fold, "the chain composite launched the map pass"
    folded_result = folded.run()
    assert torch.equal(d["state"][untouched], d["initial_pool"][untouched])
    _assert_bitwise(folded.run(), folded_result, "folded repeat")

    monkeypatch.setattr(
        registry, "_RENDERED_MAP_FOLD_ROUTES", {name: {} for name in registry.ARCHES}
    )
    mapped = _Call(d, -5.0, rows=rows)
    assert "_mapfold" not in str(mapped.prepared.schedule)
    assert not _modules(mapped.prepared) & folding
    _assert_bitwise(folded_result, mapped.run(), "folded vs map pass")


INTERCEPTORS = [
    pytest.param(
        runtime.FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerMapFoldLaunch,
        runtime._ChainMapFoldUnavailable,
        "_RENDERED_MAP_FOLD_ROUTES",
        "_chain_map_fold_factory",
        -5.0,
        "_mapfold",
        id="map-fold",
    ),
    pytest.param(
        runtime.FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerExportFoldLaunch,
        runtime._PairMapFoldUnavailable,
        "_RENDERED_FOLD_ROUTES",
        "_pair_map_fold_factory",
        None,
        "_pairfold",
        id="pair-map-fold",
    ),
    pytest.param(
        runtime.FlashKDABlackwellFP32SlabM128PDLConsumerMergeLaunch,
        runtime._CorrectionMergeUnavailable,
        "_RENDERED_MERGE_ROUTES",
        "_correction_merge_factory",
        -5.0,
        "_correction_merge",
        id="correction-merge",
    ),
]


def _unrecognised_factories():
    """Factories the fold / merge interceptors must not rewrite."""
    m128 = partial(runtime._factory, "compiled_bf16_fused_m128")
    return [
        partial(runtime._factory, "compiled_bf16_fused_m64"),
        partial(runtime._m64_pair_share_factory, "compiled_bf16_fused_m128"),
        partial(lambda factory, *a, **k: factory(*a, **k), m128),  # a wrapper
        lambda *a, **k: None,
    ]


@pytest.mark.parametrize(
    "launch_cls,unavailable,table,factory_name,lower_bound,suffix", INTERCEPTORS
)
def test_interceptor_refuses_an_unrecognised_factory(
    launch_cls, unavailable, table, factory_name, lower_bound, suffix, monkeypatch
):
    """Fail closed: an interceptor raises its ``...Unavailable`` exception for
    any factory but the plain fused M128 row, before anything is built."""
    built = []
    monkeypatch.setattr(
        runtime, "_build_kda_module", lambda factory, *a, **k: built.append(factory)
    )
    for factory in _unrecognised_factories():
        with pytest.raises(unavailable):
            launch_cls._build_kda_module(factory, 32)
    assert not built


@requires_blackwell
@pytest.mark.parametrize(
    "launch_cls,unavailable,table,factory_name,lower_bound,suffix", INTERCEPTORS
)
def test_composite_falls_back_when_its_interceptor_does_not_apply(
    launch_cls, unavailable, table, factory_name, lower_bound, suffix, monkeypatch
):
    """When the interceptor does not recognise the main / correction factory,
    or the body it built is not the build the loader names, the composite
    takes its fallback (map pass, pair-map producer, epilogue): same bits."""
    arch = registry.device_arch()
    programs = set(getattr(registry, table)[arch].values())
    d = _inputs([8192], 12)
    routed = _Call(d, lower_bound)
    schedule = str(routed.prepared.schedule)
    if suffix not in schedule:
        pytest.skip(f"{schedule}: the composite takes no {suffix} build here")
    assert _modules(routed.prepared) & programs
    routed_result = routed.run()

    # The interceptor does not recognise the factory: it raises, the
    # composite falls back.
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "_is_fused_m128_row", lambda factory: False)
        refused = _Call(d, lower_bound)
        assert suffix not in str(refused.prepared.schedule)
        assert not _modules(refused.prepared) & programs
        _assert_bitwise(refused.run(), routed_result, "interceptor refused")

    # The interceptor applies but the body is not the loader's build: the
    # post-build check falls back as well.
    with monkeypatch.context() as patch:
        patch.setattr(runtime, factory_name, runtime._factory)
        wrong_body = _Call(d, lower_bound)
        assert suffix not in str(wrong_body.prepared.schedule)
        assert not _modules(wrong_body.prepared) & programs
        _assert_bitwise(wrong_body.run(), routed_result, "body check")


def _drop_map_producer(dropped):
    """``_map_records`` without the window-map producer: the map pass, or the
    pair-map producer ahead of the apply route's prefix chain."""
    records = runtime.FlashKDABlackwellAffineSplitLaunch._map_records

    def without_producer(self):
        kept = records(self)
        if self._apply_route and self._lean_map:
            dropped.append(len(kept) - 1)
            return kept[-1:]
        dropped.append(len(kept))
        return []

    return without_producer


def _drop_merging_correction(dropped):
    """``_settled_launch_records`` without the merging correction windows."""
    settled = runtime._settled_launch_records

    def without_merge(launch):
        merging = runtime.FlashKDABlackwellFP32SlabM128PDLConsumerMergeLaunch
        if isinstance(launch, merging):
            dropped.append(len(settled(launch)))
            return []
        return settled(launch)

    return without_merge


DROPPED_PRODUCERS = [
    pytest.param("_RENDERED_MAP_FOLD_ROUTES", -5.0, "_mapfold", id="map-pass"),
    pytest.param("_RENDERED_FOLD_ROUTES", None, "_pairfold", id="pair-map-producer"),
    pytest.param(
        "_RENDERED_MERGE_ROUTES", -5.0, "_correction_merge", id="merging-correction"
    ),
]


@requires_blackwell
@pytest.mark.parametrize("table,lower_bound,suffix", DROPPED_PRODUCERS)
def test_poisoned_bitwise_compare_sees_a_dropped_producer(
    table, lower_bound, suffix, monkeypatch
):
    """Seeded negative for the bitwise composite tests.  The map-fold, pair-map
    fold and merging builds let a composite drop a producer (the map pass, the
    pair-map producer, the epilogue).  A composite that drops the producer
    without such a body leaves its maps or outputs unwritten.  A first call
    with the producer leaves the right maps in memory the allocator hands out
    again; with free memory and outputs poisoned before every arm, the bitwise
    compare still fails on the call without it."""
    d = _inputs([8192], 12)
    routed = _Call(d, lower_bound)
    if suffix not in str(routed.prepared.schedule):
        pytest.skip(f"{routed.prepared.schedule}: no {suffix} build here")
    want = routed.run()
    dropped = []
    with monkeypatch.context() as patch:
        if table == "_RENDERED_MERGE_ROUTES":
            drop = (runtime, "_settled_launch_records", _drop_merging_correction)
        else:
            # The composite without the fold build: its producer writes the maps.
            patch.setattr(registry, table, {name: {} for name in registry.ARCHES})
            drop = (
                runtime.FlashKDABlackwellAffineSplitLaunch,
                "_map_records",
                _drop_map_producer,
            )
        kept = _Call(d, lower_bound)
        if table != "_RENDERED_MERGE_ROUTES":
            assert suffix not in str(kept.prepared.schedule)
        _assert_bitwise(kept.run(), want, "producer kept")
        del kept
        patch.setattr(drop[0], drop[1], drop[2](dropped))
        without = _Call(d, lower_bound)
        got = without.run()
        assert dropped and all(dropped), f"no producer launch dropped: {dropped}"
        with pytest.raises(AssertionError, match="differs"):
            _assert_bitwise(got, want, "producer dropped")


@requires_blackwell
def test_pair_cluster_capacity_probe_caches_only_an_answer(monkeypatch):
    """A failed capacity probe returns 0 (the routed M64 program), warns once
    and is not cached; the next probe that answers is."""
    from flashinfer.jit.core import logger

    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(runtime, "_M64_PAIR_CLUSTER_CAPACITY", {})
    monkeypatch.setattr(runtime, "_M64_PAIR_PROBE_WARNED", False)
    warnings = []

    def no_device(*args, **kwargs):
        raise RuntimeError("probe failure")

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "device", no_device)
        patch.setattr(logger, "warning", lambda *args, **kwargs: warnings.append(args))
        assert runtime._m64_pair_cluster_capacity(device) == 0
        assert runtime._m64_pair_cluster_capacity(device) == 0
        assert len(warnings) == 1, warnings
        assert "capacity probe failed" in warnings[0][0]
        assert device not in runtime._M64_PAIR_CLUSTER_CAPACITY
    capacity = runtime._m64_pair_cluster_capacity(device)
    assert capacity > 0
    assert runtime._M64_PAIR_CLUSTER_CAPACITY[device] == capacity
