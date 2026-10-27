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
launches asking for work bins take instead, and ``_RENDERED_OWNER_HELPER_ROUTES``
the owner/helper program a call takes when the device holds its grid.

The CPU checks hold both tables to the units, to the shipped argument plans and
to the factory lookup. The GPU checks run the prepared BF16 prefill on the
serving shapes (FP32 in-place state pool, reversed int32 state indices, a
strided BF16 beta-logit slice, checkpoint rows every 64 tokens) with the routes
in place and with the routes cleared, each against an FP32 token-by-token
reference, and a launch with work bins bit for bit against the one-task-per-CTA
program and against the selection without work bins; replay the routed launch from a CUDA graph; and rebind it through a
plan-cache hit.

A program that takes tensor maps encodes them once per preparation into a host
buffer (``("workspace", "tensor_maps")`` in its plan); the checks hold every
stored map equal to a fresh encode of the bound tensors after preparation and
after a plan-cache rebind to new tensors.
"""

import re
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
    assert set(registry._RENDERED_FOLD_ROUTES) == set(registry.ARCHES)
    for arch, modules in registry._RENDERED_FOLD_ROUTES.items():
        for module, program in modules.items():
            # Every other launch of the module keeps its route.
            assert module in registry._RENDERED_ROUTES[arch], (arch, module)
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
            *registry._RENDERED_FOLD_ROUTES[arch].items(),
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


class _Call:
    """One prepared launch with its own output and checkpoint rows."""

    def __init__(self, d, lower_bound, *, cu_seqlens=None, plan_cache=None):
        heads = d["heads"]
        rows = _checkpoint_starts(d["lengths"])[-1]
        self.d = d
        self.out = torch.full_like(d["q"], float("nan"))
        self.checkpoints = torch.full(
            (rows, heads, HEAD_DIM, HEAD_DIM),
            float("nan"),
            device="cuda",
            dtype=_checkpoint_dtype(lower_bound),
        )
        self.prepared = prepare_bf16_kda_prefill(
            d["q"],
            d["k"],
            d["v"],
            d["g"],
            d["beta"],
            A_log=d["A_log"].reshape(heads),
            dt_bias=d["dt_bias"].reshape(heads, HEAD_DIM),
            out=self.out,
            initial_state=d["state"],
            final_state=d["state"],
            scale=None,
            lower_bound=lower_bound,
            cu_seqlens=d["cu_seqlens"] if cu_seqlens is None else cu_seqlens,
            sequence_lengths=d["lengths"],
            state_indices=d["state_indices"],
            state_checkpoints=self.checkpoints,
            checkpoint_cu_starts=d["checkpoint_cu_starts"],
            checkpoint_every_n_tokens=CHECKPOINT_EVERY,
            beta_is_logit=True,
            plan_cache=plan_cache,
        )

    def poison(self):
        self.d["state"].copy_(self.d["initial_pool"])
        self.out.fill_(float("nan"))
        self.checkpoints.fill_(float("nan"))

    def run(self):
        self.poison()
        self.prepared.launch()
        torch.cuda.synchronize()
        return self.snapshot()

    def snapshot(self):
        d = self.d
        return (
            self.out.clone(),
            d["state"][d["state_indices"].long()].clone(),
            self.checkpoints.clone(),
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
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        ), f"{label}: {name} differs"


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
            "_RENDERED_FOLD_ROUTES",
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
    # the module's one-task-per-CTA program and matches bit for bit.
    work_bin_programs = set(registry._RENDERED_WORK_BIN_ROUTES[arch].values())
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
        # early-pack body, the binned launch takes the binned N32 body).
        with monkeypatch.context() as patch:
            patch.setattr(
                runtime, "_use_direct_work_bins", lambda *args, **kwargs: False
            )
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


GRAPH_ROWS = [
    pytest.param([17, 64, 65, 127, 128, 200, 255, 96] * 2, 12, None, id="mixed-h12"),
    pytest.param([64] * 16, 16, -5.0, id="bs16x64-h16-lb5"),
    pytest.param([8128, 64, 8128, 64], 16, None, id="long-short-h16"),
]


@requires_blackwell
@pytest.mark.parametrize("lengths,heads,lower_bound", GRAPH_ROWS)
def test_routed_launch_replays_from_a_cuda_graph(lengths, heads, lower_bound):
    d = _inputs(lengths, heads)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        call = _Call(d, lower_bound)
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
@pytest.mark.parametrize("lengths,heads,lower_bound", GRAPH_ROWS)
def test_plan_cache_hit_matches_a_fresh_preparation(lengths, heads, lower_bound):
    cache = KDAPrefillPlanCache(8)
    first = _inputs(lengths, heads, seed=1)
    second = _inputs(lengths, heads, seed=2)
    miss = _Call(first, lower_bound, plan_cache=cache)
    miss.run()
    hit = _Call(second, lower_bound, plan_cache=cache)
    assert (cache.misses, cache.hits) == (1, 1)
    got = hit.run()
    fresh = _Call(second, lower_bound)
    _assert_bitwise(got, fresh.run(), "plan-cache hit")


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
