"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""Generated BF16 KDA prefill programs in the shared KDA loader.

``_RENDERED_MODULES`` registers the generated programs (each compiled from its
own device unit and host binding under ``csrc/kda/prefill/``) and
``_RENDERED_ROUTES`` points the shipped modules of each architecture at them.

The CPU checks hold both tables to the units, to the shipped argument plans and
to the factory lookup. The GPU checks run the prepared BF16 prefill on the
serving shapes (FP32 in-place state pool, reversed int32 state indices, a
strided BF16 beta-logit slice, checkpoint rows every 64 tokens) with the routes
in place and with the routes cleared, each against an FP32 token-by-token
reference; replay the routed launch from a CUDA graph; and rebind it through a
plan-cache hit.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import flashinfer.jit.cake_kda_tf32 as registry
from flashinfer import (
    KDAPrefillPlanCache,
    kda_prefill_supports_fp32_checkpoints,
    prepare_bf16_kda_prefill,
)

ROOT = Path(registry.__file__).resolve().parents[2]
UNIT_DIR = "csrc/kda/prefill"
NAME = re.compile(r"cake_kda_prefill_[0-9a-f]{20}")
LEAN_KEYS = {"sources", "ffi_entry", "compile_flags", "arg_plan", "arches"}
GRID = [["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]
PLAN_KINDS = {"buffer", "tma_buffer", "parameter", "grid"}
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
    for arch, routes in registry._RENDERED_ROUTES.items():
        for module, program in routes.items():
            shipped = {
                name: kind for kind, name in registry.MODULES[module]["arg_plan"]
            }
            for kind, name in registry._RENDERED_MODULES[program]["arg_plan"]:
                if kind == "grid":
                    continue
                where = (arch, module, program, name)
                assert name in shipped, where
                assert (kind in TENSOR_KINDS) == (shipped[name] in TENSOR_KINDS), where
                assert (kind == "parameter") == (shipped[name] == "parameter"), where


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


@requires_programs
def test_prepared_kernel_calls_run_without_a_descriptor_workspace(monkeypatch):
    name = GENERATED[0]
    record = registry._RENDERED_MODULES[name]
    calls = []
    native = SimpleNamespace(run=lambda *args: calls.append(args))
    monkeypatch.setattr(registry, "load", lambda module, arch: native)
    kernel = registry.PreparedKernel(name, record["arches"][0])
    assert kernel._prepare is None
    assert kernel.descriptor_storage is None
    names = [slot_name for kind, slot_name in record["arg_plan"] if kind != "grid"]
    bindings = {slot_name: object() for slot_name in names}
    bindings["unused_by_this_plan"] = object()
    kernel.prepare(grid=(3, 2), **bindings)
    assert calls == []
    kernel.launch(grid=(3, 2), **bindings)
    assert calls == [(*(bindings[slot_name] for slot_name in names), 3, 2, 1)]


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
        monkeypatch.setattr(
            registry, "_RENDERED_ROUTES", {arch: {} for arch in registry.ARCHES}
        )

    return clear


@requires_blackwell
@pytest.mark.parametrize("lengths,heads,lower_bound", ROWS)
def test_serving_rows_match_the_fp32_reference(lengths, heads, lower_bound, unrouted):
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
