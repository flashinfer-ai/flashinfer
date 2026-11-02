"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""Host side of the M64 owner/helper build (CPU; no GPU needed).

An M64 FP32-state value-split launch takes the loader's owner/helper build of
its module (``_RENDERED_M64_OWNER_HELPER_ROUTES``, beside the module's
pair-share build) where the part's ``M64_OWNER_HELPER_MIN_SEQ_LEN`` row takes
it and the device holds the cooperative grid ``(tasks, 2 + helpers)`` at once;
the packet ring and the zeroed ready / consumed words ride in its
``final_state`` and ``seq_order`` slots.  A launch the driver refuses as a too
large cooperative grid switches the plan for good to the pair-share or routed
program it would otherwise have taken.

The device queries are stubbed (SM100a / SM103a, 148 or 152 SMs, the
cooperative and cluster capacities), the tensors live on the meta device and
the loader's ``PreparedKernel`` is a recording stub, so the routing, the grid,
the workspace and the fallback run without a kernel.
"""

import contextlib
from functools import partial
from types import SimpleNamespace

import pytest
import torch

import flashinfer.cake_kda_tf32_runtime as runtime
import flashinfer.jit.cake_kda_tf32 as registry

pytestmark = pytest.mark.cpu_only

HEAD_DIM = 128
NAME = "compiled_bf16_fused_m64"
M64_BODY = {"state_dtype_is_fp32": True, "active_beta_f32": False}
OWNER_HELPER = "cake_kda_prefill_00000000000000000064"
REFUSED = RuntimeError(
    "kernel_cake_kda_prefill_00000000000000000064: cudaLaunchKernelExC failed: "
    "too many blocks in cooperative launch"
)
PARTS = (("sm_100a", (10, 0), 148), ("sm_103a", (10, 3), 152))
PART_IDS = ["sm100a-148", "sm103a-152"]
RULES = {"sm_100a": "M64_OWNER_HELPER_MIN_SEQ_LEN"}
RULES["sm_103a"] = "M64_OWNER_HELPER_MIN_SEQ_LEN_GB300"


def _m64_module(arch):
    return registry.FACTORIES[arch][NAME][registry._key((), M64_BODY)]


# The loader's pair-share tables, before any test stubs them.
PAIR_SHARE = {
    arch: dict(routes) for arch, routes in registry._RENDERED_PAIR_SHARE_ROUTES.items()
}


# ------------------------------------------------------------- rule and probe


@pytest.mark.parametrize(
    ("rows", "tasks", "min_seq_len", "capacity", "expected"),
    [
        ((), 16, 1 << 20, 1 << 20, 0),  # empty table: never
        (((16, 4096),), 16, 4096, 148, 2),  # default two helpers
        (((16, 4096),), 16, 4095, 148, 0),  # a sequence below the row
        (((16, 4096),), 17, 8192, 148, 0),  # beyond the last row
        (((16, 4096), (32, 8192, 3)), 17, 8192, 148, 3),  # the row's helpers
        (((16, 4096), (32, 8192, 3)), 17, 4096, 148, 0),
        (((16, 4096), (32, 8192, 3)), 30, 8192, 148, 2),  # 3 do not fit: 2
        (((16, 4096), (32, 8192, 3)), 32, 8192, 148, 2),
        (((16, 4096), (32, 8192, 3)), 32, 8192, 127, 0),  # 2 do not fit
        (((16, 4096),), 16, 4096, 64, 2),  # 16 x (2 + 2) CTAs at capacity
        (((16, 4096),), 16, 4096, 63, 0),
        (((16, 4096),), 16, 4096, 0, 0),  # the probe failed
        (((16, 4096, 5),), 8, 4096, 148, 3),  # never beyond three
    ],
)
@pytest.mark.parametrize(("arch", "capability", "sm_count"), PARTS, ids=PART_IDS)
def test_owner_helper_count_follows_the_part_rule(
    monkeypatch,
    arch,
    capability,
    sm_count,
    rows,
    tasks,
    min_seq_len,
    capacity,
    expected,
):
    probed = []

    def probe(device):
        probed.append(device)
        return capacity

    monkeypatch.setattr(runtime, "_m64_cooperative_capacity", probe)
    for table in RULES.values():
        monkeypatch.setattr(runtime, table, ())
    monkeypatch.setattr(runtime, RULES[arch], rows)
    shape = dict(total_tasks=tasks, min_seq_len=min_seq_len, device="dev")
    got = runtime._m64_owner_helper_count(gpu_arch=arch, sm_count=sm_count, **shape)
    assert got == expected
    if got:
        assert tasks * (2 + got) <= capacity
    if not rows:
        assert not probed, "an empty table must not probe the device"
    # The other part's table (and another architecture) never takes the launch.
    other = 148 if sm_count == 152 else 152
    if arch == "sm_103a":
        assert (
            runtime._m64_owner_helper_count(gpu_arch=arch, sm_count=other, **shape) == 0
        )
    assert (
        runtime._m64_owner_helper_count(gpu_arch="sm_120a", sm_count=188, **shape) == 0
    )


def test_gb300_table_is_the_152_sm_sm103a_part(monkeypatch):
    monkeypatch.setattr(runtime, "_m64_cooperative_capacity", lambda device: 148)
    monkeypatch.setattr(runtime, "M64_OWNER_HELPER_MIN_SEQ_LEN", ((16, 1024),))
    monkeypatch.setattr(runtime, "M64_OWNER_HELPER_MIN_SEQ_LEN_GB300", ((16, 2048),))
    shape = dict(total_tasks=16, min_seq_len=1024, device="dev")
    assert runtime._m64_owner_helper_count(gpu_arch="sm_100a", sm_count=148, **shape)
    assert runtime._m64_owner_helper_count(gpu_arch="sm_103a", sm_count=148, **shape)
    assert not runtime._m64_owner_helper_count(
        gpu_arch="sm_103a", sm_count=152, **shape
    )
    shape["min_seq_len"] = 2048
    assert runtime._m64_owner_helper_count(gpu_arch="sm_103a", sm_count=152, **shape)


def test_rule_tables_ship_the_measured_rows():
    assert runtime.M64_OWNER_HELPER_MIN_SEQ_LEN == ((36, 1024),)
    assert runtime.M64_OWNER_HELPER_MIN_SEQ_LEN_GB300 == ((36, 1024),)


@pytest.mark.parametrize("arch,sm_count", [("sm_100a", 148), ("sm_103a", 152)])
def test_shipped_rows_take_long_launches_up_to_36_tasks(monkeypatch, arch, sm_count):
    monkeypatch.setattr(runtime, "_m64_cooperative_capacity", lambda device: sm_count)
    count = partial(
        runtime._m64_owner_helper_count,
        gpu_arch=arch,
        sm_count=sm_count,
        device=torch.device("meta"),
    )
    assert count(total_tasks=8, min_seq_len=1024) == 2
    assert count(total_tasks=36, min_seq_len=16384) == 2
    assert count(total_tasks=36, min_seq_len=1023) == 0
    assert count(total_tasks=37, min_seq_len=4096) == 0


class _Raising(contextlib.AbstractContextManager):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("no CUDA device")

    def __exit__(self, *exc):
        return False


def test_capacity_probe_failure_keeps_the_other_programs(monkeypatch):
    warnings = []
    monkeypatch.setattr(runtime, "_M64_COOPERATIVE_CAPACITY", {})
    monkeypatch.setattr(runtime, "_M64_COOPERATIVE_PROBE_WARNED", False)
    monkeypatch.setattr(torch.cuda, "device", _Raising)
    from flashinfer.jit import core

    monkeypatch.setattr(
        core.logger, "warning", lambda *args, **kwargs: warnings.append(args)
    )
    device = torch.device("cuda", 0)
    assert runtime._probe_m64_cooperative_capacity(device) is None
    assert runtime._m64_cooperative_capacity(device) == 0
    assert runtime._m64_cooperative_capacity(device) == 0
    assert len(warnings) == 1, "one warning per process"
    assert runtime._M64_COOPERATIVE_CAPACITY == {}, "a failure is not cached"
    monkeypatch.setattr(runtime, "M64_OWNER_HELPER_MIN_SEQ_LEN", ((16, 1024),))
    assert not runtime._m64_owner_helper_count(
        gpu_arch="sm_100a",
        sm_count=148,
        total_tasks=16,
        min_seq_len=4096,
        device=device,
    )


def test_capacity_caches_answers_per_device(monkeypatch):
    answers = {"a": [None, 120], "b": [140]}
    calls = []

    def probe(device):
        calls.append(device)
        return answers[device].pop(0)

    monkeypatch.setattr(runtime, "_M64_COOPERATIVE_CAPACITY", {})
    monkeypatch.setattr(runtime, "_probe_m64_cooperative_capacity", probe)
    assert runtime._m64_cooperative_capacity("a") == 0
    assert runtime._m64_cooperative_capacity("a") == 120
    assert runtime._m64_cooperative_capacity("a") == 120
    assert runtime._m64_cooperative_capacity("b") == 140
    assert calls == ["a", "a", "b"]


@pytest.mark.parametrize(
    "message",
    [
        str(REFUSED),
        "kernel_x: cuLaunchKernelEx failed with CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE",
        "cudaErrorCooperativeLaunchTooLarge",
    ],
)
def test_cooperative_refusal_is_recognized(message):
    assert runtime._cooperative_launch_refused(RuntimeError(message))


def test_other_launch_errors_are_not_refusals():
    for message in (
        "kernel_x: cudaLaunchKernelExC failed: invalid argument",
        "kernel_x: cudaLaunchKernelExC failed: an illegal memory access",
    ):
        assert not runtime._cooperative_launch_refused(RuntimeError(message))


# ------------------------------------------------------------- the factory


class _Kernel:
    """The loader's ``PreparedKernel`` as a recorder: preparations and launches."""

    schedule = None
    owner_helper = False
    owner_helper_pool = False
    work_bins = False
    ctas_per_head = None
    refuse: dict = {}
    made: list = []

    def __init__(self, name, arch):
        self.name = name
        self.arch = arch
        self._ready = False
        self.prepared = []
        self.launches = []
        _Kernel.made.append(self)

    def prepare(self, *, grid, **bindings):
        self.prepared.append(tuple(grid))
        self._ready = True

    def _arguments(self, grid, bindings):
        return [tuple(grid), dict(bindings)]

    def _call(self, grid, bindings):
        self.launches.append((grid, bindings))
        error = _Kernel.refuse.get(self.name)
        if error is not None:
            raise error


def _stub_tables(monkeypatch, arch, *, pair_share=True, owner_helper=True):
    module = _m64_module(arch)
    monkeypatch.setattr(runtime, "detect_gpu_arch", lambda *args, **kwargs: arch)
    monkeypatch.setattr(registry, "device_arch", lambda *args, **kwargs: arch)
    ps = dict(PAIR_SHARE[arch])
    assert module in ps, "the M64 FP32-state module has a pair-share build"
    if not pair_share:
        ps.pop(module)
    monkeypatch.setattr(registry, "_RENDERED_PAIR_SHARE_ROUTES", {arch: ps})
    monkeypatch.setattr(
        registry,
        "_RENDERED_M64_OWNER_HELPER_ROUTES",
        {arch: {module: OWNER_HELPER} if owner_helper else {}},
    )
    monkeypatch.setattr(registry, "PreparedKernel", _Kernel)
    monkeypatch.setattr(_Kernel, "made", [])
    monkeypatch.setattr(_Kernel, "refuse", {})
    return module


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_factory_takes_the_build_only_beside_a_pair_share_build(monkeypatch, arch):
    module = _stub_tables(monkeypatch, arch)
    assert runtime._m64_owner_helper_program(NAME, **M64_BODY) == OWNER_HELPER
    kernel = runtime._m64_owner_helper_factory(NAME, **M64_BODY)
    assert kernel.name == OWNER_HELPER and kernel.m64_owner_helper
    # Another row of the same factory (BF16 state) has no owner/helper build.
    other = {**M64_BODY, "state_dtype_is_fp32": False}
    if registry._key((), other) in registry.FACTORIES[arch][NAME]:
        assert runtime._m64_owner_helper_program(NAME, **other) is None
        kept = runtime._m64_owner_helper_factory(NAME, **other)
        assert not getattr(kept, "m64_owner_helper", False)
    # Without the module's pair-share build the table entry is never taken.
    _stub_tables(monkeypatch, arch, pair_share=False)
    assert runtime._m64_owner_helper_program(NAME, **M64_BODY) is None
    kept = runtime._m64_owner_helper_factory(NAME, **M64_BODY)
    assert kept.name == registry._RENDERED_ROUTES[arch].get(module, module)
    assert not getattr(kept, "m64_owner_helper", False)
    _stub_tables(monkeypatch, arch, owner_helper=False)
    assert runtime._m64_owner_helper_program(NAME, **M64_BODY) is None


# ------------------------------------------------------------- the plan


class _NoDevice(contextlib.AbstractContextManager):
    def __init__(self, *args, **kwargs):
        pass

    def __exit__(self, *exc):
        return False


@pytest.fixture
def planner(monkeypatch):
    """Meta-device planning on a stubbed part (see test_kda_prefill_route_exports)."""
    part = SimpleNamespace(
        arch="sm_100a", capability=(10, 0), sm_count=148, capacity=148, clusters=74
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: part.capability
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=part.sm_count),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", _NoDevice)

    def data_ptr(tensor):
        try:
            base = tensor.untyped_storage().data_ptr()
        except Exception:  # noqa: BLE001
            base = 0
        return base + tensor.storage_offset() * tensor.element_size()

    monkeypatch.setattr(torch.Tensor, "data_ptr", data_ptr)

    def require_tensor(tensor, *, name, dtype, ndim, contiguous=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if tensor.ndim != ndim:
            raise ValueError(f"{name} must have rank {ndim}")
        if contiguous and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def upload_int_batch(device, host_lists, dtype):
        uploads = {"_host_pinned": None}
        for name, values in host_lists.items():
            uploads[name] = torch.empty(len(values), dtype=dtype, device=device)
        return uploads

    monkeypatch.setattr(runtime, "_require_tensor", require_tensor)
    monkeypatch.setattr(runtime, "_upload_int_batch", upload_int_batch)
    monkeypatch.setattr(runtime, "_device_sm_count", lambda device: part.sm_count)
    monkeypatch.setattr(runtime, "_ffi_arguments", tuple)
    monkeypatch.setattr(
        runtime, "_m64_cooperative_capacity", lambda device: part.capacity
    )
    monkeypatch.setattr(
        runtime, "_m64_pair_cluster_capacity", lambda device: part.clusters
    )
    monkeypatch.setattr(runtime, "_clear_cuda_launch_error", lambda: None)
    for table in RULES.values():
        monkeypatch.setattr(runtime, table, ())

    def configure(arch, rows=(), **values):
        for name, value in values.items():
            setattr(part, name, value)
        part.arch = arch
        part.capability = dict((a, c) for a, c, _ in PARTS)[arch]
        part.sm_count = values.get("sm_count", dict((a, n) for a, _, n in PARTS)[arch])
        monkeypatch.setattr(
            runtime,
            "M64_OWNER_HELPER_MIN_SEQ_LEN_GB300"
            if arch == "sm_103a" and part.sm_count == 152
            else "M64_OWNER_HELPER_MIN_SEQ_LEN",
            rows,
        )
        return _stub_tables(monkeypatch, arch)

    part.configure = configure
    return part


def _plan(lengths, heads, *, rows=True):
    """One M64 FP32-state value-split plan on meta tensors: dense states,
    bounded gate, BF16 checkpoint rows every 64 tokens unless ``rows=False``."""
    dev = "meta"
    tokens = sum(lengths)
    shape = (1, tokens, heads, HEAD_DIM)
    q, k, v, g = (
        torch.empty(shape, dtype=torch.bfloat16, device=dev) for _ in range(4)
    )
    state = torch.empty(
        (len(lengths), heads, HEAD_DIM, HEAD_DIM), dtype=torch.float32, device=dev
    )
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + (n + 63) // 64)
    kwargs = dict(
        initial_state=state,
        final_state=state,
        cu_seqlens=torch.empty((len(lengths) + 1,), dtype=torch.int64, device=dev),
        sequence_lengths=tuple(lengths),
    )
    if rows:
        kwargs.update(
            state_checkpoints=torch.empty(
                (starts[-1], heads, HEAD_DIM, HEAD_DIM),
                dtype=torch.bfloat16,
                device=dev,
            ),
            checkpoint_cu_starts=torch.empty(
                (len(lengths) + 1,), dtype=torch.int64, device=dev
            ),
            checkpoint_every_n_tokens=64,
        )
    return runtime.FlashKDABlackwellFP32M64Launch(
        q,
        k,
        v,
        g,
        torch.empty((1, tokens, heads), dtype=torch.bfloat16, device=dev),
        HEAD_DIM**-0.5,
        torch.empty(shape, dtype=torch.bfloat16, device=dev),
        torch.empty((heads,), dtype=torch.float32, device=dev),
        torch.empty((heads, HEAD_DIM), dtype=torch.float32, device=dev),
        -5.0,
        **kwargs,
    )


BASE = "fused_m64_independent_dvsplit_fp32_state"


@pytest.mark.parametrize("rows", [True, False], ids=["checkpoints", "no-checkpoints"])
@pytest.mark.parametrize(("arch", "capability", "sm_count"), PARTS, ids=PART_IDS)
def test_rule_takes_the_build_with_its_grid_ring_and_words(
    planner, arch, capability, sm_count, rows
):
    planner.configure(arch, rows=((16, 2048),))
    plan = _plan([2048, 4096], 8, rows=rows)
    assert plan.m64_owner_helpers == 2
    assert plan.module.name == OWNER_HELPER
    assert plan.schedule == BASE + "_owner2helper"
    assert plan.grid == (16, 4, 1)
    ring, words = plan.args["final_state"], plan.args["seq_order"]
    assert ring.dtype == torch.bfloat16 and ring.shape == (16 * 2 * 5 * 16384,)
    assert words.dtype == torch.int32 and words.shape == (2 * 3 * 5 * 16,)
    # The program a refused launch takes: the pair-share build here (16 tasks,
    # 74 clusters, 4096 tokens), with the plan's own BF16-state and seq_order
    # bindings and its cluster grid.
    module, grid, schedule, bindings = plan._m64_owner_helper_fallback
    assert module.name == registry._RENDERED_PAIR_SHARE_ROUTES[arch][_m64_module(arch)]
    assert grid == (32, 1, 1) and schedule == BASE + "_pair_share"
    assert set(bindings) == {"final_state", "seq_order"}
    assert bindings["final_state"] is not ring and bindings["seq_order"] is not words
    assert bindings["final_state"].dtype == torch.bfloat16
    assert bindings["seq_order"].dtype == torch.int32
    # FP32 states stay in their own slots.
    assert plan.args["final_state_f32"].dtype == torch.float32


@pytest.mark.parametrize(("arch", "capability", "sm_count"), PARTS, ids=PART_IDS)
def test_rule_keeps_the_pair_share_or_routed_program(
    planner, arch, capability, sm_count
):
    # Empty tables (as shipped): the launch keeps the pair-share build.
    planner.configure(arch)
    plan = _plan([2048, 4096], 8)
    assert plan.m64_owner_helpers == 0 and plan.schedule == BASE + "_pair_share"
    assert plan.grid == (32, 1, 1)
    assert not any(kernel.name == OWNER_HELPER for kernel in _Kernel.made)
    assert not hasattr(plan, "_m64_owner_helper_fallback")
    # A sequence below the row, more tasks than the row holds, or a device that
    # holds one CTA too few keep it as well.
    for lengths, heads, capacity in (
        ([2047, 4096], 8, 148),
        ([2048, 4096], 9, 148),
        ([2048, 4096], 8, 63),
    ):
        planner.configure(arch, rows=((16, 2048),), capacity=capacity)
        plan = _plan(lengths, heads)
        assert plan.m64_owner_helpers == 0, (lengths, heads, capacity)
        assert OWNER_HELPER not in {kernel.name for kernel in _Kernel.made}
    planner.configure(arch, rows=((16, 2048),), capacity=64)
    assert _plan([2048, 4096], 8).m64_owner_helpers == 2
    # A launch the pair-share rule leaves alone falls back to the routed program.
    planner.configure(arch, rows=((16, 512),))
    plan = _plan([512, 512], 8)
    assert plan.schedule == BASE + "_owner2helper"
    module, grid, schedule, _ = plan._m64_owner_helper_fallback
    assert module.name == registry._RENDERED_ROUTES[arch][_m64_module(arch)]
    assert (grid, schedule) == ((32, 1, 1), BASE)


def test_modules_without_a_pair_share_build_never_take_the_build(planner, monkeypatch):
    module = planner.configure("sm_100a", rows=((16, 1024),))
    ps = dict(registry._RENDERED_PAIR_SHARE_ROUTES["sm_100a"])
    ps.pop(module)
    monkeypatch.setattr(registry, "_RENDERED_PAIR_SHARE_ROUTES", {"sm_100a": ps})
    plan = _plan([2048, 4096], 8)
    assert plan.m64_owner_helpers == 0 and plan.schedule == BASE


# ------------------------------------------------------------- the launch


def _owner_helper_plan(planner, arch="sm_100a"):
    planner.configure(arch, rows=((16, 2048),))
    plan = _plan([2048, 4096], 8)
    plan.module.prepare(grid=plan.grid, **plan.args)
    oh = plan.module
    ps = plan._m64_owner_helper_fallback[0]
    return plan, oh, ps


def test_launch_runs_the_build(planner):
    plan, oh, ps = _owner_helper_plan(planner)
    plan._launch_in_stream()
    plan._launch_in_stream()
    assert [grid for grid, _ in oh.launches] == [(16, 4, 1)] * 2
    assert oh.launches[0][1]["final_state"] is plan.args["final_state"]
    assert not ps.launches and not ps.prepared


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_refused_launch_switches_the_plan_to_its_pair_share_program(
    planner, monkeypatch, arch
):
    cleared = []
    monkeypatch.setattr(runtime, "_clear_cuda_launch_error", lambda: cleared.append(1))
    plan, oh, ps = _owner_helper_plan(planner, arch)
    ring, words = plan.args["final_state"], plan.args["seq_order"]
    bindings = plan._m64_owner_helper_fallback[3]
    held = plan._m64_owner_helper_record()  # e.g. a composite's stored record
    _Kernel.refuse[OWNER_HELPER] = REFUSED
    plan._launch_in_stream()
    assert len(oh.launches) == 1, "the refused grid is tried once"
    assert cleared == [1]
    assert plan.m64_owner_helpers == 0
    assert plan.module is ps and plan.grid == (32, 1, 1)
    assert plan.schedule == BASE + "_pair_share"
    assert ps.prepared == [(32, 1, 1)]
    assert [grid for grid, _ in ps.launches] == [(32, 1, 1)]
    launched = ps.launches[0][1]
    assert launched["final_state"] is bindings["final_state"]
    assert launched["seq_order"] is bindings["seq_order"]
    assert launched["final_state"] is not ring and launched["seq_order"] is not words
    # For good: later launches, and a record built before the switch, run the
    # pair-share program and never retry the refused grid.
    _Kernel.refuse.clear()
    plan._launch_in_stream()
    held[0](*held[1])
    assert len(oh.launches) == 1
    assert [grid for grid, _ in ps.launches] == [(32, 1, 1)] * 3
    assert ps.prepared == [(32, 1, 1)] and cleared == [1]


def test_other_launch_errors_propagate_and_keep_the_plan(planner):
    plan, oh, ps = _owner_helper_plan(planner)
    _Kernel.refuse[OWNER_HELPER] = RuntimeError(
        "kernel_x: cudaLaunchKernelExC failed: an illegal memory access"
    )
    with pytest.raises(RuntimeError, match="illegal memory access"):
        plan._launch_in_stream()
    assert plan.m64_owner_helpers == 2 and plan.module is oh
    assert plan.schedule == BASE + "_owner2helper"
    assert not ps.launches and not ps.prepared


def test_fallback_preserves_rebound_bindings(planner):
    """A plan-cache rebind re-points the caller views of ``args`` in place;
    the fallback swaps only the ring and the words back."""
    plan, oh, ps = _owner_helper_plan(planner)
    moved = torch.empty_like(plan.args["out"])
    plan.args["out"] = moved
    _Kernel.refuse[OWNER_HELPER] = REFUSED
    plan._launch_in_stream()
    assert ps.launches[0][1]["out"] is moved


def test_build_body_passes_the_owner_helper_factory_through(planner, monkeypatch):
    """``_build_body``'s turnover rewrite only touches ``_factory`` builds."""
    planner.configure("sm_100a")
    body = object.__new__(runtime.FlashKDABlackwellBF16FusedLaunch)
    body._qkv_carrier_names = {}
    built = []
    monkeypatch.setattr(
        runtime, "_build_kda_module", lambda factory, *a, **k: built.append(factory)
    )
    factory = partial(runtime._m64_owner_helper_factory, NAME)
    body._build_body(factory, work_bin_turnover=True, **M64_BODY)
    assert built == [factory]
