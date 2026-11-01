"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""Host side of the work-bin bin-mix build (CPU; no GPU needed).

A work-binned launch that takes its module's turnover build
(``_RENDERED_WORK_BIN_TURN_ROUTES``) takes the module's bin-mix build
(``_RENDERED_WORK_BIN_MIX_ROUTES``) instead where the bin-mix rule admits its
bin table (``_work_bin_mix_plan``) on an architecture of
``WORK_BIN_MIX_ARCHES``: the plan words follow the table in ``seq_order`` and
a zeroed give ring rides in ``chunk_state`` / ``zero_workspace``.  On an
architecture of ``WORK_BIN_MIX_SPAN_ARCHES`` the plan must also shorten the
launch's longest bin and give no single chunk to a helper with a longer task
than the owner's shorter task.  Every other launch, and every launch on
another architecture, keeps the turnover build unchanged.  A launch the driver refuses as a too large cooperative grid
switches the plan for good to the turnover build with its own bindings.

The device queries are stubbed (SM100a / SM103a, 148 or 152 SMs), the tensors
live on the meta device and the loader's ``PreparedKernel`` is a recording
stub, so the rule, the routing, the workspace and the fallback run without a
kernel.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

import flashinfer.cake_kda_tf32_runtime as runtime
import flashinfer.jit.cake_kda_tf32 as registry
from flashinfer import KDAPrefillPlanCache, prepare_bf16_kda_prefill

pytestmark = pytest.mark.cpu_only

HEAD_DIM = 128
CHECKPOINT_EVERY = 64
MAGIC = 0x58313221
BIN_MIX = "cake_kda_prefill_000000000000000b1a11"
REFUSED = RuntimeError(
    f"kernel_{BIN_MIX}: cudaLaunchKernelExC failed: too many blocks in cooperative launch"
)
PARTS = (("sm_100a", (10, 0), 148), ("sm_103a", (10, 3), 152))
PART_IDS = ["sm100a-148", "sm103a-152"]
# The parts whose admitted launches take the bin-mix build.
MIX_PARTS = PARTS
MIX_PART_IDS = PART_IDS
# Launches the rule admits on a 152-SM sm_103a part whose plan fails the span
# check there; each ran slower on that part with the bin-mix build than with
# the turnover build: (lengths, heads).  The first two give a single chunk to a
# helper with a longer task than the owner's shorter task; the others leave the
# longest bin as long as before.
SM103A_REFUSED_ROWS = [
    pytest.param(
        [
            111,
            74,
            96,
            109,
            77,
            11,
            18,
            92,
            98,
            105,
            72,
            104,
            15,
            70,
            95,
            127,
            79,
            22,
            2,
        ],
        12,
        id="short-mixed-19-h12",
    ),
    pytest.param([88] * 9 + [107] * 3, 16, id="12x88-107-h16"),
    pytest.param(
        [116, 76, 76, 116, 116, 116] + [76] * 6 + [116] + [76] * 4,
        10,
        id="5x116-12x76-h10",
    ),
    pytest.param([88] * 12 + [116] + [88] * 10, 8, id="1x116-22x88-h8"),
    pytest.param([256] * 3 + [128] * 14, 12, id="3x256-14x128-h12"),
    pytest.param(
        [110, 107, 106, 229, 107, 101, 103, 99, 102, 97, 230, 97, 101, 229, 104, 104],
        12,
        id="probe-16-h12",
    ),
    pytest.param(
        [114, 104, 110, 152, 142, 114, 252, 160, 147, 120, 149], 16, id="probe-11-h16"
    ),
    pytest.param(
        [176, 182, 243, 80, 91, 100, 100, 136, 187, 235, 202, 83, 150, 120],
        16,
        id="random-14-h16",
    ),
    pytest.param(
        [106, 100, 103, 108, 107, 227, 105, 240, 105, 256, 108, 100, 102, 240],
        16,
        id="probe-14-h16",
    ),
    pytest.param([200, 200] + [103] * 8 + [200] + [103] * 2, 16, id="3x200-10x103-h16"),
]
# The span check refuses these too, and the GPU tests' mixed bin-mix row
# (tests/kda BIN_MIX_ROWS), whose plan gives single chunks to helpers with
# longer tasks than the owner's shorter task.
SM103A_SPAN_REFUSED_ROWS = SM103A_REFUSED_ROWS + [
    pytest.param([256, 224, 192, 160] * 4, 12, id="mixed16-h12"),
]
# The loader's turnover tables, before any test stubs them.
TURN = {
    arch: dict(routes)
    for arch, routes in registry._RENDERED_WORK_BIN_TURN_ROUTES.items()
}


def _table(lengths, heads, sm_count, task_cost=0):
    return runtime._make_direct_work_bins(
        tuple(lengths),
        num_heads=heads,
        worker_count=sm_count,
        chunk_tokens=runtime.BF16_M128_CHUNK,
        task_cost=task_cost,
    )


def _plan_words(lengths, heads, sm_count, task_cost=0):
    table, bins = _table(lengths, heads, sm_count, task_cost)
    return table, bins, runtime._work_bin_mix_plan(table, bins, tuple(lengths), heads)


def _bins(table, bins):
    return [table[table[index] : table[index + 1]] for index in range(bins)]


# ------------------------------------------------------------- the rule


def test_rule_constants():
    assert runtime.WORK_BIN_MIX_ARCHES == ("sm_100a", "sm_103a")
    assert runtime.WORK_BIN_MIX_SPAN_ARCHES == ("sm_103a",)
    assert runtime.WORK_BIN_MIX_MAGIC == MAGIC
    assert runtime.WORK_BIN_MIX_MAX_GIVE == 8
    assert runtime.WORK_BIN_MIX_SLOTS == 8
    assert runtime.WORK_BIN_MIX_SLOT_ELEMENTS == 16384


# 16 sequences x 12 heads = 192 tasks: two-task bins of 2 x chunks beside
# one-task bins of chunks (44 + 104 bins on 148 SMs, 40 + 112 on 152).
@pytest.mark.parametrize(
    ("tokens", "give"),
    [(32, None), (64, 1), (96, 1), (128, 2), (192, 3), (256, 5)],
    ids=["T32", "T64", "T96", "T128", "T192", "T256"],
)
@pytest.mark.parametrize(("arch", "capability", "sm_count"), PARTS, ids=PART_IDS)
def test_uniform_rows_admit_from_two_chunks(arch, capability, sm_count, tokens, give):
    table, bins, words = _plan_words([tokens] * 16, 12, sm_count)
    owners = [i for i, tasks in enumerate(_bins(table, bins)) if len(tasks) == 2]
    assert bins == sm_count
    assert len(owners) == 192 - sm_count
    if give is None:
        assert words is None, "a one-chunk difference keeps the turnover build"
        return
    chunks = (tokens + 31) // 32
    assert len(words) == 4 * bins + 1 and words[-1] == MAGIC
    helpers = [i for i, tasks in enumerate(_bins(table, bins)) if len(tasks) == 1]
    for owner, helper in zip(owners, helpers, strict=False):
        keep = 2 * chunks - give
        assert words[4 * owner : 4 * owner + 4] == (give, keep, 0, 0)
        assert words[4 * helper : 4 * helper + 4] == (-give, owner, chunks, keep)
    for helper in helpers[len(owners) :]:
        assert words[4 * helper : 4 * helper + 4] == (0, 0, 0, 0)


def test_give_formula_and_cap():
    # Owner loads 2 x c beside a helper of c: d = c, give = min(8, c//2 + c//8, 2c - 1).
    for chunks in range(1, 40):
        tokens = 32 * chunks
        _, _, words = _plan_words([tokens] * 16, 12, 148)
        expected = min(8, chunks // 2 + chunks // 8, 2 * chunks - 1)
        if expected < 1:
            assert words is None, chunks
        else:
            gives = {w for w in words[0:-1:4] if w > 0}
            assert gives == {expected}, chunks
    _, _, words = _plan_words([32 * 64] * 16, 12, 148)
    assert max(words[0:-1:4]) == 8


@pytest.mark.parametrize("sm_count", [148, 152])
def test_mixed_lengths_pair_in_order(sm_count):
    # 8, 7, 6 and 5 chunks: the shortest tasks double up (owners of 10 or 11
    # chunks) and pair in order with the longest one-task bins.
    lengths = [256, 224, 192, 160] * 4
    table, bins, words = _plan_words(lengths, 12, sm_count)
    assert words is not None and words[-1] == MAGIC
    chunks = [(n + 31) // 32 for n in lengths]
    tasks = _bins(table, bins)
    loads = [sum(chunks[t // 12] for t in ts) for ts in tasks]
    owners = [i for i, ts in enumerate(tasks) if len(ts) == 2]
    helpers = [i for i, ts in enumerate(tasks) if len(ts) == 1]
    for owner, helper in zip(owners, helpers, strict=False):
        d = loads[owner] - loads[helper]
        give = min(8, d // 2 + d // 8, loads[owner] - 1)
        if give < 1:
            assert words[4 * owner : 4 * owner + 4] == (0, 0, 0, 0)
            assert words[4 * helper : 4 * helper + 4] == (0, 0, 0, 0)
            continue
        assert words[4 * owner : 4 * owner + 4] == (give, loads[owner] - give, 0, 0)
        assert words[4 * helper : 4 * helper + 4] == (
            -give,
            owner,
            loads[helper],
            loads[owner] - give,
        )


@pytest.mark.parametrize(
    ("lengths", "heads", "sm_count"),
    [
        ([256] * 16, 12, 64),  # three-task bins
        ([256] * 16, 12, 96),  # every bin holds two tasks: no helpers
        ([256] * 20, 12, 148),  # 240 tasks: 92 owners beside 56 one-task bins
        ([256] * 8, 12, 148),  # one task per bin: no owners
        ([256, 200, 129, 64] * 4, 12, 148),  # 2-chunk tasks take a third slot
        ([256, 64] * 8, 12, 148),  # owners of 4 chunks pair with helpers of 8
    ],
    ids=[
        "three-task-bins",
        "two-task-bins-only",
        "fewer-helpers",
        "no-owners",
        "mixed-three-task-bins",
        "owners-below-helpers",
    ],
)
def test_tables_outside_the_rule_keep_the_turnover_build(lengths, heads, sm_count):
    assert _plan_words(lengths, heads, sm_count)[2] is None


def test_plan_is_cached_and_checks_the_table_length():
    table, bins = _table([256] * 16, 12, 148)
    first = runtime._work_bin_mix_plan(table, bins, (256,) * 16, 12)
    assert runtime._work_bin_mix_plan(table, bins, (256,) * 16, 12) is first
    assert runtime._work_bin_mix_plan(table[:-1], bins, (256,) * 16, 12) is None
    assert runtime._work_bin_mix_plan(table, bins, (256,) * 15, 12) is None


def test_turnover_task_cost_table():
    # The turnover-bins table (task cost charged) of the same rows bins alike.
    for tokens in (64, 128, 256):
        cost = runtime.BF16_WORK_BIN_TURNOVER_TASK_COST
        strict = _plan_words([tokens] * 16, 12, 148)
        charged = _plan_words([tokens] * 16, 12, 148, task_cost=cost)
        assert (strict[2] is None) == (charged[2] is None)


# ------------------------------------------------------------- the program


def _turnover_pair(arch):
    module, program = next(iter(TURN[arch].items()))
    return module, program


def test_program_lookup(monkeypatch):
    arch = "sm_100a"
    module, program = _turnover_pair(arch)
    monkeypatch.setattr(registry, "_RENDERED_WORK_BIN_MIX_ROUTES", {arch: {}})
    assert runtime._work_bin_mix_program(program, arch) is None
    monkeypatch.setattr(
        registry, "_RENDERED_WORK_BIN_MIX_ROUTES", {arch: {module: BIN_MIX}}
    )
    assert runtime._work_bin_mix_program(program, arch) == BIN_MIX
    assert runtime._work_bin_mix_program(module, arch) is None
    assert runtime._work_bin_mix_program(None, arch) is None
    assert runtime._work_bin_mix_program(program, "sm_103a") is None


def test_program_lookup_on_sm103a(monkeypatch):
    module, program = _turnover_pair("sm_103a")
    monkeypatch.setattr(
        registry,
        "_RENDERED_WORK_BIN_MIX_ROUTES",
        {"sm_100a": {}, "sm_103a": {module: BIN_MIX}},
    )
    assert runtime._work_bin_mix_program(program, "sm_103a") == BIN_MIX
    assert runtime._work_bin_mix_program(program, "sm_100a") is None


def test_shipped_loader_names_the_unbounded_direct_bin_mix_build():
    # One bin-mix build on each architecture of WORK_BIN_MIX_ARCHES, beside
    # the turnover build of the same module, registered like every generated
    # program; none on another architecture.
    routes = registry._RENDERED_WORK_BIN_MIX_ROUTES
    assert set(routes) == {"sm_100a", "sm_103a"}
    for arch, by_module in routes.items():
        if arch not in runtime.WORK_BIN_MIX_ARCHES:
            assert by_module == {}, arch
            continue
        assert by_module, arch
        for module, program in by_module.items():
            assert module in TURN[arch], (arch, module)
            assert program != TURN[arch][module]
            assert registry._RENDERED_MODULES[program]["arches"] == list(
                runtime.WORK_BIN_MIX_ARCHES
            )
            assert runtime._work_bin_mix_program(TURN[arch][module], arch) == program


# ------------------------------------------------------------- the plan


class _Kernel:
    """The loader's ``PreparedKernel`` as a recorder: preparations and launches."""

    schedule = None
    owner_helper = False
    owner_helper_pool = False
    ctas_per_head = None
    encodes_on_host = False
    refuse: dict = {}
    made: list = []

    def __init__(self, name, arch):
        self.name = name
        self.arch = arch
        self.work_bins = (
            name in registry._RENDERED_WORK_BIN_ROUTES.get(arch, {}).values()
            or name in registry._RENDERED_WORK_BIN_TURN_ROUTES.get(arch, {}).values()
            or name in registry._RENDERED_WORK_BIN_MIX_ROUTES.get(arch, {}).values()
        )
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


class _NoDevice(contextlib.AbstractContextManager):
    def __init__(self, *args, **kwargs):
        pass

    def __exit__(self, *exc):
        return False


@pytest.fixture
def planner(monkeypatch):
    """Meta-device planning on a stubbed part (see test_kda_prefill_route_exports)."""
    part = SimpleNamespace(arch="sm_100a", capability=(10, 0), sm_count=148)
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
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

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

    uploads = []

    def upload_int_batch(device, host_lists, dtype):
        out = {"_host_pinned": None}
        for name, values in host_lists.items():
            uploads.append((name, list(values)))
            out[name] = torch.empty(len(values), dtype=dtype, device=device)
        return out

    monkeypatch.setattr(runtime, "_require_tensor", require_tensor)
    monkeypatch.setattr(runtime, "_upload_int_batch", upload_int_batch)
    monkeypatch.setattr(runtime, "_device_sm_count", lambda device: part.sm_count)
    monkeypatch.setattr(runtime, "_fused_affine_epilogue_enabled", lambda: False)
    monkeypatch.setattr(runtime, "_clear_cuda_launch_error", lambda: None)
    monkeypatch.setattr(runtime, "_ffi_arguments", tuple)
    monkeypatch.setattr(registry, "prepare_descriptors", lambda prepared: None)
    monkeypatch.setattr(registry, "PreparedKernel", _Kernel)
    monkeypatch.setattr(_Kernel, "made", [])
    monkeypatch.setattr(_Kernel, "refuse", {})
    part.uploads = uploads

    def configure(arch, *, mix=True):
        part.arch = arch
        part.capability = dict((a, c) for a, c, _ in PARTS)[arch]
        part.sm_count = dict((a, n) for a, _, n in PARTS)[arch]
        monkeypatch.setattr(runtime, "detect_gpu_arch", lambda *args, **kwargs: arch)
        monkeypatch.setattr(registry, "device_arch", lambda *args, **kwargs: arch)
        routes = {module: BIN_MIX for module in TURN[arch]} if mix else {}
        monkeypatch.setattr(
            registry,
            "_RENDERED_WORK_BIN_MIX_ROUTES",
            {"sm_100a": {}, "sm_103a": {}, arch: routes},
        )
        uploads.clear()
        _Kernel.made.clear()

    part.configure = configure
    return part


def _prepare(lengths, heads, *, plan_cache=None):
    """A bench_tb wrapper row on meta tensors: packed sequences, unbounded gate,
    indexed FP32 pool, FP32 checkpoint rows every 64 tokens, strided beta."""
    dev = "meta"
    total = sum(lengths)
    shape = (1, total, heads, HEAD_DIM)
    q, k, v, g = (
        torch.empty(shape, dtype=torch.bfloat16, device=dev) for _ in range(4)
    )
    n = len(lengths)
    pool = torch.empty(
        (n + 5, heads, HEAD_DIM, HEAD_DIM), dtype=torch.float32, device=dev
    )
    rows = sum((t + CHECKPOINT_EVERY - 1) // CHECKPOINT_EVERY for t in lengths)
    beta = torch.empty((1, total, heads + 28), dtype=torch.bfloat16, device=dev)
    return prepare_bf16_kda_prefill(
        q,
        k,
        v,
        g,
        beta[:, :, 8 : 8 + heads],
        A_log=torch.empty((heads,), dtype=torch.float32, device=dev),
        dt_bias=torch.empty((heads, HEAD_DIM), dtype=torch.float32, device=dev),
        out=torch.empty(shape, dtype=torch.bfloat16, device=dev),
        initial_state=pool,
        final_state=pool,
        lower_bound=None,
        cu_seqlens=torch.empty((n + 1,), dtype=torch.int64, device=dev),
        sequence_lengths=tuple(lengths),
        state_indices=torch.empty((n,), dtype=torch.int32, device=dev),
        state_checkpoints=torch.empty(
            (rows, heads, HEAD_DIM, HEAD_DIM), dtype=torch.float32, device=dev
        ),
        checkpoint_cu_starts=torch.empty((n + 1,), dtype=torch.int64, device=dev),
        checkpoint_every_n_tokens=CHECKPOINT_EVERY,
        plan_cache=plan_cache,
    )


def _launch(prepared):
    """The launch object a prepared call runs (the prepared object itself, or its body)."""
    candidates = [prepared, *getattr(prepared, "__dict__", {}).values()]
    for obj in candidates:
        if isinstance(obj, runtime.FlashKDABlackwellBF16FusedLaunch):
            return obj
    raise AssertionError(f"no fused launch in {type(prepared)}")


def _seq_order_upload(part, bins):
    """The work-bin table upload (its first word is the table's first offset)."""
    uploads = [
        values
        for name, values in part.uploads
        if name == "seq_order" and values and values[0] == bins + 1
    ]
    assert len(uploads) == 1, part.uploads
    return uploads[0]


@pytest.mark.parametrize("tokens", [64, 128, 256], ids=["T64", "T128", "T256"])
@pytest.mark.parametrize(
    ("arch", "capability", "sm_count"), MIX_PARTS, ids=MIX_PART_IDS
)
def test_admitted_launch_takes_the_build_with_its_plan_and_ring(
    planner, arch, capability, sm_count, tokens
):
    planner.configure(arch)
    plan = _launch(_prepare([tokens] * 16, 12))
    assert plan.work_bin_mix, plan.schedule
    assert plan.module.name == BIN_MIX
    assert plan.schedule.endswith("_turnover_bin_mix"), plan.schedule
    assert plan.grid == (sm_count, 1, 1)
    table, bins = _table([tokens] * 16, 12, sm_count)
    words = runtime._work_bin_mix_plan(table, bins, (tokens,) * 16, 12)
    assert _seq_order_upload(planner, bins) == list(table) + list(words)
    assert plan.args["seq_order"].shape == (len(table) + 4 * bins + 1,)
    ring, ready = plan.args["chunk_state"], plan.args["zero_workspace"]
    assert ring.dtype == torch.bfloat16 and ring.shape == (bins * 8 * 16384,)
    assert ready.dtype == torch.uint32 and ready.shape == (bins * 8,)
    module, grid, schedule, bindings = plan._work_bin_mix_fallback
    assert module.name in TURN[arch].values()
    assert grid == (sm_count, 1, 1)
    assert schedule == plan.schedule.removesuffix("_bin_mix")
    assert set(bindings) == {"chunk_state", "zero_workspace", "seq_order"}
    assert bindings["chunk_state"] is not ring
    assert bindings["zero_workspace"] is not ready
    assert bindings["seq_order"].shape == (len(table),)
    assert plan.args["num_sequences"] == 16


@pytest.mark.parametrize(("arch", "capability", "sm_count"), PARTS, ids=PART_IDS)
def test_other_launches_keep_the_turnover_build_unchanged(
    planner, arch, capability, sm_count
):
    # T32 (one-chunk difference), and every admitted row while the loader
    # names no bin-mix build: the same module, grid, table and bindings.
    for tokens, mix in ((32, True), (64, False), (256, False)):
        planner.configure(arch, mix=mix)
        plan = _launch(_prepare([tokens] * 16, 12))
        assert not plan.work_bin_mix, (tokens, mix)
        assert BIN_MIX not in {kernel.name for kernel in _Kernel.made}
        assert not hasattr(plan, "_work_bin_mix_fallback")
        if plan.module.name in TURN[arch].values():
            table, bins = _table([tokens] * 16, 12, sm_count)
            assert _seq_order_upload(planner, bins) == list(table)
            assert plan.schedule.endswith("_turnover")


def _launch_table(part, lengths, heads):
    """The work-bin table a launch uploaded (its first ``bins + 1`` words are
    the bin offsets, the last of them the table's length) and the words that
    follow it."""
    tasks = len(lengths) * heads
    uploads = [
        values
        for name, values in part.uploads
        if name == "seq_order"
        and values
        and len(values) >= values[0] + tasks
        and values[values[0] - 1] == values[0] + tasks
    ]
    assert len(uploads) == 1, part.uploads
    upload = uploads[0]
    size = upload[0] + tasks
    return tuple(upload[:size]), upload[0] - 1, tuple(upload[size:])


@pytest.mark.parametrize("named", [True, False], ids=["mix-named", "mix-unnamed"])
@pytest.mark.parametrize("lengths,heads", SM103A_SPAN_REFUSED_ROWS)
def test_sm103a_span_check_keeps_the_turnover_build(planner, lengths, heads, named):
    # The rule admits the launch's own table on a 152-SM part, yet its plan
    # fails the span check of an sm_103a launch: the launch keeps the
    # turnover build, its table alone and its bindings, whether or not the
    # loader names a bin-mix build there.
    planner.configure("sm_103a", mix=named)
    plan = _launch(_prepare(lengths, heads))
    table, bins, after = _launch_table(planner, lengths, heads)
    assert runtime._work_bin_mix_plan(table, bins, tuple(lengths), heads) is not None
    assert runtime._work_bin_mix_plan(table, bins, tuple(lengths), heads, True) is None
    assert after == ()
    assert not plan.work_bin_mix, plan.schedule
    assert BIN_MIX not in {kernel.name for kernel in _Kernel.made}
    assert not hasattr(plan, "_work_bin_mix_fallback")
    assert plan.module.name in TURN["sm_103a"].values(), plan.schedule
    assert plan.schedule.endswith("_turnover"), plan.schedule
    assert plan.grid == (bins, 1, 1)


@pytest.mark.parametrize(
    "lengths,heads",
    [
        pytest.param([64] * 16, 12, id="bs16x64-h12"),
        pytest.param([128] * 16, 12, id="bs16x128-h12"),
        pytest.param([256] * 16, 12, id="bs16x256-h12"),
        pytest.param([170] * 20, 8, id="bs20x170-h8"),
        pytest.param([144] * 18, 10, id="bs18x144-h10"),
        pytest.param([270] * 14, 16, id="bs14x270-h16"),
    ],
)
def test_sm103a_span_check_keeps_shortening_plans(planner, lengths, heads):
    # A plan that shortens the longest bin without a single-chunk give to a
    # longer helper passes the span check unchanged: the sm_103a launch takes
    # the bin-mix build with the rule's plan words after its table.
    planner.configure("sm_103a")
    plan = _launch(_prepare(lengths, heads))
    table, bins, after = _launch_table(planner, lengths, heads)
    words = runtime._work_bin_mix_plan(table, bins, tuple(lengths), heads)
    assert words is not None
    assert runtime._work_bin_mix_plan(table, bins, tuple(lengths), heads, True) == words
    assert after == words
    assert plan.work_bin_mix, plan.schedule
    assert plan.module.name == BIN_MIX
    assert plan.schedule.endswith("_turnover_bin_mix"), plan.schedule


def test_sm103a_span_check_keeps_the_refused_grid_plan(planner):
    # The GPU tests' refused bin-mix launch (32 x 128 tokens at 12 heads on a
    # table built for twice the part's SMs) shortens its longest bin: on
    # sm_103a it takes the bin-mix build too.
    planner.configure("sm_103a")
    planner.sm_count *= 2
    plan = _launch(_prepare([128] * 32, 12))
    assert plan.work_bin_mix, plan.schedule
    assert plan.module.name == BIN_MIX
    assert plan.schedule.endswith("_turnover_bin_mix"), plan.schedule


@pytest.mark.parametrize("lengths,heads", SM103A_SPAN_REFUSED_ROWS)
def test_sm100a_keeps_the_rule_without_the_span_check(planner, lengths, heads):
    # On a 148-SM sm_100a part the same launches follow the rule alone: the
    # plan words after the table are the rule's, the bin-mix build runs
    # exactly where the rule admits the part's own table.
    planner.configure("sm_100a")
    plan = _launch(_prepare(lengths, heads))
    table, bins, after = _launch_table(planner, lengths, heads)
    words = runtime._work_bin_mix_plan(table, bins, tuple(lengths), heads)
    assert after == (words or ())
    assert plan.work_bin_mix == (words is not None), plan.schedule
    if words is not None:
        assert plan.module.name == BIN_MIX
        assert plan.schedule.endswith("_turnover_bin_mix"), plan.schedule


def test_sm103a_refused_row_takes_the_build_on_sm100a(planner):
    # The span check alone moves the launch: on a 148-SM sm_100a part the
    # same 12 x 88 / 107-token row at 16 heads takes the bin-mix build with
    # the rule's plan words after its table.
    lengths, heads = [88] * 9 + [107] * 3, 16
    planner.configure("sm_100a")
    plan = _launch(_prepare(lengths, heads))
    table, bins, after = _launch_table(planner, lengths, heads)
    words = runtime._work_bin_mix_plan(table, bins, tuple(lengths), heads)
    assert words is not None and after == words
    assert plan.work_bin_mix, plan.schedule
    assert plan.module.name == BIN_MIX
    assert plan.schedule.endswith("_turnover_bin_mix"), plan.schedule


def test_t32_runs_the_turnover_build(planner):
    planner.configure("sm_100a")
    plan = _launch(_prepare([32] * 16, 12))
    assert plan.module.name in TURN["sm_100a"].values(), plan.schedule


def _mix_plan(planner, arch="sm_100a"):
    planner.configure(arch)
    plan = _launch(_prepare([256] * 16, 12))
    plan.module.prepare(grid=plan.grid, **plan.args)
    return plan, plan.module, plan._work_bin_mix_fallback[0]


def test_launch_runs_the_build(planner):
    plan, mix, turnover = _mix_plan(planner)
    plan._launch_in_stream()
    plan._launch_in_stream()
    assert [grid for grid, _ in mix.launches] == [(148, 1, 1)] * 2
    assert mix.launches[0][1]["chunk_state"] is plan.args["chunk_state"]
    assert not turnover.launches


@pytest.mark.parametrize("arch", list(runtime.WORK_BIN_MIX_ARCHES))
def test_refused_launch_switches_the_plan_to_the_turnover_build(
    planner, monkeypatch, arch
):
    cleared = []
    monkeypatch.setattr(runtime, "_clear_cuda_launch_error", lambda: cleared.append(1))
    plan, mix, turnover = _mix_plan(planner, arch)
    ring, ready = plan.args["chunk_state"], plan.args["zero_workspace"]
    bindings = plan._work_bin_mix_fallback[3]
    held = plan._work_bin_mix_record()
    _Kernel.refuse[BIN_MIX] = REFUSED
    plan._launch_in_stream()
    assert len(mix.launches) == 1, "the refused grid is tried once"
    assert cleared == [1]
    assert not plan.work_bin_mix
    assert plan.module is turnover and plan.schedule.endswith("_turnover")
    sms = dict((a, n) for a, _, n in PARTS)[arch]
    assert turnover.prepared == [(sms, 1, 1)]
    launched = turnover.launches[0][1]
    for name in ("chunk_state", "zero_workspace", "seq_order"):
        assert launched[name] is bindings[name], name
    assert launched["chunk_state"] is not ring
    assert launched["zero_workspace"] is not ready
    _Kernel.refuse.clear()
    plan._launch_in_stream()
    held[0](*held[1])
    assert len(mix.launches) == 1
    assert len(turnover.launches) == 3 and cleared == [1]


def test_other_launch_errors_propagate_and_keep_the_plan(planner):
    plan, mix, turnover = _mix_plan(planner)
    _Kernel.refuse[BIN_MIX] = RuntimeError(
        "kernel_x: cudaLaunchKernelExC failed: an illegal memory access"
    )
    with pytest.raises(RuntimeError, match="illegal memory access"):
        plan._launch_in_stream()
    assert plan.work_bin_mix and plan.module is mix
    assert not turnover.launches and not turnover.prepared


def test_fallback_preserves_rebound_bindings(planner):
    plan, mix, turnover = _mix_plan(planner)
    moved = torch.empty_like(plan.args["out"])
    plan.args["out"] = moved
    _Kernel.refuse[BIN_MIX] = REFUSED
    plan._launch_in_stream()
    assert turnover.launches[0][1]["out"] is moved


def test_plan_cache_hit_keeps_the_build_and_its_ring(planner):
    planner.configure("sm_100a")
    cache = KDAPrefillPlanCache(4)
    first = _launch(_prepare([256] * 16, 12, plan_cache=cache))
    ring = first.args["chunk_state"]
    second = _launch(_prepare([256] * 16, 12, plan_cache=cache))
    assert second is first and second.work_bin_mix
    assert second.args["chunk_state"] is ring
    assert [k.name for k in _Kernel.made].count(BIN_MIX) == 1
