"""Turnover work bins of short direct M128 N32 launches (CPU; no GPU needed).

The strict bin test (``_use_direct_work_bins``) bins a direct launch only where
its largest bin, in chunks, exceeds the longest task.  A short N32 launch the
strict test keeps one task per CTA also asks for work bins, for its module's
turnover build only, where bins that charge every task
``BF16_WORK_BIN_TURNOVER_TASK_COST`` chunks pay (``_use_turnover_work_bins``).
It builds that assignment (``task_cost``) and keeps one task per CTA where its
module has no turnover build; the composite's split parts never ask.
"""

from functools import partial

import pytest

import flashinfer.cake_kda_tf32_runtime as runtime
import flashinfer.jit.cake_kda_tf32 as jit

NAME = "compiled_bf16_fused_m128"
CHUNK = runtime.BF16_M128_CHUNK
COST = runtime.BF16_WORK_BIN_TURNOVER_TASK_COST
MIXED = (17, 64, 65, 127, 128, 200, 255, 96) * 2
BIMODAL = (256, 17) * 8
PARTS = [
    pytest.param("sm_100a", 148, id="sm_100a-148"),
    pytest.param("sm_103a", 152, id="sm_103a-152"),
]

# (lengths, heads, takes turnover bins on sm_100a-148, on sm_103a-152): rows of
# the short mixed route sweep, both sides of the rule on each part.
SHAPES = [
    pytest.param(MIXED, 16, True, True, id="mixed-h16"),
    pytest.param(MIXED, 12, False, False, id="mixed-h12"),
    pytest.param(BIMODAL, 16, True, True, id="8x(256+17)-h16"),
    pytest.param(BIMODAL, 12, False, False, id="8x(256+17)-h12"),
    pytest.param((256,) * 4 + (32,) * 12, 16, True, True, id="4x256+12x32-h16"),
    pytest.param(
        (80, 149, 253, 29, 43, 155, 81, 202, 253, 43), 16, False, False, id="bs10-h16"
    ),
    pytest.param(
        (58, 45, 226, 189, 72, 106, 199, 40, 142, 142, 57, 119),
        16,
        True,
        False,
        id="bs12-h16",
    ),
    pytest.param(
        (77, 181, 220, 207, 22, 110, 250, 52, 73, 120, 111, 117, 103),
        12,
        False,
        False,
        id="bs13-h12",
    ),
    pytest.param(
        (77, 181, 220, 207, 22, 110, 250, 52, 73, 120, 111, 117, 103),
        16,
        True,
        True,
        id="bs13-h16",
    ),
    pytest.param(
        (22, 18, 16, 51, 8, 15, 57, 10, 37, 62, 33, 29, 43, 47, 8, 39),
        12,
        True,
        True,
        id="short-bs16-h12",
    ),
    # Longest sequence of 15 chunks: past the sm_100a turnover chunk count.
    pytest.param(
        (244, 179, 352, 463, 57, 193, 388, 229, 473, 50, 360, 364, 34, 372),
        16,
        False,
        True,
        id="bs14-15chunks-h16",
    ),
    pytest.param((512, 16) * 8, 16, False, True, id="8x(512+16)-h16"),
    # Already work binned by the strict test: the rule never applies.
    pytest.param((64,) * 16, 12, False, False, id="bs16x64-h12-strict"),
    pytest.param((1, 33, 64, 96) * 12, 16, False, False, id="short-mixed-h16-strict"),
]


def _strict(lengths, heads, sm_count):
    return runtime._use_direct_work_bins(
        tuple(lengths), num_heads=heads, sm_count=sm_count, chunk_tokens=CHUNK
    )


def _takes_turnover_bins(lengths, heads, arch, sm_count):
    """The launch's choice: the rule applies where the strict test keeps one
    task per CTA (the direct route's composition)."""
    return not _strict(lengths, heads, sm_count) and runtime._use_turnover_work_bins(
        tuple(lengths), num_heads=heads, sm_count=sm_count, gpu_arch=arch
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
@pytest.mark.parametrize(("lengths", "heads", "on_b200", "on_gb300"), SHAPES)
def test_turnover_bins_route_per_part(
    arch, sm_count, lengths, heads, on_b200, on_gb300
):
    want = on_b200 if sm_count == 148 else on_gb300
    assert _takes_turnover_bins(lengths, heads, arch, sm_count) == want
    if want:
        # Every launch the rule moves stays inside the turnover route.
        assert runtime._work_bin_turnover_route(
            route_work_bins=True,
            direct_n16=False,
            n32_ft_slab=False,
            max_seq_len=max(lengths),
            gpu_arch=arch,
            sm_count=sm_count,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
def test_turnover_bins_stay_within_the_part_chunk_limit(arch, sm_count):
    limit = runtime._work_bin_turnover_max_task_chunks(arch, sm_count)
    assert limit == (
        runtime.BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS_GB300
        if sm_count == 152
        else runtime.BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS
    )
    at = (CHUNK * limit, 17) * 8
    past = (CHUNK * limit + 1, 17) * 8
    assert runtime._use_turnover_work_bins(
        at, num_heads=16, sm_count=sm_count, gpu_arch=arch
    )
    assert not runtime._use_turnover_work_bins(
        past, num_heads=16, sm_count=sm_count, gpu_arch=arch
    )
    # 152 SMs of another arch take the default chunk count.
    assert runtime._work_bin_turnover_max_task_chunks("sm_100a", 152) == (
        runtime.BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
def test_turnover_bins_need_more_tasks_than_sms(arch, sm_count):
    kwargs = dict(sm_count=sm_count, gpu_arch=arch)
    lengths = (256, 17) * 4
    assert not runtime._use_turnover_work_bins(lengths, num_heads=16, **kwargs)
    assert not runtime._use_turnover_work_bins((), num_heads=16, **kwargs)
    assert not runtime._use_turnover_work_bins(
        BIMODAL[:-1] + (0,), num_heads=16, **kwargs
    )


@pytest.mark.cpu_only
def test_bin_table_charges_the_task_cost():
    """8x(256+17) H16 on 148 bins: charged its chunks only, every short task
    lands in the 20 bins without a long task (7 tasks in a bin); charged the
    task cost, those bins hold 3 short tasks and 68 long bins one more."""
    kwargs = dict(num_heads=16, worker_count=148, chunk_tokens=CHUNK)
    chunks = [(n + CHUNK - 1) // CHUNK for n in BIMODAL]

    def shape(table, num_bins):
        bins = [table[table[i] : table[i + 1]] for i in range(num_bins)]
        return (
            max(len(b) for b in bins),
            max(sum(chunks[t // 16] for t in b) for b in bins),
            sorted(t for b in bins for t in b),
        )

    plain = runtime._make_direct_work_bins(BIMODAL, **kwargs)
    assert plain == runtime._make_direct_work_bins(BIMODAL, task_cost=0, **kwargs)
    charged = runtime._make_direct_work_bins(BIMODAL, task_cost=COST, **kwargs)
    assert plain[1] == charged[1] == 148
    assert shape(*plain)[:2] == (7, 8)
    assert shape(*charged)[:2] == (3, 9)
    assert shape(*plain)[2] == shape(*charged)[2] == list(range(16 * 16))


def _stub_registry(monkeypatch, arch, *, work_bins=None, turn=None, **tables):
    monkeypatch.setattr(runtime, "detect_gpu_arch", lambda *args, **kwargs: arch)
    monkeypatch.setattr(jit, "device_arch", lambda *args, **kwargs: arch)
    monkeypatch.setattr(
        jit, "FACTORIES", {arch: {NAME: {jit._key((32,), {"a": 1}): "row_module"}}}
    )
    for table in (
        "_RENDERED_ROUTES",
        "_RENDERED_HEAD_ROUTES",
        "_RENDERED_MERGE_ROUTES",
        "_RENDERED_OWNER_HELPER_ROUTES",
        "_RENDERED_OWNER_HELPER_POOL_ROUTES",
        "_RENDERED_FOLD_ROUTES",
        "_RENDERED_MAP_FOLD_ROUTES",
        "_RENDERED_PAIR_SHARE_ROUTES",
    ):
        monkeypatch.setattr(jit, table, {arch: tables.get(table, {})})
    monkeypatch.setattr(jit, "_RENDERED_WORK_BIN_ROUTES", {arch: work_bins or {}})
    monkeypatch.setattr(jit, "_RENDERED_WORK_BIN_TURN_ROUTES", {arch: turn or {}})
    monkeypatch.setattr(
        jit, "PreparedKernel", lambda name, arch: ("prepared", name, arch)
    )
    built = []

    def build(factory, *args, **kwargs):
        built.append(factory)
        return factory

    monkeypatch.setattr(runtime, "_build_kda_module", build)
    return built


def _body(launch=runtime.FlashKDABlackwellBF16FusedLaunch):
    body = object.__new__(launch)
    body._qkv_carrier_names = {}
    return body


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_turnover_bins_take_the_turnover_build_or_no_bins(monkeypatch, arch):
    """Bins asked for the turnover build only: the turnover build where the
    loader names one, else the same factory without work bins (the
    one-task-per-CTA program the strict test chose)."""
    built = _stub_registry(
        monkeypatch,
        arch,
        work_bins={"row_module": "bins"},
        turn={"row_module": "turn"},
    )
    binned = partial(runtime._factory, NAME, route_work_bins=True)
    flags = dict(work_bin_turnover=True, turnover_bins_only=True)
    _body()._build_body(binned, 32, a=1, **flags)
    assert built[-1].func is runtime._work_bin_turnover_factory
    assert built[-1].keywords == binned.keywords
    for missing in ("_RENDERED_WORK_BIN_TURN_ROUTES", "_RENDERED_WORK_BIN_ROUTES"):
        with monkeypatch.context() as patch:
            patch.setattr(jit, missing, {arch: {}})
            _body()._build_body(binned, 32, a=1, **flags)
        assert built[-1].func is runtime._factory
        assert built[-1].args == (NAME,)
        assert built[-1].keywords == {"route_work_bins": False}
    # Owner/helper requests never take a turnover build: no bins either.
    helper = partial(runtime._factory, NAME, route_work_bins=True, owner_helper=True)
    _body()._build_body(helper, 32, a=1, **flags)
    assert built[-1].func is runtime._factory
    assert built[-1].keywords == {"route_work_bins": False, "owner_helper": True}
    # Without the flag a strict work-binned build keeps its bins.
    with monkeypatch.context() as patch:
        patch.setattr(jit, "_RENDERED_WORK_BIN_TURN_ROUTES", {arch: {}})
        _body()._build_body(binned, 32, a=1, work_bin_turnover=True)
    assert built[-1] is binned
    # Factories other than ``_factory`` pass through.
    other = partial(runtime._m64_pair_share_factory, NAME, route_work_bins=True)
    _body()._build_body(other, 32, a=1, **flags)
    assert built[-1] is other


FOLD = runtime.FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerExportFoldLaunch
MAPFOLD = runtime.FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerMapFoldLaunch
MERGE = runtime.FlashKDABlackwellFP32SlabM128PDLConsumerMergeLaunch


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("launch", "table", "built_by"),
    [
        pytest.param(
            FOLD, "_RENDERED_FOLD_ROUTES", "_pair_map_fold_factory", id="fold"
        ),
        pytest.param(
            MAPFOLD,
            "_RENDERED_MAP_FOLD_ROUTES",
            "_chain_map_fold_factory",
            id="map-fold",
        ),
        pytest.param(
            MERGE, "_RENDERED_MERGE_ROUTES", "_correction_merge_factory", id="merge"
        ),
    ],
)
def test_interceptors_build_their_program_with_turnover_bins_asked(
    monkeypatch, launch, table, built_by
):
    """The composite's split parts never ask (``_n32_ft_slab``); a body built
    with the flag still reaches its interceptor, which builds its own program."""
    assert launch._n32_ft_slab
    built = _stub_registry(
        monkeypatch,
        "sm_103a",
        work_bins={"row_module": "bins"},
        turn={"row_module": "turn"},
        **{table: {"row_module": "own"}},
    )
    binned = partial(runtime._factory, NAME, route_work_bins=True)
    _body(launch)._build_body(
        binned, 32, a=1, work_bin_turnover=False, turnover_bins_only=True
    )
    assert built[-1].func is getattr(runtime, built_by)
    assert built[-1].args == (NAME,)
