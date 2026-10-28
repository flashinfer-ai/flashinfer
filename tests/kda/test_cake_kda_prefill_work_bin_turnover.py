"""Work-bin turnover route of the direct M128 N32 body (CPU; no GPU needed).

A direct M128 launch takes the loader's turnover build of its module
(``_RENDERED_WORK_BIN_TURN_ROUTES``) only where it applies: a work-binned N32
serving launch whose longest sequence is short enough.  Affine split parts
never take it, and the composite's Fold / Merge factory interceptors keep their
builds (the pair-map fold, its ``_PairMapFoldUnavailable`` fallback, the
correction merge) whichever factory reaches them.
"""

from functools import partial
from types import SimpleNamespace

import pytest

import flashinfer.cake_kda_tf32_runtime as runtime
import flashinfer.jit.cake_kda_tf32 as jit

NAME = "compiled_bf16_fused_m128"
FOLD = runtime.FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerExportFoldLaunch
MERGE = runtime.FlashKDABlackwellFP32SlabM128PDLConsumerMergeLaunch
EXPORT_MAIN = "cake_kda_bf16_15be1db8a50ee4d9fd16"
FACTORIES = ("_factory", "_work_bin_turnover_factory")

# Unbounded gate, BF16 compute, FP32 indexed pool, FP32 checkpoint rows: the
# composite's main windows have more (window, head) tasks than SMs, a largest
# work bin longer than the longest window, and windows no longer than the
# part's turnover chunk count.
EXAMPLES = [
    ("sm_100a", 148, 4, (9472, 200), "BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS"),
    ("sm_103a", 152, 8, (9600, 200), "BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS_GB300"),
]


def _main_windows(monkeypatch, arch, sm_count, heads, lengths):
    monkeypatch.setattr(runtime, "detect_gpu_arch", lambda *args, **kwargs: arch)
    token_offsets, _ = runtime._affine_split_windows(
        sequence_lengths=lengths,
        num_heads=heads,
        sm_count=sm_count,
        fp32_indexed_state=True,
        shared_tf32_factors=False,
        checkpoints=True,
        unbounded_softplus=True,
    )
    return tuple(b - a for a, b in zip(token_offsets, token_offsets[1:], strict=False))


@pytest.mark.parametrize(("arch", "sm_count", "heads", "lengths", "limit"), EXAMPLES)
def test_composite_main_windows_are_short_work_binned_tasks(
    monkeypatch, arch, sm_count, heads, lengths, limit
):
    windows = _main_windows(monkeypatch, arch, sm_count, heads, lengths)
    assert len(windows) * heads > sm_count
    assert runtime._use_direct_work_bins(
        windows,
        num_heads=heads,
        sm_count=sm_count,
        chunk_tokens=runtime.BF16_M128_CHUNK,
    )
    assert max(windows) <= runtime.BF16_M128_CHUNK * getattr(runtime, limit)


@pytest.mark.parametrize(("arch", "sm_count", "heads", "lengths", "limit"), EXAMPLES)
def test_split_parts_never_take_the_turnover_route(
    monkeypatch, arch, sm_count, heads, lengths, limit
):
    windows = _main_windows(monkeypatch, arch, sm_count, heads, lengths)
    shape = dict(
        route_work_bins=True,
        direct_n16=False,
        max_seq_len=max(windows),
        gpu_arch=arch,
        sm_count=sm_count,
    )
    assert runtime._work_bin_turnover_route(n32_ft_slab=False, **shape)
    assert not runtime._work_bin_turnover_route(n32_ft_slab=True, **shape)
    assert FOLD._n32_ft_slab and MERGE._n32_ft_slab


@pytest.mark.parametrize(
    ("arch", "sm_count", "limit"),
    [
        ("sm_100a", 148, "BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS"),
        ("sm_103a", 152, "BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS_GB300"),
        ("sm_103a", 148, "BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS"),
    ],
)
def test_turnover_route_chunk_limit(arch, sm_count, limit):
    tokens = runtime.BF16_M128_CHUNK * getattr(runtime, limit)
    shape = dict(direct_n16=False, n32_ft_slab=False, gpu_arch=arch, sm_count=sm_count)
    assert runtime._work_bin_turnover_route(
        route_work_bins=True, max_seq_len=tokens, **shape
    )
    assert not runtime._work_bin_turnover_route(
        route_work_bins=True, max_seq_len=tokens + 1, **shape
    )
    assert not runtime._work_bin_turnover_route(
        route_work_bins=False, max_seq_len=1, **shape
    )
    assert not runtime._work_bin_turnover_route(
        route_work_bins=True,
        max_seq_len=1,
        **{**shape, "direct_n16": True},
    )


def _stub_registry(monkeypatch, arch, *, work_bins=None, fold=None, turn=None):
    assert runtime._kda_registry is jit
    monkeypatch.setattr(runtime, "detect_gpu_arch", lambda *args, **kwargs: arch)
    monkeypatch.setattr(
        jit, "FACTORIES", {arch: {NAME: {jit._key((32,), {"a": 1}): "row_module"}}}
    )
    monkeypatch.setattr(jit, "_RENDERED_HEAD_ROUTES", {arch: {}})
    monkeypatch.setattr(jit, "_RENDERED_WORK_BIN_ROUTES", {arch: work_bins or {}})
    monkeypatch.setattr(jit, "_RENDERED_FOLD_ROUTES", {arch: fold or {}})
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


@pytest.mark.parametrize("factory_name", FACTORIES)
def test_fold_interceptor_builds_the_fold_from_either_factory(
    monkeypatch, factory_name
):
    built = _stub_registry(monkeypatch, "sm_100a", fold={"row_module": "fold_program"})
    factory = partial(getattr(runtime, factory_name), NAME, route_work_bins=True)
    FOLD._build_kda_module(factory, 32, a=1)
    assert built[-1].func is runtime._pair_map_fold_factory, (
        f"fold build skipped: {built[-1].func.__name__} reached the module build"
    )


@pytest.mark.parametrize("factory_name", FACTORIES)
def test_fold_interceptor_falls_back_from_either_factory(monkeypatch, factory_name):
    _stub_registry(
        monkeypatch,
        "sm_103a",
        work_bins={"row_module": "bins_program"},
        fold={"row_module": "fold_program"},
        turn={"row_module": "turn_program"},
    )
    factory = partial(getattr(runtime, factory_name), NAME, route_work_bins=True)
    with pytest.raises(runtime._PairMapFoldUnavailable):
        FOLD._build_kda_module(factory, 32, a=1)


@pytest.mark.parametrize("factory_name", FACTORIES)
def test_merge_interceptor_builds_the_merge_from_either_factory(
    monkeypatch, factory_name
):
    built = _stub_registry(monkeypatch, "sm_103a", turn={"row_module": "turn_program"})
    factory = partial(getattr(runtime, factory_name), NAME, route_work_bins=True)
    MERGE._build_kda_module(factory, 32, a=1)
    assert built[-1].func is runtime._correction_merge_factory, (
        f"merge build skipped: {built[-1].func.__name__} reached the module build"
    )


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_fold_interceptor_with_the_loader_tables(monkeypatch, arch):
    """Real loader tables: the export main resolves the same with either factory."""
    monkeypatch.setattr(runtime, "detect_gpu_arch", lambda *args, **kwargs: arch)
    keys = [
        key
        for key, module in jit.FACTORIES.get(arch, {}).get(NAME, {}).items()
        if module == EXPORT_MAIN
    ]
    assert keys
    built = []

    def build(factory, *args, **kwargs):
        built.append(factory)
        return factory

    monkeypatch.setattr(runtime, "_build_kda_module", build)
    for args, kwargs in keys:
        outcomes = []
        for factory_name in FACTORIES:
            factory = partial(
                getattr(runtime, factory_name), NAME, route_work_bins=True
            )
            try:
                FOLD._build_kda_module(factory, *args, **dict(kwargs))
                outcomes.append((built[-1].func, built[-1].args, built[-1].keywords))
            except runtime._PairMapFoldUnavailable:
                outcomes.append("unavailable")
        assert outcomes[0] == outcomes[1], (args, kwargs, outcomes)
        assert (
            runtime._work_bin_turnover_program(
                NAME, *args, route_work_bins=True, **dict(kwargs)
            )
            is None
        )


def test_build_body_takes_the_turnover_build_only_where_named(monkeypatch):
    arch = "sm_103a"
    built = _stub_registry(
        monkeypatch,
        arch,
        work_bins={"row_module": "bins_program"},
        turn={"row_module": "turn_program"},
    )
    body = object.__new__(runtime.FlashKDABlackwellBF16FusedLaunch)
    body._qkv_carrier_names = {}
    binned = partial(runtime._factory, NAME, route_work_bins=True)
    body._build_body(binned, 32, a=1, work_bin_turnover=True)
    assert built[-1].func is runtime._work_bin_turnover_factory
    assert runtime._work_bin_turnover_factory(NAME, 32, a=1, route_work_bins=True) == (
        "prepared",
        "turn_program",
        arch,
    )
    body._build_body(binned, 32, a=1, work_bin_turnover=False)
    assert built[-1] is binned
    for factory in (
        partial(runtime._factory, NAME, route_work_bins=False),
        partial(runtime._factory, NAME, route_work_bins=True, owner_helper=True),
    ):
        body._build_body(factory, 32, a=1, work_bin_turnover=True)
        assert built[-1] is factory
    monkeypatch.setattr(jit, "_RENDERED_WORK_BIN_TURN_ROUTES", {arch: {}})
    body._build_body(binned, 32, a=1, work_bin_turnover=True)
    assert built[-1] is binned


# --- The turnover route beside the other generated-program routes (CPU). ---
#
# The fold, map-fold and merge interceptors decide by the factory they get
# (``_is_fused_m128_row``); the turnover factory is the same factory row, so
# each builds its own program from it as from ``_factory`` and refuses every
# other row or wrapper. The pair-share (M64) and owner/helper (pooled or not)
# requests never take the turnover build, and a composite checks the body it
# got against its own table before it drops another launch.

MAPFOLD = runtime.FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerMapFoldLaunch
INTERCEPTORS = [
    pytest.param(
        FOLD,
        runtime._PairMapFoldUnavailable,
        "_RENDERED_FOLD_ROUTES",
        "_pair_map_fold_factory",
        id="pair-map-fold",
    ),
    pytest.param(
        MAPFOLD,
        runtime._ChainMapFoldUnavailable,
        "_RENDERED_MAP_FOLD_ROUTES",
        "_chain_map_fold_factory",
        id="chain-map-fold",
    ),
    pytest.param(
        MERGE,
        runtime._CorrectionMergeUnavailable,
        "_RENDERED_MERGE_ROUTES",
        "_correction_merge_factory",
        id="merge",
    ),
]


class _Kernel:
    def __init__(self, name, arch):
        self.name, self.arch = name, arch


def _stub_routes(monkeypatch, arch, **tables):
    """Loader tables of one row module ``row_module`` (32-token chunk, a=1)."""
    monkeypatch.setattr(runtime, "detect_gpu_arch", lambda *args, **kwargs: arch)
    monkeypatch.setattr(jit, "device_arch", lambda *args, **kwargs: arch)
    monkeypatch.setattr(
        jit, "FACTORIES", {arch: {NAME: {jit._key((32,), {"a": 1}): "row_module"}}}
    )
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
    ):
        monkeypatch.setattr(jit, table, {arch: tables.get(table, {})})
    monkeypatch.setattr(jit, "PreparedKernel", _Kernel)
    built = []

    def build(factory, *args, **kwargs):
        built.append(factory)
        return factory

    monkeypatch.setattr(runtime, "_build_kda_module", build)
    return built


@pytest.mark.parametrize("factory_name", FACTORIES)
@pytest.mark.parametrize(("launch", "unavailable", "table", "built_by"), INTERCEPTORS)
def test_every_interceptor_builds_its_program_from_the_row_of_either_factory(
    monkeypatch, launch, unavailable, table, built_by, factory_name
):
    built = _stub_routes(monkeypatch, "sm_100a", **{table: {"row_module": "own"}})
    launch._build_kda_module(partial(getattr(runtime, factory_name), NAME), 32, a=1)
    assert built[-1].func is getattr(runtime, built_by)
    assert built[-1].args == (NAME,)


@pytest.mark.parametrize(("launch", "unavailable", "table", "built_by"), INTERCEPTORS)
def test_every_interceptor_refuses_another_row_or_a_wrapper(
    monkeypatch, launch, unavailable, table, built_by
):
    built = _stub_routes(monkeypatch, "sm_103a", **{table: {"row_module": "own"}})
    turnover = partial(runtime._work_bin_turnover_factory, NAME)
    for factory in (
        partial(runtime._work_bin_turnover_factory, "compiled_bf16_fused_m64"),
        partial(lambda factory, *a, **k: factory(*a, **k), turnover),
        partial(runtime._m64_pair_share_factory, NAME),
    ):
        with pytest.raises(unavailable):
            launch._build_kda_module(factory, 32, a=1)
    assert not built


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_turnover_build_only_where_the_factory_takes_the_work_binned_program(
    monkeypatch, arch
):
    """The turnover build replaces exactly the program ``_factory`` would take
    with work bins: never an owner/helper (pooled) or head-routed program, nor
    a module without a work-binned build."""
    full = dict(
        _RENDERED_ROUTES={"row_module": "routed"},
        _RENDERED_HEAD_ROUTES={"row_module": {12: ["banded", 8]}},
        _RENDERED_WORK_BIN_ROUTES={"row_module": "bins"},
        _RENDERED_OWNER_HELPER_ROUTES={"row_module": "owner_helper"},
        _RENDERED_OWNER_HELPER_POOL_ROUTES={"row_module": "pool"},
        _RENDERED_WORK_BIN_TURN_ROUTES={"row_module": "turn"},
    )
    _stub_routes(monkeypatch, arch, **full)
    named = 0
    for work_bins in (False, True):
        for owner_helper in (False, True):
            for pool in (False, True):
                for heads in (None, 12):
                    call = dict(
                        route_work_bins=work_bins,
                        owner_helper=owner_helper,
                        owner_helper_pool=pool,
                        route_heads=heads,
                    )
                    turn = runtime._work_bin_turnover_program(NAME, 32, a=1, **call)
                    taken = jit._factory(NAME, 32, a=1, **call).name
                    if turn is not None:
                        named += 1
                        assert (turn, taken) == ("turn", "bins"), call
                    if taken != "bins":
                        assert turn is None, call
    assert named == 1  # work bins, no owner/helper, no pool, no head route
    plain = dict(route_work_bins=True)
    assert runtime._work_bin_turnover_program(NAME, 32, a=1, **plain) == "turn"
    # A turnover build without the module's work-binned build is never taken.
    _stub_routes(monkeypatch, arch, **{**full, "_RENDERED_WORK_BIN_ROUTES": {}})
    assert runtime._work_bin_turnover_program(NAME, 32, a=1, **plain) is None
    assert jit._factory(NAME, 32, a=1, **plain).name == "routed"


def test_build_body_leaves_the_pair_share_and_owner_helper_requests_alone(
    monkeypatch,
):
    arch = "sm_100a"
    built = _stub_routes(
        monkeypatch,
        arch,
        _RENDERED_WORK_BIN_ROUTES={"row_module": "bins"},
        _RENDERED_OWNER_HELPER_ROUTES={"row_module": "owner_helper"},
        _RENDERED_OWNER_HELPER_POOL_ROUTES={"row_module": "pool"},
        _RENDERED_PAIR_SHARE_ROUTES={"row_module": "pair"},
        _RENDERED_WORK_BIN_TURN_ROUTES={"row_module": "turn"},
    )
    body = object.__new__(runtime.FlashKDABlackwellBF16FusedLaunch)
    body._qkv_carrier_names = {}
    for factory in (
        partial(runtime._m64_pair_share_factory, NAME, route_work_bins=True),
        partial(runtime._factory, NAME, route_work_bins=True, owner_helper=True),
        partial(
            runtime._factory,
            NAME,
            route_work_bins=True,
            owner_helper=True,
            owner_helper_pool=True,
        ),
        partial(runtime._factory, NAME, route_work_bins=True, owner_helper_pool=True),
    ):
        body._build_body(factory, 32, a=1, work_bin_turnover=True)
        assert built[-1] is factory
    binned = partial(runtime._factory, NAME, route_work_bins=True)
    body._build_body(binned, 32, a=1, work_bin_turnover=True)
    assert built[-1].func is runtime._work_bin_turnover_factory
    assert built[-1].args == (NAME,) and built[-1].keywords == binned.keywords


@pytest.mark.parametrize(
    "table",
    ["_RENDERED_FOLD_ROUTES", "_RENDERED_MAP_FOLD_ROUTES", "_RENDERED_MERGE_ROUTES"],
)
def test_composite_body_check_refuses_a_turnover_program(monkeypatch, table):
    _stub_routes(
        monkeypatch,
        "sm_103a",
        **{
            table: {"row_module": "own"},
            "_RENDERED_WORK_BIN_TURN_ROUTES": {"row_module": "turn"},
        },
    )

    def launch(name):
        return SimpleNamespace(module=SimpleNamespace(name=name))

    assert runtime._built_from_table(launch("own"), table)
    assert not runtime._built_from_table(launch("turn"), table)


def test_composite_launch_classes_never_take_the_turnover_route():
    shape = dict(
        route_work_bins=True,
        direct_n16=False,
        max_seq_len=32,
        gpu_arch="sm_100a",
        sm_count=148,
    )
    for launch in (FOLD, MAPFOLD, MERGE):
        assert launch._n32_ft_slab
        assert not runtime._work_bin_turnover_route(
            n32_ft_slab=launch._n32_ft_slab, **shape
        )
    assert runtime._work_bin_turnover_route(n32_ft_slab=False, **shape)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_turnover_programs_of_the_loader(monkeypatch, arch):
    """Real loader tables: each turnover program is its module's own build,
    reached through the turnover table only, beside a work-binned program of
    the same module; no fold, map-fold, merge or pair-share route names it or
    its module."""
    turn = jit._RENDERED_WORK_BIN_TURN_ROUTES[arch]
    if not turn:
        pytest.skip(f"the loader names no turnover build on {arch}")
    others = {
        table: getattr(jit, table)[arch]
        for table in (
            "_RENDERED_ROUTES",
            "_RENDERED_MERGE_ROUTES",
            "_RENDERED_WORK_BIN_ROUTES",
            "_RENDERED_OWNER_HELPER_ROUTES",
            "_RENDERED_OWNER_HELPER_POOL_ROUTES",
            "_RENDERED_FOLD_ROUTES",
            "_RENDERED_MAP_FOLD_ROUTES",
            "_RENDERED_PAIR_SHARE_ROUTES",
        )
    }
    heads = {
        program
        for by_heads in jit._RENDERED_HEAD_ROUTES[arch].values()
        for program, _ in by_heads.values()
    }
    monkeypatch.setattr(runtime, "detect_gpu_arch", lambda *args, **kwargs: arch)
    rows = jit.FACTORIES[arch][NAME]
    for module, program in turn.items():
        assert module in others["_RENDERED_WORK_BIN_ROUTES"], module
        assert program != others["_RENDERED_WORK_BIN_ROUTES"][module]
        assert arch in jit._RENDERED_MODULES[program]["arches"]
        assert program not in heads
        for table, by_module in others.items():
            assert program not in by_module.values(), (table, program)
            if table in (
                "_RENDERED_MERGE_ROUTES",
                "_RENDERED_FOLD_ROUTES",
                "_RENDERED_MAP_FOLD_ROUTES",
                "_RENDERED_PAIR_SHARE_ROUTES",
            ):
                assert module not in by_module, (table, module)
        keys = [key for key, row in rows.items() if row == module]
        assert keys, module
        for args, kwargs in keys:
            kwargs = dict(kwargs)
            got = runtime._work_bin_turnover_program(
                NAME, *args, route_work_bins=True, **kwargs
            )
            assert got == program, (args, kwargs)
            for extra in ({"owner_helper": True}, {"owner_helper_pool": True}):
                assert (
                    runtime._work_bin_turnover_program(
                        NAME, *args, route_work_bins=True, **extra, **kwargs
                    )
                    is None
                )
