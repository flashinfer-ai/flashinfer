"""Generated paged decode programs of the Cake FMHA module (``backend="cake"``).

Host-only: the pinned program tables against the generated loaders, the
launch grid, the host split plan of the quantized decode, the load-time checks
(a loader that disagrees with the host fails closed), the no-sync spy and the
argument values a stand-in program receives.  GPU: a batch without query
tokens returns before any route is chosen or any kernel launched.  Each
program family adds its shape rows, boundaries and GPU cases.
"""

from __future__ import annotations

import contextlib
import importlib
import operator
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import flashinfer.cake_fmha as cake_api
import flashinfer.decode as decode_api

_TARGET_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}


def _loader(family):
    return importlib.import_module(f"flashinfer.jit.cake_fmha_{family}")


def _served() -> dict[tuple[str, str], set[str]]:
    """``(family, identifier)`` -> the targets whose routes name it."""

    served: dict[tuple[str, str], set[str]] = {}
    for target, routes in cake_api._DECODE_PROGRAM_ROUTES.items():
        for key, identifier in routes.items():
            served.setdefault((key[0], identifier), set()).add(_TARGET_ARCH[target])
    return served


# --- The no-sync spy ----------------------------------------------------------


class HostSyncError(AssertionError):
    """A launch read a device value back to the host or synchronized."""


_READ_BACKS = (
    "item",
    "tolist",
    "cpu",
    "numpy",
    "__float__",
    "__int__",
    "__bool__",
    "__index__",
)


@contextlib.contextmanager
def no_host_sync():
    """Every host read-back of a tensor value and every host synchronization
    raises :class:`HostSyncError` inside the block."""

    def refuse(name):
        def method(*args, **kwargs):
            raise HostSyncError(f"{name} reads a device value back to the host")

        return method

    with pytest.MonkeyPatch.context() as patch:
        for name in _READ_BACKS:
            patch.setattr(torch.Tensor, name, refuse(f"Tensor.{name}"))
        patch.setattr(torch.cuda, "synchronize", refuse("torch.cuda.synchronize"))
        patch.setattr(torch.cuda.Event, "synchronize", refuse("Event.synchronize"))
        patch.setattr(torch.cuda.Stream, "synchronize", refuse("Stream.synchronize"))
        yield


def test_no_host_sync_spy_refuses_every_read_back() -> None:
    value = torch.ones(1)
    index = torch.ones((), dtype=torch.int64)
    # Each read looks its method up inside the block: a method bound before
    # the patch is the original.
    reads = (
        lambda: value.item(),
        lambda: value.tolist(),
        lambda: value.cpu(),
        lambda: value.numpy(),
        lambda: float(value),
        lambda: int(value),
        lambda: bool(value),
        lambda: operator.index(index),
        lambda: torch.cuda.synchronize(),
        lambda: torch.cuda.Event.synchronize(None),
        lambda: torch.cuda.Stream.synchronize(None),
    )
    with no_host_sync():
        for read in reads:
            with pytest.raises(HostSyncError):
                read()
    assert value.item() == 1.0 and bool(value)


# --- The program tables against the generated loaders (host-only) -----------


def test_cake_fmha_decode_program_tables_match_the_generated_loaders() -> None:
    """Every pinned record names the generated loader's program of the same
    identifier (the load-time checks: name, binding arguments, the targets the
    routes serve it on, the launch), every loader program is pinned, every
    pinned program serves a route, every reducer serves the targets of the
    programs it follows and every decode family has a launch builder."""

    from flashinfer.jit.cake_program import GRID_ARGS, CakeProgram

    served = _served()
    assert set(served) == set(cake_api._DECODE_PROGRAMS)
    for main, reducer in cake_api._DECODE_PROGRAM_REDUCERS.items():
        assert served[main] <= served[reducer], (main, reducer)
        assert cake_api._DECODE_PROGRAMS[reducer][1] == "reduce"
    reducers = set(cake_api._DECODE_PROGRAM_REDUCERS.values())
    for key, (name, launch) in cake_api._DECODE_PROGRAMS.items():
        family, identifier = key
        program = getattr(_loader(family), identifier)
        assert isinstance(program, CakeProgram), key
        assert program.name == name, key
        assert served[key] <= set(program.targets), key
        assert tuple(program.args) == (
            *cake_api._DECODE_PROGRAM_HOST_PARAMS[family],
            *GRID_ARGS,
        ), key
        assert (launch == "reduce") == (key in reducers), key
        block = (32, 1, 1) if key in reducers else (512, 1, 1)
        assert tuple(program.block) == block, key
        assert program.use_pdl is (key in reducers), key
        assert program.cooperative is False, key
        clustered = tuple(program.cluster) != (1, 1, 1)
        if launch == "cluster":
            assert clustered, key
        elif launch != "split":
            assert not clustered, key
        if key not in reducers:
            assert family in cake_api._DECODE_PROGRAM_VALUES, key
    for family in {family for family, _ in cake_api._DECODE_PROGRAMS}:
        pinned = {
            name
            for (pinned_family, _), (name, _) in cake_api._DECODE_PROGRAMS.items()
            if pinned_family == family
        }
        assert {program.name for program in _loader(family).PROGRAMS} == pinned


def test_cake_fmha_decode_launch_grid() -> None:
    grid = cake_api.cake_fmha_decode_launch_grid
    cga = SimpleNamespace(cluster=(4, 1, 1))

    def shape(batch, num_splits=1, num_kv_heads=4):
        return dict(
            batch_size=batch,
            q_len=1,
            num_q_heads=32,
            num_kv_heads=num_kv_heads,
            num_splits=num_splits,
        )

    assert grid("persistent", None, **shape(128), sm_count=148) == (148, 1, 1)
    assert grid("persistent", None, **shape(4), sm_count=148) == (16, 1, 1)
    assert grid("cluster", cga, **shape(4), sm_count=148) == (64, 1, 1)
    assert grid("cluster", cga, **shape(4), sm_count=16) == (64, 1, 1)
    # A cluster program gives each tile its own cluster; fewer SMs refuse.
    with pytest.raises(RuntimeError, match="its own cluster"):
        grid("cluster", cga, **shape(4), sm_count=15)
    assert grid("bhq", None, **shape(256), sm_count=148) == (1, 4, 256)
    assert grid("quant", None, **shape(256), sm_count=148) == (1024, 1, 1)
    assert grid("quant", None, **shape(4, 8), sm_count=152) == (128, 1, 1)
    assert grid("quant128", None, **shape(128), sm_count=148) == (128, 1, 1)
    assert grid("quant128", None, **shape(8), sm_count=148) == (32, 1, 1)
    assert grid("reduce", None, **shape(4, 8), sm_count=148) == (128, 1, 1)
    assert grid("split", None, **shape(1, 9, 1), sm_count=148) == (9, 1, 1)
    assert grid("split", None, **shape(2, 74, 1), sm_count=148) == (148, 1, 1)
    with pytest.raises(ValueError, match="unknown Cake FMHA decode program launch"):
        grid("stream", None, **shape(1), sm_count=148)


@pytest.mark.parametrize(
    ("batch", "num_kv_heads", "max_seq_len", "sm_count", "plan"),
    [
        (4, 4, 16384, 148, (8, 16)),
        (4, 4, 16384, 152, (8, 16)),
        (4, 4, 16129, 148, (8, 16)),
        (4, 4, 16128, 148, (9, 14)),
        (4, 4, 16385, 148, (8, 18)),
        (4, 4, 896, 148, (2, 4)),
        (4, 4, 768, 148, (1, 6)),
        (4, 4, 1, 148, (1, 2)),
        (37, 4, 16384, 148, (1, 128)),
        (256, 4, 8192, 148, (1, 64)),
        (128, 4, 8192, 152, (1, 64)),
    ],
)
def test_cake_fmha_decode_quant_split_plan(
    batch, num_kv_heads, max_seq_len, sm_count, plan
) -> None:
    """The host plan of the quantized decode reads host scalars only and
    agrees with the fixed-plan test of the JIT route."""

    got = cake_api._decode_program_quant_plan(
        batch, num_kv_heads, max_seq_len, sm_count
    )
    assert got == plan
    assert cake_api._decode_quant_fp8_fixed_split(
        batch, num_kv_heads, max_seq_len, sm_count
    ) == (plan == (8, 16))
    label = cake_api._decode_program_plan_label(plan)
    assert label == ("split1" if plan[0] == 1 else f"split{plan[0]}x{plan[1]}")


# --- Load-time checks against a stand-in loader (host-only) ------------------

_FAKE = "decode_fake"
_FAKE_PARAMS = ("query", "out", "num_splits")


def _fake_program(
    name, *, targets=("sm_100a", "sm_103a"), args=None, cluster=(1, 1, 1)
):
    from flashinfer.jit.cake_program import GRID_ARGS, CakeProgram

    return CakeProgram(
        name=name,
        operator_dir="cake_fmha_decode_fake",
        kernel=f"kernel_{name}",
        sources=(f"{name}_kernel.cu", f"{name}_binding.cu"),
        targets=tuple(targets),
        compile_options=tuple((target, ()) for target in targets),
        args=tuple(args or (*_FAKE_PARAMS, *GRID_ARGS)),
        tensors=("query", "out"),
        block=(512, 1, 1),
        cluster=tuple(cluster),
        dynamic_smem_bytes=0,
        use_pdl=False,
        cooperative=False,
        needs_device_linking=False,
    )


def _fake_route(identifier, target="sm100a"):
    return cake_api.CakeFmhaDecodeRoute(
        target=target,
        batch_size=4,
        q_len=1,
        num_q_heads=32,
        num_kv_heads=4,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=False,
        component="decode_native_bf16",
        program=(_FAKE, identifier),
    )


@pytest.fixture
def fake_family(monkeypatch):
    """A stand-in generated loader ``flashinfer.jit.cake_fmha_decode_fake``
    and its host tables; ``entry`` returns a recorder instead of building."""

    from flashinfer.jit.cake_program import GRID_ARGS, CakeProgram

    loader = types.ModuleType(f"flashinfer.jit.cake_fmha_{_FAKE}")
    loader.GOOD = _fake_program("fake_good")
    loader.CLUSTER = _fake_program("fake_cluster", cluster=(4, 1, 1))
    loader.RENAMED = _fake_program("fake_other")
    loader.ONE_TARGET = _fake_program("fake_one_target", targets=("sm_103a",))
    loader.ARGS = _fake_program("fake_args", args=("query", "out", *GRID_ARGS))
    loader.REDUCE = _fake_program("fake_reduce")
    loader.NOT_A_PROGRAM = "fake"
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setattr(
        CakeProgram, "entry", lambda self, target: ("entry", self.name, target)
    )
    monkeypatch.setattr(cake_api, "_DECODE_PROGRAM_MODULES", {})
    programs = {
        "GOOD": ("fake_good", "persistent"),
        "CLUSTER": ("fake_cluster", "persistent"),
        "RENAMED": ("fake_renamed", "persistent"),
        "ONE_TARGET": ("fake_one_target", "persistent"),
        "ARGS": ("fake_args", "persistent"),
        "REDUCE": ("fake_reduce", "reduce"),
        "NOT_A_PROGRAM": ("fake", "persistent"),
    }
    for identifier, record in programs.items():
        monkeypatch.setitem(cake_api._DECODE_PROGRAMS, (_FAKE, identifier), record)
        for target in ("sm100a", "sm103a"):
            routes = cake_api._DECODE_PROGRAM_ROUTES[target]
            monkeypatch.setitem(routes, (_FAKE, identifier), identifier)
    monkeypatch.setitem(cake_api._DECODE_PROGRAM_HOST_PARAMS, _FAKE, _FAKE_PARAMS)
    monkeypatch.setitem(
        cake_api._DECODE_PROGRAM_REDUCERS, (_FAKE, "GOOD"), (_FAKE, "REDUCE")
    )
    return loader


def test_cake_fmha_decode_program_loads_without_a_jit_member(
    monkeypatch, fake_family
) -> None:
    """A program route is optimized by its pinned record alone (no JIT member
    of the route's component), names its program and loads it with its
    reducer."""

    monkeypatch.setattr(
        cake_api,
        "_is_cake_fmha_decode_native_bf16_available",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    route = _fake_route("GOOD")
    assert cake_api.cake_fmha_route_is_optimized(route)
    assert cake_api._route_components(route) == (f"{_FAKE}:GOOD",)
    module, optimized = cake_api._resolve_cake_fmha_decode_module(
        torch.device("meta"), route
    )
    assert optimized
    assert isinstance(module, cake_api.CakeFmhaDecodeProgramModule)
    assert module.entry == ("entry", "fake_good", "sm_100a")
    assert module.launch == "persistent" and module.target == "sm_100a"
    assert module.reducer[:3] == (_FAKE, "REDUCE", "reduce")
    assert module.reducer[4] == ("entry", "fake_reduce", "sm_100a")
    # A route naming a program this module does not route is not optimized.
    unrouted = _fake_route("MISSING")
    assert not cake_api.cake_fmha_route_is_optimized(unrouted)
    with pytest.raises(RuntimeError, match="names no program"):
        cake_api._load_cake_fmha_decode_program(unrouted)


@pytest.mark.parametrize(
    ("identifier", "target", "match"),
    [
        ("NOT_A_PROGRAM", "sm100a", "publishes no CakeProgram"),
        ("RENAMED", "sm100a", "this module pins fake_renamed"),
        ("ONE_TARGET", "sm100a", "ships no 'sm_100a' build"),
        ("ARGS", "sm103a", "the host spells"),
        ("CLUSTER", "sm100a", "launches cluster"),
    ],
)
def test_cake_fmha_decode_program_load_fails_closed(
    fake_family, identifier, target, match
) -> None:
    with pytest.raises(RuntimeError, match=match):
        cake_api._load_cake_fmha_decode_program(_fake_route(identifier, target))


def test_cake_fmha_decode_program_loads_once_per_route(
    fake_family, monkeypatch
) -> None:
    """A route's program (and its reducer) is resolved and checked once; later
    launches of the route reuse the loaded module."""

    checked = []
    constant = cake_api._decode_program_constant

    def spy(family, identifier, target):
        checked.append(identifier)
        return constant(family, identifier, target)

    monkeypatch.setattr(cake_api, "_decode_program_constant", spy)
    route = _fake_route("GOOD")
    first = cake_api._load_cake_fmha_decode_program(route)
    again = cake_api._load_cake_fmha_decode_program(_fake_route("GOOD"))
    assert again is first
    assert checked == ["GOOD", "REDUCE"]
    other = cake_api._load_cake_fmha_decode_program(_fake_route("GOOD", "sm103a"))
    assert other is not first and other.target == "sm_103a"
    assert checked == ["GOOD", "REDUCE", "GOOD", "REDUCE"]
    # A program that fails its checks is not cached: every launch checks it.
    for _ in range(2):
        with pytest.raises(RuntimeError, match="launches cluster"):
            cake_api._load_cake_fmha_decode_program(_fake_route("CLUSTER"))
    assert checked[-2:] == ["CLUSTER", "CLUSTER"]


def test_cake_fmha_decode_program_launches_in_binding_order(
    monkeypatch, fake_family
) -> None:
    """The launches reach each entry in its binding's argument order, in
    stream order, and read nothing back from the device."""

    calls = []

    def builder(module, **kwargs):
        query, out = kwargs["query"], kwargs["out"]
        main = {"query": query, "out": out, "num_splits": 8}
        main.update(grid_x=128, grid_y=1, grid_z=1)
        reducer = {"query": query, "out": out, "num_splits": 8}
        reducer.update(grid_x=128, grid_y=1, grid_z=1)
        return [
            (module.program, lambda *values: calls.append(("main", values)), main),
            (
                module.reducer[3],
                lambda *values: calls.append(("reduce", values)),
                reducer,
            ),
        ]

    monkeypatch.setitem(cake_api._DECODE_PROGRAM_VALUES, _FAKE, builder)
    monkeypatch.setattr(
        cake_api, "_torch_stream_context", lambda device: contextlib.nullcontext()
    )
    module = cake_api._load_cake_fmha_decode_program(_fake_route("GOOD"))
    query = torch.zeros((4, 32, 128), dtype=torch.bfloat16)
    out = torch.empty_like(query)
    key = torch.zeros((8, 4, 16, 128), dtype=torch.bfloat16)
    kwargs = dict(
        query=query,
        key_cache=key,
        value_cache=key,
        out=out,
        workspace_buffer=torch.zeros(64, dtype=torch.uint8),
        counter_buffer=None,
        block_tables=torch.zeros((4, 2), dtype=torch.int32),
        seq_lens=torch.full((4,), 32, dtype=torch.int32),
        max_seq_len=32,
        bmm1_scale=0.125,
        bmm2_scale=1.0,
        sinks=None,
        lse=None,
        key_block_scales=None,
        value_block_scales=None,
        sm_count=148,
    )
    with no_host_sync():
        cake_api._run_cake_fmha_decode_program(module, **kwargs)
    assert [kind for kind, _ in calls] == ["main", "reduce"]
    for _, values in calls:
        assert values[0] is query and values[1] is out
        assert values[2:] == (8, 128, 1, 1)
    # A launch whose shapes are not its route's is refused.
    with pytest.raises(RuntimeError, match="does not match its selected route"):
        tables = torch.zeros((3, 2), dtype=torch.int32)
        cake_api._run_cake_fmha_decode_program(
            module, **{**kwargs, "block_tables": tables}
        )


# --- GPU ---------------------------------------------------------------------

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)


@_GPU
@pytest.mark.parametrize("max_seq_len", (0, 300))
@pytest.mark.parametrize("dtype", ("bf16", "fp8"))
def test_cake_decode_batch_without_query_tokens_returns_before_any_launch(
    monkeypatch, dtype, max_seq_len
) -> None:
    """A decode call with no query token returns its empty output (and LSE)
    without choosing a route or launching a kernel."""

    def refuse(*args, **kwargs):
        raise AssertionError("a decode call without query tokens chose a route")

    monkeypatch.setattr(cake_api, "select_cake_fmha_decode_route", refuse)
    monkeypatch.setattr(cake_api, "_resolve_cake_fmha_decode_module", refuse)
    device = torch.device("cuda")
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float8_e4m3fn
    query = torch.empty((0, 32, 128), dtype=torch_dtype, device=device)
    kv_cache = torch.zeros((8, 2, 4, 16, 128), dtype=torch_dtype, device=device)
    workspace = torch.zeros(1 << 20, dtype=torch.uint8, device=device)
    counter = torch.zeros(1 << 12, dtype=torch.uint8, device=device)
    out = torch.empty_like(query)
    result, lse = decode_api.trtllm_batch_decode_with_kv_cache(
        query,
        kv_cache,
        workspace,
        torch.zeros((0, 32), dtype=torch.int32, device=device),
        torch.zeros((0,), dtype=torch.int32, device=device),
        max_seq_len,
        bmm1_scale=128**-0.5,
        bmm2_scale=1.0,
        out=out,
        kv_layout="HND",
        backend="cake",
        return_lse=True,
        multi_ctas_kv_counter_buffer=counter,
    )
    assert result is out and out.shape[0] == 0
    assert lse.shape == (0, 32)
    assert not counter.any()
