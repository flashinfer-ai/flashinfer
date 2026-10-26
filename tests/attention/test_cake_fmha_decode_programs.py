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
import dataclasses
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
            *cake_api._decode_program_host_params(family, launch),
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
    # Stream-K: whole groups of q_len x num_kv_heads CTAs over the SMs, at
    # least two requests per group.
    assert grid("streamk", None, **shape(128), sm_count=148) == (148, 1, 1)
    assert grid("streamk", None, **shape(128), sm_count=152) == (152, 1, 1)
    assert grid("streamk", None, **shape(128), sm_count=150) == (148, 1, 1)
    assert grid("streamk", None, **shape(74), sm_count=148) == (148, 1, 1)
    for batch, sm_count in ((73, 148), (75, 152), (8, 148)):
        with pytest.raises(RuntimeError, match="admits batch_size >= 2 x its CTA"):
            grid("streamk", None, **shape(batch), sm_count=sm_count)
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


# --- Shape rows through route selection (host-only) ---------------------------

# The shape rows: name -> (family, kind, batch, q_len, kv_len, num_q_heads,
# num_kv_heads, page_size, the program per target).  ``kind`` names the
# dtypes and head dim of the launch (see ``_decode_kwargs``).
SHAPE_ROWS: dict[str, tuple] = {
    "nvfp4_b128_s8192": (
        "decode_nvfp4", "nvfp4", 128, 1, 8192, 32, 4, 16,
        {"sm100a": "DEC_NVFP4_B128", "sm103a": "DEC_NVFP4_B128"},
    ),
    "fp8_b256_s8192": (
        "decode_fp8", "fp8", 256, 1, 8192, 32, 4, 16,
        {"sm100a": "B256_SM100A", "sm103a": "B256_SM103A"},
    ),
    "fp8_b4_s16384": (
        "decode_fp8", "fp8", 4, 1, 16384, 32, 4, 32,
        {"sm100a": "B4_S8X16_SM100A", "sm103a": "B4_S8X16_SM103A"},
    ),
    "bf16_b128_s4096": (
        "decode_bf16", "bf16", 128, 1, 4096, 32, 4, 16,
        {"sm100a": "DEC_B128", "sm103a": "DEC_B128"},
    ),
    "bf16_b4_s1024": (
        "decode_bf16", "bf16", 4, 1, 1024, 32, 4, 16,
        {"sm100a": "DEC_B4C4", "sm103a": "DEC_B4C4"},
    ),
    "bf16_b256_s4096_sink": (
        "decode_bf16", "bf16_sink", 256, 1, 4096, 32, 4, 16,
        {"sm100a": "DEC_SINK", "sm103a": "DEC_SINK"},
    ),
    "smallm_kv60008_q8_p64": (
        "decode_smallm", "smallm", 1, 8, 60008, 8, 1, 64,
        {"sm100a": "N64_P64_Q8_S148", "sm103a": "N64_P64_Q8_S152"},
    ),
    "smallm_kv60007_q7_p64": (
        "decode_smallm", "smallm", 1, 7, 60007, 8, 1, 64,
        {"sm100a": "N64_P64_Q7_S148", "sm103a": "N64_P64_Q7_S152"},
    ),
    "smallm_kv60008_q8_p16": (
        "decode_smallm", "smallm", 1, 8, 60008, 8, 1, 16,
        {"sm100a": "N64_P16_Q8_S148", "sm103a": "N64_P16_Q8_S152"},
    ),
    "smallm_kv60004_q4_p64": (
        "decode_smallm", "smallm", 1, 4, 60004, 8, 1, 64,
        {"sm100a": "N32_P64_Q4_S148", "sm103a": "N32_P64_Q4_S152"},
    ),
    "smallm_kv60003_q3_p64": (
        "decode_smallm", "smallm", 1, 3, 60003, 8, 1, 64,
        {"sm100a": "N32_P64_Q3_S148", "sm103a": "N32_P64_Q3_S152"},
    ),
    "smallm_tp1_kv60004_q4_p64": (
        "decode_smallm", "smallm", 1, 4, 60004, 16, 2, 64,
        {"sm100a": "N32_P64_Q4_S74", "sm103a": "N32_P64_Q4_S76"},
    ),
    "smallm_kv8200_q8_p64": (
        "decode_smallm", "smallm", 1, 8, 8200, 8, 1, 64,
        {"sm100a": "N64_P64_Q8_S65", "sm103a": "N64_P64_Q8_S65"},
    ),
    "smallm_kv8199_q7_p64": (
        "decode_smallm", "smallm", 1, 7, 8199, 8, 1, 64,
        {"sm100a": "N64_P64_Q7_S65", "sm103a": "N64_P64_Q7_S65"},
    ),
    "smallm_kv1032_q8_p64": (
        "decode_smallm", "smallm", 1, 8, 1032, 8, 1, 64,
        {"sm100a": "N64_P64_Q8_C9", "sm103a": "N64_P64_Q8_C9"},
    ),
}  # fmt: skip

_SM_COUNT = {"sm100a": 148, "sm103a": 152}
_QUANT_KINDS = ("fp8", "nvfp4")
_HOST_BMM1 = 128**-0.5


def _decode_kwargs(
    kind,
    batch,
    q_len,
    kv_len,
    num_q_heads,
    num_kv_heads,
    page_size,
    *,
    device="meta",
    device_scales=False,
    width=None,
    max_seq_len=None,
    workspace_bytes=1 << 28,
):
    """``select_cake_fmha_decode_route`` inputs of one launch: empty tensors
    of its shapes (the selector reads shapes, strides, dtypes and pointer
    alignment only).  K / V are the two halves of one stacked cache (the
    contract's form) except NVFP4's separate code and scale pools."""

    head_dim = 256 if kind == "smallm" else 128
    pages = -(-kv_len // page_size)
    width = pages if width is None else width
    total_pages = batch * max(width, 1)
    bf16 = kind in ("bf16", "bf16_sink", "smallm")
    q_dtype = torch.bfloat16 if bf16 else torch.float8_e4m3fn
    query = torch.empty(
        (batch * q_len, num_q_heads, head_dim), dtype=q_dtype, device=device
    )
    key_scales = value_scales = None
    if kind == "nvfp4":
        shape = (total_pages, num_kv_heads, page_size, 64)
        key = torch.empty(shape, dtype=torch.uint8, device=device)
        value = torch.empty(shape, dtype=torch.uint8, device=device)
        scale_shape = (total_pages, num_kv_heads, page_size, 8)
        key_scales = torch.empty(scale_shape, dtype=torch.float8_e4m3fn, device=device)
        value_scales = torch.empty_like(key_scales)
    else:
        stacked = torch.empty(
            (total_pages, 2, num_kv_heads, page_size, head_dim),
            dtype=q_dtype,
            device=device,
        )
        key, value = stacked[:, 0], stacked[:, 1]
    bmm2 = 1.0 / 0.75 if kind in _QUANT_KINDS else 1.0

    def scale(host):
        if not device_scales:
            return host
        return torch.empty((1,), dtype=torch.float32, device=device)

    return dict(
        query=query,
        key_cache=key,
        value_cache=value,
        out=torch.empty_like(query),
        workspace_buffer=torch.zeros(workspace_bytes, dtype=torch.uint8, device=device),
        block_tables=torch.zeros((batch, width), dtype=torch.int32, device=device),
        seq_lens=torch.full((batch,), kv_len, dtype=torch.int32, device=device),
        batch_size=batch,
        q_len=q_len,
        max_seq_len=kv_len if max_seq_len is None else max_seq_len,
        window_left=-1,
        bmm1_scale=scale(_HOST_BMM1),
        bmm2_scale=scale(bmm2),
        o_scale=0.75 if kind in _QUANT_KINDS else 1.0,
        sinks=(
            torch.empty((num_q_heads,), dtype=torch.float32, device=device)
            if kind == "bf16_sink"
            else None
        ),
        kv_layout="HND",
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=None,
        key_block_scales=key_scales,
        value_block_scales=value_scales,
        skip_softmax_threshold_scale_factor=None,
        enable_block_sparse_attention=False,
        lse=None,
        multi_ctas_kv_counter_buffer=None,
        enable_pdl=True,
    )


def _row_kwargs(row, **overrides):
    _, kind, batch, q_len, kv_len, hq, hkv, page, _ = SHAPE_ROWS[row]
    shape = dict(
        batch=batch,
        q_len=q_len,
        kv_len=kv_len,
        num_q_heads=hq,
        num_kv_heads=hkv,
        page_size=page,
    )
    for name in tuple(overrides):
        if name in shape:
            shape[name] = overrides.pop(name)
    return _decode_kwargs(kind, *shape.values(), **overrides)


@pytest.fixture
def decode_target(monkeypatch):
    """Pin the selection's target and SM count (meta tensors have neither);
    returns a setter."""

    def pin(target, sm_count=None):
        monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
        count = _SM_COUNT[target] if sm_count is None else sm_count
        monkeypatch.setattr(cake_api, "_smallm_sm_count", lambda device: count)
        monkeypatch.setattr(
            cake_api, "_decode_program_toolkit_at_least", lambda version: True
        )

    return pin


def _select(kwargs):
    return cake_api.select_cake_fmha_decode_route(torch.device("meta"), **kwargs)


# The program a launch with device scales takes, per host-scalar program
# (a family whose programs read both forms takes the same program; a family
# without device scales takes none).
_DEVICE_SCALE_PROGRAM: dict[tuple[str, str], str | None] = {
    ("decode_bf16", "DEC_B128"): "DEC_B128_DS",
    ("decode_bf16", "DEC_B4C4"): "DEC_B4C4_DS",
    ("decode_bf16", "DEC_SINK"): "DEC_SINK_DS",
}


def _device_scale_program(family, identifier):
    if family in ("decode_nvfp4", "decode_fp8"):
        return identifier
    return _DEVICE_SCALE_PROGRAM.get((family, identifier))


@pytest.mark.parametrize("device_scales", (False, True))
@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_cake_fmha_decode_route_serves_shape_rows_as_pinned(
    decode_target, row, target, device_scales
) -> None:
    decode_target(target)
    family = SHAPE_ROWS[row][0]
    identifier = SHAPE_ROWS[row][-1][target]
    route = _select(_row_kwargs(row, device_scales=device_scales))
    if device_scales:
        identifier = _device_scale_program(family, identifier)
        if identifier is None:
            assert route is None or route.program is None
            return
    assert route is not None
    assert route.program == (family, identifier)
    assert route.device_scales is device_scales
    assert cake_api.cake_fmha_route_is_optimized(route)


# --- NVFP4 boundaries (host-only) --------------------------------------------


def test_cake_fmha_decode_nvfp4_toolkit_gate_keeps_the_jit_member(
    decode_target, monkeypatch
) -> None:
    decode_target("sm100a")
    monkeypatch.setattr(
        cake_api, "_decode_program_toolkit_at_least", lambda version: False
    )
    route = _select(_row_kwargs("nvfp4_b128_s8192"))
    assert route is not None and route.component == "decode_quant_nvfp4"
    assert route.program is None


@pytest.mark.parametrize(
    ("overrides", "hit"),
    [
        # An odd 128-token block count reads the evened width: the caller's
        # table must hold it (a tight table misses).
        ({"kv_len": 8000, "width": 512}, True),
        ({"kv_len": 8000}, False),
        ({"kv_len": 8064, "width": 512}, True),
        ({"kv_len": 1, "width": 16}, True),
        ({"kv_len": 1}, False),
        # One token more reads one block pair more.
        ({"kv_len": 8192, "max_seq_len": 8193}, False),
        ({"kv_len": 8193, "width": 528}, True),
        ({"kv_len": 8192, "width": 511}, False),
        ({"kv_len": 8192, "width": 600}, True),
        ({"batch": 127}, False),
        ({"batch": 129}, False),
        ({"page_size": 32}, False),
    ],
)
def test_cake_fmha_decode_nvfp4_boundaries(decode_target, overrides, hit) -> None:
    decode_target("sm103a")
    route = _select(_row_kwargs("nvfp4_b128_s8192", **overrides))
    if hit:
        assert route.program == ("decode_nvfp4", "DEC_NVFP4_B128"), overrides
    else:
        assert route is None or route.program is None, overrides


@pytest.mark.parametrize(
    "mutation",
    (
        "mixed_scales",
        "cpu_scale",
        "strided_query",
        "separate_tables",
        "int64_lengths",
        "small_workspace",
    ),
)
def test_cake_fmha_decode_nvfp4_misses_keep_the_jit_route(
    decode_target, mutation
) -> None:
    decode_target("sm100a")
    kwargs = _row_kwargs("nvfp4_b128_s8192")
    if mutation == "mixed_scales":
        kwargs["bmm1_scale"] = torch.empty((1,), dtype=torch.float32, device="meta")
    elif mutation == "cpu_scale":
        kwargs["bmm1_scale"] = torch.ones((1,), dtype=torch.float32)
        kwargs["bmm2_scale"] = torch.ones((1,), dtype=torch.float32)
    elif mutation == "strided_query":
        query = kwargs["query"]
        wide = torch.empty(
            (query.shape[0], query.shape[1], 256), dtype=query.dtype, device="meta"
        )
        kwargs["query"] = wide[..., :128]
    elif mutation == "separate_tables":
        kwargs["uses_shared_paged_kv_idx"] = False
        kwargs["block_tables"] = torch.zeros(
            (128, 2, 512), dtype=torch.int32, device="meta"
        )
    elif mutation == "int64_lengths":
        kwargs["seq_lens"] = kwargs["seq_lens"].to(torch.int64)
    else:
        small = torch.empty(1024, dtype=torch.uint8, device="meta")
        kwargs["workspace_buffer"] = small
    route = _select(kwargs)
    assert route is None or route.program is None, mutation


# --- The values a quantized program receives (host-only) ---------------------


def _program_values(
    monkeypatch,
    row,
    *,
    device_scales,
    sm_count=148,
    counter_buffer=None,
    rerun=False,
    **overrides,
):
    """The launches a stand-in program of ``row``'s route receives on CPU
    tensors: ``[(name, values)]`` in stream order, under the no-sync spy.
    With ``rerun``, also a function that launches the same module again with
    some launch arguments replaced and returns those launches."""

    from flashinfer.jit.cake_program import GRID_ARGS

    target = "sm100a" if sm_count == 148 else "sm103a"
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    monkeypatch.setattr(cake_api, "_smallm_sm_count", lambda device: sm_count)
    monkeypatch.setattr(
        cake_api, "_decode_program_toolkit_at_least", lambda version: True
    )
    kwargs = _row_kwargs(
        row,
        device="cpu",
        device_scales=device_scales,
        workspace_bytes=1 << 26,
        **overrides,
    )
    kwargs["multi_ctas_kv_counter_buffer"] = counter_buffer
    # Poison the workspace: whatever a launch needs zeroed, it zeroes.
    kwargs["workspace_buffer"].fill_(0xA5)
    route = cake_api.select_cake_fmha_decode_route(torch.device("cpu"), **kwargs)
    family, identifier = route.program
    calls = []

    def record(key):
        name, launch = cake_api._DECODE_PROGRAMS[key]
        args = (*cake_api._decode_program_host_params(key[0], launch), *GRID_ARGS)
        cluster = tuple(getattr(_loader(key[0]), key[1]).cluster)
        program = SimpleNamespace(name=name, args=args, cluster=cluster)
        return launch, program, lambda *values: calls.append((name, args, values))

    launch, program, entry = record((family, identifier))
    reducer = None
    reducer_key = cake_api._DECODE_PROGRAM_REDUCERS.get((family, identifier))
    if reducer_key is not None:
        reducer = (*reducer_key, *record(reducer_key))
    module = cake_api.CakeFmhaDecodeProgramModule(
        route=route,
        family=family,
        identifier=identifier,
        launch=launch,
        program=program,
        target=_TARGET_ARCH[target],
        entry=entry,
        reducer=reducer,
    )
    monkeypatch.setattr(
        cake_api, "_torch_stream_context", lambda device: contextlib.nullcontext()
    )
    launch_kwargs = dict(
        query=kwargs["query"],
        key_cache=kwargs["key_cache"],
        value_cache=kwargs["value_cache"],
        out=kwargs["out"],
        workspace_buffer=kwargs["workspace_buffer"],
        counter_buffer=kwargs["multi_ctas_kv_counter_buffer"],
        block_tables=kwargs["block_tables"],
        seq_lens=kwargs["seq_lens"],
        max_seq_len=kwargs["max_seq_len"],
        bmm1_scale=kwargs["bmm1_scale"],
        bmm2_scale=kwargs["bmm2_scale"],
        sinks=kwargs["sinks"],
        lse=kwargs["lse"],
        key_block_scales=kwargs["key_block_scales"],
        value_block_scales=kwargs["value_block_scales"],
        sm_count=sm_count,
    )

    def launch(**replaced):
        calls.clear()
        with no_host_sync():
            cake_api._run_cake_fmha_decode_program(
                module, **{**launch_kwargs, **replaced}
            )
        return [
            (name, dict(zip(args, values, strict=True))) for name, args, values in calls
        ]

    launches = launch()
    if rerun:
        return kwargs, launches, launch
    return kwargs, launches


def _same_storage(view, tensor):
    return view.data_ptr() == tensor.data_ptr()


@pytest.mark.parametrize("device_scales", (False, True))
def test_cake_decode_nvfp4_program_values_view_the_callers_tensors(
    monkeypatch, device_scales
) -> None:
    """One launch over views of the caller's tensors (no copy, no padding):
    the table in place with its width as the row stride, the partials in the
    workspace after the reserved scalar slots, null scale pointers with host
    scalars or the caller's FP32 scalars."""

    kwargs, launches = _program_values(
        monkeypatch, "nvfp4_b128_s8192", device_scales=device_scales
    )
    ((name, values),) = launches
    assert name == cake_api._DECODE_PROGRAMS[("decode_nvfp4", "DEC_NVFP4_B128")][0]
    assert values["Qt"].dtype == torch.uint8 and values["Qt"].shape == (128 * 32, 128)
    assert _same_storage(values["Qt"], kwargs["query"])
    assert values["Kp"] is kwargs["key_cache"] and values["Vp"] is kwargs["value_cache"]
    assert values["Ksf"].dtype == torch.uint8
    assert _same_storage(values["Ksf"], kwargs["key_block_scales"])
    assert _same_storage(values["Vsf"], kwargs["value_block_scales"])
    assert values["O"].dtype == torch.uint8
    assert _same_storage(values["O"], kwargs["out"])
    assert values["page_table"] is kwargs["block_tables"]
    assert values["seq_lens_kv"] is kwargs["seq_lens"]
    assert (values["pt_batch_stride"], values["pt_v_offset"]) == (512, 0)
    assert (values["num_splits"], values["blocks_per_split"]) == (1, 64)
    rows = 128 * 32
    workspace = kwargs["workspace_buffer"].data_ptr()
    first = 0 if device_scales else 16
    assert values["partial_O"].data_ptr() == workspace + first
    assert values["partial_O"].numel() == rows * 128
    assert values["partial_max"].data_ptr() == workspace + first + rows * 512
    assert values["partial_sum"].data_ptr() == workspace + first + rows * 516
    if device_scales:
        assert _same_storage(values["bmm1_scale_ptr"], kwargs["bmm1_scale"])
        assert _same_storage(values["bmm2_scale_ptr"], kwargs["bmm2_scale"])
        assert values["bmm1_is_log2"] == 1
    else:
        assert values["bmm1_scale_ptr"] is None and values["bmm2_scale_ptr"] is None
        assert values["bmm1_is_log2"] == 0
        assert values["bmm1_scalar"] == _HOST_BMM1
        assert values["bmm2_scalar"] == 1.0 / 0.75
    assert (values["grid_x"], values["grid_y"], values["grid_z"]) == (128, 1, 1)


# --- GPU: shared helpers -----------------------------------------------------


@pytest.fixture
def served_programs(monkeypatch):
    """The generated programs the decode dispatch runs, in call order (the
    resolve, not the loader: a route loaded once skips the loader)."""

    served = []
    resolve = cake_api._resolve_cake_fmha_decode_module

    def spy(device, route):
        module, loaded = resolve(device, route)
        if isinstance(module, cake_api.CakeFmhaDecodeProgramModule):
            served.append(route.program)
        return module, loaded

    monkeypatch.setattr(cake_api, "_resolve_cake_fmha_decode_module", spy)
    return served


def _keep_the_jit_route(monkeypatch):
    """Route every launch outside the program tables (the JIT members)."""

    for target in tuple(cake_api._DECODE_PROGRAM_ROUTES):
        monkeypatch.setitem(cake_api._DECODE_PROGRAM_ROUTES, target, {})


def _bits(tensor):
    dtype = torch.uint8 if tensor.element_size() == 1 else torch.int16
    return tensor.view(dtype)


def _run_decode(case, *, seq_lens=None, out=None, max_seq_len=None, lse=None):
    return decode_api.trtllm_batch_decode_with_kv_cache(
        case.query,
        case.kv_cache,
        case.workspace,
        case.block_tables,
        case.seq_lens if seq_lens is None else seq_lens,
        case.max_seq_len if max_seq_len is None else max_seq_len,
        bmm1_scale=case.bmm1,
        bmm2_scale=case.bmm2,
        out=torch.empty_like(case.out) if out is None else out,
        kv_layout="HND",
        backend="cake",
        kv_cache_sf=case.kv_cache_sf,
        o_scale=case.o_scale,
        q_len_per_req=case.q_len,
        sinks=case.sinks,
        enable_pdl=True,
        multi_ctas_kv_counter_buffer=case.counter,
        lse=lse,
    )


def _check_hit_is_sync_free_and_replays(case, lengths):
    """The hit reads nothing back (the spy), and a CUDA graph captured at the
    case's lengths replays new lengths bitwise to a direct launch."""

    run = getattr(case, "run", _run_decode)
    run(case)
    torch.cuda.synchronize()
    with no_host_sync():
        run(case)
    seq_lens = case.seq_lens.clone()
    out = torch.empty_like(case.out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(case, seq_lens=seq_lens, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(case, seq_lens=seq_lens, out=out)
    seq_lens.copy_(lengths)
    graph.replay()
    direct = run(case, seq_lens=lengths)
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(direct))


# --- GPU: NVFP4 --------------------------------------------------------------

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nvfp4_case(batch, kv_len, *, seed, lengths=None, device_scales=False):
    """FP8 queries over NVFP4 K / V pages (random codes, block scales from
    {0.5, 1, 2}, V's in FlashInfer's swizzled order), shuffled pages."""

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    hq, hkv, page = 32, 4, 16
    width = -(-kv_len // page)
    total = batch * width

    def randint(high, *shape):
        return torch.randint(
            0, high, shape, generator=generator, device=device, dtype=torch.int64
        )

    scale_values = torch.tensor((0.5, 1.0, 2.0), device=device)
    pools = []
    for _ in range(2):
        codes = randint(256, total, hkv, page, 64).to(torch.uint8)
        scales = scale_values[randint(3, total, hkv, page, 8)]
        pools.append((codes, scales.to(torch.float8_e4m3fn)))
    query = (
        torch.randn((batch, hq, 128), generator=generator, device=device) * 2.0
    ).to(torch.float8_e4m3fn)
    order = torch.randperm(total, generator=generator, device=device)
    block_tables = order.view(batch, width).to(torch.int32).contiguous()
    if lengths is None:
        lengths = [kv_len] * batch
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    bmm1, bmm2 = 128**-0.5, 1.0 / 0.75
    if device_scales:
        bmm1 = torch.tensor([bmm1], dtype=torch.float32, device=device)
        bmm2 = torch.tensor([bmm2], dtype=torch.float32, device=device)
    return SimpleNamespace(
        query=query,
        kv_cache=(pools[0][0], pools[1][0]),
        kv_cache_sf=(pools[0][1], pools[1][1]),
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=kv_len,
        bmm1=bmm1,
        bmm2=bmm2,
        o_scale=0.75,
        q_len=1,
        sinks=None,
        out=torch.empty_like(query),
        workspace=torch.zeros(1 << 28, dtype=torch.uint8, device=device),
        counter=torch.zeros(1 << 16, dtype=torch.uint8, device=device),
    )


def _nvfp4_dequantize(codes, scales, *, swizzled):
    """``[..., 16, 64]`` packed codes and ``[..., 16, 8]`` scales of whole
    pages to ``[..., 16, 128]`` FP32 values."""

    if swizzled:
        token = torch.arange(16, device=codes.device)[:, None]
        group = torch.arange(8, device=codes.device)[None, :]
        rows = (token & ~3) | (group >> 1)
        cols = ((group & 1) << 2) | (token & 3)
        scales = scales[..., rows, cols]
    table = torch.tensor(_E2M1, device=codes.device)
    low, high = (codes & 7).long(), ((codes >> 4) & 7).long()
    sign_low, sign_high = (codes & 8) != 0, (codes & 128) != 0
    values = torch.stack(
        (
            torch.where(sign_low, -table[low], table[low]),
            torch.where(sign_high, -table[high], table[high]),
        ),
        dim=-1,
    ).flatten(-2)
    return values * scales.float().repeat_interleave(16, dim=-1)


def _nvfp4_reference(case, batches):
    """FP32 ``softmax(bmm1 Q K^T) V bmm2`` of the given requests."""

    bmm1, bmm2 = (
        scale.item() if isinstance(scale, torch.Tensor) else float(scale)
        for scale in (case.bmm1, case.bmm2)
    )
    outputs = {}
    for b in batches:
        length = int(case.seq_lens[b])
        if length == 0:
            outputs[b] = torch.zeros((32, 128), device=case.query.device)
            continue
        pages = case.block_tables[b, : -(-length // 16)].long()
        k = _nvfp4_dequantize(
            case.kv_cache[0][pages], case.kv_cache_sf[0][pages], swizzled=False
        )
        v = _nvfp4_dequantize(
            case.kv_cache[1][pages], case.kv_cache_sf[1][pages], swizzled=True
        )
        k = k.transpose(0, 1).reshape(4, -1, 128)[:, :length]
        v = v.transpose(0, 1).reshape(4, -1, 128)[:, :length]
        q = case.query[b].float().view(4, 8, 128)
        p = torch.softmax(q @ k.transpose(1, 2) * bmm1, dim=-1)
        outputs[b] = (p @ v).reshape(32, 128) * bmm2
    return outputs


@_GPU
@pytest.mark.parametrize("device_scales", (False, True))
@pytest.mark.parametrize("lengths", ("uniform", "ragged"))
def test_cake_decode_nvfp4_program_matches_reference_and_the_jit_route(
    monkeypatch, served_programs, device_scales, lengths
) -> None:
    from flashinfer.jit.cpp_ext import is_cuda_version_at_least

    if not is_cuda_version_at_least("13.4"):
        pytest.skip("the NVFP4 decode program needs the CUDA 13.4 toolkit")
    ragged = [8192, 0, 1, 129, 4097, 8191, 255, 8000] * 16
    case = _nvfp4_case(
        128,
        8192,
        seed=20261026,
        lengths=None if lengths == "uniform" else ragged,
        device_scales=device_scales,
    )
    got = _run_decode(case)
    assert served_programs == [("decode_nvfp4", "DEC_NVFP4_B128")]
    batches = (0, 1, 2, 3, 5, 77, 127)
    reference = _nvfp4_reference(case, batches)
    for b in batches:
        torch.testing.assert_close(
            got[b].float(), reference[b], atol=1.0, rtol=0.1, msg=f"request {b}"
        )
    _keep_the_jit_route(monkeypatch)
    jit = _run_decode(case)
    assert len(served_programs) == 1
    assert torch.equal(_bits(got), _bits(jit))


@_GPU
def test_cake_decode_nvfp4_program_is_sync_free_and_replays(served_programs) -> None:
    from flashinfer.jit.cpp_ext import is_cuda_version_at_least

    if not is_cuda_version_at_least("13.4"):
        pytest.skip("the NVFP4 decode program needs the CUDA 13.4 toolkit")
    case = _nvfp4_case(128, 8192, seed=20261027, device_scales=True)
    lengths = torch.randint(
        0, 8193, (128,), dtype=torch.int32, device=case.seq_lens.device
    )
    _check_hit_is_sync_free_and_replays(case, lengths)
    assert set(served_programs) == {("decode_nvfp4", "DEC_NVFP4_B128")}


# --- Small-M boundaries (host-only) -------------------------------------------

_TWO_KV_HEADS = {"num_q_heads": 16, "num_kv_heads": 2}


@pytest.mark.parametrize(
    ("target", "row", "overrides", "program"),
    [
        # The cluster program takes one tile of exactly nine 128-token blocks.
        ("sm100a", "smallm_kv1032_q8_p64", {"kv_len": 1025}, "N64_P64_Q8_C9"),
        ("sm103a", "smallm_kv1032_q8_p64", {"kv_len": 1152}, "N64_P64_Q8_C9"),
        ("sm100a", "smallm_kv1032_q8_p64", {"kv_len": 1024}, None),
        ("sm100a", "smallm_kv1032_q8_p64", {"kv_len": 1153}, None),
        ("sm100a", "smallm_kv1032_q8_p64", {"batch": 2}, None),
        ("sm103a", "smallm_kv1032_q8_p64", _TWO_KV_HEADS, None),
        ("sm100a", "smallm_kv1032_q8_p64", {"q_len": 7}, None),
        ("sm100a", "smallm_kv1032_q8_p64", {"page_size": 16}, None),
        # Sixty-five blocks split 65 ways; 64 and 66 have no program.
        ("sm100a", "smallm_kv8200_q8_p64", {"kv_len": 8320}, "N64_P64_Q8_S65"),
        ("sm103a", "smallm_kv8200_q8_p64", {"kv_len": 8193}, "N64_P64_Q8_S65"),
        ("sm100a", "smallm_kv8200_q8_p64", {"kv_len": 8192}, None),
        ("sm100a", "smallm_kv8200_q8_p64", {"kv_len": 8321}, None),
        # Two tiles of 64 rows split 74 ways: no program.
        ("sm100a", "smallm_kv60008_q8_p64", {"batch": 2}, None),
        ("sm100a", "smallm_kv60008_q8_p64", _TWO_KV_HEADS, None),
    ],
)
def test_cake_fmha_decode_smallm_boundaries(
    decode_target, target, row, overrides, program
) -> None:
    decode_target(target)
    route = _select(_row_kwargs(row, **overrides))
    if program is None:
        assert route is None or route.program is None, overrides
    else:
        assert route.program == ("decode_smallm", program), overrides


def test_cake_fmha_decode_smallm_split_count_follows_the_device(
    decode_target,
) -> None:
    """A device with another SM count plans another split: an SM103 part with
    148 SMs has no 148-way program; 152 SMs on SM100 neither."""

    decode_target("sm103a", sm_count=148)
    assert _select(_row_kwargs("smallm_kv60008_q8_p64")).program is None
    decode_target("sm100a", sm_count=152)
    assert _select(_row_kwargs("smallm_kv60008_q8_p64")).program is None


@pytest.mark.parametrize("mutation", ("uint32_tables", "lse", "device_scales"))
def test_cake_fmha_decode_smallm_scale_table_and_lse_forms(
    decode_target, mutation
) -> None:
    decode_target("sm100a")
    kwargs = _row_kwargs("smallm_kv8200_q8_p64")
    if mutation == "uint32_tables":
        kwargs["block_tables"] = kwargs["block_tables"].to(torch.uint32)
        route = _select(kwargs)
        assert route is None or route.program is None
    elif mutation == "lse":
        query = kwargs["query"]
        kwargs["lse"] = torch.empty(
            (query.shape[0], query.shape[1]), dtype=torch.float32, device="meta"
        )
        assert _select(kwargs).program == ("decode_smallm", "N64_P64_Q8_S65")
    else:
        kwargs = _row_kwargs("smallm_kv8200_q8_p64", device_scales=True)
        route = _select(kwargs)
        assert route is None or route.program is None


@pytest.mark.parametrize("counters", ("caller", "workspace"))
@pytest.mark.parametrize("row", ("smallm_kv8200_q8_p64", "smallm_kv1032_q8_p64"))
def test_cake_decode_smallm_program_values(monkeypatch, row, counters) -> None:
    """The JIT route's workspace layout; the merge tickets in the caller's
    counter buffer, or zeroed in the workspace when it passes none."""

    caller = torch.full((64,), 7, dtype=torch.uint8) if counters == "caller" else None
    kwargs, launches = _program_values(
        monkeypatch, row, device_scales=False, counter_buffer=caller
    )
    ((_, values),) = launches
    num_split = 9 if row == "smallm_kv1032_q8_p64" else 65
    slots, n_rows = num_split, 64
    workspace = kwargs["workspace_buffer"]
    base = workspace.data_ptr()
    assert values["Q"] is kwargs["query"] and values["O_ptr"] is kwargs["out"]
    assert values["K"] is kwargs["key_cache"] and values["V"] is kwargs["value_cache"]
    assert values["partial_O_ptr"].data_ptr() == base
    assert values["partial_O_ptr"].dtype == torch.bfloat16
    lse_slots = (slots * n_rows * 512 + 255) // 256 * 256
    assert values["partial_LSE_ptr"].data_ptr() == base + lse_slots
    tickets = (lse_slots + slots * n_rows * 4 + 255) // 256 * 256
    if counters == "caller":
        assert values["counters"].data_ptr() == caller.data_ptr()
        assert values["counters"].numel() == 2
    else:
        assert values["counters"].data_ptr() == base + tickets
        assert not values["counters"].any()
    assert values["LSE_ptr"].data_ptr() == base + (tickets + 8 + 255) // 256 * 256
    assert values["page_table"] is kwargs["block_tables"]
    assert values["max_pages_per_seq"] == kwargs["block_tables"].shape[1]
    assert values["softmax_scale_log2"] == _HOST_BMM1 * 1.4426950408889634
    assert (values["num_q_heads"], values["num_kv_heads"]) == (8, 1)
    assert (values["grid_x"], values["grid_y"], values["grid_z"]) == (num_split, 1, 1)


@pytest.mark.parametrize("counters", ("caller", "workspace"))
def test_cake_decode_smallm_program_values_follow_the_buffers(
    monkeypatch, counters
) -> None:
    """A launch on the module's last workspace and counter buffer reuses their
    argument views; the caller's other tensors pass as given, the workspace
    tickets are zeroed at every launch, and another workspace takes its own
    views."""

    caller = torch.zeros((64,), dtype=torch.uint8) if counters == "caller" else None
    kwargs, launches, launch = _program_values(
        monkeypatch,
        "smallm_kv8200_q8_p64",
        device_scales=False,
        counter_buffer=caller,
        rerun=True,
    )
    ((_, first),) = launch()
    workspace = kwargs["workspace_buffer"]
    workspace.fill_(0xA5)
    query, out = torch.empty_like(kwargs["query"]), torch.empty_like(kwargs["out"])
    ((_, second),) = launch(query=query, out=out)
    for name in ("partial_O_ptr", "partial_LSE_ptr", "counters", "LSE_ptr"):
        assert second[name] is first[name], name
    assert second["Q"] is query and second["O_ptr"] is out
    if counters == "workspace":
        assert not second["counters"].any()
    lse = torch.empty((query.shape[0], query.shape[1]), dtype=torch.float32)
    ((_, with_lse),) = launch(lse=lse)
    assert with_lse["LSE_ptr"] is lse
    for name in ("partial_O_ptr", "partial_LSE_ptr", "counters"):
        assert with_lse[name].data_ptr() == first[name].data_ptr(), name
    ((_, third),) = launch()
    assert third["partial_O_ptr"] is first["partial_O_ptr"]
    assert launches[0][1]["partial_O_ptr"].data_ptr() == workspace.data_ptr()
    other = torch.full_like(workspace, 0xA5)
    ((_, fourth),) = launch(workspace_buffer=other)
    assert fourth["partial_O_ptr"].data_ptr() == other.data_ptr()
    assert fourth["partial_LSE_ptr"].data_ptr() != first["partial_LSE_ptr"].data_ptr()
    if caller is not None:
        assert fourth["counters"].data_ptr() == caller.data_ptr()
    else:
        assert fourth["counters"].data_ptr() != first["counters"].data_ptr()


@pytest.mark.parametrize(
    ("batch", "max_seq_len", "cluster"),
    [
        (1, 1024, False),
        (1, 1025, True),
        (1, 1152, True),
        (1, 1153, False),
        (2, 1100, False),
    ],
)
def test_cake_decode_smallm_cluster_program_precondition(
    monkeypatch, batch, max_seq_len, cluster
) -> None:
    """The nine-CTA cluster program serves one tile (one request, one KV
    head) whose KV is more than eight and at most nine 128-token blocks; its
    launch refuses any other."""

    kwargs, launches, launch = _program_values(
        monkeypatch,
        "smallm_kv1032_q8_p64",
        device_scales=False,
        rerun=True,
    )
    ((_, values),) = launches
    assert values["grid_x"] == 9
    for bound in (9 * 128 + 1, 2 * 9 * 128):
        with pytest.raises(RuntimeError, match="merges one tile"):
            launch(max_seq_len=bound)
    selected = _select(
        _row_kwargs(
            "smallm_kv1032_q8_p64",
            batch=batch,
            kv_len=min(max_seq_len, 1032),
            max_seq_len=max_seq_len,
            width=-(-max_seq_len // 64),
        )
    )
    program = None if selected is None else selected.program
    assert (program == ("decode_smallm", "N64_P64_Q8_C9")) == cluster, selected


# --- GPU: BF16 inputs and reference ------------------------------------------


def _bf16_case(row, *, seed, lengths=None, counter=True):
    """BF16 queries over one stacked BF16 cache (the contract's form: K and V
    are the two halves of each page), shuffled pages."""

    _, kind, batch, q_len, kv_len, hq, hkv, page, _ = SHAPE_ROWS[row]
    head_dim = 256 if kind == "smallm" else 128
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    width = -(-kv_len // page)
    total = batch * width
    cache = torch.randn(
        (total, 2, hkv, page, head_dim), generator=generator, device=device
    ).to(torch.bfloat16)
    query = torch.randn(
        (batch * q_len, hq, head_dim), generator=generator, device=device
    ).to(torch.bfloat16)
    order = torch.randperm(total, generator=generator, device=device)
    if lengths is None:
        lengths = [kv_len] * batch
    return SimpleNamespace(
        query=query,
        kv_cache=cache,
        kv_cache_sf=None,
        block_tables=order.view(batch, width).to(torch.int32).contiguous(),
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        max_seq_len=kv_len,
        bmm1=head_dim**-0.5,
        bmm2=1.0,
        o_scale=1.0,
        q_len=q_len,
        sinks=None,
        out=torch.empty_like(query),
        workspace=torch.zeros(1 << 28, dtype=torch.uint8, device=device),
        counter=(
            torch.zeros(1 << 16, dtype=torch.uint8, device=device) if counter else None
        ),
    )


def _bf16_reference(case, batches):
    """FP32 speculative decode of the given requests: query token ``i`` of a
    request of length ``L`` sees keys ``0 .. L - q_len + i``; a sink adds its
    logit to every row's softmax."""

    cache, q_len = case.kv_cache, case.q_len
    page = cache.shape[3]
    num_q_heads, num_kv_heads = case.query.shape[1], cache.shape[2]
    group = num_q_heads // num_kv_heads
    bmm1, bmm2 = (
        scale.item() if isinstance(scale, torch.Tensor) else float(scale)
        for scale in (case.bmm1, case.bmm2)
    )
    outputs = {}
    for b in batches:
        length = int(case.seq_lens[b])
        pages = case.block_tables[b, : -(-length // page)].long()

        def side(index):
            values = cache[pages, index].float().transpose(0, 1)
            values = values.reshape(num_kv_heads, -1, values.shape[-1])[:, :length]
            return values.repeat_interleave(group, dim=0)

        k, v = side(0), side(1)
        q = case.query[b * q_len : (b + 1) * q_len].float()
        logits = torch.einsum("qhd,hld->hql", q, k) * bmm1
        rows = torch.arange(q_len, device=q.device)[:, None] + length - q_len
        visible = torch.arange(length, device=q.device)[None, :] <= rows
        logits = logits.masked_fill(~visible[None], float("-inf"))
        if case.sinks is not None:
            sink = case.sinks.float()[:, None, None].expand(-1, q_len, 1)
            logits = torch.cat((logits, sink), dim=-1)
        p = torch.softmax(logits, dim=-1)[..., :length]
        outputs[b] = torch.einsum("hql,hld->qhd", p, v) * bmm2
    return outputs


def _require_full_device():
    """The shape rows' split counts assume the device's full SM count."""

    target = cake_api._cake_fmha_target(torch.device("cuda"))
    count = torch.cuda.get_device_properties(0).multi_processor_count
    if count != _SM_COUNT[target]:
        pytest.skip(f"{target} with {count} SMs plans other split counts")
    return target


# --- GPU: small-M ------------------------------------------------------------

_SMALLM_GPU_ROWS = (
    "smallm_kv1032_q8_p64",
    "smallm_kv8200_q8_p64",
    "smallm_kv8199_q7_p64",
    "smallm_kv60008_q8_p16",
    "smallm_tp1_kv60004_q4_p64",
    "smallm_kv60003_q3_p64",
)


@_GPU
@pytest.mark.parametrize("row", _SMALLM_GPU_ROWS)
def test_cake_decode_smallm_programs_match_reference_and_the_jit_route(
    monkeypatch, served_programs, row
) -> None:
    """Each program against the FP32 reference at the BF16 tolerance and
    against the module's JIT route (the merge order of the partials may
    differ: compared at the same tolerance), output and LSE."""

    target = _require_full_device()
    case = _bf16_case(row, seed=20261028)
    rows, heads = case.query.shape[:2]
    lse = torch.empty((rows, heads), dtype=torch.float32, device="cuda")
    got = _run_decode(case, lse=lse)
    assert served_programs == [("decode_smallm", SHAPE_ROWS[row][-1][target])]
    batch = SHAPE_ROWS[row][2]
    reference = _bf16_reference(case, range(batch))
    for b in range(batch):
        q_len = case.q_len
        torch.testing.assert_close(
            got[b * q_len : (b + 1) * q_len].float(),
            reference[b],
            atol=1e-2,
            rtol=1e-2,
        )
    _keep_the_jit_route(monkeypatch)
    jit_lse = torch.empty_like(lse)
    jit = _run_decode(case, lse=jit_lse)
    assert len(served_programs) == 1
    torch.testing.assert_close(got.float(), jit.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, jit_lse, atol=1e-3, rtol=1e-3)


@_GPU
@pytest.mark.parametrize("row", ("smallm_kv1032_q8_p64", "smallm_kv8200_q8_p64"))
def test_cake_decode_smallm_program_is_sync_free_and_replays(
    served_programs, row
) -> None:
    target = _require_full_device()
    case = _bf16_case(row, seed=20261029)
    lengths = torch.tensor([case.max_seq_len - 37], dtype=torch.int32, device="cuda")
    _check_hit_is_sync_free_and_replays(case, lengths)
    assert set(served_programs) == {("decode_smallm", SHAPE_ROWS[row][-1][target])}


@_GPU
def test_cake_decode_smallm_program_without_a_counter_buffer(served_programs) -> None:
    """Without the caller's counter buffer the tickets live in the workspace,
    zeroed on the stream first: the same bits, launch after launch."""

    _require_full_device()
    with_buffer = _bf16_case("smallm_kv8200_q8_p64", seed=20261030)
    without = _bf16_case("smallm_kv8200_q8_p64", seed=20261030, counter=False)
    without.workspace.fill_(0xA5)
    expected = _run_decode(with_buffer)
    for _ in range(3):
        assert torch.equal(_bits(_run_decode(without)), _bits(expected))
    assert not with_buffer.counter.any()


# --- BF16 boundaries (host-only) ---------------------------------------------


@pytest.mark.parametrize(
    ("row", "overrides", "program"),
    [
        # Nine 128-token blocks keep the KV in L2 (the cluster program's key);
        # the table must hold ten evened blocks of eight pages.
        ("bf16_b4_s1024", {"kv_len": 1152, "width": 80}, "DEC_B4C4"),
        ("bf16_b4_s1024", {"kv_len": 1153, "width": 80}, None),
        ("bf16_b4_s1024", {"kv_len": 1024, "max_seq_len": 1152}, None),
        ("bf16_b4_s1024", {"kv_len": 1, "width": 32}, "DEC_B4C4"),
        ("bf16_b4_s1024", {"kv_len": 1}, None),
        ("bf16_b128_s4096", {"kv_len": 4000, "width": 256}, "DEC_B128"),
        ("bf16_b128_s4096", {"kv_len": 4000}, None),
        ("bf16_b128_s4096", {"width": 255}, None),
        ("bf16_b128_s4096", {"batch": 127}, None),
        ("bf16_b128_s4096", {"batch": 129}, None),
        ("bf16_b256_s4096_sink", {"batch": 255}, None),
        ("bf16_b4_s1024", {"num_q_heads": 16, "num_kv_heads": 2}, None),
    ],
)
def test_cake_fmha_decode_bf16_boundaries(decode_target, row, overrides, program):
    decode_target("sm100a")
    route = _select(_row_kwargs(row, **overrides))
    if program is None:
        assert route is None or route.program is None, overrides
    else:
        assert route.program == ("decode_bf16", program), overrides


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize(
    ("kv_len", "max_seq_len", "width", "lse", "kept"),
    [
        (4096, 4096, 256, False, True),
        (3585, 3585, 256, False, True),
        (1300, 4096, 256, False, True),
        (3584, 3584, 256, False, False),
        (4100, 4100, 272, False, False),
        (1300, 1300, 144, False, False),
        (1300, 4096, 144, False, False),
        (1300, 1300, 82, False, False),
        # The retained-KV member (max_seq_len up to 9 blocks) keeps its
        # launches, with or without LSE; an LSE launch never runs the exact
        # member (no member writes LSE at its other flags).
        (1152, 1152, 144, False, True),
        (1152, 1152, 72, False, True),
        (1152, 1152, 144, True, True),
        (1300, 1300, 144, True, False),
    ],
)
def test_cake_fmha_decode_exact_sink_member_finishes_or_takes_compat(
    decode_target, monkeypatch, target, kv_len, max_seq_len, width, lse, kept
) -> None:
    """The exact B256 sink member finishes only for 3585 <= max_seq_len <=
    4096 with 256 pages per row: the launches outside that window that would
    run it take the compat route, and every other launch keeps its member."""

    decode_target(target)
    _keep_the_jit_route(monkeypatch)
    kwargs = _row_kwargs(
        "bf16_b256_s4096_sink", kv_len=kv_len, max_seq_len=max_seq_len, width=width
    )
    if lse:
        kwargs["lse"] = torch.empty((256, 32), dtype=torch.float32, device="meta")
    route = _select(kwargs)
    if kept:
        assert route is not None and route.component == "decode_native_bf16", route
        assert route.has_sink and route.program is None and route.return_lse == lse
        assert route.retain_kv_l2 == (max_seq_len <= 9 * 128)
    else:
        assert route is None, route


@pytest.mark.parametrize("device_scales", (False, True))
@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_cake_fmha_decode_bf16_streamk_program_admission(
    decode_target, target, device_scales
) -> None:
    """The stream-K B128 program cuts each request into at most two parts on
    adjacent CTA groups (one CTA per SM, four per group): it needs two
    requests per group, and a workspace holding its split scratch and
    counters after the LSE scratch.  Otherwise the launch keeps the module's
    route."""

    identifier = "DEC_B128_DS" if device_scales else "DEC_B128"
    kwargs = _row_kwargs("bf16_b128_s4096", device_scales=device_scales)
    decode_target(target, sm_count=259)  # 64 groups: 128 requests admitted
    assert _select(kwargs).program == ("decode_bf16", identifier)
    decode_target(target, sm_count=260)  # 65 groups
    route = _select(kwargs)
    assert route is None or route.program is None, route
    decode_target(target)
    grid = _SM_COUNT[target]
    rows = 128 * 32
    end = cake_api._decode_streamk_layout(128, rows, grid)[2]
    assert (
        end
        == ((128 * 4 + 15) // 16 * 16 + rows * 4 + 255) // 256 * 256
        + grid * 2 * 1056 * 4
        + grid * 4
    )
    fits = _row_kwargs(
        "bf16_b128_s4096", device_scales=device_scales, workspace_bytes=end
    )
    assert _select(fits).program == ("decode_bf16", identifier)
    short = _row_kwargs(
        "bf16_b128_s4096", device_scales=device_scales, workspace_bytes=end - 1
    )
    route = _select(short)
    assert route is None or route.program is None, route


@pytest.mark.parametrize("device_scales", (False, True))
@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_cake_fmha_decode_bf16_cluster_program_needs_a_cluster_per_tile(
    decode_target, target, device_scales
) -> None:
    """The cluster program gives each of its 16 tiles (4 requests x 4 KV heads)
    its own cluster and arms its reduction barrier once.  A device with fewer
    SMs (a partition) keeps the module's route; that JIT member refuses the
    launch instead of hanging."""

    kwargs = _row_kwargs("bf16_b4_s1024", device_scales=device_scales)
    identifier = SHAPE_ROWS["bf16_b4_s1024"][-1][target]
    if device_scales:
        identifier = _device_scale_program("decode_bf16", identifier)
    decode_target(target, sm_count=16)
    assert _select(kwargs).program == ("decode_bf16", identifier)
    decode_target(target, sm_count=15)
    route = _select(kwargs)
    assert route is None or route.program is None, route
    if not device_scales:
        assert route.component == "decode_native_bf16", route
        assert not cake_api._decode_route_runs_program(route)


@pytest.mark.parametrize(
    "mutation",
    (
        "lse",
        "window",
        "bmm2",
        "o_scale",
        "mixed_scales",
        "cpu_scales",
        "uint32_lengths",
        "contiguous_pools",
    ),
)
@pytest.mark.parametrize("row", ("bf16_b128_s4096", "bf16_b4_s1024"))
def test_cake_fmha_decode_bf16_misses_keep_the_other_routes(
    decode_target, row, mutation
) -> None:
    """Outside the programs' contract a launch keeps the module's routes: a
    window, a host bmm2 other than 1, an output scale, mixed or CPU scales,
    unsigned lengths and separate contiguous K / V pools (the aligned
    programs' form).  An LSE launch runs the JIT member of its route."""

    decode_target("sm100a")
    kwargs = _row_kwargs(row)
    query = kwargs["query"]
    if mutation == "lse":
        kwargs["lse"] = torch.empty(
            (query.shape[0], query.shape[1]), dtype=torch.float32, device="meta"
        )
    elif mutation == "window":
        kwargs["window_left"] = 128
    elif mutation == "bmm2":
        kwargs["bmm2_scale"] = 0.5
    elif mutation == "o_scale":
        kwargs["o_scale"] = 0.75
    elif mutation == "mixed_scales":
        kwargs["bmm1_scale"] = torch.empty((1,), dtype=torch.float32, device="meta")
    elif mutation == "cpu_scales":
        kwargs["bmm1_scale"] = torch.ones((1,), dtype=torch.float32)
        kwargs["bmm2_scale"] = torch.ones((1,), dtype=torch.float32)
    elif mutation == "uint32_lengths":
        kwargs["seq_lens"] = kwargs["seq_lens"].to(torch.uint32)
    else:
        key = kwargs["key_cache"]
        kwargs["key_cache"] = torch.empty(key.shape, dtype=key.dtype, device="meta")
        kwargs["value_cache"] = torch.empty_like(kwargs["key_cache"])
    route = _select(kwargs)
    if mutation == "lse":
        # An LSE launch takes the route of the same launch without it and runs
        # the JIT member, which writes the LSE (the programs write none).
        plain = _select(_row_kwargs(row))
        assert route == dataclasses.replace(plain, return_lse=True), route
        assert not cake_api._decode_route_runs_program(route)
        jit = dataclasses.replace(route, program=None, device_scales=False)
        assert cake_api._route_components(route) == cake_api._route_components(jit)
        assert cake_api.cake_fmha_route_is_optimized(route)
    else:
        assert route is None or route.program is None, (row, mutation, route)


@pytest.mark.parametrize("device_scales", (False, True))
@pytest.mark.parametrize(
    ("row", "grid"),
    [
        ("bf16_b128_s4096", (148, 1, 1)),
        ("bf16_b4_s1024", (64, 1, 1)),
        ("bf16_b256_s4096_sink", (1, 4, 256)),
    ],
)
def test_cake_decode_bf16_program_values(monkeypatch, row, grid, device_scales) -> None:
    """The kernels read the caller's tensors in place, the lengths raw, the
    LSE scratch at the JIT route's offset; unread pointers point at the
    lengths."""

    kwargs, launches = _program_values(monkeypatch, row, device_scales=device_scales)
    ((_, values),) = launches
    batch = kwargs["batch_size"]
    assert values["Qt"] is kwargs["query"] and values["O_ptr"] is kwargs["out"]
    assert values["K"] is kwargs["key_cache"] and values["V"] is kwargs["value_cache"]
    assert values["page_table"] is kwargs["block_tables"]
    assert values["causal_seqlens_kv_global"] is kwargs["seq_lens"]
    assert values["max_pages_per_seq"] == kwargs["block_tables"].shape[1]
    assert values["max_local_seq_len"] == kwargs["max_seq_len"]
    workspace = kwargs["workspace_buffer"].data_ptr()
    assert values["LSE_ptr"].data_ptr() == workspace + (batch * 4 + 15) // 16 * 16
    assert values["LSE_ptr"].numel() == batch * 32
    lengths = kwargs["seq_lens"].data_ptr()
    if device_scales:
        assert _same_storage(values["scale_log2_ptr"], kwargs["bmm1_scale"])
        assert _same_storage(values["bmm2_ptr"], kwargs["bmm2_scale"])
    else:
        assert values["scale_log2_ptr"].data_ptr() == lengths
        assert values["bmm2_ptr"].data_ptr() == lengths
        assert values["softmax_scale_log2"] == _HOST_BMM1 * 1.4426950408889634
    if kwargs["sinks"] is None:
        assert values["sinks_ptr"].data_ptr() == lengths
    else:
        assert values["sinks_ptr"] is kwargs["sinks"]
    assert values["window_left"] == -1
    assert (values["num_q_heads"], values["num_kv_heads"]) == (32, 4)
    assert values["batch_size"] == batch
    assert (values["grid_x"], values["grid_y"], values["grid_z"]) == grid
    if row == "bf16_b128_s4096":
        # The stream-K split scratch after the LSE scratch, and one counter
        # per CTA in the workspace (no caller buffer), zeroed for the launch.
        split_ws, counters, _ = cake_api._decode_streamk_layout(
            batch, batch * 32, grid[0]
        )
        assert values["split_ws"].data_ptr() == workspace + split_ws
        assert values["split_ws"].dtype == torch.float32
        assert values["split_ws"].numel() == grid[0] * 2 * 1056
        assert values["split_counter"].data_ptr() == workspace + counters
        assert values["split_counter"].dtype == torch.uint32
        assert values["split_counter"].numel() == grid[0]
        assert not values["split_counter"].view(torch.int32).any()
    else:
        assert "split_ws" not in values and "split_counter" not in values


@pytest.mark.parametrize("counters", ("caller", "small_caller", "workspace"))
@pytest.mark.parametrize("sm_count", (148, 152))
def test_cake_decode_bf16_streamk_counters_follow_the_buffers(
    monkeypatch, counters, sm_count
) -> None:
    """The split counters live in the caller's counter buffer when it holds
    one per CTA (the kernel leaves them zero), else in the workspace, zeroed
    before every launch.  The module keeps the views for the same buffers and
    rebuilds them for others."""

    caller = {
        "caller": torch.zeros(sm_count * 4, dtype=torch.uint8),
        "small_caller": torch.zeros(sm_count * 4 - 4, dtype=torch.uint8),
        "workspace": None,
    }[counters]
    kwargs, launches, launch = _program_values(
        monkeypatch,
        "bf16_b128_s4096",
        device_scales=False,
        sm_count=sm_count,
        counter_buffer=caller,
        rerun=True,
    )
    ((_, values),) = launches
    assert values["grid_x"] == sm_count and values["split_counter"].numel() == sm_count
    workspace = kwargs["workspace_buffer"]
    split_ws, offset, _ = cake_api._decode_streamk_layout(128, 128 * 32, sm_count)
    if counters == "caller":
        assert values["split_counter"].data_ptr() == caller.data_ptr()
    else:
        assert values["split_counter"].data_ptr() == workspace.data_ptr() + offset
    assert values["split_ws"].data_ptr() == workspace.data_ptr() + split_ws
    # Launch again after poisoning the workspace: the same views, and the
    # workspace counters zeroed again.
    workspace.fill_(0xA5)
    ((_, again),) = launch()
    assert again["split_counter"].data_ptr() == values["split_counter"].data_ptr()
    assert again["split_ws"].data_ptr() == values["split_ws"].data_ptr()
    if counters == "caller":
        assert workspace[offset : offset + sm_count * 4].eq(0xA5).all()
    else:
        assert not again["split_counter"].view(torch.int32).any()
    # Another workspace: new views.
    other = torch.zeros_like(workspace)
    ((_, moved),) = launch(workspace_buffer=other)
    assert moved["split_ws"].data_ptr() == other.data_ptr() + split_ws
    if counters != "caller":
        assert moved["split_counter"].data_ptr() == other.data_ptr() + offset


# --- GPU: BF16 ---------------------------------------------------------------

_BF16_GPU_ROWS = ("bf16_b128_s4096", "bf16_b4_s1024", "bf16_b256_s4096_sink")
# The programs whose host-scale form is the module's JIT member bit for bit
# (the four-CTA cluster program splits a tile's KV four ways instead of two,
# and the stream-K program merges the two parts of a split request).
_BF16_JIT_BITWISE = ("DEC_SINK",)


def _bf16_gpu_case(row, *, seed, device_scales=False, lengths=None):
    case = _bf16_case(row, seed=seed, lengths=lengths)
    if SHAPE_ROWS[row][1] == "bf16_sink":
        generator = torch.Generator(device="cuda").manual_seed(seed + 1)
        case.sinks = (
            torch.rand((32,), generator=generator, device="cuda", dtype=torch.float32)
            * 5.0
        )
    if device_scales:
        # The log2-domain bmm1 the host route computes, so both forms agree.
        log2_bmm1 = torch.tensor(
            [case.bmm1 * 1.4426950408889634], dtype=torch.float32, device="cuda"
        )
        case.bmm1 = log2_bmm1 / 1.4426950408889634
        case.bmm1_log2 = log2_bmm1
        case.bmm2 = torch.ones((1,), dtype=torch.float32, device="cuda")
    return case


def _run_bf16(case, **kwargs):
    log2 = getattr(case, "bmm1_log2", None)
    if log2 is None:
        return _run_decode(case, **kwargs)
    seq_lens, out = kwargs.get("seq_lens"), kwargs.get("out")
    return decode_api.trtllm_batch_decode_with_kv_cache(
        case.query,
        case.kv_cache,
        case.workspace,
        case.block_tables,
        case.seq_lens if seq_lens is None else seq_lens,
        case.max_seq_len,
        bmm1_scale=case.bmm1,
        bmm2_scale=case.bmm2,
        bmm1_scale_log2=log2,
        out=torch.empty_like(case.out) if out is None else out,
        kv_layout="HND",
        backend="cake",
        sinks=case.sinks,
        enable_pdl=True,
        multi_ctas_kv_counter_buffer=case.counter,
    )


@_GPU
@pytest.mark.parametrize("lengths", ("uniform", "ragged"))
@pytest.mark.parametrize("row", _BF16_GPU_ROWS)
def test_cake_decode_bf16_programs_match_reference_and_the_jit_route(
    monkeypatch, served_programs, row, lengths
) -> None:
    target = _require_full_device()
    batch, kv_len = SHAPE_ROWS[row][2], SHAPE_ROWS[row][4]
    ragged = None
    if lengths == "ragged":
        ragged = [kv_len, 1, 2, 129, kv_len - 1, 255, kv_len // 2, 17]
        ragged = (ragged * batch)[:batch]
    case = _bf16_gpu_case(row, seed=20261031, lengths=ragged)
    got = _run_bf16(case)
    identifier = SHAPE_ROWS[row][-1][target]
    assert served_programs == [("decode_bf16", identifier)]
    batches = sorted({0, 1, 2, 3, batch - 1})
    if identifier == "DEC_B128":
        # every request: the stream-K cuts fall between and inside requests
        batches = list(range(batch))
    reference = _bf16_reference(case, batches)
    for b in batches:
        torch.testing.assert_close(
            got[b].float(), reference[b][0], atol=1e-2, rtol=1e-2, msg=f"request {b}"
        )
    # The device-scale twin, with the same base-2 scale: the same bits.
    device_case = _bf16_gpu_case(row, seed=20261031, lengths=ragged, device_scales=True)
    device_out = _run_bf16(device_case)
    assert served_programs[-1] == ("decode_bf16", f"{identifier}_DS")
    assert torch.equal(_bits(device_out), _bits(got))
    _keep_the_jit_route(monkeypatch)
    jit = _run_bf16(case)
    assert len(served_programs) == 2
    if identifier in _BF16_JIT_BITWISE:
        assert torch.equal(_bits(got), _bits(jit))
    else:
        torch.testing.assert_close(got.float(), jit.float(), atol=1e-2, rtol=1e-2)


@_GPU
@pytest.mark.parametrize("device_scales", (False, True))
@pytest.mark.parametrize("row", _BF16_GPU_ROWS)
def test_cake_decode_bf16_program_is_sync_free_and_replays(
    served_programs, row, device_scales
) -> None:
    target = _require_full_device()
    case = _bf16_gpu_case(row, seed=20261101, device_scales=device_scales)
    batch, kv_len = SHAPE_ROWS[row][2], SHAPE_ROWS[row][4]
    lengths = torch.randint(1, kv_len + 1, (batch,), dtype=torch.int32, device="cuda")
    case.run = _run_bf16
    _check_hit_is_sync_free_and_replays(case, lengths)
    identifier = SHAPE_ROWS[row][-1][target] + ("_DS" if device_scales else "")
    assert set(served_programs) == {("decode_bf16", identifier)}


@_GPU
@pytest.mark.parametrize("counter", (True, False))
def test_cake_decode_bf16_streamk_counters_are_graph_safe(
    served_programs, counter
) -> None:
    """The stream-K program's split counters are zero after every launch (the
    second part of a split request resets its counter): the caller's buffer
    reads zero after direct and replayed launches.  Without it the counters
    live in the workspace, zeroed inside the captured work.  A graph captured
    once replays new in-place lengths bitwise to direct launches, replay after
    replay."""

    _require_full_device()
    case = _bf16_gpu_case("bf16_b128_s4096", seed=20261102)
    if not counter:
        case.counter = None
        case.workspace.fill_(0xA5)
    seq_lens = case.seq_lens.clone()
    out = torch.empty_like(case.out)
    _run_bf16(case, seq_lens=seq_lens, out=out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run_bf16(case, seq_lens=seq_lens, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_bf16(case, seq_lens=seq_lens, out=out)
    generator = torch.Generator(device="cuda").manual_seed(20261103)
    for _ in range(3):
        lengths = torch.randint(
            1,
            case.max_seq_len + 1,
            (seq_lens.numel(),),
            generator=generator,
            dtype=torch.int32,
            device="cuda",
        )
        seq_lens.copy_(lengths)
        graph.replay()
        direct = _run_bf16(case, seq_lens=lengths)
        torch.cuda.synchronize()
        assert torch.equal(_bits(out), _bits(direct))
        if counter:
            assert not case.counter.any()
    assert set(served_programs) == {("decode_bf16", "DEC_B128")}


# --- FP8 boundaries (host-only) ----------------------------------------------

# 127 blocks read the evened 128: the table must hold 512 32-token pages.
_FP8_127_BLOCKS = {"kv_len": 16129, "width": 512}


@pytest.mark.parametrize(
    ("target", "row", "overrides", "program"),
    [
        # The fixed plan is eight splits of sixteen blocks: 127 or 128 blocks.
        ("sm100a", "fp8_b4_s16384", _FP8_127_BLOCKS, "B4_S8X16_SM100A"),
        ("sm103a", "fp8_b4_s16384", _FP8_127_BLOCKS, "B4_S8X16_SM103A"),
        ("sm100a", "fp8_b4_s16384", {"kv_len": 16129}, None),
        ("sm100a", "fp8_b4_s16384", {"kv_len": 16128}, None),
        ("sm100a", "fp8_b4_s16384", {"kv_len": 16385, "width": 520}, None),
        ("sm100a", "fp8_b4_s16384", {"kv_len": 16384, "width": 511}, None),
        ("sm100a", "fp8_b4_s16384", {"batch": 3}, None),
        ("sm100a", "fp8_b4_s16384", {"page_size": 16}, None),
        # One split: any length whose table is wide enough.
        ("sm100a", "fp8_b256_s8192", {"kv_len": 8000, "width": 512}, "B256_SM100A"),
        ("sm100a", "fp8_b256_s8192", {"kv_len": 8000}, None),
        ("sm103a", "fp8_b256_s8192", {"kv_len": 512}, "B256_SM103A"),
        ("sm100a", "fp8_b256_s8192", {"batch": 255}, None),
        ("sm100a", "fp8_b256_s8192", {"batch": 257}, None),
        ("sm100a", "fp8_b256_s8192", {"page_size": 32}, None),
    ],
)
def test_cake_fmha_decode_fp8_boundaries(
    decode_target, target, row, overrides, program
) -> None:
    decode_target(target)
    route = _select(_row_kwargs(row, **overrides))
    if program is None:
        assert route is None or route.program is None, overrides
    else:
        assert route.program == ("decode_fp8", program), overrides
        assert route.num_split == (8 if row == "fp8_b4_s16384" else 1)


def test_cake_fmha_decode_fp8_fixed_plan_needs_enable_pdl(decode_target) -> None:
    """The fixed-plan programs hand their partials to a reducer launched with
    programmatic dependent launch: a call without ``enable_pdl`` keeps the
    JIT member (its reducer launches without the attribute)."""

    decode_target("sm103a")
    route = _select({**_row_kwargs("fp8_b4_s16384"), "enable_pdl": False})
    assert route.component == "decode_quant_fp8" and route.program is None
    assert route.num_split == 8


def test_cake_fmha_decode_fp8_host_scales_without_output_scale_stay_balanced(
    decode_target,
) -> None:
    """Host scales with no output scale keep the balanced FP8 route (16-token
    pages); tensor scales take the program."""

    decode_target("sm100a")
    route = _select({**_row_kwargs("fp8_b256_s8192"), "o_scale": 1.0})
    assert route.component == "decode_balanced_fp8" and route.program is None
    route = _select(
        {**_row_kwargs("fp8_b256_s8192", device_scales=True), "o_scale": 1.0}
    )
    assert route.program == ("decode_fp8", "B256_SM100A") and route.device_scales


@pytest.mark.parametrize("mutation", ("mixed_scales", "strided_query", "small_sm"))
def test_cake_fmha_decode_fp8_misses_keep_the_jit_route(decode_target, mutation):
    # 120 SMs plan seven splits of twenty blocks (128 or more plan eight of
    # sixteen, the program's plan).
    decode_target("sm100a", sm_count=120 if mutation == "small_sm" else None)
    kwargs = _row_kwargs("fp8_b4_s16384")
    if mutation == "mixed_scales":
        kwargs["bmm2_scale"] = torch.empty((1,), dtype=torch.float32, device="meta")
    elif mutation == "strided_query":
        query = kwargs["query"]
        wide = torch.empty(
            (query.shape[0], query.shape[1], 256), dtype=query.dtype, device="meta"
        )
        kwargs["query"] = wide[..., :128]
    route = _select(kwargs)
    assert route is None or route.program is None, mutation


@pytest.mark.parametrize("device_scales", (False, True))
def test_cake_decode_fp8_fixed_plan_values_pair_the_reducer(
    monkeypatch, device_scales
) -> None:
    """The fixed-plan program and its reducer share the workspace partials
    and the output; the reducer merges eight splits per (request, query
    head) with the same bmm2 scale."""

    kwargs, launches = _program_values(
        monkeypatch, "fp8_b4_s16384", device_scales=device_scales
    )
    (main_name, main), (reduce_name, reduce) = launches
    assert main_name == cake_api._DECODE_PROGRAMS[("decode_fp8", "B4_S8X16_SM100A")][0]
    assert reduce_name == cake_api._DECODE_PROGRAMS[("decode_fp8_pdl", "REDUCE_S8")][0]
    assert (main["num_splits"], main["blocks_per_split"]) == (8, 16)
    assert (main["grid_x"], main["grid_y"], main["grid_z"]) == (128, 1, 1)
    assert (reduce["grid_x"], reduce["grid_y"], reduce["grid_z"]) == (128, 1, 1)
    assert reduce["num_split"] == 8
    for name in ("partial_O", "partial_max", "partial_sum", "O"):
        assert reduce[name] is main[name], name
    assert main["K"].dtype == torch.uint8 and main["V"].dtype == torch.uint8
    assert _same_storage(main["K"], kwargs["key_cache"])
    assert _same_storage(main["V"], kwargs["value_cache"])
    assert main["K"].stride() == kwargs["key_cache"].stride()
    assert main["page_table"] is kwargs["block_tables"]
    assert main["pt_batch_stride"] == 512
    rows = 4 * 32 * 8
    first = 0 if device_scales else 16
    workspace = kwargs["workspace_buffer"].data_ptr()
    assert main["partial_O"].data_ptr() == workspace + first
    assert main["partial_max"].data_ptr() == workspace + first + rows * 512
    assert main["partial_sum"].data_ptr() % 16 == 0
    if device_scales:
        assert _same_storage(reduce["bmm2_scale_ptr"], kwargs["bmm2_scale"])
        assert main["bmm1_is_log2"] == 1
    else:
        assert reduce["bmm2_scale_ptr"] is None and main["bmm1_scale_ptr"] is None
        assert reduce["bmm2_scalar"] == main["bmm2_scalar"] == 1.0 / 0.75


# --- GPU: FP8 ----------------------------------------------------------------


def _fp8_case(row, *, seed, device_scales=False, lengths=None):
    """E4M3 queries over one stacked E4M3 cache, bmm2 folding an output scale
    of 0.75 (the contract's form)."""

    case = _bf16_case(row, seed=seed, lengths=lengths)
    case.query = (case.query.float() * 0.5).to(torch.float8_e4m3fn)
    case.kv_cache = (case.kv_cache.float() * 0.5).to(torch.float8_e4m3fn)
    case.out = torch.empty_like(case.query)
    case.bmm1, case.bmm2, case.o_scale = 128**-0.5, 1.0 / 0.75, 0.75
    if device_scales:
        case.bmm1 = torch.tensor([case.bmm1], dtype=torch.float32, device="cuda")
        case.bmm2 = torch.tensor([case.bmm2], dtype=torch.float32, device="cuda")
    return case


@_GPU
@pytest.mark.parametrize("device_scales", (False, True))
@pytest.mark.parametrize("lengths", ("uniform", "ragged"))
@pytest.mark.parametrize("row", ("fp8_b256_s8192", "fp8_b4_s16384"))
def test_cake_decode_fp8_programs_match_reference_and_the_jit_route(
    monkeypatch, served_programs, row, lengths, device_scales
) -> None:
    target = _require_full_device()
    batch, kv_len = SHAPE_ROWS[row][2], SHAPE_ROWS[row][4]
    ragged = None
    if lengths == "ragged":
        ragged = ([kv_len, 0, 1, 129, kv_len - 1, 2047, kv_len // 3, 17] * batch)[
            :batch
        ]
    case = _fp8_case(row, seed=20261102, device_scales=device_scales, lengths=ragged)
    got = _run_decode(case)
    assert served_programs == [("decode_fp8", SHAPE_ROWS[row][-1][target])]
    batches = sorted({0, 1, 2, 3, batch - 1})
    reference = _bf16_reference(case, [b for b in batches if _case_length(case, b)])
    for b, expected in reference.items():
        torch.testing.assert_close(
            got[b].float(), expected[0], atol=0.1, rtol=0.1, msg=f"request {b}"
        )
    for b in batches:
        if not _case_length(case, b):
            assert not got[b].float().any(), f"request {b} has no key"
    _keep_the_jit_route(monkeypatch)
    jit = _run_decode(case)
    assert len(served_programs) == 1
    assert torch.equal(_bits(got), _bits(jit))


def _case_length(case, b):
    return int(case.seq_lens[b])


@_GPU
@pytest.mark.parametrize("row", ("fp8_b256_s8192", "fp8_b4_s16384"))
def test_cake_decode_fp8_program_is_sync_free_and_replays(served_programs, row):
    target = _require_full_device()
    case = _fp8_case(row, seed=20261103, device_scales=True)
    batch, kv_len = SHAPE_ROWS[row][2], SHAPE_ROWS[row][4]
    lengths = torch.randint(0, kv_len + 1, (batch,), dtype=torch.int32, device="cuda")
    _check_hit_is_sync_free_and_replays(case, lengths)
    assert set(served_programs) == {("decode_fp8", SHAPE_ROWS[row][-1][target])}


@_GPU
@pytest.mark.parametrize("row", ("bf16_b128_s4096", "fp8_b4_s16384"))
def test_cake_decode_program_launches_on_the_current_torch_stream(
    served_programs, row
) -> None:
    """Every launch of a program route (the FP8 fixed plan's decode and its
    reducer too) is ordered on torch's current stream: queued behind a sleep
    on a side stream, the output is still unwritten when the default stream
    reads it, and is the direct launch's once the side stream drains."""

    target = _require_full_device()
    if row.startswith("fp8"):
        case = _fp8_case(row, seed=20261104)
    else:
        case = _bf16_case(row, seed=20261104)
    direct = _run_decode(case)
    out = torch.zeros_like(direct)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(200_000_000)
        _run_decode(case, out=out)
    early = out.clone()
    torch.cuda.synchronize()
    assert not _bits(early).any()
    assert torch.equal(_bits(out), _bits(direct))
    family = "decode_fp8" if row.startswith("fp8") else "decode_bf16"
    assert set(served_programs) == {(family, SHAPE_ROWS[row][-1][target])}
