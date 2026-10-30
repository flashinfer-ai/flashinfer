"""Generated paged context programs of the Cake FMHA module (``backend="cake"``).

Host-only: the pinned program table against the generated loaders, the launch
grid, the shape rows and their widened classes (any batch, every query length
of the same m-block count) through ``select_cake_fmha_context_route`` on both
targets (host and device scales), the routes that keep the components' JIT
modules, the FP8 runtime-size route (every measured shape on both targets),
the FP8 split-KV route (its table, the split planner and the plan a launch
passes), the FP8 one-stage route (its table and choice), the page tables the
launch passes (views of the caller's storage) and the sizes a runtime-size
program takes per launch.
GPU: the BF16 programs (the one-tile ones at uniform query lengths, the
persistent one at ragged query lengths; ragged KV lengths, CUDA graph replay
with new lengths) against an FP32 reference; the BF16 JIT route on tiles
without a visible key (an empty request, a request without keys, causal tiles
whose rows see no key) against the reference and, on such rows, against the
one-tile programs; the FP8, NVFP4, FP8 HD256 and FP16 HD256 JIT routes on such
tiles against an FP32 reference and, on batches the FP8 / NVFP4 programs
serve, against those programs bit for bit; the FP8 HD256 JIT route on batches
that end with a partial L2 section against the reference and bit for bit
against the same requests in whole sections; the FP8 / NVFP4 programs (host and
device scales) through
the shared prefill test against its reference; and
every program bitwise against the component's JIT route on its row and its
widened class, plus the invariances a launch must keep bitwise (NHD and HND
caches, shared and separate page tables, a column-sliced page table, graph
replay, device and host scales); the split-KV programs against an FP32
reference, run to run, under graph replay and with their tile counters back
at zero; the one-stage programs bitwise against the two-stage programs and
within tolerance of an FP32 reference.  Each case checks the program that
served it.
"""

from __future__ import annotations

import contextlib
import importlib
import math
from types import SimpleNamespace

import pytest
import torch
import tvm_ffi

import flashinfer.cake_fmha as cake_api
import flashinfer.prefill as prefill
from flashinfer.utils import (
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
)
from tests.test_helpers.test_helpers import assert_close_with_mismatch_tolerance

_TARGET_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
_COMPONENTS = ("context_bf16", "context_fp8", "context_nvfp4")

# The shape rows: name -> (component, batch, q_len, kv_len, num_q_heads,
# num_kv_heads, page_size, uses_shared_paged_kv_idx, the program per target).
SHAPE_ROWS = {
    "bf16_b1_q8192_k8192": (
        "context_bf16", 1, 8192, 8192, 64, 8, 16, True,
        {"sm100a": "Q8192", "sm103a": "Q8192"},
    ),
    "bf16_b128_q64_k2048": (
        "context_bf16", 128, 64, 2048, 32, 4, 16, True,
        {"sm100a": "B128", "sm103a": "B128"},
    ),
    "bf16_b4_q511_k2047": (
        "context_bf16", 4, 511, 2047, 10, 2, 32, True,
        {"sm100a": "Q511_SM100A", "sm103a": "Q511_SM103A"},
    ),
    "bf16_b4_q257_k1024_page1024": (
        "context_bf16", 4, 257, 1024, 10, 2, 1024, False,
        {"sm100a": "Q257_SM100A", "sm103a": "Q257_SM103A"},
    ),
    "fp8_b256_q32_k1024": (
        "context_fp8", 256, 32, 1024, 32, 4, 64, True,
        {"sm100a": "B256", "sm103a": "B256_PX"},
    ),
    # At most 128 keys: sm103a takes the short-KV program of the route.
    "fp8_b16_q32_k128": (
        "context_fp8", 16, 32, 128, 32, 4, 64, True,
        {"sm100a": "B256", "sm103a": "B256_KV4"},
    ),
    "nvfp4_b4_q511_k2047": (
        "context_nvfp4", 4, 511, 2047, 32, 4, 16, True,
        {"sm100a": "Q511_SM100A", "sm103a": "Q511_SM103A"},
    ),
    # FP8 calls no exact route serves: the runtime-size program of the pack
    # (the largest shipped pack dividing the JIT pack) and the page size.
    "fp8_b64_q32_k1024_g4": (
        "context_fp8", 64, 32, 1024, 32, 8, 64, True,
        {"sm100a": "RUNTIME_PACK4_PAGE64", "sm103a": "RUNTIME_PACK4_PAGE64_PX"},
    ),
    "fp8_b16_q32_k128_g4": (
        "context_fp8", 16, 32, 128, 32, 8, 64, True,
        {"sm100a": "RUNTIME_PACK4_PAGE64", "sm103a": "RUNTIME_PACK4_PAGE64"},
    ),
    "fp8_b16_q32_k1024_g16_page16": (
        "context_fp8", 16, 32, 1024, 32, 2, 16, False,
        {"sm100a": "RUNTIME_PACK8", "sm103a": "RUNTIME_PACK8_PX"},
    ),
    "fp8_b2_q256_k2048_g3_page32": (
        "context_fp8", 2, 256, 2048, 30, 10, 32, True,
        {"sm100a": "RUNTIME_PACK1", "sm103a": "RUNTIME_PACK1_PX"},
    ),
    "fp8_b4_q128_k1024_g6_page16": (
        "context_fp8", 4, 128, 1024, 30, 5, 16, True,
        {"sm100a": "RUNTIME_PACK2", "sm103a": "RUNTIME_PACK2_PX"},
    ),
}  # fmt: skip
# The device-scale program of a host-scale one; a runtime-size program serves
# both scale forms itself.
_DEVICE_SCALE_PROGRAM = {
    "B256": "B256_SCALE_PTR",
    "B256_PX": "B256_SCALE_PTR_PX",
    "B256_KV4": "B256_SCALE_PTR_KV4",
    "Q511_SM100A": "Q511_SCALE_PTR_SM100A",
    "Q511_SM103A": "Q511_SCALE_PTR_SM103A",
}
# The exact FP8 program of a target (host scales): sm103a ships the 4-stage
# K/V ring build with each stage's P tile in the other stage's S columns.
_FP8_EXACT = {"sm100a": "B256", "sm103a": "B256_PX"}
# The suffix of a target's runtime-size programs: sm103a ships the 4-stage
# K/V ring builds with each stage's P tile in the other stage's S columns.
_RUNTIME_SUFFIX = {"sm100a": "", "sm103a": "_PX"}
# runtime-size program identifier -> its pack
_RUNTIME_PACK = {
    identifier: pack
    for routes in cake_api._CONTEXT_FP8_RUNTIME_ROUTES.values()
    for (pack, _), identifier in routes.items()
}


# split-KV program identifier -> its pack
_SPLIT_PACK = {
    identifier: pack
    for routes in cake_api._CONTEXT_FP8_SPLIT_ROUTES.values()
    for (pack, _), identifier in routes.items()
}


# one-stage program identifier -> its pack
_ONE_STAGE_PACK = {
    identifier: pack
    for routes in cake_api._CONTEXT_FP8_ONE_STAGE_ROUTES.values()
    for (pack, _), identifier in routes.items()
}


def _device_scale_program(program):
    return _DEVICE_SCALE_PROGRAM.get(program, program)


def _loader(component):
    return importlib.import_module(f"flashinfer.jit.cake_fmha_{component}")


# --- The program table against the generated loaders (host-only) -------------


def test_cake_fmha_context_program_table_matches_the_generated_loaders() -> None:
    """Every pinned record names the generated loader's program of the same
    identifier (the load-time checks: name, binding argument order, the targets
    the routes serve it on), every loader program is pinned and every pinned
    program serves a route."""

    from flashinfer.jit.cake_program import GRID_ARGS, CakeProgram

    served: dict[tuple[str, str], set[str]] = {}
    for target, routes in cake_api._CONTEXT_PROGRAM_ROUTES.items():
        for key, identifier in routes.items():
            served.setdefault((key[0], identifier), set()).add(_TARGET_ARCH[target])
    for table in (
        cake_api._CONTEXT_FP8_RUNTIME_ROUTES,
        cake_api._CONTEXT_FP8_SPLIT_ROUTES,
        cake_api._CONTEXT_FP8_ONE_STAGE_ROUTES,
    ):
        for target, routes in table.items():
            for identifier in routes.values():
                served.setdefault(("context_fp8", identifier), set()).add(
                    _TARGET_ARCH[target]
                )
    for target, short in cake_api._CONTEXT_FP8_SHORT_KV_ROUTES.items():
        for identifier, _ in short.values():
            served.setdefault(("context_fp8", identifier), set()).add(
                _TARGET_ARCH[target]
            )
    assert set(served) == set(cake_api._CONTEXT_PROGRAMS)
    for (component, identifier), (
        name,
        num_m_blocks,
        one_tile,
    ) in cake_api._CONTEXT_PROGRAMS.items():
        program = getattr(_loader(component), identifier)
        assert isinstance(program, CakeProgram), identifier
        assert program.name == name, identifier
        assert served[(component, identifier)] <= set(program.targets), identifier
        # a runtime-size program (no pinned m-block count) takes the sizes; a
        # split-KV program then the split plan, its partials and counters
        params = (
            cake_api._CONTEXT_FP8_SPLIT_PARAMS
            if identifier in _SPLIT_PACK
            else cake_api._CONTEXT_FP8_RUNTIME_PARAMS
            if num_m_blocks is None
            else cake_api._CONTEXT_HOST_PARAMS[component]
        )
        assert tuple(program.args) == (*params, *GRID_ARGS), identifier
        assert tuple(program.block) == (512, 1, 1), identifier
        assert program.use_pdl is False and program.cooperative is False
        assert one_tile == (component == "context_bf16" and num_m_blocks in (11, 6))
        assert (num_m_blocks is None) == (
            identifier in _RUNTIME_PACK
            or identifier in _SPLIT_PACK
            or identifier in _ONE_STAGE_PACK
        ), identifier
    for component in _COMPONENTS:
        pinned = {
            name
            for (pinned_component, _), (
                name,
                _,
                _,
            ) in cake_api._CONTEXT_PROGRAMS.items()
            if pinned_component == component
        }
        assert {program.name for program in _loader(component).PROGRAMS} == pinned


def test_cake_fmha_context_launch_grid() -> None:
    """One-tile programs launch every tile, persistent programs at most one CTA
    per SM (the parity lane's host grid)."""

    names = {
        identifier: name
        for (component, identifier), (name, _, _) in cake_api._CONTEXT_PROGRAMS.items()
        if component == "context_bf16"
    }

    def grid(identifier, total_bh, sm_count):
        record = SimpleNamespace(name=names[identifier])
        return cake_api.cake_fmha_context_launch_grid(record, total_bh, sm_count)

    assert grid("Q511_SM103A", 8, 152) == (88, 1, 1)
    assert grid("Q511_SM100A", 20, 148) == (220, 1, 1)
    assert grid("Q257_SM100A", 8, 148) == (48, 1, 1)
    assert grid("Q8192", 8, 148) == (148, 1, 1)
    assert grid("B128", 512, 152) == (152, 1, 1)
    assert grid("B128", 4, 148) == (8, 1, 1)
    with pytest.raises(RuntimeError, match="not a Cake FMHA context program"):
        cake_api.cake_fmha_context_launch_grid(SimpleNamespace(name="x"), 1, 1)
    # the pinned m-block count, given or not; another is refused
    b128 = SimpleNamespace(name=names["B128"])
    assert cake_api.cake_fmha_context_launch_grid(b128, 4, 148, 2) == (8, 1, 1)
    with pytest.raises(RuntimeError, match="runs 2 m-blocks, not 3"):
        cake_api.cake_fmha_context_launch_grid(b128, 4, 148, 3)

    # A runtime-size program launches at most one persistent CTA per SM over
    # the launch's own m-block count, which it requires.
    runtime = SimpleNamespace(
        name=cake_api._CONTEXT_PROGRAMS[("context_fp8", "RUNTIME_PACK8")][0]
    )
    assert cake_api.cake_fmha_context_launch_grid(runtime, 12, 148, 2) == (24, 1, 1)
    assert cake_api.cake_fmha_context_launch_grid(runtime, 512, 148, 3) == (
        148,
        1,
        1,
    )
    with pytest.raises(RuntimeError, match="takes the m-block count per launch"):
        cake_api.cake_fmha_context_launch_grid(runtime, 12, 148)


# --- The shape rows through route selection (host-only) ----------------------


def _context_kwargs(row, *, device_scales=False, device="cpu", batch=None, q_len=None):
    """Host-side ``select_cake_fmha_context_route`` inputs of one shape row (a
    ``SHAPE_ROWS`` name or such a tuple; ``batch`` / ``q_len`` override the
    row's): empty tensors of the row's shapes (the selector reads shapes,
    strides and dtypes only)."""

    component, row_batch, row_q_len, kv_len, hq, hkv, page, shared, _ = (
        SHAPE_ROWS[row] if isinstance(row, str) else row
    )
    batch = row_batch if batch is None else batch
    q_len = row_q_len if q_len is None else q_len
    pages = (kv_len + page - 1) // page
    if component == "context_bf16":
        q_dtype = kv_dtype = o_dtype = torch.bfloat16
        kv_shape = (batch * pages, hkv, page, 128)
    elif component == "context_fp8":
        q_dtype = kv_dtype = o_dtype = torch.float8_e4m3fn
        kv_shape = (batch * pages, hkv, page, 128)
    else:
        q_dtype, kv_dtype, o_dtype = (
            torch.float8_e4m3fn,
            torch.uint8,
            torch.float8_e4m3fn,
        )
        kv_shape = (batch * pages, hkv, page, 64)
    query = torch.empty((batch * q_len, hq, 128), dtype=q_dtype, device=device)
    key = torch.empty(kv_shape, dtype=kv_dtype, device=device)
    scales = None
    if component == "context_nvfp4":
        scales = torch.empty(
            (batch * pages, hkv, page, 8), dtype=torch.float8_e4m3fn, device=device
        )
    tables = (batch, pages) if shared else (batch, 2, pages)

    def scale(value):
        if not device_scales:
            return value
        return torch.full((), value, dtype=torch.float32, device=device)

    return dict(
        query=query,
        key_cache=key,
        value_cache=torch.empty_like(key),
        out=torch.empty(query.shape, dtype=o_dtype, device=device),
        block_tables=torch.zeros(tables, dtype=torch.int32, device=device),
        seq_lens=torch.full((batch,), kv_len, dtype=torch.int32, device=device),
        batch_size=batch,
        max_q_len=q_len,
        max_kv_len=kv_len,
        window_left=-1,
        bmm1_scale=scale(128**-0.5),
        bmm2_scale=scale(1.0),
        sinks=None,
        uses_shared_paged_kv_idx=shared,
        cum_seq_lens_q=torch.arange(
            0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=device
        ),
        cum_seq_lens_kv=torch.arange(
            0, (batch + 1) * kv_len, kv_len, dtype=torch.int32, device=device
        ),
        key_block_scales=scales,
        value_block_scales=None if scales is None else torch.empty_like(scales),
        skip_softmax_threshold_scale_factor=None,
        is_causal=True,
        lse=None,
        kv_layout="HND",
        workspace_buffer=torch.empty(1 << 24, dtype=torch.uint8, device=device),
    )


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_cake_fmha_context_route_serves_shape_rows_as_pinned(
    monkeypatch, target, row
) -> None:
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    component = SHAPE_ROWS[row][0]
    program = SHAPE_ROWS[row][-1][target]
    # Meta tensors: route selection reads no tensor value.
    kwargs = _context_kwargs(row, device="meta")
    route = cake_api.select_cake_fmha_context_route(torch.device("meta"), **kwargs)
    assert route is not None and route.component == component
    assert route.program == program and route.device_scales is False
    assert cake_api.cake_fmha_route_is_optimized(route)
    if component == "context_bf16":
        assert route.exact_profile == (
            {"Q511": "q511", "Q257": "q257"}.get(program[:4])
        )
        return
    if program in _RUNTIME_PACK:
        # the runtime-size program's pack and the m-block count of the call
        pack = _RUNTIME_PACK[program]
        q_len = SHAPE_ROWS[row][2]
        assert route.pack_g == pack
        assert route.num_m_blocks == -(-q_len // (2 * (128 // pack)))
    device_route = cake_api.select_cake_fmha_context_route(
        torch.device("meta"), **_context_kwargs(row, device_scales=True, device="meta")
    )
    assert device_route is not None
    assert device_route.device_scales is True
    assert device_route.program == _device_scale_program(program)
    # One device scale and one host scale: the host-scalar program (the
    # prefill host path resolves the device value first).
    mixed = cake_api.select_cake_fmha_context_route(
        torch.device("meta"),
        **{
            **kwargs,
            "bmm1_scale": torch.empty((), dtype=torch.float32, device="meta"),
        },
    )
    assert mixed is not None
    assert mixed.device_scales is False and mixed.program == program


# The widened classes: name -> (shape row, batch, max_q_len).  A program
# serves any batch of its class (no L2-section or batch-parity key) and every
# planned query length of the same m-block count.
WIDENED_ROWS = {
    "bf16_b128_class_b1": ("bf16_b128_q64_k2048", 1, 64),
    "bf16_b128_class_b3": ("bf16_b128_q64_k2048", 3, 64),
    "bf16_b128_class_b127": ("bf16_b128_q64_k2048", 127, 64),
    "bf16_b128_class_b255": ("bf16_b128_q64_k2048", 255, 64),
    "bf16_b128_class_q33": ("bf16_b128_q64_k2048", 128, 33),
    "bf16_q8192_class_b2": ("bf16_b1_q8192_k8192", 2, 8192),
    "bf16_q8192_class_b3": ("bf16_b1_q8192_k8192", 3, 8192),
    "bf16_q8192_class_q8191": ("bf16_b1_q8192_k8192", 2, 8191),
    "bf16_q8192_class_q8161": ("bf16_b1_q8192_k8192", 1, 8161),
    "fp8_b256_class_b1": ("fp8_b256_q32_k1024", 1, 32),
    "fp8_b256_class_b3": ("fp8_b256_q32_k1024", 3, 32),
    "fp8_b256_class_b127": ("fp8_b256_q32_k1024", 127, 32),
    "fp8_b256_class_b255": ("fp8_b256_q32_k1024", 255, 32),
    "nvfp4_q511_class_b1": ("nvfp4_b4_q511_k2047", 1, 511),
    "nvfp4_q511_class_b3": ("nvfp4_b4_q511_k2047", 3, 511),
    "nvfp4_q511_class_b127": ("nvfp4_b4_q511_k2047", 127, 511),
    "nvfp4_q511_class_b255": ("nvfp4_b4_q511_k2047", 255, 511),
    "nvfp4_q511_class_q481": ("nvfp4_b4_q511_k2047", 4, 481),
    "fp8_runtime_g4_class_b1": ("fp8_b64_q32_k1024_g4", 1, 32),
    "fp8_runtime_g4_class_b255": ("fp8_b64_q32_k1024_g4", 255, 32),
    "fp8_runtime_g3_class_b5": ("fp8_b2_q256_k2048_g3_page32", 5, 256),
}


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize("name", sorted(WIDENED_ROWS))
def test_cake_fmha_context_widened_classes_route_to_the_row_program(
    monkeypatch, target, name
) -> None:
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    row, batch, q_len = WIDENED_ROWS[name]
    program = SHAPE_ROWS[row][-1][target]
    for device_scales in (False, True) if row[:4] != "bf16" else (False,):
        kwargs = _context_kwargs(
            row, device_scales=device_scales, device="meta", batch=batch, q_len=q_len
        )
        route = cake_api.select_cake_fmha_context_route(torch.device("meta"), **kwargs)
        assert route is not None and route.device_scales is device_scales
        expected = _device_scale_program(program) if device_scales else program
        assert route.program == expected, (name, route)


@pytest.mark.parametrize(
    ("row", "mutation"),
    [
        ("bf16_b128_q64_k2048", {"is_causal": False}),
        ("bf16_b128_q64_k2048", {"lse": "lse"}),
        ("bf16_b128_q64_k2048", {"sinks": "sinks"}),
        ("fp8_b256_q32_k1024", {"lse": "lse"}),
        ("fp8_b64_q32_k1024_g4", {"lse": "lse"}),
        ("fp8_b64_q32_k1024_g4", {"sinks": "sinks"}),
        ("fp8_b64_q32_k1024_g4", {"is_causal": False}),
        ("fp8_b4_q128_k1024_g6_page16", {"is_causal": False}),
        ("nvfp4_b4_q511_k2047", {"sinks": "sinks"}),
        ("bf16_b128_q64_k2048", {"max_q_len": 32}),
        ("bf16_b128_q64_k2048", {"max_q_len": 65}),
        ("bf16_b1_q8192_k8192", {"max_q_len": 8160}),
        ("nvfp4_b4_q511_k2047", {"max_q_len": 480}),
        ("bf16_b4_q511_k2047", {"max_kv_len": 2048}),
        ("bf16_b4_q511_k2047", {"kv_layout": "NHD"}),
        ("bf16_b4_q257_k1024_page1024", {"max_q_len": 256}),
        # sm100a, 256 x 8 (request, KV head) streams: the power limit keeps the
        # JIT module where the query tokens need two stages of its pack
        ("fp8_b256_q32_k1024", {"num_kv_heads": 8, "max_q_len": 33}),
    ],
)
def test_cake_fmha_context_routes_outside_the_programs_keep_the_jit_module(
    monkeypatch, row, mutation
) -> None:
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    kwargs = _context_kwargs(row, device="meta")
    mutation = dict(mutation)
    if mutation.pop("lse", None):
        kwargs["lse"] = torch.empty(
            (kwargs["query"].shape[0], kwargs["query"].shape[1]),
            dtype=torch.float32,
            device="meta",
        )
    if mutation.pop("sinks", None):
        kwargs["sinks"] = torch.empty(
            (kwargs["query"].shape[1],), dtype=torch.float32, device="meta"
        )
    if "num_kv_heads" in mutation:
        key = kwargs["key_cache"]
        kwargs["key_cache"] = torch.empty(
            (key.shape[0], mutation.pop("num_kv_heads"), *key.shape[2:]),
            dtype=key.dtype,
            device="meta",
        )
        kwargs["value_cache"] = torch.empty_like(kwargs["key_cache"])
    kwargs.update(mutation)
    route = cake_api.select_cake_fmha_context_route(torch.device("meta"), **kwargs)
    assert route is None or route.program is None, (row, mutation, route)


# --- The FP8 runtime-size route (host-only) ---------------------------------

# Every measured FP8 shape: (target, shape, batch, max_q_len, max_kv_len,
# num_q_heads, num_kv_heads, the JIT pack, served by the runtime-size
# program).  Each shape was timed against the JIT module in three fresh
# back-to-back processes with an A/A band; a shape within 2 % of it counts as
# a win only when it also wins under five seeded memory layouts.  The route
# gives every measured loss to the JIT module, and every measured win but
# three to the runtime-size program (sm100a ``g4_q4`` under the sm100a stream
# rule, ``e_g5_causal`` on both targets under the tile-work rule).
_FP8_RUNTIME_ROUTE_ROWS = [
    ("sm103a", "g1_q4", 256, 4, 1024, 32, 32, 1, True),
    ("sm103a", "g1_q32", 256, 32, 1024, 32, 32, 1, True),
    ("sm103a", "g1_q256", 16, 256, 2048, 32, 32, 1, True),
    ("sm103a", "g2_q4", 256, 4, 1024, 32, 16, 2, True),
    ("sm103a", "g2_q32", 256, 32, 1024, 32, 16, 2, True),
    ("sm103a", "g2_q256", 16, 256, 2048, 32, 16, 2, True),
    ("sm103a", "g3_q4", 256, 4, 1024, 30, 10, 3, False),
    ("sm103a", "g3_q32", 256, 32, 1024, 30, 10, 3, False),
    ("sm103a", "g3_q256", 16, 256, 2048, 30, 10, 1, True),
    ("sm103a", "g4_q4", 256, 4, 1024, 32, 8, 4, True),
    ("sm103a", "g4_q32", 256, 32, 1024, 32, 8, 4, True),
    ("sm103a", "g4_q256", 16, 256, 2048, 32, 8, 4, True),
    ("sm103a", "g5_q4", 256, 4, 1024, 30, 6, 5, False),
    ("sm103a", "g5_q32", 256, 32, 1024, 30, 6, 5, False),
    ("sm103a", "g5_q256", 16, 256, 2048, 30, 6, 1, True),
    ("sm103a", "g6_q4", 256, 4, 1024, 30, 5, 6, False),
    ("sm103a", "g6_q32", 256, 32, 1024, 30, 5, 6, False),
    ("sm103a", "g6_q256", 16, 256, 2048, 30, 5, 1, True),
    ("sm103a", "g7_q4", 256, 4, 1024, 28, 4, 7, False),
    ("sm103a", "g7_q32", 256, 32, 1024, 28, 4, 7, False),
    ("sm103a", "g7_q256", 16, 256, 2048, 28, 4, 1, True),
    ("sm103a", "g8_q4", 256, 4, 1024, 32, 4, 8, True),
    ("sm103a", "g8_q32", 256, 32, 1024, 32, 4, 8, True),
    ("sm103a", "g8_q256", 16, 256, 2048, 32, 4, 8, True),
    ("sm103a", "g16_q4", 256, 4, 1024, 32, 2, 16, False),
    ("sm103a", "g16_q32", 256, 32, 1024, 32, 2, 16, True),
    ("sm103a", "g16_q256", 16, 256, 2048, 32, 2, 16, True),
    ("sm103a", "g32_q4", 256, 4, 1024, 32, 1, 32, False),
    ("sm103a", "g32_q32", 256, 32, 1024, 32, 1, 32, True),
    ("sm103a", "g32_q256", 16, 256, 2048, 32, 1, 32, True),
    ("sm103a", "g64_q4", 256, 4, 1024, 64, 1, 64, False),
    ("sm103a", "g64_q32", 256, 32, 1024, 64, 1, 64, True),
    ("sm103a", "g64_q256", 16, 256, 2048, 64, 1, 64, True),
    ("sm103a", "g128_q4", 256, 4, 1024, 128, 1, 128, False),
    ("sm103a", "g128_q32", 256, 32, 1024, 128, 1, 128, True),
    ("sm103a", "g128_q256", 16, 256, 2048, 128, 1, 128, True),
    ("sm103a", "contract", 256, 32, 1024, 32, 4, 8, True),
    ("sm103a", "s_b256_g4", 256, 32, 1024, 32, 8, 4, True),
    ("sm103a", "s_b256_g2", 256, 32, 1024, 32, 16, 2, True),
    ("sm103a", "s_b64_q16_g8", 64, 16, 2048, 32, 4, 8, True),
    ("sm103a", "s_b32_q128_g8", 32, 128, 4096, 32, 4, 8, True),
    ("sm103a", "n_q1_b255_p64", 255, 1, 1024, 32, 4, 8, True),
    ("sm103a", "n_g8_h64_q32_b3", 3, 32, 2000, 64, 8, 8, True),
    ("sm103a", "e_g5_causal", 2, 200, 700, 10, 2, 5, False),
    ("sm103a", "n_g4_q8_b6", 6, 8, 1024, 16, 4, 4, True),
    ("sm103a", "n_q300_b2_p16", 2, 300, 1500, 32, 4, 8, True),
    ("sm103a", "e_tail_h64", 1, 1000, 1333, 64, 8, 8, True),
    ("sm103a", "g1_q64", 256, 64, 1024, 32, 32, 1, True),
    ("sm103a", "g1_q128", 256, 128, 1024, 32, 32, 1, True),
    ("sm103a", "g2_q64", 256, 64, 1024, 32, 16, 2, True),
    ("sm103a", "g2_q128", 256, 128, 1024, 32, 16, 2, True),
    ("sm103a", "g4_q64", 256, 64, 1024, 32, 8, 4, True),
    ("sm103a", "g4_q128", 256, 128, 1024, 32, 8, 4, True),
    ("sm103a", "g8_q64", 256, 64, 1024, 32, 4, 8, True),
    ("sm103a", "g8_q128", 256, 128, 1024, 32, 4, 8, True),
    ("sm103a", "g3_q64", 256, 64, 1024, 30, 10, 3, False),
    ("sm103a", "g3_q128", 256, 128, 1024, 30, 10, 3, False),
    ("sm103a", "g5_q64", 256, 64, 1024, 30, 6, 5, False),
    ("sm103a", "g5_q128", 256, 128, 1024, 30, 6, 5, False),
    ("sm103a", "g5_q192", 256, 192, 1024, 30, 6, 5, False),
    ("sm103a", "g6_q64", 256, 64, 1024, 30, 5, 6, False),
    ("sm103a", "g6_q128", 256, 128, 1024, 30, 5, 6, True),
    ("sm103a", "g7_q64", 256, 64, 1024, 28, 4, 7, False),
    ("sm103a", "g7_q128", 256, 128, 1024, 28, 4, 7, False),
    ("sm103a", "g7_q192", 256, 192, 1024, 28, 4, 7, False),
    ("sm103a", "g16_q16", 256, 16, 1024, 32, 2, 16, False),
    ("sm103a", "g32_q16", 256, 16, 1024, 32, 1, 32, False),
    ("sm100a", "g1_q4", 256, 4, 1024, 32, 32, 1, False),
    ("sm100a", "g1_q32", 256, 32, 1024, 32, 32, 1, False),
    ("sm100a", "g1_q256", 16, 256, 2048, 32, 32, 1, True),
    ("sm100a", "g2_q4", 256, 4, 1024, 32, 16, 2, False),
    ("sm100a", "g2_q32", 256, 32, 1024, 32, 16, 2, False),
    ("sm100a", "g2_q256", 16, 256, 2048, 32, 16, 2, True),
    ("sm100a", "g3_q4", 256, 4, 1024, 30, 10, 3, False),
    ("sm100a", "g3_q32", 256, 32, 1024, 30, 10, 3, False),
    ("sm100a", "g3_q256", 16, 256, 2048, 30, 10, 1, True),
    ("sm100a", "g4_q4", 256, 4, 1024, 32, 8, 4, False),
    ("sm100a", "g4_q32", 256, 32, 1024, 32, 8, 4, False),
    ("sm100a", "g4_q256", 16, 256, 2048, 32, 8, 4, True),
    ("sm100a", "g5_q4", 256, 4, 1024, 30, 6, 5, False),
    ("sm100a", "g5_q32", 256, 32, 1024, 30, 6, 5, False),
    ("sm100a", "g5_q256", 16, 256, 2048, 30, 6, 1, True),
    ("sm100a", "g6_q4", 256, 4, 1024, 30, 5, 6, False),
    ("sm100a", "g6_q32", 256, 32, 1024, 30, 5, 6, False),
    ("sm100a", "g6_q256", 16, 256, 2048, 30, 5, 1, True),
    ("sm100a", "g7_q4", 256, 4, 1024, 28, 4, 7, False),
    ("sm100a", "g7_q32", 256, 32, 1024, 28, 4, 7, False),
    ("sm100a", "g7_q256", 16, 256, 2048, 28, 4, 1, True),
    ("sm100a", "g8_q4", 256, 4, 1024, 32, 4, 8, True),
    ("sm100a", "g8_q32", 256, 32, 1024, 32, 4, 8, True),
    ("sm100a", "g8_q256", 16, 256, 2048, 32, 4, 8, True),
    ("sm100a", "g16_q4", 256, 4, 1024, 32, 2, 16, False),
    ("sm100a", "g16_q32", 256, 32, 1024, 32, 2, 16, True),
    ("sm100a", "g16_q256", 16, 256, 2048, 32, 2, 16, True),
    ("sm100a", "g32_q4", 256, 4, 1024, 32, 1, 32, False),
    ("sm100a", "g32_q32", 256, 32, 1024, 32, 1, 32, True),
    ("sm100a", "g32_q256", 16, 256, 2048, 32, 1, 32, True),
    ("sm100a", "g64_q4", 256, 4, 1024, 64, 1, 64, False),
    ("sm100a", "g64_q32", 256, 32, 1024, 64, 1, 64, True),
    ("sm100a", "g64_q256", 16, 256, 2048, 64, 1, 64, True),
    ("sm100a", "g128_q4", 256, 4, 1024, 128, 1, 128, False),
    ("sm100a", "g128_q32", 256, 32, 1024, 128, 1, 128, True),
    ("sm100a", "g128_q256", 16, 256, 2048, 128, 1, 128, True),
    ("sm100a", "contract", 256, 32, 1024, 32, 4, 8, True),
    ("sm100a", "s_b256_g4", 256, 32, 1024, 32, 8, 4, False),
    ("sm100a", "s_b256_g2", 256, 32, 1024, 32, 16, 2, False),
    ("sm100a", "s_b64_q16_g8", 64, 16, 2048, 32, 4, 8, True),
    ("sm100a", "s_b32_q128_g8", 32, 128, 4096, 32, 4, 8, True),
    ("sm100a", "n_q1_b255_p64", 255, 1, 1024, 32, 4, 8, True),
    ("sm100a", "n_g8_h64_q32_b3", 3, 32, 2000, 64, 8, 8, True),
    ("sm100a", "e_g5_causal", 2, 200, 700, 10, 2, 5, False),
    ("sm100a", "n_g4_q8_b6", 6, 8, 1024, 16, 4, 4, True),
    ("sm100a", "n_q300_b2_p16", 2, 300, 1500, 32, 4, 8, True),
    ("sm100a", "e_tail_h64", 1, 1000, 1333, 64, 8, 8, True),
    ("sm100a", "g1_q64", 256, 64, 1024, 32, 32, 1, False),
    ("sm100a", "g1_q128", 256, 128, 1024, 32, 32, 1, False),
    ("sm100a", "g2_q64", 256, 64, 1024, 32, 16, 2, False),
    ("sm100a", "g2_q128", 256, 128, 1024, 32, 16, 2, False),
    ("sm100a", "g4_q64", 256, 64, 1024, 32, 8, 4, False),
    ("sm100a", "g4_q128", 256, 128, 1024, 32, 8, 4, False),
    ("sm100a", "g8_q64", 256, 64, 1024, 32, 4, 8, True),
    ("sm100a", "g8_q128", 256, 128, 1024, 32, 4, 8, False),
    ("sm100a", "g1_q32_b64", 64, 32, 1024, 32, 32, 1, False),
    ("sm100a", "g1_q32_b128", 128, 32, 1024, 32, 32, 1, False),
    ("sm100a", "g2_q32_b64", 64, 32, 1024, 32, 16, 2, True),
    ("sm100a", "g2_q32_b128", 128, 32, 1024, 32, 16, 2, False),
    ("sm100a", "g4_q32_b64", 64, 32, 1024, 32, 8, 4, True),
    ("sm100a", "g4_q32_b128", 128, 32, 1024, 32, 8, 4, True),
    ("sm100a", "g8_q128_b64", 64, 128, 1024, 32, 4, 8, True),
    ("sm100a", "g8_q128_b128", 128, 128, 1024, 32, 4, 8, True),
    ("sm100a", "g4_q128_b64", 64, 128, 1024, 32, 8, 4, True),
    ("sm100a", "g4_q128_b128", 128, 128, 1024, 32, 8, 4, True),
    ("sm100a", "g3_q64", 256, 64, 1024, 30, 10, 3, False),
    ("sm100a", "g3_q128", 256, 128, 1024, 30, 10, 3, False),
    ("sm100a", "g5_q64", 256, 64, 1024, 30, 6, 5, False),
    ("sm100a", "g5_q128", 256, 128, 1024, 30, 6, 5, False),
    ("sm100a", "g5_q192", 256, 192, 1024, 30, 6, 5, False),
    ("sm100a", "g6_q64", 256, 64, 1024, 30, 5, 6, False),
    ("sm100a", "g6_q128", 256, 128, 1024, 30, 5, 6, True),
    ("sm100a", "g7_q64", 256, 64, 1024, 28, 4, 7, False),
    ("sm100a", "g7_q128", 256, 128, 1024, 28, 4, 7, False),
    ("sm100a", "g7_q192", 256, 192, 1024, 28, 4, 7, False),
    ("sm100a", "g16_q16", 256, 16, 1024, 32, 2, 16, False),
    ("sm100a", "g32_q16", 256, 16, 1024, 32, 1, 32, False),
]  # fmt: skip


@pytest.mark.parametrize(
    (
        "target",
        "shape",
        "batch",
        "max_q",
        "max_kv",
        "num_q_heads",
        "num_kv_heads",
        "pack_g",
        "serves",
    ),
    _FP8_RUNTIME_ROUTE_ROWS,
    ids=[f"{row[0]}-{row[1]}" for row in _FP8_RUNTIME_ROUTE_ROWS],
)
def test_cake_fmha_context_fp8_runtime_route(
    monkeypatch,
    target,
    shape,
    batch,
    max_q,
    max_kv,
    num_q_heads,
    num_kv_heads,
    pack_g,
    serves,
) -> None:
    """The row's JIT pack, the predicate's route and the route selection
    takes for a causal FP8 call of the row's shape with 16-token pages (no
    exact route): the runtime-size program of the largest shipped pack
    dividing the JIT pack, or the JIT module."""

    assert cake_api._context_pack_g(max_q, max_kv, num_q_heads, num_kv_heads) == pack_g
    assert (
        cake_api._context_fp8_runtime_serves(
            target, batch, max_q, max_kv, num_q_heads, num_kv_heads, pack_g
        )
        is serves
    )
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    row = (
        "context_fp8",
        batch,
        max_q,
        max_kv,
        num_q_heads,
        num_kv_heads,
        16,
        True,
        None,
    )
    route = cake_api.select_cake_fmha_context_route(
        torch.device("meta"), **_context_kwargs(row, device="meta")
    )
    assert route is not None and route.component == "context_fp8"
    if serves:
        pack = cake_api._context_fp8_runtime_pack(pack_g)
        assert pack_g % pack == 0 and pack in (1, 2, 4, 8)
        assert route.program == f"RUNTIME_PACK{pack}{_RUNTIME_SUFFIX[target]}"
        assert route.pack_g == pack
    elif (
        max_q <= 128 // pack_g and cake_api._context_fp8_runtime_pack(pack_g) == pack_g
    ):
        # the power limit keeps the JIT module, whose tiles would hold an empty
        # second stage: the one-stage program of the JIT pack (sm100a)
        assert target in cake_api._CONTEXT_FP8_POWER_ONE_STAGE_TARGETS
        assert cake_api._context_fp8_power_one_stage(
            target, batch, max_q, max_kv, num_q_heads, num_kv_heads, pack_g
        )
        assert route.program == f"RUNTIME_PACK{pack_g}{_RUNTIME_SUFFIX[target]}_Q1"
        assert route.pack_g == pack_g and route.num_m_blocks == 1
    else:
        assert not cake_api._context_fp8_power_one_stage(
            target, batch, max_q, max_kv, num_q_heads, num_kv_heads, pack_g
        )
        assert route.program is None and route.pack_g == pack_g


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_cake_fmha_context_fp8_exact_routes_come_first(monkeypatch, target) -> None:
    """The exact routes keep their keys; a call of the exact class off its
    m-block count takes the runtime-size program of its pack and page size,
    for host and device scales alike."""

    routes = cake_api._CONTEXT_PROGRAM_ROUTES[target]
    exact = _FP8_EXACT[target]
    assert routes[("context_fp8", "host", 32, 4, 8, 1, 64, None)] == exact
    assert (
        routes[("context_fp8", "device", 32, 4, 8, 1, 64, None)]
        == _DEVICE_SCALE_PROGRAM[exact]
    )
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    pack = cake_api._context_fp8_runtime_pack(cake_api._context_pack_g(33, 1024, 32, 4))
    for device_scales in (False, True):
        kwargs = _context_kwargs(
            "fp8_b256_q32_k1024", device_scales=device_scales, device="meta", q_len=33
        )
        route = cake_api.select_cake_fmha_context_route(torch.device("meta"), **kwargs)
        assert route is not None and route.device_scales is device_scales
        assert route.program == f"RUNTIME_PACK{pack}_PAGE64{_RUNTIME_SUFFIX[target]}", (
            route
        )
        assert route.num_m_blocks == -(-33 // (2 * (128 // pack)))


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_cake_fmha_context_fp8_short_kv_routes(monkeypatch, target) -> None:
    """A launch of at most the keys of its exact, runtime-size or one-stage
    route's short-KV entry, where the target names one, takes the entry's
    program (sm103a: the same schedule with each stage's P tile in its own S
    columns; 256 keys for the exact programs, 128 for the runtime-size and
    the every-page-size one-stage ones), for host and device scales alike;
    one key more keeps the route's program."""

    short = cake_api._CONTEXT_FP8_SHORT_KV_ROUTES.get(target, {})
    routed = {
        *cake_api._CONTEXT_PROGRAM_ROUTES[target].values(),
        *cake_api._CONTEXT_FP8_RUNTIME_ROUTES[target].values(),
        *cake_api._CONTEXT_FP8_ONE_STAGE_ROUTES[target].values(),
    }
    assert set(short) <= routed and (target == "sm103a") == bool(short)
    one_stage = {swapped for swapped in short if swapped.endswith("_Q1")}
    assert (
        one_stage
        == {
            identifier
            for (_, page), identifier in cake_api._CONTEXT_FP8_ONE_STAGE_ROUTES[
                target
            ].items()
            if page is None
        }
        or not short
    )
    for swapped, (identifier, keys) in short.items():
        assert keys == (256 if identifier.startswith("B256") else 128)
        assert (
            cake_api._CONTEXT_PROGRAMS[("context_fp8", swapped)][1:]
            == cake_api._CONTEXT_PROGRAMS[("context_fp8", identifier)][1:]
        )
        if swapped in one_stage:
            assert swapped == f"{identifier.removesuffix('_Q1')}_PX_Q1"
        else:
            assert swapped == f"{identifier.removesuffix('_KV4')}_PX"
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    for row in ("fp8_b16_q32_k128", "fp8_b16_q32_k128_g4"):
        component, batch, q_len, _, hq, hkv, page, shared, _ = SHAPE_ROWS[row]
        short_program = SHAPE_ROWS[row][-1][target]
        swapped = {v[0]: k for k, v in short.items()}.get(short_program)
        keys = short[swapped][1] if swapped else 128
        for kv_len in (keys, keys + 1):
            shape = (component, batch, q_len, kv_len, hq, hkv, page, shared, None)
            for device_scales in (False, True):
                kwargs = _context_kwargs(
                    shape, device_scales=device_scales, device="meta"
                )
                route = cake_api.select_cake_fmha_context_route(
                    torch.device("meta"), **kwargs
                )
                assert route is not None and route.program is not None
                program = short_program
                if kv_len > keys and swapped:
                    program = swapped
                if device_scales:
                    program = _device_scale_program(program)
                assert route.program == program, (row, kv_len, route)


# --- The FP8 split-KV route (host-only) --------------------------------------

_SM_COUNT = {"sm100a": 148, "sm103a": 152}


def test_cake_fmha_context_fp8_split_routes_mirror_the_runtime_routes() -> None:
    """Every runtime-size program has the split-KV program of its pack and page
    size on the same targets, and no other split-KV program exists."""

    runtime = cake_api._CONTEXT_FP8_RUNTIME_ROUTES
    assert set(cake_api._CONTEXT_FP8_SPLIT_ROUTES) == set(runtime)
    for target, routes in runtime.items():
        assert cake_api._CONTEXT_FP8_SPLIT_ROUTES[target] == {
            key: "RUNTIME_SK" + identifier.removeprefix("RUNTIME").removesuffix("_PX")
            for key, identifier in routes.items()
        }
    assert frozenset(_SPLIT_PACK) == cake_api._CONTEXT_FP8_SPLIT_PROGRAMS


def test_cake_fmha_context_fp8_tile_blocks_follow_the_program_tile_order() -> None:
    """The planner's tiles in the program's order: sections of 8 work units
    over every m-block, the largest m-block first; bottom-right causal KV
    blocks per m-block, every request at the planned maxima."""

    # 2 requests of 300 query tokens over 1500 keys, 32 query heads in packs
    # of 8: 8 work units x 10 m-blocks of 32 tokens, one whole section
    blocks = cake_api._context_fp8_tile_blocks(2, 300, 1500, 32, 8)
    per_m = [-(-min(1500, (m + 1) * 32 + 1200) // 128) for m in range(10)]
    assert blocks == [per_m[9 - i // 8] for i in range(80)]
    # 3 work units (no whole section): each unit's m-blocks in turn
    blocks = cake_api._context_fp8_tile_blocks(3, 256, 512, 1, 1)
    per_m = [-(-min(512, (m + 1) * 256 + 256) // 128) for m in range(1)]
    assert blocks == per_m * 3
    blocks = cake_api._context_fp8_tile_blocks(1, 1000, 1000, 3, 1)
    per_m = [
        -(-((m + 1) * 256) // 128) if (m + 1) * 256 < 1000 else 8 for m in range(4)
    ]
    assert blocks == [per_m[3 - m] for _ in range(3) for m in range(4)]


# (batch, max_q_len, max_kv_len, num_q_heads, pack, splits): launches whose
# tiles leave SMs idle split; a launch of many waves does not.
_SPLIT_PLAN_ROWS = [
    (1, 32, 16384, 32, 8, True),
    (1, 32, 4096, 32, 8, True),
    (2, 32, 16384, 32, 8, True),
    (1, 128, 16384, 32, 8, True),
    (256, 32, 1024, 32, 8, False),
    (256, 4, 1024, 32, 1, False),
    (256, 32, 1024, 32, 1, False),
    (1, 32, 128, 32, 8, False),
]


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize(
    ("batch", "max_q", "max_kv", "num_q_heads", "pack", "splits"), _SPLIT_PLAN_ROWS
)
def test_cake_fmha_context_fp8_split_plan(
    target, batch, max_q, max_kv, num_q_heads, pack, splits
) -> None:
    """The plan splits at most ``sm_count - 1`` tiles into 2 to 16 units each,
    and only where the cost model predicts it faster than the whole-tile
    schedule by both margins."""

    sm_count = _SM_COUNT[target]
    blocks = cake_api._context_fp8_tile_blocks(batch, max_q, max_kv, num_q_heads, pack)
    assert len(blocks) == -(-max_q // (2 * (128 // pack))) * batch * (
        num_q_heads // pack
    )
    plan = cake_api._context_fp8_split_plan(
        target, batch, max_q, max_kv, num_q_heads, pack, sm_count
    )
    assert (plan is not None) is splits
    if plan is None:
        return
    split_tiles, split_blocks, num_splits = plan
    assert 1 <= split_tiles <= min(len(blocks), sm_count - 1)
    assert 2 <= num_splits <= cake_api._CONTEXT_FP8_SPLIT_MAX
    assert split_blocks >= 1 and -(-max(blocks) // split_blocks) >= num_splits
    costs = cake_api._CONTEXT_FP8_SPLIT_COSTS[target]
    whole = cake_api._context_fp8_split_makespan(
        blocks,
        sm_count,
        0,
        1,
        1,
        cake_api._context_fp8_whole_costs(target, max_q, max_kv, pack, costs),
    )
    split = cake_api._context_fp8_split_makespan(blocks, sm_count, *plan, costs)
    assert split <= whole * (1 - cake_api._CONTEXT_FP8_SPLIT_REL_MARGIN)
    assert whole - split >= cake_api._CONTEXT_FP8_SPLIT_ABS_MARGIN


# (target, batch, max_q_len, max_kv_len, num_q_heads, pack, splits): launches
# near the margins (2 % and 0.5 us) whose plan differs between the targets.
_SPLIT_MARGIN_ROWS = [
    # B200: 0.58 us (2.1 %) predicted; GB300: no split predicted faster
    ("sm100a", 2, 300, 1500, 32, 8, True),
    ("sm103a", 2, 300, 1500, 32, 8, False),
    # GB300: the split's 0.84 us (3.0 % and 2.2 %) over whole tiles priced
    # like split units is gone: the whole tiles run the swapped P placement
    # (_CONTEXT_FP8_SPLIT_WHOLE_BLOCK)
    ("sm103a", 2, 300, 2000, 32, 8, False),
    ("sm103a", 2, 300, 3000, 32, 8, False),
    # 16 requests of 256 query tokens over 2048 keys, one head per tile: B200
    # 2.7 % predicted; GB300 1.5 %, below the margin
    ("sm100a", 16, 256, 2048, 32, 1, True),
    ("sm103a", 16, 256, 2048, 32, 1, False),
]


@pytest.mark.parametrize(
    ("target", "batch", "max_q", "max_kv", "num_q_heads", "pack", "splits"),
    _SPLIT_MARGIN_ROWS,
)
def test_cake_fmha_context_fp8_split_plan_margins(
    target, batch, max_q, max_kv, num_q_heads, pack, splits
) -> None:
    """Launches whose best split the cost model predicts faster by about the
    margins split exactly where it beats both (2 % and 0.5 us)."""

    sm_count = _SM_COUNT[target]
    assert cake_api._CONTEXT_FP8_SPLIT_REL_MARGIN == 0.02
    assert cake_api._CONTEXT_FP8_SPLIT_ABS_MARGIN == 0.5
    plan = cake_api._context_fp8_split_plan(
        target, batch, max_q, max_kv, num_q_heads, pack, sm_count
    )
    assert (plan is not None) is splits


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_cake_fmha_context_fp8_split_costs_scale_with_the_tile_rows(target) -> None:
    """A launch's split costs are the target's with the unit, the partial
    store and the merge scaled by the query rows of its tiles: whole for 256
    rows and for launches of several m-blocks, less for fewer rows; the
    launch's fixed cost, the KV block and the extra units' share never
    scale."""

    costs = cake_api._CONTEXT_FP8_SPLIT_COSTS[target]
    s0, u0 = cake_api._CONTEXT_FP8_SPLIT_ROWS[target]
    assert 0.0 <= s0 <= 1.0 and 0.0 <= u0 <= 1.0
    scaled = cake_api._context_fp8_split_costs
    # 32 tokens x pack 8: 256 rows; 300 tokens: several m-blocks of any pack
    assert scaled(target, 32, 8) == costs
    assert scaled(target, 300, 8) == scaled(target, 300, 1) == costs
    previous = costs
    for pack in (4, 2, 1):
        rows = 32 * pack
        got = scaled(target, 32, pack)
        assert (got[0], got[1], got[6]) == (costs[0], costs[1], costs[6])
        assert got[2] == pytest.approx(costs[2] * (u0 + (1 - u0) * rows / 256))
        for index in (3, 4, 5):
            assert got[index] == pytest.approx(
                costs[index] * (s0 + (1 - s0) * rows / 256)
            )
        assert all(cost <= more for cost, more in zip(got, previous, strict=True))
        previous = got


# (batch, max_q_len, max_kv_len, num_q_heads, pack): launches the split choice
# sees; a smaller pack runs a plan only where it holds the launch in one
# m-block (a launch of several m-blocks only on the targets of
# _CONTEXT_FP8_SPLIT_M_BLOCK_TARGETS).
_SPLIT_CHOICE_ROWS = [
    (1, 32, 16384, 32, 8),
    (1, 32, 1024, 32, 8),
    (1, 32, 2048, 32, 8),
    (1, 32, 4096, 32, 8),
    (1, 32, 32768, 32, 8),
    (1, 32, 16384, 16, 8),
    (1, 24, 8192, 32, 8),
    (2, 32, 16384, 32, 8),
    (2, 32, 4096, 32, 8),
    (4, 32, 16384, 32, 8),
    (16, 32, 16384, 32, 8),
    (1, 16, 16384, 32, 8),
    (1, 32, 16384, 64, 8),
    (1, 128, 16384, 32, 8),
    (2, 300, 6000, 32, 8),
    (2, 128, 1500, 32, 8),
    (1, 128, 1500, 32, 8),
    (1, 64, 1500, 32, 8),
    (2, 256, 2048, 32, 8),
    (12, 96, 1536, 32, 4),
    (12, 48, 2048, 64, 8),
    (1, 64, 4096, 64, 4),
    (1, 32, 16384, 32, 4),
    (1, 256, 8192, 32, 1),
    (256, 32, 1024, 32, 8),
]


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize(
    ("batch", "max_q", "max_kv", "num_q_heads", "pack"), _SPLIT_CHOICE_ROWS
)
def test_cake_fmha_context_fp8_split_choice(
    target, batch, max_q, max_kv, num_q_heads, pack
) -> None:
    """The split choice keeps the pack's plan, or runs a smaller shipped pack
    whose tiles hold every query token in one m-block: for a launch of one
    m-block, the pack's split (``split_blocks``, ``num_splits``) over its
    tiles, or its own best split; for a launch of several m-blocks (on the
    targets of ``_CONTEXT_FP8_SPLIT_M_BLOCK_TARGETS`` only) its own best
    split; only where the cost model of each pack's tile rows predicts it
    faster than the pack's plan (or whole tiles) by both margins."""

    sm_count = _SM_COUNT[target]
    choice = cake_api._context_fp8_split_choice(
        target, batch, max_q, max_kv, num_q_heads, pack, sm_count
    )
    own = cake_api._context_fp8_split_plan(
        target, batch, max_q, max_kv, num_q_heads, pack, sm_count
    )
    if choice is None:
        assert own is None
        return
    chosen, plan = choice
    if chosen == pack:
        assert plan == own
        return
    assert chosen < pack and pack % chosen == 0
    assert chosen in (*cake_api._CONTEXT_FP8_RUNTIME_PACKS, 1)
    one_m_block = max_q <= 2 * (128 // pack)
    assert max_q <= 2 * (128 // chosen)
    if one_m_block:
        assert max_q <= 128 // chosen
    assert one_m_block or (
        target in cake_api._CONTEXT_FP8_SPLIT_M_BLOCK_TARGETS
        and batch * (num_q_heads // chosen) <= sm_count
    )
    blocks = cake_api._context_fp8_tile_blocks(batch, max_q, max_kv, num_q_heads, pack)
    smaller = cake_api._context_fp8_tile_blocks(
        batch, max_q, max_kv, num_q_heads, chosen
    )
    split_tiles, split_blocks, num_splits = plan
    assert 1 <= split_tiles <= min(len(smaller), sm_count - 1)
    assert 2 <= num_splits <= cake_api._CONTEXT_FP8_SPLIT_MAX
    repack = (
        one_m_block
        and own is not None
        and own[0] == len(blocks)
        and plan == (len(smaller), own[1], own[2])
    )
    if not repack:
        found = cake_api._context_fp8_split_search(
            target, batch, max_q, max_kv, num_q_heads, chosen, sm_count, True
        )
        assert plan == found[0]
    costs = cake_api._context_fp8_split_costs
    time = cake_api._context_fp8_split_makespan(
        blocks,
        sm_count,
        *(own or (0, 1, 1)),
        costs(target, max_q, pack)
        if own
        else cake_api._context_fp8_whole_costs(
            target, max_q, max_kv, pack, costs(target, max_q, pack)
        ),
    )
    faster = cake_api._context_fp8_split_makespan(
        smaller, sm_count, *plan, costs(target, max_q, chosen)
    )
    assert faster <= time * (1 - cake_api._CONTEXT_FP8_SPLIT_REL_MARGIN)
    assert time - faster >= cake_api._CONTEXT_FP8_SPLIT_ABS_MARGIN


def _clear_split_caches() -> None:
    for cached in (
        cake_api._context_fp8_split_costs,
        cake_api._context_fp8_split_search,
        cake_api._context_fp8_split_plan,
        cake_api._context_fp8_split_choice,
        cake_api._context_fp8_one_stage_choice,
    ):
        cached.cache_clear()


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_cake_fmha_context_fp8_split_choice_bounds(monkeypatch, target) -> None:
    """With the shipped costs, launches whose query tokens span several
    m-blocks (300 tokens) or that run many waves of tiles (256 requests) keep
    pack 8, while one request of 32 query tokens over 16384 keys runs pack 8's
    split over the 8 tiles of pack 4 (the same output bytes); where the tile
    rows do not scale the costs (``S0 = U0 = 1``) no launch changes pack (a
    smaller pack only adds tiles)."""

    sm_count = _SM_COUNT[target]
    choice = cake_api._context_fp8_split_choice
    for row in ((2, 300, 6000), (256, 32, 1024)):
        result = choice(target, *row, 32, 8, sm_count)
        assert result is None or result[0] == 8, row
    own = cake_api._context_fp8_split_plan(target, 1, 32, 16384, 32, 8, sm_count)
    assert choice(target, 1, 32, 16384, 32, 8, sm_count) == (4, (8, own[1], own[2]))
    rows = (
        (1, 32, 16384),
        (2, 32, 16384),
        (1, 32, 2048),
        (1, 32, 1024),
        (1, 16, 16384),
    )
    try:
        monkeypatch.setitem(cake_api._CONTEXT_FP8_SPLIT_ROWS, target, (1.0, 1.0))
        _clear_split_caches()
        for row in rows:
            result = choice(target, *row, 32, 8, sm_count)
            assert result is None or result[0] == 8, row
    finally:
        # the cached choices of the patched costs must not outlive the test
        _clear_split_caches()


def test_cake_fmha_context_fp8_split_choice_of_several_m_blocks(monkeypatch) -> None:
    """On B200 (sm100a) a launch of several m-blocks also takes the own split
    of a smaller pack that holds every query token in one m-block: 2 requests
    of 128 query tokens over 1500 keys, heads 32/4 (4 m-blocks of pack 8)
    run pack 1's split (64, 6, 2) instead of pack 8's (32, 4, 3), and route
    to pack 1's split-KV program.  12 requests of 96 query tokens over 1536
    keys, heads 32/8 (2 m-blocks of pack 4) keep pack 4's plan: pack 2's 192
    tiles fill more than one wave.  A launch of one m-block still runs pack
    8's split over pack 4's tiles.  On GB300 (sm103a), and on B200 without
    the target in ``_CONTEXT_FP8_SPLIT_M_BLOCK_TARGETS``, the launch of
    several m-blocks keeps its pack's plan."""

    choice = cake_api._context_fp8_split_choice
    plan = cake_api._context_fp8_split_plan
    row = (2, 128, 1500, 32)
    assert plan("sm100a", *row, 8, 148) == (32, 4, 3)
    assert choice("sm100a", *row, 8, 148) == (1, (64, 6, 2))
    own = plan("sm103a", *row, 8, 152)
    assert choice("sm103a", *row, 8, 152) == (None if own is None else (8, own))
    wide = (12, 96, 1536, 32, 4, 148)
    own = plan("sm100a", *wide)
    assert choice("sm100a", *wide) == (None if own is None else (4, own))
    for target, sm_count in _SM_COUNT.items():
        own = plan(target, 1, 32, 16384, 32, 8, sm_count)
        assert choice(target, 1, 32, 16384, 32, 8, sm_count) == (4, (8, own[1], own[2]))

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: "sm100a")
    monkeypatch.setattr(cake_api, "_context_split_sm_count", lambda device: 148)
    # the split route's subject: no one-stage route takes the call (it does
    # with one, test_cake_fmha_context_fp8_one_stage_route)
    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_ONE_STAGE_ROUTES", {})
    meta = torch.device("meta")
    shape = ("context_fp8", 2, 128, 1500, 32, 4, 16, True, None)
    need = cake_api._context_fp8_split_workspace_bytes(64, 2)
    kwargs = dict(
        _context_kwargs(shape, device="meta"),
        workspace_buffer=torch.empty(need, dtype=torch.uint8, device="meta"),
    )
    route = cake_api.select_cake_fmha_context_route(meta, **kwargs)
    assert route.program == "RUNTIME_SK_PACK1", route
    assert (route.pack_g, route.num_m_blocks) == (1, 1)
    try:
        monkeypatch.setattr(cake_api, "_CONTEXT_FP8_SPLIT_M_BLOCK_TARGETS", frozenset())
        _clear_split_caches()
        assert choice("sm100a", *row, 8, 148) == (8, (32, 4, 3))
        route = cake_api.select_cake_fmha_context_route(meta, **kwargs)
        assert route.program == "RUNTIME_SK_PACK8", route
        assert (route.pack_g, route.num_m_blocks) == (8, 4)
    finally:
        # the cached choices of the patched gate must not outlive the test
        _clear_split_caches()


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_cake_fmha_context_fp8_split_route(monkeypatch, target) -> None:
    """One request of 32 query tokens over 16384 keys (B256's exact class, 4
    tiles) takes the split-KV program of its pack and page size, for host and
    device scales, where the plan splits and the workspace holds its
    partials; with a workspace one byte short, off a CUDA device, or in a
    launch of many waves it keeps its program."""

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    row = ("context_fp8", 1, 32, 16384, 32, 4, 64, True, None)
    meta = torch.device("meta")
    kwargs = _context_kwargs(row, device="meta")
    assert (
        cake_api.select_cake_fmha_context_route(meta, **kwargs).program
        == _FP8_EXACT[target]
    )
    sm_count = _SM_COUNT[target]
    monkeypatch.setattr(cake_api, "_context_split_sm_count", lambda device: sm_count)
    pack, plan = cake_api._context_fp8_split_choice(
        target, 1, 32, 16384, 32, 8, sm_count
    )
    for device_scales in (False, True):
        route = cake_api.select_cake_fmha_context_route(
            meta, **_context_kwargs(row, device="meta", device_scales=device_scales)
        )
        assert route.program == f"RUNTIME_SK_PACK{pack}_PAGE64", route
        assert (route.pack_g, route.num_m_blocks) == (pack, 1)
        assert route.device_scales is device_scales
    own = cake_api._context_fp8_split_plan(target, 1, 32, 16384, 32, 8, sm_count)
    need = cake_api._context_fp8_split_workspace_bytes(own[0], own[2])
    short = dict(
        kwargs, workspace_buffer=torch.empty(need - 1, dtype=torch.uint8, device="meta")
    )
    assert (
        cake_api.select_cake_fmha_context_route(meta, **short).program
        == _FP8_EXACT[target]
    )
    enough = dict(
        kwargs, workspace_buffer=torch.empty(need, dtype=torch.uint8, device="meta")
    )
    # the runtime pack's own plan where the workspace cannot hold the chosen
    # (smaller) pack's partials
    assert (
        cake_api.select_cake_fmha_context_route(meta, **enough).program
        == "RUNTIME_SK_PACK8_PAGE64"
    )
    need = cake_api._context_fp8_split_workspace_bytes(plan[0], plan[2])
    chosen = dict(
        kwargs, workspace_buffer=torch.empty(need, dtype=torch.uint8, device="meta")
    )
    assert (
        cake_api.select_cake_fmha_context_route(meta, **chosen).program
        == f"RUNTIME_SK_PACK{pack}_PAGE64"
    )
    many = _context_kwargs("fp8_b256_q32_k1024", device="meta")
    assert (
        cake_api.select_cake_fmha_context_route(meta, **many).program
        == _FP8_EXACT[target]
    )


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize("page", (64, 16))
@pytest.mark.parametrize("skipped", ("short_workspace", "no_split_program"))
def test_cake_fmha_context_fp8_split_route_without_a_runtime_pack_plan(
    monkeypatch, target, page, skipped
) -> None:
    """One request of 4 query tokens over 1024 keys, heads 32/4 (B256's exact
    class at 64-token pages): pack 8 has no split plan, and the split choice
    runs a smaller pack's own plan.  The call takes that pack's split-KV
    program where the workspace holds the plan's partials; where it cannot
    (a workspace one byte short, or no split-KV program of that pack) the call
    keeps its program, as pack 8 has no plan of its own."""

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    sm_count = _SM_COUNT[target]
    monkeypatch.setattr(cake_api, "_context_split_sm_count", lambda device: sm_count)
    # the split route's subject: no one-stage route takes the call
    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_ONE_STAGE_ROUTES", {})
    assert cake_api._context_fp8_split_plan(target, 1, 4, 1024, 32, 8, sm_count) is None
    pack, plan = cake_api._context_fp8_split_choice(target, 1, 4, 1024, 32, 8, sm_count)
    assert pack < 8
    need = cake_api._context_fp8_split_workspace_bytes(plan[0], plan[2])
    suffix = "_PAGE64" if page == 64 else ""
    meta = torch.device("meta")
    row = ("context_fp8", 1, 4, 1024, 32, 4, page, True, None)
    kwargs = dict(
        _context_kwargs(row, device="meta"),
        workspace_buffer=torch.empty(need, dtype=torch.uint8, device="meta"),
    )
    route = cake_api.select_cake_fmha_context_route(meta, **kwargs)
    assert route.program == f"RUNTIME_SK_PACK{pack}{suffix}", route
    assert (route.pack_g, route.num_m_blocks) == (pack, 1)
    if skipped == "short_workspace":
        kwargs["workspace_buffer"] = torch.empty(
            need - 1, dtype=torch.uint8, device="meta"
        )
    else:
        routes = cake_api._CONTEXT_FP8_SPLIT_ROUTES[target]
        monkeypatch.setitem(
            cake_api._CONTEXT_FP8_SPLIT_ROUTES,
            target,
            {key: name for key, name in routes.items() if key[0] != pack},
        )
    route = cake_api.select_cake_fmha_context_route(meta, **kwargs)
    assert route.program == (
        _FP8_EXACT[target] if page == 64 else f"RUNTIME_PACK8{_RUNTIME_SUFFIX[target]}"
    )
    assert (route.pack_g, route.num_m_blocks) == (8, 1)


def test_cake_fmha_context_fp8_one_stage_routes_mirror_the_runtime_routes() -> None:
    """Every runtime-size program has the one-stage program of its pack and
    page size on the same targets (the same schedule and P placement with one
    query stage per tile), and no other one-stage program exists."""

    runtime = cake_api._CONTEXT_FP8_RUNTIME_ROUTES
    assert set(cake_api._CONTEXT_FP8_ONE_STAGE_ROUTES) == set(runtime)
    for target, routes in runtime.items():
        assert cake_api._CONTEXT_FP8_ONE_STAGE_ROUTES[target] == {
            key: identifier + "_Q1" for key, identifier in routes.items()
        }
    assert frozenset(_ONE_STAGE_PACK) == cake_api._CONTEXT_FP8_ONE_STAGE_PROGRAMS
    assert not cake_api._CONTEXT_FP8_ONE_STAGE_PROGRAMS & set(_RUNTIME_PACK)


def test_cake_fmha_context_fp8_tile_blocks_of_one_stage_programs() -> None:
    """A one-stage program's m-block holds one query stage (128 / pack query
    tokens): its tiles' KV blocks follow the same tile order and causal
    bound."""

    # 2 requests of 128 query tokens over 1500 keys, 32 query heads in packs
    # of 1: one m-block, every tile all 12 KV blocks
    assert cake_api._context_fp8_tile_blocks(2, 128, 1500, 32, 1, 1) == [12] * 64
    assert cake_api._context_fp8_tile_blocks(2, 128, 1500, 32, 1) == [12] * 64
    # 1 request of 300 query tokens over 1000 keys, 2 query heads: 3 m-blocks
    # of 128 tokens (7, 8 and 8 KV blocks), each unit's m-blocks in turn
    assert cake_api._context_fp8_tile_blocks(1, 300, 1000, 2, 1, 1) == [8, 8, 7] * 2
    # pack 8: 16 tokens per m-block
    blocks = cake_api._context_fp8_tile_blocks(1, 40, 1000, 8, 8, 1)
    per_m = [-(-min(1000, (m + 1) * 16 + 960) // 128) for m in range(3)]
    assert blocks == [per_m[2 - m] for m in range(3)]


def _one_stage_time(target, batch, q_len, kv_len, num_q_heads, pack, sm_count):
    unit, block = cake_api._CONTEXT_FP8_ONE_STAGE_COSTS[target]
    fixed = cake_api._CONTEXT_FP8_SPLIT_COSTS[target][0]
    tiles = cake_api._context_fp8_tile_blocks(
        batch, q_len, kv_len, num_q_heads, pack, 1
    )
    return cake_api._context_fp8_split_makespan(
        tiles, sm_count, 0, 1, 1, (fixed, block, unit, 0.0, 0.0, 0.0, 0.0)
    )


def _launch_time(target, batch, q_len, kv_len, num_q_heads, current, sm_count):
    pack, plan = current
    costs = cake_api._context_fp8_split_costs(target, q_len, pack)
    tiles = cake_api._context_fp8_tile_blocks(batch, q_len, kv_len, num_q_heads, pack)
    if plan is None:
        costs = cake_api._context_fp8_whole_costs(target, q_len, kv_len, pack, costs)
    return cake_api._context_fp8_split_makespan(
        tiles, sm_count, *(plan or (0, 1, 1)), costs
    )


# (batch, max_q_len, max_kv_len, num_q_heads, runtime pack)
_ONE_STAGE_ROWS = [
    (2, 128, 1500, 32, 8),
    (1, 128, 1500, 32, 8),
    (2, 128, 6016, 32, 8),
    (4, 128, 1024, 30, 2),
    (16, 16, 1024, 32, 8),
    (128, 16, 1024, 32, 8),
    (2, 64, 4096, 64, 8),
    (8, 32, 2048, 32, 4),
    (1, 1, 384, 8, 8),
    (64, 96, 8192, 32, 8),
    (2, 200, 1500, 32, 8),
    (1, 300, 1500, 32, 8),
]


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_cake_fmha_context_fp8_one_stage_choice(target) -> None:
    """The one-stage choice of a call names a pack dividing its runtime pack
    whose tiles hold every query token in one stage, in at most
    ``_CONTEXT_FP8_SPLIT_MAX_WAVES`` waves (one wave for a smaller pack), the
    fastest under the one-stage costs and faster than the call's launch (its
    split choice, else whole tiles) by the split margins; else ``None``.  2
    requests of 128 query tokens over 1500 keys, heads 32/4, take pack 1's
    one-stage program on both targets; calls of more than 128 query tokens
    keep their launch."""

    sm_count = _SM_COUNT[target]
    choose = cake_api._context_fp8_one_stage_choice
    for batch, q_len, kv_len, hq, pack in _ONE_STAGE_ROWS:
        split = cake_api._context_fp8_split_choice(
            target, batch, q_len, kv_len, hq, pack, sm_count
        )
        current = split if split is not None else (pack, None)
        chosen = choose(target, batch, q_len, kv_len, hq, pack, sm_count, current)
        row = (batch, q_len, kv_len, hq, pack, current, chosen)
        if q_len > 128:
            assert chosen is None, row
        time = _launch_time(target, batch, q_len, kv_len, hq, current, sm_count)
        best = None
        for candidate in (8, 4, 2, 1):
            if candidate > pack or pack % candidate or q_len > 128 // candidate:
                continue
            tiles = batch * (hq // candidate)
            waves = cake_api._CONTEXT_FP8_SPLIT_MAX_WAVES if candidate == pack else 1
            if tiles > waves * sm_count:
                continue
            one = _one_stage_time(target, batch, q_len, kv_len, hq, candidate, sm_count)
            if best is None or one < best[1]:
                best = (candidate, one)
        gain = best is not None and cake_api._context_fp8_split_gain(time, best[1])
        assert chosen == (best[0] if gain else None), row
    current = cake_api._context_fp8_split_choice(target, 2, 128, 1500, 32, 8, sm_count)
    current = current if current is not None else (8, None)
    assert choose(target, 2, 128, 1500, 32, 8, sm_count, current) == 1
    # without a cost model the call keeps its launch
    assert choose("sm90a", 2, 128, 1500, 32, 8, sm_count, (8, None)) is None


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize("page", (16, 64))
def test_cake_fmha_context_fp8_one_stage_route(monkeypatch, target, page) -> None:
    """2 requests of 128 query tokens over 1500 keys, heads 32/4 (no exact
    route), take pack 1's one-stage program of the page size on a CUDA device
    (one m-block of 128 tokens), for host and device scales; off a CUDA
    device the call keeps its runtime-size program.  A call of the exact
    class keeps its program; an sm103a call of at most 128 keys takes the
    one-stage program with P in its own S columns of the smallest pack whose
    tiles number at most ``_CONTEXT_FP8_SHORT_KV_ONE_STAGE_TILES`` where the
    page size has one (every page size), else keeps its short-KV program."""

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    meta = torch.device("meta")
    suffix = "_PAGE64" if page == 64 else ""
    row = ("context_fp8", 2, 128, 1500, 32, 4, page, True, None)
    route = cake_api.select_cake_fmha_context_route(
        meta, **_context_kwargs(row, device="meta")
    )
    assert route.program == f"RUNTIME_PACK8{suffix}{_RUNTIME_SUFFIX[target]}"
    sm_count = _SM_COUNT[target]
    monkeypatch.setattr(cake_api, "_context_split_sm_count", lambda device: sm_count)
    for device_scales in (False, True):
        route = cake_api.select_cake_fmha_context_route(
            meta, **_context_kwargs(row, device="meta", device_scales=device_scales)
        )
        expected = f"RUNTIME_PACK1{suffix}{_RUNTIME_SUFFIX[target]}_Q1"
        assert route.program == expected, route
        assert (route.pack_g, route.num_m_blocks, route.l2_swizzle) == (1, 1, 8)
        assert route.device_scales is device_scales
    exact = ("context_fp8", 1, 16, 1024, 32, 4, 64, True, None)
    route = cake_api.select_cake_fmha_context_route(
        meta, **_context_kwargs(exact, device="meta")
    )
    assert route.program not in cake_api._CONTEXT_FP8_ONE_STAGE_PROGRAMS, route
    if target == "sm103a":
        short = ("context_fp8", 2, 64, 128, 32, 4, page, True, None)
        route = cake_api.select_cake_fmha_context_route(
            meta, **_context_kwargs(short, device="meta")
        )
        if page == 16:
            assert route.program == "RUNTIME_PACK1_Q1", route
            assert (route.pack_g, route.num_m_blocks) == (1, 1)
        else:
            assert route.program == f"RUNTIME_PACK{route.pack_g}{suffix}", route


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_cake_fmha_context_fp8_short_kv_one_stage_pack(monkeypatch, target) -> None:
    """An sm103a call of at most 128 keys at 16-token pages takes the
    one-stage program of the smallest pack dividing its runtime pack whose
    tiles hold every query token in one stage and number at most
    ``_CONTEXT_FP8_SHORT_KV_ONE_STAGE_TILES`` (64), and the one-stage choice
    where no pack has so few tiles; an sm100a call keeps its route (no
    short-KV programs)."""

    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    sm_count = _SM_COUNT[target]
    monkeypatch.setattr(cake_api, "_context_split_sm_count", lambda device: sm_count)
    meta = torch.device("meta")
    if target == "sm103a":
        assert cake_api._CONTEXT_FP8_SHORT_KV_ONE_STAGE_TILES == {"sm103a": 64}
    else:
        assert target not in cake_api._CONTEXT_FP8_SHORT_KV_ONE_STAGE_TILES
    # (batch, query tokens, keys, query heads, KV heads) -> sm103a pack
    cases = {
        (1, 64, 128, 32, 4): 1,
        (2, 64, 128, 32, 4): 1,
        (4, 64, 128, 32, 4): 2,
        (1, 32, 128, 32, 4): 1,
        (2, 32, 128, 64, 8): 2,
        (4, 16, 100, 32, 4): 2,
        (8, 16, 128, 32, 4): 4,
        (16, 16, 128, 32, 4): 8,
        (1, 128, 128, 32, 4): 1,
        (1, 64, 128, 32, 8): 1,
    }
    for (batch, q_len, kv_len, hq, hkv), pack in cases.items():
        row = ("context_fp8", batch, q_len, kv_len, hq, hkv, 16, True, None)
        route = cake_api.select_cake_fmha_context_route(
            meta, **_context_kwargs(row, device="meta")
        )
        if target == "sm103a":
            assert route.program == f"RUNTIME_PACK{pack}_Q1", (row, route)
            assert (route.pack_g, route.num_m_blocks) == (pack, 1), (row, route)
        else:
            with monkeypatch.context() as patch:
                patch.setattr(cake_api, "_CONTEXT_FP8_SHORT_KV_ONE_STAGE_TILES", {})
                kept = cake_api.select_cake_fmha_context_route(
                    meta, **_context_kwargs(row, device="meta")
                )
            assert route == kept, (row, route, kept)
    # 8 requests of 64 tokens: pack 1 and 2 tiles number 256 and 128, so the
    # one-stage choice names the pack
    row = ("context_fp8", 8, 64, 128, 32, 4, 16, True, None)
    route = cake_api.select_cake_fmha_context_route(
        meta, **_context_kwargs(row, device="meta")
    )
    if target == "sm103a":
        pack = cake_api._context_fp8_one_stage_choice(
            target, 8, 64, 128, 32, 8, sm_count, (8, None)
        )
        assert pack is not None
        assert route.program == f"RUNTIME_PACK{pack}_Q1", route


# --- The page tables a launch passes (host-only) -----------------------------


def _stand_in_module(
    component="context_bf16", identifier="B128", *, device_scales=False
):
    """The program module of a stand-in record (``B128`` unless ``component``
    / ``identifier`` name another) for 32 query heads over 4 KV heads and
    16-token pages, whose entry keeps every call's arguments
    (``module.calls``)."""

    name, num_m_blocks, _ = cake_api._CONTEXT_PROGRAMS[(component, identifier)]
    args = (
        *cake_api._context_program_params(component, num_m_blocks, identifier),
        "grid_x",
        "grid_y",
        "grid_z",
    )
    calls = []
    record = SimpleNamespace(
        name=name, args=args, entry=lambda target: lambda *values: calls.append(values)
    )
    route = cake_api.CakeFmhaContextRoute(
        target="sm100a",
        component=component,
        num_m_blocks=2 if num_m_blocks is None else num_m_blocks,
        num_q_heads=32,
        num_kv_heads=4,
        pack_g={**_RUNTIME_PACK, **_SPLIT_PACK, **_ONE_STAGE_PACK}.get(identifier, 8),
        page_size=16,
        l2_swizzle=1,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
        device_scales=device_scales,
        program=identifier,
    )
    module = cake_api.CakeFmhaContextProgramModule(route, record, "sm_100a")
    module.calls = calls
    return module


def _launch_values(
    monkeypatch,
    block_tables,
    *,
    shared,
    batch=3,
    component="context_bf16",
    identifier="B128",
    device_scales=False,
    device=None,
    module=None,
    **overrides,
):
    """The argument values a stand-in program record (``B128`` unless
    ``component`` / ``identifier`` name another, or the record of ``module``)
    receives from one ``cake_paged_attention_context`` on CPU tensors, by name;
    ``overrides`` replace launch keywords (e.g. device-scale tensors)."""

    if module is None:
        module = _stand_in_module(component, identifier, device_scales=device_scales)
    component = module.route.component
    monkeypatch.setattr(
        cake_api, "_torch_stream_context", lambda device: contextlib.nullcontext()
    )
    dtype = torch.bfloat16 if component == "context_bf16" else torch.float8_e4m3fn
    query = torch.zeros((batch * 64, 32, 128), dtype=dtype, device=device)
    key = torch.zeros((8, 4, 16, 128), dtype=dtype, device=device)
    keywords = dict(
        out=torch.empty_like(query),
        out_scale_factor=None,
        query=query,
        key_cache=key,
        value_cache=torch.empty_like(key),
        workspace_buffer=torch.zeros(64, dtype=torch.uint8, device=device),
        multi_ctas_kv_counter_buffer=torch.zeros(16, dtype=torch.uint8, device=device),
        block_tables=block_tables,
        seq_lens=torch.full((batch,), 64, dtype=torch.int32, device=device),
        max_q_len=64,
        max_kv_len=64,
        bmm1_scale=0.125,
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        o_sf_vec_size=0,
        o_sf_start_index=0,
        batch_size=batch,
        window_left=-1,
        cum_seq_lens_q=torch.arange(
            0, 64 * (batch + 1), 64, dtype=torch.int32, device=device
        ),
        cum_seq_lens_kv=torch.arange(
            0, 64 * (batch + 1), 64, dtype=torch.int32, device=device
        ),
        sm_count=148,
        enable_pdl=None,
        workspace_size=64,
        attention_sinks=None,
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        uses_shared_paged_kv_idx=shared,
        use_fp16_softmax=False,
        uses_spcompress=False,
        is_causal=True,
        lse=None,
        lse_stride_tokens=0,
        lse_stride_heads=0,
    )
    keywords.update(overrides)
    calls = len(module.calls)
    module.cake_paged_attention_context(**keywords)
    assert len(module.calls) == calls + 1
    # the unread pointers arrive as FFI tensors: read them back as torch views
    return {
        name: torch.from_dlpack(value) if isinstance(value, tvm_ffi.Tensor) else value
        for name, value in zip(module.program.args, module.calls[-1], strict=True)
    }


@pytest.mark.parametrize("shared", (True, False))
def test_cake_context_program_page_tables_view_the_callers_storage(
    monkeypatch, shared
) -> None:
    """A column-sliced (non-contiguous) table reaches the kernel as views of
    the caller's own storage with its true row stride: no copy."""

    batch, pages = 3, 5
    wide = torch.arange(batch * 2 * 9, dtype=torch.int32).view(batch, 2, 9)
    tables = wide[:, 0, 2 : 2 + pages] if shared else wide[:, :, 2 : 2 + pages]
    assert not tables.is_contiguous()
    values = _launch_values(monkeypatch, tables, shared=shared, batch=batch)
    table_k, table_v = values["page_table_k"], values["page_table_v"]
    row_stride = values["page_row_stride"]
    assert row_stride == tables.stride(0) == 18
    assert table_k.is_contiguous() and table_v.is_contiguous()
    assert table_k.untyped_storage().data_ptr() == wide.untyped_storage().data_ptr()
    assert table_k.data_ptr() == tables.data_ptr()
    view_k = tables if shared else tables[:, 0]
    view_v = tables if shared else tables[:, 1]
    assert table_v.data_ptr() == view_v.data_ptr()
    for b in range(batch):
        for p in range(pages):
            assert table_k[b * row_stride + p] == view_k[b, p]
            assert table_v[b * row_stride + p] == view_v[b, p]


def test_cake_context_program_arguments_follow_the_binding_order(monkeypatch) -> None:
    """Each binding argument receives its value: the lengths and offsets the
    caller passed, the work units, the grid, the ticket as uint32 and the
    unread pointers on one small buffer."""

    batch = 3
    tables = torch.arange(batch * 4, dtype=torch.int32).view(batch, 4)
    values = _launch_values(monkeypatch, tables, shared=True, batch=batch)
    _, num_m_blocks, one_tile = cake_api._CONTEXT_PROGRAMS[("context_bf16", "B128")]
    units = 32 // 8
    tiles = num_m_blocks * batch * units
    assert values["total_bh"] == batch * units
    assert values["num_ctas"] == values["grid_x"]
    assert values["grid_x"] == (tiles if one_tile else min(148, tiles))
    assert (values["grid_y"], values["grid_z"]) == (1, 1)
    assert torch.equal(
        values["seq_lens_kv"], torch.full((batch,), 64, dtype=torch.int32)
    )
    assert torch.equal(
        values["cu_seq_lens_q"],
        torch.arange(0, 64 * (batch + 1), 64, dtype=torch.int32),
    )
    assert values["dynamic_counter"].dtype == torch.uint32
    assert values["softmax_scale_log2"] == pytest.approx(0.125 * cake_api.log2e)
    assert values["LSE_ptr"].dtype == values["sinks"].dtype == torch.float32
    assert values["seq_lens_q"].dtype == torch.int32
    assert values["LSE_ptr"].data_ptr() == values["seq_lens_q"].data_ptr()
    assert (
        values["Q"].shape == (batch * 64, 32, 128)
        and values["O_ptr"].dtype == torch.bfloat16
    )


def _placeholder(device):
    """The data pointer of the per-device placeholder buffer."""

    return torch.from_dlpack(cake_api._unread_pointers(device)[0]).data_ptr()


def test_cake_context_program_unread_pointers_are_per_device_placeholders(
    monkeypatch,
) -> None:
    """The arguments a program never reads are FFI tensors on one small
    buffer per device, converted once (no DLPack conversion per launch): two
    devices in a row get two buffers, each the same objects on every call.
    Without a second GPU, two distinct CPU device keys stand in."""

    monkeypatch.setattr(cake_api, "_UNREAD_POINTERS", {})
    if torch.cuda.device_count() >= 2:
        devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
    else:
        devices = [torch.device("cpu"), torch.device("cpu", 0)]
    seen = []
    for device in devices + devices:
        f32, i32 = cake_api._unread_pointers(device)
        assert isinstance(f32, tvm_ffi.Tensor) and isinstance(i32, tvm_ffi.Tensor)
        as_f32, as_i32 = torch.from_dlpack(f32), torch.from_dlpack(i32)
        assert (as_f32.dtype, as_i32.dtype) == (torch.float32, torch.int32)
        assert as_f32.data_ptr() == as_i32.data_ptr()
        assert as_f32.device.type == device.type
        seen.append((f32, i32, as_f32.data_ptr()))
    assert seen[2][0] is seen[0][0] and seen[2][1] is seen[0][1]
    assert seen[3][0] is seen[1][0] and seen[3][1] is seen[1][1]
    assert seen[0][2] != seen[1][2]
    assert len(cake_api._UNREAD_POINTERS) == 2


def test_cake_context_program_device_scales_reach_the_binding_as_given(
    monkeypatch,
) -> None:
    """The placeholder stands only for arguments the program never reads.
    Launches in a row -- FP8 host scales, FP8 device scales, BF16 -- pass the
    caller's own tensors: the device-scale program reads the caller's bmm1
    and bmm2 tensors (a 0-d scale as a 1-element view of its storage), and
    Q / K / V / O are byte views of the caller's storage."""

    batch = 3
    tables = torch.arange(batch * 4, dtype=torch.int32).view(batch, 4)
    unread = _placeholder(torch.device("cpu"))

    host = _launch_values(
        monkeypatch,
        tables,
        shared=True,
        batch=batch,
        component="context_fp8",
        identifier="B256",
    )
    assert host["bmm1_scale_ptr"].data_ptr() == unread
    assert host["bmm2_scale_ptr"].data_ptr() == unread
    assert host["bmm1_is_log2"] == 0
    assert host["softmax_scale_log2"] == pytest.approx(0.125 * cake_api.log2e)
    assert host["output_scale"] == pytest.approx(1.0)

    bmm1 = torch.tensor([0.25], dtype=torch.float32)
    bmm2 = torch.tensor(0.5, dtype=torch.float32)
    query = torch.zeros((batch * 64, 32, 128), dtype=torch.float8_e4m3fn)
    out = torch.empty_like(query)
    scaled = _launch_values(
        monkeypatch,
        tables,
        shared=True,
        batch=batch,
        component="context_fp8",
        identifier="B256_SCALE_PTR",
        device_scales=True,
        bmm1_scale=bmm1,
        bmm2_scale=bmm2,
        query=query,
        out=out,
    )
    assert scaled["bmm1_scale_ptr"].data_ptr() == bmm1.data_ptr() != unread
    assert scaled["bmm2_scale_ptr"].data_ptr() == bmm2.data_ptr() != unread
    assert scaled["bmm1_scale_ptr"].shape == scaled["bmm2_scale_ptr"].shape == (1,)
    assert scaled["bmm1_is_log2"] == 0
    assert scaled["Q"].dtype == scaled["O_ptr"].dtype == torch.uint8
    assert scaled["Q"].data_ptr() == query.data_ptr()
    assert scaled["O_ptr"].data_ptr() == out.data_ptr()
    for name in ("LSE_ptr", "sinks", "seq_lens_q"):
        assert scaled[name].data_ptr() == unread

    plain = _launch_values(monkeypatch, tables, shared=True, batch=batch)
    assert plain["LSE_ptr"].data_ptr() == plain["seq_lens_q"].data_ptr() == unread


def test_cake_context_runtime_program_takes_the_sizes_of_each_launch(
    monkeypatch,
) -> None:
    """One runtime-size program serves launches of other head counts, page
    sizes, query lengths, batches and scale forms in a row: each passes its
    own sizes (the m-blocks of ``max_q_len``, the whole 8-unit sections, the
    units per request, the GQA group, the log2 page size) and scale source,
    and the device scales are the caller's tensors."""

    module = _stand_in_module("context_fp8", "RUNTIME_PACK8")
    unread = _placeholder(torch.device("cpu"))

    # 3 requests of 64 query tokens, 32 query heads over 4 KV heads, 16-token
    # pages, host scales
    batch = 3
    tables = torch.arange(batch * 4, dtype=torch.int32).view(batch, 4)
    first = _launch_values(monkeypatch, tables, shared=True, batch=batch, module=module)
    assert (first["num_q_heads"], first["heads_per_group"]) == (32, 8)
    assert (first["num_bh_units"], first["total_bh"], first["full_sections"]) == (
        4,
        12,
        1,
    )
    assert first["num_m_blocks"] == 2 and first["page_shift"] == 4
    assert (first["scale_source"], first["bmm1_is_log2"]) == (0, 0)
    assert first["num_ctas"] == first["grid_x"] == 24
    assert first["softmax_scale_log2"] == pytest.approx(0.125 * cake_api.log2e)
    assert first["bmm1_scale_ptr"].data_ptr() == unread
    assert first["bmm2_scale_ptr"].data_ptr() == unread

    # 5 requests of 100 query tokens, 16 query heads over 2 KV heads, 64-token
    # pages, device scales
    batch, q_len = 5, 100
    query = torch.zeros((batch * q_len, 16, 128), dtype=torch.float8_e4m3fn)
    key = torch.zeros((8, 2, 64, 128), dtype=torch.float8_e4m3fn)
    bmm1 = torch.tensor([0.25], dtype=torch.float32)
    bmm2 = torch.tensor(0.5, dtype=torch.float32)
    offsets = torch.arange(0, q_len * (batch + 1), q_len, dtype=torch.int32)
    second = _launch_values(
        monkeypatch,
        torch.arange(batch * 2, dtype=torch.int32).view(batch, 2),
        shared=True,
        batch=batch,
        module=module,
        query=query,
        out=torch.empty_like(query),
        key_cache=key,
        value_cache=torch.empty_like(key),
        seq_lens=torch.full((batch,), q_len, dtype=torch.int32),
        max_q_len=q_len,
        max_kv_len=q_len,
        cum_seq_lens_q=offsets,
        cum_seq_lens_kv=offsets,
        bmm1_scale=bmm1,
        bmm2_scale=bmm2,
    )
    assert (second["num_q_heads"], second["heads_per_group"]) == (16, 8)
    assert (second["num_bh_units"], second["total_bh"], second["full_sections"]) == (
        2,
        10,
        1,
    )
    assert second["num_m_blocks"] == 4 and second["page_shift"] == 6
    assert (second["scale_source"], second["bmm1_is_log2"]) == (1, 0)
    assert second["num_ctas"] == second["grid_x"] == 40
    assert second["bmm1_scale_ptr"].data_ptr() == bmm1.data_ptr()
    assert second["bmm2_scale_ptr"].data_ptr() == bmm2.data_ptr()
    assert second["Q"].data_ptr() == query.data_ptr()
    for name in ("LSE_ptr", "sinks", "seq_lens_q"):
        assert second[name].data_ptr() == unread
    assert len(module.calls) == 2


def test_cake_context_one_stage_program_takes_one_stage_m_blocks(
    monkeypatch,
) -> None:
    """A one-stage program's m-block holds one query stage: the launch passes
    the m-blocks of ``max_q_len`` at ``128 / pack_g`` tokens each, and its
    grid covers those tiles."""

    module = _stand_in_module("context_fp8", "RUNTIME_PACK1_Q1")
    assert (module.pack_g, module.m_block_tokens) == (1, 128)
    batch = 3
    tables = torch.arange(batch * 4, dtype=torch.int32).view(batch, 4)
    vals = _launch_values(monkeypatch, tables, shared=True, batch=batch, module=module)
    assert vals["num_m_blocks"] == 1
    assert (vals["num_bh_units"], vals["total_bh"]) == (32, 96)
    assert vals["num_ctas"] == vals["grid_x"] == 96
    module = _stand_in_module("context_fp8", "RUNTIME_PACK8_PX_Q1")
    assert (module.pack_g, module.m_block_tokens) == (8, 16)
    vals = _launch_values(monkeypatch, tables, shared=True, batch=batch, module=module)
    assert vals["num_m_blocks"] == 4
    assert vals["num_ctas"] == vals["grid_x"] == 48


def test_cake_context_split_program_takes_the_plan_of_each_launch(
    monkeypatch,
) -> None:
    """A split-KV program passes the runtime-size sizes, then its launch's plan
    (``_context_fp8_split_plan`` of the launch's SM count), the partials as
    views of the caller's workspace and the tile counters after the counter
    buffer's first word, at a grid of every work unit up to the SM count; a
    launch whose workspace cannot hold the plan runs one unit per tile."""

    module = _stand_in_module("context_fp8", "RUNTIME_SK_PACK8")
    batch, q_len, kv_len = 1, 32, 16384
    query = torch.zeros((q_len, 32, 128), dtype=torch.float8_e4m3fn)
    key = torch.zeros((kv_len // 64, 4, 64, 128), dtype=torch.float8_e4m3fn)
    workspace = torch.zeros(64 << 20, dtype=torch.uint8)
    counter = torch.zeros(4 * 152, dtype=torch.uint8)
    launch = dict(
        shared=True,
        batch=batch,
        module=module,
        query=query,
        out=torch.empty_like(query),
        key_cache=key,
        value_cache=torch.empty_like(key),
        seq_lens=torch.full((batch,), kv_len, dtype=torch.int32),
        max_q_len=q_len,
        max_kv_len=kv_len,
        cum_seq_lens_q=torch.tensor([0, q_len], dtype=torch.int32),
        cum_seq_lens_kv=torch.tensor([0, kv_len], dtype=torch.int32),
        multi_ctas_kv_counter_buffer=counter,
    )
    tables = torch.arange(kv_len // 64, dtype=torch.int32).view(1, -1)
    for sm_count in (148, 152):
        plan = cake_api._context_fp8_split_plan(
            "sm100a", batch, q_len, kv_len, 32, 8, sm_count
        )
        assert plan is not None
        split_tiles, split_blocks, num_splits = plan
        values = _launch_values(
            monkeypatch,
            tables,
            workspace_buffer=workspace,
            sm_count=sm_count,
            **launch,
        )
        assert (
            values["split_tiles"],
            values["split_blocks"],
            values["num_splits"],
        ) == plan
        units = 4 + (num_splits - 1) * split_tiles
        assert values["num_ctas"] == values["grid_x"] == min(sm_count, units)
        slots = split_tiles * num_splits
        assert values["partial_o"].dtype == values["partial_ml"].dtype == torch.float32
        assert values["partial_o"].data_ptr() == workspace.data_ptr()
        assert values["partial_o"].numel() == slots * 2 * 128 * 128
        assert values["partial_ml"].data_ptr() == (
            workspace.data_ptr() + slots * cake_api._CONTEXT_FP8_SPLIT_O_BYTES
        )
        assert values["partial_ml"].numel() == slots * 2 * 128 * 4
        assert values["dynamic_counter"].data_ptr() == counter.data_ptr()
        assert values["tile_counters"].dtype == torch.uint32
        assert values["tile_counters"].data_ptr() == counter.data_ptr() + 4
        assert values["tile_counters"].numel() == split_tiles
        assert (values["num_m_blocks"], values["total_bh"]) == (1, 4)

    short = _launch_values(
        monkeypatch,
        tables,
        workspace_buffer=torch.zeros(64, dtype=torch.uint8),
        sm_count=148,
        **launch,
    )
    assert (short["split_tiles"], short["split_blocks"], short["num_splits"]) == (
        1,
        1,
        1,
    )
    assert short["num_ctas"] == short["grid_x"] == 4


def test_cake_context_split_program_takes_the_chosen_plan_of_a_smaller_pack(
    monkeypatch,
) -> None:
    """The split-KV program of a pack smaller than the call's runtime pack
    takes the split choice's plan where the choice names its pack (pack 8's
    split over pack 4's tiles, with the shipped costs), and its own pack's
    plan otherwise (no pack change where the tile rows do not scale the
    costs)."""

    module = _stand_in_module("context_fp8", "RUNTIME_SK_PACK4")
    batch, q_len, kv_len, sm_count = 1, 32, 16384, 148
    query = torch.zeros((q_len, 32, 128), dtype=torch.float8_e4m3fn)
    key = torch.zeros((kv_len // 64, 4, 64, 128), dtype=torch.float8_e4m3fn)
    launch = dict(
        shared=True,
        batch=batch,
        module=module,
        query=query,
        out=torch.empty_like(query),
        key_cache=key,
        value_cache=torch.empty_like(key),
        seq_lens=torch.full((batch,), kv_len, dtype=torch.int32),
        max_q_len=q_len,
        max_kv_len=kv_len,
        cum_seq_lens_q=torch.tensor([0, q_len], dtype=torch.int32),
        cum_seq_lens_kv=torch.tensor([0, kv_len], dtype=torch.int32),
        multi_ctas_kv_counter_buffer=torch.zeros(4 * sm_count, dtype=torch.uint8),
        workspace_buffer=torch.zeros(64 << 20, dtype=torch.uint8),
        sm_count=sm_count,
    )
    tables = torch.arange(kv_len // 64, dtype=torch.int32).view(1, -1)
    own8 = cake_api._context_fp8_split_plan("sm100a", 1, 32, 16384, 32, 8, sm_count)
    own4 = cake_api._context_fp8_split_plan("sm100a", 1, 32, 16384, 32, 4, sm_count)
    shipped = cake_api._CONTEXT_FP8_SPLIT_ROWS["sm100a"]
    try:
        for rows, plan in ((shipped, (8, own8[1], own8[2])), ((1.0, 1.0), own4)):
            monkeypatch.setitem(cake_api._CONTEXT_FP8_SPLIT_ROWS, "sm100a", rows)
            _clear_split_caches()
            values = _launch_values(monkeypatch, tables, **launch)
            assert (
                values["split_tiles"],
                values["split_blocks"],
                values["num_splits"],
            ) == plan
            assert (values["num_m_blocks"], values["total_bh"]) == (1, 8)
            units = 8 + (plan[2] - 1) * plan[0]
            assert values["num_ctas"] == values["grid_x"] == min(sm_count, units)
    finally:
        _clear_split_caches()


@pytest.mark.parametrize("given", ("lse", "attention_sinks"))
def test_cake_context_program_refuses_an_lse_or_sinks(monkeypatch, given) -> None:
    """A program writes no LSE and reads no sinks: a launch that carries
    either is refused rather than served with the unread placeholder (route
    selection keeps such calls on the JIT module)."""

    tables = torch.arange(12, dtype=torch.int32).view(3, 4)
    tensor = torch.zeros((3 * 64, 32) if given == "lse" else (32,))
    with pytest.raises(RuntimeError, match="writes no LSE and takes no sinks"):
        _launch_values(monkeypatch, tables, shared=True, **{given: tensor})


def test_cake_context_program_passes_a_contiguous_shared_table_itself(
    monkeypatch,
) -> None:
    """A contiguous shared table reaches the binding as the caller's tensor
    (the binding reads its address): no view is built per launch."""

    batch = 3
    tables = torch.arange(batch * 4, dtype=torch.int32).view(batch, 4)
    values = _launch_values(monkeypatch, tables, shared=True, batch=batch)
    assert values["page_table_k"] is tables and values["page_table_v"] is tables
    assert values["page_row_stride"] == 4


def test_cake_context_program_refuses_a_strided_page_column(monkeypatch) -> None:
    tables = torch.zeros((3, 10), dtype=torch.int32)[:, ::2]
    with pytest.raises(RuntimeError, match="unit stride along the pages"):
        _launch_values(monkeypatch, tables, shared=True)


# --- GPU ---------------------------------------------------------------------

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)


def _skip_without_nvfp4_programs(component: str) -> None:
    """Skip a test of an NVFP4 program where the CUDA toolkit cannot build it
    (their routes fail closed to compat_v1 there, see ``cake_fmha``)."""
    if component != "context_nvfp4":
        return
    unavailable = cake_api.context_nvfp4_programs_unavailable()
    if unavailable is not None:
        pytest.skip(f"the NVFP4 context programs are not built here: {unavailable}")


@_GPU
def test_cake_context_nvfp4_programs_fail_closed_without_their_toolkit(
    monkeypatch,
) -> None:
    """On a CUDA toolkit that cannot assemble the NVFP4 programs (their PTX ISA
    9.4 packed multiply needs CUDA 13.4), a route to one of them loads
    compat_v1 with a warning instead of a build ptxas rejects, and the program
    loader itself refuses the build."""

    target = cake_api._cake_fmha_target(torch.device("cuda"))
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    kwargs = _context_kwargs("nvfp4_b4_q511_k2047", device="meta")
    route = cake_api.select_cake_fmha_context_route(torch.device("meta"), **kwargs)
    assert route is not None and route.component == "context_nvfp4"
    assert route.program is not None
    reason = "a toolkit that cannot assemble the programs"
    monkeypatch.setattr(cake_api, "context_nvfp4_programs_unavailable", lambda: reason)
    monkeypatch.setattr(cake_api, "_CONTEXT_PROGRAM_MODULES", {})
    with pytest.raises(RuntimeError, match=reason):
        cake_api._load_cake_fmha_context_program(route)
    with pytest.warns(RuntimeWarning, match="failed closed to compat_v1"):
        module = cake_api.get_cake_fmha_context_module(torch.device("cuda"), route)
    assert not isinstance(module, cake_api.CakeFmhaContextProgramModule)
    assert module is cake_api.load_cake_fmha_compat_module(target)


@pytest.fixture
def served_programs(monkeypatch):
    """The generated programs the context dispatch loads, in call order."""

    served = []
    load = cake_api._load_cake_fmha_context_program

    def spy(route):
        served.append(route.program)
        return load(route)

    monkeypatch.setattr(cake_api, "_load_cake_fmha_context_program", spy)
    return served


def _whole_tile_routes(monkeypatch):
    """No split-KV or one-stage route: the test pins the whole-tile programs
    of two query stages (a call the split plan would split, or the one-stage
    choice would take, keeps its exact or runtime-size program)."""

    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_SPLIT_ROUTES", {})
    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_ONE_STAGE_ROUTES", {})


def _reference_context(
    query, key, value, tables_k, tables_v, q_lens, kv_lens, *, scale
):
    """FP32 bottom-right causal GQA attention over the paged HND pools: request
    ``b`` has ``q_lens[b]`` new tokens and attends its first ``kv_lens[b]``
    cached tokens; query ``i`` sees keys ``j <= i + kv_len - q_len``.  An
    empty request (no query token, no key) has no rows."""

    num_q_heads, num_kv_heads, page = query.shape[1], key.shape[1], key.shape[2]
    group = num_q_heads // num_kv_heads
    out = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    start = 0
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
        if q_len == 0:
            continue
        count = (kv_len + page - 1) // page
        k = (
            key[tables_k[b, :count].long()]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, 128)[:, :kv_len]
        )
        v = (
            value[tables_v[b, :count].long()]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, 128)[:, :kv_len]
        )
        k = k.float().repeat_interleave(group, dim=0)
        v = v.float().repeat_interleave(group, dim=0)
        q = query[start : start + q_len].float().transpose(0, 1)
        logits = torch.einsum("hqd,hkd->hqk", q, k) * scale
        right = torch.arange(q_len, device=query.device)[:, None] + kv_len - q_len
        logits.masked_fill_(
            torch.arange(kv_len, device=query.device) > right, -math.inf
        )
        out[start : start + q_len] = torch.einsum(
            "hqk,hkd->hqd", torch.softmax(logits, dim=-1), v
        ).transpose(0, 1)
        start += q_len
    return out


@_GPU
@pytest.mark.parametrize(
    ("program", "heads", "q_lens", "kv_lens", "page_size", "shared"),
    [
        ("Q511", (10, 2), (511,) * 4, (2047, 1777, 1024, 600), 32, True),
        ("Q257", (10, 2), (257,) * 4, (1024, 900, 700, 300), 1024, False),
        ("B128", (32, 4), (64, 40, 17, 64), (192, 100, 64, 130), 16, True),
        # An empty request (no query token, no key) between others: the launch
        # completes and the other requests match the reference.
        ("B128", (32, 4), (64, 0, 17, 64), (192, 0, 64, 130), 16, True),
    ],
)
def test_cake_context_bf16_programs_match_reference_and_replay(
    served_programs, program, heads, q_lens, kv_lens, page_size, shared
) -> None:
    device = torch.device("cuda")
    target = cake_api._cake_fmha_target(device)
    if program in ("Q511", "Q257"):
        program = f"{program}_SM{_TARGET_ARCH[target][3:].upper()}"
    batch = len(q_lens)
    hq, hkv = heads
    max_q, max_kv = max(q_lens), max(kv_lens)
    generator = torch.Generator(device="cpu").manual_seed(20261026 + max_q)
    pages = (max_kv + page_size - 1) // page_size
    total_pages = 2 * batch * pages
    query = (torch.randn((sum(q_lens), hq, 128), generator=generator) * 2).to(
        device=device, dtype=torch.bfloat16
    )
    key = torch.randn((total_pages, hkv, page_size, 128), generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    value = torch.randn((total_pages, hkv, page_size, 128), generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    order = torch.randperm(total_pages, generator=generator).to(torch.int32)
    tables_k = order[: batch * pages].view(batch, pages)
    tables_v = tables_k if shared else order[batch * pages :].view(batch, pages)
    block_tables = (
        tables_k if shared else torch.stack((tables_k, tables_v), dim=1)
    ).to(device)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    cum_q = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.tensor(q_lens).cumsum(0).int())
    ).to(device)
    cum_kv = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.tensor(kv_lens).cumsum(0).int())
    ).to(device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    counter = torch.zeros(
        get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch, hq, get_device_sm_count(device)
        ),
        dtype=torch.uint8,
        device=device,
    )
    scale = 128**-0.5
    out = torch.empty_like(query)

    def run():
        return prefill.trtllm_batch_context_with_kv_cache(
            query,
            (key, value),
            workspace,
            block_tables,
            seq_lens,
            max_q,
            max_kv,
            scale,
            1.0,
            batch,
            cum_q,
            cum_kv,
            out=out,
            kv_layout="HND",
            uses_shared_paged_kv_idx=shared,
            multi_ctas_kv_counter_buffer=counter,
            backend="cake",
        )

    run()
    torch.cuda.synchronize()
    assert served_programs[-1] == program
    tables_k_d, tables_v_d = tables_k.to(device), tables_v.to(device)
    expected = _reference_context(
        query, key, value, tables_k_d, tables_v_d, q_lens, kv_lens, scale=scale
    )
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)
    # The persistent programs' ticket resets itself at the end of every launch.
    assert not counter.any()

    # Graph replay: route selection reads no tensor value, so the captured
    # launch serves new KV lengths under the captured maximum.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
    torch.cuda.synchronize()
    shorter = tuple(
        max(q_len, length - length // 4)
        for q_len, length in zip(q_lens, kv_lens, strict=True)
    )
    seq_lens.copy_(torch.tensor(shorter, dtype=torch.int32, device=device))
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    expected_short = _reference_context(
        query, key, value, tables_k_d, tables_v_d, q_lens, shorter, scale=scale
    )
    torch.testing.assert_close(out.float(), expected_short, atol=1e-2, rtol=1e-2)
    assert not counter.any()


def _bf16_batch(heads, q_lens, kv_lens, page_size, *, shared=True, seed=20261028):
    """Random BF16 inputs on the GPU: an HND pool of shuffled pages (one more,
    all NaN, last) behind a shared (or a K / V) page table, the query and KV
    offsets, a zeroed workspace and ticket.  A request without keys names the
    NaN page in every slot of its table rows."""

    device = torch.device("cuda")
    batch = len(q_lens)
    hq, hkv = heads
    max_q, max_kv = max(q_lens), max(kv_lens)
    generator = torch.Generator(device="cpu").manual_seed(seed + max_q)
    pages = (max_kv + page_size - 1) // page_size
    total_pages = 2 * batch * pages
    query = (torch.randn((sum(q_lens), hq, 128), generator=generator) * 2).to(
        device=device, dtype=torch.bfloat16
    )
    key, value = (
        torch.randn((total_pages + 1, hkv, page_size, 128), generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        for _ in range(2)
    )
    key[total_pages] = float("nan")
    value[total_pages] = float("nan")
    order = torch.randperm(total_pages, generator=generator).to(torch.int32)
    tables_k = order[: batch * pages].view(batch, pages).clone()
    tables_v = tables_k if shared else order[batch * pages :].view(batch, pages).clone()
    for b, kv_len in enumerate(kv_lens):
        if kv_len == 0:
            tables_k[b] = total_pages
            tables_v[b] = total_pages
    return SimpleNamespace(
        heads=heads,
        q_lens=q_lens,
        kv_lens=kv_lens,
        shared=shared,
        query=query,
        key=key,
        value=value,
        tables_k=tables_k.to(device),
        tables_v=tables_v.to(device),
        block_tables=(
            tables_k if shared else torch.stack((tables_k, tables_v), dim=1)
        ).to(device),
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        cum_q=torch.cat(
            (torch.zeros(1, dtype=torch.int32), torch.tensor(q_lens).cumsum(0).int())
        ).to(device),
        cum_kv=torch.cat(
            (torch.zeros(1, dtype=torch.int32), torch.tensor(kv_lens).cumsum(0).int())
        ).to(device),
        workspace=torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        counter=torch.zeros(
            get_trtllm_gen_multi_ctas_kv_counter_bytes(
                batch, hq, get_device_sm_count(device)
            ),
            dtype=torch.uint8,
            device=device,
        ),
    )


def _run_bf16_batch(case, *, return_lse=False):
    """One causal launch of the batch, output and LSE filled with NaN first."""

    out = torch.full_like(case.query, float("nan"))
    lse = None
    if return_lse:
        lse = torch.full(
            case.query.shape[:2], float("nan"), dtype=torch.float32, device="cuda"
        )
    prefill.trtllm_batch_context_with_kv_cache(
        case.query,
        (case.key, case.value),
        case.workspace,
        case.block_tables,
        case.seq_lens,
        max(case.q_lens),
        max(case.kv_lens),
        128**-0.5,
        1.0,
        len(case.q_lens),
        case.cum_q,
        case.cum_kv,
        out=out,
        lse=lse,
        return_lse=return_lse,
        kv_layout="HND",
        uses_shared_paged_kv_idx=case.shared,
        multi_ctas_kv_counter_buffer=case.counter,
        backend="cake",
    )
    torch.cuda.synchronize()
    assert not case.counter.any()
    return out, lse


def _rows_without_keys(q_lens, kv_lens):
    """The query rows that see no key under the bottom-right causal mask: every
    row of a request without keys, and the first ``q_len - kv_len`` rows of a
    request with fewer keys than query tokens."""

    rows, start = [], 0
    for q_len, kv_len in zip(q_lens, kv_lens, strict=True):
        rows += range(start, start + max(0, q_len - kv_len))
        start += q_len
    return rows


def _check_rows_with_and_without_keys(case, out, lse):
    """Rows with a visible key match the FP32 reference; rows without one are
    exactly 0 (their LSE -inf)."""

    blank = _rows_without_keys(case.q_lens, case.kv_lens)
    seen = torch.ones(out.shape[0], dtype=torch.bool, device=out.device)
    seen[blank] = False
    expected = _reference_context(
        case.query,
        case.key,
        case.value,
        case.tables_k,
        case.tables_v,
        case.q_lens,
        case.kv_lens,
        scale=128**-0.5,
    )
    torch.testing.assert_close(out.float()[seen], expected[seen], atol=1e-2, rtol=1e-2)
    assert torch.equal(out[blank], torch.zeros_like(out[blank]))
    if lse is not None:
        assert bool((lse[blank] == -math.inf).all())
        assert bool(torch.isfinite(lse[seen]).all())


def _tiles_without_keys(heads, q_lens, kv_lens):
    """The (request, m-block) tiles of the generic route whose causal block
    count is 0 before the clamp: KV length 0, or ``q - kv = 2T (m + 1)`` with
    ``T = 128 / pack`` query tokens per stage."""

    pack = cake_api._context_pack_g(max(q_lens), max(kv_lens), *heads)
    tokens = 128 // pack
    blocks = (max(q_lens) + 2 * tokens - 1) // (2 * tokens)
    return [
        (b, m)
        for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True))
        for m in range(blocks)
        if kv_len == 0 or q_len - kv_len == 2 * tokens * (m + 1)
    ]


@_GPU
@pytest.mark.parametrize(
    ("heads", "q_lens", "kv_lens", "page_size", "return_lse"),
    [
        # An empty request (no query token, no key) between others.
        ((10, 2), (300, 0, 37, 64), (900, 0, 100, 130), 32, False),
        # A request with query tokens but no key, with and without the LSE.
        ((32, 4), (200, 5, 64), (700, 0, 64), 16, False),
        ((32, 4), (200, 5, 64), (700, 0, 64), 16, True),
        # Causal tiles whose rows see no key (q - kv = 2 * 128 at pack 1 and
        # 2 * 16 at pack 8) in requests that hold keys.
        ((10, 2), (511, 257, 511, 511), (2047, 1, 1024, 600), 32, False),
        ((32, 4), (64, 100), (32, 300), 16, True),
    ],
)
def test_cake_context_bf16_jit_route_runs_tiles_without_keys(
    monkeypatch, served_programs, heads, q_lens, kv_lens, page_size, return_lse
) -> None:
    """The component's JIT route (no generated program serves these batches)
    on tiles without a visible key: the launch completes, rows with a visible
    key match the FP32 reference, rows without one are 0 (LSE -inf) although
    a request without keys names a NaN page, and the ticket resets."""

    loaded = []
    load = cake_api.load_cake_fmha_context_bf16_module

    def spy(*args, **kwargs):
        loaded.append(args)
        return load(*args, **kwargs)

    monkeypatch.setattr(cake_api, "load_cake_fmha_context_bf16_module", spy)
    assert _tiles_without_keys(heads, q_lens, kv_lens)
    # Every batch but the one with the empty request has rows without a key.
    assert _rows_without_keys(q_lens, kv_lens) or 0 in q_lens
    case = _bf16_batch(heads, q_lens, kv_lens, page_size)
    out, lse = _run_bf16_batch(case, return_lse=return_lse)
    assert loaded and not served_programs
    _check_rows_with_and_without_keys(case, out, lse)


@_GPU
@pytest.mark.parametrize(
    ("program", "q_lens", "kv_lens", "page_size", "shared"),
    [
        # Rows without a visible key in the exact program (pack 5: q - kv = 50)
        # and in the JIT route (pack 1: q - kv = 256), and a request without keys.
        ("Q511", (511,) * 4, (2047, 0, 461, 255), 32, True),
        ("Q257", (257,) * 4, (1024, 0, 207, 157), 1024, False),
    ],
)
def test_cake_context_bf16_program_and_jit_route_agree_on_rows_without_keys(
    monkeypatch, served_programs, program, q_lens, kv_lens, page_size, shared
) -> None:
    """A batch the one-tile BF16 program serves, with a request without keys
    and causal tiles without a visible key: the program and the JIT route
    (the exact profile off) both complete, give the same bits (0) on every row
    without a visible key and match the FP32 reference on the others."""

    target = cake_api._cake_fmha_target(torch.device("cuda"))
    program = f"{program}_SM{_TARGET_ARCH[target][3:].upper()}"
    case = _bf16_batch((10, 2), q_lens, kv_lens, page_size, shared=shared)
    out, _ = _run_bf16_batch(case)
    assert served_programs == [program]
    _check_rows_with_and_without_keys(case, out, None)
    monkeypatch.setattr(cake_api, "_context_bf16_exact_profile", lambda *a, **k: None)
    assert _tiles_without_keys((10, 2), q_lens, kv_lens)
    jit_out, _ = _run_bf16_batch(case)
    assert served_programs == [program]
    _check_rows_with_and_without_keys(case, jit_out, None)
    blank = _rows_without_keys(q_lens, kv_lens)
    assert blank
    assert torch.equal(_bits(jit_out[blank]), _bits(out[blank]))


# --- GPU: the other context JIT routes on tiles without a visible key -------

#: The context JIT routes besides BF16: KV and output dtypes, head dim, layout
#: and page tables the route takes, causal or not, (bmm1, bmm2) and the
#: (atol, rtol) of the FP32 reference check with the share of values allowed
#: outside it (the prefill tests' tolerances; an NVFP4 cache also needs a
#: cosine similarity above 0.86).  NVFP4 caches pack two E2M1 values per byte
#: under one FP8 scale per 16 values.  The HD256 routes stage the query, the cache and the metadata, and
#: reset their tile ticket there.
_JIT_ROUTES = {
    "context_fp8": SimpleNamespace(
        loader="load_cake_fmha_context_fp8_module",
        head_dim=128,
        query=torch.float8_e4m3fn,
        cache=torch.float8_e4m3fn,
        out=torch.float8_e4m3fn,
        layout="HND",
        shared=True,
        causal=True,
        scales=(0.0625, 0.75),
        tol=(0.07, 0.05),
        mismatch=0.0,
    ),
    "context_nvfp4": SimpleNamespace(
        loader="load_cake_fmha_context_nvfp4_module",
        head_dim=128,
        query=torch.float8_e4m3fn,
        cache=torch.uint8,
        out=torch.float8_e4m3fn,
        layout="HND",
        shared=True,
        causal=True,
        scales=(0.0625, 0.75),
        tol=(0.5, 0.5),
        mismatch=0.1,
    ),
    "context_fp8_hd256": SimpleNamespace(
        loader="load_cake_fmha_context_fp8_hd256_module",
        head_dim=256,
        query=torch.float8_e4m3fn,
        cache=torch.float8_e4m3fn,
        out=torch.bfloat16,
        layout="NHD",
        shared=False,
        causal=True,
        scales=(0.0625, 0.75),
        tol=(0.06, 0.04),
        mismatch=0.0,
    ),
    "context_fp16_hd256": SimpleNamespace(
        loader="load_cake_fmha_context_fp16_hd256_module",
        head_dim=256,
        query=torch.float16,
        cache=torch.float16,
        out=torch.float16,
        layout="NHD",
        shared=False,
        causal=False,
        scales=(256**-0.5, 1.0),
        tol=(1e-2, 1e-2),
        mismatch=0.0,
    ),
}
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1 = _E2M1 + tuple(-x for x in _E2M1)


def _swizzle_v_scales(scales):
    """The V block-scale order of the NVFP4 HND cache (pages, heads, page, 8)."""

    pages, heads, page_size, scale_dim = scales.shape
    return (
        scales.reshape(pages, heads, page_size // 4, 4, 4, scale_dim // 4)
        .permute(0, 1, 2, 4, 5, 3)
        .reshape_as(scales)
        .contiguous()
    )


def _jit_route_batch(component, heads, q_lens, kv_lens, page_size, *, seed=20261029):
    """Random inputs of one JIT route on the GPU: a pool of shuffled pages (one
    more, all NaN, last; NaN block scales for NVFP4) behind a shared (or a K /
    V) page table, in the route's layout.  A request without keys names the
    NaN page in every slot of its table rows."""

    spec = _JIT_ROUTES[component]
    device = torch.device("cuda")
    batch = len(q_lens)
    hq, hkv = heads
    generator = torch.Generator(device="cpu").manual_seed(seed + max(q_lens))
    pages = max(1, (max(kv_lens) + page_size - 1) // page_size)
    total_pages = 2 * batch * pages
    query = torch.randn((sum(q_lens), hq, spec.head_dim), generator=generator)
    pool = (total_pages + 1, hkv, page_size)
    scales = None
    if component == "context_nvfp4":
        key, value = (
            torch.randint(0, 256, (*pool, 64), generator=generator, dtype=torch.uint8)
            for _ in range(2)
        )
        scales = [
            (torch.rand((*pool, 8), generator=generator) * 1.5 + 0.5).to(
                torch.float8_e4m3fn
            )
            for _ in range(2)
        ]
        for sf in scales:
            sf[total_pages] = float("nan")
    else:
        key, value = (
            torch.randn((*pool, spec.head_dim), generator=generator).to(spec.cache)
            for _ in range(2)
        )
        key[total_pages] = float("nan")
        value[total_pages] = float("nan")
    order = torch.randperm(total_pages, generator=generator).to(torch.int32)
    tables_k = order[: batch * pages].view(batch, pages).clone()
    tables_v = (
        tables_k if spec.shared else order[batch * pages :].view(batch, pages).clone()
    )
    for b, kv_len in enumerate(kv_lens):
        if kv_len == 0:
            tables_k[b] = total_pages
            tables_v[b] = total_pages
    cache = (key, value)
    if spec.layout == "NHD":
        cache = tuple(x.transpose(1, 2).contiguous() for x in cache)
    return SimpleNamespace(
        component=component,
        spec=spec,
        heads=heads,
        q_lens=q_lens,
        kv_lens=kv_lens,
        page_size=page_size,
        query=query.to(device=device, dtype=spec.query),
        key=key.to(device),
        value=value.to(device),
        scales=None if scales is None else [sf.to(device) for sf in scales],
        cache=tuple(x.to(device) for x in cache),
        kv_cache_sf=(
            None
            if scales is None
            else (scales[0].to(device), _swizzle_v_scales(scales[1]).to(device))
        ),
        tables_k=tables_k.to(device),
        tables_v=tables_v.to(device),
        block_tables=(
            tables_k if spec.shared else torch.stack((tables_k, tables_v), dim=1)
        ).to(device),
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        cum_q=torch.cat(
            (torch.zeros(1, dtype=torch.int32), torch.tensor(q_lens).cumsum(0).int())
        ).to(device),
        cum_kv=torch.cat(
            (torch.zeros(1, dtype=torch.int32), torch.tensor(kv_lens).cumsum(0).int())
        ).to(device),
        workspace=torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        counter=torch.zeros(
            get_trtllm_gen_multi_ctas_kv_counter_bytes(
                batch, hq, get_device_sm_count(device)
            ),
            dtype=torch.uint8,
            device=device,
        ),
    )


def _run_jit_route_batch(case, *, return_lse=False):
    """One launch of the batch, output and LSE filled with NaN first; the
    ticket of an HD128 route is back at 0 after it."""

    spec = case.spec
    shape = (case.query.shape[0], case.heads[0], spec.head_dim)
    out = torch.full(shape, float("nan"), device="cuda").to(spec.out)
    lse = None
    if return_lse:
        lse = torch.full(shape[:2], float("nan"), dtype=torch.float32, device="cuda")
    prefill.trtllm_batch_context_with_kv_cache(
        case.query,
        case.cache,
        case.workspace,
        case.block_tables,
        case.seq_lens,
        max(case.q_lens),
        max(case.kv_lens),
        *spec.scales,
        len(case.q_lens),
        case.cum_q,
        case.cum_kv,
        out=out,
        out_dtype=spec.out,
        lse=lse,
        return_lse=return_lse,
        causal=spec.causal,
        kv_layout=spec.layout,
        kv_cache_sf=case.kv_cache_sf,
        uses_shared_paged_kv_idx=spec.shared,
        multi_ctas_kv_counter_buffer=case.counter,
        backend="cake",
    )
    torch.cuda.synchronize()
    if spec.head_dim == 128:
        assert not case.counter.any()
    return out, lse


def _jit_route_values(case, side, pages):
    """FP32 values ``(heads, tokens, head_dim)`` of the given pool pages; an
    NVFP4 value is its E2M1 code times its block scale, rounded to FP8 as the
    kernel dequantizes it."""

    data = (case.key if side == 0 else case.value)[pages]
    if case.component == "context_nvfp4":
        codes = torch.stack((data & 15, data >> 4), -1).flatten(-2).long()
        levels = torch.tensor(_E2M1, device=data.device)[codes]
        scale = case.scales[side][pages].float()
        data = (
            (levels.reshape(*data.shape[:-1], 8, 16) * scale.unsqueeze(-1))
            .to(torch.float8_e4m3fn)
            .flatten(-2)
        )
    return data.float().permute(1, 0, 2, 3).reshape(data.shape[1], -1, data.shape[-1])


def _jit_route_reference(case):
    """FP32 attention of the batch (the route's mask) and its base-2 LSE."""

    spec = case.spec
    hq, hkv = case.heads
    bmm1, bmm2 = spec.scales
    rows = case.query.shape[0]
    out = torch.zeros((rows, hq, spec.head_dim), device="cuda")
    lse = torch.full((rows, hq), -math.inf, device="cuda")
    start = 0
    for b, (q_len, kv_len) in enumerate(zip(case.q_lens, case.kv_lens, strict=True)):
        if q_len == 0 or kv_len == 0:
            start += q_len
            continue
        count = (kv_len + case.page_size - 1) // case.page_size
        k, v = (
            _jit_route_values(case, side, table[b, :count].long())[:, :kv_len]
            for side, table in ((0, case.tables_k), (1, case.tables_v))
        )
        k = k.repeat_interleave(hq // hkv, dim=0)
        v = v.repeat_interleave(hq // hkv, dim=0)
        q = case.query[start : start + q_len].float().transpose(0, 1)
        logits = torch.einsum("hqd,hkd->hqk", q, k) * bmm1
        if spec.causal:
            right = torch.arange(q_len, device="cuda")[:, None] + kv_len - q_len
            logits.masked_fill_(torch.arange(kv_len, device="cuda") > right, -math.inf)
        prob = torch.softmax(logits, dim=-1).nan_to_num(nan=0.0)
        out[start : start + q_len] = (
            torch.einsum("hqk,hkd->hqd", prob, v).transpose(0, 1) * bmm2
        )
        lse[start : start + q_len] = (
            torch.logsumexp(logits, dim=-1) / math.log(2.0)
        ).transpose(0, 1)
        start += q_len
    return out, lse


def _jit_route_rows_without_keys(case):
    """Every row of a request without keys and, causal, the first
    ``q_len - kv_len`` rows of a request with fewer keys than query tokens."""

    rows, start = [], 0
    for q_len, kv_len in zip(case.q_lens, case.kv_lens, strict=True):
        if kv_len == 0:
            rows += range(start, start + q_len)
        elif case.spec.causal:
            rows += range(start, start + max(0, q_len - kv_len))
        start += q_len
    return rows


def _check_jit_route_rows(case, out, lse):
    """Rows with a visible key match the FP32 reference; rows without one are
    0 (their LSE -inf)."""

    blank = _jit_route_rows_without_keys(case)
    seen = torch.ones(out.shape[0], dtype=torch.bool, device=out.device)
    seen[blank] = False
    expected, expected_lse = _jit_route_reference(case)
    got, want = out.float()[seen], expected[seen]
    atol, rtol = case.spec.tol
    if case.spec.mismatch:
        assert_close_with_mismatch_tolerance(
            got,
            want,
            rtol=rtol,
            atol=atol,
            max_mismatched_elements=int(case.spec.mismatch * got.numel()),
        )
        cos = torch.nn.functional.cosine_similarity(
            got.reshape(-1), want.reshape(-1), dim=0
        )
        assert cos.item() > 0.86
    else:
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
    assert bool((out.float()[blank] == 0).all())
    if lse is not None:
        assert bool((lse[blank] == -math.inf).all())
        torch.testing.assert_close(lse[seen], expected_lse[seen], atol=3e-3, rtol=1e-3)


class _RouteProbe(Exception):
    pass


def _jit_route_of(monkeypatch, case, *, return_lse=False):
    """The route the dispatch computes for the batch (no launch)."""

    def probe(device, route):
        raise _RouteProbe(route)

    with monkeypatch.context() as m:
        m.setattr(cake_api, "get_cake_fmha_context_module", probe)
        with pytest.raises(_RouteProbe) as caught:
            _run_jit_route_batch(case, return_lse=return_lse)
    return caught.value.args[0]


def _jit_route_tiles_without_keys(case, route):
    """The (request, m-block) tiles whose block count is 0 before the clamp:
    KV length 0, or (causal) a key bound of exactly 0 for the tile: ``q - kv =
    (m + 1) * rows`` with ``rows = 2 * 128 / pack`` (HD128) or 128 (HD256)."""

    rows = 128 if case.spec.head_dim == 256 else 2 * (128 // route.pack_g)
    return [
        (b, m)
        for b, (q_len, kv_len) in enumerate(zip(case.q_lens, case.kv_lens, strict=True))
        for m in range(route.num_m_blocks)
        if kv_len == 0 or (case.spec.causal and q_len - kv_len == rows * (m + 1))
    ]


@_GPU
@pytest.mark.parametrize(
    ("component", "heads", "q_lens", "kv_lens", "page_size", "return_lse"),
    [
        # An empty request (no query token, no key) between others.
        ("context_fp8", (32, 4), (300, 0, 37, 64), (900, 0, 100, 130), 16, False),
        ("context_nvfp4", (32, 4), (300, 0, 37, 64), (900, 0, 100, 130), 16, False),
        ("context_fp8_hd256", (32, 8), (300, 0, 37, 64), (900, 0, 100, 130), 16, False),
        ("context_fp16_hd256", (32, 8), (300, 0, 37, 64), (900, 0, 100, 130), 16, False),
        # A request with query tokens but no key (with and without the LSE
        # where the route returns one).
        ("context_fp8", (32, 4), (200, 5, 64), (700, 0, 64), 16, False),
        ("context_fp8", (32, 4), (200, 5, 64), (700, 0, 64), 16, True),
        ("context_nvfp4", (32, 4), (200, 5, 64), (700, 0, 64), 16, False),
        ("context_fp8_hd256", (32, 8), (200, 5, 64), (700, 0, 64), 16, False),
        ("context_fp16_hd256", (32, 8), (200, 5, 64), (700, 0, 64), 16, False),
        ("context_fp16_hd256", (16, 2), (2048, 300, 1), (2048, 0, 0), 16, False),
        # Causal tiles whose rows see no key in requests that hold keys:
        # q - kv = 2 * 128 / pack * (m + 1) (HD128) or 128 * (m + 1) (HD256).
        ("context_fp8", (8, 8), (700, 257, 385), (1500, 1, 129), 16, False),
        ("context_fp8", (32, 4), (64, 100), (32, 300), 16, True),
        ("context_nvfp4", (8, 8), (700, 257, 385), (1500, 1, 129), 16, False),
        ("context_nvfp4", (32, 4), (64, 100), (32, 300), 16, False),
        ("context_fp8_hd256", (8, 8), (511, 129, 300), (2047, 1, 44), 16, False),
    ],
)  # fmt: skip
def test_cake_context_jit_routes_run_tiles_without_keys(
    monkeypatch,
    served_programs,
    component,
    heads,
    q_lens,
    kv_lens,
    page_size,
    return_lse,
) -> None:
    """Each context JIT route (the program tables off) on tiles without a
    visible key: the launch completes, rows with a visible key match the FP32
    reference, rows without one are 0 (LSE -inf) although a request without
    keys names a NaN page, and the ticket resets."""

    _skip_without_nvfp4_programs(component)
    monkeypatch.setattr(cake_api, "_CONTEXT_PROGRAM_ROUTES", {})
    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_RUNTIME_ROUTES", {})
    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_SPLIT_ROUTES", {})
    spec = _JIT_ROUTES[component]
    loaded = []
    load = getattr(cake_api, spec.loader)

    def spy(*args, **kwargs):
        loaded.append(args)
        return load(*args, **kwargs)

    monkeypatch.setattr(cake_api, spec.loader, spy)
    case = _jit_route_batch(component, heads, q_lens, kv_lens, page_size)
    route = _jit_route_of(monkeypatch, case, return_lse=return_lse)
    assert route.component == component and route.program is None
    assert _jit_route_tiles_without_keys(case, route)
    out, lse = _run_jit_route_batch(case, return_lse=return_lse)
    assert loaded and not served_programs
    _check_jit_route_rows(case, out, lse)


def _jit_route_head(case, requests):
    """The batch's first ``requests`` requests: the same pool, query rows and
    table rows."""

    rows = sum(case.q_lens[:requests])
    return SimpleNamespace(
        **{
            **vars(case),
            "q_lens": case.q_lens[:requests],
            "kv_lens": case.kv_lens[:requests],
            "query": case.query[:rows],
            "tables_k": case.tables_k[:requests],
            "tables_v": case.tables_v[:requests],
            "block_tables": case.block_tables[:requests],
            "seq_lens": case.seq_lens[:requests],
            "cum_q": case.cum_q[: requests + 1],
            "cum_kv": case.cum_kv[: requests + 1],
        }
    )


@_GPU
@pytest.mark.parametrize(
    ("heads", "q_lens", "kv_lens", "page_size"),
    [
        # 30, 30 and 20 work units (requests x query heads) under test, at 3,
        # 4 and 3 m-blocks; the last request (one query token, one key)
        # completes 40 units.
        ((10, 2), (300, 64, 1, 1), (900, 64, 17, 1), 32),
        ((10, 2), (511, 511, 511, 1), (2047, 1024, 600, 1), 32),
        ((20, 4), (257, 1), (300, 1), 16),
    ],
)  # fmt: skip
def test_cake_context_fp8_hd256_jit_route_runs_a_partial_l2_section(
    monkeypatch, served_programs, heads, q_lens, kv_lens, page_size
) -> None:
    """The FP8 HD256 JIT route takes its tiles in sections of 8 work units.  A
    batch whose units are not a multiple of 8, with more than one m-block,
    ends with a partial section: the launch completes, every row matches the
    FP32 reference, and the rows carry the bits the same requests get in a
    batch of whole sections (one more request, the same maxima)."""

    monkeypatch.setattr(cake_api, "_CONTEXT_PROGRAM_ROUTES", {})
    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_RUNTIME_ROUTES", {})
    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_SPLIT_ROUTES", {})
    whole = _jit_route_batch("context_fp8_hd256", heads, q_lens, kv_lens, page_size)
    case = _jit_route_head(whole, len(q_lens) - 1)
    assert len(case.q_lens) * heads[0] % 8 and not len(q_lens) * heads[0] % 8
    route = _jit_route_of(monkeypatch, case)
    assert route.component == "context_fp8_hd256" and route.program is None
    assert route.num_m_blocks > 1 and not _jit_route_tiles_without_keys(case, route)
    out, _ = _run_jit_route_batch(case)
    assert not served_programs
    _check_jit_route_rows(case, out, None)
    whole_out, _ = _run_jit_route_batch(whole)
    assert torch.equal(_bits(out), _bits(whole_out[: out.shape[0]]))


@_GPU
@pytest.mark.parametrize(
    ("component", "program", "heads", "q_lens", "kv_lens", "page_size"),
    [
        # Requests without keys beside keyed ones (the exact FP8 program, pack
        # 8) and causal tiles without a visible key (NVFP4 Q511, pack 8:
        # q - kv = 32 (m + 1)).
        ("context_fp8", "B256", (32, 4), (32,) * 4, (1024, 0, 17, 0), 64),
        ("context_nvfp4", "Q511", (32, 4), (511,) * 4, (2047, 0, 479, 255), 16),
    ],
)  # fmt: skip
def test_cake_context_quantized_program_and_jit_route_agree_on_rows_without_keys(
    monkeypatch, served_programs, component, program, heads, q_lens, kv_lens, page_size
) -> None:
    """A batch an FP8 / NVFP4 program serves, with requests without keys or
    causal tiles without a visible key: the program and the component's JIT
    route (the program tables off) both complete and give the same bits on
    every row, 0 on the rows without a visible key."""

    _skip_without_nvfp4_programs(component)
    _whole_tile_routes(monkeypatch)
    target = cake_api._cake_fmha_target(torch.device("cuda"))
    if program == "B256":
        program = _FP8_EXACT[target]
    else:
        program = f"{program}_SM{_TARGET_ARCH[target][3:].upper()}"
    case = _jit_route_batch(component, heads, q_lens, kv_lens, page_size)
    out, _ = _run_jit_route_batch(case)
    assert served_programs == [program]
    _check_jit_route_rows(case, out, None)
    monkeypatch.setattr(cake_api, "_CONTEXT_PROGRAM_ROUTES", {})
    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_RUNTIME_ROUTES", {})
    route = _jit_route_of(monkeypatch, case)
    assert route.component == component and route.program is None
    assert _jit_route_tiles_without_keys(case, route)
    jit_out, _ = _run_jit_route_batch(case)
    assert served_programs == [program]
    assert torch.equal(_bits(jit_out), _bits(out))
    assert _jit_route_rows_without_keys(case)


@_GPU
@pytest.mark.parametrize(
    ("case", "program"),
    [
        (
            dict(q_dtype="fp8", o_dtype="fp8", kv_dtype="fp8", batch_size=2,
                 page_size=64, max_q_len=32, max_kv_len=159, device_scale=False),
            "B256",
        ),
        (
            dict(q_dtype="fp8", o_dtype="fp8", kv_dtype="fp8", batch_size=2,
                 page_size=64, max_q_len=32, max_kv_len=159, device_scale=True),
            "B256_SCALE_PTR",
        ),
        (
            dict(q_dtype="fp8", o_dtype="fp8", kv_dtype="nvfp4", batch_size=4,
                 page_size=16, max_q_len=511, max_kv_len=1536, device_scale=False),
            "Q511",
        ),
        (
            dict(q_dtype="fp8", o_dtype="fp8", kv_dtype="nvfp4", batch_size=4,
                 page_size=16, max_q_len=511, max_kv_len=1536, device_scale=True),
            "Q511_SCALE_PTR",
        ),
        (
            dict(q_dtype="fp8", o_dtype="fp8", kv_dtype="fp8", batch_size=2,
                 page_size=16, max_q_len=32, max_kv_len=159, device_scale=False),
            "RUNTIME_PACK",
        ),
        (
            dict(q_dtype="fp8", o_dtype="fp8", kv_dtype="fp8", batch_size=3,
                 page_size=16, max_q_len=200, max_kv_len=700, device_scale=True),
            "RUNTIME_PACK",
        ),
    ],
)  # fmt: skip
def test_cake_context_quantized_programs_match_reference(
    monkeypatch, served_programs, case, program
) -> None:
    from tests.attention.test_trtllm_gen_attention_prefill import (
        _test_trtllm_batch_prefill,
    )

    _whole_tile_routes(monkeypatch)
    target = cake_api._cake_fmha_target(torch.device("cuda"))
    if case["kv_dtype"] == "nvfp4":
        _skip_without_nvfp4_programs("context_nvfp4")
    if program.startswith("Q511"):
        program = f"{program}_SM{_TARGET_ARCH[target][3:].upper()}"
    if program.startswith("B256"):
        program = {"B256": _FP8_EXACT[target]}.get(
            program, _DEVICE_SCALE_PROGRAM[_FP8_EXACT[target]]
        )
        # a launch of at most the keys of the route's short-KV entry takes
        # the entry's program
        short = cake_api._CONTEXT_FP8_SHORT_KV_ROUTES.get(target, {}).get(program)
        if short and case["max_kv_len"] <= short[1]:
            program = short[0]
    original = prefill.trtllm_batch_context_with_kv_cache

    def cake_context(*args, **kwargs):
        return original(*args, backend="cake", **kwargs)

    monkeypatch.setattr(prefill, "trtllm_batch_context_with_kv_cache", cake_context)
    _test_trtllm_batch_prefill(
        kv_layout="HND",
        num_kv_heads=4,
        head_grp_size=8,
        causal=True,
        window_left=-1,
        enable_pdl=False,
        enable_sink=False,
        head_dim=128,
        uses_shared_paged_kv_idx=True,
        **case,
    )
    if program == "RUNTIME_PACK":
        # 16-token pages: no exact route; a runtime-size program of any page
        assert served_programs and all(
            served in _RUNTIME_PACK and "_PAGE64" not in served
            for served in served_programs
        ), served_programs
        return
    assert program in served_programs


# --- GPU: bitwise against the JIT routes and launch invariances -------------

# The rows the programs serve outside the one-tile classes: name -> (shape
# row, batch, max_q_len).  The one-tile classes have no JIT route to compare.
BITWISE_ROWS = {
    "bf16_b1_q8192_k8192": ("bf16_b1_q8192_k8192", 1, 8192),
    "bf16_b128_q64_k2048": ("bf16_b128_q64_k2048", 128, 64),
    "fp8_b256_q32_k1024": ("fp8_b256_q32_k1024", 256, 32),
    "nvfp4_b4_q511_k2047": ("nvfp4_b4_q511_k2047", 4, 511),
    "fp8_b64_q32_k1024_g4": ("fp8_b64_q32_k1024_g4", 64, 32),
    "fp8_b16_q32_k1024_g16_page16": ("fp8_b16_q32_k1024_g16_page16", 16, 32),
    "fp8_b2_q256_k2048_g3_page32": ("fp8_b2_q256_k2048_g3_page32", 2, 256),
    "fp8_b4_q128_k1024_g6_page16": ("fp8_b4_q128_k1024_g6_page16", 4, 128),
    "fp8_b16_q32_k128": ("fp8_b16_q32_k128", 16, 32),
    "fp8_b16_q32_k128_g4": ("fp8_b16_q32_k128_g4", 16, 32),
    **WIDENED_ROWS,
}
_SCALES = {"context_bf16": (128**-0.5, 1.0)}
_QUANT_SCALES = (0.0625, 0.75)


def _bits(tensor):
    return tensor.contiguous().view(torch.uint8)


def _context_case(row, batch, max_q, *, seed=20261026):
    """Random inputs of one shape row on the GPU: every request ``max_q``
    query tokens, KV lengths between ``max_q`` and the row's maximum, shuffled
    pages of an HND pool behind a shared (or a K / V) page table."""

    component, _, _, max_kv, hq, hkv, page, shared, programs = SHAPE_ROWS[row]
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed + batch + max_q)
    kv_lens = torch.randint(
        max_q, max_kv + 1, (batch,), generator=generator, device=device
    ).int()
    kv_lens[0] = max_kv
    pages = (max_kv + page - 1) // page
    total_pages = 2 * batch * pages
    order = torch.randperm(total_pages, generator=generator, device=device).int()
    tables_k = order[: batch * pages].view(batch, pages)
    tables_v = tables_k if shared else order[batch * pages :].view(batch, pages)

    def randn(*shape):
        return torch.randn(shape, generator=generator, device=device)

    case = SimpleNamespace(
        component=component,
        programs=programs,
        batch=batch,
        max_q=max_q,
        max_kv=max_kv,
        shared=shared,
        tables_k=tables_k,
        tables_v=tables_v,
        block_tables=tables_k if shared else torch.stack((tables_k, tables_v), 1),
        seq_lens=kv_lens,
        cum_q=torch.arange(
            0, (batch + 1) * max_q, max_q, dtype=torch.int32, device=device
        ),
        cum_kv=torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=device),
                kv_lens.cumsum(0, dtype=torch.int32),
            )
        ),
        workspace=torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        counter=torch.zeros(
            get_trtllm_gen_multi_ctas_kv_counter_bytes(
                batch, hq, get_device_sm_count(device)
            ),
            dtype=torch.uint8,
            device=device,
        ),
        kv_cache_sf=None,
    )
    pool = (total_pages, hkv, page)
    if component == "context_bf16":
        case.query = (randn(batch * max_q, hq, 128) * 2).bfloat16()
        case.key, case.value = (randn(*pool, 128).bfloat16() for _ in range(2))
    elif component == "context_fp8":
        case.query = randn(batch * max_q, hq, 128).to(torch.float8_e4m3fn)
        case.key, case.value = (
            randn(*pool, 128).to(torch.float8_e4m3fn) for _ in range(2)
        )
    else:
        case.query = randn(batch * max_q, hq, 128).to(torch.float8_e4m3fn)
        case.key, case.value = (
            torch.randint(0, 256, (*pool, 64), generator=generator, device=device).to(
                torch.uint8
            )
            for _ in range(2)
        )
        case.kv_cache_sf = tuple(
            (torch.rand((*pool, 8), generator=generator, device=device) * 1.5 + 0.5).to(
                torch.float8_e4m3fn
            )
            for _ in range(2)
        )
    return case


def _run_context(
    case,
    *,
    device_scales=False,
    kv_layout="HND",
    block_tables=None,
    shared=None,
):
    bmm1, bmm2 = _SCALES.get(case.component, _QUANT_SCALES)
    if device_scales:
        bmm1 = torch.tensor(bmm1, dtype=torch.float32, device="cuda")
        bmm2 = torch.tensor(bmm2, dtype=torch.float32, device="cuda")
    key, value = case.key, case.value
    if kv_layout == "NHD":
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
    out = torch.empty_like(case.query)
    prefill.trtllm_batch_context_with_kv_cache(
        case.query,
        (key, value),
        case.workspace,
        case.block_tables if block_tables is None else block_tables,
        case.seq_lens,
        case.max_q,
        case.max_kv,
        bmm1,
        bmm2,
        case.batch,
        case.cum_q,
        case.cum_kv,
        out=out,
        kv_layout=kv_layout,
        kv_cache_sf=case.kv_cache_sf,
        uses_shared_paged_kv_idx=case.shared if shared is None else shared,
        multi_ctas_kv_counter_buffer=case.counter,
        backend="cake",
    )
    torch.cuda.synchronize()
    assert not case.counter.any()
    return out


@_GPU
@pytest.mark.parametrize("name", sorted(BITWISE_ROWS))
def test_cake_context_programs_match_the_jit_route_bitwise(
    monkeypatch, served_programs, name
) -> None:
    """The program's output is bitwise the component's JIT route's (the route
    without the program tables; for a runtime-size program, the JIT module at
    its own pack) on its row and its widened class; with device scales the
    device-scale program's output is bitwise the host-scale one."""

    _whole_tile_routes(monkeypatch)
    row, batch, max_q = BITWISE_ROWS[name]
    case = _context_case(row, batch, max_q)
    _skip_without_nvfp4_programs(case.component)
    program = case.programs[cake_api._cake_fmha_target(torch.device("cuda"))]
    out = _run_context(case)
    assert served_programs == [program]
    if case.component != "context_bf16":
        device_out = _run_context(case, device_scales=True)
        assert served_programs[-1] == _device_scale_program(program)
        assert torch.equal(_bits(device_out), _bits(out))
    monkeypatch.setattr(cake_api, "_CONTEXT_PROGRAM_ROUTES", {})
    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_RUNTIME_ROUTES", {})
    served = len(served_programs)
    jit_out = _run_context(case)
    assert len(served_programs) == served
    assert torch.equal(_bits(jit_out), _bits(out)), (
        f"{name}: max abs difference "
        f"{(jit_out.float() - out.float()).abs().nan_to_num(posinf=1e9).max().item()}"
    )


#: FP8 rows with requests that hold fewer keys than query tokens: (batch,
#: max_q, KV lengths).  Under the bottom-right causal mask the first query
#: rows of such a request see no key.
SHORT_KV_ROWS = {
    "fp8_b256_q32_k1024": (3, 32, (1024, 17, 1)),
    "fp8_b64_q32_k1024_g4": (3, 32, (1024, 17, 1)),
    "fp8_b16_q32_k1024_g16_page16": (3, 32, (1024, 5, 31)),
    "fp8_b2_q256_k2048_g3_page32": (2, 256, (2048, 100)),
    "fp8_b4_q128_k1024_g6_page16": (3, 128, (1024, 40, 127)),
}


@_GPU
@pytest.mark.parametrize("name", sorted(SHORT_KV_ROWS))
def test_cake_context_programs_match_the_jit_route_with_fewer_keys_than_queries(
    monkeypatch, served_programs, name
) -> None:
    """A batch whose requests hold fewer keys than query tokens (the first
    query rows of such a request see no key): the program's output, those
    rows included, is bitwise the component's JIT route's."""

    _whole_tile_routes(monkeypatch)
    batch, max_q, kv_lens = SHORT_KV_ROWS[name]
    case = _context_case(name, batch, max_q)
    case.seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    case.cum_kv = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            case.seq_lens.cumsum(0, dtype=torch.int32),
        )
    )
    program = case.programs[cake_api._cake_fmha_target(torch.device("cuda"))]
    out = _run_context(case)
    assert served_programs == [program]
    monkeypatch.setattr(cake_api, "_CONTEXT_PROGRAM_ROUTES", {})
    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_RUNTIME_ROUTES", {})
    jit_out = _run_context(case)
    assert served_programs == [program]
    assert torch.equal(_bits(jit_out), _bits(out)), (
        f"{name}: max abs difference "
        f"{(jit_out.float() - out.float()).abs().nan_to_num(posinf=1e9).max().item()}"
    )


@_GPU
@pytest.mark.parametrize(
    ("row", "batch", "max_q"),
    [
        ("bf16_b128_q64_k2048", 3, 64),
        ("fp8_b256_q32_k1024", 3, 32),
        ("fp8_b64_q32_k1024_g4", 3, 32),
    ],
)
def test_cake_context_program_launch_invariances_are_bitwise(
    monkeypatch, served_programs, row, batch, max_q
) -> None:
    """NHD and HND caches, shared and separate page tables, a column-sliced
    page table and CUDA graph replay give the program's output bit for bit."""

    _whole_tile_routes(monkeypatch)
    case = _context_case(row, batch, max_q)
    program = case.programs[cake_api._cake_fmha_target(torch.device("cuda"))]
    out = _run_context(case)
    variants = {"nhd": _run_context(case, kv_layout="NHD")}
    separate = torch.stack((case.tables_k, case.tables_k), dim=1)
    variants["separate"] = _run_context(case, block_tables=separate, shared=False)
    pages = case.tables_k.shape[1]
    wide = torch.full(
        (batch, 2, pages + 3), -1, dtype=torch.int32, device=case.tables_k.device
    )
    wide[:, 0, 1 : 1 + pages] = case.tables_k
    wide[:, 1, 1 : 1 + pages] = case.tables_k
    sliced = wide[:, 0, 1 : 1 + pages]
    variants["sliced"] = _run_context(case, block_tables=sliced, shared=True)
    variants["sliced_separate"] = _run_context(
        case, block_tables=wide[:, :, 1 : 1 + pages], shared=False
    )
    assert served_programs == [program] * (1 + len(variants))
    for label, other in variants.items():
        assert torch.equal(_bits(other), _bits(out)), label

    # Graph replay against the direct launch (device scales for the FP8
    # program: no host synchronization in the captured call).
    device_scales = case.component != "context_bf16"
    direct = _run_context(case, device_scales=device_scales)
    graph_out = torch.empty_like(direct)
    bmm1, bmm2 = _SCALES.get(case.component, _QUANT_SCALES)
    if device_scales:
        bmm1 = torch.tensor(bmm1, dtype=torch.float32, device="cuda")
        bmm2 = torch.tensor(bmm2, dtype=torch.float32, device="cuda")

    def call():
        prefill.trtllm_batch_context_with_kv_cache(
            case.query,
            (case.key, case.value),
            case.workspace,
            case.block_tables,
            case.seq_lens,
            case.max_q,
            case.max_kv,
            bmm1,
            bmm2,
            case.batch,
            case.cum_q,
            case.cum_kv,
            out=graph_out,
            kv_layout="HND",
            uses_shared_paged_kv_idx=case.shared,
            multi_ctas_kv_counter_buffer=case.counter,
            backend="cake",
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
    graph_out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(graph_out), _bits(direct))
    assert not case.counter.any()


@_GPU
@pytest.mark.parametrize("explicit_stream", (False, True))
@pytest.mark.parametrize(
    ("row", "batch", "max_q"),
    [
        ("bf16_b128_q64_k2048", 3, 64),
        ("fp8_b256_q32_k1024", 3, 32),
        ("fp8_b64_q32_k1024_g4", 3, 32),
    ],
)
def test_cake_context_program_launches_on_the_current_torch_stream(
    monkeypatch, served_programs, row, batch, max_q, explicit_stream
) -> None:
    """The program launches on torch's current stream: queued behind a sleep
    on a side stream, its output is still unwritten when the default stream
    reads it, and is the direct launch's once the side stream drains.  Both
    ways the launch gets the stream: from the tvm-ffi call's torch tensor
    arguments, and from the launch's own stream context."""

    _whole_tile_routes(monkeypatch)
    if explicit_stream:
        monkeypatch.setattr(cake_api, "_ffi_call_takes_the_torch_stream", lambda: False)
    elif not cake_api._ffi_call_takes_the_torch_stream():
        pytest.skip("torch tensors carry no DLPack exchange API here")
    case = _context_case(row, batch, max_q)
    program = case.programs[cake_api._cake_fmha_target(torch.device("cuda"))]
    direct = _run_context(case)
    bmm1, bmm2 = _SCALES.get(case.component, _QUANT_SCALES)
    out = torch.zeros_like(direct)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(200_000_000)
        prefill.trtllm_batch_context_with_kv_cache(
            case.query,
            (case.key, case.value),
            case.workspace,
            case.block_tables,
            case.seq_lens,
            case.max_q,
            case.max_kv,
            bmm1,
            bmm2,
            case.batch,
            case.cum_q,
            case.cum_kv,
            out=out,
            kv_layout="HND",
            uses_shared_paged_kv_idx=case.shared,
            multi_ctas_kv_counter_buffer=case.counter,
            backend="cake",
        )
    early = out.clone()
    torch.cuda.synchronize()
    assert served_programs == [program] * 2
    assert not early.view(torch.uint8).any()
    assert torch.equal(_bits(out), _bits(direct))
    assert not case.counter.any()


@_GPU
@pytest.mark.parametrize("kv_len", (0, 300))
@pytest.mark.parametrize(
    "row", ("bf16_b128_q64_k2048", "fp8_b256_q32_k1024", "nvfp4_b4_q511_k2047")
)
def test_cake_context_batch_without_query_tokens_returns_before_any_launch(
    monkeypatch, row, kv_len
) -> None:
    """A batch whose requests have no query token (KV empty or not) returns its
    empty output and LSE without choosing a route or launching a kernel."""

    case = _context_case(row, 3, 1)
    device = case.query.device

    def refuse(*args, **kwargs):
        raise AssertionError("a batch without query tokens chose a route")

    monkeypatch.setattr(cake_api, "select_cake_fmha_context_route", refuse)
    monkeypatch.setattr(cake_api, "get_cake_fmha_context_module", refuse)
    query = case.query[:0]
    seq_lens = torch.full((case.batch,), kv_len, dtype=torch.int32, device=device)
    out = torch.empty_like(query)
    result, lse = prefill.trtllm_batch_context_with_kv_cache(
        query,
        (case.key, case.value),
        case.workspace,
        case.block_tables,
        seq_lens,
        0,
        kv_len,
        *_SCALES.get(case.component, _QUANT_SCALES),
        case.batch,
        torch.zeros(case.batch + 1, dtype=torch.int32, device=device),
        torch.arange(case.batch + 1, dtype=torch.int32, device=device) * kv_len,
        out=out,
        kv_layout="HND",
        kv_cache_sf=case.kv_cache_sf,
        uses_shared_paged_kv_idx=case.shared,
        multi_ctas_kv_counter_buffer=case.counter,
        return_lse=True,
        backend="cake",
    )
    assert result is out and out.shape[0] == 0
    assert lse.shape == (0, case.query.shape[1])
    assert not case.counter.any()


# FP8 calls the split plan splits on both targets: (q lengths, KV lengths,
# (query heads, KV heads), page size, shared page table).
_SPLIT_CASES = [
    # one request, 4 tiles of 128 KV blocks (B256's exact class at 16K keys)
    ((32,), (16384,), (32, 4), 64, True),
    # two requests at 16-token pages, ragged KV lengths, separate K / V tables
    ((128, 97), (8192, 6000), (32, 4), 16, False),
    # one request of 64 query heads over 8 KV heads, 32 KV heads' worth of
    # 4096 keys
    ((64,), (4096,), (64, 8), 64, True),
    # two requests of 32 query tokens over 16384 keys: a smaller pack's own
    # split (its units differ from pack 8's)
    ((32, 32), (16384, 16384), (32, 4), 64, True),
]


@_GPU
@pytest.mark.parametrize(
    ("q_lens", "kv_lens", "heads", "page_size", "shared"),
    _SPLIT_CASES,
    ids=[f"b{len(c[0])}_q{max(c[0])}_k{max(c[1])}_p{c[3]}" for c in _SPLIT_CASES],
)
def test_cake_context_split_programs_match_reference_and_replay(
    monkeypatch, served_programs, q_lens, kv_lens, heads, page_size, shared
) -> None:
    """The split-KV program of the call's pack and page size serves it; its
    output is within the FP8 contract tolerance of an FP32 reference, bitwise
    the same run to run (the split is the plan's, the merge order fixed), the
    tile counters and the ticket read zero after each launch, and a CUDA graph
    of the call serves new KV lengths under the captured maximum."""

    # the split programs' subject: no one-stage route takes the call
    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_ONE_STAGE_ROUTES", {})
    device = torch.device("cuda")
    target = cake_api._cake_fmha_target(device)
    batch = len(q_lens)
    hq, hkv = heads
    max_q, max_kv = max(q_lens), max(kv_lens)
    pack = cake_api._context_fp8_runtime_pack(
        cake_api._context_pack_g(max_q, max_kv, hq, hkv)
    )
    choice = cake_api._context_fp8_split_choice(
        target, batch, max_q, max_kv, hq, pack, get_device_sm_count(device)
    )
    assert choice is not None, "the case is chosen for a split launch"
    program = f"RUNTIME_SK_PACK{choice[0]}" + ("_PAGE64" if page_size == 64 else "")
    generator = torch.Generator(device="cpu").manual_seed(20261027 + max_kv)
    pages = (max_kv + page_size - 1) // page_size
    total_pages = 2 * batch * pages
    query = torch.randn((sum(q_lens), hq, 128), generator=generator).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    key = torch.randn((total_pages, hkv, page_size, 128), generator=generator).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    value = torch.randn((total_pages, hkv, page_size, 128), generator=generator).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    order = torch.randperm(total_pages, generator=generator).to(torch.int32)
    tables_k = order[: batch * pages].view(batch, pages).to(device)
    tables_v = (
        tables_k if shared else order[batch * pages :].view(batch, pages).to(device)
    )
    block_tables = tables_k if shared else torch.stack((tables_k, tables_v), dim=1)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    cum_q = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.tensor(q_lens).cumsum(0).int())
    ).to(device)
    cum_kv = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.tensor(kv_lens).cumsum(0).int())
    ).to(device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    counter = torch.zeros(
        get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch, hq, get_device_sm_count(device)
        ),
        dtype=torch.uint8,
        device=device,
    )
    bmm1, bmm2 = _QUANT_SCALES
    out = torch.empty_like(query)

    def run():
        return prefill.trtllm_batch_context_with_kv_cache(
            query,
            (key, value),
            workspace,
            block_tables,
            seq_lens,
            max_q,
            max_kv,
            bmm1,
            bmm2,
            batch,
            cum_q,
            cum_kv,
            out=out,
            kv_layout="HND",
            uses_shared_paged_kv_idx=shared,
            multi_ctas_kv_counter_buffer=counter,
            backend="cake",
        )

    run()
    torch.cuda.synchronize()
    assert served_programs[-1] == program
    assert not counter.any()
    expected = (
        _reference_context(
            query, key, value, tables_k, tables_v, q_lens, kv_lens, scale=bmm1
        )
        * bmm2
    )
    torch.testing.assert_close(out.float(), expected, atol=0.1, rtol=0.1)
    first = out.clone()
    out.fill_(float("nan"))
    run()
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(first))
    assert not counter.any()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
    torch.cuda.synchronize()
    shorter = tuple(
        max(q_len, length - length // 4)
        for q_len, length in zip(q_lens, kv_lens, strict=True)
    )
    seq_lens.copy_(torch.tensor(shorter, dtype=torch.int32, device=device))
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    expected_short = (
        _reference_context(
            query, key, value, tables_k, tables_v, q_lens, shorter, scale=bmm1
        )
        * bmm2
    )
    torch.testing.assert_close(out.float(), expected_short, atol=0.1, rtol=0.1)
    assert not counter.any()


@_GPU
def test_cake_context_split_choice_of_a_smaller_pack_is_bitwise_the_same(
    served_programs, monkeypatch
) -> None:
    """Pack 8's split over the tiles of pack 4 (the split choice with the
    shipped costs) writes the same output bytes as pack 8's split-KV program
    (the choice where the tile rows do not scale the costs): every query row
    meets the same KV units, merged in the same order."""

    device = torch.device("cuda")
    target = cake_api._cake_fmha_target(device)
    sm_count = get_device_sm_count(device)
    q_len, kv_len, hq, hkv, page_size = 32, 16384, 32, 4, 64
    generator = torch.Generator(device="cpu").manual_seed(20261027)
    pages = kv_len // page_size
    query = torch.randn((q_len, hq, 128), generator=generator).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    key = torch.randn((pages, hkv, page_size, 128), generator=generator).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    value = torch.randn((pages, hkv, page_size, 128), generator=generator).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    tables = torch.randperm(pages, generator=generator).to(torch.int32)
    tables = tables.view(1, pages).to(device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    counter = torch.zeros(
        get_trtllm_gen_multi_ctas_kv_counter_bytes(1, hq, sm_count),
        dtype=torch.uint8,
        device=device,
    )
    bmm1, bmm2 = _QUANT_SCALES
    cum_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    cum_kv = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
    choice = cake_api._context_fp8_split_choice
    own = cake_api._context_fp8_split_plan(target, 1, q_len, kv_len, hq, 8, sm_count)
    shipped = cake_api._CONTEXT_FP8_SPLIT_ROWS[target]
    outs = {}
    try:
        for rows in ((1.0, 1.0), shipped):
            monkeypatch.setitem(cake_api._CONTEXT_FP8_SPLIT_ROWS, target, rows)
            _clear_split_caches()
            pack, plan = choice(target, 1, q_len, kv_len, hq, 8, sm_count)
            expected = (8, own) if rows == (1.0, 1.0) else (4, (8, *own[1:]))
            assert (pack, plan) == expected
            out = torch.full_like(query, float("nan"))
            prefill.trtllm_batch_context_with_kv_cache(
                query,
                (key, value),
                workspace,
                tables,
                seq_lens,
                q_len,
                kv_len,
                bmm1,
                bmm2,
                1,
                cum_q,
                cum_kv,
                out=out,
                kv_layout="HND",
                uses_shared_paged_kv_idx=True,
                multi_ctas_kv_counter_buffer=counter,
                backend="cake",
            )
            torch.cuda.synchronize()
            assert served_programs[-1] == f"RUNTIME_SK_PACK{pack}_PAGE64"
            assert not counter.any()
            outs[pack] = out
    finally:
        _clear_split_caches()
    assert torch.equal(_bits(outs[4]), _bits(outs[8]))


# (q_lens, kv_lens, heads, page_size, shared page table, one-stage pack): calls
# whose tiles hold at most one query stage at the pack.
_ONE_STAGE_CASES = [
    # 2 requests of 128 query tokens over 1500 keys, heads 32/4 (the route's
    # own choice on both targets)
    ((128, 128), (1500, 1500), (32, 4), 16, True, 1),
    # ragged, separate K / V tables, a request shorter than its stage
    ((128, 97, 5), (1500, 700, 300), (32, 4), 64, False, 1),
    # pack 8: at most 16 query tokens per request
    ((16, 9, 1), (1024, 300, 129), (32, 4), 16, True, 8),
    # pack 4 under a runtime pack of 8, an empty request among others (16-token
    # pages: no exact route)
    ((32, 0, 20), (2048, 0, 200), (32, 4), 16, True, 4),
    # pack 2, GQA group 2
    ((64, 64), (900, 640), (32, 16), 16, True, 2),
]


@_GPU
@pytest.mark.parametrize(
    ("q_lens", "kv_lens", "heads", "page_size", "shared", "pack"),
    _ONE_STAGE_CASES,
    ids=[
        f"b{len(c[0])}_q{max(c[0])}_k{max(c[1])}_h{c[2][0]}x{c[2][1]}_p{c[3]}_g{c[5]}"
        for c in _ONE_STAGE_CASES
    ],
)
def test_cake_context_one_stage_programs_match_the_two_stage_programs_bitwise(
    monkeypatch, served_programs, q_lens, kv_lens, heads, page_size, shared, pack
) -> None:
    """The one-stage program of the pack and page size (the one-stage choice
    set to the pack) writes the same output bytes as the whole-tile program
    of two stages of the call's runtime pack: every query row meets the same
    KV blocks in the same order.  Its output is within the FP8 contract
    tolerance of an FP32 reference, bitwise the same run to run, for host and
    device scales, and the ticket reads zero after each launch."""

    device = torch.device("cuda")
    target = cake_api._cake_fmha_target(device)
    suffix = "_PAGE64" if page_size == 64 else ""
    batch = len(q_lens)
    hq, hkv = heads
    max_q, max_kv = max(q_lens), max(kv_lens)
    runtime_pack = cake_api._context_fp8_runtime_pack(
        cake_api._context_pack_g(max_q, max_kv, hq, hkv)
    )
    assert runtime_pack % pack == 0 and max_q <= 128 // pack
    generator = torch.Generator(device="cpu").manual_seed(20261029 + max_kv)
    pages = (max_kv + page_size - 1) // page_size
    total_pages = 2 * batch * pages
    query = torch.randn((sum(q_lens), hq, 128), generator=generator).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    key = torch.randn((total_pages, hkv, page_size, 128), generator=generator).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    value = torch.randn((total_pages, hkv, page_size, 128), generator=generator).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    order = torch.randperm(total_pages, generator=generator).to(torch.int32)
    tables_k = order[: batch * pages].view(batch, pages).to(device)
    tables_v = (
        tables_k if shared else order[batch * pages :].view(batch, pages).to(device)
    )
    block_tables = tables_k if shared else torch.stack((tables_k, tables_v), dim=1)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    cum_q = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.tensor(q_lens).cumsum(0).int())
    ).to(device)
    cum_kv = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.tensor(kv_lens).cumsum(0).int())
    ).to(device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    counter = torch.zeros(
        get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch, hq, get_device_sm_count(device)
        ),
        dtype=torch.uint8,
        device=device,
    )

    bmm1, bmm2 = _QUANT_SCALES

    def run(device_scales=False):
        scales = (bmm1, bmm2)
        if device_scales:
            scales = tuple(
                torch.tensor(scale, dtype=torch.float32, device=device)
                for scale in scales
            )
        out = torch.full_like(query, float("nan"))
        prefill.trtllm_batch_context_with_kv_cache(
            query,
            (key, value),
            workspace,
            block_tables,
            seq_lens,
            max_q,
            max_kv,
            *scales,
            batch,
            cum_q,
            cum_kv,
            out=out,
            kv_layout="HND",
            uses_shared_paged_kv_idx=shared,
            multi_ctas_kv_counter_buffer=counter,
            backend="cake",
        )
        torch.cuda.synchronize()
        assert not counter.any()
        return out

    one_stage = f"RUNTIME_PACK{pack}{suffix}{_RUNTIME_SUFFIX[target]}_Q1"
    with monkeypatch.context() as patch:
        patch.setattr(cake_api, "_context_fp8_one_stage_choice", lambda *args: pack)
        out = run()
        assert served_programs[-1] == one_stage
        again = run()
        device_out = run(device_scales=True)
        assert served_programs[-2:] == [one_stage, one_stage]
    assert torch.equal(_bits(again), _bits(out))
    assert torch.equal(_bits(device_out), _bits(out))
    expected = (
        _reference_context(
            query, key, value, tables_k, tables_v, q_lens, kv_lens, scale=bmm1
        )
        * bmm2
    )
    torch.testing.assert_close(out.float(), expected, atol=0.1, rtol=0.1)
    _whole_tile_routes(monkeypatch)
    two_stage = run()
    assert served_programs[-1] in _RUNTIME_PACK, served_programs[-1]
    assert _RUNTIME_PACK[served_programs[-1]] == runtime_pack
    assert torch.equal(_bits(two_stage), _bits(out)), (
        f"max abs difference {(two_stage.float() - out.float()).abs().max().item()}"
    )


@_GPU
def test_cake_context_one_stage_route_on_the_device(served_programs) -> None:
    """On the device's SM count, 2 requests of 128 query tokens over 1500
    keys, heads 32/4, take pack 1's one-stage program (the one-stage choice
    over the split choice)."""

    device = torch.device("cuda")
    target = cake_api._cake_fmha_target(device)
    kwargs = _context_kwargs(
        ("context_fp8", 2, 128, 1500, 32, 4, 16, True, None), device="cuda"
    )
    route = cake_api.select_cake_fmha_context_route(device, **kwargs)
    assert route.program == f"RUNTIME_PACK1{_RUNTIME_SUFFIX[target]}_Q1", route
    assert (route.pack_g, route.num_m_blocks) == (1, 1)


# (q_lens, kv_lens, heads, page_size, shared page table, JIT pack): calls the
# power limit keeps on the JIT module whose query tokens fit one stage of the
# JIT pack.
_POWER_ONE_STAGE_CASES = [
    # the B256 heads 32/32 rows' form at a small batch, ragged
    ((32, 32, 17), (2048, 640, 300), (32, 32), 64, True, 1),
    # few query tokens, separate K / V tables, fewer keys than a block
    ((4, 1, 3), (1024, 129, 64), (32, 32), 64, False, 1),
    # 128 query tokens: the whole stage
    ((128, 100), (1024, 700), (32, 32), 16, True, 1),
    ((64, 33), (900, 512), (32, 16), 64, True, 2),
    ((32, 0, 20), (2048, 0, 200), (32, 8), 16, True, 4),
    ((16, 9), (1024, 300), (32, 4), 16, True, 8),
]


@_GPU
@pytest.mark.parametrize(
    ("q_lens", "kv_lens", "heads", "page_size", "shared", "pack"),
    _POWER_ONE_STAGE_CASES,
    ids=[
        f"b{len(c[0])}_q{max(c[0])}_k{max(c[1])}_h{c[2][0]}x{c[2][1]}_p{c[3]}_g{c[5]}"
        for c in _POWER_ONE_STAGE_CASES
    ],
)
def test_cake_context_power_one_stage_programs_match_the_jit_module_bitwise(
    monkeypatch, served_programs, q_lens, kv_lens, heads, page_size, shared, pack
) -> None:
    """A call the power limit keeps on the JIT module (the predicate set to
    keep it, the target's power one-stage route on) takes the one-stage
    program of the JIT pack, whose output bytes are the JIT module's (every
    query row meets the same KV blocks in the same order), within the FP8
    contract tolerance of an FP32 reference, the same run to run and for
    device scales; the ticket reads zero after each launch."""

    device = torch.device("cuda")
    target = cake_api._cake_fmha_target(device)
    suffix = "_PAGE64" if page_size == 64 else ""
    batch = len(q_lens)
    hq, hkv = heads
    max_q, max_kv = max(q_lens), max(kv_lens)
    assert cake_api._context_pack_g(max_q, max_kv, hq, hkv) == pack
    assert cake_api._context_fp8_runtime_pack(pack) == pack and max_q <= 128 // pack
    monkeypatch.setattr(cake_api, "_context_fp8_runtime_serves", lambda *args: False)
    monkeypatch.setattr(
        cake_api, "_CONTEXT_FP8_POWER_ONE_STAGE_TARGETS", frozenset({target})
    )
    generator = torch.Generator(device="cpu").manual_seed(20261030 + max_kv)
    pages = (max_kv + page_size - 1) // page_size
    total_pages = 2 * batch * pages
    query = torch.randn((sum(q_lens), hq, 128), generator=generator).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    key = torch.randn((total_pages, hkv, page_size, 128), generator=generator).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    value = torch.randn((total_pages, hkv, page_size, 128), generator=generator).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    order = torch.randperm(total_pages, generator=generator).to(torch.int32)
    tables_k = order[: batch * pages].view(batch, pages).to(device)
    tables_v = (
        tables_k if shared else order[batch * pages :].view(batch, pages).to(device)
    )
    block_tables = tables_k if shared else torch.stack((tables_k, tables_v), dim=1)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    cum_q = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.tensor(q_lens).cumsum(0).int())
    ).to(device)
    cum_kv = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.tensor(kv_lens).cumsum(0).int())
    ).to(device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    counter = torch.zeros(
        get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch, hq, get_device_sm_count(device)
        ),
        dtype=torch.uint8,
        device=device,
    )

    bmm1, bmm2 = _QUANT_SCALES

    def run(device_scales=False):
        scales = (bmm1, bmm2)
        if device_scales:
            scales = tuple(
                torch.tensor(scale, dtype=torch.float32, device=device)
                for scale in scales
            )
        out = torch.full_like(query, float("nan"))
        prefill.trtllm_batch_context_with_kv_cache(
            query,
            (key, value),
            workspace,
            block_tables,
            seq_lens,
            max_q,
            max_kv,
            *scales,
            batch,
            cum_q,
            cum_kv,
            out=out,
            kv_layout="HND",
            uses_shared_paged_kv_idx=shared,
            multi_ctas_kv_counter_buffer=counter,
            backend="cake",
        )
        torch.cuda.synchronize()
        assert not counter.any()
        return out

    one_stage = f"RUNTIME_PACK{pack}{suffix}{_RUNTIME_SUFFIX[target]}_Q1"
    out = run()
    assert served_programs[-1] == one_stage
    again = run()
    device_out = run(device_scales=True)
    assert served_programs[-2:] == [one_stage, one_stage]
    assert torch.equal(_bits(again), _bits(out))
    assert torch.equal(_bits(device_out), _bits(out))
    expected = (
        _reference_context(
            query, key, value, tables_k, tables_v, q_lens, kv_lens, scale=bmm1
        )
        * bmm2
    )
    torch.testing.assert_close(out.float(), expected, atol=0.1, rtol=0.1)
    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_POWER_ONE_STAGE_TARGETS", frozenset())
    served = len(served_programs)
    jit_out = run()
    assert len(served_programs) == served
    assert torch.equal(_bits(jit_out), _bits(out)), (
        f"max abs difference {(jit_out.float() - out.float()).abs().max().item()}"
    )
