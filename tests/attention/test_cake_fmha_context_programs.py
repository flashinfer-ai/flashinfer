"""Generated paged context programs of the Cake FMHA module (``backend="cake"``).

Host-only: the pinned program table against the generated loaders, the launch
grid, the shape rows and their widened classes (any batch, every query length
of the same m-block count) through ``select_cake_fmha_context_route`` on both
targets (host and device scales), the routes that keep the components' JIT
modules, the FP8 runtime-size route (every measured shape on both targets),
the FP8 split-KV route (its table, the split planner and the plan a launch
passes), the page tables the launch passes (views of the caller's storage)
and the sizes a runtime-size program takes per launch.
GPU: the BF16 programs (the one-tile ones at uniform query lengths, the
persistent one at ragged query lengths; ragged KV lengths, CUDA graph replay
with new lengths) against an FP32 reference; the FP8 / NVFP4 programs (host
and device scales) through the shared prefill test against its reference; and
every program bitwise against the component's JIT route on its row and its
widened class, plus the invariances a launch must keep bitwise (NHD and HND
caches, shared and separate page tables, a column-sliced page table, graph
replay, device and host scales); the split-KV programs against an FP32
reference, run to run, under graph replay and with their tile counters back
at zero.  Each case checks the program that served it.
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
        {"sm100a": "RUNTIME_PACK4_PAGE64", "sm103a": "RUNTIME_PACK4_PAGE64"},
    ),
    "fp8_b16_q32_k1024_g16_page16": (
        "context_fp8", 16, 32, 1024, 32, 2, 16, False,
        {"sm100a": "RUNTIME_PACK8", "sm103a": "RUNTIME_PACK8"},
    ),
    "fp8_b2_q256_k2048_g3_page32": (
        "context_fp8", 2, 256, 2048, 30, 10, 32, True,
        {"sm100a": "RUNTIME_PACK1", "sm103a": "RUNTIME_PACK1"},
    ),
    "fp8_b4_q128_k1024_g6_page16": (
        "context_fp8", 4, 128, 1024, 30, 5, 16, True,
        {"sm100a": "RUNTIME_PACK2", "sm103a": "RUNTIME_PACK2"},
    ),
}  # fmt: skip
# The device-scale program of a host-scale one; a runtime-size program serves
# both scale forms itself.
_DEVICE_SCALE_PROGRAM = {
    "B256": "B256_SCALE_PTR",
    "B256_KV4": "B256_SCALE_PTR_KV4",
    "Q511_SM100A": "Q511_SCALE_PTR_SM100A",
    "Q511_SM103A": "Q511_SCALE_PTR_SM103A",
}
# The exact FP8 program of a target (host scales): sm103a ships the 4-stage
# K/V ring build.
_FP8_EXACT = {"sm100a": "B256", "sm103a": "B256_KV4"}
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
    ):
        for target, routes in table.items():
            for identifier in routes.values():
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
            identifier in _RUNTIME_PACK or identifier in _SPLIT_PACK
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
        ("fp8_b256_q32_k1024", {"num_kv_heads": 8}),
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
        assert route.program == f"RUNTIME_PACK{pack}" and route.pack_g == pack
    else:
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
        assert route.program == f"RUNTIME_PACK{pack}_PAGE64", route
        assert route.num_m_blocks == -(-33 // (2 * (128 // pack)))


# --- The FP8 split-KV route (host-only) --------------------------------------

_SM_COUNT = {"sm100a": 148, "sm103a": 152}


def test_cake_fmha_context_fp8_split_routes_mirror_the_runtime_routes() -> None:
    """Every runtime-size program has the split-KV program of its pack and page
    size on the same targets, and no other split-KV program exists."""

    runtime = cake_api._CONTEXT_FP8_RUNTIME_ROUTES
    assert set(cake_api._CONTEXT_FP8_SPLIT_ROUTES) == set(runtime)
    for target, routes in runtime.items():
        assert cake_api._CONTEXT_FP8_SPLIT_ROUTES[target] == {
            key: "RUNTIME_SK" + identifier.removeprefix("RUNTIME")
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
    whole = cake_api._context_fp8_split_makespan(blocks, sm_count, 0, 1, 1, costs)
    split = cake_api._context_fp8_split_makespan(blocks, sm_count, *plan, costs)
    assert split <= whole * (1 - cake_api._CONTEXT_FP8_SPLIT_REL_MARGIN)
    assert whole - split >= cake_api._CONTEXT_FP8_SPLIT_ABS_MARGIN


# (target, batch, max_q_len, max_kv_len, num_q_heads, pack, splits): launches
# near the margins (2 % and 0.5 us) whose plan differs between the targets.
_SPLIT_MARGIN_ROWS = [
    # B200: 0.58 us (2.1 %) predicted; GB300: no split predicted faster
    ("sm100a", 2, 300, 1500, 32, 8, True),
    ("sm103a", 2, 300, 1500, 32, 8, False),
    # GB300: 0.84 us (3.0 % and 2.2 %) predicted
    ("sm103a", 2, 300, 2000, 32, 8, True),
    ("sm103a", 2, 300, 3000, 32, 8, True),
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
# sees; a smaller pack runs a plan only for a launch of one m-block.
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
    """The split choice keeps the pack's plan, or, for a launch of one
    m-block, runs a smaller shipped pack whose tiles hold every query token in
    one stage: the pack's split (``split_blocks``, ``num_splits``) over its
    tiles, or its own best split; only where the cost model of each pack's
    tile rows predicts it faster than the pack's plan (or whole tiles) by both
    margins."""

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
    assert max_q <= 2 * (128 // pack) and max_q <= 128 // chosen
    blocks = cake_api._context_fp8_tile_blocks(batch, max_q, max_kv, num_q_heads, pack)
    smaller = cake_api._context_fp8_tile_blocks(
        batch, max_q, max_kv, num_q_heads, chosen
    )
    split_tiles, split_blocks, num_splits = plan
    assert 1 <= split_tiles <= min(len(smaller), sm_count - 1)
    assert 2 <= num_splits <= cake_api._CONTEXT_FP8_SPLIT_MAX
    repack = (
        own is not None
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
        blocks, sm_count, *(own or (0, 1, 1)), costs(target, max_q, pack)
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
    assert route.program == (_FP8_EXACT[target] if page == 64 else "RUNTIME_PACK8")
    assert (route.pack_g, route.num_m_blocks) == (8, 1)


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
        pack_g=_RUNTIME_PACK.get(identifier, _SPLIT_PACK.get(identifier, 8)),
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
    """No split-KV route: the test pins the whole-tile programs (a call the
    split plan would split keeps its exact or runtime-size program)."""

    monkeypatch.setattr(cake_api, "_CONTEXT_FP8_SPLIT_ROUTES", {})


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
            served in _RUNTIME_PACK and not served.endswith("_PAGE64")
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
    served_programs, q_lens, kv_lens, heads, page_size, shared
) -> None:
    """The split-KV program of the call's pack and page size serves it; its
    output is within the FP8 contract tolerance of an FP32 reference, bitwise
    the same run to run (the split is the plan's, the merge order fixed), the
    tile counters and the ticket read zero after each launch, and a CUDA graph
    of the call serves new KV lengths under the captured maximum."""

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
