"""Generated paged context programs of the Cake FMHA module (``backend="cake"``).

Host-only: the pinned program table against the generated loaders, the launch
grid, the shape rows and their widened classes (any batch, every query length
of the same m-block count) through ``select_cake_fmha_context_route`` on both
targets (host and device scales), the routes that keep the components' JIT
modules and the page tables the launch passes (views of the caller's storage).
GPU: the BF16 programs (the one-tile ones at uniform query lengths, the
persistent one at ragged query lengths; ragged KV lengths, CUDA graph replay
with new lengths) against an FP32 reference; the FP8 / NVFP4 programs (host
and device scales) through the shared prefill test against its reference; and
every program bitwise against the component's JIT route on its row and its
widened class, plus the invariances a launch must keep bitwise (NHD and HND
caches, shared and separate page tables, a column-sliced page table, graph
replay, device and host scales).  Each case checks the program that served it.
"""

from __future__ import annotations

import contextlib
import importlib
import math
from types import SimpleNamespace

import pytest
import torch

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
        {"sm100a": "B256", "sm103a": "B256"},
    ),
    "nvfp4_b4_q511_k2047": (
        "context_nvfp4", 4, 511, 2047, 32, 4, 16, True,
        {"sm100a": "Q511_SM100A", "sm103a": "Q511_SM103A"},
    ),
}  # fmt: skip
_DEVICE_SCALE_PROGRAM = {
    "B256": "B256_SCALE_PTR",
    "Q511_SM100A": "Q511_SCALE_PTR_SM100A",
    "Q511_SM103A": "Q511_SCALE_PTR_SM103A",
}


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
        assert tuple(program.args) == (
            *cake_api._CONTEXT_HOST_PARAMS[component],
            *GRID_ARGS,
        ), identifier
        assert tuple(program.block) == (512, 1, 1), identifier
        assert program.use_pdl is False and program.cooperative is False
        assert one_tile == (component == "context_bf16" and num_m_blocks in (11, 6))
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


# --- The shape rows through route selection (host-only) ----------------------


def _context_kwargs(row, *, device_scales=False, device="cpu", batch=None, q_len=None):
    """Host-side ``select_cake_fmha_context_route`` inputs of one shape row
    (``batch`` / ``q_len`` override the row's): empty tensors of the row's
    shapes (the selector reads shapes, strides and dtypes only)."""

    component, row_batch, row_q_len, kv_len, hq, hkv, page, shared, _ = SHAPE_ROWS[row]
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
    device_route = cake_api.select_cake_fmha_context_route(
        torch.device("meta"), **_context_kwargs(row, device_scales=True, device="meta")
    )
    assert device_route is not None
    assert device_route.device_scales is True
    assert device_route.program == _DEVICE_SCALE_PROGRAM[program]
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
        expected = _DEVICE_SCALE_PROGRAM[program] if device_scales else program
        assert route.program == expected, (name, route)


@pytest.mark.parametrize(
    ("row", "mutation"),
    [
        ("bf16_b128_q64_k2048", {"is_causal": False}),
        ("bf16_b128_q64_k2048", {"lse": "lse"}),
        ("bf16_b128_q64_k2048", {"max_q_len": 32}),
        ("bf16_b128_q64_k2048", {"max_q_len": 65}),
        ("bf16_b1_q8192_k8192", {"max_q_len": 8160}),
        ("nvfp4_b4_q511_k2047", {"max_q_len": 480}),
        ("fp8_b256_q32_k1024", {"max_q_len": 33}),
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


# --- The page tables a launch passes (host-only) -----------------------------


def _launch_values(monkeypatch, block_tables, *, shared, batch=3):
    """The argument values a stand-in ``B128`` program record receives from
    ``cake_paged_attention_context`` on CPU tensors, by name."""

    component, identifier = "context_bf16", "B128"
    name, num_m_blocks, _ = cake_api._CONTEXT_PROGRAMS[(component, identifier)]
    args = (*cake_api._CONTEXT_HOST_PARAMS[component], "grid_x", "grid_y", "grid_z")
    calls = []
    record = SimpleNamespace(
        name=name, args=args, entry=lambda target: lambda *values: calls.append(values)
    )
    route = cake_api.CakeFmhaContextRoute(
        target="sm100a",
        component=component,
        num_m_blocks=num_m_blocks,
        num_q_heads=32,
        num_kv_heads=4,
        pack_g=8,
        page_size=16,
        l2_swizzle=1,
        is_causal=True,
        return_lse=False,
        enable_sink=False,
        program=identifier,
    )
    module = cake_api.CakeFmhaContextProgramModule(route, record, "sm_100a")
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(
        cake_api, "tvm_ffi", SimpleNamespace(use_torch_stream=contextlib.nullcontext)
    )
    query = torch.zeros((batch * 64, 32, 128), dtype=torch.bfloat16)
    key = torch.zeros((8, 4, 16, 128), dtype=torch.bfloat16)
    module.cake_paged_attention_context(
        out=torch.empty_like(query),
        out_scale_factor=None,
        query=query,
        key_cache=key,
        value_cache=torch.empty_like(key),
        workspace_buffer=torch.zeros(64, dtype=torch.uint8),
        multi_ctas_kv_counter_buffer=torch.zeros(16, dtype=torch.uint8),
        block_tables=block_tables,
        seq_lens=torch.full((batch,), 64, dtype=torch.int32),
        max_q_len=64,
        max_kv_len=64,
        bmm1_scale=0.125,
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        o_sf_vec_size=0,
        o_sf_start_index=0,
        batch_size=batch,
        window_left=-1,
        cum_seq_lens_q=torch.arange(0, 64 * (batch + 1), 64, dtype=torch.int32),
        cum_seq_lens_kv=torch.arange(0, 64 * (batch + 1), 64, dtype=torch.int32),
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
    (values,) = calls
    return dict(zip(args, values, strict=True))


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


def _reference_context(
    query, key, value, tables_k, tables_v, q_lens, kv_lens, *, scale
):
    """FP32 bottom-right causal GQA attention over the paged HND pools: request
    ``b`` has ``q_lens[b]`` new tokens and attends its first ``kv_lens[b]``
    cached tokens; query ``i`` sees keys ``j <= i + kv_len - q_len``."""

    num_q_heads, num_kv_heads, page = query.shape[1], key.shape[1], key.shape[2]
    group = num_q_heads // num_kv_heads
    out = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    start = 0
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
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
    ],
)  # fmt: skip
def test_cake_context_quantized_programs_match_reference(
    monkeypatch, served_programs, case, program
) -> None:
    from tests.attention.test_trtllm_gen_attention_prefill import (
        _test_trtllm_batch_prefill,
    )

    target = cake_api._cake_fmha_target(torch.device("cuda"))
    if program.startswith("Q511"):
        program = f"{program}_SM{_TARGET_ARCH[target][3:].upper()}"
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
    assert program in served_programs


# --- GPU: bitwise against the JIT routes and launch invariances -------------

# The rows the programs serve outside the one-tile classes: name -> (shape
# row, batch, max_q_len).  The one-tile classes have no JIT route to compare.
BITWISE_ROWS = {
    "bf16_b1_q8192_k8192": ("bf16_b1_q8192_k8192", 1, 8192),
    "bf16_b128_q64_k2048": ("bf16_b128_q64_k2048", 128, 64),
    "fp8_b256_q32_k1024": ("fp8_b256_q32_k1024", 256, 32),
    "nvfp4_b4_q511_k2047": ("nvfp4_b4_q511_k2047", 4, 511),
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
    without the program table) on its row and its widened class; with device
    scales the device-scale program's output is bitwise the host-scale one."""

    row, batch, max_q = BITWISE_ROWS[name]
    case = _context_case(row, batch, max_q)
    program = case.programs[cake_api._cake_fmha_target(torch.device("cuda"))]
    out = _run_context(case)
    assert served_programs == [program]
    if case.component != "context_bf16":
        device_out = _run_context(case, device_scales=True)
        assert served_programs[-1] == _DEVICE_SCALE_PROGRAM[program]
        assert torch.equal(_bits(device_out), _bits(out))
    monkeypatch.setattr(cake_api, "_CONTEXT_PROGRAM_ROUTES", {})
    served = len(served_programs)
    jit_out = _run_context(case)
    assert len(served_programs) == served
    assert torch.equal(_bits(jit_out), _bits(out)), (
        f"{name}: max abs difference "
        f"{(jit_out.float() - out.float()).abs().nan_to_num(posinf=1e9).max().item()}"
    )


@_GPU
@pytest.mark.parametrize(
    ("row", "batch", "max_q"),
    [("bf16_b128_q64_k2048", 3, 64), ("fp8_b256_q32_k1024", 3, 32)],
)
def test_cake_context_program_launch_invariances_are_bitwise(
    served_programs, row, batch, max_q
) -> None:
    """NHD and HND caches, shared and separate page tables, a column-sliced
    page table and CUDA graph replay give the program's output bit for bit."""

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
