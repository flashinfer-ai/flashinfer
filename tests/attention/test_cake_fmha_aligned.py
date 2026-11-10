"""Aligned hd128 q1 decode programs of the Cake FMHA module (``backend="cake"``).

Host-only: the module's Python mirror of the source selector against the
source selector test's expectations (hard-coded here), the pinned program
records against the generated loader, and the shape rows (ten shapes x BF16 /
FP16: the seven measured shapes plus one shape per program the measured list
leaves out) through ``select_cake_fmha_decode_route``.
GPU: every shape row through ``trtllm_batch_decode_with_kv_cache(backend="cake")``
against an FP32 reference at atol = rtol = 1e-2, the route taken, and
CUDA-graph replay (also with shorter lengths under the captured KV bound).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import flashinfer.cake_fmha as cake_api
from flashinfer.decode import trtllm_batch_decode_with_kv_cache

_TORCH_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}

# The shape rows: name -> (batch, num_kv_heads, seq_lens, the aligned cluster
# size or None for the module's present route).  The first seven are the
# measured shapes of the source kernel's shape list; the last three reach the
# programs that list leaves out (cluster 2, cluster 1, and a two-request
# cluster-8 shape whose shorter-length replay changes a length).  Both targets
# and both dtypes route alike.
SHAPE_ROWS = {
    "uniform_b1_hkv8_s512": (1, 8, (512,), 8),
    "uniform_b1_hkv8_s1024": (1, 8, (1024,), 8),
    "uniform_b4_hkv8_s4096": (4, 8, (4096,) * 4, 4),
    "factor_t32_b32_hkv1_s4096": (32, 1, (4096,) * 32, 4),
    "ragged_b2_hkv8_s1024_1153": (2, 8, (1024, 1153), 4),
    "uniform_b8_hkv8_s4096": (8, 8, (4096,) * 8, None),
    "uniform_b32_hkv8_s2048": (32, 8, (2048,) * 32, None),
    "uniform_b5_hkv8_s2048": (5, 8, (2048,) * 5, 2),
    "uniform_b32_hkv8_s4096": (32, 8, (4096,) * 32, 1),
    "ragged_b2_hkv4_s1024_777": (2, 4, (1024, 777), 8),
}


# --- The selector mirror against the source selector test ------------------

TILES = [1, 2, 4, 8, 9, 16, 24, 32, 33, 40, 48, 56, 64, 65, 96, 128, 129, 144]
TILES += [256, 511, 512, 513, 1024, 4096]
MAX_KV = [16, 128, 256, 512, 1023, 1024, 2047, 2048, 2049, 3072, 4095, 4096]
MAX_KV += [6144, 8191, 8192, 12288, 16383, 16384]
MAX_KV += [32767, 32768, 32769, 49152, 65535, 65536, 65537, 98304, 131072]

# The carve-outs' edges: (tiles, max_kv) -> program, on the probed bounds and
# one length past them; identical on sm_100a and sm_103a in the source test.
EDGES = [
    (256, 2048, None),
    (256, 2049, 1),
    (144, 2048, None),
    (512, 2048, None),
    (256, 512, None),
    (256, 511, 1),
    (40, 6144, None),
    (64, 8192, None),
    (40, 8193, 2),
    (40, 6143, 2),
    (56, 4096, None),
    (64, 6143, None),
    (55, 4096, 2),
    (16, 16384, None),
    (32, 16383, 4),
    (32, 32768, None),
    (8, 65536, None),
    (8, 65535, 8),
    (1, 65536, None),
    # The long-KV and mid-band clauses: tiles > 64 at 32768..65536 take cluster
    # 4 below 512 tiles and cluster 2 from 512 tiles above 32768; past 65536,
    # in the 64 < tiles <= 128 band under 32768 and in 8 < tiles <= 64 above
    # 32768 the present route stays; cluster 1 starts past 128 tiles under
    # 32768.
    (65, 32768, 4),
    (65, 65536, 4),
    (129, 32768, 4),
    (512, 32768, 4),
    (512, 32769, 2),
    (4096, 65536, 2),
    (65, 65537, None),
    (9, 32769, None),
    (64, 32769, None),
    (65, 32767, None),
    (128, 32767, None),
    (129, 32767, 1),
]


def _select(tiles, max_kv, dtype="bf16"):
    return cake_api.aligned_cluster_size(
        tiles, 1, max_kv, q_len=1, num_q_heads=cake_api.TILE_Q, dtype=dtype
    )


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_aligned_selector_bands_locate_exactly_one_program(dtype) -> None:
    # Raises when no band or more than one band holds.
    for tiles in TILES:
        for max_kv in MAX_KV:
            assert _select(tiles, max_kv, dtype) in (8, 4, 2, 1, None), (tiles, max_kv)


@pytest.mark.parametrize(
    "kwargs",
    [dict(q_len=2), dict(num_q_heads=cake_api.TILE_Q + 1), dict(dtype="fp8")],
)
def test_aligned_selector_other_shapes_keep_the_present_route(kwargs) -> None:
    base = dict(q_len=1, num_q_heads=cake_api.TILE_Q, dtype="bf16")
    base.update(kwargs)
    assert cake_api.aligned_cluster_size(1, 1, 1024, **base) is None


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_aligned_selector_shape_rows_route_as_pinned(row, dtype) -> None:
    batch, num_kv_heads, seq_lens, program = SHAPE_ROWS[row]
    got = cake_api.aligned_cluster_size(
        batch,
        num_kv_heads,
        max(seq_lens),
        q_len=1,
        num_q_heads=cake_api.TILE_Q * num_kv_heads,
        dtype=dtype,
    )
    assert got == program


@pytest.mark.parametrize("case", EDGES, ids=lambda c: f"t{c[0]}-kv{c[1]}")
def test_aligned_selector_carve_out_edges(case) -> None:
    tiles, max_kv, program = case
    assert _select(tiles, max_kv) == program


# --- The shape rows through the module's route selection (host-only) --------


def _decode_kwargs(*, batch_size, num_kv_heads, seq_lens, dtype, stacked_kv=False):
    """Host-side ``select_cake_fmha_decode_route`` inputs of one shape row.

    A four-page pool is enough: the selector reads shapes, strides and dtypes.
    ``stacked_kv`` hands the K / V views of one ``[pages, 2, ...]`` tensor, the
    layout the aligned programs cannot address.
    """

    sm_count = 148
    num_q_heads = 8 * num_kv_heads
    query = torch.empty((batch_size, num_q_heads, 128), dtype=dtype)
    if stacked_kv:
        key, value = torch.empty((4, 2, num_kv_heads, 16, 128), dtype=dtype).unbind(1)
    else:
        key = torch.empty((4, num_kv_heads, 16, 128), dtype=dtype)
        value = torch.empty_like(key)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    max_seq_len = int(seq_lens_t.max())
    return dict(
        query=query,
        key_cache=key,
        value_cache=value,
        out=torch.empty_like(query),
        workspace_buffer=torch.empty(
            cake_api.cake_fmha_balanced_workspace_bytes(sm_count, 1), dtype=torch.uint8
        ),
        block_tables=torch.zeros(
            (batch_size, (max_seq_len + 15) // 16), dtype=torch.int32
        ),
        seq_lens=seq_lens_t,
        batch_size=batch_size,
        q_len=1,
        max_seq_len=max_seq_len,
        window_left=-1,
        bmm1_scale=128**-0.5,
        bmm2_scale=1.0,
        o_scale=1.0,
        sinks=None,
        kv_layout="HND",
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=None,
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        enable_block_sparse_attention=False,
        multi_ctas_kv_counter_buffer=torch.zeros(
            cake_api.cake_fmha_balanced_counter_bytes(sm_count), dtype=torch.uint8
        ),
    )


def _patch_host_only(monkeypatch, target):
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    monkeypatch.setattr(cake_api, "_smallm_sm_count", lambda device: 148)
    monkeypatch.setattr(
        cake_api,
        "_is_cake_fmha_decode_native_bf16_available",
        lambda *args, **kwargs: True,
    )


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_cake_fmha_aligned_route_serves_shape_rows_as_pinned(
    monkeypatch, target, dtype_name, row
) -> None:
    _patch_host_only(monkeypatch, target)
    batch_size, num_kv_heads, seq_lens, cluster = SHAPE_ROWS[row]
    kwargs = _decode_kwargs(
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        dtype=_TORCH_DTYPE[dtype_name],
    )
    route = cake_api.select_cake_fmha_decode_route(kwargs["query"].device, **kwargs)
    assert route is not None
    if cluster is None:
        # The module's present route below the balanced BF16 band / the
        # balanced FP16 route.
        present = (
            "decode_native_bf16" if dtype_name == "bf16" else "decode_balanced_fp16"
        )
        assert route.component == present
        assert route.aligned_program is None
        return
    assert route == cake_api.CakeFmhaDecodeRoute(
        target=target,
        batch_size=batch_size,
        q_len=1,
        num_q_heads=8 * num_kv_heads,
        num_kv_heads=num_kv_heads,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=(max(seq_lens) + 127) // 128 <= 9,
        component=f"decode_aligned_{dtype_name}",
        page_size=16,
        aligned_program=f"ALIGNED_C{cluster}_{dtype_name.upper()}",
    )
    assert cake_api.cake_fmha_route_is_optimized(route)
    assert cake_api._route_components(route) == (route.aligned_program,)


@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
def test_cake_fmha_aligned_route_fails_closed_outside_its_contract(
    monkeypatch, dtype_name
) -> None:
    _patch_host_only(monkeypatch, "sm100a")
    dtype = _TORCH_DTYPE[dtype_name]
    kwargs = _decode_kwargs(batch_size=1, num_kv_heads=8, seq_lens=(512,), dtype=dtype)
    route = cake_api.select_cake_fmha_decode_route(kwargs["query"].device, **kwargs)
    assert route is not None and route.component == f"decode_aligned_{dtype_name}"

    stacked = _decode_kwargs(
        batch_size=1, num_kv_heads=8, seq_lens=(512,), dtype=dtype, stacked_kv=True
    )
    assert not stacked["key_cache"].is_contiguous()
    for mutation in (
        {"sinks": torch.zeros(64, dtype=torch.float32)},
        {"window_left": 128},
        {"bmm1_scale": torch.ones(1, dtype=torch.float32)},
        {"bmm2_scale": 0.5},
        {"bmm2_scale": torch.ones(1, dtype=torch.float32)},
        {"o_scale": 0.5},
        {"kv_layout": "NHD"},
        {"lse": torch.empty((1, 64), dtype=torch.float32)},
        {"block_tables": torch.zeros((1, 32), dtype=torch.uint32)},
        {"seq_lens": torch.tensor([512], dtype=torch.uint32)},
        {"key_cache": stacked["key_cache"], "value_cache": stacked["value_cache"]},
        {"value_cache": stacked["value_cache"]},
        {"query": torch.empty((1, 64, 256), dtype=dtype)[..., :128]},
        {
            "key_cache": torch.empty((4, 8, 32, 128), dtype=dtype),
            "value_cache": torch.empty((4, 8, 32, 128), dtype=dtype),
        },
        {
            "uses_shared_paged_kv_idx": False,
            "block_tables": torch.zeros((1, 2, 32), dtype=torch.int32),
        },
        {"key_block_scales": torch.ones((4, 8, 16), dtype=torch.float32)},
        {
            "q_len": 2,
            "query": torch.empty((2, 64, 128), dtype=dtype),
            "out": torch.empty((2, 64, 128), dtype=dtype),
        },
        {
            "query": torch.empty((1, 32, 128), dtype=dtype),
            "out": torch.empty((1, 32, 128), dtype=dtype),
        },
    ):
        mutated = {**kwargs, **mutation}
        selected = cake_api.select_cake_fmha_decode_route(
            mutated["query"].device, **mutated
        )
        assert selected is None or selected.component not in (
            "decode_aligned_bf16",
            "decode_aligned_fp16",
        ), mutation
        if selected is not None:
            assert selected.aligned_program is None, mutation


def test_cake_fmha_aligned_program_table_and_launch_grid() -> None:
    """The identifier table pins one record per (cluster size, dtype), and the
    launch grid (the parity lane's host grid) spans the program's cluster
    along z."""

    table = cake_api._ALIGNED_PROGRAMS
    assert sorted((cluster, dtype) for cluster, dtype, _ in table.values()) == sorted(
        (cluster, dtype) for cluster in (8, 4, 2, 1) for dtype in ("bf16", "fp16")
    )
    names = [name for _, _, name in table.values()]
    assert len(set(names)) == len(names)
    for identifier, (cluster, dtype, name) in table.items():
        assert identifier == cake_api._aligned_program_id(cluster, dtype)
        assert name.startswith("cake_fmha_aligned_")
        digest = name.removeprefix("cake_fmha_aligned_")
        assert len(digest) == 20 and set(digest) <= set("0123456789abcdef")
        record = SimpleNamespace(cluster=(1, 1, cluster))
        assert cake_api.cake_fmha_aligned_launch_grid(record, 3, 8) == (24, 1, cluster)


def test_cake_fmha_aligned_program_table_matches_the_generated_loader() -> None:
    """Each pinned record names the generated loader's program of the same
    identifier (the load-time checks, host-only): name, cluster, binding
    argument order and both targets; the loader publishes exactly the pinned
    programs."""

    from flashinfer.jit import cake_fmha_aligned as loader
    from flashinfer.jit.cake_program import GRID_ARGS, CakeProgram

    table = cake_api._ALIGNED_PROGRAMS
    for identifier, (cluster, _dtype, name) in table.items():
        program = getattr(loader, identifier)
        assert isinstance(program, CakeProgram), identifier
        assert program.name == name, identifier
        assert tuple(program.cluster) == (1, 1, cluster), identifier
        assert tuple(program.args) == (*cake_api._ALIGNED_HOST_PARAMS, *GRID_ARGS)
        assert {"sm_100a", "sm_103a"} <= set(program.targets), identifier
    assert sorted(program.name for program in loader.PROGRAMS) == sorted(
        name for _, _, name in table.values()
    )


# --- GPU: the shape rows through backend="cake" -----------------------------


def _aligned_case(device, *, num_kv_heads, seq_lens, seed, dtype):
    """Query ``[batch, 8 x num_kv_heads, 128]``, separate contiguous K / V pools
    ``[pages, num_kv_heads, 16, 128]`` (the layout the aligned programs read in
    place), a shuffled page table and the device lengths."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = len(seq_lens)
    num_q_heads = 8 * num_kv_heads
    pages_per_req = [(s + 15) // 16 for s in seq_lens]
    total_pages = sum(pages_per_req)
    query = torch.randn((batch, num_q_heads, 128), generator=generator).to(
        device=device, dtype=dtype
    )
    key = torch.randn((total_pages, num_kv_heads, 16, 128), generator=generator).to(
        device=device, dtype=dtype
    )
    value = torch.randn((total_pages, num_kv_heads, 16, 128), generator=generator).to(
        device=device, dtype=dtype
    )
    block_tables = torch.zeros((batch, max(pages_per_req)), dtype=torch.int32)
    order = torch.randperm(total_pages, generator=generator)
    cursor = 0
    for b, n in enumerate(pages_per_req):
        block_tables[b, :n] = order[cursor : cursor + n]
        cursor += n
    return (
        query,
        key,
        value,
        block_tables.to(device),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
    )


def _reference_decode(query, key, value, block_tables, seq_lens, *, scale):
    """FP32 GQA decode over the paged HND pools: request ``b`` attends its first
    ``seq_lens[b]`` tokens, query head ``h`` reads KV head ``h // 8``."""

    batch = int(seq_lens.numel())
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    group = num_q_heads // num_kv_heads
    out = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    for b in range(batch):
        length = int(seq_lens[b])
        pages = block_tables[b, : (length + 15) // 16]
        k = key[pages].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, 128)[:, :length]
        v = value[pages].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, 128)[:, :length]
        q = query[b].float().view(num_kv_heads, group, 128)
        logits = torch.einsum("hgd,hnd->hgn", q, k.float()) * scale
        probs = torch.softmax(logits, dim=-1)
        out[b] = torch.einsum("hgn,hnd->hgd", probs, v.float()).reshape(
            num_q_heads, 128
        )
    return out


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)
@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_cake_decode_aligned_shape_rows_match_reference_and_replay(
    monkeypatch, row, dtype_name
) -> None:
    device = torch.device("cuda")
    batch_size, num_kv_heads, seq_lens, cluster = SHAPE_ROWS[row]
    query, key, value, block_tables, seq_lens_t = _aligned_case(
        device,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        seed=20261022 + len(seq_lens),
        dtype=_TORCH_DTYPE[dtype_name],
    )
    scale = 128**-0.5
    max_seq_len = max(seq_lens)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    # Record the route the dispatch resolves (decode.py imports the resolver
    # from the module at call time).
    resolved = []
    resolve = cake_api._resolve_cake_fmha_decode_module

    def spy(resolve_device, route):
        resolved.append(route)
        return resolve(resolve_device, route)

    monkeypatch.setattr(cake_api, "_resolve_cake_fmha_decode_module", spy)

    def run():
        return trtllm_batch_decode_with_kv_cache(
            query,
            (key, value),
            workspace,
            block_tables,
            seq_lens_t,
            max_seq_len,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            window_left=-1,
            kv_layout="HND",
            q_len_per_req=1,
            backend="cake",
        )

    actual = run()
    route = resolved[-1]
    if cluster is None:
        assert route is None or route.component not in (
            "decode_aligned_bf16",
            "decode_aligned_fp16",
        )
    else:
        assert route is not None
        assert route.component == f"decode_aligned_{dtype_name}"
        assert route.aligned_program == f"ALIGNED_C{cluster}_{dtype_name.upper()}"
    expected = _reference_decode(
        query, key, value, block_tables, seq_lens_t, scale=scale
    )
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)

    # Graph replay: the captured launch must reproduce the eager result and,
    # with shorter lengths under the same KV bound, follow the device lengths
    # without a re-plan.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = run()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed.float(), expected, atol=1e-2, rtol=1e-2)
    shorter = [max(1, s - s // 3) for s in seq_lens]
    longest = max(range(len(seq_lens)), key=seq_lens.__getitem__)
    shorter[longest] = seq_lens[longest]  # keep the captured KV bound
    seq_lens_t.copy_(torch.tensor(shorter, dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    expected_short = _reference_decode(
        query, key, value, block_tables, seq_lens_t, scale=scale
    )
    torch.testing.assert_close(replayed.float(), expected_short, atol=1e-2, rtol=1e-2)
    seq_lens_t.copy_(torch.tensor(seq_lens, dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed.float(), expected, atol=1e-2, rtol=1e-2)
