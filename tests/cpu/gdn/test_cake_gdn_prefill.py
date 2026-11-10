# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only tests of the Cake GDN non-CP prefill host route.

The route reads nothing back from the device: these tests pin the host-side
contract -- the ``cu_seqlens_cpu`` validation, the single host-known grid rule,
the int32 / int64 metadata path and the public forwarding of ``max_seqlen`` /
``cu_seqlens_cpu`` -- without a GPU.  The GPU behaviour (zero-sync proof, CUDA
Graph capture and replay, chunked state carry) lives in
``test_cake_gdn_prefill_gpu.py``.
"""

import heapq
import random

import pytest
import torch

from flashinfer import gdn_prefill
from flashinfer.jit import cake_gdn

_CPU_OFFSETS_CASES = (
    pytest.param([0, 128, 192], torch.int32, id="int32-tensor"),
    pytest.param([0, 128, 192], torch.int64, id="int64-tensor"),
    pytest.param([0, 128, 192], None, id="list"),
    pytest.param((0, 128, 192), None, id="tuple"),
    pytest.param([0, 0, 192], None, id="leading-empty-sequence"),
    pytest.param([0, 192, 192], None, id="trailing-empty-sequence"),
)


@pytest.mark.cpu_only
@pytest.mark.parametrize("values,dtype", _CPU_OFFSETS_CASES)
def test_host_offsets_accept_cpu_tensor_and_int_sequences(values, dtype) -> None:
    host_copy = values if dtype is None else torch.tensor(values, dtype=dtype)
    offsets = gdn_prefill._cake_gdn_prefill_host_offsets(
        host_copy, num_seqs=2, total_tokens=192
    )
    assert offsets == tuple(values)
    assert all(type(value) is int for value in offsets)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "host_copy,match",
    (
        pytest.param([0, 128], "num_seqs \\+ 1 = 3 entries", id="too-short"),
        pytest.param([0, 64, 128, 192], "num_seqs \\+ 1 = 3 entries", id="too-long"),
        pytest.param(
            [1, 128, 192], "monotonic cu_seqlens spanning", id="first-nonzero"
        ),
        pytest.param([0, 128, 191], "monotonic cu_seqlens spanning", id="last-short"),
        pytest.param([0, 128, 193], "monotonic cu_seqlens spanning", id="last-long"),
        pytest.param([0, 150, 100], "monotonic cu_seqlens spanning", id="decreasing"),
        pytest.param([0, True, 192], "not bools", id="bool-entry"),
        pytest.param([0, 128.0, 192], "sequence of ints", id="float-entry"),
        pytest.param(
            torch.tensor([[0, 128, 192]], dtype=torch.int32),
            "one-dimensional int32/int64 CPU tensor",
            id="rank-2-tensor",
        ),
        pytest.param(
            torch.tensor([0.0, 128.0, 192.0]),
            "one-dimensional int32/int64 CPU tensor",
            id="float-tensor",
        ),
        pytest.param(
            torch.empty(3, dtype=torch.int32, device="meta"),
            "one-dimensional int32/int64 CPU tensor",
            id="non-cpu-tensor",
        ),
    ),
)
def test_host_offsets_fail_closed_on_the_host(host_copy, match) -> None:
    # Every rejection happens before any launch; a non-CPU tensor is refused
    # rather than copied (the copy would be the device-to-host sync).
    with pytest.raises(cake_gdn.CakeGDNUnsupportedError, match=match):
        gdn_prefill._cake_gdn_prefill_host_offsets(
            host_copy, num_seqs=2, total_tokens=192
        )


def _residency_grid(total_tiles: int, sm_count: int, dvsplit: bool) -> int:
    """The residency rule as the test's own oracle (LANE_CONTRACT v1)."""

    if dvsplit or total_tiles <= 128:
        return min(sm_count, total_tiles)
    g_short, g_sm = min(128, total_tiles), min(sm_count, total_tiles)
    return g_sm if -(-total_tiles // g_sm) < -(-total_tiles // g_short) else g_short


# Ragged full-DV batches: (seq_lens, heads, sm_count) -> grid.  The comments give the
# modelled critical-path gain of one CTA per tile over the residency grid; a row routes
# (grid = total_tiles) when the gain reaches 4 %.
_RAGGED_GRID_CASES = (
    ([8192] + [1024] * 7, 32, 148, 256),  # 12.5 %: one long sequence first
    ([4096, 2048, 1024, 512, 256, 128, 64, 32], 32, 148, 256),  # 9.2 %
    ([2048] + [1024] * 7, 32, 148, 256),  # 25.0 %
    ([1280] + [1024] * 7, 32, 148, 256),  # 4.55 %: just over the margin
    ([1024] * 8, 32, 148, 128),  # equal lengths never route (-5.0 %)
    ([1024] * 7 + [8192], 32, 148, 128),  # long last: the chain is as short (-1.3 %)
    ([1024, 1000, 1024, 992, 1024, 1016, 1024, 1008], 32, 148, 128),  # -5.0 %
    ([512] * 4 + [4096] + [512] * 3, 32, 148, 128),  # -2.5 % on 148 SMs ...
    ([512] * 4 + [4096] + [512] * 3, 32, 152, 128),  # ... and on 152 ...
    ([512] * 4 + [4096] + [512] * 3, 32, 160, 256),  # ... but 13.75 % on 160
    ([8192] + [256] * 31, 32, 148, 1024),  # 1024 tiles, residency 148: 26.1 %
    ([8192] + [256] * 31, 32, 152, 1024),
    ([8192] + [256] * 31, 32, 160, 1024),
    ([256] * 32, 32, 148, 148),  # equal lengths keep the residency grid (-12.5 %)
    ([256] * 32, 32, 152, 152),
    ([256] * 32, 32, 160, 160),
    ([512] * 16, 32, 148, 128),  # -8.3 %
    ([4096] + [64] * 15, 32, 148, 512),  # 19.8 %
    ([2048] + [128] * 15, 16, 148, 256),  # Hv 16: 11.9 %
)


@pytest.mark.cpu_only
def test_grid_rule_mirrors_the_launch_contract() -> None:
    """The single host-known persistent-grid rule (LANE_CONTRACT v1 + the ragged route).

    grid_x(total_tiles, sm, dvsplit, bound, host_seq_lens=None): dvsplit or
    tiles <= 128 -> min(sm, tiles); else G = min(sm, tiles) when
    ceil(tiles / G) < ceil(tiles / 128), else 128 (ties keep 128) -- the
    residency grid.  With the host sequence lengths (``cu_seqlens_cpu``) a
    full-DV batch whose residency grid gives several tiles per CTA launches
    one CTA per tile when the host model of the persistent strided schedule
    predicts a critical path at least 4 % longer than dispatching one CTA per
    tile.  The bound is validated by the caller but does not change the grid;
    the result is always the residency grid or ``total_tiles``.
    """

    grid = gdn_prefill._cake_gdn_prefill_grid_x
    # DV-split: one CTA per tile up to the SM count, whatever the bound.
    assert grid(64, 148, True, 16384) == 64
    assert grid(256, 148, True, 16384) == 148
    assert grid(512, 212, True, 65536) == 212
    # Full-DV, few tiles: one CTA per tile.
    assert grid(128, 148, False, 65536) == 128
    assert grid(96, 160, False, 65536) == 96
    # Full-DV, more than 128 tiles, equal quantum: 128 CTAs whatever the bound.
    assert grid(256, 148, False, 512) == 128
    assert grid(200, 160, False, 1) == 128
    assert grid(130, 212, False, 448) == 130
    assert grid(256, 148, False, 513) == 128
    assert grid(256, 160, False, 16384) == 128
    assert grid(512, 148, False, 65536) == 128
    assert grid(200, 148, False, 1024) == 128
    assert grid(256, 128, False, 4096) == 128
    # Full-DV, the SM count gives a smaller per-CTA tile quantum.
    assert grid(384, 212, False, 8192) == 212  # ceil(384/212) = 2 < ceil(384/128) = 3
    assert grid(300, 160, False, 4096) == 160  # ceil(300/160) = 2 < ceil(300/128) = 3
    assert grid(1024, 152, False, 256) == 152
    assert grid(1024, 148, False, 256) == 148
    assert grid(2048, 152, False, 8192) == 152
    assert grid(129, 212, False, 65536) == 129
    assert grid(130, 148, False, 65536) == 130
    # The bound never changes the grid; without host lengths the rule is the
    # residency rule for every input; the shim's 0 < grid_x <= total_tiles
    # holds for every reachable input.
    for sm in (128, 132, 148, 152, 160, 212):
        for tiles in (1, 2, 64, 128, 129, 200, 256, 384, 512, 1000, 2048):
            for bound in (1, 64, 512, 513, 4096, 65536):
                for dvsplit in (False, True):
                    value = grid(tiles, sm, dvsplit, bound)
                    assert value == _residency_grid(tiles, sm, dvsplit)
                    assert value == grid(tiles, sm, dvsplit, bound, None)
                    assert 0 < value <= tiles
                    assert value <= max(sm, 128)

    # The ragged route: the host sequence lengths decide between the residency
    # grid and one CTA per tile (the table above, computed with the shared rule).
    for seq_lens, heads, sm, expected in _RAGGED_GRID_CASES:
        tiles = len(seq_lens) * heads
        value = grid(tiles, sm, False, max(seq_lens), tuple(seq_lens))
        assert value == expected, (seq_lens, heads, sm, value)
        assert value in (tiles, _residency_grid(tiles, sm, False))
        # The bound still does not change the grid; a list is accepted too.
        assert grid(tiles, sm, False, sum(seq_lens), list(seq_lens)) == expected
    long_first = (8192,) + (1024,) * 7
    # DV-split never routes, whatever the lengths.
    assert grid(512, 148, True, 8192, long_first) == 148
    assert grid(256, 148, True, 8192, long_first) == 148
    assert grid(64, 148, True, 8192, (8192,) + (1024,) * 3) == 64
    # At most 128 tiles: the residency grid already gives one CTA per tile.
    assert grid(128, 148, False, 8192, long_first) == 128
    assert grid(64, 148, False, 8192, long_first) == 64
    assert grid(148, 148, False, 8192, long_first) == 148
    # A length list that does not divide the tile count (or is empty) is
    # ignored: the residency grid.
    assert grid(256, 148, False, 8192, (8192, 1024, 1024)) == 128
    assert grid(256, 148, False, 8192, ()) == 128
    assert grid(1024, 148, False, 8192, long_first + (1024,)) == 148
    # Empty and negative lengths count as zero chunks.
    assert grid(256, 148, False, 1, (0,) * 8) == 128
    assert grid(256, 148, False, 1, (-5,) * 8) == grid(256, 148, False, 1, (0,) * 8)
    assert grid(256, 148, False, 8192, (8192, 0, 0, 0, 0, 0, 0, 0)) in (128, 256)

    # Invariants over a grid of ragged inputs: 0 < grid_x <= total_tiles and
    # the grid is the residency grid or one CTA per tile.
    rng = random.Random(2441)
    lengths = (0, 32, 64, 128, 512, 1024, 2048, 4096, 8192)
    for sm in (128, 132, 148, 152, 160, 212):
        for num_seqs, heads in (
            (3, 32),
            (4, 32),
            (8, 32),
            (12, 8),
            (16, 16),
            (16, 32),
            (32, 32),
        ):
            for _ in range(8):
                seq_lens = tuple(rng.choice(lengths) for _ in range(num_seqs))
                for dvsplit in (False, True):
                    tiles = num_seqs * heads * (2 if dvsplit else 1)
                    value = grid(tiles, sm, dvsplit, max(seq_lens) or 1, seq_lens)
                    assert 0 < value <= tiles
                    assert value in (_residency_grid(tiles, sm, dvsplit), tiles)
                    if dvsplit:
                        assert value == _residency_grid(tiles, sm, True)


# Test-only reference of the grid model: the per-tile form (tile costs expanded head by head,
# the strided sums as a pass over the tiles, the dispatched makespan as a per-tile heap).  The
# shipped rule evaluates the same model per sequence; the randomized test below holds the two
# to identical (strided, dispatch, grid) triples.
def _ref_tile_costs(seq_lens, heads):
    return [
        2 * (-(-max(int(n), 0) // 128)) + cake_gdn.TILE_OVERHEAD_CHUNKS
        for n in seq_lens
        for _ in range(heads)
    ]


def _ref_strided(costs, grid):
    per_cta = [0] * grid
    for tile, cost in enumerate(costs):
        per_cta[tile % grid] += cost
    return max(per_cta)


def _ref_dispatch(costs, sm_count):
    free = [0] * max(1, min(sm_count, len(costs)))
    finish = 0
    for cost in costs:
        start = heapq.heappop(free)
        end = start + cost + cake_gdn.DISPATCH_OVERHEAD_CHUNKS
        finish = max(finish, end)
        heapq.heappush(free, end)
    return finish


def _ref_grid(total_tiles, sm_count, dvsplit, seq_lens):
    grid = _residency_grid(total_tiles, sm_count, dvsplit)
    if dvsplit or seq_lens is None or grid >= total_tiles:
        return grid, None, None
    num_seqs = len(seq_lens)
    if num_seqs == 0 or total_tiles % num_seqs:
        return grid, None, None
    costs = _ref_tile_costs(seq_lens, total_tiles // num_seqs)
    strided = _ref_strided(costs, grid)
    dispatch = _ref_dispatch(costs, sm_count)
    gain = strided * (100 - cake_gdn.RAGGED_MIN_GAIN_PERCENT)
    return (total_tiles if gain >= dispatch * 100 else grid), strided, dispatch


def _random_batch(rng):
    """A batch of one of the shapes the rule meets: equal lengths, slightly ragged, one long
    sequence among short ones, many short with a few long ones, or uniform draws."""

    num_seqs = rng.randint(1, 64)
    kind = rng.randrange(6)
    if kind == 0:
        lens = [rng.randint(1, 65536)] * num_seqs
    elif kind == 1:
        base = rng.randint(1, 65536)
        lens = [max(1, base - rng.randint(0, 64)) for _ in range(num_seqs)]
    elif kind == 2:
        short = rng.randint(1, 2048)
        lens = [short] * num_seqs
        lens[rng.randrange(num_seqs)] = rng.randint(2048, 65536)
    elif kind == 3:
        lens = [rng.randint(1, 512) for _ in range(num_seqs)]
        for _ in range(rng.randint(1, 3)):
            lens[rng.randrange(num_seqs)] = rng.randint(4096, 65536)
    elif kind == 4:
        lens = [
            rng.choice((1, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384))
            for _ in range(num_seqs)
        ]
    else:
        lens = [rng.randint(1, 65536) for _ in range(num_seqs)]
    return tuple(lens)


@pytest.mark.cpu_only
def test_grid_rule_matches_the_per_tile_reference_on_random_batches() -> None:
    """The per-sequence rule (equal-length short-circuit, endpoint-sweep strided path, makespan
    bounds, group-wise dispatch simulation) and the per-tile reference give identical
    (strided, dispatch, grid) triples on 20,000 random batches."""

    rng = random.Random(20261019)
    rule = cake_gdn._cake_gdn_prefill_grid_x.__wrapped__  # the uncached decision
    simulated = 0
    for _ in range(20_000):
        heads = rng.choice((8, 16, 32))
        sm = rng.choice((132, 148, 152, 160))
        dvsplit = rng.random() < 0.05
        lens = _random_batch(rng)
        tiles = len(lens) * heads * (2 if dvsplit else 1)
        bound = max(lens)
        expected, ref_strided, ref_dispatch = _ref_grid(tiles, sm, dvsplit, lens)
        assert rule(tiles, sm, dvsplit, bound, lens) == expected, (
            lens,
            heads,
            sm,
            dvsplit,
        )
        if ref_strided is None:
            continue
        simulated += 1
        seq_costs = cake_gdn._prefill_seq_costs(lens)
        grid = _residency_grid(tiles, sm, dvsplit)
        assert (
            cake_gdn._prefill_strided_critical_path(seq_costs, heads, grid)
            == ref_strided
        ), (lens, heads, sm)
        assert (
            cake_gdn._prefill_dispatch_critical_path(seq_costs, heads, sm)
            == ref_dispatch
        ), (lens, heads, sm)
    assert simulated > 10_000


@pytest.mark.cpu_only
def test_grid_rule_is_the_only_grid_text_in_the_route() -> None:
    # The route must call the mirrored rule and nothing else decides grid_x
    # (the exporter's AST mirror test compares this function with Cake's).
    import inspect

    source = inspect.getsource(gdn_prefill._run_cake_gdn_prefill)
    assert "_cake_gdn_prefill_grid_x(" in source
    assert "max_chunks" not in source
    assert ".tolist()" not in source
    assert ".cpu()" not in source
    assert ".item()" not in source


@pytest.mark.cpu_only
def test_int32_metadata_is_passed_through_and_int64_is_cast_per_call() -> None:
    i32 = torch.tensor([0, 64, 128], dtype=torch.int32)
    assert gdn_prefill._cake_gdn_prefill_i32(i32) is i32
    i64 = torch.tensor([0, 64, 128], dtype=torch.int64)
    first = gdn_prefill._cake_gdn_prefill_i32(i64)
    second = gdn_prefill._cake_gdn_prefill_i32(i64)
    assert first.dtype == second.dtype == torch.int32
    assert torch.equal(first, i64.to(torch.int32))
    assert first is not second, "no identity-keyed cache"
    # The identity-keyed caches of the previous adapter are gone.
    for name in (
        "_cake_gdn_host_ints",
        "_CAKE_GDN_HOST_INTS",
        "_cake_gdn_i32",
        "_CAKE_GDN_I32_COPIES",
        "_cake_gdn_prefill_seq_lens",
    ):
        assert not hasattr(gdn_prefill, name), name


@pytest.mark.cpu_only
def test_launch_metadata_is_passed_as_given_only_for_metadata_i64_records() -> None:
    launch = gdn_prefill._cake_gdn_prefill_launch_metadata
    i32 = torch.tensor([0, 64, 128], dtype=torch.int32)
    i64 = torch.tensor([0, 64, 128], dtype=torch.int64)
    # A ``metadata_i64`` record reads either width in place: no copy, no cast.
    assert launch(i32, metadata_i64=True) is i32
    assert launch(i64, metadata_i64=True) is i64
    # Every other record takes int32: pass-through for int32, a cast for int64.
    assert launch(i32, metadata_i64=False) is i32
    cast = launch(i64, metadata_i64=False)
    assert cast.dtype == torch.int32
    assert torch.equal(cast, i32)
    # The route carries the contract and defaults to the int32-only ABI.
    assert cake_gdn.CakeGDNRoute._fields == (
        "route_id",
        "variant_name",
        "metadata_i64",
        "one_cta_per_tile",
    )
    assert cake_gdn.CakeGDNRoute("prefill.dvsplit", "variant").metadata_i64 is False
    # The launch grid comes from the grid rule unless the route says one CTA per tile.
    assert cake_gdn.CakeGDNRoute("prefill.dvsplit", "variant").one_cta_per_tile is False


def _public_dispatch_inputs() -> tuple[torch.Tensor, ...]:
    q = torch.zeros((2048, 2, 128), dtype=torch.float16)
    k = torch.zeros_like(q)
    v = torch.zeros((2048, 8, 128), dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 1024, 2048], dtype=torch.int32)
    return q, k, v, cu_seqlens


def _route_public_cake_non_cp(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(gdn_prefill, "_CAKE_GDN_AVAILABLE", True)
    monkeypatch.setattr(gdn_prefill, "_run_cake_gdn_prefill", fake_run)
    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _device: 148)
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _device: "test GPU")
    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", "13.0")
    monkeypatch.setattr(
        gdn_prefill, "should_use_cp_host", lambda *_args, **_kwargs: False
    )
    return captured


@pytest.mark.cpu_only
def test_public_api_forwards_cu_seqlens_cpu_and_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _route_public_cake_non_cp(monkeypatch)
    q, k, v, cu_seqlens = _public_dispatch_inputs()
    host_copy = [0, 1024, 2048]
    output = gdn_prefill.chunk_gated_delta_rule(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        use_cp=False,
        backend="cake_gdn",
        max_seqlen=1024,
        cu_seqlens_cpu=host_copy,
    )
    assert output.shape == (2048, 8, 128)
    assert captured["cu_seqlens_cpu"] is host_copy
    assert captured["max_seqlen"] == 1024
    assert captured["cu_seqlens"] is cu_seqlens


@pytest.mark.cpu_only
def test_public_api_defaults_the_bound_to_the_packed_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _route_public_cake_non_cp(monkeypatch)
    q, k, v, cu_seqlens = _public_dispatch_inputs()
    gdn_prefill.chunk_gated_delta_rule(
        q, k, v, cu_seqlens=cu_seqlens, use_cp=False, backend="cake_gdn"
    )
    assert captured["cu_seqlens_cpu"] is None
    assert captured["max_seqlen"] == 2048


@pytest.mark.cpu_only
def test_cu_seqlens_cpu_is_keyword_only() -> None:
    import inspect

    signature = inspect.signature(gdn_prefill.chunk_gated_delta_rule)
    parameter = signature.parameters["cu_seqlens_cpu"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    assert "cu_seqlens_cpu :" in gdn_prefill.chunk_gated_delta_rule.__doc__


@pytest.mark.cpu_only
def test_resolver_signature_has_no_device_derived_lengths() -> None:
    import inspect

    parameters = inspect.signature(cake_gdn.select_cake_gdn_prefill_variant).parameters
    assert "seq_lens" not in parameters
    assert "max_seq_len" in parameters


@pytest.mark.cpu_only
def test_resolver_gate_stream_twin_follows_the_routed_grid() -> None:
    """The gate-stream twin (a CTA owning at least two tiles, mean length at most
    512 tokens) follows the grid the rule launches.

    With the host lengths of a ragged short-mean batch the rule launches one
    CTA per tile, so the twin is not selected; without them (or with
    equal-length host lengths, which never route) the resolver sees the
    residency grid -- two tiles per CTA -- and picks the twin as before.
    """

    ragged = (2048,) + (128,) * 15  # 16 x Hv 16 = 256 full-DV tiles, mean 248
    equal = (256,) * 16
    common = dict(
        arch="sm_100a",
        io_dtype="bfloat16",
        state_dtype="bfloat16",
        num_seqs=16,
        total_seq_len=sum(ragged),
        max_seq_len=max(ragged),
        num_q_heads=8,
        num_k_heads=8,
        num_v_heads=16,
        use_initial_state=True,
        store_final_state=True,
        checkpoint_every_n_tokens=0,
        use_state_indices=False,
        gates_present=True,
    )
    assert cake_gdn._cake_gdn_prefill_grid_x(256, 148, False, 2048) == 128
    assert cake_gdn._cake_gdn_prefill_grid_x(256, 148, False, 2048, ragged) == 256
    assert cake_gdn._cake_gdn_prefill_grid_x(256, 148, False, 256, equal) == 128

    base = cake_gdn.select_cake_gdn_prefill_variant(**common)
    routed = cake_gdn.select_cake_gdn_prefill_variant(**common, host_seq_lens=ragged)
    unrouted = cake_gdn.select_cake_gdn_prefill_variant(
        **{**common, "total_seq_len": sum(equal), "max_seq_len": 256},
        host_seq_lens=equal,
    )
    for route in (base, routed, unrouted):
        assert route.route_id == "flashinfer.gdn_prefill.noncp.full_dv"
        assert route.variant_name.startswith(
            "prefill_flashinfer_blackwell_gdn_prefill_initial_bf16state"
        )
    assert "_gatestream_full" in base.variant_name
    assert "_gatestream_full" in unrouted.variant_name
    assert "_gatestream_full" not in routed.variant_name


_MIXED16 = (8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1, 1)
_UNEQ8 = (4096, 2048, 1024, 512, 256, 128, 64, 32)
_ONE_LONG8 = (8192,) + (1024,) * 7
# The head geometries (Hq, Hv) whose BF16-I/O FP32-state stateful key ships both a full-DV
# and a DV-split record on sm_107a.
_RAGGED_DVSPLIT_GEOMETRIES = (
    (2, 4),
    (2, 8),
    (4, 8),
    (4, 16),
    (8, 16),
    (8, 32),
    (16, 32),
    (16, 64),
)
# The contracts of the ragged DV-split route, (I/O dtype, state dtype, initial state), each
# with a final state -> the head geometries whose key ships both records on sm_107a.
_RAGGED_DVSPLIT_CONTRACT_GEOMETRIES = {
    ("bfloat16", "float32", True): _RAGGED_DVSPLIT_GEOMETRIES,
    ("bfloat16", "bfloat16", True): (
        (2, 8),
        (4, 8),
        (4, 16),
        (8, 16),
        (8, 32),
        (16, 32),
    ),
    ("float16", "float32", True): (
        (2, 8),
        (4, 16),
        (8, 32),
        (16, 16),
        (16, 32),
        (16, 64),
        (32, 32),
    ),
    ("bfloat16", "float32", False): ((16, 32), (16, 64)),
    ("float16", "float32", False): ((16, 32), (16, 64)),
}
_STEM = "flashinfer.cake_gdn.generated.prefill:flashinfer_blackwell_gdn_prefill_"
# contract -> (its full-DV schedules, its DV-split schedule)
_RAGGED_DVSPLIT_CONTRACT_SCHEDULES = {
    ("bfloat16", "float32", True): (
        (
            "initial_sm107a_i64meta",
            "initial_gatestream_full_sm107a_i64meta",
            "initial_gatestream_full_pf_sm107a_i64meta",
        ),
        "dvsplit_initial_sm107a_l7_vec2",
    ),
    ("bfloat16", "bfloat16", True): (
        (
            "initial_bf16state_sm107a_i64meta",
            "initial_bf16state_gatestream_full_sm107a_i64meta",
            "initial_bf16state_gatestream_full_pf_sm107a_i64meta",
        ),
        "dvsplit_initial_bf16state_sm107a_l7b_vec2lp",
    ),
    ("float16", "float32", True): (
        (
            "initial_f16io_sm107a_i64meta",
            "initial_f16io_gatestream_full_sm107a_i64meta",
            "initial_f16io_gatestream_full_pf_sm107a_i64meta",
        ),
        "dvsplit_initial_f16io_sm107a_l7_vec2",
    ),
    ("bfloat16", "float32", False): (("sm107a_i64meta",), "dvsplit_sm107a_l7_vec2"),
    ("float16", "float32", False): (
        ("f16io_sm107a_i64meta",),
        "dvsplit_f16io_sm107a_l7_vec2",
    ),
}


def _ragged_dvsplit_load_fits(num_o_heads, lens):
    """The rule of the ragged DV-split route, spelled out on the lengths."""

    limit = (
        cake_gdn.RAGGED_DVSPLIT_LOAD_LONGEST_FIRST
        if lens[0] == max(lens)
        else cake_gdn.RAGGED_DVSPLIT_LOAD
    )
    load = num_o_heads * (sum(lens) + cake_gdn.RAGGED_DVSPLIT_SEQ_TOKENS * len(lens))
    return load <= limit * max(lens)


def _sm107a_fp32_state_kwargs(heads, lens, *, host=True, **overrides):
    num_q_heads, num_v_heads = heads
    kwargs = dict(
        arch="sm_107a",
        io_dtype="bfloat16",
        state_dtype="float32",
        num_seqs=len(lens),
        total_seq_len=sum(lens),
        max_seq_len=max(lens),
        num_q_heads=num_q_heads,
        num_k_heads=min(num_q_heads, num_v_heads),
        num_v_heads=num_v_heads,
        use_initial_state=True,
        store_final_state=True,
        checkpoint_every_n_tokens=0,
        use_state_indices=False,
        gates_present=True,
        host_seq_lens=tuple(lens) if host else None,
    )
    kwargs.update(overrides)
    return kwargs


def _sm107a_fp32_state_route(heads, lens, *, host=True, **overrides):
    return cake_gdn.select_cake_gdn_prefill_variant(
        **_sm107a_fp32_state_kwargs(heads, lens, host=host, **overrides)
    )


@pytest.mark.cpu_only
def test_ragged_dvsplit_route_takes_a_ragged_full_dv_batch() -> None:
    """With the host lengths, a ragged sm_107a BF16/FP32-state batch over the DV-split
    tile budget runs the DV-split body, one CTA per half tile; the same batch without the
    lengths keeps full-DV."""

    routed = _sm107a_fp32_state_route((8, 16), _MIXED16)
    assert routed.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert routed.one_cta_per_tile is True
    assert "_dvsplit_initial_sm107a_l7_vec2_" in routed.variant_name
    assert routed.metadata_i64
    without_lengths = _sm107a_fp32_state_route((8, 16), _MIXED16, host=False)
    assert without_lengths.route_id == "flashinfer.gdn_prefill.noncp.full_dv"
    assert without_lengths.one_cta_per_tile is False
    assert routed == cake_gdn._ragged_dvsplit_routes()[without_lengths.variant_name]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("heads", "lens", "overrides"),
    (
        # equal lengths
        ((8, 16), (512,) * 32, {}),
        ((8, 32), (2048,) * 4, {}),
        ((8, 32), (256,) * 64, {}),
        # already DV-split under the tile budget (2 * 16 * 4 <= 212)
        ((2, 4), _MIXED16, {}),
        # other contracts (no final state, FP16 state, FP16 I/O with BF16 state) keep
        # their rule
        ((16, 32), _MIXED16, {"store_final_state": False}),
        ((16, 64), _MIXED16, {"use_initial_state": False, "store_final_state": False}),
        ((8, 16), _MIXED16, {"state_dtype": "float16"}),
        ((8, 32), _MIXED16, {"io_dtype": "float16", "state_dtype": "bfloat16"}),
        # contracts of the route at a head geometry without both records
        ((2, 4), (8192,) + (64,) * 30, {"state_dtype": "bfloat16"}),
        ((8, 16), _MIXED16, {"use_initial_state": False}),
        ((8, 16), _MIXED16, {"io_dtype": "float16"}),
        # other architectures
        ((8, 16), _MIXED16, {"arch": "sm_100a"}),
        ((8, 16), _MIXED16, {"arch": "sm_103a"}),
        # head geometries without both records
        ((16, 16), _MIXED16, {}),
        ((32, 16), _MIXED16, {}),
        # ragged but balanced: the doubled tile count costs more than the long sequence
        ((8, 16), (512,) * 31 + (520,), {}),
        ((4, 8), tuple(range(1024, 1024 + 64 * 40, 64)), {}),
        # the longest sequence last behind a full first wave of half tiles
        ((2, 8), (2048,) * 19 + (4096,), {}),
    ),
)
def test_ragged_dvsplit_route_keeps_every_other_batch_on_its_rule(
    heads, lens, overrides
) -> None:
    """The route (or the fail-closed error) is the one the batch resolves without the host
    lengths."""

    def outcome(host):
        try:
            return _sm107a_fp32_state_route(heads, lens, host=host, **overrides)
        except cake_gdn.CakeGDNUnsupportedError as error:
            return str(error)

    route = outcome(True)
    assert getattr(route, "one_cta_per_tile", False) is False
    assert route == outcome(False)


@pytest.mark.cpu_only
def test_ragged_dvsplit_routes_pair_the_two_records_of_each_key() -> None:
    """Each key of the map is an sm_107a-only full-DV record (base, gate-stream twin or its
    prefetch twin) of a stateful contract of the route, and its value is that contract's
    DV-split record with the same specializations; per contract the map covers exactly the
    head geometries that ship both."""

    routes = cake_gdn._ragged_dvsplit_routes()
    records = {r["name"]: r for r in cake_gdn._manifest()["variants"]}
    assert set(cake_gdn._RAGGED_DVSPLIT_CONTRACTS) == set(
        _RAGGED_DVSPLIT_CONTRACT_SCHEDULES
    )
    contract_of = {
        _STEM + full_dv: contract
        for contract, (schedules, _) in _RAGGED_DVSPLIT_CONTRACT_SCHEDULES.items()
        for full_dv in schedules
    }
    geometries = {contract: set() for contract in _RAGGED_DVSPLIT_CONTRACT_SCHEDULES}
    for name, route in routes.items():
        full_dv, split = records[name], records[route.variant_name]
        assert full_dv["architectures"] == ["sm_107a"]
        contract = contract_of[full_dv["source_schedule"]]
        assert (
            split["source_schedule"]
            == _STEM + _RAGGED_DVSPLIT_CONTRACT_SCHEDULES[contract][1]
        )
        assert split["specializations"] == full_dv["specializations"]
        assert route == cake_gdn.CakeGDNRoute(
            "flashinfer.gdn_prefill.noncp.dvsplit",
            split["name"],
            bool(split.get("metadata_i64", False)),
            True,
        )
        specs = full_dv["specializations"]
        assert (specs["USE_INITIAL_STATE"], specs["STORE_FINAL_STATE"]) == (
            int(contract[2]),
            1,
        )
        assert (specs["ENABLE_CHECKPOINTS"], specs["USE_STATE_INDICES"]) == (0, 0)
        assert specs["UNIT_GATES"] == 0
        num_o_heads = 1 << specs["NUM_O_HEADS_LOG2"]
        other = num_o_heads >> specs["HEAD_GROUP_LOG2"]
        geometries[contract].add(
            (num_o_heads, other) if specs["IS_GQA"] else (other, num_o_heads)
        )
    for contract, expected in _RAGGED_DVSPLIT_CONTRACT_GEOMETRIES.items():
        assert sorted(geometries[contract]) == sorted(expected), contract
    # BF16-I/O FP32 state: 8 base records, 7 gate-stream twins (none for Hq16/Hv64) and
    # their 7 prefetch twins; BF16 state: 6 + 6 + 6; FP16 I/O: 7 + 7 + 7; the two
    # contracts without an initial state: 2 base records each.
    assert len(routes) == 22 + 18 + 21 + 2 + 2


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("heads", "lens"),
    (
        # The nine routed rows of the shipped tables: the Qwen3.5 mixed16 rows and the four
        # chunked-engine rows.
        ((8, 16), _MIXED16),
        ((4, 8), _MIXED16),
        ((8, 32), _MIXED16),
        ((4, 16), _MIXED16),
        ((2, 8), _MIXED16),
        ((16, 32), _UNEQ8),
        ((16, 32), _ONE_LONG8),
        ((8, 16), _UNEQ8),
        ((8, 16), _ONE_LONG8),
    ),
)
def test_ragged_dvsplit_route_takes_the_routed_rows(heads, lens) -> None:
    assert _ragged_dvsplit_load_fits(max(heads), lens)
    route = _sm107a_fp32_state_route(heads, lens)
    assert route.one_cta_per_tile is True
    assert route.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert "_dvsplit_initial_sm107a_l7_vec2_" in route.variant_name


# The mixed16 key of each contract the route serves besides the BF16-I/O FP32-state one:
# (overrides, heads, the DV-split record's name suffix).
_RAGGED_DVSPLIT_MIXED16_KEYS = (
    ({"state_dtype": "bfloat16"}, (8, 32), "9a88f173a2ef"),
    ({"io_dtype": "float16"}, (8, 32), "5e4264fc928e"),
    ({"use_initial_state": False}, (16, 32), "5fa86f1bbd8f"),
    ({"io_dtype": "float16", "use_initial_state": False}, (16, 32), "0ba1a8f6eca6"),
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("overrides", "heads", "suffix"),
    _RAGGED_DVSPLIT_MIXED16_KEYS,
    ids=("bf16_state", "f16_io", "no_initial_state", "f16_io_no_initial_state"),
)
def test_ragged_dvsplit_route_serves_each_contract_at_its_mixed16_key(
    overrides, heads, suffix
) -> None:
    """The BF16-state, FP16-I/O and no-initial-state contracts take the route at their
    mixed16 key (16 lengths 8192..1; the longest first bounds the launch) under the same rule
    and constants; their balanced batches and the other architectures keep their route."""

    contract = (
        overrides.get("io_dtype", "bfloat16"),
        overrides.get("state_dtype", "float32"),
        overrides.get("use_initial_state", True),
    )
    assert _ragged_dvsplit_load_fits(max(heads), _MIXED16)
    routed = _sm107a_fp32_state_route(heads, _MIXED16, **overrides)
    assert routed.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert routed.one_cta_per_tile is True
    assert routed.metadata_i64
    split = _RAGGED_DVSPLIT_CONTRACT_SCHEDULES[contract][1]
    assert routed.variant_name == (
        f"prefill_flashinfer_blackwell_gdn_prefill_{split}_{suffix}"
    )
    full_dv = _sm107a_fp32_state_route(heads, _MIXED16, host=False, **overrides)
    assert full_dv.route_id == "flashinfer.gdn_prefill.noncp.full_dv"
    assert full_dv.one_cta_per_tile is False
    assert cake_gdn._ragged_dvsplit_routes()[full_dv.variant_name] == routed
    # balanced batches: equal lengths and slightly ragged ones
    for lens in ((1024,) * 16, (512,) * 31 + (520,), (2048,) * 4 + (2040,)):
        if 2 * len(lens) * max(heads) <= 212:
            continue
        assert not _ragged_dvsplit_load_fits(max(heads), lens)
        route = _sm107a_fp32_state_route(heads, lens, **overrides)
        assert route.one_cta_per_tile is False
        assert route == _sm107a_fp32_state_route(heads, lens, host=False, **overrides)
    # sm_100a / sm_103a: the host lengths change nothing

    def outcome(arch, host):
        try:
            return _sm107a_fp32_state_route(
                heads, _MIXED16, host=host, arch=arch, **overrides
            )
        except cake_gdn.CakeGDNUnsupportedError as error:
            return str(error)

    for arch in ("sm_100a", "sm_103a"):
        route = outcome(arch, True)
        assert getattr(route, "one_cta_per_tile", False) is False
        assert route == outcome(arch, False)


@pytest.mark.cpu_only
def test_ragged_dvsplit_route_is_the_twin_of_the_resolved_record(monkeypatch) -> None:
    """On random keys (head geometries, I/O and state dtypes, with or without an initial
    state, gates, architectures, with and without the host lengths) the resolver returns the
    route it returns without the ragged DV-split map, except on the contracts: an sm_107a
    ragged batch of a stateful contract of the route with the host lengths whose resolved
    full-DV record has a DV-split twin and whose load fits the rule, which runs that twin.
    This covers the unshipped-DV-split fallback (#8225) too: the batches it serves never
    reach the map."""

    rng = random.Random(20261102)
    resolve = cake_gdn.select_cake_gdn_prefill_variant.__wrapped__
    routes = cake_gdn._ragged_dvsplit_routes()
    heads_pairs = _RAGGED_DVSPLIT_GEOMETRIES + (
        (2, 6),
        (32, 32),
        (1, 3),
        (3, 9),
        (8, 24),
        (16, 48),
        (6, 2),
        (12, 4),
        (32, 16),
        (16, 16),
    )
    keys = []
    for _ in range(1_500):
        heads = rng.choice(heads_pairs)
        lens = _random_batch(rng)
        overrides = dict(
            io_dtype=rng.choice(("bfloat16", "bfloat16", "float16")),
            gates_present=rng.random() < 0.8,
            arch=rng.choice(("sm_107a",) * 6 + ("sm_100a", "sm_103a")),
        )
        if rng.random() < 0.15:
            overrides["state_dtype"] = "bfloat16"
        if rng.random() < 0.2:
            overrides["use_initial_state"] = False
        keys.append((heads, lens, overrides, rng.random() < 0.8))
    # one routed key per contract (Hq16/Hv32 ships both records in each)
    for io_dtype, state_dtype, use_initial_state in _RAGGED_DVSPLIT_CONTRACT_GEOMETRIES:
        overrides = dict(io_dtype=io_dtype, gates_present=True, arch="sm_107a")
        if state_dtype != "float32":
            overrides["state_dtype"] = state_dtype
        if not use_initial_state:
            overrides["use_initial_state"] = False
        keys.append(((16, 32), _MIXED16, overrides, True))

    def resolve_all():
        out = []
        for heads, lens, overrides, host in keys:
            try:
                out.append(
                    resolve(
                        **_sm107a_fp32_state_kwargs(heads, lens, host=host, **overrides)
                    )
                )
            except cake_gdn.CakeGDNUnsupportedError:
                out.append(None)
        return out

    with_map = resolve_all()
    monkeypatch.setattr(cake_gdn, "_ragged_dvsplit_routes", lambda: {})
    without_map = resolve_all()
    routed = 0
    routed_contracts = set()
    for (heads, lens, overrides, host), got, base in zip(
        keys, with_map, without_map, strict=True
    ):
        key = (
            overrides["io_dtype"],
            overrides.get("state_dtype", "float32"),
            overrides.get("use_initial_state", True),
        )
        contract = (
            host
            and len(set(lens)) > 1
            and overrides["arch"] == "sm_107a"
            and heads in _RAGGED_DVSPLIT_CONTRACT_GEOMETRIES.get(key, ())
            and 2 * len(lens) * max(heads) > 212
            and _ragged_dvsplit_load_fits(max(heads), lens)
        )
        if base is None:
            assert got is None
        elif contract:
            assert base.route_id == "flashinfer.gdn_prefill.noncp.full_dv"
            # Without the map a batch of at most 2048 tokens per sequence runs the key's
            # vec2-full twin; the map is keyed by the shared record it replaces.
            assert got == routes[_vec2full_shared_name(base.variant_name)]
            routed += 1
            routed_contracts.add(key)
        else:
            assert got == base
            assert got.one_cta_per_tile is False
    assert routed > 0
    assert routed_contracts == set(_RAGGED_DVSPLIT_CONTRACT_GEOMETRIES)


@pytest.mark.cpu_only
def test_ragged_dvsplit_route_needs_the_dvsplit_record_of_the_key(monkeypatch) -> None:
    """With the key's DV-split record removed from the catalog, the batch the route would
    move stays on its full-DV record (it never resolves to the missing record)."""

    import functools

    manifest = cake_gdn._manifest()
    routed = _sm107a_fp32_state_route((8, 16), _MIXED16)
    assert routed.one_cta_per_tile is True
    full_dv = _sm107a_fp32_state_route((8, 16), _MIXED16, host=False)
    stripped = dict(
        manifest,
        variants=[r for r in manifest["variants"] if r["name"] != routed.variant_name],
    )
    monkeypatch.setattr(cake_gdn, "_manifest", lambda: stripped)
    monkeypatch.setattr(
        cake_gdn,
        "_ragged_dvsplit_routes",
        functools.cache(cake_gdn._ragged_dvsplit_routes.__wrapped__),
    )
    assert full_dv.variant_name not in cake_gdn._ragged_dvsplit_routes()
    resolve = cake_gdn.select_cake_gdn_prefill_variant.__wrapped__
    route = resolve(**_sm107a_fp32_state_kwargs((8, 16), _MIXED16))
    assert route.one_cta_per_tile is False
    assert route == full_dv


@pytest.mark.cpu_only
@pytest.mark.parametrize("heads", ((6, 2), (12, 4)))
def test_ragged_dvsplit_route_needs_the_full_dv_record_of_the_key(heads) -> None:
    """The route chooses between two served records.  The Hq >= Hv key whose head ratio and
    output head count are not powers of two ships a DV-split record but no full-DV record, so
    over the DV-split tile budget it fails closed without the host lengths; with them it
    fails closed as well, whatever the lengths, instead of serving some length
    distributions on the DV-split record."""

    lens = (8192,) + (64,) * 20  # 2 * 21 * Hq half tiles > 212: full-DV
    kwargs = _sm107a_fp32_state_kwargs(heads, lens)
    specs = {
        "ENABLE_CHECKPOINTS": 0,
        "HEAD_GROUP_LOG2": -1,
        "IS_GQA": 1,
        "NUM_O_HEADS_LOG2": -1,
        "SINGLE_CHUNK_NO_STATE": 0,
        "STORE_FINAL_STATE": 1,
        "UNIT_GATES": 0,
        "USE_INITIAL_STATE": 1,
        "USE_STATE_INDICES": 0,
    }
    stem = "flashinfer_blackwell_gdn_prefill"
    for attr, ships in (
        (f"{stem}_dvsplit_initial_sm107a_l7_vec2", True),
        (f"{stem}_initial_sm107a_i64meta", False),
    ):
        assert (
            cake_gdn._has_variant(
                arch="sm_107a",
                domain="prefill",
                schedule_attr=attr,
                specializations=specs,
            )
            is ships
        ), attr
    records = {r["name"]: r for r in cake_gdn._manifest()["variants"]}
    assert all(
        records[name]["specializations"] != specs
        for name in cake_gdn._ragged_dvsplit_routes()
    )
    resolve = cake_gdn.select_cake_gdn_prefill_variant.__wrapped__
    for host in (True, False):
        with pytest.raises(cake_gdn.CakeGDNUnsupportedError):
            resolve(**dict(kwargs, host_seq_lens=lens if host else None))
    # Under the tile budget the key runs its DV-split record, with or without the lengths.
    short = lens[:8]
    routed = resolve(**_sm107a_fp32_state_kwargs(heads, short))
    assert routed.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert routed.variant_name.endswith("805fd7f7b98d")
    assert routed.one_cta_per_tile is False


@pytest.mark.cpu_only
def test_ragged_dvsplit_rule_limits_follow_the_position_of_the_longest_sequence() -> (
    None
):
    """The same lengths route with the longest sequence first (its half tiles start at once)
    and keep full-DV with it last, where the load must fit the smaller limit."""

    lens = (8192,) + (1024,) * 15  # Hq4/Hv16: load 16 * (23552 + 384 * 16) = 475136
    assert _ragged_dvsplit_load_fits(16, lens)
    assert not _ragged_dvsplit_load_fits(16, lens[1:] + lens[:1])
    first = _sm107a_fp32_state_route((4, 16), lens)
    last = _sm107a_fp32_state_route((4, 16), lens[1:] + lens[:1])
    assert first.one_cta_per_tile is True
    assert last.one_cta_per_tile is False
    assert last == _sm107a_fp32_state_route((4, 16), lens[1:] + lens[:1], host=False)
    # A long sequence anywhere routes once its load fits the smaller limit.
    lens = (64,) * 30 + (32768,) + (64,) * 30  # Hq2/Hv4
    assert _ragged_dvsplit_load_fits(4, lens) and lens[0] != max(lens)
    assert _sm107a_fp32_state_route((2, 4), lens).one_cta_per_tile is True


# The sm_107a vec2-full twins of the three shared full-DV records (``_VEC2FULL_SHARED_SCHEDULES``):
# contract overrides (on the BF16-I/O FP32-state kwargs) -> its twin head geometries.
_VEC2FULL_CONTRACTS = {
    "bf16_state": (
        {"state_dtype": "bfloat16"},
        ((2, 8), (4, 8), (4, 16), (8, 16), (8, 32), (16, 32)),
    ),
    "no_initial_state": ({"use_initial_state": False}, ((16, 32), (16, 64))),
    "f16_io_no_initial_state": (
        {"io_dtype": "float16", "use_initial_state": False},
        ((16, 32), (16, 64)),
    ),
}
_VEC2FULL_CASES = tuple(
    pytest.param(overrides, heads, id=f"{contract}-hq{heads[0]}-hv{heads[1]}")
    for contract, (overrides, geometries) in _VEC2FULL_CONTRACTS.items()
    for heads in geometries
)


def _vec2full_shared_name(name):
    """The shared record a vec2-full twin record name stands beside (other names unchanged)."""

    return name.replace("_sm107a_vec2full_i64meta_", "_sm107a_i64meta_")


def _vec2full_twin_name(name):
    return name.replace("_sm107a_i64meta_", "_sm107a_vec2full_i64meta_")


def _full_dv_seqs(heads):
    """A sequence count over the DV-split tile budget (``2 * n * max(Hq, Hv) > 212``)."""

    return 212 // (2 * max(heads)) + 1


@pytest.mark.cpu_only
@pytest.mark.parametrize(("overrides", "heads"), _VEC2FULL_CASES)
def test_vec2full_twin_follows_the_longest_sequence(overrides, heads) -> None:
    """A full-DV batch of the three contracts runs the key's vec2-full twin when its longest
    sequence spans at most 32 chunks (2048 tokens) and the shared record above that, with or
    without the host lengths (equal lengths never take the ragged DV-split route)."""

    num_seqs = _full_dv_seqs(heads)
    for length, twin in (
        (2048, True),
        (1985, True),
        (1, True),
        (2049, False),
        (4096, False),
        (8192, False),
    ):
        lens = (length,) * num_seqs
        for host in (True, False):
            route = _sm107a_fp32_state_route(heads, lens, host=host, **overrides)
            assert route.route_id == "flashinfer.gdn_prefill.noncp.full_dv", route
            assert route.one_cta_per_tile is False and route.metadata_i64 is True
            assert ("_sm107a_vec2full_i64meta_" in route.variant_name) is twin, (
                length,
                route,
            )
            record = cake_gdn._kernel_record(route.variant_name)
            schedule = record["source_schedule"].rsplit(":", 1)[-1]
            if twin:
                schedule = _vec2full_shared_name(route.variant_name)
                schedule = cake_gdn._kernel_record(schedule)["source_schedule"]
                schedule = schedule.rsplit(":", 1)[-1]
            assert schedule in cake_gdn._VEC2FULL_SHARED_SCHEDULES, route
    assert cake_gdn._prefill_tile_chunks(2048) == cake_gdn._VEC2FULL_MAX_TILE_CHUNKS
    assert cake_gdn._prefill_tile_chunks(2049) > cake_gdn._VEC2FULL_MAX_TILE_CHUNKS


@pytest.mark.cpu_only
@pytest.mark.parametrize(("overrides", "heads"), _VEC2FULL_CASES)
def test_vec2full_twin_needs_a_bound_of_at_most_2048_tokens(overrides, heads) -> None:
    """``max_seq_len`` is an upper bound: with the packed length as the bound (no host lengths,
    no hint) the batch keeps the shared record; with an exact hint it runs the twin."""

    lens = (1500,) * _full_dv_seqs(heads)  # the packed length is above 2048 tokens
    exact = _sm107a_fp32_state_route(heads, lens, host=False, **overrides)
    loose = _sm107a_fp32_state_route(
        heads, lens, host=False, max_seq_len=sum(lens), **overrides
    )
    assert "_sm107a_vec2full_i64meta_" in exact.variant_name, exact
    assert loose.variant_name == _vec2full_shared_name(exact.variant_name), loose
    assert loose.route_id == exact.route_id == "flashinfer.gdn_prefill.noncp.full_dv"


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("overrides", "heads", "lens"),
    (
        ({"state_dtype": "bfloat16"}, (8, 32), (2048,) + (64,) * 3),
        ({"use_initial_state": False}, (16, 32), (2048,) + (64,) * 7),
        (
            {"io_dtype": "float16", "use_initial_state": False},
            (16, 64),
            (2048, 1),
        ),
    ),
    ids=("bf16_state", "no_initial_state", "f16_io_no_initial_state"),
)
def test_vec2full_twin_leaves_the_ragged_dvsplit_route_first(
    monkeypatch, overrides, heads, lens
) -> None:
    """A ragged batch of at most 2048 tokens per sequence that the ragged DV-split rule routes
    runs the DV-split record; without the host lengths (or without the route) it runs the
    vec2-full twin of the full-DV key the route would have replaced."""

    assert _ragged_dvsplit_load_fits(max(heads), lens)
    routed = _sm107a_fp32_state_route(heads, lens, **overrides)
    assert routed.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert routed.one_cta_per_tile is True
    twin = _sm107a_fp32_state_route(heads, lens, host=False, **overrides)
    assert "_sm107a_vec2full_i64meta_" in twin.variant_name, twin
    assert (
        cake_gdn._ragged_dvsplit_routes()[_vec2full_shared_name(twin.variant_name)]
        == routed
    )
    monkeypatch.setattr(cake_gdn, "_ragged_dvsplit_routes", lambda: {})
    resolve = cake_gdn.select_cake_gdn_prefill_variant.__wrapped__
    assert resolve(**_sm107a_fp32_state_kwargs(heads, lens, **overrides)) == twin


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("heads", "lens", "overrides"),
    (
        # the gate-stream twin (and its prefetch twin) of the BF16-state contract
        ((8, 32), (256,) * 64, {"state_dtype": "bfloat16"}),
        ((2, 8), (256,) * 64, {"state_dtype": "bfloat16"}),
        # the indexed BF16-state record of the same contract
        (
            (32, 32),
            (256,) * 4,
            {"state_dtype": "bfloat16", "use_state_indices": True},
        ),
        # the FP16-I/O FP32-state checkpoint route without an initial state
        (
            (16, 64),
            (256,) * 2,
            {
                "io_dtype": "float16",
                "use_initial_state": False,
                "checkpoint_every_n_tokens": 64,
            },
        ),
        # the stateless bodies and the other contracts' full-DV records
        ((16, 64), (64,) * 2, {"use_initial_state": False, "store_final_state": False}),
        ((8, 32), (1024,) * 8, {}),
        ((8, 32), (1024,) * 8, {"io_dtype": "float16"}),
        # DV-split keys
        ((8, 32), (1024,) * 2, {"state_dtype": "bfloat16"}),
        ((16, 32), (1024,) * 2, {"use_initial_state": False}),
        # the other architectures
        ((8, 32), (1024,) * 8, {"state_dtype": "bfloat16", "arch": "sm_100a"}),
        ((8, 32), (1024,) * 8, {"state_dtype": "bfloat16", "arch": "sm_103a"}),
        ((16, 32), (1024,) * 8, {"use_initial_state": False, "arch": "sm_100a"}),
        ((16, 32), (1024,) * 8, {"use_initial_state": False, "arch": "sm_103a"}),
    ),
)
def test_vec2full_twin_leaves_every_other_route_unchanged(
    monkeypatch, heads, lens, overrides
) -> None:
    """Keys outside the three shared full-DV records resolve as they do with the twin rule off
    (or fail closed the same way), whatever the bound."""

    resolve = cake_gdn.select_cake_gdn_prefill_variant.__wrapped__

    def outcome(limit):
        monkeypatch.setattr(cake_gdn, "_VEC2FULL_MAX_TILE_CHUNKS", limit)
        try:
            return resolve(**_sm107a_fp32_state_kwargs(heads, lens, **overrides))
        except cake_gdn.CakeGDNUnsupportedError as error:
            return str(error)

    on, off = outcome(1 << 20), outcome(-1)
    assert on == off
    assert "_vec2full_" not in getattr(on, "variant_name", "")


@pytest.mark.cpu_only
def test_vec2full_twin_records_pair_the_shared_records() -> None:
    """Every vec2-full twin record stands beside the shared full-DV record of its key: the same
    specializations, metadata and launch, its own kernel symbol (``_vec2full`` suffixed) and
    unit.  Each shared record of the three schedules that stores a final state without state
    indices has exactly one twin (BF16 state 6, FP32 state without an initial state 2 + 2)."""

    records = {r["name"]: r for r in cake_gdn._manifest()["variants"]}
    twins = {n: r for n, r in records.items() if "_sm107a_vec2full_i64meta_" in n}
    counts = {}
    for name, twin in twins.items():
        shared = records[_vec2full_shared_name(name)]
        schedule = shared["source_schedule"].rsplit(":", 1)[-1]
        assert schedule in cake_gdn._VEC2FULL_SHARED_SCHEDULES
        assert twin["source_schedule"].rsplit(":", 1)[-1] == (
            cake_gdn._vec2full_twin_schedule(schedule)
        )
        counts[schedule] = counts.get(schedule, 0) + 1
        assert twin["architectures"] == shared["architectures"] == ["sm_107a"]
        for field in ("specializations", "metadata_i64", "tma_abi", "domain"):
            assert twin[field] == shared[field], (name, field)
        for field in ("params", "template"):
            assert twin["host_binding"][field] == shared["host_binding"][field]
        assert (
            twin["host_binding"]["cuda_symbol"]
            == shared["host_binding"]["cuda_symbol"] + "_vec2full"
        )
        assert "defines" not in twin and twin["outputs"] != shared["outputs"]
        assert (
            twin["specializations"]["STORE_FINAL_STATE"],
            twin["specializations"]["USE_STATE_INDICES"],
        ) == (1, 0)
    for name, record in records.items():
        schedule = record["source_schedule"].rsplit(":", 1)[-1]
        specs = record["specializations"]
        if schedule in cake_gdn._VEC2FULL_SHARED_SCHEDULES and (
            specs["STORE_FINAL_STATE"],
            specs["USE_STATE_INDICES"],
        ) == (1, 0):
            assert _vec2full_twin_name(name) in twins, name
    stem = "flashinfer_blackwell_gdn_prefill"
    assert counts == {
        f"{stem}_initial_bf16state_sm107a_i64meta": 6,
        f"{stem}_sm107a_i64meta": 2,
        f"{stem}_f16io_sm107a_i64meta": 2,
    }


@pytest.mark.cpu_only
def test_vec2full_twin_is_the_only_change_of_the_rule_on_random_keys(
    monkeypatch,
) -> None:
    """On random keys (head geometries, dtypes, with or without an initial state, indexed state,
    architectures, with and without the host lengths, exact or loose bounds) the resolver returns
    the route it returns with the twin rule off, except an sm_107a full-DV batch whose resolved
    record is one of the three shared full-DV records and whose bound spans at most 32 chunks:
    that batch runs the key's vec2-full twin (the same route otherwise)."""

    rng = random.Random(20261107)
    resolve = cake_gdn.select_cake_gdn_prefill_variant.__wrapped__
    heads_pairs = _RAGGED_DVSPLIT_GEOMETRIES + ((16, 16), (32, 32), (2, 6), (32, 16))
    keys = []
    for _ in range(1_500):
        heads = rng.choice(heads_pairs)
        num_seqs = rng.randint(1, 48)
        top = rng.choice((1024, 2048, 2048, 2112, 8192))
        lens = tuple(rng.randint(1, top) for _ in range(num_seqs))
        if rng.random() < 0.3:
            lens = (rng.choice((1984, 2048, 2049, 2176)),) * num_seqs
        overrides = dict(
            io_dtype=rng.choice(("bfloat16", "float16")),
            state_dtype=rng.choice(("float32", "float32", "bfloat16")),
            use_initial_state=rng.random() < 0.6,
            arch=rng.choice(("sm_107a",) * 6 + ("sm_100a", "sm_103a")),
        )
        if overrides["state_dtype"] == "bfloat16":
            overrides["use_initial_state"] = True
            overrides["io_dtype"] = "bfloat16"
            overrides["use_state_indices"] = rng.random() < 0.2
        if rng.random() < 0.2:
            overrides["max_seq_len"] = sum(lens)
        keys.append((heads, lens, overrides, rng.random() < 0.7))

    def resolve_all(limit):
        monkeypatch.setattr(cake_gdn, "_VEC2FULL_MAX_TILE_CHUNKS", limit)
        out = []
        for heads, lens, overrides, host in keys:
            try:
                out.append(
                    resolve(
                        **_sm107a_fp32_state_kwargs(heads, lens, host=host, **overrides)
                    )
                )
            except cake_gdn.CakeGDNUnsupportedError:
                out.append(None)
        return out

    with_rule = resolve_all(32)
    without_rule = resolve_all(-1)
    twins = set()
    for (heads, lens, overrides, _host), got, base in zip(
        keys, with_rule, without_rule, strict=True
    ):
        if base is None:
            assert got is None
            continue
        record = cake_gdn._kernel_record(base.variant_name)
        bound = overrides.get("max_seq_len", max(lens))
        eligible = (
            overrides["arch"] == "sm_107a"
            and base.route_id == "flashinfer.gdn_prefill.noncp.full_dv"
            and record["source_schedule"].rsplit(":", 1)[-1]
            in cake_gdn._VEC2FULL_SHARED_SCHEDULES
            and not overrides.get("use_state_indices", False)
            and cake_gdn._prefill_tile_chunks(bound) <= 32
            and _vec2full_twin_name(base.variant_name) in _manifest_names()
        )
        if eligible:
            assert got == cake_gdn.CakeGDNRoute(
                base.route_id,
                _vec2full_twin_name(base.variant_name),
                True,
                False,
            ), (heads, lens, overrides)
            twins.add(record["source_schedule"])
        else:
            assert got == base, (heads, lens, overrides)
    assert len(twins) == 3, twins


def _manifest_names():
    return {r["name"] for r in cake_gdn._manifest()["variants"]}
