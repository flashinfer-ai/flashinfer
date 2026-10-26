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
    assert cake_gdn.CakeGDNRoute._fields == ("route_id", "variant_name", "metadata_i64")
    assert cake_gdn.CakeGDNRoute("prefill.dvsplit", "variant").metadata_i64 is False


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
