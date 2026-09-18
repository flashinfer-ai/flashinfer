# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contract-F correctness coverage for packed DSV4 CSA FMHA."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0a0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0a0",
)

from flashinfer.attention.prims_ts import (
    prims_ts_dsv4_csa,
    prims_ts_dsv4_sparse_mla_rope_quant,
)
from flashinfer.attention.prims_ts.dsv4_csa import _get_compiled_dsv4_csa
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.config import (
    make_mla_decode_config,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.tasks import (
    dsv4_pair_ring_event_plan,
)


_REQUIRES_DSV4_CSA_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS DSV4 CSA requires SM100 or SM103",
)
_FP8 = torch.float8_e4m3fn
_HEADS = 128
_HEAD_DIM = 512
_SWA_WIDTH = 128
_COMPRESS_RATIO = 4


def test_dsv4_csa_async_config_matches_generated_reference() -> None:
    """Keep DSV4's generated async/resource contract explicit in TS config."""

    cfg = make_mla_decode_config(
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        rope_dim=0,
        page_size=1,
        qkv_dtype="e4m3",
        o_dtype="bf16",
        is_persistent=True,
        is_var_seq=True,
        is_dynamic_token_sparse=True,
        sparse_swa_topk=128,
    )

    assert cfg.cluster_shape_mnk == (2, 1, 1)
    assert cfg.threads_per_cta == 512
    assert cfg.mma_qk_tiler == (128, 128, 128)
    # Source BMM2 consumes one P[K128] operand per D256 V panel.
    assert cfg.mma_pv_tiler == (128, 256, 128)
    assert (cfg.iterations_pv_k, cfg.iterations_pv_n) == (1, 2)
    assert (cfg.load_q_stage, cfg.load_k_stage, cfg.load_v_stage) == (1, 2, 2)
    assert (cfg.mma_s_stage, cfg.p_mma_stage, cfg.p_cor_stage, cfg.mma_o_stage) == (
        2,
        2,
        2,
        1,
    )
    assert (cfg.softmax_reg_num, cfg.correction_reg_num) == (152, 144)
    assert (cfg.producer_reg_num, cfg.gather4_reg_num) == (136, 72)
    assert (cfg.load_v_warp_id, cfg.load_v_num_warps) == (12, 4)
    # Source V is four Gather4 warps × eight interleaved page quads × two
    # H256 slices.  Its CTA-group completion barrier is therefore 64 KiB.
    assert (cfg.tma_copy_k_tile_bytes, cfg.tma_copy_v_tile_bytes) == (
        65536,
        65536,
    )
    assert (cfg.scheduler_warp_id, cfg.padding_warp_id, cfg.pv_mma_warp_id) == (
        10,
        11,
        11,
    )
    assert cfg.tmem_sync_bar_threads == 512

    # Source TmemS uses one 128-float scratch per S stage and two disjoint
    # 64-thread named barriers for the W0/W2 and W1/W3 row pairs.
    assert cfg.softmax_exchange_elems == 2 * 128
    assert (cfg.softmax_sync_bar_id, cfg.softmax_sync_threads) == (1, 64)
    # Generated DSV4 CSA TMEM map: S0/S1=[0,128), softmax stats=[128,192),
    # O panels from 192.
    assert cfg.correction_factor_offset == 128
    assert cfg.tmem_o_offset == 192
    with pytest.raises(ValueError, match="qkv_dtype='e4m3'"):
        make_mla_decode_config(
            mma_qk_tiler_mn=(128, 128),
            mma_pv_tiler_mn=(128, 256),
            rope_dim=0,
            page_size=1,
            qkv_dtype="bf16",
            o_dtype="bf16",
            is_persistent=True,
            is_var_seq=True,
            is_dynamic_token_sparse=True,
            sparse_swa_topk=128,
        )

    skip_corr_cfg = make_mla_decode_config(
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        rope_dim=0,
        page_size=1,
        qkv_dtype="e4m3",
        o_dtype="bf16",
        is_persistent=True,
        is_var_seq=True,
        is_dynamic_token_sparse=True,
        sparse_swa_topk=128,
        dsv4_enable_skip_correction=True,
    )
    assert skip_corr_cfg.dsv4_enable_skip_correction
    with pytest.raises(ValueError, match="skip correction"):
        make_mla_decode_config(
            qkv_dtype="e4m3",
            dsv4_enable_skip_correction=True,
        )


@pytest.mark.parametrize("k_tiles", (0, 1, 2, 3, 6, 7, 9))
def test_dsv4_source_pair_ring_ownership_contract(k_tiles: int) -> None:
    """Pin the W9/W12--15 pair hold/reuse/release protocol before JIT porting."""

    events = dsv4_pair_ring_event_plan(k_tiles)
    pair_count = (k_tiles + 1) // 2
    for operand, stage_offset in (("K", 0), ("V", 1)):
        produced = [e for e in events if e.kind == "produce" and e.operand == operand]
        waits = [e for e in events if e.kind == "wait" and e.operand == operand]
        gathers = [e for e in events if e.kind == "gather" and e.operand == operand]
        releases = [e for e in events if e.kind == "release" and e.operand == operand]

        assert len(produced) == len(waits) == len(releases) == pair_count
        assert [e.tile for e in gathers] == list(range(k_tiles))
        assert [(e.pair, e.half) for e in gathers] == [
            divmod(tile, 2) for tile in range(k_tiles)
        ]
        assert [e.stage for e in produced] == [
            (2 * pair + stage_offset) % 6 for pair in range(pair_count)
        ]
        assert [e.stage for e in waits] == [e.stage for e in releases]
        assert [e.tile for e in releases] == [
            min(2 * pair + 1, k_tiles - 1) for pair in range(pair_count)
        ]

    if k_tiles:
        consumer_events = [e for e in events if e.kind != "produce"]
        gather_position = {
            (event.operand, event.tile): position
            for position, event in enumerate(consumer_events)
            if event.kind == "gather"
        }
        release_position = {
            (event.operand, event.tile): position
            for position, event in enumerate(consumer_events)
            if event.kind == "release"
        }
        assert gather_position[("K", 0)] < gather_position[("V", 0)]
        for tile in range(k_tiles - 1):
            assert gather_position[("K", tile + 1)] < gather_position[("V", tile)]
        assert (
            release_position[("K", k_tiles - 1)] < gather_position[("V", k_tiles - 1)]
        )


def _rand_fp8(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return (
        torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)
        * 0.25
    ).to(_FP8)


def _make_contract_e_metadata(
    cu_seqlens_q: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    *,
    sparse_capacity: int,
    compress_ratio: int = _COMPRESS_RATIO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build identity-page-table physical metadata for a CSA/HCA fixture."""

    # The generated source's ``fillDynamicSparseMlaIndices`` writes -1 for
    # inactive SWA slots, even though the Gather4 tile still reaches them.
    # Preserve that ABI instead of hiding it with a benign row-zero address:
    # the raw Gather4 descriptor/load path must supply the same masked-out
    # behavior as source.  The unselected compressed tail remains zero, which
    # matches source's zero-initialized selector allocation.
    routes = torch.full(
        (int(cu_seqlens_q[-1]), sparse_capacity),
        0,
        dtype=torch.int32,
        device="cuda",
    )
    sparse_lens = torch.empty(routes.shape[0], dtype=torch.int32, device="cuda")
    for batch_idx in range(seq_lens_kv.numel()):
        q_begin = int(cu_seqlens_q[batch_idx])
        q_end = int(cu_seqlens_q[batch_idx + 1])
        q_len = q_end - q_begin
        for packed_q in range(q_begin, q_end):
            q = packed_q - q_begin
            raw_visible = int(seq_lens_kv[batch_idx]) - q_len + q + 1
            swa_valid = min(raw_visible, _SWA_WIDTH)
            routes[packed_q, :swa_valid] = torch.arange(
                raw_visible - swa_valid, raw_visible, device="cuda", dtype=torch.int32
            )
            routes[packed_q, swa_valid:_SWA_WIDTH] = -1
            compressed_count = min(
                raw_visible // compress_ratio, sparse_capacity - _SWA_WIDTH
            )
            routes[packed_q, _SWA_WIDTH : _SWA_WIDTH + compressed_count] = torch.arange(
                compressed_count, device="cuda", dtype=torch.int32
            )
            sparse_lens[packed_q] = _SWA_WIDTH + compressed_count
    return routes, sparse_lens


def _reference_contract_f(
    query: torch.Tensor,
    swa_pool: torch.Tensor,
    compressed_pool: torch.Tensor,
    routes: torch.Tensor,
    sparse_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    skip_corr_threshold: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference TRTLLM-gen's tiled E4M3 online-softmax protocol.

    A positive skip-correction threshold changes more than the correction
    branch: it freezes sufficiently small running-max increases and changes
    the P quantization scale from 448 to 1.75.  Keep those operations together
    here so the default API is never tested against a threshold-zero oracle.
    """

    output = torch.empty_like(query, dtype=torch.float32)
    lse = torch.empty((query.shape[0], _HEADS), dtype=torch.float32, device="cuda")
    for batch_idx in range(seq_lens_kv.numel()):
        q_begin = int(cu_seqlens_q[batch_idx])
        q_end = int(cu_seqlens_q[batch_idx + 1])
        q_len = q_end - q_begin
        for packed_q in range(q_begin, q_end):
            q = packed_q - q_begin
            raw_visible = int(seq_lens_kv[batch_idx]) - q_len + q + 1
            active_width = int(sparse_lens[packed_q])
            swa_valid = min(raw_visible, _SWA_WIDTH, active_width)
            kv_tiles = [swa_pool[routes[packed_q, :swa_valid].long()].float()]
            if active_width > _SWA_WIDTH:
                compressed = routes[packed_q, _SWA_WIDTH:active_width]
                compressed = compressed[compressed >= 0]
                for tile_begin in range(0, compressed.numel(), _SWA_WIDTH):
                    tile_routes = compressed[tile_begin : tile_begin + _SWA_WIDTH]
                    kv_tiles.append(compressed_pool[tile_routes.long()].float())

            scale_log2 = bmm1_scale / math.log(2.0)
            adjusted_threshold = (
                skip_corr_threshold / scale_log2 if skip_corr_threshold > 0.0 else 0.0
            )
            p_scale = 1.75 if skip_corr_threshold > 0.0 else 448.0
            row_max = torch.full(
                (_HEADS,), -torch.finfo(torch.float32).max, device="cuda"
            )
            row_sum = torch.zeros((_HEADS,), dtype=torch.float32, device="cuda")
            row_o = torch.zeros((_HEADS, _HEAD_DIM), dtype=torch.float32, device="cuda")
            q_row = query[packed_q].float()
            for kv_tile in kv_tiles:
                scores = q_row @ kv_tile.T
                candidate_max = torch.maximum(row_max, scores.max(dim=-1).values)
                if skip_corr_threshold > 0.0:
                    candidate_max = torch.where(
                        candidate_max - row_max <= adjusted_threshold,
                        row_max,
                        candidate_max,
                    )
                correction = torch.exp(
                    (row_max - candidate_max) * scale_log2 * math.log(2.0)
                )
                p = (
                    torch.exp(
                        (scores - candidate_max[:, None]) * scale_log2 * math.log(2.0)
                    )
                    * p_scale
                )
                row_o = row_o * correction[:, None] + (
                    p.to(torch.float8_e4m3fn).float() @ kv_tile
                )
                row_sum = row_sum * correction + p.sum(dim=-1)
                row_max = candidate_max

            output[packed_q] = row_o * (bmm2_scale / row_sum[:, None])
            lse[packed_q] = (
                torch.log(row_sum / p_scale) / math.log(2.0) + row_max * scale_log2
            )
    return output, lse


@pytest.mark.parametrize("threshold", (-1.0, 8.0001, math.inf, math.nan, True))
@_REQUIRES_DSV4_CSA_GPU
def test_prims_ts_dsv4_csa_skip_correction_argument_validation(
    threshold: float,
) -> None:
    """Keep the public threshold domain identical to the E4M3 source mode."""

    query = torch.zeros((1, _HEADS, _HEAD_DIM), device="cuda", dtype=_FP8)
    pool = torch.zeros((1, _HEAD_DIM), device="cuda", dtype=_FP8)
    routes = torch.zeros((1, _SWA_WIDTH), device="cuda", dtype=torch.int32)
    sparse_lens = torch.tensor([_SWA_WIDTH], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([1], device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="skip_corr_threshold"):
        prims_ts_dsv4_csa(
            query,
            pool,
            pool,
            routes,
            sparse_lens,
            seq_lens_kv,
            cu_seqlens_q,
            max_seq_len_q=1,
            skip_corr_threshold=threshold,
        )


@_REQUIRES_DSV4_CSA_GPU
def test_prims_ts_dsv4_csa_runtime_launch_shape_reuses_compilation() -> None:
    """Keep packed B/T/maxQ/Kmax runtime, as in TRTLLM-gen's launch ABI.

    The launches share every device-code option while changing batch count,
    maximum/local Q lengths, packed T, selector row stride, active K-tile
    count, and partial/full selector capacity.  Later calls must hit the exact
    same compiled specialization, not merely produce correct results after
    silently compiling more kernels.
    """

    generator = torch.Generator(device="cuda").manual_seed(20260904)
    swa_pool = _rand_fp8((600, _HEAD_DIM), generator)
    compressed_pool = _rand_fp8((600, _HEAD_DIM), generator)
    bmm1_scale = 1.0 / math.sqrt(_HEAD_DIM)
    bmm2_scale = 0.75

    def run_case(
        *,
        q_lens: tuple[int, ...],
        max_seq_len_q: int,
        sparse_capacity: int,
        seq_lens_kv_values: tuple[int, ...],
        skip_corr_threshold: float = 8.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(q_lens) == len(seq_lens_kv_values)
        total_q = sum(q_lens)
        query = _rand_fp8((total_q, _HEADS, _HEAD_DIM), generator)
        offsets = [0]
        for q_len in q_lens:
            offsets.append(offsets[-1] + q_len)
        cu_seqlens_q = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        seq_lens_kv = torch.tensor(seq_lens_kv_values, device="cuda", dtype=torch.int32)
        routes, sparse_lens = _make_contract_e_metadata(
            cu_seqlens_q, seq_lens_kv, sparse_capacity=sparse_capacity
        )
        lse = torch.empty((total_q, _HEADS), device="cuda", dtype=torch.float32)
        actual = prims_ts_dsv4_csa(
            query,
            compressed_pool,
            swa_pool,
            routes,
            sparse_lens,
            seq_lens_kv,
            cu_seqlens_q,
            max_seq_len_q=max_seq_len_q,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            skip_corr_threshold=skip_corr_threshold,
            lse=lse,
        )
        expected, expected_lse = _reference_contract_f(
            query,
            swa_pool,
            compressed_pool,
            routes,
            sparse_lens,
            cu_seqlens_q,
            seq_lens_kv,
            bmm1_scale,
            bmm2_scale,
            skip_corr_threshold,
        )
        return actual, lse, expected, expected_lse

    first = run_case(
        q_lens=(1,),
        max_seq_len_q=1,
        sparse_capacity=192,
        seq_lens_kv_values=(3,),
    )
    cache_after_first = _get_compiled_dsv4_csa.cache_info()
    second = run_case(
        q_lens=(2,),
        max_seq_len_q=2,
        sparse_capacity=384,
        seq_lens_kv_values=(516,),
    )
    cache_after_second = _get_compiled_dsv4_csa.cache_info()
    third = run_case(
        q_lens=(2, 3),
        # Exercise a padded runtime grid row as well as B/maxQ reuse.
        max_seq_len_q=4,
        sparse_capacity=256,
        seq_lens_kv_values=(10, 260),
    )
    cache_after_third = _get_compiled_dsv4_csa.cache_info()
    fourth = run_case(
        q_lens=(1,),
        max_seq_len_q=1,
        sparse_capacity=192,
        seq_lens_kv_values=(256,),
        # Positive threshold magnitude is a runtime scalar.  Only its sign
        # selects the compiled skip-correction specialization.
        skip_corr_threshold=4.0,
    )
    cache_after_fourth = _get_compiled_dsv4_csa.cache_info()

    assert cache_after_second.misses == cache_after_first.misses
    assert cache_after_second.hits == cache_after_first.hits + 1
    assert cache_after_third.misses == cache_after_second.misses
    assert cache_after_third.hits == cache_after_second.hits + 1
    assert cache_after_fourth.misses == cache_after_third.misses
    assert cache_after_fourth.hits == cache_after_third.hits + 1
    for actual, lse, expected, expected_lse in (first, second, third, fourth):
        torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=5e-2)
        torch.testing.assert_close(lse, expected_lse, atol=1e-3, rtol=1e-3)


@_REQUIRES_DSV4_CSA_GPU
def test_prims_ts_dsv4_hca_r128_padded_workids_preserve_output() -> None:
    """Keep an HCA R128-derived K192 workload exact across padded WorkIds.

    Compression ratio is an upstream selector property rather than a kernel
    argument.  This fixture nevertheless uses the canonical HCA R128 route
    counts and verifies both the runtime launch ABI and the attention result.
    """

    generator = torch.Generator(device="cuda").manual_seed(20260908)
    q_lens = (2, 3)
    total_q = sum(q_lens)
    cu_seqlens_q = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([127, 8192], device="cuda", dtype=torch.int32)
    query = _rand_fp8((total_q, _HEADS, _HEAD_DIM), generator)
    swa_pool = _rand_fp8((8192, _HEAD_DIM), generator)
    compressed_pool = _rand_fp8((8192, _HEAD_DIM), generator)
    routes, sparse_lens = _make_contract_e_metadata(
        cu_seqlens_q,
        seq_lens_kv,
        sparse_capacity=192,
        compress_ratio=128,
    )
    assert tuple(sparse_lens.tolist()) == (128, 128, 191, 191, 192)
    bmm1_scale = 1.0 / math.sqrt(_HEAD_DIM)
    bmm2_scale = 0.75

    def run(max_seq_len_q: int) -> tuple[torch.Tensor, torch.Tensor]:
        lse = torch.empty((total_q, _HEADS), device="cuda", dtype=torch.float32)
        output = prims_ts_dsv4_csa(
            query,
            compressed_pool,
            swa_pool,
            routes,
            sparse_lens,
            seq_lens_kv,
            cu_seqlens_q,
            max_seq_len_q=max_seq_len_q,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            skip_corr_threshold=8.0,
            lse=lse,
        )
        torch.cuda.synchronize()
        return output, lse

    one_padding_output, one_padding_lse = run(3)
    cache_after_one_padding = _get_compiled_dsv4_csa.cache_info()
    three_padding_output, three_padding_lse = run(4)
    cache_after_three_padding = _get_compiled_dsv4_csa.cache_info()

    # B=2,T=5 creates one padded WorkId at maxQ=3 and three at maxQ=4.
    # Padding advances the source WorkId/throttle protocol but cannot publish
    # data or alter any logical query row.
    assert torch.equal(one_padding_output, three_padding_output)
    assert torch.equal(one_padding_lse, three_padding_lse)
    assert cache_after_three_padding.misses == cache_after_one_padding.misses
    assert cache_after_three_padding.hits == cache_after_one_padding.hits + 1
    assert torch.isfinite(one_padding_output.float()).all()
    assert torch.isfinite(one_padding_lse).all()

    expected, expected_lse = _reference_contract_f(
        query,
        swa_pool,
        compressed_pool,
        routes,
        sparse_lens,
        cu_seqlens_q,
        seq_lens_kv,
        bmm1_scale,
        bmm2_scale,
        8.0,
    )
    torch.testing.assert_close(
        one_padding_output.float(), expected, atol=1e-2, rtol=5e-2
    )
    torch.testing.assert_close(one_padding_lse, expected_lse, atol=1e-3, rtol=1e-3)


@_REQUIRES_DSV4_CSA_GPU
@pytest.mark.parametrize(
    (
        "compress_ratio",
        "sparse_capacity",
        "seq_lens_kv_values",
        "pool_rows",
        "expected_sparse_lens",
    ),
    (
        (4, 256, (10, 260), 300, None),
        (128, 192, (127, 8192), 8192, (128, 128, 191, 191, 192)),
    ),
    ids=("csa-r4-k256", "hca-r128-k192"),
)
def test_prims_ts_dsv4_rope_quant_padded_workids_preserve_output(
    compress_ratio: int,
    sparse_capacity: int,
    seq_lens_kv_values: tuple[int, int],
    pool_rows: int,
    expected_sparse_lens: tuple[int, ...] | None,
) -> None:
    """Keep CSA/HCA RopeQuant exact when padded WorkId count changes."""

    generator = torch.Generator(device="cuda").manual_seed(20260908)
    q_lens = (2, 3)
    total_q = sum(q_lens)
    cu_seqlens_q = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor(seq_lens_kv_values, device="cuda", dtype=torch.int32)
    query = _rand_fp8((total_q, _HEADS, _HEAD_DIM), generator)
    swa_pool = _rand_fp8((pool_rows, _HEAD_DIM), generator)
    compressed_pool = _rand_fp8((pool_rows, _HEAD_DIM), generator)
    routes, sparse_lens = _make_contract_e_metadata(
        cu_seqlens_q,
        seq_lens_kv,
        sparse_capacity=sparse_capacity,
        compress_ratio=compress_ratio,
    )
    if expected_sparse_lens is not None:
        assert tuple(sparse_lens.tolist()) == expected_sparse_lens
    bmm1_scale = 1.0 / math.sqrt(_HEAD_DIM)

    positions = torch.arange(pool_rows, device="cuda", dtype=torch.float32)[:, None]
    frequencies = torch.linspace(0.001, 0.032, 32, device="cuda", dtype=torch.float32)[
        None, :
    ]
    angles = positions * frequencies
    inv_rope_cos_sin = torch.cat((angles.cos(), angles.sin()), dim=1).contiguous()

    def run(max_seq_len_q: int, output_fill: int):
        output = torch.empty(
            (_HEADS // 8, total_q, 8, _HEAD_DIM), device="cuda", dtype=_FP8
        )
        # Different sentinels make a missing logical output store observable.
        output.view(torch.uint8).fill_(output_fill)
        output_scale = torch.empty(
            (_HEADS // 8, 8 * 4, 8), device="cuda", dtype=torch.float32
        )
        # The pad4(T) scale tail is outside the public logical output.  Give it
        # the same deterministic sentinel so whole-buffer equality is useful.
        output_scale.view(torch.uint8).fill_(0xDC)
        result, result_scale = prims_ts_dsv4_sparse_mla_rope_quant(
            query,
            compressed_pool,
            swa_pool,
            routes,
            sparse_lens,
            seq_lens_kv,
            cu_seqlens_q,
            inv_rope_cos_sin,
            max_seq_len_q=max_seq_len_q,
            bmm1_scale=bmm1_scale,
            skip_corr_threshold=8.0,
            out=output,
            out_scale=output_scale,
        )
        torch.cuda.synchronize()
        return result.clone(), result_scale.clone()

    one_padding_code, one_padding_scale = run(3, 0x11)
    cache_after_one_padding = _get_compiled_dsv4_csa.cache_info()
    three_padding_code, three_padding_scale = run(4, 0xEE)
    cache_after_three_padding = _get_compiled_dsv4_csa.cache_info()

    # B=2,T=5 gives one padded WorkId for maxQ=3 and three for maxQ=4.  Those
    # rows advance source's throttle/WorkQueue protocol but cannot affect data.
    assert torch.equal(one_padding_code, three_padding_code)
    assert torch.equal(one_padding_scale, three_padding_scale)
    assert cache_after_three_padding.misses == cache_after_one_padding.misses
    assert cache_after_three_padding.hits == cache_after_one_padding.hits + 1

    expected, _ = _reference_contract_f(
        query,
        swa_pool,
        compressed_pool,
        routes,
        sparse_lens,
        cu_seqlens_q,
        seq_lens_kv,
        bmm1_scale,
        1.0,
        8.0,
    )
    for batch_idx, q_len in enumerate(q_lens):
        q_begin = int(cu_seqlens_q[batch_idx])
        for local_q in range(q_len):
            packed_q = q_begin + local_q
            position = int(seq_lens_kv[batch_idx]) - q_len + local_q
            cos_sin = inv_rope_cos_sin[position]
            rope = expected[packed_q, :, 448:512].reshape(_HEADS, 32, 2)
            first = rope[..., 0].clone()
            second = rope[..., 1].clone()
            rope[..., 0] = first * cos_sin[:32] + second * cos_sin[32:]
            rope[..., 1] = second * cos_sin[:32] - first * cos_sin[32:]

    expected_blocks = expected.reshape(total_q, _HEADS, 4, 128)
    expected_scale = expected_blocks.abs().amax(dim=-1).clamp_min(1.0e-12) / 448.0
    expected_code = (expected_blocks / expected_scale[..., None]).to(_FP8)
    expected_dequant = expected_code.float() * expected_scale[..., None]

    actual_code = one_padding_code.permute(1, 0, 2, 3).reshape(
        total_q, _HEADS, _HEAD_DIM
    )
    actual_scale = (
        one_padding_scale[:, :, :total_q]
        .reshape(_HEADS // 8, 8, 4, total_q)
        .permute(3, 0, 1, 2)
        .reshape(total_q, _HEADS, 4)
    )
    assert torch.isfinite(actual_scale).all()
    assert (actual_scale > 0).all()
    actual_dequant = actual_code.reshape(total_q, _HEADS, 4, 128).float()
    actual_dequant = actual_dequant * actual_scale[..., None]
    torch.testing.assert_close(actual_dequant, expected_dequant, atol=1e-2, rtol=1e-1)
    torch.testing.assert_close(actual_scale, expected_scale, atol=1e-5, rtol=1.5e-1)


@_REQUIRES_DSV4_CSA_GPU
def test_prims_ts_dsv4_csa_packed_two_request_causal_swa_mask() -> None:
    """Check packed causal/SWA semantics on a two-request fixture."""

    generator = torch.Generator(device="cuda").manual_seed(20260903)
    # Request A's three Q rows see raw positions 7..9; B's two rows see 5..6.
    # This catches a b * max_q + q metadata mapping as well as the bug that
    # treats Lq=130 as 128 valid SWA candidates.
    cu_seqlens_q = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([10, 7], device="cuda", dtype=torch.int32)
    query = _rand_fp8((5, _HEADS, _HEAD_DIM), generator)
    swa_pool = _rand_fp8((32, _HEAD_DIM), generator)
    compressed_pool = _rand_fp8((32, _HEAD_DIM), generator)
    routes, sparse_lens = _make_contract_e_metadata(
        cu_seqlens_q, seq_lens_kv, sparse_capacity=256
    )
    bmm1_scale = 1.0 / math.sqrt(_HEAD_DIM)
    bmm2_scale = 0.75

    lse = torch.empty((5, _HEADS), device="cuda", dtype=torch.float32)
    actual = prims_ts_dsv4_csa(
        query,
        compressed_pool,
        swa_pool,
        routes,
        sparse_lens,
        seq_lens_kv,
        cu_seqlens_q,
        max_seq_len_q=3,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        lse=lse,
    )
    # The output-only specialization compiles the LSE log/store epilogue
    # away and must preserve O exactly.
    output_only = prims_ts_dsv4_csa(
        query,
        compressed_pool,
        swa_pool,
        routes,
        sparse_lens,
        seq_lens_kv,
        cu_seqlens_q,
        max_seq_len_q=3,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )
    assert torch.equal(output_only, actual)
    expected, expected_lse = _reference_contract_f(
        query,
        swa_pool,
        compressed_pool,
        routes,
        sparse_lens,
        cu_seqlens_q,
        seq_lens_kv,
        bmm1_scale,
        bmm2_scale,
    )
    assert torch.isfinite(actual.float()).all()
    assert torch.isfinite(lse).all()
    # E4M3 MMA accumulates at lower precision than the FP32 eager reference.
    # The bound matches the existing Blackwell FP8 attention coverage.
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=5e-2)
    torch.testing.assert_close(lse, expected_lse, atol=1e-3, rtol=1e-3)


@_REQUIRES_DSV4_CSA_GPU
def test_prims_ts_dsv4_csa_packed_multi_work_tile() -> None:
    """Exercise persistent task-state reuse across many packed work tiles.

    B=2 and 65 packed query rows/request create 260 logical CTAs, which
    crosses the B200 resident grid and drives the CLC WorkId/throttle state
    across scheduler responses.  This closes the earlier B=1,
    single-query-only coverage gap.
    """

    generator = torch.Generator(device="cuda").manual_seed(20260904)
    cu_seqlens_q = torch.tensor([0, 65, 130], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([131, 130], device="cuda", dtype=torch.int32)
    query = _rand_fp8((130, _HEADS, _HEAD_DIM), generator)
    swa_pool = _rand_fp8((256, _HEAD_DIM), generator)
    compressed_pool = _rand_fp8((256, _HEAD_DIM), generator)
    routes, sparse_lens = _make_contract_e_metadata(
        cu_seqlens_q, seq_lens_kv, sparse_capacity=256
    )
    bmm1_scale = 1.0 / math.sqrt(_HEAD_DIM)
    bmm2_scale = 0.75

    lse = torch.empty((130, _HEADS), device="cuda", dtype=torch.float32)
    actual = prims_ts_dsv4_csa(
        query,
        compressed_pool,
        swa_pool,
        routes,
        sparse_lens,
        seq_lens_kv,
        cu_seqlens_q,
        max_seq_len_q=65,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        lse=lse,
    )
    expected, expected_lse = _reference_contract_f(
        query,
        swa_pool,
        compressed_pool,
        routes,
        sparse_lens,
        cu_seqlens_q,
        seq_lens_kv,
        bmm1_scale,
        bmm2_scale,
    )
    assert torch.isfinite(actual.float()).all()
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=5e-2)
    torch.testing.assert_close(lse, expected_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    ("seq_len_kv_value", "sparse_capacity", "expected_sparse_len"),
    (
        # Source first fills the fixed 128-slot SWA tile, then appends
        # floor(visible / R) compressed slots.  These values exercise the W9
        # pair producer boundaries through the smoke kernel's 9 K128 tiles,
        # including the six-stage selector-ring wraparound at K=7 and K=9.
        # Kmax=192 is the source HCA ABI (128 SWA + 64 compressed slots): it
        # has a partial final K128 tile, so it must not be rejected merely
        # because the static row capacity is not a multiple of 128.
        (3, 192, 128),
        (4, 192, 129),
        (256, 192, 192),
        (3, 256, 128),
        (4, 256, 129),
        (512, 256, 256),
        (516, 384, 257),
        (3072, 896, 896),
        (4096, 1152, 1152),
    ),
)
@_REQUIRES_DSV4_CSA_GPU
def test_prims_ts_dsv4_csa_source_sparse_length_pair_boundaries(
    seq_len_kv_value: int,
    sparse_capacity: int,
    expected_sparse_len: int,
) -> None:
    """Keep source ceil(Lq/128) boundaries correct for the W9 pair port.

    This uses the source-style -1 inactive SWA slots and varies only the
    runtime sparse scan length. ``sparse_capacity`` is also a runtime tensor
    extent, and source only requires int32x4 alignment: the final K128 tile may
    be partial (notably HCA Kmax=192).
    """

    generator = torch.Generator(device="cuda").manual_seed(
        20260903 + expected_sparse_len
    )
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([seq_len_kv_value], device="cuda", dtype=torch.int32)
    query = _rand_fp8((1, _HEADS, _HEAD_DIM), generator)
    # The raw SWA route can name the final visible token; use the same pool
    # size for compressed storage to avoid adding a distinct address-boundary
    # variable to this pair-cadence test.
    pool_rows = max(520, seq_len_kv_value)
    swa_pool = _rand_fp8((pool_rows, _HEAD_DIM), generator)
    compressed_pool = _rand_fp8((pool_rows, _HEAD_DIM), generator)
    routes, sparse_lens = _make_contract_e_metadata(
        cu_seqlens_q, seq_lens_kv, sparse_capacity=sparse_capacity
    )
    assert int(sparse_lens.item()) == expected_sparse_len
    swa_valid = min(seq_len_kv_value, _SWA_WIDTH)
    assert torch.equal(
        routes[0, swa_valid:_SWA_WIDTH],
        torch.full((_SWA_WIDTH - swa_valid,), -1, device="cuda", dtype=torch.int32),
    )

    bmm1_scale = 1.0 / math.sqrt(_HEAD_DIM)
    bmm2_scale = 0.75
    lse = torch.empty((1, _HEADS), device="cuda", dtype=torch.float32)
    actual = prims_ts_dsv4_csa(
        query,
        compressed_pool,
        swa_pool,
        routes,
        sparse_lens,
        seq_lens_kv,
        cu_seqlens_q,
        max_seq_len_q=1,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        lse=lse,
    )
    expected, expected_lse = _reference_contract_f(
        query,
        swa_pool,
        compressed_pool,
        routes,
        sparse_lens,
        cu_seqlens_q,
        seq_lens_kv,
        bmm1_scale,
        bmm2_scale,
    )
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=5e-2)
    torch.testing.assert_close(lse, expected_lse, atol=1e-3, rtol=1e-3)


@_REQUIRES_DSV4_CSA_GPU
def test_prims_ts_dsv4_csa_bf16_output_satfinite() -> None:
    """Match TRTLLM-gen's BF16 epilogue when runtime scaling overflows.

    The generated source uses CUTLASS ``round_to_nearest_satfinite``.  This
    fixture makes the pre-conversion FP32 result overflow positively, so a
    plain BF16 conversion would produce ``inf`` while the source contract
    produces the largest finite BF16 value.
    """

    query = torch.zeros(
        (1, _HEADS, _HEAD_DIM), device="cuda", dtype=torch.float8_e4m3fn
    )
    max_fp8 = torch.full(
        (1, _HEAD_DIM), 448.0, device="cuda", dtype=torch.float8_e4m3fn
    )
    routes = torch.zeros((1, _SWA_WIDTH), device="cuda", dtype=torch.int32)
    sparse_lens = torch.tensor([_SWA_WIDTH], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([_SWA_WIDTH], device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    actual = prims_ts_dsv4_csa(
        query,
        max_fp8,
        max_fp8,
        routes,
        sparse_lens,
        seq_lens_kv,
        cu_seqlens_q,
        max_seq_len_q=1,
        bmm1_scale=1.0,
        bmm2_scale=3.0e38,
    )

    expected = torch.full_like(actual, torch.finfo(torch.bfloat16).max)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


@_REQUIRES_DSV4_CSA_GPU
def test_prims_ts_dsv4_csa_output_scale_ftz() -> None:
    """Match source ``--use_fast_math`` for a subnormal output scale."""

    query = torch.zeros(
        (1, _HEADS, _HEAD_DIM), device="cuda", dtype=torch.float8_e4m3fn
    )
    max_fp8 = torch.full(
        (1, _HEAD_DIM), 448.0, device="cuda", dtype=torch.float8_e4m3fn
    )
    routes = torch.zeros((1, _SWA_WIDTH), device="cuda", dtype=torch.int32)
    sparse_lens = torch.tensor([_SWA_WIDTH], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([_SWA_WIDTH], device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    actual = prims_ts_dsv4_csa(
        query,
        max_fp8,
        max_fp8,
        routes,
        sparse_lens,
        seq_lens_kv,
        cu_seqlens_q,
        max_seq_len_q=1,
        bmm1_scale=1.0,
        bmm2_scale=1.0e-38,
    )

    # Without the source FMUL.FTZ at output_scale / row_sum, the subnormal
    # normalization factor is amplified by the O accumulator and produces a
    # representable non-zero BF16 result.  TRTLLM-gen flushes it before that
    # vector multiply.
    assert torch.equal(actual, torch.zeros_like(actual))
