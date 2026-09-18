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

"""Correctness coverage for the PrimTS DeepSeek V4 sparse MLA kernels.

Covers Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA):
config/scheduler contracts, PyTorch references (exact FP32 softmax and the tiled
E4M3 online-softmax protocol with and without skip correction), CUDA graph
replay, and input validation.  Fixtures use ``[N, 512]`` row pools (page size one).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0a0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0a0",
)

from flashinfer.attention.prims_ts import (
    prims_ts_dsv4_sparse_mla,
    prims_ts_dsv4_sparse_mla_rope_quant,
    prims_ts_dsv4_sparse_mla_rope_quant_ue8m0,
)
import flashinfer.attention.prims_ts.dsv4 as dsv4_module
from flashinfer.attention.prims_ts.dsv4 import _get_compiled_dsv4_sparse_mla
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.config import (
    make_mla_decode_config,
    max_skip_corr_threshold,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.tasks import (
    dsv4_pair_ring_event_plan,
)


_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS DSV4 sparse MLA requires SM100, SM103 or SM107",
)
_FP8 = torch.float8_e4m3fn


def _force_scheduler(monkeypatch: pytest.MonkeyPatch, persistent: bool) -> None:
    """Pin the static or CLC persistent 2CTA variant regardless of shape."""

    monkeypatch.setattr(
        dsv4_module,
        "_dsv4_uses_persistent_scheduler",
        lambda *args, **kwargs: persistent,
    )


_HEADS = 128
_HEAD_DIM = 512
_SWA_WIDTH = 128
_COMPRESS_RATIO = 4


def test_dsv4_async_config_pipeline_contract() -> None:
    """Pin the DSV4 async/resource values derived by the TS config."""

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
    # BMM2 consumes one P[K128] operand per D256 V panel.
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
    # V is four Gather4 warps x eight interleaved page quads x two H256
    # slices, so its CTA-group completion barrier covers 64 KiB.
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

    # One 128-float TMEM scratch per S stage and two disjoint 64-thread named
    # barriers for the W0/W2 and W1/W3 row pairs.
    assert cfg.softmax_exchange_elems == 2 * 128
    assert (cfg.softmax_sync_bar_id, cfg.softmax_sync_threads) == (1, 64)
    # TMEM map: S0/S1=[0,128), softmax stats=[128,192), O panels from 192.
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
        enable_skip_correction=True,
    )
    assert skip_corr_cfg.enable_skip_correction
    # Skip correction is a generic throughput-2CTA specialization; only the
    # threshold bound depends on the P dtype (E4M3: 8, BF16: 64).
    for qkv_dtype in ("bf16", "e4m3"):
        dense_cfg = make_mla_decode_config(
            qkv_dtype=qkv_dtype, enable_skip_correction=True
        )
        assert dense_cfg.enable_skip_correction
        assert not dense_cfg.is_dynamic_token_sparse
    assert max_skip_corr_threshold("e4m3") == 8.0
    assert max_skip_corr_threshold("bf16") == 64.0


def test_dsv4_ue8m0_scale_requires_rope_quant_fusion() -> None:
    """The packed UE8M0 scale format is a variant of the fused RoPE/FP8 epilogue."""

    kwargs = dict(
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        rope_dim=0,
        page_size=1,
        qkv_dtype="e4m3",
        o_dtype="e4m3",
        is_persistent=True,
        is_var_seq=True,
        is_dynamic_token_sparse=True,
        sparse_swa_topk=128,
    )
    with pytest.raises(ValueError, match="UE8M0"):
        make_mla_decode_config(dsv4_uses_ue8m0_scale_o=True, **kwargs)
    cfg = make_mla_decode_config(
        dsv4_fuses_inv_rope_fp8_quant=True, dsv4_uses_ue8m0_scale_o=True, **kwargs
    )
    assert cfg.dsv4_uses_ue8m0_scale_o
    assert not make_mla_decode_config(
        dsv4_fuses_inv_rope_fp8_quant=True, **kwargs
    ).dsv4_uses_ue8m0_scale_o


@pytest.mark.parametrize("k_tiles", (0, 1, 2, 3, 6, 7, 9))
def test_dsv4_pair_ring_ownership_contract(k_tiles: int) -> None:
    """Pin the W9/W12--15 selector-pair hold/reuse/release event order."""

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

    # Inactive SWA slots hold -1 even though the Gather4 tile still reaches
    # them, so the load path must mask them rather than rely on a benign row-0
    # address.  The unselected compressed tail stays zero.
    page_idx_kv = torch.full(
        (int(cu_seqlens_q[-1]), sparse_capacity),
        0,
        dtype=torch.int32,
        device="cuda",
    )
    sparse_lens = torch.empty(page_idx_kv.shape[0], dtype=torch.int32, device="cuda")
    for batch_idx in range(seq_lens_kv.numel()):
        q_begin = int(cu_seqlens_q[batch_idx])
        q_end = int(cu_seqlens_q[batch_idx + 1])
        q_len = q_end - q_begin
        for packed_q in range(q_begin, q_end):
            q = packed_q - q_begin
            raw_visible = int(seq_lens_kv[batch_idx]) - q_len + q + 1
            swa_valid = min(raw_visible, _SWA_WIDTH)
            page_idx_kv[packed_q, :swa_valid] = torch.arange(
                raw_visible - swa_valid, raw_visible, device="cuda", dtype=torch.int32
            )
            page_idx_kv[packed_q, swa_valid:_SWA_WIDTH] = -1
            compressed_count = min(
                raw_visible // compress_ratio, sparse_capacity - _SWA_WIDTH
            )
            page_idx_kv[packed_q, _SWA_WIDTH : _SWA_WIDTH + compressed_count] = (
                torch.arange(compressed_count, device="cuda", dtype=torch.int32)
            )
            sparse_lens[packed_q] = _SWA_WIDTH + compressed_count
    return page_idx_kv, sparse_lens


def _reference_contract_f(
    query: torch.Tensor,
    swa_pool: torch.Tensor,
    compressed_pool: torch.Tensor,
    page_idx_kv: torch.Tensor,
    sparse_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    skip_corr_threshold: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Model the tiled E4M3 online-softmax protocol, including frozen-max skip correction and the 1.75 P scale."""

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
            kv_tiles = [swa_pool[page_idx_kv[packed_q, :swa_valid].long()].float()]
            if active_width > _SWA_WIDTH:
                compressed = page_idx_kv[packed_q, _SWA_WIDTH:active_width]
                compressed = compressed[compressed >= 0]
                for tile_begin in range(0, compressed.numel(), _SWA_WIDTH):
                    tile_page_idx = compressed[tile_begin : tile_begin + _SWA_WIDTH]
                    kv_tiles.append(compressed_pool[tile_page_idx.long()].float())

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
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_prims_ts_dsv4_sparse_mla_skip_correction_argument_validation(
    threshold: float,
) -> None:
    """Reject out-of-domain skip-correction thresholds for E4M3."""

    query = torch.zeros((1, _HEADS, _HEAD_DIM), device="cuda", dtype=_FP8)
    pool = torch.zeros((1, _HEAD_DIM), device="cuda", dtype=_FP8)
    page_idx_kv = torch.zeros((1, _SWA_WIDTH), device="cuda", dtype=torch.int32)
    sparse_lens = torch.tensor([_SWA_WIDTH], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([1], device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="skip_corr_threshold"):
        prims_ts_dsv4_sparse_mla(
            query,
            pool,
            pool,
            page_idx_kv,
            sparse_lens,
            seq_lens_kv,
            cu_seqlens_q,
            max_seq_len_q=1,
            skip_corr_threshold=threshold,
        )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_prims_ts_dsv4_sparse_mla_runtime_launch_shape_reuses_compilation() -> None:
    """Packed B/T/maxQ/Kmax are runtime launch parameters and must not trigger recompilation."""

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
        page_idx_kv, sparse_lens = _make_contract_e_metadata(
            cu_seqlens_q, seq_lens_kv, sparse_capacity=sparse_capacity
        )
        lse = torch.empty((total_q, _HEADS), device="cuda", dtype=torch.float32)
        actual = prims_ts_dsv4_sparse_mla(
            query,
            compressed_pool,
            swa_pool,
            page_idx_kv,
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
            page_idx_kv,
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
    cache_after_first = _get_compiled_dsv4_sparse_mla.cache_info()
    second = run_case(
        q_lens=(2,),
        max_seq_len_q=2,
        sparse_capacity=384,
        seq_lens_kv_values=(516,),
    )
    cache_after_second = _get_compiled_dsv4_sparse_mla.cache_info()
    third = run_case(
        q_lens=(2, 3),
        # Exercise a padded runtime grid row as well as B/maxQ reuse.
        max_seq_len_q=4,
        sparse_capacity=256,
        seq_lens_kv_values=(10, 260),
    )
    cache_after_third = _get_compiled_dsv4_sparse_mla.cache_info()
    fourth = run_case(
        q_lens=(1,),
        max_seq_len_q=1,
        sparse_capacity=192,
        seq_lens_kv_values=(256,),
        # Positive threshold magnitude is a runtime scalar.  Only its sign
        # selects the compiled skip-correction specialization.
        skip_corr_threshold=4.0,
    )
    cache_after_fourth = _get_compiled_dsv4_sparse_mla.cache_info()

    assert cache_after_second.misses == cache_after_first.misses
    assert cache_after_second.hits == cache_after_first.hits + 1
    assert cache_after_third.misses == cache_after_second.misses
    assert cache_after_third.hits == cache_after_second.hits + 1
    assert cache_after_fourth.misses == cache_after_third.misses
    assert cache_after_fourth.hits == cache_after_third.hits + 1
    for actual, lse, expected, expected_lse in (first, second, third, fourth):
        torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=5e-2)
        torch.testing.assert_close(lse, expected_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("persistent", (False, True), ids=("static", "persistent"))
def test_prims_ts_dsv4_hca_r128_padded_workids_preserve_output(
    persistent: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep an HCA R128 Kmax=192 workload exact across padded work tiles on both schedulers."""

    _force_scheduler(monkeypatch, persistent)
    generator = torch.Generator(device="cuda").manual_seed(20260908)
    q_lens = (2, 3)
    total_q = sum(q_lens)
    cu_seqlens_q = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([127, 8192], device="cuda", dtype=torch.int32)
    query = _rand_fp8((total_q, _HEADS, _HEAD_DIM), generator)
    swa_pool = _rand_fp8((8192, _HEAD_DIM), generator)
    compressed_pool = _rand_fp8((8192, _HEAD_DIM), generator)
    page_idx_kv, sparse_lens = _make_contract_e_metadata(
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
        output = prims_ts_dsv4_sparse_mla(
            query,
            compressed_pool,
            swa_pool,
            page_idx_kv,
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
    cache_after_one_padding = _get_compiled_dsv4_sparse_mla.cache_info()
    three_padding_output, three_padding_lse = run(4)
    cache_after_three_padding = _get_compiled_dsv4_sparse_mla.cache_info()

    # B=2,T=5 creates one padded work tile at maxQ=3 and three at maxQ=4;
    # padded tiles must not write any logical query row.
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
        page_idx_kv,
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


_UE8M0_AMAX_EPS = 1.0e-10
_FP32_SCALE_AMAX_EPS = 1.0e-12


def _expected_rope_quant(blocks: torch.Tensor, *, ue8m0: bool):
    """Return ``(scale, dequant)`` for FP32 ``[T, H, 4, 128]`` blocks (FP32 ``amax / 448`` or UE8M0 ``ceil_pow2``)."""

    amax = blocks.abs().amax(dim=-1)
    if ue8m0:
        scale = torch.exp2(
            torch.ceil(torch.log2(amax.clamp_min(_UE8M0_AMAX_EPS) / 448.0))
        )
    else:
        scale = amax.clamp_min(_FP32_SCALE_AMAX_EPS) / 448.0
    dequant = (blocks / scale[..., None]).to(_FP8).float() * scale[..., None]
    return scale, dequant.reshape(blocks.shape[0], _HEADS, _HEAD_DIM)


def _unpack_ue8m0_scale(words: torch.Tensor, total_q: int) -> torch.Tensor:
    """``[16, 8, pad4(T)]`` INT32 words -> FP32 ``[T, H, 4]`` power-of-two scales."""

    shifts = torch.arange(4, dtype=torch.int32, device=words.device) * 8
    exponents = (words[:, :, :total_q].unsqueeze(-1) >> shifts) & 0xFF
    return (
        torch.exp2(exponents.float() - 127.0)
        .permute(2, 0, 1, 3)
        .reshape(total_q, _HEADS, 4)
    )


def _rope_quant_entry(ue8m0: bool):
    return (
        prims_ts_dsv4_sparse_mla_rope_quant_ue8m0
        if ue8m0
        else prims_ts_dsv4_sparse_mla_rope_quant
    )


def _empty_rope_quant_scale(
    total_q: int, *, ue8m0: bool, device="cuda"
) -> torch.Tensor:
    scale_buf_m = (total_q + 3) // 4 * 4
    if ue8m0:
        return torch.empty(
            (_HEADS // 8, 8, scale_buf_m), device=device, dtype=torch.int32
        )
    return torch.empty(
        (_HEADS // 8, 8 * 4, scale_buf_m), device=device, dtype=torch.float32
    )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
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
@pytest.mark.parametrize("persistent", (False, True), ids=("static", "persistent"))
@pytest.mark.parametrize("ue8m0", (False, True), ids=("fp32-scale", "ue8m0-scale"))
def test_prims_ts_dsv4_rope_quant_padded_workids_preserve_output(
    compress_ratio: int,
    sparse_capacity: int,
    seq_lens_kv_values: tuple[int, int],
    pool_rows: int,
    expected_sparse_lens: tuple[int, ...] | None,
    persistent: bool,
    ue8m0: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the CSA/HCA RopeQuant output exact when the padded work-tile count changes."""

    _force_scheduler(monkeypatch, persistent)
    generator = torch.Generator(device="cuda").manual_seed(20260908)
    q_lens = (2, 3)
    total_q = sum(q_lens)
    cu_seqlens_q = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor(seq_lens_kv_values, device="cuda", dtype=torch.int32)
    query = _rand_fp8((total_q, _HEADS, _HEAD_DIM), generator)
    swa_pool = _rand_fp8((pool_rows, _HEAD_DIM), generator)
    compressed_pool = _rand_fp8((pool_rows, _HEAD_DIM), generator)
    page_idx_kv, sparse_lens = _make_contract_e_metadata(
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
        output_scale = _empty_rope_quant_scale(total_q, ue8m0=ue8m0)
        # The pad4(T) scale tail is outside the public logical output.  Give it
        # the same deterministic sentinel so whole-buffer equality is useful.
        output_scale.view(torch.uint8).fill_(0xDC)
        result, result_scale = _rope_quant_entry(ue8m0)(
            query,
            compressed_pool,
            swa_pool,
            page_idx_kv,
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
    cache_after_one_padding = _get_compiled_dsv4_sparse_mla.cache_info()
    three_padding_code, three_padding_scale = run(4, 0xEE)
    cache_after_three_padding = _get_compiled_dsv4_sparse_mla.cache_info()

    # B=2,T=5 gives one padded work tile for maxQ=3 and three for maxQ=4;
    # padded tiles must not affect data.
    assert torch.equal(one_padding_code, three_padding_code)
    assert torch.equal(one_padding_scale, three_padding_scale)
    assert cache_after_three_padding.misses == cache_after_one_padding.misses
    assert cache_after_three_padding.hits == cache_after_one_padding.hits + 1

    expected, _ = _reference_contract_f(
        query,
        swa_pool,
        compressed_pool,
        page_idx_kv,
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
    expected_scale, expected_dequant = _expected_rope_quant(
        expected_blocks, ue8m0=ue8m0
    )
    expected_dequant = expected_dequant.reshape(total_q, _HEADS, 4, 128)

    actual_code = one_padding_code.permute(1, 0, 2, 3).reshape(
        total_q, _HEADS, _HEAD_DIM
    )
    if ue8m0:
        actual_scale = _unpack_ue8m0_scale(one_padding_scale, total_q)
    else:
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
    if ue8m0:
        # Power-of-two scales are exact unless amax / 448 lands within FP32
        # rounding of a power of two, which random data does not produce.
        assert torch.equal(actual_scale, expected_scale)
    else:
        torch.testing.assert_close(actual_scale, expected_scale, atol=1e-5, rtol=1.5e-1)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_prims_ts_dsv4_sparse_mla_packed_two_request_causal_swa_mask() -> None:
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
    page_idx_kv, sparse_lens = _make_contract_e_metadata(
        cu_seqlens_q, seq_lens_kv, sparse_capacity=256
    )
    bmm1_scale = 1.0 / math.sqrt(_HEAD_DIM)
    bmm2_scale = 0.75

    lse = torch.empty((5, _HEADS), device="cuda", dtype=torch.float32)
    actual = prims_ts_dsv4_sparse_mla(
        query,
        compressed_pool,
        swa_pool,
        page_idx_kv,
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
    output_only = prims_ts_dsv4_sparse_mla(
        query,
        compressed_pool,
        swa_pool,
        page_idx_kv,
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
        page_idx_kv,
        sparse_lens,
        cu_seqlens_q,
        seq_lens_kv,
        bmm1_scale,
        bmm2_scale,
    )
    assert torch.isfinite(actual.float()).all()
    assert torch.isfinite(lse).all()
    # E4M3 MMA accumulates at lower precision than the FP32 eager reference.
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=5e-2)
    torch.testing.assert_close(lse, expected_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_prims_ts_dsv4_sparse_mla_packed_multi_work_tile() -> None:
    """Exercise persistent task-state reuse: 260 logical CTAs exceed one resident B200 wave."""

    generator = torch.Generator(device="cuda").manual_seed(20260904)
    cu_seqlens_q = torch.tensor([0, 65, 130], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([131, 130], device="cuda", dtype=torch.int32)
    query = _rand_fp8((130, _HEADS, _HEAD_DIM), generator)
    swa_pool = _rand_fp8((256, _HEAD_DIM), generator)
    compressed_pool = _rand_fp8((256, _HEAD_DIM), generator)
    page_idx_kv, sparse_lens = _make_contract_e_metadata(
        cu_seqlens_q, seq_lens_kv, sparse_capacity=256
    )
    bmm1_scale = 1.0 / math.sqrt(_HEAD_DIM)
    bmm2_scale = 0.75

    lse = torch.empty((130, _HEADS), device="cuda", dtype=torch.float32)
    actual = prims_ts_dsv4_sparse_mla(
        query,
        compressed_pool,
        swa_pool,
        page_idx_kv,
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
        page_idx_kv,
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
        # Lq = 128 SWA slots + floor(visible / R) compressed slots.  The cases
        # cross K-tile pair boundaries up to 9 tiles, including the six-stage
        # selector-ring wraparound; Kmax=192 has a partial final K128 tile.
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
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_prims_ts_dsv4_sparse_mla_sparse_length_pair_boundaries(
    seq_len_kv_value: int,
    sparse_capacity: int,
    expected_sparse_len: int,
) -> None:
    """Keep ceil(Lq/128) K-tile pair boundaries correct for runtime scan lengths and partial final tiles."""

    generator = torch.Generator(device="cuda").manual_seed(
        20260903 + expected_sparse_len
    )
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([seq_len_kv_value], device="cuda", dtype=torch.int32)
    query = _rand_fp8((1, _HEADS, _HEAD_DIM), generator)
    # The raw SWA page index can name the final visible token; use the same pool
    # size for compressed storage to avoid adding a distinct address-boundary
    # variable to this pair-cadence test.
    pool_rows = max(520, seq_len_kv_value)
    swa_pool = _rand_fp8((pool_rows, _HEAD_DIM), generator)
    compressed_pool = _rand_fp8((pool_rows, _HEAD_DIM), generator)
    page_idx_kv, sparse_lens = _make_contract_e_metadata(
        cu_seqlens_q, seq_lens_kv, sparse_capacity=sparse_capacity
    )
    assert int(sparse_lens.item()) == expected_sparse_len
    swa_valid = min(seq_len_kv_value, _SWA_WIDTH)
    assert torch.equal(
        page_idx_kv[0, swa_valid:_SWA_WIDTH],
        torch.full((_SWA_WIDTH - swa_valid,), -1, device="cuda", dtype=torch.int32),
    )

    bmm1_scale = 1.0 / math.sqrt(_HEAD_DIM)
    bmm2_scale = 0.75
    lse = torch.empty((1, _HEADS), device="cuda", dtype=torch.float32)
    actual = prims_ts_dsv4_sparse_mla(
        query,
        compressed_pool,
        swa_pool,
        page_idx_kv,
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
        page_idx_kv,
        sparse_lens,
        cu_seqlens_q,
        seq_lens_kv,
        bmm1_scale,
        bmm2_scale,
    )
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=5e-2)
    torch.testing.assert_close(lse, expected_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_prims_ts_dsv4_sparse_mla_bf16_output_satfinite() -> None:
    """The BF16 epilogue saturates to the largest finite value instead of producing ``inf``."""

    query = torch.zeros(
        (1, _HEADS, _HEAD_DIM), device="cuda", dtype=torch.float8_e4m3fn
    )
    max_fp8 = torch.full(
        (1, _HEAD_DIM), 448.0, device="cuda", dtype=torch.float8_e4m3fn
    )
    page_idx_kv = torch.zeros((1, _SWA_WIDTH), device="cuda", dtype=torch.int32)
    sparse_lens = torch.tensor([_SWA_WIDTH], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([_SWA_WIDTH], device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    actual = prims_ts_dsv4_sparse_mla(
        query,
        max_fp8,
        max_fp8,
        page_idx_kv,
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


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_prims_ts_dsv4_sparse_mla_output_scale_ftz() -> None:
    """A subnormal output scale is flushed to zero before the O multiply."""

    query = torch.zeros(
        (1, _HEADS, _HEAD_DIM), device="cuda", dtype=torch.float8_e4m3fn
    )
    max_fp8 = torch.full(
        (1, _HEAD_DIM), 448.0, device="cuda", dtype=torch.float8_e4m3fn
    )
    page_idx_kv = torch.zeros((1, _SWA_WIDTH), device="cuda", dtype=torch.int32)
    sparse_lens = torch.tensor([_SWA_WIDTH], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([_SWA_WIDTH], device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    actual = prims_ts_dsv4_sparse_mla(
        query,
        max_fp8,
        max_fp8,
        page_idx_kv,
        sparse_lens,
        seq_lens_kv,
        cu_seqlens_q,
        max_seq_len_q=1,
        bmm1_scale=1.0,
        bmm2_scale=1.0e-38,
    )

    # output_scale / row_sum is computed with FTZ; without it the subnormal
    # factor would be amplified by the O accumulator into a non-zero BF16.
    assert torch.equal(actual, torch.zeros_like(actual))


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    ("value", "expected_byte"),
    (
        # 256 / 448 in [0.5, 1): ceil -> 2**0, byte 127; codes hold 256 exactly.
        (256.0, 127),
        # 3 / 448 in [2**-8, 2**-7): ceil -> 2**-7, byte 120; codes hold 384.
        (3.0, 120),
    ),
    ids=("v256", "v3"),
)
def test_prims_ts_dsv4_rope_quant_ue8m0_scale_bytes_exact(
    value: float, expected_byte: int
) -> None:
    """Packed UE8M0 bytes are the biased exponent of ``ceil_pow2(amax / 448)``.

    Zero queries give uniform E4M3 P, and identical V rows make every O entry
    exactly ``value``; an identity cos/sin row keeps the RoPE block exact.  The
    scale and code bytes are therefore fully determined.
    """

    query = torch.zeros((1, _HEADS, _HEAD_DIM), device="cuda", dtype=_FP8)
    pool = torch.full((1, _HEAD_DIM), value, device="cuda", dtype=_FP8)
    page_idx_kv = torch.zeros((1, _SWA_WIDTH), device="cuda", dtype=torch.int32)
    sparse_lens = torch.tensor([_SWA_WIDTH], device="cuda", dtype=torch.int32)
    seq_lens_kv = torch.tensor([_SWA_WIDTH], device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cos_sin = torch.zeros((_SWA_WIDTH, 64), device="cuda", dtype=torch.float32)
    cos_sin[:, :32] = 1.0

    code, words = prims_ts_dsv4_sparse_mla_rope_quant_ue8m0(
        query,
        pool,
        pool,
        page_idx_kv,
        sparse_lens,
        seq_lens_kv,
        cu_seqlens_q,
        cos_sin,
        max_seq_len_q=1,
        bmm1_scale=1.0,
    )
    torch.cuda.synchronize()

    assert words.shape == (_HEADS // 8, 8, 4) and words.dtype == torch.int32
    expected_word = int.from_bytes(bytes([expected_byte] * 4), "little", signed=True)
    assert torch.equal(words[:, :, :1], torch.full_like(words[:, :, :1], expected_word))
    inv_scale = 2.0 ** (127 - expected_byte)
    expected_code = torch.tensor(value * inv_scale, device="cuda").to(_FP8)
    assert torch.equal(
        code.view(torch.uint8),
        torch.full_like(code.view(torch.uint8), int(expected_code.view(torch.uint8))),
    )


# ---------------------------------------------------------------------------
# PrimTS vs FP32 references, CUDA graph replay, FLASHINFER_VALIDATE_INPUTS.
# ---------------------------------------------------------------------------

_ROPE_DIM = 64
_DEFAULT_BMM1_SCALE = 1.0 / math.sqrt(_HEAD_DIM)

# Isolated E4M3 / BF16 rounding-boundary flips between device FP32 math and
# the torch reference: tolerate at most this fraction of elements.
_MAX_MISMATCH_FRACTION = 1e-5
# PrimTS vs the tiled FP32 protocol reference.
_TOL_PROTOCOL_O = dict(atol=1e-2, rtol=5e-2)
_TOL_PROTOCOL_LSE = dict(atol=1e-3, rtol=1e-3)
# PrimTS vs exact FP32 softmax; E4M3 P quantization dominates.
_TOL_EXACT_O = dict(atol=1e-1, rtol=1e-1)
# RopeQuant dequantized values against the quantized PyTorch reference.  The
# kernel and the torch reference (cuBLAS einsum, whose accumulation order
# differs by SM count / architecture, e.g. B200 vs GB200 vs GB300) round a few
# FP32 values on opposite sides of an E4M3 boundary.  One E4M3 mantissa ulp is
# at most 2**-3 == 12.5% of the value, so rtol must exceed 0.125 for such flips
# to count as agreement; _MAX_MISMATCH_FRACTION then only covers larger errors.
_TOL_ROPE_DEQUANT_VS_REF = dict(atol=1e-2, rtol=1.3e-1)


@dataclasses.dataclass(frozen=True)
class Case:
    """One packed DSV4 request batch with physical routing metadata."""

    name: str
    q_lens: tuple[int, ...]
    seq_lens_kv: tuple[int, ...]
    kmax: int
    compress_ratio: int
    rope: bool = False
    # RopeQuant scale format: FP32 ``amax / 448`` (default) or packed UE8M0.
    ue8m0: bool = False
    # Draw sparse_topk_lens below the selector fill while keeping live indices
    # past it, so both kernels must mask by scan width rather than by value.
    truncate_lens: bool = False
    bmm1_scale: float = _DEFAULT_BMM1_SCALE
    bmm2_scale: float = 1.0
    seed: int = 20260920

    def __post_init__(self) -> None:
        if len(self.q_lens) != len(self.seq_lens_kv):
            raise ValueError(f"{self.name}: q_lens/seq_lens_kv length mismatch")
        for q_len, kv_len in zip(self.q_lens, self.seq_lens_kv, strict=False):
            if q_len <= 0 or kv_len < q_len:
                raise ValueError(f"{self.name}: need 0 < q_len <= kv_len")
        if self.kmax < _SWA_WIDTH or self.kmax % 4:
            raise ValueError(f"{self.name}: Kmax must be >= 128 and divisible by 4")
        if self.rope and self.bmm2_scale != 1.0:
            raise ValueError(f"{self.name}: RopeQuant fixtures use bmm2_scale=1.0")
        if self.ue8m0 and not self.rope:
            raise ValueError(f"{self.name}: UE8M0 scales require rope=True")

    @property
    def total_q(self) -> int:
        return sum(self.q_lens)


@dataclasses.dataclass
class Fixture:
    case: Case
    query: torch.Tensor
    swa_pool: torch.Tensor
    compressed_pool: torch.Tensor
    page_idx_kv: torch.Tensor
    sparse_lens: torch.Tensor
    seq_lens_kv: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seq_len_q: int
    positions: torch.Tensor
    cos_sin: Optional[torch.Tensor]


def build_fixture(case: Case, device: torch.device) -> Fixture:
    gen = torch.Generator(device=device).manual_seed(case.seed)
    total_q = case.total_q
    max_kv = max(case.seq_lens_kv)
    # Odd pool sizes with spare rows: gather addresses must never depend on
    # the pool extent.
    swa_rows = max_kv + 37
    compressed_rows = max(1, max_kv // case.compress_ratio) + 13

    def rand_fp8(shape, scale):
        return (
            torch.randn(shape, device=device, dtype=torch.float32, generator=gen)
            * scale
        ).to(_FP8)

    query = rand_fp8((total_q, _HEADS, _HEAD_DIM), 0.5)
    swa_pool = rand_fp8((swa_rows, _HEAD_DIM), 1.0)
    compressed_pool = rand_fp8((compressed_rows, _HEAD_DIM), 1.0)

    page_idx_kv = torch.zeros((total_q, case.kmax), dtype=torch.int32, device=device)
    sparse_lens = torch.empty(total_q, dtype=torch.int32, device=device)
    positions = torch.empty(total_q, dtype=torch.int64, device=device)
    cu = [0]
    for q_len in case.q_lens:
        cu.append(cu[-1] + q_len)
    for batch_idx, (q_len, kv_len) in enumerate(
        zip(case.q_lens, case.seq_lens_kv, strict=False)
    ):
        for q in range(q_len):
            packed_q = cu[batch_idx] + q
            raw_visible = kv_len - q_len + q + 1
            positions[packed_q] = raw_visible - 1
            swa_valid = min(raw_visible, _SWA_WIDTH)
            swa_rows_for_token = torch.randperm(swa_rows, device=device, generator=gen)[
                :swa_valid
            ].to(torch.int32)
            page_idx_kv[packed_q, :swa_valid] = swa_rows_for_token
            page_idx_kv[packed_q, swa_valid:_SWA_WIDTH] = -1
            compressed_count = min(
                raw_visible // case.compress_ratio, case.kmax - _SWA_WIDTH
            )
            if compressed_count > 0:
                page_idx_kv[packed_q, _SWA_WIDTH : _SWA_WIDTH + compressed_count] = (
                    torch.randint(
                        0,
                        compressed_rows,
                        (compressed_count,),
                        device=device,
                        dtype=torch.int32,
                        generator=gen,
                    )
                )
            active = _SWA_WIDTH + compressed_count
            if case.truncate_lens and compressed_count > 0:
                active = int(
                    torch.randint(
                        _SWA_WIDTH, active + 1, (1,), device=device, generator=gen
                    ).item()
                )
            sparse_lens[packed_q] = active

    cos_sin = None
    if case.rope:
        pos = torch.arange(max_kv, device=device, dtype=torch.float32)[:, None]
        dim = torch.arange(32, device=device, dtype=torch.float32)[None, :]
        angles = (pos + 1.0) * (dim + 1.0) * 0.001
        cos_sin = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        cos_sin = cos_sin.contiguous()

    return Fixture(
        case=case,
        query=query,
        swa_pool=swa_pool,
        compressed_pool=compressed_pool,
        page_idx_kv=page_idx_kv,
        sparse_lens=sparse_lens,
        seq_lens_kv=torch.tensor(case.seq_lens_kv, dtype=torch.int32, device=device),
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=device),
        max_seq_len_q=max(case.q_lens),
        positions=positions,
        cos_sin=cos_sin,
    )


def reference_exact(fx: Fixture) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact FP32 softmax over the routed rows; LSE in log2 units."""

    case = fx.case
    total_q = case.total_q
    out = torch.empty(
        (total_q, _HEADS, _HEAD_DIM), dtype=torch.float32, device=fx.query.device
    )
    lse2 = torch.empty((total_q, _HEADS), dtype=torch.float32, device=fx.query.device)
    slot = torch.arange(case.kmax, device=fx.query.device)
    chunk = max(1, (256 * 1152) // case.kmax)
    for begin in range(0, total_q, chunk):
        end = min(total_q, begin + chunk)
        idx = fx.page_idx_kv[begin:end].long()
        lens = fx.sparse_lens[begin:end].long()
        masked = (idx < 0) | (slot[None, :] >= lens[:, None])
        safe = idx.clamp_min(0)
        kv = torch.cat(
            (
                fx.swa_pool[safe[:, :_SWA_WIDTH]].float(),
                fx.compressed_pool[safe[:, _SWA_WIDTH:]].float(),
            ),
            dim=1,
        )
        q = fx.query[begin:end].float()
        scores = torch.einsum("thd,tkd->thk", q, kv) * case.bmm1_scale
        scores = scores.masked_fill(masked[:, None, :], float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        p = torch.exp(scores - lse[..., None])
        out[begin:end] = torch.einsum("thk,tkd->thd", p, kv) * case.bmm2_scale
        lse2[begin:end] = lse / math.log(2.0)
    return out, lse2


def reference_protocol(fx: Fixture, skip_corr_threshold: float):
    """Tiled E4M3 online-softmax reference shared with the PrimTS regression."""

    return _reference_contract_f(
        fx.query,
        fx.swa_pool,
        fx.compressed_pool,
        fx.page_idx_kv,
        fx.sparse_lens,
        fx.cu_seqlens_q,
        fx.seq_lens_kv,
        fx.case.bmm1_scale,
        fx.case.bmm2_scale,
        skip_corr_threshold,
    )


def reference_protocol_batched(fx: Fixture, skip_corr_threshold: float):
    """Vectorized form of ``_reference_contract_f`` for large fixtures.

    Slot ``s`` of token ``t`` belongs to tile ``s // 128``; the SWA tile is
    masked past ``min(raw_visible, 128)`` and every tile past
    ``sparse_topk_lens[t]``.  Fixtures never place ``-1`` inside the
    compressed region, so tiling by slot position matches the loop reference,
    which compacts ``-1`` entries before tiling.
    """

    case = fx.case
    device = fx.query.device
    total_q = case.total_q
    tiles = (case.kmax + _SWA_WIDTH - 1) // _SWA_WIDTH
    scale_log2 = case.bmm1_scale / math.log(2.0)
    adjusted_threshold = (
        skip_corr_threshold / scale_log2 if skip_corr_threshold > 0.0 else 0.0
    )
    p_scale = 1.75 if skip_corr_threshold > 0.0 else 448.0
    ln2 = math.log(2.0)
    neg_max = -torch.finfo(torch.float32).max

    raw_visible = fx.positions + 1
    slot = torch.arange(case.kmax, device=device)
    out = torch.empty((total_q, _HEADS, _HEAD_DIM), dtype=torch.float32, device=device)
    lse = torch.empty((total_q, _HEADS), dtype=torch.float32, device=device)
    chunk = 512
    for begin in range(0, total_q, chunk):
        end = min(total_q, begin + chunk)
        idx = fx.page_idx_kv[begin:end].long()
        lens = fx.sparse_lens[begin:end].long()
        visible = raw_visible[begin:end]
        valid = slot[None, :] < lens[:, None]
        valid[:, :_SWA_WIDTH] &= slot[None, :_SWA_WIDTH] < visible[:, None]
        valid &= idx >= 0
        q = fx.query[begin:end].float()
        row_max = torch.full((end - begin, _HEADS), neg_max, device=device)
        row_sum = torch.zeros((end - begin, _HEADS), device=device)
        row_o = torch.zeros((end - begin, _HEADS, _HEAD_DIM), device=device)
        for tile in range(tiles):
            lo, hi = tile * _SWA_WIDTH, min(case.kmax, (tile + 1) * _SWA_WIDTH)
            tile_valid = valid[:, lo:hi]
            active_rows = tile_valid.any(dim=1)
            if not bool(active_rows.any()):
                continue
            pool = fx.swa_pool if tile == 0 else fx.compressed_pool
            kv = pool[idx[:, lo:hi].clamp_min(0)].float()
            scores = torch.einsum("thd,tkd->thk", q, kv)
            scores = scores.masked_fill(~tile_valid[:, None, :], float("-inf"))
            tile_max = scores.max(dim=-1).values
            candidate = torch.maximum(row_max, tile_max)
            if skip_corr_threshold > 0.0:
                candidate = torch.where(
                    candidate - row_max <= adjusted_threshold, row_max, candidate
                )
            # Inactive rows keep their state exactly (correction 1, p 0).
            candidate = torch.where(active_rows[:, None], candidate, row_max)
            correction = torch.exp((row_max - candidate) * scale_log2 * ln2)
            p = torch.exp((scores - candidate[..., None]) * scale_log2 * ln2) * p_scale
            p = p.masked_fill(~tile_valid[:, None, :], 0.0)
            row_o = row_o * correction[..., None] + torch.einsum(
                "thk,tkd->thd", p.to(_FP8).float(), kv
            )
            row_sum = row_sum * correction + p.sum(dim=-1)
            row_max = candidate
        out[begin:end] = row_o * (case.bmm2_scale / row_sum[..., None])
        lse[begin:end] = torch.log(row_sum / p_scale) / ln2 + row_max * scale_log2
    return out, lse


def apply_inverse_rope(out: torch.Tensor, fx: Fixture) -> torch.Tensor:
    """Rotate D[448:512] of an FP32 ``[T, H, 512]`` tensor like the epilogue."""

    assert fx.cos_sin is not None
    rotated = out.clone()
    rope = rotated[..., _HEAD_DIM - _ROPE_DIM :].reshape(out.shape[0], _HEADS, 32, 2)
    first = rope[..., 0].clone()
    second = rope[..., 1].clone()
    cos_sin = fx.cos_sin.index_select(0, fx.positions)
    cos = cos_sin[:, None, :32]
    sin = cos_sin[:, None, 32:]
    rope[..., 0] = first * cos + second * sin
    rope[..., 1] = second * cos - first * sin
    return rotated


def run_prims_ts(fx: Fixture, *, skip_corr_threshold: float, with_lse: bool):
    """Return ``(o_or_code, scale_or_None, lse_or_None)`` from PrimTS."""

    case = fx.case
    lse = (
        torch.empty((case.total_q, _HEADS), dtype=torch.float32, device=fx.query.device)
        if with_lse
        else None
    )
    common = dict(
        max_seq_len_q=fx.max_seq_len_q,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        skip_corr_threshold=skip_corr_threshold,
        lse=lse,
    )
    if case.rope:
        code, scale = _rope_quant_entry(case.ue8m0)(
            fx.query,
            fx.compressed_pool,
            fx.swa_pool,
            fx.page_idx_kv,
            fx.sparse_lens,
            fx.seq_lens_kv,
            fx.cu_seqlens_q,
            fx.cos_sin,
            **common,
        )
        return code, scale, lse
    out = prims_ts_dsv4_sparse_mla(
        fx.query,
        fx.compressed_pool,
        fx.swa_pool,
        fx.page_idx_kv,
        fx.sparse_lens,
        fx.seq_lens_kv,
        fx.cu_seqlens_q,
        **common,
    )
    return out, None, lse


def dequant_prims_ts(code: torch.Tensor, scale: torch.Tensor, total_q: int):
    """PrimTS ``[16, T, 8, 512]`` code + FP32 ``[16, 32, pad4(T)]`` or packed
    UE8M0 INT32 ``[16, 8, pad4(T)]`` scale -> ``(dequant [T, H, 512], scale [T, H, 4])``."""

    values = code.permute(1, 0, 2, 3).reshape(total_q, _HEADS, 4, 128).float()
    if scale.dtype == torch.int32:
        scales = _unpack_ue8m0_scale(scale, total_q)
    else:
        scales = (
            scale[:, :, :total_q]
            .reshape(_HEADS // 8, 8, 4, total_q)
            .permute(3, 0, 1, 2)
            .reshape(total_q, _HEADS, 4)
        )
    return (values * scales[..., None]).reshape(total_q, _HEADS, _HEAD_DIM), scales


def compare_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.float()
    expected = expected.float()
    diff = (actual - expected).abs()
    denom = expected.abs().clamp_min(1e-3)
    return {
        "max_abs": float(diff.max()),
        "max_rel": float((diff / denom).max()),
        "mean_abs": float(diff.mean()),
        "bitwise": bool(torch.equal(actual, expected)),
    }


class _Checker:
    """Collect every tolerance violation instead of stopping at the first."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def close(
        self,
        label: str,
        actual,
        expected,
        *,
        atol: float,
        rtol: float,
        max_mismatch_fraction: float = 0.0,
    ) -> None:
        actual = actual.float()
        expected = expected.float()
        diff = (actual - expected).abs()
        bad = diff > atol + rtol * expected.abs()
        n_bad = int(bad.sum())
        allowed = int(math.floor(max_mismatch_fraction * bad.numel()))
        if n_bad > allowed:
            self.violations.append(
                f"{label}: {n_bad}/{bad.numel()} elements outside "
                f"atol={atol} rtol={rtol} (allowed {allowed}); "
                f"max_abs={float(diff.max()):.6g}"
            )

    def true(self, label: str, ok: bool) -> None:
        if not ok:
            self.violations.append(label)


def check_case(case: Case, device: torch.device, *, strict: bool = True) -> dict:
    """Compare PrimTS with the PyTorch references and return the metrics.

    With ``strict`` the first violation raises ``AssertionError`` after all
    metrics have been computed; otherwise violations are returned under
    ``metrics["violations"]``.
    """

    fx = build_fixture(case, device)
    total_q = case.total_q
    metrics: dict = {"case": case.name, "total_q": total_q}
    chk = _Checker()
    exact_o, exact_lse = reference_exact(fx)
    proto0_o, proto0_lse = reference_protocol_batched(fx, 0.0)

    if not case.rope:
        ts0, _, ts0_lse = run_prims_ts(fx, skip_corr_threshold=0.0, with_lse=True)
        ts8, _, ts8_lse = run_prims_ts(fx, skip_corr_threshold=8.0, with_lse=True)
        torch.cuda.synchronize()
        chk.true("ts0 finite", bool(torch.isfinite(ts0.float()).all()))
        chk.true("ts0 lse finite", bool(torch.isfinite(ts0_lse).all()))
        chk.true("ts8 finite", bool(torch.isfinite(ts8.float()).all()))
        chk.true("ts8 lse finite", bool(torch.isfinite(ts8_lse).all()))
        proto8_o, proto8_lse = reference_protocol_batched(fx, 8.0)
        metrics["ts0_vs_proto0"] = compare_metrics(ts0, proto0_o)
        metrics["ts0_lse_vs_proto0"] = compare_metrics(ts0_lse, proto0_lse)
        metrics["ts8_vs_proto8"] = compare_metrics(ts8, proto8_o)
        metrics["ts8_lse_vs_proto8"] = compare_metrics(ts8_lse, proto8_lse)
        metrics["ts0_vs_exact"] = compare_metrics(ts0, exact_o)
        metrics["ts8_vs_exact"] = compare_metrics(ts8, exact_o)
        metrics["ts0_lse_vs_exact"] = compare_metrics(ts0_lse, exact_lse)
        # Allow isolated rounding differences between device math and the
        # tiled FP32 emulation.
        chk.close(
            "ts0 vs proto0",
            ts0,
            proto0_o,
            **_TOL_PROTOCOL_O,
            max_mismatch_fraction=_MAX_MISMATCH_FRACTION,
        )
        chk.close("ts0 lse vs proto0", ts0_lse, proto0_lse, **_TOL_PROTOCOL_LSE)
        chk.close(
            "ts8 vs proto8",
            ts8,
            proto8_o,
            **_TOL_PROTOCOL_O,
            max_mismatch_fraction=_MAX_MISMATCH_FRACTION,
        )
        chk.close("ts8 lse vs proto8", ts8_lse, proto8_lse, **_TOL_PROTOCOL_LSE)
        chk.close("ts0 vs exact", ts0.float(), exact_o, **_TOL_EXACT_O)
        metrics["violations"] = chk.violations
        if strict and chk.violations:
            raise AssertionError("; ".join(chk.violations))
        return metrics

    ts_code, ts_scale, ts_lse = run_prims_ts(fx, skip_corr_threshold=0.0, with_lse=True)
    torch.cuda.synchronize()
    ts_dq, ts_scales = dequant_prims_ts(ts_code, ts_scale, total_q)
    chk.true("ts dequant finite", bool(torch.isfinite(ts_dq).all()))
    chk.true("ts scale positive", bool((ts_scales > 0).all()))
    chk.true("ts scale finite", bool(torch.isfinite(ts_scales).all()))
    chk.true("ts lse finite", bool(torch.isfinite(ts_lse).all()))

    # Apply inverse RoPE and D128 block quantization to the FP32 reference.
    expected = apply_inverse_rope(proto0_o, fx)
    blocks = expected.reshape(total_q, _HEADS, 4, 128)
    ts_expected_scale, ts_expected_dq = _expected_rope_quant(blocks, ue8m0=case.ue8m0)

    metrics["ts_dq_vs_expected"] = compare_metrics(ts_dq, ts_expected_dq)
    metrics["ts_dq_vs_exact_rot"] = compare_metrics(
        ts_dq, apply_inverse_rope(exact_o, fx)
    )
    metrics["ts_scale_vs_expected"] = compare_metrics(ts_scales, ts_expected_scale)
    metrics["ts_lse_vs_proto0"] = compare_metrics(ts_lse, proto0_lse)

    chk.close(
        "ts dq vs expected",
        ts_dq,
        ts_expected_dq,
        **_TOL_ROPE_DEQUANT_VS_REF,
        max_mismatch_fraction=_MAX_MISMATCH_FRACTION,
    )
    if case.ue8m0:
        # Power-of-two scales match exactly except where the device amax and
        # the reference amax straddle a power of two after / 448.
        chk.close(
            "ts scale vs expected",
            ts_scales,
            ts_expected_scale,
            atol=0.0,
            rtol=0.0,
            max_mismatch_fraction=_MAX_MISMATCH_FRACTION,
        )
    else:
        chk.close(
            "ts scale vs expected", ts_scales, ts_expected_scale, atol=1e-5, rtol=1.5e-1
        )
    chk.close("ts lse vs proto0", ts_lse, proto0_lse, **_TOL_PROTOCOL_LSE)
    metrics["violations"] = chk.violations
    if strict and chk.violations:
        raise AssertionError("; ".join(chk.violations))
    return metrics


CASES = (
    Case("b1_q1_kv1_k128", (1,), (1,), 128, 4),
    Case("b1_q1_kv3_k192_r4", (1,), (3,), 192, 4),
    Case("b1_q7_kv7_k132", (7,), (7,), 132, 4),
    Case("b3_ragged_kv_around_128", (1, 3, 5), (5, 130, 127), 196, 4),
    Case("b2_q129_odd", (129, 1), (129, 300), 260, 4),
    Case("b1_q1_kv40001_r128_k516", (1,), (40001,), 516, 128),
    Case("b5_odd_kv_r3", (2, 3, 1, 4, 7), (1001, 999, 1, 777, 4097), 384, 3),
    Case("b1_q65_kv8191_csa_k1152", (65,), (8191,), 1152, 4),
    Case("b2_truncated_lens", (3, 3), (2000, 3001), 640, 8, truncate_lens=True),
    Case("b1_scales", (2,), (600,), 256, 4, bmm1_scale=0.05, bmm2_scale=0.75),
    Case("b8_q1_kv_ladder", (1,) * 8, (1, 2, 127, 128, 129, 255, 256, 4095), 384, 4),
    Case("b2_q257_two_tiles", (257, 5), (1024, 2049), 768, 4),
    Case("rope_b1_q1_kv1_k128", (1,), (1,), 128, 4, rope=True),
    Case("rope_b3_ragged", (1, 3, 5), (5, 130, 127), 196, 4, rope=True),
    Case("rope_b1_kv40001_r128", (1,), (40001,), 516, 128, rope=True),
    Case(
        "rope_b2_truncated", (3, 3), (2000, 3001), 640, 8, rope=True, truncate_lens=True
    ),
    Case(
        "rope_b5_odd_kv_r3",
        (2, 3, 1, 4, 7),
        (1001, 999, 1, 777, 4097),
        384,
        3,
        rope=True,
    ),
    Case("rope_b1_q65_csa_k1152", (65,), (8191,), 1152, 4, rope=True),
    Case(
        "rope_b8_q1_kv_ladder",
        (1,) * 8,
        (1, 2, 127, 128, 129, 255, 256, 4095),
        384,
        4,
        rope=True,
    ),
    Case("ue8m0_b1_q1_kv1_k128", (1,), (1,), 128, 4, rope=True, ue8m0=True),
    Case("ue8m0_b3_ragged", (1, 3, 5), (5, 130, 127), 196, 4, rope=True, ue8m0=True),
    Case("ue8m0_b1_kv40001_r128", (1,), (40001,), 516, 128, rope=True, ue8m0=True),
    Case(
        "ue8m0_b5_odd_kv_r3",
        (2, 3, 1, 4, 7),
        (1001, 999, 1, 777, 4097),
        384,
        3,
        rope=True,
        ue8m0=True,
    ),
    Case(
        "ue8m0_b2_truncated",
        (3, 3),
        (2000, 3001),
        640,
        8,
        rope=True,
        ue8m0=True,
        truncate_lens=True,
    ),
    Case("ue8m0_b1_q65_csa_k1152", (65,), (8191,), 1152, 4, rope=True, ue8m0=True),
)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_prims_ts_dsv4_sparse_mla_reference(case: Case) -> None:
    check_case(case, torch.device("cuda"))


# 160/320-request batches (static vs persistent grid) and long packed contexts.
LARGE_CASES = (
    Case(
        "large_b320_q1_csa_k1152",
        (1,) * 320,
        tuple(129 + (i * 613) % 20000 for i in range(320)),
        1152,
        4,
    ),
    Case(
        "large_b160_q4_hca_rope_k384",
        (4,) * 160,
        tuple(4 + (i * 1237) % 40000 for i in range(160)),
        384,
        128,
        rope=True,
    ),
    Case("long_ctx_b2_q4096_csa_k1152", (4096, 4096), (4096, 4096), 1152, 4),
    Case("long_ctx_b1_q8192_hca_rope_k192", (8192,), (8192,), 192, 128, rope=True),
    Case(
        "large_b160_q4_hca_ue8m0_k384",
        (4,) * 160,
        tuple(4 + (i * 1237) % 40000 for i in range(160)),
        384,
        128,
        rope=True,
        ue8m0=True,
    ),
    Case(
        "long_ctx_b1_q8192_csa_ue8m0_k1152",
        (8192,),
        (8192,),
        1152,
        4,
        rope=True,
        ue8m0=True,
    ),
)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_reference_protocol_batched_matches_loop() -> None:
    case = Case("ref_equiv", (1, 3, 5), (5, 130, 127), 196, 4, truncate_lens=True)
    fx = build_fixture(case, torch.device("cuda"))
    for threshold in (0.0, 8.0):
        loop_o, loop_lse = reference_protocol(fx, threshold)
        fast_o, fast_lse = reference_protocol_batched(fx, threshold)
        torch.testing.assert_close(fast_o, loop_o, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(fast_lse, loop_lse, atol=1e-5, rtol=1e-5)


@pytest.mark.long_running
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("case", LARGE_CASES, ids=[c.name for c in LARGE_CASES])
def test_prims_ts_dsv4_sparse_mla_reference_large(case: Case) -> None:
    check_case(case, torch.device("cuda"))


_GRAPH_SKIP_CORR_THRESHOLD = 8.0


def _run_ts_into(fx: Fixture, out, scale, lse):
    case = fx.case
    common = dict(
        max_seq_len_q=fx.max_seq_len_q,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        skip_corr_threshold=_GRAPH_SKIP_CORR_THRESHOLD,
        out=out,
        lse=lse,
    )
    if case.rope:
        return _rope_quant_entry(case.ue8m0)(
            fx.query,
            fx.compressed_pool,
            fx.swa_pool,
            fx.page_idx_kv,
            fx.sparse_lens,
            fx.seq_lens_kv,
            fx.cu_seqlens_q,
            fx.cos_sin,
            out_scale=scale,
            **common,
        )
    return prims_ts_dsv4_sparse_mla(
        fx.query,
        fx.compressed_pool,
        fx.swa_pool,
        fx.page_idx_kv,
        fx.sparse_lens,
        fx.seq_lens_kv,
        fx.cu_seqlens_q,
        **common,
    )


def _assert_graph_output_matches_reference(fx: Fixture, out, out_scale, lse) -> None:
    """Check replay output against PyTorch ground truth for the current inputs."""

    expected, expected_lse = reference_protocol_batched(fx, _GRAPH_SKIP_CORR_THRESHOLD)
    chk = _Checker()
    if fx.case.rope:
        assert out_scale is not None
        actual, actual_scale = dequant_prims_ts(out, out_scale, fx.case.total_q)
        blocks = apply_inverse_rope(expected, fx).reshape(
            fx.case.total_q, _HEADS, 4, 128
        )
        expected_scale, expected = _expected_rope_quant(blocks, ue8m0=fx.case.ue8m0)
        chk.true("graph scale finite", bool(torch.isfinite(actual_scale).all()))
        chk.true("graph scale positive", bool((actual_scale > 0).all()))
        if fx.case.ue8m0:
            chk.close(
                "graph scale vs reference",
                actual_scale,
                expected_scale,
                atol=0.0,
                rtol=0.0,
                max_mismatch_fraction=_MAX_MISMATCH_FRACTION,
            )
        else:
            chk.close(
                "graph scale vs reference",
                actual_scale,
                expected_scale,
                atol=1e-5,
                rtol=1.5e-1,
            )
        tolerance = _TOL_ROPE_DEQUANT_VS_REF
    else:
        actual = out.float()
        tolerance = _TOL_PROTOCOL_O

    chk.true("graph output finite", bool(torch.isfinite(actual).all()))
    chk.true("graph lse finite", bool(torch.isfinite(lse).all()))
    chk.close(
        "graph output vs reference",
        actual,
        expected,
        **tolerance,
        max_mismatch_fraction=_MAX_MISMATCH_FRACTION,
    )
    chk.close("graph lse vs reference", lse, expected_lse, **_TOL_PROTOCOL_LSE)
    assert not chk.violations, f"{fx.case.name}: {'; '.join(chk.violations)}"


_OUTPUT_MODE_PARAMS = (
    pytest.param(False, False, id="bf16"),
    pytest.param(True, False, id="rope-quant"),
    pytest.param(True, True, id="rope-quant-ue8m0"),
)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(("rope", "ue8m0"), _OUTPUT_MODE_PARAMS)
def test_prims_ts_dsv4_sparse_mla_cuda_graph_replay(rope: bool, ue8m0: bool) -> None:
    """Captured launches must read live inputs and match the PyTorch reference."""

    device = torch.device("cuda")
    case = Case("graph", (1, 3, 5), (5, 130, 4097), 260, 4, rope=rope, ue8m0=ue8m0)
    fx = build_fixture(case, device)
    # First eager call compiles and caches the Gather4 descriptor pair for
    # these pool buffers; capture must not allocate or copy on the host.
    eager = _run_ts_into(fx, None, None, None)
    if rope:
        code, scale = eager
        out = torch.empty_like(code)
        out_scale = torch.empty_like(scale)
    else:
        out = torch.empty_like(eager)
        out_scale = None
    lse = torch.empty((case.total_q, _HEADS), dtype=torch.float32, device=device)
    _run_ts_into(fx, out, out_scale, lse)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        _run_ts_into(fx, out, out_scale, lse)
    torch.cuda.synchronize()

    for seed_offset in (1, 2):
        # New payloads and page_idx_kv of identical shape, written into the same
        # buffers the graph captured.
        fresh = build_fixture(
            dataclasses.replace(case, seed=case.seed + seed_offset), device
        )
        fx.query.copy_(fresh.query)
        fx.swa_pool.copy_(fresh.swa_pool)
        fx.compressed_pool.copy_(fresh.compressed_pool)
        fx.page_idx_kv.copy_(fresh.page_idx_kv)
        fx.sparse_lens.copy_(fresh.sparse_lens)
        if rope:
            out.fill_(0)
            out_scale.fill_(0)
        else:
            out.fill_(0)
        lse.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        _assert_graph_output_matches_reference(fresh, out, out_scale, lse)


def _validated_call(fx: Fixture, **overrides):
    kwargs = dict(
        query=fx.query,
        compressed_kv_pool=fx.compressed_pool,
        sliding_window_kv_pool=fx.swa_pool,
        ptr_page_idx_kv=fx.page_idx_kv,
        ptr_sparse_mla_topk_lens=fx.sparse_lens,
        ptr_seq_lens_kv=fx.seq_lens_kv,
        ptr_cum_seq_lens_q=fx.cu_seqlens_q,
        max_seq_len_q=fx.max_seq_len_q,
    )
    kwargs.update(overrides)
    return prims_ts_dsv4_sparse_mla(**kwargs)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_prims_ts_dsv4_sparse_mla_validate_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FLASHINFER_VALIDATE_INPUTS=1 rejects inconsistent metadata and is a no-op otherwise."""

    device = torch.device("cuda")
    case = Case("validate", (2, 3), (300, 129), 260, 4)
    fx = build_fixture(case, device)
    monkeypatch.delenv("FLASHINFER_VALIDATE_INPUTS", raising=False)
    baseline = _validated_call(fx)
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    validated = _validated_call(fx)
    assert torch.equal(validated, baseline)

    i32 = dict(dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="start at zero and end at T"):
        _validated_call(fx, ptr_cum_seq_lens_q=torch.tensor([1, 2, 5], **i32))
    with pytest.raises(ValueError, match="start at zero and end at T"):
        _validated_call(fx, ptr_cum_seq_lens_q=torch.tensor([0, 2, 4], **i32))
    with pytest.raises(ValueError, match="at least one Q row"):
        _validated_call(
            fx,
            ptr_cum_seq_lens_q=torch.tensor([0, 2, 2, 5], **i32),
            ptr_seq_lens_kv=torch.tensor([300, 300, 129], **i32),
        )
    with pytest.raises(ValueError, match="raw lengths"):
        _validated_call(fx, ptr_seq_lens_kv=torch.tensor([300, 2], **i32))
    # Rows past B * max_seq_len_q would never be scheduled.
    with pytest.raises(ValueError, match="max_seq_len_q must be at least"):
        _validated_call(fx, max_seq_len_q=max(case.q_lens) - 1)
    too_short = fx.sparse_lens.clone()
    too_short[0] = _SWA_WIDTH - 1
    with pytest.raises(ValueError, match=r"\[128, Kmax\]"):
        _validated_call(fx, ptr_sparse_mla_topk_lens=too_short)
    too_long = fx.sparse_lens.clone()
    too_long[-1] = case.kmax + 4
    with pytest.raises(ValueError, match=r"\[128, Kmax\]"):
        _validated_call(fx, ptr_sparse_mla_topk_lens=too_long)

    # Route bounds: the raw Gather4 tensor maps cannot catch out-of-range rows.
    swa_rows = int(fx.swa_pool.shape[0])
    compressed_rows = int(fx.compressed_pool.shape[0])
    swa_out_of_range = fx.page_idx_kv.clone()
    swa_out_of_range[0, 0] = swa_rows
    with pytest.raises(ValueError, match="SWA slots"):
        _validated_call(fx, ptr_page_idx_kv=swa_out_of_range)
    swa_negative = fx.page_idx_kv.clone()
    swa_negative[0, 0] = -2  # only -1 marks an inactive SWA slot
    with pytest.raises(ValueError, match="SWA slots"):
        _validated_call(fx, ptr_page_idx_kv=swa_negative)
    assert int(fx.sparse_lens.min()) > _SWA_WIDTH  # slot 128 is reachable
    compressed_out_of_range = fx.page_idx_kv.clone()
    compressed_out_of_range[1, _SWA_WIDTH] = compressed_rows
    with pytest.raises(ValueError, match="compressed slots"):
        _validated_call(fx, ptr_page_idx_kv=compressed_out_of_range)
    compressed_negative = fx.page_idx_kv.clone()
    compressed_negative[1, _SWA_WIDTH] = -1
    with pytest.raises(ValueError, match="compressed slots"):
        _validated_call(fx, ptr_page_idx_kv=compressed_negative)
    # Slots beyond the scan width are unreachable and may hold anything.
    assert int(fx.sparse_lens.max()) < case.kmax
    unreachable_garbage = fx.page_idx_kv.clone()
    unreachable_garbage[:, -1] = compressed_rows + 7
    assert torch.equal(
        _validated_call(fx, ptr_page_idx_kv=unreachable_garbage), baseline
    )

    rope_case = dataclasses.replace(case, name="validate_rope", rope=True)
    rfx = build_fixture(rope_case, device)
    with pytest.raises(ValueError, match="does not cover raw KV positions"):
        prims_ts_dsv4_sparse_mla_rope_quant(
            rfx.query,
            rfx.compressed_pool,
            rfx.swa_pool,
            rfx.page_idx_kv,
            rfx.sparse_lens,
            rfx.seq_lens_kv,
            rfx.cu_seqlens_q,
            rfx.cos_sin[: int(rfx.seq_lens_kv.max()) - 1].contiguous(),
            max_seq_len_q=rfx.max_seq_len_q,
        )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(("rope", "ue8m0"), _OUTPUT_MODE_PARAMS)
def test_prims_ts_dsv4_sparse_mla_cuda_graph_replay_reloads_live_metadata(
    rope: bool, ue8m0: bool
) -> None:
    """Replays must honour KV lengths, Q offsets, and scan widths rewritten in place.

    The captured launch bakes in only the buffer pointers and
    ``max_seq_len_q``.  Every variant below keeps the same T, B, and maximum
    KV length (so the pool, page-index, and cos/sin buffers keep their shapes) but
    redistributes queries across requests, changes each request's raw KV
    length, and shrinks sparse scan widths.  The replay must match the PyTorch
    reference computed from each variant's updated metadata.
    """

    device = torch.device("cuda")
    captured = Case(
        "graph-meta", (6, 2, 4), (4097, 700, 5), 260, 4, rope=rope, ue8m0=ue8m0
    )
    variants = (
        # Fewer local queries than the captured maximum: padded work tiles.
        Case(
            "graph-meta-shift",
            (3, 5, 4),
            (130, 4097, 2049),
            260,
            4,
            rope=rope,
            ue8m0=ue8m0,
            seed=captured.seed + 1,
        ),
        # Scan widths drawn below the selector fill: mask by width, not value.
        Case(
            "graph-meta-truncate",
            (1, 5, 6),
            (7, 260, 4097),
            260,
            4,
            rope=rope,
            ue8m0=ue8m0,
            truncate_lens=True,
            seed=captured.seed + 2,
        ),
    )
    fx = build_fixture(captured, device)
    eager = _run_ts_into(fx, None, None, None)
    if rope:
        code, scale = eager
        out = torch.empty_like(code)
        out_scale = torch.empty_like(scale)
    else:
        out = torch.empty_like(eager)
        out_scale = None
    lse = torch.empty((captured.total_q, _HEADS), dtype=torch.float32, device=device)
    _run_ts_into(fx, out, out_scale, lse)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        _run_ts_into(fx, out, out_scale, lse)
    torch.cuda.synchronize()

    live = (
        "query",
        "swa_pool",
        "compressed_pool",
        "page_idx_kv",
        "sparse_lens",
        "seq_lens_kv",
        "cu_seqlens_q",
    )
    for variant in variants:
        assert variant.total_q == captured.total_q
        assert max(variant.q_lens) <= fx.max_seq_len_q
        fresh = build_fixture(variant, device)
        for name in live:
            target = getattr(fx, name)
            source = getattr(fresh, name)
            assert target.shape == source.shape, name
            target.copy_(source)
        if rope:
            # Same maximum KV length, hence the same position table.
            assert torch.equal(fx.cos_sin, fresh.cos_sin)
        assert not torch.equal(
            fx.seq_lens_kv,
            torch.tensor(captured.seq_lens_kv, dtype=torch.int32, device=device),
        )
        out.fill_(0)
        if out_scale is not None:
            out_scale.fill_(0)
        lse.fill_(0)
        graph.replay()
        torch.cuda.synchronize()

        _assert_graph_output_matches_reference(fresh, out, out_scale, lse)


def _entry_kwargs(fx: Fixture) -> dict:
    kwargs = dict(
        query=fx.query,
        compressed_kv_pool=fx.compressed_pool,
        sliding_window_kv_pool=fx.swa_pool,
        ptr_page_idx_kv=fx.page_idx_kv,
        ptr_sparse_mla_topk_lens=fx.sparse_lens,
        ptr_seq_lens_kv=fx.seq_lens_kv,
        ptr_cum_seq_lens_q=fx.cu_seqlens_q,
        max_seq_len_q=fx.max_seq_len_q,
    )
    if fx.case.rope:
        kwargs["inv_rope_cos_sin_cache"] = fx.cos_sin
    return kwargs


def _call_entry(fx: Fixture, **overrides):
    kwargs = _entry_kwargs(fx)
    kwargs.update(overrides)
    if fx.case.rope:
        return _rope_quant_entry(fx.case.ue8m0)(**kwargs)
    return prims_ts_dsv4_sparse_mla(**kwargs)


def _noncontiguous_like(tensor: torch.Tensor) -> torch.Tensor:
    """Return a same-shape, same-dtype view whose memory order is transposed."""

    swapped = torch.empty(
        (tensor.shape[1], tensor.shape[0], *tensor.shape[2:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    view = swapped.transpose(0, 1)
    assert view.shape == tensor.shape and not view.is_contiguous()
    return view


_I32 = torch.int32
# Argument mutations shared by both public entry points.  Each entry is
# (id, overrides(fx) -> kwargs, exception type, message regex).
_COMMON_INVALID_ARGUMENTS = (
    (
        "query-not-tensor",
        lambda fx: dict(query=None),
        TypeError,
        "query must be a torch.Tensor",
    ),
    (
        "query-cpu",
        lambda fx: dict(query=fx.query.cpu()),
        ValueError,
        "query must be a CUDA tensor",
    ),
    (
        "query-dtype",
        lambda fx: dict(query=fx.query.to(torch.bfloat16)),
        TypeError,
        "query must have dtype",
    ),
    (
        "query-rank",
        lambda fx: dict(query=fx.query.reshape(fx.case.total_q, -1)),
        ValueError,
        "query must have rank 3",
    ),
    (
        "query-heads",
        lambda fx: dict(query=fx.query[:, : _HEADS // 2].contiguous()),
        ValueError,
        r"query must have shape \[T, 128, 512\]",
    ),
    (
        "query-noncontiguous",
        lambda fx: dict(query=_noncontiguous_like(fx.query)),
        ValueError,
        "query must be contiguous",
    ),
    (
        "query-empty",
        lambda fx: dict(query=fx.query[:0]),
        ValueError,
        "at least one packed Q row",
    ),
    (
        "max-seq-len-q-zero",
        lambda fx: dict(max_seq_len_q=0),
        ValueError,
        "max_seq_len_q must be a positive Python int",
    ),
    (
        "max-seq-len-q-float",
        lambda fx: dict(max_seq_len_q=float(fx.max_seq_len_q)),
        ValueError,
        "max_seq_len_q must be a positive Python int",
    ),
    (
        "compressed-pool-dtype",
        lambda fx: dict(compressed_kv_pool=fx.compressed_pool.to(torch.bfloat16)),
        TypeError,
        "compressed_kv_pool must have dtype",
    ),
    (
        "compressed-pool-head-dim",
        lambda fx: dict(
            compressed_kv_pool=fx.compressed_pool[:, : _HEAD_DIM // 2].contiguous()
        ),
        ValueError,
        r"compressed_kv_pool must have shape \[N>0, 512\]",
    ),
    (
        "swa-pool-rank",
        lambda fx: dict(sliding_window_kv_pool=fx.swa_pool.view(-1)),
        ValueError,
        "sliding_window_kv_pool must have rank 2",
    ),
    (
        "swa-pool-empty",
        lambda fx: dict(sliding_window_kv_pool=fx.swa_pool[:0]),
        ValueError,
        r"sliding_window_kv_pool must have shape \[N>0, 512\]",
    ),
    (
        "page_idx_kv-dtype",
        lambda fx: dict(ptr_page_idx_kv=fx.page_idx_kv.to(torch.int64)),
        TypeError,
        "ptr_page_idx_kv must have dtype",
    ),
    (
        "page_idx_kv-rows",
        lambda fx: dict(ptr_page_idx_kv=fx.page_idx_kv[:-1]),
        ValueError,
        r"ptr_page_idx_kv must have shape \[T, Kmax\]",
    ),
    (
        "page_idx_kv-kmax-below-swa",
        lambda fx: dict(
            ptr_page_idx_kv=fx.page_idx_kv[:, : _SWA_WIDTH - 4].contiguous()
        ),
        ValueError,
        "Kmax must be at least 128 and divisible by 4",
    ),
    (
        "page_idx_kv-kmax-not-multiple-of-4",
        lambda fx: dict(
            ptr_page_idx_kv=fx.page_idx_kv[:, : _SWA_WIDTH + 2].contiguous()
        ),
        ValueError,
        "Kmax must be at least 128 and divisible by 4",
    ),
    (
        "sparse-lens-dtype",
        lambda fx: dict(ptr_sparse_mla_topk_lens=fx.sparse_lens.to(torch.int64)),
        TypeError,
        "ptr_sparse_mla_topk_lens must have dtype",
    ),
    (
        "sparse-lens-rank",
        lambda fx: dict(ptr_sparse_mla_topk_lens=fx.sparse_lens[None, :]),
        ValueError,
        "ptr_sparse_mla_topk_lens must have rank 1",
    ),
    (
        "sparse-lens-length",
        lambda fx: dict(ptr_sparse_mla_topk_lens=fx.sparse_lens[:-1]),
        ValueError,
        r"ptr_sparse_mla_topk_lens must have shape \[T\]",
    ),
    (
        "seq-lens-kv-dtype",
        lambda fx: dict(ptr_seq_lens_kv=fx.seq_lens_kv.to(torch.int64)),
        TypeError,
        "ptr_seq_lens_kv must have dtype",
    ),
    (
        "seq-lens-kv-cpu",
        lambda fx: dict(ptr_seq_lens_kv=fx.seq_lens_kv.cpu()),
        ValueError,
        "ptr_seq_lens_kv must be a CUDA tensor",
    ),
    (
        "seq-lens-kv-empty",
        lambda fx: dict(ptr_seq_lens_kv=fx.seq_lens_kv[:0]),
        ValueError,
        r"sequence metadata must have shapes \[B\] and \[B \+ 1\]",
    ),
    (
        "cu-seqlens-q-length",
        lambda fx: dict(
            ptr_cum_seq_lens_q=torch.cat((fx.cu_seqlens_q, fx.cu_seqlens_q[-1:]))
        ),
        ValueError,
        r"sequence metadata must have shapes \[B\] and \[B \+ 1\]",
    ),
    (
        "cu-seqlens-q-rank",
        lambda fx: dict(ptr_cum_seq_lens_q=fx.cu_seqlens_q[None, :]),
        ValueError,
        "ptr_cum_seq_lens_q must have rank 1",
    ),
    (
        "bmm1-scale-inf",
        lambda fx: dict(bmm1_scale=math.inf),
        ValueError,
        "bmm1_scale must be a finite Python number",
    ),
    (
        "bmm2-scale-nan",
        lambda fx: dict(bmm2_scale=math.nan),
        ValueError,
        "bmm2_scale must be a finite Python number",
    ),
    (
        "bmm2-scale-string",
        lambda fx: dict(bmm2_scale="1.0"),
        ValueError,
        "bmm2_scale must be a finite Python number",
    ),
    (
        "skip-correction-needs-positive-bmm1",
        lambda fx: dict(bmm1_scale=0.0, skip_corr_threshold=8.0),
        ValueError,
        "positive skip_corr_threshold requires a positive bmm1_scale",
    ),
    (
        "lse-dtype",
        lambda fx: dict(
            lse=torch.empty(
                (fx.case.total_q, _HEADS), dtype=torch.bfloat16, device=fx.query.device
            )
        ),
        TypeError,
        "lse must have dtype",
    ),
    (
        "lse-shape",
        lambda fx: dict(
            lse=torch.empty(
                (fx.case.total_q, _HEADS // 2),
                dtype=torch.float32,
                device=fx.query.device,
            )
        ),
        ValueError,
        "lse must have shape",
    ),
    (
        "lse-noncontiguous",
        lambda fx: dict(
            lse=_noncontiguous_like(
                torch.empty(
                    (fx.case.total_q, _HEADS),
                    dtype=torch.float32,
                    device=fx.query.device,
                )
            )
        ),
        ValueError,
        "lse must be contiguous",
    ),
)

# Output-buffer mutations for the BF16 [T, H, D] contract.
_BF16_INVALID_ARGUMENTS = (
    (
        "out-dtype",
        lambda fx: dict(
            out=torch.empty(
                (fx.case.total_q, _HEADS, _HEAD_DIM),
                dtype=torch.float32,
                device=fx.query.device,
            )
        ),
        TypeError,
        "out must have dtype",
    ),
    (
        "out-shape",
        lambda fx: dict(
            out=torch.empty(
                (fx.case.total_q + 1, _HEADS, _HEAD_DIM),
                dtype=torch.bfloat16,
                device=fx.query.device,
            )
        ),
        ValueError,
        "out must have shape",
    ),
    (
        "out-noncontiguous",
        lambda fx: dict(
            out=_noncontiguous_like(
                torch.empty(
                    (fx.case.total_q, _HEADS, _HEAD_DIM),
                    dtype=torch.bfloat16,
                    device=fx.query.device,
                )
            )
        ),
        ValueError,
        "out must be contiguous",
    ),
)

# Output-buffer and cache mutations for the fused RoPE/FP8 physical contract.
_ROPE_SHARED_INVALID_ARGUMENTS = (
    (
        "cos-sin-dtype",
        lambda fx: dict(inv_rope_cos_sin_cache=fx.cos_sin.to(torch.bfloat16)),
        TypeError,
        "inv_rope_cos_sin_cache must have dtype",
    ),
    (
        "cos-sin-width",
        lambda fx: dict(inv_rope_cos_sin_cache=fx.cos_sin[:, :32].contiguous()),
        ValueError,
        r"inv_rope_cos_sin_cache must have shape \[max_position>0, 64\]",
    ),
    (
        "cos-sin-empty",
        lambda fx: dict(inv_rope_cos_sin_cache=fx.cos_sin[:0]),
        ValueError,
        r"inv_rope_cos_sin_cache must have shape \[max_position>0, 64\]",
    ),
    (
        "cos-sin-noncontiguous",
        lambda fx: dict(inv_rope_cos_sin_cache=_noncontiguous_like(fx.cos_sin)),
        ValueError,
        "inv_rope_cos_sin_cache must be contiguous",
    ),
    (
        "out-dtype",
        lambda fx: dict(
            out=torch.empty(
                (_HEADS // 8, fx.case.total_q, 8, _HEAD_DIM),
                dtype=torch.bfloat16,
                device=fx.query.device,
            )
        ),
        TypeError,
        "out must have dtype",
    ),
    (
        "out-logical-layout",
        # The logical [T, H, D] layout is not the physical grouped contract.
        lambda fx: dict(
            out=torch.empty(
                (fx.case.total_q, _HEADS, _HEAD_DIM),
                dtype=_FP8,
                device=fx.query.device,
            )
        ),
        ValueError,
        "out must have rank 4",
    ),
    (
        "out-shape",
        lambda fx: dict(
            out=torch.empty(
                (_HEADS // 8, fx.case.total_q + 1, 8, _HEAD_DIM),
                dtype=_FP8,
                device=fx.query.device,
            )
        ),
        ValueError,
        "out must have shape",
    ),
)

_FP32_SCALE_INVALID_ARGUMENTS = (
    (
        "out-scale-dtype",
        lambda fx: dict(
            out_scale=torch.empty(
                (_HEADS // 8, 32, (fx.case.total_q + 3) // 4 * 4),
                dtype=torch.bfloat16,
                device=fx.query.device,
            )
        ),
        TypeError,
        "out_scale must have dtype",
    ),
    (
        "out-scale-unpadded",
        # The scale buffer must carry pad4(T) columns, not exactly T.
        lambda fx: dict(
            out_scale=torch.empty(
                (_HEADS // 8, 32, fx.case.total_q),
                dtype=torch.float32,
                device=fx.query.device,
            )
        ),
        ValueError,
        "out_scale must have shape",
    ),
    (
        "out-scale-rank",
        lambda fx: dict(
            out_scale=torch.empty(
                (_HEADS // 8 * 32, (fx.case.total_q + 3) // 4 * 4),
                dtype=torch.float32,
                device=fx.query.device,
            )
        ),
        ValueError,
        "out_scale must have rank 3",
    ),
)

_UE8M0_SCALE_INVALID_ARGUMENTS = (
    (
        "out-scale-dtype",
        # The FP32 scale layout is not accepted by the UE8M0 entry point.
        lambda fx: dict(
            out_scale=torch.empty(
                (_HEADS // 8, 8, (fx.case.total_q + 3) // 4 * 4),
                dtype=torch.float32,
                device=fx.query.device,
            )
        ),
        TypeError,
        "out_scale must have dtype",
    ),
    (
        "out-scale-fp32-shape",
        lambda fx: dict(
            out_scale=torch.empty(
                (_HEADS // 8, 32, (fx.case.total_q + 3) // 4 * 4),
                dtype=torch.int32,
                device=fx.query.device,
            )
        ),
        ValueError,
        "out_scale must have shape",
    ),
    (
        "out-scale-unpadded",
        lambda fx: dict(
            out_scale=torch.empty(
                (_HEADS // 8, 8, fx.case.total_q),
                dtype=torch.int32,
                device=fx.query.device,
            )
        ),
        ValueError,
        "out_scale must have shape",
    ),
)


def _invalid_argument_params():
    params = []
    for mode in _OUTPUT_MODE_PARAMS:
        rope, ue8m0 = mode.values
        tag = mode.id
        if not rope:
            specific = _BF16_INVALID_ARGUMENTS
        elif ue8m0:
            specific = (
                *_ROPE_SHARED_INVALID_ARGUMENTS,
                *_UE8M0_SCALE_INVALID_ARGUMENTS,
            )
        else:
            specific = (*_ROPE_SHARED_INVALID_ARGUMENTS, *_FP32_SCALE_INVALID_ARGUMENTS)
        for case_id, overrides, exc, message in (
            *_COMMON_INVALID_ARGUMENTS,
            *specific,
        ):
            params.append(
                pytest.param(
                    rope, ue8m0, overrides, exc, message, id=f"{tag}-{case_id}"
                )
            )
    return params


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    ("rope", "ue8m0", "overrides", "exception", "message"),
    _invalid_argument_params(),
)
def test_prims_ts_dsv4_sparse_mla_rejects_invalid_arguments(
    rope: bool,
    ue8m0: bool,
    overrides,
    exception: type[Exception],
    message: str,
) -> None:
    """Both entry points reject malformed tensors, metadata, and buffers eagerly.

    Every rejection here happens on the host before the kernel launches; the
    checks are independent of ``FLASHINFER_VALIDATE_INPUTS`` because they
    read only shapes, dtypes, devices, and Python scalars.
    """

    device = torch.device("cuda")
    # The scale buffer padding case needs T % 4 != 0.
    case = Case("invalid", (2, 3), (300, 129), 260, 4, rope=rope, ue8m0=ue8m0)
    fx = build_fixture(case, device)
    assert case.total_q % 4 != 0
    # The unmodified arguments must be accepted so a failure below is
    # attributable to the mutation alone.
    _call_entry(fx)
    with pytest.raises(exception, match=message):
        _call_entry(fx, **overrides(fx))


def test_prims_ts_dsv4_scheduler_selection_rule() -> None:
    """Static grid within one resident 2CTA wave, CLC persistent beyond it."""

    rule = dsv4_module._dsv4_uses_persistent_scheduler
    assert rule(1, 1, 74) is False
    # Exactly one wave still launches every cluster directly.
    assert rule(2, 37, 74) is False
    assert rule(1, 75, 74) is True
    assert rule(3, 25, 74) is True


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("rope", (False, True), ids=("bf16", "rope-quant"))
def test_prims_ts_dsv4_sparse_mla_static_and_persistent_schedules_match(
    rope: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The static and persistent schedulers must produce identical O, LSE, and scales.

    Uneven request lengths leave padded work tiles in the ``B x maxQ`` grid
    and truncated scan widths vary the K domain per token.
    """

    device = torch.device("cuda")
    case = Case(
        "sched",
        (1, 4, 2, 6),
        (5, 700, 4097, 130),
        260,
        4,
        rope=rope,
        truncate_lens=True,
    )
    fx = build_fixture(case, device)
    # This shape sits inside one resident wave, so the public policy picks the
    # static grid; the persistent variant is exercised by forcing it below.
    resident = dsv4_module._dsv4_max_active_clusters(torch.cuda.current_device())
    assert not dsv4_module._dsv4_uses_persistent_scheduler(
        len(case.q_lens), fx.max_seq_len_q, resident
    )

    results = {}
    for persistent in (False, True):
        _force_scheduler(monkeypatch, persistent)
        lse = torch.empty((case.total_q, _HEADS), dtype=torch.float32, device=device)
        result = _run_ts_into(fx, None, None, lse)
        torch.cuda.synchronize()
        results[persistent] = (result, lse)

    (static_out, static_lse) = results[False]
    (persistent_out, persistent_lse) = results[True]
    if rope:
        static_code, static_scale = static_out
        persistent_code, persistent_scale = persistent_out
        assert torch.equal(static_code, persistent_code)
        assert torch.equal(
            static_scale[:, :, : case.total_q],
            persistent_scale[:, :, : case.total_q],
        )
    else:
        assert torch.equal(static_out, persistent_out)
    assert torch.equal(static_lse, persistent_lse)
    assert torch.isfinite(static_lse).all()


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_prims_ts_dsv4_rope_quant_rejects_int32_flat_output_overflow() -> None:
    """T * 128 * 512 must fit the flat Int32 O extent of the fused epilogue."""

    device = torch.device("cuda")
    limit = dsv4_module._MAX_ROPE_QUANT_TOTAL_Q
    assert limit == 32767
    total_q = limit + 1
    # Uninitialized storage only; validation rejects the shape before any
    # other argument is inspected.
    query = torch.empty((total_q, _HEADS, _HEAD_DIM), dtype=_FP8, device=device)
    dummy = torch.empty((1, _HEAD_DIM), dtype=_FP8, device=device)
    i32 = dict(dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="flat T\\*128\\*512"):
        prims_ts_dsv4_sparse_mla_rope_quant(
            query,
            dummy,
            dummy,
            torch.zeros((total_q, _SWA_WIDTH), **i32),
            torch.full((total_q,), _SWA_WIDTH, **i32),
            torch.tensor([total_q], **i32),
            torch.tensor([0, total_q], **i32),
            torch.empty((1, 64), dtype=torch.float32, device=device),
            max_seq_len_q=total_q,
        )
