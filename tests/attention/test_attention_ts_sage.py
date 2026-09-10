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

"""Sage attention (per-block Q/K scales, per-channel V scales) for PrimTS decode."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl>=4.7.0",
)

from flashinfer.attention.prims_ts import sage as sage_module
from flashinfer.attention.prims_ts.sage import (
    SageAttentionParams,
    flat_scale_numel,
    flat_scale_slot,
    log2_block_size,
    validate_sage_params,
)

from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_constants import (
    FP8_P_QUANT_SCALE,
)
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources import (
    sage_scales,
)

from tests.attention.prims_ts_test_utils import (
    FP8 as _FP8,
    HEAD_DIM as _HEAD_DIM,
    REQUIRES_PRIMTS_GPU as _REQUIRES_PRIMTS_GPU,
    dense_stream_columns,
    heavy_tailed,
    make_sage_decode_config,
    make_sage_params,
)
from tests.attention.sage_quant_reference import (
    dequantize_token_blocks,
    dequantize_v_channels,
    quantize_token_blocks,
    quantize_v_channels,
)


# ---------------------------------------------------------------------------
# Flat scale layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "block_size", "expected"),
    (
        (1, 100, 16, 7),
        (2, 100, 16, 14),
        (1, 1024, 64, 16),
        (3, 1000, 64, 49),
        (2, 64, 1, 129),
    ),
)
def test_flat_scale_numel_matches_trtllm_gen(
    batch_size: int, seq_len: int, block_size: int, expected: int
) -> None:
    """One head owns ``ceil(B * S / blk) + B - 1`` scales."""

    assert flat_scale_numel(batch_size, seq_len, block_size) == expected


@pytest.mark.parametrize("seq_len", (64, 100, 1000))
@pytest.mark.parametrize("block_size", (1, 16, 64))
def test_flat_scale_index_keeps_sequences_apart_and_in_range(
    seq_len: int, block_size: int
) -> None:
    """Every token maps inside the head stride and batches never share a slot."""

    batch_size = 3
    log2_block = log2_block_size(block_size)
    numel = flat_scale_numel(batch_size, seq_len, block_size)
    previous_end = -1
    for batch_idx in range(batch_size):
        first = flat_scale_slot(batch_idx, 0, seq_len, log2_block)
        last = flat_scale_slot(batch_idx, seq_len - 1, seq_len, log2_block)
        assert previous_end < first <= last < numel
        assert last - first == (seq_len - 1) // block_size
        previous_end = last


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "block_size", "batch_idx", "token_idx", "expected"),
    (
        # cumSeqLens[b] // blk + b + t // blk with cumSeqLens[b] = b * S.
        (1, 100, 16, 0, 0, 0),
        (1, 100, 16, 0, 99, 6),
        # 100 // 16 = 6 leaves a ragged block; batch 1 starts at 6 + 1 = 7.
        (2, 100, 16, 1, 0, 7),
        (2, 100, 16, 1, 99, 13),
        # 64 divides 1024: batch 2 starts at 2 * 16 + 2 = 34.
        (3, 1024, 64, 2, 0, 34),
        (3, 1024, 64, 2, 1023, 49),
        # 1000 // 64 = 15 with a 40-token tail; batch 2 starts at 31 + 2 = 33.
        (3, 1000, 64, 2, 999, 48),
        # blk = 1 is the identity plus the batch offset.
        (2, 64, 1, 1, 5, 70),
    ),
)
def test_flat_scale_slot_matches_sage_quant_formula(
    batch_size: int,
    seq_len: int,
    block_size: int,
    batch_idx: int,
    token_idx: int,
    expected: int,
) -> None:
    """Hand-computed slots of TensorRT-LLM's ``sageQuant`` flat layout."""

    slot = flat_scale_slot(batch_idx, token_idx, seq_len, log2_block_size(block_size))
    assert slot == expected
    assert slot < flat_scale_numel(batch_size, seq_len, block_size)


@pytest.mark.parametrize("block_size", (1, 16, 32, 64, 128, 256))
def test_log2_block_size_covers_supported_blocks(block_size: int) -> None:
    assert 1 << log2_block_size(block_size) == block_size


@pytest.mark.parametrize("block_size", (0, 3, 24))
def test_log2_block_size_rejects_other_sizes(block_size: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        log2_block_size(block_size)


# ---------------------------------------------------------------------------
# Kernel-side scale addressing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tile_size_q", (64, 128))
@pytest.mark.parametrize("k_block_size", (16, 32, 64, 128, 256))
def test_sage_scale_arr_size_follows_fragment_groups(
    tile_size_q: int, k_block_size: int
) -> None:
    """Both streamed profiles own four K32 fragments; only ``blk=16`` splits them."""

    cfg = make_sage_decode_config(
        tile_size_q=tile_size_q,
        tile_size_kv=256 if tile_size_q == 64 else 128,
        sage_args={"sage_k_block_size": k_block_size},
    )
    groups = 2 if k_block_size == 16 else 1
    assert cfg.num_softmax_score_fragments == 4
    assert cfg.sage_k_groups_per_fragment == groups
    assert sage_scales.sage_scale_arr_size(cfg) == 4 * groups


# ---------------------------------------------------------------------------
# Reference quantizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (torch.int8, _FP8))
@pytest.mark.parametrize("block_size", (1, 16, 64))
def test_token_block_quantizer_round_trips_within_one_step(
    dtype: torch.dtype, block_size: int
) -> None:
    """Dequantized Q/K stay within one quantization step of the input."""

    torch.manual_seed(0)
    x = torch.randn((2, 100, 3, 16)) * torch.exp(
        (torch.rand((2, 100, 3, 16)) - 0.5) * 3.2
    )
    quantized, scales = quantize_token_blocks(x, block_size=block_size, dtype=dtype)
    assert quantized.dtype == dtype
    assert scales.shape == (3, flat_scale_numel(2, 100, block_size))
    assert (scales > 0).all()
    restored = dequantize_token_blocks(quantized, scales, block_size=block_size)
    step = torch.empty_like(x)
    for batch_idx in range(2):
        for block_begin in range(0, 100, block_size):
            block_end = min(block_begin + block_size, 100)
            flat_idx = flat_scale_slot(
                batch_idx, block_begin, 100, log2_block_size(block_size)
            )
            step[batch_idx, block_begin:block_end] = scales[:, flat_idx][None, :, None]
    if dtype == torch.int8:
        tolerance = 0.5 * step + 1e-6
    else:
        # E4M3 keeps three mantissa bits; the smallest subnormal is 2**-9.
        tolerance = x.abs() * 2.0**-4 + step * 2.0**-10 + 1e-6
    assert ((x - restored).abs() <= tolerance).all()


@pytest.mark.parametrize("smooth", (False, True))
def test_v_channel_quantizer_round_trips(smooth: bool) -> None:
    """Per-channel V scales (optionally with a mean) invert within E4M3 precision."""

    torch.manual_seed(1)
    v = torch.randn((2, 50, 4, 32)) + 3.0
    quantized, v_scale, v_mean = quantize_v_channels(v, smooth=smooth)
    assert quantized.dtype == _FP8
    assert v_scale.shape == (4, 32)
    assert (v_mean is not None) == smooth
    restored = dequantize_v_channels(quantized, v_scale, v_mean)
    centered = v - (v_mean[None, None] if smooth else 0.0)
    tolerance = centered.abs() * 2.0**-4 + v_scale[None, None] * 2.0**-10 + 1e-6
    assert ((v - restored).abs() <= tolerance).all()


# ---------------------------------------------------------------------------
# Host validation of SageAttentionParams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Geometry:
    batch_size: int = 2
    seq_len_q: int = 64
    seq_len_kv: int = 1000
    num_qo_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = _HEAD_DIM
    tile_size_q: int = 64
    q_dtype: torch.dtype = _FP8
    kv_dtype: torch.dtype = _FP8
    out_dtype: torch.dtype = torch.bfloat16


def _make_params(
    geometry: _Geometry,
    *,
    q_block_size: int = 1,
    k_block_size: int = 16,
    with_mean: bool = False,
    device: torch.device | str = "cpu",
) -> SageAttentionParams:
    return make_sage_params(
        batch_size=geometry.batch_size,
        seq_len_q=geometry.seq_len_q,
        seq_len_kv=geometry.seq_len_kv,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        with_mean=with_mean,
        device=device,
    )


def _validate(params: SageAttentionParams, geometry: _Geometry) -> None:
    validate_sage_params(
        params,
        batch_size=geometry.batch_size,
        seq_len_q=geometry.seq_len_q,
        seq_len_kv=geometry.seq_len_kv,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        tile_size_q=geometry.tile_size_q,
        q_dtype=geometry.q_dtype,
        kv_dtype=geometry.kv_dtype,
        out_dtype=geometry.out_dtype,
    )


def test_sage_params_defaults_follow_the_production_recipe() -> None:
    """The dataclass defaults are TensorRT-LLM's ``(1, 16, 1)`` recipe."""

    geometry = _Geometry()
    params = SageAttentionParams(
        q_scale=torch.rand(
            (
                geometry.num_qo_heads,
                flat_scale_numel(geometry.batch_size, geometry.seq_len_q, 1),
            )
        ),
        k_scale=torch.rand(
            (
                geometry.num_kv_heads,
                flat_scale_numel(geometry.batch_size, geometry.seq_len_kv, 16),
            )
        ),
        v_scale=torch.rand((geometry.num_kv_heads, geometry.head_dim)),
    )
    assert params.q_block_size == 1
    assert params.k_block_size == 16
    assert params.k_summary_scale is None
    assert params.v_mean is None
    _validate(params, geometry)


@pytest.mark.parametrize("k_block_size", (16, 32, 64, 128, 256))
@pytest.mark.parametrize("q_block_size", (1, 4, 64))
@pytest.mark.parametrize("with_mean", (False, True))
@pytest.mark.parametrize("out_dtype", (torch.bfloat16, torch.float16))
def test_sage_params_accept_supported_shapes(
    k_block_size: int, q_block_size: int, with_mean: bool, out_dtype: torch.dtype
) -> None:
    geometry = _Geometry(out_dtype=out_dtype)
    _validate(
        _make_params(
            geometry,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            with_mean=with_mean,
        ),
        geometry,
    )


@pytest.mark.parametrize("k_block_size", (1, 4, 8, 48, 512))
def test_sage_params_reject_unsupported_k_block_sizes(k_block_size: int) -> None:
    geometry = _Geometry()
    with pytest.raises(ValueError, match="k_block_size"):
        _validate(_make_params(geometry, k_block_size=k_block_size), geometry)


@pytest.mark.parametrize("q_block_size", (0, 3, 6, 128))
def test_sage_params_reject_unsupported_q_block_sizes(q_block_size: int) -> None:
    """``q_block_size`` must be a power of two no larger than the Q tile."""

    geometry = _Geometry(tile_size_q=64)
    params = replace(_make_params(geometry), q_block_size=q_block_size)
    with pytest.raises(ValueError, match="q_block_size"):
        _validate(params, geometry)


@pytest.mark.parametrize(
    ("q_dtype", "kv_dtype", "out_dtype", "match"),
    (
        (torch.float16, torch.float16, torch.float16, "Q and K"),
        (_FP8, torch.float16, torch.bfloat16, "Q and K"),
        (_FP8, _FP8, torch.float32, "output"),
        (_FP8, _FP8, _FP8, "output"),
    ),
)
def test_sage_params_reject_unsupported_dtypes(
    q_dtype: torch.dtype, kv_dtype: torch.dtype, out_dtype: torch.dtype, match: str
) -> None:
    geometry = _Geometry(q_dtype=q_dtype, kv_dtype=kv_dtype, out_dtype=out_dtype)
    with pytest.raises(ValueError, match=match):
        _validate(_make_params(geometry), geometry)


@pytest.mark.parametrize("kv_dtype", (torch.int8, _FP8))
def test_sage_params_report_int8_as_not_supported_yet(kv_dtype: torch.dtype) -> None:
    """INT8 Q/K names the missing feature instead of a V dtype mismatch."""

    geometry = _Geometry(q_dtype=torch.int8, kv_dtype=kv_dtype)
    with pytest.raises(NotImplementedError, match="INT8 Q/K Sage attention"):
        _validate(_make_params(geometry), geometry)


@pytest.mark.parametrize(
    ("field", "shape", "match"),
    (
        ("q_scale", (8, 130), "q_scale"),
        ("q_scale", (4, 129), "q_scale"),
        ("k_scale", (2, 127), "k_scale"),
        ("k_scale", (8, 126), "k_scale"),
        ("v_scale", (2, 64), "v_scale"),
        ("v_scale", (8, _HEAD_DIM), "v_scale"),
        ("v_mean", (2, 64), "v_mean"),
    ),
)
def test_sage_params_reject_wrong_scale_shapes(
    field: str, shape: tuple[int, ...], match: str
) -> None:
    """Q/K scales use the flat layout ``[H, ceil(B*S/blk) + B - 1]``; V is ``[Hkv, D]``."""

    geometry = _Geometry()
    params = replace(
        _make_params(geometry, with_mean=True), **{field: torch.rand(shape)}
    )
    with pytest.raises(ValueError, match=match):
        _validate(params, geometry)


def test_sage_params_reject_non_fp32_or_strided_scales() -> None:
    geometry = _Geometry()
    params = _make_params(geometry)
    with pytest.raises(ValueError, match="float32"):
        _validate(replace(params, k_scale=params.k_scale.half()), geometry)
    strided = torch.rand((geometry.num_kv_heads, 2 * geometry.head_dim))[:, ::2]
    with pytest.raises(ValueError, match="contiguous"):
        _validate(replace(params, v_scale=strided), geometry)
    # The epilogue reads V scales with 16-byte vector loads.
    misaligned = torch.rand((geometry.num_kv_heads * geometry.head_dim + 1,))[1:].view(
        geometry.num_kv_heads, geometry.head_dim
    )
    with pytest.raises(ValueError, match="16-byte aligned"):
        _validate(replace(params, v_scale=misaligned), geometry)
    with pytest.raises(ValueError, match="k_summary_scale"):
        _validate(replace(params, k_summary_scale=params.k_scale.clone()), geometry)


def test_sage_params_reject_scales_on_another_device() -> None:
    geometry = _Geometry()
    params = _make_params(geometry)
    with pytest.raises(ValueError, match="device"):
        validate_sage_params(
            params,
            batch_size=geometry.batch_size,
            seq_len_q=geometry.seq_len_q,
            seq_len_kv=geometry.seq_len_kv,
            num_qo_heads=geometry.num_qo_heads,
            num_kv_heads=geometry.num_kv_heads,
            head_dim=geometry.head_dim,
            tile_size_q=geometry.tile_size_q,
            q_dtype=geometry.q_dtype,
            kv_dtype=geometry.kv_dtype,
            out_dtype=geometry.out_dtype,
            device=torch.device("cuda", 0),
        )


def test_sage_module_exports() -> None:
    assert set(sage_module.__all__) >= {
        "SageAttentionParams",
        "flat_scale_numel",
        "flat_scale_slot",
        "log2_block_size",
        "validate_sage_params",
    }


# ---------------------------------------------------------------------------
# Dense kernel fidelity and recipe tests
# ---------------------------------------------------------------------------

_NUM_KV_INSTANCES = 2


@dataclass(frozen=True)
class _DenseSageCase:
    """One dense Sage problem: geometry selects the profile, scales the recipe."""

    name: str
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_qo_heads: int
    num_kv_heads: int
    q_block_size: int
    kv_block_size: int
    expected_kv_tile: int
    sage_q_block_size: int
    sage_k_block_size: int
    with_mean: bool
    out_dtype: torch.dtype
    mask_type: str = "dense"

    @property
    def expected_q_tile(self) -> int:
        return 64 if self.expected_kv_tile == 256 else 128


_KV256_MHA = dict(
    batch_size=2,
    seq_len_q=128,
    seq_len_kv=1000,
    num_qo_heads=4,
    num_kv_heads=4,
    q_block_size=64,
    kv_block_size=64,
    expected_kv_tile=256,
)
# Two KV heads with a ratio of eight exercise the KV-head term of the Q-head
# scale index as well as the grouped Q128 tile.
_Q128_GQA = dict(
    batch_size=2,
    seq_len_q=48,
    seq_len_kv=500,
    num_qo_heads=16,
    num_kv_heads=2,
    q_block_size=16,
    kv_block_size=64,
    expected_kv_tile=128,
)

# Four Q heads per KV head fill the Q64 tile with 16 tokens, so the KV256
# profile also runs the ``kv_head * heads_q_per_kv + local_head`` Q-scale index.
_KV256_GQA = dict(
    batch_size=2,
    seq_len_q=64,
    seq_len_kv=1000,
    num_qo_heads=8,
    num_kv_heads=2,
    q_block_size=16,
    kv_block_size=64,
    expected_kv_tile=256,
)

_DENSE_SAGE_CASES = (
    _DenseSageCase(
        "kv256_k16_q1_bf16",
        **_KV256_MHA,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=False,
        out_dtype=torch.bfloat16,
    ),
    _DenseSageCase(
        "kv256_gqa4_k16_q1_bf16",
        **_KV256_GQA,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=False,
        out_dtype=torch.bfloat16,
    ),
    _DenseSageCase(
        "kv256_k16_q1_fp16_mean_causal",
        **_KV256_MHA,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=True,
        out_dtype=torch.float16,
        mask_type="causal",
    ),
    _DenseSageCase(
        "kv256_k64_q16_bf16_mean",
        **_KV256_MHA,
        sage_q_block_size=16,
        sage_k_block_size=64,
        with_mean=True,
        out_dtype=torch.bfloat16,
    ),
    _DenseSageCase(
        "kv256_k256_q64_bf16",
        **_KV256_MHA,
        sage_q_block_size=64,
        sage_k_block_size=256,
        with_mean=False,
        out_dtype=torch.bfloat16,
    ),
    _DenseSageCase(
        "q128_k16_q1_bf16_mean",
        **_Q128_GQA,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=True,
        out_dtype=torch.bfloat16,
    ),
    _DenseSageCase(
        "q128_k32_q4_fp16_causal",
        **_Q128_GQA,
        sage_q_block_size=4,
        sage_k_block_size=32,
        with_mean=False,
        out_dtype=torch.float16,
        mask_type="causal",
    ),
)


def _cases_named(cases, *names: str):
    """Select test cases by name, in the order the names are given."""

    by_name = {case.name: case for case in cases}
    return tuple(by_name[name] for name in names)


def _random_sage_inputs(case: _DenseSageCase, device: torch.device):
    """Random E4M3 Q/K/V with random positive scales in the flat layout."""

    q = torch.randn(
        (case.batch_size, case.seq_len_q, case.num_qo_heads, _HEAD_DIM), device=device
    ).to(_FP8)
    k = torch.randn(
        (case.batch_size, case.seq_len_kv, case.num_kv_heads, _HEAD_DIM),
        device=device,
    ).to(_FP8)
    v = torch.randn_like(k.float()).to(_FP8)

    def random_scale(shape, low, high):
        return (torch.rand(shape, device=device) * (high - low) + low).contiguous()

    q_scale = random_scale(
        (
            case.num_qo_heads,
            flat_scale_numel(case.batch_size, case.seq_len_q, case.sage_q_block_size),
        ),
        0.05,
        0.2,
    )
    k_scale = random_scale(
        (
            case.num_kv_heads,
            flat_scale_numel(case.batch_size, case.seq_len_kv, case.sage_k_block_size),
        ),
        0.5,
        2.0,
    )
    v_scale = random_scale((case.num_kv_heads, _HEAD_DIM), 0.25, 1.0)
    v_mean = (
        torch.randn((case.num_kv_heads, _HEAD_DIM), device=device).contiguous()
        if case.with_mean
        else None
    )
    params = SageAttentionParams(
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        v_mean=v_mean,
        q_block_size=case.sage_q_block_size,
        k_block_size=case.sage_k_block_size,
    )
    return q, k, v, params


@torch.no_grad()
def _sage_dense_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    params: SageAttentionParams,
    *,
    sm_scale: float,
    mask_type: str,
    kv_tile_size: int,
) -> torch.Tensor:
    """FP32 attention on dequantized inputs with the kernel's P448 stream model.

    Scores are ``sfQ * sfK * (Q K^T)``; every online-softmax stream (one per
    K/V instance, and per spatial half for KV256) quantizes its 448-scaled
    probabilities to E4M3 against its own running maximum; the merged output
    is scaled per V channel and shifted by the optional V mean.
    """

    batch_size, seq_len_q, num_qo_heads, head_dim = q.shape
    seq_len_kv, num_kv_heads = k.shape[1], k.shape[2]
    group_size = num_qo_heads // num_kv_heads
    q_real = dequantize_token_blocks(q, params.q_scale, block_size=params.q_block_size)
    k_real = dequantize_token_blocks(k, params.k_scale, block_size=params.k_block_size)
    v_raw = v.float()
    half_columns = dense_stream_columns(kv_tile_size, q.device)
    num_tiles = (seq_len_kv + kv_tile_size - 1) // kv_tile_size
    stream_tiles = [
        (range(instance_idx, num_tiles, _NUM_KV_INSTANCES), columns)
        for instance_idx in range(_NUM_KV_INSTANCES)
        for columns in half_columns
    ]
    output = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for batch_idx in range(batch_size):
        keys = k_real[batch_idx].repeat_interleave(group_size, dim=1)
        values = v_raw[batch_idx].repeat_interleave(group_size, dim=1)
        for query_idx in range(seq_len_q):
            visible_end = seq_len_kv
            if mask_type == "causal":
                visible_end = seq_len_kv - seq_len_q + query_idx + 1
            scores = torch.einsum(
                "hd,thd->ht", q_real[batch_idx, query_idx], keys[:visible_end]
            )
            stream_max = []
            stream_sum = []
            stream_acc = []
            for tile_indices, columns in stream_tiles:
                running_max = None
                running_sum = torch.zeros(num_qo_heads, device=q.device)
                running_acc = torch.zeros((num_qo_heads, head_dim), device=q.device)
                for tile_idx in tile_indices:
                    tile_columns = columns + tile_idx * kv_tile_size
                    tile_columns = tile_columns[tile_columns < visible_end]
                    if tile_columns.numel() == 0:
                        continue
                    tile_scores = scores[:, tile_columns]
                    local_max = tile_scores.max(dim=-1).values
                    new_max = (
                        local_max
                        if running_max is None
                        else torch.maximum(running_max, local_max)
                    )
                    probabilities = (
                        torch.exp((tile_scores - new_max.unsqueeze(-1)) * sm_scale)
                        * FP8_P_QUANT_SCALE
                    )
                    tile_acc = torch.einsum(
                        "ht,thd->hd",
                        probabilities.to(_FP8).float(),
                        values[tile_columns],
                    )
                    if running_max is not None:
                        correction = torch.exp((running_max - new_max) * sm_scale)
                        running_sum = running_sum * correction
                        running_acc = running_acc * correction.unsqueeze(-1)
                    running_sum = running_sum + probabilities.sum(dim=-1)
                    running_acc = running_acc + tile_acc
                    running_max = new_max
                if running_max is not None:
                    stream_max.append(running_max)
                    stream_sum.append(running_sum)
                    stream_acc.append(running_acc)
            final_max = torch.stack(stream_max).max(dim=0).values
            final_sum = torch.zeros_like(final_max)
            final_acc = torch.zeros_like(stream_acc[0])
            for maximum, denominator, accumulator in zip(
                stream_max, stream_sum, stream_acc, strict=True
            ):
                correction = torch.exp((maximum - final_max) * sm_scale)
                final_sum += denominator * correction
                final_acc += accumulator * correction.unsqueeze(-1)
            normalized = final_acc / final_sum.unsqueeze(-1)
            v_scale = params.v_scale.repeat_interleave(group_size, dim=0)
            result = normalized * v_scale
            if params.v_mean is not None:
                result = result + params.v_mean.repeat_interleave(group_size, dim=0)
            output[batch_idx, query_idx] = result
    return output


def _plan_dense_sage(case: _DenseSageCase, params: SageAttentionParams, device):
    from flashinfer.attention.prims_ts import BatchDecodeTSWrapper
    from flashinfer.attention.prims_ts._block_sparse import config as sparse_config

    wrapper = BatchDecodeTSWrapper()
    sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    try:
        wrapper.plan(
            case.batch_size,
            case.seq_len_q,
            case.seq_len_kv,
            case.num_qo_heads,
            case.num_kv_heads,
            _HEAD_DIM,
            case.q_block_size,
            case.kv_block_size,
            device=device,
            use_block_sparse=False,
            mask_type=case.mask_type,
            q_data_type=_FP8,
            kv_data_type=_FP8,
            o_data_type=case.out_dtype,
            sage=params,
        )
    finally:
        sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    policy = dict(wrapper._policy)
    assert policy["tile_size_q"] == case.expected_q_tile
    assert policy["tile_size_kv"] == case.expected_kv_tile
    return wrapper


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _DENSE_SAGE_CASES, ids=lambda case: case.name)
@torch.no_grad()
def test_dense_sage_matches_dequantized_reference(case: _DenseSageCase) -> None:
    """Random scales on random E4M3 inputs reproduce the FP32 stream model."""

    torch.manual_seed(20260908)
    device = torch.device("cuda", 0)
    q, k, v, params = _random_sage_inputs(case, device)
    sm_scale = _HEAD_DIM**-0.5
    expected = _sage_dense_reference(
        q,
        k,
        v,
        params,
        sm_scale=sm_scale,
        mask_type=case.mask_type,
        kv_tile_size=case.expected_kv_tile,
    )
    wrapper = _plan_dense_sage(case, params, device)
    actual = wrapper.run(q, k, v, sm_scale=sm_scale)
    torch.cuda.synchronize()
    assert actual.dtype == case.out_dtype
    assert torch.isfinite(actual).all()
    # The reference models the kernel's quantization exactly, so the
    # remaining error is the 16-bit output rounding of values up to about two.
    if case.out_dtype == torch.bfloat16:
        rtol, atol = 8e-3, 2e-3
    else:
        rtol, atol = 2e-3, 2e-3
    torch.testing.assert_close(actual.float(), expected, rtol=rtol, atol=atol)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "q_magnitude",
    (0.25, 1.0),
    ids=("logit-std-1", "unscaled"),
)
@pytest.mark.parametrize(
    "case",
    _cases_named(_DENSE_SAGE_CASES, "kv256_k16_q1_bf16", "q128_k16_q1_bf16_mean"),
    ids=("kv256", "q128"),
)
@torch.no_grad()
def test_dense_sage_fp8_recipe_tracks_bf16_attention(
    case: _DenseSageCase, q_magnitude: float
) -> None:
    """Recipe ``(fp8, fp8, (1, 16, 1))`` on BF16 inputs tracks BF16 attention.

    Heavy-tailed inputs follow TensorRT-LLM's recipe test. With Q scaled so the
    logits spread by about one standard deviation, as in trained attention
    layers, the output meets TensorRT-LLM's elementwise bound. The unscaled
    inputs spread the logits by about four: each row concentrates on one key,
    the block-quantization error of that key becomes a full-magnitude output
    error on a fraction of a percent of the elements, and only the cosine
    similarity bound holds. The elementwise bound is therefore a property of
    the input distribution, not of the kernel.
    """

    torch.manual_seed(1)
    device = torch.device("cuda", 0)
    q_shape = (case.batch_size, case.seq_len_q, case.num_qo_heads, _HEAD_DIM)
    kv_shape = (case.batch_size, case.seq_len_kv, case.num_kv_heads, _HEAD_DIM)
    q_bf16 = heavy_tailed(q_shape, device=device, magnitude=q_magnitude)
    k_bf16 = heavy_tailed(kv_shape, device=device)
    v_bf16 = heavy_tailed(kv_shape, device=device)
    q_fp8, q_scale = quantize_token_blocks(q_bf16, block_size=1, dtype=_FP8)
    k_fp8, k_scale = quantize_token_blocks(k_bf16, block_size=16, dtype=_FP8)
    v_fp8, v_scale, _ = quantize_v_channels(v_bf16)
    params = SageAttentionParams(
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        q_block_size=1,
        k_block_size=16,
    )
    recipe_case = replace(
        case,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=False,
        out_dtype=torch.bfloat16,
    )
    sm_scale = _HEAD_DIM**-0.5
    group_size = case.num_qo_heads // case.num_kv_heads
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_bf16.float().permute(0, 2, 1, 3),
        k_bf16.float().repeat_interleave(group_size, dim=2).permute(0, 2, 1, 3),
        v_bf16.float().repeat_interleave(group_size, dim=2).permute(0, 2, 1, 3),
        scale=sm_scale,
    ).permute(0, 2, 1, 3)

    wrapper = _plan_dense_sage(recipe_case, params, device)
    actual = wrapper.run(q_fp8, k_fp8, v_fp8, sm_scale=sm_scale).float()
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    )
    assert cosine > 0.99
    if q_magnitude < 1.0:
        torch.testing.assert_close(actual, expected, rtol=2e-1, atol=3e-1)
