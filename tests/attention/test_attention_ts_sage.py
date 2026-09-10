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
    block_mean,
    dense_stream_columns,
    heavy_tailed,
    make_bsr,
    make_exact_block_bits,
    make_sage_decode_config,
    make_sage_params,
    pack_token_mask,
    token_mask_valid_sets,
    widest_bsr_row,
)
from tests.attention.sage_quant_reference import (
    dequantize_token_blocks,
    dequantize_v_channels,
    quantize_token_blocks,
    quantize_v_channels,
    quantize_v_channels_with_scale,
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


@pytest.mark.parametrize(
    ("kv_route_size", "kv_block_size", "seq_len_kv", "expected"),
    (
        # KV256, 169 summaries: tail 168 lies in atom 2, owned by threads
        # [0, 64) as their second atom, in that atom's second K32 fragment.
        (256, 64, 10800, (0, 3)),
        # KV256, 33 summaries: tail 32 opens the second fragment of atom 0.
        (256, 64, 2100, (0, 1)),
        # KV256, 100 summaries: tail 99 lies in atom 1, owned by threads
        # [64, 128) as their first atom.
        (256, 64, 6370, (1, 1)),
        # KV256, 257 summaries need two proxy groups, so the tail's route is
        # not known at compile time.
        (256, 64, 16400, None),
        # KV128 has one spatial half owning both atoms: 100 summaries put
        # tail 99 in atom 1's second fragment.
        (128, 128, 12679, (0, 3)),
        # KV128, 131 summaries exceed one route.
        (128, 64, 8323, None),
    ),
)
def test_proxy_static_tail_fragment_follows_keeps_atom_ownership(
    kv_route_size: int,
    kv_block_size: int,
    seq_len_kv: int,
    expected: tuple[int, int] | None,
) -> None:
    """A single proxy group fixes the ragged summary's half and fragment."""

    from flashinfer.attention.prims_ts._block_sparse import (
        config as block_sparse_config,
    )
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
        _configure_static_sliding_window,
    )
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.smem_block_sparse_metadata import (
        _proxy_static_tail_fragment,
    )

    key = block_sparse_config._BlockSparseCompileKey(
        device_index=0,
        batch_size=1,
        seq_len_q=64 if kv_route_size == 256 else 48,
        seq_len_kv=seq_len_kv,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        q_block_size=64 if kv_route_size == 256 else 128,
        kv_block_size=kv_block_size,
        kv_route_size=kv_route_size,
        dtype_key="float8_e4m3fn",
        mask_type="dense",
        use_kv_valid_bits=False,
        use_persistent_scheduler=False,
        use_parallel_sparse_kv_loads=False,
        sparse_format="bitmask",
        use_proxy_routes=True,
        out_dtype_key="bfloat16",
        sage_q_block_size=1,
        sage_k_block_size=16,
    )
    cfg = block_sparse_config._make_block_sparse_config(key)
    # The launch publishes the static sequence length on the config before
    # tracing; the tail geometry reads it from there.
    _configure_static_sliding_window(cfg, seq_len_kv)
    assert cfg.tile_size_kv == kv_route_size and cfg.block_sparse_kv_atom_size == 64
    assert _proxy_static_tail_fragment(cfg) == expected


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
    # ``None`` exercises the default: V follows the K dtype.
    v_dtype: torch.dtype | None = None
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
        v_dtype=geometry.v_dtype,
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


@pytest.mark.parametrize("out_dtype", (torch.bfloat16, torch.float16))
def test_sage_params_accept_int8_qk_with_e4m3_v(out_dtype: torch.dtype) -> None:
    """The INT8 recipe keeps E4M3 V, so V must be named when K is INT8."""

    geometry = _Geometry(
        q_dtype=torch.int8, kv_dtype=torch.int8, v_dtype=_FP8, out_dtype=out_dtype
    )
    _validate(_make_params(geometry), geometry)


def test_sage_params_v_dtype_defaults_to_k_dtype() -> None:
    """Omitting V keeps the E4M3 callers unchanged and rejects INT8 K alone."""

    _validate(_make_params(_Geometry()), _Geometry())
    geometry = _Geometry(q_dtype=torch.int8, kv_dtype=torch.int8)
    with pytest.raises(ValueError, match="V in"):
        _validate(_make_params(geometry), geometry)


@pytest.mark.parametrize(
    ("q_dtype", "kv_dtype", "v_dtype", "match"),
    (
        (torch.int8, _FP8, _FP8, "Q and K"),
        (_FP8, torch.int8, _FP8, "Q and K"),
        (torch.int8, torch.int8, torch.int8, "V in"),
        (_FP8, _FP8, torch.int8, "V in"),
        (_FP8, _FP8, torch.float16, "V in"),
    ),
)
def test_sage_params_reject_mixed_8bit_dtypes(
    q_dtype: torch.dtype, kv_dtype: torch.dtype, v_dtype: torch.dtype, match: str
) -> None:
    geometry = _Geometry(q_dtype=q_dtype, kv_dtype=kv_dtype, v_dtype=v_dtype)
    with pytest.raises(ValueError, match=match):
        _validate(_make_params(geometry), geometry)


@pytest.mark.parametrize("k_block_size", (1, 4))
def test_sage_params_reject_int8_recipes_with_small_k_blocks(
    k_block_size: int,
) -> None:
    """TensorRT-LLM's ``(int8, fp8, (1, 1, 1))`` and ``(1, 4, 1)`` stay unsupported."""

    geometry = _Geometry(q_dtype=torch.int8, kv_dtype=torch.int8, v_dtype=_FP8)
    with pytest.raises(ValueError, match="k_block_size"):
        _validate(_make_params(geometry, k_block_size=k_block_size), geometry)


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


def _validate_with_summaries(
    params: SageAttentionParams,
    geometry: _Geometry,
    *,
    summary_seq_len: int | None,
) -> None:
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
        summary_seq_len=summary_seq_len,
    )


def test_sage_params_require_summary_scale_with_proxy_routes() -> None:
    """``k_summary_scale`` covers the summary sequence in the flat layout."""

    geometry = _Geometry()
    params = _make_params(geometry, k_block_size=16)
    num_kv_blocks = -(-geometry.seq_len_kv // 64)
    summary_slots = flat_scale_numel(geometry.batch_size, num_kv_blocks, 16)
    with pytest.raises(ValueError, match="k_summary_scale"):
        _validate_with_summaries(params, geometry, summary_seq_len=num_kv_blocks)
    with pytest.raises(ValueError, match="k_summary_scale"):
        _validate_with_summaries(
            replace(
                params,
                k_summary_scale=torch.rand((geometry.num_kv_heads, summary_slots + 1)),
            ),
            geometry,
            summary_seq_len=num_kv_blocks,
        )
    _validate_with_summaries(
        replace(
            params,
            k_summary_scale=torch.rand((geometry.num_kv_heads, summary_slots)),
        ),
        geometry,
        summary_seq_len=num_kv_blocks,
    )


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
    qk_dtype: torch.dtype = _FP8

    @property
    def expected_q_tile(self) -> int:
        return 64 if self.expected_kv_tile == 256 else 128


# INT8 Q/K are drawn with this standard deviation and their scales divided by
# it, so the dequantized inputs match the E4M3 cases' unit-variance values.
_INT8_INPUT_STD = 40.0


def _random_qk(shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Random 8-bit Q or K values of one dtype."""

    values = torch.randn(shape, device=device)
    if dtype == torch.int8:
        return (values * _INT8_INPUT_STD).round().clamp(-127, 127).to(torch.int8)
    return values.to(dtype)


def _qk_scale_factor(dtype: torch.dtype) -> float:
    return 1.0 / _INT8_INPUT_STD if dtype == torch.int8 else 1.0


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
    # INT8 Q/K accumulate INT32 scores; the exact dot product leaves input
    # quantization as the only error, so the FP32 reference applies unchanged.
    _DenseSageCase(
        "kv256_int8_k16_q1_bf16",
        **_KV256_MHA,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=False,
        out_dtype=torch.bfloat16,
        qk_dtype=torch.int8,
    ),
    _DenseSageCase(
        "kv256_gqa4_int8_k64_q16_fp16_mean_causal",
        **_KV256_GQA,
        sage_q_block_size=16,
        sage_k_block_size=64,
        with_mean=True,
        out_dtype=torch.float16,
        mask_type="causal",
        qk_dtype=torch.int8,
    ),
    _DenseSageCase(
        "q128_int8_k16_q1_bf16_mean",
        **_Q128_GQA,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=True,
        out_dtype=torch.bfloat16,
        qk_dtype=torch.int8,
    ),
)


def _cases_named(cases, *names: str):
    """Select test cases by name, in the order the names are given."""

    by_name = {case.name: case for case in cases}
    return tuple(by_name[name] for name in names)


def _random_sage_inputs(case: _DenseSageCase | _SparseSageCase, device: torch.device):
    """Random 8-bit Q/K, E4M3 V and random positive scales in the flat layout.

    Only the geometry and recipe fields shared by both case types are read.
    """

    q = _random_qk(
        (case.batch_size, case.seq_len_q, case.num_qo_heads, _HEAD_DIM),
        case.qk_dtype,
        device,
    )
    k = _random_qk(
        (case.batch_size, case.seq_len_kv, case.num_kv_heads, _HEAD_DIM),
        case.qk_dtype,
        device,
    )
    v = torch.randn(k.shape, device=device).to(_FP8)
    qk_scale_factor = _qk_scale_factor(case.qk_dtype)

    def random_scale(shape, low, high):
        return (torch.rand(shape, device=device) * (high - low) + low).contiguous()

    q_scale = random_scale(
        (
            case.num_qo_heads,
            flat_scale_numel(case.batch_size, case.seq_len_q, case.sage_q_block_size),
        ),
        0.05 * qk_scale_factor,
        0.2 * qk_scale_factor,
    )
    k_scale = random_scale(
        (
            case.num_kv_heads,
            flat_scale_numel(case.batch_size, case.seq_len_kv, case.sage_k_block_size),
        ),
        0.5 * qk_scale_factor,
        2.0 * qk_scale_factor,
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


def _v_data_type_kwargs(qk_dtype: torch.dtype) -> dict[str, torch.dtype]:
    """Name the E4M3 V dtype only for INT8 Q/K; E4M3 callers rely on the default."""

    return {"v_data_type": _FP8} if qk_dtype == torch.int8 else {}


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
            q_data_type=case.qk_dtype,
            kv_data_type=case.qk_dtype,
            o_data_type=case.out_dtype,
            sage=params,
            **_v_data_type_kwargs(case.qk_dtype),
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
@torch.no_grad()
def test_dense_int8_extreme_scores_stay_in_the_bias_binade() -> None:
    """The largest INT8 dot products, ``+2**21`` and ``-128 * 127 * 128``, are exact.

    INT32 scores are read as FP32 after accumulating onto ``1.5 * 2**23``, which
    needs every ``bias + score`` inside ``[2**23, 2**24)``. One K row per head of
    all ``-128`` against a Q row of all ``-128`` scores exactly ``+2**21``, and a K
    row of all ``127`` scores ``-2080768``; their scales keep the reference
    softmax unsaturated so both keys carry visible weight.
    """

    (case,) = _cases_named(_DENSE_SAGE_CASES, "kv256_int8_k16_q1_bf16")
    torch.manual_seed(20260909)
    device = torch.device("cuda", 0)
    q, k, v, params = _random_sage_inputs(case, device)
    batch, q_row, k_max_token, k_min_token = 0, 5, 300, 700
    q[batch, q_row] = -128
    k[batch, k_max_token] = -128
    k[batch, k_min_token] = 127
    scores = torch.einsum("hd,thd->ht", q[batch, q_row].float(), k[batch].float())
    assert torch.equal(scores[:, k_max_token], torch.full_like(scores[:, 0], 2.0**21))
    assert torch.equal(
        scores[:, k_min_token], torch.full_like(scores[:, 0], -2080768.0)
    )
    # ``sfQ * sfK * s * sm_scale`` is about +-1.9 for the extreme keys, so the
    # softmax row keeps most of its mass on the remaining keys.
    q_slot = flat_scale_slot(
        batch, q_row, case.seq_len_q, log2_block_size(params.q_block_size)
    )
    params.q_scale[:, q_slot] = 1e-3
    for token in (k_max_token, k_min_token):
        k_slot = flat_scale_slot(
            batch, token, case.seq_len_kv, log2_block_size(params.k_block_size)
        )
        params.k_scale[:, k_slot] = 1e-2
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
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual[batch, q_row].float(), expected[batch, q_row], rtol=8e-3, atol=2e-3
    )
    torch.testing.assert_close(actual.float(), expected, rtol=8e-3, atol=2e-3)


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
@pytest.mark.parametrize("qk_dtype", (_FP8, torch.int8), ids=("fp8", "int8"))
@torch.no_grad()
def test_dense_sage_recipe_tracks_bf16_attention(
    case: _DenseSageCase, q_magnitude: float, qk_dtype: torch.dtype
) -> None:
    """Recipes ``(fp8|int8, fp8, (1, 16, 1))`` on BF16 inputs track BF16 attention.

    Heavy-tailed inputs follow TensorRT-LLM's recipe test. With Q scaled so the
    logits spread by about one standard deviation, as in trained attention
    layers, the output meets TensorRT-LLM's elementwise bound. The unscaled
    inputs spread the logits by about four: each row concentrates on one key,
    the block-quantization error of that key becomes a full-magnitude output
    error on a fraction of a percent of the elements, and only the cosine
    similarity bound holds. The elementwise bound is therefore a property of
    the input distribution, not of the kernel. INT8 uses TensorRT-LLM's
    ``TypeMax = 126.9`` quantizer and its recipe tolerance.
    """

    torch.manual_seed(1)
    device = torch.device("cuda", 0)
    q_shape = (case.batch_size, case.seq_len_q, case.num_qo_heads, _HEAD_DIM)
    kv_shape = (case.batch_size, case.seq_len_kv, case.num_kv_heads, _HEAD_DIM)
    q_bf16 = heavy_tailed(q_shape, device=device, magnitude=q_magnitude)
    k_bf16 = heavy_tailed(kv_shape, device=device)
    v_bf16 = heavy_tailed(kv_shape, device=device)
    q_quant, q_scale = quantize_token_blocks(q_bf16, block_size=1, dtype=qk_dtype)
    k_quant, k_scale = quantize_token_blocks(k_bf16, block_size=16, dtype=qk_dtype)
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
        qk_dtype=qk_dtype,
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
    actual = wrapper.run(q_quant, k_quant, v_fp8, sm_scale=sm_scale).float()
    torch.cuda.synchronize()
    _assert_recipe_close(actual, expected, q_magnitude=q_magnitude, qk_dtype=qk_dtype)


def _recipe_elementwise_tolerance(qk_dtype: torch.dtype) -> tuple[float, float]:
    """TensorRT-LLM's unit-test bounds per recipe: INT8 ``(5e-1, 5e-1)``."""

    if qk_dtype == torch.int8:
        return 5e-1, 5e-1
    return 2e-1, 3e-1


def _relative_l2_error_per_head(
    actual: torch.Tensor, expected: torch.Tensor
) -> torch.Tensor:
    """Return ``||O - ref||_2 / ||ref||_2`` per (batch, head) of ``[B, S, H, D]``."""

    difference = (actual - expected).permute(0, 2, 1, 3).flatten(2).norm(dim=-1)
    reference = expected.permute(0, 2, 1, 3).flatten(2).norm(dim=-1)
    return difference / reference


def _recipe_relative_l2_bound(q_magnitude: float, qk_dtype: torch.dtype) -> float:
    """Per-head relative L2 bound of the recipe tests.

    The INT8 elementwise bound ``(5e-1, 5e-1)`` says nothing about outputs
    below one in magnitude, so this metric holds every head to a fraction of
    its norm. The bounds leave about 1.3x to 1.5x headroom over the maxima
    measured on B200 for both profiles and both route kinds: INT8 reaches
    0.043 with logits spread by one standard deviation and 0.055 unscaled;
    E4M3 Q/K, whose three mantissa bits quantize each input about ten times
    as coarsely, reaches 0.082 and 0.133. A misaddressed scale moves a head
    by well over half its norm.
    """

    if qk_dtype == torch.int8:
        return 0.075
    return 0.10 if q_magnitude < 1.0 else 0.20


def _assert_recipe_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    q_magnitude: float,
    qk_dtype: torch.dtype,
) -> None:
    """Check a recipe output: finite, cosine-close, and elementwise-close for spread logits."""

    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    )
    assert cosine > 0.99
    relative_l2 = _relative_l2_error_per_head(actual, expected)
    assert relative_l2.max() < _recipe_relative_l2_bound(q_magnitude, qk_dtype), (
        relative_l2
    )
    if q_magnitude < 1.0:
        rtol, atol = _recipe_elementwise_tolerance(qk_dtype)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Block-sparse kernel fidelity and recipe tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SparseSageCase:
    """One block-sparse Sage problem with exact routes and optional proxies."""

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
    use_proxy_routes: bool
    use_token_mask: bool = False
    sparse_format: str = "bsr"
    qk_dtype: torch.dtype = _FP8
    # "auto" follows the planner; the test shapes are too small for it to
    # pick the persistent grid, so persistent cases force the selection.
    scheduler: str = "auto"

    @property
    def expected_q_tile(self) -> int:
        return 64 if self.expected_kv_tile == 256 else 128

    @property
    def num_kv_blocks(self) -> int:
        return -(-self.seq_len_kv // self.kv_block_size)

    @property
    def heads_q_per_kv(self) -> int:
        return self.num_qo_heads // self.num_kv_heads


# 1000 tokens leave a 40-token ragged block; 500 tokens a 52-token one.
_SPARSE_KV256_MHA = dict(
    batch_size=2,
    seq_len_q=128,
    seq_len_kv=1000,
    num_qo_heads=2,
    num_kv_heads=2,
    q_block_size=64,
    expected_kv_tile=256,
)
_SPARSE_Q128_GQA = dict(
    batch_size=2,
    seq_len_q=48,
    seq_len_kv=500,
    num_qo_heads=16,
    num_kv_heads=2,
    q_block_size=16,
    expected_kv_tile=128,
)

_SPARSE_SAGE_CASES = (
    _SparseSageCase(
        "kv256_exact_k16_q1_bf16",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=False,
        out_dtype=torch.bfloat16,
        use_proxy_routes=False,
    ),
    _SparseSageCase(
        "kv256_exact_bk128_k64_q16_fp16_mask",
        **_SPARSE_KV256_MHA,
        kv_block_size=128,
        sage_q_block_size=16,
        sage_k_block_size=64,
        with_mean=False,
        out_dtype=torch.float16,
        use_proxy_routes=False,
        use_token_mask=True,
    ),
    _SparseSageCase(
        "kv256_proxy_k16_q1_bf16_mean",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=True,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
    ),
    _SparseSageCase(
        "kv256_proxy_bk128_k64_q64_bf16_bitmask",
        **_SPARSE_KV256_MHA,
        kv_block_size=128,
        sage_q_block_size=64,
        sage_k_block_size=64,
        with_mean=False,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
        sparse_format="bitmask",
    ),
    # One 128-token K block scale spans several summary atoms of the proxy
    # sequence (16 summaries for 1000 tokens in 64-token blocks).
    _SparseSageCase(
        "kv256_proxy_k128_q16_fp16",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        sage_q_block_size=16,
        sage_k_block_size=128,
        with_mean=False,
        out_dtype=torch.float16,
        use_proxy_routes=True,
    ),
    _SparseSageCase(
        "q128_gqa8_exact_k16_q1_bf16_mask",
        **_SPARSE_Q128_GQA,
        kv_block_size=64,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=False,
        out_dtype=torch.bfloat16,
        use_proxy_routes=False,
        use_token_mask=True,
    ),
    # Without a token mask the Q128 exact route transports two atom origins per
    # record, so the load warp stages sfK from the broadcast origin pair.
    _SparseSageCase(
        "q128_gqa8_exact_k16_q1_bf16",
        **_SPARSE_Q128_GQA,
        kv_block_size=64,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=False,
        out_dtype=torch.bfloat16,
        use_proxy_routes=False,
    ),
    _SparseSageCase(
        "q128_gqa8_proxy_bk128_k32_q4_fp16_mean",
        **_SPARSE_Q128_GQA,
        kv_block_size=128,
        sage_q_block_size=4,
        sage_k_block_size=32,
        with_mean=True,
        out_dtype=torch.float16,
        use_proxy_routes=True,
    ),
    # INT8 Q/K on both routes: the exact route runs the masked INT32 store
    # path and the proxy route shifts the ragged tail summary in score units.
    _SparseSageCase(
        "kv256_int8_exact_k16_q1_bf16_mask",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=False,
        out_dtype=torch.bfloat16,
        use_proxy_routes=False,
        use_token_mask=True,
        qk_dtype=torch.int8,
    ),
    _SparseSageCase(
        "kv256_int8_proxy_bk128_k64_q64_bf16_mean",
        **_SPARSE_KV256_MHA,
        kv_block_size=128,
        sage_q_block_size=64,
        sage_k_block_size=64,
        with_mean=True,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
        qk_dtype=torch.int8,
    ),
    # 33 summaries put the ragged tail summary on lane 0: the shifted score
    # the max pass writes back is the first element the P pass reloads.
    _SparseSageCase(
        "kv256_int8_proxy_tail_lane0_k64_q64_bf16",
        **{**_SPARSE_KV256_MHA, "seq_len_kv": 2100},
        kv_block_size=64,
        sage_q_block_size=64,
        sage_k_block_size=64,
        with_mean=False,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
        qk_dtype=torch.int8,
    ),
    # 100 summaries put the ragged tail summary in atom 1, which the second
    # spatial half of the KV256 softmax owns as its first atom.
    _SparseSageCase(
        "kv256_proxy_tail_atom1_k16_q1_bf16",
        **{**_SPARSE_KV256_MHA, "seq_len_kv": 6370},
        kv_block_size=64,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=False,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
    ),
    _SparseSageCase(
        "q128_gqa8_int8_proxy_k16_q1_fp16",
        **_SPARSE_Q128_GQA,
        kv_block_size=64,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=False,
        out_dtype=torch.float16,
        use_proxy_routes=True,
        qk_dtype=torch.int8,
    ),
)
# The persistent grid resolves every tile through the work tile; cover one
# KV256 exact, one KV256 proxy, one Q128 and one INT8 case on it.
_PERSISTENT_SPARSE_SAGE_CASE_NAMES = (
    "kv256_exact_bk128_k64_q16_fp16_mask",
    "kv256_proxy_k16_q1_bf16_mean",
    "q128_gqa8_proxy_bk128_k32_q4_fp16_mean",
    "kv256_int8_proxy_bk128_k64_q64_bf16_mean",
)
_SPARSE_SAGE_CASES += tuple(
    replace(case, name=f"{case.name}_persistent", scheduler="persistent")
    for case in _SPARSE_SAGE_CASES
    if case.name in _PERSISTENT_SPARSE_SAGE_CASE_NAMES
)


def _sparse_patterns(case: _SparseSageCase, generator: torch.Generator):
    """Return exact block tuples per (batch, kv head, q row) with a ragged tail."""

    num_q_rows = -(-case.seq_len_q // case.q_block_size)
    patterns = []
    for _batch_idx in range(case.batch_size):
        heads = []
        for _head_idx in range(case.num_kv_heads):
            rows = []
            for row_idx in range(num_q_rows):
                count = 1 + int(
                    torch.randint(0, case.num_kv_blocks, (1,), generator=generator)
                )
                selected = set(
                    torch.randperm(case.num_kv_blocks, generator=generator)[
                        :count
                    ].tolist()
                )
                if row_idx % 2:
                    # Odd rows keep the ragged final block exact.
                    selected.add(case.num_kv_blocks - 1)
                rows.append(tuple(sorted(selected)))
            heads.append(tuple(rows))
        patterns.append(tuple(heads))
    return tuple(patterns)


def _sparse_routing(case: _SparseSageCase, patterns, device: torch.device):
    """Build the BSR or bitmask routing tensors of one pattern set."""

    if case.sparse_format == "bsr":
        block_indptr, block_indices = make_bsr(patterns, device)
        return {"block_indptr": block_indptr, "block_indices": block_indices}
    return {
        "exact_block_bits": make_exact_block_bits(patterns, case.num_kv_blocks, device)
    }


def _sparse_token_mask(case: _SparseSageCase, device: torch.device):
    """Return packed validity bits and the boolean mask ``[B, Skv]``."""

    valid_by_batch = token_mask_valid_sets(case.batch_size, case.seq_len_kv)
    valid = torch.tensor(
        [
            [token_idx in valid_tokens for token_idx in range(case.seq_len_kv)]
            for valid_tokens in valid_by_batch
        ],
        device=device,
    )
    return pack_token_mask(case.seq_len_kv, valid_by_batch, device), valid


def _random_sparse_sage_inputs(case: _SparseSageCase, device: torch.device):
    """Random E4M3 Q/K/V and summaries with random positive scales."""

    q, k, v, params = _random_sage_inputs(case, device)
    summaries = None
    if case.use_proxy_routes:
        summary_shape = (
            case.batch_size,
            case.num_kv_blocks,
            case.num_kv_heads,
            _HEAD_DIM,
        )
        k_summary = _random_qk(summary_shape, case.qk_dtype, device)
        v_summary = torch.randn(summary_shape, device=device).to(_FP8)
        k_summary_scale = (
            (
                torch.rand(
                    (
                        case.num_kv_heads,
                        flat_scale_numel(
                            case.batch_size, case.num_kv_blocks, case.sage_k_block_size
                        ),
                    ),
                    device=device,
                )
                * 1.5
                + 0.5
            )
            * _qk_scale_factor(case.qk_dtype)
        ).contiguous()
        params = replace(params, k_summary_scale=k_summary_scale)
        summaries = (k_summary, v_summary)
    return q, k, v, params, summaries


def _sparse_row_routes(case: _SparseSageCase, exact_blocks: tuple[int, ...]):
    """Return the kernel's route sequence of one sparse row.

    Every route is a list of columns ``(source, index, mass)``: exact routes
    pack ``route / kv_block`` selected blocks as K64 atoms of tokens, proxy
    routes follow with ``route`` summaries each. For KV256 the atoms alternate
    between the two spatial halves; a column's stream is ``(route % 2, half)``.
    """

    route_size = case.expected_kv_tile
    atom_size = min(case.kv_block_size, 64)
    blocks_per_route = route_size // case.kv_block_size
    routes = []
    for begin in range(0, len(exact_blocks), blocks_per_route):
        atoms = []
        for block_idx in exact_blocks[begin : begin + blocks_per_route]:
            block_begin = block_idx * case.kv_block_size
            block_end = min(block_begin + case.kv_block_size, case.seq_len_kv)
            for atom_begin in range(
                block_begin, block_begin + case.kv_block_size, atom_size
            ):
                atoms.append(
                    [
                        ("token", token_idx, 1)
                        for token_idx in range(
                            atom_begin, min(atom_begin + atom_size, block_end)
                        )
                    ]
                )
        routes.append(atoms)
    if case.use_proxy_routes:
        exact = set(exact_blocks)
        for begin in range(0, case.num_kv_blocks, route_size):
            atoms = []
            for atom_begin in range(begin, begin + route_size, atom_size):
                atom = []
                for summary_idx in range(
                    atom_begin, min(atom_begin + atom_size, case.num_kv_blocks)
                ):
                    if summary_idx in exact:
                        continue
                    mass = min(
                        case.kv_block_size,
                        case.seq_len_kv - summary_idx * case.kv_block_size,
                    )
                    atom.append(("summary", summary_idx, mass))
                atoms.append(atom)
            routes.append(atoms)
    return routes


@torch.no_grad()
def _sage_sparse_reference(
    case: _SparseSageCase,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    params: SageAttentionParams,
    summaries,
    patterns,
    valid_tokens: torch.Tensor | None,
    *,
    sm_scale: float,
) -> torch.Tensor:
    """FP32 sparse attention on dequantized inputs with the kernel's P448 streams.

    Exact tokens and proxy summaries form one logit set per Q row; a proxy
    summary of ``mass`` tokens contributes ``mass * exp(logit - max)``. Each
    stream quantizes its 448-scaled probabilities to E4M3 against its own
    running maximum, as the kernel does per K/V instance and spatial half.
    """

    q_real = dequantize_token_blocks(q, params.q_scale, block_size=params.q_block_size)
    k_real = dequantize_token_blocks(k, params.k_scale, block_size=params.k_block_size)
    v_raw = v.float()
    if summaries is not None:
        k_summary_real = dequantize_token_blocks(
            summaries[0], params.k_summary_scale, block_size=params.k_block_size
        )
        v_summary_raw = summaries[1].float()
    num_streams = 4 if case.expected_kv_tile == 256 else 2
    output = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for batch_idx in range(case.batch_size):
        for kv_head_idx in range(case.num_kv_heads):
            head_slice = slice(
                kv_head_idx * case.heads_q_per_kv,
                (kv_head_idx + 1) * case.heads_q_per_kv,
            )
            for row_idx, exact_blocks in enumerate(patterns[batch_idx][kv_head_idx]):
                row_begin = row_idx * case.q_block_size
                row_end = min(row_begin + case.q_block_size, case.seq_len_q)
                queries = q_real[batch_idx, row_begin:row_end, head_slice]
                streams = [None] * num_streams
                for route_idx, atoms in enumerate(
                    _sparse_row_routes(case, exact_blocks)
                ):
                    for atom_idx, atom in enumerate(atoms):
                        stream_idx = route_idx % 2
                        if num_streams == 4:
                            stream_idx = stream_idx * 2 + atom_idx % 2
                        keys = []
                        values = []
                        masses = []
                        for source, index, mass in atom:
                            if source == "token":
                                if (
                                    valid_tokens is not None
                                    and not valid_tokens[batch_idx, index]
                                ):
                                    continue
                                keys.append(k_real[batch_idx, index, kv_head_idx])
                                values.append(v_raw[batch_idx, index, kv_head_idx])
                            else:
                                keys.append(
                                    k_summary_real[batch_idx, index, kv_head_idx]
                                )
                                values.append(
                                    v_summary_raw[batch_idx, index, kv_head_idx]
                                )
                            masses.append(float(mass))
                        if not keys:
                            continue
                        key_stack = torch.stack(keys)
                        value_stack = torch.stack(values)
                        logits = (
                            torch.einsum("thd,cd->thc", queries, key_stack) * sm_scale
                        )
                        logits = logits + torch.tensor(masses, device=q.device).log()
                        local_max = logits.amax(dim=-1)
                        state = streams[stream_idx]
                        new_max = (
                            local_max
                            if state is None
                            else torch.maximum(state[0], local_max)
                        )
                        # The row sum accumulates the fp32 probabilities; only
                        # the PV operand is quantized, as in the kernel.
                        probabilities = (
                            torch.exp(logits - new_max.unsqueeze(-1))
                            * FP8_P_QUANT_SCALE
                        )
                        acc = torch.einsum(
                            "thc,cd->thd", probabilities.to(_FP8).float(), value_stack
                        )
                        total = probabilities.sum(dim=-1)
                        if state is not None:
                            correction = torch.exp(state[0] - new_max)
                            total = total + state[1] * correction
                            acc = acc + state[2] * correction.unsqueeze(-1)
                        streams[stream_idx] = (new_max, total, acc)
                live = [state for state in streams if state is not None]
                if not live:
                    continue
                final_max = torch.stack([state[0] for state in live]).amax(dim=0)
                final_sum = torch.zeros_like(final_max)
                final_acc = torch.zeros_like(live[0][2])
                for maximum, total, acc in live:
                    correction = torch.exp(maximum - final_max)
                    final_sum += total * correction
                    final_acc += acc * correction.unsqueeze(-1)
                result = (
                    final_acc / final_sum.unsqueeze(-1) * params.v_scale[kv_head_idx]
                )
                if params.v_mean is not None:
                    result = result + params.v_mean[kv_head_idx]
                output[batch_idx, row_begin:row_end, head_slice] = result
    return output


def _plan_sparse_sage(
    case: _SparseSageCase,
    params: SageAttentionParams,
    device: torch.device,
    *,
    max_blocks_per_row: int,
):
    from flashinfer.attention.prims_ts import BatchDecodeTSWrapper
    from flashinfer.attention.prims_ts._block_sparse import config as sparse_config

    wrapper = BatchDecodeTSWrapper()
    auto_select_scheduler = sparse_config._select_block_sparse_scheduler
    select_scheduler = auto_select_scheduler
    if case.scheduler != "auto":

        def select_scheduler(**kwargs):
            q_tile_size, _ = auto_select_scheduler(**kwargs)
            return q_tile_size, case.scheduler == "persistent"

    sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                sparse_config, "_select_block_sparse_scheduler", select_scheduler
            )
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
                use_block_sparse=True,
                max_blocks_per_row=max_blocks_per_row,
                use_kv_valid_bits=case.use_token_mask,
                sparse_format=case.sparse_format,
                use_proxy_routes=case.use_proxy_routes,
                q_data_type=case.qk_dtype,
                kv_data_type=case.qk_dtype,
                o_data_type=case.out_dtype,
                sage=params,
                **_v_data_type_kwargs(case.qk_dtype),
            )
    finally:
        sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    policy = dict(wrapper._policy)
    assert policy["tile_size_q"] == case.expected_q_tile
    assert policy["tile_size_kv"] == case.expected_kv_tile
    if case.scheduler != "auto":
        assert policy["scheduler"] == case.scheduler
    return wrapper


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _SPARSE_SAGE_CASES, ids=lambda case: case.name)
@torch.no_grad()
def test_block_sparse_sage_matches_dequantized_reference(case: _SparseSageCase) -> None:
    """Exact routes index ``k_scale`` by atom origin, proxies ``k_summary_scale``."""

    torch.manual_seed(20260908)
    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(20260908)
    patterns = _sparse_patterns(case, generator)
    q, k, v, params, summaries = _random_sparse_sage_inputs(case, device)
    valid_bits = None
    valid_tokens = None
    if case.use_token_mask:
        valid_bits, valid_tokens = _sparse_token_mask(case, device)
    sm_scale = _HEAD_DIM**-0.5
    expected = _sage_sparse_reference(
        case, q, k, v, params, summaries, patterns, valid_tokens, sm_scale=sm_scale
    )
    max_blocks_per_row = widest_bsr_row(patterns)
    wrapper = _plan_sparse_sage(
        case, params, device, max_blocks_per_row=max_blocks_per_row
    )
    run_kwargs = _sparse_routing(case, patterns, device)
    if summaries is not None:
        run_kwargs.update(k_summary=summaries[0], v_summary=summaries[1])
    actual = wrapper.run(
        q, k, v, kv_valid_bits=valid_bits, sm_scale=sm_scale, **run_kwargs
    )
    torch.cuda.synchronize()
    assert actual.dtype == case.out_dtype
    assert torch.isfinite(actual).all()
    # The kernel evaluates part of its exponentials with the FMA-pipe
    # emulation, whose relative error moves about one to two percent of the
    # E4M3 probabilities to the neighbouring quantization step. A sparse row
    # attends to a few hundred tokens or fewer, so those steps show up to
    # about 1e-2 in the output where the dense test's thousand-token rows
    # average them out; a misaddressed scale would move the output by far
    # more than this bound.
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


_INT8_FLOOR_SCALE_CASE = next(
    case
    for case in _SPARSE_SAGE_CASES
    if case.name == "kv256_int8_exact_k16_q1_bf16_mask"
)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("q_magnitude", (0.0, 0.01), ids=("zero_q", "small_q"))
@torch.no_grad()
def test_block_sparse_int8_masked_lanes_carry_no_mass_at_floor_scales(
    q_magnitude: float,
) -> None:
    """Masked INT8 lanes stay at zero mass when their group scale is at the floor.

    Zero K rows (padding) drive a 16-token block's scale down to the quantizer
    floor ``1e-3 / 126.9``, and a zero or tiny Q token does the same for
    ``sfQ``. Masked lanes of the INT32 path are rewritten as FP32 ``-FLT_MAX``
    and must exponentiate to zero regardless of how small ``sfQ * sfK`` gets;
    an integer sentinel scaled by the multiplier would leak most of a kept
    lane's weight here. Masked tokens
    carry a constant V well away from the kept tokens' values so any leaked
    mass shows up in the output.
    """

    case = _INT8_FLOOR_SCALE_CASE
    torch.manual_seed(20260909)
    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(20260909)
    patterns = _sparse_patterns(case, generator)
    valid_bits, valid_tokens = _sparse_token_mask(case, device)
    q_shape = (case.batch_size, case.seq_len_q, case.num_qo_heads, _HEAD_DIM)
    kv_shape = (case.batch_size, case.seq_len_kv, case.num_kv_heads, _HEAD_DIM)
    q_bf16 = (torch.randn(q_shape, device=device) * q_magnitude).to(torch.bfloat16)
    k_bf16 = torch.randn(kv_shape, device=device).to(torch.bfloat16)
    token_block = torch.arange(case.seq_len_kv, device=device) // case.sage_k_block_size
    k_bf16[:, token_block % 3 == 2] = 0
    v_bf16 = torch.randn(kv_shape, device=device).to(torch.bfloat16)
    v_bf16[~valid_tokens] = 4.0
    q_quant, q_scale = quantize_token_blocks(
        q_bf16, block_size=case.sage_q_block_size, dtype=torch.int8
    )
    k_quant, k_scale = quantize_token_blocks(
        k_bf16, block_size=case.sage_k_block_size, dtype=torch.int8
    )
    v_fp8, v_scale, _ = quantize_v_channels(v_bf16)
    params = SageAttentionParams(
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        q_block_size=case.sage_q_block_size,
        k_block_size=case.sage_k_block_size,
    )
    sm_scale = _HEAD_DIM**-0.5
    expected = _sage_sparse_reference(
        case,
        q_quant,
        k_quant,
        v_fp8,
        params,
        None,
        patterns,
        valid_tokens,
        sm_scale=sm_scale,
    )
    max_blocks_per_row = widest_bsr_row(patterns)
    wrapper = _plan_sparse_sage(
        case, params, device, max_blocks_per_row=max_blocks_per_row
    )
    actual = wrapper.run(
        q_quant,
        k_quant,
        v_fp8,
        kv_valid_bits=valid_bits,
        sm_scale=sm_scale,
        **_sparse_routing(case, patterns, device),
    )
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


@torch.no_grad()
def _sparse_bf16_reference(
    case: _SparseSageCase,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    patterns,
    *,
    sm_scale: float,
) -> torch.Tensor:
    """FP32 attention over exact tokens plus mass-weighted mean proxies."""

    k_summary = block_mean(k, case.kv_block_size)
    v_summary = block_mean(v, case.kv_block_size)
    masses = torch.full(
        (case.num_kv_blocks,), float(case.kv_block_size), device=q.device
    )
    masses[-1] = case.seq_len_kv - (case.num_kv_blocks - 1) * case.kv_block_size
    token_block = torch.arange(case.seq_len_kv, device=q.device) // case.kv_block_size
    output = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for batch_idx in range(case.batch_size):
        for kv_head_idx in range(case.num_kv_heads):
            head_slice = slice(
                kv_head_idx * case.heads_q_per_kv,
                (kv_head_idx + 1) * case.heads_q_per_kv,
            )
            keys = k[batch_idx, :, kv_head_idx].float()
            values = v[batch_idx, :, kv_head_idx].float()
            for row_idx, exact_blocks in enumerate(patterns[batch_idx][kv_head_idx]):
                row_begin = row_idx * case.q_block_size
                row_end = min(row_begin + case.q_block_size, case.seq_len_q)
                queries = q[batch_idx, row_begin:row_end, head_slice].float()
                exact_mask = torch.isin(
                    token_block, torch.tensor(exact_blocks, device=q.device)
                )
                logits = torch.einsum("thd,cd->thc", queries, keys) * sm_scale
                logits = logits.masked_fill(~exact_mask, float("-inf"))
                column_values = values
                if case.use_proxy_routes:
                    proxy_mask = ~torch.isin(
                        torch.arange(case.num_kv_blocks, device=q.device),
                        torch.tensor(exact_blocks, device=q.device),
                    )
                    proxy_logits = (
                        torch.einsum(
                            "thd,cd->thc", queries, k_summary[batch_idx, :, kv_head_idx]
                        )
                        * sm_scale
                        + masses.log()
                    ).masked_fill(~proxy_mask, float("-inf"))
                    logits = torch.cat((logits, proxy_logits), dim=-1)
                    column_values = torch.cat(
                        (values, v_summary[batch_idx, :, kv_head_idx]), dim=0
                    )
                probabilities = torch.softmax(logits, dim=-1)
                output[batch_idx, row_begin:row_end, head_slice] = torch.einsum(
                    "thc,cd->thd", probabilities, column_values
                )
    return output


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "q_magnitude",
    (0.25, 1.0),
    ids=("logit-std-1", "unscaled"),
)
@pytest.mark.parametrize(
    "case",
    _cases_named(
        _SPARSE_SAGE_CASES,
        "kv256_exact_k16_q1_bf16",
        "kv256_proxy_k16_q1_bf16_mean",
        "q128_gqa8_proxy_bk128_k32_q4_fp16_mean",
    ),
    ids=("kv256-exact", "kv256-proxy", "q128-proxy"),
)
@pytest.mark.parametrize("qk_dtype", (_FP8, torch.int8), ids=("fp8", "int8"))
@torch.no_grad()
def test_block_sparse_sage_recipe_tracks_bf16_attention(
    case: _SparseSageCase, q_magnitude: float, qk_dtype: torch.dtype
) -> None:
    """Recipes ``(fp8|int8, fp8, (1, 16, 1))`` track BF16 block-sparse attention.

    Proxy summaries are the per-block means of K and V; the summary K is
    quantized like one more K sequence and the summary V with the shared V
    scale. Tolerances follow the dense recipe test.
    """

    torch.manual_seed(1)
    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(1)
    recipe_case = replace(
        case,
        sage_q_block_size=1,
        sage_k_block_size=16,
        with_mean=False,
        out_dtype=torch.bfloat16,
        use_token_mask=False,
        qk_dtype=qk_dtype,
    )
    patterns = _sparse_patterns(recipe_case, generator)
    q_shape = (case.batch_size, case.seq_len_q, case.num_qo_heads, _HEAD_DIM)
    kv_shape = (case.batch_size, case.seq_len_kv, case.num_kv_heads, _HEAD_DIM)
    q_bf16 = heavy_tailed(q_shape, device=device, magnitude=q_magnitude)
    k_bf16 = heavy_tailed(kv_shape, device=device)
    v_bf16 = heavy_tailed(kv_shape, device=device)
    q_quant, q_scale = quantize_token_blocks(q_bf16, block_size=1, dtype=qk_dtype)
    k_quant, k_scale = quantize_token_blocks(k_bf16, block_size=16, dtype=qk_dtype)
    v_fp8, v_scale, _ = quantize_v_channels(v_bf16)
    summaries = None
    k_summary_scale = None
    if case.use_proxy_routes:
        k_summary_bf16 = block_mean(k_bf16, case.kv_block_size, torch.bfloat16)
        v_summary_bf16 = block_mean(v_bf16, case.kv_block_size, torch.bfloat16)
        k_summary_quant, k_summary_scale = quantize_token_blocks(
            k_summary_bf16, block_size=16, dtype=qk_dtype
        )
        v_summary_fp8 = quantize_v_channels_with_scale(v_summary_bf16, v_scale)
        summaries = (k_summary_quant, v_summary_fp8)
    params = SageAttentionParams(
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        k_summary_scale=k_summary_scale,
        q_block_size=1,
        k_block_size=16,
    )
    sm_scale = _HEAD_DIM**-0.5
    expected = _sparse_bf16_reference(
        recipe_case, q_bf16, k_bf16, v_bf16, patterns, sm_scale=sm_scale
    )
    max_blocks_per_row = widest_bsr_row(patterns)
    wrapper = _plan_sparse_sage(
        recipe_case, params, device, max_blocks_per_row=max_blocks_per_row
    )
    run_kwargs = _sparse_routing(recipe_case, patterns, device)
    if summaries is not None:
        run_kwargs.update(k_summary=summaries[0], v_summary=summaries[1])
    actual = wrapper.run(
        q_quant, k_quant, v_fp8, sm_scale=sm_scale, **run_kwargs
    ).float()
    torch.cuda.synchronize()
    _assert_recipe_close(actual, expected, q_magnitude=q_magnitude, qk_dtype=qk_dtype)
