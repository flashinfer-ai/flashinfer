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
    SageAttentionConfig,
    SageAttentionParams,
    flat_scale_numel,
    flat_scale_slot,
    log2_block_size,
    sage_scale_shapes,
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
    fold_stream,
    heavy_tailed,
    make_block_sparse_compile_key,
    make_bsr,
    make_exact_block_bits,
    make_sage_decode_config,
    make_sage_params,
    merge_streams,
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
@pytest.mark.parametrize("k_block_size", (1, 4, 16, 32, 64, 128, 256))
def test_sage_scale_arr_size_follows_fragment_groups(
    tile_size_q: int, k_block_size: int
) -> None:
    """Both streamed profiles own four K32 fragments; blocks below 32 split them.

    The Sage predicates follow the K block size alone: a 16-token block splits
    every K32 fragment into two scale groups, a 4-token block into eight and a
    one-token block into 32, larger blocks share one scale per fragment, and a
    profile without a K block size runs without Sage. Blocks below 16 tokens
    keep the tile's ``sfK`` words in SMEM instead of a lane register array.
    """

    cfg = make_sage_decode_config(
        tile_size_q=tile_size_q,
        tile_size_kv=256 if tile_size_q == 64 else 128,
        sage_args={"sage_k_block_size": k_block_size},
    )
    groups = max(1, 32 // k_block_size)
    assert cfg.use_sage_attention and cfg.streams_tmem_p_fragments
    assert not cfg.uses_int32_scores
    assert cfg.num_softmax_score_fragments == 4
    assert cfg.sage_k_groups_per_fragment == groups
    assert sage_scales.sage_scale_arr_size(cfg) == 4 * groups
    assert cfg.sage_k_scales_in_smem == (k_block_size < 16)
    assert sage_scales.sage_k_scale_words(cfg) == (
        (2 if tile_size_q == 64 else 1) * 4 * groups
    )

    plain = replace(cfg, sage_k_block_size=0, sage_q_block_size=0)
    assert not plain.use_sage_attention
    assert plain.sage_k_groups_per_fragment == 1


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
        # KV256, 257 summaries fill two proxy groups: tail 256 opens the
        # second group in atom 0, owned by threads [0, 64).
        (256, 64, 16400, (0, 0)),
        # KV256, 321 summaries: tail 320 is summary 64 of the second group,
        # atom 1, owned by threads [64, 128) as their first atom.
        (256, 64, 20520, (1, 0)),
        # KV128 has one spatial half owning both atoms: 100 summaries put
        # tail 99 in atom 1's second fragment.
        (128, 128, 12679, (0, 3)),
        # KV128, 131 summaries: tail 130 is summary 2 of the second group.
        (128, 64, 8323, (0, 0)),
    ),
)
def test_proxy_static_tail_fragment_follows_keeps_atom_ownership(
    kv_route_size: int,
    kv_block_size: int,
    seq_len_kv: int,
    expected: tuple[int, int],
) -> None:
    """Fixed proxy groups place the ragged summary in a compile-time fragment."""

    from flashinfer.attention.prims_ts._block_sparse import (
        config as block_sparse_config,
    )
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
        _configure_static_sliding_window,
    )

    key = make_block_sparse_compile_key(
        seq_len_q=64 if kv_route_size == 256 else 48,
        seq_len_kv=seq_len_kv,
        q_block_size=64 if kv_route_size == 256 else 128,
        kv_block_size=kv_block_size,
        kv_route_size=kv_route_size,
        dtype_key="float8_e4m3fn",
        sparse_format="bitmask",
        use_proxy_routes=True,
        out_dtype_key="bfloat16",
        sage=SageAttentionConfig(),
    )
    cfg = block_sparse_config._make_block_sparse_config(key)
    # The launch publishes the static sequence length on the config before
    # tracing; the tail geometry reads it from there.
    _configure_static_sliding_window(cfg, seq_len_kv)
    assert cfg.tile_size_kv == kv_route_size and cfg.block_sparse_kv_atom_size == 64
    assert cfg.proxy_static_tail_fragment == expected


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


_GEOMETRY = dict(
    batch_size=2,
    seq_len_q=64,
    seq_len_kv=1000,
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim=_HEAD_DIM,
)


def _make_params(**overrides) -> SageAttentionParams:
    return make_sage_params(**_GEOMETRY, **overrides)


def _validate(
    params: SageAttentionParams,
    config: SageAttentionConfig | None = None,
    *,
    device: torch.device | None = None,
    summary_seq_len: int | None = None,
) -> None:
    shapes = sage_scale_shapes(
        SageAttentionConfig() if config is None else config,
        **_GEOMETRY,
        summary_seq_len=summary_seq_len,
    )
    validate_sage_params(
        params, shapes, device=torch.device("cpu") if device is None else device
    )


def test_sage_config_defaults_follow_the_production_recipe() -> None:
    """The recipe defaults are TensorRT-LLM's ``(1, 16, 1)``; runs supply scales only."""

    config = SageAttentionConfig()
    assert (config.q_block_size, config.k_block_size, config.v_mean) == (1, 16, False)
    assert config.k_summary_block_size is None
    assert config.summary_k_block_size == 16
    params = SageAttentionParams(
        q_scale=torch.rand(
            (
                _GEOMETRY["num_qo_heads"],
                flat_scale_numel(_GEOMETRY["batch_size"], _GEOMETRY["seq_len_q"], 1),
            )
        ),
        k_scale=torch.rand(
            (
                _GEOMETRY["num_kv_heads"],
                flat_scale_numel(_GEOMETRY["batch_size"], _GEOMETRY["seq_len_kv"], 16),
            )
        ),
        v_scale=torch.rand((_GEOMETRY["num_kv_heads"], _GEOMETRY["head_dim"])),
    )
    assert params.k_summary_scale is None
    assert params.v_mean is None
    _validate(params, config)


@pytest.mark.parametrize("k_block_size", (1, 4, 16, 32, 64, 128, 256))
@pytest.mark.parametrize("q_block_size", (1, 4, 64))
@pytest.mark.parametrize("with_mean", (False, True))
def test_sage_params_accept_supported_shapes(
    k_block_size: int, q_block_size: int, with_mean: bool
) -> None:
    _validate(
        _make_params(
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            with_mean=with_mean,
        ),
        SageAttentionConfig(q_block_size, k_block_size, v_mean=with_mean),
    )


@pytest.mark.parametrize("with_mean", (False, True))
def test_sage_params_must_match_the_planned_v_mean(with_mean: bool) -> None:
    """``v_mean`` is present exactly when the plan's config asks for it."""

    with pytest.raises(ValueError, match="v_mean"):
        _validate(
            _make_params(with_mean=with_mean),
            SageAttentionConfig(v_mean=not with_mean),
        )


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

    params = replace(_make_params(with_mean=True), **{field: torch.rand(shape)})
    with pytest.raises(ValueError, match=match):
        _validate(params, SageAttentionConfig(v_mean=True))


def test_sage_params_reject_non_fp32_or_strided_scales() -> None:
    params = _make_params()
    with pytest.raises(ValueError, match="float32"):
        _validate(replace(params, k_scale=params.k_scale.half()))
    strided = torch.rand((_GEOMETRY["num_kv_heads"], 2 * _GEOMETRY["head_dim"]))[:, ::2]
    with pytest.raises(ValueError, match="contiguous"):
        _validate(replace(params, v_scale=strided))
    # The epilogue reads V scales with 16-byte vector loads.
    misaligned = torch.rand((_GEOMETRY["num_kv_heads"] * _GEOMETRY["head_dim"] + 1,))[
        1:
    ].view(_GEOMETRY["num_kv_heads"], _GEOMETRY["head_dim"])
    with pytest.raises(ValueError, match="16-byte aligned"):
        _validate(replace(params, v_scale=misaligned))
    with pytest.raises(ValueError, match="k_summary_scale"):
        _validate(replace(params, k_summary_scale=params.k_scale.clone()))


@pytest.mark.parametrize("k_summary_block_size", (1, 4, 16, 64))
def test_summary_scales_follow_the_summary_block_size(
    k_summary_block_size: int,
) -> None:
    """``k_summary_scale`` uses the recipe's summary K block size, not ``k_block_size``."""

    kv_block_size = 64
    num_kv_blocks = -(-_GEOMETRY["seq_len_kv"] // kv_block_size)
    config = SageAttentionConfig(
        k_block_size=16, k_summary_block_size=k_summary_block_size
    )
    assert config.summary_k_block_size == k_summary_block_size
    shapes = sage_scale_shapes(config, **_GEOMETRY, summary_seq_len=num_kv_blocks)
    assert shapes["k_scale"][1] == flat_scale_numel(
        _GEOMETRY["batch_size"], _GEOMETRY["seq_len_kv"], 16
    )
    assert shapes["k_summary_scale"][1] == flat_scale_numel(
        _GEOMETRY["batch_size"], num_kv_blocks, k_summary_block_size
    )


@pytest.mark.parametrize("use_proxy_routes", (False, True))
def test_summary_block_size_reaches_only_proxy_kernels(use_proxy_routes: bool) -> None:
    """Plans without proxy routes compile one kernel per recipe, whatever the summary block."""

    from flashinfer.attention.prims_ts._block_sparse import (
        config as block_sparse_config,
    )

    key = make_block_sparse_compile_key(
        seq_len_q=64,
        seq_len_kv=4096,
        q_block_size=64,
        kv_block_size=64,
        kv_route_size=256,
        dtype_key="float8_e4m3fn",
        sparse_format="bitmask",
        use_proxy_routes=use_proxy_routes,
        out_dtype_key="bfloat16",
        sage=SageAttentionConfig(k_summary_block_size=1),
    )
    cfg = block_sparse_config._make_block_sparse_config(key)
    assert cfg.sage_k_block_size == 16
    assert cfg.sage_k_summary_block_size == (1 if use_proxy_routes else 0)
    assert cfg.sage_mixed_k_geometry == use_proxy_routes
    assert cfg.sage_summary_k_groups_per_fragment == (32 if use_proxy_routes else 2)
    # One scale per summary score: the max pass writes proxy tiles back
    # dequantized and the P pass needs no summary scales.
    assert cfg.sage_summary_scores_dequantized == use_proxy_routes
    if use_proxy_routes:
        assert cfg.sage_k_scales_in_smem_for(cfg.sage_summary_k_groups_per_fragment)
        assert not cfg.sage_k_scales_in_smem
        # Proxy routes gather their words in the softmax warps; staging
        # covers the exact geometry.
        assert sage_scales.sage_staged_k_scale_words(cfg) == 2 * 4 * 2
    else:
        assert sage_scales.sage_staged_k_scale_words(cfg) == 2 * 4 * 2


def test_summary_block_size_is_validated() -> None:
    """A proxy plan needs a supported summary block; other plans must leave it unset."""

    with pytest.raises(ValueError, match="sage_k_summary_block_size"):
        make_sage_decode_config(
            tile_size_q=64,
            tile_size_kv=256,
            sage_args={"sage_k_block_size": 16, "sage_k_summary_block_size": 16},
        )
    from flashinfer.attention.prims_ts._block_sparse import (
        config as block_sparse_config,
    )

    key = make_block_sparse_compile_key(
        seq_len_q=64,
        seq_len_kv=4096,
        q_block_size=64,
        kv_block_size=64,
        kv_route_size=256,
        dtype_key="float8_e4m3fn",
        sparse_format="bitmask",
        use_proxy_routes=True,
        out_dtype_key="bfloat16",
        sage=SageAttentionConfig(k_summary_block_size=3),
    )
    with pytest.raises(ValueError, match="sage_k_summary_block_size"):
        block_sparse_config._make_block_sparse_config(key)


def test_sage_params_require_summary_scale_with_proxy_routes() -> None:
    """``k_summary_scale`` covers the summary sequence in the flat layout."""

    kv_block_size = 64
    num_kv_blocks = -(-_GEOMETRY["seq_len_kv"] // kv_block_size)
    params = _make_params(
        k_block_size=16, with_summary_scale=True, kv_block_size=kv_block_size
    )
    assert params.k_summary_scale.shape == (
        _GEOMETRY["num_kv_heads"],
        flat_scale_numel(_GEOMETRY["batch_size"], num_kv_blocks, 16),
    )
    with pytest.raises(ValueError, match="k_summary_scale"):
        _validate(
            replace(params, k_summary_scale=None),
            summary_seq_len=num_kv_blocks,
        )
    with pytest.raises(ValueError, match="k_summary_scale"):
        _validate(
            replace(
                params,
                k_summary_scale=torch.rand(
                    (_GEOMETRY["num_kv_heads"], params.k_summary_scale.shape[1] + 1)
                ),
            ),
            summary_seq_len=num_kv_blocks,
        )
    _validate(params, summary_seq_len=num_kv_blocks)


def test_sage_params_reject_scales_on_another_device() -> None:
    params = _make_params()
    with pytest.raises(ValueError, match="device"):
        _validate(params, device=torch.device("cuda", 0))


def test_sage_module_exports() -> None:
    assert set(sage_module.__all__) >= {
        "SageAttentionConfig",
        "SageAttentionParams",
        "flat_scale_numel",
        "flat_scale_slot",
        "log2_block_size",
        "validate_sage_params",
    }


# ---------------------------------------------------------------------------
# Kernel test cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class _SageCase:
    """Geometry, profile and recipe shared by the dense and block-sparse cases."""

    name: str
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_qo_heads: int
    num_kv_heads: int
    q_block_size: int
    kv_block_size: int
    expected_kv_tile: int
    out_dtype: torch.dtype
    # The recipe defaults to the production ``(1, 16, no mean)``.
    sage_q_block_size: int = 1
    sage_k_block_size: int = 16
    # ``None`` follows ``sage_k_block_size`` (one scale-group geometry).
    sage_k_summary_block_size: int | None = None
    with_mean: bool = False
    qk_dtype: torch.dtype = _FP8
    # "auto" follows the planner's heuristic; "static" and "persistent" force
    # the selection and check the published policy.
    scheduler: str = "auto"

    @property
    def expected_q_tile(self) -> int:
        return 64 if self.expected_kv_tile == 256 else 128

    @property
    def summary_k_block_size(self) -> int:
        return self.sage_config.summary_k_block_size

    @property
    def sage_config(self) -> SageAttentionConfig:
        return SageAttentionConfig(
            q_block_size=self.sage_q_block_size,
            k_block_size=self.sage_k_block_size,
            v_mean=self.with_mean,
            k_summary_block_size=self.sage_k_summary_block_size,
        )


@dataclass(frozen=True, kw_only=True)
class _DenseSageCase(_SageCase):
    """One dense Sage problem: geometry selects the profile, scales the recipe."""

    mask_type: str = "dense"


@dataclass(frozen=True, kw_only=True)
class _SparseSageCase(_SageCase):
    """One block-sparse Sage problem with exact routes and optional proxies.

    The test shapes are too small for the planner to pick the persistent
    grid, so persistent cases force the selection.
    """

    use_proxy_routes: bool
    use_token_mask: bool = False
    sparse_format: str = "bsr"

    @property
    def num_kv_blocks(self) -> int:
        return -(-self.seq_len_kv // self.kv_block_size)

    @property
    def heads_q_per_kv(self) -> int:
        return self.num_qo_heads // self.num_kv_heads


def _with_persistent_replicas(cases, names):
    """Extend ``cases`` with a persistent-scheduler replica of each named case."""

    by_name = {case.name: case for case in cases}
    return cases + tuple(
        replace(by_name[name], name=f"{name}_persistent", scheduler="persistent")
        for name in names
    )


# ---------------------------------------------------------------------------
# Dense kernel fidelity and recipe tests
# ---------------------------------------------------------------------------

_NUM_KV_INSTANCES = 2

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
        name="kv256_k16_q1_bf16",
        **_KV256_MHA,
        out_dtype=torch.bfloat16,
    ),
    _DenseSageCase(
        name="kv256_gqa4_k16_q1_bf16",
        **_KV256_GQA,
        out_dtype=torch.bfloat16,
    ),
    _DenseSageCase(
        name="kv256_k16_q1_fp16_mean_causal",
        **_KV256_MHA,
        with_mean=True,
        out_dtype=torch.float16,
        mask_type="causal",
    ),
    _DenseSageCase(
        name="kv256_k64_q16_bf16_mean",
        **_KV256_MHA,
        sage_q_block_size=16,
        sage_k_block_size=64,
        with_mean=True,
        out_dtype=torch.bfloat16,
    ),
    _DenseSageCase(
        name="kv256_k256_q64_bf16",
        **_KV256_MHA,
        sage_q_block_size=64,
        sage_k_block_size=256,
        out_dtype=torch.bfloat16,
    ),
    _DenseSageCase(
        name="q128_k16_q1_bf16_mean",
        **_Q128_GQA,
        with_mean=True,
        out_dtype=torch.bfloat16,
    ),
    _DenseSageCase(
        name="q128_k32_q4_fp16_causal",
        **_Q128_GQA,
        sage_q_block_size=4,
        sage_k_block_size=32,
        out_dtype=torch.float16,
        mask_type="causal",
    ),
    # A 4-token K block splits every K32 fragment into eight scale groups.
    _DenseSageCase(
        name="kv256_k4_q1_bf16",
        **_KV256_MHA,
        sage_k_block_size=4,
        out_dtype=torch.bfloat16,
    ),
    _DenseSageCase(
        name="q128_k4_q1_bf16_mean_causal",
        **_Q128_GQA,
        sage_k_block_size=4,
        with_mean=True,
        out_dtype=torch.bfloat16,
        mask_type="causal",
    ),
    # A one-token K block gives every score its own scale; the tile's sfK
    # words live in SMEM.
    _DenseSageCase(
        name="kv256_k1_q1_bf16",
        **_KV256_MHA,
        sage_k_block_size=1,
        out_dtype=torch.bfloat16,
    ),
    _DenseSageCase(
        name="kv256_gqa4_k1_q1_fp16_mean_causal",
        **_KV256_GQA,
        sage_k_block_size=1,
        with_mean=True,
        out_dtype=torch.float16,
        mask_type="causal",
    ),
    _DenseSageCase(
        name="q128_k1_q1_bf16",
        **_Q128_GQA,
        sage_k_block_size=1,
        out_dtype=torch.bfloat16,
    ),
    # INT8 Q/K accumulate INT32 scores; the exact dot product leaves input
    # quantization as the only error, so the FP32 reference applies unchanged.
    _DenseSageCase(
        name="kv256_int8_k16_q1_bf16",
        **_KV256_MHA,
        out_dtype=torch.bfloat16,
        qk_dtype=torch.int8,
    ),
    _DenseSageCase(
        name="kv256_gqa4_int8_k64_q16_fp16_mean_causal",
        **_KV256_GQA,
        sage_q_block_size=16,
        sage_k_block_size=64,
        with_mean=True,
        out_dtype=torch.float16,
        mask_type="causal",
        qk_dtype=torch.int8,
    ),
    _DenseSageCase(
        name="q128_int8_k16_q1_bf16_mean",
        **_Q128_GQA,
        with_mean=True,
        out_dtype=torch.bfloat16,
        qk_dtype=torch.int8,
    ),
    _DenseSageCase(
        name="kv256_int8_k4_q1_fp16_causal",
        **_KV256_MHA,
        sage_k_block_size=4,
        out_dtype=torch.float16,
        mask_type="causal",
        qk_dtype=torch.int8,
    ),
    _DenseSageCase(
        name="q128_int8_k1_q1_bf16_mean",
        **_Q128_GQA,
        sage_k_block_size=1,
        with_mean=True,
        out_dtype=torch.bfloat16,
        qk_dtype=torch.int8,
    ),
)
# The dense persistent loop resolves every tile through the work tile; cover
# one E4M3 case, one causal case with the V mean, one INT8 case, and the
# Q128/KV128 profile on it.
_PERSISTENT_DENSE_SAGE_CASE_NAMES = (
    "kv256_k16_q1_bf16",
    "kv256_k16_q1_fp16_mean_causal",
    "kv256_int8_k16_q1_bf16",
    "q128_int8_k16_q1_bf16_mean",
    "kv256_k1_q1_bf16",
)
_DENSE_SAGE_CASES = _with_persistent_replicas(
    _DENSE_SAGE_CASES, _PERSISTENT_DENSE_SAGE_CASE_NAMES
)


def _cases_named(cases, *names: str):
    """Select test cases by name, in the order the names are given."""

    by_name = {case.name: case for case in cases}
    return tuple(by_name[name] for name in names)


def _random_scale(shape, low: float, high: float, device: torch.device) -> torch.Tensor:
    """Uniform random scales in ``[low, high)``, contiguous."""

    return (torch.rand(shape, device=device) * (high - low) + low).contiguous()


def _random_sage_inputs(case: _SageCase, device: torch.device):
    """Random 8-bit Q/K, E4M3 V and random positive scales in the flat layout."""

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
    q_scale = _random_scale(
        (
            case.num_qo_heads,
            flat_scale_numel(case.batch_size, case.seq_len_q, case.sage_q_block_size),
        ),
        0.05 * qk_scale_factor,
        0.2 * qk_scale_factor,
        device,
    )
    k_scale = _random_scale(
        (
            case.num_kv_heads,
            flat_scale_numel(case.batch_size, case.seq_len_kv, case.sage_k_block_size),
        ),
        0.5 * qk_scale_factor,
        2.0 * qk_scale_factor,
        device,
    )
    v_scale = _random_scale((case.num_kv_heads, _HEAD_DIM), 0.25, 1.0, device)
    v_mean = (
        torch.randn((case.num_kv_heads, _HEAD_DIM), device=device).contiguous()
        if case.with_mean
        else None
    )
    params = SageAttentionParams(
        q_scale=q_scale, k_scale=k_scale, v_scale=v_scale, v_mean=v_mean
    )
    return q, k, v, params


@torch.no_grad()
def _sage_dense_reference(
    case: _DenseSageCase,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    params: SageAttentionParams,
    *,
    sm_scale: float,
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
    kv_tile_size = case.expected_kv_tile
    q_real = dequantize_token_blocks(
        q, params.q_scale, block_size=case.sage_q_block_size
    )
    k_real = dequantize_token_blocks(
        k, params.k_scale, block_size=case.sage_k_block_size
    )
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
            if case.mask_type == "causal":
                visible_end = seq_len_kv - seq_len_q + query_idx + 1
            logits = (
                torch.einsum(
                    "hd,thd->ht", q_real[batch_idx, query_idx], keys[:visible_end]
                )
                * sm_scale
            )
            streams = []
            for tile_indices, columns in stream_tiles:
                state = None
                for tile_idx in tile_indices:
                    tile_columns = columns + tile_idx * kv_tile_size
                    tile_columns = tile_columns[tile_columns < visible_end]
                    if tile_columns.numel() == 0:
                        continue
                    tile_logits = logits[:, tile_columns]
                    local_max = tile_logits.amax(dim=-1)
                    new_max = (
                        local_max
                        if state is None
                        else torch.maximum(state[0], local_max)
                    )
                    probabilities = (
                        torch.exp(tile_logits - new_max.unsqueeze(-1))
                        * FP8_P_QUANT_SCALE
                    )
                    tile_acc = torch.einsum(
                        "ht,thd->hd",
                        probabilities.to(_FP8).float(),
                        values[tile_columns],
                    )
                    state = fold_stream(
                        state, new_max, probabilities.sum(dim=-1), tile_acc
                    )
                streams.append(state)
            normalized = merge_streams(streams)
            v_scale = params.v_scale.repeat_interleave(group_size, dim=0)
            result = normalized * v_scale
            if params.v_mean is not None:
                result = result + params.v_mean.repeat_interleave(group_size, dim=0)
            output[batch_idx, query_idx] = result
    return output


def _plan_sage(
    case: _SageCase, device: torch.device, *, max_blocks_per_row: int | None = None
):
    """Plan one case on a fresh wrapper and check the published tile policy.

    A forced scheduler patches the planner's selector: the persistent launch
    heuristic of dense plans, or the block-sparse scheduler selection, whose
    Q tile choice is kept.
    """

    from flashinfer.attention.prims_ts import BlockSparseTSWrapper
    from flashinfer.attention.prims_ts._block_sparse import config as sparse_config

    force_persistent = case.scheduler == "persistent"
    if isinstance(case, _SparseSageCase):
        selector_name = "_select_block_sparse_scheduler"
        auto_select_scheduler = sparse_config._select_block_sparse_scheduler

        def select_scheduler(**kwargs):
            q_tile_size, _ = auto_select_scheduler(**kwargs)
            return q_tile_size, force_persistent

        plan_kwargs = {
            "use_block_sparse": True,
            "max_blocks_per_row": max_blocks_per_row,
            "use_kv_valid_bits": case.use_token_mask,
            "sparse_format": case.sparse_format,
            "use_proxy_routes": case.use_proxy_routes,
        }
    else:
        selector_name = "_select_persistent_launch"

        def select_scheduler(**_kwargs):
            return force_persistent

        plan_kwargs = {"use_block_sparse": False, "mask_type": case.mask_type}

    wrapper = BlockSparseTSWrapper()
    sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            if case.scheduler != "auto":
                monkeypatch.setattr(sparse_config, selector_name, select_scheduler)
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
                q_data_type=case.qk_dtype,
                kv_data_type=case.qk_dtype,
                # INT8 Q/K name their E4M3 V; E4M3 callers rely on the default.
                v_data_type=_FP8 if case.qk_dtype == torch.int8 else None,
                o_data_type=case.out_dtype,
                sage_config=case.sage_config,
                **plan_kwargs,
            )
    finally:
        sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    policy = dict(wrapper._policy)
    assert policy["tile_size_q"] == case.expected_q_tile
    assert policy["tile_size_kv"] == case.expected_kv_tile
    if case.scheduler != "auto":
        assert policy["scheduler"] == case.scheduler
    return wrapper


def _run_dense_sage(
    case: _DenseSageCase,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    params: SageAttentionParams,
    device: torch.device,
    *,
    sm_scale: float,
) -> torch.Tensor:
    """Plan ``case`` and run the dense kernel on ``q``, ``k``, ``v``, synchronized."""

    wrapper = _plan_sage(case, device)
    actual = wrapper.run(q, k, v, sage=params, sm_scale=sm_scale)
    torch.cuda.synchronize()
    return actual


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
    expected = _sage_dense_reference(case, q, k, v, params, sm_scale=sm_scale)
    actual = _run_dense_sage(case, q, k, v, params, device, sm_scale=sm_scale)
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
    "case",
    _cases_named(_DENSE_SAGE_CASES, "kv256_k16_q1_bf16", "kv256_int8_k16_q1_bf16"),
    ids=lambda case: case.name,
)
@torch.no_grad()
def test_dense_sage_persistent_scheduler_matches_static_grid_bitwise(
    case: _DenseSageCase,
) -> None:
    """The work-tile loop only reorders tiles, so both schedulers publish the same bits."""

    torch.manual_seed(20260908)
    device = torch.device("cuda", 0)
    q, k, v, params = _random_sage_inputs(case, device)
    sm_scale = _HEAD_DIM**-0.5
    static, persistent = (
        _run_dense_sage(
            replace(case, scheduler=scheduler),
            q,
            k,
            v,
            params,
            device,
            sm_scale=sm_scale,
        )
        for scheduler in ("static", "persistent")
    )
    assert torch.equal(static, persistent)


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
        batch, q_row, case.seq_len_q, log2_block_size(case.sage_q_block_size)
    )
    params.q_scale[:, q_slot] = 1e-3
    for token in (k_max_token, k_min_token):
        k_slot = flat_scale_slot(
            batch, token, case.seq_len_kv, log2_block_size(case.sage_k_block_size)
        )
        params.k_scale[:, k_slot] = 1e-2
    sm_scale = _HEAD_DIM**-0.5
    expected = _sage_dense_reference(case, q, k, v, params, sm_scale=sm_scale)
    actual = _run_dense_sage(case, q, k, v, params, device, sm_scale=sm_scale)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual[batch, q_row].float(), expected[batch, q_row], rtol=8e-3, atol=2e-3
    )
    torch.testing.assert_close(actual.float(), expected, rtol=8e-3, atol=2e-3)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "case",
    _cases_named(_DENSE_SAGE_CASES, "kv256_k16_q1_bf16", "q128_k16_q1_bf16_mean"),
    ids=("kv256-default-recipe", "q128-explicit-recipe"),
)
@torch.no_grad()
def test_one_shot_dense_sage_matches_the_planned_wrapper(case: _DenseSageCase) -> None:
    """``block_sparse_attention(sage=...)`` plans the recipe and runs the same launch.

    Without ``sage_config`` the recipe is the default one with a V mean exactly
    when the scales carry one; the caller-owned ``out`` fixes the output dtype.
    """

    from flashinfer.attention.prims_ts import block_sparse_attention

    torch.manual_seed(20260915)
    device = torch.device("cuda", 0)
    q, k, v, params = _random_sage_inputs(case, device)
    wrapper = _plan_sage(case, device)
    expected = wrapper.run(q, k, v, sage=params)
    out = torch.empty_like(expected)
    actual = block_sparse_attention(
        q,
        k,
        v,
        None,
        None,
        case.q_block_size,
        case.kv_block_size,
        use_block_sparse=False,
        mask_type=case.mask_type,
        out=out,
        sage=params,
        sage_config=case.sage_config if case.with_mean else None,
    )
    torch.cuda.synchronize()
    assert actual is out
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_one_shot_sage_config_requires_the_scale_tensors() -> None:
    """A recipe without scales has nothing to run; the one-shot rejects it."""

    from flashinfer.attention.prims_ts import block_sparse_attention

    q = torch.empty((1, 64, 1, _HEAD_DIM), dtype=_FP8)
    with pytest.raises(ValueError, match="sage_config requires"):
        block_sparse_attention(
            q,
            q,
            q,
            None,
            None,
            64,
            64,
            use_block_sparse=False,
            sage_config=SageAttentionConfig(),
        )


def _quantized_recipe_inputs(case: _SageCase, q_magnitude: float, device: torch.device):
    """Quantize heavy-tailed BF16 Q/K/V with the case's Q/K dtype and block sizes.

    Heavy-tailed inputs follow TensorRT-LLM's recipe test; ``q_magnitude``
    scales Q. V takes per-channel E4M3 scales without a mean. Returns the
    BF16 inputs, their quantized counterparts and the scale parameters.
    """

    q_shape = (case.batch_size, case.seq_len_q, case.num_qo_heads, _HEAD_DIM)
    kv_shape = (case.batch_size, case.seq_len_kv, case.num_kv_heads, _HEAD_DIM)
    q_bf16 = heavy_tailed(q_shape, device=device, magnitude=q_magnitude)
    k_bf16 = heavy_tailed(kv_shape, device=device)
    v_bf16 = heavy_tailed(kv_shape, device=device)
    q_quant, q_scale = quantize_token_blocks(
        q_bf16, block_size=case.sage_q_block_size, dtype=case.qk_dtype
    )
    k_quant, k_scale = quantize_token_blocks(
        k_bf16, block_size=case.sage_k_block_size, dtype=case.qk_dtype
    )
    v_fp8, v_scale, _ = quantize_v_channels(v_bf16)
    params = SageAttentionParams(q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)
    return (q_bf16, k_bf16, v_bf16), (q_quant, k_quant, v_fp8), params


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
        _DENSE_SAGE_CASES,
        "kv256_k16_q1_bf16",
        "q128_k16_q1_bf16_mean",
        "kv256_k4_q1_bf16",
        "kv256_k1_q1_bf16",
    ),
    ids=("kv256", "q128", "kv256-k4", "kv256-k1"),
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
    recipe_case = replace(
        case,
        with_mean=False,
        out_dtype=torch.bfloat16,
        qk_dtype=qk_dtype,
    )
    (q_bf16, k_bf16, v_bf16), (q_quant, k_quant, v_fp8), params = (
        _quantized_recipe_inputs(recipe_case, q_magnitude, device)
    )
    sm_scale = _HEAD_DIM**-0.5
    group_size = case.num_qo_heads // case.num_kv_heads
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_bf16.float().permute(0, 2, 1, 3),
        k_bf16.float().repeat_interleave(group_size, dim=2).permute(0, 2, 1, 3),
        v_bf16.float().repeat_interleave(group_size, dim=2).permute(0, 2, 1, 3),
        scale=sm_scale,
    ).permute(0, 2, 1, 3)

    actual = _run_dense_sage(
        recipe_case, q_quant, k_quant, v_fp8, params, device, sm_scale=sm_scale
    ).float()
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
    its norm. The bounds leave about 1.2x to 1.7x headroom over the maxima
    measured on B200 for both profiles and both route kinds: INT8 reaches
    0.043 with logits spread by one standard deviation and 0.055 unscaled
    against a bound of 0.075; E4M3 Q/K, whose three mantissa bits quantize
    each input about ten times as coarsely, reaches 0.082 and 0.133 against
    bounds of 0.10 and 0.20. A misaddressed scale moves a head by well over
    half its norm.
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


# 1000 tokens leave a 40-token ragged block; 500 tokens a 52-token one. The
# sparse cases choose their KV block size per case.
_SPARSE_KV256_MHA = dict(
    batch_size=2,
    seq_len_q=128,
    seq_len_kv=1000,
    num_qo_heads=2,
    num_kv_heads=2,
    q_block_size=64,
    expected_kv_tile=256,
)
_SPARSE_Q128_GQA = {
    key: value for key, value in _Q128_GQA.items() if key != "kv_block_size"
}

_SPARSE_SAGE_CASES = (
    _SparseSageCase(
        name="kv256_exact_k16_q1_bf16",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        out_dtype=torch.bfloat16,
        use_proxy_routes=False,
    ),
    _SparseSageCase(
        name="kv256_exact_bk128_k64_q16_fp16_mask",
        **_SPARSE_KV256_MHA,
        kv_block_size=128,
        sage_q_block_size=16,
        sage_k_block_size=64,
        out_dtype=torch.float16,
        use_proxy_routes=False,
        use_token_mask=True,
    ),
    _SparseSageCase(
        name="kv256_proxy_k16_q1_bf16_mean",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        with_mean=True,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
    ),
    _SparseSageCase(
        name="kv256_proxy_bk128_k64_q64_bf16_bitmask",
        **_SPARSE_KV256_MHA,
        kv_block_size=128,
        sage_q_block_size=64,
        sage_k_block_size=64,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
        sparse_format="bitmask",
    ),
    # One 128-token K block scale spans several summary atoms of the proxy
    # sequence (16 summaries for 1000 tokens in 64-token blocks).
    _SparseSageCase(
        name="kv256_proxy_k128_q16_fp16",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        sage_q_block_size=16,
        sage_k_block_size=128,
        out_dtype=torch.float16,
        use_proxy_routes=True,
    ),
    # 321 summaries span two proxy groups; the ragged tail (8 tokens) is
    # summary 64 of the second group, an atom owned by the upper spatial half.
    _SparseSageCase(
        name="kv256_proxy_two_groups_k16_q1_bf16",
        **{**_SPARSE_KV256_MHA, "seq_len_kv": 20520},
        kv_block_size=64,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
    ),
    _SparseSageCase(
        name="q128_gqa8_exact_k16_q1_bf16_mask",
        **_SPARSE_Q128_GQA,
        kv_block_size=64,
        out_dtype=torch.bfloat16,
        use_proxy_routes=False,
        use_token_mask=True,
    ),
    # Without a token mask the Q128 exact route transports two atom origins per
    # record, so the load warp stages sfK from the broadcast origin pair.
    _SparseSageCase(
        name="q128_gqa8_exact_k16_q1_bf16",
        **_SPARSE_Q128_GQA,
        kv_block_size=64,
        out_dtype=torch.bfloat16,
        use_proxy_routes=False,
    ),
    # A 4-token K block stages 64 (KV256) or 32 (KV128) sfK words per route,
    # two 32-lane rounds of the load warp on KV256.
    _SparseSageCase(
        name="kv256_exact_k4_q1_bf16_mask",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        sage_k_block_size=4,
        out_dtype=torch.bfloat16,
        use_proxy_routes=False,
        use_token_mask=True,
    ),
    _SparseSageCase(
        name="kv256_proxy_k4_q1_bf16_mean",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        sage_k_block_size=4,
        with_mean=True,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
    ),
    _SparseSageCase(
        name="q128_gqa8_int8_proxy_k4_q1_fp16",
        **_SPARSE_Q128_GQA,
        kv_block_size=64,
        sage_k_block_size=4,
        out_dtype=torch.float16,
        use_proxy_routes=True,
        qk_dtype=torch.int8,
    ),
    # A one-token K block stages 256 (KV256) or 128 (KV128) sfK words per
    # route and reads them from SMEM; the summary sequence gets one scale per
    # summary.
    _SparseSageCase(
        name="kv256_exact_k1_q1_bf16_mask",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        sage_k_block_size=1,
        out_dtype=torch.bfloat16,
        use_proxy_routes=False,
        use_token_mask=True,
    ),
    _SparseSageCase(
        name="kv256_proxy_k1_q1_bf16_mean",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        sage_k_block_size=1,
        with_mean=True,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
    ),
    _SparseSageCase(
        name="q128_gqa8_int8_proxy_k1_q1_fp16",
        **_SPARSE_Q128_GQA,
        kv_block_size=64,
        sage_k_block_size=1,
        out_dtype=torch.float16,
        use_proxy_routes=True,
        qk_dtype=torch.int8,
    ),
    # Summary scales with their own K block size: exact routes keep the
    # register strategy of the 16-token block while proxy routes read
    # one-token summary scales from SMEM.
    _SparseSageCase(
        name="kv256_proxy_k16_s1_q1_bf16_mean",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        sage_k_summary_block_size=1,
        with_mean=True,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
    ),
    _SparseSageCase(
        name="q128_gqa8_int8_proxy_k16_s1_q1_fp16",
        **_SPARSE_Q128_GQA,
        kv_block_size=64,
        sage_k_summary_block_size=1,
        out_dtype=torch.float16,
        use_proxy_routes=True,
        qk_dtype=torch.int8,
    ),
    # Both route kinds in SMEM with different word counts.
    _SparseSageCase(
        name="kv256_proxy_k4_s1_q1_bf16",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        sage_k_block_size=4,
        sage_k_summary_block_size=1,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
    ),
    # Exact routes in SMEM, summaries in registers.
    _SparseSageCase(
        name="kv256_proxy_k1_s16_q1_bf16",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        sage_k_block_size=1,
        sage_k_summary_block_size=16,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
    ),
    # The ragged final summary of a two-group route under the summary geometry.
    _SparseSageCase(
        name="kv256_proxy_two_groups_k16_s1_q1_bf16",
        **{**_SPARSE_KV256_MHA, "seq_len_kv": 20520},
        kv_block_size=64,
        sage_k_summary_block_size=1,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
    ),
    _SparseSageCase(
        name="q128_gqa8_proxy_bk128_k32_q4_fp16_mean",
        **_SPARSE_Q128_GQA,
        kv_block_size=128,
        sage_q_block_size=4,
        sage_k_block_size=32,
        with_mean=True,
        out_dtype=torch.float16,
        use_proxy_routes=True,
    ),
    # 131 summaries span two KV128 proxy groups; the ragged tail (3 tokens) is
    # summary 2 of the second group.
    _SparseSageCase(
        name="q128_gqa8_proxy_two_groups_k16_q1_bf16",
        **{**_SPARSE_Q128_GQA, "seq_len_kv": 8323},
        kv_block_size=64,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
    ),
    # INT8 Q/K on both routes: the exact route runs the masked INT32 store
    # path and the proxy route shifts the ragged tail summary in score units.
    _SparseSageCase(
        name="kv256_int8_exact_k16_q1_bf16_mask",
        **_SPARSE_KV256_MHA,
        kv_block_size=64,
        out_dtype=torch.bfloat16,
        use_proxy_routes=False,
        use_token_mask=True,
        qk_dtype=torch.int8,
    ),
    _SparseSageCase(
        name="kv256_int8_proxy_bk128_k64_q64_bf16_mean",
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
        name="kv256_int8_proxy_tail_lane0_k64_q64_bf16",
        **{**_SPARSE_KV256_MHA, "seq_len_kv": 2100},
        kv_block_size=64,
        sage_q_block_size=64,
        sage_k_block_size=64,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
        qk_dtype=torch.int8,
    ),
    # 100 summaries put the ragged tail summary in atom 1, which the second
    # spatial half of the KV256 softmax owns as its first atom.
    _SparseSageCase(
        name="kv256_proxy_tail_atom1_k16_q1_bf16",
        **{**_SPARSE_KV256_MHA, "seq_len_kv": 6370},
        kv_block_size=64,
        out_dtype=torch.bfloat16,
        use_proxy_routes=True,
    ),
    _SparseSageCase(
        name="q128_gqa8_int8_proxy_k16_q1_fp16",
        **_SPARSE_Q128_GQA,
        kv_block_size=64,
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
    "kv256_proxy_k1_q1_bf16_mean",
)
_SPARSE_SAGE_CASES = _with_persistent_replicas(
    _SPARSE_SAGE_CASES, _PERSISTENT_SPARSE_SAGE_CASE_NAMES
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


def _sparse_routing(
    case: _SparseSageCase, patterns, device: torch.device, summaries=None
) -> dict[str, torch.Tensor]:
    """Return the ``run`` keyword arguments routing one pattern set.

    Exact routes come as BSR or bitmask tensors; proxy routes add the K and V
    ``summaries``.
    """

    if case.sparse_format == "bsr":
        block_indptr, block_indices = make_bsr(patterns, device)
        routing = {"block_indptr": block_indptr, "block_indices": block_indices}
    else:
        routing = {
            "exact_block_bits": make_exact_block_bits(
                patterns, case.num_kv_blocks, device
            )
        }
    if summaries is not None:
        routing.update(k_summary=summaries[0], v_summary=summaries[1])
    return routing


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
    """Random 8-bit Q/K, E4M3 V and proxy summaries with random positive scales."""

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
        qk_scale_factor = _qk_scale_factor(case.qk_dtype)
        k_summary_scale = _random_scale(
            (
                case.num_kv_heads,
                flat_scale_numel(
                    case.batch_size, case.num_kv_blocks, case.summary_k_block_size
                ),
            ),
            0.5 * qk_scale_factor,
            2.0 * qk_scale_factor,
            device,
        )
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

    q_real = dequantize_token_blocks(
        q, params.q_scale, block_size=case.sage_q_block_size
    )
    k_real = dequantize_token_blocks(
        k, params.k_scale, block_size=case.sage_k_block_size
    )
    v_raw = v.float()
    if summaries is not None:
        k_summary_real = dequantize_token_blocks(
            summaries[0], params.k_summary_scale, block_size=case.summary_k_block_size
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
                        # The row sum accumulates the FP32 probabilities; only
                        # the PV operand is quantized, as in the kernel.
                        probabilities = (
                            torch.exp(logits - new_max.unsqueeze(-1))
                            * FP8_P_QUANT_SCALE
                        )
                        acc = torch.einsum(
                            "thc,cd->thd", probabilities.to(_FP8).float(), value_stack
                        )
                        streams[stream_idx] = fold_stream(
                            state, new_max, probabilities.sum(dim=-1), acc
                        )
                if all(state is None for state in streams):
                    continue
                result = merge_streams(streams) * params.v_scale[kv_head_idx]
                if params.v_mean is not None:
                    result = result + params.v_mean[kv_head_idx]
                output[batch_idx, row_begin:row_end, head_slice] = result
    return output


def _run_sparse_sage(
    case: _SparseSageCase,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    params: SageAttentionParams,
    summaries,
    patterns,
    valid_bits: torch.Tensor | None,
    device: torch.device,
    *,
    sm_scale: float,
) -> torch.Tensor:
    """Plan ``case`` for ``patterns`` and run the sparse kernel, synchronized."""

    wrapper = _plan_sage(case, device, max_blocks_per_row=widest_bsr_row(patterns))
    actual = wrapper.run(
        q,
        k,
        v,
        kv_valid_bits=valid_bits,
        sm_scale=sm_scale,
        sage=params,
        **_sparse_routing(case, patterns, device, summaries),
    )
    torch.cuda.synchronize()
    return actual


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
    actual = _run_sparse_sage(
        case,
        q,
        k,
        v,
        params,
        summaries,
        patterns,
        valid_bits,
        device,
        sm_scale=sm_scale,
    )
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
    lane's weight here. Masked tokens carry a constant V well away from the
    kept tokens' values so any leaked mass shows up in the output.
    """

    (case,) = _cases_named(_SPARSE_SAGE_CASES, "kv256_int8_exact_k16_q1_bf16_mask")
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
    params = SageAttentionParams(q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)
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
    actual = _run_sparse_sage(
        case,
        q_quant,
        k_quant,
        v_fp8,
        params,
        None,
        patterns,
        valid_bits,
        device,
        sm_scale=sm_scale,
    )
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
        "kv256_proxy_k4_q1_bf16_mean",
        "kv256_proxy_k1_q1_bf16_mean",
    ),
    ids=(
        "kv256-exact",
        "kv256-proxy",
        "q128-proxy",
        "kv256-proxy-k4",
        "kv256-proxy-k1",
    ),
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
        with_mean=False,
        out_dtype=torch.bfloat16,
        use_token_mask=False,
        qk_dtype=qk_dtype,
    )
    patterns = _sparse_patterns(recipe_case, generator)
    (q_bf16, k_bf16, v_bf16), (q_quant, k_quant, v_fp8), params = (
        _quantized_recipe_inputs(recipe_case, q_magnitude, device)
    )
    summaries = None
    if case.use_proxy_routes:
        k_summary_bf16 = block_mean(k_bf16, case.kv_block_size, torch.bfloat16)
        v_summary_bf16 = block_mean(v_bf16, case.kv_block_size, torch.bfloat16)
        k_summary_quant, k_summary_scale = quantize_token_blocks(
            k_summary_bf16,
            block_size=recipe_case.summary_k_block_size,
            dtype=qk_dtype,
        )
        v_summary_fp8 = quantize_v_channels_with_scale(v_summary_bf16, params.v_scale)
        summaries = (k_summary_quant, v_summary_fp8)
        params = replace(params, k_summary_scale=k_summary_scale)
    sm_scale = _HEAD_DIM**-0.5
    expected = _sparse_bf16_reference(
        recipe_case, q_bf16, k_bf16, v_bf16, patterns, sm_scale=sm_scale
    )
    actual = _run_sparse_sage(
        recipe_case,
        q_quant,
        k_quant,
        v_fp8,
        params,
        summaries,
        patterns,
        None,
        device,
        sm_scale=sm_scale,
    ).float()
    _assert_recipe_close(actual, expected, q_magnitude=q_magnitude, qk_dtype=qk_dtype)
