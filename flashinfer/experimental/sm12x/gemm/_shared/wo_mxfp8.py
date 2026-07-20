# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/gemm/wo_projection.py @ 32d189d4 (2026-07-18) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ..._lib.scratch_layout import (
    layout_wo_projection as _layout_wo_projection,
    materialize_scratch_strided_view as _materialize_arena_strided_view,
    materialize_scratch_view as _materialize_arena_view,
    wo_mxfp8_scale_physical_shape as _wo_mxfp8_scale_physical_shape,
)
from flashinfer.experimental.sm12x._lib.utils import cuda_stream_to_int
from flashinfer.experimental.sm12x._lib.dense_gemm import (
    _WO_SPARK_MAX_SMS,
    dense_gemm,
    dense_gemm_fused_quant_a,
    dense_gemm_fused_quant_a_grouped,
)
from flashinfer.experimental.sm12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)

FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
MXFP8_SCALE_VEC_SIZE = 32
MXFP8_SCALE_ROW_TILE = 128
MXFP8_SCALE_K_TILE = 4
WO_A_INPUT_QUANT_GROUP_SIZE = MXFP8_SCALE_VEC_SIZE
_WO_QUANT_CHUNKS_PER_PROGRAM = int(
    os.environ.get("FLASHINFER_EXP_SM12X_WO_QUANT_CHUNKS_PER_PROGRAM", "16")
)
if _WO_QUANT_CHUNKS_PER_PROGRAM not in (1, 2, 4, 8, 16, 32):
    raise ValueError(
        "FLASHINFER_EXP_SM12X_WO_QUANT_CHUNKS_PER_PROGRAM must be one of 1, 2, 4, 8, 16, or 32, got "
        f"{_WO_QUANT_CHUNKS_PER_PROGRAM}"
    )
_ALPHA_ONE_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def _should_use_exact_b16_wo(*, tokens: int, sm_count: int) -> bool:
    return int(tokens) == 16 and int(sm_count) <= _WO_SPARK_MAX_SMS


@dataclass(frozen=True)
class MXFP8Rows:
    """Row-wise MXFP8 operand and scales for `dense_gemm`.

    `values` has logical dense-GEMM shape `[M, K, L]`, but for `L > 1` it is a
    strided view over physical `[L, M, K]` storage because the current CuTe
    dense kernel consumes raw pointers and reconstructs its own K-major layout.
    `scale_rows` is the compact row/chunk view `[L, M, K/32]`, and `scale_mma`
    is the strided `[32, 4, ceil(M/128), 4, ceil(K/128), L]` view consumed by
    the CuTe kernel. `values_tiled`, when present, is an additional load-time
    packed copy for a specialized dense-GEMM RHS; `values` remains the logical
    operand used by every fallback path.
    """

    values: torch.Tensor
    scale_rows: torch.Tensor
    scale_mma: torch.Tensor
    values_tiled: torch.Tensor | None = None


@dataclass(frozen=True)
class WOProjectionMXFP8Weights:
    """MXFP8 WO-A/WO-B weights in the layouts consumed by the two GEMMs.

    `sfb_k_replicated` records scale provenance: True only when the per-32
    UE8M0 rows were expanded from 128x128 block scales, so the four SFB bytes
    per 128-wide k tile are identical by construction and the dense GEMM may
    load one byte per stage (sfb_k_reuse). Natively per-32-quantized weights
    (quantize_wo_projection_weights_mxfp8_torch) must leave this False.
    """

    wo_a: MXFP8Rows
    wo_b: MXFP8Rows
    groups: int
    group_width: int
    rank: int
    hidden: int
    sfb_k_replicated: bool = False


@dataclass(frozen=True)
class _WOProjectionScratchViews:
    x_q: MXFP8Rows
    tmp: torch.Tensor
    tmp_q: MXFP8Rows
    output: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class WOProjectionBinding:
    source_tgd: torch.Tensor
    weights: WOProjectionMXFP8Weights
    x_q: MXFP8Rows
    tmp: torch.Tensor
    tmp_q: MXFP8Rows
    output: torch.Tensor
    return_3d: bool = False
    # DeepGEMM-style regime hint forwarded to the wo_b up-projection (N=hidden,
    # the n>1536 path). None keeps the M-independent default tile.
    expected_m: int | None = None

    def run(self, *, stream: object = None) -> torch.Tensor:
        return wo_projection_mxfp8(binding=self, stream=stream)


@dataclass(frozen=True, kw_only=True)
class WOProjectionInvRopeBinding:
    o: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    weights: WOProjectionMXFP8Weights
    x_q: MXFP8Rows
    tmp: torch.Tensor
    tmp_q: MXFP8Rows
    output: torch.Tensor
    heads_per_group: int
    nope_dim: int = 448
    rope_dim: int = 64
    return_3d: bool = False
    # DeepGEMM-style regime hint forwarded to the wo_b up-projection.
    expected_m: int | None = None

    def run(self, *, stream: object = None) -> torch.Tensor:
        return wo_projection_inv_rope_mxfp8(binding=self, stream=stream)


@dataclass(frozen=True, kw_only=True)
class WOProjectionScratchCaps:
    device: torch.device | str
    max_tokens: int
    groups: int
    group_width: int
    rank: int
    hidden: int
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", max(int(self.max_tokens), 1))
        object.__setattr__(self, "groups", max(int(self.groups), 1))
        object.__setattr__(self, "group_width", max(int(self.group_width), 1))
        object.__setattr__(self, "rank", max(int(self.rank), 1))
        object.__setattr__(self, "hidden", max(int(self.hidden), 1))
        if self.dtype != torch.bfloat16:
            raise ValueError(
                "WO projection scratch currently supports torch.bfloat16 outputs, "
                f"got {self.dtype}"
            )
        _check_mxfp8_k(self.group_width)
        _check_mxfp8_k(self.rank * self.groups)


@dataclass(frozen=True)
class WOProjectionScratchPlan:
    caps: WOProjectionScratchCaps
    layout: object
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        source_tgd: torch.Tensor,
        weights: WOProjectionMXFP8Weights,
        return_3d: bool = False,
        expected_m: int | None = None,
    ) -> WOProjectionBinding:
        tokens = _validate_wo_projection_inputs(source_tgd, weights)
        self._check_live_capacity(tokens=tokens, weights=weights)
        views = self._views_from_scratch(scratch=scratch, tokens=tokens)
        return _build_wo_projection_binding_from_views(
            x_q=views.x_q,
            tmp=views.tmp,
            tmp_q=views.tmp_q,
            output=views.output,
            source_tgd=source_tgd,
            weights=weights,
            return_3d=return_3d,
            expected_m=expected_m,
        )

    def bind_inv_rope(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        weights: WOProjectionMXFP8Weights,
        heads_per_group: int,
        nope_dim: int = 448,
        rope_dim: int = 64,
        return_3d: bool = False,
        expected_m: int | None = None,
    ) -> WOProjectionInvRopeBinding:
        tokens = _validate_wo_projection_inv_rope_inputs(
            o=o,
            weights=weights,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )
        self._check_live_capacity(tokens=tokens, weights=weights)
        views = self._views_from_scratch(scratch=scratch, tokens=tokens)
        return _build_wo_projection_inv_rope_binding_from_views(
            x_q=views.x_q,
            tmp=views.tmp,
            tmp_q=views.tmp_q,
            output=views.output,
            o=o,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            weights=weights,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            return_3d=return_3d,
            expected_m=expected_m,
        )

    def _views_from_scratch(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        tokens: int | None = None,
    ) -> _WOProjectionScratchViews:
        max_tokens = int(self.caps.max_tokens)
        tokens = max_tokens if tokens is None else int(tokens)
        if tokens <= 0 or tokens > max_tokens:
            raise ValueError(
                f"WO projection tokens={tokens} exceeds scratch capacity {max_tokens}"
            )
        scratch_storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="WO projection",
        )
        groups = int(self.caps.groups)
        group_width = int(self.caps.group_width)
        rank = int(self.caps.rank)
        hidden = int(self.caps.hidden)
        layout = _layout_wo_projection(
            offset_bytes=0,
            tokens=tokens,
            groups=groups,
            group_width=group_width,
            rank=rank,
            hidden=hidden,
        )
        if int(layout.nbytes) > int(self.layout.nbytes):
            raise RuntimeError(
                "WO projection scratch layout exceeds reserved scratch "
                f"capacity: requested={layout.nbytes}, reserved={self.layout.nbytes}"
            )

        def mxfp8_rows(
            *,
            values_offset_bytes: int,
            scale_rows_offset_bytes: int,
            scale_mma_offset_bytes: int,
            m: int,
            k: int,
            num_groups: int,
        ) -> MXFP8Rows:
            if num_groups == 1:
                values, _ = _materialize_arena_view(
                    scratch_storage,
                    offset_bytes=values_offset_bytes,
                    shape=(m, k),
                    dtype=torch.float8_e4m3fn,
                )
            else:
                values, _ = _materialize_arena_strided_view(
                    scratch_storage,
                    offset_bytes=values_offset_bytes,
                    shape=(m, k, num_groups),
                    stride=(k, 1, m * k),
                    dtype=torch.float8_e4m3fn,
                )
            scale_rows, _ = _materialize_arena_view(
                scratch_storage,
                offset_bytes=scale_rows_offset_bytes,
                shape=(num_groups, m, k // MXFP8_SCALE_VEC_SIZE),
                dtype=torch.float8_e8m0fnu,
            )
            scale_physical_u8, _ = _materialize_arena_view(
                scratch_storage,
                offset_bytes=scale_mma_offset_bytes,
                shape=_wo_mxfp8_scale_physical_shape(
                    m=m,
                    k=k,
                    num_groups=num_groups,
                ),
                dtype=torch.uint8,
            )
            if m % MXFP8_SCALE_ROW_TILE:
                scale_physical_u8.fill_(127)
            scale_mma = scale_physical_u8.view(torch.float8_e8m0fnu).permute(
                3,
                4,
                1,
                5,
                2,
                0,
            )
            return MXFP8Rows(
                values=values,
                scale_rows=scale_rows,
                scale_mma=scale_mma,
            )

        x_q = mxfp8_rows(
            values_offset_bytes=layout.x_q_values_offset_bytes,
            scale_rows_offset_bytes=layout.x_q_scale_rows_offset_bytes,
            scale_mma_offset_bytes=layout.x_q_scale_mma_offset_bytes,
            m=tokens,
            k=group_width,
            num_groups=groups,
        )
        tmp, _ = _materialize_arena_strided_view(
            scratch_storage,
            offset_bytes=layout.tmp_offset_bytes,
            shape=(tokens, rank, groups),
            stride=(rank, 1, tokens * rank),
            dtype=torch.bfloat16,
        )
        tmp_q = mxfp8_rows(
            values_offset_bytes=layout.tmp_q_values_offset_bytes,
            scale_rows_offset_bytes=layout.tmp_q_scale_rows_offset_bytes,
            scale_mma_offset_bytes=layout.tmp_q_scale_mma_offset_bytes,
            m=tokens,
            k=rank * groups,
            num_groups=1,
        )
        output, _ = _materialize_arena_view(
            scratch_storage,
            offset_bytes=layout.output_offset_bytes,
            shape=(tokens, hidden, 1),
            dtype=torch.bfloat16,
        )
        return _WOProjectionScratchViews(
            x_q=x_q,
            tmp=tmp,
            tmp_q=tmp_q,
            output=output,
        )

    def _check_live_capacity(
        self,
        *,
        tokens: int,
        weights: WOProjectionMXFP8Weights,
    ) -> None:
        if tokens > int(self.caps.max_tokens):
            raise ValueError(
                f"WO projection tokens {tokens} exceed scratch capacity {self.caps.max_tokens}"
            )
        expected = (
            int(self.caps.groups),
            int(self.caps.group_width),
            int(self.caps.rank),
            int(self.caps.hidden),
        )
        actual = (
            int(weights.groups),
            int(weights.group_width),
            int(weights.rank),
            int(weights.hidden),
        )
        if actual != expected:
            raise ValueError(
                "WO projection weights do not match scratch caps: "
                f"weights={actual}, caps={expected}"
            )


def _check_gpu_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA")


def _as_grouped_mkl(source: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    _check_gpu_tensor("source", source)
    if source.ndim == 2:
        m, k = source.shape
        grouped = source.reshape(m, k, 1).permute(2, 0, 1).contiguous()
        return grouped, m, k, 1
    if source.ndim == 3:
        m, k, groups = source.shape
        grouped = source.permute(2, 0, 1).contiguous()
        return grouped, m, k, groups
    raise ValueError(
        f"source must have shape [M,K] or [M,K,L], got {tuple(source.shape)}"
    )


def _check_mxfp8_k(k: int) -> None:
    if k <= 0 or k % 128 != 0:
        raise ValueError(
            f"MXFP8 dense_gemm K must be a positive multiple of 128, got {k}"
        )


def _wo_quant_chunks_per_program(k: int) -> int:
    chunks = k // MXFP8_SCALE_VEC_SIZE
    # tl.arange requires a power-of-two extent. K is only required to be a
    # multiple of 128, so first round down before finding a divisor (for
    # example, K=384 has 12 scale chunks and must use four per program).
    chunks_per_program = min(
        _WO_QUANT_CHUNKS_PER_PROGRAM,
        1 << (chunks.bit_length() - 1),
    )
    while chunks % chunks_per_program:
        chunks_per_program //= 2
    return chunks_per_program


def _cached_alpha_one(device: torch.device | str) -> torch.Tensor:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    key = (resolved.type, resolved.index)
    alpha = _ALPHA_ONE_CACHE.get(key)
    if alpha is None or alpha.device != resolved:
        alpha = torch.ones((1,), dtype=torch.float32, device=resolved)
        _ALPHA_ONE_CACHE[key] = alpha
    return alpha


@triton.jit
def _quantize_grouped_tgd_to_tdg_kernel(
    source,
    values,
    scale_rows,
    scale_mma,
    tokens,
    groups: tl.constexpr,
    group_width: tl.constexpr,
    source_stride_t,
    source_stride_g,
    source_stride_d,
    values_stride_t,
    values_stride_d,
    values_stride_g,
    scale_mma_s0,
    scale_mma_s1,
    scale_mma_s2,
    scale_mma_s3,
    scale_mma_s4,
    scale_mma_s5,
    CHUNKS_PER_PROGRAM: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    group = tl.program_id(1)
    chunk_block = tl.program_id(2)
    chunk = chunk_block * CHUNKS_PER_PROGRAM + tl.arange(0, CHUNKS_PER_PROGRAM)
    d = chunk[:, None] * 32 + tl.arange(0, 32)[None, :]

    src = tl.load(
        source
        + token * source_stride_t
        + group * source_stride_g
        + d * source_stride_d,
    ).to(tl.float32)
    max_abs = tl.max(tl.abs(src), axis=1)
    quant_scale = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    scale_exp = tl.minimum(tl.maximum(tl.ceil(tl.log2(quant_scale)), -127.0), 127.0)
    scale = tl.exp2(scale_exp)
    scale_u8 = (scale_exp + 127.0).to(tl.uint8)

    tl.store(
        values
        + token * values_stride_t
        + d * values_stride_d
        + group * values_stride_g,
        (src / scale[:, None]).to(tl.float8e4nv),
    )

    sf_cols = group_width // 32
    tl.store(
        scale_rows + group * tokens * sf_cols + token * sf_cols + chunk,
        scale_u8,
    )

    row32 = token % 32
    row4 = (token // 32) % 4
    tile_m = token // 128
    k4 = chunk % 4
    tile_k = chunk // 4
    tl.store(
        scale_mma
        + row32 * scale_mma_s0
        + row4 * scale_mma_s1
        + tile_m * scale_mma_s2
        + k4 * scale_mma_s3
        + tile_k * scale_mma_s4
        + group * scale_mma_s5,
        scale_u8,
    )


@triton.jit
def _quantize_attention_inv_rope_to_tdg_kernel(
    o,
    positions,
    cos_sin_cache,
    values,
    scale_rows,
    scale_mma,
    clear_output,
    tokens,
    groups: tl.constexpr,
    heads_per_group: tl.constexpr,
    group_width: tl.constexpr,
    o_stride_t,
    o_stride_h,
    o_stride_d,
    cos_sin_stride_pos,
    values_stride_t,
    values_stride_d,
    values_stride_g,
    scale_mma_s0,
    scale_mma_s1,
    scale_mma_s2,
    scale_mma_s3,
    scale_mma_s4,
    scale_mma_s5,
    clear_output_stride_t,
    clear_output_stride_n,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    CHUNKS_PER_PROGRAM: tl.constexpr,
    FAST_B16_SCALE: tl.constexpr,
    CLEAR_OUTPUT: tl.constexpr,
    CLEAR_HIDDEN: tl.constexpr,
    CLEAR_BLOCK_SIZE: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    group = tl.program_id(1)
    chunk_block = tl.program_id(2)
    chunk = chunk_block * CHUNKS_PER_PROGRAM + tl.arange(0, CHUNKS_PER_PROGRAM)
    d = chunk[:, None] * 32 + tl.arange(0, 32)[None, :]

    if CLEAR_OUTPUT:
        chunk_blocks = group_width // 32 // CHUNKS_PER_PROGRAM
        clear_block = group * chunk_blocks + chunk_block
        clear_n = clear_block * CLEAR_BLOCK_SIZE + tl.arange(0, CLEAR_BLOCK_SIZE)
        tl.store(
            clear_output
            + token * clear_output_stride_t
            + clear_n * clear_output_stride_n,
            0.0,
            mask=clear_n < CLEAR_HIDDEN,
        )

    head_in_group = d // HEAD_DIM
    head_d = d - head_in_group * HEAD_DIM
    head = group * heads_per_group + head_in_group

    src = tl.load(o + token * o_stride_t + head * o_stride_h + head_d * o_stride_d).to(
        tl.float32
    )

    is_rope = head_d >= NOPE_DIM
    rope_local = head_d - NOPE_DIM
    partner_d = NOPE_DIM + (rope_local ^ 1)
    partner = tl.load(
        o + token * o_stride_t + head * o_stride_h + partner_d * o_stride_d,
        mask=is_rope,
        other=0.0,
    ).to(tl.float32)

    pos = tl.load(positions + token)
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cache_base = cos_sin_cache + pos * cos_sin_stride_pos
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE_DIM + cs_idx, mask=is_rope, other=0.0)
    x_add = src * cos_v + partner * sin_v
    x_sub = src * cos_v - partner * sin_v
    rotated = tl.where((rope_local & 1) == 0, x_add, x_sub)
    src = tl.where(is_rope, rotated, src)

    max_abs = tl.max(tl.abs(src), axis=1)
    if FAST_B16_SCALE:
        max_bits = max_abs.to(tl.uint32, bitcast=True)
        biased_exp = (max_bits >> 23) & 0xFF
        mantissa = max_bits & 0x7FFFFF
        scale_exp = biased_exp.to(tl.int32) - 135 + (mantissa > 0x600000).to(tl.int32)
        scale_exp = tl.minimum(tl.maximum(scale_exp, -127), 127)
        scale_exp = tl.where(max_abs > 0.0, scale_exp, 0)
        inv_scale_bits = tl.where(
            scale_exp == 127,
            max_bits * 0 + 0x00400000,
            (127 - scale_exp).to(tl.uint32) << 23,
        )
        inv_scale = inv_scale_bits.to(tl.float32, bitcast=True)
        scaled_src = src * inv_scale[:, None]
    else:
        quant_scale = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
        scale_exp = tl.minimum(tl.maximum(tl.ceil(tl.log2(quant_scale)), -127.0), 127.0)
        scale = tl.exp2(scale_exp)
        scaled_src = src / scale[:, None]
    scale_u8 = (scale_exp + 127.0).to(tl.uint8)

    tl.store(
        values
        + token * values_stride_t
        + d * values_stride_d
        + group * values_stride_g,
        scaled_src.to(tl.float8e4nv),
    )

    sf_cols = group_width // 32
    tl.store(
        scale_rows + group * tokens * sf_cols + token * sf_cols + chunk,
        scale_u8,
    )

    row32 = token % 32
    row4 = (token // 32) % 4
    tile_m = token // 128
    k4 = chunk % 4
    tile_k = chunk // 4
    tl.store(
        scale_mma
        + row32 * scale_mma_s0
        + row4 * scale_mma_s1
        + tile_m * scale_mma_s2
        + k4 * scale_mma_s3
        + tile_k * scale_mma_s4
        + group * scale_mma_s5,
        scale_u8,
    )


@triton.jit
def _quantize_group_major_trg_to_tk_kernel(
    source,
    values,
    scale_rows,
    scale_mma,
    tokens,
    rank: tl.constexpr,
    groups: tl.constexpr,
    source_stride_t,
    source_stride_r,
    source_stride_g,
    scale_mma_s0,
    scale_mma_s1,
    scale_mma_s2,
    scale_mma_s3,
    scale_mma_s4,
    scale_mma_s5,
    CHUNKS_PER_PROGRAM: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    chunk_block = tl.program_id(1)
    chunk = chunk_block * CHUNKS_PER_PROGRAM + tl.arange(0, CHUNKS_PER_PROGRAM)
    cols = chunk[:, None] * 32 + tl.arange(0, 32)[None, :]
    g = cols // rank
    r = cols - g * rank

    src = tl.load(
        source + token * source_stride_t + r * source_stride_r + g * source_stride_g,
    ).to(tl.float32)
    max_abs = tl.max(tl.abs(src), axis=1)
    safe = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    scale_exp = tl.minimum(tl.maximum(tl.ceil(tl.log2(safe)), -127.0), 127.0)
    scale = tl.exp2(scale_exp)
    scale_u8 = (scale_exp + 127.0).to(tl.uint8)

    width = rank * groups
    tl.store(values + token * width + cols, (src / scale[:, None]).to(tl.float8e4nv))

    sf_cols = width // 32
    tl.store(scale_rows + token * sf_cols + chunk, scale_u8)

    row32 = token % 32
    row4 = (token // 32) % 4
    tile_m = token // 128
    k4 = chunk % 4
    tile_k = chunk // 4
    tl.store(
        scale_mma
        + row32 * scale_mma_s0
        + row4 * scale_mma_s1
        + tile_m * scale_mma_s2
        + k4 * scale_mma_s3
        + tile_k * scale_mma_s4
        + scale_mma_s5 * 0,
        scale_u8,
    )


def empty_dense_gemm_mnl_view(
    m: int,
    n: int,
    l: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate an `[M,N,L]` view backed by dense-GEMM physical `[L,M,N]` storage."""

    if m <= 0 or n <= 0 or l <= 0:
        raise ValueError("m, n, and l must be positive")
    if l == 1:
        return torch.empty((m, n, 1), device=device, dtype=dtype)
    physical = torch.empty((l, m, n), device=device, dtype=dtype)
    return physical.as_strided((m, n, l), (n, 1, m * n))


def empty_mxfp8_rows_bases(
    m: int,
    k: int,
    *,
    num_groups: int,
    device: torch.device | str,
    initialize_scales: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the three CONTIGUOUS base buffers behind an MXFP8Rows.

    Returns ``(values_base, scale_rows_base_u8, scale_physical_base_u8)`` -- all
    contiguous. ``mxfp8_rows_from_bases`` rebuilds the (strided/permuted)
    dense-GEMM views from them. Functional custom ops return these bases instead
    of the views: an opaque op that *returns a view* surfaces a symbolic-shaped
    view as a downstream graph input, which trips torch's AOT
    `merge_view_inputs` synthetic-base path under dynamic shapes. Returning bases
    keeps op outputs contiguous; the views are rebuilt in normal traced code.

    ``initialize_scales=False`` is for an immediately following quantization
    kernel that overwrites every logical row scale and every physical scale
    entry consumed by dense GEMM. Physical M-padding remains unspecified; dense
    GEMM must not observe it for logical output rows.
    """
    if m <= 0 or k <= 0 or num_groups <= 0:
        raise ValueError("m, k, and num_groups must be positive")
    _check_mxfp8_k(k)
    if num_groups == 1:
        values_base = torch.empty((m, k), device=device, dtype=torch.float8_e4m3fn)
    else:
        # Physical [L, M, K]; mxfp8_rows_from_bases views it as the [M,K,L] mnl layout.
        values_base = torch.empty(
            (num_groups, m, k), device=device, dtype=torch.float8_e4m3fn
        )
    scale_shape = (num_groups, m, k // MXFP8_SCALE_VEC_SIZE)
    if initialize_scales:
        scale_rows_base = torch.full(
            scale_shape,
            127,
            dtype=torch.uint8,
            device=device,
        )
    else:
        scale_rows_base = torch.empty(scale_shape, dtype=torch.uint8, device=device)
    m_tiles = math.ceil(m / MXFP8_SCALE_ROW_TILE)
    k_tiles = math.ceil((k // MXFP8_SCALE_VEC_SIZE) / MXFP8_SCALE_K_TILE)
    physical_shape = (num_groups, m_tiles, k_tiles, 32, 4, 4)
    if initialize_scales:
        scale_physical_base = torch.full(
            physical_shape,
            127,
            dtype=torch.uint8,
            device=device,
        )
    else:
        scale_physical_base = torch.empty(
            physical_shape, dtype=torch.uint8, device=device
        )
    return values_base, scale_rows_base, scale_physical_base


def mxfp8_rows_from_bases(
    values_base: torch.Tensor,
    scale_rows_base: torch.Tensor,
    scale_physical_base: torch.Tensor,
    m: int,
    k: int,
    *,
    num_groups: int,
) -> MXFP8Rows:
    """Rebuild the dense-GEMM MXFP8Rows views over contiguous bases (pure views)."""
    if num_groups == 1:
        values = values_base
    else:
        values = values_base.as_strided((m, k, num_groups), (k, 1, m * k))
    scale_rows = scale_rows_base.view(torch.float8_e8m0fnu)
    scale_mma = scale_physical_base.view(torch.float8_e8m0fnu).permute(3, 4, 1, 5, 2, 0)
    return MXFP8Rows(values=values, scale_rows=scale_rows, scale_mma=scale_mma)


def empty_mxfp8_rows_for_dense_gemm(
    m: int,
    k: int,
    *,
    num_groups: int = 1,
    device: torch.device | str,
) -> MXFP8Rows:
    """Allocate MXFP8 row storage in the layout consumed by `dense_gemm`."""

    values_base, scale_rows_base, scale_physical_base = empty_mxfp8_rows_bases(
        m, k, num_groups=num_groups, device=device
    )
    return mxfp8_rows_from_bases(
        values_base, scale_rows_base, scale_physical_base, m, k, num_groups=num_groups
    )


def _check_dense_gemm_mnl_view(name: str, tensor: torch.Tensor) -> None:
    _check_gpu_tensor(name, tensor)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [M,N,L], got {tuple(tensor.shape)}")
    m, n, l = tensor.shape
    expected_stride = (n, 1, m * n) if l > 1 else tensor.stride()
    if l > 1 and tensor.stride() != expected_stride:
        raise ValueError(
            f"{name} must be backed by dense-GEMM physical [L,M,N] storage: "
            f"expected stride {expected_stride}, got {tensor.stride()}"
        )


def _scale_u8_from_max_abs(max_abs: torch.Tensor) -> torch.Tensor:
    safe = torch.where(
        max_abs > 0,
        max_abs.to(torch.float32) / FP8_E4M3_MAX,
        torch.ones_like(max_abs, dtype=torch.float32),
    )
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    return (exponent + 127).to(torch.uint8)


def _scale_to_e8m0_u8(scale: torch.Tensor) -> torch.Tensor:
    _check_gpu_tensor("scale", scale)
    if scale.dtype == torch.float8_e8m0fnu:
        return scale.view(torch.uint8)
    if scale.dtype == torch.uint8:
        return scale
    if not scale.is_floating_point():
        raise ValueError(
            f"scale must be e8m0, uint8, or floating-point, got {scale.dtype}"
        )
    safe = torch.where(
        scale > 0,
        scale.to(torch.float32),
        torch.ones_like(scale, dtype=torch.float32),
    )
    exponent = torch.round(torch.log2(safe)).clamp(-127, 127)
    return (exponent + 127).to(torch.uint8)


def _expand_block_scales_to_mxfp8_rows(
    scale: torch.Tensor,
    *,
    m: int,
    k: int,
    num_groups: int,
) -> torch.Tensor:
    _check_gpu_tensor("scale", scale)
    if m <= 0 or k <= 0 or num_groups <= 0:
        raise ValueError("m, k, and num_groups must be positive")
    _check_mxfp8_k(k)

    m_tiles = math.ceil(m / MXFP8_SCALE_ROW_TILE)
    k_tiles = math.ceil((k // MXFP8_SCALE_VEC_SIZE) / MXFP8_SCALE_K_TILE)
    expected_2d = (num_groups * m_tiles, k_tiles)
    expected_3d = (num_groups, m_tiles, k_tiles)
    if scale.shape == expected_2d:
        block_u8 = _scale_to_e8m0_u8(scale).reshape(num_groups, m_tiles, k_tiles)
    elif scale.shape == expected_3d:
        block_u8 = _scale_to_e8m0_u8(scale).reshape(expected_3d)
    elif num_groups == 1 and scale.shape == (m_tiles, k_tiles):
        block_u8 = _scale_to_e8m0_u8(scale).reshape(1, m_tiles, k_tiles)
    else:
        raise ValueError(
            "block scale must have shape "
            f"{expected_2d}, {expected_3d}"
            + (f", or {(m_tiles, k_tiles)}" if num_groups == 1 else "")
            + f"; got {tuple(scale.shape)}"
        )

    scale_rows_u8 = (
        block_u8[:, :, None, :, None]
        .expand(num_groups, m_tiles, MXFP8_SCALE_ROW_TILE, k_tiles, MXFP8_SCALE_K_TILE)
        .reshape(
            num_groups,
            m_tiles * MXFP8_SCALE_ROW_TILE,
            k_tiles * MXFP8_SCALE_K_TILE,
        )[:, :m, : k // MXFP8_SCALE_VEC_SIZE]
        .contiguous()
    )
    return scale_rows_u8.view(torch.float8_e8m0fnu)


def pack_mxfp8_scales_for_dense_gemm(
    scale_rows: torch.Tensor,
    *,
    m: int,
    k: int,
    num_groups: int = 1,
) -> torch.Tensor:
    """Pack compact MXFP8 row/chunk scales into sm12x dense-GEMM MMA layout.

    `scale_rows` must be UE8M0 scales in either `[M, K/32]`,
    `[num_groups, M, K/32]`, or `[num_groups * M, K/32]` form. Missing padded
    rows/chunks are filled with UE8M0 1.0, so the kernel can safely read the
    fixed 128-row scale tile for small-M contracts.
    """

    _check_gpu_tensor("scale_rows", scale_rows)
    if scale_rows.dtype == torch.uint8:
        scale_rows = scale_rows.view(torch.float8_e8m0fnu)
    if scale_rows.dtype != torch.float8_e8m0fnu:
        raise ValueError(f"scale_rows must be uint8/e8m0, got {scale_rows.dtype}")
    if m <= 0 or k <= 0 or num_groups <= 0:
        raise ValueError("m, k, and num_groups must be positive")
    _check_mxfp8_k(k)

    sf_k = k // MXFP8_SCALE_VEC_SIZE
    if scale_rows.ndim == 2:
        if scale_rows.shape == (m, sf_k):
            grouped = scale_rows.reshape(1, m, sf_k)
            if num_groups != 1:
                raise ValueError(
                    "2D scale_rows with shape [M,K/32] requires num_groups=1"
                )
        elif scale_rows.shape == (num_groups * m, sf_k):
            grouped = scale_rows.reshape(num_groups, m, sf_k)
        else:
            raise ValueError(
                "scale_rows must have shape [M,K/32] or [num_groups*M,K/32], "
                f"got {tuple(scale_rows.shape)} for m={m}, k={k}, num_groups={num_groups}"
            )
    elif scale_rows.ndim == 3:
        if scale_rows.shape != (num_groups, m, sf_k):
            raise ValueError(
                f"scale_rows must have shape {(num_groups, m, sf_k)}, "
                f"got {tuple(scale_rows.shape)}"
            )
        grouped = scale_rows
    else:
        raise ValueError(
            "scale_rows must have shape [M,K/32], [num_groups*M,K/32], "
            f"or [num_groups,M,K/32], got {tuple(scale_rows.shape)}"
        )

    m_tiles = math.ceil(m / MXFP8_SCALE_ROW_TILE)
    k_tiles = math.ceil(sf_k / MXFP8_SCALE_K_TILE)
    padded_m = m_tiles * MXFP8_SCALE_ROW_TILE
    padded_sf_k = k_tiles * MXFP8_SCALE_K_TILE

    padded_u8 = torch.full(
        (num_groups, padded_m, padded_sf_k),
        127,
        dtype=torch.uint8,
        device=scale_rows.device,
    )
    padded = padded_u8.view(torch.float8_e8m0fnu)
    padded[:, :m, :sf_k] = grouped

    physical = (
        padded.view(num_groups, m_tiles, 4, 32, k_tiles, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
    )
    return physical.permute(3, 4, 1, 5, 2, 0)


def _scale_is_exact_ue8m0(scale: torch.Tensor) -> bool:
    """True if `scale` is already an exact UE8M0 scale (no re-quant needed).

    UE8M0-typed tensors (`float8_e8m0fnu`/`uint8`) qualify by construction. A
    floating-point tensor qualifies only if every entry is a positive power of
    two (zero mantissa) — e.g. a previously-UE8M0 scale widened to fp32. Genuine
    checkpoint scales (`weight_scale_inv = block_amax / 448`) are not powers of
    two and fail this test, so they get re-quantized. Runs once at weight load.
    """

    if scale.dtype in (torch.uint8, torch.float8_e8m0fnu):
        return True
    if not scale.is_floating_point():
        return False
    bits = scale.detach().to(torch.float32).view(torch.int32)
    mantissa = bits & 0x7FFFFF
    positive = scale.detach().to(torch.float32) > 0
    return bool(torch.all(positive & (mantissa == 0)).item())


def _requantize_block_fp8_to_ue8m0(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    m: int,
    k: int,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize an FP8 checkpoint weight onto exact UE8M0 block scales.

    DeepSeek-style checkpoints carry `(w_fp8, s_fp32)` where `s_fp32 =
    block_amax / 448` is an *arbitrary* fp32 128x128 block scale. The sm12x MMA
    can only apply power-of-two (UE8M0) scales. Rounding `s_fp32` to the nearest
    power of two while keeping the original `w_fp8` values leaves the values
    matched to the *unrounded* scale, baking in a per-block scale error up to
    sqrt(2) (~10% weight RMS error, measured).

    Parity with DeepGEMM's `per_block_cast_to_fp8(use_ue8m0=True)`: reconstruct
    the recovered weight `w_fp8 * s_fp32` and re-cast it onto a fresh ceil-UE8M0
    block scale, so the FP8 values are re-derived *consistently* with the
    power-of-two scale (~3.7% RMS error, near the irreducible e4m3 floor). This
    runs once at weight load, not in the serving path.

    Returns `(weight_e4m3 [num_groups*m, k] or [m, k], block_scale_e8m0_u8
    [num_groups, m_tiles, k_tiles])`.
    """

    gk = MXFP8_SCALE_ROW_TILE  # 128
    m_tiles = math.ceil(m / gk)
    k_tiles = math.ceil((k // MXFP8_SCALE_VEC_SIZE) / MXFP8_SCALE_K_TILE)

    # Normalize the FP8 weight to per-group [num_groups, m, k].
    if num_groups == 1:
        w_g = weight.reshape(1, m, k)
    elif tuple(weight.shape) == (num_groups * m, k):
        w_g = weight.reshape(num_groups, m, k)
    elif tuple(weight.shape) == (m, k, num_groups):
        w_g = weight.permute(2, 0, 1)
    else:
        raise ValueError(
            f"weight must have shape {(num_groups * m, k)} or {(m, k, num_groups)} "
            f"or {(m, k)}, got {tuple(weight.shape)}"
        )

    # Normalize the arbitrary fp32 block scale to [num_groups, m_tiles, k_tiles].
    s = scale.to(torch.float32)
    if tuple(s.shape) == (num_groups * m_tiles, k_tiles):
        s_g = s.reshape(num_groups, m_tiles, k_tiles)
    elif tuple(s.shape) == (num_groups, m_tiles, k_tiles):
        s_g = s
    elif num_groups == 1 and tuple(s.shape) == (m_tiles, k_tiles):
        s_g = s.reshape(1, m_tiles, k_tiles)
    else:
        raise ValueError(
            f"block scale must have shape {(num_groups * m_tiles, k_tiles)}, "
            f"{(num_groups, m_tiles, k_tiles)}"
            + (f", or {(m_tiles, k_tiles)}" if num_groups == 1 else "")
            + f"; got {tuple(scale.shape)}"
        )

    # Reconstruct the recovered weight: w_fp8 * (block scale broadcast to elems).
    s_elem = s_g.repeat_interleave(gk, dim=1).repeat_interleave(gk, dim=2)[:, :m, :k]
    w_rec = w_g.to(torch.float32) * s_elem

    # Pad to whole 128x128 blocks, take per-block amax, ceil-UE8M0 cast.
    m_pad, k_pad = m_tiles * gk, k_tiles * gk
    w_pad = w_rec.new_zeros(num_groups, m_pad, k_pad)
    w_pad[:, :m, :k] = w_rec
    blocks = w_pad.view(num_groups, m_tiles, gk, k_tiles, gk)
    amax = blocks.abs().amax(dim=(2, 4), keepdim=True)  # [g, m_tiles, 1, k_tiles, 1]
    safe = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax))
    exponent = torch.clamp(torch.ceil(torch.log2(safe)), -127.0, 127.0)
    sf = torch.exp2(exponent)
    new_vals = (
        (blocks / sf)
        .to(torch.float8_e4m3fn)
        .view(num_groups, m_pad, k_pad)[:, :m, :k]
        .contiguous()
    )
    new_scale_u8 = (exponent.view(num_groups, m_tiles, k_tiles) + 127).to(torch.uint8)

    if num_groups == 1:
        new_weight = new_vals.reshape(m, k)
    else:
        new_weight = new_vals.reshape(num_groups * m, k)
    return new_weight, new_scale_u8


def pack_fp8_block_scaled_weight_mxfp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    m: int,
    k: int,
    num_groups: int = 1,
) -> MXFP8Rows:
    """Pack checkpoint FP8 block-scaled weights for native MXFP8 dense GEMM.

    `weight` is FP8 E4M3 and `scale` is the DSV4-style 128x128 block scale.

    When `scale` is an *arbitrary* fp32 block scale (the DeepSeek
    `weight_scale_inv` checkpoint format), the weight is **re-quantized** onto an
    exact ceil-UE8M0 block scale so the FP8 values stay consistent with the
    power-of-two scale the MMA applies (parity with DeepGEMM
    `per_block_cast_to_fp8`). When `scale` is already UE8M0 (e8m0/uint8), the
    FP8 values are kept verbatim and the scale is expanded as-is.
    """

    _check_gpu_tensor("weight", weight)
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if m <= 0 or k <= 0 or num_groups <= 0:
        raise ValueError("m, k, and num_groups must be positive")
    _check_mxfp8_k(k)

    # Arbitrary fp32 checkpoint scales (e.g. DeepSeek `weight_scale_inv`) are not
    # powers of two; re-quantize onto exact UE8M0 instead of rounding the scale
    # and leaving the FP8 values stale (which costs ~2.7x more error). Scales
    # that are already exact UE8M0 (e8m0/uint8, or a float tensor holding only
    # powers of two) keep their FP8 values verbatim.
    if not _scale_is_exact_ue8m0(scale):
        _check_gpu_tensor("scale", scale)
        weight, scale = _requantize_block_fp8_to_ue8m0(
            weight, scale, m=m, k=k, num_groups=num_groups
        )

    if num_groups == 1:
        if weight.shape != (m, k):
            raise ValueError(
                f"weight must have shape {(m, k)}, got {tuple(weight.shape)}"
            )
        values = weight.contiguous()
    else:
        if weight.shape == (num_groups * m, k):
            values = weight.contiguous().view(num_groups, m, k).permute(1, 2, 0)
        elif weight.shape == (m, k, num_groups):
            values = weight
            _check_dense_gemm_mnl_view("weight", values)
        else:
            raise ValueError(
                f"weight must have shape {(num_groups * m, k)} or {(m, k, num_groups)}, "
                f"got {tuple(weight.shape)}"
            )

    scale_rows = _expand_block_scales_to_mxfp8_rows(
        scale,
        m=m,
        k=k,
        num_groups=num_groups,
    )
    scale_mma = pack_mxfp8_scales_for_dense_gemm(
        scale_rows,
        m=m,
        k=k,
        num_groups=num_groups,
    )
    values_tiled = None
    tile_n = 0
    if (num_groups, m, k) == (4, 1024, 4096):
        tile_n = 64
    elif (num_groups, m, k) == (1, 4096, 4096):
        tile_n = 128
    if tile_n:
        # Physical [L, Ntile, Ktile, Ninner, Kinner], specialized for the
        # production decode WO Ntile/BK128 GEMMs. Keep `values` in its logical
        # layout for all other schedules and callers.
        values_lnk = values.permute(2, 0, 1) if num_groups > 1 else values.unsqueeze(0)
        values_tiled = (
            values_lnk.reshape(num_groups, m // tile_n, tile_n, k // 128, 128)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
    return MXFP8Rows(
        values=values,
        scale_rows=scale_rows,
        scale_mma=scale_mma,
        values_tiled=values_tiled,
    )


def pack_wo_projection_fp8_block_scaled_weights_mxfp8(
    wo_a_weight: torch.Tensor,
    wo_a_scale: torch.Tensor,
    wo_b_weight: torch.Tensor,
    wo_b_scale: torch.Tensor,
    *,
    groups: int,
    group_width: int,
    rank: int,
    hidden: int,
) -> WOProjectionMXFP8Weights:
    """Pack local DSV4 WO-A/WO-B checkpoint FP8 weights for the sm12x WO path."""

    wo_a = pack_fp8_block_scaled_weight_mxfp8(
        wo_a_weight,
        wo_a_scale,
        m=rank,
        k=group_width,
        num_groups=groups,
    )
    wo_b = pack_fp8_block_scaled_weight_mxfp8(
        wo_b_weight,
        wo_b_scale,
        m=hidden,
        k=groups * rank,
        num_groups=1,
    )
    return WOProjectionMXFP8Weights(
        wo_a=wo_a,
        wo_b=wo_b,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
        sfb_k_replicated=True,
    )


def _check_mxfp8_rows_storage(
    out: MXFP8Rows,
    *,
    m: int,
    k: int,
    num_groups: int,
) -> None:
    _check_gpu_tensor("out.values", out.values)
    _check_gpu_tensor("out.scale_rows", out.scale_rows)
    _check_gpu_tensor("out.scale_mma", out.scale_mma)
    if out.values.dtype != torch.float8_e4m3fn:
        raise ValueError(f"out.values must be float8_e4m3fn, got {out.values.dtype}")
    if out.scale_rows.dtype != torch.float8_e8m0fnu:
        raise ValueError(
            f"out.scale_rows must be float8_e8m0fnu, got {out.scale_rows.dtype}"
        )
    if out.scale_mma.dtype != torch.float8_e8m0fnu:
        raise ValueError(
            f"out.scale_mma must be float8_e8m0fnu, got {out.scale_mma.dtype}"
        )
    if out.values_tiled is not None:
        _check_gpu_tensor("out.values_tiled", out.values_tiled)
        if out.values_tiled.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"out.values_tiled must be float8_e4m3fn, got {out.values_tiled.dtype}"
            )
        expected_tiled_shapes = {
            (4, 1024, 4096): (4, 16, 32, 64, 128),
            (1, 4096, 4096): (1, 32, 32, 128, 128),
        }
        expected_tiled_shape = expected_tiled_shapes.get((num_groups, m, k))
        if (
            expected_tiled_shape is None
            or tuple(out.values_tiled.shape) != expected_tiled_shape
        ):
            raise ValueError(
                "out.values_tiled is only supported for production WO-A/WO-B; "
                f"got dimensions {(num_groups, m, k)} and shape "
                f"{tuple(out.values_tiled.shape)}, expected {expected_tiled_shape}"
            )
        if out.values_tiled.device != out.values.device:
            raise ValueError(
                "out.values_tiled and out.values must be on the same device"
            )
        if not out.values_tiled.is_contiguous():
            raise ValueError("out.values_tiled must be contiguous")
    if num_groups == 1:
        if out.values.shape != (m, k):
            raise ValueError(
                f"out.values must have shape {(m, k)}, got {tuple(out.values.shape)}"
            )
    else:
        if out.values.shape != (m, k, num_groups):
            raise ValueError(
                f"out.values must have shape {(m, k, num_groups)}, got {tuple(out.values.shape)}"
            )
        _check_dense_gemm_mnl_view("out.values", out.values)
    sf_k = k // MXFP8_SCALE_VEC_SIZE
    if out.scale_rows.shape != (num_groups, m, sf_k):
        raise ValueError(
            f"out.scale_rows must have shape {(num_groups, m, sf_k)}, "
            f"got {tuple(out.scale_rows.shape)}"
        )
    expected_scale_mma = (
        32,
        4,
        math.ceil(m / MXFP8_SCALE_ROW_TILE),
        4,
        math.ceil(sf_k / MXFP8_SCALE_K_TILE),
        num_groups,
    )
    if out.scale_mma.shape != expected_scale_mma:
        raise ValueError(
            f"out.scale_mma must have shape {expected_scale_mma}, got {tuple(out.scale_mma.shape)}"
        )


def quantize_wo_a_input_mxfp8(
    source_tgd: torch.Tensor,
    *,
    out: MXFP8Rows | None = None,
) -> MXFP8Rows:
    """Quantize grouped WO-A input into per-32-column MXFP8 rows."""

    _check_gpu_tensor("source_tgd", source_tgd)
    if source_tgd.ndim != 3:
        raise ValueError(
            f"source_tgd must have shape [tokens, groups, group_width], got {tuple(source_tgd.shape)}"
        )
    tokens, groups, group_width = source_tgd.shape
    _check_mxfp8_k(group_width)
    if out is None:
        out = empty_mxfp8_rows_for_dense_gemm(
            tokens,
            group_width,
            num_groups=groups,
            device=source_tgd.device,
        )
    else:
        _check_mxfp8_rows_storage(out, m=tokens, k=group_width, num_groups=groups)
    # Measured on RTX PRO 6000 at M=8192, gw=4096, groups=4: this Triton
    # kernel (232.8us) beats the CuTe grouped port (250.1us) -- unlike the old
    # per-32-group dense quantizer, it is already tiled and bandwidth-bound.
    # quantize_wo_grouped_rows_cute stays available but is not routed.
    values_stride_g = out.values.stride(2) if out.values.ndim == 3 else 0
    chunks_per_program = _wo_quant_chunks_per_program(group_width)
    _quantize_grouped_tgd_to_tdg_kernel[
        (
            tokens,
            groups,
            group_width // MXFP8_SCALE_VEC_SIZE // chunks_per_program,
        )
    ](
        source_tgd,
        out.values,
        out.scale_rows.view(torch.uint8),
        out.scale_mma.view(torch.uint8),
        tokens,
        groups,
        group_width,
        source_tgd.stride(0),
        source_tgd.stride(1),
        source_tgd.stride(2),
        out.values.stride(0),
        out.values.stride(1),
        values_stride_g,
        out.scale_mma.stride(0),
        out.scale_mma.stride(1),
        out.scale_mma.stride(2),
        out.scale_mma.stride(3),
        out.scale_mma.stride(4),
        out.scale_mma.stride(5),
        CHUNKS_PER_PROGRAM=chunks_per_program,
        num_warps=4,
    )
    return out


def _run_wo_a_quant_kernel(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    out_values: torch.Tensor,
    out_scale_rows: torch.Tensor,
    out_scale_mma: torch.Tensor,
    tokens: int,
    groups: int,
    heads_per_group: int,
    group_width: int,
    head_dim: int,
    nope_dim: int,
    rope_dim: int,
    clear_output: torch.Tensor | None = None,
) -> None:
    # Measured on RTX PRO 6000 at the DS4-Flash TP2 prefill shape (M=8192):
    # this Triton kernel (263us) beats the branchless CuTe inverse-RoPE port
    # (279us); both are near the read+write roofline. The CuTe variant
    # (quantize_wo_grouped_rows_cute with positions) stays available but is
    # not routed.
    out_scale_mma_u8 = out_scale_mma.view(torch.uint8)
    values_stride_g = out_values.stride(2) if out_values.ndim == 3 else 0
    chunks_per_program = _wo_quant_chunks_per_program(group_width)
    fast_b16_scale = (
        tokens == 16
        and groups == 4
        and heads_per_group == 8
        and group_width == 4096
        and head_dim == 512
        and nope_dim == 448
        and rope_dim == 64
        and chunks_per_program == 16
    )
    clear_programs = groups * group_width // MXFP8_SCALE_VEC_SIZE // chunks_per_program
    if clear_output is None:
        clear_output_arg = out_values
        clear_output_stride_t = 0
        clear_output_stride_n = 0
        clear_hidden = 0
        clear_block_size = 1
    else:
        if (
            clear_output.ndim != 3
            or int(clear_output.shape[0]) != tokens
            or int(clear_output.shape[2]) != 1
            or clear_output.dtype != torch.bfloat16
        ):
            raise ValueError(
                "quantizer clear output must be BF16 [tokens, hidden, 1], got "
                f"{clear_output.dtype} {tuple(clear_output.shape)}"
            )
        clear_output_arg = clear_output
        clear_output_stride_t = clear_output.stride(0)
        clear_output_stride_n = clear_output.stride(1)
        clear_hidden = int(clear_output.shape[1])
        clear_block_size = triton.next_power_of_2(
            math.ceil(clear_hidden / clear_programs)
        )
    _quantize_attention_inv_rope_to_tdg_kernel[
        (
            tokens,
            groups,
            group_width // MXFP8_SCALE_VEC_SIZE // chunks_per_program,
        )
    ](
        o,
        positions,
        cos_sin_cache,
        out_values,
        out_scale_rows.view(torch.uint8),
        out_scale_mma_u8,
        clear_output_arg,
        tokens,
        groups,
        heads_per_group,
        group_width,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        cos_sin_cache.stride(0),
        out_values.stride(0),
        out_values.stride(1),
        values_stride_g,
        out_scale_mma.stride(0),
        out_scale_mma.stride(1),
        out_scale_mma.stride(2),
        out_scale_mma.stride(3),
        out_scale_mma.stride(4),
        out_scale_mma.stride(5),
        clear_output_stride_t,
        clear_output_stride_n,
        HEAD_DIM=head_dim,
        NOPE_DIM=nope_dim,
        HALF_ROPE_DIM=rope_dim // 2,
        CHUNKS_PER_PROGRAM=chunks_per_program,
        FAST_B16_SCALE=fast_b16_scale,
        CLEAR_OUTPUT=clear_output is not None,
        CLEAR_HIDDEN=clear_hidden,
        CLEAR_BLOCK_SIZE=clear_block_size,
        num_warps=2 if tokens == 16 else 4,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::quantize_wo_a_input_inv_rope_mxfp8_alloc",
    mutates_args=(),
)
def _quantize_wo_a_input_inv_rope_mxfp8_alloc_op(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    tokens: int,
    groups: int,
    heads_per_group: int,
    group_width: int,
    head_dim: int,
    nope_dim: int,
    rope_dim: int,
    initialize_scales: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Functional (allocate + return) quantizer. The MXFP8 output views (notably
    # the permuted scale_mma) are materialized and written INSIDE this opaque op,
    # so torch.compile never sees a mutation of an as_strided/permute view (which
    # it bans). Used whenever the caller does not pass a pre-owned `out`.
    values_base, scale_rows_base, scale_physical_base = empty_mxfp8_rows_bases(
        tokens,
        group_width,
        num_groups=groups,
        device=o.device,
        initialize_scales=initialize_scales,
    )
    out = mxfp8_rows_from_bases(
        values_base,
        scale_rows_base,
        scale_physical_base,
        tokens,
        group_width,
        num_groups=groups,
    )
    _run_wo_a_quant_kernel(
        o,
        positions,
        cos_sin_cache,
        out.values,
        out.scale_rows,
        out.scale_mma,
        tokens,
        groups,
        heads_per_group,
        group_width,
        head_dim,
        nope_dim,
        rope_dim,
    )
    # Return the CONTIGUOUS bases (not the views) -- see empty_mxfp8_rows_bases.
    return values_base, scale_rows_base, scale_physical_base


@_quantize_wo_a_input_inv_rope_mxfp8_alloc_op.register_fake
def _quantize_wo_a_input_inv_rope_mxfp8_alloc_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    tokens: int,
    groups: int,
    heads_per_group: int,
    group_width: int,
    head_dim: int,
    nope_dim: int,
    rope_dim: int,
    initialize_scales: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return empty_mxfp8_rows_bases(
        tokens,
        group_width,
        num_groups=groups,
        device=o.device,
        initialize_scales=initialize_scales,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::quantize_wo_a_input_inv_rope_mxfp8_launch",
    mutates_args=("out_values", "out_scale_rows", "out_scale_mma"),
)
def _quantize_wo_a_input_inv_rope_mxfp8_launch_op(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    out_values: torch.Tensor,
    out_scale_rows: torch.Tensor,
    out_scale_mma: torch.Tensor,
    tokens: int,
    groups: int,
    heads_per_group: int,
    group_width: int,
    head_dim: int,
    nope_dim: int,
    rope_dim: int,
) -> None:
    # Mutating variant for the eager `out`-provided path (caller owns base
    # buffers). NOT compile-safe when out_* are strided views; the torch.compile
    # path uses the _alloc op above instead.
    _run_wo_a_quant_kernel(
        o,
        positions,
        cos_sin_cache,
        out_values,
        out_scale_rows,
        out_scale_mma,
        tokens,
        groups,
        heads_per_group,
        group_width,
        head_dim,
        nope_dim,
        rope_dim,
    )


@_quantize_wo_a_input_inv_rope_mxfp8_launch_op.register_fake
def _quantize_wo_a_input_inv_rope_mxfp8_launch_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    out_values: torch.Tensor,
    out_scale_rows: torch.Tensor,
    out_scale_mma: torch.Tensor,
    tokens: int,
    groups: int,
    heads_per_group: int,
    group_width: int,
    head_dim: int,
    nope_dim: int,
    rope_dim: int,
) -> None:
    return None


def quantize_wo_a_input_inv_rope_mxfp8(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    out: MXFP8Rows | None = None,
    _initialize_scales: bool = True,
) -> MXFP8Rows:
    """Inverse-RoPE attention output and quantize into per-32-column MXFP8 rows."""

    _check_gpu_tensor("o", o)
    _check_gpu_tensor("positions", positions)
    _check_gpu_tensor("cos_sin_cache", cos_sin_cache)
    if o.ndim != 3:
        raise ValueError(
            f"o must have shape [tokens, heads, head_dim], got {tuple(o.shape)}"
        )
    tokens, heads, head_dim = o.shape
    if head_dim != nope_dim + rope_dim:
        raise ValueError(
            f"o head_dim must equal nope_dim + rope_dim ({nope_dim + rope_dim}), "
            f"got {head_dim}"
        )
    if heads != groups * heads_per_group:
        raise ValueError(
            f"o heads must equal groups * heads_per_group ({groups * heads_per_group}), "
            f"got {heads}"
        )
    if positions.shape != (tokens,):
        raise ValueError(
            f"positions must have shape {(tokens,)}, got {tuple(positions.shape)}"
        )
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[-1] != rope_dim:
        raise ValueError(
            "cos_sin_cache must have shape [max_position, rope_dim], "
            f"got {tuple(cos_sin_cache.shape)}"
        )
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")

    group_width = heads_per_group * head_dim
    _check_mxfp8_k(group_width)
    if out is None:
        values_base, scale_rows_base, scale_physical_base = (
            torch.ops.flashinfer_sm12x.quantize_wo_a_input_inv_rope_mxfp8_alloc(
                o,
                positions,
                cos_sin_cache,
                tokens,
                groups,
                heads_per_group,
                group_width,
                head_dim,
                nope_dim,
                rope_dim,
                _initialize_scales,
            )
        )
        return mxfp8_rows_from_bases(
            values_base,
            scale_rows_base,
            scale_physical_base,
            tokens,
            group_width,
            num_groups=groups,
        )

    _check_mxfp8_rows_storage(out, m=tokens, k=group_width, num_groups=groups)
    torch.ops.flashinfer_sm12x.quantize_wo_a_input_inv_rope_mxfp8_launch(
        o,
        positions,
        cos_sin_cache,
        out.values,
        out.scale_rows,
        out.scale_mma,
        tokens,
        groups,
        heads_per_group,
        group_width,
        head_dim,
        nope_dim,
        rope_dim,
    )
    return out


def _run_wo_b_quant_kernel(
    source_trg: torch.Tensor,
    out_values: torch.Tensor,
    out_scale_rows: torch.Tensor,
    out_scale_mma: torch.Tensor,
    tokens: int,
    rank: int,
    groups: int,
    width: int,
) -> None:
    # Measured on RTX PRO 6000 at M=8192, rank=1024, groups=4: the CuTe
    # group-major port wins isolated (37us vs 41us) but loses inside the live
    # WO chain (62.5us vs 46.6us right after the WO-A GEMM), so this stays on
    # Triton. quantize_wo_group_major_rows_cute remains available unrouted.
    chunks_per_program = _wo_quant_chunks_per_program(width)
    _quantize_group_major_trg_to_tk_kernel[
        (
            tokens,
            width // MXFP8_SCALE_VEC_SIZE // chunks_per_program,
        )
    ](
        source_trg,
        out_values,
        out_scale_rows.view(torch.uint8),
        out_scale_mma.view(torch.uint8),
        tokens,
        rank,
        groups,
        source_trg.stride(0),
        source_trg.stride(1),
        source_trg.stride(2),
        out_scale_mma.stride(0),
        out_scale_mma.stride(1),
        out_scale_mma.stride(2),
        out_scale_mma.stride(3),
        out_scale_mma.stride(4),
        out_scale_mma.stride(5),
        CHUNKS_PER_PROGRAM=chunks_per_program,
        num_warps=4,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::quantize_wo_b_input_mxfp8_alloc",
    mutates_args=(),
)
def _quantize_wo_b_input_mxfp8_alloc_op(
    source_trg: torch.Tensor,
    tokens: int,
    rank: int,
    groups: int,
    width: int,
    initialize_scales: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Functional (allocate + return) quantizer; materializes + writes the MXFP8
    # output views INSIDE the op, returns the contiguous bases (see WO-A _alloc).
    values_base, scale_rows_base, scale_physical_base = empty_mxfp8_rows_bases(
        tokens,
        width,
        num_groups=1,
        device=source_trg.device,
        initialize_scales=initialize_scales,
    )
    out = mxfp8_rows_from_bases(
        values_base, scale_rows_base, scale_physical_base, tokens, width, num_groups=1
    )
    _run_wo_b_quant_kernel(
        source_trg,
        out.values,
        out.scale_rows,
        out.scale_mma,
        tokens,
        rank,
        groups,
        width,
    )
    return values_base, scale_rows_base, scale_physical_base


@_quantize_wo_b_input_mxfp8_alloc_op.register_fake
def _quantize_wo_b_input_mxfp8_alloc_fake(
    source_trg: torch.Tensor,
    tokens: int,
    rank: int,
    groups: int,
    width: int,
    initialize_scales: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return empty_mxfp8_rows_bases(
        tokens,
        width,
        num_groups=1,
        device=source_trg.device,
        initialize_scales=initialize_scales,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::quantize_wo_b_input_mxfp8_launch",
    mutates_args=("out_values", "out_scale_rows", "out_scale_mma"),
)
def _quantize_wo_b_input_mxfp8_launch_op(
    source_trg: torch.Tensor,
    out_values: torch.Tensor,
    out_scale_rows: torch.Tensor,
    out_scale_mma: torch.Tensor,
    tokens: int,
    rank: int,
    groups: int,
    width: int,
) -> None:
    # Mutating variant for the eager `out`-provided path (see WO-A _launch).
    _run_wo_b_quant_kernel(
        source_trg,
        out_values,
        out_scale_rows,
        out_scale_mma,
        tokens,
        rank,
        groups,
        width,
    )


@_quantize_wo_b_input_mxfp8_launch_op.register_fake
def _quantize_wo_b_input_mxfp8_launch_fake(
    source_trg: torch.Tensor,
    out_values: torch.Tensor,
    out_scale_rows: torch.Tensor,
    out_scale_mma: torch.Tensor,
    tokens: int,
    rank: int,
    groups: int,
    width: int,
) -> None:
    return None


def quantize_wo_b_input_mxfp8(
    source_trg: torch.Tensor,
    *,
    out: MXFP8Rows | None = None,
    _initialize_scales: bool = True,
) -> MXFP8Rows:
    """Quantize WO-A intermediate `[tokens, rank, groups]` into group-major `[tokens, groups * rank]`."""

    _check_gpu_tensor("source_trg", source_trg)
    if source_trg.ndim != 3:
        raise ValueError(
            f"source_trg must have shape [tokens, rank, groups], got {tuple(source_trg.shape)}"
        )
    tokens, rank, groups = source_trg.shape
    width = rank * groups
    _check_mxfp8_k(width)
    if out is None:
        values_base, scale_rows_base, scale_physical_base = (
            torch.ops.flashinfer_sm12x.quantize_wo_b_input_mxfp8_alloc(
                source_trg,
                tokens,
                rank,
                groups,
                width,
                _initialize_scales,
            )
        )
        return mxfp8_rows_from_bases(
            values_base,
            scale_rows_base,
            scale_physical_base,
            tokens,
            width,
            num_groups=1,
        )

    _check_mxfp8_rows_storage(out, m=tokens, k=width, num_groups=1)
    torch.ops.flashinfer_sm12x.quantize_wo_b_input_mxfp8_launch(
        source_trg,
        out.values,
        out.scale_rows,
        out.scale_mma,
        tokens,
        rank,
        groups,
        width,
    )
    return out


def quantize_mxfp8_rows_torch(source: torch.Tensor) -> MXFP8Rows:
    """Quantize `[M,K]` or `[M,K,L]` rows to MXFP8 on GPU.

    This is a graph-capturable GPU Torch prep/reference helper. It is not the
    final production activation-quant kernel for WO-A/WO-B.
    """

    grouped, m, k, num_groups = _as_grouped_mkl(source)
    _check_mxfp8_k(k)

    chunks = k // MXFP8_SCALE_VEC_SIZE
    grouped_f32 = grouped.to(torch.float32)
    blocked = grouped_f32.reshape(num_groups, m, chunks, MXFP8_SCALE_VEC_SIZE)
    max_abs = blocked.abs().amax(dim=-1)
    scale_u8 = _scale_u8_from_max_abs(max_abs)
    scale_rows = scale_u8.view(torch.float8_e8m0fnu)
    scale = scale_rows.to(torch.float32)
    quant_grouped = (
        (blocked / scale[..., None])
        .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .reshape(num_groups, m, k)
        .contiguous()
    )
    if source.ndim == 2:
        values = quant_grouped.reshape(m, k).contiguous()
    else:
        values = quant_grouped.as_strided((m, k, num_groups), (k, 1, m * k))
    scale_mma = pack_mxfp8_scales_for_dense_gemm(
        scale_rows,
        m=m,
        k=k,
        num_groups=num_groups,
    )
    return MXFP8Rows(values=values, scale_rows=scale_rows, scale_mma=scale_mma)


def dequantize_mxfp8_rows_torch(
    values: torch.Tensor, scale_rows: torch.Tensor
) -> torch.Tensor:
    """Dequantize compact row-wise MXFP8 values on GPU for tests/oracles."""

    grouped, m, k, num_groups = _as_grouped_mkl(values)
    _check_mxfp8_k(k)
    sf_k = k // MXFP8_SCALE_VEC_SIZE
    if scale_rows.dtype == torch.uint8:
        scale_rows = scale_rows.view(torch.float8_e8m0fnu)
    if scale_rows.shape != (num_groups, m, sf_k):
        raise ValueError(
            f"scale_rows must have shape {(num_groups, m, sf_k)}, got {tuple(scale_rows.shape)}"
        )
    scale = scale_rows.to(torch.float32)
    out_grouped = (
        grouped.to(torch.float32).reshape(num_groups, m, sf_k, MXFP8_SCALE_VEC_SIZE)
        * scale[..., None]
    ).reshape(num_groups, m, k)
    out = out_grouped.permute(1, 2, 0).contiguous()
    if values.ndim == 2:
        out = out[:, :, 0].contiguous()
    return out


def quantize_wo_projection_weights_mxfp8_torch(
    wo_a_grd: torch.Tensor,
    wo_b_hgr: torch.Tensor,
) -> WOProjectionMXFP8Weights:
    """Quantize BF16 WO weights into the native MXFP8 two-GEMM layouts.

    This is a GPU Torch setup helper for tests and benchmarks. Serving should
    prepare the same layouts at model load from checkpoint FP8 weights/scales.
    """

    _check_gpu_tensor("wo_a_grd", wo_a_grd)
    _check_gpu_tensor("wo_b_hgr", wo_b_hgr)
    if wo_a_grd.ndim != 3:
        raise ValueError(
            f"wo_a_grd must have shape [groups, rank, group_width], got {tuple(wo_a_grd.shape)}"
        )
    if wo_b_hgr.ndim != 2:
        raise ValueError(
            f"wo_b_hgr must have shape [hidden, groups * rank], got {tuple(wo_b_hgr.shape)}"
        )
    groups, rank, group_width = wo_a_grd.shape
    hidden, wo_b_width = wo_b_hgr.shape
    if wo_b_width != groups * rank:
        raise ValueError(
            f"wo_b_hgr width must equal groups * rank, got {wo_b_width} vs {groups * rank}"
        )
    _check_mxfp8_k(group_width)
    _check_mxfp8_k(groups * rank)

    wo_a_source = wo_a_grd.permute(1, 2, 0).contiguous()
    # MXFP8Rows intentionally stores singleton-L values as compact [M,K].
    # Preserve that convention for the benchmark/test setup helper so TP8
    # (o_groups=8 -> one local output group) matches checkpoint packing.
    if groups == 1:
        wo_a_source = wo_a_source[:, :, 0]
    wo_a = quantize_mxfp8_rows_torch(wo_a_source)
    wo_b = quantize_mxfp8_rows_torch(wo_b_hgr)
    return WOProjectionMXFP8Weights(
        wo_a=wo_a,
        wo_b=wo_b,
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
    )


def _check_wo_projection_weights(weights: WOProjectionMXFP8Weights) -> None:
    if not isinstance(weights, WOProjectionMXFP8Weights):
        raise TypeError("weights must be a WOProjectionMXFP8Weights instance")
    if (
        weights.groups <= 0
        or weights.group_width <= 0
        or weights.rank <= 0
        or weights.hidden <= 0
    ):
        raise ValueError("WO projection dimensions must be positive")
    _check_mxfp8_rows_storage(
        weights.wo_a,
        m=weights.rank,
        k=weights.group_width,
        num_groups=weights.groups,
    )
    _check_mxfp8_rows_storage(
        weights.wo_b,
        m=weights.hidden,
        k=weights.rank * weights.groups,
        num_groups=1,
    )


def _check_wo_projection_views(
    *,
    x_q: MXFP8Rows,
    tmp: torch.Tensor,
    tmp_q: MXFP8Rows,
    output: torch.Tensor,
    tokens: int,
    weights: WOProjectionMXFP8Weights,
) -> None:
    _check_mxfp8_rows_storage(
        x_q,
        m=tokens,
        k=weights.group_width,
        num_groups=weights.groups,
    )
    _check_dense_gemm_mnl_view("tmp", tmp)
    if tmp.shape != (tokens, weights.rank, weights.groups):
        raise ValueError(
            "tmp must have shape "
            f"{(tokens, weights.rank, weights.groups)}, got {tuple(tmp.shape)}"
        )
    if tmp.dtype != torch.bfloat16:
        raise ValueError(f"tmp must be bfloat16, got {tmp.dtype}")
    _check_mxfp8_rows_storage(
        tmp_q,
        m=tokens,
        k=weights.rank * weights.groups,
        num_groups=1,
    )
    _check_dense_gemm_mnl_view("output", output)
    if output.shape != (tokens, weights.hidden, 1):
        raise ValueError(
            "output must have shape "
            f"{(tokens, weights.hidden, 1)}, got {tuple(output.shape)}"
        )
    if output.dtype != torch.bfloat16:
        raise ValueError(f"output must be bfloat16, got {output.dtype}")


def _validate_wo_projection_inputs(
    source_tgd: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
) -> int:
    _check_gpu_tensor("source_tgd", source_tgd)
    _check_wo_projection_weights(weights)
    if source_tgd.ndim != 3:
        raise ValueError(
            f"source_tgd must have shape [tokens, groups, group_width], got {tuple(source_tgd.shape)}"
        )
    tokens, groups, group_width = source_tgd.shape
    if (groups, group_width) != (weights.groups, weights.group_width):
        raise ValueError(
            "source_tgd shape does not match weights: "
            f"source={(groups, group_width)}, weights={(weights.groups, weights.group_width)}"
        )
    return int(tokens)


def _validate_wo_projection_inv_rope_inputs(
    *,
    o: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> int:
    _check_gpu_tensor("o", o)
    _check_wo_projection_weights(weights)
    heads_per_group = int(heads_per_group)
    nope_dim = int(nope_dim)
    rope_dim = int(rope_dim)
    if heads_per_group <= 0 or nope_dim <= 0 or rope_dim <= 0:
        raise ValueError("heads_per_group, nope_dim, and rope_dim must be positive")
    if o.ndim != 3:
        raise ValueError(
            f"o must have shape [tokens, heads, head_dim], got {tuple(o.shape)}"
        )
    tokens, heads, head_dim = o.shape
    if head_dim != nope_dim + rope_dim:
        raise ValueError(
            f"o head_dim must equal nope_dim + rope_dim ({nope_dim + rope_dim}), "
            f"got {head_dim}"
        )
    if heads != weights.groups * heads_per_group:
        raise ValueError(
            f"o heads must equal weights.groups * heads_per_group "
            f"({weights.groups * heads_per_group}), got {heads}"
        )
    if weights.group_width != heads_per_group * head_dim:
        raise ValueError(
            "weights.group_width does not match heads_per_group * head_dim: "
            f"{weights.group_width} != {heads_per_group * head_dim}"
        )
    return int(tokens)


def build_wo_projection_binding(
    *,
    x_q: MXFP8Rows,
    tmp: torch.Tensor,
    tmp_q: MXFP8Rows,
    output: torch.Tensor,
    source_tgd: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
    return_3d: bool = False,
    expected_m: int | None = None,
) -> WOProjectionBinding:
    return _build_wo_projection_binding_from_views(
        x_q=x_q,
        tmp=tmp,
        tmp_q=tmp_q,
        output=output,
        source_tgd=source_tgd,
        weights=weights,
        return_3d=return_3d,
        expected_m=expected_m,
    )


def _build_wo_projection_binding_from_views(
    *,
    x_q: MXFP8Rows,
    tmp: torch.Tensor,
    tmp_q: MXFP8Rows,
    output: torch.Tensor,
    source_tgd: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
    return_3d: bool = False,
    expected_m: int | None = None,
) -> WOProjectionBinding:
    tokens = _validate_wo_projection_inputs(source_tgd, weights)
    _check_wo_projection_views(
        x_q=x_q,
        tmp=tmp,
        tmp_q=tmp_q,
        output=output,
        tokens=tokens,
        weights=weights,
    )
    return WOProjectionBinding(
        source_tgd=source_tgd,
        weights=weights,
        x_q=x_q,
        tmp=tmp,
        tmp_q=tmp_q,
        output=output,
        return_3d=bool(return_3d),
        expected_m=expected_m,
    )


def build_wo_projection_inv_rope_binding(
    *,
    x_q: MXFP8Rows,
    tmp: torch.Tensor,
    tmp_q: MXFP8Rows,
    output: torch.Tensor,
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    return_3d: bool = False,
    expected_m: int | None = None,
) -> WOProjectionInvRopeBinding:
    tokens = _validate_wo_projection_inv_rope_inputs(
        o=o,
        weights=weights,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    _check_gpu_tensor("positions", positions)
    _check_gpu_tensor("cos_sin_cache", cos_sin_cache)
    _check_wo_projection_views(
        x_q=x_q,
        tmp=tmp,
        tmp_q=tmp_q,
        output=output,
        tokens=tokens,
        weights=weights,
    )
    return _build_wo_projection_inv_rope_binding_from_views(
        x_q=x_q,
        tmp=tmp,
        tmp_q=tmp_q,
        output=output,
        o=o,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=weights,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        return_3d=return_3d,
        expected_m=expected_m,
    )


def _build_wo_projection_inv_rope_binding_from_views(
    *,
    x_q: MXFP8Rows,
    tmp: torch.Tensor,
    tmp_q: MXFP8Rows,
    output: torch.Tensor,
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: WOProjectionMXFP8Weights,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    return_3d: bool = False,
    expected_m: int | None = None,
) -> WOProjectionInvRopeBinding:
    tokens = _validate_wo_projection_inv_rope_inputs(
        o=o,
        weights=weights,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    _check_gpu_tensor("positions", positions)
    _check_gpu_tensor("cos_sin_cache", cos_sin_cache)
    _check_wo_projection_views(
        x_q=x_q,
        tmp=tmp,
        tmp_q=tmp_q,
        output=output,
        tokens=tokens,
        weights=weights,
    )
    return WOProjectionInvRopeBinding(
        o=o,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        weights=weights,
        x_q=x_q,
        tmp=tmp,
        tmp_q=tmp_q,
        output=output,
        heads_per_group=int(heads_per_group),
        nope_dim=int(nope_dim),
        rope_dim=int(rope_dim),
        return_3d=bool(return_3d),
        expected_m=expected_m,
    )


def plan_wo_projection_scratch(
    caps: WOProjectionScratchCaps,
) -> WOProjectionScratchPlan:
    layout = _layout_wo_projection(
        offset_bytes=0,
        tokens=caps.max_tokens,
        groups=caps.groups,
        group_width=caps.group_width,
        rank=caps.rank,
        hidden=caps.hidden,
    )
    return WOProjectionScratchPlan(
        caps=caps,
        layout=layout,
        _scratch_specs=(
            scratch_buffer_spec(
                "wo_projection.scratch",
                nbytes=layout.nbytes,
                device=caps.device,
            ),
        ),
    )


def wo_a_dense_gemm_mxfp8(
    x_tdg: MXFP8Rows,
    wo_a_rdg: MXFP8Rows,
    *,
    out: torch.Tensor | None = None,
    quantized_out: MXFP8Rows | None = None,
    alpha: torch.Tensor | None = None,
    expected_m: int | None = None,
    sfb_k_replicated: bool = False,
    stream: object = None,
) -> torch.Tensor:
    """Run WO-A as grouped MXFP8 dense GEMM.

    Inputs are `x.values [tokens, group_width, groups]` and
    `wo_a.values [rank, group_width, groups]`; singleton groups use compact
    `[tokens, group_width]` / `[rank, group_width]` storage. Output is
    `[tokens, rank, groups]`.
    """

    if x_tdg.values.ndim not in (2, 3) or wo_a_rdg.values.ndim != x_tdg.values.ndim:
        raise ValueError(
            "WO-A operands must both have shape [M,K] for one group or "
            "[M,K,groups] for multiple groups"
        )
    if x_tdg.values.shape[1:] != wo_a_rdg.values.shape[1:]:
        raise ValueError(
            f"WO-A K/groups mismatch: x={tuple(x_tdg.values.shape)} "
            f"wo_a={tuple(wo_a_rdg.values.shape)}"
        )
    if out is not None:
        _check_dense_gemm_mnl_view("out", out)
    tokens = int(x_tdg.values.shape[0])
    rank = int(wo_a_rdg.values.shape[0])
    groups = int(wo_a_rdg.values.shape[2]) if wo_a_rdg.values.ndim == 3 else 1
    if quantized_out is not None:
        _check_mxfp8_rows_storage(
            quantized_out,
            m=tokens,
            k=rank * groups,
            num_groups=1,
        )
    # When out is None, dense_gemm allocates + returns functionally (no caller
    # view mutated in the compile graph). The returned [M,N,L] is read downstream
    # via strides, so its physical layout does not matter.
    mma_tiler_mn = None
    if expected_m is not None and wo_a_rdg.values.shape[0] <= 1536:
        if 1 <= expected_m <= 8:
            mma_tiler_mn = (16, 64)
        # B16 is the existing generic decode policy; the B9-15 extension is
        # measured for the DSV4 TP2 WO-A shape only.
        elif expected_m == 16 or (
            9 <= expected_m <= 15
            and rank == 1024
            and int(x_tdg.values.shape[1]) == 512
            and groups == 4
        ):
            mma_tiler_mn = (32, 64)
    x_values = x_tdg.values
    wo_a_values = wo_a_rdg.values
    if x_values.ndim == 2:
        x_values = x_values.unsqueeze(-1)
        wo_a_values = wo_a_values.unsqueeze(-1)
    return dense_gemm(
        (x_values, x_tdg.scale_mma),
        (wo_a_values, wo_a_rdg.scale_mma),
        rhs_values_tiled=(
            wo_a_rdg.values_tiled
            if mma_tiler_mn in ((16, 64), (32, 64), (64, 64))
            else None
        ),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        out=out,
        alpha=alpha,
        mma_tiler_mn=mma_tiler_mn,
        expected_m=expected_m,
        sfb_k_replicated=sfb_k_replicated,
        _quantized_c=(
            (
                quantized_out.values,
                quantized_out.scale_rows.reshape(tokens, -1),
                quantized_out.scale_mma,
            )
            if quantized_out is not None
            else None
        ),
        stream=stream,
    )


def wo_b_dense_gemm_mxfp8(
    tmp_tgr_group_major: MXFP8Rows,
    wo_b_hgr: MXFP8Rows,
    *,
    out: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    expected_m: int | None = None,
    sfb_k_replicated: bool = False,
    stream: object = None,
) -> torch.Tensor:
    """Run group-major WO-B as MXFP8 dense GEMM.

    Inputs are `tmp.values [tokens, rank * groups]` and
    `wo_b.values [hidden, rank * groups]`; output is `[tokens, hidden, 1]`.
    """

    if tmp_tgr_group_major.values.ndim != 2 or wo_b_hgr.values.ndim != 2:
        raise ValueError("WO-B operands must have shape [M,K] and [N,K]")
    if tmp_tgr_group_major.values.shape[1] != wo_b_hgr.values.shape[1]:
        raise ValueError(
            f"WO-B K mismatch: tmp={tuple(tmp_tgr_group_major.values.shape)} "
            f"wo_b={tuple(wo_b_hgr.values.shape)}"
        )
    if out is not None:
        _check_dense_gemm_mnl_view("out", out)
    # Preserve the generic B16 policy while limiting the B9-15 packed tile to
    # the measured DSV4 TP2 WO-B shape.
    use_decode_tile = expected_m == 16 or (
        expected_m is not None
        and 9 <= expected_m <= 15
        and tuple(wo_b_hgr.values.shape) == (4096, 4096)
    )
    mma_tiler_mn = (32, 64) if use_decode_tile else None
    # out=None -> dense_gemm allocates + returns functionally (no caller view
    # mutated in the compile graph).
    return dense_gemm(
        (
            tmp_tgr_group_major.values.reshape(
                tmp_tgr_group_major.values.shape[0],
                tmp_tgr_group_major.values.shape[1],
                1,
            ),
            tmp_tgr_group_major.scale_mma,
        ),
        (
            wo_b_hgr.values.reshape(
                wo_b_hgr.values.shape[0],
                wo_b_hgr.values.shape[1],
                1,
            ),
            wo_b_hgr.scale_mma,
        ),
        rhs_values_tiled=(wo_b_hgr.values_tiled if use_decode_tile else None),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=MXFP8_SCALE_VEC_SIZE,
        out=out,
        alpha=alpha,
        mma_tiler_mn=mma_tiler_mn,
        expected_m=expected_m,
        sfb_k_replicated=sfb_k_replicated,
        stream=stream,
    )


def _wo_a_fused_mma_tiler(expected_m: int | None, rank: int) -> tuple[int, int] | None:
    if expected_m is not None and 1 <= expected_m <= 8 and rank <= 1536:
        return (16, 64)
    return None


def wo_a_dense_gemm_fused_quant_mxfp8(
    source_tgd: torch.Tensor,
    wo_a_rdg: MXFP8Rows,
    *,
    out: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    head_dim: int = 0,
    nope_dim: int = 0,
    rope_dim: int = 0,
    expected_m: int | None = None,
    sfb_k_replicated: bool = False,
    stream: object = None,
) -> torch.Tensor:
    """Run WO-A at small M with (optionally inverse-RoPE) activation
    quantization fused into the grouped GEMM's DMA warp.

    `source_tgd` is the BF16 `[tokens, groups, group_width]` view of the
    attention output (rows may be strided; each `[groups, group_width]` row
    must be contiguous). Output is `[tokens, rank, groups]`.
    """

    if source_tgd.ndim != 3:
        raise ValueError(
            f"source_tgd must have shape [tokens, groups, group_width], got {tuple(source_tgd.shape)}"
        )
    groups = int(source_tgd.shape[1])
    wo_a_values = wo_a_rdg.values
    if wo_a_values.ndim == 2:
        wo_a_values = wo_a_values.unsqueeze(-1)
    rank = int(wo_a_values.shape[0])
    if out is not None:
        _check_dense_gemm_mnl_view("out", out)
    return dense_gemm_fused_quant_a_grouped(
        source_tgd,
        wo_a_values,
        wo_a_rdg.scale_mma,
        groups=groups,
        out=out,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        head_dim=head_dim,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        expected_m=expected_m,
        sfb_k_replicated=sfb_k_replicated,
        mma_tiler_mn=_wo_a_fused_mma_tiler(expected_m, rank),
        stream=stream,
    )


def wo_b_dense_gemm_fused_quant_mxfp8(
    tmp_trg: torch.Tensor,
    wo_b_hgr: MXFP8Rows,
    *,
    out: torch.Tensor | None = None,
    expected_m: int | None = None,
    sfb_k_replicated: bool = False,
    _atomic_output_precleared: bool = False,
    stream: object = None,
) -> torch.Tensor:
    """Run WO-B at small M with group-major quantization fused into the GEMM.

    `tmp_trg` is the BF16 WO-A output `[tokens, rank, groups]` dense-GEMM mnl
    view; the GEMM's DMA warp quantizes the group-major `[tokens, groups*rank]`
    rows in-CTA, so no standalone quant kernel or MXFP8 scratch is needed.
    """

    if tmp_trg.ndim != 3:
        raise ValueError(
            f"tmp_trg must have shape [tokens, rank, groups], got {tuple(tmp_trg.shape)}"
        )
    tokens, rank, groups = map(int, tmp_trg.shape)
    width = rank * groups
    if wo_b_hgr.values.ndim != 2 or int(wo_b_hgr.values.shape[1]) != width:
        raise ValueError(
            f"WO-B weight must have shape [N,{width}], got {tuple(wo_b_hgr.values.shape)}"
        )
    hidden = int(wo_b_hgr.values.shape[0])
    if out is not None:
        _check_dense_gemm_mnl_view("out", out)
    if groups == 1:
        if tmp_trg.stride(0) != rank or tmp_trg.stride(1) != 1:
            raise ValueError(
                f"tmp_trg rows must be contiguous [tokens, rank], got strides {tmp_trg.stride()}"
            )
        source: torch.Tensor = tmp_trg.as_strided((tokens, rank), (rank, 1))
        inner_span = 0
    else:
        source = tmp_trg
        inner_span = rank
    return dense_gemm_fused_quant_a(
        source,
        wo_b_hgr.values.reshape(hidden, width, 1),
        wo_b_hgr.scale_mma,
        out=out,
        expected_m=expected_m,
        sfb_k_replicated=sfb_k_replicated,
        rhs_values_tiled=(
            wo_b_hgr.values_tiled
            if expected_m is not None and 1 <= expected_m <= 8
            else None
        ),
        a_inner_span=inner_span,
        _atomic_output_precleared=_atomic_output_precleared,
        stream=stream,
    )


def wo_projection_mxfp8(
    source_tgd: torch.Tensor | None = None,
    weights: WOProjectionMXFP8Weights | None = None,
    *,
    return_3d: bool = False,
    binding: WOProjectionBinding | None = None,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run the native MXFP8 WO-A/WO-B projection.

    `source_tgd` is `[tokens, groups, group_width]`. The default return value
    is the SGLang-friendly `[tokens, hidden]` view over the binding output.
    """

    if binding is not None:
        extras = [
            name
            for name, value in (
                ("source_tgd", source_tgd),
                ("weights", weights),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "WO projection binding owns source_tgd, weights, and scratch tensors; "
                f"do not also pass {', '.join(extras)}"
            )
        if return_3d:
            raise ValueError(
                "WO projection binding owns return_3d; do not also pass return_3d=True"
            )
        source_tgd = binding.source_tgd
        weights = binding.weights
        x_q = binding.x_q
        tmp = binding.tmp
        tmp_q = binding.tmp_q
        output = binding.output
        return_3d = binding.return_3d
        expected_m = binding.expected_m
    else:
        x_q = None
        tmp = None
        tmp_q = None
        output = None
    if source_tgd is None or weights is None:
        raise TypeError(
            "wo_projection_mxfp8 requires source_tgd and weights or binding"
        )
    tokens = _validate_wo_projection_inputs(source_tgd, weights)
    # WO auto-defaults the regime hint to the (capture-fixed) token count so the
    # wo_b up-projection (N=hidden>1536) picks the decode tile (32x128) at small
    # M and the prefill tile (64x128) at large M, with no caller change.
    # Freeze-safe: the binding is built per-forward / per-capture, so `tokens`
    # is fixed per compiled kernel. Pass expected_m explicitly to override.
    if expected_m is None:
        expected_m = int(tokens)
    if x_q is None or tmp is None or tmp_q is None or output is None:
        raise TypeError("wo_projection_mxfp8 requires binding for caller-owned scratch")

    alpha_one = _cached_alpha_one(source_tgd.device)
    # WO-A stays on the standalone quantizer + GEMM; fusing quant into the
    # WO-A GEMM measured ~2x slower at M=1 (small N, deep K, per-n-tile
    # requantization) -- see wo_a_dense_gemm_fused_quant_mxfp8.
    quantize_wo_a_input_mxfp8(source_tgd, out=x_q)
    wo_a_dense_gemm_mxfp8(
        x_q,
        weights.wo_a,
        out=tmp,
        alpha=alpha_one,
        expected_m=expected_m,
        sfb_k_replicated=weights.sfb_k_replicated,
        stream=stream,
    )
    if tokens <= 8 and tmp.dtype == torch.bfloat16:
        # Decode band: group-major WO-B quantization runs inside the GEMM's
        # DMA warp; the bound tmp_q scratch is intentionally unused here.
        wo_b_dense_gemm_fused_quant_mxfp8(
            tmp,
            weights.wo_b,
            out=output,
            expected_m=expected_m,
            sfb_k_replicated=weights.sfb_k_replicated,
            stream=stream,
        )
    else:
        quantize_wo_b_input_mxfp8(tmp, out=tmp_q)
        wo_b_dense_gemm_mxfp8(
            tmp_q,
            weights.wo_b,
            out=output,
            alpha=alpha_one,
            expected_m=expected_m,
            sfb_k_replicated=weights.sfb_k_replicated,
            stream=stream,
        )
    if return_3d:
        return output
    return output[:, :, 0]


@torch.library.custom_op(
    "flashinfer_sm12x::wo_projection_inv_rope_mxfp8_fused",
    mutates_args=(),
)
def _wo_projection_inv_rope_mxfp8_fused_op(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a_values: torch.Tensor,
    wo_a_values_tiled: torch.Tensor | None,
    wo_a_scale_rows: torch.Tensor,
    wo_a_scale_mma: torch.Tensor,
    wo_b_values: torch.Tensor,
    wo_b_values_tiled: torch.Tensor | None,
    wo_b_scale_rows: torch.Tensor,
    wo_b_scale_mma: torch.Tensor,
    groups: int,
    group_width: int,
    rank: int,
    hidden: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    expected_m: int,
    sfb_k_replicated: bool,
    stream_int: int | None,
) -> torch.Tensor:
    # Fully opaque fused inv-rope WO: the entire quantize -> wo_a gemm -> quantize
    # -> wo_b gemm chain runs INSIDE this one op, so every token-shaped activation
    # MXFP8 view (x_q, tmp_q) stays internal and never becomes a symbolic-shaped
    # view graph value (which trips torch AOT merge_view_inputs under dynamic
    # shapes). Weight views passed in are static-shaped. Returns the contiguous
    # [tokens, hidden, 1] base.
    weights = WOProjectionMXFP8Weights(
        wo_a=MXFP8Rows(
            values=wo_a_values,
            scale_rows=wo_a_scale_rows,
            scale_mma=wo_a_scale_mma,
            values_tiled=wo_a_values_tiled,
        ),
        wo_b=MXFP8Rows(
            values=wo_b_values,
            scale_rows=wo_b_scale_rows,
            scale_mma=wo_b_scale_mma,
            values_tiled=wo_b_values_tiled,
        ),
        groups=groups,
        group_width=group_width,
        rank=rank,
        hidden=hidden,
        sfb_k_replicated=sfb_k_replicated,
    )
    alpha_one = _cached_alpha_one(o.device)
    tokens = int(o.shape[0])
    # Tiny-M WO-B uses atomic split-K. Clear its output inside the initial
    # inverse-RoPE quantizer so the decode graph does not need a fill launch.
    # WO-A stays on the standalone quantizer + GEMM: fusing the quant into the
    # WO-A GEMM's DMA warp measured ~2x slower at M=1 (N=rank is small, K is
    # deep, and every n-tile CTA re-quantizes the row without a source
    # prefetch pipeline) -- see wo_a_dense_gemm_fused_quant_mxfp8.
    # Both quantizers overwrite every logical scale before either GEMM reads
    # it. Leave physical M-padding unspecified to avoid four graph-replayed
    # uint8 unity-fill launches; poisoned-padding tests pin that it cannot
    # affect a logical output row. Standalone quantizers retain initialized
    # padding.
    atomic_output_precleared = tokens <= 8
    output = None
    if atomic_output_precleared:
        output = torch.empty((tokens, hidden, 1), dtype=torch.bfloat16, device=o.device)
        values_base, scale_rows_base, scale_physical_base = empty_mxfp8_rows_bases(
            tokens,
            group_width,
            num_groups=groups,
            device=o.device,
            initialize_scales=False,
        )
        x_q = mxfp8_rows_from_bases(
            values_base,
            scale_rows_base,
            scale_physical_base,
            tokens,
            group_width,
            num_groups=groups,
        )
        _run_wo_a_quant_kernel(
            o,
            positions,
            cos_sin_cache,
            x_q.values,
            x_q.scale_rows,
            x_q.scale_mma,
            tokens,
            groups,
            heads_per_group,
            group_width,
            nope_dim + rope_dim,
            nope_dim,
            rope_dim,
            clear_output=output,
        )
    else:
        x_q = quantize_wo_a_input_inv_rope_mxfp8(
            o,
            positions,
            cos_sin_cache,
            groups=groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            _initialize_scales=False,
        )
    sm_count = torch.cuda.get_device_properties(o.device).multi_processor_count
    # Keep upstream's Spark-only B16 policy. The B9-15 extension is likewise
    # retained only for the measured DSV4 TP2 inverse-RoPE contract.
    use_quantized_intermediate = _should_use_exact_b16_wo(
        tokens=tokens,
        sm_count=sm_count,
    ) or (
        9 <= tokens <= 15
        and sm_count <= _WO_SPARK_MAX_SMS
        and groups == 4
        and group_width == 512
        and rank == 1024
        and hidden == 4096
        and heads_per_group == 1
        and nope_dim == 448
        and rope_dim == 64
    )
    if use_quantized_intermediate:
        tmp_q_bases = empty_mxfp8_rows_bases(
            tokens,
            rank * groups,
            num_groups=1,
            device=o.device,
            initialize_scales=False,
        )
        tmp_q = mxfp8_rows_from_bases(
            *tmp_q_bases,
            tokens,
            rank * groups,
            num_groups=1,
        )
        wo_a_dense_gemm_mxfp8(
            x_q,
            weights.wo_a,
            quantized_out=tmp_q,
            expected_m=expected_m,
            sfb_k_replicated=weights.sfb_k_replicated,
            stream=stream_int,
        )
        return wo_b_dense_gemm_mxfp8(
            tmp_q,
            weights.wo_b,
            expected_m=expected_m,
            sfb_k_replicated=weights.sfb_k_replicated,
            stream=stream_int,
        )
    tmp = wo_a_dense_gemm_mxfp8(
        x_q,
        weights.wo_a,
        alpha=alpha_one,
        expected_m=expected_m,
        sfb_k_replicated=weights.sfb_k_replicated,
        stream=stream_int,
    )
    if tokens <= 8 and tmp.dtype == torch.bfloat16:
        # Decode band: quantize the group-major WO-B input inside the GEMM's
        # DMA warp instead of launching a standalone quant kernel.
        return wo_b_dense_gemm_fused_quant_mxfp8(
            tmp,
            weights.wo_b,
            out=output,
            expected_m=expected_m,
            sfb_k_replicated=weights.sfb_k_replicated,
            _atomic_output_precleared=atomic_output_precleared,
            stream=stream_int,
        )
    tmp_q = quantize_wo_b_input_mxfp8(tmp, _initialize_scales=False)
    return wo_b_dense_gemm_mxfp8(
        tmp_q,
        weights.wo_b,
        alpha=alpha_one,
        expected_m=expected_m,
        sfb_k_replicated=weights.sfb_k_replicated,
        stream=stream_int,
    )


@_wo_projection_inv_rope_mxfp8_fused_op.register_fake
def _wo_projection_inv_rope_mxfp8_fused_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a_values: torch.Tensor,
    wo_a_values_tiled: torch.Tensor | None,
    wo_a_scale_rows: torch.Tensor,
    wo_a_scale_mma: torch.Tensor,
    wo_b_values: torch.Tensor,
    wo_b_values_tiled: torch.Tensor | None,
    wo_b_scale_rows: torch.Tensor,
    wo_b_scale_mma: torch.Tensor,
    groups: int,
    group_width: int,
    rank: int,
    hidden: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    expected_m: int,
    sfb_k_replicated: bool,
    stream_int: int | None,
) -> torch.Tensor:
    del stream_int
    return torch.empty((o.shape[0], hidden, 1), dtype=o.dtype, device=o.device)


def wo_projection_inv_rope_mxfp8(
    o: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    weights: WOProjectionMXFP8Weights | None = None,
    *,
    heads_per_group: int | None = None,
    nope_dim: int = 448,
    rope_dim: int = 64,
    return_3d: bool = False,
    binding: WOProjectionInvRopeBinding | None = None,
    expected_m: int | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run WO projection from attention output without BF16 inverse-RoPE storage."""

    if binding is not None:
        extras = [
            name
            for name, value in (
                ("o", o),
                ("positions", positions),
                ("cos_sin_cache", cos_sin_cache),
                ("weights", weights),
                ("heads_per_group", heads_per_group),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "WO projection inverse-RoPE binding owns runtime tensors, weights, "
                f"scratch tensors, and heads_per_group; do not also pass {', '.join(extras)}"
            )
        extra_options = []
        if nope_dim != 448:
            extra_options.append("nope_dim")
        if rope_dim != 64:
            extra_options.append("rope_dim")
        if return_3d:
            extra_options.append("return_3d")
        if extra_options:
            raise ValueError(
                "WO projection inverse-RoPE binding owns options; "
                f"do not also pass {', '.join(extra_options)}"
            )
        o = binding.o
        positions = binding.positions
        cos_sin_cache = binding.cos_sin_cache
        weights = binding.weights
        heads_per_group = binding.heads_per_group
        nope_dim = binding.nope_dim
        rope_dim = binding.rope_dim
        return_3d = binding.return_3d
        expected_m = binding.expected_m
    if (
        o is None
        or positions is None
        or cos_sin_cache is None
        or weights is None
        or heads_per_group is None
    ):
        raise TypeError(
            "wo_projection_inv_rope_mxfp8 requires o, positions, cos_sin_cache, "
            "weights, and heads_per_group or binding"
        )
    tokens = _validate_wo_projection_inv_rope_inputs(
        o=o,
        weights=weights,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    _check_gpu_tensor("positions", positions)
    _check_gpu_tensor("cos_sin_cache", cos_sin_cache)
    # Auto-default the regime hint to the (capture-fixed) token count (see
    # wo_projection_mxfp8); decode -> 32x128, prefill -> 64x128, no caller change.
    if expected_m is None:
        expected_m = int(tokens)
    # One fully opaque fused op runs the whole quantize -> gemm -> quantize ->
    # gemm chain internally, so the token-shaped activation MXFP8 views never
    # become graph values (see _wo_projection_inv_rope_mxfp8_fused_op). Any
    # bound scratch is intentionally unused here.
    output = torch.ops.flashinfer_sm12x.wo_projection_inv_rope_mxfp8_fused(
        o,
        positions,
        cos_sin_cache,
        weights.wo_a.values,
        weights.wo_a.values_tiled,
        weights.wo_a.scale_rows,
        weights.wo_a.scale_mma,
        weights.wo_b.values,
        weights.wo_b.values_tiled,
        weights.wo_b.scale_rows,
        weights.wo_b.scale_mma,
        weights.groups,
        weights.group_width,
        weights.rank,
        weights.hidden,
        heads_per_group,
        nope_dim,
        rope_dim,
        expected_m,
        weights.sfb_k_replicated,
        cuda_stream_to_int(stream),
    )
    if return_3d:
        return output
    return output[:, :, 0]


__all__ = [
    "FP8_E4M3_MAX",
    "MXFP8Rows",
    "MXFP8_SCALE_VEC_SIZE",
    "WO_A_INPUT_QUANT_GROUP_SIZE",
    "WOProjectionBinding",
    "WOProjectionInvRopeBinding",
    "WOProjectionMXFP8Weights",
    "WOProjectionScratchCaps",
    "WOProjectionScratchPlan",
    "build_wo_projection_binding",
    "build_wo_projection_inv_rope_binding",
    "dequantize_mxfp8_rows_torch",
    "empty_dense_gemm_mnl_view",
    "empty_mxfp8_rows_for_dense_gemm",
    "pack_fp8_block_scaled_weight_mxfp8",
    "pack_mxfp8_scales_for_dense_gemm",
    "pack_wo_projection_fp8_block_scaled_weights_mxfp8",
    "plan_wo_projection_scratch",
    "quantize_mxfp8_rows_torch",
    "quantize_wo_a_input_inv_rope_mxfp8",
    "quantize_wo_a_input_mxfp8",
    "quantize_wo_b_input_mxfp8",
    "quantize_wo_projection_weights_mxfp8_torch",
    "wo_a_dense_gemm_fused_quant_mxfp8",
    "wo_a_dense_gemm_mxfp8",
    "wo_b_dense_gemm_fused_quant_mxfp8",
    "wo_b_dense_gemm_mxfp8",
    "wo_projection_inv_rope_mxfp8",
    "wo_projection_mxfp8",
]
