# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/gemm/wo_quant_cute.py @ 7183f674 (2026-07-02) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
from __future__ import annotations

import functools
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Uint8, Uint32

from flashinfer.experimental.sm12x._lib.compiler import (
    KernelCompileSpec,
    compile as sm12x_compile,
)
from flashinfer.experimental.sm12x._lib.intrinsics import (
    FLOAT8_E4M3_MAX,
    bfloat2_to_float2_scaled,
    cvt_f32x4_to_e4m3x4,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_nc_u32,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)
from flashinfer.experimental.sm12x._lib.runtime_control import (
    raise_if_kernel_resolution_frozen,
)
from flashinfer.experimental.sm12x._lib.utils import current_cuda_stream, make_ptr

_THREADS = 256
_GRID_CTAS_PER_SM = 4


class _WOQuantCuTeLaunch:
    """Tiled MXFP8 quantizer for the WO activation layouts.

    NOT ROUTED in serving: measured on RTX PRO 6000 at DS4-Flash TP2 prefill
    shapes (M=8192), the already-tiled Triton WO quantizers win or tie every
    layout in live context -- grouped 232.8us vs 250.1us, inverse-RoPE 263us
    vs 279us (branchless), group-major 46.6us vs 62.5us inside the WO chain
    (the CuTe group-major wins isolated, 37us vs 41us, but not E2E). Kept
    bit-exact and tested as the starting point for a future fused/producer
    prefetch rework.

    Mirrors the dense `_MXFP8RowsQuantLaunch` subgroup scheme (four 8-lane
    subgroups per warp, one 32-value scale block each, four adjacent values
    per lane) over two WO-specific addressings:

    - grouped (`mode="grouped"`): BF16 source rows `[m, groups * group_width]`
      (flat contiguous attention output), optionally applying inverse RoPE to
      the trailing `rope_dim` of every `head_dim` block; writes the grouped
      dense-GEMM operand (values physical `[g, m, group_width]`, per-group
      scale_rows / swizzled scale_mma).
    - group-major (`mode="group_major"`): BF16 source physical
      `[g, m, rank]` (the WO-A output) regathered as flat group-major rows
      `[m, groups * rank]`; writes the singleton-group dense-GEMM operand.
    """

    def __init__(
        self,
        mode: str,
        total_k: int,
        span: int,
        source_type: type[cutlass.Numeric],
        inv_rope: bool,
        head_dim: int,
        nope_dim: int,
        rope_dim: int,
        positions_type: type[cutlass.Numeric],
        cos_sin_type: type[cutlass.Numeric],
        threads: int,
    ) -> None:
        if mode not in ("grouped", "group_major"):
            raise ValueError(f"unsupported WO quant mode {mode!r}")
        self._mode = mode
        self._total_k = int(total_k)
        # Per-group width: group_width for grouped, rank for group-major.
        self._span = int(span)
        self._groups = self._total_k // self._span
        self._groups_k = self._total_k // 32
        self._span_groups_k = self._span // 32
        self._source_type = source_type
        self._inv_rope = bool(inv_rope)
        self._head_dim = int(head_dim)
        self._nope_dim = int(nope_dim)
        self._rope_dim = int(rope_dim)
        self._positions_type = positions_type
        self._cos_sin_type = cos_sin_type
        self._threads = int(threads)
        self._warps_per_cta = self._threads // 32

    @cute.jit
    def __call__(
        self,
        source_ptr: cute.Pointer,
        positions_ptr: cute.Pointer,
        cos_sin_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        scale_rows_ptr: cute.Pointer,
        scale_mma_ptr: cute.Pointer,
        m: Int32,
        cos_sin_len: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        source = cute.make_tensor(
            source_ptr,
            cute.make_layout((m * self._total_k,)),
        )
        positions = cute.make_tensor(positions_ptr, cute.make_layout((m,)))
        cos_sin = cute.make_tensor(cos_sin_ptr, cute.make_layout((cos_sin_len,)))
        values_u32 = cute.make_tensor(
            values_ptr,
            cute.make_layout((m * (self._total_k // 4),)),
        )
        scale_rows = cute.make_tensor(
            scale_rows_ptr,
            cute.make_layout((m * self._groups_k,)),
        )
        scale_mma = cute.make_tensor(
            scale_mma_ptr,
            cute.make_layout((max(512, ((self._groups_k + 3) // 4) * 512),)),
        )
        self.kernel(
            source, positions, cos_sin, values_u32, scale_rows, scale_mma, m
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        positions: cute.Tensor,
        cos_sin: cute.Tensor,
        values_u32: cute.Tensor,
        scale_rows: cute.Tensor,
        scale_mma: cute.Tensor,
        m: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        # Four 8-lane subgroups per warp each quantize one 32-value block;
        # every lane owns four adjacent values. total_k % 128 == 0, so all
        # four blocks of a task exist and the warp never diverges at the
        # butterfly reductions.
        warp = Int32(tidx) // Int32(32)
        lane = Int32(tidx) % Int32(32)
        subgroup = lane // Int32(8)
        lane8 = lane % Int32(8)
        group_tiles = Int32(self._groups_k // 4)
        task = Int32(bidx) * Int32(self._warps_per_cta) + warp
        total_tasks = m * group_tiles
        while task < total_tasks:
            row = task // group_tiles
            block = (task % group_tiles) * Int32(4) + subgroup
            k0 = block * Int32(32) + lane8 * Int32(4)
            g = k0 // Int32(self._span)
            inner0 = k0 - g * Int32(self._span)

            if cutlass.const_expr(self._mode == "group_major"):
                # Source physical [g, m, span]: regather group-major flat k.
                src0 = g * (m * Int32(self._span)) + row * Int32(self._span) + inner0
            else:
                src0 = row * Int32(self._total_k) + k0

            # Pure SSA scalars (no rmem array): divergent updates under the
            # rope branch then merge in registers instead of spilling.
            v0 = cutlass.Float32(source[src0 + Int32(0)])
            v1 = cutlass.Float32(source[src0 + Int32(1)])
            v2 = cutlass.Float32(source[src0 + Int32(2)])
            v3 = cutlass.Float32(source[src0 + Int32(3)])

            if cutlass.const_expr(self._inv_rope):
                # Branchless: every lane loads (clamped) cos/sin and computes
                # the rotation; a full-width bit mask selects rotated vs raw
                # values, so no divergent path lengthens the task loop and
                # -0.0 payloads survive untouched on nope lanes.
                head_d0 = k0 % Int32(self._head_dim)
                is_rope = head_d0 >= Int32(self._nope_dim)
                pos = Int32(positions[row])
                half_rope = Int32(self._rope_dim // 2)
                cs_base = pos * Int32(self._rope_dim)
                rl_half0 = (head_d0 - Int32(self._nope_dim)) // Int32(2)
                if rl_half0 < Int32(0):
                    rl_half0 = Int32(0)
                if cutlass.const_expr(self._cos_sin_type == cutlass.BFloat16):
                    # rl_half0 is even (head_d0 and nope_dim are multiples of
                    # 4), so each cos/sin pair sits on one aligned 4B word.
                    cos_w = ld_global_nc_u32(
                        get_ptr_as_int64(cos_sin, cs_base + rl_half0)
                    )
                    sin_w = ld_global_nc_u32(
                        get_ptr_as_int64(cos_sin, cs_base + rl_half0 + half_rope)
                    )
                    cos0, cos1 = bfloat2_to_float2_scaled(cos_w, cutlass.Float32(1.0))
                    sin0, sin1 = bfloat2_to_float2_scaled(sin_w, cutlass.Float32(1.0))
                else:
                    cos0 = cutlass.Float32(cos_sin[cs_base + rl_half0])
                    sin0 = cutlass.Float32(cos_sin[cs_base + rl_half0 + half_rope])
                    cos1 = cutlass.Float32(cos_sin[cs_base + rl_half0 + Int32(1)])
                    sin1 = cutlass.Float32(
                        cos_sin[cs_base + rl_half0 + Int32(1) + half_rope]
                    )
                r0 = v0 * cos0 + v1 * sin0
                r1 = v1 * cos0 - v0 * sin0
                r2 = v2 * cos1 + v3 * sin1
                r3 = v3 * cos1 - v2 * sin1
                mask = Uint32(0) - Uint32(is_rope)
                keep = mask ^ Uint32(0xFFFFFFFF)
                v0 = cutlass.Uint32(
                    (r0.bitcast(Uint32) & mask) | (v0.bitcast(Uint32) & keep)
                ).bitcast(cutlass.Float32)
                v1 = cutlass.Uint32(
                    (r1.bitcast(Uint32) & mask) | (v1.bitcast(Uint32) & keep)
                ).bitcast(cutlass.Float32)
                v2 = cutlass.Uint32(
                    (r2.bitcast(Uint32) & mask) | (v2.bitcast(Uint32) & keep)
                ).bitcast(cutlass.Float32)
                v3 = cutlass.Uint32(
                    (r3.bitcast(Uint32) & mask) | (v3.bitcast(Uint32) & keep)
                ).bitcast(cutlass.Float32)

            max_abs = fabs_f32(v0)
            max_abs = fmax_f32(max_abs, fabs_f32(v1))
            max_abs = fmax_f32(max_abs, fabs_f32(v2))
            max_abs = fmax_f32(max_abs, fabs_f32(v3))
            for shift in cutlass.range_constexpr(3):
                max_abs = fmax_f32(
                    max_abs,
                    cute.arch.shuffle_sync_bfly(max_abs, offset=1 << shift),
                )

            _, scale_byte = pow2_ceil_ue8m0(
                max_abs * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX)
            )
            if max_abs == cutlass.Float32(0.0):
                scale_byte = Uint32(127)
            inv_scale = ue8m0_to_output_scale(scale_byte)
            payload = cvt_f32x4_to_e4m3x4(
                v0 * inv_scale,
                v1 * inv_scale,
                v2 * inv_scale,
                v3 * inv_scale,
            )

            if cutlass.const_expr(self._mode == "grouped"):
                # Values physical [g, m, span]: transpose grouped rows.
                word = (
                    g * (m * Int32(self._span // 4))
                    + row * Int32(self._span // 4)
                    + inner0 // Int32(4)
                )
            else:
                word = row * Int32(self._total_k // 4) + k0 // Int32(4)
            values_u32[word] = payload

            if lane8 == Int32(0):
                if cutlass.const_expr(self._mode == "grouped"):
                    chunk = inner0 // Int32(32)
                    span_gk = Int32(self._span_groups_k)
                    scale_rows[g * (m * span_gk) + row * span_gk + chunk] = Uint8(
                        scale_byte
                    )
                    span_tiles_k = Int32((self._span_groups_k + 3) // 4)
                    m_tiles = (m + Int32(127)) // Int32(128)
                    self._store_scale_mma(
                        scale_mma,
                        g * (m_tiles * span_tiles_k * Int32(512)),
                        span_tiles_k,
                        row,
                        chunk,
                        scale_byte,
                    )
                else:
                    chunk = block
                    scale_rows[row * Int32(self._groups_k) + chunk] = Uint8(scale_byte)
                    self._store_scale_mma(
                        scale_mma,
                        Int32(0),
                        Int32((self._groups_k + 3) // 4),
                        row,
                        chunk,
                        scale_byte,
                    )
            task += Int32(gdim) * Int32(self._warps_per_cta)

    @cute.jit
    def _store_scale_mma(
        self,
        scale_mma: cute.Tensor,
        base: Int32,
        tiles_k: Int32,
        row: Int32,
        chunk: Int32,
        scale_byte: Uint32,
    ) -> None:
        row32 = row % Int32(32)
        row4 = (row // Int32(32)) % Int32(4)
        tile_m = row // Int32(128)
        k4 = chunk % Int32(4)
        tile_k = chunk // Int32(4)
        offset = (
            base
            + row32 * Int32(16)
            + row4 * Int32(4)
            + tile_m * (tiles_k * Int32(512))
            + k4
            + tile_k * Int32(512)
        )
        scale_mma[offset] = Uint8(scale_byte)


def _cutlass_source_type(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    raise TypeError(f"WO CuTe quantizer requires BF16/FP16 input, got {dtype}")


def _cutlass_positions_type(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.int64:
        return cutlass.Int64
    if dtype == torch.int32:
        return cutlass.Int32
    raise TypeError(f"WO CuTe quantizer positions must be int32/int64, got {dtype}")


def _cutlass_cos_sin_type(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"WO CuTe quantizer cos/sin must be bf16/fp32, got {dtype}")


@functools.cache
def _get_compiled_wo_quant(
    mode: str,
    total_k: int,
    span: int,
    source_dtype: torch.dtype,
    inv_rope: bool,
    head_dim: int,
    nope_dim: int,
    rope_dim: int,
    positions_dtype: torch.dtype,
    cos_sin_dtype: torch.dtype,
) -> Callable:
    source_type = _cutlass_source_type(source_dtype)
    positions_type = _cutlass_positions_type(positions_dtype)
    cos_sin_type = _cutlass_cos_sin_type(cos_sin_dtype)
    launch = _WOQuantCuTeLaunch(
        mode,
        total_k,
        span,
        source_type,
        inv_rope,
        head_dim,
        nope_dim,
        rope_dim,
        positions_type,
        cos_sin_type,
        _THREADS,
    )
    cache_key = (
        mode,
        int(total_k),
        int(span),
        str(source_dtype),
        bool(inv_rope),
        int(head_dim),
        int(nope_dim),
        int(rope_dim),
        str(positions_dtype),
        str(cos_sin_dtype),
        _THREADS,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=cache_key,
    )
    raw = sm12x_compile(
        launch,
        make_ptr(source_type, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(positions_type, 16, cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cos_sin_type, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.wo_quant_cute",
            2,
            cache_key,
        ),
    )

    def launch_tensors(
        source: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        values: torch.Tensor,
        scale_rows: torch.Tensor,
        scale_mma: torch.Tensor,
        m: int,
    ) -> None:
        groups_per_warp_tile = 4
        total_tasks = m * (total_k // 32 // groups_per_warp_tile)
        warps_per_cta = _THREADS // 32
        natural_grid = max(1, (total_tasks + warps_per_cta - 1) // warps_per_cta)
        sm_count = torch.cuda.get_device_properties(source.device).multi_processor_count
        grid_x = min(natural_grid, sm_count * _GRID_CTAS_PER_SM)
        raw(
            make_ptr(
                source_type,
                source.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                positions_type,
                positions.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                cos_sin_type,
                cos_sin.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Uint32,
                values.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Uint8,
                scale_rows.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Uint8,
                scale_mma.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            int(m),
            int(cos_sin.numel()),
            grid_x,
            current_cuda_stream(),
        )

    return launch_tensors


def quantize_wo_grouped_rows_cute(
    source_flat: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    *,
    m: int,
    groups: int,
    group_width: int,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    head_dim: int = 0,
    nope_dim: int = 0,
    rope_dim: int = 0,
) -> None:
    """Quantize flat `[m, groups*group_width]` rows into the grouped WO-A
    MXFP8 operand, optionally applying inverse RoPE first."""

    inv_rope = positions is not None
    if inv_rope:
        assert cos_sin_cache is not None
        pos_t: torch.Tensor = positions
        cs_t: torch.Tensor = cos_sin_cache
    else:
        pos_t = source_flat
        cs_t = source_flat
    _get_compiled_wo_quant(
        "grouped",
        groups * group_width,
        group_width,
        source_flat.dtype,
        inv_rope,
        head_dim if inv_rope else 0,
        nope_dim if inv_rope else 0,
        rope_dim if inv_rope else 0,
        pos_t.dtype if inv_rope else torch.int64,
        cs_t.dtype if inv_rope else torch.bfloat16,
    )(source_flat, pos_t, cs_t, values, scale_rows, scale_mma, m)


def quantize_wo_group_major_rows_cute(
    source_gmr: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    *,
    m: int,
    groups: int,
    rank: int,
) -> None:
    """Quantize the WO-A output (physical `[groups, m, rank]`) as group-major
    flat `[m, groups*rank]` MXFP8 rows for WO-B."""

    _get_compiled_wo_quant(
        "group_major",
        groups * rank,
        rank,
        source_gmr.dtype,
        False,
        0,
        0,
        0,
        torch.int64,
        torch.bfloat16,
    )(source_gmr, source_gmr, source_gmr, values, scale_rows, scale_mma, m)


__all__ = [
    "quantize_wo_group_major_rows_cute",
    "quantize_wo_grouped_rows_cute",
]
