# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/gemm/mxfp8_quant_cute.py @ a611e365 (2026-07-02) -- one-time curated port.
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
    cvt_f32x4_to_e4m3x4,
    fabs_f32,
    fmax_f32,
    max_abs_32,
    pow2_ceil_ue8m0,
    quantize_block_fp8_mx,
    ue8m0_to_output_scale,
)
from flashinfer.experimental.sm12x._lib.runtime_control import (
    raise_if_kernel_resolution_frozen,
)
from flashinfer.experimental.sm12x._lib.utils import current_cuda_stream, make_ptr


_THREADS = 256
_GRID_CTAS_PER_SM = 4
_WARP_SUBGROUP_WIDTH = 4


class _MXFP8RowsQuantLaunch:
    def __init__(
        self,
        k: int,
        source_type: type[cutlass.Numeric],
        subgroup_width: int,
        threads: int,
    ) -> None:
        self._k = int(k)
        self._groups_k = self._k // 32
        self._source_type = source_type
        self._subgroup_width = int(subgroup_width)
        self._threads = int(threads)
        self._warps_per_cta = self._threads // 32

    @cute.jit
    def __call__(
        self,
        source_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        scale_rows_ptr: cute.Pointer,
        scale_mma_ptr: cute.Pointer,
        m: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        source = cute.make_tensor(
            source_ptr,
            cute.make_ordered_layout((m, self._k), order=(1, 0)),
        )
        values_u32 = cute.make_tensor(
            values_ptr,
            cute.make_ordered_layout((m, self._k // 4), order=(1, 0)),
        )
        scale_rows = cute.make_tensor(
            scale_rows_ptr,
            cute.make_ordered_layout((m, self._groups_k), order=(1, 0)),
        )
        scale_mma = cute.make_tensor(
            scale_mma_ptr,
            cute.make_layout((max(512, ((self._groups_k + 3) // 4) * 512),)),
        )
        self.kernel(source, values_u32, scale_rows, scale_mma, m).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        values_u32: cute.Tensor,
        scale_rows: cute.Tensor,
        scale_mma: cute.Tensor,
        m: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        if cutlass.const_expr(self._subgroup_width == 4):
            # Eight 4-lane subgroups per warp each quantize one 32-value block.
            # Each lane owns eight adjacent values and emits two packed words.
            warp = Int32(tidx) // Int32(32)
            lane = Int32(tidx) % Int32(32)
            subgroup = lane // Int32(4)
            lane8 = lane % Int32(4)
            group_tiles = Int32((self._groups_k + 7) // 8)
            task = Int32(bidx) * Int32(self._warps_per_cta) + warp
            total_tasks = m * group_tiles
            while task < total_tasks:
                row = task // group_tiles
                group = (task % group_tiles) * Int32(8) + subgroup
                if group < Int32(self._groups_k):
                    values = cute.make_rmem_tensor((8,), cutlass.Float32)
                    k0 = group * Int32(32) + lane8 * Int32(8)
                    for elem in cutlass.range_constexpr(8):
                        values[elem] = cutlass.Float32(source[row, k0 + Int32(elem)])

                    max_abs = fabs_f32(values[0])
                    for elem in cutlass.range_constexpr(1, 8):
                        max_abs = fmax_f32(max_abs, fabs_f32(values[elem]))
                    for shift in cutlass.range_constexpr(2):
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
                    word0 = group * Int32(8) + lane8 * Int32(2)
                    values_u32[row, word0] = cvt_f32x4_to_e4m3x4(
                        values[0] * inv_scale,
                        values[1] * inv_scale,
                        values[2] * inv_scale,
                        values[3] * inv_scale,
                    )
                    values_u32[row, word0 + Int32(1)] = cvt_f32x4_to_e4m3x4(
                        values[4] * inv_scale,
                        values[5] * inv_scale,
                        values[6] * inv_scale,
                        values[7] * inv_scale,
                    )

                    if lane8 == Int32(0):
                        self._store_scale(scale_rows, scale_mma, row, group, scale_byte)
                task += Int32(gdim) * Int32(self._warps_per_cta)
        elif cutlass.const_expr(self._subgroup_width == 8):
            # Four 8-lane subgroups per warp each quantize one 32-value block.
            # Every lane owns four adjacent values, giving coalesced 128-value
            # warp loads/stores and a cheap width-8 butterfly max reduction.
            warp = Int32(tidx) // Int32(32)
            lane = Int32(tidx) % Int32(32)
            subgroup = lane // Int32(8)
            lane4 = lane % Int32(8)
            group_tiles = Int32((self._groups_k + 3) // 4)
            task = Int32(bidx) * Int32(self._warps_per_cta) + warp
            total_tasks = m * group_tiles
            while task < total_tasks:
                row = task // group_tiles
                group = (task % group_tiles) * Int32(4) + subgroup
                if group < Int32(self._groups_k):
                    values = cute.make_rmem_tensor((4,), cutlass.Float32)
                    k0 = group * Int32(32) + lane4 * Int32(4)
                    for elem in cutlass.range_constexpr(4):
                        values[elem] = cutlass.Float32(source[row, k0 + Int32(elem)])

                    max_abs = fabs_f32(values[0])
                    for elem in cutlass.range_constexpr(1, 4):
                        max_abs = fmax_f32(max_abs, fabs_f32(values[elem]))
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
                        values[0] * inv_scale,
                        values[1] * inv_scale,
                        values[2] * inv_scale,
                        values[3] * inv_scale,
                    )
                    values_u32[row, group * Int32(8) + lane4] = payload

                    if lane4 == Int32(0):
                        self._store_scale(scale_rows, scale_mma, row, group, scale_byte)
                task += Int32(gdim) * Int32(self._warps_per_cta)
        else:
            block = Int32(bidx) * Int32(self._threads) + Int32(tidx)
            total_blocks = m * Int32(self._groups_k)
            while block < total_blocks:
                row = block // Int32(self._groups_k)
                group = block % Int32(self._groups_k)
                values = cute.make_rmem_tensor((32,), cutlass.Float32)
                k0 = group * Int32(32)
                for elem in cutlass.range_constexpr(32):
                    values[elem] = cutlass.Float32(source[row, k0 + Int32(elem)])

                max_abs = max_abs_32(values)
                payload, scale_byte = quantize_block_fp8_mx(values, max_abs)
                if max_abs == cutlass.Float32(0.0):
                    scale_byte = Uint32(127)

                word0 = group * Int32(8)
                for word in cutlass.range_constexpr(8):
                    values_u32[row, word0 + Int32(word)] = payload[word]
                self._store_scale(scale_rows, scale_mma, row, group, scale_byte)
                block += Int32(gdim) * Int32(self._threads)

    @cute.jit
    def _store_scale(
        self,
        scale_rows: cute.Tensor,
        scale_mma: cute.Tensor,
        row: Int32,
        group: Int32,
        scale_byte: Uint32,
    ) -> None:
        scale_u8 = Uint8(scale_byte)
        scale_rows[row, group] = scale_u8
        row32 = row % Int32(32)
        row4 = (row // Int32(32)) % Int32(4)
        tile_m = row // Int32(128)
        k4 = group % Int32(4)
        tile_k = group // Int32(4)
        scale_mma_offset = (
            row32 * Int32(16)
            + row4 * Int32(4)
            + tile_m * Int32(((self._groups_k + 3) // 4) * 512)
            + k4
            + tile_k * Int32(512)
        )
        scale_mma[scale_mma_offset] = scale_u8


@functools.cache
def _get_compiled_mxfp8_rows_quant(
    k: int,
    source_dtype: torch.dtype,
    subgroup_width: int,
    threads: int,
) -> Callable:
    k = int(k)
    if k <= 0 or k % 32 != 0:
        raise ValueError(f"MXFP8 CuTe quantizer requires K divisible by 32, got {k}")
    if source_dtype == torch.bfloat16:
        source_type = cutlass.BFloat16
        source_dtype_name = "bf16"
    elif source_dtype == torch.float16:
        source_type = cutlass.Float16
        source_dtype_name = "fp16"
    else:
        raise TypeError(
            f"CuTe MXFP8 quantizer requires BF16 or FP16 input, got {source_dtype}"
        )
    if subgroup_width not in (0, 4, 8):
        raise ValueError(
            f"MXFP8 CuTe quantizer subgroup width must be 0, 4, or 8, got {subgroup_width}"
        )
    if threads <= 0 or threads % 32 != 0:
        raise ValueError(
            f"MXFP8 CuTe quantizer threads must be a positive multiple of 32, got {threads}"
        )
    launch = _MXFP8RowsQuantLaunch(k, source_type, subgroup_width, threads)
    cache_key = (k, source_dtype_name, int(subgroup_width), int(threads))
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=cache_key,
    )
    raw = sm12x_compile(
        launch,
        make_ptr(source_type, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.mxfp8_quant_cute",
            2,
            cache_key,
        ),
    )

    def launch_tensors(
        source: torch.Tensor,
        values: torch.Tensor,
        scale_rows: torch.Tensor,
        scale_mma: torch.Tensor,
    ) -> None:
        if subgroup_width:
            groups_per_warp = 32 // subgroup_width
            total_tasks = int(source.shape[0]) * (
                (k // 32 + groups_per_warp - 1) // groups_per_warp
            )
            warps_per_cta = threads // 32
            natural_grid = max(1, (total_tasks + warps_per_cta - 1) // warps_per_cta)
            sm_count = torch.cuda.get_device_properties(
                source.device
            ).multi_processor_count
            grid_x = min(natural_grid, sm_count * _GRID_CTAS_PER_SM)
        else:
            total_blocks = int(source.shape[0]) * (k // 32)
            grid_x = max(1, (total_blocks + threads - 1) // threads)
        raw(
            make_ptr(
                source_type,
                source.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
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
            int(source.shape[0]),
            grid_x,
            current_cuda_stream(),
        )

    return launch_tensors


def quantize_mxfp8_rows_cute(
    source: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
) -> None:
    """Quantize contiguous BF16 rows into the dense-GEMM MXFP8 layouts."""

    if source.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"CuTe MXFP8 quantizer requires BF16 or FP16 input, got {source.dtype}"
        )
    if source.ndim != 2 or not source.is_contiguous():
        raise ValueError("CuTe MXFP8 quantizer requires contiguous [M,K] input")
    subgroup_width = _WARP_SUBGROUP_WIDTH if int(source.shape[0]) > 8 else 0
    _get_compiled_mxfp8_rows_quant(
        int(source.shape[1]), source.dtype, subgroup_width, _THREADS
    )(
        source,
        values,
        scale_rows,
        scale_mma,
    )
