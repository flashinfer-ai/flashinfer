from __future__ import annotations

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Uint8, Uint32

from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as b12x_compile,
)
from b12x._lib.intrinsics import (
    FLOAT8_E4M3_MAX,
    cvt_f32x4_to_e4m3x4,
    fabs_f32,
    fmax_f32,
    max_abs_32,
    pow2_ceil_ue8m0,
    quantize_block_fp8_mx,
    ue8m0_to_output_scale,
)
from b12x._lib.runtime_control import (
    raise_if_kernel_resolution_frozen,
)
from b12x._lib.utils import current_cuda_stream, make_ptr


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
        trellis_native_mma_order: bool,
        min_amax: float,
    ) -> None:
        self._k = int(k)
        self._groups_k = self._k // 32
        self._source_type = source_type
        self._subgroup_width = int(subgroup_width)
        self._threads = int(threads)
        self._warps_per_cta = self._threads // 32
        self._trellis_native_mma_order = bool(trellis_native_mma_order)
        self._min_amax = float(min_amax)

    @cute.jit
    def __call__(
        self,
        source_ptr: cute.Pointer,
        values_ptr: cute.Pointer,
        scale_rows_ptr: cute.Pointer,
        scale_mma_ptr: cute.Pointer,
        m: Int32,
        source_k: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        source = cute.make_tensor(
            source_ptr,
            cute.make_ordered_layout((m, source_k), order=(1, 0)),
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
        self.kernel(source, values_u32, scale_rows, scale_mma, m, source_k).launch(
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
        source_k: Int32,
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
                        values[elem] = cutlass.Float32(0.0)
                        if k0 + Int32(elem) < source_k:
                            values[elem] = cutlass.Float32(
                                source[row, k0 + Int32(elem)]
                            )
                    max_abs = fabs_f32(values[0])
                    for elem in cutlass.range_constexpr(1, 8):
                        max_abs = fmax_f32(max_abs, fabs_f32(values[elem]))
                    for shift in cutlass.range_constexpr(2):
                        max_abs = fmax_f32(
                            max_abs,
                            cute.arch.shuffle_sync_bfly(max_abs, offset=1 << shift),
                        )

                    if cutlass.const_expr(self._min_amax > 0.0):
                        max_abs = fmax_f32(max_abs, cutlass.Float32(self._min_amax))
                    if k0 >= source_k:
                        max_abs = cutlass.Float32(0.0)
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
                    k0 = group * Int32(32)
                    if cutlass.const_expr(self._trellis_native_mma_order):
                        # Match the direct native-trellis B assignment.  Each
                        # output word contains one native EXL lane's four K
                        # values; the eight words are a permutation wholly
                        # within this K32 scale group.
                        r = Int32(0)
                        tile = Int32(0)
                        if lane4 < Int32(4):
                            r = ((lane4 & Int32(1)) << Int32(1)) | (
                                lane4 >> Int32(1)
                            )
                        else:
                            q = lane4 - Int32(4)
                            r = (
                                ((Int32(1) - (q & Int32(1))) << Int32(1))
                                | (q >> Int32(1))
                            )
                            tile = Int32(16)
                        base = k0 + tile + (r << Int32(1))
                        values[0] = cutlass.Float32(0.0)
                        values[1] = cutlass.Float32(0.0)
                        values[2] = cutlass.Float32(0.0)
                        values[3] = cutlass.Float32(0.0)
                        if base < source_k:
                            values[0] = cutlass.Float32(source[row, base])
                        if base + Int32(1) < source_k:
                            values[1] = cutlass.Float32(source[row, base + Int32(1)])
                        if base + Int32(8) < source_k:
                            values[2] = cutlass.Float32(source[row, base + Int32(8)])
                        if base + Int32(9) < source_k:
                            values[3] = cutlass.Float32(source[row, base + Int32(9)])
                    else:
                        k0 += lane4 * Int32(4)
                        for elem in cutlass.range_constexpr(4):
                            values[elem] = cutlass.Float32(0.0)
                            if k0 + Int32(elem) < source_k:
                                values[elem] = cutlass.Float32(
                                    source[row, k0 + Int32(elem)]
                                )

                    max_abs = fabs_f32(values[0])
                    for elem in cutlass.range_constexpr(1, 4):
                        max_abs = fmax_f32(max_abs, fabs_f32(values[elem]))
                    for shift in cutlass.range_constexpr(3):
                        max_abs = fmax_f32(
                            max_abs,
                            cute.arch.shuffle_sync_bfly(max_abs, offset=1 << shift),
                        )

                    if cutlass.const_expr(self._min_amax > 0.0):
                        max_abs = fmax_f32(max_abs, cutlass.Float32(self._min_amax))
                    if k0 >= source_k:
                        max_abs = cutlass.Float32(0.0)
                    _, scale_byte = pow2_ceil_ue8m0(
                        max_abs * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX)
                    )
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
                k0 = group * Int32(32)
                values = cute.make_rmem_tensor((32,), cutlass.Float32)
                for elem in cutlass.range_constexpr(32):
                    values[elem] = cutlass.Float32(0.0)
                    if k0 + Int32(elem) < source_k:
                        values[elem] = cutlass.Float32(source[row, k0 + Int32(elem)])
                max_abs = max_abs_32(values)
                if cutlass.const_expr(self._min_amax > 0.0):
                    max_abs = fmax_f32(max_abs, cutlass.Float32(self._min_amax))
                if k0 >= source_k:
                    max_abs = cutlass.Float32(0.0)
                payload, scale_byte = quantize_block_fp8_mx(values, max_abs)

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


@program_cache
def _get_compiled_mxfp8_rows_quant(
    k: int,
    source_dtype: torch.dtype,
    subgroup_width: int,
    threads: int,
    value_order: str,
    min_amax: float = 0.0,
    *,
    device_ordinal: int | None = None,
    sm_count: int | None = None,
) -> Callable:
    k = int(k)
    device_ordinal = torch.cuda.current_device() if device_ordinal is None else device_ordinal
    sm_count = torch.cuda.get_device_properties(device_ordinal).multi_processor_count if sm_count is None else sm_count
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
    if value_order not in {"linear", "trellis_native_mma"}:
        raise ValueError(
            "MXFP8 CuTe quantizer value_order must be 'linear' or "
            f"'trellis_native_mma', got {value_order!r}"
        )
    if value_order == "trellis_native_mma" and subgroup_width != 8:
        raise ValueError(
            "trellis_native_mma MXFP8 ordering requires subgroup_width=8"
        )
    if min_amax not in (0.0, 1e-4):
        raise ValueError("MXFP8 min_amax must be 0.0 or 1e-4")
    launch = _MXFP8RowsQuantLaunch(
        k,
        source_type,
        subgroup_width,
        threads,
        value_order == "trellis_native_mma",
        min_amax,
    )
    cache_key = (
        k,
        source_dtype_name,
        int(subgroup_width),
        int(threads),
        value_order,
        min_amax,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=cache_key,
    )
    with torch.cuda.device(device_ordinal):
        raw = b12x_compile(
            launch,
            make_ptr(source_type, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
            1,
            1,
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "gemm.mxfp8_quant_cute",
                4,
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
            int(source.shape[1]),
            grid_x,
            current_cuda_stream(),
        )

    return attach_programs(launch_tensors, raw)


def mxfp8_rows_quant_launch_options(
    planned_rows: int,
    value_order: str,
) -> tuple[int, int]:
    """Return the existing row-quantizer specialization for a declared M bound."""
    if type(planned_rows) is not int or planned_rows <= 0:
        raise ValueError("MXFP8 planned rows must be a positive integer")
    if value_order == "trellis_native_mma":
        return 8, _THREADS
    if value_order != "linear":
        raise ValueError(f"unsupported MXFP8 value order {value_order!r}")
    if planned_rows <= 8:
        return 8, 128
    return _WARP_SUBGROUP_WIDTH, _THREADS


def quantize_mxfp8_rows_cute(
    source: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    *,
    value_order: str = "linear",
    expected_m: int | None = None,
    min_amax: float = 0.0,
    physical_k: int | None = None,
) -> None:
    """Quantize contiguous BF16 rows into dense-GEMM MXFP8 layouts.

    ``trellis_native_mma`` applies the fixed within-K32 byte permutation used
    by direct native-trellis E4M3 B fragments.  It changes neither values nor
    scale groups and avoids a separate activation transpose kernel.

    expected_m fixes the row bound used for lane-layout specialization.
    min_amax=1e-4 selects the DeepSeek V4.1 activation scale floor.
    """

    if source.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"CuTe MXFP8 quantizer requires BF16 or FP16 input, got {source.dtype}"
        )
    if source.ndim != 2 or not source.is_contiguous():
        raise ValueError("CuTe MXFP8 quantizer requires contiguous [M,K] input")
    planned_rows = int(source.shape[0]) if expected_m is None else expected_m
    subgroup_width, threads = mxfp8_rows_quant_launch_options(planned_rows, value_order)
    physical_k = int(source.shape[1]) if physical_k is None else int(physical_k)
    if physical_k < int(source.shape[1]) or physical_k % 32:
        raise ValueError(
            "MXFP8 CuTe quantizer physical K must be a multiple of 32 no smaller "
            f"than source K, got physical={physical_k}, source={int(source.shape[1])}"
        )
    _get_compiled_mxfp8_rows_quant(
        physical_k,
        source.dtype,
        subgroup_width,
        threads,
        value_order,
        min_amax,
        device_ordinal=source.device.index,
    )(
        source,
        values,
        scale_rows,
        scale_mma,
    )
