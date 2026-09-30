"""CuTeDSL BF16 projections for Qwen3.8 Flash Next MTP prefill."""

from __future__ import annotations

from threading import RLock
from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.utils import LayoutEnum

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream

from ._cute_prefill_config import supports_prefill, tensors_support_prefill
from b12x._lib.program_cache import register_program_cache


_TILE_N = 16
_TILE_K = 64
_STAGES = 3
_BUFFER_ALIGN_BYTES = 1_024
_CACHE_LOCK = RLock()
_KERNEL_CACHE: Dict[Tuple[int, int, int, int, int, bool], object] = {}
register_program_cache(_KERNEL_CACHE, lock=_CACHE_LOCK)


def _cutlass_runtime_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the exact Tensor type required by the CUTLASS runtime adapter."""
    if type(tensor) is torch.Tensor:
        return tensor
    result = tensor.detach()
    if type(result) is not torch.Tensor:
        raise TypeError(
            "MTP CuTe arguments must be torch.Tensor or detach to torch.Tensor, "
            f"got {type(tensor)!r}"
        )
    return result


def _convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    column_major = cute.make_layout(acc_layout.shape)
    shape = (
        (column_major.shape[0][1], column_major.shape[1]),
        (
            column_major.shape[0][0],
            *column_major.shape[0][2:],
            column_major.shape[2],
        ),
        *column_major.shape[3:],
    )
    stride = (
        (column_major.stride[0][1], column_major.stride[1]),
        (
            column_major.stride[0][0],
            *column_major.stride[0][2:],
            column_major.stride[2],
        ),
        *column_major.stride[3:],
    )
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def _reshape_acc_to_mn(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, _convert_layout_acc_mn(acc.layout))


@cute.jit
def _warp_mma_gemm(
    tiled_mma: cute.TiledMma,
    accumulator: cute.Tensor,
    register_a: cute.Tensor,
    register_b: cute.Tensor,
    shared_a: cute.Tensor,
    shared_b: cute.Tensor,
    copy_a: cute.TiledCopy,
    copy_b: cute.TiledCopy,
):
    register_a_copy = copy_a.retile(register_a)
    register_b_copy = copy_b.retile(register_b)
    cute.copy(copy_a, shared_a[None, None, 0], register_a_copy[None, None, 0])
    cute.copy(copy_b, shared_b[None, None, 0], register_b_copy[None, None, 0])
    for k_step in cutlass.range_constexpr(cute.size(shared_a.shape[2])):
        if k_step < cute.size(shared_a.shape[2]) - 1:
            cute.copy(
                copy_a,
                shared_a[None, None, k_step + 1],
                register_a_copy[None, None, k_step + 1],
            )
            cute.copy(
                copy_b,
                shared_b[None, None, k_step + 1],
                register_b_copy[None, None, k_step + 1],
            )
        cute.gemm(
            tiled_mma,
            accumulator,
            register_a[None, None, k_step],
            register_b[None, None, k_step],
            accumulator,
        )


class MTPPrefillBf16GemmKernel:
    """TMA-fed BF16 GEMM with an optional token broadcast-add epilogue."""

    def __init__(
        self,
        *,
        rows: int,
        output_columns: int,
        reduction_width: int,
        streams: int,
        add_token_path: bool,
        tile_m: int = 128,
    ) -> None:
        self.rows = int(rows)
        self.output_columns = int(output_columns)
        self.reduction_width = int(reduction_width)
        self.streams = int(streams)
        self.add_token_path = bool(add_token_path)
        self.tile_m = int(tile_m)
        if self.tile_m not in (16, 32, 64, 128):
            raise ValueError(f"tile_m must be 16, 32, 64, or 128, got {self.tile_m}")
        if self.rows % self.tile_m:
            raise ValueError(f"rows={self.rows} must be divisible by {self.tile_m}")
        if self.output_columns % _TILE_N:
            raise ValueError(
                f"output_columns={self.output_columns} must be divisible by {_TILE_N}"
            )
        if self.reduction_width % _TILE_K:
            raise ValueError(
                f"reduction_width={self.reduction_width} must be divisible by {_TILE_K}"
            )
        if self.add_token_path and self.rows % self.streams:
            raise ValueError(
                f"rows={self.rows} must be divisible by streams={self.streams}"
            )
        self.reduction_tiles = self.reduction_width // _TILE_K
        self.compute_warps = self.tile_m // 16
        self.producer_warp = self.compute_warps
        self.threads = (self.compute_warps + 1) * 32

    def _tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (self.compute_warps, 1, 1),
            permutation_mnk=(self.compute_warps * 16, _TILE_N, 16),
        )

    def _shared_layouts(self) -> tuple[cute.ComposedLayout, cute.ComposedLayout]:
        layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                cutlass.BFloat16,
                _TILE_K,
            ),
            cutlass.BFloat16,
        )
        layout_a = cute.tile_to_shape(
            layout_atom,
            (self.tile_m, _TILE_K, _STAGES),
            order=(0, 1, 2),
        )
        layout_b = cute.tile_to_shape(
            layout_atom,
            (_TILE_N, _TILE_K, _STAGES),
            order=(0, 1, 2),
        )
        return layout_a, layout_b

    def _shared_storage(
        self,
        layout_a: cute.ComposedLayout,
        layout_b: cute.ComposedLayout,
    ):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "barriers": cute.struct.MemRange[cutlass.Int64, _STAGES * 2],
            "a": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(layout_a)],
                _BUFFER_ALIGN_BYTES,
            ],
            "b": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(layout_b)],
                _BUFFER_ALIGN_BYTES,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        inputs: cute.Tensor,
        weight: cute.Tensor,
        token_path: cute.Tensor,
        output: cute.Tensor,
        live_rows: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(inputs.element_type != cutlass.BFloat16):
            raise TypeError("inputs must be BFloat16")
        if const_expr(weight.element_type != cutlass.BFloat16):
            raise TypeError("weight must be BFloat16")
        if const_expr(token_path.element_type != cutlass.BFloat16):
            raise TypeError("token_path must be BFloat16")
        if const_expr(output.element_type != cutlass.BFloat16):
            raise TypeError("output must be BFloat16")

        layout_a, layout_b = self._shared_layouts()
        tiled_mma = self._tiled_mma()
        SharedStorage = self._shared_storage(layout_a, layout_b)
        tma_layout_a = cute.slice_(layout_a, (None, None, 0))
        tma_layout_b = cute.slice_(layout_b, (None, None, 0))
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            inputs,
            tma_layout_a,
            (self.tile_m, _TILE_K),
            num_multicast=1,
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            weight,
            tma_layout_b,
            (_TILE_N, _TILE_K),
            num_multicast=1,
        )
        self.kernel(
            tma_tensor_a,
            tma_tensor_b,
            token_path,
            output,
            live_rows,
            tma_atom_a,
            tma_atom_b,
            layout_a,
            layout_b,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=(
                (live_rows + Int32(self.tile_m - 1)) // Int32(self.tile_m),
                self.output_columns // _TILE_N,
                1,
            ),
            block=[self.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        inputs: cute.Tensor,
        weight: cute.Tensor,
        token_path: cute.Tensor,
        output: cute.Tensor,
        live_rows: Int32,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        layout_a: cute.ComposedLayout,
        layout_b: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, _ = cute.arch.block_idx()
        warp_index = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_index == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        allocator = cutlass_utils.SmemAllocator()
        storage = allocator.allocate(SharedStorage)
        shared_a = storage.a.get_tensor(layout_a.outer, swizzle=layout_a.inner)
        shared_b = storage.b.get_tensor(layout_b.outer, swizzle=layout_b.inner)
        copy_bytes = (
            (self.tile_m + _TILE_N) * _TILE_K * cutlass.BFloat16.width // 8
        )
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.compute_warps,
            ),
            tx_count=copy_bytes,
            barrier_storage=storage.barriers.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()

        global_a = cute.local_tile(
            inputs,
            (self.tile_m, _TILE_K),
            (None, None),
        )
        global_b = cute.local_tile(
            weight,
            (_TILE_N, _TILE_K),
            (None, None),
        )
        cta_layout = cute.make_layout(1)
        partition_shared_a, partition_global_a = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(shared_a, 0, 2),
            cute.group_modes(global_a, 0, 2),
        )
        partition_shared_b, partition_global_b = cpasync.tma_partition(
            tma_atom_b,
            0,
            cta_layout,
            cute.group_modes(shared_b, 0, 2),
            cute.group_modes(global_b, 0, 2),
        )

        if warp_index < Int32(self.compute_warps):
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _STAGES,
            )
            thread_mma = tiled_mma.get_slice(tidx)
            thread_shared_a = thread_mma.partition_A(shared_a)
            thread_shared_b = thread_mma.partition_B(shared_b)
            register_a = thread_mma.make_fragment_A(
                thread_shared_a[None, None, None, 0]
            )
            register_b = thread_mma.make_fragment_B(
                thread_shared_b[None, None, None, 0]
            )
            accumulator_shape = thread_mma.partition_shape_C((self.tile_m, _TILE_N))
            accumulator = cute.make_rmem_tensor(accumulator_shape, Float32)
            accumulator.fill(0.0)
            copy_atom_a = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            copy_atom_b = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma).get_slice(tidx)
            copy_b = cute.make_tiled_copy_B(copy_atom_b, tiled_mma).get_slice(tidx)
            copy_source_a = copy_a.partition_S(shared_a)
            copy_source_b = copy_b.partition_S(shared_b)

            for _ in cutlass.range_constexpr(self.reduction_tiles):
                load_pipeline.consumer_wait(consumer_state)
                _warp_mma_gemm(
                    tiled_mma,
                    accumulator,
                    register_a,
                    register_b,
                    copy_source_a[None, None, None, consumer_state.index],
                    copy_source_b[None, None, None, consumer_state.index],
                    copy_a,
                    copy_b,
                )
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            accumulator_mn = _reshape_acc_to_mn(accumulator)
            coordinate_mn = _reshape_acc_to_mn(
                thread_mma.partition_C(
                    cute.make_identity_tensor((self.tile_m, _TILE_N))
                )
            )
            for accumulator_m in cutlass.range_constexpr(
                cute.size(accumulator_mn.shape[0])
            ):
                for accumulator_n in cutlass.range_constexpr(
                    cute.size(accumulator_mn.shape[1])
                ):
                    coordinate = coordinate_mn[accumulator_m, accumulator_n]
                    row = m_tile * Int32(self.tile_m) + coordinate[0]
                    column = n_tile * Int32(_TILE_N) + coordinate[1]
                    value = accumulator_mn[accumulator_m, accumulator_n]
                    if cutlass.const_expr(self.add_token_path):
                        token = row // Int32(self.streams)
                        token_offset = (
                            token.to(Int64) * Int64(self.output_columns)
                            + column.to(Int64)
                        )
                        token_value = token_path[token_offset]
                        value = Float32(cutlass.BFloat16(value)) + Float32(token_value)
                    if row < live_rows:
                        output_offset = (
                            row.to(Int64) * Int64(self.output_columns)
                            + column.to(Int64)
                        )
                        output[output_offset] = cutlass.BFloat16(value)

        elif warp_index == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _STAGES,
            )
            for reduction_tile in cutlass.range_constexpr(self.reduction_tiles):
                load_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_a,
                    partition_global_a[(None, m_tile, reduction_tile)],
                    partition_shared_a[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_b,
                    partition_global_b[(None, n_tile, reduction_tile)],
                    partition_shared_b[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


def _cache_key(
    rows: int,
    output_columns: int,
    reduction_width: int,
    *,
    device: torch.device,
    streams: int,
    add_token_path: bool,
) -> tuple[int, int, int, int, int, bool]:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        int(device_index),
        int(rows),
        int(output_columns),
        int(reduction_width),
        int(streams),
        bool(add_token_path),
    )


def compile_mtp_prefill_bf16_gemm(
    rows: int,
    output_columns: int,
    reduction_width: int,
    *,
    device: torch.device,
    streams: int,
    add_token_path: bool,
):
    """Compile one capacity-shaped projection into caller-owned output."""
    cache_key = _cache_key(
        rows,
        output_columns,
        reduction_width,
        device=device,
        streams=streams,
        add_token_path=add_token_path,
    )
    with _CACHE_LOCK:
        cached = _KERNEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        tile_m = next(
            (candidate for candidate in (128, 64, 32, 16) if rows % candidate == 0),
            None,
        )
        if tile_m is None:
            raise ValueError(
                "MTP CuTe rows must be a multiple of 16 and at least 16, "
                f"got {rows}"
            )
        kernel = MTPPrefillBf16GemmKernel(
            rows=rows,
            output_columns=output_columns,
            reduction_width=reduction_width,
            streams=streams,
            add_token_path=add_token_path,
            tile_m=tile_m,
        )
        token_rows = rows // streams if add_token_path else 1
        inputs_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (rows, reduction_width),
            stride_order=(1, 0),
            assumed_align=16,
        )
        weight_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (output_columns, reduction_width),
            stride_order=(1, 0),
            assumed_align=16,
        )
        token_path_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (token_rows * output_columns,),
            assumed_align=16,
        )
        output_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (rows * output_columns,),
            assumed_align=16,
        )
        raise_if_kernel_resolution_frozen(
            "cute.compile",
            target=kernel,
            cache_key=cache_key,
        )
        with torch.cuda.device(device):
            raw = b12x_compile(
                kernel,
                inputs_fake,
                weight_fake,
                token_path_fake,
                output_fake,
                Int32(rows),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "sequence.mtp_feedback.prefill_bf16_gemm",
                    3,
                    cache_key[1:],
                ),
            )

        def launch(
            inputs: torch.Tensor,
            weight: torch.Tensor,
            output: torch.Tensor,
            *,
            token_path: torch.Tensor | None = None,
            live_rows: int | None = None,
        ) -> None:
            if add_token_path and token_path is None:
                raise ValueError(
                    "token_path is required for the broadcast-add epilogue"
                )
            token_argument = output if token_path is None else token_path
            live = rows if live_rows is None else int(live_rows)
            if live < 0 or live > rows:
                raise ValueError(f"live_rows must be in [0, {rows}], got {live}")
            if live == 0:
                return
            runtime_inputs = _cutlass_runtime_tensor(inputs)
            runtime_weight = _cutlass_runtime_tensor(weight)
            runtime_token = _cutlass_runtime_tensor(token_argument)
            runtime_output = _cutlass_runtime_tensor(output)
            with torch.cuda.device(runtime_inputs.device):
                capturing = torch.cuda.is_current_stream_capturing()
                if capturing and not launch.__b12x_warmed__:
                    raise RuntimeError(
                        "MTP CuTe prefill kernels must be warm-run before CUDA graph "
                        "capture"
                    )
                raw(
                    runtime_inputs,
                    runtime_weight,
                    runtime_token.view(-1),
                    runtime_output.view(-1),
                    Int32(live),
                    current_cuda_stream(),
                )
            if not capturing:
                launch.__b12x_warmed__ = True

        attach_programs(launch, raw)
        launch.__b12x_warmed__ = False
        _KERNEL_CACHE[cache_key] = launch
        return launch


def compile_mtp_prefill_capacity(
    token_rows: int,
    state_rows: int,
    hidden_size: int,
    *,
    device: torch.device,
    streams: int,
) -> tuple:
    """Return both capacity-specialized Qwen projection entry points."""
    token = compile_mtp_prefill_bf16_gemm(
        token_rows,
        hidden_size,
        hidden_size,
        device=device,
        streams=streams,
        add_token_path=False,
    )
    state = compile_mtp_prefill_bf16_gemm(
        state_rows,
        hidden_size,
        hidden_size,
        device=device,
        streams=streams,
        add_token_path=True,
    )
    return token, state


def precompile_mtp_prefill_capacity(
    token_rows: int,
    state_rows: int,
    hidden_size: int,
    *,
    device: torch.device,
    streams: int,
) -> None:
    compile_mtp_prefill_capacity(
        token_rows, state_rows, hidden_size, device=device, streams=streams
    )


def get_cached_mtp_prefill_bf16_gemm(
    rows: int,
    output_columns: int,
    reduction_width: int,
    *,
    device: torch.device,
    streams: int,
    add_token_path: bool,
):
    """Return an already compiled specialization without triggering JIT."""
    cache_key = _cache_key(
        rows,
        output_columns,
        reduction_width,
        device=device,
        streams=streams,
        add_token_path=add_token_path,
    )
    with _CACHE_LOCK:
        return _KERNEL_CACHE.get(cache_key)


def is_mtp_prefill_bf16_gemm_warmed(
    rows: int,
    output_columns: int,
    reduction_width: int,
    *,
    device: torch.device,
    streams: int,
    add_token_path: bool,
) -> bool:
    """Return whether the cached callable has launched outside graph capture."""
    cache_key = _cache_key(
        rows,
        output_columns,
        reduction_width,
        device=device,
        streams=streams,
        add_token_path=add_token_path,
    )
    with _CACHE_LOCK:
        cached = _KERNEL_CACHE.get(cache_key)
        return cached is not None and cached.__b12x_warmed__


__all__ = [
    "compile_mtp_prefill_bf16_gemm",
    "get_cached_mtp_prefill_bf16_gemm",
    "is_mtp_prefill_bf16_gemm_warmed",
    "precompile_mtp_prefill_capacity",
    "supports_prefill",
    "tensors_support_prefill",
]
