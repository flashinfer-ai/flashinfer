"""Materialize uniform-rate intermediate-Hadamard trellis payloads into BF16 weight banks.

The kernel consumes the same native trellis tensors and scale tables as the
fused MoE path.  Each CTA reconstructs one final 128x128 weight tile, applies
the ordinary and intermediate Hadamard transforms in FP32 on chip, and publishes
the BF16 tile with a TMA shared-to-global store.  No dense FP32 matrix is
written to global memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    fp8_e4m3_to_f32,
    get_ptr_as_int64,
    ld_global_nc_u32,
    shared_ptr_to_u32,
    st_shared_u32,
)
from b12x._lib.quant.lut_e4m3 import lut_e4m3_value_table
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_had128_quad,
    _w4a8_trellis_decode_both,
    _w4a8_trellis_lane_geom,
)


_TILE = 128
_TILE16 = _TILE // 16
_THREADS = 256
_WARPS = _THREADS // 32
_MAX_WORDS_PER_TILE = 24
_WORK_ELEMENTS = _TILE * _TILE
_RING_WORDS = _WARPS * _MAX_WORDS_PER_TILE


@dataclass(frozen=True)
class IntermediateHadamardUniformMaterializerInputs:
    """Packed weights and transform tables for one complete expert extent."""

    upstream_packed: torch.Tensor
    upstream_suh: torch.Tensor
    upstream_svh: torch.Tensor
    upstream_signs_k: torch.Tensor
    upstream_signs_n: torch.Tensor
    down_packed: torch.Tensor
    down_suh: torch.Tensor
    down_svh: torch.Tensor
    down_signs_k: torch.Tensor
    down_signs_n: torch.Tensor


def _cute_tensor(value: torch.Tensor, dtype) -> cute.Tensor:
    tensor = from_dlpack(value, assumed_align=16)
    tensor.element_type = dtype
    return tensor


@cute.jit
def _h4_coefficient(source_group: cutlass.Constexpr, output_group: Int32):
    coefficient = cutlass.Float32(0.5)
    if cutlass.const_expr(source_group == 1):
        if (output_group & Int32(1)) != Int32(0):
            coefficient = cutlass.Float32(-0.5)
    elif cutlass.const_expr(source_group == 2):
        if output_group >= Int32(2):
            coefficient = cutlass.Float32(-0.5)
    elif cutlass.const_expr(source_group == 3):
        if output_group == Int32(1) or output_group == Int32(2):
            coefficient = cutlass.Float32(-0.5)
    return coefficient


class IntermediateHadamardUniformMaterializer:
    """Decode one uniform K2 or K3 matrix family into its final BF16 basis.

    The stored tensor has shape ``[M, K/16, N/16, 16*bits]`` in int16 units. The
    output is the transposed dense matrix family ``[E*N_out, K]``.  Down
    projections have ``M=E`` and ``N_out=N``.  Upstream extents with an
    intermediate Hadamard have
    ``M=2E`` and ``N_out=2N``: the two physical FC1 slots are reassembled into
    the interleaved gate/up coordinate order consumed by the intermediate H128.

    ``hidden_axis`` selects the dimension carrying the H512.  The other
    dimension carries the sign-pattern-dependent sign vector after its second
    H128.  This covers both upstream matrices (hidden K) and W2 (hidden N).
    """

    def __init__(
        self,
        *,
        experts: int,
        k: int,
        n: int,
        bits: int,
        hidden_axis: str,
        joined_upstream: bool = False,
    ) -> None:
        self.experts = int(experts)
        self.k = int(k)
        self.n = int(n)
        self.bits = int(bits)
        self.hidden_axis = str(hidden_axis)
        self.joined_upstream = bool(joined_upstream)
        if self.experts <= 0:
            raise ValueError("experts must be positive")
        if self.bits not in (2, 3):
            raise ValueError("uniform materialization supports K2 or K3")
        if self.k <= 0 or self.k % _TILE or self.n <= 0 or self.n % _TILE:
            raise ValueError("uniform materialization requires K,N multiples of 128")
        if self.hidden_axis not in {"k", "n"}:
            raise ValueError("hidden_axis must be 'k' or 'n'")
        hidden = self.k if self.hidden_axis == "k" else self.n
        if hidden % 512:
            raise ValueError("the hidden axis must close H512 blocks")
        if self.joined_upstream and self.hidden_axis != "k":
            raise ValueError("joined upstream payloads require the K hidden axis")
        self.source_matrices = self.experts * (2 if self.joined_upstream else 1)
        self.scale_k_banks = 2 if self.joined_upstream else 1
        self.logical_n = self.n * (2 if self.joined_upstream else 1)
        if self.logical_n % _TILE:
            raise ValueError("materialized output width must be a multiple of 128")
        self.k16 = self.k // 16
        self.source_n16 = self.n // 16
        self.k128 = self.k // _TILE
        self.source_n128 = self.n // _TILE
        self.n128 = self.logical_n // _TILE
        self.hidden128 = self.k128 if self.hidden_axis == "k" else self.n128
        self.other128 = self.n128 if self.hidden_axis == "k" else self.k128
        self.hidden_pairs = self.hidden128 // 2
        self.blocks = self.experts * self.hidden_pairs * self.other128

    @property
    def __cache_key__(self):
        return (
            self.experts,
            self.k,
            self.n,
            self.bits,
            self.hidden_axis,
            self.joined_upstream,
        )

    @cute.jit
    def __call__(
        self,
        packed: cute.Tensor,
        suh: cute.Tensor,
        svh: cute.Tensor,
        signs_k: cute.Tensor,
        signs_n: cute.Tensor,
        value_table: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        output_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR,
                cutlass.BFloat16,
                _TILE,
            ),
            cutlass.BFloat16,
        )
        output_staged_layout = cute.tile_to_shape(
            output_atom,
            (_TILE, _TILE, 1),
            order=(0, 1, 2),
        )
        output_smem_layout = cute.slice_(output_staged_layout, (None, None, 0))
        tma_store, tma_output = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            output,
            output_smem_layout,
            (_TILE, _TILE),
        )
        self.kernel(
            packed,
            suh,
            svh,
            signs_k,
            signs_n,
            value_table,
            tma_store,
            tma_output,
            output_staged_layout,
            cute.cosize(output_staged_layout),
        ).launch(
            grid=(self.blocks, 1, 1),
            block=[_THREADS, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _decode_tile(
        self,
        work: cute.Tensor,
        ring_base: Int32,
        lut_addr: Int64,
        packed_base: Int64,
        source_matrix: Int32,
        kb: Int32,
        nb: Int32,
        tid: Int32,
        lane: Int32,
        warp: Int32,
    ):
        ia, ib, shift = _w4a8_trellis_lane_geom(lane, self.bits)
        words_per_tile = Int32(8 * self.bits)
        for iteration in cutlass.range_constexpr(8):
            local_tile = warp + Int32(iteration * _WARPS)
            k16_local = local_tile >> Int32(3)
            n16_local = local_tile & Int32(7)
            k16 = kb * Int32(_TILE16) + k16_local
            n16 = nb * Int32(_TILE16) + n16_local
            tile = (
                (source_matrix * Int32(self.k16) + k16)
                * Int32(self.source_n16)
                + n16
            )
            warp_ring = ring_base + warp * Int32(_MAX_WORDS_PER_TILE * 4)
            if lane < words_per_tile:
                word = Int64(tile) * Int64(8 * self.bits) + Int64(lane)
                st_shared_u32(
                    warp_ring + (lane << Int32(2)),
                    ld_global_nc_u32(packed_base + (word << Int64(2))),
                )
            cute.arch.sync_warp()
            lo, hi = _w4a8_trellis_decode_both(
                warp_ring,
                Int32(0),
                ia,
                ib,
                shift,
                self.bits,
                lut_addr,
                lut_in_smem=False,
            )
            row0 = (lane & Int32(3)) << Int32(1)
            col0 = lane >> Int32(3)
            parity = (lane >> Int32(2)) & Int32(1)
            base_row = k16_local * Int32(16)
            base_col = n16_local * Int32(16)
            for value_index in cutlass.range_constexpr(8):
                values = lo if cutlass.const_expr(value_index < 4) else hi
                byte_index = value_index if value_index < 4 else value_index - 4
                byte = (values >> Uint32(byte_index * 8)) & Uint32(0xFF)
                local_value = Int32(value_index & 3)
                row = row0 + local_value
                if local_value >= Int32(2):
                    row += Int32(6)
                col_group = col0
                if cutlass.const_expr(value_index >= 4):
                    col_group += Int32(4)
                col = (col_group << Int32(1)) + parity
                work[base_row + row, base_col + col] = fp8_e4m3_to_f32(byte)
            cute.arch.sync_warp()

    @cute.jit
    def _decode_upstream_tile(
        self,
        work: cute.Tensor,
        staging: cute.Tensor,
        ring_base: Int32,
        lut_addr: Int64,
        packed_base: Int64,
        expert: Int32,
        kb: Int32,
        pre_block: Int32,
        tid: Int32,
        lane: Int32,
        warp: Int32,
    ):
        source_nb = pre_block >> Int32(1)
        source_half = pre_block & Int32(1)
        for slot in cutlass.range_constexpr(2):
            source_matrix = Int32(slot * self.experts) + expert
            self._decode_tile(
                work,
                ring_base,
                lut_addr,
                packed_base,
                source_matrix,
                kb,
                source_nb,
                tid,
                lane,
                warp,
            )
            cute.arch.sync_threads()
            for iteration in cutlass.range_constexpr(32):
                element = tid + Int32(iteration * _THREADS)
                row = element >> Int32(6)
                local_col = element & Int32(63)
                source_col = (source_half << Int32(6)) + local_col
                slot_in_pair = local_col >> Int32(5)
                within_slot = local_col & Int32(31)
                logical_col = (
                    slot_in_pair * Int32(64)
                    + Int32(slot * 32)
                    + within_slot
                )
                staging[row, logical_col] = cutlass.BFloat16(
                    work[row, source_col]
                )
            cute.arch.sync_threads()
        for iteration in cutlass.range_constexpr(64):
            element = tid + Int32(iteration * _THREADS)
            row = element >> Int32(7)
            col = element & Int32(127)
            work[row, col] = staging[row, col].to(cutlass.Float32)
        cute.arch.sync_threads()

    @cute.jit
    def _hadamard_k(
        self,
        work: cute.Tensor,
        scale: cute.Tensor,
        signs_k: cute.Tensor,
        expert: Int32,
        kb: Int32,
        output_nb: Int32,
        lane: Int32,
        warp: Int32,
        *,
        first: cutlass.Constexpr,
    ):
        row = lane << Int32(2)
        for iteration in cutlass.range_constexpr(16):
            col = warp + Int32(iteration * _WARPS)
            v0 = work[row + Int32(0), col]
            v1 = work[row + Int32(1), col]
            v2 = work[row + Int32(2), col]
            v3 = work[row + Int32(3), col]
            v0, v1, v2, v3 = _w4a8_had128_quad(v0, v1, v2, v3, lane)
            scale_expert = expert
            if cutlass.const_expr(self.joined_upstream):
                projection = output_nb // Int32(self.source_n128)
                scale_expert = projection * Int32(self.experts) + expert
            scale_base = (
                scale_expert * Int32(self.k) + kb * Int32(_TILE) + row
            )
            sign_base = expert * Int32(self.k) + kb * Int32(_TILE) + row
            if cutlass.const_expr(first):
                v0 *= scale[scale_base + Int32(0)].to(cutlass.Float32)
                v1 *= scale[scale_base + Int32(1)].to(cutlass.Float32)
                v2 *= scale[scale_base + Int32(2)].to(cutlass.Float32)
                v3 *= scale[scale_base + Int32(3)].to(cutlass.Float32)
            elif cutlass.const_expr(self.hidden_axis == "n"):
                v0 *= signs_k[sign_base + Int32(0)].to(cutlass.Float32)
                v1 *= signs_k[sign_base + Int32(1)].to(cutlass.Float32)
                v2 *= signs_k[sign_base + Int32(2)].to(cutlass.Float32)
                v3 *= signs_k[sign_base + Int32(3)].to(cutlass.Float32)
            work[row + Int32(0), col] = v0
            work[row + Int32(1), col] = v1
            work[row + Int32(2), col] = v2
            work[row + Int32(3), col] = v3
        cute.arch.sync_threads()

    @cute.jit
    def _hadamard_n(
        self,
        work: cute.Tensor,
        scale: cute.Tensor,
        signs_n: cute.Tensor,
        expert: Int32,
        source_nb: Int32,
        output_nb: Int32,
        lane: Int32,
        warp: Int32,
        *,
        first: cutlass.Constexpr,
    ):
        col = lane << Int32(2)
        for iteration in cutlass.range_constexpr(16):
            row = warp + Int32(iteration * _WARPS)
            v0 = work[row, col + Int32(0)]
            v1 = work[row, col + Int32(1)]
            v2 = work[row, col + Int32(2)]
            v3 = work[row, col + Int32(3)]
            v0, v1, v2, v3 = _w4a8_had128_quad(v0, v1, v2, v3, lane)
            scale_nb = source_nb
            if cutlass.const_expr(self.joined_upstream):
                scale_nb = output_nb
            base = (
                expert * Int32(self.logical_n)
                + scale_nb * Int32(_TILE)
                + col
            )
            if cutlass.const_expr(first):
                v0 *= scale[base + Int32(0)].to(cutlass.Float32)
                v1 *= scale[base + Int32(1)].to(cutlass.Float32)
                v2 *= scale[base + Int32(2)].to(cutlass.Float32)
                v3 *= scale[base + Int32(3)].to(cutlass.Float32)
            elif cutlass.const_expr(self.hidden_axis == "k"):
                sign_base = (
                    expert * Int32(self.logical_n)
                    + output_nb * Int32(_TILE)
                    + col
                )
                v0 *= signs_n[sign_base + Int32(0)].to(cutlass.Float32)
                v1 *= signs_n[sign_base + Int32(1)].to(cutlass.Float32)
                v2 *= signs_n[sign_base + Int32(2)].to(cutlass.Float32)
                v3 *= signs_n[sign_base + Int32(3)].to(cutlass.Float32)
            work[row, col + Int32(0)] = v0
            work[row, col + Int32(1)] = v1
            work[row, col + Int32(2)] = v2
            work[row, col + Int32(3)] = v3
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        packed: cute.Tensor,
        suh: cute.Tensor,
        svh: cute.Tensor,
        signs_k: cute.Tensor,
        signs_n: cute.Tensor,
        value_table: cute.Tensor,
        tma_store: cute.CopyAtom,
        output: cute.Tensor,
        output_staged_layout: cute.ComposedLayout,
        output_smem_elements: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        lane = tid & Int32(31)
        warp = tid >> Int32(5)

        tiles_per_matrix = Int32(self.hidden_pairs * self.other128)
        expert = Int32(bidx) // tiles_per_matrix
        block = Int32(bidx) - expert * tiles_per_matrix
        if cutlass.const_expr(self.hidden_axis == "k"):
            hidden_pair = block // Int32(self.n128)
            out_nb = block - hidden_pair * Int32(self.n128)
            out_kb = hidden_pair << Int32(1)
        else:
            out_kb = block // Int32(self.hidden_pairs)
            hidden_pair = block - out_kb * Int32(self.hidden_pairs)
            out_nb = hidden_pair << Int32(1)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            work: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, _WORK_ELEMENTS],
                1024,
            ]
            output: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, output_smem_elements],
                1024,
            ]
            ring: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, _RING_WORDS],
                128,
            ]

        storage = smem.allocate(Storage)
        work = storage.work.get_tensor(
            cute.make_layout((_TILE, _TILE), stride=(_TILE, 1))
        )
        output_smem_full = storage.output.get_tensor(
            output_staged_layout.outer,
            swizzle=output_staged_layout.inner,
        )
        output_smem = output_smem_full[None, None, 0]
        ring_base = shared_ptr_to_u32(storage.ring.data_ptr())
        lut_base = get_ptr_as_int64(value_table, Int32(0))
        if tid == Int32(0):
            cpasync.prefetch_descriptor(tma_store)
        cute.arch.sync_threads()

        accum0 = cute.make_rmem_tensor((64,), cutlass.Float32)
        accum1 = cute.make_rmem_tensor((64,), cutlass.Float32)
        for index in cutlass.range_constexpr(64):
            accum0[index] = cutlass.Float32(0.0)
            accum1[index] = cutlass.Float32(0.0)

        if cutlass.const_expr(self.hidden_axis == "k"):
            hidden_base = (out_kb >> Int32(2)) << Int32(2)
            output_group = out_kb & Int32(3)
        else:
            hidden_base = (out_nb >> Int32(2)) << Int32(2)
            output_group = out_nb & Int32(3)

        packed_base = get_ptr_as_int64(packed, Int32(0))
        for source_group in cutlass.range_constexpr(4):
            source_kb = out_kb
            source_nb = out_nb
            if cutlass.const_expr(self.hidden_axis == "k"):
                source_kb = hidden_base + Int32(source_group)
            else:
                source_nb = hidden_base + Int32(source_group)
            if cutlass.const_expr(self.joined_upstream):
                self._decode_upstream_tile(
                    work,
                    output_smem,
                    ring_base,
                    lut_base,
                    packed_base,
                    expert,
                    source_kb,
                    out_nb,
                    tid,
                    lane,
                    warp,
                )
            else:
                self._decode_tile(
                    work,
                    ring_base,
                    lut_base,
                    packed_base,
                    expert,
                    source_kb,
                    source_nb,
                    tid,
                    lane,
                    warp,
                )
            cute.arch.sync_threads()
            self._hadamard_k(
                work,
                suh,
                signs_k,
                expert,
                source_kb,
                out_nb,
                lane,
                warp,
                first=True,
            )
            self._hadamard_n(
                work,
                svh,
                signs_n,
                expert,
                source_nb,
                out_nb,
                lane,
                warp,
                first=True,
            )
            self._hadamard_k(
                work,
                suh,
                signs_k,
                expert,
                source_kb,
                out_nb,
                lane,
                warp,
                first=False,
            )
            self._hadamard_n(
                work,
                svh,
                signs_n,
                expert,
                source_nb,
                out_nb,
                lane,
                warp,
                first=False,
            )
            coefficient0 = _h4_coefficient(source_group, output_group)
            coefficient1 = _h4_coefficient(
                source_group, output_group + Int32(1)
            )
            for index in cutlass.range_constexpr(64):
                element = tid + Int32(index * _THREADS)
                output_row = element // Int32(_TILE)
                output_col = element - output_row * Int32(_TILE)
                value = work[output_col, output_row]
                accum0[index] += coefficient0 * value
                accum1[index] += coefficient1 * value
            cute.arch.sync_threads()

        global_output = cute.local_tile(
            output,
            (_TILE, _TILE),
            (None, None, None),
        )
        shared_partition, global_partition = cpasync.tma_partition(
            tma_store,
            0,
            cute.make_layout(1),
            cute.group_modes(output_smem_full, 0, 2),
            cute.group_modes(global_output, 0, 2),
        )
        store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _THREADS,
            ),
        )
        for output_offset in cutlass.range_constexpr(2):
            target_kb = out_kb
            target_nb = out_nb
            if cutlass.const_expr(self.hidden_axis == "k"):
                target_kb += Int32(output_offset)
            else:
                target_nb += Int32(output_offset)
            for index in cutlass.range_constexpr(64):
                element = tid + Int32(index * _THREADS)
                output_row = element // Int32(_TILE)
                output_col = element - output_row * Int32(_TILE)
                value = (
                    accum0[index]
                    if cutlass.const_expr(output_offset == 0)
                    else accum1[index]
                )
                if cutlass.const_expr(self.hidden_axis == "k"):
                    sign_index = (
                        expert * Int32(self.k)
                        + target_kb * Int32(_TILE)
                        + output_col
                    )
                    value *= signs_k[sign_index].to(cutlass.Float32)
                else:
                    sign_index = (
                        expert * Int32(self.logical_n)
                        + target_nb * Int32(_TILE)
                        + output_row
                    )
                    value *= signs_n[sign_index].to(cutlass.Float32)
                output_smem[output_row, output_col] = cutlass.BFloat16(value)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if warp == Int32(0):
                global_row_block = expert * Int32(self.n128) + target_nb
                cute.copy(
                    tma_store,
                    shared_partition[(None, Int32(0))],
                    global_partition[
                        (None, global_row_block, target_kb, Int32(0))
                    ],
                )
                store_pipeline.producer_commit()
                store_pipeline.producer_acquire()
            cute.arch.sync_threads()


def _compile_materializer(
    materializer: IntermediateHadamardUniformMaterializer,
    *,
    device: torch.device,
):
    packed = torch.empty(
        (
            materializer.source_matrices,
            materializer.k16,
            materializer.source_n16,
            16 * materializer.bits,
        ),
        dtype=torch.int16,
        device=device,
    )
    suh = torch.empty(
        (materializer.scale_k_banks, materializer.experts, materializer.k),
        dtype=torch.float16,
        device=device,
    )
    svh = torch.empty(
        (materializer.experts, materializer.logical_n),
        dtype=torch.float16,
        device=device,
    )
    signs_k = torch.empty_like(suh)
    signs_n = torch.empty_like(svh)
    output = torch.empty(
        (materializer.experts * materializer.logical_n, materializer.k),
        dtype=torch.bfloat16,
        device=device,
    )
    args = (
        _cute_tensor(packed.view(torch.int32), cutlass.Uint32),
        _cute_tensor(suh.reshape(-1), cutlass.Float16),
        _cute_tensor(svh.reshape(-1), cutlass.Float16),
        _cute_tensor(signs_k.reshape(-1), cutlass.Float16),
        _cute_tensor(signs_n.reshape(-1), cutlass.Float16),
        _cute_tensor(lut_e4m3_value_table(device), cutlass.Uint8),
        _cute_tensor(output.view(output.shape[0], output.shape[1], 1), cutlass.BFloat16),
        current_cuda_stream(),
    )
    return b12x_compile(
        materializer,
        *args,
        compile_spec=KernelCompileSpec.from_key(
            "moe.trellis.intermediate_hadamard_materialize",
            1,
            materializer.__cache_key__,
            labels=("experts", "k", "n", "bits", "hidden_axis", "joined_upstream"),
        ),
    )


def prepare_intermediate_hadamard_uniform_materializer_inputs(
    lower,
    upper,
) -> IntermediateHadamardUniformMaterializerInputs:
    """Join two 48-slot intermediate-Hadamard extents into complete experts.

    Uniform checkpoints with an intermediate Hadamard have a transform barrier
    at the midpoint of a 96-slot intermediate axis. ``lower`` and ``upper`` are the two
    prepared EXL3 extents on the same CUDA device. The returned tensors retain
    native uniform payloads and arrange only the small transform tables needed by
    the complete-matrix materializers.
    """

    values = (lower, upper)
    if any(not bool(value.intermediate_hadamard) for value in values):
        raise ValueError("materializer inputs require intermediate-Hadamard extents")
    bits = int(lower.trellis_bits)
    if bits not in (2, 3) or any(int(value.trellis_bits) != bits for value in values):
        raise ValueError("materializer inputs require matching uniform K2 or K3 extents")
    experts = int(lower.num_experts)
    hidden = int(lower.hidden_size)
    half_intermediate = int(lower.intermediate_size)
    if (
        int(upper.num_experts) != experts
        or int(upper.hidden_size) != hidden
        or int(upper.intermediate_size) != half_intermediate
    ):
        raise ValueError("extent geometry does not match")
    if lower.w13.device != upper.w13.device:
        raise ValueError("extents must share one CUDA device")
    if half_intermediate % 32:
        raise ValueError("extent width must close 32-channel slots")

    hidden16 = hidden // 16
    half16 = half_intermediate // 16
    upstream_packed = torch.cat(
        tuple(
            value.w13.view(torch.int16).reshape(
                2,
                experts,
                hidden16,
                half16,
                16 * bits,
            )
            for value in values
        ),
        dim=3,
    ).contiguous()
    down_packed = torch.cat(
        tuple(
            value.w2.view(torch.int16).reshape(
                experts,
                half16,
                hidden16,
                16 * bits,
            )
            for value in values
        ),
        dim=1,
    ).contiguous()

    def expand_rows(value: torch.Tensor, width: int) -> torch.Tensor:
        if value.dtype != torch.float16 or value.dim() != 2:
            raise TypeError("transform tables must be rank-two FP16")
        if int(value.shape[1]) != width or int(value.shape[0]) not in (1, experts):
            raise ValueError("transform table geometry is invalid")
        return value.expand(experts, width).contiguous()

    def rotation_parts(value) -> tuple[torch.Tensor, ...]:
        rotations = value.intermediate_rotations
        if rotations is None or rotations.dtype != torch.float16:
            raise TypeError("extents require FP16 rotation rows")
        if tuple(rotations.shape) != (experts, 6 * half_intermediate):
            raise ValueError("rotation rows have incompatible geometry")
        return tuple(rotations.split(half_intermediate, dim=1))

    lower_parts = rotation_parts(lower)
    upper_parts = rotation_parts(upper)

    def interleave_slots(slot0: torch.Tensor, slot1: torch.Tensor) -> torch.Tensor:
        slot_count = half_intermediate // 32
        return torch.stack(
            (
                slot0.reshape(experts, slot_count, 32),
                slot1.reshape(experts, slot_count, 32),
            ),
            dim=2,
        ).reshape(experts, 2 * half_intermediate)

    upstream_suh = torch.stack(
        (
            expand_rows(lower.gate_suh, hidden),
            expand_rows(upper.gate_suh, hidden),
        ),
        dim=0,
    ).contiguous()
    upstream_svh = torch.cat(
        (
            interleave_slots(lower_parts[0], lower_parts[1]),
            interleave_slots(upper_parts[0], upper_parts[1]),
        ),
        dim=1,
    ).contiguous()
    upstream_signs_n = torch.cat(
        (
            lower_parts[3],
            lower_parts[4],
            upper_parts[3],
            upper_parts[4],
        ),
        dim=1,
    ).contiguous()
    upstream_signs_k = torch.ones_like(upstream_suh)

    down_suh = torch.cat(
        (lower_parts[2], upper_parts[2]),
        dim=1,
    ).contiguous()
    down_svh = expand_rows(lower.down_svh, hidden)
    down_signs_k = torch.cat(
        (lower_parts[5], upper_parts[5]),
        dim=1,
    ).contiguous()
    down_signs_n = torch.ones_like(down_svh)

    return IntermediateHadamardUniformMaterializerInputs(
        upstream_packed=upstream_packed,
        upstream_suh=upstream_suh,
        upstream_svh=upstream_svh,
        upstream_signs_k=upstream_signs_k,
        upstream_signs_n=upstream_signs_n,
        down_packed=down_packed,
        down_suh=down_suh,
        down_svh=down_svh,
        down_signs_k=down_signs_k,
        down_signs_n=down_signs_n,
    )


def compile_intermediate_hadamard_uniform_upstream_materializer(
    *,
    experts: int,
    hidden: int,
    intermediate: int,
    bits: int,
    device: torch.device | str,
):
    """Compile the joined W1/W3 materializer for one expert shard."""

    materializer = IntermediateHadamardUniformMaterializer(
        experts=int(experts),
        k=int(hidden),
        n=int(intermediate),
        bits=int(bits),
        hidden_axis="k",
        joined_upstream=True,
    )
    return _compile_materializer(materializer, device=torch.device(device))


def compile_intermediate_hadamard_uniform_down_materializer(
    *,
    experts: int,
    hidden: int,
    intermediate: int,
    bits: int,
    device: torch.device | str,
):
    """Compile the W2 materializer for one expert shard."""

    materializer = IntermediateHadamardUniformMaterializer(
        experts=int(experts),
        k=int(intermediate),
        n=int(hidden),
        bits=int(bits),
        hidden_axis="n",
    )
    return _compile_materializer(materializer, device=torch.device(device))


def materialize_intermediate_hadamard_uniform(
    compiled,
    *,
    packed: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    signs_k: torch.Tensor,
    signs_n: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Launch a compiled materializer into caller-owned BF16 storage."""

    if packed.dtype != torch.int16 or not packed.is_contiguous():
        raise ValueError("packed payload must be contiguous int16")
    if any(
        value.dtype != torch.float16
        for value in (suh, svh, signs_k, signs_n)
    ):
        raise ValueError("materializer scales and signs must be FP16")
    if output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise ValueError("materializer output must be contiguous BF16")
    device = packed.device
    if any(
        value.device != device
        for value in (suh, svh, signs_k, signs_n, output)
    ):
        raise ValueError("materializer tensors must share one CUDA device")
    compiled(
        _cute_tensor(packed.view(torch.int32), cutlass.Uint32),
        _cute_tensor(suh.reshape(-1), cutlass.Float16),
        _cute_tensor(svh.reshape(-1), cutlass.Float16),
        _cute_tensor(signs_k.reshape(-1), cutlass.Float16),
        _cute_tensor(signs_n.reshape(-1), cutlass.Float16),
        _cute_tensor(lut_e4m3_value_table(device), cutlass.Uint8),
        _cute_tensor(output.view(output.shape[0], output.shape[1], 1), cutlass.BFloat16),
        current_cuda_stream(),
    )


IntermediateHadamardUniformK2MaterializerInputs = IntermediateHadamardUniformMaterializerInputs


class IntermediateHadamardUniformK2Materializer(IntermediateHadamardUniformMaterializer):
    """Compatibility constructor for the uniform-K2 materializer."""

    def __init__(
        self,
        *,
        experts: int,
        k: int,
        n: int,
        hidden_axis: str,
        joined_upstream: bool = False,
    ) -> None:
        super().__init__(
            experts=experts,
            k=k,
            n=n,
            bits=2,
            hidden_axis=hidden_axis,
            joined_upstream=joined_upstream,
        )


def prepare_intermediate_hadamard_uniform_k2_materializer_inputs(
    lower,
    upper,
) -> IntermediateHadamardUniformK2MaterializerInputs:
    """Join two uniform-K2 intermediate-Hadamard extents."""

    if int(lower.trellis_bits) != 2 or int(upper.trellis_bits) != 2:
        raise ValueError("K2 materializer inputs require uniform K2 extents")
    return prepare_intermediate_hadamard_uniform_materializer_inputs(lower, upper)


def compile_intermediate_hadamard_uniform_k2_upstream_materializer(
    *,
    experts: int,
    hidden: int,
    intermediate: int,
    device: torch.device | str,
):
    """Compile the uniform-K2 joined W1/W3 materializer."""

    return compile_intermediate_hadamard_uniform_upstream_materializer(
        experts=experts,
        hidden=hidden,
        intermediate=intermediate,
        bits=2,
        device=device,
    )


def compile_intermediate_hadamard_uniform_k2_down_materializer(
    *,
    experts: int,
    hidden: int,
    intermediate: int,
    device: torch.device | str,
):
    """Compile the uniform-K2 W2 materializer."""

    return compile_intermediate_hadamard_uniform_down_materializer(
        experts=experts,
        hidden=hidden,
        intermediate=intermediate,
        bits=2,
        device=device,
    )


def materialize_intermediate_hadamard_uniform_k2(
    compiled,
    **kwargs,
) -> None:
    """Launch a compiled uniform-K2 materializer."""

    materialize_intermediate_hadamard_uniform(compiled, **kwargs)


__all__ = [
    "IntermediateHadamardUniformK2MaterializerInputs",
    "IntermediateHadamardUniformK2Materializer",
    "IntermediateHadamardUniformMaterializerInputs",
    "IntermediateHadamardUniformMaterializer",
    "compile_intermediate_hadamard_uniform_down_materializer",
    "compile_intermediate_hadamard_uniform_k2_down_materializer",
    "compile_intermediate_hadamard_uniform_k2_upstream_materializer",
    "compile_intermediate_hadamard_uniform_upstream_materializer",
    "materialize_intermediate_hadamard_uniform",
    "materialize_intermediate_hadamard_uniform_k2",
    "prepare_intermediate_hadamard_uniform_materializer_inputs",
    "prepare_intermediate_hadamard_uniform_k2_materializer_inputs",
]
