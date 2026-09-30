"""Standalone BF16→MX-FP6 TMA quantize kernel (32-elt UE8M0 blocks, 4-in-3-byte pack)."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cuda.bindings.driver as cuda
from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu import warp as _nvgpu_warp

from b12x._lib.intrinsics import (
    cvt_f32_to_bf16_bits,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    shared_ptr_to_u32,
    st_global_u8,
    st_global_u64,
    st_shared_u8,
    u32_as_f32,
)
from b12x._lib.fp6 import (
    quantize_block_fp6_e2m3_bytes,
    quantize_block_fp6_e2m3_fast,
    quantize_block_fp6_e3m2_bytes,
    quantize_block_fp6_e3m2_fast,
    quantize_block_fp8_e4m3_bytes,
)
from b12x._lib.utils import make_ptr, sm120_make_smem_layout_sfa

# The pinned cutlass-dsl ships the native SM120 MXF8 block-scaled warp op.
_MmaMXF8Op = _nvgpu_warp.MmaMXF8Op

_NUM_STAGES = 1
_SF_VEC_SIZE = 32
_FP6_BLOCK_ELEMS = 32


# These vectorized global-store helpers are local because they only depend on
# _lib intrinsics and are also useful outside the fused-MoE implementation.
@cute.jit
def moe_mxfp6_store_packed_global(
    storage: cute.Tensor,
    byte_offset: Int64,
    lo: Uint64,
    mid: Uint64,
    hi: Uint64,
) -> None:
    """Store 24 packed MX-FP6 bytes (three u64 lanes) to global uint8 storage."""
    base = get_ptr_as_int64(storage, byte_offset)
    st_global_u64(base, lo)
    st_global_u64(base + Int64(8), mid)
    st_global_u64(base + Int64(16), hi)


@cute.jit
def moe_mxfp6_store_bytes_u64_global(
    storage: cute.Tensor,
    byte_offset: Int64,
    q0: Uint64,
    q1: Uint64,
    q2: Uint64,
    q3: Uint64,
) -> None:
    """Store 32 FP6 byte-containers (four u64 lanes) to global uint8 storage.

    Byte-container counterpart of :func:`moe_mxfp6_store_packed_global`: 32
    bytes (one code each) instead of 24 packed bytes, same vectorized
    ``st.global.u64`` path. ``byte_offset`` must stay 8-byte aligned (the
    bytes-mode block stride of 32 guarantees it).
    """
    base = get_ptr_as_int64(storage, byte_offset)
    st_global_u64(base, q0)
    st_global_u64(base + Int64(8), q1)
    st_global_u64(base + Int64(16), q2)
    st_global_u64(base + Int64(24), q3)


class TestKernel:
    def __init__(self, fmt: str = "e3m2", emit: str = "packed", per_row: bool = False):
        assert fmt in ("e3m2", "e2m3", "e4m3"), f"unsupported act/weight fmt: {fmt}"
        # "packed": 3:4-packed wire format (M, 3K/4) - weights/storage (FP6 only).
        # "bytes":  one code per byte (M, K) - the mxf8f6f4 GEMM operand layout,
        #           so the per-call activation expansion is skipped entirely.
        # E4M3 is activation-only (W6A8) and only supports emit="bytes".
        assert emit in ("packed", "bytes"), f"unsupported FP6 emit: {emit}"
        if fmt == "e4m3" and emit != "bytes":
            raise ValueError("e4m3 activations require emit='bytes' (no 3:4 pack)")
        # per_row: large-M half of the fused per-row global-scale recipe (see
        # fp6_row_gs). ``global_scale`` becomes the (M,) f32 per-row gs from
        # RowGsKernel; each element is pre-scaled bf16(f32(x) * gs_row) in
        # registers (cvt.rn.bf16.f32 — the exact host-chain rounding) and
        # quantized with a UNIT gs, bit-identical to the host per-row path.
        # Activations only, so bytes-emit only.
        if per_row and emit != "bytes":
            raise ValueError("per_row quantization requires emit='bytes'")
        self.fmt = fmt
        self.emit = emit
        self.per_row = per_row
        self.tile_shape_mnk = (128, 128, 128)
        self.threads_per_cta = 160
        self.num_mma_warps = 4
        self.tma_warp = 4
        self.num_stages = _NUM_STAGES
        self.load_register_requirement = 40

    @cute.jit
    def __call__(
        self,
        bf16_input: cute.Tensor,
        global_scale: cute.Tensor,
        packed_a: cute.Tensor,
        sfa_ptr: cute.Pointer,
        mac: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        bf = cutlass.BFloat16
        sf = cutlass.Float8E8M0FNU
        # FP6 output (3:4-packed, 96 bytes per 128-code row) is written straight to
        # global memory with direct vectorized u64 stores from registers -- never via
        # TMA. A cp.async.bulk.tensor S2G store of the 96-byte (non-power-of-2) packed
        # row is rejected as an illegal instruction on this sm_120/ptxas/driver, so the
        # packed output bypasses smem/TMA entirely. Only the SFA scales use a TMA store.
        bf16_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.COL_MAJOR, bf, 128),
            bf,
        )
        bf16_staged = cute.tile_to_shape(bf16_atom, (128, 128, self.num_stages), order=(0, 1, 2))
        mma_op = _MmaMXF8Op(
            cutlass.Float8E4M3FN,
            cutlass.Float32,
            sf,
        )
        perm = sm120_utils.get_permutation_mnk(self.tile_shape_mnk, _SF_VEC_SIZE, True)
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((4, 2, 1)), permutation_mnk=perm)
        sfa_staged = sm120_make_smem_layout_sfa(
            tiled_mma, self.tile_shape_mnk, _SF_VEC_SIZE, 1
        )
        sfa_logical_shape = (
            packed_a.shape[0],
            self.tile_shape_mnk[2],
            packed_a.shape[2],
        )
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            sfa_logical_shape, _SF_VEC_SIZE
        )
        # SFA scales are written directly to global memory as bytes (no TMA store):
        # the FP6 SFA TMA store (sf_vec=32, Int16 internal type) miscompiles to an
        # illegal instruction on this sm_120/ptxas build, so scale bytes are stored
        # element-wise into the swizzled global layout, exactly like the MoE path.
        sfa_flat = cute.make_tensor(sfa_ptr, cute.make_layout(cute.cosize(sfa_layout)))
        bf16_smem1 = cute.slice_(bf16_staged, (None, None, 0))
        tma_load, gInput = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            bf16_input,
            bf16_smem1,
            (128, 128),
            num_multicast=1,
        )
        self.kernel(
            tma_load,
            gInput,
            bf16_input,
            packed_a,
            sfa_flat,
            global_scale,
            bf16_staged,
            cute.cosize(bf16_staged),
        ).launch(
            grid=(mac, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_load: cute.CopyAtom,
        mInput: cute.Tensor,
        mRaw: cute.Tensor,
        mOA: cute.Tensor,
        mSFA: cute.Tensor,
        global_scale: cute.Tensor,
        bf16_smem: cute.ComposedLayout,
        bf16_cs: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        warp_idx = tidx // Int32(32)

        k_tiles = Int32(mInput.shape[1]) // Int32(128)
        total_tiles = (Int32(mInput.shape[0]) // Int32(128)) * k_tiles

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class S:
            pmem: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sBF16: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, bf16_cs], 1024]
            # Packed-byte smem region (128*96). Required: without it the cutlass-dsl
            # backend miscompiles the quantize/load path to an illegal instruction on
            # this sm_120/ptxas build (codegen sensitive to smem layout). The packed
            # output itself is written directly to global, so this region is otherwise
            # unused, but its presence keeps codegen valid.
            sFP6PAD: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 12288], 1024]

        st = smem.allocate(S)
        sBF16 = st.sBF16.get_tensor(bf16_smem.outer, swizzle=bf16_smem.inner)
        if cutlass.const_expr(self.per_row):
            # Per-row mode: values are pre-scaled per row below, so the
            # quantizer's own global scale is exactly 1.0 (same collapse as
            # the host chain, which quantizes the pre-scaled input with a
            # unit gs — see fp6_dense_weights).
            gs_value = cutlass.Float32(1.0)
        else:
            gs_value = global_scale[Int32(0)].to(cutlass.Float32)

        cta_layout = cute.make_layout(1)
        gI = cute.local_tile(mInput, (128, 128), (None, None))
        tLsI, tLgI = cpasync.tma_partition(
            tma_load,
            0,
            cta_layout,
            cute.group_modes(sBF16, 0, 2),
            cute.group_modes(gI, 0, 2),
        )

        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_warps
            ),
            tx_count=128 * 128 * cutlass.BFloat16.width // 8,
            barrier_storage=st.pmem.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if tidx == Int32(0):
            cpasync.prefetch_descriptor(tma_load)
        cute.arch.sync_threads()

        tile_idx = Int32(bidx)
        blocks_per_row = Int32(128 // _FP6_BLOCK_ELEMS)

        if warp_idx < Int32(self.num_mma_warps):
            cs = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            while tile_idx < total_tiles:
                mt = tile_idx // k_tiles
                kt = tile_idx % k_tiles
                load_pipeline.consumer_wait(cs)
                stage = cs.index

                blk = Int32(tidx)
                while blk < Int32(128 * blocks_per_row):
                    row = blk // blocks_per_row
                    sf_block = blk % blocks_per_row
                    col0 = sf_block * Int32(_FP6_BLOCK_ELEMS)
                    grow = mt * Int32(128) + row
                    gcol0 = kt * Int32(128) + col0
                    # Read the 32 bf16 inputs straight from global memory. Dynamic column
                    # indexing into the swizzled bf16 smem tensor collapses to a single
                    # value per row on this build (stride lost), so the staged TMA copy is
                    # kept only to preserve the validated smem layout/codegen while the
                    # quantizer sources its data directly from the resident global input.
                    vals = cute.make_rmem_tensor((_FP6_BLOCK_ELEMS,), cutlass.Float32)
                    bmax = cutlass.Float32(0.0)
                    if cutlass.const_expr(self.per_row):
                        # Per-row pre-scale with the SAME rounding chain as the
                        # host: f32 multiply (RN), round through bf16 (RN),
                        # widen back (exact) — identical to the small-M
                        # per-row branch. global_scale is the (M,) f32 row
                        # scales from RowGsKernel.
                        gs_row = cutlass.Float32(global_scale[grow])
                        for e in cutlass.range_constexpr(_FP6_BLOCK_ELEMS):
                            v_raw = cutlass.Float32(mRaw[grow, gcol0 + Int32(e)])
                            pre_bits = cvt_f32_to_bf16_bits(v_raw * gs_row)
                            v = u32_as_f32(pre_bits << Uint32(16))
                            vals[e] = v
                            bmax = fmax_f32(bmax, fabs_f32(v))
                    else:
                        for e in cutlass.range_constexpr(_FP6_BLOCK_ELEMS):
                            v = cutlass.Float32(mRaw[grow, gcol0 + Int32(e)])
                            vals[e] = v
                            bmax = fmax_f32(bmax, fabs_f32(v))
                    if cutlass.const_expr(self.emit == "bytes"):
                        # Byte-container output (M, K): one code per byte, the
                        # mxf8f6f4 GEMM operand layout. Same quantization math as
                        # the packed path, four u64 lanes instead of three; row
                        # stride is the full byte width 128*k_tiles (= K), each
                        # 128-wide K-tile contributing 128 bytes. Offsets stay
                        # 8-byte aligned (128 and 32 are multiples of 8).
                        if cutlass.const_expr(self.fmt == "e4m3"):
                            q0, q1, q2, q3, sbyte = quantize_block_fp8_e4m3_bytes(
                                vals, bmax, gs_value
                            )
                        elif cutlass.const_expr(self.fmt == "e2m3"):
                            q0, q1, q2, q3, sbyte = quantize_block_fp6_e2m3_bytes(
                                vals, bmax, gs_value
                            )
                        else:
                            q0, q1, q2, q3, sbyte = quantize_block_fp6_e3m2_bytes(
                                vals, bmax, gs_value
                            )
                        # Row ID * stride in Int64 (coding guideline): the
                        # bytes-mode offset is ~M*K and must not wrap.
                        byte_offset = (
                            Int64(grow) * Int64(128) * Int64(k_tiles)
                            + Int64(kt * Int32(128))
                            + Int64(sf_block * Int32(32))
                        )
                        moe_mxfp6_store_bytes_u64_global(
                            mOA, byte_offset, q0, q1, q2, q3
                        )
                    else:
                        if cutlass.const_expr(self.fmt == "e2m3"):
                            lo, mid, hi, sbyte = quantize_block_fp6_e2m3_fast(
                                vals, bmax, gs_value
                            )
                        else:
                            lo, mid, hi, sbyte = quantize_block_fp6_e3m2_fast(
                                vals, bmax, gs_value
                            )
                        # Write the 24 packed bytes (lo|mid|hi) straight to global via the
                        # proven inline-asm st.global.u64 helper. The packed output is a
                        # row-major (M, 3K/4) buffer, so the per-row stride is the FULL
                        # packed width 96*k_tiles (= 3K/4), NOT 96: each 128-wide K-tile
                        # contributes 96 bytes. Using grow*96 only works for k_tiles==1
                        # (K<=128); for K>128 it collapses every row into the first tile's
                        # 96 bytes, leaving most rows unwritten (zero) and scrambling the
                        # rest. The flat byte offset stays 8-byte aligned (96 and 24 are
                        # multiples of 8), matching the three u64 lanes.
                        byte_offset = (
                            Int64(grow) * Int64(96) * Int64(k_tiles)
                            + Int64(kt * Int32(96))
                            + Int64(sf_block * Int32(24))
                        )
                        moe_mxfp6_store_packed_global(mOA, byte_offset, lo, mid, hi)
                    # Scale byte -> global, directly into the swizzled blockscaled SFA
                    # layout (same offset math + flat uint8 buffer as the MoE path):
                    #   tile base = (mt*k_tiles + kt) * 512
                    #   within-tile = (row%32)*16 + (row//32)*4 + sf_block
                    outer_m_idx = row % Int32(32)
                    inner_m_idx = row // Int32(32)
                    sf_offset = (
                        Int64(mt) * Int64(k_tiles) * Int64(512)
                        + Int64(kt * Int32(512))
                        + Int64(outer_m_idx * Int32(16))
                        + Int64(inner_m_idx * Int32(4))
                        + Int64(sf_block)
                    )
                    st_global_u8(get_ptr_as_int64(mSFA, sf_offset), sbyte)
                    blk += Int32(self.num_mma_warps * 32)

                load_pipeline.consumer_release(cs)
                cs.advance()

                tile_idx += Int32(gdim)

        elif warp_idx == Int32(self.tma_warp):
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            ps = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            while tile_idx < total_tiles:
                mt = tile_idx // k_tiles
                kt = tile_idx % k_tiles
                load_pipeline.producer_acquire(ps)
                cute.copy(
                    tma_load,
                    tLgI[(None, mt, kt)],
                    tLsI[(None, ps.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(ps),
                )
                load_pipeline.producer_commit(ps)
                ps.advance()
                tile_idx += Int32(gdim)
            load_pipeline.producer_tail(ps)
