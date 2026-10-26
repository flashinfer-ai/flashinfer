"""Standalone BF16→FP4 quantization with a TMA packed-output store."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cuda.bindings.driver as cuda
from cutlass.cutlass_dsl import Int32, Uint8, Uint32, Uint64, Float32, T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync


@dsl_user_op
def fabs_f32(a, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmax_f32(a, b, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmin_f32(a, b, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def rcp_approx_ftz(a, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def rcp_rn_f32(a, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "{\n.reg .f32 one;\nmov.f32 one, 0f3F800000;\ndiv.rn.f32 $0, one, $1;\n}",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ld_global_bf16x2(tensor, coord, *, loc=None, ip=None):
    """Load two adjacent BF16 values as one raw 32-bit register."""
    elem_ptr = tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)
    address = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [address],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_bf16x2_to_f32x2(packed, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; "
        "cvt.f32.bf16 $0, lo; cvt.f32.bf16 $1, hi; }",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_bf16x2_to_f32x2_scaled(packed, scale, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            Uint32(packed).ir_value(loc=loc, ip=ip),
            Float32(scale).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b16 lo, hi; .reg .f32 flo, fhi; "
        "mov.b32 {lo, hi}, $2; cvt.f32.bf16 flo, lo; "
        "cvt.f32.bf16 fhi, hi; mul.rn.f32 $0, flo, $3; "
        "mul.rn.f32 $1, fhi, $3; }",
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_f32_to_e4m3(a, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 fp8_pair;\n.reg .f32 zero;\nmov.f32 zero, 0f00000000;\ncvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;\ncvt.u32.u16 $0, fp8_pair;\n}",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fp8_e4m3_to_f32_and_effective_rcp(
    fp8_val, global_scale_recip, *, loc=None, ip=None
):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(fp8_val).ir_value(loc=loc, ip=ip),
                Float32(global_scale_recip).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .pred p_zero, p_subnormal;\n.reg .u32 exp_u, mant_u;\n.reg .s32 exp_s;\n.reg .f32 exp_f, mant_f, fp8_float, one, denom, result;\n"
            "setp.eq.u32 p_zero, $1, 0;\nand.b32 mant_u, $1, 7;\nshr.b32 exp_u, $1, 3;\nand.b32 exp_u, exp_u, 15;\n"
            "setp.eq.u32 p_subnormal, exp_u, 0;\nsub.s32 exp_s, exp_u, 7;\nselp.s32 exp_s, -6, exp_s, p_subnormal;\n"
            "cvt.rn.f32.s32 exp_f, exp_s;\nex2.approx.f32 exp_f, exp_f;\n"
            "cvt.rn.f32.u32 mant_f, mant_u;\nselp.f32 fp8_float, 0f00000000, 0f3F800000, p_subnormal;\n"
            "fma.rn.f32 mant_f, mant_f, 0f3E000000, fp8_float;\n"
            "mul.f32 fp8_float, exp_f, mant_f;\nmov.f32 one, 0f3F800000;\n"
            "mul.rn.f32 denom, fp8_float, $2;\ndiv.rn.f32 result, one, denom;\n"
            "selp.f32 $0, 0f00000000, result, p_zero;\n}",
            "=f,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_e2m1x8_f32(v0, v1, v2, v3, v4, v5, v6, v7, *, loc=None, ip=None):
    args = [
        Float32(v).ir_value(loc=loc, ip=ip) for v in [v0, v1, v2, v3, v4, v5, v6, v7]
    ]
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            args,
            "{\n.reg .b8 byte0, byte1, byte2, byte3;\n"
            "cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;\ncvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;\n"
            "cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;\ncvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;\n"
            "mov.b32 $0, {byte0, byte1, byte2, byte3};\n}",
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def canonicalize_fp4_zero_signs(packed: Uint32) -> Uint32:
    """Clear the sign bit on FP4 values whose magnitude rounded to zero."""
    magnitude = packed & Uint32(0x77777777)
    nonzero = magnitude | (magnitude >> Uint32(1)) | (magnitude >> Uint32(2))
    nonzero = nonzero & Uint32(0x11111111)
    return packed & (Uint32(0x77777777) | (nonzero << Uint32(3)))


@cute.jit
def quantize_scale_fp4_fast(max_abs, global_scale_val, global_scale_recip):
    scale_u32 = Uint32(0)
    scale_byte = Uint8(0)
    effective_recip = Float32(0.0)
    if global_scale_val != Float32(0.0):
        fp4_max_rcp = rcp_approx_ftz(Float32(6.0))
        scale_float = global_scale_val * (max_abs * fp4_max_rcp)
        scale_float = fmin_f32(scale_float, Float32(448.0))
        scale_u32 = cvt_f32_to_e4m3(scale_float)
        scale_byte = Uint8(scale_u32 & Uint32(0xFF))
        effective_recip = fp8_e4m3_to_f32_and_effective_rcp(
            scale_u32,
            global_scale_recip,
        )
    return effective_recip, scale_byte


class Bf16ToFp4TmaKernel:
    def __init__(self, liveness_strategy: str):
        if liveness_strategy not in {"retain", "packed"}:
            raise ValueError(
                f"unsupported BF16->FP4 liveness strategy: {liveness_strategy}"
            )
        self.liveness_strategy = liveness_strategy
        self.tile_shape_mnk = (128, 128, 128)
        self.threads_per_cta = 128
        self.num_mma_warps = 4
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * 32,
        )

    @cute.jit
    def __call__(
        self,
        bf16_input: cute.Tensor,
        global_scale: cute.Tensor,
        packed_a: cute.Tensor,
        scale_storage: cute.Tensor,
        mac: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        ab = cutlass.Float4E2M1FN
        fp4_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, ab, 128), ab
        )
        fp4_staged = cute.tile_to_shape(fp4_atom, (128, 128, 1), order=(0, 1, 2))
        fp4_smem1 = cute.slice_(fp4_staged, (None, None, 0))
        tile_mk = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        tma_store_a, gOA = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), packed_a, fp4_smem1, tile_mk
        )
        self.kernel(
            bf16_input,
            tma_store_a,
            gOA,
            global_scale,
            scale_storage,
            fp4_staged,
            cute.cosize(fp4_staged),
        ).launch(
            grid=(mac, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        tma_store_a: cute.CopyAtom,
        mOA: cute.Tensor,
        global_scale: cute.Tensor,
        scale_storage: cute.Tensor,
        fp4_smem: cute.ComposedLayout,
        fp4_cs: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        warp_idx = tidx // Int32(32)

        M = Int32(mInput.shape[0])
        K = Int32(mInput.shape[1])
        k_tiles = K // Int32(128)
        total_tiles = (M // Int32(128)) * k_tiles

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class S:
            sFP4: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float4E2M1FN, fp4_cs], 1024
            ]
            # Four scale bytes are written directly to global memory so this
            # rounded reciprocal slot stays inside the existing 9 KiB bucket.
            sScale: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 1020], 1024]
            sGlobalScaleRecip: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 1], 4
            ]

        st = smem.allocate(S)
        sFP4 = st.sFP4.get_tensor(fp4_smem.outer, swizzle=fp4_smem.inner)
        sScale = st.sScale.get_tensor(cute.make_layout(1020))
        sGlobalScaleRecip = st.sGlobalScaleRecip.get_tensor(cute.make_layout(1))
        sA_u8 = cute.recast_tensor(sFP4[None, None, 0], cutlass.Uint8)

        cta_layout = cute.make_layout(1)
        tile_mk = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        gOA = cute.local_tile(mOA, tile_mk, (None, None, None))
        bSsA, bSgA = cpasync.tma_partition(
            tma_store_a,
            0,
            cta_layout,
            cute.group_modes(sFP4, 0, 2),
            cute.group_modes(gOA, 0, 2),
        )

        store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_warps * 32
            ),
        )

        if tidx == Int32(0):
            cta_global_scale = global_scale[Int32(0)].to(cutlass.Float32)
            global_scale_recip = cutlass.Float32(0.0)
            if cta_global_scale != cutlass.Float32(0.0):
                global_scale_recip = rcp_rn_f32(cta_global_scale)
            sGlobalScaleRecip[Int32(0)] = global_scale_recip
            cpasync.prefetch_descriptor(tma_store_a)
        cute.arch.sync_threads()

        tile_idx = Int32(bidx)

        while tile_idx < total_tiles:
            mt = tile_idx // k_tiles
            kt = tile_idx % k_tiles
            scale_tile_base = mt * (K * Int32(8)) + kt * Int32(1024)

            blk = Int32(tidx)
            while blk < Int32(128 * 8):
                row = blk // Int32(8)
                sf_col = blk % Int32(8)
                col0 = sf_col * Int32(16)
                bmax = cutlass.Float32(0.0)
                packed_bf16x2 = Uint32(0)
                if cutlass.const_expr(self.liveness_strategy == "packed"):
                    # Keep the final adjacent pair in its lossless packed
                    # representation across the exact scale division.  This
                    # shortens two FP32 live ranges without adding loads.
                    for e in cutlass.range_constexpr(14):
                        coord = (
                            mt * Int32(128) + row,
                            kt * Int32(128) + col0 + Int32(e),
                        )
                        v = cutlass.Float32(mInput[coord])
                        bmax = fmax_f32(bmax, fabs_f32(v))
                    packed_bf16x2 = ld_global_bf16x2(
                        mInput,
                        (
                            mt * Int32(128) + row,
                            kt * Int32(128) + col0 + Int32(14),
                        ),
                    )
                    pair0, pair1 = cvt_bf16x2_to_f32x2(packed_bf16x2)
                    bmax = fmax_f32(bmax, fabs_f32(pair0))
                    bmax = fmax_f32(bmax, fabs_f32(pair1))
                else:
                    # The 4.6 compiler already improves M == 128 codegen, so
                    # retain its original instruction schedule unchanged.
                    for e in cutlass.range_constexpr(16):
                        coord = (
                            mt * Int32(128) + row,
                            kt * Int32(128) + col0 + Int32(e),
                        )
                        v = cutlass.Float32(mInput[coord])
                        bmax = fmax_f32(bmax, fabs_f32(v))
                block_global_scale = global_scale[Int32(0)].to(cutlass.Float32)
                global_scale_recip = sGlobalScaleRecip[Int32(0)]
                effective_recip, sbyte = quantize_scale_fp4_fast(
                    bmax,
                    block_global_scale,
                    global_scale_recip,
                )
                p64 = Uint64(0)
                if effective_recip != Float32(0.0):
                    q = cute.make_rmem_tensor((8,), Float32)
                    for e in cutlass.range_constexpr(8):
                        q[e] = (
                            cutlass.Float32(
                                mInput[
                                    mt * Int32(128) + row,
                                    kt * Int32(128) + col0 + Int32(e),
                                ]
                            )
                            * effective_recip
                        )
                    packed_lo = canonicalize_fp4_zero_signs(
                        cvt_e2m1x8_f32(
                            q[0],
                            q[1],
                            q[2],
                            q[3],
                            q[4],
                            q[5],
                            q[6],
                            q[7],
                        )
                    )
                    if cutlass.const_expr(self.liveness_strategy == "packed"):
                        for e in cutlass.range_constexpr(6):
                            q[e] = (
                                cutlass.Float32(
                                    mInput[
                                        mt * Int32(128) + row,
                                        kt * Int32(128) + col0 + Int32(8 + e),
                                    ]
                                )
                                * effective_recip
                            )
                        q[6], q[7] = cvt_bf16x2_to_f32x2_scaled(
                            packed_bf16x2,
                            effective_recip,
                        )
                    else:
                        for e in cutlass.range_constexpr(8):
                            q[e] = (
                                cutlass.Float32(
                                    mInput[
                                        mt * Int32(128) + row,
                                        kt * Int32(128) + col0 + Int32(8 + e),
                                    ]
                                )
                                * effective_recip
                            )
                    packed_hi = canonicalize_fp4_zero_signs(
                        cvt_e2m1x8_f32(
                            q[0],
                            q[1],
                            q[2],
                            q[3],
                            q[4],
                            q[5],
                            q[6],
                            q[7],
                        )
                    )
                    p64 = (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)
                om = row % Int32(32)
                im = row // Int32(32)
                ik = sf_col % Int32(4)
                ktile = sf_col // Int32(4)
                sf_off = ktile * Int32(512) + om * Int32(16) + im * Int32(4) + ik
                if sf_off < Int32(1020):
                    sScale[sf_off] = sbyte
                else:
                    scale_storage[scale_tile_base + sf_off] = sbyte
                pb = sf_col << Int32(3)
                dpc = row & Int32(63)
                xor = ((dpc >> Int32(1)) & Int32(0x3)) << Int32(4)
                rhi = row >> Int32(6)
                for bi in cutlass.range_constexpr(8):
                    spc = pb + Int32(bi)
                    dr = ((spc ^ xor) << Int32(1)) + rhi
                    flat = dr * Int32(64) + dpc
                    sA_u8[flat] = Uint8((p64 >> Uint64(bi * 8)) & Uint64(0xFF))
                blk += Int32(self.num_mma_warps * 32)

            cute.arch.fence_proxy("async.shared", space="cta")
            self.epilog_sync_barrier.arrive_and_wait()
            if warp_idx == Int32(0):
                cute.copy(
                    tma_store_a, bSsA[(None, Int32(0))], bSgA[(None, mt, kt, Int32(0))]
                )
                store_pipeline.producer_commit()
                store_pipeline.producer_acquire()
            scale_copy = Int32(tidx)
            while scale_copy < Int32(1020):
                scale_storage[scale_tile_base + scale_copy] = sScale[scale_copy]
                scale_copy += Int32(self.num_mma_warps * 32)
            self.epilog_sync_barrier.arrive_and_wait()

            tile_idx += Int32(gdim)
