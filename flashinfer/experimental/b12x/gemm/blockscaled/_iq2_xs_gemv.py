"""Direct descriptor decoding and SIMT contraction for small-row IQ2_XS GEMM."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.intrinsics import get_ptr_as_int64, warp_reduce, iq2_xxs_descriptor_pair, q8_0_pair_to_bf16x2
from b12x._lib.program_cache import program_cache
from b12x._lib.utils import cuda_stream_from_int_or_current, current_cuda_stream, make_ptr
from b12x.gemm.bf16_gemv._kernel import _dot_bf16x8, _fadd, _flat


@dsl_user_op
def _decode_eight(descriptor, base, subscale, lut_addr, *, loc=None, ip=None):
    pairs = []
    for i in range(4):
        pairs.append(f"""
            shl.b32 lo, w{i}, 16;
            and.b32 hi, w{i}, 0xffff0000;
            mul.f32 lo, lo, scale;
            mul.f32 hi, hi, scale;
            cvt.rn.satfinite.bf16x2.f32 ${i}, hi, lo;
            shr.u32 pair_signs, signs, {i * 2};
            mul.lo.u32 pair_signs, pair_signs, 0x40008000;
            lop3.b32 ${i}, ${i}, pair_signs, 0x80008000, 0x78;
        """)
    asm = """{
        .reg .b16 dh;
        .reg .u32 grid, signs, parity, w<4>, pair_signs;
        .reg .u64 address;
        .reg .f32 scale, nibble, lo, hi;
        cvt.u16.u32 dh, $5;
        cvt.f32.f16 scale, dh;
        cvt.rn.f32.u32 nibble, $6;
        add.f32 nibble, nibble, 0f3f000000;
        mul.f32 scale, scale, nibble;
        mul.f32 scale, scale, 0f3e800000;
        shr.u32 signs, $4, 9;
        popc.b32 parity, signs;
        mad.lo.u32 signs, parity, 128, signs;
        and.b32 grid, $4, 0x1ff;
        mad.wide.u32 address, grid, 16, $7;
        ld.global.nc.v4.u32 {w0, w1, w2, w3}, [address];
    """ + "\n".join(pairs) + "\n}"
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Uint32(descriptor).ir_value(loc=loc, ip=ip), Uint32(base).ir_value(loc=loc, ip=ip),
         Uint32(subscale).ir_value(loc=loc, ip=ip), Int64(lut_addr).ir_value(loc=loc, ip=ip)],
        asm, "=r,=r,=r,=r,r,r,r,l", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


class IQ2XSGemv:
    def __init__(self, n, k, tile_m, codec="iq2_xs"):
        self.codec = codec
        self.q8 = codec == "q8_0"
        self.iq2_xxs = codec == "iq2_xxs"
        self.n, self.k, self.tile_m = n, k, tile_m
        self.descriptor_tile_n = 128 if n % 128 == 0 else 8

    @cute.jit
    def __call__(self, source: cute.Pointer, descriptors: cute.Pointer,
                 metadata: cute.Pointer, lut: cute.Pointer, output: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(_flat(source), _flat(descriptors), _flat(metadata), _flat(lut),
                    _flat(output), rows).launch(
            grid=((self.n + 3) // 4, (rows + self.tile_m - 1) // self.tile_m, 1),
            block=(128, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, source: cute.Tensor, descriptors: cute.Tensor,
               metadata: cute.Tensor, lut: cute.Tensor, output: cute.Tensor, rows: Int32):
        thread, _, _ = cute.arch.thread_idx()
        tile_n, tile_m, _ = cute.arch.block_idx()
        lane = Int32(thread) % 32
        column = Int64(tile_n) * 4 + Int64(Int32(thread) // 32)
        first_row = Int64(tile_m) * self.tile_m
        accum = cute.make_rmem_tensor((self.tile_m,), Float32)
        for r in cutlass.range_constexpr(self.tile_m):
            accum[r] = Float32(0)
        if column < self.n:
            group = lane
            while group < self.k // 8:
                if cutlass.const_expr(self.q8):
                    meta = (Int64(group // 4) * self.n + column) * 2
                    base = Uint32(metadata[meta]) | (Uint32(metadata[meta + 1]) << 8)
                    offset = ((column // self.descriptor_tile_n * (self.k // 32) + Int64(group // 4)) * (self.descriptor_tile_n * 16)
                              + column % self.descriptor_tile_n * 16 + Int64(group % 4) * 4)
                    w0 = q8_0_pair_to_bf16x2(Uint32(descriptors[offset]), base)
                    w1 = q8_0_pair_to_bf16x2(Uint32(descriptors[offset + 1]), base)
                    w2 = q8_0_pair_to_bf16x2(Uint32(descriptors[offset + 2]), base)
                    w3 = q8_0_pair_to_bf16x2(Uint32(descriptors[offset + 3]), base)
                else:
                    if cutlass.const_expr(self.iq2_xxs):
                        meta = (Int64(group // 32) * self.n + column) * 2
                        base = Uint32(metadata[meta]) | (Uint32(metadata[meta + 1]) << 8)
                        record = ((column // self.descriptor_tile_n * (self.k // 256) + Int64(group // 32)) * (self.descriptor_tile_n * 32)
                                  + column % self.descriptor_tile_n * 32 + Int64(group % 32 // 4) * 4)
                        grids = Uint32(descriptors[record]) | (Uint32(descriptors[record + 1]) << 16)
                        signs_scale = Uint32(descriptors[record + 2]) | (Uint32(descriptors[record + 3]) << 16)
                        pair = iq2_xxs_descriptor_pair(grids, signs_scale, group % 4 // 2)
                        descriptor = (pair >> (group % 2 * 16)) & Uint32(65535)
                        subscale = signs_scale >> 28
                    else:
                        meta = (column // 128 * (self.k // 256) + Int64(group // 32)) * 1280
                        base = (Uint32(metadata[meta + column % 128 * 2])
                                | Uint32(metadata[meta + column % 128 * 2 + 1]) << 8)
                        scale_pair = Uint32(metadata[meta + 256 + Int64(group % 32 // 4) * 128 + column % 128])
                        subscale = (scale_pair >> (group % 4 // 2 * 4)) & 15
                        descriptor_offset = (
                            (column // self.descriptor_tile_n * (self.k // 256) + Int64(group // 32)) * (self.descriptor_tile_n * 32)
                            + (column % self.descriptor_tile_n) * 32 + Int64(group % 32)
                        )
                        descriptor = Uint32(descriptors[descriptor_offset])
                    address = get_ptr_as_int64(lut, 0)
                    w0, w1, w2, w3 = _decode_eight(descriptor, base, subscale, address)
                for r in cutlass.range_constexpr(self.tile_m):
                    row = first_row + r
                    if row < Int64(rows):
                        accum[r] = _dot_bf16x8(source, row * self.k + Int64(group) * 8,
                                              w0, w1, w2, w3, accum[r])
                group += 32
            for r in cutlass.range_constexpr(self.tile_m):
                row = first_row + r
                if row < Int64(rows):
                    total = warp_reduce(accum[r], _fadd)
                    if lane == 0:
                        output[row * self.n + column] = total.to(BFloat16)


@program_cache
def compile_gemv(ordinal, n, k, tile_m, codec="iq2_xs"):
    types = (BFloat16, Uint16, Uint8, Uint16, BFloat16)
    fake = tuple(make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16) for dtype in types)
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            IQ2XSGemv(n, k, tile_m, codec), *fake, Int32(1), current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key("gemm.iq2_xs.simt", 5, (ordinal, n, k, tile_m, codec)),
        )

    def run(source, values, lut, metadata, output, alpha, stream):
        del alpha
        tensors = (source, values, metadata, lut, output)
        pointers = tuple(make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
                         for dtype, tensor in zip(types, tensors))
        run_compiled(raw, (*pointers, Int32(source.shape[0]), cuda_stream_from_int_or_current(stream)))

    return attach_programs(run, raw)
