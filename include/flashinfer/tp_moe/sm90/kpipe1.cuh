/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// KPipe cut to its first k32 slice (the FC2 tail k-tile whose other slices are zero).
// Machine-generated inline PTX (one specialization per accumulator width the kernels use); the
// register / barrier protocol is part of the kernel's validated schedule.
#pragma once
#include <cstdint>

namespace flashinfer {
namespace tp_moe {
namespace sm90 {
namespace kpipe {

template <int N>
struct KPipe1;

template <>
struct KPipe1<8> {
  __device__ __forceinline__ static void run(float (&C)[4], float (&D)[4], uint32_t frag,
                                             uint32_t scl, uint64_t desc, uint32_t bar,
                                             uint32_t arrive) {
    asm volatile(
        "{\n"
        ".reg .pred pz, pt, pa;\n"
        ".reg .b32 q0, q1, oa, ob, la, ha, lb, hb, s0, s1, m0, m1, t0, t1;\n"
        ".reg .b32 a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, "
        "a33;\n"
        ".reg .b64 dsc;\n"
        "setp.ne.b32 pz, 0, 0;\n"
        "setp.eq.b32 pt, 0, 0;\n"
        "setp.ne.b32 pa, %12, 0;\n"
        "wgmma.wait_group.sync.aligned 3;\n"
        "ld.shared.v2.u32 {q0, q1}, [%8 + 0];\n"
        "ld.shared.u8 oa, [%9 + 0];\n"
        "ld.shared.u8 ob, [%9 + 1];\n"
        "mad.lo.u32 la, oa, 0x08080800, 0x0C080000;\n"
        "mad.lo.u32 ha, oa, 0x08080808, 0x1C181410;\n"
        "mad.lo.u32 lb, ob, 0x08080800, 0x0C080000;\n"
        "mad.lo.u32 hb, ob, 0x08080808, 0x1C181410;\n"
        "and.b32 s0, q0, 0x77777777;\n"
        "shr.u32 s1, s0, 16;\n"
        "prmt.b32 m0, la, ha, s0;\n"
        "prmt.b32 m1, la, ha, s1;\n"
        "shl.b32 t0, q0, 4;\n"
        "and.b32 t0, t0, 0x80808080;\n"
        "or.b32 a00, t0, m0;\n"
        "and.b32 t1, q0, 0x80808080;\n"
        "or.b32 a02, t1, m1;\n"
        "and.b32 s0, q1, 0x77777777;\n"
        "shr.u32 s1, s0, 16;\n"
        "prmt.b32 m0, lb, hb, s0;\n"
        "prmt.b32 m1, lb, hb, s1;\n"
        "shl.b32 t0, q1, 4;\n"
        "and.b32 t0, t0, 0x80808080;\n"
        "or.b32 a01, t0, m0;\n"
        "and.b32 t1, q1, 0x80808080;\n"
        "or.b32 a03, t1, m1;\n"
        "wgmma.wait_group.sync.aligned 0;\n"
        "add.rn.f32 %4, %4, %0;\n"
        "add.rn.f32 %5, %5, %1;\n"
        "add.rn.f32 %6, %6, %2;\n"
        "add.rn.f32 %7, %7, %3;\n"
        "@pa mbarrier.arrive.shared::cta.b64 _, [%11];\n"
        "wgmma.fence.sync.aligned;\n"
        "add.s64 dsc, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {a00, a01, a02, "
        "a03}, dsc, pz, 1, 1;\n"
        "wgmma.commit_group.sync.aligned;\n"
        "}\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(D[0]), "+f"(D[1]), "+f"(D[2]),
          "+f"(D[3])
        : "r"(frag), "r"(scl), "l"(desc), "r"(bar), "r"(arrive)
        : "memory");
  }
};

template <>
struct KPipe1<16> {
  __device__ __forceinline__ static void run(float (&C)[8], float (&D)[8], uint32_t frag,
                                             uint32_t scl, uint64_t desc, uint32_t bar,
                                             uint32_t arrive) {
    asm volatile(
        "{\n"
        ".reg .pred pz, pt, pa;\n"
        ".reg .b32 q0, q1, oa, ob, la, ha, lb, hb, s0, s1, m0, m1, t0, t1;\n"
        ".reg .b32 a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, "
        "a33;\n"
        ".reg .b64 dsc;\n"
        "setp.ne.b32 pz, 0, 0;\n"
        "setp.eq.b32 pt, 0, 0;\n"
        "setp.ne.b32 pa, %20, 0;\n"
        "wgmma.wait_group.sync.aligned 3;\n"
        "ld.shared.v2.u32 {q0, q1}, [%16 + 0];\n"
        "ld.shared.u8 oa, [%17 + 0];\n"
        "ld.shared.u8 ob, [%17 + 1];\n"
        "mad.lo.u32 la, oa, 0x08080800, 0x0C080000;\n"
        "mad.lo.u32 ha, oa, 0x08080808, 0x1C181410;\n"
        "mad.lo.u32 lb, ob, 0x08080800, 0x0C080000;\n"
        "mad.lo.u32 hb, ob, 0x08080808, 0x1C181410;\n"
        "and.b32 s0, q0, 0x77777777;\n"
        "shr.u32 s1, s0, 16;\n"
        "prmt.b32 m0, la, ha, s0;\n"
        "prmt.b32 m1, la, ha, s1;\n"
        "shl.b32 t0, q0, 4;\n"
        "and.b32 t0, t0, 0x80808080;\n"
        "or.b32 a00, t0, m0;\n"
        "and.b32 t1, q0, 0x80808080;\n"
        "or.b32 a02, t1, m1;\n"
        "and.b32 s0, q1, 0x77777777;\n"
        "shr.u32 s1, s0, 16;\n"
        "prmt.b32 m0, lb, hb, s0;\n"
        "prmt.b32 m1, lb, hb, s1;\n"
        "shl.b32 t0, q1, 4;\n"
        "and.b32 t0, t0, 0x80808080;\n"
        "or.b32 a01, t0, m0;\n"
        "and.b32 t1, q1, 0x80808080;\n"
        "or.b32 a03, t1, m1;\n"
        "wgmma.wait_group.sync.aligned 0;\n"
        "add.rn.f32 %8, %8, %0;\n"
        "add.rn.f32 %9, %9, %1;\n"
        "add.rn.f32 %10, %10, %2;\n"
        "add.rn.f32 %11, %11, %3;\n"
        "add.rn.f32 %12, %12, %4;\n"
        "add.rn.f32 %13, %13, %5;\n"
        "add.rn.f32 %14, %14, %6;\n"
        "add.rn.f32 %15, %15, %7;\n"
        "@pa mbarrier.arrive.shared::cta.b64 _, [%19];\n"
        "wgmma.fence.sync.aligned;\n"
        "add.s64 dsc, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{a00, a01, a02, a03}, dsc, pz, 1, 1;\n"
        "wgmma.commit_group.sync.aligned;\n"
        "}\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), "+f"(C[5]), "+f"(C[6]),
          "+f"(C[7]), "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]),
          "+f"(D[6]), "+f"(D[7])
        : "r"(frag), "r"(scl), "l"(desc), "r"(bar), "r"(arrive)
        : "memory");
  }
};

template <>
struct KPipe1<24> {
  __device__ __forceinline__ static void run(float (&C)[12], float (&D)[12], uint32_t frag,
                                             uint32_t scl, uint64_t desc, uint32_t bar,
                                             uint32_t arrive) {
    asm volatile(
        "{\n"
        ".reg .pred pz, pt, pa;\n"
        ".reg .b32 q0, q1, oa, ob, la, ha, lb, hb, s0, s1, m0, m1, t0, t1;\n"
        ".reg .b32 a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, "
        "a33;\n"
        ".reg .b64 dsc;\n"
        "setp.ne.b32 pz, 0, 0;\n"
        "setp.eq.b32 pt, 0, 0;\n"
        "setp.ne.b32 pa, %28, 0;\n"
        "wgmma.wait_group.sync.aligned 3;\n"
        "ld.shared.v2.u32 {q0, q1}, [%24 + 0];\n"
        "ld.shared.u8 oa, [%25 + 0];\n"
        "ld.shared.u8 ob, [%25 + 1];\n"
        "mad.lo.u32 la, oa, 0x08080800, 0x0C080000;\n"
        "mad.lo.u32 ha, oa, 0x08080808, 0x1C181410;\n"
        "mad.lo.u32 lb, ob, 0x08080800, 0x0C080000;\n"
        "mad.lo.u32 hb, ob, 0x08080808, 0x1C181410;\n"
        "and.b32 s0, q0, 0x77777777;\n"
        "shr.u32 s1, s0, 16;\n"
        "prmt.b32 m0, la, ha, s0;\n"
        "prmt.b32 m1, la, ha, s1;\n"
        "shl.b32 t0, q0, 4;\n"
        "and.b32 t0, t0, 0x80808080;\n"
        "or.b32 a00, t0, m0;\n"
        "and.b32 t1, q0, 0x80808080;\n"
        "or.b32 a02, t1, m1;\n"
        "and.b32 s0, q1, 0x77777777;\n"
        "shr.u32 s1, s0, 16;\n"
        "prmt.b32 m0, lb, hb, s0;\n"
        "prmt.b32 m1, lb, hb, s1;\n"
        "shl.b32 t0, q1, 4;\n"
        "and.b32 t0, t0, 0x80808080;\n"
        "or.b32 a01, t0, m0;\n"
        "and.b32 t1, q1, 0x80808080;\n"
        "or.b32 a03, t1, m1;\n"
        "wgmma.wait_group.sync.aligned 0;\n"
        "add.rn.f32 %12, %12, %0;\n"
        "add.rn.f32 %13, %13, %1;\n"
        "add.rn.f32 %14, %14, %2;\n"
        "add.rn.f32 %15, %15, %3;\n"
        "add.rn.f32 %16, %16, %4;\n"
        "add.rn.f32 %17, %17, %5;\n"
        "add.rn.f32 %18, %18, %6;\n"
        "add.rn.f32 %19, %19, %7;\n"
        "add.rn.f32 %20, %20, %8;\n"
        "add.rn.f32 %21, %21, %9;\n"
        "add.rn.f32 %22, %22, %10;\n"
        "add.rn.f32 %23, %23, %11;\n"
        "@pa mbarrier.arrive.shared::cta.b64 _, [%27];\n"
        "wgmma.fence.sync.aligned;\n"
        "add.s64 dsc, %26, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, "
        "%9, %10, %11}, {a00, a01, a02, a03}, dsc, pz, 1, 1;\n"
        "wgmma.commit_group.sync.aligned;\n"
        "}\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), "+f"(C[5]), "+f"(C[6]),
          "+f"(C[7]), "+f"(C[8]), "+f"(C[9]), "+f"(C[10]), "+f"(C[11]), "+f"(D[0]), "+f"(D[1]),
          "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]),
          "+f"(D[9]), "+f"(D[10]), "+f"(D[11])
        : "r"(frag), "r"(scl), "l"(desc), "r"(bar), "r"(arrive)
        : "memory");
  }
};

template <>
struct KPipe1<32> {
  __device__ __forceinline__ static void run(float (&C)[16], float (&D)[16], uint32_t frag,
                                             uint32_t scl, uint64_t desc, uint32_t bar,
                                             uint32_t arrive) {
    asm volatile(
        "{\n"
        ".reg .pred pz, pt, pa;\n"
        ".reg .b32 q0, q1, oa, ob, la, ha, lb, hb, s0, s1, m0, m1, t0, t1;\n"
        ".reg .b32 a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, "
        "a33;\n"
        ".reg .b64 dsc;\n"
        "setp.ne.b32 pz, 0, 0;\n"
        "setp.eq.b32 pt, 0, 0;\n"
        "setp.ne.b32 pa, %36, 0;\n"
        "wgmma.wait_group.sync.aligned 3;\n"
        "ld.shared.v2.u32 {q0, q1}, [%32 + 0];\n"
        "ld.shared.u8 oa, [%33 + 0];\n"
        "ld.shared.u8 ob, [%33 + 1];\n"
        "mad.lo.u32 la, oa, 0x08080800, 0x0C080000;\n"
        "mad.lo.u32 ha, oa, 0x08080808, 0x1C181410;\n"
        "mad.lo.u32 lb, ob, 0x08080800, 0x0C080000;\n"
        "mad.lo.u32 hb, ob, 0x08080808, 0x1C181410;\n"
        "and.b32 s0, q0, 0x77777777;\n"
        "shr.u32 s1, s0, 16;\n"
        "prmt.b32 m0, la, ha, s0;\n"
        "prmt.b32 m1, la, ha, s1;\n"
        "shl.b32 t0, q0, 4;\n"
        "and.b32 t0, t0, 0x80808080;\n"
        "or.b32 a00, t0, m0;\n"
        "and.b32 t1, q0, 0x80808080;\n"
        "or.b32 a02, t1, m1;\n"
        "and.b32 s0, q1, 0x77777777;\n"
        "shr.u32 s1, s0, 16;\n"
        "prmt.b32 m0, lb, hb, s0;\n"
        "prmt.b32 m1, lb, hb, s1;\n"
        "shl.b32 t0, q1, 4;\n"
        "and.b32 t0, t0, 0x80808080;\n"
        "or.b32 a01, t0, m0;\n"
        "and.b32 t1, q1, 0x80808080;\n"
        "or.b32 a03, t1, m1;\n"
        "wgmma.wait_group.sync.aligned 0;\n"
        "add.rn.f32 %16, %16, %0;\n"
        "add.rn.f32 %17, %17, %1;\n"
        "add.rn.f32 %18, %18, %2;\n"
        "add.rn.f32 %19, %19, %3;\n"
        "add.rn.f32 %20, %20, %4;\n"
        "add.rn.f32 %21, %21, %5;\n"
        "add.rn.f32 %22, %22, %6;\n"
        "add.rn.f32 %23, %23, %7;\n"
        "add.rn.f32 %24, %24, %8;\n"
        "add.rn.f32 %25, %25, %9;\n"
        "add.rn.f32 %26, %26, %10;\n"
        "add.rn.f32 %27, %27, %11;\n"
        "add.rn.f32 %28, %28, %12;\n"
        "add.rn.f32 %29, %29, %13;\n"
        "add.rn.f32 %30, %30, %14;\n"
        "add.rn.f32 %31, %31, %15;\n"
        "@pa mbarrier.arrive.shared::cta.b64 _, [%35];\n"
        "wgmma.fence.sync.aligned;\n"
        "add.s64 dsc, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, "
        "%9, %10, %11, %12, %13, %14, %15}, {a00, a01, a02, a03}, dsc, pz, 1, 1;\n"
        "wgmma.commit_group.sync.aligned;\n"
        "}\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), "+f"(C[5]), "+f"(C[6]),
          "+f"(C[7]), "+f"(C[8]), "+f"(C[9]), "+f"(C[10]), "+f"(C[11]), "+f"(C[12]), "+f"(C[13]),
          "+f"(C[14]), "+f"(C[15]), "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]),
          "+f"(D[5]), "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]),
          "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15])
        : "r"(frag), "r"(scl), "l"(desc), "r"(bar), "r"(arrive)
        : "memory");
  }
};

}  // namespace kpipe
}  // namespace sm90
}  // namespace tp_moe
}  // namespace flashinfer
