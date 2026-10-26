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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
template <typename T, int Capacity = 8>
struct __align__(16) CakePeerPointerTable { T* ptrs[Capacity]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 448
#define TMEM_ACC_OFFSET 0
#define TMEM_A_BF16_OFFSET 64
#define NUM_RAW_RING_STAGES 6
#define NUM_B_RING_STAGES 6
#define NUM_A_RING_STAGES 3
#define NUM_ACC_RING_STAGES 2
#define NUM_STORE_RING_STAGES 2
#define SMEM_RAW_SMEM_OFF 1024
#define SMEM_RAW_SMEM_STAGE_BYTES 16384
#define SMEM_RAW_SMEM_STRIDE 16384
#define SMEM_SF_SMEM_OFF 99328
#define SMEM_SF_SMEM_STAGE_BYTES 2048
#define SMEM_SF_SMEM_STRIDE 2048
#define SMEM_SMEM_B_OFF 111616
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_STAGING_OFF 209920
#define SMEM_STAGING_STAGE_BYTES 4096
#define SMEM_STAGING_STRIDE 4096
#define SMEM_SLOT_OFF 218112
#define SMEM_SLOT_STAGE_BYTES 12288
#define SMEM_SLOT_STRIDE 12288
#define SMEM_HIST_S_OFF 111616
#define SMEM_HIST_S_STAGE_BYTES 6144
#define SMEM_HIST_S_STRIDE 6144
#define SMEM_WBASE_S_OFF 117760
#define SMEM_WBASE_S_STAGE_BYTES 6144
#define SMEM_WBASE_S_STRIDE 6144
#define SMEM_PRE_S_OFF 123904
#define SMEM_PRE_S_STAGE_BYTES 6144
#define SMEM_PRE_S_STRIDE 6144
#define SMEM_SUF_S_OFF 130048
#define SMEM_SUF_S_STAGE_BYTES 6144
#define SMEM_SUF_S_STRIDE 6144
#define SMEM_TOTAL_S_OFF 136192
#define SMEM_TOTAL_S_STAGE_BYTES 1024
#define SMEM_TOTAL_S_STRIDE 1024
#define SMEM_CBASE_S_OFF 137216
#define SMEM_CBASE_S_STAGE_BYTES 1024
#define SMEM_CBASE_S_STRIDE 1024
#define SMEM_SEG_S_OFF 138240
#define SMEM_SEG_S_STAGE_BYTES 1040
#define SMEM_SEG_S_STRIDE 1040
#define SMEM_META_S_OFF 139280
#define SMEM_META_S_STAGE_BYTES 32
#define SMEM_META_S_STRIDE 32
#define SMEM_TOTAL 230400
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

__device__ __forceinline__ uint32_t cvt_bf16x2_e4m3x2_p112(uint16_t src) {
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(dst) : "h"(src));
#else
    uint32_t _f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(src));
    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
    float _f0;
    float _f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
    asm("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(dst) : "f"(_f1), "f"(_f0));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst) : "r"(dst), "r"(0x77807780u));
#endif
    return dst;
}

__device__ __forceinline__ uint32_t cvt_bf16x2_e2m1x2_prescaled(uint32_t src) {
    uint16_t _fp4_u16 = (uint16_t)(src & 0xFFu);
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
        : "=r"(dst) : "h"(_fp4_u16));
#else
    uint32_t _fp4_f16x2;
    asm("{ .reg .b8 _fp4b, _fp4z0, _fp4z1, _fp4z2;      \n\t"
        "  mov.b32 {_fp4b, _fp4z0, _fp4z1, _fp4z2}, %1;   \n\t"
        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;              }"
        : "=r"(_fp4_f16x2) : "r"((uint32_t)_fp4_u16));
    dst = ((_fp4_f16x2 & 0x7FFF7FFFu) >> 3) | (_fp4_f16x2 & 0x80008000u);
#endif
    return dst;
}

__device__ __forceinline__ void cvt_bf16x16_e2m1x32_e4m3x2_lo(uint32_t packed0, uint32_t packed1, uint32_t packed2, uint32_t packed3, uint32_t scale_word, uint32_t (&dst)[16]) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7, _q8, _q9, _q10, _q11, _q12, _q13, _q14, _q15; \n\t"
        "  .reg .b16 _s01, _s23, _sc0, _sc1;             \n\t"
        "  .reg .b32 _sc, _bc0, _bc1;                    \n\t"
        "  mov.b32 {_q0, _q1, _q2, _q3}, %16;   \n\t"
        "  mov.b32 {_q4, _q5, _q6, _q7}, %17;   \n\t"
        "  mov.b32 {_q8, _q9, _q10, _q11}, %18;   \n\t"
        "  mov.b32 {_q12, _q13, _q14, _q15}, %19;   \n\t"
        "  mov.b32 {_s01, _s23}, %20;                   \n\t"
        "  cvt.rn.bf16x2.e4m3x2 _sc, _s01;           \n\t"
        "  mov.b32 {_sc0, _sc1}, _sc;                   \n\t"
        "  mov.b32 _bc0, {_sc0, _sc0};                  \n\t"
        "  mov.b32 _bc1, {_sc1, _sc1};                  \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _q0;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %1, _q1;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %2, _q2;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %3, _q3;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %4, _q4;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %5, _q5;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %6, _q6;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %7, _q7;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %8, _q8;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %9, _q9;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %10, _q10;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %11, _q11;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %12, _q12;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %13, _q13;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %14, _q14;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %15, _q15;        \n\t"
        "  mul.rn.bf16x2 %0, %0, _bc0;           \n\t"
        "  mul.rn.bf16x2 %1, %1, _bc0;           \n\t"
        "  mul.rn.bf16x2 %2, %2, _bc0;           \n\t"
        "  mul.rn.bf16x2 %3, %3, _bc0;           \n\t"
        "  mul.rn.bf16x2 %4, %4, _bc0;           \n\t"
        "  mul.rn.bf16x2 %5, %5, _bc0;           \n\t"
        "  mul.rn.bf16x2 %6, %6, _bc0;           \n\t"
        "  mul.rn.bf16x2 %7, %7, _bc0;           \n\t"
        "  mul.rn.bf16x2 %8, %8, _bc1;           \n\t"
        "  mul.rn.bf16x2 %9, %9, _bc1;           \n\t"
        "  mul.rn.bf16x2 %10, %10, _bc1;           \n\t"
        "  mul.rn.bf16x2 %11, %11, _bc1;           \n\t"
        "  mul.rn.bf16x2 %12, %12, _bc1;           \n\t"
        "  mul.rn.bf16x2 %13, %13, _bc1;           \n\t"
        "  mul.rn.bf16x2 %14, %14, _bc1;           \n\t"
        "  mul.rn.bf16x2 %15, %15, _bc1;           \n\t"
        "}"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]), "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7]), "=r"(dst[8]), "=r"(dst[9]), "=r"(dst[10]), "=r"(dst[11]), "=r"(dst[12]), "=r"(dst[13]), "=r"(dst[14]), "=r"(dst[15])
        : "r"(packed0), "r"(packed1), "r"(packed2), "r"(packed3), "r"(scale_word));
#else
    uint32_t _scale_a = (scale_word >> 0) & 0xFFu;
    uint32_t _scale_b = (scale_word >> 8) & 0xFFu;
    uint32_t _scale_a_p112 = cvt_bf16x2_e4m3x2_p112((uint16_t)(_scale_a | (_scale_a << 8)));
    uint32_t _scale_b_p112 = cvt_bf16x2_e4m3x2_p112((uint16_t)(_scale_b | (_scale_b << 8)));
#pragma unroll
    for (int _k = 0; _k < 4; ++_k) {
        uint32_t _a0 = cvt_bf16x2_e2m1x2_prescaled(packed0 >> (8 * _k));
        uint32_t _a1 = cvt_bf16x2_e2m1x2_prescaled(packed1 >> (8 * _k));
        uint32_t _b0 = cvt_bf16x2_e2m1x2_prescaled(packed2 >> (8 * _k));
        uint32_t _b1 = cvt_bf16x2_e2m1x2_prescaled(packed3 >> (8 * _k));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[_k]) : "r"(_a0), "r"(_scale_a_p112));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[4 + _k]) : "r"(_a1), "r"(_scale_a_p112));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[8 + _k]) : "r"(_b0), "r"(_scale_b_p112));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[12 + _k]) : "r"(_b1), "r"(_scale_b_p112));
    }
#endif
}

__device__ __forceinline__ void cvt_bf16x16_e2m1x32_e4m3x2_hi(uint32_t packed0, uint32_t packed1, uint32_t packed2, uint32_t packed3, uint32_t scale_word, uint32_t (&dst)[16]) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7, _q8, _q9, _q10, _q11, _q12, _q13, _q14, _q15; \n\t"
        "  .reg .b16 _s01, _s23, _sc0, _sc1;             \n\t"
        "  .reg .b32 _sc, _bc0, _bc1;                    \n\t"
        "  mov.b32 {_q0, _q1, _q2, _q3}, %16;   \n\t"
        "  mov.b32 {_q4, _q5, _q6, _q7}, %17;   \n\t"
        "  mov.b32 {_q8, _q9, _q10, _q11}, %18;   \n\t"
        "  mov.b32 {_q12, _q13, _q14, _q15}, %19;   \n\t"
        "  mov.b32 {_s01, _s23}, %20;                   \n\t"
        "  cvt.rn.bf16x2.e4m3x2 _sc, _s23;           \n\t"
        "  mov.b32 {_sc0, _sc1}, _sc;                   \n\t"
        "  mov.b32 _bc0, {_sc0, _sc0};                  \n\t"
        "  mov.b32 _bc1, {_sc1, _sc1};                  \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _q0;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %1, _q1;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %2, _q2;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %3, _q3;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %4, _q4;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %5, _q5;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %6, _q6;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %7, _q7;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %8, _q8;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %9, _q9;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %10, _q10;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %11, _q11;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %12, _q12;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %13, _q13;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %14, _q14;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %15, _q15;        \n\t"
        "  mul.rn.bf16x2 %0, %0, _bc0;           \n\t"
        "  mul.rn.bf16x2 %1, %1, _bc0;           \n\t"
        "  mul.rn.bf16x2 %2, %2, _bc0;           \n\t"
        "  mul.rn.bf16x2 %3, %3, _bc0;           \n\t"
        "  mul.rn.bf16x2 %4, %4, _bc0;           \n\t"
        "  mul.rn.bf16x2 %5, %5, _bc0;           \n\t"
        "  mul.rn.bf16x2 %6, %6, _bc0;           \n\t"
        "  mul.rn.bf16x2 %7, %7, _bc0;           \n\t"
        "  mul.rn.bf16x2 %8, %8, _bc1;           \n\t"
        "  mul.rn.bf16x2 %9, %9, _bc1;           \n\t"
        "  mul.rn.bf16x2 %10, %10, _bc1;           \n\t"
        "  mul.rn.bf16x2 %11, %11, _bc1;           \n\t"
        "  mul.rn.bf16x2 %12, %12, _bc1;           \n\t"
        "  mul.rn.bf16x2 %13, %13, _bc1;           \n\t"
        "  mul.rn.bf16x2 %14, %14, _bc1;           \n\t"
        "  mul.rn.bf16x2 %15, %15, _bc1;           \n\t"
        "}"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]), "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7]), "=r"(dst[8]), "=r"(dst[9]), "=r"(dst[10]), "=r"(dst[11]), "=r"(dst[12]), "=r"(dst[13]), "=r"(dst[14]), "=r"(dst[15])
        : "r"(packed0), "r"(packed1), "r"(packed2), "r"(packed3), "r"(scale_word));
#else
    uint32_t _scale_a = (scale_word >> 16) & 0xFFu;
    uint32_t _scale_b = (scale_word >> 24) & 0xFFu;
    uint32_t _scale_a_p112 = cvt_bf16x2_e4m3x2_p112((uint16_t)(_scale_a | (_scale_a << 8)));
    uint32_t _scale_b_p112 = cvt_bf16x2_e4m3x2_p112((uint16_t)(_scale_b | (_scale_b << 8)));
#pragma unroll
    for (int _k = 0; _k < 4; ++_k) {
        uint32_t _a0 = cvt_bf16x2_e2m1x2_prescaled(packed0 >> (8 * _k));
        uint32_t _a1 = cvt_bf16x2_e2m1x2_prescaled(packed1 >> (8 * _k));
        uint32_t _b0 = cvt_bf16x2_e2m1x2_prescaled(packed2 >> (8 * _k));
        uint32_t _b1 = cvt_bf16x2_e2m1x2_prescaled(packed3 >> (8 * _k));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[_k]) : "r"(_a0), "r"(_scale_a_p112));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[4 + _k]) : "r"(_a1), "r"(_scale_a_p112));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[8 + _k]) : "r"(_b0), "r"(_scale_b_p112));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[12 + _k]) : "r"(_b1), "r"(_scale_b_p112));
    }
#endif
}

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}





__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}







__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_mega_moe_w4a16_bd1657f1c8c759cb8748(__nv_bfloat16* __restrict__ hidden, int* __restrict__ topk_ids, float* __restrict__ topk_weights, const __grid_constant__ CUtensorMap W13, const __grid_constant__ CUtensorMap S13, const __grid_constant__ CUtensorMap W2, const __grid_constant__ CUtensorMap S2, const __grid_constant__ CUtensorMap pool, __nv_bfloat16* __restrict__ pool_ptr, const __grid_constant__ CUtensorMap fc1_out_st, const __grid_constant__ CUtensorMap fc1_out_ld, int* __restrict__ route_src, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ tile_count, int* __restrict__ expert_seg, int* cta_hist, unsigned int* __restrict__ sync_words, unsigned int* __restrict__ epoch, unsigned int* __restrict__ fc1_ready, unsigned int* __restrict__ fc1_done, unsigned int* __restrict__ fc2_done, float* __restrict__ fc1_alpha, float* __restrict__ fc2_alpha, float swiglu_limit, int T_r, int cap, int H, int I, int E_local, int fc1_m_tiles, CakeFastDivmod fc1_m_div, int fc2_m_tiles, CakeFastDivmod fc2_m_div, int32_t mega_pg_world, int32_t mega_pg_rank, CakePeerPointerTable<unsigned int> mega_pg_flags, __nv_bfloat16* published_hidden, CakePeerPointerTable<__nv_bfloat16> published_hidden_peers, int* published_ids, CakePeerPointerTable<int> published_ids_peers, float* published_weights, CakePeerPointerTable<float> published_weights_peers, int* published_meta, CakePeerPointerTable<int> published_meta_peers, __nv_bfloat16* route_terms, CakePeerPointerTable<__nv_bfloat16> route_terms_peers, unsigned int* completion, CakePeerPointerTable<unsigned int> completion_peers)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define raw_full_addr (mbar_base + 0)
    #define raw_empty_addr (mbar_base + 48)
    #define b_full_addr (mbar_base + 96)
    #define b_empty_addr (mbar_base + 144)
    #define a_full_addr (mbar_base + 192)
    #define a_empty_addr (mbar_base + 216)
    #define acc_full_addr (mbar_base + 240)
    #define acc_empty_addr (mbar_base + 256)
    #define slot_full_addr (mbar_base + 272)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* raw_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int raw_smem_addr = smem + 1024;
    uint8_t* sf_smem = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int sf_smem_addr = smem + 99328;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 111616);
    const int smem_b_addr = smem + 111616;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 209920);
    const int staging_addr = smem + 209920;
    uint8_t* slot = reinterpret_cast<uint8_t*>(smem_raw + 218112);
    const int slot_addr = smem + 218112;
    int* hist_s = reinterpret_cast<int*>(smem_raw + 111616);
    const int hist_s_addr = smem + 111616;
    int* wbase_s = reinterpret_cast<int*>(smem_raw + 117760);
    const int wbase_s_addr = smem + 117760;
    int* pre_s = reinterpret_cast<int*>(smem_raw + 123904);
    const int pre_s_addr = smem + 123904;
    int* suf_s = reinterpret_cast<int*>(smem_raw + 130048);
    const int suf_s_addr = smem + 130048;
    int* total_s = reinterpret_cast<int*>(smem_raw + 136192);
    const int total_s_addr = smem + 136192;
    int* cbase_s = reinterpret_cast<int*>(smem_raw + 137216);
    const int cbase_s_addr = smem + 137216;
    int* seg_s = reinterpret_cast<int*>(smem_raw + 138240);
    const int seg_s_addr = smem + 138240;
    int* meta_s = reinterpret_cast<int*>(smem_raw + 139280);
    const int meta_s_addr = smem + 139280;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 35 barriers)
    // Mbarriers at smem_raw[0..280)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'raw_ring' ---
            // raw_full: 6 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // raw_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'b_ring' ---
            // b_full: 6 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // b_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // --- pipeline 'a_ring' ---
            // a_full: 3 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // a_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // --- pipeline 'acc_ring' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // acc_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 256, 4);
            mbarrier_init(smem + 264, 4);
            // slot_full: 1 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 448 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 280);
    if (warp == 0) {
        int _tmem_hold = smem + 280;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_a_bf16 = taddr + 64;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // epilogue_main
            const int world = mega_pg_world;
            const int rank = mega_pg_rank;
            const int fw = warp;
            const int ftid = fw * 32 + lane;
            const int num_ctas = num_bids;
            const int bid_0 = bid;
            const int linear = bid_0 * 192 + ftid;
            const int stride = num_ctas * 192;
            const int max_tiles = (world * cap * 8 + 31 * E_local) / 32;
            const unsigned int lane_lt = (1 << (unsigned int)lane) - 1;
            const int zero_i = 0;
            const int hist_quad_words = E_local * 4;
            #pragma unroll 1
            for (int t = linear; t < max_tiles; t += stride) {
                asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(fc1_ready + t)), "r"(static_cast<unsigned int>(0)) : "memory");
                asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(fc1_done + t)), "r"(static_cast<unsigned int>(0)) : "memory");
                asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(fc2_done + t)), "r"(static_cast<unsigned int>(0)) : "memory");
            }
            if (linear == 0) {
                reinterpret_cast<int*>(published_meta)[0] = T_r;
            }
            if (world > 1) {
                #pragma unroll 1
                for (int index = linear * 8; index < T_r * H; index += stride * 8) {
                    float _vec_load_2[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(hidden + index + 0);
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    unsigned int _packed_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_2[_lp*2 + 0], _vec_load_2[_lp*2+1 + 0]));
                        _packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_2[_lp*2 + 2], _vec_load_2[_lp*2+1 + 2]));
                        _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_2[_lp*2 + 4], _vec_load_2[_lp*2+1 + 4]));
                        _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_2[_lp*2 + 6], _vec_load_2[_lp*2+1 + 6]));
                        _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    {
                        auto _gst_1 = reinterpret_cast<__nv_bfloat16*>(published_hidden) + index;
                        __builtin_assume(__isGlobal(_gst_1));
                        reinterpret_cast<int4*>(_gst_1)[0] = reinterpret_cast<int4*>(_packed_2)[0];
                    }
                }
                #pragma unroll 1
                for (int route = linear; route < T_r * 8; route += stride) {
                    int route_id = topk_ids[route];
                    float route_weight = topk_weights[route];
                    reinterpret_cast<int*>(published_ids)[route] = route_id;
                    reinterpret_cast<float*>(published_weights)[route] = route_weight;
                }
                asm volatile("barrier.sync 11, 192;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sync_words) + (0))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
                if (warp == 0) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(sync_words) + (0))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>((unsigned int)num_ctas)) >= static_cast<unsigned int>(2147483648u));
                    }
                }
                asm volatile("barrier.sync 11, 192;" ::: "memory");
                if (bid_0 == 0 && warp == 0) {
                    // nvlink_barrier(mega_pg_flags) phase=0 owner_warp=0
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __warp = warp;
                        const int __lane = lane;
                        if (__warp == 0) {
                            unsigned* __local_epoch = mega_pg_flags.ptrs[__me] + 0;
                            unsigned __previous_epoch;
                            asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                                : "=r"(__previous_epoch) : "l"(__local_epoch) : "memory");
                            const unsigned __epoch = __previous_epoch + 1u;
                            const int __bank = (int)(__epoch & 1u);
                            const int __mailbox_base = 2 + (0 * 2 + __bank) * __ws;
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            if (__lane < __ws) {
                                unsigned* __peer_mailbox = mega_pg_flags.ptrs[__lane] + __mailbox_base + __me;
                                asm volatile("st.release.sys.global.u32 [%0], %1;"
                                    :: "l"(__peer_mailbox), "r"(__epoch) : "memory");
                                unsigned* __local_mailbox = mega_pg_flags.ptrs[__me] + __mailbox_base + __lane;
                                while (true) {
                                    unsigned __v;
                                    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_mailbox) : "memory");
                                    if (__v != __epoch) continue;
                                    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_mailbox) : "memory");
                                    if (__v == __epoch) break;
                                }
                                asm volatile("fence.proxy.alias;" ::: "memory");
                            }
                            __syncwarp();
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            if (__lane == 0) {
                                asm volatile("st.release.sys.global.u32 [%0], %1;"
                                    :: "l"(__local_epoch), "r"(__epoch) : "memory");
                            }
                        }
                    }
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sync_words) + (1))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
                if (warp == 0) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(sync_words) + (1))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(1)) >= static_cast<unsigned int>(2147483648u));
                    }
                }
                asm volatile("barrier.sync 11, 192;" ::: "memory");
            }
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                }
            }
            if (warp == 0) {
                if (world > 1) {
                    if (world > lane) {
                        int peer_tokens = reinterpret_cast<int*>(published_meta_peers.ptrs[lane])[0];
                        {
                            uint32_t _ival_2 = static_cast<uint32_t>(peer_tokens);
                            uint32_t _addr_2 = static_cast<uint32_t>(meta_s_addr + (unsigned int)(lane * 4));
                            asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_2), "r"(_ival_2) : "memory");
                        }
                    }
                } else {
                    if (elect_sync()) {
                        int own_tokens = T_r;
                        {
                            uint32_t _ival_3 = static_cast<uint32_t>(own_tokens);
                            uint32_t _addr_3 = static_cast<uint32_t>(meta_s_addr);
                            asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_3), "r"(_ival_3) : "memory");
                        }
                    }
                }
            }
            #pragma unroll 1
            for (int e = lane; e < E_local; e += 32) {
                {
                    uint32_t _ival_4 = static_cast<uint32_t>(zero_i);
                    uint32_t _addr_4 = static_cast<uint32_t>(hist_s_addr + (unsigned int)((fw * 256 + e) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_4), "r"(_ival_4) : "memory");
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            const int tokens_total = world * cap;
            const int tokens_per_cta = (tokens_total + num_ctas - 1) / num_ctas;
            const int cta_tok0 = bid_0 * tokens_per_cta;
            int cta_tok1_raw = cta_tok0 + tokens_per_cta;
            int cta_tok1 = ((cta_tok1_raw < tokens_total) ? cta_tok1_raw : tokens_total);
            const int tokens_per_warp = (tokens_per_cta + 6 - 1) / 6;
            const int w_tok0 = cta_tok0 + fw * tokens_per_warp;
            int w_tok1_raw = w_tok0 + tokens_per_warp;
            int w_tok1 = ((w_tok1_raw < cta_tok1) ? w_tok1_raw : cta_tok1);
            const int my_slot = lane % 8;
            const int my_tok_off = lane / 8;
            const int scan_all = ((world == 1 && T_r * 8 <= 4096) ? 1 : 0);
            const int scan_pre1 = ((cta_tok0 < T_r) ? cta_tok0 : T_r);
            if (scan_all != 0) {
                #pragma unroll 1
                for (int e_1 = lane; e_1 < E_local; e_1 += 32) {
                    {
                        uint32_t _ival_5 = static_cast<uint32_t>(zero_i);
                        uint32_t _addr_5 = static_cast<uint32_t>(pre_s_addr + (unsigned int)((fw * 256 + e_1) * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_5), "r"(_ival_5) : "memory");
                    }
                    {
                        uint32_t _ival_6 = static_cast<uint32_t>(zero_i);
                        uint32_t _addr_6 = static_cast<uint32_t>(suf_s_addr + (unsigned int)((fw * 256 + e_1) * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_6), "r"(_ival_6) : "memory");
                    }
                }
                __syncwarp();
            }
            #pragma unroll 1
            for (int tok = w_tok0; tok < w_tok1; tok += 4) {
                int my_tok = tok + my_tok_off;
                int src = my_tok / cap;
                int token = my_tok - src * cap;
                int g = my_tok * 8 + my_slot;
                int ident = -1;
                if (my_tok < w_tok1) {
                    if (token < meta_s[src]) {
                        if (world > 1) {
                            ident = reinterpret_cast<int*>(published_ids_peers.ptrs[src])[token * 8 + my_slot];
                        } else {
                            ident = topk_ids[token * 8 + my_slot];
                        }
                    }
                }
                int local = 0;
                int expert = 0;
                if (ident >= 0) {
                    if (ident / E_local == rank) {
                        local = 1;
                        expert = ident - rank * E_local;
                    }
                }
                int key = ((local != 0) ? expert : -1);
                unsigned int live = __ballot_sync(0xFFFFFFFF, local != 0);
                int _count_leading_zeros_7 = __clz(live);
                int trips = 32 - _count_leading_zeros_7;
                unsigned int peers = 0;
                #pragma unroll 1
                for (int lane0 = 0; lane0 < trips; lane0 += 4) {
                    int held_0 = __shfl_sync(0xFFFFFFFF, key, lane0 + 0);
                    unsigned int same_0 = ((held_0 == key) ? 1 : 0);
                    peers = peers | same_0 << (unsigned int)(lane0 + 0);
                    int held_1 = __shfl_sync(0xFFFFFFFF, key, lane0 + 1);
                    unsigned int same_1 = ((held_1 == key) ? 1 : 0);
                    peers = peers | same_1 << (unsigned int)(lane0 + 1);
                    int held_2 = __shfl_sync(0xFFFFFFFF, key, lane0 + 2);
                    unsigned int same_2 = ((held_2 == key) ? 1 : 0);
                    peers = peers | same_2 << (unsigned int)(lane0 + 2);
                    int held_3 = __shfl_sync(0xFFFFFFFF, key, lane0 + 3);
                    unsigned int same_3 = ((held_3 == key) ? 1 : 0);
                    peers = peers | same_3 << (unsigned int)(lane0 + 3);
                }
                unsigned int group = peers & live;
                int members = (int)(unsigned int)__popc(group);
                int lower = (int)(unsigned int)__popc(group & lane_lt);
                if (local != 0 && lower == 0) {
                    unsigned int count_addr = hist_s_addr + (unsigned int)((fw * 256 + expert) * 4);
                    int counted = hist_s[fw * 256 + expert];
                    {
                        uint32_t _ival_8 = static_cast<uint32_t>(counted + members);
                        uint32_t _addr_8 = static_cast<uint32_t>(count_addr);
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_8), "r"(_ival_8) : "memory");
                    }
                }
                __syncwarp();
            }
            if (scan_all != 0) {
                #pragma unroll 1
                for (int tok_1 = fw * 4; tok_1 < scan_pre1; tok_1 += 24) {
                    int my_tok_1 = tok_1 + my_tok_off;
                    int src_1 = my_tok_1 / cap;
                    int token_1 = my_tok_1 - src_1 * cap;
                    int g_1 = my_tok_1 * 8 + my_slot;
                    int ident_1 = -1;
                    if (my_tok_1 < scan_pre1) {
                        if (token_1 < meta_s[src_1]) {
                            if (world > 1) {
                                ident_1 = reinterpret_cast<int*>(published_ids_peers.ptrs[src_1])[token_1 * 8 + my_slot];
                            } else {
                                ident_1 = topk_ids[token_1 * 8 + my_slot];
                            }
                        }
                    }
                    int local_1 = 0;
                    int expert_1 = 0;
                    if (ident_1 >= 0) {
                        if (ident_1 / E_local == rank) {
                            local_1 = 1;
                            expert_1 = ident_1 - rank * E_local;
                        }
                    }
                    int _atomic_old_4 = 0;
                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.s32 %0, [%1], %2;\n}"
                        : "+r"(_atomic_old_4) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&pre_s[fw * 256 + expert_1]))), "r"(static_cast<int>(1)), "r"(static_cast<uint32_t>(local_1 != 0)) : "memory");
                }
                #pragma unroll 1
                for (int tok_2 = cta_tok1 + fw * 4; tok_2 < T_r; tok_2 += 24) {
                    int my_tok_2 = tok_2 + my_tok_off;
                    int src_2 = my_tok_2 / cap;
                    int token_2 = my_tok_2 - src_2 * cap;
                    int g_2 = my_tok_2 * 8 + my_slot;
                    int ident_2 = -1;
                    if (my_tok_2 < T_r) {
                        if (token_2 < meta_s[src_2]) {
                            if (world > 1) {
                                ident_2 = reinterpret_cast<int*>(published_ids_peers.ptrs[src_2])[token_2 * 8 + my_slot];
                            } else {
                                ident_2 = topk_ids[token_2 * 8 + my_slot];
                            }
                        }
                    }
                    int local_2 = 0;
                    int expert_2 = 0;
                    if (ident_2 >= 0) {
                        if (ident_2 / E_local == rank) {
                            local_2 = 1;
                            expert_2 = ident_2 - rank * E_local;
                        }
                    }
                    int _atomic_old_5 = 0;
                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.s32 %0, [%1], %2;\n}"
                        : "+r"(_atomic_old_5) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&suf_s[fw * 256 + expert_2]))), "r"(static_cast<int>(1)), "r"(static_cast<uint32_t>(local_2 != 0)) : "memory");
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            #pragma unroll 1
            for (int e_2 = ftid; e_2 < E_local; e_2 += 192) {
                int running = 0;
                {
                    uint32_t _ival_9 = static_cast<uint32_t>(running);
                    uint32_t _addr_9 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)(e_2 * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_9), "r"(_ival_9) : "memory");
                }
                running = running + hist_s[e_2];
                {
                    uint32_t _ival_10 = static_cast<uint32_t>(running);
                    uint32_t _addr_10 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((256 + e_2) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_10), "r"(_ival_10) : "memory");
                }
                running = running + hist_s[256 + e_2];
                {
                    uint32_t _ival_11 = static_cast<uint32_t>(running);
                    uint32_t _addr_11 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((512 + e_2) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_11), "r"(_ival_11) : "memory");
                }
                running = running + hist_s[512 + e_2];
                {
                    uint32_t _ival_12 = static_cast<uint32_t>(running);
                    uint32_t _addr_12 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((768 + e_2) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_12), "r"(_ival_12) : "memory");
                }
                running = running + hist_s[768 + e_2];
                {
                    uint32_t _ival_13 = static_cast<uint32_t>(running);
                    uint32_t _addr_13 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((1024 + e_2) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_13), "r"(_ival_13) : "memory");
                }
                running = running + hist_s[1024 + e_2];
                {
                    uint32_t _ival_14 = static_cast<uint32_t>(running);
                    uint32_t _addr_14 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((1280 + e_2) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_14), "r"(_ival_14) : "memory");
                }
                running = running + hist_s[1280 + e_2];
                if (scan_all != 0) {
                    int before = 0;
                    int after = 0;
                    before = before + pre_s[e_2];
                    after = after + suf_s[e_2];
                    before = before + pre_s[256 + e_2];
                    after = after + suf_s[256 + e_2];
                    before = before + pre_s[512 + e_2];
                    after = after + suf_s[512 + e_2];
                    before = before + pre_s[768 + e_2];
                    after = after + suf_s[768 + e_2];
                    before = before + pre_s[1024 + e_2];
                    after = after + suf_s[1024 + e_2];
                    before = before + pre_s[1280 + e_2];
                    after = after + suf_s[1280 + e_2];
                    {
                        uint32_t _ival_15 = static_cast<uint32_t>(before);
                        uint32_t _addr_15 = static_cast<uint32_t>(cbase_s_addr + (unsigned int)(e_2 * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_15), "r"(_ival_15) : "memory");
                    }
                    {
                        uint32_t _ival_16 = static_cast<uint32_t>(before + running + after);
                        uint32_t _addr_16 = static_cast<uint32_t>(total_s_addr + (unsigned int)(e_2 * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_16), "r"(_ival_16) : "memory");
                    }
                } else {
                    cta_hist[bid_0 / 4 * hist_quad_words + e_2 * 4 + bid_0 % 4] = running;
                }
            }
            int quads[32];
            if (scan_all == 0) {
                asm volatile("barrier.sync 11, 192;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sync_words) + (2))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
                if (warp == 0) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(sync_words) + (2))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>((unsigned int)num_ctas)) >= static_cast<unsigned int>(2147483648u));
                    }
                }
                asm volatile("barrier.sync 11, 192;" ::: "memory");
                #pragma unroll 1
                for (int e_3 = ftid; e_3 < E_local; e_3 += 192) {
                    int total = 0;
                    int base = 0;
                    #pragma unroll 1
                    for (int c0 = 0; c0 < num_ctas; c0 += 32) {
                        {
                            const int4* _ivptr_17 = reinterpret_cast<const int4*>(cta_hist + (c0 / 4 * hist_quad_words + e_3 * 4) + 0);
                            int4 _ivld_17;
                            _ivld_17 = *_ivptr_17;
                            quads[0 + 0] = _ivld_17.x;
                            quads[0 + 1] = _ivld_17.y;
                            quads[0 + 2] = _ivld_17.z;
                            quads[0 + 3] = _ivld_17.w;
                        }
                        {
                            const int4* _ivptr_18 = reinterpret_cast<const int4*>(cta_hist + ((c0 / 4 + 1) * hist_quad_words + e_3 * 4) + 0);
                            int4 _ivld_18;
                            _ivld_18 = *_ivptr_18;
                            quads[4 + 0] = _ivld_18.x;
                            quads[4 + 1] = _ivld_18.y;
                            quads[4 + 2] = _ivld_18.z;
                            quads[4 + 3] = _ivld_18.w;
                        }
                        {
                            const int4* _ivptr_19 = reinterpret_cast<const int4*>(cta_hist + ((c0 / 4 + 2) * hist_quad_words + e_3 * 4) + 0);
                            int4 _ivld_19;
                            _ivld_19 = *_ivptr_19;
                            quads[8 + 0] = _ivld_19.x;
                            quads[8 + 1] = _ivld_19.y;
                            quads[8 + 2] = _ivld_19.z;
                            quads[8 + 3] = _ivld_19.w;
                        }
                        {
                            const int4* _ivptr_20 = reinterpret_cast<const int4*>(cta_hist + ((c0 / 4 + 3) * hist_quad_words + e_3 * 4) + 0);
                            int4 _ivld_20;
                            _ivld_20 = *_ivptr_20;
                            quads[12 + 0] = _ivld_20.x;
                            quads[12 + 1] = _ivld_20.y;
                            quads[12 + 2] = _ivld_20.z;
                            quads[12 + 3] = _ivld_20.w;
                        }
                        {
                            const int4* _ivptr_21 = reinterpret_cast<const int4*>(cta_hist + ((c0 / 4 + 4) * hist_quad_words + e_3 * 4) + 0);
                            int4 _ivld_21;
                            _ivld_21 = *_ivptr_21;
                            quads[16 + 0] = _ivld_21.x;
                            quads[16 + 1] = _ivld_21.y;
                            quads[16 + 2] = _ivld_21.z;
                            quads[16 + 3] = _ivld_21.w;
                        }
                        {
                            const int4* _ivptr_22 = reinterpret_cast<const int4*>(cta_hist + ((c0 / 4 + 5) * hist_quad_words + e_3 * 4) + 0);
                            int4 _ivld_22;
                            _ivld_22 = *_ivptr_22;
                            quads[20 + 0] = _ivld_22.x;
                            quads[20 + 1] = _ivld_22.y;
                            quads[20 + 2] = _ivld_22.z;
                            quads[20 + 3] = _ivld_22.w;
                        }
                        {
                            const int4* _ivptr_23 = reinterpret_cast<const int4*>(cta_hist + ((c0 / 4 + 6) * hist_quad_words + e_3 * 4) + 0);
                            int4 _ivld_23;
                            _ivld_23 = *_ivptr_23;
                            quads[24 + 0] = _ivld_23.x;
                            quads[24 + 1] = _ivld_23.y;
                            quads[24 + 2] = _ivld_23.z;
                            quads[24 + 3] = _ivld_23.w;
                        }
                        {
                            const int4* _ivptr_24 = reinterpret_cast<const int4*>(cta_hist + ((c0 / 4 + 7) * hist_quad_words + e_3 * 4) + 0);
                            int4 _ivld_24;
                            _ivld_24 = *_ivptr_24;
                            quads[28 + 0] = _ivld_24.x;
                            quads[28 + 1] = _ivld_24.y;
                            quads[28 + 2] = _ivld_24.z;
                            quads[28 + 3] = _ivld_24.w;
                        }
                        total = total + ((num_ctas > c0) ? quads[0] : 0);
                        base = base + ((bid_0 > c0) ? quads[0] : 0);
                        total = total + ((num_ctas > c0 + 1) ? quads[1] : 0);
                        base = base + ((bid_0 > c0 + 1) ? quads[1] : 0);
                        total = total + ((num_ctas > c0 + 2) ? quads[2] : 0);
                        base = base + ((bid_0 > c0 + 2) ? quads[2] : 0);
                        total = total + ((num_ctas > c0 + 3) ? quads[3] : 0);
                        base = base + ((bid_0 > c0 + 3) ? quads[3] : 0);
                        total = total + ((num_ctas > c0 + 4) ? quads[4] : 0);
                        base = base + ((bid_0 > c0 + 4) ? quads[4] : 0);
                        total = total + ((num_ctas > c0 + 5) ? quads[5] : 0);
                        base = base + ((bid_0 > c0 + 5) ? quads[5] : 0);
                        total = total + ((num_ctas > c0 + 6) ? quads[6] : 0);
                        base = base + ((bid_0 > c0 + 6) ? quads[6] : 0);
                        total = total + ((num_ctas > c0 + 7) ? quads[7] : 0);
                        base = base + ((bid_0 > c0 + 7) ? quads[7] : 0);
                        total = total + ((num_ctas > c0 + 8) ? quads[8] : 0);
                        base = base + ((bid_0 > c0 + 8) ? quads[8] : 0);
                        total = total + ((num_ctas > c0 + 9) ? quads[9] : 0);
                        base = base + ((bid_0 > c0 + 9) ? quads[9] : 0);
                        total = total + ((num_ctas > c0 + 10) ? quads[10] : 0);
                        base = base + ((bid_0 > c0 + 10) ? quads[10] : 0);
                        total = total + ((num_ctas > c0 + 11) ? quads[11] : 0);
                        base = base + ((bid_0 > c0 + 11) ? quads[11] : 0);
                        total = total + ((num_ctas > c0 + 12) ? quads[12] : 0);
                        base = base + ((bid_0 > c0 + 12) ? quads[12] : 0);
                        total = total + ((num_ctas > c0 + 13) ? quads[13] : 0);
                        base = base + ((bid_0 > c0 + 13) ? quads[13] : 0);
                        total = total + ((num_ctas > c0 + 14) ? quads[14] : 0);
                        base = base + ((bid_0 > c0 + 14) ? quads[14] : 0);
                        total = total + ((num_ctas > c0 + 15) ? quads[15] : 0);
                        base = base + ((bid_0 > c0 + 15) ? quads[15] : 0);
                        total = total + ((num_ctas > c0 + 16) ? quads[16] : 0);
                        base = base + ((bid_0 > c0 + 16) ? quads[16] : 0);
                        total = total + ((num_ctas > c0 + 17) ? quads[17] : 0);
                        base = base + ((bid_0 > c0 + 17) ? quads[17] : 0);
                        total = total + ((num_ctas > c0 + 18) ? quads[18] : 0);
                        base = base + ((bid_0 > c0 + 18) ? quads[18] : 0);
                        total = total + ((num_ctas > c0 + 19) ? quads[19] : 0);
                        base = base + ((bid_0 > c0 + 19) ? quads[19] : 0);
                        total = total + ((num_ctas > c0 + 20) ? quads[20] : 0);
                        base = base + ((bid_0 > c0 + 20) ? quads[20] : 0);
                        total = total + ((num_ctas > c0 + 21) ? quads[21] : 0);
                        base = base + ((bid_0 > c0 + 21) ? quads[21] : 0);
                        total = total + ((num_ctas > c0 + 22) ? quads[22] : 0);
                        base = base + ((bid_0 > c0 + 22) ? quads[22] : 0);
                        total = total + ((num_ctas > c0 + 23) ? quads[23] : 0);
                        base = base + ((bid_0 > c0 + 23) ? quads[23] : 0);
                        total = total + ((num_ctas > c0 + 24) ? quads[24] : 0);
                        base = base + ((bid_0 > c0 + 24) ? quads[24] : 0);
                        total = total + ((num_ctas > c0 + 25) ? quads[25] : 0);
                        base = base + ((bid_0 > c0 + 25) ? quads[25] : 0);
                        total = total + ((num_ctas > c0 + 26) ? quads[26] : 0);
                        base = base + ((bid_0 > c0 + 26) ? quads[26] : 0);
                        total = total + ((num_ctas > c0 + 27) ? quads[27] : 0);
                        base = base + ((bid_0 > c0 + 27) ? quads[27] : 0);
                        total = total + ((num_ctas > c0 + 28) ? quads[28] : 0);
                        base = base + ((bid_0 > c0 + 28) ? quads[28] : 0);
                        total = total + ((num_ctas > c0 + 29) ? quads[29] : 0);
                        base = base + ((bid_0 > c0 + 29) ? quads[29] : 0);
                        total = total + ((num_ctas > c0 + 30) ? quads[30] : 0);
                        base = base + ((bid_0 > c0 + 30) ? quads[30] : 0);
                        total = total + ((num_ctas > c0 + 31) ? quads[31] : 0);
                        base = base + ((bid_0 > c0 + 31) ? quads[31] : 0);
                    }
                    {
                        uint32_t _ival_25 = static_cast<uint32_t>(total);
                        uint32_t _addr_25 = static_cast<uint32_t>(total_s_addr + (unsigned int)(e_3 * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_25), "r"(_ival_25) : "memory");
                    }
                    {
                        uint32_t _ival_26 = static_cast<uint32_t>(base);
                        uint32_t _addr_26 = static_cast<uint32_t>(cbase_s_addr + (unsigned int)(e_3 * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_26), "r"(_ival_26) : "memory");
                    }
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            if (warp == 0) {
                int carry = 0;
                #pragma unroll 1
                for (int chunk = 0; chunk < E_local; chunk += 32) {
                    int e_4 = chunk + lane;
                    int padded = 0;
                    if (e_4 < E_local) {
                        padded = (total_s[e_4] + 31) / 32 * 32;
                    }
                    int inclusive = padded;
                    int above = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
                    if (lane >= 1) {
                        inclusive = inclusive + above;
                    }
                    int above_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
                    if (lane >= 2) {
                        inclusive = inclusive + above_0;
                    }
                    int above_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
                    if (lane >= 4) {
                        inclusive = inclusive + above_1;
                    }
                    int above_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
                    if (lane >= 8) {
                        inclusive = inclusive + above_2;
                    }
                    int above_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
                    if (lane >= 16) {
                        inclusive = inclusive + above_3;
                    }
                    if (e_4 < E_local) {
                        {
                            uint32_t _ival_27 = static_cast<uint32_t>(carry + inclusive - padded);
                            uint32_t _addr_27 = static_cast<uint32_t>(seg_s_addr + (unsigned int)(e_4 * 4));
                            asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_27), "r"(_ival_27) : "memory");
                        }
                    }
                    int _shfl_sync_28 = __shfl_sync(0xFFFFFFFF, inclusive, 31);
                    carry = carry + _shfl_sync_28;
                }
                if (elect_sync()) {
                    {
                        uint32_t _ival_29 = static_cast<uint32_t>(carry);
                        uint32_t _addr_29 = static_cast<uint32_t>(seg_s_addr + (unsigned int)(E_local * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_29), "r"(_ival_29) : "memory");
                    }
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            const int active_rows = seg_s[E_local];
            const int active_tiles = active_rows / 32;
            if (bid_0 == 0) {
                if (warp == 0) {
                    if (elect_sync()) {
                        tile_count[0] = active_tiles;
                    }
                }
                #pragma unroll 1
                for (int e_5 = ftid; e_5 < E_local + 1; e_5 += 192) {
                    expert_seg[e_5] = seg_s[e_5];
                }
                #pragma unroll 1
                for (int e_6 = ftid; e_6 < E_local; e_6 += 192) {
                    int rows = total_s[e_6];
                    int seg0 = seg_s[e_6];
                    int first = seg0 / 32;
                    int tiles = (rows + 31) / 32;
                    #pragma unroll 1
                    for (int k = 0; k < tiles; k++) {
                        tile_expert[first + k] = e_6;
                        int limit = (k + 1) * 32;
                        tile_mn_limit[first + k] = seg0 + ((rows < limit) ? rows : limit);
                    }
                    #pragma unroll 1
                    for (int pad_row = seg0 + rows; pad_row < seg0 + tiles * 32; pad_row++) {
                        route_src[pad_row] = -1;
                    }
                }
            }
            #pragma unroll 1
            for (int e_7 = lane; e_7 < E_local; e_7 += 32) {
                {
                    uint32_t _ival_30 = static_cast<uint32_t>(zero_i);
                    uint32_t _addr_30 = static_cast<uint32_t>(hist_s_addr + (unsigned int)((fw * 256 + e_7) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_30), "r"(_ival_30) : "memory");
                }
            }
            __syncwarp();
            #pragma unroll 1
            for (int tok_3 = w_tok0; tok_3 < w_tok1; tok_3 += 4) {
                int my_tok_3 = tok_3 + my_tok_off;
                int src_3 = my_tok_3 / cap;
                int token_3 = my_tok_3 - src_3 * cap;
                int g_3 = my_tok_3 * 8 + my_slot;
                int ident_3 = -1;
                if (my_tok_3 < w_tok1) {
                    if (token_3 < meta_s[src_3]) {
                        if (world > 1) {
                            ident_3 = reinterpret_cast<int*>(published_ids_peers.ptrs[src_3])[token_3 * 8 + my_slot];
                        } else {
                            ident_3 = topk_ids[token_3 * 8 + my_slot];
                        }
                    }
                }
                int local_3 = 0;
                int expert_3 = 0;
                if (ident_3 >= 0) {
                    if (ident_3 / E_local == rank) {
                        local_3 = 1;
                        expert_3 = ident_3 - rank * E_local;
                    }
                }
                int key_1 = ((local_3 != 0) ? expert_3 : -1);
                unsigned int live_1 = __ballot_sync(0xFFFFFFFF, local_3 != 0);
                int _count_leading_zeros_31 = __clz(live_1);
                int trips_1 = 32 - _count_leading_zeros_31;
                unsigned int peers_1 = 0;
                #pragma unroll 1
                for (int lane0_1 = 0; lane0_1 < trips_1; lane0_1 += 4) {
                    int held_0_1 = __shfl_sync(0xFFFFFFFF, key_1, lane0_1 + 0);
                    unsigned int same_0_1 = ((held_0_1 == key_1) ? 1 : 0);
                    peers_1 = peers_1 | same_0_1 << (unsigned int)(lane0_1 + 0);
                    int held_1_1 = __shfl_sync(0xFFFFFFFF, key_1, lane0_1 + 1);
                    unsigned int same_1_1 = ((held_1_1 == key_1) ? 1 : 0);
                    peers_1 = peers_1 | same_1_1 << (unsigned int)(lane0_1 + 1);
                    int held_2_1 = __shfl_sync(0xFFFFFFFF, key_1, lane0_1 + 2);
                    unsigned int same_2_1 = ((held_2_1 == key_1) ? 1 : 0);
                    peers_1 = peers_1 | same_2_1 << (unsigned int)(lane0_1 + 2);
                    int held_3_1 = __shfl_sync(0xFFFFFFFF, key_1, lane0_1 + 3);
                    unsigned int same_3_1 = ((held_3_1 == key_1) ? 1 : 0);
                    peers_1 = peers_1 | same_3_1 << (unsigned int)(lane0_1 + 3);
                }
                unsigned int group_1 = peers_1 & live_1;
                int members_1 = (int)(unsigned int)__popc(group_1);
                int lower_1 = (int)(unsigned int)__popc(group_1 & lane_lt);
                int running_1 = 0;
                if (local_3 != 0) {
                    running_1 = hist_s[fw * 256 + expert_3];
                    int my_row = seg_s[expert_3] + cbase_s[expert_3] + wbase_s[fw * 256 + expert_3] + running_1 + lower_1;
                    route_src[my_row] = g_3;
                }
                __syncwarp();
                if (local_3 != 0 && lower_1 == 0) {
                    {
                        uint32_t _ival_32 = static_cast<uint32_t>(running_1 + members_1);
                        uint32_t _addr_32 = static_cast<uint32_t>(hist_s_addr + (unsigned int)((fw * 256 + expert_3) * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_32), "r"(_ival_32) : "memory");
                    }
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sync_words) + (3))), "r"(static_cast<unsigned int>(1)) : "memory");
                }
            }
            if (warp == 0) {
                {
                unsigned int _acquire_observed;
                do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(sync_words) + (3))) : "memory");
                } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>((unsigned int)num_ctas)) >= static_cast<unsigned int>(2147483648u));
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            asm volatile("barrier.sync 12, 512;" ::: "memory");
            int tiles_1 = tile_count[0];
            int fc1_items = tiles_1 * fc1_m_tiles;
            int wl_b = (int)bid;
            int wl_P = (int)num_bids;
            int wl_fc2 = tiles_1 * fc2_m_tiles;
            int wl_r = fc1_items % wl_P;
            int wl_L = wl_P - wl_r;
            int wl_delta = (H / 256 + I / 256 / 2) / (I / 256);
            int wl_n1 = wl_delta * wl_L;
            if (wl_n1 > wl_fc2) {
                wl_n1 = wl_fc2;
            }
            int wl_c1 = (fc1_items - wl_b + wl_P - 1) / wl_P;
            int wl_p1 = ((wl_b >= wl_r) ? (wl_n1 - (wl_b - wl_r) + wl_L - 1) / wl_L : 0);
            int wl_p2 = (wl_fc2 - wl_n1 - wl_b + wl_P - 1) / wl_P;
            int wl_items = wl_c1 + wl_p1 + wl_p2;
            const unsigned long long H64 = (unsigned long long)H;
            const int routes_per_rank = cap * 8;
            const int quarter = warp % 4;
            const int etid = quarter * 32 + lane;
            const unsigned int gate_lanes = (unsigned int)(quarter * 32) << 16;
            const unsigned int up_lanes = (unsigned int)(quarter * 32 + 16) << 16;
            const int thread_off = lane / 2 % 2 * 128 + (quarter * 16 + lane / 4 + lane % 2 * 8) * 2;
            const unsigned int ch_off = (unsigned int)(etid * 2);
            const int piece_row = etid / 8;
            const int piece_off = etid % 8 * 32;
            const int piece_words = etid % 8 * 8;
            float _neg_33;
            asm("neg.f32 %0, %1;" : "=f"(_neg_33) : "f"(swiglu_limit));
            const float neg_limit = _neg_33;
            unsigned int acc_cur = 0;
            unsigned int out_cur = 0;
            unsigned int piece[8];
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int w = 0; w < wl_items; w++) {
                int w_i = (int)w;
                int wl_j = w_i - wl_c1;
                if (wl_j < 0) {
                    w_i = bid + w_i * num_bids;
                } else if (wl_j < wl_p1) {
                    w_i = fc1_items + wl_j * (num_bids - wl_r) + (bid - wl_r);
                } else {
                    w_i = fc1_items + wl_n1 + (wl_j - wl_p1) * num_bids + bid;
                }
                int is_fc1 = 0;
                int local_4 = w_i - fc1_items;
                if (w_i < fc1_items) {
                    is_fc1 = 1;
                    local_4 = w_i;
                }
                unsigned int ww = (unsigned int)local_4;
                uint32_t _fast_div_q_8 = (fc1_m_div.divisor != 1) ? (__umulhi((uint32_t)(ww), fc1_m_div.multiplier) >> fc1_m_div.shift_right) : (uint32_t)(ww);
                uint32_t _fast_div_r_8 = (uint32_t)(ww) - _fast_div_q_8 * (uint32_t)(fc1_m_div.divisor);
                uint32_t _fast_div_q_9 = (fc2_m_div.divisor != 1) ? (__umulhi((uint32_t)(ww), fc2_m_div.multiplier) >> fc2_m_div.shift_right) : (uint32_t)(ww);
                uint32_t _fast_div_r_9 = (uint32_t)(ww) - _fast_div_q_9 * (uint32_t)(fc2_m_div.divisor);
                int t_1 = (int)_fast_div_q_9;
                int m = (int)_fast_div_r_9;
                if (is_fc1 != 0) {
                    t_1 = (int)_fast_div_q_8;
                    m = (int)_fast_div_r_8;
                }
                int expert_4 = tile_expert[t_1];
                int row0 = t_1 * 32;
                mbarrier_wait(acc_full_addr + (acc_cur) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (is_fc1 != 0) {
                    float alpha_e = fc1_alpha[expert_4];
                    int feature0 = m * 64;
                    #pragma unroll
                    for (int sub = 0; sub < 1; sub++) {
                        unsigned int col = acc_cur * 32 + (unsigned int)(sub * 32);
                        float _tmem_load_0[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x64b.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                            : "r"(taddr + gate_lanes + col));
                        float _tmem_load_1[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x64b.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                            : "r"(taddr + up_lanes + col));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (sub == 0) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            if (elect_sync()) {
                                mbarrier_arrive(acc_empty_addr + (acc_cur) * 8);
                            }
                        }
                        unsigned int out_base = staging_addr + out_cur * 4096 + (unsigned int)thread_off;
                        float g_4;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_4) : "f"(_tmem_load_0[0]), "f"(alpha_e));
                        float u;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u) : "f"(_tmem_load_1[0]), "f"(alpha_e));
                        float _min_34;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_34) : "f"(g_4), "f"(swiglu_limit));
                        g_4 = _min_34;
                        float _min_36;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_36) : "f"(u), "f"(swiglu_limit));
                        float _max_35;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_35) : "f"(_min_36), "f"(neg_limit));
                        u = _max_35;
                        float _exp2_39 = approx_exp2(mul_f32_rn_noftz(g_4, -1.442695f));
                        float _add_38;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(_exp2_39), "f"(1.0f));
                        float _rcp_37 = approx_rcp(_add_38);
                        {
                            __nv_bfloat16 _bval_40 = __float2bfloat16_rn(mul_f32_rn_noftz(u, mul_f32_rn_noftz(g_4, _rcp_37)));
                            uint16_t _bits_40 = *(uint16_t*)&_bval_40;
                            uint32_t _addr_40 = static_cast<uint32_t>(out_base);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_40), "h"(_bits_40) : "memory");
                        }
                        float g_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_0) : "f"(_tmem_load_0[1]), "f"(alpha_e));
                        float u_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_1) : "f"(_tmem_load_1[1]), "f"(alpha_e));
                        float _min_41;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_41) : "f"(g_0), "f"(swiglu_limit));
                        g_0 = _min_41;
                        float _min_43;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_43) : "f"(u_1), "f"(swiglu_limit));
                        float _max_42;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_42) : "f"(_min_43), "f"(neg_limit));
                        u_1 = _max_42;
                        float _exp2_46 = approx_exp2(mul_f32_rn_noftz(g_0, -1.442695f));
                        float _add_45;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(_exp2_46), "f"(1.0f));
                        float _rcp_44 = approx_rcp(_add_45);
                        {
                            __nv_bfloat16 _bval_47 = __float2bfloat16_rn(mul_f32_rn_noftz(u_1, mul_f32_rn_noftz(g_0, _rcp_44)));
                            uint16_t _bits_47 = *(uint16_t*)&_bval_47;
                            uint32_t _addr_47 = static_cast<uint32_t>(out_base + 256);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_47), "h"(_bits_47) : "memory");
                        }
                        float g_2_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_2_1) : "f"(_tmem_load_0[2]), "f"(alpha_e));
                        float u_3;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_3) : "f"(_tmem_load_1[2]), "f"(alpha_e));
                        float _min_48;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_48) : "f"(g_2_1), "f"(swiglu_limit));
                        g_2_1 = _min_48;
                        float _min_50;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_50) : "f"(u_3), "f"(swiglu_limit));
                        float _max_49;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_49) : "f"(_min_50), "f"(neg_limit));
                        u_3 = _max_49;
                        float _exp2_53 = approx_exp2(mul_f32_rn_noftz(g_2_1, -1.442695f));
                        float _add_52;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(_exp2_53), "f"(1.0f));
                        float _rcp_51 = approx_rcp(_add_52);
                        {
                            __nv_bfloat16 _bval_54 = __float2bfloat16_rn(mul_f32_rn_noftz(u_3, mul_f32_rn_noftz(g_2_1, _rcp_51)));
                            uint16_t _bits_54 = *(uint16_t*)&_bval_54;
                            uint32_t _addr_54 = static_cast<uint32_t>(out_base + 512);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_54), "h"(_bits_54) : "memory");
                        }
                        float g_4_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_4_1) : "f"(_tmem_load_0[3]), "f"(alpha_e));
                        float u_5;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_5) : "f"(_tmem_load_1[3]), "f"(alpha_e));
                        float _min_55;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_55) : "f"(g_4_1), "f"(swiglu_limit));
                        g_4_1 = _min_55;
                        float _min_57;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_57) : "f"(u_5), "f"(swiglu_limit));
                        float _max_56;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_56) : "f"(_min_57), "f"(neg_limit));
                        u_5 = _max_56;
                        float _exp2_60 = approx_exp2(mul_f32_rn_noftz(g_4_1, -1.442695f));
                        float _add_59;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(_exp2_60), "f"(1.0f));
                        float _rcp_58 = approx_rcp(_add_59);
                        {
                            __nv_bfloat16 _bval_61 = __float2bfloat16_rn(mul_f32_rn_noftz(u_5, mul_f32_rn_noftz(g_4_1, _rcp_58)));
                            uint16_t _bits_61 = *(uint16_t*)&_bval_61;
                            uint32_t _addr_61 = static_cast<uint32_t>(out_base + 768);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_61), "h"(_bits_61) : "memory");
                        }
                        float g_6;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_6) : "f"(_tmem_load_0[4]), "f"(alpha_e));
                        float u_7;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_7) : "f"(_tmem_load_1[4]), "f"(alpha_e));
                        float _min_62;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_62) : "f"(g_6), "f"(swiglu_limit));
                        g_6 = _min_62;
                        float _min_64;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_64) : "f"(u_7), "f"(swiglu_limit));
                        float _max_63;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_63) : "f"(_min_64), "f"(neg_limit));
                        u_7 = _max_63;
                        float _exp2_67 = approx_exp2(mul_f32_rn_noftz(g_6, -1.442695f));
                        float _add_66;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(_exp2_67), "f"(1.0f));
                        float _rcp_65 = approx_rcp(_add_66);
                        {
                            __nv_bfloat16 _bval_68 = __float2bfloat16_rn(mul_f32_rn_noftz(u_7, mul_f32_rn_noftz(g_6, _rcp_65)));
                            uint16_t _bits_68 = *(uint16_t*)&_bval_68;
                            uint32_t _addr_68 = static_cast<uint32_t>(out_base + 1024);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_68), "h"(_bits_68) : "memory");
                        }
                        float g_8;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_8) : "f"(_tmem_load_0[5]), "f"(alpha_e));
                        float u_9;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_9) : "f"(_tmem_load_1[5]), "f"(alpha_e));
                        float _min_69;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_69) : "f"(g_8), "f"(swiglu_limit));
                        g_8 = _min_69;
                        float _min_71;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_71) : "f"(u_9), "f"(swiglu_limit));
                        float _max_70;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_70) : "f"(_min_71), "f"(neg_limit));
                        u_9 = _max_70;
                        float _exp2_74 = approx_exp2(mul_f32_rn_noftz(g_8, -1.442695f));
                        float _add_73;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(_exp2_74), "f"(1.0f));
                        float _rcp_72 = approx_rcp(_add_73);
                        {
                            __nv_bfloat16 _bval_75 = __float2bfloat16_rn(mul_f32_rn_noftz(u_9, mul_f32_rn_noftz(g_8, _rcp_72)));
                            uint16_t _bits_75 = *(uint16_t*)&_bval_75;
                            uint32_t _addr_75 = static_cast<uint32_t>(out_base + 1280);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_75), "h"(_bits_75) : "memory");
                        }
                        float g_10;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_10) : "f"(_tmem_load_0[6]), "f"(alpha_e));
                        float u_11;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_11) : "f"(_tmem_load_1[6]), "f"(alpha_e));
                        float _min_76;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_76) : "f"(g_10), "f"(swiglu_limit));
                        g_10 = _min_76;
                        float _min_78;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_78) : "f"(u_11), "f"(swiglu_limit));
                        float _max_77;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_77) : "f"(_min_78), "f"(neg_limit));
                        u_11 = _max_77;
                        float _exp2_81 = approx_exp2(mul_f32_rn_noftz(g_10, -1.442695f));
                        float _add_80;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(_exp2_81), "f"(1.0f));
                        float _rcp_79 = approx_rcp(_add_80);
                        {
                            __nv_bfloat16 _bval_82 = __float2bfloat16_rn(mul_f32_rn_noftz(u_11, mul_f32_rn_noftz(g_10, _rcp_79)));
                            uint16_t _bits_82 = *(uint16_t*)&_bval_82;
                            uint32_t _addr_82 = static_cast<uint32_t>(out_base + 1536);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_82), "h"(_bits_82) : "memory");
                        }
                        float g_12;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_12) : "f"(_tmem_load_0[7]), "f"(alpha_e));
                        float u_13;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_13) : "f"(_tmem_load_1[7]), "f"(alpha_e));
                        float _min_83;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_83) : "f"(g_12), "f"(swiglu_limit));
                        g_12 = _min_83;
                        float _min_85;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_85) : "f"(u_13), "f"(swiglu_limit));
                        float _max_84;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_84) : "f"(_min_85), "f"(neg_limit));
                        u_13 = _max_84;
                        float _exp2_88 = approx_exp2(mul_f32_rn_noftz(g_12, -1.442695f));
                        float _add_87;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(_exp2_88), "f"(1.0f));
                        float _rcp_86 = approx_rcp(_add_87);
                        {
                            __nv_bfloat16 _bval_89 = __float2bfloat16_rn(mul_f32_rn_noftz(u_13, mul_f32_rn_noftz(g_12, _rcp_86)));
                            uint16_t _bits_89 = *(uint16_t*)&_bval_89;
                            uint32_t _addr_89 = static_cast<uint32_t>(out_base + 1792);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_89), "h"(_bits_89) : "memory");
                        }
                        float g_14;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_14) : "f"(_tmem_load_0[8]), "f"(alpha_e));
                        float u_15;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_15) : "f"(_tmem_load_1[8]), "f"(alpha_e));
                        float _min_90;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_90) : "f"(g_14), "f"(swiglu_limit));
                        g_14 = _min_90;
                        float _min_92;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_92) : "f"(u_15), "f"(swiglu_limit));
                        float _max_91;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_91) : "f"(_min_92), "f"(neg_limit));
                        u_15 = _max_91;
                        float _exp2_95 = approx_exp2(mul_f32_rn_noftz(g_14, -1.442695f));
                        float _add_94;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(_exp2_95), "f"(1.0f));
                        float _rcp_93 = approx_rcp(_add_94);
                        {
                            __nv_bfloat16 _bval_96 = __float2bfloat16_rn(mul_f32_rn_noftz(u_15, mul_f32_rn_noftz(g_14, _rcp_93)));
                            uint16_t _bits_96 = *(uint16_t*)&_bval_96;
                            uint32_t _addr_96 = static_cast<uint32_t>(out_base + 2048);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_96), "h"(_bits_96) : "memory");
                        }
                        float g_16;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_16) : "f"(_tmem_load_0[9]), "f"(alpha_e));
                        float u_17;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_17) : "f"(_tmem_load_1[9]), "f"(alpha_e));
                        float _min_97;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_97) : "f"(g_16), "f"(swiglu_limit));
                        g_16 = _min_97;
                        float _min_99;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_99) : "f"(u_17), "f"(swiglu_limit));
                        float _max_98;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_98) : "f"(_min_99), "f"(neg_limit));
                        u_17 = _max_98;
                        float _exp2_102 = approx_exp2(mul_f32_rn_noftz(g_16, -1.442695f));
                        float _add_101;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(_exp2_102), "f"(1.0f));
                        float _rcp_100 = approx_rcp(_add_101);
                        {
                            __nv_bfloat16 _bval_103 = __float2bfloat16_rn(mul_f32_rn_noftz(u_17, mul_f32_rn_noftz(g_16, _rcp_100)));
                            uint16_t _bits_103 = *(uint16_t*)&_bval_103;
                            uint32_t _addr_103 = static_cast<uint32_t>(out_base + 2304);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_103), "h"(_bits_103) : "memory");
                        }
                        float g_18;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_18) : "f"(_tmem_load_0[10]), "f"(alpha_e));
                        float u_19;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_19) : "f"(_tmem_load_1[10]), "f"(alpha_e));
                        float _min_104;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_104) : "f"(g_18), "f"(swiglu_limit));
                        g_18 = _min_104;
                        float _min_106;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_106) : "f"(u_19), "f"(swiglu_limit));
                        float _max_105;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_105) : "f"(_min_106), "f"(neg_limit));
                        u_19 = _max_105;
                        float _exp2_109 = approx_exp2(mul_f32_rn_noftz(g_18, -1.442695f));
                        float _add_108;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(_exp2_109), "f"(1.0f));
                        float _rcp_107 = approx_rcp(_add_108);
                        {
                            __nv_bfloat16 _bval_110 = __float2bfloat16_rn(mul_f32_rn_noftz(u_19, mul_f32_rn_noftz(g_18, _rcp_107)));
                            uint16_t _bits_110 = *(uint16_t*)&_bval_110;
                            uint32_t _addr_110 = static_cast<uint32_t>(out_base + 2560);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_110), "h"(_bits_110) : "memory");
                        }
                        float g_20;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_20) : "f"(_tmem_load_0[11]), "f"(alpha_e));
                        float u_21;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_21) : "f"(_tmem_load_1[11]), "f"(alpha_e));
                        float _min_111;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_111) : "f"(g_20), "f"(swiglu_limit));
                        g_20 = _min_111;
                        float _min_113;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_113) : "f"(u_21), "f"(swiglu_limit));
                        float _max_112;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_112) : "f"(_min_113), "f"(neg_limit));
                        u_21 = _max_112;
                        float _exp2_116 = approx_exp2(mul_f32_rn_noftz(g_20, -1.442695f));
                        float _add_115;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(_exp2_116), "f"(1.0f));
                        float _rcp_114 = approx_rcp(_add_115);
                        {
                            __nv_bfloat16 _bval_117 = __float2bfloat16_rn(mul_f32_rn_noftz(u_21, mul_f32_rn_noftz(g_20, _rcp_114)));
                            uint16_t _bits_117 = *(uint16_t*)&_bval_117;
                            uint32_t _addr_117 = static_cast<uint32_t>(out_base + 2816);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_117), "h"(_bits_117) : "memory");
                        }
                        float g_22;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_22) : "f"(_tmem_load_0[12]), "f"(alpha_e));
                        float u_23;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_23) : "f"(_tmem_load_1[12]), "f"(alpha_e));
                        float _min_118;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_118) : "f"(g_22), "f"(swiglu_limit));
                        g_22 = _min_118;
                        float _min_120;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_120) : "f"(u_23), "f"(swiglu_limit));
                        float _max_119;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_119) : "f"(_min_120), "f"(neg_limit));
                        u_23 = _max_119;
                        float _exp2_123 = approx_exp2(mul_f32_rn_noftz(g_22, -1.442695f));
                        float _add_122;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(_exp2_123), "f"(1.0f));
                        float _rcp_121 = approx_rcp(_add_122);
                        {
                            __nv_bfloat16 _bval_124 = __float2bfloat16_rn(mul_f32_rn_noftz(u_23, mul_f32_rn_noftz(g_22, _rcp_121)));
                            uint16_t _bits_124 = *(uint16_t*)&_bval_124;
                            uint32_t _addr_124 = static_cast<uint32_t>(out_base + 3072);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_124), "h"(_bits_124) : "memory");
                        }
                        float g_24;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_24) : "f"(_tmem_load_0[13]), "f"(alpha_e));
                        float u_25;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_25) : "f"(_tmem_load_1[13]), "f"(alpha_e));
                        float _min_125;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_125) : "f"(g_24), "f"(swiglu_limit));
                        g_24 = _min_125;
                        float _min_127;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_127) : "f"(u_25), "f"(swiglu_limit));
                        float _max_126;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_126) : "f"(_min_127), "f"(neg_limit));
                        u_25 = _max_126;
                        float _exp2_130 = approx_exp2(mul_f32_rn_noftz(g_24, -1.442695f));
                        float _add_129;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(_exp2_130), "f"(1.0f));
                        float _rcp_128 = approx_rcp(_add_129);
                        {
                            __nv_bfloat16 _bval_131 = __float2bfloat16_rn(mul_f32_rn_noftz(u_25, mul_f32_rn_noftz(g_24, _rcp_128)));
                            uint16_t _bits_131 = *(uint16_t*)&_bval_131;
                            uint32_t _addr_131 = static_cast<uint32_t>(out_base + 3328);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_131), "h"(_bits_131) : "memory");
                        }
                        float g_26;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_26) : "f"(_tmem_load_0[14]), "f"(alpha_e));
                        float u_27;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_27) : "f"(_tmem_load_1[14]), "f"(alpha_e));
                        float _min_132;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_132) : "f"(g_26), "f"(swiglu_limit));
                        g_26 = _min_132;
                        float _min_134;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_134) : "f"(u_27), "f"(swiglu_limit));
                        float _max_133;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_133) : "f"(_min_134), "f"(neg_limit));
                        u_27 = _max_133;
                        float _exp2_137 = approx_exp2(mul_f32_rn_noftz(g_26, -1.442695f));
                        float _add_136;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(_exp2_137), "f"(1.0f));
                        float _rcp_135 = approx_rcp(_add_136);
                        {
                            __nv_bfloat16 _bval_138 = __float2bfloat16_rn(mul_f32_rn_noftz(u_27, mul_f32_rn_noftz(g_26, _rcp_135)));
                            uint16_t _bits_138 = *(uint16_t*)&_bval_138;
                            uint32_t _addr_138 = static_cast<uint32_t>(out_base + 3584);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_138), "h"(_bits_138) : "memory");
                        }
                        float g_28;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_28) : "f"(_tmem_load_0[15]), "f"(alpha_e));
                        float u_29;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_29) : "f"(_tmem_load_1[15]), "f"(alpha_e));
                        float _min_139;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_139) : "f"(g_28), "f"(swiglu_limit));
                        g_28 = _min_139;
                        float _min_141;
                        asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_141) : "f"(u_29), "f"(swiglu_limit));
                        float _max_140;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_140) : "f"(_min_141), "f"(neg_limit));
                        u_29 = _max_140;
                        float _exp2_144 = approx_exp2(mul_f32_rn_noftz(g_28, -1.442695f));
                        float _add_143;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(_exp2_144), "f"(1.0f));
                        float _rcp_142 = approx_rcp(_add_143);
                        {
                            __nv_bfloat16 _bval_145 = __float2bfloat16_rn(mul_f32_rn_noftz(u_29, mul_f32_rn_noftz(g_28, _rcp_142)));
                            uint16_t _bits_145 = *(uint16_t*)&_bval_145;
                            uint32_t _addr_145 = static_cast<uint32_t>(out_base + 3840);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_145), "h"(_bits_145) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&fc1_out_st)), "r"(feature0), "r"(row0 + sub * 32), "r"(staging_addr + out_cur * 4096), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        out_cur += 1;
                        if (out_cur == 2) { out_cur = 0; }
                    }
                    if (warp == 0) {
                        if (elect_sync()) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(fc1_done) + (t_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                } else {
                    float alpha_e_1 = fc2_alpha[expert_4];
                    int col_m = m * 128;
                    int live_2 = tile_mn_limit[t_1] - row0;
                    #pragma unroll
                    for (int sub_1 = 0; sub_1 < 2; sub_1++) {
                        unsigned int col_1 = acc_cur * 32 + (unsigned int)(sub_1 * 16);
                        float _tmem_load_2[16];
                        tmem_ld_x16(&_tmem_load_2[0], taddr + gate_lanes + col_1);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (sub_1 == 1) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            if (elect_sync()) {
                                mbarrier_arrive(acc_empty_addr + (acc_cur) * 8);
                            }
                        }
                        unsigned int base_1 = staging_addr + out_cur * 4096;
                        {
                            __nv_bfloat16 _bval_146 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[0], alpha_e_1));
                            uint16_t _bits_146 = *(uint16_t*)&_bval_146;
                            uint32_t _addr_146 = static_cast<uint32_t>(base_1 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_146), "h"(_bits_146) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_147 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[1], alpha_e_1));
                            uint16_t _bits_147 = *(uint16_t*)&_bval_147;
                            uint32_t _addr_147 = static_cast<uint32_t>(base_1 + 256 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_147), "h"(_bits_147) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_148 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[2], alpha_e_1));
                            uint16_t _bits_148 = *(uint16_t*)&_bval_148;
                            uint32_t _addr_148 = static_cast<uint32_t>(base_1 + 512 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_148), "h"(_bits_148) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_149 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[3], alpha_e_1));
                            uint16_t _bits_149 = *(uint16_t*)&_bval_149;
                            uint32_t _addr_149 = static_cast<uint32_t>(base_1 + 768 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_149), "h"(_bits_149) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_150 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[4], alpha_e_1));
                            uint16_t _bits_150 = *(uint16_t*)&_bval_150;
                            uint32_t _addr_150 = static_cast<uint32_t>(base_1 + 1024 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_150), "h"(_bits_150) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_151 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[5], alpha_e_1));
                            uint16_t _bits_151 = *(uint16_t*)&_bval_151;
                            uint32_t _addr_151 = static_cast<uint32_t>(base_1 + 1280 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_151), "h"(_bits_151) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_152 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[6], alpha_e_1));
                            uint16_t _bits_152 = *(uint16_t*)&_bval_152;
                            uint32_t _addr_152 = static_cast<uint32_t>(base_1 + 1536 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_152), "h"(_bits_152) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_153 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[7], alpha_e_1));
                            uint16_t _bits_153 = *(uint16_t*)&_bval_153;
                            uint32_t _addr_153 = static_cast<uint32_t>(base_1 + 1792 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_153), "h"(_bits_153) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_154 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[8], alpha_e_1));
                            uint16_t _bits_154 = *(uint16_t*)&_bval_154;
                            uint32_t _addr_154 = static_cast<uint32_t>(base_1 + 2048 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_154), "h"(_bits_154) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_155 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[9], alpha_e_1));
                            uint16_t _bits_155 = *(uint16_t*)&_bval_155;
                            uint32_t _addr_155 = static_cast<uint32_t>(base_1 + 2304 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_155), "h"(_bits_155) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_156 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[10], alpha_e_1));
                            uint16_t _bits_156 = *(uint16_t*)&_bval_156;
                            uint32_t _addr_156 = static_cast<uint32_t>(base_1 + 2560 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_156), "h"(_bits_156) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_157 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[11], alpha_e_1));
                            uint16_t _bits_157 = *(uint16_t*)&_bval_157;
                            uint32_t _addr_157 = static_cast<uint32_t>(base_1 + 2816 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_157), "h"(_bits_157) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_158 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[12], alpha_e_1));
                            uint16_t _bits_158 = *(uint16_t*)&_bval_158;
                            uint32_t _addr_158 = static_cast<uint32_t>(base_1 + 3072 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_158), "h"(_bits_158) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_159 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[13], alpha_e_1));
                            uint16_t _bits_159 = *(uint16_t*)&_bval_159;
                            uint32_t _addr_159 = static_cast<uint32_t>(base_1 + 3328 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_159), "h"(_bits_159) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_160 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[14], alpha_e_1));
                            uint16_t _bits_160 = *(uint16_t*)&_bval_160;
                            uint32_t _addr_160 = static_cast<uint32_t>(base_1 + 3584 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_160), "h"(_bits_160) : "memory");
                        }
                        {
                            __nv_bfloat16 _bval_161 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_2[15], alpha_e_1));
                            uint16_t _bits_161 = *(uint16_t*)&_bval_161;
                            uint32_t _addr_161 = static_cast<uint32_t>(base_1 + 3840 + ch_off);
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_161), "h"(_bits_161) : "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int sub_row = sub_1 * 16 + piece_row;
                        if (sub_row < live_2) {
                            unsigned int piece_addr = base_1 + (unsigned int)(piece_row * 256 + piece_off);
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&piece[0])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(0) + 3]))
                                : "r"(piece_addr));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&piece[4])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&piece[(4) + 3]))
                                : "r"(piece_addr + 16));
                            int route_g = route_src[row0 + sub_row];
                            int src_4 = route_g / routes_per_rank;
                            int src_route = route_g - src_4 * routes_per_rank;
                            unsigned int* remote_terms = reinterpret_cast<unsigned int*>(route_terms_peers.ptrs[src_4]);
                            unsigned long long term_word = ((unsigned long long)src_route * H64 + (unsigned long long)col_m) / 2 + (unsigned long long)piece_words;
                            {
                                auto _gst_162 = remote_terms + term_word;
                                __builtin_assume(__isGlobal(_gst_162));
                                reinterpret_cast<int4*>(_gst_162)[0] = reinterpret_cast<int4*>(piece)[0];
                            }
                            {
                                auto _gst_163 = remote_terms + (term_word + 4);
                                __builtin_assume(__isGlobal(_gst_163));
                                reinterpret_cast<int4*>(_gst_163)[0] = reinterpret_cast<int4*>(piece + (4))[0];
                            }
                        }
                        out_cur += 1;
                        if (out_cur == 2) { out_cur = 0; }
                    }
                }
                acc_cur += 1;
                if (acc_cur == 2) { acc_cur = 0; _phase_acc_full ^= 1; }
            }
            if (warp == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("fence.release.sys;" ::: "memory");
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sync_words) + (4))), "r"(static_cast<unsigned int>(1)) : "memory");
                }
            }
            if (bid_0 == 0 && warp == 0) {
                if (elect_sync()) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(sync_words) + (4))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>((unsigned int)num_ctas)) >= static_cast<unsigned int>(2147483648u));
                    }
                }
                __syncwarp();
                if (world > lane) {
                    asm volatile("red.release.sys.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(completion_peers.ptrs[lane]))), "r"(static_cast<unsigned int>(1)) : "memory");
                }
                if (elect_sync()) {
                    asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(sync_words + 0)), "r"(static_cast<unsigned int>(0)) : "memory");
                    asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(sync_words + 1)), "r"(static_cast<unsigned int>(0)) : "memory");
                    asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(sync_words + 2)), "r"(static_cast<unsigned int>(0)) : "memory");
                    asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(sync_words + 3)), "r"(static_cast<unsigned int>(0)) : "memory");
                    asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(sync_words + 4)), "r"(static_cast<unsigned int>(0)) : "memory");
                    asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(epoch)), "r"(static_cast<unsigned int>(1)) : "memory");
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            asm volatile("barrier.sync 12, 512;" ::: "memory");
            int tiles_2 = tile_count[0];
            int fc1_items_1 = tiles_2 * fc1_m_tiles;
            int wl_b_1 = (int)bid;
            int wl_P_1 = (int)num_bids;
            int wl_fc2_1 = tiles_2 * fc2_m_tiles;
            int wl_r_1 = fc1_items_1 % wl_P_1;
            int wl_L_1 = wl_P_1 - wl_r_1;
            int wl_delta_1 = (H / 256 + I / 256 / 2) / (I / 256);
            int wl_n1_1 = wl_delta_1 * wl_L_1;
            if (wl_n1_1 > wl_fc2_1) {
                wl_n1_1 = wl_fc2_1;
            }
            int wl_c1_1 = (fc1_items_1 - wl_b_1 + wl_P_1 - 1) / wl_P_1;
            int wl_p1_1 = ((wl_b_1 >= wl_r_1) ? (wl_n1_1 - (wl_b_1 - wl_r_1) + wl_L_1 - 1) / wl_L_1 : 0);
            int wl_p2_1 = (wl_fc2_1 - wl_n1_1 - wl_b_1 + wl_P_1 - 1) / wl_P_1;
            int wl_items_1 = wl_c1_1 + wl_p1_1 + wl_p2_1;
            unsigned int a_cur = 0;
            unsigned int b_cur = 0;
            unsigned int acc_cur_1 = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int w_1 = 0; w_1 < wl_items_1; w_1++) {
                    int w_i_1 = (int)w_1;
                    int wl_j_1 = w_i_1 - wl_c1_1;
                    if (wl_j_1 < 0) {
                        w_i_1 = bid + w_i_1 * num_bids;
                    } else if (wl_j_1 < wl_p1_1) {
                        w_i_1 = fc1_items_1 + wl_j_1 * (num_bids - wl_r_1) + (bid - wl_r_1);
                    } else {
                        w_i_1 = fc1_items_1 + wl_n1_1 + (wl_j_1 - wl_p1_1) * num_bids + bid;
                    }
                    int is_fc1_1 = 0;
                    int local_5 = w_i_1 - fc1_items_1;
                    if (w_i_1 < fc1_items_1) {
                        is_fc1_1 = 1;
                        local_5 = w_i_1;
                    }
                    unsigned int ww_1 = (unsigned int)local_5;
                    uint32_t _fast_div_q_6 = (fc1_m_div.divisor != 1) ? (__umulhi((uint32_t)(ww_1), fc1_m_div.multiplier) >> fc1_m_div.shift_right) : (uint32_t)(ww_1);
                    uint32_t _fast_div_r_6 = (uint32_t)(ww_1) - _fast_div_q_6 * (uint32_t)(fc1_m_div.divisor);
                    uint32_t _fast_div_q_7 = (fc2_m_div.divisor != 1) ? (__umulhi((uint32_t)(ww_1), fc2_m_div.multiplier) >> fc2_m_div.shift_right) : (uint32_t)(ww_1);
                    uint32_t _fast_div_r_7 = (uint32_t)(ww_1) - _fast_div_q_7 * (uint32_t)(fc2_m_div.divisor);
                    int t_2 = (int)_fast_div_q_7;
                    int m_1 = (int)_fast_div_r_7;
                    if (is_fc1_1 != 0) {
                        t_2 = (int)_fast_div_q_6;
                        m_1 = (int)_fast_div_r_6;
                    }
                    int num_k = I / 256;
                    if (is_fc1_1 != 0) {
                        num_k = H / 256;
                    }
                    mbarrier_wait(acc_empty_addr + (acc_cur_1) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int kt = 0; kt < num_k; kt++) {
                        mbarrier_wait(a_full_addr + (a_cur) * 8, _phase_a_full);
                        mbarrier_wait(b_full_addr + (b_cur) * 8, _phase_b_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (b_cur) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_acc + (acc_cur_1 * 32))), "r"(_smem_desc_lo_0), "r"((unsigned int)tmem_a_bf16 + a_cur * 128), "r"(((((kt == 0) ? 1 : 0)) ? 0 : 1)) : "memory");
                        tcgen05_commit(b_empty_addr + (b_cur) * 8);
                        tcgen05_commit(a_empty_addr + (a_cur) * 8);
                        a_cur += 1;
                        if (a_cur == 3) { a_cur = 0; _phase_a_full ^= 1; }
                        b_cur += 1;
                        if (b_cur == 6) { b_cur = 0; _phase_b_full ^= 1; }
                    }
                    tcgen05_commit(acc_full_addr + (acc_cur_1) * 8);
                    acc_cur_1 += 1;
                    if (acc_cur_1 == 2) { acc_cur_1 = 0; _phase_acc_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: puller ----
    if (warp == 5) {
        { // puller_main
            const int world_1 = mega_pg_world;
            const int rank_1 = mega_pg_rank;
            const int fw_1 = 4;
            const int ftid_1 = fw_1 * 32 + lane;
            const int num_ctas_1 = num_bids;
            const int bid_0_1 = bid;
            const int linear_1 = bid_0_1 * 192 + ftid_1;
            const int stride_1 = num_ctas_1 * 192;
            const int max_tiles_1 = (world_1 * cap * 8 + 31 * E_local) / 32;
            const unsigned int lane_lt_1 = (1 << (unsigned int)lane) - 1;
            const int zero_i_1 = 0;
            const int hist_quad_words_1 = E_local * 4;
            #pragma unroll 1
            for (int t_3 = linear_1; t_3 < max_tiles_1; t_3 += stride_1) {
                asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(fc1_ready + t_3)), "r"(static_cast<unsigned int>(0)) : "memory");
                asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(fc1_done + t_3)), "r"(static_cast<unsigned int>(0)) : "memory");
                asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(fc2_done + t_3)), "r"(static_cast<unsigned int>(0)) : "memory");
            }
            if (linear_1 == 0) {
                reinterpret_cast<int*>(published_meta)[0] = T_r;
            }
            if (world_1 > 1) {
                #pragma unroll 1
                for (int index_1 = linear_1 * 8; index_1 < T_r * H; index_1 += stride_1 * 8) {
                    float _vec_load_0[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(hidden + index_1 + 0);
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    unsigned int _packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_0[_lp*2 + 0], _vec_load_0[_lp*2+1 + 0]));
                        _packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_0[_lp*2 + 2], _vec_load_0[_lp*2+1 + 2]));
                        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_0[_lp*2 + 4], _vec_load_0[_lp*2+1 + 4]));
                        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_0[_lp*2 + 6], _vec_load_0[_lp*2+1 + 6]));
                        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    {
                        auto _gst_1 = reinterpret_cast<__nv_bfloat16*>(published_hidden) + index_1;
                        __builtin_assume(__isGlobal(_gst_1));
                        reinterpret_cast<int4*>(_gst_1)[0] = reinterpret_cast<int4*>(_packed_0)[0];
                    }
                }
                #pragma unroll 1
                for (int route_1 = linear_1; route_1 < T_r * 8; route_1 += stride_1) {
                    int route_id_1 = topk_ids[route_1];
                    float route_weight_1 = topk_weights[route_1];
                    reinterpret_cast<int*>(published_ids)[route_1] = route_id_1;
                    reinterpret_cast<float*>(published_weights)[route_1] = route_weight_1;
                }
                asm volatile("barrier.sync 11, 192;" ::: "memory");
                asm volatile("barrier.sync 11, 192;" ::: "memory");
                asm volatile("barrier.sync 11, 192;" ::: "memory");
            }
            #pragma unroll 1
            for (int e_8 = lane; e_8 < E_local; e_8 += 32) {
                {
                    uint32_t _ival_2 = static_cast<uint32_t>(zero_i_1);
                    uint32_t _addr_2 = static_cast<uint32_t>(hist_s_addr + (unsigned int)((fw_1 * 256 + e_8) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_2), "r"(_ival_2) : "memory");
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            const int tokens_total_1 = world_1 * cap;
            const int tokens_per_cta_1 = (tokens_total_1 + num_ctas_1 - 1) / num_ctas_1;
            const int cta_tok0_1 = bid_0_1 * tokens_per_cta_1;
            int cta_tok1_raw_1 = cta_tok0_1 + tokens_per_cta_1;
            int cta_tok1_1 = ((cta_tok1_raw_1 < tokens_total_1) ? cta_tok1_raw_1 : tokens_total_1);
            const int tokens_per_warp_1 = (tokens_per_cta_1 + 6 - 1) / 6;
            const int w_tok0_1 = cta_tok0_1 + fw_1 * tokens_per_warp_1;
            int w_tok1_raw_1 = w_tok0_1 + tokens_per_warp_1;
            int w_tok1_1 = ((w_tok1_raw_1 < cta_tok1_1) ? w_tok1_raw_1 : cta_tok1_1);
            const int my_slot_1 = lane % 8;
            const int my_tok_off_1 = lane / 8;
            const int scan_all_1 = ((world_1 == 1 && T_r * 8 <= 4096) ? 1 : 0);
            const int scan_pre1_1 = ((cta_tok0_1 < T_r) ? cta_tok0_1 : T_r);
            if (scan_all_1 != 0) {
                #pragma unroll 1
                for (int e_9 = lane; e_9 < E_local; e_9 += 32) {
                    {
                        uint32_t _ival_3 = static_cast<uint32_t>(zero_i_1);
                        uint32_t _addr_3 = static_cast<uint32_t>(pre_s_addr + (unsigned int)((fw_1 * 256 + e_9) * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_3), "r"(_ival_3) : "memory");
                    }
                    {
                        uint32_t _ival_4 = static_cast<uint32_t>(zero_i_1);
                        uint32_t _addr_4 = static_cast<uint32_t>(suf_s_addr + (unsigned int)((fw_1 * 256 + e_9) * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_4), "r"(_ival_4) : "memory");
                    }
                }
                __syncwarp();
            }
            #pragma unroll 1
            for (int tok_4 = w_tok0_1; tok_4 < w_tok1_1; tok_4 += 4) {
                int my_tok_4 = tok_4 + my_tok_off_1;
                int src_5 = my_tok_4 / cap;
                int token_4 = my_tok_4 - src_5 * cap;
                int g_5 = my_tok_4 * 8 + my_slot_1;
                int ident_4 = -1;
                if (my_tok_4 < w_tok1_1) {
                    if (token_4 < meta_s[src_5]) {
                        if (world_1 > 1) {
                            ident_4 = reinterpret_cast<int*>(published_ids_peers.ptrs[src_5])[token_4 * 8 + my_slot_1];
                        } else {
                            ident_4 = topk_ids[token_4 * 8 + my_slot_1];
                        }
                    }
                }
                int local_6 = 0;
                int expert_5 = 0;
                if (ident_4 >= 0) {
                    if (ident_4 / E_local == rank_1) {
                        local_6 = 1;
                        expert_5 = ident_4 - rank_1 * E_local;
                    }
                }
                int key_2 = ((local_6 != 0) ? expert_5 : -1);
                unsigned int live_3 = __ballot_sync(0xFFFFFFFF, local_6 != 0);
                int _count_leading_zeros_5 = __clz(live_3);
                int trips_2 = 32 - _count_leading_zeros_5;
                unsigned int peers_2 = 0;
                #pragma unroll 1
                for (int lane0_2 = 0; lane0_2 < trips_2; lane0_2 += 4) {
                    int held_0_2 = __shfl_sync(0xFFFFFFFF, key_2, lane0_2 + 0);
                    unsigned int same_0_2 = ((held_0_2 == key_2) ? 1 : 0);
                    peers_2 = peers_2 | same_0_2 << (unsigned int)(lane0_2 + 0);
                    int held_1_2 = __shfl_sync(0xFFFFFFFF, key_2, lane0_2 + 1);
                    unsigned int same_1_2 = ((held_1_2 == key_2) ? 1 : 0);
                    peers_2 = peers_2 | same_1_2 << (unsigned int)(lane0_2 + 1);
                    int held_2_2 = __shfl_sync(0xFFFFFFFF, key_2, lane0_2 + 2);
                    unsigned int same_2_2 = ((held_2_2 == key_2) ? 1 : 0);
                    peers_2 = peers_2 | same_2_2 << (unsigned int)(lane0_2 + 2);
                    int held_3_2 = __shfl_sync(0xFFFFFFFF, key_2, lane0_2 + 3);
                    unsigned int same_3_2 = ((held_3_2 == key_2) ? 1 : 0);
                    peers_2 = peers_2 | same_3_2 << (unsigned int)(lane0_2 + 3);
                }
                unsigned int group_2 = peers_2 & live_3;
                int members_2 = (int)(unsigned int)__popc(group_2);
                int lower_2 = (int)(unsigned int)__popc(group_2 & lane_lt_1);
                if (local_6 != 0 && lower_2 == 0) {
                    unsigned int count_addr_1 = hist_s_addr + (unsigned int)((fw_1 * 256 + expert_5) * 4);
                    int counted_1 = hist_s[fw_1 * 256 + expert_5];
                    {
                        uint32_t _ival_6 = static_cast<uint32_t>(counted_1 + members_2);
                        uint32_t _addr_6 = static_cast<uint32_t>(count_addr_1);
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_6), "r"(_ival_6) : "memory");
                    }
                }
                __syncwarp();
            }
            if (scan_all_1 != 0) {
                #pragma unroll 1
                for (int tok_5 = fw_1 * 4; tok_5 < scan_pre1_1; tok_5 += 24) {
                    int my_tok_5 = tok_5 + my_tok_off_1;
                    int src_6 = my_tok_5 / cap;
                    int token_5 = my_tok_5 - src_6 * cap;
                    int g_7 = my_tok_5 * 8 + my_slot_1;
                    int ident_5 = -1;
                    if (my_tok_5 < scan_pre1_1) {
                        if (token_5 < meta_s[src_6]) {
                            if (world_1 > 1) {
                                ident_5 = reinterpret_cast<int*>(published_ids_peers.ptrs[src_6])[token_5 * 8 + my_slot_1];
                            } else {
                                ident_5 = topk_ids[token_5 * 8 + my_slot_1];
                            }
                        }
                    }
                    int local_7 = 0;
                    int expert_6 = 0;
                    if (ident_5 >= 0) {
                        if (ident_5 / E_local == rank_1) {
                            local_7 = 1;
                            expert_6 = ident_5 - rank_1 * E_local;
                        }
                    }
                    int _atomic_old_0 = 0;
                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.s32 %0, [%1], %2;\n}"
                        : "+r"(_atomic_old_0) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&pre_s[fw_1 * 256 + expert_6]))), "r"(static_cast<int>(1)), "r"(static_cast<uint32_t>(local_7 != 0)) : "memory");
                }
                #pragma unroll 1
                for (int tok_6 = cta_tok1_1 + fw_1 * 4; tok_6 < T_r; tok_6 += 24) {
                    int my_tok_6 = tok_6 + my_tok_off_1;
                    int src_7 = my_tok_6 / cap;
                    int token_6 = my_tok_6 - src_7 * cap;
                    int g_9 = my_tok_6 * 8 + my_slot_1;
                    int ident_6 = -1;
                    if (my_tok_6 < T_r) {
                        if (token_6 < meta_s[src_7]) {
                            if (world_1 > 1) {
                                ident_6 = reinterpret_cast<int*>(published_ids_peers.ptrs[src_7])[token_6 * 8 + my_slot_1];
                            } else {
                                ident_6 = topk_ids[token_6 * 8 + my_slot_1];
                            }
                        }
                    }
                    int local_8 = 0;
                    int expert_7 = 0;
                    if (ident_6 >= 0) {
                        if (ident_6 / E_local == rank_1) {
                            local_8 = 1;
                            expert_7 = ident_6 - rank_1 * E_local;
                        }
                    }
                    int _atomic_old_1 = 0;
                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.s32 %0, [%1], %2;\n}"
                        : "+r"(_atomic_old_1) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&suf_s[fw_1 * 256 + expert_7]))), "r"(static_cast<int>(1)), "r"(static_cast<uint32_t>(local_8 != 0)) : "memory");
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            #pragma unroll 1
            for (int e_10 = ftid_1; e_10 < E_local; e_10 += 192) {
                int running_2 = 0;
                {
                    uint32_t _ival_7 = static_cast<uint32_t>(running_2);
                    uint32_t _addr_7 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)(e_10 * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_7), "r"(_ival_7) : "memory");
                }
                running_2 = running_2 + hist_s[e_10];
                {
                    uint32_t _ival_8 = static_cast<uint32_t>(running_2);
                    uint32_t _addr_8 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((256 + e_10) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_8), "r"(_ival_8) : "memory");
                }
                running_2 = running_2 + hist_s[256 + e_10];
                {
                    uint32_t _ival_9 = static_cast<uint32_t>(running_2);
                    uint32_t _addr_9 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((512 + e_10) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_9), "r"(_ival_9) : "memory");
                }
                running_2 = running_2 + hist_s[512 + e_10];
                {
                    uint32_t _ival_10 = static_cast<uint32_t>(running_2);
                    uint32_t _addr_10 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((768 + e_10) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_10), "r"(_ival_10) : "memory");
                }
                running_2 = running_2 + hist_s[768 + e_10];
                {
                    uint32_t _ival_11 = static_cast<uint32_t>(running_2);
                    uint32_t _addr_11 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((1024 + e_10) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_11), "r"(_ival_11) : "memory");
                }
                running_2 = running_2 + hist_s[1024 + e_10];
                {
                    uint32_t _ival_12 = static_cast<uint32_t>(running_2);
                    uint32_t _addr_12 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((1280 + e_10) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_12), "r"(_ival_12) : "memory");
                }
                running_2 = running_2 + hist_s[1280 + e_10];
                if (scan_all_1 != 0) {
                    int before_1 = 0;
                    int after_1 = 0;
                    before_1 = before_1 + pre_s[e_10];
                    after_1 = after_1 + suf_s[e_10];
                    before_1 = before_1 + pre_s[256 + e_10];
                    after_1 = after_1 + suf_s[256 + e_10];
                    before_1 = before_1 + pre_s[512 + e_10];
                    after_1 = after_1 + suf_s[512 + e_10];
                    before_1 = before_1 + pre_s[768 + e_10];
                    after_1 = after_1 + suf_s[768 + e_10];
                    before_1 = before_1 + pre_s[1024 + e_10];
                    after_1 = after_1 + suf_s[1024 + e_10];
                    before_1 = before_1 + pre_s[1280 + e_10];
                    after_1 = after_1 + suf_s[1280 + e_10];
                    {
                        uint32_t _ival_13 = static_cast<uint32_t>(before_1);
                        uint32_t _addr_13 = static_cast<uint32_t>(cbase_s_addr + (unsigned int)(e_10 * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_13), "r"(_ival_13) : "memory");
                    }
                    {
                        uint32_t _ival_14 = static_cast<uint32_t>(before_1 + running_2 + after_1);
                        uint32_t _addr_14 = static_cast<uint32_t>(total_s_addr + (unsigned int)(e_10 * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_14), "r"(_ival_14) : "memory");
                    }
                } else {
                    cta_hist[bid_0_1 / 4 * hist_quad_words_1 + e_10 * 4 + bid_0_1 % 4] = running_2;
                }
            }
            int quads_1[32];
            if (scan_all_1 == 0) {
                asm volatile("barrier.sync 11, 192;" ::: "memory");
                asm volatile("barrier.sync 11, 192;" ::: "memory");
                #pragma unroll 1
                for (int e_11 = ftid_1; e_11 < E_local; e_11 += 192) {
                    int total_1 = 0;
                    int base_2 = 0;
                    #pragma unroll 1
                    for (int c0_1 = 0; c0_1 < num_ctas_1; c0_1 += 32) {
                        {
                            const int4* _ivptr_15 = reinterpret_cast<const int4*>(cta_hist + (c0_1 / 4 * hist_quad_words_1 + e_11 * 4) + 0);
                            int4 _ivld_15;
                            _ivld_15 = *_ivptr_15;
                            quads_1[0 + 0] = _ivld_15.x;
                            quads_1[0 + 1] = _ivld_15.y;
                            quads_1[0 + 2] = _ivld_15.z;
                            quads_1[0 + 3] = _ivld_15.w;
                        }
                        {
                            const int4* _ivptr_16 = reinterpret_cast<const int4*>(cta_hist + ((c0_1 / 4 + 1) * hist_quad_words_1 + e_11 * 4) + 0);
                            int4 _ivld_16;
                            _ivld_16 = *_ivptr_16;
                            quads_1[4 + 0] = _ivld_16.x;
                            quads_1[4 + 1] = _ivld_16.y;
                            quads_1[4 + 2] = _ivld_16.z;
                            quads_1[4 + 3] = _ivld_16.w;
                        }
                        {
                            const int4* _ivptr_17 = reinterpret_cast<const int4*>(cta_hist + ((c0_1 / 4 + 2) * hist_quad_words_1 + e_11 * 4) + 0);
                            int4 _ivld_17;
                            _ivld_17 = *_ivptr_17;
                            quads_1[8 + 0] = _ivld_17.x;
                            quads_1[8 + 1] = _ivld_17.y;
                            quads_1[8 + 2] = _ivld_17.z;
                            quads_1[8 + 3] = _ivld_17.w;
                        }
                        {
                            const int4* _ivptr_18 = reinterpret_cast<const int4*>(cta_hist + ((c0_1 / 4 + 3) * hist_quad_words_1 + e_11 * 4) + 0);
                            int4 _ivld_18;
                            _ivld_18 = *_ivptr_18;
                            quads_1[12 + 0] = _ivld_18.x;
                            quads_1[12 + 1] = _ivld_18.y;
                            quads_1[12 + 2] = _ivld_18.z;
                            quads_1[12 + 3] = _ivld_18.w;
                        }
                        {
                            const int4* _ivptr_19 = reinterpret_cast<const int4*>(cta_hist + ((c0_1 / 4 + 4) * hist_quad_words_1 + e_11 * 4) + 0);
                            int4 _ivld_19;
                            _ivld_19 = *_ivptr_19;
                            quads_1[16 + 0] = _ivld_19.x;
                            quads_1[16 + 1] = _ivld_19.y;
                            quads_1[16 + 2] = _ivld_19.z;
                            quads_1[16 + 3] = _ivld_19.w;
                        }
                        {
                            const int4* _ivptr_20 = reinterpret_cast<const int4*>(cta_hist + ((c0_1 / 4 + 5) * hist_quad_words_1 + e_11 * 4) + 0);
                            int4 _ivld_20;
                            _ivld_20 = *_ivptr_20;
                            quads_1[20 + 0] = _ivld_20.x;
                            quads_1[20 + 1] = _ivld_20.y;
                            quads_1[20 + 2] = _ivld_20.z;
                            quads_1[20 + 3] = _ivld_20.w;
                        }
                        {
                            const int4* _ivptr_21 = reinterpret_cast<const int4*>(cta_hist + ((c0_1 / 4 + 6) * hist_quad_words_1 + e_11 * 4) + 0);
                            int4 _ivld_21;
                            _ivld_21 = *_ivptr_21;
                            quads_1[24 + 0] = _ivld_21.x;
                            quads_1[24 + 1] = _ivld_21.y;
                            quads_1[24 + 2] = _ivld_21.z;
                            quads_1[24 + 3] = _ivld_21.w;
                        }
                        {
                            const int4* _ivptr_22 = reinterpret_cast<const int4*>(cta_hist + ((c0_1 / 4 + 7) * hist_quad_words_1 + e_11 * 4) + 0);
                            int4 _ivld_22;
                            _ivld_22 = *_ivptr_22;
                            quads_1[28 + 0] = _ivld_22.x;
                            quads_1[28 + 1] = _ivld_22.y;
                            quads_1[28 + 2] = _ivld_22.z;
                            quads_1[28 + 3] = _ivld_22.w;
                        }
                        total_1 = total_1 + ((num_ctas_1 > c0_1) ? quads_1[0] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1) ? quads_1[0] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 1) ? quads_1[1] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 1) ? quads_1[1] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 2) ? quads_1[2] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 2) ? quads_1[2] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 3) ? quads_1[3] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 3) ? quads_1[3] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 4) ? quads_1[4] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 4) ? quads_1[4] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 5) ? quads_1[5] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 5) ? quads_1[5] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 6) ? quads_1[6] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 6) ? quads_1[6] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 7) ? quads_1[7] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 7) ? quads_1[7] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 8) ? quads_1[8] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 8) ? quads_1[8] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 9) ? quads_1[9] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 9) ? quads_1[9] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 10) ? quads_1[10] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 10) ? quads_1[10] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 11) ? quads_1[11] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 11) ? quads_1[11] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 12) ? quads_1[12] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 12) ? quads_1[12] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 13) ? quads_1[13] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 13) ? quads_1[13] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 14) ? quads_1[14] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 14) ? quads_1[14] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 15) ? quads_1[15] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 15) ? quads_1[15] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 16) ? quads_1[16] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 16) ? quads_1[16] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 17) ? quads_1[17] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 17) ? quads_1[17] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 18) ? quads_1[18] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 18) ? quads_1[18] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 19) ? quads_1[19] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 19) ? quads_1[19] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 20) ? quads_1[20] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 20) ? quads_1[20] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 21) ? quads_1[21] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 21) ? quads_1[21] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 22) ? quads_1[22] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 22) ? quads_1[22] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 23) ? quads_1[23] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 23) ? quads_1[23] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 24) ? quads_1[24] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 24) ? quads_1[24] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 25) ? quads_1[25] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 25) ? quads_1[25] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 26) ? quads_1[26] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 26) ? quads_1[26] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 27) ? quads_1[27] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 27) ? quads_1[27] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 28) ? quads_1[28] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 28) ? quads_1[28] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 29) ? quads_1[29] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 29) ? quads_1[29] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 30) ? quads_1[30] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 30) ? quads_1[30] : 0);
                        total_1 = total_1 + ((num_ctas_1 > c0_1 + 31) ? quads_1[31] : 0);
                        base_2 = base_2 + ((bid_0_1 > c0_1 + 31) ? quads_1[31] : 0);
                    }
                    {
                        uint32_t _ival_23 = static_cast<uint32_t>(total_1);
                        uint32_t _addr_23 = static_cast<uint32_t>(total_s_addr + (unsigned int)(e_11 * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_23), "r"(_ival_23) : "memory");
                    }
                    {
                        uint32_t _ival_24 = static_cast<uint32_t>(base_2);
                        uint32_t _addr_24 = static_cast<uint32_t>(cbase_s_addr + (unsigned int)(e_11 * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_24), "r"(_ival_24) : "memory");
                    }
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            const int active_rows_1 = seg_s[E_local];
            const int active_tiles_1 = active_rows_1 / 32;
            if (bid_0_1 == 0) {
                #pragma unroll 1
                for (int e_12 = ftid_1; e_12 < E_local + 1; e_12 += 192) {
                    expert_seg[e_12] = seg_s[e_12];
                }
                #pragma unroll 1
                for (int e_13 = ftid_1; e_13 < E_local; e_13 += 192) {
                    int rows_1 = total_s[e_13];
                    int seg0_1 = seg_s[e_13];
                    int first_1 = seg0_1 / 32;
                    int tiles_3 = (rows_1 + 31) / 32;
                    #pragma unroll 1
                    for (int k_1 = 0; k_1 < tiles_3; k_1++) {
                        tile_expert[first_1 + k_1] = e_13;
                        int limit_1 = (k_1 + 1) * 32;
                        tile_mn_limit[first_1 + k_1] = seg0_1 + ((rows_1 < limit_1) ? rows_1 : limit_1);
                    }
                    #pragma unroll 1
                    for (int pad_row_1 = seg0_1 + rows_1; pad_row_1 < seg0_1 + tiles_3 * 32; pad_row_1++) {
                        route_src[pad_row_1] = -1;
                    }
                }
            }
            #pragma unroll 1
            for (int e_14 = lane; e_14 < E_local; e_14 += 32) {
                {
                    uint32_t _ival_25 = static_cast<uint32_t>(zero_i_1);
                    uint32_t _addr_25 = static_cast<uint32_t>(hist_s_addr + (unsigned int)((fw_1 * 256 + e_14) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_25), "r"(_ival_25) : "memory");
                }
            }
            __syncwarp();
            #pragma unroll 1
            for (int tok_7 = w_tok0_1; tok_7 < w_tok1_1; tok_7 += 4) {
                int my_tok_7 = tok_7 + my_tok_off_1;
                int src_8 = my_tok_7 / cap;
                int token_7 = my_tok_7 - src_8 * cap;
                int g_11 = my_tok_7 * 8 + my_slot_1;
                int ident_7 = -1;
                if (my_tok_7 < w_tok1_1) {
                    if (token_7 < meta_s[src_8]) {
                        if (world_1 > 1) {
                            ident_7 = reinterpret_cast<int*>(published_ids_peers.ptrs[src_8])[token_7 * 8 + my_slot_1];
                        } else {
                            ident_7 = topk_ids[token_7 * 8 + my_slot_1];
                        }
                    }
                }
                int local_9 = 0;
                int expert_8 = 0;
                if (ident_7 >= 0) {
                    if (ident_7 / E_local == rank_1) {
                        local_9 = 1;
                        expert_8 = ident_7 - rank_1 * E_local;
                    }
                }
                int key_3 = ((local_9 != 0) ? expert_8 : -1);
                unsigned int live_4 = __ballot_sync(0xFFFFFFFF, local_9 != 0);
                int _count_leading_zeros_26 = __clz(live_4);
                int trips_3 = 32 - _count_leading_zeros_26;
                unsigned int peers_3 = 0;
                #pragma unroll 1
                for (int lane0_3 = 0; lane0_3 < trips_3; lane0_3 += 4) {
                    int held_0_3 = __shfl_sync(0xFFFFFFFF, key_3, lane0_3 + 0);
                    unsigned int same_0_3 = ((held_0_3 == key_3) ? 1 : 0);
                    peers_3 = peers_3 | same_0_3 << (unsigned int)(lane0_3 + 0);
                    int held_1_3 = __shfl_sync(0xFFFFFFFF, key_3, lane0_3 + 1);
                    unsigned int same_1_3 = ((held_1_3 == key_3) ? 1 : 0);
                    peers_3 = peers_3 | same_1_3 << (unsigned int)(lane0_3 + 1);
                    int held_2_3 = __shfl_sync(0xFFFFFFFF, key_3, lane0_3 + 2);
                    unsigned int same_2_3 = ((held_2_3 == key_3) ? 1 : 0);
                    peers_3 = peers_3 | same_2_3 << (unsigned int)(lane0_3 + 2);
                    int held_3_3 = __shfl_sync(0xFFFFFFFF, key_3, lane0_3 + 3);
                    unsigned int same_3_3 = ((held_3_3 == key_3) ? 1 : 0);
                    peers_3 = peers_3 | same_3_3 << (unsigned int)(lane0_3 + 3);
                }
                unsigned int group_3 = peers_3 & live_4;
                int members_3 = (int)(unsigned int)__popc(group_3);
                int lower_3 = (int)(unsigned int)__popc(group_3 & lane_lt_1);
                int running_3 = 0;
                if (local_9 != 0) {
                    running_3 = hist_s[fw_1 * 256 + expert_8];
                    int my_row_1 = seg_s[expert_8] + cbase_s[expert_8] + wbase_s[fw_1 * 256 + expert_8] + running_3 + lower_3;
                    route_src[my_row_1] = g_11;
                }
                __syncwarp();
                if (local_9 != 0 && lower_3 == 0) {
                    {
                        uint32_t _ival_27 = static_cast<uint32_t>(running_3 + members_3);
                        uint32_t _addr_27 = static_cast<uint32_t>(hist_s_addr + (unsigned int)((fw_1 * 256 + expert_8) * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_27), "r"(_ival_27) : "memory");
                    }
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            asm volatile("barrier.sync 12, 512;" ::: "memory");
            const int row_bytes = H * 2;
            const unsigned long long H64_1 = (unsigned long long)H;
            const int routes_per_rank_1 = cap * 8;
            int prev_row = -1;
            unsigned int _phase_slot_full_0 = 0;
            #pragma unroll 1
            for (int row = bid_0_1; row < active_rows_1; row += num_ctas_1) {
                int route_g_1 = route_src[row];
                if (route_g_1 >= 0) {
                    int src_9 = route_g_1 / routes_per_rank_1;
                    int token_8 = (route_g_1 - src_9 * routes_per_rank_1) / 8;
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        mbarrier_arrive_expect_tx(slot_full_addr, (unsigned int)row_bytes);
                        if (world_1 > 1) {
                            cp_async_bulk_gmem2smem(slot_addr, reinterpret_cast<uint8_t*>(published_hidden_peers.ptrs[src_9]) + (unsigned long long)((unsigned long long)token_8 * (unsigned long long)row_bytes), (unsigned int)row_bytes, slot_full_addr);
                        } else {
                            cp_async_bulk_gmem2smem(slot_addr, hidden + ((unsigned long long)token_8 * H64_1), (unsigned int)row_bytes, slot_full_addr);
                        }
                        if (prev_row >= 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(fc1_ready) + (prev_row / 32))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                    mbarrier_wait(slot_full_addr, _phase_slot_full_0);
                    _phase_slot_full_0 ^= 1;
                    if (elect_sync()) {
                        {
                            void* _cpbulk_dst_28 = reinterpret_cast<void*>(pool_ptr + ((unsigned long long)row * H64_1));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_28), "r"(slot_addr), "r"((uint32_t)((unsigned int)row_bytes))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    prev_row = row;
                }
            }
            if (elect_sync()) {
                if (prev_row >= 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(fc1_ready) + (prev_row / 32))), "r"(static_cast<unsigned int>(1)) : "memory");
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
        }
    }
    // ---- Role: weight_tma ----
    if (warp == 6) {
        { // weight_tma_main
            asm volatile("barrier.sync 12, 512;" ::: "memory");
            int tiles_4 = tile_count[0];
            int fc1_items_2 = tiles_4 * fc1_m_tiles;
            int wl_b_2 = (int)bid;
            int wl_P_2 = (int)num_bids;
            int wl_fc2_2 = tiles_4 * fc2_m_tiles;
            int wl_r_2 = fc1_items_2 % wl_P_2;
            int wl_L_2 = wl_P_2 - wl_r_2;
            int wl_delta_2 = (H / 256 + I / 256 / 2) / (I / 256);
            int wl_n1_2 = wl_delta_2 * wl_L_2;
            if (wl_n1_2 > wl_fc2_2) {
                wl_n1_2 = wl_fc2_2;
            }
            int wl_c1_2 = (fc1_items_2 - wl_b_2 + wl_P_2 - 1) / wl_P_2;
            int wl_p1_2 = ((wl_b_2 >= wl_r_2) ? (wl_n1_2 - (wl_b_2 - wl_r_2) + wl_L_2 - 1) / wl_L_2 : 0);
            int wl_p2_2 = (wl_fc2_2 - wl_n1_2 - wl_b_2 + wl_P_2 - 1) / wl_P_2;
            int wl_items_2 = wl_c1_2 + wl_p1_2 + wl_p2_2;
            unsigned int raw_cur = 0;
            unsigned int _phase_raw_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int w_2 = 0; w_2 < wl_items_2; w_2++) {
                    int w_i_2 = (int)w_2;
                    int wl_j_2 = w_i_2 - wl_c1_2;
                    if (wl_j_2 < 0) {
                        w_i_2 = bid + w_i_2 * num_bids;
                    } else if (wl_j_2 < wl_p1_2) {
                        w_i_2 = fc1_items_2 + wl_j_2 * (num_bids - wl_r_2) + (bid - wl_r_2);
                    } else {
                        w_i_2 = fc1_items_2 + wl_n1_2 + (wl_j_2 - wl_p1_2) * num_bids + bid;
                    }
                    int is_fc1_2 = 0;
                    int local_10 = w_i_2 - fc1_items_2;
                    if (w_i_2 < fc1_items_2) {
                        is_fc1_2 = 1;
                        local_10 = w_i_2;
                    }
                    unsigned int ww_2 = (unsigned int)local_10;
                    uint32_t _fast_div_q_2 = (fc1_m_div.divisor != 1) ? (__umulhi((uint32_t)(ww_2), fc1_m_div.multiplier) >> fc1_m_div.shift_right) : (uint32_t)(ww_2);
                    uint32_t _fast_div_r_2 = (uint32_t)(ww_2) - _fast_div_q_2 * (uint32_t)(fc1_m_div.divisor);
                    uint32_t _fast_div_q_3 = (fc2_m_div.divisor != 1) ? (__umulhi((uint32_t)(ww_2), fc2_m_div.multiplier) >> fc2_m_div.shift_right) : (uint32_t)(ww_2);
                    uint32_t _fast_div_r_3 = (uint32_t)(ww_2) - _fast_div_q_3 * (uint32_t)(fc2_m_div.divisor);
                    int t_4 = (int)_fast_div_q_3;
                    int m_2 = (int)_fast_div_r_3;
                    if (is_fc1_2 != 0) {
                        t_4 = (int)_fast_div_q_2;
                        m_2 = (int)_fast_div_r_2;
                    }
                    int num_k_1 = I / 256;
                    if (is_fc1_2 != 0) {
                        num_k_1 = H / 256;
                    }
                    int expert_9 = tile_expert[t_4];
                    if (is_fc1_2 != 0) {
                        int gate_row = m_2 * 64;
                        int up_row = I + m_2 * 64;
                        #pragma unroll 1
                        for (int kt_1 = 0; kt_1 < num_k_1; kt_1++) {
                            mbarrier_wait(raw_empty_addr + (raw_cur) * 8, _phase_raw_empty);
                            unsigned int raw_base = raw_smem_addr + raw_cur * 16384;
                            unsigned int sf_base = sf_smem_addr + raw_cur * 2048;
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(raw_base), "l"((&W13)), "r"(kt_1 * 128), "r"(gate_row), "r"(expert_9),
                                   "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(raw_base + 8192), "l"((&W13)), "r"(kt_1 * 128), "r"(up_row), "r"(expert_9),
                                   "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(sf_base), "l"((&S13)), "r"(kt_1 * 16), "r"(gate_row), "r"(expert_9),
                                   "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(sf_base + 1024), "l"((&S13)), "r"(kt_1 * 16), "r"(up_row), "r"(expert_9),
                                   "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            mbarrier_arrive_expect_tx(raw_full_addr + (raw_cur) * 8, 18432);
                            raw_cur += 1;
                            if (raw_cur == 6) { raw_cur = 0; _phase_raw_empty ^= 1; }
                        }
                    } else {
                        int m_row = m_2 * 128;
                        #pragma unroll 1
                        for (int kt_2 = 0; kt_2 < num_k_1; kt_2++) {
                            mbarrier_wait(raw_empty_addr + (raw_cur) * 8, _phase_raw_empty);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(raw_smem_addr + raw_cur * 16384), "l"((&W2)), "r"(kt_2 * 128), "r"(m_row), "r"(expert_9),
                                   "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(sf_smem_addr + raw_cur * 2048), "l"((&S2)), "r"(kt_2 * 16), "r"(m_row), "r"(expert_9),
                                   "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            mbarrier_arrive_expect_tx(raw_full_addr + (raw_cur) * 8, 18432);
                            raw_cur += 1;
                            if (raw_cur == 6) { raw_cur = 0; _phase_raw_empty ^= 1; }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: token_tma ----
    if (warp == 7) {
        { // token_tma_main
            const int world_2 = mega_pg_world;
            const int rank_2 = mega_pg_rank;
            const int fw_2 = 5;
            const int ftid_2 = fw_2 * 32 + lane;
            const int num_ctas_2 = num_bids;
            const int bid_0_2 = bid;
            const int linear_2 = bid_0_2 * 192 + ftid_2;
            const int stride_2 = num_ctas_2 * 192;
            const int max_tiles_2 = (world_2 * cap * 8 + 31 * E_local) / 32;
            const unsigned int lane_lt_2 = (1 << (unsigned int)lane) - 1;
            const int zero_i_2 = 0;
            const int hist_quad_words_2 = E_local * 4;
            #pragma unroll 1
            for (int t_5 = linear_2; t_5 < max_tiles_2; t_5 += stride_2) {
                asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(fc1_ready + t_5)), "r"(static_cast<unsigned int>(0)) : "memory");
                asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(fc1_done + t_5)), "r"(static_cast<unsigned int>(0)) : "memory");
                asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(fc2_done + t_5)), "r"(static_cast<unsigned int>(0)) : "memory");
            }
            if (linear_2 == 0) {
                reinterpret_cast<int*>(published_meta)[0] = T_r;
            }
            if (world_2 > 1) {
                #pragma unroll 1
                for (int index_2 = linear_2 * 8; index_2 < T_r * H; index_2 += stride_2 * 8) {
                    float _vec_load_1[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(hidden + index_2 + 0);
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    unsigned int _packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_1[_lp*2 + 0], _vec_load_1[_lp*2+1 + 0]));
                        _packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_1[_lp*2 + 2], _vec_load_1[_lp*2+1 + 2]));
                        _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_1[_lp*2 + 4], _vec_load_1[_lp*2+1 + 4]));
                        _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_1[_lp*2 + 6], _vec_load_1[_lp*2+1 + 6]));
                        _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    {
                        auto _gst_1 = reinterpret_cast<__nv_bfloat16*>(published_hidden) + index_2;
                        __builtin_assume(__isGlobal(_gst_1));
                        reinterpret_cast<int4*>(_gst_1)[0] = reinterpret_cast<int4*>(_packed_1)[0];
                    }
                }
                #pragma unroll 1
                for (int route_2 = linear_2; route_2 < T_r * 8; route_2 += stride_2) {
                    int route_id_2 = topk_ids[route_2];
                    float route_weight_2 = topk_weights[route_2];
                    reinterpret_cast<int*>(published_ids)[route_2] = route_id_2;
                    reinterpret_cast<float*>(published_weights)[route_2] = route_weight_2;
                }
                asm volatile("barrier.sync 11, 192;" ::: "memory");
                asm volatile("barrier.sync 11, 192;" ::: "memory");
                asm volatile("barrier.sync 11, 192;" ::: "memory");
            }
            #pragma unroll 1
            for (int e_15 = lane; e_15 < E_local; e_15 += 32) {
                {
                    uint32_t _ival_2 = static_cast<uint32_t>(zero_i_2);
                    uint32_t _addr_2 = static_cast<uint32_t>(hist_s_addr + (unsigned int)((fw_2 * 256 + e_15) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_2), "r"(_ival_2) : "memory");
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            const int tokens_total_2 = world_2 * cap;
            const int tokens_per_cta_2 = (tokens_total_2 + num_ctas_2 - 1) / num_ctas_2;
            const int cta_tok0_2 = bid_0_2 * tokens_per_cta_2;
            int cta_tok1_raw_2 = cta_tok0_2 + tokens_per_cta_2;
            int cta_tok1_2 = ((cta_tok1_raw_2 < tokens_total_2) ? cta_tok1_raw_2 : tokens_total_2);
            const int tokens_per_warp_2 = (tokens_per_cta_2 + 6 - 1) / 6;
            const int w_tok0_2 = cta_tok0_2 + fw_2 * tokens_per_warp_2;
            int w_tok1_raw_2 = w_tok0_2 + tokens_per_warp_2;
            int w_tok1_2 = ((w_tok1_raw_2 < cta_tok1_2) ? w_tok1_raw_2 : cta_tok1_2);
            const int my_slot_2 = lane % 8;
            const int my_tok_off_2 = lane / 8;
            const int scan_all_2 = ((world_2 == 1 && T_r * 8 <= 4096) ? 1 : 0);
            const int scan_pre1_2 = ((cta_tok0_2 < T_r) ? cta_tok0_2 : T_r);
            if (scan_all_2 != 0) {
                #pragma unroll 1
                for (int e_16 = lane; e_16 < E_local; e_16 += 32) {
                    {
                        uint32_t _ival_3 = static_cast<uint32_t>(zero_i_2);
                        uint32_t _addr_3 = static_cast<uint32_t>(pre_s_addr + (unsigned int)((fw_2 * 256 + e_16) * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_3), "r"(_ival_3) : "memory");
                    }
                    {
                        uint32_t _ival_4 = static_cast<uint32_t>(zero_i_2);
                        uint32_t _addr_4 = static_cast<uint32_t>(suf_s_addr + (unsigned int)((fw_2 * 256 + e_16) * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_4), "r"(_ival_4) : "memory");
                    }
                }
                __syncwarp();
            }
            #pragma unroll 1
            for (int tok_8 = w_tok0_2; tok_8 < w_tok1_2; tok_8 += 4) {
                int my_tok_8 = tok_8 + my_tok_off_2;
                int src_10 = my_tok_8 / cap;
                int token_9 = my_tok_8 - src_10 * cap;
                int g_13 = my_tok_8 * 8 + my_slot_2;
                int ident_8 = -1;
                if (my_tok_8 < w_tok1_2) {
                    if (token_9 < meta_s[src_10]) {
                        if (world_2 > 1) {
                            ident_8 = reinterpret_cast<int*>(published_ids_peers.ptrs[src_10])[token_9 * 8 + my_slot_2];
                        } else {
                            ident_8 = topk_ids[token_9 * 8 + my_slot_2];
                        }
                    }
                }
                int local_11 = 0;
                int expert_10 = 0;
                if (ident_8 >= 0) {
                    if (ident_8 / E_local == rank_2) {
                        local_11 = 1;
                        expert_10 = ident_8 - rank_2 * E_local;
                    }
                }
                int key_4 = ((local_11 != 0) ? expert_10 : -1);
                unsigned int live_5 = __ballot_sync(0xFFFFFFFF, local_11 != 0);
                int _count_leading_zeros_5_1 = __clz(live_5);
                int trips_4 = 32 - _count_leading_zeros_5_1;
                unsigned int peers_4 = 0;
                #pragma unroll 1
                for (int lane0_4 = 0; lane0_4 < trips_4; lane0_4 += 4) {
                    int held_0_4 = __shfl_sync(0xFFFFFFFF, key_4, lane0_4 + 0);
                    unsigned int same_0_4 = ((held_0_4 == key_4) ? 1 : 0);
                    peers_4 = peers_4 | same_0_4 << (unsigned int)(lane0_4 + 0);
                    int held_1_4 = __shfl_sync(0xFFFFFFFF, key_4, lane0_4 + 1);
                    unsigned int same_1_4 = ((held_1_4 == key_4) ? 1 : 0);
                    peers_4 = peers_4 | same_1_4 << (unsigned int)(lane0_4 + 1);
                    int held_2_4 = __shfl_sync(0xFFFFFFFF, key_4, lane0_4 + 2);
                    unsigned int same_2_4 = ((held_2_4 == key_4) ? 1 : 0);
                    peers_4 = peers_4 | same_2_4 << (unsigned int)(lane0_4 + 2);
                    int held_3_4 = __shfl_sync(0xFFFFFFFF, key_4, lane0_4 + 3);
                    unsigned int same_3_4 = ((held_3_4 == key_4) ? 1 : 0);
                    peers_4 = peers_4 | same_3_4 << (unsigned int)(lane0_4 + 3);
                }
                unsigned int group_4 = peers_4 & live_5;
                int members_4 = (int)(unsigned int)__popc(group_4);
                int lower_4 = (int)(unsigned int)__popc(group_4 & lane_lt_2);
                if (local_11 != 0 && lower_4 == 0) {
                    unsigned int count_addr_2 = hist_s_addr + (unsigned int)((fw_2 * 256 + expert_10) * 4);
                    int counted_2 = hist_s[fw_2 * 256 + expert_10];
                    {
                        uint32_t _ival_6 = static_cast<uint32_t>(counted_2 + members_4);
                        uint32_t _addr_6 = static_cast<uint32_t>(count_addr_2);
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_6), "r"(_ival_6) : "memory");
                    }
                }
                __syncwarp();
            }
            if (scan_all_2 != 0) {
                #pragma unroll 1
                for (int tok_9 = fw_2 * 4; tok_9 < scan_pre1_2; tok_9 += 24) {
                    int my_tok_9 = tok_9 + my_tok_off_2;
                    int src_11 = my_tok_9 / cap;
                    int token_10 = my_tok_9 - src_11 * cap;
                    int g_15 = my_tok_9 * 8 + my_slot_2;
                    int ident_9 = -1;
                    if (my_tok_9 < scan_pre1_2) {
                        if (token_10 < meta_s[src_11]) {
                            if (world_2 > 1) {
                                ident_9 = reinterpret_cast<int*>(published_ids_peers.ptrs[src_11])[token_10 * 8 + my_slot_2];
                            } else {
                                ident_9 = topk_ids[token_10 * 8 + my_slot_2];
                            }
                        }
                    }
                    int local_12 = 0;
                    int expert_11 = 0;
                    if (ident_9 >= 0) {
                        if (ident_9 / E_local == rank_2) {
                            local_12 = 1;
                            expert_11 = ident_9 - rank_2 * E_local;
                        }
                    }
                    int _atomic_old_2 = 0;
                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.s32 %0, [%1], %2;\n}"
                        : "+r"(_atomic_old_2) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&pre_s[fw_2 * 256 + expert_11]))), "r"(static_cast<int>(1)), "r"(static_cast<uint32_t>(local_12 != 0)) : "memory");
                }
                #pragma unroll 1
                for (int tok_10 = cta_tok1_2 + fw_2 * 4; tok_10 < T_r; tok_10 += 24) {
                    int my_tok_10 = tok_10 + my_tok_off_2;
                    int src_12 = my_tok_10 / cap;
                    int token_11 = my_tok_10 - src_12 * cap;
                    int g_17 = my_tok_10 * 8 + my_slot_2;
                    int ident_10 = -1;
                    if (my_tok_10 < T_r) {
                        if (token_11 < meta_s[src_12]) {
                            if (world_2 > 1) {
                                ident_10 = reinterpret_cast<int*>(published_ids_peers.ptrs[src_12])[token_11 * 8 + my_slot_2];
                            } else {
                                ident_10 = topk_ids[token_11 * 8 + my_slot_2];
                            }
                        }
                    }
                    int local_13 = 0;
                    int expert_12 = 0;
                    if (ident_10 >= 0) {
                        if (ident_10 / E_local == rank_2) {
                            local_13 = 1;
                            expert_12 = ident_10 - rank_2 * E_local;
                        }
                    }
                    int _atomic_old_3 = 0;
                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.s32 %0, [%1], %2;\n}"
                        : "+r"(_atomic_old_3) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&suf_s[fw_2 * 256 + expert_12]))), "r"(static_cast<int>(1)), "r"(static_cast<uint32_t>(local_13 != 0)) : "memory");
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            #pragma unroll 1
            for (int e_17 = ftid_2; e_17 < E_local; e_17 += 192) {
                int running_4 = 0;
                {
                    uint32_t _ival_7 = static_cast<uint32_t>(running_4);
                    uint32_t _addr_7 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)(e_17 * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_7), "r"(_ival_7) : "memory");
                }
                running_4 = running_4 + hist_s[e_17];
                {
                    uint32_t _ival_8 = static_cast<uint32_t>(running_4);
                    uint32_t _addr_8 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((256 + e_17) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_8), "r"(_ival_8) : "memory");
                }
                running_4 = running_4 + hist_s[256 + e_17];
                {
                    uint32_t _ival_9 = static_cast<uint32_t>(running_4);
                    uint32_t _addr_9 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((512 + e_17) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_9), "r"(_ival_9) : "memory");
                }
                running_4 = running_4 + hist_s[512 + e_17];
                {
                    uint32_t _ival_10 = static_cast<uint32_t>(running_4);
                    uint32_t _addr_10 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((768 + e_17) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_10), "r"(_ival_10) : "memory");
                }
                running_4 = running_4 + hist_s[768 + e_17];
                {
                    uint32_t _ival_11 = static_cast<uint32_t>(running_4);
                    uint32_t _addr_11 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((1024 + e_17) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_11), "r"(_ival_11) : "memory");
                }
                running_4 = running_4 + hist_s[1024 + e_17];
                {
                    uint32_t _ival_12 = static_cast<uint32_t>(running_4);
                    uint32_t _addr_12 = static_cast<uint32_t>(wbase_s_addr + (unsigned int)((1280 + e_17) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_12), "r"(_ival_12) : "memory");
                }
                running_4 = running_4 + hist_s[1280 + e_17];
                if (scan_all_2 != 0) {
                    int before_2 = 0;
                    int after_2 = 0;
                    before_2 = before_2 + pre_s[e_17];
                    after_2 = after_2 + suf_s[e_17];
                    before_2 = before_2 + pre_s[256 + e_17];
                    after_2 = after_2 + suf_s[256 + e_17];
                    before_2 = before_2 + pre_s[512 + e_17];
                    after_2 = after_2 + suf_s[512 + e_17];
                    before_2 = before_2 + pre_s[768 + e_17];
                    after_2 = after_2 + suf_s[768 + e_17];
                    before_2 = before_2 + pre_s[1024 + e_17];
                    after_2 = after_2 + suf_s[1024 + e_17];
                    before_2 = before_2 + pre_s[1280 + e_17];
                    after_2 = after_2 + suf_s[1280 + e_17];
                    {
                        uint32_t _ival_13 = static_cast<uint32_t>(before_2);
                        uint32_t _addr_13 = static_cast<uint32_t>(cbase_s_addr + (unsigned int)(e_17 * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_13), "r"(_ival_13) : "memory");
                    }
                    {
                        uint32_t _ival_14 = static_cast<uint32_t>(before_2 + running_4 + after_2);
                        uint32_t _addr_14 = static_cast<uint32_t>(total_s_addr + (unsigned int)(e_17 * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_14), "r"(_ival_14) : "memory");
                    }
                } else {
                    cta_hist[bid_0_2 / 4 * hist_quad_words_2 + e_17 * 4 + bid_0_2 % 4] = running_4;
                }
            }
            int quads_2[32];
            if (scan_all_2 == 0) {
                asm volatile("barrier.sync 11, 192;" ::: "memory");
                asm volatile("barrier.sync 11, 192;" ::: "memory");
                #pragma unroll 1
                for (int e_18 = ftid_2; e_18 < E_local; e_18 += 192) {
                    int total_2 = 0;
                    int base_3 = 0;
                    #pragma unroll 1
                    for (int c0_2 = 0; c0_2 < num_ctas_2; c0_2 += 32) {
                        {
                            const int4* _ivptr_15 = reinterpret_cast<const int4*>(cta_hist + (c0_2 / 4 * hist_quad_words_2 + e_18 * 4) + 0);
                            int4 _ivld_15;
                            _ivld_15 = *_ivptr_15;
                            quads_2[0 + 0] = _ivld_15.x;
                            quads_2[0 + 1] = _ivld_15.y;
                            quads_2[0 + 2] = _ivld_15.z;
                            quads_2[0 + 3] = _ivld_15.w;
                        }
                        {
                            const int4* _ivptr_16 = reinterpret_cast<const int4*>(cta_hist + ((c0_2 / 4 + 1) * hist_quad_words_2 + e_18 * 4) + 0);
                            int4 _ivld_16;
                            _ivld_16 = *_ivptr_16;
                            quads_2[4 + 0] = _ivld_16.x;
                            quads_2[4 + 1] = _ivld_16.y;
                            quads_2[4 + 2] = _ivld_16.z;
                            quads_2[4 + 3] = _ivld_16.w;
                        }
                        {
                            const int4* _ivptr_17 = reinterpret_cast<const int4*>(cta_hist + ((c0_2 / 4 + 2) * hist_quad_words_2 + e_18 * 4) + 0);
                            int4 _ivld_17;
                            _ivld_17 = *_ivptr_17;
                            quads_2[8 + 0] = _ivld_17.x;
                            quads_2[8 + 1] = _ivld_17.y;
                            quads_2[8 + 2] = _ivld_17.z;
                            quads_2[8 + 3] = _ivld_17.w;
                        }
                        {
                            const int4* _ivptr_18 = reinterpret_cast<const int4*>(cta_hist + ((c0_2 / 4 + 3) * hist_quad_words_2 + e_18 * 4) + 0);
                            int4 _ivld_18;
                            _ivld_18 = *_ivptr_18;
                            quads_2[12 + 0] = _ivld_18.x;
                            quads_2[12 + 1] = _ivld_18.y;
                            quads_2[12 + 2] = _ivld_18.z;
                            quads_2[12 + 3] = _ivld_18.w;
                        }
                        {
                            const int4* _ivptr_19 = reinterpret_cast<const int4*>(cta_hist + ((c0_2 / 4 + 4) * hist_quad_words_2 + e_18 * 4) + 0);
                            int4 _ivld_19;
                            _ivld_19 = *_ivptr_19;
                            quads_2[16 + 0] = _ivld_19.x;
                            quads_2[16 + 1] = _ivld_19.y;
                            quads_2[16 + 2] = _ivld_19.z;
                            quads_2[16 + 3] = _ivld_19.w;
                        }
                        {
                            const int4* _ivptr_20 = reinterpret_cast<const int4*>(cta_hist + ((c0_2 / 4 + 5) * hist_quad_words_2 + e_18 * 4) + 0);
                            int4 _ivld_20;
                            _ivld_20 = *_ivptr_20;
                            quads_2[20 + 0] = _ivld_20.x;
                            quads_2[20 + 1] = _ivld_20.y;
                            quads_2[20 + 2] = _ivld_20.z;
                            quads_2[20 + 3] = _ivld_20.w;
                        }
                        {
                            const int4* _ivptr_21 = reinterpret_cast<const int4*>(cta_hist + ((c0_2 / 4 + 6) * hist_quad_words_2 + e_18 * 4) + 0);
                            int4 _ivld_21;
                            _ivld_21 = *_ivptr_21;
                            quads_2[24 + 0] = _ivld_21.x;
                            quads_2[24 + 1] = _ivld_21.y;
                            quads_2[24 + 2] = _ivld_21.z;
                            quads_2[24 + 3] = _ivld_21.w;
                        }
                        {
                            const int4* _ivptr_22 = reinterpret_cast<const int4*>(cta_hist + ((c0_2 / 4 + 7) * hist_quad_words_2 + e_18 * 4) + 0);
                            int4 _ivld_22;
                            _ivld_22 = *_ivptr_22;
                            quads_2[28 + 0] = _ivld_22.x;
                            quads_2[28 + 1] = _ivld_22.y;
                            quads_2[28 + 2] = _ivld_22.z;
                            quads_2[28 + 3] = _ivld_22.w;
                        }
                        total_2 = total_2 + ((num_ctas_2 > c0_2) ? quads_2[0] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2) ? quads_2[0] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 1) ? quads_2[1] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 1) ? quads_2[1] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 2) ? quads_2[2] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 2) ? quads_2[2] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 3) ? quads_2[3] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 3) ? quads_2[3] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 4) ? quads_2[4] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 4) ? quads_2[4] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 5) ? quads_2[5] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 5) ? quads_2[5] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 6) ? quads_2[6] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 6) ? quads_2[6] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 7) ? quads_2[7] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 7) ? quads_2[7] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 8) ? quads_2[8] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 8) ? quads_2[8] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 9) ? quads_2[9] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 9) ? quads_2[9] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 10) ? quads_2[10] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 10) ? quads_2[10] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 11) ? quads_2[11] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 11) ? quads_2[11] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 12) ? quads_2[12] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 12) ? quads_2[12] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 13) ? quads_2[13] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 13) ? quads_2[13] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 14) ? quads_2[14] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 14) ? quads_2[14] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 15) ? quads_2[15] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 15) ? quads_2[15] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 16) ? quads_2[16] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 16) ? quads_2[16] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 17) ? quads_2[17] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 17) ? quads_2[17] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 18) ? quads_2[18] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 18) ? quads_2[18] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 19) ? quads_2[19] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 19) ? quads_2[19] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 20) ? quads_2[20] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 20) ? quads_2[20] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 21) ? quads_2[21] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 21) ? quads_2[21] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 22) ? quads_2[22] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 22) ? quads_2[22] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 23) ? quads_2[23] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 23) ? quads_2[23] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 24) ? quads_2[24] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 24) ? quads_2[24] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 25) ? quads_2[25] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 25) ? quads_2[25] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 26) ? quads_2[26] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 26) ? quads_2[26] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 27) ? quads_2[27] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 27) ? quads_2[27] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 28) ? quads_2[28] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 28) ? quads_2[28] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 29) ? quads_2[29] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 29) ? quads_2[29] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 30) ? quads_2[30] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 30) ? quads_2[30] : 0);
                        total_2 = total_2 + ((num_ctas_2 > c0_2 + 31) ? quads_2[31] : 0);
                        base_3 = base_3 + ((bid_0_2 > c0_2 + 31) ? quads_2[31] : 0);
                    }
                    {
                        uint32_t _ival_23 = static_cast<uint32_t>(total_2);
                        uint32_t _addr_23 = static_cast<uint32_t>(total_s_addr + (unsigned int)(e_18 * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_23), "r"(_ival_23) : "memory");
                    }
                    {
                        uint32_t _ival_24 = static_cast<uint32_t>(base_3);
                        uint32_t _addr_24 = static_cast<uint32_t>(cbase_s_addr + (unsigned int)(e_18 * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_24), "r"(_ival_24) : "memory");
                    }
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            const int active_rows_2 = seg_s[E_local];
            const int active_tiles_2 = active_rows_2 / 32;
            if (bid_0_2 == 0) {
                #pragma unroll 1
                for (int e_19 = ftid_2; e_19 < E_local + 1; e_19 += 192) {
                    expert_seg[e_19] = seg_s[e_19];
                }
                #pragma unroll 1
                for (int e_20 = ftid_2; e_20 < E_local; e_20 += 192) {
                    int rows_2 = total_s[e_20];
                    int seg0_2 = seg_s[e_20];
                    int first_2 = seg0_2 / 32;
                    int tiles_5 = (rows_2 + 31) / 32;
                    #pragma unroll 1
                    for (int k_2 = 0; k_2 < tiles_5; k_2++) {
                        tile_expert[first_2 + k_2] = e_20;
                        int limit_2 = (k_2 + 1) * 32;
                        tile_mn_limit[first_2 + k_2] = seg0_2 + ((rows_2 < limit_2) ? rows_2 : limit_2);
                    }
                    #pragma unroll 1
                    for (int pad_row_2 = seg0_2 + rows_2; pad_row_2 < seg0_2 + tiles_5 * 32; pad_row_2++) {
                        route_src[pad_row_2] = -1;
                    }
                }
            }
            #pragma unroll 1
            for (int e_21 = lane; e_21 < E_local; e_21 += 32) {
                {
                    uint32_t _ival_25 = static_cast<uint32_t>(zero_i_2);
                    uint32_t _addr_25 = static_cast<uint32_t>(hist_s_addr + (unsigned int)((fw_2 * 256 + e_21) * 4));
                    asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_25), "r"(_ival_25) : "memory");
                }
            }
            __syncwarp();
            #pragma unroll 1
            for (int tok_11 = w_tok0_2; tok_11 < w_tok1_2; tok_11 += 4) {
                int my_tok_11 = tok_11 + my_tok_off_2;
                int src_13 = my_tok_11 / cap;
                int token_12 = my_tok_11 - src_13 * cap;
                int g_19 = my_tok_11 * 8 + my_slot_2;
                int ident_11 = -1;
                if (my_tok_11 < w_tok1_2) {
                    if (token_12 < meta_s[src_13]) {
                        if (world_2 > 1) {
                            ident_11 = reinterpret_cast<int*>(published_ids_peers.ptrs[src_13])[token_12 * 8 + my_slot_2];
                        } else {
                            ident_11 = topk_ids[token_12 * 8 + my_slot_2];
                        }
                    }
                }
                int local_14 = 0;
                int expert_13 = 0;
                if (ident_11 >= 0) {
                    if (ident_11 / E_local == rank_2) {
                        local_14 = 1;
                        expert_13 = ident_11 - rank_2 * E_local;
                    }
                }
                int key_5 = ((local_14 != 0) ? expert_13 : -1);
                unsigned int live_6 = __ballot_sync(0xFFFFFFFF, local_14 != 0);
                int _count_leading_zeros_26_1 = __clz(live_6);
                int trips_5 = 32 - _count_leading_zeros_26_1;
                unsigned int peers_5 = 0;
                #pragma unroll 1
                for (int lane0_5 = 0; lane0_5 < trips_5; lane0_5 += 4) {
                    int held_0_5 = __shfl_sync(0xFFFFFFFF, key_5, lane0_5 + 0);
                    unsigned int same_0_5 = ((held_0_5 == key_5) ? 1 : 0);
                    peers_5 = peers_5 | same_0_5 << (unsigned int)(lane0_5 + 0);
                    int held_1_5 = __shfl_sync(0xFFFFFFFF, key_5, lane0_5 + 1);
                    unsigned int same_1_5 = ((held_1_5 == key_5) ? 1 : 0);
                    peers_5 = peers_5 | same_1_5 << (unsigned int)(lane0_5 + 1);
                    int held_2_5 = __shfl_sync(0xFFFFFFFF, key_5, lane0_5 + 2);
                    unsigned int same_2_5 = ((held_2_5 == key_5) ? 1 : 0);
                    peers_5 = peers_5 | same_2_5 << (unsigned int)(lane0_5 + 2);
                    int held_3_5 = __shfl_sync(0xFFFFFFFF, key_5, lane0_5 + 3);
                    unsigned int same_3_5 = ((held_3_5 == key_5) ? 1 : 0);
                    peers_5 = peers_5 | same_3_5 << (unsigned int)(lane0_5 + 3);
                }
                unsigned int group_5 = peers_5 & live_6;
                int members_5 = (int)(unsigned int)__popc(group_5);
                int lower_5 = (int)(unsigned int)__popc(group_5 & lane_lt_2);
                int running_5 = 0;
                if (local_14 != 0) {
                    running_5 = hist_s[fw_2 * 256 + expert_13];
                    int my_row_2 = seg_s[expert_13] + cbase_s[expert_13] + wbase_s[fw_2 * 256 + expert_13] + running_5 + lower_5;
                    route_src[my_row_2] = g_19;
                }
                __syncwarp();
                if (local_14 != 0 && lower_5 == 0) {
                    {
                        uint32_t _ival_27 = static_cast<uint32_t>(running_5 + members_5);
                        uint32_t _addr_27 = static_cast<uint32_t>(hist_s_addr + (unsigned int)((fw_2 * 256 + expert_13) * 4));
                        asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_27), "r"(_ival_27) : "memory");
                    }
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            asm volatile("barrier.sync 11, 192;" ::: "memory");
            asm volatile("barrier.sync 12, 512;" ::: "memory");
            int tiles_6 = tile_count[0];
            int fc1_items_3 = tiles_6 * fc1_m_tiles;
            int wl_b_3 = (int)bid;
            int wl_P_3 = (int)num_bids;
            int wl_fc2_3 = tiles_6 * fc2_m_tiles;
            int wl_r_3 = fc1_items_3 % wl_P_3;
            int wl_L_3 = wl_P_3 - wl_r_3;
            int wl_delta_3 = (H / 256 + I / 256 / 2) / (I / 256);
            int wl_n1_3 = wl_delta_3 * wl_L_3;
            if (wl_n1_3 > wl_fc2_3) {
                wl_n1_3 = wl_fc2_3;
            }
            int wl_c1_3 = (fc1_items_3 - wl_b_3 + wl_P_3 - 1) / wl_P_3;
            int wl_p1_3 = ((wl_b_3 >= wl_r_3) ? (wl_n1_3 - (wl_b_3 - wl_r_3) + wl_L_3 - 1) / wl_L_3 : 0);
            int wl_p2_3 = (wl_fc2_3 - wl_n1_3 - wl_b_3 + wl_P_3 - 1) / wl_P_3;
            int wl_items_3 = wl_c1_3 + wl_p1_3 + wl_p2_3;
            unsigned int b_cur_1 = 0;
            unsigned int _phase_b_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int w_3 = 0; w_3 < wl_items_3; w_3++) {
                    int w_i_3 = (int)w_3;
                    int wl_j_3 = w_i_3 - wl_c1_3;
                    if (wl_j_3 < 0) {
                        w_i_3 = bid + w_i_3 * num_bids;
                    } else if (wl_j_3 < wl_p1_3) {
                        w_i_3 = fc1_items_3 + wl_j_3 * (num_bids - wl_r_3) + (bid - wl_r_3);
                    } else {
                        w_i_3 = fc1_items_3 + wl_n1_3 + (wl_j_3 - wl_p1_3) * num_bids + bid;
                    }
                    int is_fc1_3 = 0;
                    int local_15 = w_i_3 - fc1_items_3;
                    if (w_i_3 < fc1_items_3) {
                        is_fc1_3 = 1;
                        local_15 = w_i_3;
                    }
                    unsigned int ww_3 = (unsigned int)local_15;
                    uint32_t _fast_div_q_0 = (fc1_m_div.divisor != 1) ? (__umulhi((uint32_t)(ww_3), fc1_m_div.multiplier) >> fc1_m_div.shift_right) : (uint32_t)(ww_3);
                    uint32_t _fast_div_r_0 = (uint32_t)(ww_3) - _fast_div_q_0 * (uint32_t)(fc1_m_div.divisor);
                    uint32_t _fast_div_q_1 = (fc2_m_div.divisor != 1) ? (__umulhi((uint32_t)(ww_3), fc2_m_div.multiplier) >> fc2_m_div.shift_right) : (uint32_t)(ww_3);
                    uint32_t _fast_div_r_1 = (uint32_t)(ww_3) - _fast_div_q_1 * (uint32_t)(fc2_m_div.divisor);
                    int t_6 = (int)_fast_div_q_1;
                    int m_3 = (int)_fast_div_r_1;
                    if (is_fc1_3 != 0) {
                        t_6 = (int)_fast_div_q_0;
                        m_3 = (int)_fast_div_r_0;
                    }
                    int num_k_2 = I / 256;
                    if (is_fc1_3 != 0) {
                        num_k_2 = H / 256;
                    }
                    int row0_1 = t_6 * 32;
                    if (is_fc1_3 != 0) {
                        int live_7 = tile_mn_limit[t_6] - t_6 * 32;
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(fc1_ready) + (t_6))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>((unsigned int)live_7)) >= static_cast<unsigned int>(2147483648u));
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        #pragma unroll 1
                        for (int kt_3 = 0; kt_3 < num_k_2; kt_3++) {
                            mbarrier_wait(b_empty_addr + (b_cur_1) * 8, _phase_b_empty);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_b_addr + b_cur_1 * 16384), "l"((&pool)), "r"(0), "r"(row0_1), "r"(kt_3 * 4),
                                   "r"(b_full_addr + (b_cur_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            mbarrier_arrive_expect_tx(b_full_addr + (b_cur_1) * 8, 16384);
                            b_cur_1 += 1;
                            if (b_cur_1 == 6) { b_cur_1 = 0; _phase_b_empty ^= 1; }
                        }
                    } else {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(fc1_done) + (t_6))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>((unsigned int)fc1_m_tiles)) >= static_cast<unsigned int>(2147483648u));
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        #pragma unroll 1
                        for (int kt_4 = 0; kt_4 < num_k_2; kt_4++) {
                            mbarrier_wait(b_empty_addr + (b_cur_1) * 8, _phase_b_empty);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_b_addr + b_cur_1 * 16384), "l"((&fc1_out_ld)), "r"(0), "r"(row0_1), "r"(kt_4 * 4),
                                   "r"(b_full_addr + (b_cur_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            mbarrier_arrive_expect_tx(b_full_addr + (b_cur_1) * 8, 16384);
                            b_cur_1 += 1;
                            if (b_cur_1 == 6) { b_cur_1 = 0; _phase_b_empty ^= 1; }
                        }
                    }
                }
                #pragma unroll 1
                for (int _ = 0; _ < 6; _++) {
                    mbarrier_wait(b_empty_addr + (b_cur_1) * 8, _phase_b_empty);
                    b_cur_1 += 1;
                    if (b_cur_1 == 6) { b_cur_1 = 0; _phase_b_empty ^= 1; }
                }
            }
            asm volatile("barrier.sync 11, 192;" ::: "memory");
        }
    }
    // ---- Role: transform ----
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // transform_main
            asm volatile("barrier.sync 12, 512;" ::: "memory");
            int tiles_7 = tile_count[0];
            int fc1_items_4 = tiles_7 * fc1_m_tiles;
            int wl_b_4 = (int)bid;
            int wl_P_4 = (int)num_bids;
            int wl_fc2_4 = tiles_7 * fc2_m_tiles;
            int wl_r_4 = fc1_items_4 % wl_P_4;
            int wl_L_4 = wl_P_4 - wl_r_4;
            int wl_delta_4 = (H / 256 + I / 256 / 2) / (I / 256);
            int wl_n1_4 = wl_delta_4 * wl_L_4;
            if (wl_n1_4 > wl_fc2_4) {
                wl_n1_4 = wl_fc2_4;
            }
            int wl_c1_4 = (fc1_items_4 - wl_b_4 + wl_P_4 - 1) / wl_P_4;
            int wl_p1_4 = ((wl_b_4 >= wl_r_4) ? (wl_n1_4 - (wl_b_4 - wl_r_4) + wl_L_4 - 1) / wl_L_4 : 0);
            int wl_p2_4 = (wl_fc2_4 - wl_n1_4 - wl_b_4 + wl_P_4 - 1) / wl_P_4;
            int wl_items_4 = wl_c1_4 + wl_p1_4 + wl_p2_4;
            const int quarter_1 = warp % 4;
            const int half = warp / 4 - 2;
            const int fc1_row = ((lane < 16) ? quarter_1 * 16 + lane : 48 + quarter_1 * 16 + lane);
            const int fc2_row = quarter_1 * 32 + lane;
            const unsigned int lane_base = (unsigned int)(quarter_1 * 32) << 16;
            unsigned int raw_cur_1 = 0;
            unsigned int a_cur_1 = 0;
            unsigned int raw_words[16];
            unsigned int scale_words[2];
            unsigned int _phase_raw_full = 0;
            unsigned int _phase_a_empty = 1;
            #pragma unroll 1
            for (unsigned int w_4 = 0; w_4 < wl_items_4; w_4++) {
                int w_i_4 = (int)w_4;
                int wl_j_4 = w_i_4 - wl_c1_4;
                if (wl_j_4 < 0) {
                    w_i_4 = bid + w_i_4 * num_bids;
                } else if (wl_j_4 < wl_p1_4) {
                    w_i_4 = fc1_items_4 + wl_j_4 * (num_bids - wl_r_4) + (bid - wl_r_4);
                } else {
                    w_i_4 = fc1_items_4 + wl_n1_4 + (wl_j_4 - wl_p1_4) * num_bids + bid;
                }
                int is_fc1_4 = 0;
                int local_16 = w_i_4 - fc1_items_4;
                if (w_i_4 < fc1_items_4) {
                    is_fc1_4 = 1;
                    local_16 = w_i_4;
                }
                unsigned int ww_4 = (unsigned int)local_16;
                uint32_t _fast_div_q_4 = (fc1_m_div.divisor != 1) ? (__umulhi((uint32_t)(ww_4), fc1_m_div.multiplier) >> fc1_m_div.shift_right) : (uint32_t)(ww_4);
                uint32_t _fast_div_r_4 = (uint32_t)(ww_4) - _fast_div_q_4 * (uint32_t)(fc1_m_div.divisor);
                uint32_t _fast_div_q_5 = (fc2_m_div.divisor != 1) ? (__umulhi((uint32_t)(ww_4), fc2_m_div.multiplier) >> fc2_m_div.shift_right) : (uint32_t)(ww_4);
                uint32_t _fast_div_r_5 = (uint32_t)(ww_4) - _fast_div_q_5 * (uint32_t)(fc2_m_div.divisor);
                int t_7 = (int)_fast_div_q_5;
                int m_4 = (int)_fast_div_r_5;
                if (is_fc1_4 != 0) {
                    t_7 = (int)_fast_div_q_4;
                    m_4 = (int)_fast_div_r_4;
                }
                int num_k_3 = I / 256;
                if (is_fc1_4 != 0) {
                    num_k_3 = H / 256;
                }
                int src_row = fc2_row;
                if (is_fc1_4 != 0) {
                    src_row = fc1_row;
                }
                #pragma unroll 1
                for (int _kt = 0; _kt < num_k_3; _kt++) {
                    mbarrier_wait(raw_full_addr + (raw_cur_1) * 8, _phase_raw_full);
                    mbarrier_wait(a_empty_addr + (a_cur_1) * 8, _phase_a_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int raw_row = raw_smem_addr + raw_cur_1 * 16384 + (unsigned int)(src_row * 128);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(0) + 3]))
                        : "r"(raw_row + (unsigned int)(half * 64 ^ (src_row & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(4) + 3]))
                        : "r"(raw_row + (unsigned int)(half * 64 + 16 ^ (src_row & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(8) + 3]))
                        : "r"(raw_row + (unsigned int)(half * 64 + 32 ^ (src_row & 7) << 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(12) + 3]))
                        : "r"(raw_row + (unsigned int)(half * 64 + 48 ^ (src_row & 7) << 4)));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&scale_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&scale_words[(0) + 1]))
                        : "r"(sf_smem_addr + raw_cur_1 * 2048 + (unsigned int)(src_row * 16 + half * 8)));
                    unsigned int a_base = taddr + 64 + lane_base + (a_cur_1 * 128 + (unsigned int)(half * 64));
                    uint32_t _fp4_dequant_block32_bf16_0[16];
                    cvt_bf16x16_e2m1x32_e4m3x2_lo((uint32_t)(raw_words[0]), (uint32_t)(raw_words[1]), (uint32_t)(raw_words[2]), (uint32_t)(raw_words[3]), (uint32_t)(scale_words[0]), _fp4_dequant_block32_bf16_0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(a_base), "r"(_fp4_dequant_block32_bf16_0[0]), "r"(_fp4_dequant_block32_bf16_0[1]), "r"(_fp4_dequant_block32_bf16_0[2]), "r"(_fp4_dequant_block32_bf16_0[3]), "r"(_fp4_dequant_block32_bf16_0[4]), "r"(_fp4_dequant_block32_bf16_0[5]), "r"(_fp4_dequant_block32_bf16_0[6]), "r"(_fp4_dequant_block32_bf16_0[7]), "r"(_fp4_dequant_block32_bf16_0[8]), "r"(_fp4_dequant_block32_bf16_0[9]), "r"(_fp4_dequant_block32_bf16_0[10]), "r"(_fp4_dequant_block32_bf16_0[11]), "r"(_fp4_dequant_block32_bf16_0[12]), "r"(_fp4_dequant_block32_bf16_0[13]), "r"(_fp4_dequant_block32_bf16_0[14]), "r"(_fp4_dequant_block32_bf16_0[15]));
                    uint32_t _fp4_dequant_block32_bf16_1[16];
                    cvt_bf16x16_e2m1x32_e4m3x2_hi((uint32_t)(raw_words[4]), (uint32_t)(raw_words[5]), (uint32_t)(raw_words[6]), (uint32_t)(raw_words[7]), (uint32_t)(scale_words[0]), _fp4_dequant_block32_bf16_1);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(a_base + 16), "r"(_fp4_dequant_block32_bf16_1[0]), "r"(_fp4_dequant_block32_bf16_1[1]), "r"(_fp4_dequant_block32_bf16_1[2]), "r"(_fp4_dequant_block32_bf16_1[3]), "r"(_fp4_dequant_block32_bf16_1[4]), "r"(_fp4_dequant_block32_bf16_1[5]), "r"(_fp4_dequant_block32_bf16_1[6]), "r"(_fp4_dequant_block32_bf16_1[7]), "r"(_fp4_dequant_block32_bf16_1[8]), "r"(_fp4_dequant_block32_bf16_1[9]), "r"(_fp4_dequant_block32_bf16_1[10]), "r"(_fp4_dequant_block32_bf16_1[11]), "r"(_fp4_dequant_block32_bf16_1[12]), "r"(_fp4_dequant_block32_bf16_1[13]), "r"(_fp4_dequant_block32_bf16_1[14]), "r"(_fp4_dequant_block32_bf16_1[15]));
                    uint32_t _fp4_dequant_block32_bf16_2[16];
                    cvt_bf16x16_e2m1x32_e4m3x2_lo((uint32_t)(raw_words[8]), (uint32_t)(raw_words[9]), (uint32_t)(raw_words[10]), (uint32_t)(raw_words[11]), (uint32_t)(scale_words[1]), _fp4_dequant_block32_bf16_2);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(a_base + 32), "r"(_fp4_dequant_block32_bf16_2[0]), "r"(_fp4_dequant_block32_bf16_2[1]), "r"(_fp4_dequant_block32_bf16_2[2]), "r"(_fp4_dequant_block32_bf16_2[3]), "r"(_fp4_dequant_block32_bf16_2[4]), "r"(_fp4_dequant_block32_bf16_2[5]), "r"(_fp4_dequant_block32_bf16_2[6]), "r"(_fp4_dequant_block32_bf16_2[7]), "r"(_fp4_dequant_block32_bf16_2[8]), "r"(_fp4_dequant_block32_bf16_2[9]), "r"(_fp4_dequant_block32_bf16_2[10]), "r"(_fp4_dequant_block32_bf16_2[11]), "r"(_fp4_dequant_block32_bf16_2[12]), "r"(_fp4_dequant_block32_bf16_2[13]), "r"(_fp4_dequant_block32_bf16_2[14]), "r"(_fp4_dequant_block32_bf16_2[15]));
                    uint32_t _fp4_dequant_block32_bf16_3[16];
                    cvt_bf16x16_e2m1x32_e4m3x2_hi((uint32_t)(raw_words[12]), (uint32_t)(raw_words[13]), (uint32_t)(raw_words[14]), (uint32_t)(raw_words[15]), (uint32_t)(scale_words[1]), _fp4_dequant_block32_bf16_3);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(a_base + 48), "r"(_fp4_dequant_block32_bf16_3[0]), "r"(_fp4_dequant_block32_bf16_3[1]), "r"(_fp4_dequant_block32_bf16_3[2]), "r"(_fp4_dequant_block32_bf16_3[3]), "r"(_fp4_dequant_block32_bf16_3[4]), "r"(_fp4_dequant_block32_bf16_3[5]), "r"(_fp4_dequant_block32_bf16_3[6]), "r"(_fp4_dequant_block32_bf16_3[7]), "r"(_fp4_dequant_block32_bf16_3[8]), "r"(_fp4_dequant_block32_bf16_3[9]), "r"(_fp4_dequant_block32_bf16_3[10]), "r"(_fp4_dequant_block32_bf16_3[11]), "r"(_fp4_dequant_block32_bf16_3[12]), "r"(_fp4_dequant_block32_bf16_3[13]), "r"(_fp4_dequant_block32_bf16_3[14]), "r"(_fp4_dequant_block32_bf16_3[15]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (warp == 8) {
                        asm volatile("barrier.sync %0, 256;" :: "r"(3 + a_cur_1) : "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            mbarrier_arrive(a_full_addr + (a_cur_1) * 8);
                            mbarrier_arrive(raw_empty_addr + (raw_cur_1) * 8);
                        }
                    } else {
                        asm volatile("barrier.arrive %0, 256;" :: "r"(3 + a_cur_1) : "memory");
                    }
                    raw_cur_1 += 1;
                    if (raw_cur_1 == 6) { raw_cur_1 = 0; _phase_raw_full ^= 1; }
                    a_cur_1 += 1;
                    if (a_cur_1 == 3) { a_cur_1 = 0; _phase_a_empty ^= 1; }
                }
            }
            #pragma unroll 1
            for (int __1 = 0; __1 < 3; __1++) {
                mbarrier_wait(a_empty_addr + (a_cur_1) * 8, _phase_a_empty);
                a_cur_1 += 1;
                if (a_cur_1 == 3) { a_cur_1 = 0; _phase_a_empty ^= 1; }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
