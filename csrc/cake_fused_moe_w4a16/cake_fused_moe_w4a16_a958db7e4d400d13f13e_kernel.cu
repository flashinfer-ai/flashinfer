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
#define NUM_RAW_PIPE_STAGES 8
#define NUM_B_PIPE_STAGES 8
#define NUM_A_PIPE_STAGES 3
#define NUM_ACC_PIPE_STAGES 2
#define NUM_TILE_PIPE_STAGES 3
#define NUM_STORE_PIPE_STAGES 2
#define SMEM_RAW_OFF 1024
#define SMEM_RAW_STAGE_BYTES 16384
#define SMEM_RAW_STRIDE 16384
#define SMEM_SF_OFF 132096
#define SMEM_SF_STAGE_BYTES 2048
#define SMEM_SF_STRIDE 2048
#define SMEM_XB_OFF 148480
#define SMEM_XB_STAGE_BYTES 8192
#define SMEM_XB_STRIDE 8192
#define SMEM_STAGING_OFF 214016
#define SMEM_STAGING_STAGE_BYTES 4096
#define SMEM_STAGING_STRIDE 4096
#define SMEM_TILE_INFO_OFF 222208
#define SMEM_TILE_INFO_STAGE_BYTES 16
#define SMEM_TILE_INFO_STRIDE 16
#define SMEM_TOTAL 222336
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

__device__ __forceinline__ void cvt_bf16x8_e2m1x16_e4m3_sel0(uint32_t packed0, uint32_t packed1, uint32_t scale_word, uint32_t (&dst)[8]) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7; \n\t"
        "  .reg .b16 _sx2, _sxh;                      \n\t"
        "  .reg .b32 _sc, _sp, _z;                    \n\t"
        "  mov.b32 {_q0, _q1, _q2, _q3}, %8;          \n\t"
        "  mov.b32 {_q4, _q5, _q6, _q7}, %9;          \n\t"
        "  mov.b32 _z, 0;                             \n\t"
        "  prmt.b32 _sp, %10, _z, 0x7700;           \n\t"
        "  mov.b32 {_sx2, _sxh}, _sp;                 \n\t"
        "  cvt.rn.bf16x2.e4m3x2 _sc, _sx2;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _q0;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %1, _q1;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %2, _q2;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %3, _q3;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %4, _q4;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %5, _q5;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %6, _q6;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %7, _q7;        \n\t"
        "  mul.rn.bf16x2 %0, %0, _sc;            \n\t"
        "  mul.rn.bf16x2 %1, %1, _sc;            \n\t"
        "  mul.rn.bf16x2 %2, %2, _sc;            \n\t"
        "  mul.rn.bf16x2 %3, %3, _sc;            \n\t"
        "  mul.rn.bf16x2 %4, %4, _sc;            \n\t"
        "  mul.rn.bf16x2 %5, %5, _sc;            \n\t"
        "  mul.rn.bf16x2 %6, %6, _sc;            \n\t"
        "  mul.rn.bf16x2 %7, %7, _sc;            \n\t"
        "}"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]), "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
        : "r"(packed0), "r"(packed1), "r"(scale_word));
#else
    uint32_t _scale_byte = (scale_word >> 0) & 0xFFu;
    uint32_t _scale_p112 = cvt_bf16x2_e4m3x2_p112((uint16_t)(_scale_byte | (_scale_byte << 8)));
#pragma unroll
    for (int _k = 0; _k < 4; ++_k) {
        uint32_t _lo = cvt_bf16x2_e2m1x2_prescaled(packed0 >> (8 * _k));
        uint32_t _hi = cvt_bf16x2_e2m1x2_prescaled(packed1 >> (8 * _k));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[_k]) : "r"(_lo), "r"(_scale_p112));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[4 + _k]) : "r"(_hi), "r"(_scale_p112));
    }
#endif
}

__device__ __forceinline__ void cvt_bf16x8_e2m1x16_e4m3_sel1(uint32_t packed0, uint32_t packed1, uint32_t scale_word, uint32_t (&dst)[8]) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7; \n\t"
        "  .reg .b16 _sx2, _sxh;                      \n\t"
        "  .reg .b32 _sc, _sp, _z;                    \n\t"
        "  mov.b32 {_q0, _q1, _q2, _q3}, %8;          \n\t"
        "  mov.b32 {_q4, _q5, _q6, _q7}, %9;          \n\t"
        "  mov.b32 _z, 0;                             \n\t"
        "  prmt.b32 _sp, %10, _z, 0x7711;           \n\t"
        "  mov.b32 {_sx2, _sxh}, _sp;                 \n\t"
        "  cvt.rn.bf16x2.e4m3x2 _sc, _sx2;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _q0;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %1, _q1;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %2, _q2;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %3, _q3;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %4, _q4;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %5, _q5;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %6, _q6;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %7, _q7;        \n\t"
        "  mul.rn.bf16x2 %0, %0, _sc;            \n\t"
        "  mul.rn.bf16x2 %1, %1, _sc;            \n\t"
        "  mul.rn.bf16x2 %2, %2, _sc;            \n\t"
        "  mul.rn.bf16x2 %3, %3, _sc;            \n\t"
        "  mul.rn.bf16x2 %4, %4, _sc;            \n\t"
        "  mul.rn.bf16x2 %5, %5, _sc;            \n\t"
        "  mul.rn.bf16x2 %6, %6, _sc;            \n\t"
        "  mul.rn.bf16x2 %7, %7, _sc;            \n\t"
        "}"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]), "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
        : "r"(packed0), "r"(packed1), "r"(scale_word));
#else
    uint32_t _scale_byte = (scale_word >> 8) & 0xFFu;
    uint32_t _scale_p112 = cvt_bf16x2_e4m3x2_p112((uint16_t)(_scale_byte | (_scale_byte << 8)));
#pragma unroll
    for (int _k = 0; _k < 4; ++_k) {
        uint32_t _lo = cvt_bf16x2_e2m1x2_prescaled(packed0 >> (8 * _k));
        uint32_t _hi = cvt_bf16x2_e2m1x2_prescaled(packed1 >> (8 * _k));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[_k]) : "r"(_lo), "r"(_scale_p112));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[4 + _k]) : "r"(_hi), "r"(_scale_p112));
    }
#endif
}

__device__ __forceinline__ void cvt_bf16x8_e2m1x16_e4m3_sel2(uint32_t packed0, uint32_t packed1, uint32_t scale_word, uint32_t (&dst)[8]) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7; \n\t"
        "  .reg .b16 _sx2, _sxh;                      \n\t"
        "  .reg .b32 _sc, _sp, _z;                    \n\t"
        "  mov.b32 {_q0, _q1, _q2, _q3}, %8;          \n\t"
        "  mov.b32 {_q4, _q5, _q6, _q7}, %9;          \n\t"
        "  mov.b32 _z, 0;                             \n\t"
        "  prmt.b32 _sp, %10, _z, 0x7722;           \n\t"
        "  mov.b32 {_sx2, _sxh}, _sp;                 \n\t"
        "  cvt.rn.bf16x2.e4m3x2 _sc, _sx2;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _q0;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %1, _q1;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %2, _q2;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %3, _q3;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %4, _q4;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %5, _q5;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %6, _q6;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %7, _q7;        \n\t"
        "  mul.rn.bf16x2 %0, %0, _sc;            \n\t"
        "  mul.rn.bf16x2 %1, %1, _sc;            \n\t"
        "  mul.rn.bf16x2 %2, %2, _sc;            \n\t"
        "  mul.rn.bf16x2 %3, %3, _sc;            \n\t"
        "  mul.rn.bf16x2 %4, %4, _sc;            \n\t"
        "  mul.rn.bf16x2 %5, %5, _sc;            \n\t"
        "  mul.rn.bf16x2 %6, %6, _sc;            \n\t"
        "  mul.rn.bf16x2 %7, %7, _sc;            \n\t"
        "}"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]), "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
        : "r"(packed0), "r"(packed1), "r"(scale_word));
#else
    uint32_t _scale_byte = (scale_word >> 16) & 0xFFu;
    uint32_t _scale_p112 = cvt_bf16x2_e4m3x2_p112((uint16_t)(_scale_byte | (_scale_byte << 8)));
#pragma unroll
    for (int _k = 0; _k < 4; ++_k) {
        uint32_t _lo = cvt_bf16x2_e2m1x2_prescaled(packed0 >> (8 * _k));
        uint32_t _hi = cvt_bf16x2_e2m1x2_prescaled(packed1 >> (8 * _k));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[_k]) : "r"(_lo), "r"(_scale_p112));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[4 + _k]) : "r"(_hi), "r"(_scale_p112));
    }
#endif
}

__device__ __forceinline__ void cvt_bf16x8_e2m1x16_e4m3_sel3(uint32_t packed0, uint32_t packed1, uint32_t scale_word, uint32_t (&dst)[8]) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7; \n\t"
        "  .reg .b16 _sx2, _sxh;                      \n\t"
        "  .reg .b32 _sc, _sp, _z;                    \n\t"
        "  mov.b32 {_q0, _q1, _q2, _q3}, %8;          \n\t"
        "  mov.b32 {_q4, _q5, _q6, _q7}, %9;          \n\t"
        "  mov.b32 _z, 0;                             \n\t"
        "  prmt.b32 _sp, %10, _z, 0x7733;           \n\t"
        "  mov.b32 {_sx2, _sxh}, _sp;                 \n\t"
        "  cvt.rn.bf16x2.e4m3x2 _sc, _sx2;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _q0;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %1, _q1;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %2, _q2;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %3, _q3;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %4, _q4;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %5, _q5;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %6, _q6;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %7, _q7;        \n\t"
        "  mul.rn.bf16x2 %0, %0, _sc;            \n\t"
        "  mul.rn.bf16x2 %1, %1, _sc;            \n\t"
        "  mul.rn.bf16x2 %2, %2, _sc;            \n\t"
        "  mul.rn.bf16x2 %3, %3, _sc;            \n\t"
        "  mul.rn.bf16x2 %4, %4, _sc;            \n\t"
        "  mul.rn.bf16x2 %5, %5, _sc;            \n\t"
        "  mul.rn.bf16x2 %6, %6, _sc;            \n\t"
        "  mul.rn.bf16x2 %7, %7, _sc;            \n\t"
        "}"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]), "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
        : "r"(packed0), "r"(packed1), "r"(scale_word));
#else
    uint32_t _scale_byte = (scale_word >> 24) & 0xFFu;
    uint32_t _scale_p112 = cvt_bf16x2_e4m3x2_p112((uint16_t)(_scale_byte | (_scale_byte << 8)));
#pragma unroll
    for (int _k = 0; _k < 4; ++_k) {
        uint32_t _lo = cvt_bf16x2_e2m1x2_prescaled(packed0 >> (8 * _k));
        uint32_t _hi = cvt_bf16x2_e2m1x2_prescaled(packed1 >> (8 * _k));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[_k]) : "r"(_lo), "r"(_scale_p112));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[4 + _k]) : "r"(_hi), "r"(_scale_p112));
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






__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
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



__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_fused_moe_w4a16_a958db7e4d400d13f13e(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap S, const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap C, int* __restrict__ tile_idx_to_expert_idx, int* __restrict__ tile_idx_to_mn_limit, int* __restrict__ num_non_exiting_tiles, float* __restrict__ alpha, int K, int num_m_tiles, CakeFastDivmod m_tile_div)
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
    #define raw_empty_addr (mbar_base + 64)
    #define b_full_addr (mbar_base + 128)
    #define b_empty_addr (mbar_base + 192)
    #define a_full_addr (mbar_base + 256)
    #define a_empty_addr (mbar_base + 280)
    #define acc_full_addr (mbar_base + 304)
    #define acc_empty_addr (mbar_base + 320)
    #define tile_full_addr (mbar_base + 336)
    #define tile_empty_addr (mbar_base + 360)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* raw = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int raw_addr = smem + 1024;
    uint8_t* sf = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int sf_addr = smem + 132096;
    __nv_bfloat16* xb = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int xb_addr = smem + 148480;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 214016);
    const int staging_addr = smem + 214016;
    int* tile_info = reinterpret_cast<int*>(smem_raw + 222208);
    const int tile_info_addr = smem + 222208;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 48 barriers)
    // Mbarriers at smem_raw[0..384)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'raw_pipe' ---
            // raw_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // raw_empty: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // --- pipeline 'b_pipe' ---
            // b_full: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // b_empty: 8 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // --- pipeline 'a_pipe' ---
            // a_full: 3 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // a_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // acc_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 320, 4);
            mbarrier_init(smem + 328, 4);
            // --- pipeline 'tile_pipe' ---
            // tile_full: 3 barriers, init_count=1
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            // tile_empty: 3 barriers, init_count=5
            mbarrier_init(smem + 360, 5);
            mbarrier_init(smem + 368, 5);
            mbarrier_init(smem + 376, 5);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 448 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 384);
    if (warp == 0) {
        int _tmem_hold = smem + 384;
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
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // epilogue_main
            int nnet = num_non_exiting_tiles[0];
            int end = nnet * num_m_tiles;
            const int quarter = warp % 4;
            const unsigned int row_base = (unsigned int)(quarter * 32 << 16);
            const unsigned int ch_off = (unsigned int)((quarter * 32 + lane) * 2);
            unsigned int ti = 0;
            unsigned int accs = 0;
            unsigned int out = 0;
            int info[4];
            unsigned int _phase_tile_full = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int item = bid; item < end; item += num_bids) {
                uint32_t _fast_div_q_3 = (m_tile_div.divisor != 1) ? (__umulhi((uint32_t)(item), m_tile_div.multiplier) >> m_tile_div.shift_right) : (uint32_t)(item);
                uint32_t _fast_div_r_3 = (uint32_t)(item) - _fast_div_q_3 * (uint32_t)(m_tile_div.divisor);
                mbarrier_wait(tile_full_addr + (ti) * 8, _phase_tile_full);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&info[0])), "=r"(*reinterpret_cast<uint32_t*>(&info[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&info[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&info[(0) + 3]))
                    : "r"(tile_info_addr + ti * 16));
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(tile_empty_addr + (ti) * 8);
                }
                ti += 1;
                if (ti == 3) { ti = 0; _phase_tile_full ^= 1; }
                int expert = info[0];
                float alpha_e = alpha[expert];
                int col_m = (int)_fast_div_r_3 * 128;
                int row_t = (int)_fast_div_q_3 * 16;
                mbarrier_wait(acc_full_addr + (accs) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int s = 0; s < 1; s++) {
                    unsigned int col = accs * 32 + (unsigned int)(s * 32);
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], taddr + row_base + col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (s == 0) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (elect_sync()) {
                            mbarrier_arrive(acc_empty_addr + (accs) * 8);
                        }
                    }
                    {
                        __nv_bfloat16 _bval_0 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[0], alpha_e));
                        uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                        uint32_t _addr_0 = static_cast<uint32_t>((staging_addr + out * 4096 + ch_off));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_1 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[1], alpha_e));
                        uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                        uint32_t _addr_1 = static_cast<uint32_t>((staging_addr + out * 4096 + (256 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_2 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[2], alpha_e));
                        uint16_t _bits_2 = *(uint16_t*)&_bval_2;
                        uint32_t _addr_2 = static_cast<uint32_t>((staging_addr + out * 4096 + (512 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_2), "h"(_bits_2) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_3 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[3], alpha_e));
                        uint16_t _bits_3 = *(uint16_t*)&_bval_3;
                        uint32_t _addr_3 = static_cast<uint32_t>((staging_addr + out * 4096 + (768 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_3), "h"(_bits_3) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_4 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[4], alpha_e));
                        uint16_t _bits_4 = *(uint16_t*)&_bval_4;
                        uint32_t _addr_4 = static_cast<uint32_t>((staging_addr + out * 4096 + (1024 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_4), "h"(_bits_4) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_5 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[5], alpha_e));
                        uint16_t _bits_5 = *(uint16_t*)&_bval_5;
                        uint32_t _addr_5 = static_cast<uint32_t>((staging_addr + out * 4096 + (1280 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_5), "h"(_bits_5) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_6 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[6], alpha_e));
                        uint16_t _bits_6 = *(uint16_t*)&_bval_6;
                        uint32_t _addr_6 = static_cast<uint32_t>((staging_addr + out * 4096 + (1536 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_6), "h"(_bits_6) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_7 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[7], alpha_e));
                        uint16_t _bits_7 = *(uint16_t*)&_bval_7;
                        uint32_t _addr_7 = static_cast<uint32_t>((staging_addr + out * 4096 + (1792 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_7), "h"(_bits_7) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_8 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[8], alpha_e));
                        uint16_t _bits_8 = *(uint16_t*)&_bval_8;
                        uint32_t _addr_8 = static_cast<uint32_t>((staging_addr + out * 4096 + (2048 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_8), "h"(_bits_8) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_9 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[9], alpha_e));
                        uint16_t _bits_9 = *(uint16_t*)&_bval_9;
                        uint32_t _addr_9 = static_cast<uint32_t>((staging_addr + out * 4096 + (2304 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_9), "h"(_bits_9) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_10 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[10], alpha_e));
                        uint16_t _bits_10 = *(uint16_t*)&_bval_10;
                        uint32_t _addr_10 = static_cast<uint32_t>((staging_addr + out * 4096 + (2560 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_10), "h"(_bits_10) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_11 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[11], alpha_e));
                        uint16_t _bits_11 = *(uint16_t*)&_bval_11;
                        uint32_t _addr_11 = static_cast<uint32_t>((staging_addr + out * 4096 + (2816 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_11), "h"(_bits_11) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_12 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[12], alpha_e));
                        uint16_t _bits_12 = *(uint16_t*)&_bval_12;
                        uint32_t _addr_12 = static_cast<uint32_t>((staging_addr + out * 4096 + (3072 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_12), "h"(_bits_12) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_13 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[13], alpha_e));
                        uint16_t _bits_13 = *(uint16_t*)&_bval_13;
                        uint32_t _addr_13 = static_cast<uint32_t>((staging_addr + out * 4096 + (3328 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_13), "h"(_bits_13) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_14 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[14], alpha_e));
                        uint16_t _bits_14 = *(uint16_t*)&_bval_14;
                        uint32_t _addr_14 = static_cast<uint32_t>((staging_addr + out * 4096 + (3584 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_14), "h"(_bits_14) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_15 = __float2bfloat16_rn(mul_f32_rn_noftz(_tmem_load_0[15], alpha_e));
                        uint16_t _bits_15 = *(uint16_t*)&_bval_15;
                        uint32_t _addr_15 = static_cast<uint32_t>((staging_addr + out * 4096 + (3840 + ch_off)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_15), "h"(_bits_15) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            tma_store_2d((&C), col_m, row_t + s * 32, staging_addr + out * 4096);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    out += 1;
                    if (out == 2) { out = 0; }
                }
                accs += 1;
                if (accs == 2) { accs = 0; _phase_acc_full ^= 1; }
            }
            if (warp == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int nnet_1 = num_non_exiting_tiles[0];
            int end_1 = nnet_1 * num_m_tiles;
            int num_k = K / 256;
            unsigned int bs = 0;
            unsigned int a_s = 0;
            unsigned int accs_1 = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int _item = bid; _item < end_1; _item += num_bids) {
                    mbarrier_wait(acc_empty_addr + (accs_1) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int kt = 0; kt < num_k; kt++) {
                        mbarrier_wait(a_full_addr + (a_s) * 8, _phase_a_full);
                        mbarrier_wait(b_full_addr + (bs) * 8, _phase_b_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int first_stage = ((kt == 0) ? 1 : 0);
                        int _smem_desc_lo_0 = (((xb_addr) >> 4) & 0x3FFF) + (bs) * 512;
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
                    "mov.b32 id, 134481040;\n\t"
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
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 blo, blo, 122;\n\t"
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
                    :: "r"((tmem_acc + (accs_1 * 32))), "r"(_smem_desc_lo_0), "r"((unsigned int)tmem_a_bf16 + a_s * 128), "r"(((first_stage) ? 0 : 1)) : "memory");
                        tcgen05_commit(b_empty_addr + (bs) * 8);
                        tcgen05_commit(a_empty_addr + (a_s) * 8);
                        bs += 1;
                        if (bs == 8) { bs = 0; _phase_b_full ^= 1; }
                        a_s += 1;
                        if (a_s == 3) { a_s = 0; _phase_a_full ^= 1; }
                    }
                    tcgen05_commit(acc_full_addr + (accs_1) * 8);
                    accs_1 += 1;
                    if (accs_1 == 2) { accs_1 = 0; _phase_acc_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 5) {
        { // scheduler_main
            int nnet_2 = num_non_exiting_tiles[0];
            int end_2 = nnet_2 * num_m_tiles;
            unsigned int ti_1 = 0;
            unsigned int _phase_tile_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int item_1 = bid; item_1 < end_2; item_1 += num_bids) {
                    uint32_t _fast_div_q_0 = (m_tile_div.divisor != 1) ? (__umulhi((uint32_t)(item_1), m_tile_div.multiplier) >> m_tile_div.shift_right) : (uint32_t)(item_1);
                    uint32_t _fast_div_r_0 = (uint32_t)(item_1) - _fast_div_q_0 * (uint32_t)(m_tile_div.divisor);
                    int expert_1 = tile_idx_to_expert_idx[_fast_div_q_0];
                    int dist = tile_idx_to_mn_limit[_fast_div_q_0] - (int)_fast_div_q_0 * 16;
                    mbarrier_wait(tile_empty_addr + (ti_1) * 8, _phase_tile_empty);
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(tile_info_addr + ti_1 * 16), "r"(expert_1), "r"(dist) : "memory");
                    mbarrier_arrive(tile_full_addr + (ti_1) * 8);
                    ti_1 += 1;
                    if (ti_1 == 3) { ti_1 = 0; _phase_tile_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: weight_tma ----
    if (warp == 6) {
        { // weight_tma_main
            int nnet_3 = num_non_exiting_tiles[0];
            int end_3 = nnet_3 * num_m_tiles;
            int num_k_1 = K / 256;
            unsigned int ti_2 = 0;
            unsigned int rs = 0;
            int info_1[4];
            unsigned int _phase_tile_full_1 = 0;
            unsigned int _phase_raw_empty = 1;
            #pragma unroll 1
            for (unsigned int item_2 = bid; item_2 < end_3; item_2 += num_bids) {
                uint32_t _fast_div_q_1 = (m_tile_div.divisor != 1) ? (__umulhi((uint32_t)(item_2), m_tile_div.multiplier) >> m_tile_div.shift_right) : (uint32_t)(item_2);
                uint32_t _fast_div_r_1 = (uint32_t)(item_2) - _fast_div_q_1 * (uint32_t)(m_tile_div.divisor);
                mbarrier_wait(tile_full_addr + (ti_2) * 8, _phase_tile_full_1);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&info_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&info_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&info_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&info_1[(0) + 3]))
                    : "r"(tile_info_addr + ti_2 * 16));
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(tile_empty_addr + (ti_2) * 8);
                }
                ti_2 += 1;
                if (ti_2 == 3) { ti_2 = 0; _phase_tile_full_1 ^= 1; }
                int expert_2 = info_1[0];
                int m_row = (int)_fast_div_r_1 * 128;
                #pragma unroll 1
                for (int kt_1 = 0; kt_1 < num_k_1; kt_1++) {
                    mbarrier_wait(raw_empty_addr + (rs) * 8, _phase_raw_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(raw_addr + rs * 16384), "l"((&W)), "r"(kt_1 * 128), "r"(m_row), "r"(expert_2),
                               "r"(raw_full_addr + (rs) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(sf_addr + rs * 2048), "l"((&S)), "r"(kt_1 * 16), "r"(m_row), "r"(expert_2),
                               "r"(raw_full_addr + (rs) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(raw_full_addr + (rs) * 8, 18432);
                    }
                    rs += 1;
                    if (rs == 8) { rs = 0; _phase_raw_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: b_tma ----
    if (warp == 7) {
        { // b_tma_main
            int nnet_4 = num_non_exiting_tiles[0];
            int end_4 = nnet_4 * num_m_tiles;
            int num_k_2 = K / 256;
            unsigned int bs_1 = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_b_empty = 1;
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                #pragma unroll 1
                for (unsigned int item_3 = bid; item_3 < end_4; item_3 += num_bids) {
                    uint32_t _fast_div_q_2 = (m_tile_div.divisor != 1) ? (__umulhi((uint32_t)(item_3), m_tile_div.multiplier) >> m_tile_div.shift_right) : (uint32_t)(item_3);
                    uint32_t _fast_div_r_2 = (uint32_t)(item_3) - _fast_div_q_2 * (uint32_t)(m_tile_div.divisor);
                    int row0 = (int)_fast_div_q_2 * 16;
                    #pragma unroll 1
                    for (int kt_2 = 0; kt_2 < num_k_2; kt_2++) {
                        mbarrier_wait(b_empty_addr + (bs_1) * 8, _phase_b_empty);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(xb_addr + bs_1 * 8192), "l"((&X)), "r"(0), "r"(row0), "r"(kt_2 * 4),
                               "r"(b_full_addr + (bs_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(b_full_addr + (bs_1) * 8, 8192);
                        bs_1 += 1;
                        if (bs_1 == 8) { bs_1 = 0; _phase_b_empty ^= 1; }
                    }
                }
                #pragma unroll 1
                for (int _ = 0; _ < 8; _++) {
                    mbarrier_wait(b_empty_addr + (bs_1) * 8, _phase_b_empty);
                    bs_1 += 1;
                    if (bs_1 == 8) { bs_1 = 0; _phase_b_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: transform ----
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // transform_main
            int nnet_5 = num_non_exiting_tiles[0];
            int end_5 = nnet_5 * num_m_tiles;
            int num_k_3 = K / 256;
            const int quarter_1 = warp % 4;
            const int half = (warp - 8) / 4;
            const int row = quarter_1 * 32 + lane;
            const unsigned int row_base_1 = (unsigned int)(quarter_1 * 32 << 16);
            const int ld_off0 = row * 128 + (half * 64 ^ (row & 7) << 4);
            const int ld_off1 = row * 128 + (half * 64 + 16 ^ (row & 7) << 4);
            const int ld_off2 = row * 128 + (half * 64 + 32 ^ (row & 7) << 4);
            const int ld_off3 = row * 128 + (half * 64 + 48 ^ (row & 7) << 4);
            const int sf_off_thread = row * 16 + half * 8;
            const unsigned int a_half = (unsigned int)(half * 64);
            unsigned int raw_w[16];
            unsigned int sc_w[2];
            unsigned int rs_1 = 0;
            unsigned int a_s_1 = 0;
            int items_cta = (end_5 - (int)bid + (int)num_bids - 1) / (int)num_bids;
            int total = items_cta * num_k_3;
            unsigned int raw_pub = 0;
            unsigned int a_pub = 0;
            unsigned int _phase_raw_full = 0;
            unsigned int _phase_a_empty = 1;
            #pragma unroll 1
            for (int s_1 = 0; s_1 < total; s_1++) {
                mbarrier_wait(raw_full_addr + (rs_1) * 8, _phase_raw_full);
                mbarrier_wait(a_empty_addr + (a_s_1) * 8, _phase_a_empty);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int raw_base = raw_addr + rs_1 * 16384;
                int sf_base = sf_addr + rs_1 * 2048;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_w[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_w[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_w[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_w[(0) + 3]))
                    : "r"(raw_base + ld_off0));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_w[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw_w[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_w[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_w[(4) + 3]))
                    : "r"(raw_base + ld_off1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_w[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw_w[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_w[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_w[(8) + 3]))
                    : "r"(raw_base + ld_off2));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_w[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw_w[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_w[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_w[(12) + 3]))
                    : "r"(raw_base + ld_off3));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sc_w[0])), "=r"(*reinterpret_cast<uint32_t*>(&sc_w[(0) + 1]))
                    : "r"(sf_base + sf_off_thread));
                if (s_1 > 0) {
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (warp == 8) {
                        if (elect_sync()) {
                            mbarrier_arrive(a_full_addr + (a_pub) * 8);
                            mbarrier_arrive(raw_empty_addr + (raw_pub) * 8);
                        }
                    }
                    raw_pub += 1;
                    if (raw_pub == 8) { raw_pub = 0; }
                    a_pub += 1;
                    if (a_pub == 3) { a_pub = 0; }
                }
                unsigned int a_addr = taddr + 64 + row_base_1 + (a_s_1 * 128 + a_half);
                uint32_t _fp4_dequant_block16_sel_bf16_0[8];
                cvt_bf16x8_e2m1x16_e4m3_sel0((uint32_t)(raw_w[0]), (uint32_t)(raw_w[1]), (uint32_t)(sc_w[0]), _fp4_dequant_block16_sel_bf16_0);
                tmem_st_x8_u32(a_addr, (const uint32_t*)_fp4_dequant_block16_sel_bf16_0);
                uint32_t _fp4_dequant_block16_sel_bf16_1[8];
                cvt_bf16x8_e2m1x16_e4m3_sel1((uint32_t)(raw_w[2]), (uint32_t)(raw_w[3]), (uint32_t)(sc_w[0]), _fp4_dequant_block16_sel_bf16_1);
                tmem_st_x8_u32(a_addr + 8, (const uint32_t*)_fp4_dequant_block16_sel_bf16_1);
                uint32_t _fp4_dequant_block16_sel_bf16_2[8];
                cvt_bf16x8_e2m1x16_e4m3_sel2((uint32_t)(raw_w[4]), (uint32_t)(raw_w[5]), (uint32_t)(sc_w[0]), _fp4_dequant_block16_sel_bf16_2);
                tmem_st_x8_u32(a_addr + 16, (const uint32_t*)_fp4_dequant_block16_sel_bf16_2);
                uint32_t _fp4_dequant_block16_sel_bf16_3[8];
                cvt_bf16x8_e2m1x16_e4m3_sel3((uint32_t)(raw_w[6]), (uint32_t)(raw_w[7]), (uint32_t)(sc_w[0]), _fp4_dequant_block16_sel_bf16_3);
                tmem_st_x8_u32(a_addr + 24, (const uint32_t*)_fp4_dequant_block16_sel_bf16_3);
                uint32_t _fp4_dequant_block16_sel_bf16_4[8];
                cvt_bf16x8_e2m1x16_e4m3_sel0((uint32_t)(raw_w[8]), (uint32_t)(raw_w[9]), (uint32_t)(sc_w[1]), _fp4_dequant_block16_sel_bf16_4);
                tmem_st_x8_u32(a_addr + 32, (const uint32_t*)_fp4_dequant_block16_sel_bf16_4);
                uint32_t _fp4_dequant_block16_sel_bf16_5[8];
                cvt_bf16x8_e2m1x16_e4m3_sel1((uint32_t)(raw_w[10]), (uint32_t)(raw_w[11]), (uint32_t)(sc_w[1]), _fp4_dequant_block16_sel_bf16_5);
                tmem_st_x8_u32(a_addr + 40, (const uint32_t*)_fp4_dequant_block16_sel_bf16_5);
                uint32_t _fp4_dequant_block16_sel_bf16_6[8];
                cvt_bf16x8_e2m1x16_e4m3_sel2((uint32_t)(raw_w[12]), (uint32_t)(raw_w[13]), (uint32_t)(sc_w[1]), _fp4_dequant_block16_sel_bf16_6);
                tmem_st_x8_u32(a_addr + 48, (const uint32_t*)_fp4_dequant_block16_sel_bf16_6);
                uint32_t _fp4_dequant_block16_sel_bf16_7[8];
                cvt_bf16x8_e2m1x16_e4m3_sel3((uint32_t)(raw_w[14]), (uint32_t)(raw_w[15]), (uint32_t)(sc_w[1]), _fp4_dequant_block16_sel_bf16_7);
                tmem_st_x8_u32(a_addr + 56, (const uint32_t*)_fp4_dequant_block16_sel_bf16_7);
                rs_1 += 1;
                if (rs_1 == 8) { rs_1 = 0; _phase_raw_full ^= 1; }
                a_s_1 += 1;
                if (a_s_1 == 3) { a_s_1 = 0; _phase_a_empty ^= 1; }
            }
            if (total > 0) {
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 2, 256;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(a_full_addr + (a_pub) * 8);
                        mbarrier_arrive(raw_empty_addr + (raw_pub) * 8);
                    }
                }
                raw_pub += 1;
                if (raw_pub == 8) { raw_pub = 0; }
                a_pub += 1;
                if (a_pub == 3) { a_pub = 0; }
            }
            #pragma unroll 1
            for (int __1 = 0; __1 < 3; __1++) {
                mbarrier_wait(a_empty_addr + (a_s_1) * 8, _phase_a_empty);
                a_s_1 += 1;
                if (a_s_1 == 3) { a_s_1 = 0; _phase_a_empty ^= 1; }
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
