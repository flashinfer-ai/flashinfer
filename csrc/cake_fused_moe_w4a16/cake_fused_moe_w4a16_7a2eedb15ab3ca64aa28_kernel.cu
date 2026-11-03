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
#define NUM_RAW_RING_STAGES 10
#define NUM_B_RING_STAGES 10
#define NUM_A_RING_STAGES 3
#define NUM_ACC_RING_STAGES 2
#define NUM_TILE_RING_STAGES 3
#define NUM_STORE_RING_STAGES 2
#define SMEM_RAW_SMEM_OFF 1024
#define SMEM_RAW_SMEM_STAGE_BYTES 16384
#define SMEM_RAW_SMEM_STRIDE 16384
#define SMEM_SF_SMEM_OFF 164864
#define SMEM_SF_SMEM_STAGE_BYTES 2048
#define SMEM_SF_SMEM_STRIDE 2048
#define SMEM_SMEM_B_OFF 185344
#define SMEM_SMEM_B_STAGE_BYTES 4096
#define SMEM_SMEM_B_STRIDE 4096
#define SMEM_STAGING_OFF 226304
#define SMEM_STAGING_STAGE_BYTES 1024
#define SMEM_STAGING_STRIDE 1024
#define SMEM_TILE_INFO_OFF 228352
#define SMEM_TILE_INFO_STAGE_BYTES 16
#define SMEM_TILE_INFO_STRIDE 16
#define SMEM_TOTAL 228480
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
kernel_cake_fused_moe_w4a16_7a2eedb15ab3ca64aa28(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap S, const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap C, int* __restrict__ tile_idx_to_expert_idx, int* __restrict__ tile_idx_to_mn_limit, int* __restrict__ num_non_exiting_tiles, float* __restrict__ alpha, float swiglu_limit, int K, int num_m_tiles, CakeFastDivmod m_tile_div, int* __restrict__ permuted_idx_to_expanded_idx, CakeFastDivmod top_k_div)
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
    #define raw_empty_addr (mbar_base + 80)
    #define b_full_addr (mbar_base + 160)
    #define b_empty_addr (mbar_base + 240)
    #define a_full_addr (mbar_base + 320)
    #define a_empty_addr (mbar_base + 344)
    #define acc_full_addr (mbar_base + 368)
    #define acc_empty_addr (mbar_base + 384)
    #define tile_full_addr (mbar_base + 400)
    #define tile_empty_addr (mbar_base + 424)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* raw_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int raw_smem_addr = smem + 1024;
    uint8_t* sf_smem = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int sf_smem_addr = smem + 164864;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 185344);
    const int smem_b_addr = smem + 185344;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 226304);
    const int staging_addr = smem + 226304;
    int* tile_info = reinterpret_cast<int*>(smem_raw + 228352);
    const int tile_info_addr = smem + 228352;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 56 barriers)
    // Mbarriers at smem_raw[0..448)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'raw_ring' ---
            // raw_full: 10 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // raw_empty: 10 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'b_ring' ---
            // b_full: 10 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // b_empty: 10 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // --- pipeline 'a_ring' ---
            // a_full: 3 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            // a_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            // --- pipeline 'acc_ring' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // acc_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 384, 4);
            mbarrier_init(smem + 392, 4);
            // --- pipeline 'tile_ring' ---
            // tile_full: 3 barriers, init_count=1
            mbarrier_init(smem + 400, 1);
            mbarrier_init(smem + 408, 1);
            mbarrier_init(smem + 416, 1);
            // tile_empty: 3 barriers, init_count=5
            mbarrier_init(smem + 424, 5);
            mbarrier_init(smem + 432, 5);
            mbarrier_init(smem + 440, 5);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 448 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 448);
    if (warp == 0) {
        int _tmem_hold = smem + 448;
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
            int valid_tiles = num_non_exiting_tiles[0];
            int end = valid_tiles * num_m_tiles;
            const int quarter = warp % 4;
            const unsigned int gate_lanes = (unsigned int)(quarter * 32) << 16;
            const unsigned int up_lanes = (unsigned int)(quarter * 32 + 16) << 16;
            const int thread_off = lane / 2 % 2 * 128 + (quarter * 16 + lane / 4 + lane % 2 * 8) * 2;
            float _neg_0;
            asm("neg.f32 %0, %1;" : "=f"(_neg_0) : "f"(swiglu_limit));
            const float neg_limit = _neg_0;
            unsigned int tile_cur = 0;
            unsigned int acc_cur = 0;
            unsigned int out_cur = 0;
            int record[2];
            unsigned int _phase_tile_full = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int w = bid; w < end; w += num_bids) {
                uint32_t _fast_div_q_11 = (m_tile_div.divisor != 1) ? (__umulhi((uint32_t)(w), m_tile_div.multiplier) >> m_tile_div.shift_right) : (uint32_t)(w);
                uint32_t _fast_div_r_11 = (uint32_t)(w) - _fast_div_q_11 * (uint32_t)(m_tile_div.divisor);
                int t = (int)_fast_div_q_11;
                int m_cta = (int)_fast_div_r_11;
                mbarrier_wait(tile_full_addr + (tile_cur) * 8, _phase_tile_full);
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&record[0])), "=r"(*reinterpret_cast<uint32_t*>(&record[(0) + 1]))
                    : "r"(tile_info_addr + tile_cur * 16));
                int expert = record[0];
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(tile_empty_addr + (tile_cur) * 8);
                }
                tile_cur += 1;
                if (tile_cur == 3) { tile_cur = 0; _phase_tile_full ^= 1; }
                float alpha_e = alpha[expert];
                int feature0 = m_cta * 64;
                int row0 = t * 8;
                mbarrier_wait(acc_full_addr + (acc_cur) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int sub = 0; sub < 1; sub++) {
                    unsigned int col = acc_cur * 32 + (unsigned int)(sub * 32);
                    float _tmem_load_0[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x64b.x4.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                        : "r"(taddr + gate_lanes + col));
                    float _tmem_load_1[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x64b.x4.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                        : "r"(taddr + up_lanes + col));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (sub == 0) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (elect_sync()) {
                            mbarrier_arrive(acc_empty_addr + (acc_cur) * 8);
                        }
                    }
                    unsigned int out_base = staging_addr + out_cur * 1024 + (unsigned int)thread_off;
                    float g;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g) : "f"(_tmem_load_0[0]), "f"(alpha_e));
                    float u;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u) : "f"(_tmem_load_1[0]), "f"(alpha_e));
                    float _min_1;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_1) : "f"(g), "f"(swiglu_limit));
                    g = _min_1;
                    float _min_3;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_3) : "f"(u), "f"(swiglu_limit));
                    float _max_2;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_2) : "f"(_min_3), "f"(neg_limit));
                    u = _max_2;
                    float _exp2_6 = approx_exp2(mul_f32_rn_noftz(g, -1.442695f));
                    float _add_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(_exp2_6), "f"(1.0f));
                    float _rcp_4 = approx_rcp(_add_5);
                    {
                        __nv_bfloat16 _bval_7 = __float2bfloat16_rn(mul_f32_rn_noftz(u, mul_f32_rn_noftz(g, _rcp_4)));
                        uint16_t _bits_7 = *(uint16_t*)&_bval_7;
                        uint32_t _addr_7 = static_cast<uint32_t>(out_base);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_7), "h"(_bits_7) : "memory");
                    }
                    float g_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_0) : "f"(_tmem_load_0[1]), "f"(alpha_e));
                    float u_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_1) : "f"(_tmem_load_1[1]), "f"(alpha_e));
                    float _min_8;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_8) : "f"(g_0), "f"(swiglu_limit));
                    g_0 = _min_8;
                    float _min_10;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_10) : "f"(u_1), "f"(swiglu_limit));
                    float _max_9;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_9) : "f"(_min_10), "f"(neg_limit));
                    u_1 = _max_9;
                    float _exp2_13 = approx_exp2(mul_f32_rn_noftz(g_0, -1.442695f));
                    float _add_12;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(_exp2_13), "f"(1.0f));
                    float _rcp_11 = approx_rcp(_add_12);
                    {
                        __nv_bfloat16 _bval_14 = __float2bfloat16_rn(mul_f32_rn_noftz(u_1, mul_f32_rn_noftz(g_0, _rcp_11)));
                        uint16_t _bits_14 = *(uint16_t*)&_bval_14;
                        uint32_t _addr_14 = static_cast<uint32_t>(out_base + 256);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_14), "h"(_bits_14) : "memory");
                    }
                    float g_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_2) : "f"(_tmem_load_0[2]), "f"(alpha_e));
                    float u_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_3) : "f"(_tmem_load_1[2]), "f"(alpha_e));
                    float _min_15;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_15) : "f"(g_2), "f"(swiglu_limit));
                    g_2 = _min_15;
                    float _min_17;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_17) : "f"(u_3), "f"(swiglu_limit));
                    float _max_16;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_16) : "f"(_min_17), "f"(neg_limit));
                    u_3 = _max_16;
                    float _exp2_20 = approx_exp2(mul_f32_rn_noftz(g_2, -1.442695f));
                    float _add_19;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(_exp2_20), "f"(1.0f));
                    float _rcp_18 = approx_rcp(_add_19);
                    {
                        __nv_bfloat16 _bval_21 = __float2bfloat16_rn(mul_f32_rn_noftz(u_3, mul_f32_rn_noftz(g_2, _rcp_18)));
                        uint16_t _bits_21 = *(uint16_t*)&_bval_21;
                        uint32_t _addr_21 = static_cast<uint32_t>(out_base + 512);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_21), "h"(_bits_21) : "memory");
                    }
                    float g_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(g_4) : "f"(_tmem_load_0[3]), "f"(alpha_e));
                    float u_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(u_5) : "f"(_tmem_load_1[3]), "f"(alpha_e));
                    float _min_22;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_22) : "f"(g_4), "f"(swiglu_limit));
                    g_4 = _min_22;
                    float _min_24;
                    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_24) : "f"(u_5), "f"(swiglu_limit));
                    float _max_23;
                    asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_23) : "f"(_min_24), "f"(neg_limit));
                    u_5 = _max_23;
                    float _exp2_27 = approx_exp2(mul_f32_rn_noftz(g_4, -1.442695f));
                    float _add_26;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(_exp2_27), "f"(1.0f));
                    float _rcp_25 = approx_rcp(_add_26);
                    {
                        __nv_bfloat16 _bval_28 = __float2bfloat16_rn(mul_f32_rn_noftz(u_5, mul_f32_rn_noftz(g_4, _rcp_25)));
                        uint16_t _bits_28 = *(uint16_t*)&_bval_28;
                        uint32_t _addr_28 = static_cast<uint32_t>(out_base + 768);
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_28), "h"(_bits_28) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2}], [%3], %4;"
                                :: "l"((&C)), "r"(feature0), "r"(row0 + sub * 32), "r"(staging_addr + out_cur * 1024), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    out_cur += 1;
                    if (out_cur == 2) { out_cur = 0; }
                }
                acc_cur += 1;
                if (acc_cur == 2) { acc_cur = 0; _phase_acc_full ^= 1; }
            }
            if (warp == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int valid_tiles_1 = num_non_exiting_tiles[0];
            int end_1 = valid_tiles_1 * num_m_tiles;
            int num_k = K / 256;
            unsigned int a_cur = 0;
            unsigned int b_cur = 0;
            unsigned int acc_cur_1 = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int _w = bid; _w < end_1; _w += num_bids) {
                    mbarrier_wait(acc_empty_addr + (acc_cur_1) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int kt = 0; kt < num_k; kt++) {
                        mbarrier_wait(a_full_addr + (a_cur) * 8, _phase_a_full);
                        mbarrier_wait(b_full_addr + (b_cur) * 8, _phase_b_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (b_cur) * 256;
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
                    "mov.b32 id, 134349968;\n\t"
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
                    "add.u32 blo, blo, 58;\n\t"
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
                    "add.u32 blo, blo, 58;\n\t"
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
                    "add.u32 blo, blo, 58;\n\t"
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
                        if (b_cur == 10) { b_cur = 0; _phase_b_full ^= 1; }
                    }
                    tcgen05_commit(acc_full_addr + (acc_cur_1) * 8);
                    acc_cur_1 += 1;
                    if (acc_cur_1 == 2) { acc_cur_1 = 0; _phase_acc_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 5) {
        { // scheduler_main
            int valid_tiles_2 = num_non_exiting_tiles[0];
            int end_2 = valid_tiles_2 * num_m_tiles;
            unsigned int tile_cur_1 = 0;
            unsigned int _phase_tile_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int w_1 = bid; w_1 < end_2; w_1 += num_bids) {
                    uint32_t _fast_div_q_0 = (m_tile_div.divisor != 1) ? (__umulhi((uint32_t)(w_1), m_tile_div.multiplier) >> m_tile_div.shift_right) : (uint32_t)(w_1);
                    uint32_t _fast_div_r_0 = (uint32_t)(w_1) - _fast_div_q_0 * (uint32_t)(m_tile_div.divisor);
                    int t_1 = (int)_fast_div_q_0;
                    int m_cta_1 = (int)_fast_div_r_0;
                    int expert_1 = tile_idx_to_expert_idx[t_1];
                    int valid_rows = tile_idx_to_mn_limit[t_1] - t_1 * 8;
                    mbarrier_wait(tile_empty_addr + (tile_cur_1) * 8, _phase_tile_empty);
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(tile_info_addr + tile_cur_1 * 16), "r"(expert_1), "r"(valid_rows) : "memory");
                    mbarrier_arrive(tile_full_addr + (tile_cur_1) * 8);
                    tile_cur_1 += 1;
                    if (tile_cur_1 == 3) { tile_cur_1 = 0; _phase_tile_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: weight_tma ----
    if (warp == 6) {
        { // weight_tma_main
            int valid_tiles_3 = num_non_exiting_tiles[0];
            int end_3 = valid_tiles_3 * num_m_tiles;
            const int features_i = num_m_tiles * 64;
            int num_k_1 = K / 256;
            unsigned int tile_cur_2 = 0;
            unsigned int raw_cur = 0;
            int record_1[2];
            unsigned int _phase_tile_full_1 = 0;
            unsigned int _phase_raw_empty = 1;
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                #pragma unroll 1
                for (unsigned int w_2 = bid; w_2 < end_3; w_2 += num_bids) {
                    uint32_t _fast_div_q_1 = (m_tile_div.divisor != 1) ? (__umulhi((uint32_t)(w_2), m_tile_div.multiplier) >> m_tile_div.shift_right) : (uint32_t)(w_2);
                    uint32_t _fast_div_r_1 = (uint32_t)(w_2) - _fast_div_q_1 * (uint32_t)(m_tile_div.divisor);
                    int t_2 = (int)_fast_div_q_1;
                    int m_cta_2 = (int)_fast_div_r_1;
                    mbarrier_wait(tile_full_addr + (tile_cur_2) * 8, _phase_tile_full_1);
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&record_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&record_1[(0) + 1]))
                        : "r"(tile_info_addr + tile_cur_2 * 16));
                    int expert_2 = record_1[0];
                    mbarrier_arrive(tile_empty_addr + (tile_cur_2) * 8);
                    tile_cur_2 += 1;
                    if (tile_cur_2 == 3) { tile_cur_2 = 0; _phase_tile_full_1 ^= 1; }
                    int gate_row = m_cta_2 * 64;
                    int up_row = features_i + m_cta_2 * 64;
                    #pragma unroll 1
                    for (int kt_1 = 0; kt_1 < num_k_1; kt_1++) {
                        mbarrier_wait(raw_empty_addr + (raw_cur) * 8, _phase_raw_empty);
                        unsigned int raw_base = raw_smem_addr + raw_cur * 16384;
                        unsigned int sf_base = sf_smem_addr + raw_cur * 2048;
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(raw_base), "l"((&W)), "r"(kt_1 * 128), "r"(gate_row), "r"(expert_2),
                               "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(raw_base + 8192), "l"((&W)), "r"(kt_1 * 128), "r"(up_row), "r"(expert_2),
                               "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(sf_base), "l"((&S)), "r"(kt_1 * 16), "r"(gate_row), "r"(expert_2),
                               "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(sf_base + 1024), "l"((&S)), "r"(kt_1 * 16), "r"(up_row), "r"(expert_2),
                               "r"(raw_full_addr + (raw_cur) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(raw_full_addr + (raw_cur) * 8, 18432);
                        raw_cur += 1;
                        if (raw_cur == 10) { raw_cur = 0; _phase_raw_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: token_tma ----
    if (warp == 7) {
        { // token_tma_main
            int valid_tiles_4 = num_non_exiting_tiles[0];
            int end_4 = valid_tiles_4 * num_m_tiles;
            int num_k_2 = K / 256;
            unsigned int b_cur_1 = 0;
            int tokens[8];
            unsigned int _phase_b_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int w_3 = bid; w_3 < end_4; w_3 += num_bids) {
                    uint32_t _fast_div_q_2 = (m_tile_div.divisor != 1) ? (__umulhi((uint32_t)(w_3), m_tile_div.multiplier) >> m_tile_div.shift_right) : (uint32_t)(w_3);
                    uint32_t _fast_div_r_2 = (uint32_t)(w_3) - _fast_div_q_2 * (uint32_t)(m_tile_div.divisor);
                    int t_3 = (int)_fast_div_q_2;
                    int m_cta_3 = (int)_fast_div_r_2;
                    int row0_1 = t_3 * 8;
                    int valid_rows_1 = tile_idx_to_mn_limit[t_3] - row0_1;
                    unsigned int expanded = (unsigned int)permuted_idx_to_expanded_idx[row0_1];
                    uint32_t _fast_div_q_3 = (top_k_div.divisor != 1) ? (__umulhi((uint32_t)(expanded), top_k_div.multiplier) >> top_k_div.shift_right) : (uint32_t)(expanded);
                    uint32_t _fast_div_r_3 = (uint32_t)(expanded) - _fast_div_q_3 * (uint32_t)(top_k_div.divisor);
                    tokens[0] = ((valid_rows_1 > 0) ? (int)_fast_div_q_3 : 0);
                    unsigned int expanded_0 = (unsigned int)permuted_idx_to_expanded_idx[row0_1 + 1];
                    uint32_t _fast_div_q_4 = (top_k_div.divisor != 1) ? (__umulhi((uint32_t)(expanded_0), top_k_div.multiplier) >> top_k_div.shift_right) : (uint32_t)(expanded_0);
                    uint32_t _fast_div_r_4 = (uint32_t)(expanded_0) - _fast_div_q_4 * (uint32_t)(top_k_div.divisor);
                    tokens[1] = ((valid_rows_1 > 1) ? (int)_fast_div_q_4 : 0);
                    unsigned int expanded_1 = (unsigned int)permuted_idx_to_expanded_idx[row0_1 + 2];
                    uint32_t _fast_div_q_5 = (top_k_div.divisor != 1) ? (__umulhi((uint32_t)(expanded_1), top_k_div.multiplier) >> top_k_div.shift_right) : (uint32_t)(expanded_1);
                    uint32_t _fast_div_r_5 = (uint32_t)(expanded_1) - _fast_div_q_5 * (uint32_t)(top_k_div.divisor);
                    tokens[2] = ((valid_rows_1 > 2) ? (int)_fast_div_q_5 : 0);
                    unsigned int expanded_2 = (unsigned int)permuted_idx_to_expanded_idx[row0_1 + 3];
                    uint32_t _fast_div_q_6 = (top_k_div.divisor != 1) ? (__umulhi((uint32_t)(expanded_2), top_k_div.multiplier) >> top_k_div.shift_right) : (uint32_t)(expanded_2);
                    uint32_t _fast_div_r_6 = (uint32_t)(expanded_2) - _fast_div_q_6 * (uint32_t)(top_k_div.divisor);
                    tokens[3] = ((valid_rows_1 > 3) ? (int)_fast_div_q_6 : 0);
                    unsigned int expanded_3 = (unsigned int)permuted_idx_to_expanded_idx[row0_1 + 4];
                    uint32_t _fast_div_q_7 = (top_k_div.divisor != 1) ? (__umulhi((uint32_t)(expanded_3), top_k_div.multiplier) >> top_k_div.shift_right) : (uint32_t)(expanded_3);
                    uint32_t _fast_div_r_7 = (uint32_t)(expanded_3) - _fast_div_q_7 * (uint32_t)(top_k_div.divisor);
                    tokens[4] = ((valid_rows_1 > 4) ? (int)_fast_div_q_7 : 0);
                    unsigned int expanded_4 = (unsigned int)permuted_idx_to_expanded_idx[row0_1 + 5];
                    uint32_t _fast_div_q_8 = (top_k_div.divisor != 1) ? (__umulhi((uint32_t)(expanded_4), top_k_div.multiplier) >> top_k_div.shift_right) : (uint32_t)(expanded_4);
                    uint32_t _fast_div_r_8 = (uint32_t)(expanded_4) - _fast_div_q_8 * (uint32_t)(top_k_div.divisor);
                    tokens[5] = ((valid_rows_1 > 5) ? (int)_fast_div_q_8 : 0);
                    unsigned int expanded_5 = (unsigned int)permuted_idx_to_expanded_idx[row0_1 + 6];
                    uint32_t _fast_div_q_9 = (top_k_div.divisor != 1) ? (__umulhi((uint32_t)(expanded_5), top_k_div.multiplier) >> top_k_div.shift_right) : (uint32_t)(expanded_5);
                    uint32_t _fast_div_r_9 = (uint32_t)(expanded_5) - _fast_div_q_9 * (uint32_t)(top_k_div.divisor);
                    tokens[6] = ((valid_rows_1 > 6) ? (int)_fast_div_q_9 : 0);
                    unsigned int expanded_6 = (unsigned int)permuted_idx_to_expanded_idx[row0_1 + 7];
                    uint32_t _fast_div_q_10 = (top_k_div.divisor != 1) ? (__umulhi((uint32_t)(expanded_6), top_k_div.multiplier) >> top_k_div.shift_right) : (uint32_t)(expanded_6);
                    uint32_t _fast_div_r_10 = (uint32_t)(expanded_6) - _fast_div_q_10 * (uint32_t)(top_k_div.divisor);
                    tokens[7] = ((valid_rows_1 > 7) ? (int)_fast_div_q_10 : 0);
                    #pragma unroll 1
                    for (int kt_2 = 0; kt_2 < num_k_2; kt_2++) {
                        mbarrier_wait(b_empty_addr + (b_cur_1) * 8, _phase_b_empty);
                        unsigned int stage_base = smem_b_addr + b_cur_1 * 4096;
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(stage_base), "l"((&X)), "r"(kt_2 * 256), "r"(tokens[0]), "r"(tokens[1]), "r"(tokens[2]), "r"(tokens[3]), "r"(b_full_addr + (b_cur_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(stage_base + 512), "l"((&X)), "r"(kt_2 * 256), "r"(tokens[4]), "r"(tokens[5]), "r"(tokens[6]), "r"(tokens[7]), "r"(b_full_addr + (b_cur_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(stage_base + 1024), "l"((&X)), "r"(kt_2 * 256 + 64), "r"(tokens[0]), "r"(tokens[1]), "r"(tokens[2]), "r"(tokens[3]), "r"(b_full_addr + (b_cur_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(stage_base + 1024 + 512), "l"((&X)), "r"(kt_2 * 256 + 64), "r"(tokens[4]), "r"(tokens[5]), "r"(tokens[6]), "r"(tokens[7]), "r"(b_full_addr + (b_cur_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(stage_base + 2048), "l"((&X)), "r"(kt_2 * 256 + 128), "r"(tokens[0]), "r"(tokens[1]), "r"(tokens[2]), "r"(tokens[3]), "r"(b_full_addr + (b_cur_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(stage_base + 2048 + 512), "l"((&X)), "r"(kt_2 * 256 + 128), "r"(tokens[4]), "r"(tokens[5]), "r"(tokens[6]), "r"(tokens[7]), "r"(b_full_addr + (b_cur_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(stage_base + 3072), "l"((&X)), "r"(kt_2 * 256 + 192), "r"(tokens[0]), "r"(tokens[1]), "r"(tokens[2]), "r"(tokens[3]), "r"(b_full_addr + (b_cur_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(stage_base + 3072 + 512), "l"((&X)), "r"(kt_2 * 256 + 192), "r"(tokens[4]), "r"(tokens[5]), "r"(tokens[6]), "r"(tokens[7]), "r"(b_full_addr + (b_cur_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        mbarrier_arrive_expect_tx(b_full_addr + (b_cur_1) * 8, 4096);
                        b_cur_1 += 1;
                        if (b_cur_1 == 10) { b_cur_1 = 0; _phase_b_empty ^= 1; }
                    }
                }
                #pragma unroll 1
                for (int _ = 0; _ < 10; _++) {
                    mbarrier_wait(b_empty_addr + (b_cur_1) * 8, _phase_b_empty);
                    b_cur_1 += 1;
                    if (b_cur_1 == 10) { b_cur_1 = 0; _phase_b_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: transform ----
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // transform_main
            int valid_tiles_5 = num_non_exiting_tiles[0];
            int end_5 = valid_tiles_5 * num_m_tiles;
            int num_k_3 = K / 256;
            const int quarter_1 = warp % 4;
            const int half = warp / 4 - 2;
            const int src_row = ((lane < 16) ? quarter_1 * 16 + lane : 48 + quarter_1 * 16 + lane);
            const unsigned int lane_base = (unsigned int)(quarter_1 * 32) << 16;
            unsigned int raw_cur_1 = 0;
            unsigned int a_cur_1 = 0;
            unsigned int raw_words[16];
            unsigned int scale_words[2];
            unsigned int _phase_raw_full = 0;
            unsigned int _phase_a_empty = 1;
            #pragma unroll 1
            for (unsigned int w_4 = bid; w_4 < end_5; w_4 += num_bids) {
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
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                    if (warp == 8) {
                        if (elect_sync()) {
                            mbarrier_arrive(a_full_addr + (a_cur_1) * 8);
                            mbarrier_arrive(raw_empty_addr + (raw_cur_1) * 8);
                        }
                    }
                    raw_cur_1 += 1;
                    if (raw_cur_1 == 10) { raw_cur_1 = 0; _phase_raw_full ^= 1; }
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
