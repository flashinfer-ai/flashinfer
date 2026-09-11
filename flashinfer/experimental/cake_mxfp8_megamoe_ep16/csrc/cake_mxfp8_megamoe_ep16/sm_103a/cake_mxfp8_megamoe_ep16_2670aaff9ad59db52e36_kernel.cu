/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TRANSFORMED_A_OFFSET 256
#define NUM_A_PIPE_STAGES 4
#define NUM_SCALE_PIPE_STAGES 4
#define NUM_B_PIPE_STAGES 2
#define NUM_TRANSFORMED_PIPE_STAGES 4
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_DISPATCH_ROWS_I32_OFF 1024
#define SMEM_DISPATCH_ROWS_I32_STAGE_BYTES 36
#define SMEM_DISPATCH_ROWS_I32_STRIDE 36
#define SMEM_RAW_SMEM_OFF 1024
#define SMEM_RAW_SMEM_STAGE_BYTES 16384
#define SMEM_RAW_SMEM_STRIDE 16384
#define SMEM_SCALE_SMEM_OFF 66560
#define SMEM_SCALE_SMEM_STAGE_BYTES 512
#define SMEM_SCALE_SMEM_STRIDE 512
#define SMEM_B_SMEM_OFF 68608
#define SMEM_B_SMEM_STAGE_BYTES 4096
#define SMEM_B_SMEM_STRIDE 4096
#define SMEM_B_SMEM_N16_OFF 68608
#define SMEM_B_SMEM_N16_STAGE_BYTES 2048
#define SMEM_B_SMEM_N16_STRIDE 4096
#define SMEM_TOTAL 76800
#define THREADS 384

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


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}


__device__ __forceinline__ void tcgen05_mma_f16_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ts_step_cg2(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], db, %4, "
        "{m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
}


__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
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


__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
}


__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}


__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
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


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(384, 1) __cluster_dims__(2,1,1) void
kernel_cake_mxfp8_megamoe_ep16_2670aaff9ad59db52e36(__nv_bfloat16* __restrict__ source_hidden_bf16, long long* __restrict__ source_topk_ids_i64, float* __restrict__ source_topk_weights_f32, CakeTensorMap const* fc1_weight_e4m3, CakeTensorMap const* fc1_blocked_e8m0, CakeTensorMap const* activation_bf16, CakeTensorMap const* activation_bf16_n16, __nv_bfloat16* __restrict__ activation_bf16_ptr, CakeTensorMap const* fc2_weight_e4m3, CakeTensorMap const* fc2_blocked_e8m0, CakeTensorMap const* fc1_workspace_bf16_tma, CakeTensorMap const* fc1_workspace_bf16_tma_n16, __nv_bfloat16* __restrict__ fc1_workspace_bf16, __nv_bfloat16* __restrict__ fc2_output_bf16, int* __restrict__ route_map_i32, float* __restrict__ route_scale_f32, unsigned int* __restrict__ route_counts_u32, unsigned int* __restrict__ fc1_done, unsigned int* __restrict__ publication_done, unsigned int* __restrict__ publication_visible, unsigned int* __restrict__ dispatch_done, unsigned int* __restrict__ compute_done, unsigned int* __restrict__ return_done, unsigned int* __restrict__ return_visible, int launch_epoch, int tokens_per_rank, int32_t mega_pg_world, int32_t mega_pg_rank, unsigned* const* __restrict__ mega_pg_flags, __nv_bfloat16* __restrict__ published_hidden_bf16, __nv_bfloat16* const* __restrict__ published_hidden_bf16_peers, int* __restrict__ published_topk_ids_i32, int* const* __restrict__ published_topk_ids_i32_peers, float* __restrict__ published_topk_weights_f32, float* const* __restrict__ published_topk_weights_f32_peers, __nv_bfloat16* __restrict__ route_terms_bf16, __nv_bfloat16* const* __restrict__ route_terms_bf16_peers)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define a_full_addr (mbar_base + 0)
    #define a_free_addr (mbar_base + 32)
    #define scale_full_addr (mbar_base + 64)
    #define scale_free_addr (mbar_base + 96)
    #define b_full_addr (mbar_base + 128)
    #define b_free_addr (mbar_base + 144)
    #define transformed_full_addr (mbar_base + 160)
    #define transformed_free_addr (mbar_base + 192)
    #define acc_full_addr (mbar_base + 224)
    #define acc_free_addr (mbar_base + 240)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(fc1_weight_e4m3)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(fc1_blocked_e8m0)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(activation_bf16)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(activation_bf16_n16)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(fc2_weight_e4m3)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(fc2_blocked_e8m0)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(fc1_workspace_bf16_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(fc1_workspace_bf16_tma_n16)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    int* dispatch_rows_i32 = reinterpret_cast<int*>(smem_raw + 1024);
    const int dispatch_rows_i32_addr = smem + 1024;
    uint8_t* raw_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int raw_smem_addr = smem + 1024;
    uint8_t* scale_smem = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int scale_smem_addr = smem + 66560;
    __nv_bfloat16* b_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int b_smem_addr = smem + 68608;
    __nv_bfloat16* b_smem_n16 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int b_smem_n16_addr = smem + 68608;

    // Mbarrier init (10 groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'a_pipe' ---
            // a_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // a_free: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'scale_pipe' ---
            // scale_full: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // scale_free: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // --- pipeline 'b_pipe' ---
            // b_full: 2 barriers, init_count=2
            mbarrier_init(smem + 128, 2);
            mbarrier_init(smem + 136, 2);
            // b_free: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'transformed_pipe' ---
            // transformed_full: 4 barriers, init_count=2
            mbarrier_init(smem + 160, 2);
            mbarrier_init(smem + 168, 2);
            mbarrier_init(smem + 176, 2);
            mbarrier_init(smem + 184, 2);
            // transformed_free: 4 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // acc_free: 2 barriers, init_count=8
            mbarrier_init(smem + 240, 8);
            mbarrier_init(smem + 248, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
    if (warp == 0) {
        int _tmem_hold = smem + 256;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_transformed_a = taddr + 256;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // epilogue_main
            int linear_thread = bid * 384 + tid;
            int thread_stride = num_bids * 384;
            #pragma unroll 1
            for (int index = linear_thread; index < 2048; index += thread_stride) {
                route_map_i32[index] = -1;
                route_scale_f32[index] = 0.0f;
            }
            #pragma unroll 1
            for (int expert = linear_thread; expert < 32; expert += thread_stride) {
                route_counts_u32[expert] = 0;
            }
            #pragma unroll 1
            for (int expert_row_tile = linear_thread; expert_row_tile < 96; expert_row_tile += thread_stride) {
                {
                    unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(fc1_done) + (expert_row_tile);
                    asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
                }
            }
            #pragma unroll 1
            for (int index_1 = linear_thread * 8; index_1 < tokens_per_rank * 3072; index_1 += thread_stride * 8) {
                float _vec_load_10[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(source_hidden_bf16 + index_1 + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(_vec_load_10[0 + 0], _vec_load_10[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(_vec_load_10[0 + 2], _vec_load_10[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(_vec_load_10[0 + 4], _vec_load_10[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(_vec_load_10[0 + 6], _vec_load_10[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(published_hidden_bf16) + index_1))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
            #pragma unroll 1
            for (int route = linear_thread; route < tokens_per_rank * 8; route += thread_stride) {
                int route_id = (int)source_topk_ids_i64[route];
                float route_weight = source_topk_weights_f32[route];
                *(reinterpret_cast<int*>(reinterpret_cast<int*>(published_topk_ids_i32) + route) + (0)) = route_id;
                *(reinterpret_cast<float*>(reinterpret_cast<float*>(published_topk_weights_f32) + route) + (0)) = route_weight;
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(publication_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(publication_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            if (bid == 0 && warp == 0) {
                if (elect_sync()) {
                    // nvlink_barrier(mega_pg_flags) phase=0
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __slot = 0;
                        unsigned* __local_flag = mega_pg_flags[__me] + __slot;
                        unsigned __old_sense = 0u;
                        const unsigned __delta = (__me == 0) ? (0x80000000u - (unsigned)(__ws - 1)) : 1u;
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        for (int __r = 0; __r < __ws; ++__r) {
                            unsigned* __peer_flag = mega_pg_flags[__r] + __slot;
                            unsigned __old_peer;
                            asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                                : "=r"(__old_peer) : "l"(__peer_flag), "r"(__delta) : "memory");
                            if (__r == __me) __old_sense = __old_peer;
                        }
                        asm volatile("fence.proxy.alias;" ::: "memory");
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                            if (((__old_sense ^ __v) & 0x80000000u) != 0u) break;
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(publication_visible) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(publication_visible) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(launch_epoch + 1)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            int source_rank = bid % 16;
            int source_token_begin = bid / 16;
            int* remote_ids = reinterpret_cast<int*>(published_topk_ids_i32_peers[source_rank]);
            float* remote_weights = reinterpret_cast<float*>(published_topk_weights_f32_peers[source_rank]);
            __nv_bfloat16* remote_hidden = reinterpret_cast<__nv_bfloat16*>(published_hidden_bf16_peers[source_rank]);
            #pragma unroll 1
            for (int source_token = source_token_begin; source_token < tokens_per_rank; source_token += 9) {
                int routes_per_source = tokens_per_rank * 8;
                int source_route_lane = source_token * 8 + lane;
                int row_index_lane = -1;
                int row_lane = 0;
                if (warp == 0 && lane < 8) {
                    int global_expert = remote_ids[source_route_lane];
                    int owner = global_expert / 32;
                    if (owner == mega_pg_rank) {
                        int local_expert_lane = global_expert - mega_pg_rank * 32;
                        unsigned int _atomic_old_5 = atomicAdd(&route_counts_u32[local_expert_lane], 1);
                        row_lane = (int)_atomic_old_5;
                        if (row_lane < 64) {
                            row_index_lane = local_expert_lane * 64 + row_lane;
                            int global_route_lane = source_rank * routes_per_source + source_route_lane;
                            route_map_i32[row_index_lane] = global_route_lane;
                            route_scale_f32[row_index_lane] = remote_weights[source_route_lane];
                        }
                    }
                    dispatch_rows_i32[lane] = row_index_lane;
                }
                if (warp == 0) {
                    int _vote_5 = __any_sync(0xFFFFFFFF, row_index_lane >= 0);
                    int any_owned = _vote_5;
                    if (lane == 0) {
                        dispatch_rows_i32[8] = any_owned;
                    }
                }
                asm volatile("barrier.sync 11, 384;" ::: "memory");
                int dispatch_any = dispatch_rows_i32[8];
                if (dispatch_any != 0) {
                    int column = tid * 8;
                    float _vec_load_11[8];
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(remote_hidden + (source_token * 3072 + column) + 0);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int route_slot = 0; route_slot < 8; route_slot++) {
                        int row_index = dispatch_rows_i32[route_slot];
                        if (row_index >= 0) {
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(_vec_load_11[0 + 0], _vec_load_11[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_vec_load_11[0 + 2], _vec_load_11[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_vec_load_11[0 + 4], _vec_load_11[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_vec_load_11[0 + 6], _vec_load_11[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(activation_bf16_ptr + (row_index * 3072 + column)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                }
                asm volatile("barrier.sync 11, 384;" ::: "memory");
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            asm volatile("fence.release.gpu;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            unsigned int acc_stage = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll
            for (int phase = 0; phase < 2; phase++) {
                int work_begin = ((phase == 0) ? (int)cluster_id : ((int)cluster_id + 16) % 72);
                int work_end = ((phase == 0) ? 1280 : 384);
                int m_tiles_per_expert = ((phase == 0) ? 40 : 12);
                #pragma unroll 1
                for (int work = work_begin; work < work_end; work += (int)num_clusters) {
                    int expert_1 = work / m_tiles_per_expert;
                    int weight_m_tile = work - expert_1 * m_tiles_per_expert;
                    int route_count = (int)route_counts_u32[expert_1];
                    mbarrier_wait(acc_full_addr + (acc_stage) * 8, _phase_acc_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int row_tile = 0; row_tile < 1; row_tile++) {
                        int packed_token = row_tile * 32 + lane;
                        int expert_row_tile_1 = row_tile * 32 + expert_1;
                        int acc_offset = (acc_stage * 3 + (unsigned int)row_tile) * 32;
                        if (phase == 0) {
                            int warp_row_ptr = taddr + (unsigned int)acc_offset + (unsigned int)(warp * 32 << 16);
                            int feature_base = weight_m_tile * 128 + cta_rank * 64 + warp * 16;
                            int half_ptr = warp_row_ptr;
                            float _tmem_load_0[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x64b.x16.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                                : "r"(half_ptr));
                            float _tmem_load_1[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x64b.x16.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                                : "r"(half_ptr + 1048576));
                            float folded[16];
                            #pragma unroll
                            for (int pair_idx = 0; pair_idx < 8; pair_idx++) {
                                int elem = pair_idx * 2;
                                float2 _f2_0 = make_float2(_tmem_load_1[elem], _tmem_load_1[elem + 1]);
                                float2 up_pair = _f2_0;
                                float2 _f2_1 = make_float2(_tmem_load_0[elem], _tmem_load_0[elem + 1]);
                                float2 gate_pair = _f2_1;
                                float2 _mul_f32x2_0;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&up_pair), "l"(*(const unsigned long long*)&gate_pair));
                                float2 up_gate = _mul_f32x2_0;
                                float2 _f2_2 = make_float2(-1.4426950408889634f, -1.4426950408889634f);
                                float2 neg_log2e_pair = _f2_2;
                                float2 _mul_f32x2_1;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&gate_pair), "l"(*(const unsigned long long*)&neg_log2e_pair));
                                float2 neg_gate_log2e = _mul_f32x2_1;
                                float _exp2_0 = approx_exp2(neg_gate_log2e.x);
                                float _exp2_1 = approx_exp2(neg_gate_log2e.y);
                                float2 _f2_3 = make_float2(_exp2_0, _exp2_1);
                                float2 exp_pair = _f2_3;
                                float2 _f2_4 = make_float2(1.0f, 1.0f);
                                float2 one_pair = _f2_4;
                                float2 one_plus_exp = add_f32x2(exp_pair, one_pair);
                                float _rcp_0 = approx_rcp(one_plus_exp.x);
                                float _rcp_1 = approx_rcp(one_plus_exp.y);
                                float2 _f2_5 = make_float2(_rcp_0, _rcp_1);
                                float2 reciprocal_pair = _f2_5;
                                float2 _mul_f32x2_2;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&up_gate), "l"(*(const unsigned long long*)&reciprocal_pair));
                                float2 result_pair = _mul_f32x2_2;
                                folded[elem] = result_pair.x;
                                folded[elem + 1] = result_pair.y;
                            }
                            float transposed[16];
                            float r1[16];
                            r1[0] = folded[0];
                            r1[1] = folded[8];
                            r1[2] = folded[2];
                            r1[3] = folded[10];
                            r1[4] = folded[4];
                            r1[5] = folded[12];
                            r1[6] = folded[6];
                            r1[7] = folded[14];
                            r1[8] = folded[1];
                            r1[9] = folded[9];
                            r1[10] = folded[3];
                            r1[11] = folded[11];
                            r1[12] = folded[5];
                            r1[13] = folded[13];
                            r1[14] = folded[7];
                            r1[15] = folded[15];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(half_ptr), "r"(*reinterpret_cast<const uint32_t*>(&r1[0])), "r"(*reinterpret_cast<const uint32_t*>(&r1[1])), "r"(*reinterpret_cast<const uint32_t*>(&r1[2])), "r"(*reinterpret_cast<const uint32_t*>(&r1[3])), "r"(*reinterpret_cast<const uint32_t*>(&r1[4])), "r"(*reinterpret_cast<const uint32_t*>(&r1[5])), "r"(*reinterpret_cast<const uint32_t*>(&r1[6])), "r"(*reinterpret_cast<const uint32_t*>(&r1[7])), "r"(*reinterpret_cast<const uint32_t*>(&r1[8])), "r"(*reinterpret_cast<const uint32_t*>(&r1[9])), "r"(*reinterpret_cast<const uint32_t*>(&r1[10])), "r"(*reinterpret_cast<const uint32_t*>(&r1[11])), "r"(*reinterpret_cast<const uint32_t*>(&r1[12])), "r"(*reinterpret_cast<const uint32_t*>(&r1[13])), "r"(*reinterpret_cast<const uint32_t*>(&r1[14])), "r"(*reinterpret_cast<const uint32_t*>(&r1[15])));
                            float _tmem_load_2[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x64b.x16.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                                : "r"(half_ptr));
                            tmem_st_x16_f32(half_ptr, _tmem_load_2);
                            float r3_raw[16];
                            float _tmem_load_3[8];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7]))
                                : "r"(half_ptr));
                            float _tmem_load_4[8];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7]))
                                : "r"(half_ptr + 1048576));
                            #pragma unroll
                            for (int elem_1 = 0; elem_1 < 8; elem_1++) {
                                r3_raw[elem_1] = _tmem_load_3[elem_1];
                                r3_raw[elem_1 + 8] = _tmem_load_4[elem_1];
                            }
                            float r3[16];
                            r3[0] = r3_raw[0];
                            r3[1] = r3_raw[1];
                            r3[2] = r3_raw[4];
                            r3[3] = r3_raw[5];
                            r3[4] = r3_raw[2];
                            r3[5] = r3_raw[3];
                            r3[6] = r3_raw[6];
                            r3[7] = r3_raw[7];
                            r3[8] = r3_raw[8];
                            r3[9] = r3_raw[9];
                            r3[10] = r3_raw[12];
                            r3[11] = r3_raw[13];
                            r3[12] = r3_raw[10];
                            r3[13] = r3_raw[11];
                            r3[14] = r3_raw[14];
                            r3[15] = r3_raw[15];
                            tmem_st_x16_f32(half_ptr, r3);
                            float r4_raw[16];
                            float _tmem_load_5[8];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x128b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7]))
                                : "r"(half_ptr));
                            float _tmem_load_6[8];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x128b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7]))
                                : "r"(half_ptr + 1048576));
                            #pragma unroll
                            for (int elem_2 = 0; elem_2 < 8; elem_2++) {
                                r4_raw[elem_2] = _tmem_load_5[elem_2];
                                r4_raw[elem_2 + 8] = _tmem_load_6[elem_2];
                            }
                            transposed[0] = r4_raw[0];
                            transposed[1] = r4_raw[8];
                            transposed[2] = r4_raw[2];
                            transposed[3] = r4_raw[10];
                            transposed[4] = r4_raw[4];
                            transposed[5] = r4_raw[12];
                            transposed[6] = r4_raw[6];
                            transposed[7] = r4_raw[14];
                            transposed[8] = r4_raw[1];
                            transposed[9] = r4_raw[9];
                            transposed[10] = r4_raw[3];
                            transposed[11] = r4_raw[11];
                            transposed[12] = r4_raw[5];
                            transposed[13] = r4_raw[13];
                            transposed[14] = r4_raw[7];
                            transposed[15] = r4_raw[15];
                            float route_scale = route_scale_f32[expert_1 * 64 + packed_token];
                            int valid_route = route_map_i32[expert_1 * 64 + packed_token];
                            float scaled[16];
                            if (valid_route >= 0) {
                                #pragma unroll
                                for (int pair_idx_1 = 0; pair_idx_1 < 8; pair_idx_1++) {
                                    int elem_3 = pair_idx_1 * 2;
                                    float2 _f2_6 = make_float2(transposed[elem_3], transposed[elem_3 + 1]);
                                    float2 value_pair = _f2_6;
                                    float2 _f2_7 = make_float2(route_scale, route_scale);
                                    float2 scale_pair = _f2_7;
                                    float2 _mul_f32x2_3;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&value_pair), "l"(*(const unsigned long long*)&scale_pair));
                                    float2 scaled_pair = _mul_f32x2_3;
                                    scaled[elem_3] = scaled_pair.x;
                                    scaled[elem_3 + 1] = scaled_pair.y;
                                }
                            } else {
                                #pragma unroll
                                for (int element = 0; element < 16; element++) {
                                    scaled[element] = 0.0f;
                                }
                            }
                            {
                                {
                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(scaled[0 + 0], scaled[0 + 1]);
                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(scaled[0 + 2], scaled[0 + 3]);
                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(scaled[0 + 4], scaled[0 + 5]);
                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(scaled[0 + 6], scaled[0 + 7]);
                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(scaled[0 + 8], scaled[0 + 9]);
                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(scaled[0 + 10], scaled[0 + 11]);
                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(scaled[0 + 12], scaled[0 + 13]);
                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(scaled[0 + 14], scaled[0 + 15]);
                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(&((__nv_bfloat16*)(fc1_workspace_bf16 + ((expert_1 * 64 + packed_token) * 5120 + feature_base)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                }
                            }
                        } else {
                            int warp_row_ptr_1 = taddr + (unsigned int)acc_offset + (unsigned int)(warp * 32 << 16);
                            int hidden_base = weight_m_tile * 256 + cta_rank * 128 + warp * 32;
                            unsigned int hidden_lo[8];
                            unsigned int hidden_hi[8];
                            float _tmem_load_7[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x64b.x16.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                                : "r"(warp_row_ptr_1));
                            float _tmem_load_8[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x64b.x16.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                                : "r"(warp_row_ptr_1 + 1048576));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float casted_input[32];
                            #pragma unroll
                            for (int elem_4 = 0; elem_4 < 16; elem_4++) {
                                casted_input[elem_4] = _tmem_load_7[elem_4];
                                casted_input[16 + elem_4] = _tmem_load_8[elem_4];
                            }
                            unsigned int casted_bf16[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(casted_input[_lp*2 + 0], casted_input[_lp*2+1 + 0]));
                                casted_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            float packed[16];
                            #pragma unroll
                            for (int pair = 0; pair < 8; pair++) {
                                uint32_t _prmt_b32_4;
                                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_4) : "r"(casted_bf16[pair]), "r"(casted_bf16[8 + pair]));
                                packed[pair * 2] = __uint_as_float(_prmt_b32_4);
                                uint32_t _prmt_b32_5;
                                asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_b32_5) : "r"(casted_bf16[pair]), "r"(casted_bf16[8 + pair]));
                                packed[pair * 2 + 1] = __uint_as_float(_prmt_b32_5);
                            }
                            float transposed_1[16];
                            float r1_1[16];
                            r1_1[0] = packed[0];
                            r1_1[1] = packed[8];
                            r1_1[2] = packed[2];
                            r1_1[3] = packed[10];
                            r1_1[4] = packed[4];
                            r1_1[5] = packed[12];
                            r1_1[6] = packed[6];
                            r1_1[7] = packed[14];
                            r1_1[8] = packed[1];
                            r1_1[9] = packed[9];
                            r1_1[10] = packed[3];
                            r1_1[11] = packed[11];
                            r1_1[12] = packed[5];
                            r1_1[13] = packed[13];
                            r1_1[14] = packed[7];
                            r1_1[15] = packed[15];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(warp_row_ptr_1), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&r1_1[15])));
                            float _tmem_load_9[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x64b.x16.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15]))
                                : "r"(warp_row_ptr_1));
                            tmem_st_x16_f32(warp_row_ptr_1, _tmem_load_9);
                            float r3_raw_1[16];
                            float _tmem_load_10[8];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7]))
                                : "r"(warp_row_ptr_1));
                            float _tmem_load_11[8];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7]))
                                : "r"(warp_row_ptr_1 + 1048576));
                            #pragma unroll
                            for (int elem_5 = 0; elem_5 < 8; elem_5++) {
                                r3_raw_1[elem_5] = _tmem_load_10[elem_5];
                                r3_raw_1[elem_5 + 8] = _tmem_load_11[elem_5];
                            }
                            float r3_1[16];
                            r3_1[0] = r3_raw_1[0];
                            r3_1[1] = r3_raw_1[1];
                            r3_1[2] = r3_raw_1[4];
                            r3_1[3] = r3_raw_1[5];
                            r3_1[4] = r3_raw_1[2];
                            r3_1[5] = r3_raw_1[3];
                            r3_1[6] = r3_raw_1[6];
                            r3_1[7] = r3_raw_1[7];
                            r3_1[8] = r3_raw_1[8];
                            r3_1[9] = r3_raw_1[9];
                            r3_1[10] = r3_raw_1[12];
                            r3_1[11] = r3_raw_1[13];
                            r3_1[12] = r3_raw_1[10];
                            r3_1[13] = r3_raw_1[11];
                            r3_1[14] = r3_raw_1[14];
                            r3_1[15] = r3_raw_1[15];
                            tmem_st_x16_f32(warp_row_ptr_1, r3_1);
                            float r4_raw_1[16];
                            float _tmem_load_12[8];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x128b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7]))
                                : "r"(warp_row_ptr_1));
                            float _tmem_load_13[8];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x128b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7]))
                                : "r"(warp_row_ptr_1 + 1048576));
                            #pragma unroll
                            for (int elem_6 = 0; elem_6 < 8; elem_6++) {
                                r4_raw_1[elem_6] = _tmem_load_12[elem_6];
                                r4_raw_1[elem_6 + 8] = _tmem_load_13[elem_6];
                            }
                            transposed_1[0] = r4_raw_1[0];
                            transposed_1[1] = r4_raw_1[8];
                            transposed_1[2] = r4_raw_1[2];
                            transposed_1[3] = r4_raw_1[10];
                            transposed_1[4] = r4_raw_1[4];
                            transposed_1[5] = r4_raw_1[12];
                            transposed_1[6] = r4_raw_1[6];
                            transposed_1[7] = r4_raw_1[14];
                            transposed_1[8] = r4_raw_1[1];
                            transposed_1[9] = r4_raw_1[9];
                            transposed_1[10] = r4_raw_1[3];
                            transposed_1[11] = r4_raw_1[11];
                            transposed_1[12] = r4_raw_1[5];
                            transposed_1[13] = r4_raw_1[13];
                            transposed_1[14] = r4_raw_1[7];
                            transposed_1[15] = r4_raw_1[15];
                            #pragma unroll
                            for (int pair_1 = 0; pair_1 < 8; pair_1++) {
                                float word_a_f32 = transposed_1[pair_1 * 2];
                                float word_b_f32 = transposed_1[pair_1 * 2 + 1];
                                unsigned int word_a = __as_u32(word_a_f32);
                                unsigned int word_b = __as_u32(word_b_f32);
                                uint32_t _prmt_b32_6;
                                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_6) : "r"(word_a), "r"(word_b));
                                hidden_lo[pair_1] = _prmt_b32_6;
                                uint32_t _prmt_b32_7;
                                asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_b32_7) : "r"(word_a), "r"(word_b));
                                hidden_hi[pair_1] = _prmt_b32_7;
                            }
                            int global_route = route_map_i32[expert_1 * 64 + packed_token];
                            if (global_route >= 0) {
                                int routes_per_source_1 = tokens_per_rank * 8;
                                int source_rank_0 = global_route / routes_per_source_1;
                                int source_route = global_route - source_rank_0 * routes_per_source_1;
                                __nv_bfloat16* remote_terms = reinterpret_cast<__nv_bfloat16*>(route_terms_bf16_peers[source_rank_0]);
                                {
                                    const unsigned* _raw_stv8_2 = reinterpret_cast<const unsigned*>(hidden_lo);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(remote_terms + ((unsigned long long)source_route * 3072 + (unsigned long long)hidden_base))), "r"(_raw_stv8_2[0]), "r"(_raw_stv8_2[1]), "r"(_raw_stv8_2[2]), "r"(_raw_stv8_2[3]), "r"(_raw_stv8_2[4]), "r"(_raw_stv8_2[5]), "r"(_raw_stv8_2[6]), "r"(_raw_stv8_2[7]) : "memory");
                                }
                                {
                                    const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(hidden_hi);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(remote_terms + ((unsigned long long)source_route * 3072 + (unsigned long long)hidden_base + 16))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                                }
                            }
                        }
                        asm volatile("barrier.sync 4, 128;" ::: "memory");
                        if (phase == 0) {
                            if (warp == 0) {
                                if (elect_sync()) {
                                    {
                                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(fc1_done) + (expert_row_tile_1);
                                        unsigned int _gc_old;
                                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int tail = 0; tail < 2; tail++) {
                        if (route_count > 32 + tail * 16) {
                            int packed_token_1 = 32 + tail * 16 + lane;
                            int expert_row_tile_2 = (tail + 1) * 32 + expert_1;
                            int acc_offset_1 = (acc_stage * 3 + (unsigned int)tail + 1) * 32;
                            int warp_row_ptr_2 = taddr + (unsigned int)acc_offset_1 + (unsigned int)(warp * 32 << 16);
                            if (phase == 0) {
                                int feature_base_1 = weight_m_tile * 128 + cta_rank * 64 + warp * 16;
                                float _tmem_load_14[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x64b.x8.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7]))
                                    : "r"(warp_row_ptr_2));
                                float _tmem_load_15[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x64b.x8.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7]))
                                    : "r"(warp_row_ptr_2 + 1048576));
                                float folded_1[16];
                                #pragma unroll
                                for (int pair_idx_2 = 0; pair_idx_2 < 4; pair_idx_2++) {
                                    int elem_7 = pair_idx_2 * 2;
                                    float2 _f2_8 = make_float2(_tmem_load_15[elem_7], _tmem_load_15[elem_7 + 1]);
                                    float2 up_pair_1 = _f2_8;
                                    float2 _f2_9 = make_float2(_tmem_load_14[elem_7], _tmem_load_14[elem_7 + 1]);
                                    float2 gate_pair_1 = _f2_9;
                                    float2 _mul_f32x2_4;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&up_pair_1), "l"(*(const unsigned long long*)&gate_pair_1));
                                    float2 up_gate_1 = _mul_f32x2_4;
                                    float2 _f2_10 = make_float2(-1.4426950408889634f, -1.4426950408889634f);
                                    float2 neg_log2e_pair_1 = _f2_10;
                                    float2 _mul_f32x2_5;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&gate_pair_1), "l"(*(const unsigned long long*)&neg_log2e_pair_1));
                                    float2 neg_gate_log2e_1 = _mul_f32x2_5;
                                    float _exp2_2 = approx_exp2(neg_gate_log2e_1.x);
                                    float _exp2_3 = approx_exp2(neg_gate_log2e_1.y);
                                    float2 _f2_11 = make_float2(_exp2_2, _exp2_3);
                                    float2 exp_pair_1 = _f2_11;
                                    float2 _f2_12 = make_float2(1.0f, 1.0f);
                                    float2 one_pair_1 = _f2_12;
                                    float2 one_plus_exp_1 = add_f32x2(exp_pair_1, one_pair_1);
                                    float _rcp_2 = approx_rcp(one_plus_exp_1.x);
                                    float _rcp_3 = approx_rcp(one_plus_exp_1.y);
                                    float2 _f2_13 = make_float2(_rcp_2, _rcp_3);
                                    float2 reciprocal_pair_1 = _f2_13;
                                    float2 _mul_f32x2_6;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&up_gate_1), "l"(*(const unsigned long long*)&reciprocal_pair_1));
                                    float2 result_pair_1 = _mul_f32x2_6;
                                    folded_1[elem_7] = result_pair_1.x;
                                    folded_1[elem_7 + 1] = result_pair_1.y;
                                }
                                #pragma unroll
                                for (int elem_8 = 8; elem_8 < 16; elem_8++) {
                                    folded_1[elem_8] = 0.0f;
                                }
                                float transposed_2[16];
                                float r1_2[16];
                                r1_2[0] = folded_1[0];
                                r1_2[1] = folded_1[8];
                                r1_2[2] = folded_1[2];
                                r1_2[3] = folded_1[10];
                                r1_2[4] = folded_1[4];
                                r1_2[5] = folded_1[12];
                                r1_2[6] = folded_1[6];
                                r1_2[7] = folded_1[14];
                                r1_2[8] = folded_1[1];
                                r1_2[9] = folded_1[9];
                                r1_2[10] = folded_1[3];
                                r1_2[11] = folded_1[11];
                                r1_2[12] = folded_1[5];
                                r1_2[13] = folded_1[13];
                                r1_2[14] = folded_1[7];
                                r1_2[15] = folded_1[15];
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                    :: "r"(warp_row_ptr_2), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&r1_2[15])));
                                float _tmem_load_16[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x64b.x16.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[15]))
                                    : "r"(warp_row_ptr_2));
                                tmem_st_x16_f32(warp_row_ptr_2, _tmem_load_16);
                                float r3_raw_2[16];
                                float _tmem_load_17[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[7]))
                                    : "r"(warp_row_ptr_2));
                                float _tmem_load_18[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[7]))
                                    : "r"(warp_row_ptr_2 + 1048576));
                                #pragma unroll
                                for (int elem_9 = 0; elem_9 < 8; elem_9++) {
                                    r3_raw_2[elem_9] = _tmem_load_17[elem_9];
                                    r3_raw_2[elem_9 + 8] = _tmem_load_18[elem_9];
                                }
                                float r3_2[16];
                                r3_2[0] = r3_raw_2[0];
                                r3_2[1] = r3_raw_2[1];
                                r3_2[2] = r3_raw_2[4];
                                r3_2[3] = r3_raw_2[5];
                                r3_2[4] = r3_raw_2[2];
                                r3_2[5] = r3_raw_2[3];
                                r3_2[6] = r3_raw_2[6];
                                r3_2[7] = r3_raw_2[7];
                                r3_2[8] = r3_raw_2[8];
                                r3_2[9] = r3_raw_2[9];
                                r3_2[10] = r3_raw_2[12];
                                r3_2[11] = r3_raw_2[13];
                                r3_2[12] = r3_raw_2[10];
                                r3_2[13] = r3_raw_2[11];
                                r3_2[14] = r3_raw_2[14];
                                r3_2[15] = r3_raw_2[15];
                                tmem_st_x16_f32(warp_row_ptr_2, r3_2);
                                float r4_raw_2[16];
                                float _tmem_load_19[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x128b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[7]))
                                    : "r"(warp_row_ptr_2));
                                float _tmem_load_20[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x128b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[7]))
                                    : "r"(warp_row_ptr_2 + 1048576));
                                #pragma unroll
                                for (int elem_10 = 0; elem_10 < 8; elem_10++) {
                                    r4_raw_2[elem_10] = _tmem_load_19[elem_10];
                                    r4_raw_2[elem_10 + 8] = _tmem_load_20[elem_10];
                                }
                                transposed_2[0] = r4_raw_2[0];
                                transposed_2[1] = r4_raw_2[8];
                                transposed_2[2] = r4_raw_2[2];
                                transposed_2[3] = r4_raw_2[10];
                                transposed_2[4] = r4_raw_2[4];
                                transposed_2[5] = r4_raw_2[12];
                                transposed_2[6] = r4_raw_2[6];
                                transposed_2[7] = r4_raw_2[14];
                                transposed_2[8] = r4_raw_2[1];
                                transposed_2[9] = r4_raw_2[9];
                                transposed_2[10] = r4_raw_2[3];
                                transposed_2[11] = r4_raw_2[11];
                                transposed_2[12] = r4_raw_2[5];
                                transposed_2[13] = r4_raw_2[13];
                                transposed_2[14] = r4_raw_2[7];
                                transposed_2[15] = r4_raw_2[15];
                                if (lane < 16) {
                                    float route_scale_1 = route_scale_f32[expert_1 * 64 + packed_token_1];
                                    int valid_route_1 = route_map_i32[expert_1 * 64 + packed_token_1];
                                    float scaled_1[16];
                                    if (valid_route_1 >= 0) {
                                        #pragma unroll
                                        for (int pair_idx_3 = 0; pair_idx_3 < 8; pair_idx_3++) {
                                            int elem_11 = pair_idx_3 * 2;
                                            float2 _f2_14 = make_float2(transposed_2[elem_11], transposed_2[elem_11 + 1]);
                                            float2 value_pair_1 = _f2_14;
                                            float2 _f2_15 = make_float2(route_scale_1, route_scale_1);
                                            float2 scale_pair_1 = _f2_15;
                                            float2 _mul_f32x2_7;
                                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&value_pair_1), "l"(*(const unsigned long long*)&scale_pair_1));
                                            float2 scaled_pair_1 = _mul_f32x2_7;
                                            scaled_1[elem_11] = scaled_pair_1.x;
                                            scaled_1[elem_11 + 1] = scaled_pair_1.y;
                                        }
                                    } else {
                                        #pragma unroll
                                        for (int element_1 = 0; element_1 < 16; element_1++) {
                                            scaled_1[element_1] = 0.0f;
                                        }
                                    }
                                    {
                                        {
                                            __nv_bfloat162 _pk0 = __floats2bfloat162_rn(scaled_1[0 + 0], scaled_1[0 + 1]);
                                            unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                            __nv_bfloat162 _pk1 = __floats2bfloat162_rn(scaled_1[0 + 2], scaled_1[0 + 3]);
                                            unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                            __nv_bfloat162 _pk2 = __floats2bfloat162_rn(scaled_1[0 + 4], scaled_1[0 + 5]);
                                            unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                            __nv_bfloat162 _pk3 = __floats2bfloat162_rn(scaled_1[0 + 6], scaled_1[0 + 7]);
                                            unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                            __nv_bfloat162 _pk4 = __floats2bfloat162_rn(scaled_1[0 + 8], scaled_1[0 + 9]);
                                            unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                            __nv_bfloat162 _pk5 = __floats2bfloat162_rn(scaled_1[0 + 10], scaled_1[0 + 11]);
                                            unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                            __nv_bfloat162 _pk6 = __floats2bfloat162_rn(scaled_1[0 + 12], scaled_1[0 + 13]);
                                            unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                            __nv_bfloat162 _pk7 = __floats2bfloat162_rn(scaled_1[0 + 14], scaled_1[0 + 15]);
                                            unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                            asm volatile(
                                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                :: "l"((void*)(&((__nv_bfloat16*)(fc1_workspace_bf16 + ((expert_1 * 64 + packed_token_1) * 5120 + feature_base_1)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                        }
                                    }
                                }
                            } else {
                                int hidden_base_1 = weight_m_tile * 256 + cta_rank * 128 + warp * 32;
                                unsigned int hidden_lo_1[8];
                                unsigned int hidden_hi_1[8];
                                float _tmem_load_21[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x64b.x8.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[7]))
                                    : "r"(warp_row_ptr_2));
                                float _tmem_load_22[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x64b.x8.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[7]))
                                    : "r"(warp_row_ptr_2 + 1048576));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                float casted_input_1[16];
                                #pragma unroll
                                for (int elem_12 = 0; elem_12 < 8; elem_12++) {
                                    casted_input_1[elem_12] = _tmem_load_21[elem_12];
                                    casted_input_1[8 + elem_12] = _tmem_load_22[elem_12];
                                }
                                unsigned int casted_bf16_1[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 8; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(casted_input_1[_lp*2 + 0], casted_input_1[_lp*2+1 + 0]));
                                    casted_bf16_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                float packed_1[16];
                                #pragma unroll
                                for (int pair_2 = 0; pair_2 < 4; pair_2++) {
                                    uint32_t _prmt_b32_8;
                                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_8) : "r"(casted_bf16_1[pair_2]), "r"(casted_bf16_1[4 + pair_2]));
                                    packed_1[pair_2 * 2] = __uint_as_float(_prmt_b32_8);
                                    uint32_t _prmt_b32_9;
                                    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_b32_9) : "r"(casted_bf16_1[pair_2]), "r"(casted_bf16_1[4 + pair_2]));
                                    packed_1[pair_2 * 2 + 1] = __uint_as_float(_prmt_b32_9);
                                }
                                #pragma unroll
                                for (int elem_13 = 8; elem_13 < 16; elem_13++) {
                                    packed_1[elem_13] = 0.0f;
                                }
                                float transposed_3[16];
                                float r1_3[16];
                                r1_3[0] = packed_1[0];
                                r1_3[1] = packed_1[8];
                                r1_3[2] = packed_1[2];
                                r1_3[3] = packed_1[10];
                                r1_3[4] = packed_1[4];
                                r1_3[5] = packed_1[12];
                                r1_3[6] = packed_1[6];
                                r1_3[7] = packed_1[14];
                                r1_3[8] = packed_1[1];
                                r1_3[9] = packed_1[9];
                                r1_3[10] = packed_1[3];
                                r1_3[11] = packed_1[11];
                                r1_3[12] = packed_1[5];
                                r1_3[13] = packed_1[13];
                                r1_3[14] = packed_1[7];
                                r1_3[15] = packed_1[15];
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                    :: "r"(warp_row_ptr_2), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[7])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[11])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&r1_3[15])));
                                float _tmem_load_23[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x64b.x16.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[15]))
                                    : "r"(warp_row_ptr_2));
                                tmem_st_x16_f32(warp_row_ptr_2, _tmem_load_23);
                                float r3_raw_3[16];
                                float _tmem_load_24[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[7]))
                                    : "r"(warp_row_ptr_2));
                                float _tmem_load_25[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[7]))
                                    : "r"(warp_row_ptr_2 + 1048576));
                                #pragma unroll
                                for (int elem_14 = 0; elem_14 < 8; elem_14++) {
                                    r3_raw_3[elem_14] = _tmem_load_24[elem_14];
                                    r3_raw_3[elem_14 + 8] = _tmem_load_25[elem_14];
                                }
                                float r3_3[16];
                                r3_3[0] = r3_raw_3[0];
                                r3_3[1] = r3_raw_3[1];
                                r3_3[2] = r3_raw_3[4];
                                r3_3[3] = r3_raw_3[5];
                                r3_3[4] = r3_raw_3[2];
                                r3_3[5] = r3_raw_3[3];
                                r3_3[6] = r3_raw_3[6];
                                r3_3[7] = r3_raw_3[7];
                                r3_3[8] = r3_raw_3[8];
                                r3_3[9] = r3_raw_3[9];
                                r3_3[10] = r3_raw_3[12];
                                r3_3[11] = r3_raw_3[13];
                                r3_3[12] = r3_raw_3[10];
                                r3_3[13] = r3_raw_3[11];
                                r3_3[14] = r3_raw_3[14];
                                r3_3[15] = r3_raw_3[15];
                                tmem_st_x16_f32(warp_row_ptr_2, r3_3);
                                float r4_raw_3[16];
                                float _tmem_load_26[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x128b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[7]))
                                    : "r"(warp_row_ptr_2));
                                float _tmem_load_27[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x128b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[7]))
                                    : "r"(warp_row_ptr_2 + 1048576));
                                #pragma unroll
                                for (int elem_15 = 0; elem_15 < 8; elem_15++) {
                                    r4_raw_3[elem_15] = _tmem_load_26[elem_15];
                                    r4_raw_3[elem_15 + 8] = _tmem_load_27[elem_15];
                                }
                                transposed_3[0] = r4_raw_3[0];
                                transposed_3[1] = r4_raw_3[8];
                                transposed_3[2] = r4_raw_3[2];
                                transposed_3[3] = r4_raw_3[10];
                                transposed_3[4] = r4_raw_3[4];
                                transposed_3[5] = r4_raw_3[12];
                                transposed_3[6] = r4_raw_3[6];
                                transposed_3[7] = r4_raw_3[14];
                                transposed_3[8] = r4_raw_3[1];
                                transposed_3[9] = r4_raw_3[9];
                                transposed_3[10] = r4_raw_3[3];
                                transposed_3[11] = r4_raw_3[11];
                                transposed_3[12] = r4_raw_3[5];
                                transposed_3[13] = r4_raw_3[13];
                                transposed_3[14] = r4_raw_3[7];
                                transposed_3[15] = r4_raw_3[15];
                                #pragma unroll
                                for (int pair_3 = 0; pair_3 < 8; pair_3++) {
                                    float word_a_f32_1 = transposed_3[pair_3 * 2];
                                    float word_b_f32_1 = transposed_3[pair_3 * 2 + 1];
                                    unsigned int word_a_1 = __as_u32(word_a_f32_1);
                                    unsigned int word_b_1 = __as_u32(word_b_f32_1);
                                    uint32_t _prmt_b32_10;
                                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_10) : "r"(word_a_1), "r"(word_b_1));
                                    hidden_lo_1[pair_3] = _prmt_b32_10;
                                    uint32_t _prmt_b32_11;
                                    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_b32_11) : "r"(word_a_1), "r"(word_b_1));
                                    hidden_hi_1[pair_3] = _prmt_b32_11;
                                }
                                if (lane < 16) {
                                    int global_route_1 = route_map_i32[expert_1 * 64 + packed_token_1];
                                    if (global_route_1 >= 0) {
                                        int routes_per_source_2 = tokens_per_rank * 8;
                                        int source_rank_0_1 = global_route_1 / routes_per_source_2;
                                        int source_route_1 = global_route_1 - source_rank_0_1 * routes_per_source_2;
                                        __nv_bfloat16* remote_terms_1 = reinterpret_cast<__nv_bfloat16*>(route_terms_bf16_peers[source_rank_0_1]);
                                        {
                                            const unsigned* _raw_stv8_4 = reinterpret_cast<const unsigned*>(hidden_lo_1);
                                            asm volatile(
                                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                :: "l"((void*)(remote_terms_1 + ((unsigned long long)source_route_1 * 3072 + (unsigned long long)hidden_base_1))), "r"(_raw_stv8_4[0]), "r"(_raw_stv8_4[1]), "r"(_raw_stv8_4[2]), "r"(_raw_stv8_4[3]), "r"(_raw_stv8_4[4]), "r"(_raw_stv8_4[5]), "r"(_raw_stv8_4[6]), "r"(_raw_stv8_4[7]) : "memory");
                                        }
                                        {
                                            const unsigned* _raw_stv8_5 = reinterpret_cast<const unsigned*>(hidden_hi_1);
                                            asm volatile(
                                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                :: "l"((void*)(remote_terms_1 + ((unsigned long long)source_route_1 * 3072 + (unsigned long long)hidden_base_1 + 16))), "r"(_raw_stv8_5[0]), "r"(_raw_stv8_5[1]), "r"(_raw_stv8_5[2]), "r"(_raw_stv8_5[3]), "r"(_raw_stv8_5[4]), "r"(_raw_stv8_5[5]), "r"(_raw_stv8_5[6]), "r"(_raw_stv8_5[7]) : "memory");
                                        }
                                    }
                                }
                            }
                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                            if (phase == 0) {
                                if (warp == 0) {
                                    if (elect_sync()) {
                                        {
                                            unsigned int* _gc_p = reinterpret_cast<unsigned int*>(fc1_done) + (expert_row_tile_2);
                                            unsigned int _gc_old;
                                            asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(acc_free_addr + acc_stage * 8), "r"(0) : "memory");
                    }
                    acc_stage += 1;
                    if (acc_stage == 2) { acc_stage = 0; _phase_acc_full ^= 1; }
                }
            }
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(return_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(return_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            if (bid == 0 && warp == 0) {
                if (elect_sync()) {
                    // nvlink_barrier(mega_pg_flags) phase=1
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __slot = 1;
                        unsigned* __local_flag = mega_pg_flags[__me] + __slot;
                        unsigned __old_sense = 0u;
                        const unsigned __delta = (__me == 0) ? (0x80000000u - (unsigned)(__ws - 1)) : 1u;
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        for (int __r = 0; __r < __ws; ++__r) {
                            unsigned* __peer_flag = mega_pg_flags[__r] + __slot;
                            unsigned __old_peer;
                            asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                                : "=r"(__old_peer) : "l"(__peer_flag), "r"(__delta) : "memory");
                            if (__r == __me) __old_sense = __old_peer;
                        }
                        asm volatile("fence.proxy.alias;" ::: "memory");
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                            if (((__old_sense ^ __v) & 0x80000000u) != 0u) break;
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(return_visible) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(return_visible) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(launch_epoch + 1)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int linear_thread_1 = bid * 384 + tid;
            int thread_stride_1 = num_bids * 384;
            #pragma unroll 1
            for (int index_2 = linear_thread_1; index_2 < 2048; index_2 += thread_stride_1) {
                route_map_i32[index_2] = -1;
                route_scale_f32[index_2] = 0.0f;
            }
            #pragma unroll 1
            for (int expert_2 = linear_thread_1; expert_2 < 32; expert_2 += thread_stride_1) {
                route_counts_u32[expert_2] = 0;
            }
            #pragma unroll 1
            for (int expert_row_tile_3 = linear_thread_1; expert_row_tile_3 < 96; expert_row_tile_3 += thread_stride_1) {
                {
                    unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(fc1_done) + (expert_row_tile_3);
                    asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
                }
            }
            #pragma unroll 1
            for (int index_3 = linear_thread_1 * 8; index_3 < tokens_per_rank * 3072; index_3 += thread_stride_1 * 8) {
                float _vec_load_8[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(source_hidden_bf16 + index_3 + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(_vec_load_8[0 + 0], _vec_load_8[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(_vec_load_8[0 + 2], _vec_load_8[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(_vec_load_8[0 + 4], _vec_load_8[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(_vec_load_8[0 + 6], _vec_load_8[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(published_hidden_bf16) + index_3))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
            #pragma unroll 1
            for (int route_1 = linear_thread_1; route_1 < tokens_per_rank * 8; route_1 += thread_stride_1) {
                int route_id_1 = (int)source_topk_ids_i64[route_1];
                float route_weight_1 = source_topk_weights_f32[route_1];
                *(reinterpret_cast<int*>(reinterpret_cast<int*>(published_topk_ids_i32) + route_1) + (0)) = route_id_1;
                *(reinterpret_cast<float*>(reinterpret_cast<float*>(published_topk_weights_f32) + route_1) + (0)) = route_weight_1;
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(publication_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(publication_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            if (bid == 0 && warp == 0) {
                if (elect_sync()) {
                    // nvlink_barrier(mega_pg_flags) phase=0
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __slot = 0;
                        unsigned* __local_flag = mega_pg_flags[__me] + __slot;
                        unsigned __old_sense = 0u;
                        const unsigned __delta = (__me == 0) ? (0x80000000u - (unsigned)(__ws - 1)) : 1u;
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        for (int __r = 0; __r < __ws; ++__r) {
                            unsigned* __peer_flag = mega_pg_flags[__r] + __slot;
                            unsigned __old_peer;
                            asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                                : "=r"(__old_peer) : "l"(__peer_flag), "r"(__delta) : "memory");
                            if (__r == __me) __old_sense = __old_peer;
                        }
                        asm volatile("fence.proxy.alias;" ::: "memory");
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                            if (((__old_sense ^ __v) & 0x80000000u) != 0u) break;
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(publication_visible) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(publication_visible) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(launch_epoch + 1)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            int source_rank_1 = bid % 16;
            int source_token_begin_1 = bid / 16;
            int* remote_ids_1 = reinterpret_cast<int*>(published_topk_ids_i32_peers[source_rank_1]);
            float* remote_weights_1 = reinterpret_cast<float*>(published_topk_weights_f32_peers[source_rank_1]);
            __nv_bfloat16* remote_hidden_1 = reinterpret_cast<__nv_bfloat16*>(published_hidden_bf16_peers[source_rank_1]);
            #pragma unroll 1
            for (int source_token_1 = source_token_begin_1; source_token_1 < tokens_per_rank; source_token_1 += 9) {
                int routes_per_source_3 = tokens_per_rank * 8;
                int source_route_lane_1 = source_token_1 * 8 + lane;
                int row_index_lane_1 = -1;
                int row_lane_1 = 0;
                if (warp == 0 && lane < 8) {
                    int global_expert_1 = remote_ids_1[source_route_lane_1];
                    int owner_1 = global_expert_1 / 32;
                    if (owner_1 == mega_pg_rank) {
                        int local_expert_lane_1 = global_expert_1 - mega_pg_rank * 32;
                        unsigned int _atomic_old_4 = atomicAdd(&route_counts_u32[local_expert_lane_1], 1);
                        row_lane_1 = (int)_atomic_old_4;
                        if (row_lane_1 < 64) {
                            row_index_lane_1 = local_expert_lane_1 * 64 + row_lane_1;
                            int global_route_lane_1 = source_rank_1 * routes_per_source_3 + source_route_lane_1;
                            route_map_i32[row_index_lane_1] = global_route_lane_1;
                            route_scale_f32[row_index_lane_1] = remote_weights_1[source_route_lane_1];
                        }
                    }
                    dispatch_rows_i32[lane] = row_index_lane_1;
                }
                if (warp == 0) {
                    int _vote_4 = __any_sync(0xFFFFFFFF, row_index_lane_1 >= 0);
                    int any_owned_1 = _vote_4;
                    if (lane == 0) {
                        dispatch_rows_i32[8] = any_owned_1;
                    }
                }
                asm volatile("barrier.sync 11, 384;" ::: "memory");
                int dispatch_any_1 = dispatch_rows_i32[8];
                if (dispatch_any_1 != 0) {
                    int column_1 = tid * 8;
                    float _vec_load_9[8];
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(remote_hidden_1 + (source_token_1 * 3072 + column_1) + 0);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int route_slot_1 = 0; route_slot_1 < 8; route_slot_1++) {
                        int row_index_1 = dispatch_rows_i32[route_slot_1];
                        if (row_index_1 >= 0) {
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(_vec_load_9[0 + 0], _vec_load_9[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_vec_load_9[0 + 2], _vec_load_9[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_vec_load_9[0 + 4], _vec_load_9[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_vec_load_9[0 + 6], _vec_load_9[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(activation_bf16_ptr + (row_index_1 * 3072 + column_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                }
                asm volatile("barrier.sync 11, 384;" ::: "memory");
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            asm volatile("fence.release.gpu;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            unsigned int transformed_stage = 0;
            unsigned int b_stage = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int _phase_acc_free = 1;
            unsigned int _phase_transformed_full = 0;
            unsigned int _phase_b_full = 0;
            if (cta_rank == 0) {
                #pragma unroll
                for (int phase_1 = 0; phase_1 < 2; phase_1++) {
                    int work_begin_1 = ((phase_1 == 0) ? (int)cluster_id : ((int)cluster_id + 16) % 72);
                    int work_end_1 = ((phase_1 == 0) ? 1280 : 384);
                    int m_tiles_per_expert_1 = ((phase_1 == 0) ? 40 : 12);
                    #pragma unroll 1
                    for (int work_1 = work_begin_1; work_1 < work_end_1; work_1 += (int)num_clusters) {
                        int expert_3 = work_1 / m_tiles_per_expert_1;
                        int weight_m_tile_1 = work_1 - expert_3 * m_tiles_per_expert_1;
                        int route_count_1 = (int)route_counts_u32[expert_3];
                        int num_k_tiles = ((phase_1 == 0) ? 24 : 40);
                        mbarrier_wait(acc_free_addr + (acc_stage_1) * 8, _phase_acc_free);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < num_k_tiles; iter_k++) {
                            mbarrier_wait_cluster(transformed_full_addr + (transformed_stage) * 8, _phase_transformed_full);
                            mbarrier_wait_cluster(b_full_addr + (b_stage) * 8, _phase_b_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_0 = (((b_smem_addr) >> 4) & 0x3FFF) + (b_stage) * 256;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 268960912;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_accum + (acc_stage_1 * 3 * 32))), "r"(_mma_b_lo_0), "r"((unsigned int)tmem_transformed_a + transformed_stage * 64), "r"(((((iter_k == 0) ? 1 : 0)) ? 0 : 1)));
                            elect_commit_cg2_multicast(b_free_addr + (b_stage) * 8, (uint16_t)(3));
                            b_stage += 1;
                            if (b_stage == 2) { b_stage = 0; _phase_b_full ^= 1; }
                            #pragma unroll
                            for (int tail_1 = 0; tail_1 < 2; tail_1++) {
                                if (route_count_1 > 32 + tail_1 * 16) {
                                    mbarrier_wait_cluster(b_full_addr + (b_stage) * 8, _phase_b_full);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    int _mma_b_lo_1 = (((b_smem_n16_addr) >> 4) & 0x3FFF) + (b_stage) * 256;
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 268698768;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_accum + ((acc_stage_1 * 3 + (unsigned int)tail_1 + 1) * 32))), "r"(_mma_b_lo_1), "r"((unsigned int)tmem_transformed_a + transformed_stage * 64), "r"(((((iter_k == 0) ? 1 : 0)) ? 0 : 1)));
                                    elect_commit_cg2_multicast(b_free_addr + (b_stage) * 8, (uint16_t)(3));
                                    b_stage += 1;
                                    if (b_stage == 2) { b_stage = 0; _phase_b_full ^= 1; }
                                }
                            }
                            elect_commit_cg2_multicast(transformed_free_addr + (transformed_stage) * 8, (uint16_t)(3));
                            transformed_stage += 1;
                            if (transformed_stage == 4) { transformed_stage = 0; _phase_transformed_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(acc_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                        acc_stage_1 += 1;
                        if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_acc_free ^= 1; }
                    }
                }
            }
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(return_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(return_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            if (bid == 0 && warp == 0) {
                if (elect_sync()) {
                    // nvlink_barrier(mega_pg_flags) phase=1
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __slot = 1;
                        unsigned* __local_flag = mega_pg_flags[__me] + __slot;
                        unsigned __old_sense = 0u;
                        const unsigned __delta = (__me == 0) ? (0x80000000u - (unsigned)(__ws - 1)) : 1u;
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        for (int __r = 0; __r < __ws; ++__r) {
                            unsigned* __peer_flag = mega_pg_flags[__r] + __slot;
                            unsigned __old_peer;
                            asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                                : "=r"(__old_peer) : "l"(__peer_flag), "r"(__delta) : "memory");
                            if (__r == __me) __old_sense = __old_peer;
                        }
                        asm volatile("fence.proxy.alias;" ::: "memory");
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                            if (((__old_sense ^ __v) & 0x80000000u) != 0u) break;
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(return_visible) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(return_visible) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(launch_epoch + 1)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
        }
    }
    // ---- Role: tma_a_scale ----
    if (warp == 5) {
        { // tma_a_scale_main
            int linear_thread_2 = bid * 384 + tid;
            int thread_stride_2 = num_bids * 384;
            #pragma unroll 1
            for (int index_4 = linear_thread_2; index_4 < 2048; index_4 += thread_stride_2) {
                route_map_i32[index_4] = -1;
                route_scale_f32[index_4] = 0.0f;
            }
            #pragma unroll 1
            for (int expert_4 = linear_thread_2; expert_4 < 32; expert_4 += thread_stride_2) {
                route_counts_u32[expert_4] = 0;
            }
            #pragma unroll 1
            for (int expert_row_tile_4 = linear_thread_2; expert_row_tile_4 < 96; expert_row_tile_4 += thread_stride_2) {
                {
                    unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(fc1_done) + (expert_row_tile_4);
                    asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
                }
            }
            #pragma unroll 1
            for (int index_5 = linear_thread_2 * 8; index_5 < tokens_per_rank * 3072; index_5 += thread_stride_2 * 8) {
                float _vec_load_0[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(source_hidden_bf16 + index_5 + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(_vec_load_0[0 + 0], _vec_load_0[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(_vec_load_0[0 + 2], _vec_load_0[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(_vec_load_0[0 + 4], _vec_load_0[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(_vec_load_0[0 + 6], _vec_load_0[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(published_hidden_bf16) + index_5))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
            #pragma unroll 1
            for (int route_2 = linear_thread_2; route_2 < tokens_per_rank * 8; route_2 += thread_stride_2) {
                int route_id_2 = (int)source_topk_ids_i64[route_2];
                float route_weight_2 = source_topk_weights_f32[route_2];
                *(reinterpret_cast<int*>(reinterpret_cast<int*>(published_topk_ids_i32) + route_2) + (0)) = route_id_2;
                *(reinterpret_cast<float*>(reinterpret_cast<float*>(published_topk_weights_f32) + route_2) + (0)) = route_weight_2;
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(publication_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(publication_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            if (bid == 0 && warp == 0) {
                if (elect_sync()) {
                    // nvlink_barrier(mega_pg_flags) phase=0
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __slot = 0;
                        unsigned* __local_flag = mega_pg_flags[__me] + __slot;
                        unsigned __old_sense = 0u;
                        const unsigned __delta = (__me == 0) ? (0x80000000u - (unsigned)(__ws - 1)) : 1u;
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        for (int __r = 0; __r < __ws; ++__r) {
                            unsigned* __peer_flag = mega_pg_flags[__r] + __slot;
                            unsigned __old_peer;
                            asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                                : "=r"(__old_peer) : "l"(__peer_flag), "r"(__delta) : "memory");
                            if (__r == __me) __old_sense = __old_peer;
                        }
                        asm volatile("fence.proxy.alias;" ::: "memory");
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                            if (((__old_sense ^ __v) & 0x80000000u) != 0u) break;
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(publication_visible) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(publication_visible) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(launch_epoch + 1)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            int source_rank_2 = bid % 16;
            int source_token_begin_2 = bid / 16;
            int* remote_ids_2 = reinterpret_cast<int*>(published_topk_ids_i32_peers[source_rank_2]);
            float* remote_weights_2 = reinterpret_cast<float*>(published_topk_weights_f32_peers[source_rank_2]);
            __nv_bfloat16* remote_hidden_2 = reinterpret_cast<__nv_bfloat16*>(published_hidden_bf16_peers[source_rank_2]);
            #pragma unroll 1
            for (int source_token_2 = source_token_begin_2; source_token_2 < tokens_per_rank; source_token_2 += 9) {
                int routes_per_source_4 = tokens_per_rank * 8;
                int source_route_lane_2 = source_token_2 * 8 + lane;
                int row_index_lane_2 = -1;
                int row_lane_2 = 0;
                if (warp == 0 && lane < 8) {
                    int global_expert_2 = remote_ids_2[source_route_lane_2];
                    int owner_2 = global_expert_2 / 32;
                    if (owner_2 == mega_pg_rank) {
                        int local_expert_lane_2 = global_expert_2 - mega_pg_rank * 32;
                        unsigned int _atomic_old_0 = atomicAdd(&route_counts_u32[local_expert_lane_2], 1);
                        row_lane_2 = (int)_atomic_old_0;
                        if (row_lane_2 < 64) {
                            row_index_lane_2 = local_expert_lane_2 * 64 + row_lane_2;
                            int global_route_lane_2 = source_rank_2 * routes_per_source_4 + source_route_lane_2;
                            route_map_i32[row_index_lane_2] = global_route_lane_2;
                            route_scale_f32[row_index_lane_2] = remote_weights_2[source_route_lane_2];
                        }
                    }
                    dispatch_rows_i32[lane] = row_index_lane_2;
                }
                if (warp == 0) {
                    int _vote_0 = __any_sync(0xFFFFFFFF, row_index_lane_2 >= 0);
                    int any_owned_2 = _vote_0;
                    if (lane == 0) {
                        dispatch_rows_i32[8] = any_owned_2;
                    }
                }
                asm volatile("barrier.sync 11, 384;" ::: "memory");
                int dispatch_any_2 = dispatch_rows_i32[8];
                if (dispatch_any_2 != 0) {
                    int column_2 = tid * 8;
                    float _vec_load_1[8];
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(remote_hidden_2 + (source_token_2 * 3072 + column_2) + 0);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int route_slot_2 = 0; route_slot_2 < 8; route_slot_2++) {
                        int row_index_2 = dispatch_rows_i32[route_slot_2];
                        if (row_index_2 >= 0) {
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(_vec_load_1[0 + 0], _vec_load_1[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_vec_load_1[0 + 2], _vec_load_1[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_vec_load_1[0 + 4], _vec_load_1[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_vec_load_1[0 + 6], _vec_load_1[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(activation_bf16_ptr + (row_index_2 * 3072 + column_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                }
                asm volatile("barrier.sync 11, 384;" ::: "memory");
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            asm volatile("fence.release.gpu;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            unsigned int a_stage = 0;
            unsigned int scale_stage = 0;
            unsigned int _phase_a_free = 1;
            unsigned int _phase_scale_free = 1;
            #pragma unroll
            for (int phase_2 = 0; phase_2 < 2; phase_2++) {
                int work_begin_2 = ((phase_2 == 0) ? (int)cluster_id : ((int)cluster_id + 16) % 72);
                int work_end_2 = ((phase_2 == 0) ? 1280 : 384);
                int m_tiles_per_expert_2 = ((phase_2 == 0) ? 40 : 12);
                #pragma unroll 1
                for (int work_2 = work_begin_2; work_2 < work_end_2; work_2 += (int)num_clusters) {
                    int expert_5 = work_2 / m_tiles_per_expert_2;
                    int weight_m_tile_2 = work_2 - expert_5 * m_tiles_per_expert_2;
                    int num_k_tiles_1 = ((phase_2 == 0) ? 24 : 40);
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < num_k_tiles_1; iter_k_1++) {
                        mbarrier_wait(a_free_addr + (a_stage) * 8, _phase_a_free);
                        mbarrier_wait(scale_free_addr + (scale_stage) * 8, _phase_scale_free);
                        if (elect_sync()) {
                            if (phase_2 == 0) {
                                tma_4d_gmem2smem(raw_smem_addr + a_stage * 16384, fc1_weight_e4m3, 0, weight_m_tile_2 * 256 + cta_rank * 128, iter_k_1, expert_5, a_full_addr + (a_stage) * 8);
                            } else {
                                tma_4d_gmem2smem(raw_smem_addr + a_stage * 16384, fc2_weight_e4m3, 0, weight_m_tile_2 * 256 + cta_rank * 128, iter_k_1, expert_5, a_full_addr + (a_stage) * 8);
                            }
                            mbarrier_arrive_expect_tx(a_full_addr + (a_stage) * 8, 16384);
                            int scale_atom = (weight_m_tile_2 * 2 + cta_rank) * num_k_tiles_1 + iter_k_1;
                            if (phase_2 == 0) {
                                scale_atom = expert_5 * 1920 + scale_atom;
                            } else {
                                scale_atom = expert_5 * 960 + scale_atom;
                            }
                            if (phase_2 == 0) {
                                tma_2d_gmem2smem(scale_smem_addr + scale_stage * 512, fc1_blocked_e8m0, 0, scale_atom * 4, scale_full_addr + (scale_stage) * 8);
                            } else {
                                tma_2d_gmem2smem(scale_smem_addr + scale_stage * 512, fc2_blocked_e8m0, 0, scale_atom * 4, scale_full_addr + (scale_stage) * 8);
                            }
                            mbarrier_arrive_expect_tx(scale_full_addr + (scale_stage) * 8, 512);
                        }
                        a_stage += 1;
                        if (a_stage == 4) { a_stage = 0; _phase_a_free ^= 1; }
                        scale_stage += 1;
                        if (scale_stage == 4) { scale_stage = 0; _phase_scale_free ^= 1; }
                    }
                }
            }
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(return_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(return_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            if (bid == 0 && warp == 0) {
                if (elect_sync()) {
                    // nvlink_barrier(mega_pg_flags) phase=1
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __slot = 1;
                        unsigned* __local_flag = mega_pg_flags[__me] + __slot;
                        unsigned __old_sense = 0u;
                        const unsigned __delta = (__me == 0) ? (0x80000000u - (unsigned)(__ws - 1)) : 1u;
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        for (int __r = 0; __r < __ws; ++__r) {
                            unsigned* __peer_flag = mega_pg_flags[__r] + __slot;
                            unsigned __old_peer;
                            asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                                : "=r"(__old_peer) : "l"(__peer_flag), "r"(__delta) : "memory");
                            if (__r == __me) __old_sense = __old_peer;
                        }
                        asm volatile("fence.proxy.alias;" ::: "memory");
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                            if (((__old_sense ^ __v) & 0x80000000u) != 0u) break;
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(return_visible) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(return_visible) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(launch_epoch + 1)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
        }
    }
    // ---- Role: tma_b ----
    if (warp == 6) {
        { // tma_b_main
            int linear_thread_3 = bid * 384 + tid;
            int thread_stride_3 = num_bids * 384;
            #pragma unroll 1
            for (int index_6 = linear_thread_3; index_6 < 2048; index_6 += thread_stride_3) {
                route_map_i32[index_6] = -1;
                route_scale_f32[index_6] = 0.0f;
            }
            #pragma unroll 1
            for (int expert_6 = linear_thread_3; expert_6 < 32; expert_6 += thread_stride_3) {
                route_counts_u32[expert_6] = 0;
            }
            #pragma unroll 1
            for (int expert_row_tile_5 = linear_thread_3; expert_row_tile_5 < 96; expert_row_tile_5 += thread_stride_3) {
                {
                    unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(fc1_done) + (expert_row_tile_5);
                    asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
                }
            }
            #pragma unroll 1
            for (int index_7 = linear_thread_3 * 8; index_7 < tokens_per_rank * 3072; index_7 += thread_stride_3 * 8) {
                float _vec_load_2[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(source_hidden_bf16 + index_7 + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(_vec_load_2[0 + 0], _vec_load_2[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(_vec_load_2[0 + 2], _vec_load_2[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(_vec_load_2[0 + 4], _vec_load_2[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(_vec_load_2[0 + 6], _vec_load_2[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(published_hidden_bf16) + index_7))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
            #pragma unroll 1
            for (int route_3 = linear_thread_3; route_3 < tokens_per_rank * 8; route_3 += thread_stride_3) {
                int route_id_3 = (int)source_topk_ids_i64[route_3];
                float route_weight_3 = source_topk_weights_f32[route_3];
                *(reinterpret_cast<int*>(reinterpret_cast<int*>(published_topk_ids_i32) + route_3) + (0)) = route_id_3;
                *(reinterpret_cast<float*>(reinterpret_cast<float*>(published_topk_weights_f32) + route_3) + (0)) = route_weight_3;
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(publication_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(publication_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            if (bid == 0 && warp == 0) {
                if (elect_sync()) {
                    // nvlink_barrier(mega_pg_flags) phase=0
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __slot = 0;
                        unsigned* __local_flag = mega_pg_flags[__me] + __slot;
                        unsigned __old_sense = 0u;
                        const unsigned __delta = (__me == 0) ? (0x80000000u - (unsigned)(__ws - 1)) : 1u;
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        for (int __r = 0; __r < __ws; ++__r) {
                            unsigned* __peer_flag = mega_pg_flags[__r] + __slot;
                            unsigned __old_peer;
                            asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                                : "=r"(__old_peer) : "l"(__peer_flag), "r"(__delta) : "memory");
                            if (__r == __me) __old_sense = __old_peer;
                        }
                        asm volatile("fence.proxy.alias;" ::: "memory");
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                            if (((__old_sense ^ __v) & 0x80000000u) != 0u) break;
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(publication_visible) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(publication_visible) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(launch_epoch + 1)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            int source_rank_3 = bid % 16;
            int source_token_begin_3 = bid / 16;
            int* remote_ids_3 = reinterpret_cast<int*>(published_topk_ids_i32_peers[source_rank_3]);
            float* remote_weights_3 = reinterpret_cast<float*>(published_topk_weights_f32_peers[source_rank_3]);
            __nv_bfloat16* remote_hidden_3 = reinterpret_cast<__nv_bfloat16*>(published_hidden_bf16_peers[source_rank_3]);
            #pragma unroll 1
            for (int source_token_3 = source_token_begin_3; source_token_3 < tokens_per_rank; source_token_3 += 9) {
                int routes_per_source_5 = tokens_per_rank * 8;
                int source_route_lane_3 = source_token_3 * 8 + lane;
                int row_index_lane_3 = -1;
                int row_lane_3 = 0;
                if (warp == 0 && lane < 8) {
                    int global_expert_3 = remote_ids_3[source_route_lane_3];
                    int owner_3 = global_expert_3 / 32;
                    if (owner_3 == mega_pg_rank) {
                        int local_expert_lane_3 = global_expert_3 - mega_pg_rank * 32;
                        unsigned int _atomic_old_1 = atomicAdd(&route_counts_u32[local_expert_lane_3], 1);
                        row_lane_3 = (int)_atomic_old_1;
                        if (row_lane_3 < 64) {
                            row_index_lane_3 = local_expert_lane_3 * 64 + row_lane_3;
                            int global_route_lane_3 = source_rank_3 * routes_per_source_5 + source_route_lane_3;
                            route_map_i32[row_index_lane_3] = global_route_lane_3;
                            route_scale_f32[row_index_lane_3] = remote_weights_3[source_route_lane_3];
                        }
                    }
                    dispatch_rows_i32[lane] = row_index_lane_3;
                }
                if (warp == 0) {
                    int _vote_1 = __any_sync(0xFFFFFFFF, row_index_lane_3 >= 0);
                    int any_owned_3 = _vote_1;
                    if (lane == 0) {
                        dispatch_rows_i32[8] = any_owned_3;
                    }
                }
                asm volatile("barrier.sync 11, 384;" ::: "memory");
                int dispatch_any_3 = dispatch_rows_i32[8];
                if (dispatch_any_3 != 0) {
                    int column_3 = tid * 8;
                    float _vec_load_3[8];
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(remote_hidden_3 + (source_token_3 * 3072 + column_3) + 0);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int route_slot_3 = 0; route_slot_3 < 8; route_slot_3++) {
                        int row_index_3 = dispatch_rows_i32[route_slot_3];
                        if (row_index_3 >= 0) {
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(_vec_load_3[0 + 0], _vec_load_3[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_vec_load_3[0 + 2], _vec_load_3[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_vec_load_3[0 + 4], _vec_load_3[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_vec_load_3[0 + 6], _vec_load_3[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(activation_bf16_ptr + (row_index_3 * 3072 + column_3)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                }
                asm volatile("barrier.sync 11, 384;" ::: "memory");
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            asm volatile("fence.release.gpu;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            unsigned int b_stage_1 = 0;
            unsigned int _phase_b_free = 1;
            #pragma unroll
            for (int phase_3 = 0; phase_3 < 2; phase_3++) {
                int work_begin_3 = ((phase_3 == 0) ? (int)cluster_id : ((int)cluster_id + 16) % 72);
                int work_end_3 = ((phase_3 == 0) ? 1280 : 384);
                int m_tiles_per_expert_3 = ((phase_3 == 0) ? 40 : 12);
                #pragma unroll 1
                for (int work_3 = work_begin_3; work_3 < work_end_3; work_3 += (int)num_clusters) {
                    int expert_7 = work_3 / m_tiles_per_expert_3;
                    int weight_m_tile_3 = work_3 - expert_7 * m_tiles_per_expert_3;
                    int route_count_2 = (int)route_counts_u32[expert_7];
                    if (phase_3 == 1) {
                        {
                            unsigned int* _gca_p = reinterpret_cast<unsigned int*>(fc1_done) + (expert_7);
                            while (true) {
                                unsigned int _gca_v;
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                if (_gca_v >= (unsigned int)(80)) break;
                            }
                        }
                        #pragma unroll
                        for (int tail_2 = 0; tail_2 < 2; tail_2++) {
                            if (route_count_2 > 32 + tail_2 * 16) {
                                {
                                    unsigned int* _gca_p = reinterpret_cast<unsigned int*>(fc1_done) + ((tail_2 + 1) * 32 + expert_7);
                                    while (true) {
                                        unsigned int _gca_v;
                                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                        if (_gca_v >= (unsigned int)(80)) break;
                                    }
                                }
                            }
                        }
                    }
                    int num_k_tiles_2 = ((phase_3 == 0) ? 24 : 40);
                    #pragma unroll 1
                    for (int iter_k_2 = 0; iter_k_2 < num_k_tiles_2; iter_k_2++) {
                        mbarrier_wait(b_free_addr + (b_stage_1) * 8, _phase_b_free);
                        if (elect_sync()) {
                            if (phase_3 == 0) {
                                tma_4d_gmem2smem_cta2(b_smem_addr + b_stage_1 * 4096, activation_bf16, 0, cta_rank * 16, iter_k_2 * 2, expert_7, ((b_full_addr + (b_stage_1) * 8) & 0xFEFFFFFF));
                            } else {
                                tma_4d_gmem2smem_cta2(b_smem_addr + b_stage_1 * 4096, fc1_workspace_bf16_tma, 0, cta_rank * 16, iter_k_2 * 2, expert_7, ((b_full_addr + (b_stage_1) * 8) & 0xFEFFFFFF));
                            }
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((b_full_addr + (b_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(4096)) : "memory");
                        }
                        b_stage_1 += 1;
                        if (b_stage_1 == 2) { b_stage_1 = 0; _phase_b_free ^= 1; }
                        #pragma unroll
                        for (int tail_3 = 0; tail_3 < 2; tail_3++) {
                            if (route_count_2 > 32 + tail_3 * 16) {
                                mbarrier_wait(b_free_addr + (b_stage_1) * 8, _phase_b_free);
                                if (elect_sync()) {
                                    int tail_row = 32 + tail_3 * 16;
                                    if (phase_3 == 0) {
                                        tma_4d_gmem2smem_cta2(b_smem_n16_addr + b_stage_1 * 4096, activation_bf16_n16, 0, tail_row + cta_rank * 8, iter_k_2 * 2, expert_7, ((b_full_addr + (b_stage_1) * 8) & 0xFEFFFFFF));
                                    } else {
                                        tma_4d_gmem2smem_cta2(b_smem_n16_addr + b_stage_1 * 4096, fc1_workspace_bf16_tma_n16, 0, tail_row + cta_rank * 8, iter_k_2 * 2, expert_7, ((b_full_addr + (b_stage_1) * 8) & 0xFEFFFFFF));
                                    }
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((b_full_addr + (b_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(2048)) : "memory");
                                }
                                b_stage_1 += 1;
                                if (b_stage_1 == 2) { b_stage_1 = 0; _phase_b_free ^= 1; }
                            }
                        }
                    }
                    if (phase_3 == 1) {
                    }
                }
            }
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(return_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(return_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            if (bid == 0 && warp == 0) {
                if (elect_sync()) {
                    // nvlink_barrier(mega_pg_flags) phase=1
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __slot = 1;
                        unsigned* __local_flag = mega_pg_flags[__me] + __slot;
                        unsigned __old_sense = 0u;
                        const unsigned __delta = (__me == 0) ? (0x80000000u - (unsigned)(__ws - 1)) : 1u;
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        for (int __r = 0; __r < __ws; ++__r) {
                            unsigned* __peer_flag = mega_pg_flags[__r] + __slot;
                            unsigned __old_peer;
                            asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                                : "=r"(__old_peer) : "l"(__peer_flag), "r"(__delta) : "memory");
                            if (__r == __me) __old_sense = __old_peer;
                        }
                        asm volatile("fence.proxy.alias;" ::: "memory");
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                            if (((__old_sense ^ __v) & 0x80000000u) != 0u) break;
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(return_visible) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(return_visible) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(launch_epoch + 1)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
        }
    }
    // ---- Role: scheduler ----
    if (warp == 7) {
        { // scheduler_main
            int linear_thread_4 = bid * 384 + tid;
            int thread_stride_4 = num_bids * 384;
            #pragma unroll 1
            for (int index_8 = linear_thread_4; index_8 < 2048; index_8 += thread_stride_4) {
                route_map_i32[index_8] = -1;
                route_scale_f32[index_8] = 0.0f;
            }
            #pragma unroll 1
            for (int expert_8 = linear_thread_4; expert_8 < 32; expert_8 += thread_stride_4) {
                route_counts_u32[expert_8] = 0;
            }
            #pragma unroll 1
            for (int expert_row_tile_6 = linear_thread_4; expert_row_tile_6 < 96; expert_row_tile_6 += thread_stride_4) {
                {
                    unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(fc1_done) + (expert_row_tile_6);
                    asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
                }
            }
            #pragma unroll 1
            for (int index_9 = linear_thread_4 * 8; index_9 < tokens_per_rank * 3072; index_9 += thread_stride_4 * 8) {
                float _vec_load_4[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(source_hidden_bf16 + index_9 + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(_vec_load_4[0 + 0], _vec_load_4[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(_vec_load_4[0 + 2], _vec_load_4[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(_vec_load_4[0 + 4], _vec_load_4[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(_vec_load_4[0 + 6], _vec_load_4[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(published_hidden_bf16) + index_9))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
            #pragma unroll 1
            for (int route_4 = linear_thread_4; route_4 < tokens_per_rank * 8; route_4 += thread_stride_4) {
                int route_id_4 = (int)source_topk_ids_i64[route_4];
                float route_weight_4 = source_topk_weights_f32[route_4];
                *(reinterpret_cast<int*>(reinterpret_cast<int*>(published_topk_ids_i32) + route_4) + (0)) = route_id_4;
                *(reinterpret_cast<float*>(reinterpret_cast<float*>(published_topk_weights_f32) + route_4) + (0)) = route_weight_4;
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(publication_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(publication_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            if (bid == 0 && warp == 0) {
                if (elect_sync()) {
                    // nvlink_barrier(mega_pg_flags) phase=0
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __slot = 0;
                        unsigned* __local_flag = mega_pg_flags[__me] + __slot;
                        unsigned __old_sense = 0u;
                        const unsigned __delta = (__me == 0) ? (0x80000000u - (unsigned)(__ws - 1)) : 1u;
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        for (int __r = 0; __r < __ws; ++__r) {
                            unsigned* __peer_flag = mega_pg_flags[__r] + __slot;
                            unsigned __old_peer;
                            asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                                : "=r"(__old_peer) : "l"(__peer_flag), "r"(__delta) : "memory");
                            if (__r == __me) __old_sense = __old_peer;
                        }
                        asm volatile("fence.proxy.alias;" ::: "memory");
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                            if (((__old_sense ^ __v) & 0x80000000u) != 0u) break;
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(publication_visible) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(publication_visible) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(launch_epoch + 1)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            int source_rank_4 = bid % 16;
            int source_token_begin_4 = bid / 16;
            int* remote_ids_4 = reinterpret_cast<int*>(published_topk_ids_i32_peers[source_rank_4]);
            float* remote_weights_4 = reinterpret_cast<float*>(published_topk_weights_f32_peers[source_rank_4]);
            __nv_bfloat16* remote_hidden_4 = reinterpret_cast<__nv_bfloat16*>(published_hidden_bf16_peers[source_rank_4]);
            #pragma unroll 1
            for (int source_token_4 = source_token_begin_4; source_token_4 < tokens_per_rank; source_token_4 += 9) {
                int routes_per_source_6 = tokens_per_rank * 8;
                int source_route_lane_4 = source_token_4 * 8 + lane;
                int row_index_lane_4 = -1;
                int row_lane_4 = 0;
                if (warp == 0 && lane < 8) {
                    int global_expert_4 = remote_ids_4[source_route_lane_4];
                    int owner_4 = global_expert_4 / 32;
                    if (owner_4 == mega_pg_rank) {
                        int local_expert_lane_4 = global_expert_4 - mega_pg_rank * 32;
                        unsigned int _atomic_old_2 = atomicAdd(&route_counts_u32[local_expert_lane_4], 1);
                        row_lane_4 = (int)_atomic_old_2;
                        if (row_lane_4 < 64) {
                            row_index_lane_4 = local_expert_lane_4 * 64 + row_lane_4;
                            int global_route_lane_4 = source_rank_4 * routes_per_source_6 + source_route_lane_4;
                            route_map_i32[row_index_lane_4] = global_route_lane_4;
                            route_scale_f32[row_index_lane_4] = remote_weights_4[source_route_lane_4];
                        }
                    }
                    dispatch_rows_i32[lane] = row_index_lane_4;
                }
                if (warp == 0) {
                    int _vote_2 = __any_sync(0xFFFFFFFF, row_index_lane_4 >= 0);
                    int any_owned_4 = _vote_2;
                    if (lane == 0) {
                        dispatch_rows_i32[8] = any_owned_4;
                    }
                }
                asm volatile("barrier.sync 11, 384;" ::: "memory");
                int dispatch_any_4 = dispatch_rows_i32[8];
                if (dispatch_any_4 != 0) {
                    int column_4 = tid * 8;
                    float _vec_load_5[8];
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(remote_hidden_4 + (source_token_4 * 3072 + column_4) + 0);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int route_slot_4 = 0; route_slot_4 < 8; route_slot_4++) {
                        int row_index_4 = dispatch_rows_i32[route_slot_4];
                        if (row_index_4 >= 0) {
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(_vec_load_5[0 + 0], _vec_load_5[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_vec_load_5[0 + 2], _vec_load_5[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_vec_load_5[0 + 4], _vec_load_5[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_vec_load_5[0 + 6], _vec_load_5[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(activation_bf16_ptr + (row_index_4 * 3072 + column_4)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                }
                asm volatile("barrier.sync 11, 384;" ::: "memory");
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            asm volatile("fence.release.gpu;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            int scheduler_present = 0;
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(return_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(return_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            if (bid == 0 && warp == 0) {
                if (elect_sync()) {
                    // nvlink_barrier(mega_pg_flags) phase=1
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __slot = 1;
                        unsigned* __local_flag = mega_pg_flags[__me] + __slot;
                        unsigned __old_sense = 0u;
                        const unsigned __delta = (__me == 0) ? (0x80000000u - (unsigned)(__ws - 1)) : 1u;
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        for (int __r = 0; __r < __ws; ++__r) {
                            unsigned* __peer_flag = mega_pg_flags[__r] + __slot;
                            unsigned __old_peer;
                            asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                                : "=r"(__old_peer) : "l"(__peer_flag), "r"(__delta) : "memory");
                            if (__r == __me) __old_sense = __old_peer;
                        }
                        asm volatile("fence.proxy.alias;" ::: "memory");
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                            if (((__old_sense ^ __v) & 0x80000000u) != 0u) break;
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(return_visible) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(return_visible) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(launch_epoch + 1)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
        }
    }
    // ---- Role: transform ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 80;");
        { // transform_main
            int linear_thread_5 = bid * 384 + tid;
            int thread_stride_5 = num_bids * 384;
            #pragma unroll 1
            for (int index_10 = linear_thread_5; index_10 < 2048; index_10 += thread_stride_5) {
                route_map_i32[index_10] = -1;
                route_scale_f32[index_10] = 0.0f;
            }
            #pragma unroll 1
            for (int expert_9 = linear_thread_5; expert_9 < 32; expert_9 += thread_stride_5) {
                route_counts_u32[expert_9] = 0;
            }
            #pragma unroll 1
            for (int expert_row_tile_7 = linear_thread_5; expert_row_tile_7 < 96; expert_row_tile_7 += thread_stride_5) {
                {
                    unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(fc1_done) + (expert_row_tile_7);
                    asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
                }
            }
            #pragma unroll 1
            for (int index_11 = linear_thread_5 * 8; index_11 < tokens_per_rank * 3072; index_11 += thread_stride_5 * 8) {
                float _vec_load_6[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(source_hidden_bf16 + index_11 + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(_vec_load_6[0 + 0], _vec_load_6[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(_vec_load_6[0 + 2], _vec_load_6[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(_vec_load_6[0 + 4], _vec_load_6[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(_vec_load_6[0 + 6], _vec_load_6[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(published_hidden_bf16) + index_11))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
            #pragma unroll 1
            for (int route_5 = linear_thread_5; route_5 < tokens_per_rank * 8; route_5 += thread_stride_5) {
                int route_id_5 = (int)source_topk_ids_i64[route_5];
                float route_weight_5 = source_topk_weights_f32[route_5];
                *(reinterpret_cast<int*>(reinterpret_cast<int*>(published_topk_ids_i32) + route_5) + (0)) = route_id_5;
                *(reinterpret_cast<float*>(reinterpret_cast<float*>(published_topk_weights_f32) + route_5) + (0)) = route_weight_5;
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(publication_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(publication_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            if (bid == 0 && warp == 0) {
                if (elect_sync()) {
                    // nvlink_barrier(mega_pg_flags) phase=0
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __slot = 0;
                        unsigned* __local_flag = mega_pg_flags[__me] + __slot;
                        unsigned __old_sense = 0u;
                        const unsigned __delta = (__me == 0) ? (0x80000000u - (unsigned)(__ws - 1)) : 1u;
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        for (int __r = 0; __r < __ws; ++__r) {
                            unsigned* __peer_flag = mega_pg_flags[__r] + __slot;
                            unsigned __old_peer;
                            asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                                : "=r"(__old_peer) : "l"(__peer_flag), "r"(__delta) : "memory");
                            if (__r == __me) __old_sense = __old_peer;
                        }
                        asm volatile("fence.proxy.alias;" ::: "memory");
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                            if (((__old_sense ^ __v) & 0x80000000u) != 0u) break;
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(publication_visible) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(publication_visible) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(launch_epoch + 1)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            int source_rank_5 = bid % 16;
            int source_token_begin_5 = bid / 16;
            int* remote_ids_5 = reinterpret_cast<int*>(published_topk_ids_i32_peers[source_rank_5]);
            float* remote_weights_5 = reinterpret_cast<float*>(published_topk_weights_f32_peers[source_rank_5]);
            __nv_bfloat16* remote_hidden_5 = reinterpret_cast<__nv_bfloat16*>(published_hidden_bf16_peers[source_rank_5]);
            #pragma unroll 1
            for (int source_token_5 = source_token_begin_5; source_token_5 < tokens_per_rank; source_token_5 += 9) {
                int routes_per_source_7 = tokens_per_rank * 8;
                int source_route_lane_5 = source_token_5 * 8 + lane;
                int row_index_lane_5 = -1;
                int row_lane_5 = 0;
                if (warp == 0 && lane < 8) {
                    int global_expert_5 = remote_ids_5[source_route_lane_5];
                    int owner_5 = global_expert_5 / 32;
                    if (owner_5 == mega_pg_rank) {
                        int local_expert_lane_5 = global_expert_5 - mega_pg_rank * 32;
                        unsigned int _atomic_old_3 = atomicAdd(&route_counts_u32[local_expert_lane_5], 1);
                        row_lane_5 = (int)_atomic_old_3;
                        if (row_lane_5 < 64) {
                            row_index_lane_5 = local_expert_lane_5 * 64 + row_lane_5;
                            int global_route_lane_5 = source_rank_5 * routes_per_source_7 + source_route_lane_5;
                            route_map_i32[row_index_lane_5] = global_route_lane_5;
                            route_scale_f32[row_index_lane_5] = remote_weights_5[source_route_lane_5];
                        }
                    }
                    dispatch_rows_i32[lane] = row_index_lane_5;
                }
                if (warp == 0) {
                    int _vote_3 = __any_sync(0xFFFFFFFF, row_index_lane_5 >= 0);
                    int any_owned_5 = _vote_3;
                    if (lane == 0) {
                        dispatch_rows_i32[8] = any_owned_5;
                    }
                }
                asm volatile("barrier.sync 11, 384;" ::: "memory");
                int dispatch_any_5 = dispatch_rows_i32[8];
                if (dispatch_any_5 != 0) {
                    int column_5 = tid * 8;
                    float _vec_load_7[8];
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(remote_hidden_5 + (source_token_5 * 3072 + column_5) + 0);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int route_slot_5 = 0; route_slot_5 < 8; route_slot_5++) {
                        int row_index_5 = dispatch_rows_i32[route_slot_5];
                        if (row_index_5 >= 0) {
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(_vec_load_7[0 + 0], _vec_load_7[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_vec_load_7[0 + 2], _vec_load_7[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_vec_load_7[0 + 4], _vec_load_7[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_vec_load_7[0 + 6], _vec_load_7[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(activation_bf16_ptr + (row_index_5 * 3072 + column_5)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                }
                asm volatile("barrier.sync 11, 384;" ::: "memory");
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            asm volatile("fence.release.gpu;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 12, 384;" ::: "memory");
            int row = (warp - 8) * 32 + lane;
            int local_scale_word = row % 32 * 4 + row / 32;
            unsigned int a_stage_1 = 0;
            unsigned int scale_stage_1 = 0;
            unsigned int transformed_stage_1 = 0;
            unsigned int scale_reg[1];
            unsigned int raw_fragment[16];
            unsigned int scale_fragment[8];
            unsigned int products[64];
            int tmem_row_base = (warp - 8) * 32 << 16;
            unsigned int _phase_scale_full = 0;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_transformed_free = 1;
            #pragma unroll
            for (int phase_4 = 0; phase_4 < 2; phase_4++) {
                int work_begin_4 = ((phase_4 == 0) ? (int)cluster_id : ((int)cluster_id + 16) % 72);
                int work_end_4 = ((phase_4 == 0) ? 1280 : 384);
                int m_tiles_per_expert_4 = ((phase_4 == 0) ? 40 : 12);
                #pragma unroll 1
                for (int work_4 = work_begin_4; work_4 < work_end_4; work_4 += (int)num_clusters) {
                    int expert_10 = work_4 / m_tiles_per_expert_4;
                    int weight_m_tile_4 = work_4 - expert_10 * m_tiles_per_expert_4;
                    int num_k_tiles_3 = ((phase_4 == 0) ? 24 : 40);
                    #pragma unroll 1
                    for (int _iter_k = 0; _iter_k < num_k_tiles_3; _iter_k++) {
                        mbarrier_wait(scale_full_addr + (scale_stage_1) * 8, _phase_scale_full);
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_reg[0])) : "r"(scale_smem_addr + scale_stage_1 * 512 + (unsigned int)(local_scale_word * 4)));
                        mbarrier_wait(a_full_addr + (a_stage_1) * 8, _phase_a_full);
                        mbarrier_wait(transformed_free_addr + (transformed_stage_1) * 8, _phase_transformed_free);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(0) + 3]))
                            : "r"(raw_smem_addr + a_stage_1 * 16384 + (unsigned int)(row * 128) + (unsigned int)(0 ^ (row & 7) << 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(4) + 3]))
                            : "r"(raw_smem_addr + a_stage_1 * 16384 + (unsigned int)(row * 128) + (unsigned int)(16 ^ (row & 7) << 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(8) + 3]))
                            : "r"(raw_smem_addr + a_stage_1 * 16384 + (unsigned int)(row * 128) + (unsigned int)(32 ^ (row & 7) << 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(12) + 3]))
                            : "r"(raw_smem_addr + a_stage_1 * 16384 + (unsigned int)(row * 128) + (unsigned int)(48 ^ (row & 7) << 4)));
                        uint32_t _prmt_b32_0;
                        asm("prmt.b32 %0, %1, %2, 0x0000;" : "=r"(_prmt_b32_0) : "r"(scale_reg[0]), "r"(scale_reg[0]));
                        uint32_t _bf16x2_from_ue8m0x2_0;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_0) : "h"((uint16_t)(_prmt_b32_0 & 65535)));
                        scale_fragment[0] = _bf16x2_from_ue8m0x2_0;
                        uint32_t _bf16x2_from_ue8m0x2_1;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_1) : "h"((uint16_t)(_prmt_b32_0 >> 16)));
                        scale_fragment[1] = _bf16x2_from_ue8m0x2_1;
                        uint32_t _bf16x2_from_ue8m0x2_2;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_2) : "h"((uint16_t)(_prmt_b32_0 & 65535)));
                        scale_fragment[2] = _bf16x2_from_ue8m0x2_2;
                        uint32_t _bf16x2_from_ue8m0x2_3;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_3) : "h"((uint16_t)(_prmt_b32_0 >> 16)));
                        scale_fragment[3] = _bf16x2_from_ue8m0x2_3;
                        uint32_t _prmt_b32_1;
                        asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(_prmt_b32_1) : "r"(scale_reg[0]), "r"(scale_reg[0]));
                        uint32_t _bf16x2_from_ue8m0x2_4;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_4) : "h"((uint16_t)(_prmt_b32_1 & 65535)));
                        scale_fragment[4] = _bf16x2_from_ue8m0x2_4;
                        uint32_t _bf16x2_from_ue8m0x2_5;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_5) : "h"((uint16_t)(_prmt_b32_1 >> 16)));
                        scale_fragment[5] = _bf16x2_from_ue8m0x2_5;
                        uint32_t _bf16x2_from_ue8m0x2_6;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_6) : "h"((uint16_t)(_prmt_b32_1 & 65535)));
                        scale_fragment[6] = _bf16x2_from_ue8m0x2_6;
                        uint32_t _bf16x2_from_ue8m0x2_7;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_7) : "h"((uint16_t)(_prmt_b32_1 >> 16)));
                        scale_fragment[7] = _bf16x2_from_ue8m0x2_7;
                        uint32_t _bf16x2_from_e4m3x2_0;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_0) : "h"((uint16_t)(raw_fragment[0] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_1;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_1) : "h"((uint16_t)(raw_fragment[0] >> 16)));
                        uint32_t _bf16x2_mul_0;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(_bf16x2_from_e4m3x2_0), "r"(scale_fragment[0]));
                        products[0] = _bf16x2_mul_0;
                        uint32_t _bf16x2_mul_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(_bf16x2_from_e4m3x2_1), "r"(scale_fragment[0]));
                        products[1] = _bf16x2_mul_1;
                        uint32_t _bf16x2_from_e4m3x2_2;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_2) : "h"((uint16_t)(raw_fragment[1] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_3;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_3) : "h"((uint16_t)(raw_fragment[1] >> 16)));
                        uint32_t _bf16x2_mul_2;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(_bf16x2_from_e4m3x2_2), "r"(scale_fragment[0]));
                        products[2] = _bf16x2_mul_2;
                        uint32_t _bf16x2_mul_3;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(_bf16x2_from_e4m3x2_3), "r"(scale_fragment[0]));
                        products[3] = _bf16x2_mul_3;
                        uint32_t _bf16x2_from_e4m3x2_4;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_4) : "h"((uint16_t)(raw_fragment[2] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_5;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_5) : "h"((uint16_t)(raw_fragment[2] >> 16)));
                        uint32_t _bf16x2_mul_4;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(_bf16x2_from_e4m3x2_4), "r"(scale_fragment[1]));
                        products[4] = _bf16x2_mul_4;
                        uint32_t _bf16x2_mul_5;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(_bf16x2_from_e4m3x2_5), "r"(scale_fragment[1]));
                        products[5] = _bf16x2_mul_5;
                        uint32_t _bf16x2_from_e4m3x2_6;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_6) : "h"((uint16_t)(raw_fragment[3] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_7;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_7) : "h"((uint16_t)(raw_fragment[3] >> 16)));
                        uint32_t _bf16x2_mul_6;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(_bf16x2_from_e4m3x2_6), "r"(scale_fragment[1]));
                        products[6] = _bf16x2_mul_6;
                        uint32_t _bf16x2_mul_7;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(_bf16x2_from_e4m3x2_7), "r"(scale_fragment[1]));
                        products[7] = _bf16x2_mul_7;
                        uint32_t _bf16x2_from_e4m3x2_8;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_8) : "h"((uint16_t)(raw_fragment[4] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_9;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_9) : "h"((uint16_t)(raw_fragment[4] >> 16)));
                        uint32_t _bf16x2_mul_8;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(_bf16x2_from_e4m3x2_8), "r"(scale_fragment[2]));
                        products[8] = _bf16x2_mul_8;
                        uint32_t _bf16x2_mul_9;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(_bf16x2_from_e4m3x2_9), "r"(scale_fragment[2]));
                        products[9] = _bf16x2_mul_9;
                        uint32_t _bf16x2_from_e4m3x2_10;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_10) : "h"((uint16_t)(raw_fragment[5] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_11;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_11) : "h"((uint16_t)(raw_fragment[5] >> 16)));
                        uint32_t _bf16x2_mul_10;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(_bf16x2_from_e4m3x2_10), "r"(scale_fragment[2]));
                        products[10] = _bf16x2_mul_10;
                        uint32_t _bf16x2_mul_11;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(_bf16x2_from_e4m3x2_11), "r"(scale_fragment[2]));
                        products[11] = _bf16x2_mul_11;
                        uint32_t _bf16x2_from_e4m3x2_12;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_12) : "h"((uint16_t)(raw_fragment[6] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_13;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_13) : "h"((uint16_t)(raw_fragment[6] >> 16)));
                        uint32_t _bf16x2_mul_12;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(_bf16x2_from_e4m3x2_12), "r"(scale_fragment[3]));
                        products[12] = _bf16x2_mul_12;
                        uint32_t _bf16x2_mul_13;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(_bf16x2_from_e4m3x2_13), "r"(scale_fragment[3]));
                        products[13] = _bf16x2_mul_13;
                        uint32_t _bf16x2_from_e4m3x2_14;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_14) : "h"((uint16_t)(raw_fragment[7] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_15;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_15) : "h"((uint16_t)(raw_fragment[7] >> 16)));
                        uint32_t _bf16x2_mul_14;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(_bf16x2_from_e4m3x2_14), "r"(scale_fragment[3]));
                        products[14] = _bf16x2_mul_14;
                        uint32_t _bf16x2_mul_15;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(_bf16x2_from_e4m3x2_15), "r"(scale_fragment[3]));
                        products[15] = _bf16x2_mul_15;
                        uint32_t _bf16x2_from_e4m3x2_16;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_16) : "h"((uint16_t)(raw_fragment[8] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_17;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_17) : "h"((uint16_t)(raw_fragment[8] >> 16)));
                        uint32_t _bf16x2_mul_16;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(_bf16x2_from_e4m3x2_16), "r"(scale_fragment[4]));
                        products[16] = _bf16x2_mul_16;
                        uint32_t _bf16x2_mul_17;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(_bf16x2_from_e4m3x2_17), "r"(scale_fragment[4]));
                        products[17] = _bf16x2_mul_17;
                        uint32_t _bf16x2_from_e4m3x2_18;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_18) : "h"((uint16_t)(raw_fragment[9] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_19;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_19) : "h"((uint16_t)(raw_fragment[9] >> 16)));
                        uint32_t _bf16x2_mul_18;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(_bf16x2_from_e4m3x2_18), "r"(scale_fragment[4]));
                        products[18] = _bf16x2_mul_18;
                        uint32_t _bf16x2_mul_19;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(_bf16x2_from_e4m3x2_19), "r"(scale_fragment[4]));
                        products[19] = _bf16x2_mul_19;
                        uint32_t _bf16x2_from_e4m3x2_20;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_20) : "h"((uint16_t)(raw_fragment[10] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_21;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_21) : "h"((uint16_t)(raw_fragment[10] >> 16)));
                        uint32_t _bf16x2_mul_20;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(_bf16x2_from_e4m3x2_20), "r"(scale_fragment[5]));
                        products[20] = _bf16x2_mul_20;
                        uint32_t _bf16x2_mul_21;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(_bf16x2_from_e4m3x2_21), "r"(scale_fragment[5]));
                        products[21] = _bf16x2_mul_21;
                        uint32_t _bf16x2_from_e4m3x2_22;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_22) : "h"((uint16_t)(raw_fragment[11] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_23;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_23) : "h"((uint16_t)(raw_fragment[11] >> 16)));
                        uint32_t _bf16x2_mul_22;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(_bf16x2_from_e4m3x2_22), "r"(scale_fragment[5]));
                        products[22] = _bf16x2_mul_22;
                        uint32_t _bf16x2_mul_23;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(_bf16x2_from_e4m3x2_23), "r"(scale_fragment[5]));
                        products[23] = _bf16x2_mul_23;
                        uint32_t _bf16x2_from_e4m3x2_24;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_24) : "h"((uint16_t)(raw_fragment[12] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_25;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_25) : "h"((uint16_t)(raw_fragment[12] >> 16)));
                        uint32_t _bf16x2_mul_24;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(_bf16x2_from_e4m3x2_24), "r"(scale_fragment[6]));
                        products[24] = _bf16x2_mul_24;
                        uint32_t _bf16x2_mul_25;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(_bf16x2_from_e4m3x2_25), "r"(scale_fragment[6]));
                        products[25] = _bf16x2_mul_25;
                        uint32_t _bf16x2_from_e4m3x2_26;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_26) : "h"((uint16_t)(raw_fragment[13] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_27;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_27) : "h"((uint16_t)(raw_fragment[13] >> 16)));
                        uint32_t _bf16x2_mul_26;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(_bf16x2_from_e4m3x2_26), "r"(scale_fragment[6]));
                        products[26] = _bf16x2_mul_26;
                        uint32_t _bf16x2_mul_27;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(_bf16x2_from_e4m3x2_27), "r"(scale_fragment[6]));
                        products[27] = _bf16x2_mul_27;
                        uint32_t _bf16x2_from_e4m3x2_28;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_28) : "h"((uint16_t)(raw_fragment[14] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_29;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_29) : "h"((uint16_t)(raw_fragment[14] >> 16)));
                        uint32_t _bf16x2_mul_28;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(_bf16x2_from_e4m3x2_28), "r"(scale_fragment[7]));
                        products[28] = _bf16x2_mul_28;
                        uint32_t _bf16x2_mul_29;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(_bf16x2_from_e4m3x2_29), "r"(scale_fragment[7]));
                        products[29] = _bf16x2_mul_29;
                        uint32_t _bf16x2_from_e4m3x2_30;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_30) : "h"((uint16_t)(raw_fragment[15] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_31;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_31) : "h"((uint16_t)(raw_fragment[15] >> 16)));
                        uint32_t _bf16x2_mul_30;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(_bf16x2_from_e4m3x2_30), "r"(scale_fragment[7]));
                        products[30] = _bf16x2_mul_30;
                        uint32_t _bf16x2_mul_31;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(_bf16x2_from_e4m3x2_31), "r"(scale_fragment[7]));
                        products[31] = _bf16x2_mul_31;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(0) + 3]))
                            : "r"(raw_smem_addr + a_stage_1 * 16384 + (unsigned int)(row * 128) + (unsigned int)(64 ^ (row & 7) << 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(4) + 3]))
                            : "r"(raw_smem_addr + a_stage_1 * 16384 + (unsigned int)(row * 128) + (unsigned int)(80 ^ (row & 7) << 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(8) + 3]))
                            : "r"(raw_smem_addr + a_stage_1 * 16384 + (unsigned int)(row * 128) + (unsigned int)(96 ^ (row & 7) << 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_fragment[(12) + 3]))
                            : "r"(raw_smem_addr + a_stage_1 * 16384 + (unsigned int)(row * 128) + (unsigned int)(112 ^ (row & 7) << 4)));
                        uint32_t _prmt_b32_2;
                        asm("prmt.b32 %0, %1, %2, 0x2222;" : "=r"(_prmt_b32_2) : "r"(scale_reg[0]), "r"(scale_reg[0]));
                        uint32_t _bf16x2_from_ue8m0x2_8;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_8) : "h"((uint16_t)(_prmt_b32_2 & 65535)));
                        scale_fragment[0] = _bf16x2_from_ue8m0x2_8;
                        uint32_t _bf16x2_from_ue8m0x2_9;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_9) : "h"((uint16_t)(_prmt_b32_2 >> 16)));
                        scale_fragment[1] = _bf16x2_from_ue8m0x2_9;
                        uint32_t _bf16x2_from_ue8m0x2_10;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_10) : "h"((uint16_t)(_prmt_b32_2 & 65535)));
                        scale_fragment[2] = _bf16x2_from_ue8m0x2_10;
                        uint32_t _bf16x2_from_ue8m0x2_11;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_11) : "h"((uint16_t)(_prmt_b32_2 >> 16)));
                        scale_fragment[3] = _bf16x2_from_ue8m0x2_11;
                        uint32_t _prmt_b32_3;
                        asm("prmt.b32 %0, %1, %2, 0x3333;" : "=r"(_prmt_b32_3) : "r"(scale_reg[0]), "r"(scale_reg[0]));
                        uint32_t _bf16x2_from_ue8m0x2_12;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_12) : "h"((uint16_t)(_prmt_b32_3 & 65535)));
                        scale_fragment[4] = _bf16x2_from_ue8m0x2_12;
                        uint32_t _bf16x2_from_ue8m0x2_13;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_13) : "h"((uint16_t)(_prmt_b32_3 >> 16)));
                        scale_fragment[5] = _bf16x2_from_ue8m0x2_13;
                        uint32_t _bf16x2_from_ue8m0x2_14;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_14) : "h"((uint16_t)(_prmt_b32_3 & 65535)));
                        scale_fragment[6] = _bf16x2_from_ue8m0x2_14;
                        uint32_t _bf16x2_from_ue8m0x2_15;
                        asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(_bf16x2_from_ue8m0x2_15) : "h"((uint16_t)(_prmt_b32_3 >> 16)));
                        scale_fragment[7] = _bf16x2_from_ue8m0x2_15;
                        uint32_t _bf16x2_from_e4m3x2_32;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_32) : "h"((uint16_t)(raw_fragment[0] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_33;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_33) : "h"((uint16_t)(raw_fragment[0] >> 16)));
                        uint32_t _bf16x2_mul_32;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(_bf16x2_from_e4m3x2_32), "r"(scale_fragment[0]));
                        products[32] = _bf16x2_mul_32;
                        uint32_t _bf16x2_mul_33;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(_bf16x2_from_e4m3x2_33), "r"(scale_fragment[0]));
                        products[33] = _bf16x2_mul_33;
                        uint32_t _bf16x2_from_e4m3x2_34;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_34) : "h"((uint16_t)(raw_fragment[1] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_35;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_35) : "h"((uint16_t)(raw_fragment[1] >> 16)));
                        uint32_t _bf16x2_mul_34;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(_bf16x2_from_e4m3x2_34), "r"(scale_fragment[0]));
                        products[34] = _bf16x2_mul_34;
                        uint32_t _bf16x2_mul_35;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(_bf16x2_from_e4m3x2_35), "r"(scale_fragment[0]));
                        products[35] = _bf16x2_mul_35;
                        uint32_t _bf16x2_from_e4m3x2_36;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_36) : "h"((uint16_t)(raw_fragment[2] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_37;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_37) : "h"((uint16_t)(raw_fragment[2] >> 16)));
                        uint32_t _bf16x2_mul_36;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(_bf16x2_from_e4m3x2_36), "r"(scale_fragment[1]));
                        products[36] = _bf16x2_mul_36;
                        uint32_t _bf16x2_mul_37;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(_bf16x2_from_e4m3x2_37), "r"(scale_fragment[1]));
                        products[37] = _bf16x2_mul_37;
                        uint32_t _bf16x2_from_e4m3x2_38;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_38) : "h"((uint16_t)(raw_fragment[3] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_39;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_39) : "h"((uint16_t)(raw_fragment[3] >> 16)));
                        uint32_t _bf16x2_mul_38;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(_bf16x2_from_e4m3x2_38), "r"(scale_fragment[1]));
                        products[38] = _bf16x2_mul_38;
                        uint32_t _bf16x2_mul_39;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(_bf16x2_from_e4m3x2_39), "r"(scale_fragment[1]));
                        products[39] = _bf16x2_mul_39;
                        uint32_t _bf16x2_from_e4m3x2_40;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_40) : "h"((uint16_t)(raw_fragment[4] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_41;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_41) : "h"((uint16_t)(raw_fragment[4] >> 16)));
                        uint32_t _bf16x2_mul_40;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(_bf16x2_from_e4m3x2_40), "r"(scale_fragment[2]));
                        products[40] = _bf16x2_mul_40;
                        uint32_t _bf16x2_mul_41;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(_bf16x2_from_e4m3x2_41), "r"(scale_fragment[2]));
                        products[41] = _bf16x2_mul_41;
                        uint32_t _bf16x2_from_e4m3x2_42;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_42) : "h"((uint16_t)(raw_fragment[5] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_43;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_43) : "h"((uint16_t)(raw_fragment[5] >> 16)));
                        uint32_t _bf16x2_mul_42;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(_bf16x2_from_e4m3x2_42), "r"(scale_fragment[2]));
                        products[42] = _bf16x2_mul_42;
                        uint32_t _bf16x2_mul_43;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(_bf16x2_from_e4m3x2_43), "r"(scale_fragment[2]));
                        products[43] = _bf16x2_mul_43;
                        uint32_t _bf16x2_from_e4m3x2_44;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_44) : "h"((uint16_t)(raw_fragment[6] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_45;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_45) : "h"((uint16_t)(raw_fragment[6] >> 16)));
                        uint32_t _bf16x2_mul_44;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(_bf16x2_from_e4m3x2_44), "r"(scale_fragment[3]));
                        products[44] = _bf16x2_mul_44;
                        uint32_t _bf16x2_mul_45;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(_bf16x2_from_e4m3x2_45), "r"(scale_fragment[3]));
                        products[45] = _bf16x2_mul_45;
                        uint32_t _bf16x2_from_e4m3x2_46;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_46) : "h"((uint16_t)(raw_fragment[7] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_47;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_47) : "h"((uint16_t)(raw_fragment[7] >> 16)));
                        uint32_t _bf16x2_mul_46;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(_bf16x2_from_e4m3x2_46), "r"(scale_fragment[3]));
                        products[46] = _bf16x2_mul_46;
                        uint32_t _bf16x2_mul_47;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(_bf16x2_from_e4m3x2_47), "r"(scale_fragment[3]));
                        products[47] = _bf16x2_mul_47;
                        uint32_t _bf16x2_from_e4m3x2_48;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_48) : "h"((uint16_t)(raw_fragment[8] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_49;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_49) : "h"((uint16_t)(raw_fragment[8] >> 16)));
                        uint32_t _bf16x2_mul_48;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(_bf16x2_from_e4m3x2_48), "r"(scale_fragment[4]));
                        products[48] = _bf16x2_mul_48;
                        uint32_t _bf16x2_mul_49;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(_bf16x2_from_e4m3x2_49), "r"(scale_fragment[4]));
                        products[49] = _bf16x2_mul_49;
                        uint32_t _bf16x2_from_e4m3x2_50;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_50) : "h"((uint16_t)(raw_fragment[9] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_51;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_51) : "h"((uint16_t)(raw_fragment[9] >> 16)));
                        uint32_t _bf16x2_mul_50;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(_bf16x2_from_e4m3x2_50), "r"(scale_fragment[4]));
                        products[50] = _bf16x2_mul_50;
                        uint32_t _bf16x2_mul_51;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(_bf16x2_from_e4m3x2_51), "r"(scale_fragment[4]));
                        products[51] = _bf16x2_mul_51;
                        uint32_t _bf16x2_from_e4m3x2_52;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_52) : "h"((uint16_t)(raw_fragment[10] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_53;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_53) : "h"((uint16_t)(raw_fragment[10] >> 16)));
                        uint32_t _bf16x2_mul_52;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(_bf16x2_from_e4m3x2_52), "r"(scale_fragment[5]));
                        products[52] = _bf16x2_mul_52;
                        uint32_t _bf16x2_mul_53;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(_bf16x2_from_e4m3x2_53), "r"(scale_fragment[5]));
                        products[53] = _bf16x2_mul_53;
                        uint32_t _bf16x2_from_e4m3x2_54;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_54) : "h"((uint16_t)(raw_fragment[11] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_55;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_55) : "h"((uint16_t)(raw_fragment[11] >> 16)));
                        uint32_t _bf16x2_mul_54;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(_bf16x2_from_e4m3x2_54), "r"(scale_fragment[5]));
                        products[54] = _bf16x2_mul_54;
                        uint32_t _bf16x2_mul_55;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(_bf16x2_from_e4m3x2_55), "r"(scale_fragment[5]));
                        products[55] = _bf16x2_mul_55;
                        uint32_t _bf16x2_from_e4m3x2_56;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_56) : "h"((uint16_t)(raw_fragment[12] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_57;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_57) : "h"((uint16_t)(raw_fragment[12] >> 16)));
                        uint32_t _bf16x2_mul_56;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(_bf16x2_from_e4m3x2_56), "r"(scale_fragment[6]));
                        products[56] = _bf16x2_mul_56;
                        uint32_t _bf16x2_mul_57;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(_bf16x2_from_e4m3x2_57), "r"(scale_fragment[6]));
                        products[57] = _bf16x2_mul_57;
                        uint32_t _bf16x2_from_e4m3x2_58;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_58) : "h"((uint16_t)(raw_fragment[13] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_59;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_59) : "h"((uint16_t)(raw_fragment[13] >> 16)));
                        uint32_t _bf16x2_mul_58;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(_bf16x2_from_e4m3x2_58), "r"(scale_fragment[6]));
                        products[58] = _bf16x2_mul_58;
                        uint32_t _bf16x2_mul_59;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(_bf16x2_from_e4m3x2_59), "r"(scale_fragment[6]));
                        products[59] = _bf16x2_mul_59;
                        uint32_t _bf16x2_from_e4m3x2_60;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_60) : "h"((uint16_t)(raw_fragment[14] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_61;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_61) : "h"((uint16_t)(raw_fragment[14] >> 16)));
                        uint32_t _bf16x2_mul_60;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(_bf16x2_from_e4m3x2_60), "r"(scale_fragment[7]));
                        products[60] = _bf16x2_mul_60;
                        uint32_t _bf16x2_mul_61;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(_bf16x2_from_e4m3x2_61), "r"(scale_fragment[7]));
                        products[61] = _bf16x2_mul_61;
                        uint32_t _bf16x2_from_e4m3x2_62;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_62) : "h"((uint16_t)(raw_fragment[15] & 65535)));
                        uint32_t _bf16x2_from_e4m3x2_63;
                        asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_bf16x2_from_e4m3x2_63) : "h"((uint16_t)(raw_fragment[15] >> 16)));
                        uint32_t _bf16x2_mul_62;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(_bf16x2_from_e4m3x2_62), "r"(scale_fragment[7]));
                        products[62] = _bf16x2_mul_62;
                        uint32_t _bf16x2_mul_63;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(_bf16x2_from_e4m3x2_63), "r"(scale_fragment[7]));
                        products[63] = _bf16x2_mul_63;
                        int tmem_base = taddr + 256 + (unsigned int)tmem_row_base + transformed_stage_1 * 64;
                        #pragma unroll
                        for (int chunk = 0; chunk < 8; chunk++) {
                            tmem_st_x8_u32(tmem_base + chunk * 8, (const uint32_t*)(products + chunk * 8));
                        }
                        asm volatile("barrier.sync 3, 128;" ::: "memory");
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        if (warp == 8) {
                            if (elect_sync()) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(transformed_full_addr + transformed_stage_1 * 8), "r"(0) : "memory");
                                mbarrier_arrive(a_free_addr + (a_stage_1) * 8);
                                mbarrier_arrive(scale_free_addr + (scale_stage_1) * 8);
                            }
                        }
                        a_stage_1 += 1;
                        if (a_stage_1 == 4) { a_stage_1 = 0; _phase_a_full ^= 1; }
                        scale_stage_1 += 1;
                        if (scale_stage_1 == 4) { scale_stage_1 = 0; _phase_scale_full ^= 1; }
                        transformed_stage_1 += 1;
                        if (transformed_stage_1 == 4) { transformed_stage_1 = 0; _phase_transformed_free ^= 1; }
                    }
                }
            }
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            asm volatile("fence.release.sys;" ::: "memory");
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(return_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(return_done) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)((launch_epoch + 1) * num_bids)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
            if (bid == 0 && warp == 0) {
                if (elect_sync()) {
                    // nvlink_barrier(mega_pg_flags) phase=1
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __slot = 1;
                        unsigned* __local_flag = mega_pg_flags[__me] + __slot;
                        unsigned __old_sense = 0u;
                        const unsigned __delta = (__me == 0) ? (0x80000000u - (unsigned)(__ws - 1)) : 1u;
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        for (int __r = 0; __r < __ws; ++__r) {
                            unsigned* __peer_flag = mega_pg_flags[__r] + __slot;
                            unsigned __old_peer;
                            asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                                : "=r"(__old_peer) : "l"(__peer_flag), "r"(__delta) : "memory");
                            if (__r == __me) __old_sense = __old_peer;
                        }
                        asm volatile("fence.proxy.alias;" ::: "memory");
                        while (true) {
                            unsigned __v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_flag) : "memory");
                            if (((__old_sense ^ __v) & 0x80000000u) != 0u) break;
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(return_visible) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(return_visible) + (0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(launch_epoch + 1)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 13, 384;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
