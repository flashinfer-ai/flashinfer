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
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 432
#define TMEM_SCORES_OFFSET 0
#define TMEM_OUTPUT_0_OFFSET 128
#define TMEM_OUTPUT_1_OFFSET 256
#define TMEM_TMEM_SFA_QK0_OFFSET 384
#define TMEM_TMEM_SFB_QK0_OFFSET 392
#define TMEM_TMEM_SFA_QK1_OFFSET 400
#define TMEM_TMEM_SFA_PV0_LO_OFFSET 408
#define TMEM_TMEM_SFA_PV0_HI_OFFSET 412
#define TMEM_TMEM_SFA_PV1_LO_OFFSET 416
#define TMEM_TMEM_SFA_PV1_HI_OFFSET 420
#define TMEM_TMEM_SFB_PV_LO_OFFSET 424
#define TMEM_TMEM_SFB_PV_HI_OFFSET 428
#define NUM_V_PIPE_STAGES 5
#define NUM_K_PIPE_STAGES 4
#define SMEM_ROW_STATE_OFF 1024
#define SMEM_ROW_STATE_STAGE_BYTES 2048
#define SMEM_ROW_STATE_STRIDE 2048
#define SMEM_SMEM_Q_OFF 3072
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_SFQ_OFF 19456
#define SMEM_SMEM_SFQ_STAGE_BYTES 1024
#define SMEM_SMEM_SFQ_STRIDE 1024
#define SMEM_SMEM_K_OFF 21504
#define SMEM_SMEM_K_STAGE_BYTES 4096
#define SMEM_SMEM_K_STRIDE 4096
#define SMEM_SMEM_SFK_OFF 37888
#define SMEM_SMEM_SFK_STAGE_BYTES 1024
#define SMEM_SMEM_SFK_STRIDE 1024
#define SMEM_SMEM_VT_OFF 41984
#define SMEM_SMEM_VT_STAGE_BYTES 4096
#define SMEM_SMEM_VT_STRIDE 4096
#define SMEM_SMEM_SFVT_LO_OFF 62464
#define SMEM_SMEM_SFVT_LO_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_LO_STRIDE 1024
#define SMEM_SMEM_SFVT_HI_OFF 62976
#define SMEM_SMEM_SFVT_HI_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_HI_STRIDE 1024
#define SMEM_SMEM_P_OFF 67584
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_TOTAL 83968
#define THREADS 512
#define USE_TMEM_LD_RED 1
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128

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

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
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

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
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


__device__ __forceinline__ void tmem_ld_x32(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
        : "r"(tmem_addr));
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}


__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}


__device__ __forceinline__ float2 ex2_emulation_f32x2_value(float2 value) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(value.x, -127.0f), x1 = max_noftz(value.y, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    return make_float2(r0, r1);
}

__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    float2 result = ex2_emulation_f32x2_value(make_float2(*x0_ptr, *x1_ptr));
    *x0_ptr = result.x; *x1_ptr = result.y;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
    }
}



__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void fma_f32x2_noftz_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("add.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("sub.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.f32x2 %0, %1, %2;"
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

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
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
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 add_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

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


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo128(int addr) {
    const int SBO = 128;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo256(int addr) {
    const int SBO = 256;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo256(int lo) {
    const int SBO = 256;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,1,1) void
kernel_cake_minimax_h3_varlen_attention_39a37bc532bd45f66386(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, const __grid_constant__ CUtensorMap SFQ, const __grid_constant__ CUtensorMap SFK, const __grid_constant__ CUtensorMap SFVtLo, const __grid_constant__ CUtensorMap SFVtHi, __nv_bfloat16* __restrict__ O, int* __restrict__ cl_head, int* __restrict__ cl_seg_begin, int* __restrict__ cl_seg_len, int* __restrict__ cl_kv_base, int* __restrict__ cl_q_block, int* __restrict__ cl_kv_begin, int* __restrict__ cl_kv_blocks, int* __restrict__ cl_ws_slot, __half* __restrict__ partial_O, float* __restrict__ partial_ML, unsigned int num_tiles, int total_clusters, int heads, int PB, float softmax_scale_log2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define k_full_addr (mbar_base + 24)
    #define k_empty_addr (mbar_base + 56)
    #define v_full_addr (mbar_base + 88)
    #define v_empty_addr (mbar_base + 128)
    #define s_full_addr (mbar_base + 168)
    #define s_empty_addr (mbar_base + 184)
    #define p_full_addr (mbar_base + 200)
    #define p_full_2_addr (mbar_base + 216)
    #define p_empty_addr (mbar_base + 232)
    #define corr_sig_lo_addr (mbar_base + 248)
    #define corr_sig_hi_addr (mbar_base + 264)
    #define corr_done_addr (mbar_base + 280)
    #define o_full_addr (mbar_base + 296)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* row_state = reinterpret_cast<float*>(smem_raw + 1024);
    const int row_state_addr = smem + 1024;
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_q_addr = smem + 3072;
    uint8_t* smem_sfq = reinterpret_cast<uint8_t*>(smem_raw + 19456);
    const int smem_sfq_addr = smem + 19456;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 21504);
    const int smem_k_addr = smem + 21504;
    uint8_t* smem_sfk = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_sfk_addr = smem + 37888;
    uint8_t* smem_vt = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int smem_vt_addr = smem + 41984;
    uint8_t* smem_sfvt_lo = reinterpret_cast<uint8_t*>(smem_raw + 62464);
    const int smem_sfvt_lo_addr = smem + 62464;
    uint8_t* smem_sfvt_hi = reinterpret_cast<uint8_t*>(smem_raw + 62976);
    const int smem_sfvt_hi_addr = smem + 62976;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_p_addr = smem + 67584;

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 39 barriers)
    // Mbarriers at smem_raw[0..312)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 2 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 4 barriers, init_count=2
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // k_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 5 barriers, init_count=2
            mbarrier_init(smem + 88, 2);
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            mbarrier_init(smem + 120, 2);
            // v_empty: 5 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            // s_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 184, 256);
            mbarrier_init(smem + 192, 256);
            // p_full: 2 barriers, init_count=512
            mbarrier_init(smem + 200, 512);
            mbarrier_init(smem + 208, 512);
            // p_full_2: 2 barriers, init_count=256
            mbarrier_init(smem + 216, 256);
            mbarrier_init(smem + 224, 256);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            // corr_sig_lo: 2 barriers, init_count=64
            mbarrier_init(smem + 248, 64);
            mbarrier_init(smem + 256, 64);
            // corr_sig_hi: 2 barriers, init_count=64
            mbarrier_init(smem + 264, 64);
            mbarrier_init(smem + 272, 64);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 280, 128);
            mbarrier_init(smem + 288, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 432 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 312);
    if (warp == 0) {
        int _tmem_hold = smem + 312;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_output_0 = taddr + 128;
    const int tmem_output_1 = taddr + 256;
    const int tmem_tmem_sfa_qk0 = taddr + 384;
    const int tmem_tmem_sfb_qk0 = taddr + 392;
    const int tmem_tmem_sfa_qk1 = taddr + 400;
    const int tmem_tmem_sfa_pv0_lo = taddr + 408;
    const int tmem_tmem_sfa_pv0_hi = taddr + 412;
    const int tmem_tmem_sfa_pv1_lo = taddr + 416;
    const int tmem_tmem_sfa_pv1_hi = taddr + 420;
    const int tmem_tmem_sfb_pv_lo = taddr + 424;
    const int tmem_tmem_sfb_pv_hi = taddr + 428;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // softmax_main
            unsigned int total_tiles = num_tiles;
            unsigned int stage = make_warp_uniform(warp / 4);
            int scale_off = make_warp_uniform(stage * 128);
            int p_col = make_warp_uniform(0);
            unsigned int _phase_s_full = 0;
            unsigned int _phase_p_empty = 1;
            unsigned int _phase_corr_done = 0;
            unsigned int _phase_o_full = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < total_tiles; tile_idx += num_clusters) {
                int head = cl_head[tile_idx];
                int seg_begin = cl_seg_begin[tile_idx];
                int seg_len = cl_seg_len[tile_idx];
                int kv_base = cl_kv_base[tile_idx];
                int m_block = cl_q_block[tile_idx] + cta_rank * 2;
                unsigned int num_n_blocks = cl_kv_blocks[tile_idx];
                int kv_begin = cl_kv_begin[tile_idx];
                int ws_slot = cl_ws_slot[tile_idx];
                int tail_base = seg_len - kv_begin * 128;
                int kv_len = num_n_blocks * 128;
                unsigned int n_count = (unsigned int)((kv_len + 128 - 1) / 128);
                float row_max = -CAKE_INF;
                float row_max_scaled = 0.0f;
                float row_sum = 0.0f;
                #pragma unroll 1
                for (unsigned int n_iter = 0; n_iter < n_count; n_iter++) {
                    int n_block = n_count - 1 - n_iter;
                    mbarrier_wait_hint(s_full_addr + (stage) * 8, _phase_s_full, 10000000);
                    _phase_s_full ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_addr = taddr + (unsigned int)TMEM_SCORES_OFFSET + (unsigned int)(warp % 4 * 32 << 16);
                    float sv[128];
                    float tile_max = -CAKE_INF;
                    {
                        float lo_max = -CAKE_INF;
                        float hi_max = -CAKE_INF;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                            : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(lo_max)
                            : "r"(tmem_scores + (warp % 4 * 32 << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                            : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95]), "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127]), "=f"(hi_max)
                            : "r"(tmem_scores + 64 + (warp % 4 * 32 << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float _max_0 = max_noftz(lo_max, hi_max);
                        tile_max = _max_0;
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((s_empty_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    int tail_valid = tail_base - n_block * 128;
                    if (tail_valid < 128) {
                        uint32_t _slice_lo_mask_0;
                        {
                            int _lim_0 = tail_valid;
                            if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                            else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                            }
                        }
                        if (!(_slice_lo_mask_0 & (1u << 0))) sv[0] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 1))) sv[1] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 2))) sv[2] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 3))) sv[3] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 4))) sv[4] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 5))) sv[5] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 6))) sv[6] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 7))) sv[7] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 8))) sv[8] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 9))) sv[9] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 10))) sv[10] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 11))) sv[11] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 12))) sv[12] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 13))) sv[13] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 14))) sv[14] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 15))) sv[15] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 16))) sv[16] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 17))) sv[17] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 18))) sv[18] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 19))) sv[19] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 20))) sv[20] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 21))) sv[21] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 22))) sv[22] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 23))) sv[23] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 24))) sv[24] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 25))) sv[25] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 26))) sv[26] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 27))) sv[27] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 28))) sv[28] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 29))) sv[29] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 30))) sv[30] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 31))) sv[31] = -CAKE_INF;
                        uint32_t _slice_lo_mask_1;
                        {
                            int _lim_1 = tail_valid - 32;
                            if (_lim_1 <= 0) { _slice_lo_mask_1 = 0u; }
                            else if (_lim_1 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_1));
                            }
                        }
                        if (!(_slice_lo_mask_1 & (1u << 0))) sv[32] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 1))) sv[33] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 2))) sv[34] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 3))) sv[35] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 4))) sv[36] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 5))) sv[37] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 6))) sv[38] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 7))) sv[39] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 8))) sv[40] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 9))) sv[41] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 10))) sv[42] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 11))) sv[43] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 12))) sv[44] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 13))) sv[45] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 14))) sv[46] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 15))) sv[47] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 16))) sv[48] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 17))) sv[49] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 18))) sv[50] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 19))) sv[51] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 20))) sv[52] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 21))) sv[53] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 22))) sv[54] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 23))) sv[55] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 24))) sv[56] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 25))) sv[57] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 26))) sv[58] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 27))) sv[59] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 28))) sv[60] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 29))) sv[61] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 30))) sv[62] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 31))) sv[63] = -CAKE_INF;
                        uint32_t _slice_lo_mask_2;
                        {
                            int _lim_2 = tail_valid - 64;
                            if (_lim_2 <= 0) { _slice_lo_mask_2 = 0u; }
                            else if (_lim_2 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_2));
                            }
                        }
                        if (!(_slice_lo_mask_2 & (1u << 0))) sv[64] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 1))) sv[65] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 2))) sv[66] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 3))) sv[67] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 4))) sv[68] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 5))) sv[69] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 6))) sv[70] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 7))) sv[71] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 8))) sv[72] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 9))) sv[73] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 10))) sv[74] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 11))) sv[75] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 12))) sv[76] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 13))) sv[77] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 14))) sv[78] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 15))) sv[79] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 16))) sv[80] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 17))) sv[81] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 18))) sv[82] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 19))) sv[83] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 20))) sv[84] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 21))) sv[85] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 22))) sv[86] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 23))) sv[87] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 24))) sv[88] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 25))) sv[89] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 26))) sv[90] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 27))) sv[91] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 28))) sv[92] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 29))) sv[93] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 30))) sv[94] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 31))) sv[95] = -CAKE_INF;
                        uint32_t _slice_lo_mask_3;
                        {
                            int _lim_3 = tail_valid - 96;
                            if (_lim_3 <= 0) { _slice_lo_mask_3 = 0u; }
                            else if (_lim_3 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_3));
                            }
                        }
                        if (!(_slice_lo_mask_3 & (1u << 0))) sv[96] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 1))) sv[97] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 2))) sv[98] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 3))) sv[99] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 4))) sv[100] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 5))) sv[101] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 6))) sv[102] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 7))) sv[103] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 8))) sv[104] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 9))) sv[105] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 10))) sv[106] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 11))) sv[107] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 12))) sv[108] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 13))) sv[109] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 14))) sv[110] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 15))) sv[111] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 16))) sv[112] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 17))) sv[113] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 18))) sv[114] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 19))) sv[115] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 20))) sv[116] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 21))) sv[117] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 22))) sv[118] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 23))) sv[119] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 24))) sv[120] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 25))) sv[121] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 26))) sv[122] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 27))) sv[123] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 28))) sv[124] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 29))) sv[125] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 30))) sv[126] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 31))) sv[127] = -CAKE_INF;
                        float2 _reg_reduce_max2_4 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&sv[0], _reg_reduce_max2_4);
                        row_max_x32_accum(&sv[32], _reg_reduce_max2_4);
                        row_max_x32_accum(&sv[64], _reg_reduce_max2_4);
                        row_max_x32_accum(&sv[96], _reg_reduce_max2_4);
                        float sv_max = row_max_reduce(_reg_reduce_max2_4);
                        tile_max = sv_max;
                    }
                    float _max_1 = max_noftz(tile_max, row_max);
                    float new_max = _max_1;
                    float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                    float new_max_scaled = safe_max * softmax_scale_log2;
                    float _fma_0 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                    float acc_scale_log2 = _fma_0;
                    float acc_scale;
                    if (acc_scale_log2 >= -8.0f) {
                        acc_scale = 1.0f;
                        new_max_scaled = row_max_scaled;
                    } else {
                        float _exp2_0 = approx_exp2(acc_scale_log2);
                        acc_scale = ((row_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                        row_max = new_max;
                        row_max_scaled = new_max_scaled;
                    }
                    row_state[warp % 4 * 32 + lane + scale_off] = acc_scale;
                    if (warp % 4 < 2) {
                        mbarrier_arrive(corr_sig_lo_addr + (stage) * 8);
                    } else {
                        mbarrier_arrive(corr_sig_hi_addr + (stage) * 8);
                    }
                    mbarrier_wait(p_empty_addr + (stage) * 8, _phase_p_empty);
                    _phase_p_empty ^= 1;
                    float block_sum = 0.0f;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float2 _f2_0 = make_float2(0.0f, 0.0f);
                    float2 block_sum2 = _f2_0;
                    float block_max = tile_max;
                    float block_max_scaled = ((block_max > -CAKE_INF) ? block_max * softmax_scale_log2 : 0.0f);
                    float _exp2_1 = approx_exp2(block_max_scaled - new_max_scaled - 2.584962500721156f);
                    float p_scale = ((block_max > -CAKE_INF) ? _exp2_1 : 0.0f);
                    uint16_t _e4m3x2_f32_0;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(p_scale), "f"(p_scale));
                    unsigned int sf_packed[1];
                    sf_packed[0] = (unsigned int)_e4m3x2_f32_0 * 65537;
                    const float2 _fma_b2_5 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_6 = {2.584962500721156f - block_max_scaled, 2.584962500721156f - block_max_scaled};
                    float2 _fma_pair_7 = fma_f32x2(make_float2(((sv + 0))[0], ((sv + 0))[1]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[0] = _fma_pair_7.x;
                    (sv + 0)[1] = _fma_pair_7.y;
                    float2 _fma_pair_8 = fma_f32x2(make_float2(((sv + 0))[2], ((sv + 0))[3]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[2] = _fma_pair_8.x;
                    (sv + 0)[3] = _fma_pair_8.y;
                    float2 _fma_pair_9 = fma_f32x2(make_float2(((sv + 0))[4], ((sv + 0))[5]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[4] = _fma_pair_9.x;
                    (sv + 0)[5] = _fma_pair_9.y;
                    float2 _fma_pair_10 = fma_f32x2(make_float2(((sv + 0))[6], ((sv + 0))[7]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[6] = _fma_pair_10.x;
                    (sv + 0)[7] = _fma_pair_10.y;
                    float2 _fma_pair_11 = fma_f32x2(make_float2(((sv + 0))[8], ((sv + 0))[9]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[8] = _fma_pair_11.x;
                    (sv + 0)[9] = _fma_pair_11.y;
                    float2 _fma_pair_12 = fma_f32x2(make_float2(((sv + 0))[10], ((sv + 0))[11]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[10] = _fma_pair_12.x;
                    (sv + 0)[11] = _fma_pair_12.y;
                    float2 _fma_pair_13 = fma_f32x2(make_float2(((sv + 0))[12], ((sv + 0))[13]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[12] = _fma_pair_13.x;
                    (sv + 0)[13] = _fma_pair_13.y;
                    float2 _fma_pair_14 = fma_f32x2(make_float2(((sv + 0))[14], ((sv + 0))[15]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[14] = _fma_pair_14.x;
                    (sv + 0)[15] = _fma_pair_14.y;
                    float2 _fma_pair_15 = fma_f32x2(make_float2(((sv + 0))[16], ((sv + 0))[17]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[16] = _fma_pair_15.x;
                    (sv + 0)[17] = _fma_pair_15.y;
                    float2 _fma_pair_16 = fma_f32x2(make_float2(((sv + 0))[18], ((sv + 0))[19]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[18] = _fma_pair_16.x;
                    (sv + 0)[19] = _fma_pair_16.y;
                    float2 _fma_pair_17 = fma_f32x2(make_float2(((sv + 0))[20], ((sv + 0))[21]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[20] = _fma_pair_17.x;
                    (sv + 0)[21] = _fma_pair_17.y;
                    float2 _fma_pair_18 = fma_f32x2(make_float2(((sv + 0))[22], ((sv + 0))[23]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[22] = _fma_pair_18.x;
                    (sv + 0)[23] = _fma_pair_18.y;
                    float2 _fma_pair_19 = fma_f32x2(make_float2(((sv + 0))[24], ((sv + 0))[25]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[24] = _fma_pair_19.x;
                    (sv + 0)[25] = _fma_pair_19.y;
                    float2 _fma_pair_20 = fma_f32x2(make_float2(((sv + 0))[26], ((sv + 0))[27]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[26] = _fma_pair_20.x;
                    (sv + 0)[27] = _fma_pair_20.y;
                    float2 _fma_pair_21 = fma_f32x2(make_float2(((sv + 0))[28], ((sv + 0))[29]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[28] = _fma_pair_21.x;
                    (sv + 0)[29] = _fma_pair_21.y;
                    float2 _fma_pair_22 = fma_f32x2(make_float2(((sv + 0))[30], ((sv + 0))[31]), _fma_b2_5, _fma_c2_6);
                    (sv + 0)[30] = _fma_pair_22.x;
                    (sv + 0)[31] = _fma_pair_22.y;
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        if (USE_TMEM_LD_RED == 0 && _le >= 12) {
                            float2 _exp2_pair_23 = ex2_emulation_f32x2_value(make_float2(sv[_le*2], sv[_le*2 + 1]));
                            sv[_le*2] = _exp2_pair_23.x;
                            sv[_le*2 + 1] = _exp2_pair_23.y;
                        } else {
                            sv[_le*2] = approx_exp2(sv[_le*2]);
                            sv[_le*2 + 1] = approx_exp2(sv[_le*2 + 1]);
                        }
                    }
                    float2 _f2_1 = make_float2(sv[0], sv[1]);
                    float2 partial = _f2_1;
                    #pragma unroll
                    for (int pair = 2; pair < 16; pair += 2) {
                        float2 _f2_2 = make_float2((sv + 0)[pair], (sv + 0)[pair + 1]);
                        partial = add_f32x2(partial, _f2_2);
                    }
                    float2 frag_sum2 = partial;
                    float2 _f2_3 = make_float2(p_scale, p_scale);
                    float2 scale2 = _f2_3;
                    block_sum2 = fma_f32x2_rn_ftz(frag_sum2, scale2, block_sum2);
                    float2 _f2_4 = make_float2(sv[16], sv[17]);
                    float2 partial_0 = _f2_4;
                    #pragma unroll
                    for (int pair_1 = 2; pair_1 < 16; pair_1 += 2) {
                        float2 _f2_5 = make_float2((sv + 16)[pair_1], (sv + 16)[pair_1 + 1]);
                        partial_0 = add_f32x2(partial_0, _f2_5);
                    }
                    frag_sum2 = partial_0;
                    block_sum2 = fma_f32x2_rn_ftz(frag_sum2, scale2, block_sum2);
                    uint32_t _fp4_0[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(sv[0]), "f"(sv[1]), "f"(sv[2]), "f"(sv[3]), "f"(sv[4]), "f"(sv[5]), "f"(sv[6]), "f"(sv[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[1]) : "f"(sv[8]), "f"(sv[9]), "f"(sv[10]), "f"(sv[11]), "f"(sv[12]), "f"(sv[13]), "f"(sv[14]), "f"(sv[15]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[2]) : "f"(sv[16]), "f"(sv[17]), "f"(sv[18]), "f"(sv[19]), "f"(sv[20]), "f"(sv[21]), "f"(sv[22]), "f"(sv[23]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[3]) : "f"(sv[24]), "f"(sv[25]), "f"(sv[26]), "f"(sv[27]), "f"(sv[28]), "f"(sv[29]), "f"(sv[30]), "f"(sv[31]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + stage * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 ^ ((warp % 4 * 32 + lane) * 64 >> 7 & 3) << 4))), "r"(_fp4_0[0]), "r"(_fp4_0[1]), "r"(_fp4_0[2]), "r"(_fp4_0[3]) : "memory");
                    const float2 _fma_b2_24 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_25 = {2.584962500721156f - block_max_scaled, 2.584962500721156f - block_max_scaled};
                    float2 _fma_pair_26 = fma_f32x2(make_float2(((sv + 32))[0], ((sv + 32))[1]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[0] = _fma_pair_26.x;
                    (sv + 32)[1] = _fma_pair_26.y;
                    float2 _fma_pair_27 = fma_f32x2(make_float2(((sv + 32))[2], ((sv + 32))[3]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[2] = _fma_pair_27.x;
                    (sv + 32)[3] = _fma_pair_27.y;
                    float2 _fma_pair_28 = fma_f32x2(make_float2(((sv + 32))[4], ((sv + 32))[5]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[4] = _fma_pair_28.x;
                    (sv + 32)[5] = _fma_pair_28.y;
                    float2 _fma_pair_29 = fma_f32x2(make_float2(((sv + 32))[6], ((sv + 32))[7]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[6] = _fma_pair_29.x;
                    (sv + 32)[7] = _fma_pair_29.y;
                    float2 _fma_pair_30 = fma_f32x2(make_float2(((sv + 32))[8], ((sv + 32))[9]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[8] = _fma_pair_30.x;
                    (sv + 32)[9] = _fma_pair_30.y;
                    float2 _fma_pair_31 = fma_f32x2(make_float2(((sv + 32))[10], ((sv + 32))[11]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[10] = _fma_pair_31.x;
                    (sv + 32)[11] = _fma_pair_31.y;
                    float2 _fma_pair_32 = fma_f32x2(make_float2(((sv + 32))[12], ((sv + 32))[13]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[12] = _fma_pair_32.x;
                    (sv + 32)[13] = _fma_pair_32.y;
                    float2 _fma_pair_33 = fma_f32x2(make_float2(((sv + 32))[14], ((sv + 32))[15]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[14] = _fma_pair_33.x;
                    (sv + 32)[15] = _fma_pair_33.y;
                    float2 _fma_pair_34 = fma_f32x2(make_float2(((sv + 32))[16], ((sv + 32))[17]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[16] = _fma_pair_34.x;
                    (sv + 32)[17] = _fma_pair_34.y;
                    float2 _fma_pair_35 = fma_f32x2(make_float2(((sv + 32))[18], ((sv + 32))[19]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[18] = _fma_pair_35.x;
                    (sv + 32)[19] = _fma_pair_35.y;
                    float2 _fma_pair_36 = fma_f32x2(make_float2(((sv + 32))[20], ((sv + 32))[21]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[20] = _fma_pair_36.x;
                    (sv + 32)[21] = _fma_pair_36.y;
                    float2 _fma_pair_37 = fma_f32x2(make_float2(((sv + 32))[22], ((sv + 32))[23]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[22] = _fma_pair_37.x;
                    (sv + 32)[23] = _fma_pair_37.y;
                    float2 _fma_pair_38 = fma_f32x2(make_float2(((sv + 32))[24], ((sv + 32))[25]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[24] = _fma_pair_38.x;
                    (sv + 32)[25] = _fma_pair_38.y;
                    float2 _fma_pair_39 = fma_f32x2(make_float2(((sv + 32))[26], ((sv + 32))[27]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[26] = _fma_pair_39.x;
                    (sv + 32)[27] = _fma_pair_39.y;
                    float2 _fma_pair_40 = fma_f32x2(make_float2(((sv + 32))[28], ((sv + 32))[29]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[28] = _fma_pair_40.x;
                    (sv + 32)[29] = _fma_pair_40.y;
                    float2 _fma_pair_41 = fma_f32x2(make_float2(((sv + 32))[30], ((sv + 32))[31]), _fma_b2_24, _fma_c2_25);
                    (sv + 32)[30] = _fma_pair_41.x;
                    (sv + 32)[31] = _fma_pair_41.y;
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        if (USE_TMEM_LD_RED == 0 && _le >= 12) {
                            float2 _exp2_pair_42 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 32], sv[_le*2 + 1 + 32]));
                            sv[_le*2 + 32] = _exp2_pair_42.x;
                            sv[_le*2 + 1 + 32] = _exp2_pair_42.y;
                        } else {
                            sv[_le*2 + 32] = approx_exp2(sv[_le*2 + 32]);
                            sv[_le*2 + 1 + 32] = approx_exp2(sv[_le*2 + 1 + 32]);
                        }
                    }
                    float2 _f2_6 = make_float2(sv[32], sv[33]);
                    float2 partial_1 = _f2_6;
                    #pragma unroll
                    for (int pair_2 = 2; pair_2 < 16; pair_2 += 2) {
                        float2 _f2_7 = make_float2((sv + 32)[pair_2], (sv + 32)[pair_2 + 1]);
                        partial_1 = add_f32x2(partial_1, _f2_7);
                    }
                    float2 frag_sum2_2 = partial_1;
                    float2 _f2_8 = make_float2(p_scale, p_scale);
                    float2 scale2_3 = _f2_8;
                    block_sum2 = fma_f32x2_rn_ftz(frag_sum2_2, scale2_3, block_sum2);
                    float2 _f2_9 = make_float2(sv[48], sv[49]);
                    float2 partial_4 = _f2_9;
                    #pragma unroll
                    for (int pair_3 = 2; pair_3 < 16; pair_3 += 2) {
                        float2 _f2_10 = make_float2((sv + 48)[pair_3], (sv + 48)[pair_3 + 1]);
                        partial_4 = add_f32x2(partial_4, _f2_10);
                    }
                    frag_sum2_2 = partial_4;
                    block_sum2 = fma_f32x2_rn_ftz(frag_sum2_2, scale2_3, block_sum2);
                    uint32_t _fp4_1[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[0]) : "f"(sv[32]), "f"(sv[33]), "f"(sv[34]), "f"(sv[35]), "f"(sv[36]), "f"(sv[37]), "f"(sv[38]), "f"(sv[39]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[1]) : "f"(sv[40]), "f"(sv[41]), "f"(sv[42]), "f"(sv[43]), "f"(sv[44]), "f"(sv[45]), "f"(sv[46]), "f"(sv[47]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[2]) : "f"(sv[48]), "f"(sv[49]), "f"(sv[50]), "f"(sv[51]), "f"(sv[52]), "f"(sv[53]), "f"(sv[54]), "f"(sv[55]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[3]) : "f"(sv[56]), "f"(sv[57]), "f"(sv[58]), "f"(sv[59]), "f"(sv[60]), "f"(sv[61]), "f"(sv[62]), "f"(sv[63]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + stage * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 16 ^ ((warp % 4 * 32 + lane) * 64 + 16 >> 7 & 3) << 4))), "r"(_fp4_1[0]), "r"(_fp4_1[1]), "r"(_fp4_1[2]), "r"(_fp4_1[3]) : "memory");
                    int pv_sfa_lo_col = (unsigned int)TMEM_TMEM_SFA_PV0_LO_OFFSET + stage * 8 + (unsigned int)(warp % 4);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)pv_sfa_lo_col), "r"(sf_packed[0]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    const float2 _fma_b2_43 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_44 = {2.584962500721156f - block_max_scaled, 2.584962500721156f - block_max_scaled};
                    float2 _fma_pair_45 = fma_f32x2(make_float2(((sv + 64))[0], ((sv + 64))[1]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[0] = _fma_pair_45.x;
                    (sv + 64)[1] = _fma_pair_45.y;
                    float2 _fma_pair_46 = fma_f32x2(make_float2(((sv + 64))[2], ((sv + 64))[3]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[2] = _fma_pair_46.x;
                    (sv + 64)[3] = _fma_pair_46.y;
                    float2 _fma_pair_47 = fma_f32x2(make_float2(((sv + 64))[4], ((sv + 64))[5]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[4] = _fma_pair_47.x;
                    (sv + 64)[5] = _fma_pair_47.y;
                    float2 _fma_pair_48 = fma_f32x2(make_float2(((sv + 64))[6], ((sv + 64))[7]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[6] = _fma_pair_48.x;
                    (sv + 64)[7] = _fma_pair_48.y;
                    float2 _fma_pair_49 = fma_f32x2(make_float2(((sv + 64))[8], ((sv + 64))[9]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[8] = _fma_pair_49.x;
                    (sv + 64)[9] = _fma_pair_49.y;
                    float2 _fma_pair_50 = fma_f32x2(make_float2(((sv + 64))[10], ((sv + 64))[11]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[10] = _fma_pair_50.x;
                    (sv + 64)[11] = _fma_pair_50.y;
                    float2 _fma_pair_51 = fma_f32x2(make_float2(((sv + 64))[12], ((sv + 64))[13]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[12] = _fma_pair_51.x;
                    (sv + 64)[13] = _fma_pair_51.y;
                    float2 _fma_pair_52 = fma_f32x2(make_float2(((sv + 64))[14], ((sv + 64))[15]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[14] = _fma_pair_52.x;
                    (sv + 64)[15] = _fma_pair_52.y;
                    float2 _fma_pair_53 = fma_f32x2(make_float2(((sv + 64))[16], ((sv + 64))[17]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[16] = _fma_pair_53.x;
                    (sv + 64)[17] = _fma_pair_53.y;
                    float2 _fma_pair_54 = fma_f32x2(make_float2(((sv + 64))[18], ((sv + 64))[19]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[18] = _fma_pair_54.x;
                    (sv + 64)[19] = _fma_pair_54.y;
                    float2 _fma_pair_55 = fma_f32x2(make_float2(((sv + 64))[20], ((sv + 64))[21]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[20] = _fma_pair_55.x;
                    (sv + 64)[21] = _fma_pair_55.y;
                    float2 _fma_pair_56 = fma_f32x2(make_float2(((sv + 64))[22], ((sv + 64))[23]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[22] = _fma_pair_56.x;
                    (sv + 64)[23] = _fma_pair_56.y;
                    float2 _fma_pair_57 = fma_f32x2(make_float2(((sv + 64))[24], ((sv + 64))[25]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[24] = _fma_pair_57.x;
                    (sv + 64)[25] = _fma_pair_57.y;
                    float2 _fma_pair_58 = fma_f32x2(make_float2(((sv + 64))[26], ((sv + 64))[27]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[26] = _fma_pair_58.x;
                    (sv + 64)[27] = _fma_pair_58.y;
                    float2 _fma_pair_59 = fma_f32x2(make_float2(((sv + 64))[28], ((sv + 64))[29]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[28] = _fma_pair_59.x;
                    (sv + 64)[29] = _fma_pair_59.y;
                    float2 _fma_pair_60 = fma_f32x2(make_float2(((sv + 64))[30], ((sv + 64))[31]), _fma_b2_43, _fma_c2_44);
                    (sv + 64)[30] = _fma_pair_60.x;
                    (sv + 64)[31] = _fma_pair_60.y;
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        if (USE_TMEM_LD_RED == 0 && _le >= 12) {
                            float2 _exp2_pair_61 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 64], sv[_le*2 + 1 + 64]));
                            sv[_le*2 + 64] = _exp2_pair_61.x;
                            sv[_le*2 + 1 + 64] = _exp2_pair_61.y;
                        } else {
                            sv[_le*2 + 64] = approx_exp2(sv[_le*2 + 64]);
                            sv[_le*2 + 1 + 64] = approx_exp2(sv[_le*2 + 1 + 64]);
                        }
                    }
                    float2 _f2_11 = make_float2(sv[64], sv[65]);
                    float2 partial_5 = _f2_11;
                    #pragma unroll
                    for (int pair_4 = 2; pair_4 < 16; pair_4 += 2) {
                        float2 _f2_12 = make_float2((sv + 64)[pair_4], (sv + 64)[pair_4 + 1]);
                        partial_5 = add_f32x2(partial_5, _f2_12);
                    }
                    float2 _f2_13 = make_float2(p_scale, p_scale);
                    float2 scale2_6 = _f2_13;
                    block_sum2 = fma_f32x2_rn_ftz(partial_5, scale2_6, block_sum2);
                    float2 _f2_14 = make_float2(sv[80], sv[81]);
                    float2 partial_7 = _f2_14;
                    #pragma unroll
                    for (int pair_5 = 2; pair_5 < 16; pair_5 += 2) {
                        float2 _f2_15 = make_float2((sv + 80)[pair_5], (sv + 80)[pair_5 + 1]);
                        partial_7 = add_f32x2(partial_7, _f2_15);
                    }
                    block_sum2 = fma_f32x2_rn_ftz(partial_7, scale2_6, block_sum2);
                    uint32_t _fp4_2[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[0]) : "f"(sv[64]), "f"(sv[65]), "f"(sv[66]), "f"(sv[67]), "f"(sv[68]), "f"(sv[69]), "f"(sv[70]), "f"(sv[71]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[1]) : "f"(sv[72]), "f"(sv[73]), "f"(sv[74]), "f"(sv[75]), "f"(sv[76]), "f"(sv[77]), "f"(sv[78]), "f"(sv[79]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[2]) : "f"(sv[80]), "f"(sv[81]), "f"(sv[82]), "f"(sv[83]), "f"(sv[84]), "f"(sv[85]), "f"(sv[86]), "f"(sv[87]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[3]) : "f"(sv[88]), "f"(sv[89]), "f"(sv[90]), "f"(sv[91]), "f"(sv[92]), "f"(sv[93]), "f"(sv[94]), "f"(sv[95]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + stage * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 32 ^ ((warp % 4 * 32 + lane) * 64 + 32 >> 7 & 3) << 4))), "r"(_fp4_2[0]), "r"(_fp4_2[1]), "r"(_fp4_2[2]), "r"(_fp4_2[3]) : "memory");
                    const float2 _fma_b2_62 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_63 = {2.584962500721156f - block_max_scaled, 2.584962500721156f - block_max_scaled};
                    float2 _fma_pair_64 = fma_f32x2(make_float2(((sv + 96))[0], ((sv + 96))[1]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[0] = _fma_pair_64.x;
                    (sv + 96)[1] = _fma_pair_64.y;
                    float2 _fma_pair_65 = fma_f32x2(make_float2(((sv + 96))[2], ((sv + 96))[3]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[2] = _fma_pair_65.x;
                    (sv + 96)[3] = _fma_pair_65.y;
                    float2 _fma_pair_66 = fma_f32x2(make_float2(((sv + 96))[4], ((sv + 96))[5]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[4] = _fma_pair_66.x;
                    (sv + 96)[5] = _fma_pair_66.y;
                    float2 _fma_pair_67 = fma_f32x2(make_float2(((sv + 96))[6], ((sv + 96))[7]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[6] = _fma_pair_67.x;
                    (sv + 96)[7] = _fma_pair_67.y;
                    float2 _fma_pair_68 = fma_f32x2(make_float2(((sv + 96))[8], ((sv + 96))[9]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[8] = _fma_pair_68.x;
                    (sv + 96)[9] = _fma_pair_68.y;
                    float2 _fma_pair_69 = fma_f32x2(make_float2(((sv + 96))[10], ((sv + 96))[11]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[10] = _fma_pair_69.x;
                    (sv + 96)[11] = _fma_pair_69.y;
                    float2 _fma_pair_70 = fma_f32x2(make_float2(((sv + 96))[12], ((sv + 96))[13]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[12] = _fma_pair_70.x;
                    (sv + 96)[13] = _fma_pair_70.y;
                    float2 _fma_pair_71 = fma_f32x2(make_float2(((sv + 96))[14], ((sv + 96))[15]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[14] = _fma_pair_71.x;
                    (sv + 96)[15] = _fma_pair_71.y;
                    float2 _fma_pair_72 = fma_f32x2(make_float2(((sv + 96))[16], ((sv + 96))[17]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[16] = _fma_pair_72.x;
                    (sv + 96)[17] = _fma_pair_72.y;
                    float2 _fma_pair_73 = fma_f32x2(make_float2(((sv + 96))[18], ((sv + 96))[19]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[18] = _fma_pair_73.x;
                    (sv + 96)[19] = _fma_pair_73.y;
                    float2 _fma_pair_74 = fma_f32x2(make_float2(((sv + 96))[20], ((sv + 96))[21]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[20] = _fma_pair_74.x;
                    (sv + 96)[21] = _fma_pair_74.y;
                    float2 _fma_pair_75 = fma_f32x2(make_float2(((sv + 96))[22], ((sv + 96))[23]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[22] = _fma_pair_75.x;
                    (sv + 96)[23] = _fma_pair_75.y;
                    float2 _fma_pair_76 = fma_f32x2(make_float2(((sv + 96))[24], ((sv + 96))[25]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[24] = _fma_pair_76.x;
                    (sv + 96)[25] = _fma_pair_76.y;
                    float2 _fma_pair_77 = fma_f32x2(make_float2(((sv + 96))[26], ((sv + 96))[27]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[26] = _fma_pair_77.x;
                    (sv + 96)[27] = _fma_pair_77.y;
                    float2 _fma_pair_78 = fma_f32x2(make_float2(((sv + 96))[28], ((sv + 96))[29]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[28] = _fma_pair_78.x;
                    (sv + 96)[29] = _fma_pair_78.y;
                    float2 _fma_pair_79 = fma_f32x2(make_float2(((sv + 96))[30], ((sv + 96))[31]), _fma_b2_62, _fma_c2_63);
                    (sv + 96)[30] = _fma_pair_79.x;
                    (sv + 96)[31] = _fma_pair_79.y;
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        if (USE_TMEM_LD_RED == 0 && _le >= 12) {
                            float2 _exp2_pair_80 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 96], sv[_le*2 + 1 + 96]));
                            sv[_le*2 + 96] = _exp2_pair_80.x;
                            sv[_le*2 + 1 + 96] = _exp2_pair_80.y;
                        } else {
                            sv[_le*2 + 96] = approx_exp2(sv[_le*2 + 96]);
                            sv[_le*2 + 1 + 96] = approx_exp2(sv[_le*2 + 1 + 96]);
                        }
                    }
                    float2 _f2_16 = make_float2(sv[96], sv[97]);
                    float2 partial_8 = _f2_16;
                    #pragma unroll
                    for (int pair_6 = 2; pair_6 < 16; pair_6 += 2) {
                        float2 _f2_17 = make_float2((sv + 96)[pair_6], (sv + 96)[pair_6 + 1]);
                        partial_8 = add_f32x2(partial_8, _f2_17);
                    }
                    float2 _f2_18 = make_float2(p_scale, p_scale);
                    float2 scale2_9 = _f2_18;
                    block_sum2 = fma_f32x2_rn_ftz(partial_8, scale2_9, block_sum2);
                    float2 _f2_19 = make_float2(sv[112], sv[113]);
                    float2 partial_10 = _f2_19;
                    #pragma unroll
                    for (int pair_7 = 2; pair_7 < 16; pair_7 += 2) {
                        float2 _f2_20 = make_float2((sv + 112)[pair_7], (sv + 112)[pair_7 + 1]);
                        partial_10 = add_f32x2(partial_10, _f2_20);
                    }
                    block_sum2 = fma_f32x2_rn_ftz(partial_10, scale2_9, block_sum2);
                    uint32_t _fp4_3[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[0]) : "f"(sv[96]), "f"(sv[97]), "f"(sv[98]), "f"(sv[99]), "f"(sv[100]), "f"(sv[101]), "f"(sv[102]), "f"(sv[103]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[1]) : "f"(sv[104]), "f"(sv[105]), "f"(sv[106]), "f"(sv[107]), "f"(sv[108]), "f"(sv[109]), "f"(sv[110]), "f"(sv[111]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[2]) : "f"(sv[112]), "f"(sv[113]), "f"(sv[114]), "f"(sv[115]), "f"(sv[116]), "f"(sv[117]), "f"(sv[118]), "f"(sv[119]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[3]) : "f"(sv[120]), "f"(sv[121]), "f"(sv[122]), "f"(sv[123]), "f"(sv[124]), "f"(sv[125]), "f"(sv[126]), "f"(sv[127]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + stage * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 48 ^ ((warp % 4 * 32 + lane) * 64 + 48 >> 7 & 3) << 4))), "r"(_fp4_3[0]), "r"(_fp4_3[1]), "r"(_fp4_3[2]), "r"(_fp4_3[3]) : "memory");
                    int pv_sfa_hi_col = (unsigned int)TMEM_TMEM_SFA_PV0_HI_OFFSET + stage * 8 + (unsigned int)(warp % 4);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)pv_sfa_hi_col), "r"(sf_packed[0]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_2_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    block_sum = block_sum2.x + block_sum2.y;
                    row_sum = row_sum * acc_scale + block_sum;
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                }
                if (warp % 4 < 2) {
                    mbarrier_arrive(corr_sig_lo_addr + (stage) * 8);
                } else {
                    mbarrier_arrive(corr_sig_hi_addr + (stage) * 8);
                }
                mbarrier_wait(o_full_addr + (stage) * 8, _phase_o_full);
                _phase_o_full ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float final_scale = 0.0f;
                float _rcp_0 = approx_rcp(row_sum);
                final_scale = ((row_sum != 0.0f && row_sum == row_sum) ? _rcp_0 : 0.0f);
                int seg_len_out = cl_seg_len[tile_idx];
                int ws_slot_out = cl_ws_slot[tile_idx];
                int local_row = ((unsigned int)m_block + stage) * 128 + (unsigned int)(warp % 4 * 32 + lane);
                int token = seg_begin + local_row;
                long long out_off = ((long long)token * (long long)heads + (long long)head) * 128;
                int tmem_o_off = make_warp_uniform((unsigned int)TMEM_OUTPUT_0_OFFSET + stage * 128);
                int partial_row = ws_slot_out * 512 + cta_rank * 256 + scale_off + (warp % 4 * 32 + lane);
                if (ws_slot_out >= 0) {
                    if (local_row < seg_len_out) {
                        partial_ML[partial_row * 2] = row_max_scaled;
                        partial_ML[partial_row * 2 + 1] = row_sum;
                    }
                }
                #pragma unroll
                for (int col = 0; col < 8; col++) {
                    int addr = taddr + (unsigned int)tmem_o_off + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col * 16);
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], addr);
                    if (local_row < seg_len_out) {
                        if (ws_slot_out >= 0) {
                            {
                                const float2 _prescale2_81 = {final_scale, final_scale};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_0[0])[_ps], _prescale2_81);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 16; _ps++)
                                    _tmem_load_0[0 + _ps] *= final_scale;
                                #endif
                                __half2 _pk[8];
                                _pk[0] = __floats2half2_rn(_tmem_load_0[0 + 0], _tmem_load_0[0 + 1]);
                                _pk[1] = __floats2half2_rn(_tmem_load_0[0 + 2], _tmem_load_0[0 + 3]);
                                _pk[2] = __floats2half2_rn(_tmem_load_0[0 + 4], _tmem_load_0[0 + 5]);
                                _pk[3] = __floats2half2_rn(_tmem_load_0[0 + 6], _tmem_load_0[0 + 7]);
                                _pk[4] = __floats2half2_rn(_tmem_load_0[0 + 8], _tmem_load_0[0 + 9]);
                                _pk[5] = __floats2half2_rn(_tmem_load_0[0 + 10], _tmem_load_0[0 + 11]);
                                _pk[6] = __floats2half2_rn(_tmem_load_0[0 + 12], _tmem_load_0[0 + 13]);
                                _pk[7] = __floats2half2_rn(_tmem_load_0[0 + 14], _tmem_load_0[0 + 15]);
                                *reinterpret_cast<uint4*>(&((__half*)(partial_O + (partial_row * 128 + col * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                *reinterpret_cast<uint4*>(&((__half*)(partial_O + (partial_row * 128 + col * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                            }
                        } else {
                            {
                                const float2 _prescale2_82 = {final_scale, final_scale};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_0[0])[_ps], _prescale2_82);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 16; _ps++)
                                    _tmem_load_0[0 + _ps] *= final_scale;
                                #endif
                                __nv_bfloat162 _pk[8];
                                _pk[0] = __floats2bfloat162_rn(_tmem_load_0[0 + 0], _tmem_load_0[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_tmem_load_0[0 + 2], _tmem_load_0[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_tmem_load_0[0 + 4], _tmem_load_0[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_tmem_load_0[0 + 6], _tmem_load_0[0 + 7]);
                                _pk[4] = __floats2bfloat162_rn(_tmem_load_0[0 + 8], _tmem_load_0[0 + 9]);
                                _pk[5] = __floats2bfloat162_rn(_tmem_load_0[0 + 10], _tmem_load_0[0 + 11]);
                                _pk[6] = __floats2bfloat162_rn(_tmem_load_0[0 + 12], _tmem_load_0[0 + 13]);
                                _pk[7] = __floats2bfloat162_rn(_tmem_load_0[0 + 14], _tmem_load_0[0 + 15]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (out_off + (long long)(col * 16))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (out_off + (long long)(col * 16))))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: correction_lo ----
    if (warp >= 8 && warp <= 9) {
        { // correction_lo_main
            unsigned int total_tiles_1 = num_tiles;
            int owned_row_base = make_warp_uniform(warp % 2 * 32);
            unsigned int _phase_p_empty_0 = 1;
            unsigned int _phase_p_empty_1 = 1;
            unsigned int _phase_corr_sig_lo_0 = 0;
            unsigned int _phase_corr_sig_lo_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = cluster_id; tile_idx_1 < total_tiles_1; tile_idx_1 += num_clusters) {
                int head_1 = cl_head[tile_idx_1];
                int seg_begin_1 = cl_seg_begin[tile_idx_1];
                int seg_len_1 = cl_seg_len[tile_idx_1];
                int kv_base_1 = cl_kv_base[tile_idx_1];
                int m_block_1 = cl_q_block[tile_idx_1] + cta_rank * 2;
                unsigned int num_n_blocks_1 = cl_kv_blocks[tile_idx_1];
                int kv_begin_1 = cl_kv_begin[tile_idx_1];
                int ws_slot_1 = cl_ws_slot[tile_idx_1];
                mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                _phase_p_empty_0 ^= 1;
                mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1);
                _phase_p_empty_1 ^= 1;
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                _phase_corr_sig_lo_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                mbarrier_wait(corr_sig_lo_addr + 8, _phase_corr_sig_lo_1);
                _phase_corr_sig_lo_1 ^= 1;
                mbarrier_arrive(corr_done_addr + 8);
                #pragma unroll 1
                for (unsigned int n_iter_1 = 1; n_iter_1 < num_n_blocks_1; n_iter_1++) {
                    mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                    _phase_corr_sig_lo_0 ^= 1;
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                    _phase_p_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int logical_row = owned_row_base + lane;
                    int tmem_row_base = make_warp_uniform(owned_row_base << 16);
                    float scale = row_state[logical_row];
                    int _vote_0 = __all_sync(0xFFFFFFFF, scale == 1.0f);
                    int skip_rescale = _vote_0;
                    if (skip_rescale == 0) {
                        #pragma unroll
                        for (int col_1 = 0; col_1 < 8; col_1++) {
                            int addr_1 = make_warp_uniform(taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)tmem_row_base + (unsigned int)(col_1 * 16));
                            float _tmem_load_1[16];
                            tmem_ld_x16(&_tmem_load_1[0], addr_1);
                            const float2 _scale2_0 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_0);
                            tmem_st_x16_f32(addr_1, _tmem_load_1);
                        }
                    }
                    int skip_rescale_0 = skip_rescale;
                    if (skip_rescale_0 == 0) {
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(corr_sig_lo_addr + 8, _phase_corr_sig_lo_1);
                    _phase_corr_sig_lo_1 ^= 1;
                    mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1);
                    _phase_p_empty_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int logical_row_1 = owned_row_base + lane;
                    int tmem_row_base_2 = make_warp_uniform(owned_row_base << 16);
                    float scale_3 = row_state[logical_row_1 + 128];
                    int _vote_1 = __all_sync(0xFFFFFFFF, scale_3 == 1.0f);
                    int skip_rescale_4 = _vote_1;
                    if (skip_rescale_4 == 0) {
                        #pragma unroll
                        for (int col_2 = 0; col_2 < 8; col_2++) {
                            int addr_2 = make_warp_uniform(taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)tmem_row_base_2 + (unsigned int)(col_2 * 16));
                            float _tmem_load_2[16];
                            tmem_ld_x16(&_tmem_load_2[0], addr_2);
                            const float2 _scale2_1 = {scale_3, scale_3};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_1);
                            tmem_st_x16_f32(addr_2, _tmem_load_2);
                        }
                    }
                    skip_rescale_0 = skip_rescale_4;
                    if (skip_rescale_0 == 0) {
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(corr_done_addr + 8);
                }
                mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                _phase_corr_sig_lo_0 ^= 1;
                mbarrier_wait(corr_sig_lo_addr + 8, _phase_corr_sig_lo_1);
                _phase_corr_sig_lo_1 ^= 1;
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
                _phase_o_full_1 ^= 1;
            }
        }
    }
    // ---- Role: correction_hi ----
    if (warp >= 10 && warp <= 11) {
        { // correction_hi_main
            unsigned int total_tiles_2 = num_tiles;
            int owned_row_base_1 = make_warp_uniform(64 + warp % 2 * 32);
            unsigned int _phase_p_empty_0_1 = 1;
            unsigned int _phase_p_empty_1_1 = 1;
            unsigned int _phase_corr_sig_hi_0 = 0;
            unsigned int _phase_corr_sig_hi_1 = 0;
            unsigned int _phase_o_full_0_1 = 0;
            unsigned int _phase_o_full_1_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < total_tiles_2; tile_idx_2 += num_clusters) {
                int head_2 = cl_head[tile_idx_2];
                int seg_begin_2 = cl_seg_begin[tile_idx_2];
                int seg_len_2 = cl_seg_len[tile_idx_2];
                int kv_base_2 = cl_kv_base[tile_idx_2];
                int m_block_2 = cl_q_block[tile_idx_2] + cta_rank * 2;
                unsigned int num_n_blocks_2 = cl_kv_blocks[tile_idx_2];
                int kv_begin_2 = cl_kv_begin[tile_idx_2];
                int ws_slot_2 = cl_ws_slot[tile_idx_2];
                mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                _phase_p_empty_0_1 ^= 1;
                mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1_1);
                _phase_p_empty_1_1 ^= 1;
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                _phase_corr_sig_hi_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                mbarrier_wait(corr_sig_hi_addr + 8, _phase_corr_sig_hi_1);
                _phase_corr_sig_hi_1 ^= 1;
                mbarrier_arrive(corr_done_addr + 8);
                #pragma unroll 1
                for (unsigned int n_iter_2 = 1; n_iter_2 < num_n_blocks_2; n_iter_2++) {
                    mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                    _phase_corr_sig_hi_0 ^= 1;
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                    _phase_p_empty_0_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int logical_row_2 = owned_row_base_1 + lane;
                    int tmem_row_base_1 = make_warp_uniform(owned_row_base_1 << 16);
                    float scale_1 = row_state[logical_row_2];
                    int _vote_2 = __all_sync(0xFFFFFFFF, scale_1 == 1.0f);
                    int skip_rescale_1 = _vote_2;
                    if (skip_rescale_1 == 0) {
                        #pragma unroll
                        for (int col_3 = 0; col_3 < 8; col_3++) {
                            int addr_3 = make_warp_uniform(taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)tmem_row_base_1 + (unsigned int)(col_3 * 16));
                            float _tmem_load_3[16];
                            tmem_ld_x16(&_tmem_load_3[0], addr_3);
                            const float2 _scale2_0 = {scale_1, scale_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_0);
                            tmem_st_x16_f32(addr_3, _tmem_load_3);
                        }
                    }
                    int skip_rescale_0_1 = skip_rescale_1;
                    if (skip_rescale_0_1 == 0) {
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(corr_sig_hi_addr + 8, _phase_corr_sig_hi_1);
                    _phase_corr_sig_hi_1 ^= 1;
                    mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1_1);
                    _phase_p_empty_1_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int logical_row_1_1 = owned_row_base_1 + lane;
                    int tmem_row_base_2_1 = make_warp_uniform(owned_row_base_1 << 16);
                    float scale_3_1 = row_state[logical_row_1_1 + 128];
                    int _vote_3 = __all_sync(0xFFFFFFFF, scale_3_1 == 1.0f);
                    int skip_rescale_4_1 = _vote_3;
                    if (skip_rescale_4_1 == 0) {
                        #pragma unroll
                        for (int col_4 = 0; col_4 < 8; col_4++) {
                            int addr_4 = make_warp_uniform(taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)tmem_row_base_2_1 + (unsigned int)(col_4 * 16));
                            float _tmem_load_4[16];
                            tmem_ld_x16(&_tmem_load_4[0], addr_4);
                            const float2 _scale2_1 = {scale_3_1, scale_3_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_1);
                            tmem_st_x16_f32(addr_4, _tmem_load_4);
                        }
                    }
                    skip_rescale_0_1 = skip_rescale_4_1;
                    if (skip_rescale_0_1 == 0) {
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(corr_done_addr + 8);
                }
                mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                _phase_corr_sig_hi_0 ^= 1;
                mbarrier_wait(corr_sig_hi_addr + 8, _phase_corr_sig_hi_1);
                _phase_corr_sig_hi_1 ^= 1;
                mbarrier_wait(o_full_addr, _phase_o_full_0_1);
                _phase_o_full_0_1 ^= 1;
                mbarrier_wait(o_full_addr + 8, _phase_o_full_1_1);
                _phase_o_full_1_1 ^= 1;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_s_empty_0 = 0;
            unsigned int _phase_s_empty_1 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_p_full_2_0 = 0;
            unsigned int _phase_p_full_2_1 = 0;
            if (cta_rank == 0) {
                unsigned int total_tiles_3 = num_tiles;
                unsigned int k_stage = 0;
                unsigned int k_phase = 0;
                unsigned int v_stage = 0;
                unsigned int v_phase = 0;
                #pragma unroll 1
                for (unsigned int tile_idx_3 = cluster_id; tile_idx_3 < total_tiles_3; tile_idx_3 += num_clusters) {
                    int head_3 = cl_head[tile_idx_3];
                    int seg_begin_3 = cl_seg_begin[tile_idx_3];
                    int seg_len_3 = cl_seg_len[tile_idx_3];
                    int kv_base_3 = cl_kv_base[tile_idx_3];
                    int m_block_3 = cl_q_block[tile_idx_3] + cta_rank * 2;
                    unsigned int num_n_blocks_3 = cl_kv_blocks[tile_idx_3];
                    int kv_begin_3 = cl_kv_begin[tile_idx_3];
                    int ws_slot_3 = cl_ws_slot[tile_idx_3];
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    mbarrier_wait(q_full_addr + 8, _phase_q_full_1);
                    _phase_q_full_1 ^= 1;
                    mbarrier_wait(k_full_addr + (k_stage) * 8, k_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4))));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4) + 8)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                    }
                    int _mma_a_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                    int _mma_b_lo_0 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                        }
                    }
                    elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                    mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                    _phase_s_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_qk1, make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4) + 64)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa_qk1 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4) + 64 + 8)));
                    }
                    int _mma_a_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512;
                    int _mma_b_lo_1 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_qk1 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                0x10200480U, tmem_tmem_sfa_qk1 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                        }
                    }
                    elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                    k_stage += 1;
                    if (k_stage == 4) { k_stage = 0; k_phase ^= 1; }
                    unsigned int first_pv = 1;
                    #pragma unroll 1
                    for (unsigned int n_iter_3 = 0; n_iter_3 < num_n_blocks_3 - 1; n_iter_3++) {
                        int first_pv_flag = first_pv;
                        mbarrier_wait(v_full_addr + (v_stage) * 8, v_phase);
                        mbarrier_wait(k_full_addr + (k_stage) * 8, k_phase);
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                        }
                        mbarrier_wait(s_empty_addr + 8, _phase_s_empty_1);
                        _phase_s_empty_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_2 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int _mma_b_lo_2 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                            }
                        }
                        elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                        mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                        _phase_s_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_3 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512;
                        int _mma_b_lo_3 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk1 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk1 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                            }
                        }
                        elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                        k_stage += 1;
                        if (k_stage == 4) { k_stage = 0; k_phase ^= 1; }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_lo, make_sf_cp_desc_lo_sbo128((((smem_sfvt_lo_addr) >> 4) + (v_stage) * 64)));
                        }
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        _phase_p_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_4 = (((smem_p_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int _mma_b_lo_4 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag) ? 0 : 1));
                            }
                        }
                        mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                        _phase_p_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_5 = (((smem_p_addr) >> 4) & 0x3FFF) + (1) * 512;
                        int _mma_b_lo_5 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag) ? 0 : 1));
                            }
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_hi, make_sf_cp_desc_lo_sbo128((((smem_sfvt_hi_addr) >> 4) + (v_stage) * 64)));
                        }
                        mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                        _phase_p_full_2_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_6 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (0) * 512;
                        int _mma_b_lo_6 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                            }
                        }
                        elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                        mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                        _phase_p_full_2_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_7 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (1) * 512;
                        int _mma_b_lo_7 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                            }
                        }
                        elect_commit_cg2_multicast(p_empty_addr + 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                        v_stage += 1;
                        if (v_stage == 5) { v_stage = 0; v_phase ^= 1; }
                        first_pv = 0;
                    }
                    int first_pv_flag_1 = first_pv;
                    mbarrier_wait(v_full_addr + (v_stage) * 8, v_phase);
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_lo, make_sf_cp_desc_lo_sbo128((((smem_sfvt_lo_addr) >> 4) + (v_stage) * 64)));
                    }
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_8 = (((smem_p_addr) >> 4) & 0x3FFF) + (0) * 512;
                    int _mma_b_lo_8 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_8) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_8) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                        }
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_hi, make_sf_cp_desc_lo_sbo128((((smem_sfvt_hi_addr) >> 4) + (v_stage) * 64)));
                    }
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                    _phase_p_full_2_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_9 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (0) * 512;
                    int _mma_b_lo_9 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_9) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_9) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                        }
                    }
                    elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_10 = (((smem_p_addr) >> 4) & 0x3FFF) + (1) * 512;
                    int _mma_b_lo_10 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_10) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_10) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                        }
                    }
                    mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                    _phase_p_full_2_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_11 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (1) * 512;
                    int _mma_b_lo_11 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_11) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_11) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                        }
                    }
                    elect_commit_cg2_multicast(p_empty_addr + 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                    v_stage += 1;
                    if (v_stage == 5) { v_stage = 0; v_phase ^= 1; }
                    mbarrier_wait(s_empty_addr + 8, _phase_s_empty_1);
                    _phase_s_empty_1 ^= 1;
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(o_full_addr + 8, (uint16_t)(3));
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            unsigned int total_tiles_4 = num_tiles;
            unsigned int k_load_stage = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_k_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_4 = cluster_id; tile_idx_4 < total_tiles_4; tile_idx_4 += num_clusters) {
                int head_4 = cl_head[tile_idx_4];
                int seg_begin_4 = cl_seg_begin[tile_idx_4];
                int seg_len_4 = cl_seg_len[tile_idx_4];
                int kv_base_4 = cl_kv_base[tile_idx_4];
                int m_block_4 = cl_q_block[tile_idx_4] + cta_rank * 2;
                unsigned int num_n_blocks_4 = cl_kv_blocks[tile_idx_4];
                int kv_begin_4 = cl_kv_begin[tile_idx_4];
                int ws_slot_4 = cl_ws_slot[tile_idx_4];
                int q_tile = kv_base_4 + m_block_4;
                int q_sf_tile = head_4 * PB + q_tile;
                int q_row = q_sf_tile * 128;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    tma_2d_gmem2smem_cta2(smem_q_addr, (&Q), 0, q_row, ((q_full_addr) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_sfq_addr, (&SFQ), 0, q_sf_tile * 32, ((q_full_addr) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_q_addr + 8192, (&Q), 0, q_row + 128, ((q_full_addr + 8) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_sfq_addr + 1024, (&SFQ), 0, (q_sf_tile + 1) * 32, ((q_full_addr + 8) & 0xFEFFFFFF));
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(9216)) : "memory");
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr + 8) & 0xFEFFFFFF), "r"((uint32_t)(9216)) : "memory");
                }
                #pragma unroll 1
                for (unsigned int ni = 0; ni < num_n_blocks_4; ni++) {
                    unsigned int n = num_n_blocks_4 - 1 - ni;
                    int kv_sf_tile = (unsigned int)(head_4 * PB + kv_base_4 + kv_begin_4) + n;
                    int kv_row = kv_sf_tile * 128;
                    mbarrier_wait(k_empty_addr + (k_load_stage) * 8, _phase_k_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem_cta2(smem_k_addr + k_load_stage * 4096, (&K), 0, kv_row + cta_rank * 64, ((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfk_addr + k_load_stage * 1024, (&SFK), 0, kv_sf_tile * 32, ((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(5120)) : "memory");
                    }
                    k_load_stage += 1;
                    if (k_load_stage == 4) { k_load_stage = 0; _phase_k_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: v_load ----
    if (warp == 14) {
        { // v_load_main
            unsigned int total_tiles_5 = num_tiles;
            unsigned int v_load_stage = 0;
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_5 = cluster_id; tile_idx_5 < total_tiles_5; tile_idx_5 += num_clusters) {
                int head_5 = cl_head[tile_idx_5];
                int seg_begin_5 = cl_seg_begin[tile_idx_5];
                int seg_len_5 = cl_seg_len[tile_idx_5];
                int kv_base_5 = cl_kv_base[tile_idx_5];
                int m_block_5 = cl_q_block[tile_idx_5] + cta_rank * 2;
                unsigned int num_n_blocks_5 = cl_kv_blocks[tile_idx_5];
                int kv_begin_5 = cl_kv_begin[tile_idx_5];
                int ws_slot_5 = cl_ws_slot[tile_idx_5];
                #pragma unroll 1
                for (unsigned int ni_1 = 0; ni_1 < num_n_blocks_5; ni_1++) {
                    unsigned int n_1 = num_n_blocks_5 - 1 - ni_1;
                    int kv_tile = (unsigned int)(kv_base_5 + kv_begin_5) + n_1;
                    int kv_sf_tile_1 = head_5 * PB + kv_tile;
                    mbarrier_wait(v_empty_addr + (v_load_stage) * 8, _phase_v_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem_cta2(smem_vt_addr + v_load_stage * 4096, (&Vt), kv_tile * 64, head_5 * 128 + cta_rank * 64, ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfvt_lo_addr + v_load_stage * 1024, (&SFVtLo), 0, kv_sf_tile_1 * 16, ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfvt_hi_addr + v_load_stage * 1024, (&SFVtHi), 0, kv_sf_tile_1 * 16, ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(5120)) : "memory");
                    }
                    v_load_stage += 1;
                    if (v_load_stage == 5) { v_load_stage = 0; _phase_v_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 15) {
        { // idle_main
            __syncwarp();
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
