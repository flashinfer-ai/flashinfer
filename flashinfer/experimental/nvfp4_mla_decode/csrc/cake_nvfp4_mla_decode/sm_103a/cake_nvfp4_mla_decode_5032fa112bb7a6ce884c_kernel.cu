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
#define TMEM_NCOLS 480
#define TMEM_SCORES0_OFFSET 0
#define TMEM_SCORES1_OFFSET 64
#define TMEM_ACC_OFFSET 128
#define TMEM_TMEM_SFA0_OFFSET 384
#define TMEM_TMEM_SFA1_OFFSET 392
#define TMEM_TMEM_SFA2_OFFSET 400
#define TMEM_TMEM_SFA3_OFFSET 408
#define TMEM_TMEM_SFB0_OFFSET 416
#define TMEM_TMEM_SFB1_OFFSET 424
#define TMEM_TMEM_SFB2_OFFSET 432
#define TMEM_TMEM_SFB3_OFFSET 440
#define TMEM_TMEM_SFB4_OFFSET 448
#define TMEM_TMEM_SFB5_OFFSET 456
#define TMEM_TMEM_SFB6_OFFSET 464
#define TMEM_TMEM_SFB7_OFFSET 472
#define NUM_RAW_PIPE_STAGES 6
#define NUM_V_PIPE_STAGES 4
#define NUM_P_PIPE_STAGES 2
#define NUM_S_PIPE_STAGES 2
#define NUM_SF_PIPE_STAGES 2
#define SMEM_SMEM_Q0_OFF 1024
#define SMEM_SMEM_Q0_STAGE_BYTES 8192
#define SMEM_SMEM_Q0_STRIDE 8192
#define SMEM_SMEM_Q1_OFF 9216
#define SMEM_SMEM_Q1_STAGE_BYTES 8192
#define SMEM_SMEM_Q1_STRIDE 8192
#define SMEM_SMEM_Q2_OFF 17408
#define SMEM_SMEM_Q2_STAGE_BYTES 8192
#define SMEM_SMEM_Q2_STRIDE 8192
#define SMEM_SMEM_Q3_OFF 25600
#define SMEM_SMEM_Q3_STAGE_BYTES 8192
#define SMEM_SMEM_Q3_STRIDE 8192
#define SMEM_SMEM_QS_OFF 33792
#define SMEM_SMEM_QS_STAGE_BYTES 4096
#define SMEM_SMEM_QS_STRIDE 4096
#define SMEM_SMEM_K0_OFF 37888
#define SMEM_SMEM_K0_STAGE_BYTES 4096
#define SMEM_SMEM_K0_STRIDE 18432
#define SMEM_SMEM_K1_OFF 41984
#define SMEM_SMEM_K1_STAGE_BYTES 4096
#define SMEM_SMEM_K1_STRIDE 18432
#define SMEM_SMEM_K2_OFF 46080
#define SMEM_SMEM_K2_STAGE_BYTES 4096
#define SMEM_SMEM_K2_STRIDE 18432
#define SMEM_SMEM_K3_OFF 50176
#define SMEM_SMEM_K3_STAGE_BYTES 4096
#define SMEM_SMEM_K3_STRIDE 18432
#define SMEM_SMEM_KS_OFF 54272
#define SMEM_SMEM_KS_STAGE_BYTES 2048
#define SMEM_SMEM_KS_STRIDE 18432
#define SMEM_SMEM_RAW_FLAT_OFF 37888
#define SMEM_SMEM_RAW_FLAT_STAGE_BYTES 110592
#define SMEM_SMEM_RAW_FLAT_STRIDE 110592
#define SMEM_SMEM_V_OFF 148480
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_P_OFF 214016
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_ROW_STATE_OFF 230400
#define SMEM_ROW_STATE_STAGE_BYTES 512
#define SMEM_ROW_STATE_STRIDE 512
#define SMEM_SMEM_PAGES_OFF 230912
#define SMEM_SMEM_PAGES_STAGE_BYTES 128
#define SMEM_SMEM_PAGES_STRIDE 128
#define SMEM_TOTAL 231424
#define THREADS 512
#define SKIP_XFORM 0
#define LAZY_RESCALE 0
#define EXP2_EMU 1
#define IDENTITY_PAGES 0
#define SKIP_EXP 0
#define SKIP_KPUB 0
#define SKIP_PSTORE 0
#define SKIP_QK 0
#define SKIP_PV 0
#define SKIP_SMATH 0
#define TRACE 0

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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
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


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
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


__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
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



__device__ __forceinline__ void softmax_block_sum(const float* sv, float2* acc) {
    const float2* sv2 = reinterpret_cast<const float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        asm("add.f32x2 %0, %1, %2;"
            : "+l"(reinterpret_cast<uint64_t&>(*acc))
            : "l"(reinterpret_cast<uint64_t&>(*acc)),
              "l"(reinterpret_cast<const uint64_t&>(sv2[j])));
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512) __cluster_dims__(2,1,1) void
kernel_cake_nvfp4_mla_decode_5032fa112bb7a6ce884c(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap QS, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KVS, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, float* __restrict__ partial_o, float* __restrict__ partial_lse, int* __restrict__ work_table, int* __restrict__ unit_first, int* __restrict__ page_table, int* __restrict__ seq_lens, int* __restrict__ q_indptr, float* __restrict__ sinks, int num_heads, int q_len, int max_pages, int max_splits, float scale_log2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define raw_full_addr (mbar_base + 16)
    #define raw_empty_addr (mbar_base + 64)
    #define peer_empty_addr (mbar_base + 112)
    #define sfa_full_addr (mbar_base + 160)
    #define sf_full_addr (mbar_base + 168)
    #define sf_empty_addr (mbar_base + 184)
    #define s_full_addr (mbar_base + 200)
    #define s_empty_addr (mbar_base + 216)
    #define v_full_addr (mbar_base + 232)
    #define v_empty_addr (mbar_base + 264)
    #define corr_sig_addr (mbar_base + 296)
    #define corr_empty_addr (mbar_base + 304)
    #define p_full_addr (mbar_base + 312)
    #define p_empty_addr (mbar_base + 328)
    #define pv_done_addr (mbar_base + 344)
    #define o_full_addr (mbar_base + 352)
    #define o_empty_addr (mbar_base + 360)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_q0 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q0_addr = smem + 1024;
    uint8_t* smem_q1 = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_q1_addr = smem + 9216;
    uint8_t* smem_q2 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_q2_addr = smem + 17408;
    uint8_t* smem_q3 = reinterpret_cast<uint8_t*>(smem_raw + 25600);
    const int smem_q3_addr = smem + 25600;
    uint8_t* smem_qs = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_qs_addr = smem + 33792;
    uint8_t* smem_k0 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_k0_addr = smem + 37888;
    uint8_t* smem_k1 = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int smem_k1_addr = smem + 41984;
    uint8_t* smem_k2 = reinterpret_cast<uint8_t*>(smem_raw + 46080);
    const int smem_k2_addr = smem + 46080;
    uint8_t* smem_k3 = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int smem_k3_addr = smem + 50176;
    uint8_t* smem_ks = reinterpret_cast<uint8_t*>(smem_raw + 54272);
    const int smem_ks_addr = smem + 54272;
    uint8_t* smem_raw_flat = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_raw_flat_addr = smem + 37888;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int smem_v_addr = smem + 148480;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 214016);
    const int smem_p_addr = smem + 214016;
    float* row_state = reinterpret_cast<float*>(smem_raw + 230400);
    const int row_state_addr = smem + 230400;
    int* smem_pages = reinterpret_cast<int*>(smem_raw + 230912);
    const int smem_pages_addr = smem + 230912;

    // Mbarrier init (19 pipeline groups, 0 ordered-sequence groups, 46 barriers)
    // Mbarriers at smem_raw[0..368)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'raw_pipe' ---
            // raw_full: 6 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // raw_empty: 6 barriers, init_count=321
            mbarrier_init(smem + 64, 321);
            mbarrier_init(smem + 72, 321);
            mbarrier_init(smem + 80, 321);
            mbarrier_init(smem + 88, 321);
            mbarrier_init(smem + 96, 321);
            mbarrier_init(smem + 104, 321);
            // peer_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // sfa_full: 1 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            // --- pipeline 'sf_pipe' ---
            // sf_full: 2 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            // sf_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // --- pipeline 's_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // s_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 216, 128);
            mbarrier_init(smem + 224, 128);
            // --- pipeline 'v_pipe' ---
            // v_full: 4 barriers, init_count=192
            mbarrier_init(smem + 232, 192);
            mbarrier_init(smem + 240, 192);
            mbarrier_init(smem + 248, 192);
            mbarrier_init(smem + 256, 192);
            // v_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            // corr_sig: 1 barriers, init_count=128
            mbarrier_init(smem + 296, 128);
            // corr_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 304, 128);
            // --- pipeline 'p_pipe' ---
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 312, 256);
            mbarrier_init(smem + 320, 256);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            // pv_done: 1 barriers, init_count=1
            mbarrier_init(smem + 344, 1);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 352, 1);
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 360, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 480 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 368);
    if (warp == 0) {
        int _tmem_hold = smem + 368;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores0 = taddr;
    const int tmem_scores1 = taddr + 64;
    const int tmem_acc = taddr + 128;
    const int tmem_tmem_sfa0 = taddr + 384;
    const int tmem_tmem_sfa1 = taddr + 392;
    const int tmem_tmem_sfa2 = taddr + 400;
    const int tmem_tmem_sfa3 = taddr + 408;
    const int tmem_tmem_sfb0 = taddr + 416;
    const int tmem_tmem_sfb1 = taddr + 424;
    const int tmem_tmem_sfb2 = taddr + 432;
    const int tmem_tmem_sfb3 = taddr + 440;
    const int tmem_tmem_sfb4 = taddr + 448;
    const int tmem_tmem_sfb5 = taddr + 456;
    const int tmem_tmem_sfb6 = taddr + 464;
    const int tmem_tmem_sfb7 = taddr + 472;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            int quadrant = make_warp_uniform(warp % 4);
            int my_row = quadrant * 32 + lane;
            int lane_base = quadrant * 32 << 16;
            int p_stage_s = 0;
            int p_phase_s = 1;
            int s_stage_s = 0;
            int s_phase_s = 0;
            int unit_s = cluster_id;
            int item_lo_s = unit_first[unit_s] + cta_rank;
            int item_hi_s = unit_first[unit_s + 1];
            int tr_s = blockIdx.x == 0 && warp == 0 && lane == 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_corr_empty_0 = 1;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (int item_s = item_lo_s; item_s < item_hi_s; item_s += 2) {
                int tr_item_s = tr_s & (int)(item_s == item_lo_s);
                int b_s = work_table[item_s * 8];
                int m_tile_s = work_table[item_s * 8 + 1];
                int v_half_s = work_table[item_s * 8 + 2];
                int tile_start_s = work_table[item_s * 8 + 3];
                int tile_end_s = work_table[item_s * 8 + 4];
                int split_s = work_table[item_s * 8 + 5];
                int num_splits_s = work_table[item_s * 8 + 6];
                int flags_s = work_table[item_s * 8 + 7];
                int direct_out = (flags_s & 2) != 0;
                int num_tiles_s = tile_end_s - tile_start_s;
                int kv_len_s = seq_lens[b_s];
                int q_start_s = q_indptr[b_s];
                int row_in_req = m_tile_s * 128 + my_row;
                int token_s = row_in_req / num_heads;
                int head_s = row_in_req - token_s * num_heads;
                int row_valid = row_in_req < q_len * num_heads;
                int _min_0 = ((kv_len_s - q_len + token_s + 1) < (kv_len_s) ? (kv_len_s - q_len + token_s + 1) : (kv_len_s));
                int row_limit = _min_0;
                float row_max = -CAKE_INF;
                float row_sum = 0.0f;
                if ((flags_s & 1) != 0) {
                    int _min_1 = ((head_s) < (num_heads - 1) ? (head_s) : (num_heads - 1));
                    int sink_head = _min_1;
                    row_max = sinks[sink_head] * 1.4426950408889634f;
                    row_sum = 256.0f;
                }
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                unsigned int q_words[8];
                int qs_addr = smem_qs_addr + (unsigned int)(my_row * 32);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words[(0) + 3]))
                    : "r"(qs_addr));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words[(4) + 3]))
                    : "r"(qs_addr + 16));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base + (unsigned int)TMEM_TMEM_SFA0_OFFSET + (unsigned int)quadrant), "r"(q_words[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base + (unsigned int)TMEM_TMEM_SFA0_OFFSET + 4 + (unsigned int)quadrant), "r"((q_words + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base + (unsigned int)TMEM_TMEM_SFA1_OFFSET + (unsigned int)quadrant), "r"((q_words + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base + (unsigned int)TMEM_TMEM_SFA1_OFFSET + 4 + (unsigned int)quadrant), "r"((q_words + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base + (unsigned int)TMEM_TMEM_SFA2_OFFSET + (unsigned int)quadrant), "r"((q_words + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base + (unsigned int)TMEM_TMEM_SFA2_OFFSET + 4 + (unsigned int)quadrant), "r"((q_words + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base + (unsigned int)TMEM_TMEM_SFA3_OFFSET + (unsigned int)quadrant), "r"((q_words + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base + (unsigned int)TMEM_TMEM_SFA3_OFFSET + 4 + (unsigned int)quadrant), "r"((q_words + 7)[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(sfa_full_addr);
                #pragma unroll 1
                for (int it_s = 0; it_s < num_tiles_s; it_s++) {
                    mbarrier_wait(s_full_addr + (s_stage_s) * 8, s_phase_s);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float sv[64];
                    {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                            : "r"(taddr + (unsigned int)lane_base + (unsigned int)(s_stage_s * 64)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63])
                            : "r"(taddr + (unsigned int)lane_base + (unsigned int)(s_stage_s * 64) + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(s_empty_addr + (s_stage_s) * 8);
                    s_stage_s += 1;
                    if (s_stage_s == 2) { s_stage_s = 0; s_phase_s ^= 1; }
                    int _max_0 = ((row_limit - (tile_start_s + it_s) * 64) > (0) ? (row_limit - (tile_start_s + it_s) * 64) : (0));
                    int _min_2 = ((_max_0) < (64) ? (_max_0) : (64));
                    int valid_s = _min_2;
                    float tile_max = 0.0f;
                    {
                        if (valid_s < 64) {
                            uint32_t _slice_lo_mask_0;
                            {
                                int _lim_0 = valid_s;
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
                                int _lim_1 = valid_s - 32;
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
                        }
                        float2 _reg_reduce_max2_2 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&sv[0], _reg_reduce_max2_2);
                        row_max_x32_accum(&sv[32], _reg_reduce_max2_2);
                        float sv_max = row_max_reduce(_reg_reduce_max2_2);
                        tile_max = sv_max;
                    }
                    float tile_max_scaled = ((tile_max > -CAKE_INF) ? tile_max * scale_log2 : -CAKE_INF);
                    float _max_1 = max_noftz(row_max, tile_max_scaled);
                    float new_max = _max_1;
                    float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                    float _exp2_0 = approx_exp2(row_max - safe_max);
                    float alpha = ((row_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                    row_max = new_max;
                    mbarrier_wait(corr_empty_addr, _phase_corr_empty_0);
                    _phase_corr_empty_0 ^= 1;
                    row_state[my_row] = alpha;
                    mbarrier_arrive(corr_sig_addr);
                    float block_sum = 0.0f;
                    {
                        {
                            const float2 _fma_b2_3 = {scale_log2, scale_log2};
                            const float2 _fma_c2_4 = {8.0f - safe_max, 8.0f - safe_max};
                            #pragma unroll
                            for (int _lf = 0; _lf < 32; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(sv)[_lf], _fma_b2_3, _fma_c2_4);
                            {
                                #pragma unroll
                                for (int _le = 0; _le < 32; _le++) {
                                    if (1 && _le >= 24) {
                                        float2 _exp2_pair_5 = ex2_emulation_f32x2_value(make_float2(sv[_le*2], sv[_le*2 + 1]));
                                        sv[_le*2] = _exp2_pair_5.x;
                                        sv[_le*2 + 1] = _exp2_pair_5.y;
                                    } else {
                                        sv[_le*2] = approx_exp2(sv[_le*2]);
                                        sv[_le*2 + 1] = approx_exp2(sv[_le*2 + 1]);
                                    }
                                }
                            }
                            float2 _reg_reduce_sum2_6 = make_float2(0.0f, 0.0f);
                            softmax_block_sum(&sv[0], &_reg_reduce_sum2_6);
                            softmax_block_sum(&sv[32], &_reg_reduce_sum2_6);
                            float sv_sum = _reg_reduce_sum2_6.x + _reg_reduce_sum2_6.y;
                            block_sum = sv_sum;
                        }
                    }
                    row_sum = row_sum * alpha + block_sum;
                    mbarrier_wait(p_empty_addr + (p_stage_s) * 8, p_phase_s);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        {
                            uint32_t _fp8_0[16];
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[0]), "f"(sv[1]),
                                                       "f"(sv[2]), "f"(sv[3]));
                                _fp8_0[0] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[4]), "f"(sv[5]),
                                                       "f"(sv[6]), "f"(sv[7]));
                                _fp8_0[1] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[8]), "f"(sv[9]),
                                                       "f"(sv[10]), "f"(sv[11]));
                                _fp8_0[2] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[12]), "f"(sv[13]),
                                                       "f"(sv[14]), "f"(sv[15]));
                                _fp8_0[3] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[16]), "f"(sv[17]),
                                                       "f"(sv[18]), "f"(sv[19]));
                                _fp8_0[4] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[20]), "f"(sv[21]),
                                                       "f"(sv[22]), "f"(sv[23]));
                                _fp8_0[5] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[24]), "f"(sv[25]),
                                                       "f"(sv[26]), "f"(sv[27]));
                                _fp8_0[6] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[28]), "f"(sv[29]),
                                                       "f"(sv[30]), "f"(sv[31]));
                                _fp8_0[7] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[32]), "f"(sv[33]),
                                                       "f"(sv[34]), "f"(sv[35]));
                                _fp8_0[8] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[36]), "f"(sv[37]),
                                                       "f"(sv[38]), "f"(sv[39]));
                                _fp8_0[9] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[40]), "f"(sv[41]),
                                                       "f"(sv[42]), "f"(sv[43]));
                                _fp8_0[10] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[44]), "f"(sv[45]),
                                                       "f"(sv[46]), "f"(sv[47]));
                                _fp8_0[11] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[48]), "f"(sv[49]),
                                                       "f"(sv[50]), "f"(sv[51]));
                                _fp8_0[12] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[52]), "f"(sv[53]),
                                                       "f"(sv[54]), "f"(sv[55]));
                                _fp8_0[13] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[56]), "f"(sv[57]),
                                                       "f"(sv[58]), "f"(sv[59]));
                                _fp8_0[14] = _packed;
                            }
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[60]), "f"(sv[61]),
                                                       "f"(sv[62]), "f"(sv[63]));
                                _fp8_0[15] = _packed;
                            }
                            int p_base = smem_p_addr + (unsigned int)(p_stage_s * 8192);
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (my_row * 64 ^ (my_row * 64 >> 7 & 3) << 4))), "r"(_fp8_0[0]), "r"(_fp8_0[1]), "r"(_fp8_0[2]), "r"(_fp8_0[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (my_row * 64 + 16 ^ (my_row * 64 + 16 >> 7 & 3) << 4))), "r"(_fp8_0[4]), "r"(_fp8_0[5]), "r"(_fp8_0[6]), "r"(_fp8_0[7]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (my_row * 64 + 32 ^ (my_row * 64 + 32 >> 7 & 3) << 4))), "r"(_fp8_0[8]), "r"(_fp8_0[9]), "r"(_fp8_0[10]), "r"(_fp8_0[11]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (my_row * 64 + 48 ^ (my_row * 64 + 48 >> 7 & 3) << 4))), "r"(_fp8_0[12]), "r"(_fp8_0[13]), "r"(_fp8_0[14]), "r"(_fp8_0[15]) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + (p_stage_s) * 8);
                    p_stage_s += 1;
                    if (p_stage_s == 2) { p_stage_s = 0; p_phase_s ^= 1; }
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _rcp_0 = approx_rcp(row_sum);
                float inv_sum = ((row_sum > 0.0f) ? _rcp_0 : 0.0f);
                int out_row = (q_start_s + token_s) * num_heads + head_s;
                int o_base = out_row * 512 + v_half_s * 256;
                int part_base = (out_row * max_splits + split_s) * 512 + v_half_s * 256;
                float _log2_0;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(row_sum));
                float lse2_s = row_max + _log2_0 - 8.0f;
                #pragma unroll
                for (int col = 0; col < 256; col += 32) {
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(taddr + (unsigned int)lane_base + (unsigned int)TMEM_ACC_OFFSET + (unsigned int)col));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (row_valid != 0) {
                        if (direct_out != 0) {
                            {
                                const float2 _prescale2_7 = {inv_sum, inv_sum};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_0[0])[_ps], _prescale2_7);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 16; _ps++)
                                    _tmem_load_0[0 + _ps] *= inv_sum;
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
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (o_base + col)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (o_base + col)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                            }
                            {
                                const float2 _prescale2_8 = {inv_sum, inv_sum};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_0[16])[_ps], _prescale2_8);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 16; _ps++)
                                    _tmem_load_0[16 + _ps] *= inv_sum;
                                #endif
                                __nv_bfloat162 _pk[8];
                                _pk[0] = __floats2bfloat162_rn(_tmem_load_0[16 + 0], _tmem_load_0[16 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_tmem_load_0[16 + 2], _tmem_load_0[16 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_tmem_load_0[16 + 4], _tmem_load_0[16 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_tmem_load_0[16 + 6], _tmem_load_0[16 + 7]);
                                _pk[4] = __floats2bfloat162_rn(_tmem_load_0[16 + 8], _tmem_load_0[16 + 9]);
                                _pk[5] = __floats2bfloat162_rn(_tmem_load_0[16 + 10], _tmem_load_0[16 + 11]);
                                _pk[6] = __floats2bfloat162_rn(_tmem_load_0[16 + 12], _tmem_load_0[16 + 13]);
                                _pk[7] = __floats2bfloat162_rn(_tmem_load_0[16 + 14], _tmem_load_0[16 + 15]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (o_base + col + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (o_base + col + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                            }
                        } else {
                            const float2 _scale2_9 = {inv_sum, inv_sum};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_9);
                            #pragma unroll
                            for (int vec = 0; vec < 32; vec += 4) {
                                {
                                    float4 _v4 = make_float4(_tmem_load_0[vec + 0], _tmem_load_0[vec + 1], _tmem_load_0[vec + 2], _tmem_load_0[vec + 3]);
                                    *reinterpret_cast<float4*>(partial_o + (part_base + col + vec) + 0) = _v4;
                                }
                            }
                        }
                    }
                }
                if ((row_valid & (int)(v_half_s == 0)) != 0) {
                    if (direct_out != 0) {
                        LSE[out_row] = lse2_s * 0.6931471805599453f;
                    } else {
                        partial_lse[out_row * max_splits + split_s] = lse2_s;
                        #pragma unroll 1
                        for (int extra = num_splits_s; extra < max_splits; extra++) {
                            partial_lse[out_row * max_splits + extra] = -CAKE_INF;
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_empty_addr);
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // correction_main
            int quadrant_c = make_warp_uniform(warp % 4);
            int my_row_c = quadrant_c * 32 + lane;
            int lane_base_c = quadrant_c * 32 << 16;
            int p_stage_c = 0;
            int p_phase_c = 0;
            int raw_stage_c = 0;
            int raw_phase_c = 0;
            int sf_stage_c = 0;
            int sf_phase_c = 1;
            int unit_c = cluster_id;
            int item_lo_c = unit_first[unit_c] + cta_rank;
            int item_hi_c = unit_first[unit_c + 1];
            int tr_c = blockIdx.x == 0 && warp == 4 && lane == 0;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_pv_done_0 = 0;
            #pragma unroll 1
            for (int item_c = item_lo_c; item_c < item_hi_c; item_c += 2) {
                int tr_item_c = tr_c & (int)(item_c == item_lo_c);
                int tile_start_c = work_table[item_c * 8 + 3];
                int tile_end_c = work_table[item_c * 8 + 4];
                int num_tiles_c = tile_end_c - tile_start_c;
                {
                    mbarrier_wait_cluster_hint(raw_full_addr + (raw_stage_c) * 8, raw_phase_c, 10000000);
                    mbarrier_wait(sf_empty_addr + (sf_stage_c) * 8, sf_phase_c);
                    int sfb_base_c = taddr + (unsigned int)lane_base_c + 416 + (unsigned int)(sf_stage_c * 32);
                    unsigned int k_words_c[8];
                    int ks_addr_c = smem_ks_addr + (unsigned int)(raw_stage_c * 18432) + (unsigned int)(lane * 32);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(0) + 3]))
                        : "r"(ks_addr_c));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(4) + 3]))
                        : "r"(ks_addr_c + 16));
                    {
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c), "r"(k_words_c[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 4), "r"((k_words_c + 1)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 8), "r"((k_words_c + 2)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 8 + 4), "r"((k_words_c + 3)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 16), "r"((k_words_c + 4)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 16 + 4), "r"((k_words_c + 5)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 24), "r"((k_words_c + 6)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 24 + 4), "r"((k_words_c + 7)[0]));
                    }
                    unsigned int k_words_c_0[8];
                    int ks_addr_c_1 = smem_ks_addr + (unsigned int)(raw_stage_c * 18432) + (unsigned int)((32 + lane) * 32);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[(0) + 3]))
                        : "r"(ks_addr_c_1));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[(4) + 3]))
                        : "r"(ks_addr_c_1 + 16));
                    {
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 1), "r"(k_words_c_0[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 4 + 1), "r"((k_words_c_0 + 1)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 8 + 1), "r"((k_words_c_0 + 2)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 8 + 4 + 1), "r"((k_words_c_0 + 3)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 16 + 1), "r"((k_words_c_0 + 4)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 16 + 4 + 1), "r"((k_words_c_0 + 5)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 24 + 1), "r"((k_words_c_0 + 6)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c + 24 + 4 + 1), "r"((k_words_c_0 + 7)[0]));
                    }
                    mbarrier_arrive(raw_empty_addr + (raw_stage_c) * 8);
                    raw_stage_c += 1;
                    if (raw_stage_c == 6) { raw_stage_c = 0; raw_phase_c ^= 1; }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(sf_full_addr + (sf_stage_c) * 8);
                    sf_stage_c += 1;
                    if (sf_stage_c == 2) { sf_stage_c = 0; sf_phase_c ^= 1; }
                    if (num_tiles_c > 1) {
                        mbarrier_wait_cluster_hint(raw_full_addr + (raw_stage_c) * 8, raw_phase_c, 10000000);
                        mbarrier_wait(sf_empty_addr + (sf_stage_c) * 8, sf_phase_c);
                        int sfb_base_c_0 = taddr + (unsigned int)lane_base_c + 416 + (unsigned int)(sf_stage_c * 32);
                        unsigned int k_words_c_1[8];
                        int ks_addr_c_2 = smem_ks_addr + (unsigned int)(raw_stage_c * 18432) + (unsigned int)(lane * 32);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(0) + 3]))
                            : "r"(ks_addr_c_2));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(4) + 3]))
                            : "r"(ks_addr_c_2 + 16));
                        {
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0), "r"(k_words_c_1[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 4), "r"((k_words_c_1 + 1)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 8), "r"((k_words_c_1 + 2)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 8 + 4), "r"((k_words_c_1 + 3)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 16), "r"((k_words_c_1 + 4)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 16 + 4), "r"((k_words_c_1 + 5)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 24), "r"((k_words_c_1 + 6)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 24 + 4), "r"((k_words_c_1 + 7)[0]));
                        }
                        unsigned int k_words_c_3[8];
                        int ks_addr_c_4 = smem_ks_addr + (unsigned int)(raw_stage_c * 18432) + (unsigned int)((32 + lane) * 32);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[(0) + 3]))
                            : "r"(ks_addr_c_4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[(4) + 3]))
                            : "r"(ks_addr_c_4 + 16));
                        {
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 1), "r"(k_words_c_3[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 4 + 1), "r"((k_words_c_3 + 1)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 8 + 1), "r"((k_words_c_3 + 2)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 8 + 4 + 1), "r"((k_words_c_3 + 3)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 16 + 1), "r"((k_words_c_3 + 4)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 16 + 4 + 1), "r"((k_words_c_3 + 5)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 24 + 1), "r"((k_words_c_3 + 6)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0 + 24 + 4 + 1), "r"((k_words_c_3 + 7)[0]));
                        }
                        mbarrier_arrive(raw_empty_addr + (raw_stage_c) * 8);
                        raw_stage_c += 1;
                        if (raw_stage_c == 6) { raw_stage_c = 0; raw_phase_c ^= 1; }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(sf_full_addr + (sf_stage_c) * 8);
                        sf_stage_c += 1;
                        if (sf_stage_c == 2) { sf_stage_c = 0; sf_phase_c ^= 1; }
                    }
                }
                #pragma unroll 1
                for (int it_c = 0; it_c < num_tiles_c; it_c++) {
                    {
                        if (num_tiles_c > it_c + 2) {
                            mbarrier_wait_cluster_hint(raw_full_addr + (raw_stage_c) * 8, raw_phase_c, 10000000);
                            mbarrier_wait(sf_empty_addr + (sf_stage_c) * 8, sf_phase_c);
                            int sfb_base_c_1 = taddr + (unsigned int)lane_base_c + 416 + (unsigned int)(sf_stage_c * 32);
                            unsigned int k_words_c_2[8];
                            int ks_addr_c_3 = smem_ks_addr + (unsigned int)(raw_stage_c * 18432) + (unsigned int)(lane * 32);
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[(0) + 3]))
                                : "r"(ks_addr_c_3));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[(4) + 3]))
                                : "r"(ks_addr_c_3 + 16));
                            {
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1), "r"(k_words_c_2[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 4), "r"((k_words_c_2 + 1)[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 8), "r"((k_words_c_2 + 2)[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 8 + 4), "r"((k_words_c_2 + 3)[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 16), "r"((k_words_c_2 + 4)[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 16 + 4), "r"((k_words_c_2 + 5)[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 24), "r"((k_words_c_2 + 6)[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 24 + 4), "r"((k_words_c_2 + 7)[0]));
                            }
                            unsigned int k_words_c_0_1[8];
                            int ks_addr_c_1_1 = smem_ks_addr + (unsigned int)(raw_stage_c * 18432) + (unsigned int)((32 + lane) * 32);
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0_1[(0) + 3]))
                                : "r"(ks_addr_c_1_1));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0_1[(4) + 3]))
                                : "r"(ks_addr_c_1_1 + 16));
                            {
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 1), "r"(k_words_c_0_1[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 4 + 1), "r"((k_words_c_0_1 + 1)[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 8 + 1), "r"((k_words_c_0_1 + 2)[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 8 + 4 + 1), "r"((k_words_c_0_1 + 3)[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 16 + 1), "r"((k_words_c_0_1 + 4)[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 16 + 4 + 1), "r"((k_words_c_0_1 + 5)[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 24 + 1), "r"((k_words_c_0_1 + 6)[0]));
                                asm volatile(
                                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                    " [%0], {%1};"
                                    :: "r"(sfb_base_c_1 + 24 + 4 + 1), "r"((k_words_c_0_1 + 7)[0]));
                            }
                            mbarrier_arrive(raw_empty_addr + (raw_stage_c) * 8);
                            raw_stage_c += 1;
                            if (raw_stage_c == 6) { raw_stage_c = 0; raw_phase_c ^= 1; }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(sf_full_addr + (sf_stage_c) * 8);
                            sf_stage_c += 1;
                            if (sf_stage_c == 2) { sf_stage_c = 0; sf_phase_c ^= 1; }
                        }
                    }
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    float alpha_c = row_state[my_row_c];
                    mbarrier_arrive(corr_empty_addr);
                    if (it_c > 0) {
                        mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                        _phase_pv_done_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _vote_0 = __all_sync(0xFFFFFFFF, alpha_c == 1.0f);
                        int skip_rescale = _vote_0;
                        if (skip_rescale == 0) {
                            #pragma unroll
                            for (int col_1 = 0; col_1 < 256; col_1 += 32) {
                                int o_addr = taddr + (unsigned int)lane_base_c + (unsigned int)TMEM_ACC_OFFSET + (unsigned int)col_1;
                                float _tmem_load_1[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                                    : "r"(o_addr));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                const float2 _scale2_0 = {alpha_c, alpha_c};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_0);
                                tmem_st_x32_f32(o_addr, _tmem_load_1);
                            }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + (p_stage_c) * 8);
                    p_stage_c += 1;
                    if (p_stage_c == 2) { p_stage_c = 0; p_phase_c ^= 1; }
                }
                mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                _phase_pv_done_0 ^= 1;
            }
        }
    }
    // ---- Role: transform ----
    if (warp >= 8 && warp <= 13) {
        { // transform_main
            int raw_stage_t = 0;
            int raw_phase_t = 0;
            int v_stage_t = 0;
            int v_phase_t = 1;
            int role_tid = (warp - 8) * 32 + lane;
            int quadrant_t = make_warp_uniform(warp % 4);
            int lane_base_t = quadrant_t * 32 << 16;
            int sf_stage_t = 0;
            int sf_phase_t = 1;
            int unit_t = cluster_id;
            int item_lo_t = unit_first[unit_t] + cta_rank;
            int item_hi_t = unit_first[unit_t + 1];
            int tr_t = blockIdx.x == 0 && warp == 8 && lane == 0;
            #pragma unroll 1
            for (int item_t = item_lo_t; item_t < item_hi_t; item_t += 2) {
                int tr_item_t = tr_t & (int)(item_t == item_lo_t);
                int v_half_t = work_table[item_t * 8 + 2];
                int tile_start_t = work_table[item_t * 8 + 3];
                int tile_end_t = work_table[item_t * 8 + 4];
                int num_tiles_t = tile_end_t - tile_start_t;
                #pragma unroll 1
                for (int it_t = 0; it_t < num_tiles_t; it_t++) {
                    mbarrier_wait_cluster_hint(raw_full_addr + (raw_stage_t) * 8, raw_phase_t, 10000000);
                    mbarrier_wait(v_empty_addr + (v_stage_t) * 8, v_phase_t);
                    sf_stage_t += 1;
                    if (sf_stage_t == 2) { sf_stage_t = 0; sf_phase_t ^= 1; }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int raw_base_t = smem_k0_addr + (unsigned int)(raw_stage_t * 18432);
                    int scale_off_t = raw_stage_t * 18432 + 16384;
                    int v_base_t = smem_v_addr + (unsigned int)(v_stage_t * 16384);
                    #pragma unroll
                    for (int rnd = 0; rnd < 2; rnd++) {
                        int pair_t = role_tid + rnd * 192;
                        {
                            int token = pair_t / 8;
                            int pair_in_token = pair_t - token * 8;
                            int atom = v_half_t * 2 + pair_in_token / 4;
                            int chunk = pair_in_token % 4 ^ token >> 1 & 3;
                            unsigned int packed[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                                : "r"(raw_base_t + atom * 4096 + token * 64 + chunk * 16));
                            int scale_index = scale_off_t + token * 32 + v_half_t * 16 + pair_in_token * 2;
                            unsigned int scale0 = smem_raw_flat[scale_index];
                            unsigned int scale1 = smem_raw_flat[scale_index + 1];
                            unsigned int converted[8];
                            {
                                uint16_t _qmul4_pair_0_sb0 = (uint16_t)((scale0) & 0xFFu);
                                uint16_t _qmul4_pair_0_s0 = (uint16_t)(_qmul4_pair_0_sb0 | (_qmul4_pair_0_sb0 << 8));
                                uint16_t _qmul4_pair_0_sb1 = (uint16_t)((scale1) & 0xFFu);
                                uint16_t _qmul4_pair_0_s1 = (uint16_t)(_qmul4_pair_0_sb1 | (_qmul4_pair_0_sb1 << 8));
                                asm volatile("{\n"
                                    ".reg .b8 b0, b1, b2, b3;\n"
                                    ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                                    ".reg .b16 e0, e1, e2, e3;\n"
                                    "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                                    "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                                    "mov.b32 {b0, b1, b2, b3}, %10;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                    "mul.rn.f16x2 ah0, ah0, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                    "mul.rn.f16x2 ah1, ah1, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                    "mul.rn.f16x2 ah2, ah2, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                    "mul.rn.f16x2 ah3, ah3, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                    "mov.b32 %0, {e0, e1};\n"
                                    "mov.b32 %1, {e2, e3};\n"
                                    "mov.b32 {b0, b1, b2, b3}, %11;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                    "mul.rn.f16x2 ah0, ah0, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                    "mul.rn.f16x2 ah1, ah1, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                    "mul.rn.f16x2 ah2, ah2, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                    "mul.rn.f16x2 ah3, ah3, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                    "mov.b32 %2, {e0, e1};\n"
                                    "mov.b32 %3, {e2, e3};\n"
                                    "mov.b32 {b0, b1, b2, b3}, %12;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                    "mul.rn.f16x2 ah0, ah0, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                    "mul.rn.f16x2 ah1, ah1, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                    "mul.rn.f16x2 ah2, ah2, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                    "mul.rn.f16x2 ah3, ah3, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                    "mov.b32 %4, {e0, e1};\n"
                                    "mov.b32 %5, {e2, e3};\n"
                                    "mov.b32 {b0, b1, b2, b3}, %13;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                    "mul.rn.f16x2 ah0, ah0, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                    "mul.rn.f16x2 ah1, ah1, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                    "mul.rn.f16x2 ah2, ah2, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                    "mul.rn.f16x2 ah3, ah3, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                    "mov.b32 %6, {e0, e1};\n"
                                    "mov.b32 %7, {e2, e3};\n"
                                    "}\n"
                                    : "=&r"(converted[0]), "=&r"(converted[1]), "=&r"(converted[2]), "=&r"(converted[3]), "=&r"(converted[4]), "=&r"(converted[5]), "=&r"(converted[6]), "=&r"(converted[7])
                                    : "h"(_qmul4_pair_0_s0), "h"(_qmul4_pair_0_s1), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]));
                            }
                            int panel = pair_in_token / 4;
                            int col_bytes = (pair_in_token - panel * 4) * 32;
                            int panel_base = v_base_t + panel * 8192;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((panel_base + (token * 128 + col_bytes ^ (token * 128 + col_bytes >> 7 & 7) << 4))), "r"(converted[0]), "r"(converted[1]), "r"(converted[2]), "r"(converted[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((panel_base + (token * 128 + (col_bytes + 16) ^ (token * 128 + (col_bytes + 16) >> 7 & 7) << 4))), "r"(converted[4]), "r"(converted[5]), "r"(converted[6]), "r"(converted[7]) : "memory");
                        }
                    }
                    if (role_tid < 128) {
                        int tail_pair_t = role_tid + 384;
                        {
                            int token_1 = tail_pair_t / 8;
                            int pair_in_token_1 = tail_pair_t - token_1 * 8;
                            int atom_1 = v_half_t * 2 + pair_in_token_1 / 4;
                            int chunk_1 = pair_in_token_1 % 4 ^ token_1 >> 1 & 3;
                            unsigned int packed_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 3]))
                                : "r"(raw_base_t + atom_1 * 4096 + token_1 * 64 + chunk_1 * 16));
                            int scale_index_1 = scale_off_t + token_1 * 32 + v_half_t * 16 + pair_in_token_1 * 2;
                            unsigned int scale0_1 = smem_raw_flat[scale_index_1];
                            unsigned int scale1_1 = smem_raw_flat[scale_index_1 + 1];
                            unsigned int converted_1[8];
                            {
                                uint16_t _qmul4_pair_1_sb0 = (uint16_t)((scale0_1) & 0xFFu);
                                uint16_t _qmul4_pair_1_s0 = (uint16_t)(_qmul4_pair_1_sb0 | (_qmul4_pair_1_sb0 << 8));
                                uint16_t _qmul4_pair_1_sb1 = (uint16_t)((scale1_1) & 0xFFu);
                                uint16_t _qmul4_pair_1_s1 = (uint16_t)(_qmul4_pair_1_sb1 | (_qmul4_pair_1_sb1 << 8));
                                asm volatile("{\n"
                                    ".reg .b8 b0, b1, b2, b3;\n"
                                    ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                                    ".reg .b16 e0, e1, e2, e3;\n"
                                    "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                                    "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                                    "mov.b32 {b0, b1, b2, b3}, %10;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                    "mul.rn.f16x2 ah0, ah0, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                    "mul.rn.f16x2 ah1, ah1, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                    "mul.rn.f16x2 ah2, ah2, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                    "mul.rn.f16x2 ah3, ah3, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                    "mov.b32 %0, {e0, e1};\n"
                                    "mov.b32 %1, {e2, e3};\n"
                                    "mov.b32 {b0, b1, b2, b3}, %11;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                    "mul.rn.f16x2 ah0, ah0, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                    "mul.rn.f16x2 ah1, ah1, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                    "mul.rn.f16x2 ah2, ah2, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                    "mul.rn.f16x2 ah3, ah3, sh0;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                    "mov.b32 %2, {e0, e1};\n"
                                    "mov.b32 %3, {e2, e3};\n"
                                    "mov.b32 {b0, b1, b2, b3}, %12;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                    "mul.rn.f16x2 ah0, ah0, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                    "mul.rn.f16x2 ah1, ah1, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                    "mul.rn.f16x2 ah2, ah2, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                    "mul.rn.f16x2 ah3, ah3, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                    "mov.b32 %4, {e0, e1};\n"
                                    "mov.b32 %5, {e2, e3};\n"
                                    "mov.b32 {b0, b1, b2, b3}, %13;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                    "mul.rn.f16x2 ah0, ah0, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                    "mul.rn.f16x2 ah1, ah1, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                    "mul.rn.f16x2 ah2, ah2, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                    "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                    "mul.rn.f16x2 ah3, ah3, sh1;\n"
                                    "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                    "mov.b32 %6, {e0, e1};\n"
                                    "mov.b32 %7, {e2, e3};\n"
                                    "}\n"
                                    : "=&r"(converted_1[0]), "=&r"(converted_1[1]), "=&r"(converted_1[2]), "=&r"(converted_1[3]), "=&r"(converted_1[4]), "=&r"(converted_1[5]), "=&r"(converted_1[6]), "=&r"(converted_1[7])
                                    : "h"(_qmul4_pair_1_s0), "h"(_qmul4_pair_1_s1), "r"(packed_1[0]), "r"(packed_1[1]), "r"(packed_1[2]), "r"(packed_1[3]));
                            }
                            int panel_1 = pair_in_token_1 / 4;
                            int col_bytes_1 = (pair_in_token_1 - panel_1 * 4) * 32;
                            int panel_base_1 = v_base_t + panel_1 * 8192;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((panel_base_1 + (token_1 * 128 + col_bytes_1 ^ (token_1 * 128 + col_bytes_1 >> 7 & 7) << 4))), "r"(converted_1[0]), "r"(converted_1[1]), "r"(converted_1[2]), "r"(converted_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((panel_base_1 + (token_1 * 128 + (col_bytes_1 + 16) ^ (token_1 * 128 + (col_bytes_1 + 16) >> 7 & 7) << 4))), "r"(converted_1[4]), "r"(converted_1[5]), "r"(converted_1[6]), "r"(converted_1[7]) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(v_full_addr + (v_stage_t) * 8);
                    mbarrier_arrive(raw_empty_addr + (raw_stage_t) * 8);
                    raw_stage_t += 1;
                    if (raw_stage_t == 6) { raw_stage_t = 0; raw_phase_t ^= 1; }
                    v_stage_t += 1;
                    if (v_stage_t == 4) { v_stage_t = 0; v_phase_t ^= 1; }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 14) {
        { // load_main
            int raw_stage_l = 0;
            int raw_phase_l = 1;
            int unit_l = cluster_id;
            int item_lo_l = unit_first[unit_l] + cta_rank;
            int item_hi_l = unit_first[unit_l + 1];
            int tr_l = blockIdx.x == 0 && lane == 0;
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (int item_l = item_lo_l; item_l < item_hi_l; item_l += 2) {
                int tr_item_l = tr_l & (int)(item_l == item_lo_l);
                int b_l = work_table[item_l * 8];
                int m_tile_l = work_table[item_l * 8 + 1];
                int tile_start_l = work_table[item_l * 8 + 3];
                int tile_end_l = work_table[item_l * 8 + 4];
                int num_tiles_l = tile_end_l - tile_start_l;
                int q_row0 = q_indptr[b_l] * num_heads + m_tile_l * 128;
                int first_page_l = page_table[b_l * max_pages];
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 36864);
                    tma_2d_gmem2smem(smem_q0_addr, (&Q), 0, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_q1_addr, (&Q), 64, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_q2_addr, (&Q), 128, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_q3_addr, (&Q), 192, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_qs_addr, (&QS), 0, q_row0, q_full_addr);
                }
                #pragma unroll 1
                for (int it_l = 0; it_l < num_tiles_l; it_l++) {
                    int batch_pos = it_l % 32;
                    if (batch_pos == 0) {
                        int fetch_tile = tile_start_l + it_l + lane;
                        int fetched = ((fetch_tile < tile_end_l) ? page_table[b_l * max_pages + fetch_tile] : 0);
                        smem_pages[lane] = fetched;
                        __syncwarp();
                    }
                    int page_l = smem_pages[batch_pos];
                    int kv_row0 = page_l * 64;
                    mbarrier_wait(raw_empty_addr + (raw_stage_l) * 8, raw_phase_l);
                    if (cta_rank == 0) {
                        int peer_phase_l = 1 - raw_phase_l;
                        mbarrier_wait_cluster_hint(peer_empty_addr + (raw_stage_l) * 8, peer_phase_l, 10000000);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(raw_full_addr + (raw_stage_l) * 8, 18432);
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_k0_addr + (unsigned int)(raw_stage_l * 18432)), "l"((&KV)), "r"(0), "r"(kv_row0),
                                   "r"(raw_full_addr + (raw_stage_l) * 8), "h"((uint16_t)(3)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_k1_addr + (unsigned int)(raw_stage_l * 18432)), "l"((&KV)), "r"(64), "r"(kv_row0),
                                   "r"(raw_full_addr + (raw_stage_l) * 8), "h"((uint16_t)(3)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_k2_addr + (unsigned int)(raw_stage_l * 18432)), "l"((&KV)), "r"(128), "r"(kv_row0),
                                   "r"(raw_full_addr + (raw_stage_l) * 8), "h"((uint16_t)(3)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_k3_addr + (unsigned int)(raw_stage_l * 18432)), "l"((&KV)), "r"(192), "r"(kv_row0),
                                   "r"(raw_full_addr + (raw_stage_l) * 8), "h"((uint16_t)(3)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_ks_addr + (unsigned int)(raw_stage_l * 18432)), "l"((&KVS)), "r"(0), "r"(kv_row0),
                                   "r"(raw_full_addr + (raw_stage_l) * 8), "h"((uint16_t)(3)) : "memory");
                        }
                    } else {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(raw_full_addr + (raw_stage_l) * 8, 18432);
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(peer_empty_addr + raw_stage_l * 8), "r"(0) : "memory");
                        }
                    }
                    raw_stage_l += 1;
                    if (raw_stage_l == 6) { raw_stage_l = 0; raw_phase_l ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 15) {
        { // mma_main
            int raw_stage_m = 0;
            int raw_phase_m = 0;
            int v_stage_m = 0;
            int v_phase_m = 0;
            int p_stage_m = 0;
            int p_phase_m = 0;
            int s_stage_m = 0;
            int s_phase_m = 1;
            int sf_stage_m = 0;
            int sf_phase_m = 0;
            int unit_m = cluster_id;
            int item_lo_m = unit_first[unit_m] + cta_rank;
            int item_hi_m = unit_first[unit_m + 1];
            int tr_m = blockIdx.x == 0 && lane == 0;
            unsigned int _phase_q_full_0_1 = 0;
            unsigned int _phase_sfa_full_0 = 0;
            unsigned int _phase_o_empty_0 = 1;
            #pragma unroll 1
            for (int item_m = item_lo_m; item_m < item_hi_m; item_m += 2) {
                int tr_item_m = tr_m & (int)(item_m == item_lo_m);
                int tile_start_m = work_table[item_m * 8 + 3];
                int tile_end_m = work_table[item_m * 8 + 4];
                int num_tiles_m = tile_end_m - tile_start_m;
                mbarrier_wait(q_full_addr, _phase_q_full_0_1);
                _phase_q_full_0_1 ^= 1;
                mbarrier_wait(sfa_full_addr, _phase_sfa_full_0);
                _phase_sfa_full_0 ^= 1;
                mbarrier_wait_cluster_hint(raw_full_addr + (raw_stage_m) * 8, raw_phase_m, 10000000);
                mbarrier_wait(sf_full_addr + (sf_stage_m) * 8, sf_phase_m);
                mbarrier_wait(s_empty_addr + (s_stage_m) * 8, s_phase_m);
                asm volatile("tcgen05.fence::after_thread_sync;");
                {
                    if (s_stage_m == 0) {
                        int _mma_a_lo_0 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_0 = make_warp_uniform((((smem_k0_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa0 + 0, tmem_tmem_sfb0 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 2, b_desc + 2,
                                    0x8100480U, tmem_tmem_sfa0 + 4, tmem_tmem_sfb0 + 4, 1);
                            }
                        }
                        int _mma_a_lo_1 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_1 = make_warp_uniform((((smem_k1_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa1 + 0, tmem_tmem_sfb1 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 2, b_desc + 2,
                                    0x8100480U, tmem_tmem_sfa1 + 4, tmem_tmem_sfb1 + 4, 1);
                            }
                        }
                        int _mma_a_lo_2 = make_warp_uniform(((smem_q2_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_2 = make_warp_uniform((((smem_k2_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa2 + 0, tmem_tmem_sfb2 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 2, b_desc + 2,
                                    0x8100480U, tmem_tmem_sfa2 + 4, tmem_tmem_sfb2 + 4, 1);
                            }
                        }
                        int _mma_a_lo_3 = make_warp_uniform(((smem_q3_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_3 = make_warp_uniform((((smem_k3_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa3 + 0, tmem_tmem_sfb3 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 2, b_desc + 2,
                                    0x8100480U, tmem_tmem_sfa3 + 4, tmem_tmem_sfb3 + 4, 1);
                            }
                        }
                    } else {
                        int _mma_a_lo_4 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_4 = make_warp_uniform((((smem_k0_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa0 + 0, tmem_tmem_sfb4 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 2, b_desc + 2,
                                    0x8100480U, tmem_tmem_sfa0 + 4, tmem_tmem_sfb4 + 4, 1);
                            }
                        }
                        int _mma_a_lo_5 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_5 = make_warp_uniform((((smem_k1_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa1 + 0, tmem_tmem_sfb5 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 2, b_desc + 2,
                                    0x8100480U, tmem_tmem_sfa1 + 4, tmem_tmem_sfb5 + 4, 1);
                            }
                        }
                        int _mma_a_lo_6 = make_warp_uniform(((smem_q2_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_6 = make_warp_uniform((((smem_k2_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa2 + 0, tmem_tmem_sfb6 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 2, b_desc + 2,
                                    0x8100480U, tmem_tmem_sfa2 + 4, tmem_tmem_sfb6 + 4, 1);
                            }
                        }
                        int _mma_a_lo_7 = make_warp_uniform(((smem_q3_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_7 = make_warp_uniform((((smem_k3_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa3 + 0, tmem_tmem_sfb7 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 2, b_desc + 2,
                                    0x8100480U, tmem_tmem_sfa3 + 4, tmem_tmem_sfb7 + 4, 1);
                            }
                        }
                    }
                }
                elect_commit(s_full_addr + (s_stage_m) * 8);
                elect_commit(sf_empty_addr + (sf_stage_m) * 8);
                elect_commit(raw_empty_addr + (raw_stage_m) * 8);
                raw_stage_m += 1;
                if (raw_stage_m == 6) { raw_stage_m = 0; raw_phase_m ^= 1; }
                sf_stage_m += 1;
                if (sf_stage_m == 2) { sf_stage_m = 0; sf_phase_m ^= 1; }
                s_stage_m += 1;
                if (s_stage_m == 2) { s_stage_m = 0; s_phase_m ^= 1; }
                int first_pv = 1;
                mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                _phase_o_empty_0 ^= 1;
                #pragma unroll 1
                for (int it_m = 0; it_m < num_tiles_m; it_m++) {
                    if (num_tiles_m > it_m + 1) {
                        mbarrier_wait_cluster_hint(raw_full_addr + (raw_stage_m) * 8, raw_phase_m, 10000000);
                        mbarrier_wait(sf_full_addr + (sf_stage_m) * 8, sf_phase_m);
                        mbarrier_wait(s_empty_addr + (s_stage_m) * 8, s_phase_m);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        {
                            if (s_stage_m == 0) {
                                int _mma_a_lo_8 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                                int _mma_b_lo_8 = make_warp_uniform((((smem_k0_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_8) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_8) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 0, b_desc + 0,
                                            0x8100480U, tmem_tmem_sfa0 + 0, tmem_tmem_sfb0 + 0, 0);
                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 2, b_desc + 2,
                                            0x8100480U, tmem_tmem_sfa0 + 4, tmem_tmem_sfb0 + 4, 1);
                                    }
                                }
                                int _mma_a_lo_9 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                                int _mma_b_lo_9 = make_warp_uniform((((smem_k1_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_9) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_9) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 0, b_desc + 0,
                                            0x8100480U, tmem_tmem_sfa1 + 0, tmem_tmem_sfb1 + 0, 1);
                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 2, b_desc + 2,
                                            0x8100480U, tmem_tmem_sfa1 + 4, tmem_tmem_sfb1 + 4, 1);
                                    }
                                }
                                int _mma_a_lo_10 = make_warp_uniform(((smem_q2_addr) >> 4) & 0x3FFF);
                                int _mma_b_lo_10 = make_warp_uniform((((smem_k2_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_10) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_10) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 0, b_desc + 0,
                                            0x8100480U, tmem_tmem_sfa2 + 0, tmem_tmem_sfb2 + 0, 1);
                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 2, b_desc + 2,
                                            0x8100480U, tmem_tmem_sfa2 + 4, tmem_tmem_sfb2 + 4, 1);
                                    }
                                }
                                int _mma_a_lo_11 = make_warp_uniform(((smem_q3_addr) >> 4) & 0x3FFF);
                                int _mma_b_lo_11 = make_warp_uniform((((smem_k3_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_11) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_11) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 0, b_desc + 0,
                                            0x8100480U, tmem_tmem_sfa3 + 0, tmem_tmem_sfb3 + 0, 1);
                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores0, a_desc + 2, b_desc + 2,
                                            0x8100480U, tmem_tmem_sfa3 + 4, tmem_tmem_sfb3 + 4, 1);
                                    }
                                }
                            } else {
                                int _mma_a_lo_12 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                                int _mma_b_lo_12 = make_warp_uniform((((smem_k0_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_12) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_12) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 0, b_desc + 0,
                                            0x8100480U, tmem_tmem_sfa0 + 0, tmem_tmem_sfb4 + 0, 0);
                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 2, b_desc + 2,
                                            0x8100480U, tmem_tmem_sfa0 + 4, tmem_tmem_sfb4 + 4, 1);
                                    }
                                }
                                int _mma_a_lo_13 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                                int _mma_b_lo_13 = make_warp_uniform((((smem_k1_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_13) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_13) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 0, b_desc + 0,
                                            0x8100480U, tmem_tmem_sfa1 + 0, tmem_tmem_sfb5 + 0, 1);
                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 2, b_desc + 2,
                                            0x8100480U, tmem_tmem_sfa1 + 4, tmem_tmem_sfb5 + 4, 1);
                                    }
                                }
                                int _mma_a_lo_14 = make_warp_uniform(((smem_q2_addr) >> 4) & 0x3FFF);
                                int _mma_b_lo_14 = make_warp_uniform((((smem_k2_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_14) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_14) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 0, b_desc + 0,
                                            0x8100480U, tmem_tmem_sfa2 + 0, tmem_tmem_sfb6 + 0, 1);
                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 2, b_desc + 2,
                                            0x8100480U, tmem_tmem_sfa2 + 4, tmem_tmem_sfb6 + 4, 1);
                                    }
                                }
                                int _mma_a_lo_15 = make_warp_uniform(((smem_q3_addr) >> 4) & 0x3FFF);
                                int _mma_b_lo_15 = make_warp_uniform((((smem_k3_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1152);
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_15) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_15) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 0, b_desc + 0,
                                            0x8100480U, tmem_tmem_sfa3 + 0, tmem_tmem_sfb7 + 0, 1);
                                        tcgen05_mma_mxf4nvf4_bs(tmem_scores1, a_desc + 2, b_desc + 2,
                                            0x8100480U, tmem_tmem_sfa3 + 4, tmem_tmem_sfb7 + 4, 1);
                                    }
                                }
                            }
                        }
                        elect_commit(s_full_addr + (s_stage_m) * 8);
                        elect_commit(sf_empty_addr + (sf_stage_m) * 8);
                        elect_commit(raw_empty_addr + (raw_stage_m) * 8);
                        raw_stage_m += 1;
                        if (raw_stage_m == 6) { raw_stage_m = 0; raw_phase_m ^= 1; }
                        sf_stage_m += 1;
                        if (sf_stage_m == 2) { sf_stage_m = 0; sf_phase_m ^= 1; }
                        s_stage_m += 1;
                        if (s_stage_m == 2) { s_stage_m = 0; s_phase_m ^= 1; }
                    } else {
                        elect_commit(q_empty_addr);
                    }
                    int first_pv_flag = first_pv;
                    mbarrier_wait(v_full_addr + (v_stage_m) * 8, v_phase_m);
                    mbarrier_wait(p_full_addr + (p_stage_m) * 8, p_phase_m);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    {
                        int _mma_a_lo_16 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) + (p_stage_m) * 512);
                        int _mma_b_lo_16 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage_m) * 1024);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_16), "r"(_mma_b_lo_16), "r"(tmem_acc), "r"(((first_pv_flag) ? 0 : 1)));
                    }
                    elect_commit(pv_done_addr);
                    elect_commit(p_empty_addr + (p_stage_m) * 8);
                    elect_commit(v_empty_addr + (v_stage_m) * 8);
                    p_stage_m += 1;
                    if (p_stage_m == 2) { p_stage_m = 0; p_phase_m ^= 1; }
                    v_stage_m += 1;
                    if (v_stage_m == 4) { v_stage_m = 0; v_phase_m ^= 1; }
                    first_pv = 0;
                }
                elect_commit(o_full_addr);
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
