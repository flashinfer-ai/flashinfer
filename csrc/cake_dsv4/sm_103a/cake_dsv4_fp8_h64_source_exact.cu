/*
 * Copyright (c) 2023 by FlashInfer team.
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
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_OFFSET 0
#define NUM_Q_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 4
#define NUM_PAGE_PIPE_STAGES 6
#define NUM_SCORE_PIPE_STAGES 2
#define NUM_STATS_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 2
#define NUM_THROTTLE_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 0
#define SMEM_SMEM_Q_STAGE_BYTES 32768
#define SMEM_SMEM_Q_STRIDE 32768
#define SMEM_SMEM_K_OFF 65536
#define SMEM_SMEM_K_STAGE_BYTES 32768
#define SMEM_SMEM_K_STRIDE 32768
#define SMEM_SMEM_V_OFF 65536
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_PAGE_OFFSETS_OFF 196608
#define SMEM_SMEM_PAGE_OFFSETS_STAGE_BYTES 1024
#define SMEM_SMEM_PAGE_OFFSETS_STRIDE 1024
#define SMEM_WORK_RESPONSE_OFF 202752
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 203136
#define THREADS 512

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


__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}


__device__ __forceinline__ void mma_ts_step(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], db, %4, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_gather4_gmem2smem(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Canonical .shared::cta form for non-multicast gather4, matching
    // trtllm-gen / cuda_ptx and the PTX ISA qualifier order
    // (dim.dst.src.load_mode.completion_mechanism). Per the PTX grammar,
    // .shared::cluster is reserved for the multicast variant (ctaMask).
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}


template <int Stages>
struct CakePipelineState {
    uint32_t index_;
    uint32_t phase_;
    uint32_t count_;

    __device__ __forceinline__ uint32_t index() const { return index_; }
    __device__ __forceinline__ uint32_t phase() const { return phase_; }
    __device__ __forceinline__ uint32_t count() const { return count_; }
    __device__ __forceinline__ CakePipelineState& operator++() {
        ++count_;
        ++index_;
        if (index_ == Stages) {
            index_ = 0;
            phase_ ^= 1;
        }
        return *this;
    }
};

template <int Offset, int Stages>
__device__ __forceinline__ CakePipelineState<Stages>
cake_pipeline_state_at(CakePipelineState<Stages> state) {
    #pragma unroll
    for (int step = 0; step < Offset; ++step) {
        ++state;
    }
    return state;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_cake_dsv4_fp8_h64_source_exact(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, __nv_bfloat16* __restrict__ O, int* __restrict__ cum_seq_lens_q, int* __restrict__ sparse_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int sparse_topk, int has_sinks, int total_work_items)
{
    // PTX global compiler scheduling controls
    asm volatile(".pragma \"global knob ForceLateCommoning=1\";\n" : : : "memory");
    asm volatile(".pragma \"global knob HoistLate=3\";\n" : : : "memory");
    asm volatile(".pragma \"global knob MbarrierInitRegMapping=1\";\n" : : : "memory");

    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 202784;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 64)
    #define page_full_addr (mbar_base + 96)
    #define page_empty_addr (mbar_base + 144)
    #define score_full_addr (mbar_base + 192)
    #define score_empty_addr (mbar_base + 208)
    #define stats_full_addr (mbar_base + 224)
    #define stats_empty_addr (mbar_base + 240)
    #define o_full_addr (mbar_base + 256)
    #define o_empty_addr (mbar_base + 264)
    #define work_full_addr (mbar_base + 272)
    #define work_empty_addr (mbar_base + 288)
    #define throttle_full_addr (mbar_base + 304)
    #define throttle_empty_addr (mbar_base + 320)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int smem_q_addr = smem + 0;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 65536);
    const int smem_k_addr = smem + 65536;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 65536);
    const int smem_v_addr = smem + 65536;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 196608);
    const int smem_page_offsets_addr = smem + 196608;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 202752);
    const int work_response_addr = smem + 202752;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_swa_kv))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_compressed_kv))) : "memory");

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 42 barriers)
    // Mbarriers at smem_raw[202784..203120)

    if (warp == 12) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 202784, 1);
            mbarrier_init(smem + 202792, 1);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 202800, 1);
            mbarrier_init(smem + 202808, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'kv_pipe' ---
            // kv_full: 4 barriers, init_count=4
            mbarrier_init(smem + 202816, 4);
            mbarrier_init(smem + 202824, 4);
            mbarrier_init(smem + 202832, 4);
            mbarrier_init(smem + 202840, 4);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 202848, 1);
            mbarrier_init(smem + 202856, 1);
            mbarrier_init(smem + 202864, 1);
            mbarrier_init(smem + 202872, 1);
            // --- pipeline 'page_pipe' ---
            // page_full: 6 barriers, init_count=32
            mbarrier_init(smem + 202880, 32);
            mbarrier_init(smem + 202888, 32);
            mbarrier_init(smem + 202896, 32);
            mbarrier_init(smem + 202904, 32);
            mbarrier_init(smem + 202912, 32);
            mbarrier_init(smem + 202920, 32);
            // page_empty: 6 barriers, init_count=128
            mbarrier_init(smem + 202928, 128);
            mbarrier_init(smem + 202936, 128);
            mbarrier_init(smem + 202944, 128);
            mbarrier_init(smem + 202952, 128);
            mbarrier_init(smem + 202960, 128);
            mbarrier_init(smem + 202968, 128);
            // --- pipeline 'score_pipe' ---
            // score_full: 2 barriers, init_count=1
            mbarrier_init(smem + 202976, 1);
            mbarrier_init(smem + 202984, 1);
            // score_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 202992, 128);
            mbarrier_init(smem + 203000, 128);
            // --- pipeline 'stats_pipe' ---
            // stats_full: 2 barriers, init_count=128
            mbarrier_init(smem + 203008, 128);
            mbarrier_init(smem + 203016, 128);
            // stats_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 203024, 128);
            mbarrier_init(smem + 203032, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 203040, 1);
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 203048, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 203056, 1);
            mbarrier_init(smem + 203064, 1);
            // work_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 203072, 512);
            mbarrier_init(smem + 203080, 512);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 2 barriers, init_count=128
            mbarrier_init(smem + 203088, 128);
            mbarrier_init(smem + 203096, 128);
            // throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 203104, 32);
            mbarrier_init(smem + 203112, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 203120);
    if (warp == 0) {
        int _tmem_hold = smem + 203120;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Inc phase consumes the registers released above.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;");
        { // softmax_main
            const int warp_rank = warp;
            const int logical_row_origin = warp_rank * 16;
            const int my_row = (unsigned int)logical_row_origin + lane % 16;
            const int col_half = lane / 16;
            int score_cursor = 0;
            int stats_cursor = 0;
            unsigned int softmax_work_stage = 0;
            unsigned int softmax_q = blockIdx.x;
            unsigned int softmax_head = blockIdx.y;
            unsigned int softmax_batch = blockIdx.z;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _work = 0; _work < total_work_items; _work++) {
                int q_begin = cum_seq_lens_q[(int)softmax_batch];
                int q_end = cum_seq_lens_q[(int)(softmax_batch + 1)];
                int q_local = (int)softmax_q;
                int work_valid = q_local < q_end - q_begin;
                asm volatile("griddepcontrol.wait;" ::: "memory");
                float softmax_rescale_log2 = bmm1_scale[0] * 1.4426950408889634f;
                asm volatile("griddepcontrol.wait;" ::: "memory");
                float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
                if (work_valid != 0) {
                    int query_idx = q_begin + q_local;
                    int active_topk = sparse_topk_lens[query_idx];
                    int num_tiles = (active_topk + 128 - 1) / 128;
                    float row_max = -CAKE_INF;
                    float row_sum = 0.0f;
                    #pragma unroll 1
                    for (int tile = 0; tile < num_tiles; tile++) {
                        int pipeline_tile = score_cursor + tile;
                        int score_stage = pipeline_tile & 1;
                        int score_phase = pipeline_tile >> 1 & 1;
                        mbarrier_wait(score_full_addr + (score_stage) * 8, score_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int score_col = ((score_stage != 0) ? 128 : 0);
                        float _tmem_load_0[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                            : "r"(taddr + (unsigned int)score_col));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
                            : "r"(taddr + (unsigned int)score_col + 32));
                        int valid_cols = active_topk - tile * 128 - col_half * 64;
                        if (valid_cols < 0) {
                            valid_cols = 0;
                        }
                        if (valid_cols > 64) {
                            valid_cols = 64;
                        }
                        uint32_t _slice_lo_mask_0;
                        {
                            int _lim_0 = valid_cols;
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
                        if (!(_slice_lo_mask_0 & (1u << 0))) _tmem_load_0[0] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 1))) _tmem_load_0[1] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 2))) _tmem_load_0[2] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 3))) _tmem_load_0[3] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 4))) _tmem_load_0[4] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 5))) _tmem_load_0[5] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 6))) _tmem_load_0[6] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 7))) _tmem_load_0[7] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 8))) _tmem_load_0[8] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 9))) _tmem_load_0[9] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 10))) _tmem_load_0[10] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 11))) _tmem_load_0[11] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 12))) _tmem_load_0[12] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 13))) _tmem_load_0[13] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 14))) _tmem_load_0[14] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 15))) _tmem_load_0[15] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 16))) _tmem_load_0[16] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 17))) _tmem_load_0[17] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 18))) _tmem_load_0[18] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 19))) _tmem_load_0[19] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 20))) _tmem_load_0[20] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 21))) _tmem_load_0[21] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 22))) _tmem_load_0[22] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 23))) _tmem_load_0[23] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 24))) _tmem_load_0[24] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 25))) _tmem_load_0[25] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 26))) _tmem_load_0[26] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 27))) _tmem_load_0[27] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 28))) _tmem_load_0[28] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 29))) _tmem_load_0[29] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 30))) _tmem_load_0[30] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 31))) _tmem_load_0[31] = -CAKE_INF;
                        uint32_t _slice_lo_mask_1;
                        {
                            int _lim_1 = valid_cols - 32;
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
                        if (!(_slice_lo_mask_1 & (1u << 0))) _tmem_load_0[32] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 1))) _tmem_load_0[33] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 2))) _tmem_load_0[34] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 3))) _tmem_load_0[35] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 4))) _tmem_load_0[36] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 5))) _tmem_load_0[37] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 6))) _tmem_load_0[38] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 7))) _tmem_load_0[39] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 8))) _tmem_load_0[40] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 9))) _tmem_load_0[41] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 10))) _tmem_load_0[42] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 11))) _tmem_load_0[43] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 12))) _tmem_load_0[44] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 13))) _tmem_load_0[45] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 14))) _tmem_load_0[46] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 15))) _tmem_load_0[47] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 16))) _tmem_load_0[48] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 17))) _tmem_load_0[49] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 18))) _tmem_load_0[50] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 19))) _tmem_load_0[51] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 20))) _tmem_load_0[52] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 21))) _tmem_load_0[53] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 22))) _tmem_load_0[54] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 23))) _tmem_load_0[55] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 24))) _tmem_load_0[56] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 25))) _tmem_load_0[57] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 26))) _tmem_load_0[58] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 27))) _tmem_load_0[59] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 28))) _tmem_load_0[60] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 29))) _tmem_load_0[61] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 30))) _tmem_load_0[62] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 31))) _tmem_load_0[63] = -CAKE_INF;
                        float2 _reg_reduce_max2_2 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_2);
                        row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_2);
                        float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_2);
                        float _max_0 = max_noftz(row_max, _tmem_load_0_max);
                        float new_max = _max_0;
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, new_max, 16);
                        float _max_1 = max_noftz(new_max, _shfl_xor_0);
                        new_max = _max_1;
                        float _fma_0 = __fmaf_rn(row_max, softmax_rescale_log2, (-new_max) * softmax_rescale_log2);
                        float delta = _fma_0;
                        float _exp2_0 = approx_exp2(delta);
                        float acc_scale = ((row_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                        int stats_tile = stats_cursor + tile;
                        int stats_stage = stats_tile & 1;
                        int stats_phase = stats_tile >> 1 & 1;
                        mbarrier_wait(stats_empty_addr + (stats_stage) * 8, 1 - stats_phase);
                        float stats_pair[2];
                        stats_pair[0] = row_max;
                        stats_pair[1] = new_max;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x2.b32"
                            " [%0], {%1, %2};"
                            :: "r"(taddr + (unsigned int)(stats_stage * 128)), "f"(stats_pair[0]), "f"(stats_pair[1]));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(stats_full_addr + (stats_stage) * 8);
                        float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                        float max_scaled = safe_max * softmax_scale_log2;
                        const float2 _fma_b2_3 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_4 = {-max_scaled + 8.8073549f, -max_scaled + 8.8073549f};
                        #pragma unroll
                        for (int _lf = 0; _lf < 32; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_3, _fma_c2_4);
                        #pragma unroll
                        for (int _le = 0; _le < 64; _le++) {
                            _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                        }
                        float2 _reg_reduce_sum2_5 = make_float2(0.0f, 0.0f);
                        softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_5);
                        softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_5);
                        float _tmem_load_0_sum = _reg_reduce_sum2_5.x + _reg_reduce_sum2_5.y;
                        float block_sum = _tmem_load_0_sum;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, block_sum, 16);
                        block_sum = block_sum + _shfl_xor_1;
                        float _fma_1 = __fmaf_rn(row_sum, acc_scale, block_sum);
                        row_sum = _fma_1;
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
                                : "=r"(_packed) : "f"(_tmem_load_0[0]), "f"(_tmem_load_0[1]),
                                                   "f"(_tmem_load_0[2]), "f"(_tmem_load_0[3]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[4]), "f"(_tmem_load_0[5]),
                                                   "f"(_tmem_load_0[6]), "f"(_tmem_load_0[7]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[8]), "f"(_tmem_load_0[9]),
                                                   "f"(_tmem_load_0[10]), "f"(_tmem_load_0[11]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[12]), "f"(_tmem_load_0[13]),
                                                   "f"(_tmem_load_0[14]), "f"(_tmem_load_0[15]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[16]), "f"(_tmem_load_0[17]),
                                                   "f"(_tmem_load_0[18]), "f"(_tmem_load_0[19]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[20]), "f"(_tmem_load_0[21]),
                                                   "f"(_tmem_load_0[22]), "f"(_tmem_load_0[23]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[24]), "f"(_tmem_load_0[25]),
                                                   "f"(_tmem_load_0[26]), "f"(_tmem_load_0[27]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[28]), "f"(_tmem_load_0[29]),
                                                   "f"(_tmem_load_0[30]), "f"(_tmem_load_0[31]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[32]), "f"(_tmem_load_0[33]),
                                                   "f"(_tmem_load_0[34]), "f"(_tmem_load_0[35]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[36]), "f"(_tmem_load_0[37]),
                                                   "f"(_tmem_load_0[38]), "f"(_tmem_load_0[39]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[40]), "f"(_tmem_load_0[41]),
                                                   "f"(_tmem_load_0[42]), "f"(_tmem_load_0[43]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[44]), "f"(_tmem_load_0[45]),
                                                   "f"(_tmem_load_0[46]), "f"(_tmem_load_0[47]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[48]), "f"(_tmem_load_0[49]),
                                                   "f"(_tmem_load_0[50]), "f"(_tmem_load_0[51]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[52]), "f"(_tmem_load_0[53]),
                                                   "f"(_tmem_load_0[54]), "f"(_tmem_load_0[55]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[56]), "f"(_tmem_load_0[57]),
                                                   "f"(_tmem_load_0[58]), "f"(_tmem_load_0[59]));
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
                                : "=r"(_packed) : "f"(_tmem_load_0[60]), "f"(_tmem_load_0[61]),
                                                   "f"(_tmem_load_0[62]), "f"(_tmem_load_0[63]));
                            _fp8_0[15] = _packed;
                        }
                        int p_col = ((score_stage != 0) ? 160 : 32);
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                            " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + (unsigned int)p_col), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                            " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + 1048576 + (unsigned int)p_col), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (num_tiles > tile + 1) {
                            mbarrier_arrive(score_empty_addr + (score_stage) * 8);
                        }
                        row_max = new_max;
                    }
                    int final_stats_tile = stats_cursor + num_tiles;
                    int final_stats_stage = final_stats_tile & 1;
                    int final_stats_phase = final_stats_tile >> 1 & 1;
                    mbarrier_wait(stats_empty_addr + (final_stats_stage) * 8, 1 - final_stats_phase);
                    float final_stats[2];
                    final_stats[0] = row_sum;
                    final_stats[1] = row_max;
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)(final_stats_stage * 128)), "f"(final_stats[0]), "f"(final_stats[1]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(stats_full_addr + (final_stats_stage) * 8);
                    int tail_stats_tile = final_stats_tile + 1;
                    int tail_stats_stage = tail_stats_tile & 1;
                    int tail_stats_phase = tail_stats_tile >> 1 & 1;
                    mbarrier_wait(stats_empty_addr + (tail_stats_stage) * 8, 1 - tail_stats_phase);
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(stats_full_addr + (tail_stats_stage) * 8);
                    int last_score_tile = score_cursor + num_tiles - 1;
                    mbarrier_arrive(score_empty_addr + (last_score_tile & 1) * 8);
                    stats_cursor = stats_cursor + num_tiles + 2;
                    score_cursor = score_cursor + num_tiles;
                }
                mbarrier_wait(work_full_addr + (softmax_work_stage) * 8, _phase_work_full);
                uint32_t _clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_3)
                    : "r"(work_response_addr + softmax_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_9 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_9)
                    : "r"(work_response_addr + softmax_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_10 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_10)
                    : "r"(work_response_addr + softmax_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_11 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_11)
                    : "r"(work_response_addr + softmax_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (softmax_work_stage) * 8);
                softmax_work_stage += 1;
                if (softmax_work_stage == 2) { softmax_work_stage = 0; _phase_work_full ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                softmax_q = _clc_ctaid_9;
                softmax_head = _clc_ctaid_10;
                softmax_batch = _clc_ctaid_11;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // correction_main
            const int warp_rank_1 = warp - 4;
            const int logical_row_origin_1 = warp_rank_1 * 16;
            const int my_row_1 = (unsigned int)logical_row_origin_1 + lane % 16;
            const int col_half_1 = lane / 16;
            int score_cursor_1 = 0;
            int stats_cursor_1 = 0;
            unsigned int correction_work_stage = 0;
            unsigned int correction_q = blockIdx.x;
            unsigned int correction_head = blockIdx.y;
            unsigned int correction_batch = blockIdx.z;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _work_1 = 0; _work_1 < total_work_items; _work_1++) {
                int q_begin_1 = cum_seq_lens_q[(int)correction_batch];
                int q_end_1 = cum_seq_lens_q[(int)(correction_batch + 1)];
                int q_local_1 = (int)correction_q;
                int work_valid_1 = q_local_1 < q_end_1 - q_begin_1;
                asm volatile("griddepcontrol.wait;" ::: "memory");
                float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
                float output_scale = bmm2_scale[0];
                if (work_valid_1 != 0) {
                    int query_idx_1 = q_begin_1 + q_local_1;
                    int head_base = (int)correction_head * 64;
                    int head_idx = head_base + my_row_1;
                    int active_topk_1 = sparse_topk_lens[query_idx_1];
                    int num_tiles_1 = (active_topk_1 + 128 - 1) / 128;
                    int first_stats_tile = stats_cursor_1;
                    int first_stats_stage = first_stats_tile & 1;
                    int first_stats_phase = first_stats_tile >> 1 & 1;
                    mbarrier_wait(stats_full_addr + (first_stats_stage) * 8, first_stats_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(stats_empty_addr + (first_stats_stage) * 8);
                    #pragma unroll 1
                    for (int tile_1 = 1; tile_1 < num_tiles_1; tile_1++) {
                        int stats_tile_1 = stats_cursor_1 + tile_1;
                        int stats_stage_1 = stats_tile_1 & 1;
                        int stats_phase_1 = stats_tile_1 >> 1 & 1;
                        mbarrier_wait(stats_full_addr + (stats_stage_1) * 8, stats_phase_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_1[2];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x2.b32"
                            " {%0, %1}, [%2];"
                            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1])
                            : "r"(taddr + (unsigned int)(stats_stage_1 * 128)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        mbarrier_arrive(stats_empty_addr + (stats_stage_1) * 8);
                        int output_tile = score_cursor_1 + tile_1 - 1;
                        mbarrier_wait(o_full_addr, output_tile & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float max_delta = _tmem_load_1[0] - _tmem_load_1[1];
                        float _exp2_1 = approx_exp2(max_delta * softmax_scale_log2_1);
                        float acc_scale_1 = ((max_delta != 0.0f) ? _exp2_1 : 1.0f);
                        int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                        int any_rescale = _vote_0;
                        if (any_rescale != 0) {
                            #pragma unroll
                            for (int output_half = 0; output_half < 2; output_half++) {
                                int row_bank = ((output_half != 0) ? 1048576 : 0);
                                #pragma unroll
                                for (int col = 0; col < 128; col += 64) {
                                    float _tmem_load_2[64];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 128;"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                                        : "r"(taddr + 256 + (unsigned int)row_bank + (unsigned int)col));
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 128;"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[63]))
                                        : "r"(taddr + 256 + (unsigned int)row_bank + (unsigned int)col + 32));
                                    const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 32; _ls++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_0);
                                    asm volatile(
                                        "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                                        " [%0], 128, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                                        :: "r"(taddr + 256 + (unsigned int)row_bank + (unsigned int)col), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[63])));
                                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                }
                            }
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(o_empty_addr);
                    }
                    int final_stats_tile_1 = stats_cursor_1 + num_tiles_1;
                    int final_stats_stage_1 = final_stats_tile_1 & 1;
                    int final_stats_phase_1 = final_stats_tile_1 >> 1 & 1;
                    mbarrier_wait(stats_full_addr + (final_stats_stage_1) * 8, final_stats_phase_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_3[2];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x2.b32"
                        " {%0, %1}, [%2];"
                        : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1])
                        : "r"(taddr + (unsigned int)(final_stats_stage_1 * 128)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    mbarrier_arrive(stats_empty_addr + (final_stats_stage_1) * 8);
                    int tail_stats_tile_1 = final_stats_tile_1 + 1;
                    int tail_stats_stage_1 = tail_stats_tile_1 & 1;
                    int tail_stats_phase_1 = tail_stats_tile_1 >> 1 & 1;
                    mbarrier_wait(stats_full_addr + (tail_stats_stage_1) * 8, tail_stats_phase_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(stats_empty_addr + (tail_stats_stage_1) * 8);
                    int last_output_tile = score_cursor_1 + num_tiles_1 - 1;
                    mbarrier_wait(o_full_addr, last_output_tile & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float total_sum = _tmem_load_3[0];
                    float final_max = _tmem_load_3[1];
                    int output_in_bounds = head_idx < num_heads;
                    if (((int)(has_sinks != 0) & output_in_bounds) != 0) {
                        float _exp2_2 = approx_exp2(sinks[head_idx] * 1.4426950408889634f - final_max * softmax_scale_log2_1);
                        total_sum = total_sum + _exp2_2 * 448.0f;
                    }
                    float _rcp_0 = approx_rcp(total_sum);
                    float inv_sum = ((total_sum > 0.0f) ? output_scale * _rcp_0 : 0.0f);
                    int output_base = (query_idx_1 * num_heads + head_idx) * 512;
                    #pragma unroll
                    for (int output_half_1 = 0; output_half_1 < 2; output_half_1++) {
                        int row_bank_1 = ((output_half_1 != 0) ? 1048576 : 0);
                        int col_base = col_half_1 * 128;
                        #pragma unroll 2
                        for (int col_1 = 0; col_1 < 128; col_1 += 8) {
                            float _tmem_load_4[8];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 128;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7]))
                                : "r"(taddr + 256 + (unsigned int)row_bank_1 + (unsigned int)col_1));
                            const float2 _scale2_1 = {inv_sum, inv_sum};
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_1);
                            uint32_t _tmem_load_4_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                                _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            if (output_in_bounds != 0) {
                                reinterpret_cast<int4*>(O + (output_base + output_half_1 * 256 + col_base + col_1))[0] = reinterpret_cast<int4*>(_tmem_load_4_bf16)[0];
                            }
                        }
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(o_empty_addr);
                    stats_cursor_1 = stats_cursor_1 + num_tiles_1 + 2;
                    score_cursor_1 = score_cursor_1 + num_tiles_1;
                }
                mbarrier_wait(work_full_addr + (correction_work_stage) * 8, _phase_work_full_1);
                uint32_t _clc_valid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_4)
                    : "r"(work_response_addr + correction_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_12 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_12)
                    : "r"(work_response_addr + correction_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_13 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_13)
                    : "r"(work_response_addr + correction_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_14 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_14)
                    : "r"(work_response_addr + correction_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (correction_work_stage) * 8);
                correction_work_stage += 1;
                if (correction_work_stage == 2) { correction_work_stage = 0; _phase_work_full_1 ^= 1; }
                if (_clc_valid_4 == 0) {
                    break;
                }
                correction_q = _clc_ctaid_12;
                correction_head = _clc_ctaid_13;
                correction_batch = _clc_ctaid_14;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("barrier.sync 7, 128;" ::: "memory");
            if (warp == 4) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            CakePipelineState<4> _kv_cons_state_state{0u, 0u, 0u};
            unsigned int q_cons_stage = 0;
            int score_prod_stage = 0;
            int score_prod_phase = 0;
            int p_cons_stage = 0;
            int p_cons_phase = 0;
            int o_prod_phase = 0;
            unsigned int mma_work_stage = 0;
            unsigned int mma_q = blockIdx.x;
            unsigned int mma_head = blockIdx.y;
            unsigned int mma_batch = blockIdx.z;
            #pragma unroll
            for (int score_lookahead = 0; score_lookahead < 2; score_lookahead++) {
                int lookahead_stage = score_lookahead & 1;
                int lookahead_phase = score_lookahead >> 1 & 1;
                mbarrier_wait(score_empty_addr + (lookahead_stage) * 8, 1 - lookahead_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            unsigned int _phase_q_full = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _work_2 = 0; _work_2 < total_work_items; _work_2++) {
                int q_begin_2 = cum_seq_lens_q[(int)mma_batch];
                int q_end_2 = cum_seq_lens_q[(int)(mma_batch + 1)];
                int q_local_2 = (int)mma_q;
                int work_valid_2 = q_local_2 < q_end_2 - q_begin_2;
                if (work_valid_2 != 0) {
                    int query_idx_2 = q_begin_2 + q_local_2;
                    int active_topk_2 = sparse_topk_lens[query_idx_2];
                    int num_tiles_2 = (active_topk_2 + 128 - 1) / 128;
                    if (num_tiles_2 > 0) {
                        mbarrier_wait(q_full_addr + (q_cons_stage) * 8, _phase_q_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int score_col_1 = ((score_prod_stage != 0) ? 128 : 0);
                        mbarrier_wait(kv_full_addr + (_kv_cons_state_state.index()) * 8, _kv_cons_state_state.phase());
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 2048);
                        int _mma_b_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (_kv_cons_state_state.index()) * 2048);
                        {
                            uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                            uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 506U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 1018U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                            }
                        }
                        elect_commit(kv_empty_addr + (_kv_cons_state_state.index()) * 8);
                        ++_kv_cons_state_state;
                        mbarrier_wait(kv_full_addr + (_kv_cons_state_state.index()) * 8, _kv_cons_state_state.phase());
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_1 = make_warp_uniform((((smem_q_addr + 16384) >> 4) & 0x3FFF) + (q_cons_stage) * 2048);
                        int _mma_b_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (_kv_cons_state_state.index()) * 2048);
                        {
                            uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                            uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 506U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 1018U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                            }
                        }
                        elect_commit(score_full_addr + (score_prod_stage) * 8);
                        elect_commit(kv_empty_addr + (_kv_cons_state_state.index()) * 8);
                        ++_kv_cons_state_state;
                        score_prod_stage ^= 1;
                        if (score_prod_stage == 0) {
                            score_prod_phase ^= 1;
                        }
                        int first_pv = 1;
                        #pragma unroll 1
                        for (int _steady = 0; _steady < num_tiles_2 - 1; _steady++) {
                            score_col_1 = ((score_prod_stage != 0) ? 128 : 0);
                            mbarrier_wait(kv_full_addr + (_kv_cons_state_state.index()) * 8, _kv_cons_state_state.phase());
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_2 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 2048);
                            int _mma_b_lo_2 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (_kv_cons_state_state.index()) * 2048);
                            {
                                uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                                uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 0);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 506U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 1018U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                                }
                            }
                            elect_commit(kv_empty_addr + (_kv_cons_state_state.index()) * 8);
                            ++_kv_cons_state_state;
                            mbarrier_wait(kv_full_addr + (_kv_cons_state_state.index()) * 8, _kv_cons_state_state.phase());
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_3 = make_warp_uniform((((smem_q_addr + 16384) >> 4) & 0x3FFF) + (q_cons_stage) * 2048);
                            int _mma_b_lo_3 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (_kv_cons_state_state.index()) * 2048);
                            {
                                uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                                uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 506U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 1018U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem + (score_col_1)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                                }
                            }
                            elect_commit(score_full_addr + (score_prod_stage) * 8);
                            elect_commit(kv_empty_addr + (_kv_cons_state_state.index()) * 8);
                            ++_kv_cons_state_state;
                            score_prod_stage ^= 1;
                            if (score_prod_stage == 0) {
                                score_prod_phase ^= 1;
                            }
                            mbarrier_wait(score_empty_addr + (score_prod_stage) * 8, 1 - score_prod_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            mbarrier_wait(o_empty_addr, 1 - o_prod_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            #pragma unroll
                            for (int v_half = 0; v_half < 2; v_half++) {
                                mbarrier_wait(kv_full_addr + (_kv_cons_state_state.index()) * 8, _kv_cons_state_state.phase());
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int row_bank_2 = ((v_half != 0) ? 1048576 : 0);
                                int p_col_1 = ((p_cons_stage != 0) ? 160 : 32) + row_bank_2;
                                int _mma_b_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (_kv_cons_state_state.index()) * 2048);
                                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 71368720;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem + (256 + row_bank_2))), "r"(_mma_b_lo_4), "r"(tmem_tmem + p_col_1), "r"(((first_pv) ? 0 : 1)));
                                elect_commit(kv_empty_addr + (_kv_cons_state_state.index()) * 8);
                                ++_kv_cons_state_state;
                            }
                            first_pv = 0;
                            elect_commit(o_full_addr);
                            p_cons_stage ^= 1;
                            if (p_cons_stage == 0) {
                                p_cons_phase ^= 1;
                            }
                            o_prod_phase ^= 1;
                        }
                        elect_commit(q_empty_addr + (q_cons_stage) * 8);
                        q_cons_stage += 1;
                        if (q_cons_stage == 2) { q_cons_stage = 0; _phase_q_full ^= 1; }
                        int tail_score_stage = score_prod_stage ^ 1;
                        int tail_score_phase = score_prod_phase ^ score_prod_stage;
                        mbarrier_wait(score_empty_addr + (tail_score_stage) * 8, 1 - tail_score_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        mbarrier_wait(o_empty_addr, 1 - o_prod_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int v_half_1 = 0; v_half_1 < 2; v_half_1++) {
                            mbarrier_wait(kv_full_addr + (_kv_cons_state_state.index()) * 8, _kv_cons_state_state.phase());
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int row_bank_3 = ((v_half_1 != 0) ? 1048576 : 0);
                            int p_col_2 = ((p_cons_stage != 0) ? 160 : 32) + row_bank_3;
                            int _mma_b_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (_kv_cons_state_state.index()) * 2048);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 71368720;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem + (256 + row_bank_3))), "r"(_mma_b_lo_5), "r"(tmem_tmem + p_col_2), "r"(((first_pv) ? 0 : 1)));
                            elect_commit(kv_empty_addr + (_kv_cons_state_state.index()) * 8);
                            ++_kv_cons_state_state;
                        }
                        elect_commit(o_full_addr);
                        p_cons_stage ^= 1;
                        if (p_cons_stage == 0) {
                            p_cons_phase ^= 1;
                        }
                        o_prod_phase ^= 1;
                    }
                }
                mbarrier_wait(work_full_addr + (mma_work_stage) * 8, _phase_work_full_2);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + mma_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_6)
                    : "r"(work_response_addr + mma_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_7)
                    : "r"(work_response_addr + mma_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_8 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_8)
                    : "r"(work_response_addr + mma_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (mma_work_stage) * 8);
                mma_work_stage += 1;
                if (mma_work_stage == 2) { mma_work_stage = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                mma_q = _clc_ctaid_6;
                mma_head = _clc_ctaid_7;
                mma_batch = _clc_ctaid_8;
            }
        }
    }
    // ---- Role: page_loader ----
    if (warp == 9) {
        { // page_loader_main
            unsigned int page_prod_stage = 0;
            unsigned int page_prod_phase = 1;
            unsigned int page_work_stage = 0;
            unsigned int page_q = blockIdx.x;
            unsigned int page_head = blockIdx.y;
            unsigned int page_batch = blockIdx.z;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _work_3 = 0; _work_3 < total_work_items; _work_3++) {
                int q_begin_3 = cum_seq_lens_q[(int)page_batch];
                int q_end_3 = cum_seq_lens_q[(int)(page_batch + 1)];
                int q_local_3 = (int)page_q;
                int work_valid_3 = q_local_3 < q_end_3 - q_begin_3;
                if (work_valid_3 != 0) {
                    int query_idx_3 = q_begin_3 + q_local_3;
                    int active_topk_3 = sparse_topk_lens[query_idx_3];
                    int num_tiles_3 = (active_topk_3 + 128 - 1) / 128;
                    int num_page_passes = (num_tiles_3 + 1) / 2;
                    int sparse_base = query_idx_3 * sparse_topk;
                    int last_vec_base = (active_topk_3 - 1) / 4 * 4;
                    #pragma unroll 1
                    for (int page_pass = 0; page_pass < num_page_passes; page_pass++) {
                        #pragma unroll
                        for (int _duplicate = 0; _duplicate < 2; _duplicate++) {
                            mbarrier_wait(page_empty_addr + (page_prod_stage) * 8, page_prod_phase);
                            int page_dst = smem_page_offsets_addr + page_prod_stage * 1024;
                            #pragma unroll
                            for (int page_half = 0; page_half < 2; page_half++) {
                                int page_offset = (unsigned int)(page_pass * 256 + page_half * 128) + lane * 4;
                                int clamped_offset = ((page_offset < active_topk_3) ? page_offset : last_vec_base);
                                asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16;"
                                    :: "r"((unsigned int)page_dst + ((unsigned int)(page_half * 128) + lane * 4) * 4), "l"(sparse_indices + (sparse_base + clamped_offset)));
                            }
                            asm volatile("cp.async.commit_group;");
                            asm volatile("cp.async.wait_group 0;");
                            mbarrier_arrive(page_full_addr + (page_prod_stage) * 8);
                            page_prod_stage += 1;
                            if (page_prod_stage == 6) { page_prod_stage = 0; page_prod_phase ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (page_work_stage) * 8, _phase_work_full_3);
                uint32_t _clc_valid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_0)
                    : "r"(work_response_addr + page_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_0)
                    : "r"(work_response_addr + page_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_1)
                    : "r"(work_response_addr + page_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + page_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (page_work_stage) * 8);
                page_work_stage += 1;
                if (page_work_stage == 2) { page_work_stage = 0; _phase_work_full_3 ^= 1; }
                if (_clc_valid_0 == 0) {
                    break;
                }
                page_q = _clc_ctaid_0;
                page_head = _clc_ctaid_1;
                page_batch = _clc_ctaid_2;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        { // scheduler_main
            unsigned int scheduler_work_stage = 0;
            unsigned int scheduler_throttle_stage = 0;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _work_4 = 0; _work_4 < total_work_items; _work_4++) {
                mbarrier_wait(throttle_full_addr + (scheduler_throttle_stage) * 8, _phase_throttle_full);
                mbarrier_arrive(throttle_empty_addr + (scheduler_throttle_stage) * 8);
                scheduler_throttle_stage += 1;
                if (scheduler_throttle_stage == 2) { scheduler_throttle_stage = 0; _phase_throttle_full ^= 1; }
                if (elect_sync()) {
                    mbarrier_wait(work_empty_addr + (scheduler_work_stage) * 8, _phase_work_empty);
                    mbarrier_arrive_expect_tx(work_full_addr + (scheduler_work_stage) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + scheduler_work_stage * 16 + 0 * 16), "r"(work_full_addr + scheduler_work_stage * 8)
                        : "memory");
                }
                mbarrier_wait(work_full_addr + (scheduler_work_stage) * 8, _phase_work_full_4);
                uint32_t _clc_valid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_5)
                    : "r"(work_response_addr + scheduler_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_15 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_15)
                    : "r"(work_response_addr + scheduler_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_16 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_16)
                    : "r"(work_response_addr + scheduler_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_17 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_17)
                    : "r"(work_response_addr + scheduler_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (scheduler_work_stage) * 8);
                scheduler_work_stage += 1;
                if (scheduler_work_stage == 2) { scheduler_work_stage = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                if (_clc_valid_5 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 11) {
        { // padding_main
            unsigned int padding_work_stage = 0;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _work_5 = 0; _work_5 < total_work_items; _work_5++) {
                mbarrier_wait(work_full_addr + (padding_work_stage) * 8, _phase_work_full_5);
                uint32_t _clc_valid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_6)
                    : "r"(work_response_addr + padding_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_18 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_18)
                    : "r"(work_response_addr + padding_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_19 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_19)
                    : "r"(work_response_addr + padding_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_20 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_20)
                    : "r"(work_response_addr + padding_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (padding_work_stage) * 8);
                padding_work_stage += 1;
                if (padding_work_stage == 2) { padding_work_stage = 0; _phase_work_full_5 ^= 1; }
                if (_clc_valid_6 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // load_main
            const int load_rank = warp - 12;
            unsigned int q_prod_stage = 0;
            unsigned int kv_prod_stage = 0;
            unsigned int page_k_stage = 0;
            unsigned int page_k_phase = 0;
            unsigned int page_v_stage = 1;
            unsigned int page_v_phase = 0;
            unsigned int load_work_stage = 0;
            unsigned int throttle_stage = 0;
            unsigned int load_q = blockIdx.x;
            unsigned int load_head = blockIdx.y;
            unsigned int load_batch = blockIdx.z;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_kv_empty = 1;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _work_6 = 0; _work_6 < total_work_items; _work_6++) {
                mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                throttle_stage += 1;
                if (throttle_stage == 2) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                int q_begin_4 = cum_seq_lens_q[(int)load_batch];
                int q_end_4 = cum_seq_lens_q[(int)(load_batch + 1)];
                int q_local_4 = (int)load_q;
                int work_valid_4 = q_local_4 < q_end_4 - q_begin_4;
                if (work_valid_4 != 0) {
                    int query_idx_4 = q_begin_4 + q_local_4;
                    int active_topk_4 = sparse_topk_lens[query_idx_4];
                    int num_tiles_4 = (active_topk_4 + 128 - 1) / 128;
                    mbarrier_wait(q_empty_addr + (q_prod_stage) * 8, _phase_q_empty);
                    if (load_rank == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(q_full_addr + (q_prod_stage) * 8, 32768);
                        }
                    }
                    asm volatile("griddepcontrol.wait;" ::: "memory");
                    if (load_rank == 0) {
                        if (elect_sync()) {
                            #pragma unroll
                            for (int q_chunk = 0; q_chunk < 4; q_chunk++) {
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6];"
                                    :: "r"(smem_q_addr + q_prod_stage * 32768 + (unsigned int)(q_chunk * 8192)), "l"((&tmap_q)), "r"(q_chunk * 128), "r"(0), "r"((int)load_head), "r"(query_idx_4), "r"(q_full_addr + (q_prod_stage) * 8) : "memory");
                            }
                        }
                    }
                    q_prod_stage += 1;
                    if (q_prod_stage == 2) { q_prod_stage = 0; _phase_q_empty ^= 1; }
                    #pragma unroll 1
                    for (int tile_2 = 0; tile_2 < num_tiles_4; tile_2++) {
                        if ((tile_2 & 1) == 0) {
                            mbarrier_wait(page_full_addr + (page_k_stage) * 8, page_k_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        int page_k_base = smem_page_offsets_addr + page_k_stage * 1024;
                        #pragma unroll
                        for (int k_half = 0; k_half < 2; k_half++) {
                            mbarrier_wait(kv_empty_addr + (kv_prod_stage) * 8, _phase_kv_empty);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (kv_prod_stage) * 8, 8192);
                                if (tile_2 == 0) {
                                    #pragma unroll
                                    for (int local_group = 0; local_group < 8; local_group++) {
                                        int group = load_rank * 8 + local_group;
                                        int group_offset = (tile_2 & 1) * 128 + group * 4;
                                        int rows[4];
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rows[(0) + 3]))
                                            : "r"(page_k_base + group_offset * 4));
                                        int k_dst = smem_k_addr + kv_prod_stage * 32768;
                                        tma_gather4_gmem2smem(k_dst + group * 512, (&tmap_swa_kv), k_half * 256, rows[0], rows[1], rows[2], rows[3], kv_full_addr + (kv_prod_stage) * 8);
                                        tma_gather4_gmem2smem(k_dst + 16384 + group * 512, (&tmap_swa_kv), k_half * 256 + 128, rows[0], rows[1], rows[2], rows[3], kv_full_addr + (kv_prod_stage) * 8);
                                    }
                                } else {
                                    #pragma unroll
                                    for (int local_group_1 = 0; local_group_1 < 8; local_group_1++) {
                                        int group_1 = load_rank * 8 + local_group_1;
                                        int group_offset_1 = (tile_2 & 1) * 128 + group_1 * 4;
                                        int rows_1[4];
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rows_1[(0) + 3]))
                                            : "r"(page_k_base + group_offset_1 * 4));
                                        int k_dst_1 = smem_k_addr + kv_prod_stage * 32768;
                                        tma_gather4_gmem2smem(k_dst_1 + group_1 * 512, (&tmap_compressed_kv), k_half * 256, rows_1[0], rows_1[1], rows_1[2], rows_1[3], kv_full_addr + (kv_prod_stage) * 8);
                                        tma_gather4_gmem2smem(k_dst_1 + 16384 + group_1 * 512, (&tmap_compressed_kv), k_half * 256 + 128, rows_1[0], rows_1[1], rows_1[2], rows_1[3], kv_full_addr + (kv_prod_stage) * 8);
                                    }
                                }
                            }
                            kv_prod_stage += 1;
                            if (kv_prod_stage == 4) { kv_prod_stage = 0; _phase_kv_empty ^= 1; }
                        }
                        if ((tile_2 & 1) != 0 || tile_2 == num_tiles_4 - 1) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(page_empty_addr + (page_k_stage) * 8);
                            page_k_stage += 1;
                            if (page_k_stage == 6) { page_k_stage = 0; page_k_phase ^= 1; }
                            page_k_stage += 1;
                            if (page_k_stage == 6) { page_k_stage = 0; page_k_phase ^= 1; }
                        }
                        if (tile_2 > 0) {
                            int prev_tile = tile_2 - 1;
                            if ((prev_tile & 1) == 0) {
                                mbarrier_wait(page_full_addr + (page_v_stage) * 8, page_v_phase);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            int page_v_base = smem_page_offsets_addr + page_v_stage * 1024;
                            #pragma unroll
                            for (int v_half_2 = 0; v_half_2 < 2; v_half_2++) {
                                mbarrier_wait(kv_empty_addr + (kv_prod_stage) * 8, _phase_kv_empty);
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_prod_stage) * 8, 8192);
                                    if (prev_tile == 0) {
                                        #pragma unroll
                                        for (int local_group_2 = 0; local_group_2 < 8; local_group_2++) {
                                            int group_2 = load_rank * 8 + local_group_2;
                                            int group_offset_2 = (prev_tile & 1) * 128 + group_2 * 4;
                                            int rows_2[4];
                                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&rows_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rows_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rows_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rows_2[(0) + 3]))
                                                : "r"(page_v_base + group_offset_2 * 4));
                                            int v_dst = smem_v_addr + kv_prod_stage * 32768;
                                            tma_gather4_gmem2smem(v_dst + group_2 * 512, (&tmap_swa_kv), v_half_2 * 256, rows_2[0], rows_2[1], rows_2[2], rows_2[3], kv_full_addr + (kv_prod_stage) * 8);
                                            tma_gather4_gmem2smem(v_dst + 16384 + group_2 * 512, (&tmap_swa_kv), v_half_2 * 256 + 128, rows_2[0], rows_2[1], rows_2[2], rows_2[3], kv_full_addr + (kv_prod_stage) * 8);
                                        }
                                    } else {
                                        #pragma unroll
                                        for (int local_group_3 = 0; local_group_3 < 8; local_group_3++) {
                                            int group_3 = load_rank * 8 + local_group_3;
                                            int group_offset_3 = (prev_tile & 1) * 128 + group_3 * 4;
                                            int rows_3[4];
                                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&rows_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rows_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rows_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rows_3[(0) + 3]))
                                                : "r"(page_v_base + group_offset_3 * 4));
                                            int v_dst_1 = smem_v_addr + kv_prod_stage * 32768;
                                            tma_gather4_gmem2smem(v_dst_1 + group_3 * 512, (&tmap_compressed_kv), v_half_2 * 256, rows_3[0], rows_3[1], rows_3[2], rows_3[3], kv_full_addr + (kv_prod_stage) * 8);
                                            tma_gather4_gmem2smem(v_dst_1 + 16384 + group_3 * 512, (&tmap_compressed_kv), v_half_2 * 256 + 128, rows_3[0], rows_3[1], rows_3[2], rows_3[3], kv_full_addr + (kv_prod_stage) * 8);
                                        }
                                    }
                                }
                                kv_prod_stage += 1;
                                if (kv_prod_stage == 4) { kv_prod_stage = 0; _phase_kv_empty ^= 1; }
                            }
                            if ((prev_tile & 1) != 0) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(page_empty_addr + (page_v_stage) * 8);
                                page_v_stage += 1;
                                if (page_v_stage == 6) { page_v_stage = 0; page_v_phase ^= 1; }
                                page_v_stage += 1;
                                if (page_v_stage == 6) { page_v_stage = 0; page_v_phase ^= 1; }
                            }
                        }
                    }
                    int last_tile = num_tiles_4 - 1;
                    mbarrier_wait(page_full_addr + (page_v_stage) * 8, page_v_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int page_v_base_1 = smem_page_offsets_addr + page_v_stage * 1024;
                    #pragma unroll
                    for (int v_half_3 = 0; v_half_3 < 2; v_half_3++) {
                        mbarrier_wait(kv_empty_addr + (kv_prod_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (kv_prod_stage) * 8, 8192);
                            if (last_tile == 0) {
                                #pragma unroll
                                for (int local_group_4 = 0; local_group_4 < 8; local_group_4++) {
                                    int group_4 = load_rank * 8 + local_group_4;
                                    int group_offset_4 = (last_tile & 1) * 128 + group_4 * 4;
                                    int rows_4[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&rows_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&rows_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rows_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rows_4[(0) + 3]))
                                        : "r"(page_v_base_1 + group_offset_4 * 4));
                                    int v_dst_2 = smem_v_addr + kv_prod_stage * 32768;
                                    tma_gather4_gmem2smem(v_dst_2 + group_4 * 512, (&tmap_swa_kv), v_half_3 * 256, rows_4[0], rows_4[1], rows_4[2], rows_4[3], kv_full_addr + (kv_prod_stage) * 8);
                                    tma_gather4_gmem2smem(v_dst_2 + 16384 + group_4 * 512, (&tmap_swa_kv), v_half_3 * 256 + 128, rows_4[0], rows_4[1], rows_4[2], rows_4[3], kv_full_addr + (kv_prod_stage) * 8);
                                }
                            } else {
                                #pragma unroll
                                for (int local_group_5 = 0; local_group_5 < 8; local_group_5++) {
                                    int group_5 = load_rank * 8 + local_group_5;
                                    int group_offset_5 = (last_tile & 1) * 128 + group_5 * 4;
                                    int rows_5[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&rows_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&rows_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rows_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rows_5[(0) + 3]))
                                        : "r"(page_v_base_1 + group_offset_5 * 4));
                                    int v_dst_3 = smem_v_addr + kv_prod_stage * 32768;
                                    tma_gather4_gmem2smem(v_dst_3 + group_5 * 512, (&tmap_compressed_kv), v_half_3 * 256, rows_5[0], rows_5[1], rows_5[2], rows_5[3], kv_full_addr + (kv_prod_stage) * 8);
                                    tma_gather4_gmem2smem(v_dst_3 + 16384 + group_5 * 512, (&tmap_compressed_kv), v_half_3 * 256 + 128, rows_5[0], rows_5[1], rows_5[2], rows_5[3], kv_full_addr + (kv_prod_stage) * 8);
                                }
                            }
                        }
                        kv_prod_stage += 1;
                        if (kv_prod_stage == 4) { kv_prod_stage = 0; _phase_kv_empty ^= 1; }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(page_empty_addr + (page_v_stage) * 8);
                    page_v_stage += 1;
                    if (page_v_stage == 6) { page_v_stage = 0; page_v_phase ^= 1; }
                    page_v_stage += 1;
                    if (page_v_stage == 6) { page_v_stage = 0; page_v_phase ^= 1; }
                }
                mbarrier_wait(work_full_addr + (load_work_stage) * 8, _phase_work_full_6);
                uint32_t _clc_valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_1)
                    : "r"(work_response_addr + load_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + load_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_4)
                    : "r"(work_response_addr + load_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_5)
                    : "r"(work_response_addr + load_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (load_work_stage) * 8);
                load_work_stage += 1;
                if (load_work_stage == 2) { load_work_stage = 0; _phase_work_full_6 ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                load_q = _clc_ctaid_3;
                load_head = _clc_ctaid_4;
                load_batch = _clc_ctaid_5;
            }
        }
    }

    // Cleanup
}

} // extern "C"
