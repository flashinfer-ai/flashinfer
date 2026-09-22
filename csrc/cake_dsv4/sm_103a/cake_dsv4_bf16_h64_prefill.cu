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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_OFFSET 0
#define NUM_KV_PIPE_STAGES 4
#define NUM_PAGE_PIPE_STAGES 6
#define NUM_WORK_PIPE_STAGES 2
#define NUM_THROTTLE_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_KV_OFF 66560
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 66560
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_PAGE_OFFSETS_OFF 197632
#define SMEM_SMEM_PAGE_OFFSETS_STAGE_BYTES 1024
#define SMEM_SMEM_PAGE_OFFSETS_STRIDE 1024
#define SMEM_WORK_RESPONSE_OFF 203776
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 203904
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


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
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
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %3, p;\n\t"
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
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], db, %4, p;\n\t"
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_cake_dsv4_bf16_h64_prefill(CakeTensorMap const* tmap_q, CakeTensorMap const* tmap_swa_kv, CakeTensorMap const* tmap_compressed_kv, __nv_bfloat16* __restrict__ O, int* __restrict__ sparse_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int sparse_topk, int total_work_items, int has_sinks)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 48)
    #define s_full_addr (mbar_base + 80)
    #define p_full_addr (mbar_base + 96)
    #define stats_addr (mbar_base + 112)
    #define stats_empty_addr (mbar_base + 128)
    #define o_empty_addr (mbar_base + 144)
    #define o_full_addr (mbar_base + 152)
    #define page_full_addr (mbar_base + 160)
    #define page_empty_addr (mbar_base + 208)
    #define work_full_addr (mbar_base + 256)
    #define work_empty_addr (mbar_base + 272)
    #define throttle_full_addr (mbar_base + 288)
    #define throttle_empty_addr (mbar_base + 304)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_swa_kv)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_compressed_kv)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_kv_addr = smem + 66560;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_v_addr = smem + 66560;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 197632);
    const int smem_page_offsets_addr = smem + 197632;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 203776);
    const int work_response_addr = smem + 203776;

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 40 barriers)
    // Mbarriers at smem_raw[0..320)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 96, 128);
            mbarrier_init(smem + 104, 128);
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            mbarrier_init(smem + 120, 128);
            // stats_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 144, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'page_pipe' ---
            // page_full: 6 barriers, init_count=32
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            mbarrier_init(smem + 192, 32);
            mbarrier_init(smem + 200, 32);
            // page_empty: 6 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            mbarrier_init(smem + 216, 128);
            mbarrier_init(smem + 224, 128);
            mbarrier_init(smem + 232, 128);
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            // work_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 272, 512);
            mbarrier_init(smem + 280, 512);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 2 barriers, init_count=128
            mbarrier_init(smem + 288, 128);
            mbarrier_init(smem + 296, 128);
            // throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 304, 32);
            mbarrier_init(smem + 312, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 320);
    if (warp == 0) {
        int _tmem_hold = smem + 320;
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
            asm volatile("griddepcontrol.wait;" ::: "memory");
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            unsigned int work_stage = 0;
            unsigned int query_idx = blockIdx.x;
            unsigned int global_tile = 0;
            unsigned int stats_cursor = 0;
            const int warp_in_compute = warp;
            const int tmem_row_origin = warp_in_compute * 32;
            const int logical_row_origin = warp_in_compute * 16;
            const int my_row = (unsigned int)logical_row_origin + lane % 16;
            const int col_half = lane / 16;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int work_iter = 0; work_iter < total_work_items; work_iter++) {
                if (query_idx >= (unsigned int)total_work_items) {
                    break;
                }
                int _max_4 = ((sparse_topk_lens[query_idx]) > (0) ? (sparse_topk_lens[query_idx]) : (0));
                int _min_2 = ((_max_4) < (640) ? (_max_4) : (640));
                int active_topk = _min_2;
                int _max_5 = (((active_topk + 128 - 1) / 128) > (1) ? ((active_topk + 128 - 1) / 128) : (1));
                int num_loop_steps = _max_5;
                float row_max = -CAKE_INF;
                float row_sum = 0.0f;
                if (has_sinks != 0 && my_row < num_heads) {
                    row_max = sinks[my_row] * 1.4426950408889634f / softmax_scale_log2;
                    row_sum = 1.0f;
                }
                if (num_loop_steps > 0) {
                    unsigned int first_stats_stage = stats_cursor & 1;
                    unsigned int first_stats_empty_phase = stats_cursor >> 1 & 1 ^ 1;
                    mbarrier_wait(stats_empty_addr + (first_stats_stage) * 8, first_stats_empty_phase);
                }
                #pragma unroll 1
                for (int tile = 0; tile < num_loop_steps; tile++) {
                    unsigned int sync_stage = global_tile & 1;
                    unsigned int sync_phase = global_tile >> 1 & 1;
                    mbarrier_wait(s_full_addr + (sync_stage) * 8, sync_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int score_col = ((sync_stage != 0) ? 128 : 0);
                    int score_addr = taddr + (unsigned int)score_col + (unsigned int)(tmem_row_origin << 16);
                    float _tmem_load_0[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                        : "r"(score_addr));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
                        : "r"(score_addr + 32));
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
                    float tile_max = _tmem_load_0_max;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, tile_max, 16);
                    float _max_6 = max_noftz(tile_max, _shfl_xor_0);
                    tile_max = _max_6;
                    float old_max = row_max;
                    float _max_7 = max_noftz(row_max, tile_max);
                    float new_max = _max_7;
                    float _fma_0 = __fmaf_rn(old_max, softmax_scale_log2, (-new_max) * softmax_scale_log2);
                    float delta = _fma_0;
                    float _exp2_0 = approx_exp2(delta);
                    float acc_scale = ((old_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                    unsigned int stats_stage = stats_cursor & 1;
                    int stats_col = ((stats_stage != 0) ? 128 : 0);
                    float max_stats[2];
                    max_stats[0] = old_max;
                    max_stats[1] = new_max;
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)stats_col), "f"(max_stats[0]), "f"(max_stats[1]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(stats_addr + (stats_stage) * 8);
                    stats_cursor = stats_cursor + 1;
                    unsigned int next_stats_stage = stats_cursor & 1;
                    unsigned int next_stats_empty_phase = stats_cursor >> 1 & 1 ^ 1;
                    mbarrier_wait(stats_empty_addr + (next_stats_stage) * 8, next_stats_empty_phase);
                    bool empty_row = new_max == -CAKE_INF;
                    float safe_max = ((empty_row) ? 0.0f : new_max);
                    float max_scaled = safe_max * softmax_scale_log2;
                    const float2 _fma_b2_3 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_4 = {-max_scaled, -max_scaled};
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
                    row_max = new_max;
                    if (tile == num_loop_steps - 1) {
                        float final_stats[2];
                        float _max_8 = max_noftz(row_sum, 1.1754943508222875e-38f);
                        final_stats[0] = _max_8;
                        final_stats[1] = row_max;
                        int final_stats_col = ((next_stats_stage != 0) ? 128 : 0);
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x2.b32"
                            " [%0], {%1, %2};"
                            :: "r"(taddr + (unsigned int)final_stats_col), "f"(final_stats[0]), "f"(final_stats[1]));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(stats_addr + (next_stats_stage) * 8);
                        stats_cursor = stats_cursor + 1;
                        unsigned int tail_stats_stage = stats_cursor & 1;
                        unsigned int tail_stats_empty_phase = stats_cursor >> 1 & 1 ^ 1;
                        mbarrier_wait(stats_empty_addr + (tail_stats_stage) * 8, tail_stats_empty_phase);
                        mbarrier_arrive(stats_addr + (tail_stats_stage) * 8);
                        stats_cursor = stats_cursor + 1;
                    }
                    unsigned int packed_p[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        packed_p[_lp] = *(uint32_t*)&_bf2;
                    }
                    int p_addr = taddr + (unsigned int)score_col + 32 + (unsigned int)(tmem_row_origin << 16);
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 16), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[15])));
                    int p_addr_high = p_addr + 1048576;
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr_high), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr_high + 16), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + (sync_stage) * 8);
                    global_tile = global_tile + 1;
                }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
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
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage) * 8);
                work_stage += 1;
                if (work_stage == 2) { work_stage = 0; _phase_work_full ^= 1; }
                unsigned int valid = _clc_valid_2;
                query_idx = _clc_ctaid_2;
                if (valid == 0) {
                    break;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // correction_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            unsigned int work_stage_1 = 0;
            unsigned int query_idx_1 = blockIdx.x;
            unsigned int global_tile_1 = 0;
            unsigned int stats_cursor_1 = 0;
            const int warp_in_role = warp - 4;
            const int tmem_row_origin_1 = warp_in_role * 32;
            const int logical_row_origin_1 = warp_in_role * 16;
            const int my_row_1 = (unsigned int)logical_row_origin_1 + lane % 16;
            const int col_half_1 = lane / 16;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int work_iter_1 = 0; work_iter_1 < total_work_items; work_iter_1++) {
                if (query_idx_1 >= (unsigned int)total_work_items) {
                    break;
                }
                int _max_9 = ((sparse_topk_lens[query_idx_1]) > (0) ? (sparse_topk_lens[query_idx_1]) : (0));
                int _min_3 = ((_max_9) < (640) ? (_max_9) : (640));
                int active_topk_1 = _min_3;
                int _max_10 = (((active_topk_1 + 128 - 1) / 128) > (1) ? ((active_topk_1 + 128 - 1) / 128) : (1));
                int num_loop_steps_1 = _max_10;
                unsigned int first_stats_stage_1 = stats_cursor_1 & 1;
                unsigned int first_stats_phase = stats_cursor_1 >> 1 & 1;
                mbarrier_wait(stats_addr + (first_stats_stage_1) * 8, first_stats_phase);
                mbarrier_arrive(stats_empty_addr + (first_stats_stage_1) * 8);
                stats_cursor_1 = stats_cursor_1 + 1;
                global_tile_1 = global_tile_1 + 1;
                #pragma unroll 1
                for (int tile_1 = 0; tile_1 < num_loop_steps_1 - 1; tile_1++) {
                    unsigned int stats_stage_1 = stats_cursor_1 & 1;
                    unsigned int stats_phase = stats_cursor_1 >> 1 & 1;
                    mbarrier_wait(stats_addr + (stats_stage_1) * 8, stats_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int stats_col_1 = ((stats_stage_1 != 0) ? 128 : 0);
                    float _tmem_load_1[2];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x2.b32"
                        " {%0, %1}, [%2];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1])
                        : "r"(taddr + (unsigned int)stats_col_1));
                    mbarrier_arrive(stats_empty_addr + (stats_stage_1) * 8);
                    stats_cursor_1 = stats_cursor_1 + 1;
                    float max_diff = _tmem_load_1[0] - _tmem_load_1[1];
                    float _exp2_1 = approx_exp2(softmax_scale_log2_1 * max_diff);
                    float acc_scale_1 = ((max_diff != 0.0f) ? _exp2_1 : 1.0f);
                    mbarrier_wait(o_full_addr, global_tile_1 - 1 & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                    int any_rescale = _vote_0;
                    if (any_rescale != 0) {
                        #pragma unroll
                        for (int v_stage = 0; v_stage < 4; v_stage++) {
                            int output_addr = taddr + 256 + (unsigned int)((v_stage & 1) * 128) + (unsigned int)((v_stage >> 1) * 16 << 16) + (unsigned int)(tmem_row_origin_1 << 16);
                            float _tmem_load_2[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                                : "r"(output_addr));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[63]))
                                : "r"(output_addr + 32));
                            const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_0);
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                                " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                                :: "r"(output_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[63])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(o_empty_addr);
                    global_tile_1 = global_tile_1 + 1;
                }
                unsigned int final_stats_stage = stats_cursor_1 & 1;
                unsigned int final_stats_phase = stats_cursor_1 >> 1 & 1;
                mbarrier_wait(stats_addr + (final_stats_stage) * 8, final_stats_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int final_stats_col_1 = ((final_stats_stage != 0) ? 128 : 0);
                float _tmem_load_3[2];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x2.b32"
                    " {%0, %1}, [%2];"
                    : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1])
                    : "r"(taddr + (unsigned int)final_stats_col_1));
                mbarrier_arrive(stats_empty_addr + (final_stats_stage) * 8);
                stats_cursor_1 = stats_cursor_1 + 1;
                unsigned int tail_stats_stage_1 = stats_cursor_1 & 1;
                unsigned int tail_stats_phase = stats_cursor_1 >> 1 & 1;
                mbarrier_wait(stats_addr + (tail_stats_stage_1) * 8, tail_stats_phase);
                mbarrier_arrive(stats_empty_addr + (tail_stats_stage_1) * 8);
                stats_cursor_1 = stats_cursor_1 + 1;
                mbarrier_wait(o_full_addr, global_tile_1 - 1 & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float final_sum = _tmem_load_3[0];
                float _rcp_0 = approx_rcp(final_sum);
                float inv_sum = _rcp_0;
                int output_base = (query_idx_1 * (unsigned int)num_heads + (unsigned int)my_row_1) * 512;
                #pragma unroll
                for (int v_stage_1 = 0; v_stage_1 < 4; v_stage_1++) {
                    int output_addr_1 = taddr + 256 + (unsigned int)((v_stage_1 & 1) * 128) + (unsigned int)((v_stage_1 >> 1) * 16 << 16) + (unsigned int)(tmem_row_origin_1 << 16);
                    float _tmem_load_4[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                        : "r"(output_addr_1));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[63]))
                        : "r"(output_addr_1 + 32));
                    int col_base = v_stage_1 * 128 + col_half_1 * 64;
                    if (my_row_1 < num_heads) {
                        #pragma unroll
                        for (int offset = 0; offset < 64; offset += 8) {
                            {
                                const float2 _prescale2_1 = {inv_sum * output_scale, inv_sum * output_scale};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 4; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[offset])[_ps], _prescale2_1);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    _tmem_load_4[offset + _ps] *= inv_sum * output_scale;
                                #endif
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(_tmem_load_4[offset + 0], _tmem_load_4[offset + 1]);
                                _pk[1] = __floats2bfloat162_rn(_tmem_load_4[offset + 2], _tmem_load_4[offset + 3]);
                                _pk[2] = __floats2bfloat162_rn(_tmem_load_4[offset + 4], _tmem_load_4[offset + 5]);
                                _pk[3] = __floats2bfloat162_rn(_tmem_load_4[offset + 6], _tmem_load_4[offset + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (output_base + col_base + offset)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_empty_addr);
                mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
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
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
                work_stage_1 += 1;
                if (work_stage_1 == 2) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                unsigned int valid_1 = _clc_valid_3;
                query_idx_1 = _clc_ctaid_3;
                if (valid_1 == 0) {
                    break;
                }
            }
            asm volatile("barrier.sync 7, 128;" ::: "memory");
            if (warp == 4) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            unsigned int work_stage_2 = 0;
            unsigned int query_idx_2 = blockIdx.x;
            unsigned int global_tile_2 = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int work_iter_2 = 0; work_iter_2 < total_work_items; work_iter_2++) {
                if (query_idx_2 >= (unsigned int)total_work_items) {
                    break;
                }
                int _max_2 = ((sparse_topk_lens[query_idx_2]) > (0) ? (sparse_topk_lens[query_idx_2]) : (0));
                int _min_1 = ((_max_2) < (640) ? (_max_2) : (640));
                int active_topk_2 = _min_1;
                int _max_3 = (((active_topk_2 + 128 - 1) / 128) > (1) ? ((active_topk_2 + 128 - 1) / 128) : (1));
                int num_loop_steps_2 = _max_3;
                unsigned int query_phase = work_iter_2 & 1;
                mbarrier_wait(q_full_addr, query_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (num_loop_steps_2 > 0) {
                    unsigned int score_stage = global_tile_2 & 1;
                    int score_col_1 = ((score_stage != 0) ? 128 : 0);
                    #pragma unroll
                    for (int k_stage = 0; k_stage < 4; k_stage++) {
                        mbarrier_wait(kv_full_addr + (k_stage) * 8, 0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                        int _mma_b_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                        {
                            uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                            uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, ((k_stage == 0) ? 0 : 1));
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 506U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 1018U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
                            }
                        }
                        if (k_stage == 3) {
                            elect_commit(s_full_addr + (score_stage) * 8);
                        }
                        elect_commit(kv_empty_addr + (k_stage) * 8);
                    }
                    #pragma unroll 1
                    for (int tile_2 = 0; tile_2 < num_loop_steps_2 - 1; tile_2++) {
                        unsigned int next_global_tile = global_tile_2 + 1;
                        unsigned int next_score_stage = next_global_tile & 1;
                        int next_score_col = ((next_score_stage != 0) ? 128 : 0);
                        #pragma unroll
                        for (int k_stage_1 = 0; k_stage_1 < 4; k_stage_1++) {
                            mbarrier_wait(kv_full_addr + (k_stage_1) * 8, 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_1 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (k_stage_1) * 1024);
                            int _mma_b_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage_1) * 2048);
                            {
                                uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                                uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem + (next_score_col)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, ((k_stage_1 == 0) ? 0 : 1));
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem + (next_score_col)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem + (next_score_col)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem + (next_score_col)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 506U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 1018U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem + (next_score_col)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem + (next_score_col)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem + (next_score_col)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem + (next_score_col)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
                                }
                            }
                            if (k_stage_1 == 3) {
                                elect_commit(s_full_addr + (next_score_stage) * 8);
                            }
                            elect_commit(kv_empty_addr + (k_stage_1) * 8);
                        }
                        unsigned int sync_stage_1 = global_tile_2 & 1;
                        unsigned int sync_phase_1 = global_tile_2 >> 1 & 1;
                        int current_score_col = ((sync_stage_1 != 0) ? 128 : 0);
                        mbarrier_wait(p_full_addr + (sync_stage_1) * 8, sync_phase_1);
                        mbarrier_wait(o_empty_addr, 1 - (global_tile_2 & 1));
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int v_stage_2 = 0; v_stage_2 < 4; v_stage_2++) {
                            mbarrier_wait(kv_full_addr + (v_stage_2) * 8, 0);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int output_addr_2 = 256 + (v_stage_2 & 1) * 128 + ((v_stage_2 >> 1) * 16 << 16);
                            int _mma_b_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_2) * 2048);
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem + (output_addr_2))), "r"(_mma_b_lo_2), "r"(tmem_tmem + (current_score_col + 32 + ((v_stage_2 >> 1) * 16 << 16))), "r"(((tile_2 == 0) ? 0 : 1)));
                            elect_commit(kv_empty_addr + (v_stage_2) * 8);
                        }
                        elect_commit(o_full_addr);
                        global_tile_2 = next_global_tile;
                    }
                    elect_commit(q_empty_addr);
                    int last_tile = num_loop_steps_2 - 1;
                    unsigned int last_sync_stage = global_tile_2 & 1;
                    unsigned int last_sync_phase = global_tile_2 >> 1 & 1;
                    int last_score_col = ((last_sync_stage != 0) ? 128 : 0);
                    mbarrier_wait(p_full_addr + (last_sync_stage) * 8, last_sync_phase);
                    mbarrier_wait(o_empty_addr, 1 - (global_tile_2 & 1));
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int v_stage_3 = 0; v_stage_3 < 4; v_stage_3++) {
                        mbarrier_wait(kv_full_addr + (v_stage_3) * 8, 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int output_addr_3 = 256 + (v_stage_3 & 1) * 128 + ((v_stage_3 >> 1) * 16 << 16);
                        int _mma_b_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_3) * 2048);
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem + (output_addr_3))), "r"(_mma_b_lo_3), "r"(tmem_tmem + (last_score_col + 32 + ((v_stage_3 >> 1) * 16 << 16))), "r"(((last_tile == 0) ? 0 : 1)));
                        elect_commit(kv_empty_addr + (v_stage_3) * 8);
                    }
                    elect_commit(o_full_addr);
                    global_tile_2 = global_tile_2 + 1;
                } else {
                    elect_commit(q_empty_addr);
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_1)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
                work_stage_2 += 1;
                if (work_stage_2 == 2) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                unsigned int valid_2 = _clc_valid_1;
                query_idx_2 = _clc_ctaid_1;
                if (valid_2 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: page_warp ----
    if (warp == 9) {
        { // page_warp_main
            unsigned int page_stage = 0;
            unsigned int work_stage_3 = 0;
            unsigned int query_idx_3 = blockIdx.x;
            unsigned int _phase_page_empty = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int work_iter_3 = 0; work_iter_3 < total_work_items; work_iter_3++) {
                if (query_idx_3 >= (unsigned int)total_work_items) {
                    break;
                }
                int sparse_base = query_idx_3 * (unsigned int)sparse_topk;
                int _max_11 = ((sparse_topk_lens[query_idx_3]) > (0) ? (sparse_topk_lens[query_idx_3]) : (0));
                int _min_4 = ((_max_11) < (640) ? (_max_11) : (640));
                int active_topk_3 = _min_4;
                int _max_12 = (((active_topk_3 + 128 - 1) / 128) > (1) ? ((active_topk_3 + 128 - 1) / 128) : (1));
                int num_loop_steps_3 = _max_12;
                int num_page_pairs = (num_loop_steps_3 + 1) / 2;
                int _max_13 = ((active_topk_3 - 1 & -4) > (0) ? (active_topk_3 - 1 & -4) : (0));
                int last_aligned = _max_13;
                #pragma unroll 1
                for (int pair = 0; pair < num_page_pairs; pair++) {
                    int pair_base = pair * 256;
                    #pragma unroll
                    for (int duplicate = 0; duplicate < 2; duplicate++) {
                        mbarrier_wait(page_empty_addr + (page_stage) * 8, _phase_page_empty);
                        int lane_base = lane * 4;
                        int page0 = pair_base + lane_base;
                        int page1 = page0 + 128;
                        int safe0 = ((page0 <= last_aligned) ? page0 : last_aligned);
                        int safe1 = ((page1 <= last_aligned) ? page1 : last_aligned);
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16;"
                            :: "r"(smem_page_offsets_addr + page_stage * 1024 + (unsigned int)(lane_base * 4)), "l"(sparse_indices + (sparse_base + safe0)));
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16;"
                            :: "r"(smem_page_offsets_addr + page_stage * 1024 + (unsigned int)((128 + lane_base) * 4)), "l"(sparse_indices + (sparse_base + safe1)));
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(page_full_addr + (page_stage) * 8) : "memory");
                        page_stage += 1;
                        if (page_stage == 6) { page_stage = 0; _phase_page_empty ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_4)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
                work_stage_3 += 1;
                if (work_stage_3 == 2) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                unsigned int valid_3 = _clc_valid_4;
                query_idx_3 = _clc_ctaid_4;
                if (valid_3 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: scheduler_warp ----
    if (warp == 10) {
        { // scheduler_warp_main
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage = 0;
            unsigned int query_idx_4 = blockIdx.x;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int work_iter_4 = 0; work_iter_4 < total_work_items; work_iter_4++) {
                if (query_idx_4 >= (unsigned int)total_work_items) {
                    break;
                }
                mbarrier_wait(throttle_full_addr + (throttle_stage) * 8, _phase_throttle_full);
                mbarrier_arrive(throttle_empty_addr + (throttle_stage) * 8);
                throttle_stage += 1;
                if (throttle_stage == 2) { throttle_stage = 0; _phase_throttle_full ^= 1; }
                if (elect_sync()) {
                    mbarrier_wait(work_empty_addr + (work_stage_4) * 8, _phase_work_empty);
                    mbarrier_arrive_expect_tx(work_full_addr + (work_stage_4) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + work_stage_4 * 16 + 0 * 16), "r"(work_full_addr + work_stage_4 * 8)
                        : "memory");
                }
                mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
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
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_5)
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
                work_stage_4 += 1;
                if (work_stage_4 == 2) { work_stage_4 = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                unsigned int valid_4 = _clc_valid_5;
                query_idx_4 = _clc_ctaid_5;
                if (valid_4 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: padding_warp ----
    if (warp == 11) {
        { // padding_warp_main
            unsigned int work_stage_5 = 0;
            unsigned int query_idx_5 = blockIdx.x;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int work_iter_5 = 0; work_iter_5 < total_work_items; work_iter_5++) {
                if (query_idx_5 >= (unsigned int)total_work_items) {
                    break;
                }
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_6)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
                work_stage_5 += 1;
                if (work_stage_5 == 2) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                unsigned int valid_5 = _clc_valid_6;
                query_idx_5 = _clc_ctaid_6;
                if (valid_5 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // load_warp_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            const int load_warp_rank = warp - 12;
            unsigned int load_page_stage = 0;
            unsigned int page_k_stage = 0;
            unsigned int page_v_stage = 0;
            int page_k_addr = 0;
            int page_v_addr = 0;
            unsigned int work_stage_6 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int query_idx_6 = blockIdx.x;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_page_full = 0;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int work_iter_6 = 0; work_iter_6 < total_work_items; work_iter_6++) {
                if (query_idx_6 >= (unsigned int)total_work_items) {
                    break;
                }
                int _max_0 = ((sparse_topk_lens[query_idx_6]) > (0) ? (sparse_topk_lens[query_idx_6]) : (0));
                int _min_0 = ((_max_0) < (640) ? (_max_0) : (640));
                int active_topk_4 = _min_0;
                int _max_1 = (((active_topk_4 + 128 - 1) / 128) > (1) ? ((active_topk_4 + 128 - 1) / 128) : (1));
                int num_loop_steps_4 = _max_1;
                mbarrier_wait(throttle_empty_addr + (throttle_stage_1) * 8, _phase_throttle_empty);
                mbarrier_arrive(throttle_full_addr + (throttle_stage_1) * 8);
                throttle_stage_1 += 1;
                if (throttle_stage_1 == 2) { throttle_stage_1 = 0; _phase_throttle_empty ^= 1; }
                unsigned int query_phase_1 = work_iter_6 & 1;
                mbarrier_wait(q_empty_addr, 1 - query_phase_1);
                if (warp == 12) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr, 65536);
                        #pragma unroll
                        for (int q_subload = 0; q_subload < 8; q_subload++) {
                            tma_4d_gmem2smem(smem_q_addr + (unsigned int)(q_subload * 8192), tmap_q, 0, 0, q_subload, query_idx_6, q_full_addr);
                        }
                    }
                }
                if (num_loop_steps_4 > 0) {
                    mbarrier_wait(page_full_addr + (load_page_stage) * 8, _phase_page_full);
                    page_k_stage = load_page_stage;
                    page_k_addr = smem_page_offsets_addr + page_k_stage * 1024;
                    load_page_stage += 1;
                    if (load_page_stage == 6) { load_page_stage = 0; _phase_page_full ^= 1; }
                    #pragma unroll
                    for (int kv_stage = 0; kv_stage < 4; kv_stage++) {
                        mbarrier_wait(kv_empty_addr + (kv_stage) * 8, 1);
                        if (warp == 12) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                            }
                        }
                        #pragma unroll
                        for (int group = 0; group < 8; group++) {
                            int row_offset = load_warp_rank * 4 + group * 16;
                            int raw_rows[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 3]))
                                : "r"(page_k_addr + row_offset * 4));
                            int row0 = raw_rows[0];
                            int row1 = raw_rows[1];
                            int row2 = raw_rows[2];
                            int row3 = raw_rows[3];
                            int dst = smem_kv_addr + (unsigned int)(kv_stage * 32768) + (unsigned int)(row_offset * 64 * 2);
                            if (elect_sync()) {
                                {
                                    tma_gather4_gmem2smem(dst, tmap_swa_kv, kv_stage * 128, row0, row1, row2, row3, kv_full_addr + (kv_stage) * 8);
                                    tma_gather4_gmem2smem(dst + 16384, tmap_swa_kv, kv_stage * 128 + 64, row0, row1, row2, row3, kv_full_addr + (kv_stage) * 8);
                                }
                            }
                        }
                    }
                    if (num_loop_steps_4 > 1) {
                        #pragma unroll
                        for (int kv_stage_1 = 0; kv_stage_1 < 4; kv_stage_1++) {
                            mbarrier_wait(kv_empty_addr + (kv_stage_1) * 8, 0);
                            if (warp == 12) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_1) * 8, 32768);
                                }
                            }
                            #pragma unroll
                            for (int group_1 = 0; group_1 < 8; group_1++) {
                                int row_offset_1 = load_warp_rank * 4 + group_1 * 16;
                                int raw_rows_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 3]))
                                    : "r"(page_k_addr + 512 + row_offset_1 * 4));
                                int row0_1 = raw_rows_1[0];
                                int row1_1 = raw_rows_1[1];
                                int row2_1 = raw_rows_1[2];
                                int row3_1 = raw_rows_1[3];
                                int dst_1 = smem_kv_addr + (unsigned int)(kv_stage_1 * 32768) + (unsigned int)(row_offset_1 * 64 * 2);
                                if (elect_sync()) {
                                    {
                                        tma_gather4_gmem2smem(dst_1, tmap_compressed_kv, kv_stage_1 * 128, row0_1, row1_1, row2_1, row3_1, kv_full_addr + (kv_stage_1) * 8);
                                        tma_gather4_gmem2smem(dst_1 + 16384, tmap_compressed_kv, kv_stage_1 * 128 + 64, row0_1, row1_1, row2_1, row3_1, kv_full_addr + (kv_stage_1) * 8);
                                    }
                                }
                            }
                        }
                        if (num_loop_steps_4 > 2) {
                            mbarrier_arrive(page_empty_addr + (page_k_stage) * 8);
                        }
                        mbarrier_wait(page_full_addr + (load_page_stage) * 8, _phase_page_full);
                        page_v_stage = load_page_stage;
                        page_v_addr = smem_page_offsets_addr + page_v_stage * 1024;
                        load_page_stage += 1;
                        if (load_page_stage == 6) { load_page_stage = 0; _phase_page_full ^= 1; }
                        #pragma unroll
                        for (int kv_stage_2 = 0; kv_stage_2 < 4; kv_stage_2++) {
                            mbarrier_wait(kv_empty_addr + (kv_stage_2) * 8, 1);
                            if (warp == 12) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_2) * 8, 32768);
                                }
                            }
                            #pragma unroll
                            for (int group_2 = 0; group_2 < 8; group_2++) {
                                int row_offset_2 = load_warp_rank * 4 + group_2 * 16;
                                int raw_rows_2[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 3]))
                                    : "r"(page_v_addr + row_offset_2 * 4));
                                int row0_2 = raw_rows_2[0];
                                int row1_2 = raw_rows_2[1];
                                int row2_2 = raw_rows_2[2];
                                int row3_2 = raw_rows_2[3];
                                int dst_2 = smem_v_addr + (unsigned int)(kv_stage_2 * 32768) + (unsigned int)(row_offset_2 * 64 * 2);
                                if (elect_sync()) {
                                    {
                                        tma_gather4_gmem2smem(dst_2, tmap_swa_kv, kv_stage_2 * 128, row0_2, row1_2, row2_2, row3_2, kv_full_addr + (kv_stage_2) * 8);
                                        tma_gather4_gmem2smem(dst_2 + 16384, tmap_swa_kv, kv_stage_2 * 128 + 64, row0_2, row1_2, row2_2, row3_2, kv_full_addr + (kv_stage_2) * 8);
                                    }
                                }
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int tile_3 = 1; tile_3 < num_loop_steps_4 - 1; tile_3++) {
                        int next_tile = tile_3 + 1;
                        if ((next_tile & 1) == 0) {
                            mbarrier_wait(page_full_addr + (load_page_stage) * 8, _phase_page_full);
                            page_k_stage = load_page_stage;
                            page_k_addr = smem_page_offsets_addr + page_k_stage * 1024;
                            load_page_stage += 1;
                            if (load_page_stage == 6) { load_page_stage = 0; _phase_page_full ^= 1; }
                        }
                        #pragma unroll
                        for (int kv_stage_3 = 0; kv_stage_3 < 4; kv_stage_3++) {
                            mbarrier_wait(kv_empty_addr + (kv_stage_3) * 8, 0);
                            if (warp == 12) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_3) * 8, 32768);
                                }
                            }
                            #pragma unroll
                            for (int group_3 = 0; group_3 < 8; group_3++) {
                                int row_offset_3 = load_warp_rank * 4 + group_3 * 16;
                                int raw_rows_3[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 3]))
                                    : "r"(page_k_addr + (next_tile & 1) * 128 * 4 + row_offset_3 * 4));
                                int row0_3 = raw_rows_3[0];
                                int row1_3 = raw_rows_3[1];
                                int row2_3 = raw_rows_3[2];
                                int row3_3 = raw_rows_3[3];
                                int dst_3 = smem_kv_addr + (unsigned int)(kv_stage_3 * 32768) + (unsigned int)(row_offset_3 * 64 * 2);
                                if (elect_sync()) {
                                    {
                                        tma_gather4_gmem2smem(dst_3, tmap_compressed_kv, kv_stage_3 * 128, row0_3, row1_3, row2_3, row3_3, kv_full_addr + (kv_stage_3) * 8);
                                        tma_gather4_gmem2smem(dst_3 + 16384, tmap_compressed_kv, kv_stage_3 * 128 + 64, row0_3, row1_3, row2_3, row3_3, kv_full_addr + (kv_stage_3) * 8);
                                    }
                                }
                            }
                        }
                        if ((next_tile & 1) != 0 && tile_3 < num_loop_steps_4 - 2) {
                            mbarrier_arrive(page_empty_addr + (page_k_stage) * 8);
                        }
                        if ((tile_3 & 1) == 0) {
                            mbarrier_wait(page_full_addr + (load_page_stage) * 8, _phase_page_full);
                            page_v_stage = load_page_stage;
                            page_v_addr = smem_page_offsets_addr + page_v_stage * 1024;
                            load_page_stage += 1;
                            if (load_page_stage == 6) { load_page_stage = 0; _phase_page_full ^= 1; }
                        }
                        #pragma unroll
                        for (int kv_stage_4 = 0; kv_stage_4 < 4; kv_stage_4++) {
                            mbarrier_wait(kv_empty_addr + (kv_stage_4) * 8, 1);
                            if (warp == 12) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_4) * 8, 32768);
                                }
                            }
                            #pragma unroll
                            for (int group_4 = 0; group_4 < 8; group_4++) {
                                int row_offset_4 = load_warp_rank * 4 + group_4 * 16;
                                int raw_rows_4[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_4[(0) + 3]))
                                    : "r"(page_v_addr + (tile_3 & 1) * 128 * 4 + row_offset_4 * 4));
                                int row0_4 = raw_rows_4[0];
                                int row1_4 = raw_rows_4[1];
                                int row2_4 = raw_rows_4[2];
                                int row3_4 = raw_rows_4[3];
                                int dst_4 = smem_v_addr + (unsigned int)(kv_stage_4 * 32768) + (unsigned int)(row_offset_4 * 64 * 2);
                                if (elect_sync()) {
                                    {
                                        tma_gather4_gmem2smem(dst_4, tmap_compressed_kv, kv_stage_4 * 128, row0_4, row1_4, row2_4, row3_4, kv_full_addr + (kv_stage_4) * 8);
                                        tma_gather4_gmem2smem(dst_4 + 16384, tmap_compressed_kv, kv_stage_4 * 128 + 64, row0_4, row1_4, row2_4, row3_4, kv_full_addr + (kv_stage_4) * 8);
                                    }
                                }
                            }
                        }
                        if ((tile_3 & 1) != 0) {
                            mbarrier_arrive(page_empty_addr + (page_v_stage) * 8);
                        }
                    }
                    int last_tile_1 = num_loop_steps_4 - 1;
                    mbarrier_arrive(page_empty_addr + (page_k_stage) * 8);
                    if ((last_tile_1 & 1) == 0) {
                        mbarrier_wait(page_full_addr + (load_page_stage) * 8, _phase_page_full);
                        page_v_stage = load_page_stage;
                        page_v_addr = smem_page_offsets_addr + page_v_stage * 1024;
                        load_page_stage += 1;
                        if (load_page_stage == 6) { load_page_stage = 0; _phase_page_full ^= 1; }
                    }
                    if (num_loop_steps_4 == 1) {
                        #pragma unroll
                        for (int kv_stage_5 = 0; kv_stage_5 < 4; kv_stage_5++) {
                            mbarrier_wait(kv_empty_addr + (kv_stage_5) * 8, 0);
                            if (warp == 12) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_5) * 8, 32768);
                                }
                            }
                            #pragma unroll
                            for (int group_5 = 0; group_5 < 8; group_5++) {
                                int row_offset_5 = load_warp_rank * 4 + group_5 * 16;
                                int raw_rows_5[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_5[(0) + 3]))
                                    : "r"(page_v_addr + row_offset_5 * 4));
                                int row0_5 = raw_rows_5[0];
                                int row1_5 = raw_rows_5[1];
                                int row2_5 = raw_rows_5[2];
                                int row3_5 = raw_rows_5[3];
                                int dst_5 = smem_v_addr + (unsigned int)(kv_stage_5 * 32768) + (unsigned int)(row_offset_5 * 64 * 2);
                                if (elect_sync()) {
                                    {
                                        tma_gather4_gmem2smem(dst_5, tmap_swa_kv, kv_stage_5 * 128, row0_5, row1_5, row2_5, row3_5, kv_full_addr + (kv_stage_5) * 8);
                                        tma_gather4_gmem2smem(dst_5 + 16384, tmap_swa_kv, kv_stage_5 * 128 + 64, row0_5, row1_5, row2_5, row3_5, kv_full_addr + (kv_stage_5) * 8);
                                    }
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int kv_stage_6 = 0; kv_stage_6 < 4; kv_stage_6++) {
                            mbarrier_wait(kv_empty_addr + (kv_stage_6) * 8, 0);
                            if (warp == 12) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_6) * 8, 32768);
                                }
                            }
                            #pragma unroll
                            for (int group_6 = 0; group_6 < 8; group_6++) {
                                int row_offset_6 = load_warp_rank * 4 + group_6 * 16;
                                int raw_rows_6[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_6[(0) + 3]))
                                    : "r"(page_v_addr + (last_tile_1 & 1) * 128 * 4 + row_offset_6 * 4));
                                int row0_6 = raw_rows_6[0];
                                int row1_6 = raw_rows_6[1];
                                int row2_6 = raw_rows_6[2];
                                int row3_6 = raw_rows_6[3];
                                int dst_6 = smem_v_addr + (unsigned int)(kv_stage_6 * 32768) + (unsigned int)(row_offset_6 * 64 * 2);
                                if (elect_sync()) {
                                    {
                                        tma_gather4_gmem2smem(dst_6, tmap_compressed_kv, kv_stage_6 * 128, row0_6, row1_6, row2_6, row3_6, kv_full_addr + (kv_stage_6) * 8);
                                        tma_gather4_gmem2smem(dst_6 + 16384, tmap_compressed_kv, kv_stage_6 * 128 + 64, row0_6, row1_6, row2_6, row3_6, kv_full_addr + (kv_stage_6) * 8);
                                    }
                                }
                            }
                        }
                    }
                    mbarrier_arrive(page_empty_addr + (page_v_stage) * 8);
                }
                mbarrier_wait(work_full_addr + (work_stage_6) * 8, _phase_work_full_6);
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
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_0)
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                work_stage_6 += 1;
                if (work_stage_6 == 2) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
                unsigned int valid_6 = _clc_valid_0;
                query_idx_6 = _clc_ctaid_0;
                if (valid_6 == 0) {
                    break;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
