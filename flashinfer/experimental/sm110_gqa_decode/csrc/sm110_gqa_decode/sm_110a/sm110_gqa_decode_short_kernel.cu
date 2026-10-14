// Copyright (c) 2026 by FlashInfer team.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// clang-format off
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "FlashInfer requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) Sm110GqaTensorMap { uint64_t opaque[16]; };
struct __align__(64) Sm110GqaTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(Sm110GqaTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(Sm110GqaTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

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

#define SM110_GQA_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_SCORES_OFFSET 0
#define TMEM_OUTPUT_OFFSET 128
#define NUM_MAIN_STAGES 1
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_KV_SMEM_OFF 17408
#define SMEM_KV_SMEM_STAGE_BYTES 16384
#define SMEM_KV_SMEM_STRIDE 16384
#define SMEM_V_SMEM_OFF 17408
#define SMEM_V_SMEM_STAGE_BYTES 16384
#define SMEM_V_SMEM_STRIDE 16384
#define SMEM_TOTAL 50176
#define THREADS 256

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
           "r"(i_desc), "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_sm110_gqa_decode_short(const __grid_constant__ Sm110GqaTensorMap64 Q, const __grid_constant__ Sm110GqaTensorMap64 K, const __grid_constant__ Sm110GqaTensorMap64 V, __half* __restrict__ O, int* __restrict__ sequence_lengths, float softmax_scale_log2)
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
    #define kv_full_addr (mbar_base + 8)
    #define s_full_addr (mbar_base + 24)
    #define p_full_addr (mbar_base + 32)
    #define o_full_addr (mbar_base + 40)
    #define tile_done_addr (mbar_base + 48)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __half* q_smem = reinterpret_cast<__half*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __half* kv_smem = reinterpret_cast<__half*>(smem_raw + 17408);
    const int kv_smem_addr = smem + 17408;
    __half* v_smem = reinterpret_cast<__half*>(smem_raw + 17408);
    const int v_smem_addr = smem + 17408;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 7 barriers)
    // Mbarriers at smem_raw[0..56)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // kv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // p_full: 1 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // tile_done: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 56);
    if (warp == 0) {
        int _tmem_hold = smem + 56;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_output = taddr + 128;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax_and_drain ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_and_drain_main
            int tile = blockIdx.x;
            int batch = tile / 8;
            int kv_head = tile - batch * 8;
            int seqlen_kv = sequence_lengths[batch];
            const int warp_in_role = warp;
            const int tmem_row_origin = warp_in_role * 32;
            const int logical_row_origin = warp_in_role * 16;
            int my_row = (unsigned int)logical_row_origin + lane % 16;
            int col_half = lane / 16;
            int row_valid = ((my_row < 4) ? 1 : 0);
            unsigned int packed_p[16];
            float block_sum = 0.0f;
            unsigned int _phase_s_full_0 = 0;
            if (seqlen_kv == 1) {
                packed_p[0] = 0;
                packed_p[1] = 0;
                packed_p[2] = 0;
                packed_p[3] = 0;
                packed_p[4] = 0;
                packed_p[5] = 0;
                packed_p[6] = 0;
                packed_p[7] = 0;
                packed_p[8] = 0;
                packed_p[9] = 0;
                packed_p[10] = 0;
                packed_p[11] = 0;
                packed_p[12] = 0;
                packed_p[13] = 0;
                packed_p[14] = 0;
                packed_p[15] = 0;
                if (row_valid != 0 && col_half == 0) {
                    packed_p[0] = 15360;
                }
                block_sum = ((row_valid != 0) ? 1.0f : 0.0f);
            } else {
                float score_values[32];
                mbarrier_wait(s_full_addr, _phase_s_full_0);
                _phase_s_full_0 ^= 1;
                int valid_cols = ((row_valid != 0) ? seqlen_kv : 0);
                int score_addr = taddr + (unsigned int)(tmem_row_origin << 16);
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 32;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&score_values[0])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[1])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[2])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[3])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[4])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[5])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[6])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[7])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[8])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[9])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[10])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[11])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[12])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[13])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[14])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[15])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[16])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[17])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[18])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[19])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[20])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[21])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[22])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[23])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[24])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[25])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[26])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[27])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[28])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[29])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[30])), "=r"(*reinterpret_cast<uint32_t*>(&score_values[31]))
                    : "r"(score_addr));
                int half_valid = valid_cols - col_half * 32;
                if (half_valid < 0) {
                    half_valid = 0;
                }
                if (half_valid > 32) {
                    half_valid = 32;
                }
                if (half_valid <= 0) {
                    if (!(0 & (1u << 0))) score_values[0] = -SM110_GQA_INF;
                    if (!(0 & (1u << 1))) score_values[1] = -SM110_GQA_INF;
                    if (!(0 & (1u << 2))) score_values[2] = -SM110_GQA_INF;
                    if (!(0 & (1u << 3))) score_values[3] = -SM110_GQA_INF;
                    if (!(0 & (1u << 4))) score_values[4] = -SM110_GQA_INF;
                    if (!(0 & (1u << 5))) score_values[5] = -SM110_GQA_INF;
                    if (!(0 & (1u << 6))) score_values[6] = -SM110_GQA_INF;
                    if (!(0 & (1u << 7))) score_values[7] = -SM110_GQA_INF;
                    if (!(0 & (1u << 8))) score_values[8] = -SM110_GQA_INF;
                    if (!(0 & (1u << 9))) score_values[9] = -SM110_GQA_INF;
                    if (!(0 & (1u << 10))) score_values[10] = -SM110_GQA_INF;
                    if (!(0 & (1u << 11))) score_values[11] = -SM110_GQA_INF;
                    if (!(0 & (1u << 12))) score_values[12] = -SM110_GQA_INF;
                    if (!(0 & (1u << 13))) score_values[13] = -SM110_GQA_INF;
                    if (!(0 & (1u << 14))) score_values[14] = -SM110_GQA_INF;
                    if (!(0 & (1u << 15))) score_values[15] = -SM110_GQA_INF;
                    if (!(0 & (1u << 16))) score_values[16] = -SM110_GQA_INF;
                    if (!(0 & (1u << 17))) score_values[17] = -SM110_GQA_INF;
                    if (!(0 & (1u << 18))) score_values[18] = -SM110_GQA_INF;
                    if (!(0 & (1u << 19))) score_values[19] = -SM110_GQA_INF;
                    if (!(0 & (1u << 20))) score_values[20] = -SM110_GQA_INF;
                    if (!(0 & (1u << 21))) score_values[21] = -SM110_GQA_INF;
                    if (!(0 & (1u << 22))) score_values[22] = -SM110_GQA_INF;
                    if (!(0 & (1u << 23))) score_values[23] = -SM110_GQA_INF;
                    if (!(0 & (1u << 24))) score_values[24] = -SM110_GQA_INF;
                    if (!(0 & (1u << 25))) score_values[25] = -SM110_GQA_INF;
                    if (!(0 & (1u << 26))) score_values[26] = -SM110_GQA_INF;
                    if (!(0 & (1u << 27))) score_values[27] = -SM110_GQA_INF;
                    if (!(0 & (1u << 28))) score_values[28] = -SM110_GQA_INF;
                    if (!(0 & (1u << 29))) score_values[29] = -SM110_GQA_INF;
                    if (!(0 & (1u << 30))) score_values[30] = -SM110_GQA_INF;
                    if (!(0 & (1u << 31))) score_values[31] = -SM110_GQA_INF;
                } else if (half_valid < 32) {
                    uint32_t _slice_lo_mask_0;
                    {
                        int _lim_0 = half_valid;
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
                    if (!(_slice_lo_mask_0 & (1u << 0))) score_values[0] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 1))) score_values[1] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 2))) score_values[2] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 3))) score_values[3] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 4))) score_values[4] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 5))) score_values[5] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 6))) score_values[6] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 7))) score_values[7] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 8))) score_values[8] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 9))) score_values[9] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 10))) score_values[10] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 11))) score_values[11] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 12))) score_values[12] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 13))) score_values[13] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 14))) score_values[14] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 15))) score_values[15] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 16))) score_values[16] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 17))) score_values[17] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 18))) score_values[18] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 19))) score_values[19] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 20))) score_values[20] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 21))) score_values[21] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 22))) score_values[22] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 23))) score_values[23] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 24))) score_values[24] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 25))) score_values[25] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 26))) score_values[26] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 27))) score_values[27] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 28))) score_values[28] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 29))) score_values[29] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 30))) score_values[30] = -SM110_GQA_INF;
                    if (!(_slice_lo_mask_0 & (1u << 31))) score_values[31] = -SM110_GQA_INF;
                }
                float2 _reg_reduce_max2_1 = {-SM110_GQA_INF, -SM110_GQA_INF};
                _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(score_values[0], score_values[1]));
                _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(score_values[2], score_values[3]));
                _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(score_values[4], score_values[5]));
                _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(score_values[6], score_values[7]));
                _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(score_values[8], score_values[9]));
                _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(score_values[10], score_values[11]));
                _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(score_values[12], score_values[13]));
                _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(score_values[14], score_values[15]));
                _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(score_values[16], score_values[17]));
                _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(score_values[18], score_values[19]));
                _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(score_values[20], score_values[21]));
                _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(score_values[22], score_values[23]));
                _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(score_values[24], score_values[25]));
                _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(score_values[26], score_values[27]));
                _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(score_values[28], score_values[29]));
                _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(score_values[30], score_values[31]));
                float score_values_max = row_max_reduce(_reg_reduce_max2_1);
                float tile_max = score_values_max;
                if (half_valid <= 0) {
                    tile_max = -SM110_GQA_INF;
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, tile_max, 16);
                float _max_0 = max_noftz(tile_max, _shfl_xor_0);
                tile_max = _max_0;
                float safe_max = ((tile_max == -SM110_GQA_INF) ? 0.0f : tile_max);
                float max_scaled = safe_max * softmax_scale_log2;
                float score_bias = ((valid_cols > 0) ? -max_scaled : -SM110_GQA_INF);
                const float2 _fma_b2_2 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_3 = {score_bias, score_bias};
                float2 _fma_pair_4 = fma_f32x2(make_float2(score_values[0], score_values[1]), _fma_b2_2, _fma_c2_3);
                score_values[0] = _fma_pair_4.x;
                score_values[1] = _fma_pair_4.y;
                float2 _fma_pair_5 = fma_f32x2(make_float2(score_values[2], score_values[3]), _fma_b2_2, _fma_c2_3);
                score_values[2] = _fma_pair_5.x;
                score_values[3] = _fma_pair_5.y;
                float2 _fma_pair_6 = fma_f32x2(make_float2(score_values[4], score_values[5]), _fma_b2_2, _fma_c2_3);
                score_values[4] = _fma_pair_6.x;
                score_values[5] = _fma_pair_6.y;
                float2 _fma_pair_7 = fma_f32x2(make_float2(score_values[6], score_values[7]), _fma_b2_2, _fma_c2_3);
                score_values[6] = _fma_pair_7.x;
                score_values[7] = _fma_pair_7.y;
                float2 _fma_pair_8 = fma_f32x2(make_float2(score_values[8], score_values[9]), _fma_b2_2, _fma_c2_3);
                score_values[8] = _fma_pair_8.x;
                score_values[9] = _fma_pair_8.y;
                float2 _fma_pair_9 = fma_f32x2(make_float2(score_values[10], score_values[11]), _fma_b2_2, _fma_c2_3);
                score_values[10] = _fma_pair_9.x;
                score_values[11] = _fma_pair_9.y;
                float2 _fma_pair_10 = fma_f32x2(make_float2(score_values[12], score_values[13]), _fma_b2_2, _fma_c2_3);
                score_values[12] = _fma_pair_10.x;
                score_values[13] = _fma_pair_10.y;
                float2 _fma_pair_11 = fma_f32x2(make_float2(score_values[14], score_values[15]), _fma_b2_2, _fma_c2_3);
                score_values[14] = _fma_pair_11.x;
                score_values[15] = _fma_pair_11.y;
                float2 _fma_pair_12 = fma_f32x2(make_float2(score_values[16], score_values[17]), _fma_b2_2, _fma_c2_3);
                score_values[16] = _fma_pair_12.x;
                score_values[17] = _fma_pair_12.y;
                float2 _fma_pair_13 = fma_f32x2(make_float2(score_values[18], score_values[19]), _fma_b2_2, _fma_c2_3);
                score_values[18] = _fma_pair_13.x;
                score_values[19] = _fma_pair_13.y;
                float2 _fma_pair_14 = fma_f32x2(make_float2(score_values[20], score_values[21]), _fma_b2_2, _fma_c2_3);
                score_values[20] = _fma_pair_14.x;
                score_values[21] = _fma_pair_14.y;
                float2 _fma_pair_15 = fma_f32x2(make_float2(score_values[22], score_values[23]), _fma_b2_2, _fma_c2_3);
                score_values[22] = _fma_pair_15.x;
                score_values[23] = _fma_pair_15.y;
                float2 _fma_pair_16 = fma_f32x2(make_float2(score_values[24], score_values[25]), _fma_b2_2, _fma_c2_3);
                score_values[24] = _fma_pair_16.x;
                score_values[25] = _fma_pair_16.y;
                float2 _fma_pair_17 = fma_f32x2(make_float2(score_values[26], score_values[27]), _fma_b2_2, _fma_c2_3);
                score_values[26] = _fma_pair_17.x;
                score_values[27] = _fma_pair_17.y;
                float2 _fma_pair_18 = fma_f32x2(make_float2(score_values[28], score_values[29]), _fma_b2_2, _fma_c2_3);
                score_values[28] = _fma_pair_18.x;
                score_values[29] = _fma_pair_18.y;
                float2 _fma_pair_19 = fma_f32x2(make_float2(score_values[30], score_values[31]), _fma_b2_2, _fma_c2_3);
                score_values[30] = _fma_pair_19.x;
                score_values[31] = _fma_pair_19.y;
                #pragma unroll
                for (int _le = 0; _le < 32; _le++) {
                    score_values[_le] = approx_exp2(score_values[_le]);
                }
                float2 _reg_reduce_sum2_20 = make_float2(0.0f, 0.0f);
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[0], score_values[1]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[2], score_values[3]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[4], score_values[5]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[6], score_values[7]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[8], score_values[9]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[10], score_values[11]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[12], score_values[13]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[14], score_values[15]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[16], score_values[17]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[18], score_values[19]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[20], score_values[21]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[22], score_values[23]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[24], score_values[25]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[26], score_values[27]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[28], score_values[29]));
                _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(score_values[30], score_values[31]));
                float score_values_sum = _reg_reduce_sum2_20.x + _reg_reduce_sum2_20.y;
                float block_half = score_values_sum;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, block_half, 16);
                block_sum = block_half + _shfl_xor_1;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(score_values[_lp*2 + 0], score_values[_lp*2+1 + 0]));
                    packed_p[_lp] = *(uint32_t*)&_h2;
                }
            }
            int p_addr = taddr + 32 + (unsigned int)(tmem_row_origin << 16);
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[15])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            mbarrier_arrive(p_full_addr);
            unsigned int _phase_o_full_0 = 0;
            mbarrier_wait(o_full_addr, _phase_o_full_0);
            _phase_o_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float inv_sum = 1.0f;
            if (seqlen_kv != 1) {
                float _rcp_0 = approx_rcp(block_sum);
                inv_sum = ((block_sum > 0.0f && block_sum == block_sum) ? _rcp_0 : 0.0f);
            }
            int row_addr = tmem_row_origin << 16;
            float _tmem_load_0[64];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                : "r"(taddr + 128 + (unsigned int)row_addr));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
                : "r"(taddr + 128 + (unsigned int)row_addr + 32));
            int col_base = col_half * 64;
            if (my_row < 4) {
                int q_head = kv_head * 4 + my_row;
                int output_row = (batch * 32 + q_head) * 128;
                #pragma unroll
                for (int offset = 0; offset < 64; offset += 8) {
                    {
                        const float2 _prescale2_21 = {inv_sum, inv_sum};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_0[offset])[_ps], _prescale2_21);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            _tmem_load_0[offset + _ps] *= inv_sum;
                        #endif
                        __half2 _pk[4];
                        _pk[0] = __floats2half2_rn(_tmem_load_0[offset + 0], _tmem_load_0[offset + 1]);
                        _pk[1] = __floats2half2_rn(_tmem_load_0[offset + 2], _tmem_load_0[offset + 3]);
                        _pk[2] = __floats2half2_rn(_tmem_load_0[offset + 4], _tmem_load_0[offset + 5]);
                        _pk[3] = __floats2half2_rn(_tmem_load_0[offset + 6], _tmem_load_0[offset + 7]);
                        *reinterpret_cast<uint4*>(&((__half*)(O + (output_row + col_base + offset)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tile_done_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 4) {
        { // mma_warp_main
            int tile_1 = blockIdx.x;
            int batch_1 = tile_1 / 8;
            int kv_head_1 = tile_1 - batch_1 * 8;
            int seqlen_kv_1 = sequence_lengths[batch_1];
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_kv_full_0 = 0;
            if (seqlen_kv_1 != 1) {
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(kv_full_addr, _phase_kv_full_0);
                _phase_kv_full_0 ^= 1;
                if (elect_sync()) {
                    int _mma_a_lo_0 = ((q_smem_addr) >> 4) & 0x3FFF;
                    int _mma_b_lo_0 = (((kv_smem_addr) >> 4) & 0x3FFF) + (0) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68157456;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores), "r"(0));
                    tcgen05_commit(s_full_addr);
                }
            }
            unsigned int _phase_kv_full_1 = 0;
            mbarrier_wait(kv_full_addr + 8, _phase_kv_full_1);
            _phase_kv_full_1 ^= 1;
            unsigned int _phase_p_full_0 = 0;
            mbarrier_wait(p_full_addr, _phase_p_full_0);
            _phase_p_full_0 ^= 1;
            if (elect_sync()) {
                int _mma_b_lo_1 = ((((v_smem_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 1024;
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
                    "mov.b32 id, 69271568;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_1), "r"(tmem_scores + 32), "r"(0));
                tcgen05_commit(o_full_addr);
            }
            unsigned int _phase_tile_done_0 = 0;
            mbarrier_wait(tile_done_addr, _phase_tile_done_0);
            _phase_tile_done_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    }
    // ---- Role: load_warp ----
    if (warp == 5) {
        { // load_warp_main
            int tile_2 = blockIdx.x;
            int batch_2 = tile_2 / 8;
            int kv_head_2 = tile_2 - batch_2 * 8;
            int seqlen_kv_2 = sequence_lengths[batch_2];
            if (seqlen_kv_2 != 1) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 16384);
                    tma_5d_gmem2smem(q_smem_addr, (&Q), 0, 0, 0, kv_head_2, batch_2, q_full_addr);
                    tma_5d_gmem2smem(q_smem_addr + 8192, (&Q), 64, 0, 0, kv_head_2, batch_2, q_full_addr);
                }
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr, 16384);
                    tma_4d_gmem2smem(kv_smem_addr, (&K), 0, 0, kv_head_2, batch_2, kv_full_addr);
                    tma_4d_gmem2smem(kv_smem_addr + 8192, (&K), 64, 0, kv_head_2, batch_2, kv_full_addr);
                }
            }
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(kv_full_addr + 8, 16384);
                tma_4d_gmem2smem(kv_smem_addr + 16384, (&V), 0, 0, kv_head_2, batch_2, kv_full_addr + 8);
                tma_4d_gmem2smem(kv_smem_addr + 16384 + 8192, (&V), 64, 0, kv_head_2, batch_2, kv_full_addr + 8);
            }
        }
    }
    // ---- Role: empty ----
    if (warp >= 6 && warp <= 7) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
// clang-format on
