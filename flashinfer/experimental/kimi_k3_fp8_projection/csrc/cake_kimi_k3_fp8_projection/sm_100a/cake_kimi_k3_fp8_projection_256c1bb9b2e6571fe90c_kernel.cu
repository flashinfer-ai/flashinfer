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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 80
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFW_OFFSET 64
#define TMEM_TMEM_SFX_OFFSET 72
#define NUM_TMA_PIPE_STAGES 2
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_XB_PIPE_STAGES 3
#define SMEM_SMEM_W_OFF 1024
#define SMEM_SMEM_W_STAGE_BYTES 32768
#define SMEM_SMEM_W_STRIDE 51200
#define SMEM_SMEM_SFW0_OFF 50176
#define SMEM_SMEM_SFW0_STAGE_BYTES 512
#define SMEM_SMEM_SFW0_STRIDE 51200
#define SMEM_SMEM_SFW1_OFF 50688
#define SMEM_SMEM_SFW1_STAGE_BYTES 512
#define SMEM_SMEM_SFW1_STRIDE 51200
#define SMEM_SMEM_X_OFF 33792
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 51200
#define SMEM_SMEM_SFX_ALL_OFF 51200
#define SMEM_SMEM_SFX_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_SFX_ALL_STRIDE 51200
#define SMEM_SMEM_SFX0_OFF 51200
#define SMEM_SMEM_SFX0_STAGE_BYTES 512
#define SMEM_SMEM_SFX0_STRIDE 51200
#define SMEM_SMEM_SFX1_OFF 51712
#define SMEM_SMEM_SFX1_STAGE_BYTES 512
#define SMEM_SMEM_SFX1_STRIDE 51200
#define SMEM_SMEM_XB_OFF 103424
#define SMEM_SMEM_XB_STAGE_BYTES 32768
#define SMEM_SMEM_XB_STRIDE 32768
#define SMEM_SMEM_XB0_OFF 103424
#define SMEM_SMEM_XB0_STAGE_BYTES 16384
#define SMEM_SMEM_XB0_STRIDE 32768
#define SMEM_SMEM_XB1_OFF 119808
#define SMEM_SMEM_XB1_STAGE_BYTES 16384
#define SMEM_SMEM_XB1_STRIDE 32768
#define SMEM_SMEM_EPI_OFF 201728
#define SMEM_SMEM_EPI_STAGE_BYTES 16384
#define SMEM_SMEM_EPI_STRIDE 16384
#define SMEM_TOTAL 218112
#define THREADS 736

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
        :: "l"(mbar_addr), "r"(count) : "memory");
}


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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


__device__ __forceinline__ void tcgen05_mma_mxf8_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale"
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


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(736) void
kernel_cake_kimi_k3_fp8_projection_256c1bb9b2e6571fe90c(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap SFW, const __grid_constant__ CUtensorMap SFX, __nv_bfloat16* __restrict__ out, float* __restrict__ partials, unsigned int* __restrict__ counters, int M, int n_tiles, int n_valid, int ldo, int num_k_iters, int sf_k_tiles, int split, int tok_per_cta, int total_work, int store_vec, __nv_bfloat16* __restrict__ x, int K, const __grid_constant__ CUtensorMap XB)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 16)
    #define xb_full_addr (mbar_base + 32)
    #define xb_empty_addr (mbar_base + 56)
    #define xb_full1_addr (mbar_base + 80)
    #define xb_empty1_addr (mbar_base + 104)
    #define mainloop_done_addr (mbar_base + 128)
    #define epilogue_done_addr (mbar_base + 136)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_w = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w_addr = smem + 1024;
    uint8_t* smem_sfw0 = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int smem_sfw0_addr = smem + 50176;
    uint8_t* smem_sfw1 = reinterpret_cast<uint8_t*>(smem_raw + 50688);
    const int smem_sfw1_addr = smem + 50688;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_x_addr = smem + 33792;
    uint8_t* smem_sfx_all = reinterpret_cast<uint8_t*>(smem_raw + 51200);
    const int smem_sfx_all_addr = smem + 51200;
    uint8_t* smem_sfx0 = reinterpret_cast<uint8_t*>(smem_raw + 51200);
    const int smem_sfx0_addr = smem + 51200;
    uint8_t* smem_sfx1 = reinterpret_cast<uint8_t*>(smem_raw + 51712);
    const int smem_sfx1_addr = smem + 51712;
    uint8_t* smem_xb = reinterpret_cast<uint8_t*>(smem_raw + 103424);
    const int smem_xb_addr = smem + 103424;
    uint8_t* smem_xb0 = reinterpret_cast<uint8_t*>(smem_raw + 103424);
    const int smem_xb0_addr = smem + 103424;
    uint8_t* smem_xb1 = reinterpret_cast<uint8_t*>(smem_raw + 119808);
    const int smem_xb1_addr = smem + 119808;
    uint8_t* smem_epi = reinterpret_cast<uint8_t*>(smem_raw + 201728);
    const int smem_epi_addr = smem + 201728;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 18 barriers)
    // Mbarriers at smem_raw[0..144)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 2 barriers, init_count=17
            mbarrier_init(smem + 0, 17);
            mbarrier_init(smem + 8, 17);
            // mma_done: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'xb_pipe' ---
            // xb_full: 3 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // xb_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 56, 8);
            mbarrier_init(smem + 64, 8);
            mbarrier_init(smem + 72, 8);
            // xb_full1: 3 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            // xb_empty1: 3 barriers, init_count=8
            mbarrier_init(smem + 104, 8);
            mbarrier_init(smem + 112, 8);
            mbarrier_init(smem + 120, 8);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // epilogue_done: 1 barriers, init_count=4
            mbarrier_init(smem + 136, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 80 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfw = taddr + 64;
    const int tmem_tmem_sfx = taddr + 72;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                int w0 = bid;
                int ws = num_bids;
                int n_items = (total_work - bid + num_bids - 1) / num_bids;
                unsigned int load_stage = 0;
                #pragma unroll 1
                for (int it = 0; it < n_items; it++) {
                    int work = w0 + it * ws;
                    int tile = work / split;
                    int rank = work - tile * split;
                    int k_begin = num_k_iters * rank / split;
                    int k_end = num_k_iters * (rank + 1) / split;
                    int k_count = k_end - k_begin;
                    int n_tile = tile % n_tiles;
                    int m_tile = tile / n_tiles;
                    int w_tile0 = n_tile * (2 * num_k_iters);
                    int x_row = m_tile * 64;
                    int sfw_unit0 = n_tile / 2 * sf_k_tiles * 2 + n_tile % 2;
                    int sfx_unit0 = m_tile * sf_k_tiles;
                    int work_n = work + ws;
                    int tile_0 = work_n / split;
                    int rank_1 = work_n - tile_0 * split;
                    int k_begin_2 = num_k_iters * rank_1 / split;
                    int k_end_3 = num_k_iters * (rank_1 + 1) / split;
                    int k_count_4 = k_end_3 - k_begin_2;
                    int n_tile_n = tile_0 % n_tiles;
                    int m_tile_n = tile_0 / n_tiles;
                    int w_tile0_n = n_tile_n * (2 * num_k_iters);
                    int x_row_n = m_tile_n * 64;
                    int sfw_unit0_n = n_tile_n / 2 * sf_k_tiles * 2 + n_tile_n % 2;
                    #pragma unroll 1
                    for (int i = 0; i < k_count; i++) {
                        int iter_k = k_begin + i;
                        int k_pfi = k_begin_2 + (iter_k + 4 - (k_begin + k_count));
                        if (iter_k + 4 >= k_begin + k_count && n_items > it + 1 && k_pfi < k_begin_2 + k_count_4) {
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(0)), "r"((int)(w_tile0_n + k_pfi * 2)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0_n + k_pfi * 4)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0_n + k_pfi * 4 + 2)) : "memory");
                        }
                        if (i == 0) {
                            if (k_begin < k_begin + k_count) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(0)), "r"((int)(w_tile0 + k_begin * 2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + k_begin * 4)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + k_begin * 4 + 2)) : "memory");
                            }
                            if (k_begin + 1 < k_begin + k_count) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(0)), "r"((int)(w_tile0 + (k_begin + 1) * 2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + (k_begin + 1) * 4)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + (k_begin + 1) * 4 + 2)) : "memory");
                            }
                            if (k_begin + 2 < k_begin + k_count) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(0)), "r"((int)(w_tile0 + (k_begin + 2) * 2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + (k_begin + 2) * 4)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + (k_begin + 2) * 4 + 2)) : "memory");
                            }
                            if (k_begin + 3 < k_begin + k_count) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(0)), "r"((int)(w_tile0 + (k_begin + 3) * 2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + (k_begin + 3) * 4)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + (k_begin + 3) * 4 + 2)) : "memory");
                            }
                        }
                        int k_pf = iter_k + 4;
                        if (k_pf < k_begin + k_count) {
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(0)), "r"((int)(w_tile0 + k_pf * 2)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + k_pf * 4)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + k_pf * 4 + 2)) : "memory");
                        }
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k_group = iter_k * 2;
                        tma_3d_gmem2smem(smem_w_addr + load_stage * 51200, (&W), 0, 0, w_tile0 + k_group, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw0_addr + load_stage * 51200, (&SFW), 0, 0, sfw_unit0 + k_group * 2, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw1_addr + load_stage * 51200, (&SFW), 0, 0, sfw_unit0 + k_group * 2 + 2, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 33792);
                        load_stage += 1;
                        if (load_stage == 2) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            int w0_m = bid;
            int ws_m = num_bids;
            int n_items_m = (total_work - bid + num_bids - 1) / num_bids;
            unsigned int mma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int xq_m = 0;
            unsigned int res_stage_m = 0;
            int m_prev_m = -1;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (int it_m = 0; it_m < n_items_m; it_m++) {
                int work_m = w0_m + it_m * ws_m;
                int tile_1 = work_m / split;
                int rank_2 = work_m - tile_1 * split;
                int k_begin_1 = num_k_iters * rank_2 / split;
                int k_end_1 = num_k_iters * (rank_2 + 1) / split;
                int k_count_1 = k_end_1 - k_begin_1;
                mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
                #pragma unroll 1
                for (int i_1 = 0; i_1 < k_count_1; i_1++) {
                    mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((i_1 == 0) ? 1 : 0);
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfw, make_sf_cp_desc_lo_sbo128((((smem_sfw0_addr) >> 4) + (mma_stage) * 3200)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfx, make_sf_cp_desc_lo_sbo128((((smem_sfx0_addr) >> 4) + (mma_stage) * 3200)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfw + 4, make_sf_cp_desc_lo_sbo128((((smem_sfw1_addr) >> 4) + (mma_stage) * 3200)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfx + 4, make_sf_cp_desc_lo_sbo128((((smem_sfx1_addr) >> 4) + (mma_stage) * 3200)));
                        int _mma_a_lo_0 = (((smem_w_addr) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                        int _mma_b_lo_0 = (((smem_x_addr) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 64)), a_desc + 0, b_desc + 0,
                                0x8900000U, tmem_tmem_sfw, tmem_tmem_sfx, ((init_flag) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 64)), a_desc + 2, b_desc + 2,
                                0x28900010U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 64)), a_desc + 4, b_desc + 4,
                                0x48900020U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 64)), a_desc + 6, b_desc + 6,
                                0x68900030U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                        }
                        int _mma_a_lo_1 = (((smem_w_addr + 16384) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                        int _mma_b_lo_1 = (((smem_x_addr + 8192) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 64)), a_desc + 0, b_desc + 0,
                                0x8900000U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 64)), a_desc + 2, b_desc + 2,
                                0x28900010U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 64)), a_desc + 4, b_desc + 4,
                                0x48900020U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 64)), a_desc + 6, b_desc + 6,
                                0x68900030U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                        }
                    }
                    elect_commit(mma_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 2) { mma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(mainloop_done_addr + (acc_stage) * 8);
                _phase_epilogue_done ^= 1;
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int epi_group = (warp - 2) / 4;
            int tok_g0 = epi_group * 64;
            const int lane_row = epi_warp * 32 + lane;
            const int epi_tid = epi_warp * 32 + lane;
            int w0_e = bid;
            int ws_e = num_bids;
            int n_items_e = (total_work - bid + num_bids - 1) / num_bids;
            unsigned int acc_stage_e = 0;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (int it_e = 0; it_e < n_items_e; it_e++) {
                int work_e = w0_e + it_e * ws_e;
                int tile_2 = work_e / split;
                int rank_3 = work_e - tile_2 * split;
                int k_begin_3 = num_k_iters * rank_3 / split;
                int k_end_2 = num_k_iters * (rank_3 + 1) / split;
                int k_count_2 = k_end_2 - k_begin_3;
                int n_tile_e = tile_2 % n_tiles;
                int m_tile_e = tile_2 / n_tiles;
                int feature = n_tile_e * 128 + lane_row;
                int tok0 = m_tile_e * 64;
                unsigned long long col = (unsigned long long)feature;
                mbarrier_wait(mainloop_done_addr + (acc_stage_e) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_col_e = 0;
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    mbarrier_arrive(epilogue_done_addr + (acc_stage_e) * 8);
                }
                _phase_mainloop_done ^= 1;
                unsigned int epi_base = smem_epi_addr + (unsigned int)(epi_group * 16384);
                if (split == 1) {
                    #pragma unroll
                    for (int t = 0; t < 32; t++) {
                        {
                            uint32_t _addr_0 = static_cast<uint32_t>(smem_epi_addr + (unsigned int)(epi_group * 16384 + t * 512 + lane_row * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_0), "f"(_tmem_load_0[t]) : "memory");
                        }
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(2 + epi_group) : "memory");
                    if (store_vec == 8) {
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            int idx8 = j * 128 + epi_tid;
                            int row8 = idx8 >> 4;
                            int f8 = (idx8 & 15) * 8;
                            unsigned int words8[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words8[0])), "=r"(*reinterpret_cast<uint32_t*>(&words8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words8[(0) + 3]))
                                : "r"(epi_base + (unsigned int)(row8 * 512 + f8 * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words8[4])), "=r"(*reinterpret_cast<uint32_t*>(&words8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words8[(4) + 3]))
                                : "r"(epi_base + (unsigned int)(row8 * 512 + f8 * 4 + 16)));
                            float vals8[8];
                            #pragma unroll
                            for (int q = 0; q < 8; q++) {
                                float v8 = 0.0f;
                                v8 = reinterpret_cast<float*>(&words8[q])[0];
                                vals8[q] = v8;
                            }
                            int tok8 = tok0 + tok_g0 + row8;
                            int feat8 = n_tile_e * 128 + f8;
                            if (tok_g0 + row8 >= 0 && tok_g0 + row8 < 64 && tok8 < M && feat8 < n_valid) {
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(vals8[0 + 0], vals8[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(vals8[0 + 2], vals8[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(vals8[0 + 4], vals8[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(vals8[0 + 6], vals8[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + ((unsigned long long)tok8 * (unsigned long long)ldo + (unsigned long long)feat8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    } else if (store_vec == 4) {
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 8; j_1++) {
                            int idx4 = j_1 * 128 + epi_tid;
                            int row4 = idx4 >> 5;
                            int f4 = (idx4 & 31) * 4;
                            unsigned int words4[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words4[0])), "=r"(*reinterpret_cast<uint32_t*>(&words4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words4[(0) + 3]))
                                : "r"(epi_base + (unsigned int)(row4 * 512 + f4 * 4)));
                            float vals4[4];
                            #pragma unroll
                            for (int q_1 = 0; q_1 < 4; q_1++) {
                                float v4 = 0.0f;
                                v4 = reinterpret_cast<float*>(&words4[q_1])[0];
                                vals4[q_1] = v4;
                            }
                            int tok4 = tok0 + tok_g0 + row4;
                            int feat4 = n_tile_e * 128 + f4;
                            if (tok_g0 + row4 >= 0 && tok_g0 + row4 < 64 && tok4 < M && feat4 < n_valid) {
                                {
                                    uint2 _pk2;
                                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                    _pk[0] = __floats2bfloat162_rn(vals4[0 + 0], vals4[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(vals4[0 + 2], vals4[0 + 3]);
                                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out + ((unsigned long long)tok4 * (unsigned long long)ldo + (unsigned long long)feat4)))[0]) = _pk2;
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 16; j_2++) {
                            int idx2 = j_2 * 128 + epi_tid;
                            int row2 = idx2 >> 6;
                            int f2 = (idx2 & 63) * 2;
                            unsigned int words2[2];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words2[0])) : "r"(epi_base + (unsigned int)(row2 * 512 + f2 * 4)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words2[1])) : "r"(epi_base + (unsigned int)(row2 * 512 + f2 * 4 + 4)));
                            float vals2[2];
                            #pragma unroll
                            for (int q_2 = 0; q_2 < 2; q_2++) {
                                float v2 = 0.0f;
                                v2 = reinterpret_cast<float*>(&words2[q_2])[0];
                                vals2[q_2] = v2;
                            }
                            int tok2 = tok0 + tok_g0 + row2;
                            int feat2 = n_tile_e * 128 + f2;
                            if (tok_g0 + row2 >= 0 && tok_g0 + row2 < 64 && tok2 < M && feat2 < n_valid) {
                                {
                                    __nv_bfloat162 _pk = __floats2bfloat162_rn(vals2[0 + 0], vals2[0 + 1]);
                                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out + ((unsigned long long)tok2 * (unsigned long long)ldo + (unsigned long long)feat2)))[0]) = _pk;
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(2 + epi_group) : "memory");
                    #pragma unroll
                    for (int t_1 = 0; t_1 < 32; t_1++) {
                        {
                            uint32_t _addr_1 = static_cast<uint32_t>(smem_epi_addr + (unsigned int)(epi_group * 16384 + t_1 * 512 + lane_row * 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_1), "f"(_tmem_load_0[32 + t_1]) : "memory");
                        }
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(2 + epi_group) : "memory");
                    if (store_vec == 8) {
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 4; j_3++) {
                            int idx8_1 = j_3 * 128 + epi_tid;
                            int row8_1 = idx8_1 >> 4;
                            int f8_1 = (idx8_1 & 15) * 8;
                            unsigned int words8_1[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words8_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words8_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words8_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words8_1[(0) + 3]))
                                : "r"(epi_base + (unsigned int)(row8_1 * 512 + f8_1 * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words8_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&words8_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words8_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words8_1[(4) + 3]))
                                : "r"(epi_base + (unsigned int)(row8_1 * 512 + f8_1 * 4 + 16)));
                            float vals8_1[8];
                            #pragma unroll
                            for (int q_3 = 0; q_3 < 8; q_3++) {
                                float v8_1 = 0.0f;
                                v8_1 = reinterpret_cast<float*>(&words8_1[q_3])[0];
                                vals8_1[q_3] = v8_1;
                            }
                            int tok8_1 = tok0 + (tok_g0 + 32) + row8_1;
                            int feat8_1 = n_tile_e * 128 + f8_1;
                            if (tok_g0 + 32 + row8_1 >= 0 && tok_g0 + 32 + row8_1 < 64 && tok8_1 < M && feat8_1 < n_valid) {
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(vals8_1[0 + 0], vals8_1[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(vals8_1[0 + 2], vals8_1[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(vals8_1[0 + 4], vals8_1[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(vals8_1[0 + 6], vals8_1[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + ((unsigned long long)tok8_1 * (unsigned long long)ldo + (unsigned long long)feat8_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    } else if (store_vec == 4) {
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 8; j_4++) {
                            int idx4_1 = j_4 * 128 + epi_tid;
                            int row4_1 = idx4_1 >> 5;
                            int f4_1 = (idx4_1 & 31) * 4;
                            unsigned int words4_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words4_1[(0) + 3]))
                                : "r"(epi_base + (unsigned int)(row4_1 * 512 + f4_1 * 4)));
                            float vals4_1[4];
                            #pragma unroll
                            for (int q_4 = 0; q_4 < 4; q_4++) {
                                float v4_1 = 0.0f;
                                v4_1 = reinterpret_cast<float*>(&words4_1[q_4])[0];
                                vals4_1[q_4] = v4_1;
                            }
                            int tok4_1 = tok0 + (tok_g0 + 32) + row4_1;
                            int feat4_1 = n_tile_e * 128 + f4_1;
                            if (tok_g0 + 32 + row4_1 >= 0 && tok_g0 + 32 + row4_1 < 64 && tok4_1 < M && feat4_1 < n_valid) {
                                {
                                    uint2 _pk2;
                                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                    _pk[0] = __floats2bfloat162_rn(vals4_1[0 + 0], vals4_1[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(vals4_1[0 + 2], vals4_1[0 + 3]);
                                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out + ((unsigned long long)tok4_1 * (unsigned long long)ldo + (unsigned long long)feat4_1)))[0]) = _pk2;
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int j_5 = 0; j_5 < 16; j_5++) {
                            int idx2_1 = j_5 * 128 + epi_tid;
                            int row2_1 = idx2_1 >> 6;
                            int f2_1 = (idx2_1 & 63) * 2;
                            unsigned int words2_1[2];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words2_1[0])) : "r"(epi_base + (unsigned int)(row2_1 * 512 + f2_1 * 4)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words2_1[1])) : "r"(epi_base + (unsigned int)(row2_1 * 512 + f2_1 * 4 + 4)));
                            float vals2_1[2];
                            #pragma unroll
                            for (int q_5 = 0; q_5 < 2; q_5++) {
                                float v2_1 = 0.0f;
                                v2_1 = reinterpret_cast<float*>(&words2_1[q_5])[0];
                                vals2_1[q_5] = v2_1;
                            }
                            int tok2_1 = tok0 + (tok_g0 + 32) + row2_1;
                            int feat2_1 = n_tile_e * 128 + f2_1;
                            if (tok_g0 + 32 + row2_1 >= 0 && tok_g0 + 32 + row2_1 < 64 && tok2_1 < M && feat2_1 < n_valid) {
                                {
                                    __nv_bfloat162 _pk = __floats2bfloat162_rn(vals2_1[0 + 0], vals2_1[0 + 1]);
                                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out + ((unsigned long long)tok2_1 * (unsigned long long)ldo + (unsigned long long)feat2_1)))[0]) = _pk;
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(2 + epi_group) : "memory");
                } else {
                    unsigned long long pbase = (unsigned long long)(tile_2 * split + rank_3) * 8192 + (unsigned long long)lane_row + (unsigned long long)tok_g0 * 128;
                    #pragma unroll
                    for (int t_2 = 0; t_2 < 64; t_2++) {
                        partials[pbase + (unsigned long long)(t_2 * 128)] = _tmem_load_0[t_2];
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    if (tid == 64) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(counters) + (tile_2 * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(counters) + (tile_2 * 2))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>((unsigned int)split)) >= static_cast<unsigned int>(1));
                        }
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    int t_first = rank_3 * tok_per_cta;
                    int _min_2 = ((64) < (t_first + tok_per_cta) ? (64) : (t_first + tok_per_cta));
                    int t_last = _min_2;
                    unsigned long long rbase = (unsigned long long)(tile_2 * split) * 8192 + (unsigned long long)lane_row;
                    #pragma unroll
                    for (int tb = 0; tb < 64; tb += 8) {
                        if (t_first < tb + 8 && t_last > tb && tok_g0 <= tb && tb < tok_g0 + 64) {
                            float total[8];
                            #pragma unroll
                            for (int u = 0; u < 8; u++) {
                                total[u] = 0.0f;
                            }
                            #pragma unroll 1
                            for (int s_0 = 0; s_0 < split; s_0 += 8) {
                                float parts[64];
                                #pragma unroll
                                for (int j_6 = 0; j_6 < 8; j_6++) {
                                    int _min_3 = ((s_0 + j_6) < (split - 1) ? (s_0 + j_6) : (split - 1));
                                    int s_j = _min_3;
                                    unsigned long long rank_off = rbase + (unsigned long long)s_j * 8192;
                                    #pragma unroll
                                    for (int u_1 = 0; u_1 < 8; u_1++) {
                                        parts[u_1 * 8 + j_6] = partials[rank_off + (unsigned long long)((tb + u_1) * 128)];
                                    }
                                }
                                #pragma unroll
                                for (int j_7 = 0; j_7 < 8; j_7++) {
                                    if (s_0 + j_7 < split) {
                                        #pragma unroll
                                        for (int u_2 = 0; u_2 < 8; u_2++) {
                                            total[u_2] = total[u_2] + parts[u_2 * 8 + j_7];
                                        }
                                    }
                                }
                            }
                            if (feature < n_valid) {
                                #pragma unroll
                                for (int u_3 = 0; u_3 < 8; u_3++) {
                                    int tok_r = tok0 + tb + u_3;
                                    if (t_first <= tb + u_3 && t_last > tb + u_3 && tok_r < M) {
                                        *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_r * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total[u_3]);
                                    }
                                }
                            }
                        }
                    }
                    if (tid == 64) {
                        unsigned int _atomic_old_0 = atomicAdd(&counters[tile_2 * 2 + 1], 1);
                        unsigned int finished = _atomic_old_0;
                        if ((int)finished == split - 1) {
                            counters[(unsigned long long)(tile_2 * 2)] = 0;
                            counters[(unsigned long long)(tile_2 * 2 + 1)] = 0;
                        }
                    }
                }
            }
        }
    }
    // ---- Role: quant ----
    if (warp >= 6 && warp <= 21) {
        { // quant_main
            int w0_q = bid;
            int ws_q = num_bids;
            int n_items_q = (total_work - bid + num_bids - 1) / num_bids;
            const int qwarp = warp - 6;
            const int half = lane >> 4;
            const int half_lane = lane & 15;
            const int lane_q = lane;
            unsigned int q_stage = 0;
            unsigned int xb_q = 0;
            unsigned int xq_q = 0;
            unsigned int _phase_mma_done_1 = 1;
            unsigned int _phase_xb_full = 0;
            unsigned int _phase_xb_full1 = 0;
            #pragma unroll 1
            for (int it_q = 0; it_q < n_items_q; it_q++) {
                int work_q = w0_q + it_q * ws_q;
                int tile_3 = work_q / split;
                int rank_4 = work_q - tile_3 * split;
                int k_begin_4 = num_k_iters * rank_4 / split;
                int k_end_4 = num_k_iters * (rank_4 + 1) / split;
                int k_count_3 = k_end_4 - k_begin_4;
                int m_tile_q = tile_3 / n_tiles;
                int tok0_q = m_tile_q * 64;
                #pragma unroll 1
                for (int i_2 = 0; i_2 < k_count_3; i_2++) {
                    mbarrier_wait(mma_done_addr + (q_stage) * 8, _phase_mma_done_1);
                    unsigned int x_stage = smem_x_addr + q_stage * 51200;
                    unsigned int sf_stage_off = q_stage * 51200;
                    const int grp = lane_q / 4;
                    const int sub = lane_q - grp * 4;
                    const int stag = (grp & 1) * 4;
                    if (qwarp >= 0 && qwarp < 8) {
                        mbarrier_wait(xb_full_addr + (xb_q) * 8, _phase_xb_full);
                        unsigned int xb_half = smem_xb0_addr + xb_q * 32768;
                        unsigned int panel_h = x_stage;
                        #pragma unroll
                        for (int itn = 0; itn < 1; itn++) {
                            int tok_h = (itn * 16 + qwarp) * 8 + grp;
                            int tok_gh = tok0_q + tok_h;
                            unsigned int row_h = xb_half + (unsigned int)(tok_h * 256);
                            unsigned int wordsh[16];
                            #pragma unroll
                            for (int ci = 0; ci < 4; ci++) {
                                int chunk_h = sub + 4 * ci + stag & 15;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&wordsh[4 * ci])), "=r"(*reinterpret_cast<uint32_t*>(&wordsh[(4 * ci) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wordsh[(4 * ci) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wordsh[(4 * ci) + 3]))
                                    : "r"(row_h + (unsigned int)(chunk_h * 16)));
                            }
                            if (elect_sync()) {
                                mbarrier_arrive(xb_empty_addr + (xb_q) * 8);
                            }
                            float amax_h = 0.0f;
                            #pragma unroll
                            for (int ci_1 = 0; ci_1 < 4; ci_1++) {
                                float mags_h[8];
                                #pragma unroll
                                for (int j_8 = 0; j_8 < 4; j_8++) {
                                    unsigned int lo_bh = wordsh[4 * ci_1 + j_8] << 16;
                                    unsigned int hi_bh = wordsh[4 * ci_1 + j_8] & 4294901760u;
                                    float lo_fh = 0.0f;
                                    float hi_fh = 0.0f;
                                    lo_fh = reinterpret_cast<float*>(&lo_bh)[0];
                                    hi_fh = reinterpret_cast<float*>(&hi_bh)[0];
                                    mags_h[2 * j_8] = lo_fh;
                                    mags_h[2 * j_8 + 1] = hi_fh;
                                }
                                float _fabs_0 = fabsf(mags_h[0]);
                                mags_h[0] = _fabs_0;
                                float _fabs_1 = fabsf(mags_h[1]);
                                mags_h[1] = _fabs_1;
                                float _fabs_2 = fabsf(mags_h[2]);
                                mags_h[2] = _fabs_2;
                                float _fabs_3 = fabsf(mags_h[3]);
                                mags_h[3] = _fabs_3;
                                float _fabs_4 = fabsf(mags_h[4]);
                                mags_h[4] = _fabs_4;
                                float _fabs_5 = fabsf(mags_h[5]);
                                mags_h[5] = _fabs_5;
                                float _fabs_6 = fabsf(mags_h[6]);
                                mags_h[6] = _fabs_6;
                                float _fabs_7 = fabsf(mags_h[7]);
                                mags_h[7] = _fabs_7;
                                float mags_h_max = mags_h[0];
                                #pragma unroll
                                for (int _lr = 1; _lr < 8; _lr++) {
                                    mags_h_max = max_noftz(mags_h_max, mags_h[_lr]);
                                }
                                float _max_0 = max_noftz(amax_h, mags_h_max);
                                amax_h = _max_0;
                            }
                            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, amax_h, 1);
                            float o_h = _shfl_xor_0;
                            float _max_1 = max_noftz(amax_h, o_h);
                            amax_h = _max_1;
                            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, amax_h, 2);
                            float o_h_0 = _shfl_xor_1;
                            float _max_2 = max_noftz(amax_h, o_h_0);
                            amax_h = _max_2;
                            float _max_3 = max_noftz(amax_h, 0.0001f);
                            amax_h = _max_3;
                            unsigned int amax_bits_h = __as_u32(amax_h);
                            unsigned int sb_raw_h = (amax_bits_h + 2097151 >> 23) - 8;
                            unsigned int _max_4 = ((sb_raw_h) > (1) ? (sb_raw_h) : (1));
                            unsigned int _min_0 = ((_max_4) < (254) ? (_max_4) : (254));
                            unsigned int scale_byte_h = _min_0;
                            unsigned int inverse_bits_h = 254 - scale_byte_h << 23;
                            float inverse_h = 0.0f;
                            inverse_h = reinterpret_cast<float*>(&inverse_bits_h)[0];
                            unsigned int sf4_h = scale_byte_h * 16843009;
                            float sf4_bits_h = 0.0f;
                            sf4_bits_h = reinterpret_cast<float*>(&sf4_h)[0];
                            unsigned int sf_off_h = sf_stage_off + (unsigned int)((tok_h & 31) * 16 + (tok_h >> 5) * 4);
                            #pragma unroll
                            for (int ci_2 = 0; ci_2 < 4; ci_2++) {
                                int chunk_sh = sub + 4 * ci_2 + stag & 15;
                                float vals_h[8];
                                #pragma unroll
                                for (int j_9 = 0; j_9 < 4; j_9++) {
                                    unsigned int lo_ch = wordsh[4 * ci_2 + j_9] << 16;
                                    unsigned int hi_ch = wordsh[4 * ci_2 + j_9] & 4294901760u;
                                    float lo_vh = 0.0f;
                                    float hi_vh = 0.0f;
                                    lo_vh = reinterpret_cast<float*>(&lo_ch)[0];
                                    hi_vh = reinterpret_cast<float*>(&hi_ch)[0];
                                    vals_h[2 * j_9] = lo_vh;
                                    vals_h[2 * j_9 + 1] = hi_vh;
                                }
                                const float2 _scale2_0 = {inverse_h, inverse_h};
                                #pragma unroll
                                for (int _ls = 0; _ls < 4; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(vals_h)[_ls], _scale2_0);
                                uint16_t _e4m3x2_f32_0;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(vals_h[1]), "f"(vals_h[0]));
                                uint16_t eh0 = _e4m3x2_f32_0;
                                uint16_t _e4m3x2_f32_1;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(vals_h[3]), "f"(vals_h[2]));
                                uint16_t eh1 = _e4m3x2_f32_1;
                                uint16_t _e4m3x2_f32_2;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(vals_h[5]), "f"(vals_h[4]));
                                uint16_t eh2 = _e4m3x2_f32_2;
                                uint16_t _e4m3x2_f32_3;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(vals_h[7]), "f"(vals_h[6]));
                                uint16_t eh3 = _e4m3x2_f32_3;
                                unsigned int wh0 = (unsigned int)eh0 | (unsigned int)eh1 << 16;
                                unsigned int wh1 = (unsigned int)eh2 | (unsigned int)eh3 << 16;
                                if (tok_gh < M) {
                                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((panel_h + (unsigned int)(tok_h * 128 + chunk_sh * 8 ^ (tok_h * 128 + chunk_sh * 8 >> 7 & 7) << 4))), "r"(wh0), "r"(wh1) : "memory");
                                }
                            }
                            if (tok_gh < M) {
                                if (sub == 0) {
                                    {
                                        uint32_t _addr_1 = static_cast<uint32_t>(smem_sfx0_addr + sf_off_h);
                                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_1), "f"(sf4_bits_h) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    if (qwarp >= 8 && qwarp < 16) {
                        mbarrier_wait(xb_full1_addr + (xb_q) * 8, _phase_xb_full1);
                        unsigned int xb_half_1 = smem_xb1_addr + xb_q * 32768;
                        unsigned int panel_h_1 = x_stage + 8192;
                        #pragma unroll
                        for (int itn_1 = 0; itn_1 < 1; itn_1++) {
                            int tok_h_1 = (itn_1 * 16 + qwarp) * 8 + grp - 64;
                            int tok_gh_1 = tok0_q + tok_h_1;
                            unsigned int row_h_1 = xb_half_1 + (unsigned int)(tok_h_1 * 256);
                            unsigned int wordsh_1[16];
                            #pragma unroll
                            for (int ci_3 = 0; ci_3 < 4; ci_3++) {
                                int chunk_h_1 = sub + 4 * ci_3 + stag & 15;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&wordsh_1[4 * ci_3])), "=r"(*reinterpret_cast<uint32_t*>(&wordsh_1[(4 * ci_3) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wordsh_1[(4 * ci_3) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wordsh_1[(4 * ci_3) + 3]))
                                    : "r"(row_h_1 + (unsigned int)(chunk_h_1 * 16)));
                            }
                            if (elect_sync()) {
                                mbarrier_arrive(xb_empty1_addr + (xb_q) * 8);
                            }
                            float amax_h_1 = 0.0f;
                            #pragma unroll
                            for (int ci_4 = 0; ci_4 < 4; ci_4++) {
                                float mags_h_1[8];
                                #pragma unroll
                                for (int j_10 = 0; j_10 < 4; j_10++) {
                                    unsigned int lo_bh_1 = wordsh_1[4 * ci_4 + j_10] << 16;
                                    unsigned int hi_bh_1 = wordsh_1[4 * ci_4 + j_10] & 4294901760u;
                                    float lo_fh_1 = 0.0f;
                                    float hi_fh_1 = 0.0f;
                                    lo_fh_1 = reinterpret_cast<float*>(&lo_bh_1)[0];
                                    hi_fh_1 = reinterpret_cast<float*>(&hi_bh_1)[0];
                                    mags_h_1[2 * j_10] = lo_fh_1;
                                    mags_h_1[2 * j_10 + 1] = hi_fh_1;
                                }
                                float _fabs_8 = fabsf(mags_h_1[0]);
                                mags_h_1[0] = _fabs_8;
                                float _fabs_9 = fabsf(mags_h_1[1]);
                                mags_h_1[1] = _fabs_9;
                                float _fabs_10 = fabsf(mags_h_1[2]);
                                mags_h_1[2] = _fabs_10;
                                float _fabs_11 = fabsf(mags_h_1[3]);
                                mags_h_1[3] = _fabs_11;
                                float _fabs_12 = fabsf(mags_h_1[4]);
                                mags_h_1[4] = _fabs_12;
                                float _fabs_13 = fabsf(mags_h_1[5]);
                                mags_h_1[5] = _fabs_13;
                                float _fabs_14 = fabsf(mags_h_1[6]);
                                mags_h_1[6] = _fabs_14;
                                float _fabs_15 = fabsf(mags_h_1[7]);
                                mags_h_1[7] = _fabs_15;
                                float mags_h_max_1 = mags_h_1[0];
                                #pragma unroll
                                for (int _lr = 1; _lr < 8; _lr++) {
                                    mags_h_max_1 = max_noftz(mags_h_max_1, mags_h_1[_lr]);
                                }
                                float _max_5 = max_noftz(amax_h_1, mags_h_max_1);
                                amax_h_1 = _max_5;
                            }
                            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, amax_h_1, 1);
                            float o_h_1 = _shfl_xor_2;
                            float _max_6 = max_noftz(amax_h_1, o_h_1);
                            amax_h_1 = _max_6;
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, amax_h_1, 2);
                            float o_h_0_1 = _shfl_xor_3;
                            float _max_7 = max_noftz(amax_h_1, o_h_0_1);
                            amax_h_1 = _max_7;
                            float _max_8 = max_noftz(amax_h_1, 0.0001f);
                            amax_h_1 = _max_8;
                            unsigned int amax_bits_h_1 = __as_u32(amax_h_1);
                            unsigned int sb_raw_h_1 = (amax_bits_h_1 + 2097151 >> 23) - 8;
                            unsigned int _max_9 = ((sb_raw_h_1) > (1) ? (sb_raw_h_1) : (1));
                            unsigned int _min_1 = ((_max_9) < (254) ? (_max_9) : (254));
                            unsigned int scale_byte_h_1 = _min_1;
                            unsigned int inverse_bits_h_1 = 254 - scale_byte_h_1 << 23;
                            float inverse_h_1 = 0.0f;
                            inverse_h_1 = reinterpret_cast<float*>(&inverse_bits_h_1)[0];
                            unsigned int sf4_h_1 = scale_byte_h_1 * 16843009;
                            float sf4_bits_h_1 = 0.0f;
                            sf4_bits_h_1 = reinterpret_cast<float*>(&sf4_h_1)[0];
                            unsigned int sf_off_h_1 = sf_stage_off + (unsigned int)((tok_h_1 & 31) * 16 + (tok_h_1 >> 5) * 4);
                            #pragma unroll
                            for (int ci_5 = 0; ci_5 < 4; ci_5++) {
                                int chunk_sh_1 = sub + 4 * ci_5 + stag & 15;
                                float vals_h_1[8];
                                #pragma unroll
                                for (int j_11 = 0; j_11 < 4; j_11++) {
                                    unsigned int lo_ch_1 = wordsh_1[4 * ci_5 + j_11] << 16;
                                    unsigned int hi_ch_1 = wordsh_1[4 * ci_5 + j_11] & 4294901760u;
                                    float lo_vh_1 = 0.0f;
                                    float hi_vh_1 = 0.0f;
                                    lo_vh_1 = reinterpret_cast<float*>(&lo_ch_1)[0];
                                    hi_vh_1 = reinterpret_cast<float*>(&hi_ch_1)[0];
                                    vals_h_1[2 * j_11] = lo_vh_1;
                                    vals_h_1[2 * j_11 + 1] = hi_vh_1;
                                }
                                const float2 _scale2_2 = {inverse_h_1, inverse_h_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 4; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(vals_h_1)[_ls], _scale2_2);
                                uint16_t _e4m3x2_f32_4;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_4) : "f"(vals_h_1[1]), "f"(vals_h_1[0]));
                                uint16_t eh0_1 = _e4m3x2_f32_4;
                                uint16_t _e4m3x2_f32_5;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_5) : "f"(vals_h_1[3]), "f"(vals_h_1[2]));
                                uint16_t eh1_1 = _e4m3x2_f32_5;
                                uint16_t _e4m3x2_f32_6;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(vals_h_1[5]), "f"(vals_h_1[4]));
                                uint16_t eh2_1 = _e4m3x2_f32_6;
                                uint16_t _e4m3x2_f32_7;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_7) : "f"(vals_h_1[7]), "f"(vals_h_1[6]));
                                uint16_t eh3_1 = _e4m3x2_f32_7;
                                unsigned int wh0_1 = (unsigned int)eh0_1 | (unsigned int)eh1_1 << 16;
                                unsigned int wh1_1 = (unsigned int)eh2_1 | (unsigned int)eh3_1 << 16;
                                if (tok_gh_1 < M) {
                                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((panel_h_1 + (unsigned int)(tok_h_1 * 128 + chunk_sh_1 * 8 ^ (tok_h_1 * 128 + chunk_sh_1 * 8 >> 7 & 7) << 4))), "r"(wh0_1), "r"(wh1_1) : "memory");
                                }
                            }
                            if (tok_gh_1 < M) {
                                if (sub == 0) {
                                    {
                                        uint32_t _addr_3 = static_cast<uint32_t>(smem_sfx1_addr + sf_off_h_1);
                                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(sf4_bits_h_1) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(tma_full_addr + (q_stage) * 8);
                    }
                    q_stage += 1;
                    if (q_stage == 2) { q_stage = 0; _phase_mma_done_1 ^= 1; }
                    xb_q += 1;
                    if (xb_q == 3) { xb_q = 0; _phase_xb_full ^= 1; _phase_xb_full1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: xload ----
    if (warp == 22) {
        { // xload_main
            unsigned int _phase_xb_empty = 1;
            unsigned int _phase_xb_empty1 = 1;
            if (elect_sync()) {
                int w0_x = bid;
                int ws_x = num_bids;
                int n_items_x = (total_work - bid + num_bids - 1) / num_bids;
                unsigned int xb_stage_l = 0;
                #pragma unroll 1
                for (int it_x = 0; it_x < n_items_x; it_x++) {
                    int work_x = w0_x + it_x * ws_x;
                    int tile_4 = work_x / split;
                    int rank_5 = work_x - tile_4 * split;
                    int k_begin_5 = num_k_iters * rank_5 / split;
                    int k_end_5 = num_k_iters * (rank_5 + 1) / split;
                    int k_count_5 = k_end_5 - k_begin_5;
                    int m_tile_x = tile_4 / n_tiles;
                    int x_row_x = m_tile_x * 64;
                    #pragma unroll 1
                    for (int i_3 = 0; i_3 < k_count_5; i_3++) {
                        int k_group_x = (k_begin_5 + i_3) * 2;
                        mbarrier_wait(xb_empty_addr + (xb_stage_l) * 8, _phase_xb_empty);
                        if (it_x == 0 && i_3 == 0) {
                            asm volatile("griddepcontrol.wait;" ::: "memory");
                        }
                        tma_3d_gmem2smem(smem_xb0_addr + xb_stage_l * 32768, (&XB), 0, x_row_x, k_group_x, xb_full_addr + (xb_stage_l) * 8);
                        mbarrier_arrive_expect_tx(xb_full_addr + (xb_stage_l) * 8, 16384);
                        mbarrier_wait(xb_empty1_addr + (xb_stage_l) * 8, _phase_xb_empty1);
                        tma_3d_gmem2smem(smem_xb1_addr + xb_stage_l * 32768, (&XB), 0, x_row_x, k_group_x + 1, xb_full1_addr + (xb_stage_l) * 8);
                        mbarrier_arrive_expect_tx(xb_full1_addr + (xb_stage_l) * 8, 16384);
                        xb_stage_l += 1;
                        if (xb_stage_l == 3) { xb_stage_l = 0; _phase_xb_empty ^= 1; _phase_xb_empty1 ^= 1; }
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
