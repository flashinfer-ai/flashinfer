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

// clang-format off
// Frozen FlashInfer-generated CUDA device kernel.
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) BlackwellMsaTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) BlackwellMsaTensorMapPack { BlackwellMsaTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } BlackwellMsaGeneratedTensorMap;

#include <cuda_bf16.h>

#define BLACKWELL_MSA_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 16
#define TMEM_TMEM_O0_OFFSET 32
#define TMEM_TMEM_O1_OFFSET 48
#define TMEM_TMEM_STATS0_OFFSET 64
#define TMEM_TMEM_STATS1_OFFSET 80
#define TMEM_PREFILL_SCORES_0_OFFSET 0
#define TMEM_PREFILL_SCORES_1_OFFSET 128
#define TMEM_PREFILL_OUTPUT_0_OFFSET 256
#define TMEM_PREFILL_OUTPUT_1_OFFSET 384
#define NUM_DECODE_KV_STAGES 4
#define NUM_PREFILL_KV_PIPELINE_STAGES 2
#define SMEM_SMEM_CORR0_OFF 1024
#define SMEM_SMEM_CORR0_STAGE_BYTES 64
#define SMEM_SMEM_CORR0_STRIDE 64
#define SMEM_SMEM_CORR1_OFF 1088
#define SMEM_SMEM_CORR1_STAGE_BYTES 64
#define SMEM_SMEM_CORR1_STRIDE 64
#define SMEM_SMEM_EXCH0_OFF 1152
#define SMEM_SMEM_EXCH0_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_STRIDE 256
#define SMEM_SMEM_EXCH1_OFF 1408
#define SMEM_SMEM_EXCH1_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_STRIDE 256
#define SMEM_SMEM_QT_OFF 1664
#define SMEM_SMEM_QT_STAGE_BYTES 4096
#define SMEM_SMEM_QT_STRIDE 4096
#define SMEM_SMEM_KV_OFF 6144
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 6144
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_KV_FP8_OFF 6144
#define SMEM_SMEM_KV_FP8_STAGE_BYTES 16384
#define SMEM_SMEM_KV_FP8_STRIDE 16384
#define SMEM_SMEM_P0_OFF 137216
#define SMEM_SMEM_P0_STAGE_BYTES 4096
#define SMEM_SMEM_P0_STRIDE 4096
#define SMEM_SMEM_P1_OFF 141312
#define SMEM_SMEM_P1_STAGE_BYTES 4096
#define SMEM_SMEM_P1_STRIDE 4096
#define SMEM_SMEM_PAGE_INDICES_OFF 145408
#define SMEM_SMEM_PAGE_INDICES_STAGE_BYTES 2048
#define SMEM_SMEM_PAGE_INDICES_STRIDE 2048
#define SMEM_PREFILL_SCALE_OFF 1024
#define SMEM_PREFILL_SCALE_STAGE_BYTES 3072
#define SMEM_PREFILL_SCALE_STRIDE 3072
#define SMEM_PREFILL_Q0_OFF 4096
#define SMEM_PREFILL_Q0_STAGE_BYTES 32768
#define SMEM_PREFILL_Q0_STRIDE 32768
#define SMEM_PREFILL_PARTIAL_TILE_OFF 4096
#define SMEM_PREFILL_PARTIAL_TILE_STAGE_BYTES 65536
#define SMEM_PREFILL_PARTIAL_TILE_STRIDE 65536
#define SMEM_PREFILL_SPLIT_WEIGHTS_OFF 4096
#define SMEM_PREFILL_SPLIT_WEIGHTS_STAGE_BYTES 4096
#define SMEM_PREFILL_SPLIT_WEIGHTS_STRIDE 4096
#define SMEM_PREFILL_Q1_OFF 36864
#define SMEM_PREFILL_Q1_STAGE_BYTES 32768
#define SMEM_PREFILL_Q1_STRIDE 32768
#define SMEM_PREFILL_KV_OFF 69632
#define SMEM_PREFILL_KV_STAGE_BYTES 32768
#define SMEM_PREFILL_KV_STRIDE 32768
#define SMEM_PREFILL_V_OFF 69632
#define SMEM_PREFILL_V_STAGE_BYTES 32768
#define SMEM_PREFILL_V_STRIDE 32768
#define SMEM_TASK_OFFSETS_OFF 147968
#define SMEM_TASK_OFFSETS_STAGE_BYTES 2052
#define SMEM_TASK_OFFSETS_STRIDE 2052
#define SMEM_WORK_DESC_SLOTS_OFF 150032
#define SMEM_WORK_DESC_SLOTS_STAGE_BYTES 104
#define SMEM_WORK_DESC_SLOTS_STRIDE 104
#define SMEM_DECODE_ROW_MAX_OFF 150144
#define SMEM_DECODE_ROW_MAX_STAGE_BYTES 512
#define SMEM_DECODE_ROW_MAX_STRIDE 512
#define SMEM_DECODE_ROW_SUM_OFF 150656
#define SMEM_DECODE_ROW_SUM_STAGE_BYTES 512
#define SMEM_DECODE_ROW_SUM_STRIDE 512
#define SMEM_SPLIT_REDUCE_FLAG_OFF 145408
#define SMEM_SPLIT_REDUCE_FLAG_STAGE_BYTES 4
#define SMEM_SPLIT_REDUCE_FLAG_STRIDE 4
#define SMEM_TOTAL 151168
#define THREADS 512
#define MAX_REQUESTS_CONST 512

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
        :: "r"(mbar_addr), "r"(count));
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

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
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


__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
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


__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(*x0_ptr, -127.0f), x1 = max_noftz(*x1_ptr, -127.0f);
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
    *x0_ptr = r0; *x1_ptr = r1;
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

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        "}\n"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
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


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
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

__global__ __launch_bounds__(512) void
kernel_blackwell_msa_decode_fp16_flat(const __grid_constant__ BlackwellMsaTensorMap Q_value, const __grid_constant__ BlackwellMsaTensorMap Q_prefill_value, __half* __restrict__ Q_prefill_raw, const __grid_constant__ BlackwellMsaTensorMap K_value, const __grid_constant__ BlackwellMsaTensorMap K_prefill_pair_value, const __grid_constant__ BlackwellMsaTensorMap V_value, const __grid_constant__ BlackwellMsaTensorMap V_prefill_pair_value, const __grid_constant__ BlackwellMsaTensorMap KV_value, __half* __restrict__ O, float* __restrict__ partial_O, float* __restrict__ partial_M, float* __restrict__ partial_D, int* __restrict__ split_completion, float* __restrict__ msa_lse, int* __restrict__ kv_indices, int* __restrict__ qo_indptr, int* __restrict__ kv_indptr, int* __restrict__ kv_len_arr, int* __restrict__ task_kind, int* __restrict__ task_request, int* __restrict__ task_kv_head, int* __restrict__ task_q_tile, int* __restrict__ task_split, int* __restrict__ task_kv_tile_begin, int* __restrict__ task_kv_tile_end, int* __restrict__ task_qo_begin, int* __restrict__ task_qo_end, int* __restrict__ task_page_begin, int* __restrict__ task_page_end, int* __restrict__ status, int num_requests, int num_q_heads, int num_kv_heads, int max_kv_tiles, int max_splits, int max_task_claims, float softmax_scale_log2, int attention_mode, int is_causal, int derive_q_offset, int record_tasks, int msa_max_pages, int msa_split_policy)
{
    BlackwellMsaTensorMap const* Q = &Q_value;
    BlackwellMsaTensorMap const* Q_prefill = &Q_prefill_value;
    BlackwellMsaTensorMap const* K = &K_value;
    BlackwellMsaTensorMap const* K_prefill_pair = &K_prefill_pair_value;
    BlackwellMsaTensorMap const* V = &V_value;
    BlackwellMsaTensorMap const* V_prefill_pair = &V_prefill_pair_value;
    BlackwellMsaTensorMap const* KV = &KV_value;
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    __syncthreads();


    // Kernel setup ops
    float* smem_corr0 = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_corr0_addr = smem + 1024;
    float* smem_corr1 = reinterpret_cast<float*>(smem_raw + 1088);
    const int smem_corr1_addr = smem + 1088;
    float* smem_exch0 = reinterpret_cast<float*>(smem_raw + 1152);
    const int smem_exch0_addr = smem + 1152;
    float* smem_exch1 = reinterpret_cast<float*>(smem_raw + 1408);
    const int smem_exch1_addr = smem + 1408;
    __half* smem_qt = reinterpret_cast<__half*>(smem_raw + 1664);
    const int smem_qt_addr = smem + 1664;
    __half* smem_kv = reinterpret_cast<__half*>(smem_raw + 6144);
    const int smem_kv_addr = smem + 6144;
    __half* smem_v = reinterpret_cast<__half*>(smem_raw + 6144);
    const int smem_v_addr = smem + 6144;
    uint8_t* smem_kv_fp8 = reinterpret_cast<uint8_t*>(smem_raw + 6144);
    const int smem_kv_fp8_addr = smem + 6144;
    __half* smem_p0 = reinterpret_cast<__half*>(smem_raw + 137216);
    const int smem_p0_addr = smem + 137216;
    __half* smem_p1 = reinterpret_cast<__half*>(smem_raw + 141312);
    const int smem_p1_addr = smem + 141312;
    int* smem_page_indices = reinterpret_cast<int*>(smem_raw + 145408);
    const int smem_page_indices_addr = smem + 145408;
    float* prefill_scale = reinterpret_cast<float*>(smem_raw + 1024);
    const int prefill_scale_addr = smem + 1024;
    __half* prefill_q0 = reinterpret_cast<__half*>(smem_raw + 4096);
    const int prefill_q0_addr = smem + 4096;
    float* prefill_partial_tile = reinterpret_cast<float*>(smem_raw + 4096);
    const int prefill_partial_tile_addr = smem + 4096;
    float* prefill_split_weights = reinterpret_cast<float*>(smem_raw + 4096);
    const int prefill_split_weights_addr = smem + 4096;
    __half* prefill_q1 = reinterpret_cast<__half*>(smem_raw + 36864);
    const int prefill_q1_addr = smem + 36864;
    __half* prefill_kv = reinterpret_cast<__half*>(smem_raw + 69632);
    const int prefill_kv_addr = smem + 69632;
    __half* prefill_v = reinterpret_cast<__half*>(smem_raw + 69632);
    const int prefill_v_addr = smem + 69632;
    int* task_offsets = reinterpret_cast<int*>(smem_raw + 147968);
    const int task_offsets_addr = smem + 147968;
    int* work_desc_slots = reinterpret_cast<int*>(smem_raw + 150032);
    const int work_desc_slots_addr = smem + 150032;
    float* decode_row_max = reinterpret_cast<float*>(smem_raw + 150144);
    const int decode_row_max_addr = smem + 150144;
    float* decode_row_sum = reinterpret_cast<float*>(smem_raw + 150656);
    const int decode_row_sum_addr = smem + 150656;
    int* split_reduce_flag = reinterpret_cast<int*>(smem_raw + 145408);
    const int split_reduce_flag_addr = smem + 145408;

    // Mbarrier init (19 groups, 38 barriers)
    // Mbarriers at smem_raw[0..304)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // work_full_0: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // work_full_1: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // work_empty_0: 1 barriers, init_count=15
        mbarrier_init_pred(smem + 16, 15, leader);
        // work_empty_1: 1 barriers, init_count=15
        mbarrier_init_pred(smem + 24, 15, leader);
        // q_full: 2 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        mbarrier_init_pred(smem + 40, 1, leader);
        // q_tail_full: 2 barriers, init_count=32
        mbarrier_init_pred(smem + 48, 32, leader);
        mbarrier_init_pred(smem + 56, 32, leader);
        // --- pipeline 'decode_kv' ---
        // kv_full: 4 barriers, init_count=1
        mbarrier_init_pred(smem + 64, 1, leader);
        mbarrier_init_pred(smem + 72, 1, leader);
        mbarrier_init_pred(smem + 80, 1, leader);
        mbarrier_init_pred(smem + 88, 1, leader);
        // kv_src_full: 4 barriers, init_count=1
        mbarrier_init_pred(smem + 96, 1, leader);
        mbarrier_init_pred(smem + 104, 1, leader);
        mbarrier_init_pred(smem + 112, 1, leader);
        mbarrier_init_pred(smem + 120, 1, leader);
        // kv_empty: 4 barriers, init_count=1
        mbarrier_init_pred(smem + 128, 1, leader);
        mbarrier_init_pred(smem + 136, 1, leader);
        mbarrier_init_pred(smem + 144, 1, leader);
        mbarrier_init_pred(smem + 152, 1, leader);
        // --- pipeline 'prefill_kv_pipeline' ---
        // prefill_kv_full: 2 barriers, init_count=1
        mbarrier_init_pred(smem + 160, 1, leader);
        mbarrier_init_pred(smem + 168, 1, leader);
        // prefill_kv_empty: 2 barriers, init_count=1
        mbarrier_init_pred(smem + 176, 1, leader);
        mbarrier_init_pred(smem + 184, 1, leader);
        // s_full: 2 barriers, init_count=1
        mbarrier_init_pred(smem + 192, 1, leader);
        mbarrier_init_pred(smem + 200, 1, leader);
        // p_full: 2 barriers, init_count=256
        mbarrier_init_pred(smem + 208, 256, leader);
        mbarrier_init_pred(smem + 216, 256, leader);
        // p_full_tail: 2 barriers, init_count=256
        mbarrier_init_pred(smem + 224, 256, leader);
        mbarrier_init_pred(smem + 232, 256, leader);
        // corr_sig: 2 barriers, init_count=128
        mbarrier_init_pred(smem + 240, 128, leader);
        mbarrier_init_pred(smem + 248, 128, leader);
        // corr_done: 2 barriers, init_count=128
        mbarrier_init_pred(smem + 256, 128, leader);
        mbarrier_init_pred(smem + 264, 128, leader);
        // o_full: 2 barriers, init_count=1
        mbarrier_init_pred(smem + 272, 1, leader);
        mbarrier_init_pred(smem + 280, 1, leader);
        // decode_done: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 288, 128, leader);
        // prefill_partial_ready: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 296, 128, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 304);
    if (warp == 0) {
        int _tmem_hold = smem + 304;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define work_full_0_addr (mbar_base + 0)
    #define work_full_1_addr (mbar_base + 8)
    #define work_empty_0_addr (mbar_base + 16)
    #define work_empty_1_addr (mbar_base + 24)
    #define q_full_addr (mbar_base + 32)
    #define q_tail_full_addr (mbar_base + 48)
    #define kv_full_addr (mbar_base + 64)
    #define kv_src_full_addr (mbar_base + 96)
    #define kv_empty_addr (mbar_base + 128)
    #define prefill_kv_full_addr (mbar_base + 160)
    #define prefill_kv_empty_addr (mbar_base + 176)
    #define s_full_addr (mbar_base + 192)
    #define p_full_addr (mbar_base + 208)
    #define p_full_tail_addr (mbar_base + 224)
    #define corr_sig_addr (mbar_base + 240)
    #define corr_done_addr (mbar_base + 256)
    #define o_full_addr (mbar_base + 272)
    #define decode_done_addr (mbar_base + 288)
    #define prefill_partial_ready_addr (mbar_base + 296)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_s1 = taddr + 16;
    const int tmem_tmem_o0 = taddr + 32;
    const int tmem_tmem_o1 = taddr + 48;
    const int tmem_tmem_stats0 = taddr + 64;
    const int tmem_tmem_stats1 = taddr + 80;
    const int tmem_prefill_scores_0 = taddr;
    const int tmem_prefill_scores_1 = taddr + 128;
    const int tmem_prefill_output_0 = taddr + 256;
    const int tmem_prefill_output_1 = taddr + 384;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            int is_wg1 = ((warp >= 4) ? 1 : 0);
            int group_size = num_q_heads / num_kv_heads;
            const int tmem_row_base_v = warp % 4 * 32;
            int my_tmem_s = taddr + (unsigned int)(((is_wg1 != 0) ? 16 : 0)) + (unsigned int)(tmem_row_base_v << 16);
            int my_tmem_stats = taddr + (unsigned int)(((is_wg1 != 0) ? 80 : 64)) + (unsigned int)(tmem_row_base_v << 16);
            const int warp_in_wg = warp % 4;
            const int wg_tid = warp_in_wg * 32 + lane;
            float* my_exch_ptr = ((is_wg1 != 0) ? smem_exch1 : smem_exch0);
            float* my_corr_ptr = ((is_wg1 != 0) ? smem_corr1 : smem_corr0);
            unsigned int* base = ((is_wg1 != 0) ? reinterpret_cast<unsigned int*>(smem_p1) : reinterpret_cast<unsigned int*>(smem_p0));
            int direct_decode = ((attention_mode == 1 && gridDim.x * gridDim.y >= num_requests * num_kv_heads) ? 1 : 0);
            unsigned int _phase_work_full_0_0 = 0;
            unsigned int _phase_work_full_1_0 = 0;
            unsigned int _phase_s_full_1 = 0;
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_corr_done = 0;
            #pragma unroll 1
            for (int task_iter = 0; task_iter < max_task_claims + 1; task_iter++) {
                int work_slot = task_iter % 2;
                int* work_desc = work_desc_slots + (work_slot * 13);
                if (direct_decode == 0) {
                    if (work_slot == 0) {
                        mbarrier_wait(work_full_0_addr, _phase_work_full_0_0);
                        _phase_work_full_0_0 ^= 1;
                    } else {
                        mbarrier_wait(work_full_1_addr, _phase_work_full_1_0);
                        _phase_work_full_1_0 ^= 1;
                    }
                    asm volatile("barrier.sync 8, 480;" ::: "memory");
                }
                int ticket = -1;
                if (direct_decode != 0) {
                    if (task_iter == 0) {
                        ticket = blockIdx.x * num_kv_heads + blockIdx.y;
                    }
                } else {
                    ticket = work_desc[0];
                }
                if (ticket < 0) {
                    if (direct_decode == 0) {
                        if (elect_sync()) {
                            if (work_slot == 0) {
                                mbarrier_arrive(work_empty_0_addr);
                            } else {
                                mbarrier_arrive(work_empty_1_addr);
                            }
                        }
                    }
                    break;
                }
                int kind = ((direct_decode != 0) ? 1 : -1);
                if (direct_decode == 0 && attention_mode != 0) {
                    kind = work_desc[1];
                }
                int direct_request = 0;
                if (direct_decode != 0) {
                    direct_request = blockIdx.x;
                } else {
                    direct_request = ticket / num_kv_heads;
                    {
                        direct_request = 0;
                    }
                }
                int kv_tile_begin = 0;
                int direct_batch = direct_request;
                {
                    direct_batch = direct_request / record_tasks;
                }
                int direct_kv_len = kv_len_arr[direct_batch];
                int kv_tile_end = (direct_kv_len + 128 - 1) / 128;
                {
                    kv_tile_end = max_kv_tiles;
                }
                if (direct_decode == 0) {
                    kv_tile_begin = work_desc[6];
                    kv_tile_end = work_desc[7];
                }
                int num_n_blocks = kv_tile_end - kv_tile_begin;
                if (kind == 1) {
                    int num_pairs = num_n_blocks / 2;
                    const int row_state_base = warp * 16;
                    #pragma unroll
                    for (int c = 0; c < 16; c++) {
                        decode_row_max[row_state_base + c] = -BLACKWELL_MSA_INF;
                        decode_row_sum[row_state_base + c] = 0.0f;
                    }
                    int max_decode_pairs = max_kv_tiles / 2;
                    #pragma unroll 1
                    for (int pair = 0; pair < max_decode_pairs; pair++) {
                        if (num_pairs <= pair) {
                            break;
                        }
                        if (is_wg1 != 0) {
                            mbarrier_wait(s_full_addr + 8, _phase_s_full_1);
                            _phase_s_full_1 ^= 1;
                        } else {
                            mbarrier_wait(s_full_addr, _phase_s_full_0);
                            _phase_s_full_0 ^= 1;
                        }
                        float _tmem_load_0[16];
                        tmem_ld_x16(&_tmem_load_0[0], my_tmem_s);
                        {
                            int valid_cols = smem_page_indices[pair * 2 + is_wg1];
                            int token_in_block = warp_in_wg * 32 + lane;
                            if (token_in_block >= valid_cols) {
                                #pragma unroll
                                for (int c_1 = 0; c_1 < 16; c_1++) {
                                    _tmem_load_0[c_1] = -BLACKWELL_MSA_INF;
                                }
                            }
                        }
                        float partial_max[16];
                        #pragma unroll
                        for (int c_2 = 0; c_2 < 16; c_2++) {
                            partial_max[c_2] = _tmem_load_0[c_2];
                        }
                        #pragma unroll
                        for (int c_3 = 0; c_3 < 16; c_3++) {
                            float _warp_reduce_0 = partial_max[c_3];
                            #pragma unroll
                            for (int offset = 16; offset > 0; offset >>= 1)
                                _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                            partial_max[c_3] = _warp_reduce_0;
                        }
                        if (lane < 16) {
                            my_exch_ptr[warp_in_wg * 16 + lane] = partial_max[lane];
                        }
                        if (is_wg1 != 0) {
                            asm volatile("barrier.sync 12, 128;" ::: "memory");
                        } else {
                            asm volatile("barrier.sync 11, 128;" ::: "memory");
                        }
                        float tile_max[16];
                        if (lane < 16) {
                            float _max_0 = max_noftz(my_exch_ptr[lane], my_exch_ptr[16 + lane]);
                            float _max_1 = max_noftz(my_exch_ptr[32 + lane], my_exch_ptr[48 + lane]);
                            float _max_2 = max_noftz(_max_0, _max_1);
                            tile_max[lane] = _max_2;
                        }
                        #pragma unroll
                        for (int c_4 = 0; c_4 < 16; c_4++) {
                            float _shfl_0 = __shfl_sync(0xFFFFFFFF, tile_max[c_4], c_4);
                            tile_max[c_4] = _shfl_0;
                        }
                        float acc_scale[16];
                        #pragma unroll
                        for (int c_5 = 0; c_5 < 16; c_5++) {
                            float old_max = decode_row_max[row_state_base + c_5];
                            float _max_3 = max_noftz(old_max, tile_max[c_5]);
                            float new_max = _max_3;
                            decode_row_max[row_state_base + c_5] = new_max;
                            float delta = softmax_scale_log2 * (old_max - new_max);
                            float _exp2_0 = approx_exp2(delta);
                            acc_scale[c_5] = ((old_max > -BLACKWELL_MSA_INF) ? _exp2_0 : 1.0f);
                        }
                        tmem_st_x16_f32(my_tmem_stats, acc_scale);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (is_wg1 != 0) {
                            mbarrier_arrive(corr_sig_addr + 8);
                        } else {
                            mbarrier_arrive(corr_sig_addr);
                        }
                        float exp_vals[16];
                        #pragma unroll
                        for (int c_6 = 0; c_6 < 16; c_6++) {
                            float new_max_1 = decode_row_max[row_state_base + c_6];
                            float safe_max = ((new_max_1 == -BLACKWELL_MSA_INF) ? 0.0f : new_max_1);
                            float max_scaled = safe_max * softmax_scale_log2;
                            float _exp2_1 = approx_exp2(_tmem_load_0[c_6] * softmax_scale_log2 - max_scaled);
                            exp_vals[c_6] = _exp2_1;
                        }
                        float warp_sum[16];
                        #pragma unroll
                        for (int c_7 = 0; c_7 < 16; c_7++) {
                            warp_sum[c_7] = exp_vals[c_7];
                        }
                        #pragma unroll
                        for (int c_8 = 0; c_8 < 16; c_8++) {
                            float _warp_reduce_1 = warp_sum[c_8];
                            #pragma unroll
                            for (int offset = 16; offset > 0; offset >>= 1)
                                _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                            warp_sum[c_8] = _warp_reduce_1;
                        }
                        #pragma unroll
                        for (int c_9 = 0; c_9 < 16; c_9++) {
                            float old_sum = decode_row_sum[row_state_base + c_9];
                            decode_row_sum[row_state_base + c_9] = old_sum * acc_scale[c_9] + warp_sum[c_9];
                        }
                        #pragma unroll
                        for (int h = 0; h < 16; h++) {
                            {
                                __half _hval_1945028688 = __float2half_rn(exp_vals[h]);
                                uint16_t _bits_1945028688 = *(uint16_t*)&_hval_1945028688;
                                const void* _ptr_1945028688 = reinterpret_cast<const void*>((reinterpret_cast<uint8_t*>(base) + (wg_tid % 64 / 64 * 2048 + (wg_tid / 64 * 16 + h) * 128 + wg_tid % 64 % 64 * 2 ^ (wg_tid % 64 / 64 * 2048 + (wg_tid / 64 * 16 + h) * 128 + wg_tid % 64 % 64 * 2 >> 7 & 7) << 4)));
                                uint64_t _addr64_1945028688;
                                asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_addr64_1945028688) : "l"(_ptr_1945028688));
                                uint32_t _addr_1945028688;
                                asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_addr_1945028688) : "l"(_addr64_1945028688));
                                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1945028688), "h"(_bits_1945028688) : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async;");
                        if (is_wg1 != 0) {
                            mbarrier_arrive(p_full_addr + 8);
                        } else {
                            mbarrier_arrive(p_full_addr);
                        }
                    }
                    if (is_wg1 != 0) {
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                    }
                    if (lane < 16) {
                        my_exch_ptr[warp_in_wg * 16 + lane] = decode_row_sum[row_state_base + lane];
                    }
                    if (is_wg1 != 0) {
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                    }
                    float total_sum[16];
                    if (lane < 16) {
                        total_sum[lane] = my_exch_ptr[lane] + my_exch_ptr[16 + lane] + my_exch_ptr[32 + lane] + my_exch_ptr[48 + lane];
                    }
                    if (is_wg1 != 0) {
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                    }
                    if (warp_in_wg == 0 && lane < 16) {
                        my_corr_ptr[lane] = total_sum[lane];
                        my_exch_ptr[lane] = decode_row_max[row_state_base + lane];
                    }
                    if (is_wg1 != 0) {
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                    }
                    if (is_wg1 != 0) {
                        mbarrier_arrive(corr_sig_addr + 8);
                    } else {
                        mbarrier_arrive(corr_sig_addr);
                    }
                } else if (kind == 0) {
                    int request = work_desc[2];
                    int q_tile = work_desc[4];
                    int qo_begin = work_desc[8];
                    int qo_end = work_desc[9];
                    int q_len = qo_end - qo_begin;
                    int token_tiles = (q_len + 256 - 1) / 256;
                    int q_token_base = q_tile % token_tiles * 256;
                    int packed_gqa = 0;
                    if (group_size > 1) {
                        if (q_len * group_size <= 128) {
                            packed_gqa = 1;
                        }
                    }
                    int q_stages = ((q_len <= q_token_base + 128) ? 1 : 2);
                    if (is_wg1 == 0 || q_stages == 2) {
                        unsigned int stage = make_warp_uniform(warp / 4);
                        int tmem_s_off = make_warp_uniform(stage * 128);
                        int tmem_p_off = make_warp_uniform(stage * 128 + 64);
                        int scale_off = make_warp_uniform(stage * 128);
                        const int tmem_row_base = warp % 4 * 32 << 16;
                        int my_row = warp % 4 * 32 + lane;
                        int local_q_token = (unsigned int)q_token_base + stage * 128 + (unsigned int)my_row;
                        if (packed_gqa != 0) {
                            local_q_token = my_row / group_size;
                        }
                        int causal_row = kv_len_arr[request] - q_len + local_q_token;
                        int num_masked_iters = 0;
                        if (is_causal != 0) {
                            int causal_row_min = (unsigned int)(kv_len_arr[request] - q_len + q_token_base) + stage * 128;
                            if (packed_gqa != 0) {
                                causal_row_min = kv_len_arr[request] - q_len;
                            }
                            int n_block_no_mask_limit = (causal_row_min + 1) / 128;
                            int local_no_mask_blocks = n_block_no_mask_limit - kv_tile_begin;
                            if (local_no_mask_blocks < 0) {
                                local_no_mask_blocks = 0;
                            }
                            if (local_no_mask_blocks > num_n_blocks) {
                                local_no_mask_blocks = num_n_blocks;
                            }
                            num_masked_iters = num_n_blocks - local_no_mask_blocks;
                            if (num_masked_iters > num_n_blocks) {
                                num_masked_iters = num_n_blocks;
                            }
                        }
                        float row_max_val = -BLACKWELL_MSA_INF;
                        float row_sum_val = 0.0f;
                        #pragma unroll 1
                        for (int n_iter = 0; n_iter < num_n_blocks; n_iter++) {
                            int n_block = kv_tile_begin + num_n_blocks - 1 - n_iter;
                            if (stage == 0) {
                                mbarrier_wait(s_full_addr, _phase_s_full_0);
                                _phase_s_full_0 ^= 1;
                            } else {
                                mbarrier_wait(s_full_addr + 8, _phase_s_full_1);
                                _phase_s_full_1 ^= 1;
                            }
                            int s_base = taddr + (unsigned int)tmem_s_off + (unsigned int)tmem_row_base;
                            int valid_count = 128;
                            if (num_masked_iters > n_iter) {
                                valid_count = causal_row - n_block * 128 + 1;
                            }
                            float _tmem_load_1[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                                : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                                : "r"(s_base)
                                : "memory");
                            int body_valid = valid_count;
                            if (body_valid < 0) {
                                body_valid = 0;
                            }
                            if (body_valid < 64) {
                                uint32_t _slice_lo_mask_0;
                                {
                                    int _lim_0 = body_valid;
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
                                #pragma unroll
                                for (int _i_1 = 0; _i_1 < 32; _i_1++) {
                                    if (!(_slice_lo_mask_0 & (1u << _i_1))) _tmem_load_1[0 + _i_1] = -BLACKWELL_MSA_INF;
                                }
                                uint32_t _slice_lo_mask_1;
                                {
                                    int _lim_2 = body_valid - 32;
                                    if (_lim_2 <= 0) { _slice_lo_mask_1 = 0u; }
                                    else if (_lim_2 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_2));
                                    }
                                }
                                #pragma unroll
                                for (int _i_3 = 0; _i_3 < 32; _i_3++) {
                                    if (!(_slice_lo_mask_1 & (1u << _i_3))) _tmem_load_1[32 + _i_3] = -BLACKWELL_MSA_INF;
                                }
                            }
                            float2 _reg_reduce_max2_4 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                            row_max_x32_accum(&_tmem_load_1[0], _reg_reduce_max2_4);
                            row_max_x32_accum(&_tmem_load_1[32], _reg_reduce_max2_4);
                            float _tmem_load_1_max = row_max_reduce(_reg_reduce_max2_4);
                            float tile_max_1 = _tmem_load_1_max;
                            float _tmem_load_2[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                                : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31]), "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                                : "r"(s_base + 64)
                                : "memory");
                            int tail_valid = valid_count - 64;
                            if (tail_valid < 0) {
                                tail_valid = 0;
                            }
                            if (tail_valid < 64) {
                                uint32_t _slice_lo_mask_2;
                                {
                                    int _lim_5 = tail_valid;
                                    if (_lim_5 <= 0) { _slice_lo_mask_2 = 0u; }
                                    else if (_lim_5 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_5));
                                    }
                                }
                                #pragma unroll
                                for (int _i_6 = 0; _i_6 < 32; _i_6++) {
                                    if (!(_slice_lo_mask_2 & (1u << _i_6))) _tmem_load_2[0 + _i_6] = -BLACKWELL_MSA_INF;
                                }
                                uint32_t _slice_lo_mask_3;
                                {
                                    int _lim_7 = tail_valid - 32;
                                    if (_lim_7 <= 0) { _slice_lo_mask_3 = 0u; }
                                    else if (_lim_7 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_7));
                                    }
                                }
                                #pragma unroll
                                for (int _i_8 = 0; _i_8 < 32; _i_8++) {
                                    if (!(_slice_lo_mask_3 & (1u << _i_8))) _tmem_load_2[32 + _i_8] = -BLACKWELL_MSA_INF;
                                }
                            }
                            float2 _reg_reduce_max2_9 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                            row_max_x32_accum(&_tmem_load_2[0], _reg_reduce_max2_9);
                            row_max_x32_accum(&_tmem_load_2[32], _reg_reduce_max2_9);
                            float _tmem_load_2_max = row_max_reduce(_reg_reduce_max2_9);
                            float _max_4 = max_noftz(tile_max_1, _tmem_load_2_max);
                            tile_max_1 = _max_4;
                            float _max_5 = max_noftz(tile_max_1, row_max_val);
                            float new_max_2 = _max_5;
                            float safe_max_1 = ((new_max_2 == -BLACKWELL_MSA_INF) ? 0.0f : new_max_2);
                            float new_max_scaled = safe_max_1 * softmax_scale_log2;
                            float _fma_0 = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max_scaled);
                            float acc_scale_log2 = _fma_0;
                            float acc_scale_1;
                            float selected_max;
                            if (acc_scale_log2 >= -8.0f) {
                                selected_max = row_max_val;
                                safe_max_1 = ((row_max_val == -BLACKWELL_MSA_INF) ? 0.0f : row_max_val);
                                acc_scale_1 = 1.0f;
                                new_max_scaled = safe_max_1 * softmax_scale_log2;
                            } else {
                                selected_max = new_max_2;
                                float _exp2_2 = approx_exp2(acc_scale_log2);
                                acc_scale_1 = ((row_max_val > -BLACKWELL_MSA_INF) ? _exp2_2 : 1.0f);
                            }
                            row_max_val = selected_max;
                            prefill_scale[my_row + scale_off] = acc_scale_1;
                            mbarrier_arrive(corr_sig_addr + (stage) * 8);
                            int p_base = taddr + (unsigned int)tmem_p_off + (unsigned int)tmem_row_base;
                            const float2 _fma_b2_10 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_11 = {-new_max_scaled, -new_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 32; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_lf], _fma_b2_10, _fma_c2_11);
                            #pragma unroll
                            for (int _le = 0; _le < 64; _le++) {
                                _tmem_load_2[_le] = approx_exp2(_tmem_load_2[_le]);
                            }
                            unsigned int tail_probability[32];
                            #pragma unroll
                            for (int _lp = 0; _lp < 32; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                                tail_probability[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x32.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                :: "r"(p_base + 32), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[0])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[1])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[2])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[3])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[4])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[5])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[6])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[7])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[8])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[9])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[10])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[11])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[12])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[13])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[14])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[15])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[16])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[17])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[18])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[19])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[20])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[21])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[22])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[23])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[24])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[25])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[26])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[27])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[28])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[29])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[30])), "r"(*reinterpret_cast<const uint32_t*>(&tail_probability[31]))
                                : "memory");
                            float2 _reg_reduce_sum2_12 = make_float2(0.0f, 0.0f);
                            softmax_block_sum(&_tmem_load_2[0], &_reg_reduce_sum2_12);
                            softmax_block_sum(&_tmem_load_2[32], &_reg_reduce_sum2_12);
                            float _tmem_load_2_sum = _reg_reduce_sum2_12.x + _reg_reduce_sum2_12.y;
                            float block_sum = _tmem_load_2_sum;
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            mbarrier_arrive(p_full_tail_addr + (stage) * 8);
                            float _tmem_load_3[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                                : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31]), "=f"(_tmem_load_3[32]), "=f"(_tmem_load_3[33]), "=f"(_tmem_load_3[34]), "=f"(_tmem_load_3[35]), "=f"(_tmem_load_3[36]), "=f"(_tmem_load_3[37]), "=f"(_tmem_load_3[38]), "=f"(_tmem_load_3[39]), "=f"(_tmem_load_3[40]), "=f"(_tmem_load_3[41]), "=f"(_tmem_load_3[42]), "=f"(_tmem_load_3[43]), "=f"(_tmem_load_3[44]), "=f"(_tmem_load_3[45]), "=f"(_tmem_load_3[46]), "=f"(_tmem_load_3[47]), "=f"(_tmem_load_3[48]), "=f"(_tmem_load_3[49]), "=f"(_tmem_load_3[50]), "=f"(_tmem_load_3[51]), "=f"(_tmem_load_3[52]), "=f"(_tmem_load_3[53]), "=f"(_tmem_load_3[54]), "=f"(_tmem_load_3[55]), "=f"(_tmem_load_3[56]), "=f"(_tmem_load_3[57]), "=f"(_tmem_load_3[58]), "=f"(_tmem_load_3[59]), "=f"(_tmem_load_3[60]), "=f"(_tmem_load_3[61]), "=f"(_tmem_load_3[62]), "=f"(_tmem_load_3[63])
                                : "r"(s_base)
                                : "memory");
                            if (body_valid < 64) {
                                uint32_t _slice_lo_mask_4;
                                {
                                    int _lim_13 = body_valid;
                                    if (_lim_13 <= 0) { _slice_lo_mask_4 = 0u; }
                                    else if (_lim_13 >= 32) { _slice_lo_mask_4 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_4) : "r"(_lim_13));
                                    }
                                }
                                #pragma unroll
                                for (int _i_14 = 0; _i_14 < 32; _i_14++) {
                                    if (!(_slice_lo_mask_4 & (1u << _i_14))) _tmem_load_3[0 + _i_14] = -BLACKWELL_MSA_INF;
                                }
                                uint32_t _slice_lo_mask_5;
                                {
                                    int _lim_15 = body_valid - 32;
                                    if (_lim_15 <= 0) { _slice_lo_mask_5 = 0u; }
                                    else if (_lim_15 >= 32) { _slice_lo_mask_5 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_5) : "r"(_lim_15));
                                    }
                                }
                                #pragma unroll
                                for (int _i_16 = 0; _i_16 < 32; _i_16++) {
                                    if (!(_slice_lo_mask_5 & (1u << _i_16))) _tmem_load_3[32 + _i_16] = -BLACKWELL_MSA_INF;
                                }
                            }
                            const float2 _fma_b2_17 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_18 = {-new_max_scaled, -new_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 32; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_lf], _fma_b2_17, _fma_c2_18);
                            #pragma unroll
                            for (int _le = 0; _le < 64; _le++) {
                                _tmem_load_3[_le] = approx_exp2(_tmem_load_3[_le]);
                            }
                            unsigned int body_probability[32];
                            #pragma unroll
                            for (int _lp = 0; _lp < 32; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                                body_probability[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x32.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                :: "r"(p_base), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[0])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[1])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[2])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[3])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[4])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[5])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[6])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[7])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[8])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[9])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[10])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[11])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[12])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[13])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[14])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[15])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[16])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[17])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[18])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[19])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[20])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[21])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[22])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[23])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[24])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[25])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[26])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[27])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[28])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[29])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[30])), "r"(*reinterpret_cast<const uint32_t*>(&body_probability[31]))
                                : "memory");
                            float2 _reg_reduce_sum2_19 = make_float2(0.0f, 0.0f);
                            softmax_block_sum(&_tmem_load_3[0], &_reg_reduce_sum2_19);
                            softmax_block_sum(&_tmem_load_3[32], &_reg_reduce_sum2_19);
                            float _tmem_load_3_sum = _reg_reduce_sum2_19.x + _reg_reduce_sum2_19.y;
                            block_sum += _tmem_load_3_sum;
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            mbarrier_arrive(p_full_addr + (stage) * 8);
                            mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                            _phase_corr_done ^= 1;
                            row_sum_val = row_sum_val * acc_scale_1 + block_sum;
                        }
                        prefill_scale[my_row + scale_off + 256] = row_sum_val;
                        prefill_scale[my_row + scale_off + 512] = row_max_val;
                        mbarrier_arrive(corr_sig_addr + (stage) * 8);
                    }
                }
                if (direct_decode == 0) {
                    if (elect_sync()) {
                        if (work_slot == 0) {
                            mbarrier_arrive(work_empty_0_addr);
                        } else {
                            mbarrier_arrive(work_empty_1_addr);
                        }
                    }
                }
            }
        }
    // ---- Role: correction ----
    } else if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // correction_main
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            int d_idx = warp % 4 * 32 + lane;
            int direct_decode_1 = ((attention_mode == 1 && gridDim.x * gridDim.y >= num_requests * num_kv_heads) ? 1 : 0);
            unsigned int _phase_work_full_0_0_1 = 0;
            unsigned int _phase_work_full_1_0_1 = 0;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (int task_iter_1 = 0; task_iter_1 < max_task_claims + 1; task_iter_1++) {
                int work_slot_1 = task_iter_1 % 2;
                int* work_desc_1 = work_desc_slots + (work_slot_1 * 13);
                if (direct_decode_1 == 0) {
                    if (work_slot_1 == 0) {
                        mbarrier_wait(work_full_0_addr, _phase_work_full_0_0_1);
                        _phase_work_full_0_0_1 ^= 1;
                    } else {
                        mbarrier_wait(work_full_1_addr, _phase_work_full_1_0_1);
                        _phase_work_full_1_0_1 ^= 1;
                    }
                    asm volatile("barrier.sync 8, 480;" ::: "memory");
                }
                int ticket_1 = -1;
                if (direct_decode_1 != 0) {
                    if (task_iter_1 == 0) {
                        ticket_1 = blockIdx.x * num_kv_heads + blockIdx.y;
                    }
                } else {
                    ticket_1 = work_desc_1[0];
                }
                if (ticket_1 < 0) {
                    if (direct_decode_1 == 0) {
                        if (elect_sync()) {
                            if (work_slot_1 == 0) {
                                mbarrier_arrive(work_empty_0_addr);
                            } else {
                                mbarrier_arrive(work_empty_1_addr);
                            }
                        }
                    }
                    break;
                }
                int kind_1 = ((direct_decode_1 != 0) ? 1 : -1);
                if (direct_decode_1 == 0 && attention_mode != 0) {
                    kind_1 = work_desc_1[1];
                }
                int request_1 = 0;
                int kv_head = 0;
                int split = 0;
                int splits = 1;
                if (direct_decode_1 != 0) {
                    request_1 = blockIdx.x;
                    kv_head = blockIdx.y;
                } else {
                    request_1 = ticket_1 / num_kv_heads;
                    kv_head = ticket_1 % num_kv_heads;
                    {
                        request_1 = 0;
                    }
                }
                int kv_tile_begin_1 = 0;
                int direct_batch_1 = request_1;
                {
                    direct_batch_1 = request_1 / record_tasks;
                }
                int direct_kv_len_1 = kv_len_arr[direct_batch_1];
                int kv_tile_end_1 = (direct_kv_len_1 + 128 - 1) / 128;
                int qo_begin_1 = qo_indptr[request_1];
                {
                    kv_tile_end_1 = max_kv_tiles;
                    qo_begin_1 = request_1;
                }
                if (direct_decode_1 == 0) {
                    request_1 = work_desc_1[2];
                    kv_head = work_desc_1[3];
                    split = work_desc_1[5];
                    splits = work_desc_1[12];
                    kv_tile_begin_1 = work_desc_1[6];
                    kv_tile_end_1 = work_desc_1[7];
                    qo_begin_1 = work_desc_1[8];
                }
                int num_n_blocks_1 = kv_tile_end_1 - kv_tile_begin_1;
                int group_size_1 = num_q_heads / num_kv_heads;
                if (kind_1 == 1) {
                    int num_pairs_1 = num_n_blocks_1 / 2;
                    int max_decode_pairs_1 = max_kv_tiles / 2;
                    #pragma unroll 1
                    for (int pair_1 = 0; pair_1 < max_decode_pairs_1; pair_1++) {
                        if (num_pairs_1 <= pair_1) {
                            break;
                        }
                        mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                        _phase_corr_sig_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_4[16];
                        tmem_ld_x16(&_tmem_load_4[0], taddr + 64 + (unsigned int)corr_row);
                        float _tmem_load_5[16];
                        tmem_ld_x16(&_tmem_load_5[0], taddr + 32 + (unsigned int)corr_row);
                        #pragma unroll
                        for (int h_1 = 0; h_1 < 16; h_1++) {
                            _tmem_load_5[h_1] = _tmem_load_5[h_1] * _tmem_load_4[h_1];
                        }
                        tmem_st_x16_f32(taddr + 32 + (unsigned int)corr_row, _tmem_load_5);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(p_full_addr);
                        mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                        _phase_corr_sig_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_6[16];
                        tmem_ld_x16(&_tmem_load_6[0], taddr + 80 + (unsigned int)corr_row);
                        float _tmem_load_7[16];
                        tmem_ld_x16(&_tmem_load_7[0], taddr + 48 + (unsigned int)corr_row);
                        #pragma unroll
                        for (int h_2 = 0; h_2 < 16; h_2++) {
                            _tmem_load_7[h_2] = _tmem_load_7[h_2] * _tmem_load_6[h_2];
                        }
                        tmem_st_x16_f32(taddr + 48 + (unsigned int)corr_row, _tmem_load_7);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(p_full_addr + 8);
                    }
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                    _phase_corr_sig_1 ^= 1;
                    float scale0[16];
                    float scale1[16];
                    float final_sum[16];
                    float final_max[16];
                    #pragma unroll
                    for (int c_10 = 0; c_10 < 16; c_10++) {
                        float _shfl_1 = __shfl_sync(0xFFFFFFFF, smem_exch0[c_10], c_10);
                        float _shfl_2 = __shfl_sync(0xFFFFFFFF, smem_exch1[c_10], c_10);
                        float _shfl_3 = __shfl_sync(0xFFFFFFFF, smem_corr0[c_10], c_10);
                        float _shfl_4 = __shfl_sync(0xFFFFFFFF, smem_corr1[c_10], c_10);
                        float _max_6 = max_noftz(_shfl_1, _shfl_2);
                        float fm = _max_6;
                        final_max[c_10] = fm;
                        float d0 = ((_shfl_1 == -BLACKWELL_MSA_INF) ? 0.0f : softmax_scale_log2 * (_shfl_1 - fm));
                        float d1 = ((_shfl_2 == -BLACKWELL_MSA_INF) ? 0.0f : softmax_scale_log2 * (_shfl_2 - fm));
                        float _exp2_3 = approx_exp2(d0);
                        scale0[c_10] = _exp2_3;
                        float _exp2_4 = approx_exp2(d1);
                        scale1[c_10] = _exp2_4;
                        final_sum[c_10] = _shfl_3 * scale0[c_10] + _shfl_4 * scale1[c_10];
                    }
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float inv_sum[16];
                    #pragma unroll
                    for (int h_3 = 0; h_3 < 16; h_3++) {
                        float _rcp_0 = approx_rcp(final_sum[h_3]);
                        inv_sum[h_3] = ((final_sum[h_3] > 0.0f) ? _rcp_0 : 0.0f);
                    }
                    float _tmem_load_8[16];
                    tmem_ld_x16(&_tmem_load_8[0], taddr + 32 + (unsigned int)corr_row);
                    float _tmem_load_9[16];
                    tmem_ld_x16(&_tmem_load_9[0], taddr + 48 + (unsigned int)corr_row);
                    #pragma unroll
                    for (int h_4 = 0; h_4 < 16; h_4++) {
                        if (group_size_1 > h_4) {
                            float merged = _tmem_load_8[h_4] * scale0[h_4] + _tmem_load_9[h_4] * scale1[h_4];
                            int q_row = qo_begin_1 * num_q_heads + kv_head * group_size_1 + h_4;
                            int out_idx = q_row * 128 + d_idx;
                            if (splits == 1) {
                                {
                                    if (d_idx == 0) {
                                        float natural_lse = -BLACKWELL_MSA_INF;
                                        if (final_sum[h_4] > 0.0f) {
                                            float _log2_0;
                                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(final_sum[h_4]));
                                            natural_lse = final_max[h_4] * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f;
                                        }
                                        *((float*)(msa_lse + q_row)) = natural_lse;
                                    }
                                }
                                *((__half*)(O + out_idx)) = __float2half_rn(merged * inv_sum[h_4]);
                            } else {
                                int logical_output = request_1 * num_kv_heads + kv_head;
                                int partial_slot = logical_output * max_splits + split;
                                int partial_o_idx = (partial_slot * 128 + h_4) * 128 + d_idx;
                                *((float*)(partial_O + partial_o_idx)) = merged;
                            }
                        }
                    }
                    if (splits > 1) {
                        int logical_output_1 = request_1 * num_kv_heads + kv_head;
                        int partial_slot_1 = logical_output_1 * max_splits + split;
                        if (d_idx < group_size_1) {
                            int stat_idx = partial_slot_1 * 128 + d_idx;
                            float stat_m0 = smem_exch0[d_idx];
                            float stat_m1 = smem_exch1[d_idx];
                            float _max_7 = max_noftz(stat_m0, stat_m1);
                            float stat_m = _max_7;
                            float _exp2_5 = approx_exp2(softmax_scale_log2 * (stat_m0 - stat_m));
                            float stat_scale0 = ((stat_m0 == -BLACKWELL_MSA_INF) ? 0.0f : _exp2_5);
                            float _exp2_6 = approx_exp2(softmax_scale_log2 * (stat_m1 - stat_m));
                            float stat_scale1 = ((stat_m1 == -BLACKWELL_MSA_INF) ? 0.0f : _exp2_6);
                            float stat_d = smem_corr0[d_idx] * stat_scale0 + smem_corr1[d_idx] * stat_scale1;
                            *((float*)(partial_M + stat_idx)) = stat_m;
                            *((float*)(partial_D + stat_idx)) = stat_d;
                        }
                        __threadfence();
                        asm volatile("barrier.sync 13, 128;" ::: "memory");
                        if (d_idx == 0) {
                            int _atomic_old_0 = atomicAdd(&split_completion[logical_output_1], 1);
                            int old_count = _atomic_old_0;
                            split_reduce_flag[0] = ((old_count + 1 == splits) ? 1 : 0);
                        }
                        asm volatile("barrier.sync 13, 128;" ::: "memory");
                        if (split_reduce_flag[0] != 0) {
                            __threadfence();
                            #pragma unroll
                            for (int h_5 = 0; h_5 < 16; h_5++) {
                                if (group_size_1 > h_5) {
                                    float reduce_m = -BLACKWELL_MSA_INF;
                                    float reduce_d = 0.0f;
                                    float reduce_o = 0.0f;
                                    #pragma unroll 1
                                    for (int reduce_split = 0; reduce_split < max_splits; reduce_split++) {
                                        if (splits <= reduce_split) {
                                            break;
                                        }
                                        int reduce_slot = logical_output_1 * max_splits + reduce_split;
                                        int reduce_stat_idx = reduce_slot * 128 + h_5;
                                        float split_m = partial_M[reduce_stat_idx];
                                        float split_d = partial_D[reduce_stat_idx];
                                        int split_o_idx = reduce_stat_idx * 128 + d_idx;
                                        float split_o = partial_O[split_o_idx];
                                        float _max_8 = max_noftz(reduce_m, split_m);
                                        float new_m = _max_8;
                                        float _exp2_7 = approx_exp2(softmax_scale_log2 * (reduce_m - new_m));
                                        float old_scale = ((reduce_m == -BLACKWELL_MSA_INF) ? 0.0f : _exp2_7);
                                        float _exp2_8 = approx_exp2(softmax_scale_log2 * (split_m - new_m));
                                        float split_scale = ((split_m == -BLACKWELL_MSA_INF) ? 0.0f : _exp2_8);
                                        reduce_o = reduce_o * old_scale + split_o * split_scale;
                                        reduce_d = reduce_d * old_scale + split_d * split_scale;
                                        reduce_m = new_m;
                                    }
                                    int q_row_1 = qo_begin_1 * num_q_heads + kv_head * group_size_1 + h_5;
                                    int out_idx_1 = q_row_1 * 128 + d_idx;
                                    {
                                        if (d_idx == 0) {
                                            float natural_lse_1 = -BLACKWELL_MSA_INF;
                                            if (reduce_d > 0.0f) {
                                                float _log2_1;
                                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(reduce_d));
                                                natural_lse_1 = reduce_m * softmax_scale_log2 * 0.6931471805599453f + _log2_1 * 0.6931471805599453f;
                                            }
                                            *((float*)(msa_lse + q_row_1)) = natural_lse_1;
                                        }
                                    }
                                    float _rcp_1 = approx_rcp(reduce_d);
                                    *((__half*)(O + out_idx_1)) = __float2half_rn(((reduce_d > 0.0f) ? reduce_o * _rcp_1 : 0.0f));
                                }
                            }
                        }
                    }
                    mbarrier_arrive(decode_done_addr);
                } else if (kind_1 == 0) {
                    int request_0 = work_desc_1[2];
                    int q_tile_1 = work_desc_1[4];
                    int qo_end_1 = work_desc_1[9];
                    int q_len_1 = qo_end_1 - qo_begin_1;
                    int token_tiles_1 = (q_len_1 + 256 - 1) / 256;
                    int q_token_base_1 = q_tile_1 % token_tiles_1 * 256;
                    int q_head_local = q_tile_1 / token_tiles_1;
                    int q_head = kv_head * group_size_1 + q_head_local;
                    int packed_gqa_1 = 0;
                    if (group_size_1 > 1) {
                        if (q_len_1 * group_size_1 <= 128) {
                            packed_gqa_1 = 1;
                        }
                    }
                    int q_stages_1 = ((q_len_1 <= q_token_base_1 + 128) ? 1 : 2);
                    if (q_stages_1 == 1) {
                        int tmem_row_base_1 = warp % 4 * 32 << 16;
                        int my_row_1 = warp % 4 * 32 + lane;
                        mbarrier_arrive(p_full_addr);
                        mbarrier_arrive(p_full_tail_addr);
                        mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                        _phase_corr_sig_0 ^= 1;
                        mbarrier_arrive(corr_done_addr);
                        #pragma unroll 1
                        for (int _ = 1; _ < num_n_blocks_1; _++) {
                            mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                            _phase_corr_sig_0 ^= 1;
                            float scale0_1 = prefill_scale[my_row_1];
                            int _vote_0 = __any_sync(0xFFFFFFFF, scale0_1 < 1.0f);
                            if (_vote_0 != 0) {
                                #pragma unroll
                                for (int col = 0; col < 8; col++) {
                                    int addr0 = taddr + 256 + (unsigned int)tmem_row_base_1 + (unsigned int)(col * 16);
                                    float _tmem_load_10[16];
                                    tmem_ld_x16(&_tmem_load_10[0], addr0);
                                    const float2 _scale2_0 = {scale0_1, scale0_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_10)[_ls], _scale2_0);
                                    tmem_st_x16_f32(addr0, _tmem_load_10);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            mbarrier_arrive(p_full_addr);
                            mbarrier_arrive(p_full_tail_addr);
                            mbarrier_arrive(corr_done_addr);
                        }
                        mbarrier_wait(o_full_addr, _phase_o_full_0);
                        _phase_o_full_0 ^= 1;
                        mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                        _phase_corr_sig_0 ^= 1;
                        float final_sum_1 = prefill_scale[my_row_1 + 256];
                        float _rcp_2 = approx_rcp(final_sum_1);
                        float final_scale = ((final_sum_1 != 0.0f && final_sum_1 == final_sum_1) ? _rcp_2 : 0.0f);
                        int local_q_row = q_token_base_1 + my_row_1;
                        int output_q_head = q_head;
                        int valid_row = ((local_q_row < q_len_1) ? 1 : 0);
                        if (packed_gqa_1 != 0) {
                            local_q_row = my_row_1 / group_size_1;
                            output_q_head = kv_head * group_size_1 + my_row_1 % group_size_1;
                            valid_row = ((my_row_1 < q_len_1 * group_size_1) ? 1 : 0);
                        }
                        int output_row = (qo_begin_1 + local_q_row) * num_q_heads + output_q_head;
                        int partial_row_base = my_row_1 * 128;
                        #pragma unroll
                        for (int col_1 = 0; col_1 < 8; col_1++) {
                            int addr = taddr + 256 + (unsigned int)tmem_row_base_1 + (unsigned int)(col_1 * 16);
                            float _tmem_load_11[16];
                            tmem_ld_x16(&_tmem_load_11[0], addr);
                            if (valid_row != 0) {
                                if (splits > 1) {
                                    int partial_addr = prefill_partial_tile_addr + (unsigned int)((partial_row_base + col_1 * 16) * 4);
                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partial_addr), "f"(_tmem_load_11[0]), "f"(_tmem_load_11[1]), "f"(_tmem_load_11[2]), "f"(_tmem_load_11[3]) : "memory");
                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partial_addr + 16), "f"(_tmem_load_11[4]), "f"(_tmem_load_11[5]), "f"(_tmem_load_11[6]), "f"(_tmem_load_11[7]) : "memory");
                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partial_addr + 32), "f"(_tmem_load_11[8]), "f"(_tmem_load_11[9]), "f"(_tmem_load_11[10]), "f"(_tmem_load_11[11]) : "memory");
                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partial_addr + 48), "f"(_tmem_load_11[12]), "f"(_tmem_load_11[13]), "f"(_tmem_load_11[14]), "f"(_tmem_load_11[15]) : "memory");
                                }
                                {
                                    const float2 _prescale2_1 = {final_scale, final_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_11[0])[_ps], _prescale2_1);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_11[0 + _ps] *= final_scale;
                                    #endif
                                    __nv_bfloat162 _pk[8];
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_11[0 + 0], _tmem_load_11[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_11[0 + 2], _tmem_load_11[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_tmem_load_11[0 + 4], _tmem_load_11[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_tmem_load_11[0 + 6], _tmem_load_11[0 + 7]);
                                    _pk[4] = __floats2bfloat162_rn(_tmem_load_11[0 + 8], _tmem_load_11[0 + 9]);
                                    _pk[5] = __floats2bfloat162_rn(_tmem_load_11[0 + 10], _tmem_load_11[0 + 11]);
                                    _pk[6] = __floats2bfloat162_rn(_tmem_load_11[0 + 12], _tmem_load_11[0 + 13]);
                                    _pk[7] = __floats2bfloat162_rn(_tmem_load_11[0 + 14], _tmem_load_11[0 + 15]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (output_row * 128 + col_1 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (output_row * 128 + col_1 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                                }
                            }
                        }
                        if (splits > 1) {
                            mbarrier_arrive(prefill_partial_ready_addr);
                        }
                    } else {
                        int tmem_row_base_2 = warp % 4 * 32 << 16;
                        int my_row_2 = warp % 4 * 32 + lane;
                        mbarrier_arrive(p_full_addr);
                        mbarrier_arrive(p_full_addr + 8);
                        mbarrier_arrive(p_full_tail_addr);
                        mbarrier_arrive(p_full_tail_addr + 8);
                        mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                        _phase_corr_sig_0 ^= 1;
                        mbarrier_arrive(corr_done_addr);
                        mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                        _phase_corr_sig_1 ^= 1;
                        #pragma unroll 1
                        for (int __1 = 1; __1 < num_n_blocks_1; __1++) {
                            mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                            _phase_corr_sig_0 ^= 1;
                            float scale0_2 = prefill_scale[my_row_2];
                            int _vote_1 = __any_sync(0xFFFFFFFF, scale0_2 < 1.0f);
                            if (_vote_1 != 0) {
                                #pragma unroll
                                for (int col_2 = 0; col_2 < 8; col_2++) {
                                    int addr0_1 = taddr + 256 + (unsigned int)tmem_row_base_2 + (unsigned int)(col_2 * 16);
                                    float _tmem_load_12[16];
                                    tmem_ld_x16(&_tmem_load_12[0], addr0_1);
                                    const float2 _scale2_2 = {scale0_2, scale0_2};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_12)[_ls], _scale2_2);
                                    tmem_st_x16_f32(addr0_1, _tmem_load_12);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            mbarrier_arrive(p_full_addr);
                            mbarrier_arrive(p_full_tail_addr);
                            mbarrier_arrive(corr_done_addr + 8);
                            mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                            _phase_corr_sig_1 ^= 1;
                            float scale1_1 = prefill_scale[my_row_2 + 128];
                            int _vote_2 = __any_sync(0xFFFFFFFF, scale1_1 < 1.0f);
                            if (_vote_2 != 0) {
                                #pragma unroll
                                for (int col_3 = 0; col_3 < 8; col_3++) {
                                    int addr1 = taddr + 384 + (unsigned int)tmem_row_base_2 + (unsigned int)(col_3 * 16);
                                    float _tmem_load_13[16];
                                    tmem_ld_x16(&_tmem_load_13[0], addr1);
                                    const float2 _scale2_3 = {scale1_1, scale1_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_13)[_ls], _scale2_3);
                                    tmem_st_x16_f32(addr1, _tmem_load_13);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            mbarrier_arrive(p_full_addr + 8);
                            mbarrier_arrive(p_full_tail_addr + 8);
                            mbarrier_arrive(corr_done_addr);
                        }
                        mbarrier_arrive(corr_done_addr + 8);
                        mbarrier_wait(o_full_addr, _phase_o_full_0);
                        _phase_o_full_0 ^= 1;
                        mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
                        _phase_o_full_1 ^= 1;
                        mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                        _phase_corr_sig_0 ^= 1;
                        mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                        _phase_corr_sig_1 ^= 1;
                        #pragma unroll
                        for (int stage_1 = 0; stage_1 < 2; stage_1++) {
                            int output_offset = 256 + stage_1 * 128;
                            int scale_offset = stage_1 * 128;
                            float final_sum_2 = prefill_scale[my_row_2 + scale_offset + 256];
                            float _rcp_3 = approx_rcp(final_sum_2);
                            float final_scale_1 = ((final_sum_2 != 0.0f && final_sum_2 == final_sum_2) ? _rcp_3 : 0.0f);
                            int local_q_row_1 = q_token_base_1 + stage_1 * 128 + my_row_2;
                            int output_row_1 = (qo_begin_1 + local_q_row_1) * num_q_heads + q_head;
                            #pragma unroll
                            for (int col_4 = 0; col_4 < 8; col_4++) {
                                int addr_1 = taddr + (unsigned int)output_offset + (unsigned int)tmem_row_base_2 + (unsigned int)(col_4 * 16);
                                float _tmem_load_14[16];
                                tmem_ld_x16(&_tmem_load_14[0], addr_1);
                                if (local_q_row_1 < q_len_1) {
                                    {
                                        const float2 _prescale2_4 = {final_scale_1, final_scale_1};
                                        #if __CUDA_ARCH__ >= 1000
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 8; _ps++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_14[0])[_ps], _prescale2_4);
                                        #else
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 16; _ps++)
                                            _tmem_load_14[0 + _ps] *= final_scale_1;
                                        #endif
                                        __nv_bfloat162 _pk[8];
                                        _pk[0] = __floats2bfloat162_rn(_tmem_load_14[0 + 0], _tmem_load_14[0 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(_tmem_load_14[0 + 2], _tmem_load_14[0 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(_tmem_load_14[0 + 4], _tmem_load_14[0 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(_tmem_load_14[0 + 6], _tmem_load_14[0 + 7]);
                                        _pk[4] = __floats2bfloat162_rn(_tmem_load_14[0 + 8], _tmem_load_14[0 + 9]);
                                        _pk[5] = __floats2bfloat162_rn(_tmem_load_14[0 + 10], _tmem_load_14[0 + 11]);
                                        _pk[6] = __floats2bfloat162_rn(_tmem_load_14[0 + 12], _tmem_load_14[0 + 13]);
                                        _pk[7] = __floats2bfloat162_rn(_tmem_load_14[0 + 14], _tmem_load_14[0 + 15]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (output_row_1 * 128 + col_4 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (output_row_1 * 128 + col_4 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                                    }
                                }
                            }
                        }
                    }
                }
                if (direct_decode_1 == 0) {
                    if (elect_sync()) {
                        if (work_slot_1 == 0) {
                            mbarrier_arrive(work_empty_0_addr);
                        } else {
                            mbarrier_arrive(work_empty_1_addr);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 12) {
        { // mma_main
            int direct_decode_2 = ((attention_mode == 1 && gridDim.x * gridDim.y >= num_requests * num_kv_heads) ? 1 : 0);
            unsigned int _phase_work_full_0_0_2 = 0;
            unsigned int _phase_work_full_1_0_2 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_decode_done_0 = 0;
            unsigned int _phase_q_tail_full_0 = 0;
            unsigned int _phase_p_full_tail_0 = 0;
            unsigned int _phase_q_tail_full_1 = 0;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_p_full_tail_1 = 0;
            #pragma unroll 1
            for (int task_iter_2 = 0; task_iter_2 < max_task_claims + 1; task_iter_2++) {
                int work_slot_2 = task_iter_2 % 2;
                int* work_desc_2 = work_desc_slots + (work_slot_2 * 13);
                if (direct_decode_2 == 0) {
                    if (work_slot_2 == 0) {
                        mbarrier_wait(work_full_0_addr, _phase_work_full_0_0_2);
                        _phase_work_full_0_0_2 ^= 1;
                    } else {
                        mbarrier_wait(work_full_1_addr, _phase_work_full_1_0_2);
                        _phase_work_full_1_0_2 ^= 1;
                    }
                    asm volatile("barrier.sync 8, 480;" ::: "memory");
                }
                int ticket_2 = -1;
                if (direct_decode_2 != 0) {
                    if (task_iter_2 == 0) {
                        ticket_2 = blockIdx.x * num_kv_heads + blockIdx.y;
                    }
                } else {
                    ticket_2 = work_desc_2[0];
                }
                if (ticket_2 < 0) {
                    if (direct_decode_2 == 0) {
                        if (elect_sync()) {
                            if (work_slot_2 == 0) {
                                mbarrier_arrive(work_empty_0_addr);
                            } else {
                                mbarrier_arrive(work_empty_1_addr);
                            }
                        }
                    }
                    break;
                }
                int kind_2 = ((direct_decode_2 != 0) ? 1 : -1);
                if (direct_decode_2 == 0 && attention_mode != 0) {
                    kind_2 = work_desc_2[1];
                }
                int direct_request_1 = 0;
                if (direct_decode_2 != 0) {
                    direct_request_1 = blockIdx.x;
                } else {
                    direct_request_1 = ticket_2 / num_kv_heads;
                    {
                        direct_request_1 = 0;
                    }
                }
                int kv_tile_begin_2 = 0;
                int direct_batch_2 = direct_request_1;
                {
                    direct_batch_2 = direct_request_1 / record_tasks;
                }
                int direct_kv_len_2 = kv_len_arr[direct_batch_2];
                int kv_tile_end_2 = (direct_kv_len_2 + 128 - 1) / 128;
                {
                    kv_tile_end_2 = max_kv_tiles;
                }
                if (direct_decode_2 == 0) {
                    kv_tile_begin_2 = work_desc_2[6];
                    kv_tile_end_2 = work_desc_2[7];
                }
                int num_n_blocks_2 = kv_tile_end_2 - kv_tile_begin_2;
                if (kind_2 == 1) {
                    int num_pairs_2 = num_n_blocks_2 / 2;
                    int inst0_stage = 0;
                    int first_pv0 = 1;
                    int first_pv1 = 1;
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    mbarrier_wait(kv_full_addr, 0);
                    int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_tmem_s0), "r"(0));
                    elect_commit(s_full_addr);
                    elect_commit(kv_empty_addr);
                    int max_decode_pairs_2 = max_kv_tiles / 2;
                    #pragma unroll 1
                    for (int pair_2 = 0; pair_2 < max_decode_pairs_2 - 1; pair_2++) {
                        if (pair_2 >= num_pairs_2 - 1) {
                            break;
                        }
                        int s0 = inst0_stage;
                        int s1 = (inst0_stage + 1) % 4;
                        int s0_next = (inst0_stage + 2) % 4;
                        mbarrier_wait(kv_full_addr + (s1) * 8, 0);
                        int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1) * 2048);
                        int _mma_b_lo_1 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_tmem_s1), "r"(0));
                        elect_commit(s_full_addr + 8);
                        elect_commit(kv_empty_addr + (s1) * 8);
                        mbarrier_wait(kv_full_addr + (s0) * 8, 1);
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        _phase_p_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0) * 2048);
                        int _mma_b_lo_2 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134512656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
                        elect_commit(kv_empty_addr + (s0) * 8);
                        mbarrier_wait(kv_full_addr + (s0_next) * 8, 0);
                        int _mma_a_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s0_next) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_1), "r"(tmem_tmem_s0), "r"(0));
                        elect_commit(s_full_addr);
                        elect_commit(kv_empty_addr + (s0_next) * 8);
                        mbarrier_wait(kv_full_addr + (s1) * 8, 1);
                        mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                        _phase_p_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1) * 2048);
                        int _mma_b_lo_4 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134512656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
                        elect_commit(kv_empty_addr + (s1) * 8);
                        inst0_stage = s0_next;
                        first_pv0 = 0;
                        first_pv1 = 0;
                    }
                    int s0_last = inst0_stage;
                    int s1_last = (inst0_stage + 1) % 4;
                    mbarrier_wait(kv_full_addr + (s1_last) * 8, 0);
                    int _mma_a_lo_5 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1_last) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_0), "r"(tmem_tmem_s1), "r"(0));
                    elect_commit(s_full_addr + 8);
                    elect_commit(kv_empty_addr + (s1_last) * 8);
                    mbarrier_wait(kv_full_addr + (s0_last) * 8, 1);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_last) * 2048);
                    int _mma_b_lo_6 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134512656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
                    elect_commit(kv_empty_addr + (s0_last) * 8);
                    mbarrier_wait(kv_full_addr + (s1_last) * 8, 1);
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_last) * 2048);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134512656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
                    elect_commit(kv_empty_addr + (s1_last) * 8);
                    elect_commit(o_full_addr);
                    mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                    _phase_decode_done_0 ^= 1;
                } else if (kind_2 == 0) {
                    int q_tile_2 = work_desc_2[4];
                    int qo_begin_2 = work_desc_2[8];
                    int qo_end_2 = work_desc_2[9];
                    int q_len_2 = qo_end_2 - qo_begin_2;
                    int token_tiles_2 = (q_len_2 + 256 - 1) / 256;
                    int q_token_base_2 = q_tile_2 % token_tiles_2 * 256;
                    int q_stages_2 = ((q_len_2 <= q_token_base_2 + 128) ? 1 : 2);
                    if (q_stages_2 == 1) {
                        unsigned int kv_stage = 0;
                        unsigned int kv_phase = 0;
                        if (((q_len_2 < q_token_base_2 + 256) ? 1 : 0) != 0) {
                            mbarrier_wait(q_tail_full_addr, _phase_q_tail_full_0);
                            _phase_q_tail_full_0 ^= 1;
                        } else {
                            mbarrier_wait(q_full_addr, _phase_q_full_0);
                            _phase_q_full_0 ^= 1;
                        }
                        mbarrier_wait(prefill_kv_full_addr + (kv_stage) * 8, kv_phase);
                        int _mma_a_lo_8 = make_warp_uniform(((prefill_q0_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_8 = make_warp_uniform((((prefill_kv_addr) >> 4) & 0x3FFF) + (kv_stage) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"(tmem_prefill_scores_0), "r"(0));
                        elect_commit(s_full_addr);
                        elect_commit(prefill_kv_empty_addr + (kv_stage) * 8);
                        kv_stage += 1;
                        if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
                        unsigned int first_pv = 1;
                        #pragma unroll 1
                        for (int __2 = 0; __2 < num_n_blocks_2 - 1; __2++) {
                            unsigned int v_stage = kv_stage;
                            unsigned int v_phase = kv_phase;
                            kv_stage += 1;
                            if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
                            mbarrier_wait(prefill_kv_full_addr + (v_stage) * 8, v_phase);
                            int first_pv_flag = first_pv;
                            mbarrier_wait(p_full_tail_addr, _phase_p_full_tail_0);
                            _phase_p_full_tail_0 ^= 1;
                            int _mma_b_lo_9 = make_warp_uniform(((((prefill_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_prefill_output_0), "r"(_mma_b_lo_9), "r"(tmem_prefill_scores_0 + 64), "r"(((first_pv_flag) ? 0 : 1)));
                            mbarrier_wait(p_full_addr, _phase_p_full_0);
                            _phase_p_full_0 ^= 1;
                            int _mma_b_lo_10 = make_warp_uniform(((((prefill_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_prefill_output_0), "r"(_mma_b_lo_10), "r"(tmem_prefill_scores_0 + 64), "r"(1));
                            elect_commit(prefill_kv_empty_addr + (v_stage) * 8);
                            unsigned int k_stage = kv_stage;
                            unsigned int k_phase = kv_phase;
                            kv_stage += 1;
                            if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
                            mbarrier_wait(prefill_kv_full_addr + (k_stage) * 8, k_phase);
                            int _mma_a_lo_11 = make_warp_uniform(((prefill_q0_addr) >> 4) & 0x3FFF);
                            int _mma_b_lo_11 = make_warp_uniform((((prefill_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"(tmem_prefill_scores_0), "r"(0));
                            elect_commit(s_full_addr);
                            elect_commit(prefill_kv_empty_addr + (k_stage) * 8);
                            first_pv = 0;
                        }
                        mbarrier_wait(prefill_kv_full_addr + (kv_stage) * 8, kv_phase);
                        int first_pv_flag_1 = first_pv;
                        mbarrier_wait(p_full_tail_addr, _phase_p_full_tail_0);
                        _phase_p_full_tail_0 ^= 1;
                        int _mma_b_lo_12 = make_warp_uniform(((((prefill_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_prefill_output_0), "r"(_mma_b_lo_12), "r"(tmem_prefill_scores_0 + 64), "r"(((first_pv_flag_1) ? 0 : 1)));
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        _phase_p_full_0 ^= 1;
                        int _mma_b_lo_13 = make_warp_uniform(((((prefill_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_prefill_output_0), "r"(_mma_b_lo_13), "r"(tmem_prefill_scores_0 + 64), "r"(1));
                        elect_commit(prefill_kv_empty_addr + (kv_stage) * 8);
                        elect_commit(o_full_addr);
                    } else {
                        unsigned int kv_stage_1 = 0;
                        unsigned int kv_phase_1 = 0;
                        if (((q_len_2 < q_token_base_2 + 256) ? 1 : 0) != 0) {
                            mbarrier_wait(q_tail_full_addr, _phase_q_tail_full_0);
                            _phase_q_tail_full_0 ^= 1;
                            mbarrier_wait(q_tail_full_addr + 8, _phase_q_tail_full_1);
                            _phase_q_tail_full_1 ^= 1;
                        } else {
                            mbarrier_wait(q_full_addr, _phase_q_full_0);
                            _phase_q_full_0 ^= 1;
                            mbarrier_wait(q_full_addr + 8, _phase_q_full_1);
                            _phase_q_full_1 ^= 1;
                        }
                        mbarrier_wait(prefill_kv_full_addr + (kv_stage_1) * 8, kv_phase_1);
                        int _mma_a_lo_14 = make_warp_uniform(((prefill_q0_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_14 = make_warp_uniform((((prefill_kv_addr) >> 4) & 0x3FFF) + (kv_stage_1) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_14), "r"(_mma_b_lo_14), "r"(tmem_prefill_scores_0), "r"(0));
                        elect_commit(s_full_addr);
                        int _mma_a_lo_15 = make_warp_uniform(((prefill_q1_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_15 = make_warp_uniform((((prefill_kv_addr) >> 4) & 0x3FFF) + (kv_stage_1) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_15), "r"(_mma_b_lo_15), "r"(tmem_prefill_scores_1), "r"(0));
                        elect_commit(s_full_addr + 8);
                        elect_commit(prefill_kv_empty_addr + (kv_stage_1) * 8);
                        kv_stage_1 += 1;
                        if (kv_stage_1 == 2) { kv_stage_1 = 0; kv_phase_1 ^= 1; }
                        unsigned int first_pv_1 = 1;
                        #pragma unroll 1
                        for (int __3 = 0; __3 < num_n_blocks_2 - 1; __3++) {
                            unsigned int v_stage_1 = kv_stage_1;
                            unsigned int v_phase_1 = kv_phase_1;
                            kv_stage_1 += 1;
                            if (kv_stage_1 == 2) { kv_stage_1 = 0; kv_phase_1 ^= 1; }
                            mbarrier_wait(prefill_kv_full_addr + (v_stage_1) * 8, v_phase_1);
                            int first_pv_flag_2 = first_pv_1;
                            mbarrier_wait(p_full_tail_addr, _phase_p_full_tail_0);
                            _phase_p_full_tail_0 ^= 1;
                            int _mma_b_lo_16 = make_warp_uniform(((((prefill_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_1) * 2048);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_prefill_output_0), "r"(_mma_b_lo_16), "r"(tmem_prefill_scores_0 + 64), "r"(((first_pv_flag_2) ? 0 : 1)));
                            mbarrier_wait(p_full_addr, _phase_p_full_0);
                            _phase_p_full_0 ^= 1;
                            int _mma_b_lo_17 = make_warp_uniform(((((prefill_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_1) * 2048);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_prefill_output_0), "r"(_mma_b_lo_17), "r"(tmem_prefill_scores_0 + 64), "r"(1));
                            unsigned int k_stage_1 = kv_stage_1;
                            unsigned int k_phase_1 = kv_phase_1;
                            kv_stage_1 += 1;
                            if (kv_stage_1 == 2) { kv_stage_1 = 0; kv_phase_1 ^= 1; }
                            mbarrier_wait(prefill_kv_full_addr + (k_stage_1) * 8, k_phase_1);
                            int _mma_a_lo_18 = make_warp_uniform(((prefill_q0_addr) >> 4) & 0x3FFF);
                            int _mma_b_lo_18 = make_warp_uniform((((prefill_kv_addr) >> 4) & 0x3FFF) + (k_stage_1) * 2048);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_18), "r"(_mma_b_lo_18), "r"(tmem_prefill_scores_0), "r"(0));
                            elect_commit(s_full_addr);
                            mbarrier_wait(p_full_tail_addr + 8, _phase_p_full_tail_1);
                            _phase_p_full_tail_1 ^= 1;
                            int _mma_b_lo_19 = make_warp_uniform(((((prefill_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_1) * 2048);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_prefill_output_1), "r"(_mma_b_lo_19), "r"(tmem_prefill_scores_1 + 64), "r"(((first_pv_flag_2) ? 0 : 1)));
                            mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                            _phase_p_full_1 ^= 1;
                            int _mma_b_lo_20 = make_warp_uniform(((((prefill_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_1) * 2048);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_prefill_output_1), "r"(_mma_b_lo_20), "r"(tmem_prefill_scores_1 + 64), "r"(1));
                            elect_commit(prefill_kv_empty_addr + (v_stage_1) * 8);
                            int _mma_a_lo_21 = make_warp_uniform(((prefill_q1_addr) >> 4) & 0x3FFF);
                            int _mma_b_lo_21 = make_warp_uniform((((prefill_kv_addr) >> 4) & 0x3FFF) + (k_stage_1) * 2048);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_21), "r"(_mma_b_lo_21), "r"(tmem_prefill_scores_1), "r"(0));
                            elect_commit(s_full_addr + 8);
                            elect_commit(prefill_kv_empty_addr + (k_stage_1) * 8);
                            first_pv_1 = 0;
                        }
                        mbarrier_wait(prefill_kv_full_addr + (kv_stage_1) * 8, kv_phase_1);
                        int first_pv_flag_3 = first_pv_1;
                        mbarrier_wait(p_full_tail_addr, _phase_p_full_tail_0);
                        _phase_p_full_tail_0 ^= 1;
                        int _mma_b_lo_22 = make_warp_uniform(((((prefill_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_1) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_prefill_output_0), "r"(_mma_b_lo_22), "r"(tmem_prefill_scores_0 + 64), "r"(((first_pv_flag_3) ? 0 : 1)));
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        _phase_p_full_0 ^= 1;
                        int _mma_b_lo_23 = make_warp_uniform(((((prefill_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_1) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_prefill_output_0), "r"(_mma_b_lo_23), "r"(tmem_prefill_scores_0 + 64), "r"(1));
                        mbarrier_wait(p_full_tail_addr + 8, _phase_p_full_tail_1);
                        _phase_p_full_tail_1 ^= 1;
                        int _mma_b_lo_24 = make_warp_uniform(((((prefill_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_1) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_prefill_output_1), "r"(_mma_b_lo_24), "r"(tmem_prefill_scores_1 + 64), "r"(((first_pv_flag_3) ? 0 : 1)));
                        mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                        _phase_p_full_1 ^= 1;
                        int _mma_b_lo_25 = make_warp_uniform(((((prefill_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_1) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_prefill_output_1), "r"(_mma_b_lo_25), "r"(tmem_prefill_scores_1 + 64), "r"(1));
                        elect_commit(prefill_kv_empty_addr + (kv_stage_1) * 8);
                        elect_commit2(o_full_addr, o_full_addr + 8);
                    }
                }
                if (direct_decode_2 == 0) {
                    if (elect_sync()) {
                        if (work_slot_2 == 0) {
                            mbarrier_arrive(work_empty_0_addr);
                        } else {
                            mbarrier_arrive(work_empty_1_addr);
                        }
                    }
                }
            }
        }
    // ---- Role: scheduler ----
    } else if (warp == 13) {
        { // scheduler_main
            int lane_0 = lane;
            int group_size_2 = num_q_heads / num_kv_heads;
            int target_per_request = (num_bids + num_requests - 1) / num_requests;
            int unsplit_lower_bound = num_requests * num_kv_heads;
            int probe_unsplit = 0;
            if (unsplit_lower_bound <= num_bids) {
                if (unsplit_lower_bound * 3 >= num_bids) {
                    probe_unsplit = 1;
                }
            }
            int carry = 0;
            if (attention_mode == 1) {
                carry = num_requests * num_kv_heads;
            } else {
                int prefix_chunks = (num_requests + 31) / 32;
                #pragma unroll
                for (int prefix_pass = 0; prefix_pass < 2; prefix_pass++) {
                    int run_prefix = 1;
                    if (prefix_pass != 0) {
                        if (probe_unsplit == 0) {
                            run_prefix = 0;
                        } else if (carry <= num_bids) {
                            run_prefix = 0;
                        }
                    }
                    if (run_prefix != 0) {
                        carry = 0;
                        #pragma unroll 1
                        for (int chunk = 0; chunk < prefix_chunks; chunk++) {
                            int prefix_request = chunk * 32 + lane_0;
                            int request_tasks = 0;
                            if (prefix_request < num_requests) {
                                int prefix_q_len = qo_indptr[prefix_request + 1] - qo_indptr[prefix_request];
                                {
                                    prefix_q_len = 1;
                                }
                                int prefix_packed_q = prefix_q_len * group_size_2;
                                int prefix_kv_len = kv_len_arr[prefix_request];
                                int prefix_kv_tiles = (prefix_kv_len + 128 - 1) / 128;
                                {
                                    prefix_kv_tiles = max_kv_tiles;
                                }
                                int prefix_is_decode = 0;
                                if (prefix_q_len == 1) {
                                    if (prefix_packed_q <= 16) {
                                        prefix_is_decode = 1;
                                    }
                                }
                                if (prefix_is_decode != 0) {
                                    int prefix_q_tiles = (prefix_packed_q + 16 - 1) / 16;
                                    int prefix_base_tasks = num_kv_heads * prefix_q_tiles;
                                    int prefix_splits = 1;
                                    int use_split_prefix = 0;
                                    if (probe_unsplit == 0) {
                                        use_split_prefix = 1;
                                    }
                                    if (prefix_pass != 0) {
                                        use_split_prefix = 1;
                                    }
                                    {
                                        if (msa_split_policy == 1) {
                                            use_split_prefix = 1;
                                        }
                                    }
                                    if (use_split_prefix != 0) {
                                        prefix_splits = (target_per_request + prefix_base_tasks - 1) / prefix_base_tasks;
                                        int prefix_kv_pairs = prefix_kv_tiles / 2;
                                        if (prefix_splits < 1) {
                                            prefix_splits = 1;
                                        }
                                        if (prefix_splits > prefix_kv_pairs) {
                                            prefix_splits = prefix_kv_pairs;
                                        }
                                        {
                                            if (msa_split_policy == 1) {
                                                prefix_splits = prefix_kv_pairs;
                                            }
                                        }
                                    }
                                    request_tasks = prefix_base_tasks * prefix_splits;
                                } else {
                                    int prefix_token_tiles = (prefix_q_len + 256 - 1) / 256;
                                    int prefix_q_tiles_1 = group_size_2 * prefix_token_tiles;
                                    if (prefix_packed_q <= 128) {
                                        prefix_q_tiles_1 = 1;
                                    }
                                    int prefix_base_tasks_1 = num_kv_heads * prefix_q_tiles_1;
                                    int prefix_splits_1 = 1;
                                    int prefix_can_split = 0;
                                    if (prefix_q_len <= 8) {
                                        if (prefix_packed_q <= 128) {
                                            prefix_can_split = 1;
                                        }
                                    }
                                    if (prefix_can_split != 0) {
                                        int use_split_prefix_1 = 0;
                                        if (probe_unsplit == 0) {
                                            use_split_prefix_1 = 1;
                                        }
                                        if (prefix_pass != 0) {
                                            use_split_prefix_1 = 1;
                                        }
                                        if (use_split_prefix_1 != 0) {
                                            prefix_splits_1 = (target_per_request + prefix_base_tasks_1 - 1) / prefix_base_tasks_1;
                                            int prefix_split_cap = 8;
                                            if (prefix_packed_q > 64) {
                                                prefix_split_cap = 4;
                                            }
                                            if (prefix_splits_1 < 1) {
                                                prefix_splits_1 = 1;
                                            }
                                            if (prefix_splits_1 > prefix_kv_tiles) {
                                                prefix_splits_1 = prefix_kv_tiles;
                                            }
                                            if (prefix_splits_1 > prefix_split_cap) {
                                                prefix_splits_1 = prefix_split_cap;
                                            }
                                        }
                                    }
                                    request_tasks = prefix_base_tasks_1 * prefix_splits_1;
                                }
                            }
                            int inclusive = request_tasks;
                            int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
                            int peer = _shfl_up_0;
                            inclusive = ((lane_0 >= 1) ? inclusive + peer : inclusive);
                            int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
                            int peer_0 = _shfl_up_1;
                            inclusive = ((lane_0 >= 2) ? inclusive + peer_0 : inclusive);
                            int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
                            int peer_1 = _shfl_up_2;
                            inclusive = ((lane_0 >= 4) ? inclusive + peer_1 : inclusive);
                            int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
                            int peer_2 = _shfl_up_3;
                            inclusive = ((lane_0 >= 8) ? inclusive + peer_2 : inclusive);
                            int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
                            int peer_3 = _shfl_up_4;
                            inclusive = ((lane_0 >= 16) ? inclusive + peer_3 : inclusive);
                            if (prefix_request < num_requests) {
                                task_offsets[prefix_request] = carry + inclusive - request_tasks;
                            }
                            int _shfl_5 = __shfl_sync(0xFFFFFFFF, inclusive, 31);
                            carry += _shfl_5;
                        }
                    }
                }
                if (lane_0 == 0) {
                    task_offsets[num_requests] = carry;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
            }
            int device_total = ((carry <= max_task_claims) ? carry : max_task_claims);
            if (blockIdx.x == 0 && lane_0 == 0 && carry > max_task_claims) {
                *((int*)(status + 1)) = 1;
            }
            unsigned int _phase_work_empty_0_0 = 1;
            unsigned int _phase_work_empty_1_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int claim_iter = 0; claim_iter < max_task_claims + 1; claim_iter++) {
                    if (attention_mode == 1 && device_total <= gridDim.x * gridDim.y) {
                        break;
                    }
                    int work_slot_3 = claim_iter % 2;
                    int* work_desc_3 = work_desc_slots + (work_slot_3 * 13);
                    if (work_slot_3 == 0) {
                        mbarrier_wait(work_empty_0_addr, _phase_work_empty_0_0);
                        _phase_work_empty_0_0 ^= 1;
                    } else {
                        mbarrier_wait(work_empty_1_addr, _phase_work_empty_1_0);
                        _phase_work_empty_1_0 ^= 1;
                    }
                    int direct_decode_3 = ((attention_mode == 1 && device_total <= gridDim.x * gridDim.y) ? 1 : 0);
                    int ticket_3 = 0;
                    if (direct_decode_3 != 0) {
                        ticket_3 = ((claim_iter == 0) ? blockIdx.x * num_kv_heads + blockIdx.y : device_total);
                    } else {
                        int _atomic_old_3 = atomicAdd(&status[0], 1);
                        ticket_3 = _atomic_old_3;
                    }
                    work_desc_3[0] = ((ticket_3 < device_total) ? ticket_3 : -1);
                    if (ticket_3 < device_total) {
                        int kind_3 = 1;
                        int request_2 = 0;
                        int kv_head_1 = 0;
                        int q_tile_3 = 0;
                        int split_1 = 0;
                        int splits_1 = 1;
                        int kv_tile_begin_3 = 0;
                        int kv_tile_end_3 = 0;
                        if (attention_mode == 1) {
                            request_2 = ticket_3 / num_kv_heads;
                            kv_head_1 = ticket_3 % num_kv_heads;
                            int fast_kv_len = kv_len_arr[request_2];
                            kv_tile_end_3 = (fast_kv_len + 128 - 1) / 128;
                            {
                                kv_tile_end_3 = max_kv_tiles;
                            }
                        } else {
                            int lo = 0;
                            int hi = num_requests;
                            #pragma unroll
                            for (int __4 = 0; __4 < 10; __4++) {
                                int mid = (lo + hi) / 2;
                                int safe_mid = ((mid < num_requests) ? mid : num_requests - 1);
                                int advance = ((ticket_3 >= task_offsets[safe_mid + 1] && mid < num_requests) ? 1 : 0);
                                lo = ((advance != 0) ? mid + 1 : lo);
                                hi = ((advance == 0) ? mid : hi);
                            }
                            request_2 = lo;
                            int local = ticket_3 - task_offsets[request_2];
                            int q_len_3 = qo_indptr[request_2 + 1] - qo_indptr[request_2];
                            {
                                q_len_3 = 1;
                            }
                            int packed_q = q_len_3 * group_size_2;
                            int kv_len = kv_len_arr[request_2];
                            int kv_tiles = (kv_len + 128 - 1) / 128;
                            {
                                kv_tiles = max_kv_tiles;
                            }
                            int kv_pairs = kv_tiles / 2;
                            kind_3 = 0;
                            int prefill_token_tiles = (q_len_3 + 256 - 1) / 256;
                            int q_tiles = group_size_2 * prefill_token_tiles;
                            if (packed_q <= 128) {
                                q_tiles = 1;
                            }
                            int prefill_base_tasks = num_kv_heads * q_tiles;
                            if (q_len_3 <= 8) {
                                if (packed_q <= 128) {
                                    if (probe_unsplit == 0 || device_total > num_bids) {
                                        splits_1 = (target_per_request + prefill_base_tasks - 1) / prefill_base_tasks;
                                        int speculative_split_cap = 8;
                                        if (packed_q > 64) {
                                            speculative_split_cap = 4;
                                        }
                                        if (splits_1 < 1) {
                                            splits_1 = 1;
                                        }
                                        if (splits_1 > kv_tiles) {
                                            splits_1 = kv_tiles;
                                        }
                                        if (splits_1 > speculative_split_cap) {
                                            splits_1 = speculative_split_cap;
                                        }
                                    }
                                }
                            }
                            if (q_len_3 == 1) {
                                if (packed_q <= 16) {
                                    kind_3 = 1;
                                    q_tiles = (packed_q + 16 - 1) / 16;
                                    int base_tasks = num_kv_heads * q_tiles;
                                    splits_1 = 1;
                                    if (probe_unsplit == 0 || device_total > num_bids) {
                                        splits_1 = (target_per_request + base_tasks - 1) / base_tasks;
                                        if (splits_1 < 1) {
                                            splits_1 = 1;
                                        }
                                        if (splits_1 > kv_pairs) {
                                            splits_1 = kv_pairs;
                                        }
                                    }
                                    {
                                        if (msa_split_policy == 1) {
                                            splits_1 = kv_pairs;
                                        }
                                    }
                                }
                            }
                            split_1 = local % splits_1;
                            int head_q = local / splits_1;
                            q_tile_3 = head_q % q_tiles;
                            kv_head_1 = head_q / q_tiles;
                            if (kind_3 == 1) {
                                kv_tile_begin_3 = 2 * (kv_pairs * split_1 / splits_1);
                                kv_tile_end_3 = 2 * (kv_pairs * (split_1 + 1) / splits_1);
                            } else {
                                kv_tile_begin_3 = kv_tiles * split_1 / splits_1;
                                kv_tile_end_3 = kv_tiles * (split_1 + 1) / splits_1;
                            }
                        }
                        work_desc_3[0] = ticket_3;
                        work_desc_3[1] = kind_3;
                        work_desc_3[2] = request_2;
                        work_desc_3[3] = kv_head_1;
                        work_desc_3[4] = q_tile_3;
                        work_desc_3[5] = split_1;
                        work_desc_3[6] = kv_tile_begin_3;
                        work_desc_3[7] = kv_tile_end_3;
                        {
                            work_desc_3[8] = request_2;
                            work_desc_3[9] = request_2 + 1;
                        }
                        {
                            work_desc_3[10] = 0;
                            work_desc_3[11] = 0;
                        }
                        work_desc_3[12] = splits_1;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (work_slot_3 == 0) {
                        mbarrier_arrive(work_full_0_addr);
                    } else {
                        mbarrier_arrive(work_full_1_addr);
                    }
                    if (ticket_3 >= device_total) {
                        break;
                    }
                }
            }
        }
    // ---- Role: producer ----
    } else if (warp == 14) {
        { // producer_main
            int direct_decode_4 = ((attention_mode == 1 && gridDim.x * gridDim.y >= num_requests * num_kv_heads) ? 1 : 0);
            unsigned int _phase_work_full_0_0_3 = 0;
            unsigned int _phase_work_full_1_0_3 = 0;
            unsigned int _phase_prefill_partial_ready_0 = 0;
            #pragma unroll 1
            for (int task_iter_3 = 0; task_iter_3 < max_task_claims + 1; task_iter_3++) {
                int work_slot_4 = task_iter_3 % 2;
                int* work_desc_4 = work_desc_slots + (work_slot_4 * 13);
                if (direct_decode_4 == 0) {
                    if (work_slot_4 == 0) {
                        mbarrier_wait(work_full_0_addr, _phase_work_full_0_0_3);
                        _phase_work_full_0_0_3 ^= 1;
                    } else {
                        mbarrier_wait(work_full_1_addr, _phase_work_full_1_0_3);
                        _phase_work_full_1_0_3 ^= 1;
                    }
                    asm volatile("barrier.sync 8, 480;" ::: "memory");
                }
                int ticket_4 = -1;
                if (direct_decode_4 != 0) {
                    if (task_iter_3 == 0) {
                        ticket_4 = blockIdx.x * num_kv_heads + blockIdx.y;
                    }
                } else {
                    ticket_4 = work_desc_4[0];
                }
                if (ticket_4 < 0) {
                    if (direct_decode_4 == 0) {
                        if (elect_sync()) {
                            if (work_slot_4 == 0) {
                                mbarrier_arrive(work_empty_0_addr);
                            } else {
                                mbarrier_arrive(work_empty_1_addr);
                            }
                        }
                    }
                    break;
                }
                int kind_4 = ((direct_decode_4 != 0) ? 1 : -1);
                if (direct_decode_4 == 0 && attention_mode != 0) {
                    kind_4 = work_desc_4[1];
                }
                int direct_request_2 = 0;
                int kv_head_2 = 0;
                if (direct_decode_4 != 0) {
                    direct_request_2 = blockIdx.x;
                    kv_head_2 = blockIdx.y;
                } else {
                    direct_request_2 = ticket_4 / num_kv_heads;
                    kv_head_2 = ticket_4 % num_kv_heads;
                    {
                        direct_request_2 = 0;
                    }
                }
                int q_tile_4 = 0;
                int kv_tile_begin_4 = 0;
                int direct_batch_3 = direct_request_2;
                {
                    direct_batch_3 = direct_request_2 / record_tasks;
                }
                int direct_kv_len_3 = kv_len_arr[direct_batch_3];
                int kv_tile_end_4 = (direct_kv_len_3 + 128 - 1) / 128;
                int qo_begin_3 = qo_indptr[direct_request_2];
                int page_begin = kv_indptr[direct_request_2];
                {
                    kv_tile_end_4 = max_kv_tiles;
                    qo_begin_3 = direct_request_2;
                    page_begin = kv_indptr[direct_batch_3];
                }
                if (direct_decode_4 == 0) {
                    kv_head_2 = work_desc_4[3];
                    q_tile_4 = work_desc_4[4];
                    kv_tile_begin_4 = work_desc_4[6];
                    kv_tile_end_4 = work_desc_4[7];
                    qo_begin_3 = work_desc_4[8];
                    page_begin = work_desc_4[10];
                    {
                        direct_request_2 = work_desc_4[2];
                    }
                }
                int num_n_blocks_3 = kv_tile_end_4 - kv_tile_begin_4;
                int group_size_3 = num_q_heads / num_kv_heads;
                if (kind_4 == 1) {
                    if (elect_sync()) {
                        int q_row_2 = qo_begin_3 * num_q_heads + kv_head_2 * group_size_3 + q_tile_4 * 16;
                        mbarrier_arrive_expect_tx(q_full_addr, 4096);
                        tma_3d_gmem2smem(smem_qt_addr, Q, 0, q_row_2, 0, q_full_addr);
                    }
                    int native_num_n_blocks = num_n_blocks_3;
                    if (elect_sync()) {
                        int native_kv_stage = 0;
                        int native_kv_phase = 1;
                        int native_prefill = ((native_num_n_blocks < 4) ? native_num_n_blocks : 4);
                        #pragma unroll
                        for (int native_ni = 0; native_ni < 4; native_ni++) {
                            if (native_prefill <= native_ni) {
                                break;
                            }
                            int native_n_block = kv_tile_end_4 - 1 - native_ni;
                            int native_pg0 = 0;
                            int native_pg1 = 0;
                            int msa_token_base = 0;
                            int msa_page_head = 0;
                            int msa_valid_cols = 128;
                            {
                                int batch = direct_request_2 / record_tasks;
                                int query_in_batch = direct_request_2 - batch * record_tasks;
                                int selected_block = task_kind[(kv_head_2 * num_requests + direct_request_2) * max_kv_tiles + native_n_block];
                                int kv_len_1 = task_kv_head[batch];
                                {
                                    kv_len_1 = kv_indptr[batch + 1] - kv_indptr[batch];
                                }
                                int valid_cols_1 = 0;
                                if (selected_block >= 0) {
                                    int block_start = selected_block * 128;
                                    valid_cols_1 = kv_len_1 - block_start;
                                    if (valid_cols_1 > 128) {
                                        valid_cols_1 = 128;
                                    }
                                    if (valid_cols_1 < 0) {
                                        valid_cols_1 = 0;
                                    }
                                    if (is_causal != 0) {
                                        int query_position = kv_len_1 - record_tasks + query_in_batch;
                                        if (derive_q_offset == 0) {
                                            query_position = task_request[batch] + query_in_batch;
                                        }
                                        int causal_cols = query_position - block_start + 1;
                                        if (valid_cols_1 > causal_cols) {
                                            valid_cols_1 = causal_cols;
                                        }
                                        if (valid_cols_1 < 0) {
                                            valid_cols_1 = 0;
                                        }
                                    }
                                }
                                int token_base = 0;
                                int page_head = 0;
                                {
                                    int safe_block = ((selected_block >= 0) ? selected_block : 0);
                                    token_base = kv_indptr[batch] + safe_block * 128;
                                    page_head = kv_head_2;
                                }
                                msa_token_base = token_base;
                                msa_page_head = page_head;
                                msa_valid_cols = valid_cols_1;
                                smem_page_indices[native_ni] = msa_valid_cols;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                            mbarrier_wait(kv_empty_addr + (native_kv_stage) * 8, native_kv_phase);
                            {
                                mbarrier_arrive_expect_tx(kv_full_addr + (native_kv_stage) * 8, 32768);
                            }
                            int native_dst = smem_kv_addr + (unsigned int)(native_kv_stage * 32768);
                            {
                                {
                                    int token0 = msa_token_base;
                                    int token1 = msa_token_base + 64;
                                    tma_4d_gmem2smem(native_dst, K, 0, token0, 0, msa_page_head, kv_full_addr + (native_kv_stage) * 8);
                                    tma_4d_gmem2smem(native_dst + 8192, K, 0, token1, 0, msa_page_head, kv_full_addr + (native_kv_stage) * 8);
                                    tma_4d_gmem2smem(native_dst + 16384, K, 0, token0, 1, msa_page_head, kv_full_addr + (native_kv_stage) * 8);
                                    tma_4d_gmem2smem(native_dst + 24576, K, 0, token1, 1, msa_page_head, kv_full_addr + (native_kv_stage) * 8);
                                }
                            }
                            native_kv_stage += 1;
                            if (native_kv_stage == 4) { native_kv_stage = 0; native_kv_phase ^= 1; }
                        }
                        #pragma unroll 1
                        for (int native_ni_1 = 0; native_ni_1 < max_kv_tiles; native_ni_1++) {
                            if (native_num_n_blocks <= native_ni_1) {
                                break;
                            }
                            int native_stage = native_ni_1 % 4;
                            int native_n_block_1 = kv_tile_end_4 - 1 - native_ni_1;
                            int native_pg0_1 = 0;
                            int native_pg1_1 = 0;
                            int msa_token_base_1 = 0;
                            int msa_page_head_1 = 0;
                            int msa_valid_cols_1 = 128;
                            {
                                int batch_1 = direct_request_2 / record_tasks;
                                int query_in_batch_1 = direct_request_2 - batch_1 * record_tasks;
                                int selected_block_1 = task_kind[(kv_head_2 * num_requests + direct_request_2) * max_kv_tiles + native_n_block_1];
                                int kv_len_2 = task_kv_head[batch_1];
                                {
                                    kv_len_2 = kv_indptr[batch_1 + 1] - kv_indptr[batch_1];
                                }
                                int valid_cols_2 = 0;
                                if (selected_block_1 >= 0) {
                                    int block_start_1 = selected_block_1 * 128;
                                    valid_cols_2 = kv_len_2 - block_start_1;
                                    if (valid_cols_2 > 128) {
                                        valid_cols_2 = 128;
                                    }
                                    if (valid_cols_2 < 0) {
                                        valid_cols_2 = 0;
                                    }
                                    if (is_causal != 0) {
                                        int query_position_1 = kv_len_2 - record_tasks + query_in_batch_1;
                                        if (derive_q_offset == 0) {
                                            query_position_1 = task_request[batch_1] + query_in_batch_1;
                                        }
                                        int causal_cols_1 = query_position_1 - block_start_1 + 1;
                                        if (valid_cols_2 > causal_cols_1) {
                                            valid_cols_2 = causal_cols_1;
                                        }
                                        if (valid_cols_2 < 0) {
                                            valid_cols_2 = 0;
                                        }
                                    }
                                }
                                int token_base_1 = 0;
                                int page_head_1 = 0;
                                {
                                    int safe_block_1 = ((selected_block_1 >= 0) ? selected_block_1 : 0);
                                    token_base_1 = kv_indptr[batch_1] + safe_block_1 * 128;
                                    page_head_1 = kv_head_2;
                                }
                                msa_token_base_1 = token_base_1;
                                msa_page_head_1 = page_head_1;
                                msa_valid_cols_1 = valid_cols_2;
                            }
                            mbarrier_wait(kv_empty_addr + (native_stage) * 8, 0);
                            {
                                mbarrier_arrive_expect_tx(kv_full_addr + (native_stage) * 8, 32768);
                            }
                            int native_dst_1 = smem_kv_addr + (unsigned int)(native_stage * 32768);
                            {
                                {
                                    int token0_1 = msa_token_base_1;
                                    int token1_1 = msa_token_base_1 + 64;
                                    tma_4d_gmem2smem(native_dst_1, V, 0, token0_1, 0, msa_page_head_1, kv_full_addr + (native_stage) * 8);
                                    tma_4d_gmem2smem(native_dst_1 + 8192, V, 0, token1_1, 0, msa_page_head_1, kv_full_addr + (native_stage) * 8);
                                    tma_4d_gmem2smem(native_dst_1 + 16384, V, 0, token0_1, 1, msa_page_head_1, kv_full_addr + (native_stage) * 8);
                                    tma_4d_gmem2smem(native_dst_1 + 24576, V, 0, token1_1, 1, msa_page_head_1, kv_full_addr + (native_stage) * 8);
                                }
                            }
                            int native_next_ni = native_ni_1 + 4;
                            if (native_next_ni < native_num_n_blocks) {
                                int native_next_n = kv_tile_end_4 - 1 - native_next_ni;
                                int native_npg0 = 0;
                                int native_npg1 = 0;
                                int msa_next_token_base = 0;
                                int msa_next_page_head = 0;
                                int msa_next_valid_cols = 128;
                                {
                                    int batch_2 = direct_request_2 / record_tasks;
                                    int query_in_batch_2 = direct_request_2 - batch_2 * record_tasks;
                                    int selected_block_2 = task_kind[(kv_head_2 * num_requests + direct_request_2) * max_kv_tiles + native_next_n];
                                    int kv_len_3 = task_kv_head[batch_2];
                                    {
                                        kv_len_3 = kv_indptr[batch_2 + 1] - kv_indptr[batch_2];
                                    }
                                    int valid_cols_3 = 0;
                                    if (selected_block_2 >= 0) {
                                        int block_start_2 = selected_block_2 * 128;
                                        valid_cols_3 = kv_len_3 - block_start_2;
                                        if (valid_cols_3 > 128) {
                                            valid_cols_3 = 128;
                                        }
                                        if (valid_cols_3 < 0) {
                                            valid_cols_3 = 0;
                                        }
                                        if (is_causal != 0) {
                                            int query_position_2 = kv_len_3 - record_tasks + query_in_batch_2;
                                            if (derive_q_offset == 0) {
                                                query_position_2 = task_request[batch_2] + query_in_batch_2;
                                            }
                                            int causal_cols_2 = query_position_2 - block_start_2 + 1;
                                            if (valid_cols_3 > causal_cols_2) {
                                                valid_cols_3 = causal_cols_2;
                                            }
                                            if (valid_cols_3 < 0) {
                                                valid_cols_3 = 0;
                                            }
                                        }
                                    }
                                    int token_base_2 = 0;
                                    int page_head_2 = 0;
                                    {
                                        int safe_block_2 = ((selected_block_2 >= 0) ? selected_block_2 : 0);
                                        token_base_2 = kv_indptr[batch_2] + safe_block_2 * 128;
                                        page_head_2 = kv_head_2;
                                    }
                                    msa_next_token_base = token_base_2;
                                    msa_next_page_head = page_head_2;
                                    msa_next_valid_cols = valid_cols_3;
                                    smem_page_indices[native_next_ni] = msa_next_valid_cols;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                }
                                mbarrier_wait(kv_empty_addr + (native_stage) * 8, 1);
                                {
                                    mbarrier_arrive_expect_tx(kv_full_addr + (native_stage) * 8, 32768);
                                }
                                int native_kdst = smem_kv_addr + (unsigned int)(native_stage * 32768);
                                {
                                    {
                                        int token0_2 = msa_next_token_base;
                                        int token1_2 = msa_next_token_base + 64;
                                        tma_4d_gmem2smem(native_kdst, K, 0, token0_2, 0, msa_next_page_head, kv_full_addr + (native_stage) * 8);
                                        tma_4d_gmem2smem(native_kdst + 8192, K, 0, token1_2, 0, msa_next_page_head, kv_full_addr + (native_stage) * 8);
                                        tma_4d_gmem2smem(native_kdst + 16384, K, 0, token0_2, 1, msa_next_page_head, kv_full_addr + (native_stage) * 8);
                                        tma_4d_gmem2smem(native_kdst + 24576, K, 0, token1_2, 1, msa_next_page_head, kv_full_addr + (native_stage) * 8);
                                    }
                                }
                            }
                        }
                    }
                    num_n_blocks_3 = 0;
                    int kv_stage_2 = 0;
                    int kv_phase_2 = 1;
                    int prefill = ((num_n_blocks_3 < 4) ? num_n_blocks_3 : 4);
                    #pragma unroll 1
                    for (int ni = 0; ni < num_n_blocks_3; ni++) {
                        if (prefill <= ni) {
                            break;
                        }
                        int n_block_1 = kv_tile_end_4 - 1 - ni;
                        #pragma unroll
                        for (int pp = 0; pp < 4; pp++) {
                            int pt_off = pp * 32 + lane;
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                                :: "r"(smem_page_indices_addr + (unsigned int)((kv_stage_2 * 128 + pt_off) * 4)), "l"(kv_indices + (page_begin + n_block_1 * 128 + pt_off)));
                        }
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 0;");
                        asm volatile("barrier.sync 10, 32;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_wait(kv_empty_addr + (kv_stage_2) * 8, kv_phase_2);
                            mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_2) * 8, 32768);
                        }
                        int dst = smem_kv_addr + (unsigned int)(kv_stage_2 * 32768);
                        #pragma unroll
                        for (int g = 0; g < 32; g++) {
                            int off_g = g * 4;
                            int k_stage_index = kv_stage_2 * 128 + off_g;
                            int r0 = smem_page_indices[k_stage_index] * (2 * num_kv_heads) + kv_head_2;
                            int r1 = smem_page_indices[k_stage_index + 1] * (2 * num_kv_heads) + kv_head_2;
                            int r2 = smem_page_indices[k_stage_index + 2] * (2 * num_kv_heads) + kv_head_2;
                            int r3 = smem_page_indices[k_stage_index + 3] * (2 * num_kv_heads) + kv_head_2;
                            if (elect_sync()) {
                                tma_gather4_gmem2smem(dst + g * 512, KV, 0, r0, r1, r2, r3, kv_full_addr + (kv_stage_2) * 8);
                                tma_gather4_gmem2smem(dst + 16384 + g * 512, KV, 64, r0, r1, r2, r3, kv_full_addr + (kv_stage_2) * 8);
                            }
                        }
                        kv_stage_2 += 1;
                        if (kv_stage_2 == 4) { kv_stage_2 = 0; kv_phase_2 ^= 1; }
                    }
                    #pragma unroll 1
                    for (int ni_1 = 0; ni_1 < num_n_blocks_3; ni_1++) {
                        if (num_n_blocks_3 <= ni_1) {
                            break;
                        }
                        int stage_2 = ni_1 % 4;
                        if (elect_sync()) {
                            mbarrier_wait(kv_empty_addr + (stage_2) * 8, 0);
                            mbarrier_arrive_expect_tx(kv_full_addr + (stage_2) * 8, 32768);
                        }
                        int dst_1 = smem_kv_addr + (unsigned int)(stage_2 * 32768);
                        #pragma unroll
                        for (int g_1 = 0; g_1 < 32; g_1++) {
                            int off_gv = g_1 * 4;
                            int v_stage_index = stage_2 * 128 + off_gv;
                            int r0_1 = smem_page_indices[v_stage_index] * (2 * num_kv_heads) + num_kv_heads + kv_head_2;
                            int r1_1 = smem_page_indices[v_stage_index + 1] * (2 * num_kv_heads) + num_kv_heads + kv_head_2;
                            int r2_1 = smem_page_indices[v_stage_index + 2] * (2 * num_kv_heads) + num_kv_heads + kv_head_2;
                            int r3_1 = smem_page_indices[v_stage_index + 3] * (2 * num_kv_heads) + num_kv_heads + kv_head_2;
                            if (elect_sync()) {
                                tma_gather4_gmem2smem(dst_1 + g_1 * 512, KV, 0, r0_1, r1_1, r2_1, r3_1, kv_full_addr + (stage_2) * 8);
                                tma_gather4_gmem2smem(dst_1 + 16384 + g_1 * 512, KV, 64, r0_1, r1_1, r2_1, r3_1, kv_full_addr + (stage_2) * 8);
                            }
                        }
                        int next_ni = ni_1 + 4;
                        if (next_ni < num_n_blocks_3) {
                            int next_n = kv_tile_end_4 - 1 - next_ni;
                            #pragma unroll
                            for (int pp_1 = 0; pp_1 < 4; pp_1++) {
                                int pt_off_1 = pp_1 * 32 + lane;
                                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                                    :: "r"(smem_page_indices_addr + (unsigned int)((stage_2 * 128 + pt_off_1) * 4)), "l"(kv_indices + (page_begin + next_n * 128 + pt_off_1)));
                            }
                            asm volatile("cp.async.commit_group;");
                            asm volatile("cp.async.wait_group 0;");
                            asm volatile("barrier.sync 10, 32;" ::: "memory");
                            if (elect_sync()) {
                                mbarrier_wait(kv_empty_addr + (stage_2) * 8, 1);
                                mbarrier_arrive_expect_tx(kv_full_addr + (stage_2) * 8, 32768);
                            }
                            int kdst = smem_kv_addr + (unsigned int)(stage_2 * 32768);
                            #pragma unroll
                            for (int g_2 = 0; g_2 < 32; g_2++) {
                                int off_gk = g_2 * 4;
                                int next_k_stage_index = stage_2 * 128 + off_gk;
                                int r0_2 = smem_page_indices[next_k_stage_index] * (2 * num_kv_heads) + kv_head_2;
                                int r1_2 = smem_page_indices[next_k_stage_index + 1] * (2 * num_kv_heads) + kv_head_2;
                                int r2_2 = smem_page_indices[next_k_stage_index + 2] * (2 * num_kv_heads) + kv_head_2;
                                int r3_2 = smem_page_indices[next_k_stage_index + 3] * (2 * num_kv_heads) + kv_head_2;
                                if (elect_sync()) {
                                    tma_gather4_gmem2smem(kdst + g_2 * 512, KV, 0, r0_2, r1_2, r2_2, r3_2, kv_full_addr + (stage_2) * 8);
                                    tma_gather4_gmem2smem(kdst + 16384 + g_2 * 512, KV, 64, r0_2, r1_2, r2_2, r3_2, kv_full_addr + (stage_2) * 8);
                                }
                            }
                        }
                    }
                } else if (kind_4 == 0) {
                    int qo_end_3 = work_desc_4[9];
                    int q_len_4 = qo_end_3 - qo_begin_3;
                    int token_tiles_3 = (q_len_4 + 256 - 1) / 256;
                    int q_token_base_3 = q_tile_4 % token_tiles_3 * 256;
                    int packed_gqa_2 = 0;
                    if (group_size_3 > 1) {
                        if (q_len_4 * group_size_3 <= 128) {
                            packed_gqa_2 = 1;
                        }
                    }
                    int q_stages_3 = ((q_len_4 <= q_token_base_3 + 128) ? 1 : 2);
                    int token_tiles_0 = (q_len_4 + 256 - 1) / 256;
                    int token_tile = q_tile_4 % token_tiles_0;
                    int q_head_local_1 = q_tile_4 / token_tiles_0;
                    int q_head_1 = kv_head_2 * (num_q_heads / num_kv_heads) + q_head_local_1;
                    int q_token_base_1_1 = qo_begin_3 + token_tile * 256;
                    int q_remaining = q_len_4 - token_tile * 256;
                    if (packed_gqa_2 != 0) {
                        int lane_group = lane / 8;
                        int lane_chunk = lane % 8;
                        int packed_q_1 = q_len_4 * (num_q_heads / num_kv_heads);
                        #pragma unroll
                        for (int row_group = 0; row_group < 32; row_group++) {
                            int local_row = row_group * 4 + lane_group;
                            int q_token = local_row / (num_q_heads / num_kv_heads);
                            int q_head_local_0 = local_row % (num_q_heads / num_kv_heads);
                            int global_row = qo_begin_3 + q_token;
                            int q_head_1_1 = kv_head_2 * (num_q_heads / num_kv_heads) + q_head_local_0;
                            int valid = ((local_row < packed_q_1) ? 1 : 0);
                            #pragma unroll
                            for (int d_group = 0; d_group < 2; d_group++) {
                                int swizzled_chunk = (lane_chunk * 16 ^ (local_row & 7) << 4) / 16;
                                int dst_b128 = d_group * 128 * 8 + local_row * 8 + swizzled_chunk;
                                int src_elem = (global_row * num_q_heads + q_head_1_1) * 128 + d_group * 64 + lane_chunk * 8;
                                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16, %2;"
                                    :: "r"(prefill_q0_addr + (unsigned int)(dst_b128 * 16)), "l"(Q_prefill_raw + src_elem), "r"((valid) ? 16 : 0));
                            }
                        }
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(q_tail_full_addr) : "memory");
                        mbarrier_arrive(q_tail_full_addr);
                    } else if (q_remaining >= 256) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(q_full_addr, 32768);
                            tma_4d_gmem2smem(prefill_q0_addr, Q_prefill, 0, q_token_base_1_1, q_head_1, 0, q_full_addr);
                            if (q_stages_3 == 2) {
                                mbarrier_arrive_expect_tx(q_full_addr + 8, 32768);
                                tma_4d_gmem2smem(prefill_q1_addr, Q_prefill, 0, q_token_base_1_1 + 128, q_head_1, 0, q_full_addr + 8);
                            }
                        }
                    } else {
                        int q0_valid = ((q_remaining < 128) ? q_remaining : 128);
                        int q1_valid = q_remaining - 128;
                        if (q1_valid < 0) {
                            q1_valid = 0;
                        }
                        int lane_group_1 = lane / 8;
                        int lane_chunk_1 = lane % 8;
                        #pragma unroll
                        for (int row_group_1 = 0; row_group_1 < 32; row_group_1++) {
                            int local_row_1 = row_group_1 * 4 + lane_group_1;
                            int global_row_1 = q_token_base_1_1 + local_row_1;
                            int valid_1 = ((local_row_1 < q0_valid) ? 1 : 0);
                            #pragma unroll
                            for (int d_group_1 = 0; d_group_1 < 2; d_group_1++) {
                                int d_chunk = d_group_1 * 8 + lane_chunk_1;
                                int swizzled_chunk_1 = (lane_chunk_1 * 16 ^ (local_row_1 & 7) << 4) / 16;
                                int dst_b128_1 = d_group_1 * 128 * 8 + local_row_1 * 8 + swizzled_chunk_1;
                                int src_elem_1 = (global_row_1 * num_q_heads + q_head_1) * 128 + d_group_1 * 64 + lane_chunk_1 * 8;
                                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16, %2;"
                                    :: "r"(prefill_q0_addr + (unsigned int)(dst_b128_1 * 16)), "l"(Q_prefill_raw + src_elem_1), "r"((valid_1) ? 16 : 0));
                            }
                        }
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(q_tail_full_addr) : "memory");
                        mbarrier_arrive(q_tail_full_addr);
                        if (q_stages_3 == 2) {
                            int lane_group_0 = lane / 8;
                            int lane_chunk_1_1 = lane % 8;
                            #pragma unroll
                            for (int row_group_2 = 0; row_group_2 < 32; row_group_2++) {
                                int local_row_2 = row_group_2 * 4 + lane_group_0;
                                int global_row_2 = q_token_base_1_1 + 128 + local_row_2;
                                int valid_2 = ((local_row_2 < q1_valid) ? 1 : 0);
                                #pragma unroll
                                for (int d_group_2 = 0; d_group_2 < 2; d_group_2++) {
                                    int d_chunk_1 = d_group_2 * 8 + lane_chunk_1_1;
                                    int swizzled_chunk_2 = (lane_chunk_1_1 * 16 ^ (local_row_2 & 7) << 4) / 16;
                                    int dst_b128_2 = d_group_2 * 128 * 8 + local_row_2 * 8 + swizzled_chunk_2;
                                    int src_elem_2 = (global_row_2 * num_q_heads + q_head_1) * 128 + d_group_2 * 64 + lane_chunk_1_1 * 8;
                                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16, %2;"
                                        :: "r"(prefill_q1_addr + (unsigned int)(dst_b128_2 * 16)), "l"(Q_prefill_raw + src_elem_2), "r"((valid_2) ? 16 : 0));
                                }
                            }
                            asm volatile(
                                "{\n\t"
                                "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                                "}"
                                :: "r"(q_tail_full_addr + 8) : "memory");
                            mbarrier_arrive(q_tail_full_addr + 8);
                        }
                    }
                    if (elect_sync()) {
                        int kv_stage_3 = 0;
                        int kv_phase_3 = 1;
                        int num_n_blocks_0 = kv_tile_end_4 - kv_tile_begin_4;
                        #pragma unroll 1
                        for (int ni_2 = 0; ni_2 < num_n_blocks_0; ni_2++) {
                            int n_block_2 = kv_tile_end_4 - 1 - ni_2;
                            int page0_id = kv_indices[page_begin + n_block_2 * 2];
                            int page1_id = kv_indices[page_begin + n_block_2 * 2 + 1];
                            int page0 = page0_id * num_kv_heads + kv_head_2;
                            int page1 = page1_id * num_kv_heads + kv_head_2;
                            mbarrier_wait(prefill_kv_empty_addr + (kv_stage_3) * 8, kv_phase_3);
                            mbarrier_arrive_expect_tx(prefill_kv_full_addr + (kv_stage_3) * 8, 32768);
                            int k_dst = prefill_kv_addr + (unsigned int)(kv_stage_3 * 32768);
                            if (page1_id == page0_id + 1) {
                                tma_5d_gmem2smem(k_dst, K_prefill_pair, 0, 0, page0_id, 0, kv_head_2, prefill_kv_full_addr + (kv_stage_3) * 8);
                            } else {
                                tma_4d_gmem2smem(k_dst, K, 0, 0, 0, page0, prefill_kv_full_addr + (kv_stage_3) * 8);
                                tma_4d_gmem2smem(k_dst + 8192, K, 0, 0, 0, page1, prefill_kv_full_addr + (kv_stage_3) * 8);
                                tma_4d_gmem2smem(k_dst + 16384, K, 0, 0, 1, page0, prefill_kv_full_addr + (kv_stage_3) * 8);
                                tma_4d_gmem2smem(k_dst + 24576, K, 0, 0, 1, page1, prefill_kv_full_addr + (kv_stage_3) * 8);
                            }
                            kv_stage_3 += 1;
                            if (kv_stage_3 == 2) { kv_stage_3 = 0; kv_phase_3 ^= 1; }
                            mbarrier_wait(prefill_kv_empty_addr + (kv_stage_3) * 8, kv_phase_3);
                            mbarrier_arrive_expect_tx(prefill_kv_full_addr + (kv_stage_3) * 8, 32768);
                            int v_dst = prefill_kv_addr + (unsigned int)(kv_stage_3 * 32768);
                            if (page1_id == page0_id + 1) {
                                tma_5d_gmem2smem(v_dst, V_prefill_pair, 0, 0, page0_id, 0, kv_head_2, prefill_kv_full_addr + (kv_stage_3) * 8);
                            } else {
                                tma_4d_gmem2smem(v_dst, V, 0, 0, 0, page0, prefill_kv_full_addr + (kv_stage_3) * 8);
                                tma_4d_gmem2smem(v_dst + 8192, V, 0, 0, 0, page1, prefill_kv_full_addr + (kv_stage_3) * 8);
                                tma_4d_gmem2smem(v_dst + 16384, V, 0, 0, 1, page0, prefill_kv_full_addr + (kv_stage_3) * 8);
                                tma_4d_gmem2smem(v_dst + 24576, V, 0, 0, 1, page1, prefill_kv_full_addr + (kv_stage_3) * 8);
                            }
                            kv_stage_3 += 1;
                            if (kv_stage_3 == 2) { kv_stage_3 = 0; kv_phase_3 ^= 1; }
                        }
                    }
                    int splits_2 = work_desc_4[12];
                    if (splits_2 > 1 && q_stages_3 == 1) {
                        mbarrier_wait(prefill_partial_ready_addr, _phase_prefill_partial_ready_0);
                        _phase_prefill_partial_ready_0 ^= 1;
                    }
                    if (splits_2 > 1 && q_stages_3 == 1) {
                        int request_3 = work_desc_4[2];
                        int q_head_local_0_1 = q_tile_4 / token_tiles_3;
                        int q_head_1_2 = kv_head_2 * group_size_3 + q_head_local_0_1;
                        int lane_2 = lane;
                        int warp_in_pair = warp - 14;
                        int reduce_worker = warp_in_pair * 32 + lane_2;
                        int valid_rows = q_len_4;
                        if (packed_gqa_2 != 0) {
                            valid_rows = q_len_4 * group_size_3;
                        }
                        int logical_output_2 = request_3 * num_kv_heads + kv_head_2;
                        int partial_slot_2 = logical_output_2 * max_splits + work_desc_4[5];
                        #pragma unroll 1
                        for (int publish_pass = 0; publish_pass < 2; publish_pass++) {
                            int publish_group = publish_pass * 2 + warp_in_pair;
                            int publish_row = publish_group * 32 + lane_2;
                            if (publish_row < valid_rows) {
                                int partial_stat_idx = partial_slot_2 * 128 + publish_row;
                                *((float*)(partial_M + partial_stat_idx)) = prefill_scale[publish_row + 512];
                                *((float*)(partial_D + partial_stat_idx)) = prefill_scale[publish_row + 256];
                                #pragma unroll
                                for (int col_5 = 0; col_5 < 8; col_5++) {
                                    int smem_addr = prefill_partial_tile_addr + (unsigned int)((publish_row * 128 + col_5 * 16) * 4);
                                    int partial_o_idx_1 = partial_stat_idx * 128 + col_5 * 16;
                                    float values_lo[8];
                                    float values_hi[8];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&values_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo[(0) + 3]))
                                        : "r"(smem_addr));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&values_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo[(4) + 3]))
                                        : "r"(smem_addr + 16));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&values_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi[(0) + 3]))
                                        : "r"(smem_addr + 32));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&values_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi[(4) + 3]))
                                        : "r"(smem_addr + 48));
                                    {
                                        unsigned _stv8_0_0 = __float_as_uint(values_lo[0 + 0]);
                                        unsigned _stv8_0_1 = __float_as_uint(values_lo[0 + 1]);
                                        unsigned _stv8_0_2 = __float_as_uint(values_lo[0 + 2]);
                                        unsigned _stv8_0_3 = __float_as_uint(values_lo[0 + 3]);
                                        unsigned _stv8_0_4 = __float_as_uint(values_lo[0 + 4]);
                                        unsigned _stv8_0_5 = __float_as_uint(values_lo[0 + 5]);
                                        unsigned _stv8_0_6 = __float_as_uint(values_lo[0 + 6]);
                                        unsigned _stv8_0_7 = __float_as_uint(values_lo[0 + 7]);
                                        asm volatile(
                                            "st.global.v4.b32 [%0], {%1, %2, %3, %4};\n\t"
                                            "st.global.v4.b32 [%0+16], {%5, %6, %7, %8};"
                                            :: "l"((void*)(partial_O + partial_o_idx_1 + (0))), "r"(_stv8_0_0), "r"(_stv8_0_1), "r"(_stv8_0_2), "r"(_stv8_0_3), "r"(_stv8_0_4), "r"(_stv8_0_5), "r"(_stv8_0_6), "r"(_stv8_0_7) : "memory");
                                    }
                                    {
                                        unsigned _stv8_1_0 = __float_as_uint(values_hi[0 + 0]);
                                        unsigned _stv8_1_1 = __float_as_uint(values_hi[0 + 1]);
                                        unsigned _stv8_1_2 = __float_as_uint(values_hi[0 + 2]);
                                        unsigned _stv8_1_3 = __float_as_uint(values_hi[0 + 3]);
                                        unsigned _stv8_1_4 = __float_as_uint(values_hi[0 + 4]);
                                        unsigned _stv8_1_5 = __float_as_uint(values_hi[0 + 5]);
                                        unsigned _stv8_1_6 = __float_as_uint(values_hi[0 + 6]);
                                        unsigned _stv8_1_7 = __float_as_uint(values_hi[0 + 7]);
                                        asm volatile(
                                            "st.global.v4.b32 [%0], {%1, %2, %3, %4};\n\t"
                                            "st.global.v4.b32 [%0+16], {%5, %6, %7, %8};"
                                            :: "l"((void*)(partial_O + (partial_o_idx_1 + 8) + (0))), "r"(_stv8_1_0), "r"(_stv8_1_1), "r"(_stv8_1_2), "r"(_stv8_1_3), "r"(_stv8_1_4), "r"(_stv8_1_5), "r"(_stv8_1_6), "r"(_stv8_1_7) : "memory");
                                    }
                                }
                            }
                        }
                        __threadfence();
                        asm volatile("barrier.sync 9, 64;" ::: "memory");
                        if (reduce_worker == 0) {
                            int _atomic_old_1 = atomicAdd(&split_completion[logical_output_2], 1);
                            int old_count_1 = _atomic_old_1;
                            split_reduce_flag[0] = ((old_count_1 + 1 == splits_2) ? 1 : 0);
                        }
                        asm volatile("barrier.sync 9, 64;" ::: "memory");
                        if (split_reduce_flag[0] != 0) {
                            __threadfence();
                            #pragma unroll 1
                            for (int row_group_3 = 0; row_group_3 < 2; row_group_3++) {
                                int reduce_row = row_group_3 * 64 + reduce_worker;
                                if (reduce_row < valid_rows) {
                                    float reduce_m_1 = -BLACKWELL_MSA_INF;
                                    #pragma unroll 1
                                    for (int reduce_split_1 = 0; reduce_split_1 < max_splits; reduce_split_1++) {
                                        if (splits_2 <= reduce_split_1) {
                                            break;
                                        }
                                        int reduce_slot_1 = logical_output_2 * max_splits + reduce_split_1;
                                        int reduce_stat_idx_1 = reduce_slot_1 * 128 + reduce_row;
                                        float split_m_1 = partial_M[reduce_stat_idx_1];
                                        float _max_9 = max_noftz(reduce_m_1, split_m_1);
                                        reduce_m_1 = _max_9;
                                    }
                                    float reduce_d_1 = 0.0f;
                                    #pragma unroll 1
                                    for (int reduce_split_2 = 0; reduce_split_2 < max_splits; reduce_split_2++) {
                                        if (splits_2 <= reduce_split_2) {
                                            break;
                                        }
                                        int reduce_slot_2 = logical_output_2 * max_splits + reduce_split_2;
                                        int reduce_stat_idx_2 = reduce_slot_2 * 128 + reduce_row;
                                        float split_m_2 = partial_M[reduce_stat_idx_2];
                                        float split_d_1 = partial_D[reduce_stat_idx_2];
                                        float _exp2_9 = approx_exp2(softmax_scale_log2 * (split_m_2 - reduce_m_1));
                                        float split_scale_1 = ((split_m_2 == -BLACKWELL_MSA_INF) ? 0.0f : _exp2_9);
                                        prefill_split_weights[reduce_split_2 * 128 + reduce_row] = split_scale_1;
                                        reduce_d_1 += split_d_1 * split_scale_1;
                                    }
                                    float _rcp_4 = approx_rcp(reduce_d_1);
                                    prefill_scale[reduce_row] = _rcp_4;
                                }
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("barrier.sync 9, 64;" ::: "memory");
                            int reduce_row_lane = reduce_worker / 16;
                            int reduce_d_chunk = reduce_worker % 16;
                            int num_reduce_groups = (valid_rows + 3) / 4;
                            #pragma unroll 1
                            for (int reduce_group = 0; reduce_group < 32; reduce_group++) {
                                if (num_reduce_groups <= reduce_group) {
                                    break;
                                }
                                int reduce_row_1 = reduce_group * 4 + reduce_row_lane;
                                if (reduce_row_1 < valid_rows) {
                                    float reduce_o_1[8];
                                    #pragma unroll
                                    for (int elem = 0; elem < 8; elem++) {
                                        reduce_o_1[elem] = 0.0f;
                                    }
                                    #pragma unroll 1
                                    for (int reduce_split_3 = 0; reduce_split_3 < max_splits; reduce_split_3++) {
                                        if (splits_2 <= reduce_split_3) {
                                            break;
                                        }
                                        int reduce_slot_3 = logical_output_2 * max_splits + reduce_split_3;
                                        int reduce_stat_idx_3 = reduce_slot_3 * 128 + reduce_row_1;
                                        int split_o_idx_1 = reduce_stat_idx_3 * 128 + reduce_d_chunk * 8;
                                        float split_o_1[8];
                                        {
                                            unsigned _ldv8_2_0;
                                            unsigned _ldv8_2_1;
                                            unsigned _ldv8_2_2;
                                            unsigned _ldv8_2_3;
                                            unsigned _ldv8_2_4;
                                            unsigned _ldv8_2_5;
                                            unsigned _ldv8_2_6;
                                            unsigned _ldv8_2_7;
                                            asm volatile(
                                                "ld.global.v4.b32 {%0, %1, %2, %3}, [%8];\n\t"
                                                "ld.global.v4.b32 {%4, %5, %6, %7}, [%8+16];"
                                                : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(partial_O + (split_o_idx_1))) : "memory");
                                            split_o_1[0 + 0] = __uint_as_float(_ldv8_2_0);
                                            split_o_1[0 + 1] = __uint_as_float(_ldv8_2_1);
                                            split_o_1[0 + 2] = __uint_as_float(_ldv8_2_2);
                                            split_o_1[0 + 3] = __uint_as_float(_ldv8_2_3);
                                            split_o_1[0 + 4] = __uint_as_float(_ldv8_2_4);
                                            split_o_1[0 + 5] = __uint_as_float(_ldv8_2_5);
                                            split_o_1[0 + 6] = __uint_as_float(_ldv8_2_6);
                                            split_o_1[0 + 7] = __uint_as_float(_ldv8_2_7);
                                        }
                                        float split_scale_2 = prefill_split_weights[reduce_split_3 * 128 + reduce_row_1];
                                        #pragma unroll
                                        for (int elem_1 = 0; elem_1 < 8; elem_1++) {
                                            float _fma_1 = __fmaf_rn(split_o_1[elem_1], split_scale_2, reduce_o_1[elem_1]);
                                            reduce_o_1[elem_1] = _fma_1;
                                        }
                                    }
                                    int reduce_local_q_row = q_token_base_3 + reduce_row_1;
                                    int reduce_q_head = q_head_1_2;
                                    if (packed_gqa_2 != 0) {
                                        reduce_local_q_row = reduce_row_1 / group_size_3;
                                        reduce_q_head = kv_head_2 * group_size_3 + reduce_row_1 % group_size_3;
                                    }
                                    int reduce_output_row = (qo_begin_3 + reduce_local_q_row) * num_q_heads + reduce_q_head;
                                    {
                                        const float2 _prescale2_3 = {prefill_scale[reduce_row_1], prefill_scale[reduce_row_1]};
                                        #if __CUDA_ARCH__ >= 1000
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 4; _ps++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&reduce_o_1[0])[_ps], _prescale2_3);
                                        #else
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 8; _ps++)
                                            reduce_o_1[0 + _ps] *= prefill_scale[reduce_row_1];
                                        #endif
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(reduce_o_1[0 + 0], reduce_o_1[0 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(reduce_o_1[0 + 2], reduce_o_1[0 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(reduce_o_1[0 + 4], reduce_o_1[0 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(reduce_o_1[0 + 6], reduce_o_1[0 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (reduce_output_row * 128 + reduce_d_chunk * 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                }
                            }
                        }
                    }
                }
                if (direct_decode_4 == 0) {
                    if (elect_sync()) {
                        if (work_slot_4 == 0) {
                            mbarrier_arrive(work_empty_0_addr);
                        } else {
                            mbarrier_arrive(work_empty_1_addr);
                        }
                    }
                }
            }
        }
    // ---- Role: producer_aux ----
    } else if (warp == 15) {
        { // producer_aux_main
            int direct_decode_5 = ((attention_mode == 1 && gridDim.x * gridDim.y >= num_requests * num_kv_heads) ? 1 : 0);
            unsigned int _phase_work_full_0_0_4 = 0;
            unsigned int _phase_work_full_1_0_4 = 0;
            unsigned int _phase_prefill_partial_ready_0_1 = 0;
            if (direct_decode_5 == 0) {
                #pragma unroll 1
                for (int task_iter_4 = 0; task_iter_4 < max_task_claims + 1; task_iter_4++) {
                    int work_slot_5 = task_iter_4 % 2;
                    int* work_desc_5 = work_desc_slots + (work_slot_5 * 13);
                    if (work_slot_5 == 0) {
                        mbarrier_wait(work_full_0_addr, _phase_work_full_0_0_4);
                        _phase_work_full_0_0_4 ^= 1;
                    } else {
                        mbarrier_wait(work_full_1_addr, _phase_work_full_1_0_4);
                        _phase_work_full_1_0_4 ^= 1;
                    }
                    asm volatile("barrier.sync 8, 480;" ::: "memory");
                    int ticket_5 = work_desc_5[0];
                    if (ticket_5 < 0) {
                        if (elect_sync()) {
                            if (work_slot_5 == 0) {
                                mbarrier_arrive(work_empty_0_addr);
                            } else {
                                mbarrier_arrive(work_empty_1_addr);
                            }
                        }
                        break;
                    }
                    int kind_5 = -1;
                    if (attention_mode != 0) {
                        kind_5 = work_desc_5[1];
                    }
                    if (kind_5 == 0) {
                        int splits_3 = work_desc_5[12];
                        if (splits_3 > 1) {
                            int request_4 = work_desc_5[2];
                            int kv_head_3 = work_desc_5[3];
                            int q_tile_5 = work_desc_5[4];
                            int qo_begin_4 = work_desc_5[8];
                            int qo_end_4 = work_desc_5[9];
                            int q_len_5 = qo_end_4 - qo_begin_4;
                            int group_size_4 = num_q_heads / num_kv_heads;
                            int token_tiles_4 = (q_len_5 + 256 - 1) / 256;
                            int q_token_base_4 = q_tile_5 % token_tiles_4 * 256;
                            int q_head_local_2 = q_tile_5 / token_tiles_4;
                            int q_head_2 = kv_head_3 * group_size_4 + q_head_local_2;
                            int packed_gqa_3 = 0;
                            if (group_size_4 > 1) {
                                if (q_len_5 * group_size_4 <= 128) {
                                    packed_gqa_3 = 1;
                                }
                            }
                            mbarrier_wait(prefill_partial_ready_addr, _phase_prefill_partial_ready_0_1);
                            _phase_prefill_partial_ready_0_1 ^= 1;
                            int lane_0_1 = lane;
                            int warp_in_pair_1 = warp - 14;
                            int reduce_worker_1 = warp_in_pair_1 * 32 + lane_0_1;
                            int valid_rows_1 = q_len_5;
                            if (packed_gqa_3 != 0) {
                                valid_rows_1 = q_len_5 * group_size_4;
                            }
                            int logical_output_3 = request_4 * num_kv_heads + kv_head_3;
                            int partial_slot_3 = logical_output_3 * max_splits + work_desc_5[5];
                            #pragma unroll 1
                            for (int publish_pass_1 = 0; publish_pass_1 < 2; publish_pass_1++) {
                                int publish_group_1 = publish_pass_1 * 2 + warp_in_pair_1;
                                int publish_row_1 = publish_group_1 * 32 + lane_0_1;
                                if (publish_row_1 < valid_rows_1) {
                                    int partial_stat_idx_1 = partial_slot_3 * 128 + publish_row_1;
                                    *((float*)(partial_M + partial_stat_idx_1)) = prefill_scale[publish_row_1 + 512];
                                    *((float*)(partial_D + partial_stat_idx_1)) = prefill_scale[publish_row_1 + 256];
                                    #pragma unroll
                                    for (int col_6 = 0; col_6 < 8; col_6++) {
                                        int smem_addr_1 = prefill_partial_tile_addr + (unsigned int)((publish_row_1 * 128 + col_6 * 16) * 4);
                                        int partial_o_idx_2 = partial_stat_idx_1 * 128 + col_6 * 16;
                                        float values_lo_1[8];
                                        float values_hi_1[8];
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&values_lo_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo_1[(0) + 3]))
                                            : "r"(smem_addr_1));
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&values_lo_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo_1[(4) + 3]))
                                            : "r"(smem_addr_1 + 16));
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&values_hi_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi_1[(0) + 3]))
                                            : "r"(smem_addr_1 + 32));
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&values_hi_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi_1[(4) + 3]))
                                            : "r"(smem_addr_1 + 48));
                                        {
                                            unsigned _stv8_0_0 = __float_as_uint(values_lo_1[0 + 0]);
                                            unsigned _stv8_0_1 = __float_as_uint(values_lo_1[0 + 1]);
                                            unsigned _stv8_0_2 = __float_as_uint(values_lo_1[0 + 2]);
                                            unsigned _stv8_0_3 = __float_as_uint(values_lo_1[0 + 3]);
                                            unsigned _stv8_0_4 = __float_as_uint(values_lo_1[0 + 4]);
                                            unsigned _stv8_0_5 = __float_as_uint(values_lo_1[0 + 5]);
                                            unsigned _stv8_0_6 = __float_as_uint(values_lo_1[0 + 6]);
                                            unsigned _stv8_0_7 = __float_as_uint(values_lo_1[0 + 7]);
                                            asm volatile(
                                                "st.global.v4.b32 [%0], {%1, %2, %3, %4};\n\t"
                                                "st.global.v4.b32 [%0+16], {%5, %6, %7, %8};"
                                                :: "l"((void*)(partial_O + partial_o_idx_2 + (0))), "r"(_stv8_0_0), "r"(_stv8_0_1), "r"(_stv8_0_2), "r"(_stv8_0_3), "r"(_stv8_0_4), "r"(_stv8_0_5), "r"(_stv8_0_6), "r"(_stv8_0_7) : "memory");
                                        }
                                        {
                                            unsigned _stv8_1_0 = __float_as_uint(values_hi_1[0 + 0]);
                                            unsigned _stv8_1_1 = __float_as_uint(values_hi_1[0 + 1]);
                                            unsigned _stv8_1_2 = __float_as_uint(values_hi_1[0 + 2]);
                                            unsigned _stv8_1_3 = __float_as_uint(values_hi_1[0 + 3]);
                                            unsigned _stv8_1_4 = __float_as_uint(values_hi_1[0 + 4]);
                                            unsigned _stv8_1_5 = __float_as_uint(values_hi_1[0 + 5]);
                                            unsigned _stv8_1_6 = __float_as_uint(values_hi_1[0 + 6]);
                                            unsigned _stv8_1_7 = __float_as_uint(values_hi_1[0 + 7]);
                                            asm volatile(
                                                "st.global.v4.b32 [%0], {%1, %2, %3, %4};\n\t"
                                                "st.global.v4.b32 [%0+16], {%5, %6, %7, %8};"
                                                :: "l"((void*)(partial_O + (partial_o_idx_2 + 8) + (0))), "r"(_stv8_1_0), "r"(_stv8_1_1), "r"(_stv8_1_2), "r"(_stv8_1_3), "r"(_stv8_1_4), "r"(_stv8_1_5), "r"(_stv8_1_6), "r"(_stv8_1_7) : "memory");
                                        }
                                    }
                                }
                            }
                            __threadfence();
                            asm volatile("barrier.sync 9, 64;" ::: "memory");
                            if (reduce_worker_1 == 0) {
                                int _atomic_old_2 = atomicAdd(&split_completion[logical_output_3], 1);
                                int old_count_2 = _atomic_old_2;
                                split_reduce_flag[0] = ((old_count_2 + 1 == splits_3) ? 1 : 0);
                            }
                            asm volatile("barrier.sync 9, 64;" ::: "memory");
                            if (split_reduce_flag[0] != 0) {
                                __threadfence();
                                #pragma unroll 1
                                for (int row_group_4 = 0; row_group_4 < 2; row_group_4++) {
                                    int reduce_row_2 = row_group_4 * 64 + reduce_worker_1;
                                    if (reduce_row_2 < valid_rows_1) {
                                        float reduce_m_2 = -BLACKWELL_MSA_INF;
                                        #pragma unroll 1
                                        for (int reduce_split_4 = 0; reduce_split_4 < max_splits; reduce_split_4++) {
                                            if (splits_3 <= reduce_split_4) {
                                                break;
                                            }
                                            int reduce_slot_4 = logical_output_3 * max_splits + reduce_split_4;
                                            int reduce_stat_idx_4 = reduce_slot_4 * 128 + reduce_row_2;
                                            float split_m_3 = partial_M[reduce_stat_idx_4];
                                            float _max_10 = max_noftz(reduce_m_2, split_m_3);
                                            reduce_m_2 = _max_10;
                                        }
                                        float reduce_d_2 = 0.0f;
                                        #pragma unroll 1
                                        for (int reduce_split_5 = 0; reduce_split_5 < max_splits; reduce_split_5++) {
                                            if (splits_3 <= reduce_split_5) {
                                                break;
                                            }
                                            int reduce_slot_5 = logical_output_3 * max_splits + reduce_split_5;
                                            int reduce_stat_idx_5 = reduce_slot_5 * 128 + reduce_row_2;
                                            float split_m_4 = partial_M[reduce_stat_idx_5];
                                            float split_d_2 = partial_D[reduce_stat_idx_5];
                                            float _exp2_10 = approx_exp2(softmax_scale_log2 * (split_m_4 - reduce_m_2));
                                            float split_scale_3 = ((split_m_4 == -BLACKWELL_MSA_INF) ? 0.0f : _exp2_10);
                                            prefill_split_weights[reduce_split_5 * 128 + reduce_row_2] = split_scale_3;
                                            reduce_d_2 += split_d_2 * split_scale_3;
                                        }
                                        float _rcp_5 = approx_rcp(reduce_d_2);
                                        prefill_scale[reduce_row_2] = _rcp_5;
                                    }
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile("barrier.sync 9, 64;" ::: "memory");
                                int reduce_row_lane_1 = reduce_worker_1 / 16;
                                int reduce_d_chunk_1 = reduce_worker_1 % 16;
                                int num_reduce_groups_1 = (valid_rows_1 + 3) / 4;
                                #pragma unroll 1
                                for (int reduce_group_1 = 0; reduce_group_1 < 32; reduce_group_1++) {
                                    if (num_reduce_groups_1 <= reduce_group_1) {
                                        break;
                                    }
                                    int reduce_row_3 = reduce_group_1 * 4 + reduce_row_lane_1;
                                    if (reduce_row_3 < valid_rows_1) {
                                        float reduce_o_2[8];
                                        #pragma unroll
                                        for (int elem_2 = 0; elem_2 < 8; elem_2++) {
                                            reduce_o_2[elem_2] = 0.0f;
                                        }
                                        #pragma unroll 1
                                        for (int reduce_split_6 = 0; reduce_split_6 < max_splits; reduce_split_6++) {
                                            if (splits_3 <= reduce_split_6) {
                                                break;
                                            }
                                            int reduce_slot_6 = logical_output_3 * max_splits + reduce_split_6;
                                            int reduce_stat_idx_6 = reduce_slot_6 * 128 + reduce_row_3;
                                            int split_o_idx_2 = reduce_stat_idx_6 * 128 + reduce_d_chunk_1 * 8;
                                            float split_o_2[8];
                                            {
                                                unsigned _ldv8_2_0;
                                                unsigned _ldv8_2_1;
                                                unsigned _ldv8_2_2;
                                                unsigned _ldv8_2_3;
                                                unsigned _ldv8_2_4;
                                                unsigned _ldv8_2_5;
                                                unsigned _ldv8_2_6;
                                                unsigned _ldv8_2_7;
                                                asm volatile(
                                                    "ld.global.v4.b32 {%0, %1, %2, %3}, [%8];\n\t"
                                                    "ld.global.v4.b32 {%4, %5, %6, %7}, [%8+16];"
                                                    : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(partial_O + (split_o_idx_2))) : "memory");
                                                split_o_2[0 + 0] = __uint_as_float(_ldv8_2_0);
                                                split_o_2[0 + 1] = __uint_as_float(_ldv8_2_1);
                                                split_o_2[0 + 2] = __uint_as_float(_ldv8_2_2);
                                                split_o_2[0 + 3] = __uint_as_float(_ldv8_2_3);
                                                split_o_2[0 + 4] = __uint_as_float(_ldv8_2_4);
                                                split_o_2[0 + 5] = __uint_as_float(_ldv8_2_5);
                                                split_o_2[0 + 6] = __uint_as_float(_ldv8_2_6);
                                                split_o_2[0 + 7] = __uint_as_float(_ldv8_2_7);
                                            }
                                            float split_scale_4 = prefill_split_weights[reduce_split_6 * 128 + reduce_row_3];
                                            #pragma unroll
                                            for (int elem_3 = 0; elem_3 < 8; elem_3++) {
                                                float _fma_2 = __fmaf_rn(split_o_2[elem_3], split_scale_4, reduce_o_2[elem_3]);
                                                reduce_o_2[elem_3] = _fma_2;
                                            }
                                        }
                                        int reduce_local_q_row_1 = q_token_base_4 + reduce_row_3;
                                        int reduce_q_head_1 = q_head_2;
                                        if (packed_gqa_3 != 0) {
                                            reduce_local_q_row_1 = reduce_row_3 / group_size_4;
                                            reduce_q_head_1 = kv_head_3 * group_size_4 + reduce_row_3 % group_size_4;
                                        }
                                        int reduce_output_row_1 = (qo_begin_4 + reduce_local_q_row_1) * num_q_heads + reduce_q_head_1;
                                        {
                                            const float2 _prescale2_3 = {prefill_scale[reduce_row_3], prefill_scale[reduce_row_3]};
                                            #if __CUDA_ARCH__ >= 1000
                                            #pragma unroll
                                            for (int _ps = 0; _ps < 4; _ps++)
                                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&reduce_o_2[0])[_ps], _prescale2_3);
                                            #else
                                            #pragma unroll
                                            for (int _ps = 0; _ps < 8; _ps++)
                                                reduce_o_2[0 + _ps] *= prefill_scale[reduce_row_3];
                                            #endif
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(reduce_o_2[0 + 0], reduce_o_2[0 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(reduce_o_2[0 + 2], reduce_o_2[0 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(reduce_o_2[0 + 4], reduce_o_2[0 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(reduce_o_2[0 + 6], reduce_o_2[0 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (reduce_output_row_1 * 128 + reduce_d_chunk_1 * 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (elect_sync()) {
                        if (work_slot_5 == 0) {
                            mbarrier_arrive(work_empty_0_addr);
                        } else {
                            mbarrier_arrive(work_empty_1_addr);
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"
// clang-format on
