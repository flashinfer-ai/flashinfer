/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeFmhaTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeFmhaTensorMapPack { CakeFmhaTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 384
#define TMEM_SCORES_0_OFFSET 0
#define TMEM_SOFTMAX_0_OFFSET 64
#define TMEM_OUTPUT_HI_OFFSET 128
#define TMEM_OUTPUT_LO_OFFSET 256
#define NUM_Q_STAGES 1
#define NUM_KV_STAGES 2
#define NUM_PO_STAGES 6
#define SMEM_SSCALE_OFF 1024
#define SMEM_SSCALE_STAGE_BYTES 3072
#define SMEM_SSCALE_STRIDE 3072
#define SMEM_SMEM_Q_HI_OFF 4096
#define SMEM_SMEM_Q_HI_STAGE_BYTES 32768
#define SMEM_SMEM_Q_HI_STRIDE 32768
#define SMEM_SMEM_Q_LO_OFF 36864
#define SMEM_SMEM_Q_LO_STAGE_BYTES 32768
#define SMEM_SMEM_Q_LO_STRIDE 32768
#define SMEM_SMEM_KV_HI_OFF 69632
#define SMEM_SMEM_KV_HI_STAGE_BYTES 32768
#define SMEM_SMEM_KV_HI_STRIDE 32768
#define SMEM_SMEM_KV_LO_OFF 135168
#define SMEM_SMEM_KV_LO_STAGE_BYTES 32768
#define SMEM_SMEM_KV_LO_STRIDE 32768
#define SMEM_SMEM_V_HI_OFF 69632
#define SMEM_SMEM_V_HI_STAGE_BYTES 32768
#define SMEM_SMEM_V_HI_STRIDE 32768
#define SMEM_SMEM_V_LO_OFF 135168
#define SMEM_SMEM_V_LO_STAGE_BYTES 32768
#define SMEM_SMEM_V_LO_STRIDE 32768
#define SMEM_SMEM_PAGES_OFF 200704
#define SMEM_SMEM_PAGES_STAGE_BYTES 192
#define SMEM_SMEM_PAGES_STRIDE 192
#define SMEM_WORK_ID_SLOT_OFF 200896
#define SMEM_WORK_ID_SLOT_STAGE_BYTES 4
#define SMEM_WORK_ID_SLOT_STRIDE 4
#define SMEM_TOTAL 200960
#define IS_CAUSAL 0
#ifndef NUM_M_BLOCKS
#define NUM_M_BLOCKS 1
#endif
#ifndef HEADS_PER_GROUP
#define HEADS_PER_GROUP 5
#endif
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 256
#define HEAD_DIM_HALF 128
#define MMA_K 16
#define PAGE_SIZE 16

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


__device__ __forceinline__ void tmem_st_x16(int tmem_addr, uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "r"(src[0]),  "r"(src[1]),  "r"(src[2]),  "r"(src[3]),
           "r"(src[4]),  "r"(src[5]),  "r"(src[6]),  "r"(src[7]),
           "r"(src[8]),  "r"(src[9]),  "r"(src[10]), "r"(src[11]),
           "r"(src[12]), "r"(src[13]), "r"(src[14]), "r"(src[15]));
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
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

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_cake_fmha_context_fp16_hd256(CakeFmhaTensorMap const* Q, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V, __half* __restrict__ O_ptr, int* __restrict__ page_table, int* __restrict__ seq_lens_q, int* __restrict__ seq_lens_kv, int* __restrict__ cu_seq_lens_q, float softmax_scale_log2, int total_bh, int max_pages_per_seq, unsigned int* __restrict__ dynamic_counter)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(V)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* sScale = reinterpret_cast<float*>(smem_raw + 1024);
    const int sScale_addr = smem + 1024;
    __half* smem_q_hi = reinterpret_cast<__half*>(smem_raw + 4096);
    const int smem_q_hi_addr = smem + 4096;
    __half* smem_q_lo = reinterpret_cast<__half*>(smem_raw + 36864);
    const int smem_q_lo_addr = smem + 36864;
    __half* smem_kv_hi = reinterpret_cast<__half*>(smem_raw + 69632);
    const int smem_kv_hi_addr = smem + 69632;
    __half* smem_kv_lo = reinterpret_cast<__half*>(smem_raw + 135168);
    const int smem_kv_lo_addr = smem + 135168;
    __half* smem_v_hi = reinterpret_cast<__half*>(smem_raw + 69632);
    const int smem_v_hi_addr = smem + 69632;
    __half* smem_v_lo = reinterpret_cast<__half*>(smem_raw + 135168);
    const int smem_v_lo_addr = smem + 135168;
    int* smem_pages = reinterpret_cast<int*>(smem_raw + 200704);
    const int smem_pages_addr = smem + 200704;
    unsigned int* work_id_slot = reinterpret_cast<unsigned int*>(smem_raw + 200896);
    const int work_id_slot_addr = smem + 200896;

    // Mbarrier init (14 groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv' ---
            // kv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // kv_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // p_full: 1 barriers, init_count=256
            mbarrier_init(smem + 48, 256);
            // p_full_2: 1 barriers, init_count=128
            mbarrier_init(smem + 56, 128);
            // corr_sig: 1 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            // corr_done: 1 barriers, init_count=128
            mbarrier_init(smem + 72, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'po' ---
            // page_full: 6 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // page_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // work_id_full: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // work_id_empty: 1 barriers, init_count=11
            mbarrier_init(smem + 200, 11);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 0) {
        int _tmem_hold = smem + 208;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define kv_full_addr (mbar_base + 8)
    #define kv_empty_addr (mbar_base + 24)
    #define s_full_addr (mbar_base + 40)
    #define p_full_addr (mbar_base + 48)
    #define p_full_2_addr (mbar_base + 56)
    #define corr_sig_addr (mbar_base + 64)
    #define corr_done_addr (mbar_base + 72)
    #define o_full_addr (mbar_base + 80)
    #define q_empty_addr (mbar_base + 88)
    #define page_full_addr (mbar_base + 96)
    #define page_empty_addr (mbar_base + 144)
    #define work_id_full_addr (mbar_base + 192)
    #define work_id_empty_addr (mbar_base + 200)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores_0 = taddr;
    const int tmem_softmax_0 = taddr + 64;
    const int tmem_output_hi = taddr + 128;
    const int tmem_output_lo = taddr + 256;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // softmax_main
            unsigned int total_tiles = NUM_M_BLOCKS * total_bh;
            unsigned int sv_f16[16];
            unsigned int _phase_work_id_full_0 = 0;
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_corr_done_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_iter = 0; tile_iter < total_tiles; tile_iter++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0);
                _phase_work_id_full_0 ^= 1;
                unsigned int tile_idx = work_id_slot[0];
                if (elect_sync()) {
                    mbarrier_arrive(work_id_empty_addr);
                }
                if (tile_idx >= total_tiles) {
                    break;
                }
                unsigned int m_block;
                unsigned int bh;
                {
                    m_block = tile_idx % (unsigned int)NUM_M_BLOCKS;
                    bh = tile_idx / (unsigned int)NUM_M_BLOCKS;
                }
                int seqlen_kv_bh = seq_lens_kv[bh];
                unsigned int num_n_blocks = (seqlen_kv_bh + BLOCK_N - 1) / BLOCK_N;
                int causal_row;
                unsigned int num_masked_iters;
                {
                    num_masked_iters = 0;
                }
                float row_max_val = -LOOM_INF;
                float row_sum_val = 0.0f;
                #pragma unroll 1
                for (unsigned int n_iter = 0; n_iter < num_masked_iters; n_iter++) {
                    int n_block = num_n_blocks - 1 - n_iter;
                    mbarrier_wait(s_full_addr, _phase_s_full_0);
                    _phase_s_full_0 ^= 1;
                    int s_base = taddr + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_0[128];
                    tmem_ld_x32(&_tmem_load_0[0], s_base);
                    tmem_ld_x32(&_tmem_load_0[32], s_base + 32);
                    tmem_ld_x32(&_tmem_load_0[64], s_base + 64);
                    tmem_ld_x32(&_tmem_load_0[96], s_base + 96);
                    int tail_valid = seqlen_kv_bh - n_block * BLOCK_N;
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
                    #pragma unroll
                    for (int _i_1 = 0; _i_1 < 32; _i_1++) {
                        if (!(_slice_lo_mask_0 & (1u << _i_1))) _tmem_load_0[0 + _i_1] = -LOOM_INF;
                    }
                    uint32_t _slice_lo_mask_1;
                    {
                        int _lim_2 = tail_valid - 32;
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
                        if (!(_slice_lo_mask_1 & (1u << _i_3))) _tmem_load_0[32 + _i_3] = -LOOM_INF;
                    }
                    uint32_t _slice_lo_mask_2;
                    {
                        int _lim_4 = tail_valid - 64;
                        if (_lim_4 <= 0) { _slice_lo_mask_2 = 0u; }
                        else if (_lim_4 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_4));
                        }
                    }
                    #pragma unroll
                    for (int _i_5 = 0; _i_5 < 32; _i_5++) {
                        if (!(_slice_lo_mask_2 & (1u << _i_5))) _tmem_load_0[64 + _i_5] = -LOOM_INF;
                    }
                    uint32_t _slice_lo_mask_3;
                    {
                        int _lim_6 = tail_valid - 96;
                        if (_lim_6 <= 0) { _slice_lo_mask_3 = 0u; }
                        else if (_lim_6 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_6));
                        }
                    }
                    #pragma unroll
                    for (int _i_7 = 0; _i_7 < 32; _i_7++) {
                        if (!(_slice_lo_mask_3 & (1u << _i_7))) _tmem_load_0[96 + _i_7] = -LOOM_INF;
                    }
                    int valid_count = causal_row - n_block * BLOCK_N + 1;
                    uint32_t _slice_lo_mask_4;
                    {
                        int _lim_8 = valid_count;
                        if (_lim_8 <= 0) { _slice_lo_mask_4 = 0u; }
                        else if (_lim_8 >= 32) { _slice_lo_mask_4 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_4) : "r"(_lim_8));
                        }
                    }
                    #pragma unroll
                    for (int _i_9 = 0; _i_9 < 32; _i_9++) {
                        if (!(_slice_lo_mask_4 & (1u << _i_9))) _tmem_load_0[0 + _i_9] = -LOOM_INF;
                    }
                    uint32_t _slice_lo_mask_5;
                    {
                        int _lim_10 = valid_count - 32;
                        if (_lim_10 <= 0) { _slice_lo_mask_5 = 0u; }
                        else if (_lim_10 >= 32) { _slice_lo_mask_5 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_5) : "r"(_lim_10));
                        }
                    }
                    #pragma unroll
                    for (int _i_11 = 0; _i_11 < 32; _i_11++) {
                        if (!(_slice_lo_mask_5 & (1u << _i_11))) _tmem_load_0[32 + _i_11] = -LOOM_INF;
                    }
                    uint32_t _slice_lo_mask_6;
                    {
                        int _lim_12 = valid_count - 64;
                        if (_lim_12 <= 0) { _slice_lo_mask_6 = 0u; }
                        else if (_lim_12 >= 32) { _slice_lo_mask_6 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_6) : "r"(_lim_12));
                        }
                    }
                    #pragma unroll
                    for (int _i_13 = 0; _i_13 < 32; _i_13++) {
                        if (!(_slice_lo_mask_6 & (1u << _i_13))) _tmem_load_0[64 + _i_13] = -LOOM_INF;
                    }
                    uint32_t _slice_lo_mask_7;
                    {
                        int _lim_14 = valid_count - 96;
                        if (_lim_14 <= 0) { _slice_lo_mask_7 = 0u; }
                        else if (_lim_14 >= 32) { _slice_lo_mask_7 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_7) : "r"(_lim_14));
                        }
                    }
                    #pragma unroll
                    for (int _i_15 = 0; _i_15 < 32; _i_15++) {
                        if (!(_slice_lo_mask_7 & (1u << _i_15))) _tmem_load_0[96 + _i_15] = -LOOM_INF;
                    }
                    float2 _reg_reduce_max2_16 = {-LOOM_INF, -LOOM_INF};
                    row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_16);
                    row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_16);
                    row_max_x32_accum(&_tmem_load_0[64], _reg_reduce_max2_16);
                    row_max_x32_accum(&_tmem_load_0[96], _reg_reduce_max2_16);
                    float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_16);
                    float new_max = _tmem_load_0_max;
                    float _max_0 = max_noftz(new_max, row_max_val);
                    new_max = _max_0;
                    float new_max_scaled = ((new_max == -LOOM_INF) ? 0.0f : new_max) * softmax_scale_log2;
                    float acc_scale;
                    float selected_max;
                    float _fma_0 = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max_scaled);
                    if (_fma_0 >= -8.0f) {
                        selected_max = row_max_val;
                        acc_scale = 1.0f;
                        new_max_scaled = ((row_max_val == -LOOM_INF) ? 0.0f : row_max_val) * softmax_scale_log2;
                    } else {
                        selected_max = new_max;
                        if (row_max_val > -LOOM_INF) {
                            float _exp2_0 = approx_exp2(_fma_0);
                            acc_scale = _exp2_0;
                        } else {
                            acc_scale = 1.0f;
                        }
                    }
                    row_max_val = selected_max;
                    sScale[warp % 4 * 32 + lane] = acc_scale;
                    mbarrier_arrive(corr_sig_addr);
                    const float2 _fma_b2_17 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_18 = {-new_max_scaled, -new_max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_17, _fma_c2_18);
                    int p_base = taddr + 64 + (unsigned int)(warp % 4 * 32 << 16);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                    }
                    uint32_t _tmem_load_0_f16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_f16[_lp] = *(uint32_t*)&_h2;
                    }
                    tmem_st_x16(p_base, _tmem_load_0_f16);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_0[_le + 32] = approx_exp2(_tmem_load_0[_le + 32]);
                    }
                    uint32_t _tmem_load_0_f16_0[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 32], _tmem_load_0[_lp*2+1 + 32]));
                        _tmem_load_0_f16_0[_lp] = *(uint32_t*)&_h2;
                    }
                    tmem_st_x16(p_base + 16, _tmem_load_0_f16_0);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_0[_le + 64] = approx_exp2(_tmem_load_0[_le + 64]);
                    }
                    uint32_t _tmem_load_0_f16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 64], _tmem_load_0[_lp*2+1 + 64]));
                        _tmem_load_0_f16_1[_lp] = *(uint32_t*)&_h2;
                    }
                    tmem_st_x16(p_base + 32, _tmem_load_0_f16_1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_0[_le + 96] = approx_exp2(_tmem_load_0[_le + 96]);
                    }
                    uint32_t _tmem_load_0_f16_2[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 96], _tmem_load_0[_lp*2+1 + 96]));
                        _tmem_load_0_f16_2[_lp] = *(uint32_t*)&_h2;
                    }
                    tmem_st_x16(p_base + 48, _tmem_load_0_f16_2);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_2_addr);
                    mbarrier_wait(corr_done_addr, _phase_corr_done_0);
                    _phase_corr_done_0 ^= 1;
                    float2 _reg_reduce_sum2_19 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_19);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_19);
                    softmax_block_sum(&_tmem_load_0[64], &_reg_reduce_sum2_19);
                    softmax_block_sum(&_tmem_load_0[96], &_reg_reduce_sum2_19);
                    float _tmem_load_0_sum = _reg_reduce_sum2_19.x + _reg_reduce_sum2_19.y;
                    row_sum_val = row_sum_val * acc_scale + _tmem_load_0_sum;
                }
                #pragma unroll 1
                for (unsigned int n_iter_1 = num_masked_iters; n_iter_1 < num_n_blocks; n_iter_1++) {
                    int n_block_1 = num_n_blocks - 1 - n_iter_1;
                    mbarrier_wait(s_full_addr, _phase_s_full_0);
                    _phase_s_full_0 ^= 1;
                    int s_base_1 = taddr + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_1[128];
                    tmem_ld_x32(&_tmem_load_1[0], s_base_1);
                    tmem_ld_x32(&_tmem_load_1[32], s_base_1 + 32);
                    tmem_ld_x32(&_tmem_load_1[64], s_base_1 + 64);
                    tmem_ld_x32(&_tmem_load_1[96], s_base_1 + 96);
                    int tail_valid_1 = seqlen_kv_bh - n_block_1 * BLOCK_N;
                    uint32_t _slice_lo_mask_8;
                    {
                        int _lim_20 = tail_valid_1;
                        if (_lim_20 <= 0) { _slice_lo_mask_8 = 0u; }
                        else if (_lim_20 >= 32) { _slice_lo_mask_8 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_8) : "r"(_lim_20));
                        }
                    }
                    #pragma unroll
                    for (int _i_21 = 0; _i_21 < 32; _i_21++) {
                        if (!(_slice_lo_mask_8 & (1u << _i_21))) _tmem_load_1[0 + _i_21] = -LOOM_INF;
                    }
                    uint32_t _slice_lo_mask_9;
                    {
                        int _lim_22 = tail_valid_1 - 32;
                        if (_lim_22 <= 0) { _slice_lo_mask_9 = 0u; }
                        else if (_lim_22 >= 32) { _slice_lo_mask_9 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_9) : "r"(_lim_22));
                        }
                    }
                    #pragma unroll
                    for (int _i_23 = 0; _i_23 < 32; _i_23++) {
                        if (!(_slice_lo_mask_9 & (1u << _i_23))) _tmem_load_1[32 + _i_23] = -LOOM_INF;
                    }
                    uint32_t _slice_lo_mask_10;
                    {
                        int _lim_24 = tail_valid_1 - 64;
                        if (_lim_24 <= 0) { _slice_lo_mask_10 = 0u; }
                        else if (_lim_24 >= 32) { _slice_lo_mask_10 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_10) : "r"(_lim_24));
                        }
                    }
                    #pragma unroll
                    for (int _i_25 = 0; _i_25 < 32; _i_25++) {
                        if (!(_slice_lo_mask_10 & (1u << _i_25))) _tmem_load_1[64 + _i_25] = -LOOM_INF;
                    }
                    uint32_t _slice_lo_mask_11;
                    {
                        int _lim_26 = tail_valid_1 - 96;
                        if (_lim_26 <= 0) { _slice_lo_mask_11 = 0u; }
                        else if (_lim_26 >= 32) { _slice_lo_mask_11 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_11) : "r"(_lim_26));
                        }
                    }
                    #pragma unroll
                    for (int _i_27 = 0; _i_27 < 32; _i_27++) {
                        if (!(_slice_lo_mask_11 & (1u << _i_27))) _tmem_load_1[96 + _i_27] = -LOOM_INF;
                    }
                    float2 _reg_reduce_max2_28 = {-LOOM_INF, -LOOM_INF};
                    row_max_x32_accum(&_tmem_load_1[0], _reg_reduce_max2_28);
                    row_max_x32_accum(&_tmem_load_1[32], _reg_reduce_max2_28);
                    row_max_x32_accum(&_tmem_load_1[64], _reg_reduce_max2_28);
                    row_max_x32_accum(&_tmem_load_1[96], _reg_reduce_max2_28);
                    float _tmem_load_1_max = row_max_reduce(_reg_reduce_max2_28);
                    float new_max_1 = _tmem_load_1_max;
                    float _max_1 = max_noftz(new_max_1, row_max_val);
                    new_max_1 = _max_1;
                    float new_max_scaled_1 = ((new_max_1 == -LOOM_INF) ? 0.0f : new_max_1) * softmax_scale_log2;
                    float acc_scale_1;
                    float selected_max_1;
                    float _fma_1 = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max_scaled_1);
                    if (_fma_1 >= -8.0f) {
                        selected_max_1 = row_max_val;
                        acc_scale_1 = 1.0f;
                        new_max_scaled_1 = ((row_max_val == -LOOM_INF) ? 0.0f : row_max_val) * softmax_scale_log2;
                    } else {
                        selected_max_1 = new_max_1;
                        if (row_max_val > -LOOM_INF) {
                            float _exp2_1 = approx_exp2(_fma_1);
                            acc_scale_1 = _exp2_1;
                        } else {
                            acc_scale_1 = 1.0f;
                        }
                    }
                    row_max_val = selected_max_1;
                    sScale[warp % 4 * 32 + lane] = acc_scale_1;
                    mbarrier_arrive(corr_sig_addr);
                    const float2 _fma_b2_29 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_30 = {-new_max_scaled_1, -new_max_scaled_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_lf], _fma_b2_29, _fma_c2_30);
                    int p_base_1 = taddr + 64 + (unsigned int)(warp % 4 * 32 << 16);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_1[_le] = approx_exp2(_tmem_load_1[_le]);
                    }
                    uint32_t _tmem_load_1_f16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                        _tmem_load_1_f16[_lp] = *(uint32_t*)&_h2;
                    }
                    tmem_st_x16(p_base_1, _tmem_load_1_f16);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_1[_le + 32] = approx_exp2(_tmem_load_1[_le + 32]);
                    }
                    uint32_t _tmem_load_1_f16_0[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_1[_lp*2 + 32], _tmem_load_1[_lp*2+1 + 32]));
                        _tmem_load_1_f16_0[_lp] = *(uint32_t*)&_h2;
                    }
                    tmem_st_x16(p_base_1 + 16, _tmem_load_1_f16_0);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_1[_le + 64] = approx_exp2(_tmem_load_1[_le + 64]);
                    }
                    uint32_t _tmem_load_1_f16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_1[_lp*2 + 64], _tmem_load_1[_lp*2+1 + 64]));
                        _tmem_load_1_f16_1[_lp] = *(uint32_t*)&_h2;
                    }
                    tmem_st_x16(p_base_1 + 32, _tmem_load_1_f16_1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_1[_le + 96] = approx_exp2(_tmem_load_1[_le + 96]);
                    }
                    uint32_t _tmem_load_1_f16_2[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_1[_lp*2 + 96], _tmem_load_1[_lp*2+1 + 96]));
                        _tmem_load_1_f16_2[_lp] = *(uint32_t*)&_h2;
                    }
                    tmem_st_x16(p_base_1 + 48, _tmem_load_1_f16_2);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_2_addr);
                    mbarrier_wait(corr_done_addr, _phase_corr_done_0);
                    _phase_corr_done_0 ^= 1;
                    float2 _reg_reduce_sum2_31 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_1[0], &_reg_reduce_sum2_31);
                    softmax_block_sum(&_tmem_load_1[32], &_reg_reduce_sum2_31);
                    softmax_block_sum(&_tmem_load_1[64], &_reg_reduce_sum2_31);
                    softmax_block_sum(&_tmem_load_1[96], &_reg_reduce_sum2_31);
                    float _tmem_load_1_sum = _reg_reduce_sum2_31.x + _reg_reduce_sum2_31.y;
                    row_sum_val = row_sum_val * acc_scale_1 + _tmem_load_1_sum;
                }
                sScale[warp % 4 * 32 + lane + 2 * BLOCK_M] = row_sum_val;
                sScale[warp % 4 * 32 + lane + 4 * BLOCK_M] = row_max_val;
                mbarrier_arrive(corr_sig_addr);
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // correction_main
            unsigned int total_tiles_1 = NUM_M_BLOCKS * total_bh;
            unsigned int _phase_work_id_full_0_1 = 0;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_iter_1 = 0; tile_iter_1 < total_tiles_1; tile_iter_1++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_1);
                _phase_work_id_full_0_1 ^= 1;
                unsigned int tile_idx_1 = work_id_slot[0];
                if (elect_sync()) {
                    mbarrier_arrive(work_id_empty_addr);
                }
                if (tile_idx_1 >= total_tiles_1) {
                    break;
                }
                unsigned int m_block_1;
                unsigned int bh_1;
                {
                    m_block_1 = tile_idx_1 % (unsigned int)NUM_M_BLOCKS;
                    bh_1 = tile_idx_1 / (unsigned int)NUM_M_BLOCKS;
                }
                int seqlen_kv_bh_1 = seq_lens_kv[bh_1];
                unsigned int num_n_blocks_1 = (seqlen_kv_bh_1 + BLOCK_N - 1) / BLOCK_N;
                int off_q = (unsigned int)cu_seq_lens_q[bh_1] + m_block_1 * (unsigned int)BLOCK_M;
                int seqlen_q_bh = seq_lens_q[bh_1];
                mbarrier_arrive(p_full_addr);
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                #pragma unroll 1
                for (unsigned int n_iter_2 = 1; n_iter_2 < num_n_blocks_1; n_iter_2++) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    float scale = sScale[warp % 4 * 32 + lane];
                    int _vote_0 = __any_sync(0xFFFFFFFF, scale < 1.0f);
                    if (_vote_0 != 0) {
                        #pragma unroll
                        for (int cr_col = 0; cr_col < HEAD_DIM_HALF / 16; cr_col++) {
                            int cr_addr_hi = taddr + (unsigned int)TMEM_OUTPUT_HI_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(cr_col * 16);
                            float _tmem_load_2[16];
                            tmem_ld_x16(&_tmem_load_2[0], cr_addr_hi);
                            const float2 _scale2_0 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_0);
                            tmem_st_x16_f32(cr_addr_hi, _tmem_load_2);
                            int cr_addr_lo = taddr + (unsigned int)TMEM_OUTPUT_LO_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(cr_col * 16);
                            float _tmem_load_3[16];
                            tmem_ld_x16(&_tmem_load_3[0], cr_addr_lo);
                            const float2 _scale2_1 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_1);
                            tmem_st_x16_f32(cr_addr_lo, _tmem_load_3);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr);
                    mbarrier_arrive(corr_done_addr);
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                float final_sum = sScale[warp % 4 * 32 + lane + 2 * BLOCK_M];
                float final_max = sScale[warp % 4 * 32 + lane + 4 * BLOCK_M];
                float final_scale;
                if (final_sum != 0.0f && final_sum == final_sum) {
                    float _rcp_0 = approx_rcp(final_sum);
                    final_scale = _rcp_0;
                } else {
                    final_scale = 0.0f;
                }
                int q_off = m_block_1 * (unsigned int)BLOCK_M;
                if (q_off < seqlen_q_bh) {
                    #pragma unroll
                    for (int ce_col = 0; ce_col < HEAD_DIM_HALF / 8; ce_col++) {
                        int ce_addr = taddr + (unsigned int)TMEM_OUTPUT_HI_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(ce_col * 8);
                        float _tmem_load_4[8];
                        tmem_ld_x8(&_tmem_load_4[0], ce_addr);
                        int o_elem = (off_q + (warp % 4 * 32 + lane)) * HEAD_DIM + ce_col * 8;
                        {
                            const float2 _prescale2_2 = {final_scale, final_scale};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 4; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[0])[_ps], _prescale2_2);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                _tmem_load_4[0 + _ps] *= final_scale;
                            #endif
                            __half2 _pk[4];
                            _pk[0] = __floats2half2_rn(_tmem_load_4[0 + 0], _tmem_load_4[0 + 1]);
                            _pk[1] = __floats2half2_rn(_tmem_load_4[0 + 2], _tmem_load_4[0 + 3]);
                            _pk[2] = __floats2half2_rn(_tmem_load_4[0 + 4], _tmem_load_4[0 + 5]);
                            _pk[3] = __floats2half2_rn(_tmem_load_4[0 + 6], _tmem_load_4[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__half*)(O_ptr + o_elem))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                    #pragma unroll
                    for (int ce_col_1 = 0; ce_col_1 < HEAD_DIM_HALF / 8; ce_col_1++) {
                        int ce_addr_1 = taddr + (unsigned int)TMEM_OUTPUT_LO_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(ce_col_1 * 8);
                        float _tmem_load_5[8];
                        tmem_ld_x8(&_tmem_load_5[0], ce_addr_1);
                        int o_elem_1 = (off_q + (warp % 4 * 32 + lane)) * HEAD_DIM + HEAD_DIM_HALF + ce_col_1 * 8;
                        {
                            const float2 _prescale2_3 = {final_scale, final_scale};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 4; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_5[0])[_ps], _prescale2_3);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                _tmem_load_5[0 + _ps] *= final_scale;
                            #endif
                            __half2 _pk[4];
                            _pk[0] = __floats2half2_rn(_tmem_load_5[0 + 0], _tmem_load_5[0 + 1]);
                            _pk[1] = __floats2half2_rn(_tmem_load_5[0 + 2], _tmem_load_5[0 + 3]);
                            _pk[2] = __floats2half2_rn(_tmem_load_5[0 + 4], _tmem_load_5[0 + 5]);
                            _pk[3] = __floats2half2_rn(_tmem_load_5[0 + 6], _tmem_load_5[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__half*)(O_ptr + o_elem_1))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            unsigned int total_tiles_2 = NUM_M_BLOCKS * total_bh;
            unsigned int mma_kv_stage = 0;
            unsigned int mma_kv_phase = 0;
            unsigned int _phase_work_id_full_0_2 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_2_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_iter_2 = 0; tile_iter_2 < total_tiles_2; tile_iter_2++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_2);
                _phase_work_id_full_0_2 ^= 1;
                unsigned int tile_idx_2 = work_id_slot[0];
                if (elect_sync()) {
                    mbarrier_arrive(work_id_empty_addr);
                }
                if (tile_idx_2 >= total_tiles_2) {
                    break;
                }
                unsigned int m_block_2;
                unsigned int bh_2;
                {
                    m_block_2 = tile_idx_2 % (unsigned int)NUM_M_BLOCKS;
                    bh_2 = tile_idx_2 / (unsigned int)NUM_M_BLOCKS;
                }
                int seqlen_kv_bh_2 = seq_lens_kv[bh_2];
                unsigned int num_n_blocks_2 = (seqlen_kv_bh_2 + BLOCK_N - 1) / BLOCK_N;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase);
                int _mma_a_lo_0 = make_warp_uniform(((smem_q_hi_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((((smem_kv_hi_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores_0), "r"(0));
                int _mma_a_lo_1 = make_warp_uniform(((smem_q_lo_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_1 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_scores_0), "r"(1));
                elect_commit(s_full_addr);
                elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                mma_kv_stage += 1;
                if (mma_kv_stage == 2) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                unsigned int first_pv = 1;
                #pragma unroll 1
                for (unsigned int n_iter_3 = 0; n_iter_3 < num_n_blocks_2 - 1; n_iter_3++) {
                    unsigned int v_stage = mma_kv_stage;
                    unsigned int v_phase = mma_kv_phase;
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 2) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
                    int first_pv_flag = first_pv;
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    int _mma_b_lo_2 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_hi), "r"(_mma_b_lo_2), "r"(tmem_softmax_0), "r"(((first_pv_flag) ? 0 : 1)));
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_lo), "r"(_mma_b_lo_3), "r"(tmem_softmax_0), "r"(((first_pv_flag) ? 0 : 1)));
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                    _phase_p_full_2_0 ^= 1;
                    int _mma_b_lo_4 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_hi), "r"(_mma_b_lo_4), "r"(tmem_softmax_0), "r"(1));
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_lo), "r"(_mma_b_lo_5), "r"(tmem_softmax_0), "r"(1));
                    unsigned int k_stage = mma_kv_stage;
                    unsigned int k_phase = mma_kv_phase;
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 2) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
                    int _mma_a_lo_6 = make_warp_uniform(((smem_q_hi_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_6 = make_warp_uniform((((smem_kv_hi_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
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
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_scores_0), "r"(0));
                    int _mma_a_lo_7 = make_warp_uniform(((smem_q_lo_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
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
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_scores_0), "r"(1));
                    elect_commit(s_full_addr);
                    elect_commit(kv_empty_addr + (v_stage) * 8);
                    elect_commit(kv_empty_addr + (k_stage) * 8);
                    first_pv = 0;
                }
                elect_commit(q_empty_addr);
                mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase);
                int first_pv_flag_1 = first_pv;
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                int _mma_b_lo_8 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 2048);
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
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_hi), "r"(_mma_b_lo_8), "r"(tmem_softmax_0), "r"(((first_pv_flag_1) ? 0 : 1)));
                int _mma_b_lo_9 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 2048);
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
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_lo), "r"(_mma_b_lo_9), "r"(tmem_softmax_0), "r"(((first_pv_flag_1) ? 0 : 1)));
                mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                _phase_p_full_2_0 ^= 1;
                int _mma_b_lo_10 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 2048);
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
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_hi), "r"(_mma_b_lo_10), "r"(tmem_softmax_0), "r"(1));
                int _mma_b_lo_11 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 2048);
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
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_lo), "r"(_mma_b_lo_11), "r"(tmem_softmax_0), "r"(1));
                elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                mma_kv_stage += 1;
                if (mma_kv_stage == 2) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                elect_commit(o_full_addr);
            }
        }
    }
    // ---- Role: page_offsets ----
    if (warp == 9) {
        { // page_offsets_main
            unsigned int total_tiles_3 = NUM_M_BLOCKS * total_bh;
            unsigned int po_stage = 0;
            unsigned int _phase_work_id_full_0_3 = 0;
            unsigned int _phase_page_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_iter_3 = 0; tile_iter_3 < total_tiles_3; tile_iter_3++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_3);
                _phase_work_id_full_0_3 ^= 1;
                unsigned int tile_idx_3 = work_id_slot[0];
                if (elect_sync()) {
                    mbarrier_arrive(work_id_empty_addr);
                }
                if (tile_idx_3 >= total_tiles_3) {
                    break;
                }
                unsigned int m_block_3;
                unsigned int bh_3;
                {
                    m_block_3 = tile_idx_3 % (unsigned int)NUM_M_BLOCKS;
                    bh_3 = tile_idx_3 / (unsigned int)NUM_M_BLOCKS;
                }
                int seqlen_kv_bh_3 = seq_lens_kv[bh_3];
                unsigned int num_n_blocks_3 = (seqlen_kv_bh_3 + BLOCK_N - 1) / BLOCK_N;
                unsigned int kv_bh = bh_3 / (unsigned int)HEADS_PER_GROUP;
                int pt_base = kv_bh * (unsigned int)max_pages_per_seq;
                #pragma unroll 1
                for (unsigned int ni = 0; ni < num_n_blocks_3; ni++) {
                    unsigned int n = num_n_blocks_3 - 1 - ni;
                    int pt_off = (unsigned int)pt_base + n * 8;
                    mbarrier_wait(page_empty_addr + (po_stage) * 8, _phase_page_empty);
                    if (elect_sync()) {
                        int pg_idx = po_stage * 8;
                        smem_pages[pg_idx] = page_table[pt_off];
                        smem_pages[pg_idx + 1] = page_table[pt_off + 1];
                        smem_pages[pg_idx + 2] = page_table[pt_off + 2];
                        smem_pages[pg_idx + 3] = page_table[pt_off + 3];
                        smem_pages[pg_idx + 4] = page_table[pt_off + 4];
                        smem_pages[pg_idx + 5] = page_table[pt_off + 5];
                        smem_pages[pg_idx + 6] = page_table[pt_off + 6];
                        smem_pages[pg_idx + 7] = page_table[pt_off + 7];
                        mbarrier_arrive(page_full_addr + (po_stage) * 8);
                    }
                    po_stage += 1;
                    if (po_stage == 6) { po_stage = 0; _phase_page_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        { // scheduler_main
            unsigned int total_tiles_4 = NUM_M_BLOCKS * total_bh;
            unsigned int _phase_work_id_empty_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int tile_iter_4 = 0; tile_iter_4 < total_tiles_4; tile_iter_4++) {
                    mbarrier_wait(work_id_empty_addr, _phase_work_id_empty_0);
                    _phase_work_id_empty_0 ^= 1;
                    unsigned int _atomic_old_0 = atomicAdd(dynamic_counter, 1);
                    work_id_slot[0] = _atomic_old_0;
                    mbarrier_arrive(work_id_full_addr);
                    if (_atomic_old_0 >= total_tiles_4) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 11) {
        { // load_main
            unsigned int total_tiles_5 = NUM_M_BLOCKS * total_bh;
            unsigned int load_kv_stage = 0;
            unsigned int load_po_stage = 0;
            unsigned int _phase_work_id_full_0_4 = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_page_full = 0;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_iter_5 = 0; tile_iter_5 < total_tiles_5; tile_iter_5++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_4);
                _phase_work_id_full_0_4 ^= 1;
                unsigned int tile_idx_4 = work_id_slot[0];
                if (elect_sync()) {
                    mbarrier_arrive(work_id_empty_addr);
                }
                if (tile_idx_4 >= total_tiles_5) {
                    break;
                }
                unsigned int m_block_4;
                unsigned int bh_4;
                {
                    m_block_4 = tile_idx_4 % (unsigned int)NUM_M_BLOCKS;
                    bh_4 = tile_idx_4 / (unsigned int)NUM_M_BLOCKS;
                }
                int seqlen_kv_bh_4 = seq_lens_kv[bh_4];
                unsigned int num_n_blocks_4 = (seqlen_kv_bh_4 + BLOCK_N - 1) / BLOCK_N;
                int off_q_1 = (unsigned int)cu_seq_lens_q[bh_4] + m_block_4 * (unsigned int)BLOCK_M;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 65536);
                    tma_3d_gmem2smem(smem_q_hi_addr, Q, 0, off_q_1, 0, q_full_addr);
                    tma_3d_gmem2smem(smem_q_lo_addr, Q, 0, off_q_1, 2, q_full_addr);
                }
                #pragma unroll 1
                for (unsigned int ni_1 = 0; ni_1 < num_n_blocks_4; ni_1++) {
                    unsigned int n_1 = num_n_blocks_4 - 1 - ni_1;
                    mbarrier_wait(page_full_addr + (load_po_stage) * 8, _phase_page_full);
                    int pg_idx_1 = load_po_stage * 8;
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 65536);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768, K, 0, 0, 0, smem_pages[pg_idx_1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 16384, K, 0, 0, 1, smem_pages[pg_idx_1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768, K, 0, 0, 2, smem_pages[pg_idx_1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 16384, K, 0, 0, 3, smem_pages[pg_idx_1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 2048, K, 0, 0, 0, smem_pages[pg_idx_1 + 1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 18432, K, 0, 0, 1, smem_pages[pg_idx_1 + 1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 2048, K, 0, 0, 2, smem_pages[pg_idx_1 + 1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 18432, K, 0, 0, 3, smem_pages[pg_idx_1 + 1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 4096, K, 0, 0, 0, smem_pages[pg_idx_1 + 2], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 20480, K, 0, 0, 1, smem_pages[pg_idx_1 + 2], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 4096, K, 0, 0, 2, smem_pages[pg_idx_1 + 2], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 20480, K, 0, 0, 3, smem_pages[pg_idx_1 + 2], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 6144, K, 0, 0, 0, smem_pages[pg_idx_1 + 3], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 22528, K, 0, 0, 1, smem_pages[pg_idx_1 + 3], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 6144, K, 0, 0, 2, smem_pages[pg_idx_1 + 3], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 22528, K, 0, 0, 3, smem_pages[pg_idx_1 + 3], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 8192, K, 0, 0, 0, smem_pages[pg_idx_1 + 4], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 24576, K, 0, 0, 1, smem_pages[pg_idx_1 + 4], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 8192, K, 0, 0, 2, smem_pages[pg_idx_1 + 4], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 24576, K, 0, 0, 3, smem_pages[pg_idx_1 + 4], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 10240, K, 0, 0, 0, smem_pages[pg_idx_1 + 5], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 26624, K, 0, 0, 1, smem_pages[pg_idx_1 + 5], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 10240, K, 0, 0, 2, smem_pages[pg_idx_1 + 5], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 26624, K, 0, 0, 3, smem_pages[pg_idx_1 + 5], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 12288, K, 0, 0, 0, smem_pages[pg_idx_1 + 6], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 28672, K, 0, 0, 1, smem_pages[pg_idx_1 + 6], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 12288, K, 0, 0, 2, smem_pages[pg_idx_1 + 6], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 28672, K, 0, 0, 3, smem_pages[pg_idx_1 + 6], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 14336, K, 0, 0, 0, smem_pages[pg_idx_1 + 7], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 30720, K, 0, 0, 1, smem_pages[pg_idx_1 + 7], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 14336, K, 0, 0, 2, smem_pages[pg_idx_1 + 7], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 30720, K, 0, 0, 3, smem_pages[pg_idx_1 + 7], kv_full_addr + (load_kv_stage) * 8);
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 2) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 65536);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768, V, 0, 0, 0, smem_pages[pg_idx_1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 16384, V, 0, 0, 1, smem_pages[pg_idx_1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768, V, 0, 0, 2, smem_pages[pg_idx_1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 16384, V, 0, 0, 3, smem_pages[pg_idx_1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 2048, V, 0, 0, 0, smem_pages[pg_idx_1 + 1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 18432, V, 0, 0, 1, smem_pages[pg_idx_1 + 1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 2048, V, 0, 0, 2, smem_pages[pg_idx_1 + 1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 18432, V, 0, 0, 3, smem_pages[pg_idx_1 + 1], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 4096, V, 0, 0, 0, smem_pages[pg_idx_1 + 2], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 20480, V, 0, 0, 1, smem_pages[pg_idx_1 + 2], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 4096, V, 0, 0, 2, smem_pages[pg_idx_1 + 2], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 20480, V, 0, 0, 3, smem_pages[pg_idx_1 + 2], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 6144, V, 0, 0, 0, smem_pages[pg_idx_1 + 3], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 22528, V, 0, 0, 1, smem_pages[pg_idx_1 + 3], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 6144, V, 0, 0, 2, smem_pages[pg_idx_1 + 3], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 22528, V, 0, 0, 3, smem_pages[pg_idx_1 + 3], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 8192, V, 0, 0, 0, smem_pages[pg_idx_1 + 4], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 24576, V, 0, 0, 1, smem_pages[pg_idx_1 + 4], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 8192, V, 0, 0, 2, smem_pages[pg_idx_1 + 4], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 24576, V, 0, 0, 3, smem_pages[pg_idx_1 + 4], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 10240, V, 0, 0, 0, smem_pages[pg_idx_1 + 5], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 26624, V, 0, 0, 1, smem_pages[pg_idx_1 + 5], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 10240, V, 0, 0, 2, smem_pages[pg_idx_1 + 5], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 26624, V, 0, 0, 3, smem_pages[pg_idx_1 + 5], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 12288, V, 0, 0, 0, smem_pages[pg_idx_1 + 6], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 28672, V, 0, 0, 1, smem_pages[pg_idx_1 + 6], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 12288, V, 0, 0, 2, smem_pages[pg_idx_1 + 6], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 28672, V, 0, 0, 3, smem_pages[pg_idx_1 + 6], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 14336, V, 0, 0, 0, smem_pages[pg_idx_1 + 7], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_hi_addr + load_kv_stage * 32768 + 30720, V, 0, 0, 1, smem_pages[pg_idx_1 + 7], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 14336, V, 0, 0, 2, smem_pages[pg_idx_1 + 7], kv_full_addr + (load_kv_stage) * 8);
                        tma_4d_gmem2smem(smem_kv_lo_addr + load_kv_stage * 32768 + 30720, V, 0, 0, 3, smem_pages[pg_idx_1 + 7], kv_full_addr + (load_kv_stage) * 8);
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 2) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    if (elect_sync()) {
                        mbarrier_arrive(page_empty_addr + (load_po_stage) * 8);
                    }
                    load_po_stage += 1;
                    if (load_po_stage == 6) { load_po_stage = 0; _phase_page_full ^= 1; }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
