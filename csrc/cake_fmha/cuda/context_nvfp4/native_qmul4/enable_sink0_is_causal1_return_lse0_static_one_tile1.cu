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
#define TMEM_NCOLS 512
#define TMEM_SCORES_0_OFFSET 0
#define TMEM_SOFTMAX_0_OFFSET 64
#define TMEM_SCORES_1_OFFSET 128
#define TMEM_SOFTMAX_1_OFFSET 192
#define TMEM_OUTPUT_0_OFFSET 256
#define TMEM_OUTPUT_1_OFFSET 384
#define NUM_Q_STAGES 2
#define NUM_KV_STAGES 3
#define NUM_PK_STAGES 2
#define NUM_PO_STAGES 6
#define NUM_ACC_STAGES 1
#define NUM_ORDER_P01_STAGES 1
#define SMEM_SSCALE_OFF 1024
#define SMEM_SSCALE_STAGE_BYTES 3072
#define SMEM_SSCALE_STRIDE 3072
#define SMEM_SMEM_Q0_OFF 4096
#define SMEM_SMEM_Q0_STAGE_BYTES 16384
#define SMEM_SMEM_Q0_STRIDE 16384
#define SMEM_SMEM_Q1_OFF 20480
#define SMEM_SMEM_Q1_STAGE_BYTES 16384
#define SMEM_SMEM_Q1_STRIDE 16384
#define SMEM_SMEM_KV_OFF 36864
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 36864
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_PAGES_OFF 86016
#define SMEM_SMEM_PAGES_STAGE_BYTES 384
#define SMEM_SMEM_PAGES_STRIDE 384
#define SMEM_WORK_ID_SLOT_OFF 86400
#define SMEM_WORK_ID_SLOT_STAGE_BYTES 4
#define SMEM_WORK_ID_SLOT_STRIDE 4
#define SMEM_SMEM_PK_OFF 86528
#define SMEM_SMEM_PK_STAGE_BYTES 9216
#define SMEM_SMEM_PK_STRIDE 9216
#define SMEM_TOTAL 104960
#define IS_CAUSAL 1
#ifndef NUM_M_BLOCKS
#define NUM_M_BLOCKS 2
#endif
#ifndef NUM_Q_HEADS
#define NUM_Q_HEADS 8
#endif
#ifndef HEADS_PER_GROUP
#define HEADS_PER_GROUP 4
#endif
#ifndef PACK_G
#define PACK_G 1
#endif
#ifndef TOK_PER_STAGE
#define TOK_PER_STAGE 128
#endif
#ifndef L2_SWIZZLE
#define L2_SWIZZLE 8
#endif
#define STATIC_ONE_TILE 1
#define CHUNK_MAJOR_DEQUANT 0
#define RETURN_LSE 0
#define ENABLE_SINK 0
#define UNIFORM_Q_LEN -1
#define UNIFORM_KV_LEN -1
#define SINGLE_MASK_LOOP 0
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128
#define MMA_K 32
#ifndef PAGE_SIZE
#define PAGE_SIZE 16
#endif

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


__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
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


__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_cake_fmha_context_nvfp4(CakeFmhaTensorMap const* Q, CakeFmhaTensorMap const* Kp, CakeFmhaTensorMap const* Vp, CakeFmhaTensorMap const* Ksf, CakeFmhaTensorMap const* Vsf, uint8_t* __restrict__ O_ptr, float* __restrict__ LSE_ptr, float* __restrict__ sinks, int* __restrict__ page_table_k, int* __restrict__ page_table_v, int* __restrict__ seq_lens_q, int* __restrict__ seq_lens_kv, int* __restrict__ cu_seq_lens_q, float softmax_scale_log2, float output_scale, int total_bh, int page_row_stride, int num_ctas, unsigned int* __restrict__ dynamic_counter)
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
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Kp)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Vp)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Ksf)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Vsf)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* sScale = reinterpret_cast<float*>(smem_raw + 1024);
    const int sScale_addr = smem + 1024;
    uint8_t* smem_q0 = reinterpret_cast<uint8_t*>(smem_raw + 4096);
    const int smem_q0_addr = smem + 4096;
    uint8_t* smem_q1 = reinterpret_cast<uint8_t*>(smem_raw + 20480);
    const int smem_q1_addr = smem + 20480;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_kv_addr = smem + 36864;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_v_addr = smem + 36864;
    int* smem_pages = reinterpret_cast<int*>(smem_raw + 86016);
    const int smem_pages_addr = smem + 86016;
    unsigned int* work_id_slot = reinterpret_cast<unsigned int*>(smem_raw + 86400);
    const int work_id_slot_addr = smem + 86400;
    uint8_t* smem_pk = reinterpret_cast<uint8_t*>(smem_raw + 86528);
    const int smem_pk_addr = smem + 86528;

    // Mbarrier init (18 groups, 41 barriers)
    // Mbarriers at smem_raw[0..328)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // q_full: 2 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        mbarrier_init_pred(smem + 8, 1, leader);
        // --- pipeline 'kv' ---
        // kv_full: 3 barriers, init_count=4
        mbarrier_init_pred(smem + 16, 4, leader);
        mbarrier_init_pred(smem + 24, 4, leader);
        mbarrier_init_pred(smem + 32, 4, leader);
        // kv_empty: 3 barriers, init_count=1
        mbarrier_init_pred(smem + 40, 1, leader);
        mbarrier_init_pred(smem + 48, 1, leader);
        mbarrier_init_pred(smem + 56, 1, leader);
        // --- pipeline 'pk' ---
        // pk_full: 2 barriers, init_count=1
        mbarrier_init_pred(smem + 64, 1, leader);
        mbarrier_init_pred(smem + 72, 1, leader);
        // pk_empty: 2 barriers, init_count=4
        mbarrier_init_pred(smem + 80, 4, leader);
        mbarrier_init_pred(smem + 88, 4, leader);
        // s_full: 2 barriers, init_count=1
        mbarrier_init_pred(smem + 96, 1, leader);
        mbarrier_init_pred(smem + 104, 1, leader);
        // p_full: 2 barriers, init_count=256
        mbarrier_init_pred(smem + 112, 256, leader);
        mbarrier_init_pred(smem + 120, 256, leader);
        // p_full_2: 2 barriers, init_count=128
        mbarrier_init_pred(smem + 128, 128, leader);
        mbarrier_init_pred(smem + 136, 128, leader);
        // corr_sig: 2 barriers, init_count=128
        mbarrier_init_pred(smem + 144, 128, leader);
        mbarrier_init_pred(smem + 152, 128, leader);
        // corr_done: 2 barriers, init_count=128
        mbarrier_init_pred(smem + 160, 128, leader);
        mbarrier_init_pred(smem + 168, 128, leader);
        // order_p01_0: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 176, 128, leader);
        // order_p01_1: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 184, 128, leader);
        // o_full: 2 barriers, init_count=1
        mbarrier_init_pred(smem + 192, 1, leader);
        mbarrier_init_pred(smem + 200, 1, leader);
        // q_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 208, 1, leader);
        // --- pipeline 'po' ---
        // page_full: 6 barriers, init_count=8
        mbarrier_init_pred(smem + 216, 8, leader);
        mbarrier_init_pred(smem + 224, 8, leader);
        mbarrier_init_pred(smem + 232, 8, leader);
        mbarrier_init_pred(smem + 240, 8, leader);
        mbarrier_init_pred(smem + 248, 8, leader);
        mbarrier_init_pred(smem + 256, 8, leader);
        // page_empty: 6 barriers, init_count=1
        mbarrier_init_pred(smem + 264, 1, leader);
        mbarrier_init_pred(smem + 272, 1, leader);
        mbarrier_init_pred(smem + 280, 1, leader);
        mbarrier_init_pred(smem + 288, 1, leader);
        mbarrier_init_pred(smem + 296, 1, leader);
        mbarrier_init_pred(smem + 304, 1, leader);
        // work_id_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 312, 1, leader);
        // work_id_empty: 1 barriers, init_count=15
        mbarrier_init_pred(smem + 320, 15, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 328);
    if (warp == 0) {
        int _tmem_hold = smem + 328;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 40)
    #define pk_full_addr (mbar_base + 64)
    #define pk_empty_addr (mbar_base + 80)
    #define s_full_addr (mbar_base + 96)
    #define p_full_addr (mbar_base + 112)
    #define p_full_2_addr (mbar_base + 128)
    #define corr_sig_addr (mbar_base + 144)
    #define corr_done_addr (mbar_base + 160)
    #define order_p01_0_addr (mbar_base + 176)
    #define order_p01_1_addr (mbar_base + 184)
    #define o_full_addr (mbar_base + 192)
    #define q_empty_addr (mbar_base + 208)
    #define page_full_addr (mbar_base + 216)
    #define page_empty_addr (mbar_base + 264)
    #define work_id_full_addr (mbar_base + 312)
    #define work_id_empty_addr (mbar_base + 320)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores_0 = taddr;
    const int tmem_softmax_0 = taddr + 64;
    const int tmem_scores_1 = taddr + 128;
    const int tmem_softmax_1 = taddr + 192;
    const int tmem_output_0 = taddr + 256;
    const int tmem_output_1 = taddr + 384;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // softmax_main
            unsigned int stage = make_warp_uniform(warp / 4);
            int tmem_s_off = make_warp_uniform(stage * 128);
            int tmem_p_off = make_warp_uniform(stage * 128 + 64);
            int scale_off = make_warp_uniform(stage * (unsigned int)BLOCK_M);
            unsigned int total_tiles = NUM_M_BLOCKS * total_bh;
            unsigned int max_rounds = total_tiles + 1;
            {
                max_rounds = 1;
            }
            int order_p01_phase = ((stage == 0) ? 1 : 0);
            int order_p01_stage = 0;
            unsigned int _phase_work_id_full_0 = 0;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_corr_done = 0;
            #pragma unroll 1
            for (unsigned int tile_iter = 0; tile_iter < max_rounds; tile_iter++) {
                unsigned int tile_idx;
                {
                    tile_idx = blockIdx.x;
                }
                if (tile_idx >= total_tiles) {
                    break;
                }
                unsigned int m_block;
                unsigned int bh;
                {
                    unsigned int l2section = L2_SWIZZLE * NUM_M_BLOCKS;
                    unsigned int section = tile_idx / l2section;
                    unsigned int l2_mod = tile_idx % l2section;
                    m_block = l2_mod / (unsigned int)L2_SWIZZLE;
                    bh = section * (unsigned int)L2_SWIZZLE + l2_mod % (unsigned int)L2_SWIZZLE;
                    m_block = (unsigned int)(NUM_M_BLOCKS - 1) - m_block;
                }
                int seqlen_kv_bh = UNIFORM_KV_LEN;
                int seqlen_q_bh = UNIFORM_Q_LEN;
                {
                    seqlen_kv_bh = seq_lens_kv[bh];
                }
                {
                    seqlen_q_bh = seq_lens_q[bh];
                }
                unsigned int num_n_blocks = (seqlen_kv_bh + BLOCK_N - 1) / BLOCK_N;
                {
                    int kv_shift = seqlen_kv_bh - seqlen_q_bh;
                    unsigned int max_n = (m_block * 2 + 2) * (unsigned int)TOK_PER_STAGE + (unsigned int)kv_shift;
                    if (max_n < (unsigned int)seqlen_kv_bh) {
                        num_n_blocks = (max_n + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                    }
                }
                int causal_row;
                unsigned int num_masked_iters;
                {
                    int kv_shift_1 = seqlen_kv_bh - seqlen_q_bh;
                    causal_row = (m_block * 2 + stage) * (unsigned int)TOK_PER_STAGE + (unsigned int)((warp % 4 * 32 + lane) / PACK_G) + (unsigned int)kv_shift_1;
                    int causal_row_min = (m_block * 2 + stage) * (unsigned int)TOK_PER_STAGE + (unsigned int)kv_shift_1;
                    unsigned int n_block_no_mask_limit = (causal_row_min + 1) / BLOCK_N;
                    num_masked_iters = num_n_blocks - n_block_no_mask_limit;
                    if (num_masked_iters > num_n_blocks) {
                        num_masked_iters = num_n_blocks;
                    }
                }
                float row_max_val = -LOOM_INF;
                float row_sum_val = 0.0f;
                unsigned int masked_loop_end = num_masked_iters;
                unsigned int unmasked_loop_start = num_masked_iters;
                #pragma unroll 1
                for (unsigned int n_iter = 0; n_iter < masked_loop_end; n_iter++) {
                    int n_block = num_n_blocks - 1 - n_iter;
                    mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                    _phase_s_full ^= 1;
                    int s_base = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_0[128];
                    tmem_ld_x32(&_tmem_load_0[0], s_base);
                    tmem_ld_x32(&_tmem_load_0[32], s_base + 32);
                    tmem_ld_x32(&_tmem_load_0[64], s_base + 64);
                    tmem_ld_x32(&_tmem_load_0[96], s_base + 96);
                    int tail_valid = seqlen_kv_bh - n_block * BLOCK_N;
                    if (tail_valid < BLOCK_N) {
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
                    float _fma_0 = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max_scaled);
                    if (row_max_val > -LOOM_INF) {
                        float _exp2_0 = approx_exp2(_fma_0);
                        acc_scale = _exp2_0;
                    } else {
                        acc_scale = 1.0f;
                    }
                    row_max_val = new_max;
                    sScale[warp % 4 * 32 + lane + scale_off] = acc_scale;
                    mbarrier_arrive(corr_sig_addr + (stage) * 8);
                    int p_base = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                    float p_log2_bias = 8.8073549f - new_max_scaled;
                    unsigned int pv0[16];
                    unsigned int pv1[8];
                    unsigned int pv2[8];
                    float block_sum0;
                    float block_sum1;
                    float block_sum2;
                    if (stage == 0) {
                        mbarrier_wait(order_p01_0_addr, order_p01_phase);
                    } else {
                        mbarrier_wait(order_p01_1_addr, order_p01_phase);
                    }
                    const float2 _fma_b2_17 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_18 = {p_log2_bias, p_log2_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_17, _fma_c2_18);
                    #pragma unroll
                    for (int _le = 0; _le < 122; _le++) {
                        _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                    }
                    if (stage == 0) {
                        mbarrier_arrive(order_p01_1_addr);
                    } else {
                        mbarrier_arrive(order_p01_0_addr);
                    }
                    order_p01_stage += 1;
                    if (order_p01_stage == 1) { order_p01_stage = 0; order_p01_phase ^= 1; }
                    #pragma unroll
                    for (int _le = 0; _le < 6; _le++) {
                        _tmem_load_0[_le + 122] = approx_exp2(_tmem_load_0[_le + 122]);
                    }
                    float2 _reg_reduce_sum2_19 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_19);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_19);
                    float _tmem_load_0_sum = _reg_reduce_sum2_19.x + _reg_reduce_sum2_19.y;
                    block_sum0 = _tmem_load_0_sum;
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
                        pv0[0] = _packed;
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
                        pv0[1] = _packed;
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
                        pv0[2] = _packed;
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
                        pv0[3] = _packed;
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
                        pv0[4] = _packed;
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
                        pv0[5] = _packed;
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
                        pv0[6] = _packed;
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
                        pv0[7] = _packed;
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
                        pv0[8] = _packed;
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
                        pv0[9] = _packed;
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
                        pv0[10] = _packed;
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
                        pv0[11] = _packed;
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
                        pv0[12] = _packed;
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
                        pv0[13] = _packed;
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
                        pv0[14] = _packed;
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
                        pv0[15] = _packed;
                    }
                    tmem_st_x16(p_base, pv0);
                    float2 _reg_reduce_sum2_20 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[(64) + 0], &_reg_reduce_sum2_20);
                    float _tmem_load_0_sum_0 = _reg_reduce_sum2_20.x + _reg_reduce_sum2_20.y;
                    block_sum1 = _tmem_load_0_sum_0;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_0[64]), "f"(_tmem_load_0[65]),
                                               "f"(_tmem_load_0[66]), "f"(_tmem_load_0[67]));
                        pv1[0] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[68]), "f"(_tmem_load_0[69]),
                                               "f"(_tmem_load_0[70]), "f"(_tmem_load_0[71]));
                        pv1[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[72]), "f"(_tmem_load_0[73]),
                                               "f"(_tmem_load_0[74]), "f"(_tmem_load_0[75]));
                        pv1[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[76]), "f"(_tmem_load_0[77]),
                                               "f"(_tmem_load_0[78]), "f"(_tmem_load_0[79]));
                        pv1[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[80]), "f"(_tmem_load_0[81]),
                                               "f"(_tmem_load_0[82]), "f"(_tmem_load_0[83]));
                        pv1[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[84]), "f"(_tmem_load_0[85]),
                                               "f"(_tmem_load_0[86]), "f"(_tmem_load_0[87]));
                        pv1[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[88]), "f"(_tmem_load_0[89]),
                                               "f"(_tmem_load_0[90]), "f"(_tmem_load_0[91]));
                        pv1[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[92]), "f"(_tmem_load_0[93]),
                                               "f"(_tmem_load_0[94]), "f"(_tmem_load_0[95]));
                        pv1[7] = _packed;
                    }
                    tmem_st_x8_u32(p_base + 16, (const uint32_t*)pv1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + (stage) * 8);
                    float2 _reg_reduce_sum2_21 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[(96) + 0], &_reg_reduce_sum2_21);
                    float _tmem_load_0_sum_1 = _reg_reduce_sum2_21.x + _reg_reduce_sum2_21.y;
                    block_sum2 = _tmem_load_0_sum_1;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_0[96]), "f"(_tmem_load_0[97]),
                                               "f"(_tmem_load_0[98]), "f"(_tmem_load_0[99]));
                        pv2[0] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[100]), "f"(_tmem_load_0[101]),
                                               "f"(_tmem_load_0[102]), "f"(_tmem_load_0[103]));
                        pv2[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[104]), "f"(_tmem_load_0[105]),
                                               "f"(_tmem_load_0[106]), "f"(_tmem_load_0[107]));
                        pv2[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[108]), "f"(_tmem_load_0[109]),
                                               "f"(_tmem_load_0[110]), "f"(_tmem_load_0[111]));
                        pv2[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[112]), "f"(_tmem_load_0[113]),
                                               "f"(_tmem_load_0[114]), "f"(_tmem_load_0[115]));
                        pv2[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[116]), "f"(_tmem_load_0[117]),
                                               "f"(_tmem_load_0[118]), "f"(_tmem_load_0[119]));
                        pv2[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[120]), "f"(_tmem_load_0[121]),
                                               "f"(_tmem_load_0[122]), "f"(_tmem_load_0[123]));
                        pv2[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[124]), "f"(_tmem_load_0[125]),
                                               "f"(_tmem_load_0[126]), "f"(_tmem_load_0[127]));
                        pv2[7] = _packed;
                    }
                    tmem_st_x8_u32(p_base + 24, (const uint32_t*)pv2);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_2_addr + (stage) * 8);
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                    row_sum_val = row_sum_val * acc_scale + (block_sum0 + block_sum1 + block_sum2);
                }
                #pragma unroll 1
                for (unsigned int n_iter_1 = unmasked_loop_start; n_iter_1 < num_n_blocks; n_iter_1++) {
                    int n_block_1 = num_n_blocks - 1 - n_iter_1;
                    mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                    _phase_s_full ^= 1;
                    int s_base_1 = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_1[128];
                    tmem_ld_x32(&_tmem_load_1[0], s_base_1);
                    tmem_ld_x32(&_tmem_load_1[32], s_base_1 + 32);
                    tmem_ld_x32(&_tmem_load_1[64], s_base_1 + 64);
                    tmem_ld_x32(&_tmem_load_1[96], s_base_1 + 96);
                    int tail_valid_1 = seqlen_kv_bh - n_block_1 * BLOCK_N;
                    {
                        if (tail_valid_1 < BLOCK_N) {
                            uint32_t _slice_lo_mask_12;
                            {
                                int _lim_22 = tail_valid_1;
                                if (_lim_22 <= 0) { _slice_lo_mask_12 = 0u; }
                                else if (_lim_22 >= 32) { _slice_lo_mask_12 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_12) : "r"(_lim_22));
                                }
                            }
                            #pragma unroll
                            for (int _i_23 = 0; _i_23 < 32; _i_23++) {
                                if (!(_slice_lo_mask_12 & (1u << _i_23))) _tmem_load_1[0 + _i_23] = -LOOM_INF;
                            }
                            uint32_t _slice_lo_mask_13;
                            {
                                int _lim_24 = tail_valid_1 - 32;
                                if (_lim_24 <= 0) { _slice_lo_mask_13 = 0u; }
                                else if (_lim_24 >= 32) { _slice_lo_mask_13 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_13) : "r"(_lim_24));
                                }
                            }
                            #pragma unroll
                            for (int _i_25 = 0; _i_25 < 32; _i_25++) {
                                if (!(_slice_lo_mask_13 & (1u << _i_25))) _tmem_load_1[32 + _i_25] = -LOOM_INF;
                            }
                            uint32_t _slice_lo_mask_14;
                            {
                                int _lim_26 = tail_valid_1 - 64;
                                if (_lim_26 <= 0) { _slice_lo_mask_14 = 0u; }
                                else if (_lim_26 >= 32) { _slice_lo_mask_14 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_14) : "r"(_lim_26));
                                }
                            }
                            #pragma unroll
                            for (int _i_27 = 0; _i_27 < 32; _i_27++) {
                                if (!(_slice_lo_mask_14 & (1u << _i_27))) _tmem_load_1[64 + _i_27] = -LOOM_INF;
                            }
                            uint32_t _slice_lo_mask_15;
                            {
                                int _lim_28 = tail_valid_1 - 96;
                                if (_lim_28 <= 0) { _slice_lo_mask_15 = 0u; }
                                else if (_lim_28 >= 32) { _slice_lo_mask_15 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_15) : "r"(_lim_28));
                                }
                            }
                            #pragma unroll
                            for (int _i_29 = 0; _i_29 < 32; _i_29++) {
                                if (!(_slice_lo_mask_15 & (1u << _i_29))) _tmem_load_1[96 + _i_29] = -LOOM_INF;
                            }
                        }
                    }
                    float2 _reg_reduce_max2_30 = {-LOOM_INF, -LOOM_INF};
                    row_max_x32_accum(&_tmem_load_1[0], _reg_reduce_max2_30);
                    row_max_x32_accum(&_tmem_load_1[32], _reg_reduce_max2_30);
                    row_max_x32_accum(&_tmem_load_1[64], _reg_reduce_max2_30);
                    row_max_x32_accum(&_tmem_load_1[96], _reg_reduce_max2_30);
                    float _tmem_load_1_max = row_max_reduce(_reg_reduce_max2_30);
                    float new_max_1 = _tmem_load_1_max;
                    float _max_1 = max_noftz(new_max_1, row_max_val);
                    new_max_1 = _max_1;
                    float new_max_scaled_1 = ((new_max_1 == -LOOM_INF) ? 0.0f : new_max_1) * softmax_scale_log2;
                    float acc_scale_1;
                    float _fma_1 = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max_scaled_1);
                    if (row_max_val > -LOOM_INF) {
                        float _exp2_1 = approx_exp2(_fma_1);
                        acc_scale_1 = _exp2_1;
                    } else {
                        acc_scale_1 = 1.0f;
                    }
                    row_max_val = new_max_1;
                    sScale[warp % 4 * 32 + lane + scale_off] = acc_scale_1;
                    mbarrier_arrive(corr_sig_addr + (stage) * 8);
                    int p_base_1 = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                    float p_log2_bias_1 = 8.8073549f - new_max_scaled_1;
                    unsigned int pv0_1[16];
                    unsigned int pv1_1[8];
                    unsigned int pv2_1[8];
                    float block_sum0_1;
                    float block_sum1_1;
                    float block_sum2_1;
                    if (stage == 0) {
                        mbarrier_wait(order_p01_0_addr, order_p01_phase);
                    } else {
                        mbarrier_wait(order_p01_1_addr, order_p01_phase);
                    }
                    const float2 _fma_b2_31 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_32 = {p_log2_bias_1, p_log2_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_lf], _fma_b2_31, _fma_c2_32);
                    #pragma unroll
                    for (int _le = 0; _le < 122; _le++) {
                        _tmem_load_1[_le] = approx_exp2(_tmem_load_1[_le]);
                    }
                    if (stage == 0) {
                        mbarrier_arrive(order_p01_1_addr);
                    } else {
                        mbarrier_arrive(order_p01_0_addr);
                    }
                    order_p01_stage += 1;
                    if (order_p01_stage == 1) { order_p01_stage = 0; order_p01_phase ^= 1; }
                    #pragma unroll
                    for (int _le = 0; _le < 6; _le++) {
                        _tmem_load_1[_le + 122] = approx_exp2(_tmem_load_1[_le + 122]);
                    }
                    float2 _reg_reduce_sum2_33 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_1[0], &_reg_reduce_sum2_33);
                    softmax_block_sum(&_tmem_load_1[32], &_reg_reduce_sum2_33);
                    float _tmem_load_1_sum = _reg_reduce_sum2_33.x + _reg_reduce_sum2_33.y;
                    block_sum0_1 = _tmem_load_1_sum;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_1[0]), "f"(_tmem_load_1[1]),
                                               "f"(_tmem_load_1[2]), "f"(_tmem_load_1[3]));
                        pv0_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[4]), "f"(_tmem_load_1[5]),
                                               "f"(_tmem_load_1[6]), "f"(_tmem_load_1[7]));
                        pv0_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[8]), "f"(_tmem_load_1[9]),
                                               "f"(_tmem_load_1[10]), "f"(_tmem_load_1[11]));
                        pv0_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[12]), "f"(_tmem_load_1[13]),
                                               "f"(_tmem_load_1[14]), "f"(_tmem_load_1[15]));
                        pv0_1[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[16]), "f"(_tmem_load_1[17]),
                                               "f"(_tmem_load_1[18]), "f"(_tmem_load_1[19]));
                        pv0_1[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[20]), "f"(_tmem_load_1[21]),
                                               "f"(_tmem_load_1[22]), "f"(_tmem_load_1[23]));
                        pv0_1[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[24]), "f"(_tmem_load_1[25]),
                                               "f"(_tmem_load_1[26]), "f"(_tmem_load_1[27]));
                        pv0_1[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[28]), "f"(_tmem_load_1[29]),
                                               "f"(_tmem_load_1[30]), "f"(_tmem_load_1[31]));
                        pv0_1[7] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[32]), "f"(_tmem_load_1[33]),
                                               "f"(_tmem_load_1[34]), "f"(_tmem_load_1[35]));
                        pv0_1[8] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[36]), "f"(_tmem_load_1[37]),
                                               "f"(_tmem_load_1[38]), "f"(_tmem_load_1[39]));
                        pv0_1[9] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[40]), "f"(_tmem_load_1[41]),
                                               "f"(_tmem_load_1[42]), "f"(_tmem_load_1[43]));
                        pv0_1[10] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[44]), "f"(_tmem_load_1[45]),
                                               "f"(_tmem_load_1[46]), "f"(_tmem_load_1[47]));
                        pv0_1[11] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[48]), "f"(_tmem_load_1[49]),
                                               "f"(_tmem_load_1[50]), "f"(_tmem_load_1[51]));
                        pv0_1[12] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[52]), "f"(_tmem_load_1[53]),
                                               "f"(_tmem_load_1[54]), "f"(_tmem_load_1[55]));
                        pv0_1[13] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[56]), "f"(_tmem_load_1[57]),
                                               "f"(_tmem_load_1[58]), "f"(_tmem_load_1[59]));
                        pv0_1[14] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[60]), "f"(_tmem_load_1[61]),
                                               "f"(_tmem_load_1[62]), "f"(_tmem_load_1[63]));
                        pv0_1[15] = _packed;
                    }
                    tmem_st_x16(p_base_1, pv0_1);
                    float2 _reg_reduce_sum2_34 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_1[(64) + 0], &_reg_reduce_sum2_34);
                    float _tmem_load_1_sum_0 = _reg_reduce_sum2_34.x + _reg_reduce_sum2_34.y;
                    block_sum1_1 = _tmem_load_1_sum_0;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_1[64]), "f"(_tmem_load_1[65]),
                                               "f"(_tmem_load_1[66]), "f"(_tmem_load_1[67]));
                        pv1_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[68]), "f"(_tmem_load_1[69]),
                                               "f"(_tmem_load_1[70]), "f"(_tmem_load_1[71]));
                        pv1_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[72]), "f"(_tmem_load_1[73]),
                                               "f"(_tmem_load_1[74]), "f"(_tmem_load_1[75]));
                        pv1_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[76]), "f"(_tmem_load_1[77]),
                                               "f"(_tmem_load_1[78]), "f"(_tmem_load_1[79]));
                        pv1_1[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[80]), "f"(_tmem_load_1[81]),
                                               "f"(_tmem_load_1[82]), "f"(_tmem_load_1[83]));
                        pv1_1[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[84]), "f"(_tmem_load_1[85]),
                                               "f"(_tmem_load_1[86]), "f"(_tmem_load_1[87]));
                        pv1_1[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[88]), "f"(_tmem_load_1[89]),
                                               "f"(_tmem_load_1[90]), "f"(_tmem_load_1[91]));
                        pv1_1[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[92]), "f"(_tmem_load_1[93]),
                                               "f"(_tmem_load_1[94]), "f"(_tmem_load_1[95]));
                        pv1_1[7] = _packed;
                    }
                    tmem_st_x8_u32(p_base_1 + 16, (const uint32_t*)pv1_1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + (stage) * 8);
                    float2 _reg_reduce_sum2_35 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_1[(96) + 0], &_reg_reduce_sum2_35);
                    float _tmem_load_1_sum_1 = _reg_reduce_sum2_35.x + _reg_reduce_sum2_35.y;
                    block_sum2_1 = _tmem_load_1_sum_1;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_1[96]), "f"(_tmem_load_1[97]),
                                               "f"(_tmem_load_1[98]), "f"(_tmem_load_1[99]));
                        pv2_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[100]), "f"(_tmem_load_1[101]),
                                               "f"(_tmem_load_1[102]), "f"(_tmem_load_1[103]));
                        pv2_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[104]), "f"(_tmem_load_1[105]),
                                               "f"(_tmem_load_1[106]), "f"(_tmem_load_1[107]));
                        pv2_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[108]), "f"(_tmem_load_1[109]),
                                               "f"(_tmem_load_1[110]), "f"(_tmem_load_1[111]));
                        pv2_1[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[112]), "f"(_tmem_load_1[113]),
                                               "f"(_tmem_load_1[114]), "f"(_tmem_load_1[115]));
                        pv2_1[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[116]), "f"(_tmem_load_1[117]),
                                               "f"(_tmem_load_1[118]), "f"(_tmem_load_1[119]));
                        pv2_1[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[120]), "f"(_tmem_load_1[121]),
                                               "f"(_tmem_load_1[122]), "f"(_tmem_load_1[123]));
                        pv2_1[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[124]), "f"(_tmem_load_1[125]),
                                               "f"(_tmem_load_1[126]), "f"(_tmem_load_1[127]));
                        pv2_1[7] = _packed;
                    }
                    tmem_st_x8_u32(p_base_1 + 24, (const uint32_t*)pv2_1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_2_addr + (stage) * 8);
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                    row_sum_val = row_sum_val * acc_scale_1 + (block_sum0_1 + block_sum1_1 + block_sum2_1);
                }
                sScale[warp % 4 * 32 + lane + scale_off + 2 * BLOCK_M] = row_sum_val;
                sScale[warp % 4 * 32 + lane + scale_off + 4 * BLOCK_M] = row_max_val;
                mbarrier_arrive(corr_sig_addr + (stage) * 8);
            }
        }
    // ---- Role: correction ----
    } else if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            unsigned int total_tiles_1 = NUM_M_BLOCKS * total_bh;
            unsigned int max_rounds_1 = total_tiles_1 + 1;
            {
                max_rounds_1 = 1;
            }
            unsigned int dq_pk_stage = 0;
            unsigned int dq_kv_stage = 0;
            int dq_warp_base = (warp - 8) * 32;
            int dq_tok = dq_warp_base + lane;
            unsigned int _phase_work_id_full_0_1 = 0;
            unsigned int _phase_pk_full = 0;
            unsigned int _phase_kv_empty = 1;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_iter_1 = 0; tile_iter_1 < max_rounds_1; tile_iter_1++) {
                unsigned int tile_idx_1;
                {
                    tile_idx_1 = blockIdx.x;
                }
                if (tile_idx_1 >= total_tiles_1) {
                    break;
                }
                unsigned int m_block_1;
                unsigned int bh_1;
                {
                    unsigned int l2section_1 = L2_SWIZZLE * NUM_M_BLOCKS;
                    unsigned int section_1 = tile_idx_1 / l2section_1;
                    unsigned int l2_mod_1 = tile_idx_1 % l2section_1;
                    m_block_1 = l2_mod_1 / (unsigned int)L2_SWIZZLE;
                    bh_1 = section_1 * (unsigned int)L2_SWIZZLE + l2_mod_1 % (unsigned int)L2_SWIZZLE;
                    m_block_1 = (unsigned int)(NUM_M_BLOCKS - 1) - m_block_1;
                }
                int seqlen_kv_bh_1 = UNIFORM_KV_LEN;
                int seqlen_q_bh_1 = UNIFORM_Q_LEN;
                {
                    seqlen_kv_bh_1 = seq_lens_kv[bh_1];
                }
                {
                    seqlen_q_bh_1 = seq_lens_q[bh_1];
                }
                unsigned int num_n_blocks_1 = (seqlen_kv_bh_1 + BLOCK_N - 1) / BLOCK_N;
                {
                    int kv_shift_2 = seqlen_kv_bh_1 - seqlen_q_bh_1;
                    unsigned int max_n_1 = (m_block_1 * 2 + 2) * (unsigned int)TOK_PER_STAGE + (unsigned int)kv_shift_2;
                    if (max_n_1 < (unsigned int)seqlen_kv_bh_1) {
                        num_n_blocks_1 = (max_n_1 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                    }
                }
                int unit = bh_1 % (unsigned int)(NUM_Q_HEADS / PACK_G);
                int q_head_base = unit * PACK_G;
                int row_tok = (warp % 4 * 32 + lane) / PACK_G;
                int row_head = q_head_base + (warp % 4 * 32 + lane) % PACK_G;
                int tok_base = cu_seq_lens_q[bh_1];
                mbarrier_arrive(p_full_addr);
                mbarrier_arrive(p_full_addr + 8);
                mbarrier_wait(pk_full_addr + (dq_pk_stage) * 8, _phase_pk_full);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_wait(kv_empty_addr + (dq_kv_stage) * 8, _phase_kv_empty);
                int pk_base = smem_pk_addr + dq_pk_stage * 9216;
                int sf_base = pk_base + 8192;
                int out_base = smem_kv_addr + dq_kv_stage * 16384;
                unsigned int ksf_words[2];
                {
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&ksf_words[0])) : "r"(sf_base + dq_tok * 8));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&ksf_words[1])) : "r"(sf_base + dq_tok * 8 + 4));
                }
                #pragma unroll
                for (int chunk_wave = 0; chunk_wave < 4; chunk_wave++) {
                    int mapped_tok = dq_tok;
                    int group_pair = chunk_wave;
                    int pk_row = pk_base + mapped_tok * 64;
                    int out_row = out_base + mapped_tok * 128;
                    int row_swizzle = mapped_tok % 8;
                    unsigned int packed_words[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words[(0) + 3]))
                        : "r"(pk_row + group_pair * 16));
                    unsigned int ksf_word[1];
                    #pragma unroll
                    for (int in_pair = 0; in_pair < 2; in_pair++) {
                        int group = group_pair * 2 + in_pair;
                        int packed_idx = in_pair * 2;
                        unsigned int sf = 0;
                        {
                            sf = ksf_words[group / 4] >> (unsigned int)(group % 4 * 8);
                        }
                        unsigned int encoded[4];
                        {
                            uint32_t _qmul4_marker_0;
                            // qmul4 patch marker 0x51A7E005
                            asm volatile("fma.rn.f32 %0, %1, 0f51A7E005, %2;" : "=r"(_qmul4_marker_0) : "r"(packed_words[packed_idx]), "r"(sf));
                            encoded[0] = _qmul4_marker_0;
                        }
                        {
                            uint32_t _qmul4_marker_1;
                            // qmul4 patch marker 0x51A7E006
                            asm volatile("fma.rn.f32 %0, %1, 0f51A7E006, %2;" : "=r"(_qmul4_marker_1) : "r"(packed_words[packed_idx]), "r"(sf));
                            encoded[1] = _qmul4_marker_1;
                        }
                        {
                            uint32_t _qmul4_marker_2;
                            // qmul4 patch marker 0x51A7E005
                            asm volatile("fma.rn.f32 %0, %1, 0f51A7E005, %2;" : "=r"(_qmul4_marker_2) : "r"(packed_words[packed_idx + 1]), "r"(sf));
                            encoded[2] = _qmul4_marker_2;
                        }
                        {
                            uint32_t _qmul4_marker_3;
                            // qmul4 patch marker 0x51A7E006
                            asm volatile("fma.rn.f32 %0, %1, 0f51A7E006, %2;" : "=r"(_qmul4_marker_3) : "r"(packed_words[packed_idx + 1]), "r"(sf));
                            encoded[3] = _qmul4_marker_3;
                        }
                        int out_group = out_row + (group ^ row_swizzle) * 16;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group), "r"((encoded[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group + 4), "r"((encoded[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group + 8), "r"((encoded[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group + 12), "r"((encoded[3])));
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(kv_full_addr + (dq_kv_stage) * 8);
                    mbarrier_arrive(pk_empty_addr + (dq_pk_stage) * 8);
                }
                dq_pk_stage += 1;
                if (dq_pk_stage == 2) { dq_pk_stage = 0; _phase_pk_full ^= 1; }
                dq_kv_stage += 1;
                if (dq_kv_stage == 3) { dq_kv_stage = 0; _phase_kv_empty ^= 1; }
                mbarrier_wait(pk_full_addr + (dq_pk_stage) * 8, _phase_pk_full);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_wait(kv_empty_addr + (dq_kv_stage) * 8, _phase_kv_empty);
                int pk_base_0 = smem_pk_addr + dq_pk_stage * 9216;
                int sf_base_1 = pk_base_0 + 8192;
                int out_base_2 = smem_kv_addr + dq_kv_stage * 16384;
                unsigned int vsf_words[8];
                {
                    int vsf_row = (dq_tok & -4) * 8;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&vsf_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words[(0) + 3]))
                        : "r"(sf_base_1 + vsf_row));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&vsf_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words[(4) + 3]))
                        : "r"(sf_base_1 + vsf_row + 16));
                }
                #pragma unroll
                for (int chunk_wave_1 = 0; chunk_wave_1 < 4; chunk_wave_1++) {
                    int mapped_tok_1 = dq_tok;
                    int group_pair_1 = chunk_wave_1;
                    int pk_row_1 = pk_base_0 + mapped_tok_1 * 64;
                    int out_row_1 = out_base_2 + mapped_tok_1 * 128;
                    int row_swizzle_1 = mapped_tok_1 % 8;
                    unsigned int packed_words_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_1[(0) + 3]))
                        : "r"(pk_row_1 + group_pair_1 * 16));
                    unsigned int chunk_vsf_words[2];
                    #pragma unroll
                    for (int in_pair_1 = 0; in_pair_1 < 2; in_pair_1++) {
                        int group_1 = group_pair_1 * 2 + in_pair_1;
                        int packed_idx_1 = in_pair_1 * 2;
                        int vshift = mapped_tok_1 % 4 * 8;
                        unsigned int sf_1 = 0;
                        {
                            sf_1 = vsf_words[group_1] >> (unsigned int)vshift;
                        }
                        unsigned int encoded_1[4];
                        {
                            uint32_t _qmul4_marker_4;
                            // qmul4 patch marker 0x51A7E005
                            asm volatile("fma.rn.f32 %0, %1, 0f51A7E005, %2;" : "=r"(_qmul4_marker_4) : "r"(packed_words_1[packed_idx_1]), "r"(sf_1));
                            encoded_1[0] = _qmul4_marker_4;
                        }
                        {
                            uint32_t _qmul4_marker_5;
                            // qmul4 patch marker 0x51A7E006
                            asm volatile("fma.rn.f32 %0, %1, 0f51A7E006, %2;" : "=r"(_qmul4_marker_5) : "r"(packed_words_1[packed_idx_1]), "r"(sf_1));
                            encoded_1[1] = _qmul4_marker_5;
                        }
                        {
                            uint32_t _qmul4_marker_6;
                            // qmul4 patch marker 0x51A7E005
                            asm volatile("fma.rn.f32 %0, %1, 0f51A7E005, %2;" : "=r"(_qmul4_marker_6) : "r"(packed_words_1[packed_idx_1 + 1]), "r"(sf_1));
                            encoded_1[2] = _qmul4_marker_6;
                        }
                        {
                            uint32_t _qmul4_marker_7;
                            // qmul4 patch marker 0x51A7E006
                            asm volatile("fma.rn.f32 %0, %1, 0f51A7E006, %2;" : "=r"(_qmul4_marker_7) : "r"(packed_words_1[packed_idx_1 + 1]), "r"(sf_1));
                            encoded_1[3] = _qmul4_marker_7;
                        }
                        int out_group_1 = out_row_1 + (group_1 ^ row_swizzle_1) * 16;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_1), "r"((encoded_1[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_1 + 4), "r"((encoded_1[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_1 + 8), "r"((encoded_1[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_1 + 12), "r"((encoded_1[3])));
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(kv_full_addr + (dq_kv_stage) * 8);
                    mbarrier_arrive(pk_empty_addr + (dq_pk_stage) * 8);
                }
                dq_pk_stage += 1;
                if (dq_pk_stage == 2) { dq_pk_stage = 0; _phase_pk_full ^= 1; }
                dq_kv_stage += 1;
                if (dq_kv_stage == 3) { dq_kv_stage = 0; _phase_kv_empty ^= 1; }
                if (num_n_blocks_1 > 1) {
                    mbarrier_wait(pk_full_addr + (dq_pk_stage) * 8, _phase_pk_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_wait(kv_empty_addr + (dq_kv_stage) * 8, _phase_kv_empty);
                    int pk_base_1 = smem_pk_addr + dq_pk_stage * 9216;
                    int sf_base_2 = pk_base_1 + 8192;
                    int out_base_3 = smem_kv_addr + dq_kv_stage * 16384;
                    unsigned int ksf_words_4[2];
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&ksf_words_4[0])) : "r"(sf_base_2 + dq_tok * 8));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&ksf_words_4[1])) : "r"(sf_base_2 + dq_tok * 8 + 4));
                    }
                    #pragma unroll
                    for (int chunk_wave_2 = 0; chunk_wave_2 < 4; chunk_wave_2++) {
                        int mapped_tok_2 = dq_tok;
                        int group_pair_2 = chunk_wave_2;
                        int pk_row_2 = pk_base_1 + mapped_tok_2 * 64;
                        int out_row_2 = out_base_3 + mapped_tok_2 * 128;
                        int row_swizzle_2 = mapped_tok_2 % 8;
                        unsigned int packed_words_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_2[(0) + 3]))
                            : "r"(pk_row_2 + group_pair_2 * 16));
                        unsigned int ksf_word_1[1];
                        #pragma unroll
                        for (int in_pair_2 = 0; in_pair_2 < 2; in_pair_2++) {
                            int group_2 = group_pair_2 * 2 + in_pair_2;
                            int packed_idx_2 = in_pair_2 * 2;
                            unsigned int sf_2 = 0;
                            {
                                sf_2 = ksf_words_4[group_2 / 4] >> (unsigned int)(group_2 % 4 * 8);
                            }
                            unsigned int encoded_2[4];
                            {
                                uint32_t _qmul4_marker_8;
                                // qmul4 patch marker 0x51A7E005
                                asm volatile("fma.rn.f32 %0, %1, 0f51A7E005, %2;" : "=r"(_qmul4_marker_8) : "r"(packed_words_2[packed_idx_2]), "r"(sf_2));
                                encoded_2[0] = _qmul4_marker_8;
                            }
                            {
                                uint32_t _qmul4_marker_9;
                                // qmul4 patch marker 0x51A7E006
                                asm volatile("fma.rn.f32 %0, %1, 0f51A7E006, %2;" : "=r"(_qmul4_marker_9) : "r"(packed_words_2[packed_idx_2]), "r"(sf_2));
                                encoded_2[1] = _qmul4_marker_9;
                            }
                            {
                                uint32_t _qmul4_marker_10;
                                // qmul4 patch marker 0x51A7E005
                                asm volatile("fma.rn.f32 %0, %1, 0f51A7E005, %2;" : "=r"(_qmul4_marker_10) : "r"(packed_words_2[packed_idx_2 + 1]), "r"(sf_2));
                                encoded_2[2] = _qmul4_marker_10;
                            }
                            {
                                uint32_t _qmul4_marker_11;
                                // qmul4 patch marker 0x51A7E006
                                asm volatile("fma.rn.f32 %0, %1, 0f51A7E006, %2;" : "=r"(_qmul4_marker_11) : "r"(packed_words_2[packed_idx_2 + 1]), "r"(sf_2));
                                encoded_2[3] = _qmul4_marker_11;
                            }
                            int out_group_2 = out_row_2 + (group_2 ^ row_swizzle_2) * 16;
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_2), "r"((encoded_2[0])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_2 + 4), "r"((encoded_2[1])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_2 + 8), "r"((encoded_2[2])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_2 + 12), "r"((encoded_2[3])));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(kv_full_addr + (dq_kv_stage) * 8);
                        mbarrier_arrive(pk_empty_addr + (dq_pk_stage) * 8);
                    }
                    dq_pk_stage += 1;
                    if (dq_pk_stage == 2) { dq_pk_stage = 0; _phase_pk_full ^= 1; }
                    dq_kv_stage += 1;
                    if (dq_kv_stage == 3) { dq_kv_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(pk_full_addr + (dq_pk_stage) * 8, _phase_pk_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_wait(kv_empty_addr + (dq_kv_stage) * 8, _phase_kv_empty);
                    int pk_base_5 = smem_pk_addr + dq_pk_stage * 9216;
                    int sf_base_6 = pk_base_5 + 8192;
                    int out_base_7 = smem_kv_addr + dq_kv_stage * 16384;
                    unsigned int vsf_words_8[8];
                    {
                        int vsf_row_1 = (dq_tok & -4) * 8;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_8[(0) + 3]))
                            : "r"(sf_base_6 + vsf_row_1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_8[(4) + 3]))
                            : "r"(sf_base_6 + vsf_row_1 + 16));
                    }
                    #pragma unroll
                    for (int chunk_wave_3 = 0; chunk_wave_3 < 4; chunk_wave_3++) {
                        int mapped_tok_3 = dq_tok;
                        int group_pair_3 = chunk_wave_3;
                        int pk_row_3 = pk_base_5 + mapped_tok_3 * 64;
                        int out_row_3 = out_base_7 + mapped_tok_3 * 128;
                        int row_swizzle_3 = mapped_tok_3 % 8;
                        unsigned int packed_words_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_3[(0) + 3]))
                            : "r"(pk_row_3 + group_pair_3 * 16));
                        unsigned int chunk_vsf_words_1[2];
                        #pragma unroll
                        for (int in_pair_3 = 0; in_pair_3 < 2; in_pair_3++) {
                            int group_3 = group_pair_3 * 2 + in_pair_3;
                            int packed_idx_3 = in_pair_3 * 2;
                            int vshift_1 = mapped_tok_3 % 4 * 8;
                            unsigned int sf_3 = 0;
                            {
                                sf_3 = vsf_words_8[group_3] >> (unsigned int)vshift_1;
                            }
                            unsigned int encoded_3[4];
                            {
                                uint32_t _qmul4_marker_12;
                                // qmul4 patch marker 0x51A7E005
                                asm volatile("fma.rn.f32 %0, %1, 0f51A7E005, %2;" : "=r"(_qmul4_marker_12) : "r"(packed_words_3[packed_idx_3]), "r"(sf_3));
                                encoded_3[0] = _qmul4_marker_12;
                            }
                            {
                                uint32_t _qmul4_marker_13;
                                // qmul4 patch marker 0x51A7E006
                                asm volatile("fma.rn.f32 %0, %1, 0f51A7E006, %2;" : "=r"(_qmul4_marker_13) : "r"(packed_words_3[packed_idx_3]), "r"(sf_3));
                                encoded_3[1] = _qmul4_marker_13;
                            }
                            {
                                uint32_t _qmul4_marker_14;
                                // qmul4 patch marker 0x51A7E005
                                asm volatile("fma.rn.f32 %0, %1, 0f51A7E005, %2;" : "=r"(_qmul4_marker_14) : "r"(packed_words_3[packed_idx_3 + 1]), "r"(sf_3));
                                encoded_3[2] = _qmul4_marker_14;
                            }
                            {
                                uint32_t _qmul4_marker_15;
                                // qmul4 patch marker 0x51A7E006
                                asm volatile("fma.rn.f32 %0, %1, 0f51A7E006, %2;" : "=r"(_qmul4_marker_15) : "r"(packed_words_3[packed_idx_3 + 1]), "r"(sf_3));
                                encoded_3[3] = _qmul4_marker_15;
                            }
                            int out_group_3 = out_row_3 + (group_3 ^ row_swizzle_3) * 16;
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_3), "r"((encoded_3[0])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_3 + 4), "r"((encoded_3[1])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_3 + 8), "r"((encoded_3[2])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_3 + 12), "r"((encoded_3[3])));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(kv_full_addr + (dq_kv_stage) * 8);
                        mbarrier_arrive(pk_empty_addr + (dq_pk_stage) * 8);
                    }
                    dq_pk_stage += 1;
                    if (dq_pk_stage == 2) { dq_pk_stage = 0; _phase_pk_full ^= 1; }
                    dq_kv_stage += 1;
                    if (dq_kv_stage == 3) { dq_kv_stage = 0; _phase_kv_empty ^= 1; }
                }
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                #pragma unroll 1
                for (unsigned int n_iter_2 = 1; n_iter_2 < num_n_blocks_1; n_iter_2++) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    float scale = sScale[warp % 4 * 32 + lane];
                    int _vote_0 = __any_sync(0xFFFFFFFF, scale < 1.0f);
                    if (_vote_0 != 0) {
                        #pragma unroll
                        for (int cr_col = 0; cr_col < HEAD_DIM / 16; cr_col++) {
                            int cr_addr = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(cr_col * 16);
                            float _tmem_load_2[16];
                            tmem_ld_x16(&_tmem_load_2[0], cr_addr);
                            const float2 _scale2_16 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_16);
                            tmem_st_x16_f32(cr_addr, _tmem_load_2);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr);
                    mbarrier_arrive(corr_done_addr + 8);
                    if (n_iter_2 < num_n_blocks_1 - 1) {
                        mbarrier_wait(pk_full_addr + (dq_pk_stage) * 8, _phase_pk_full);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_wait(kv_empty_addr + (dq_kv_stage) * 8, _phase_kv_empty);
                        int pk_base_1_1 = smem_pk_addr + dq_pk_stage * 9216;
                        int sf_base_2_1 = pk_base_1_1 + 8192;
                        int out_base_3_1 = smem_kv_addr + dq_kv_stage * 16384;
                        unsigned int ksf_words_4_1[2];
                        {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&ksf_words_4_1[0])) : "r"(sf_base_2_1 + dq_tok * 8));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&ksf_words_4_1[1])) : "r"(sf_base_2_1 + dq_tok * 8 + 4));
                        }
                        #pragma unroll
                        for (int chunk_wave_4 = 0; chunk_wave_4 < 4; chunk_wave_4++) {
                            int mapped_tok_4 = dq_tok;
                            int group_pair_4 = chunk_wave_4;
                            int pk_row_4 = pk_base_1_1 + mapped_tok_4 * 64;
                            int out_row_4 = out_base_3_1 + mapped_tok_4 * 128;
                            int row_swizzle_4 = mapped_tok_4 % 8;
                            unsigned int packed_words_4[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_4[(0) + 3]))
                                : "r"(pk_row_4 + group_pair_4 * 16));
                            unsigned int ksf_word_2[1];
                            #pragma unroll
                            for (int in_pair_4 = 0; in_pair_4 < 2; in_pair_4++) {
                                int group_4 = group_pair_4 * 2 + in_pair_4;
                                int packed_idx_4 = in_pair_4 * 2;
                                unsigned int sf_4 = 0;
                                {
                                    sf_4 = ksf_words_4_1[group_4 / 4] >> (unsigned int)(group_4 % 4 * 8);
                                }
                                unsigned int encoded_4[4];
                                {
                                    uint32_t _qmul4_marker_17;
                                    // qmul4 patch marker 0x51A7E005
                                    asm volatile("fma.rn.f32 %0, %1, 0f51A7E005, %2;" : "=r"(_qmul4_marker_17) : "r"(packed_words_4[packed_idx_4]), "r"(sf_4));
                                    encoded_4[0] = _qmul4_marker_17;
                                }
                                {
                                    uint32_t _qmul4_marker_18;
                                    // qmul4 patch marker 0x51A7E006
                                    asm volatile("fma.rn.f32 %0, %1, 0f51A7E006, %2;" : "=r"(_qmul4_marker_18) : "r"(packed_words_4[packed_idx_4]), "r"(sf_4));
                                    encoded_4[1] = _qmul4_marker_18;
                                }
                                {
                                    uint32_t _qmul4_marker_19;
                                    // qmul4 patch marker 0x51A7E005
                                    asm volatile("fma.rn.f32 %0, %1, 0f51A7E005, %2;" : "=r"(_qmul4_marker_19) : "r"(packed_words_4[packed_idx_4 + 1]), "r"(sf_4));
                                    encoded_4[2] = _qmul4_marker_19;
                                }
                                {
                                    uint32_t _qmul4_marker_20;
                                    // qmul4 patch marker 0x51A7E006
                                    asm volatile("fma.rn.f32 %0, %1, 0f51A7E006, %2;" : "=r"(_qmul4_marker_20) : "r"(packed_words_4[packed_idx_4 + 1]), "r"(sf_4));
                                    encoded_4[3] = _qmul4_marker_20;
                                }
                                int out_group_4 = out_row_4 + (group_4 ^ row_swizzle_4) * 16;
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_4), "r"((encoded_4[0])));
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_4 + 4), "r"((encoded_4[1])));
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_4 + 8), "r"((encoded_4[2])));
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_4 + 12), "r"((encoded_4[3])));
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(kv_full_addr + (dq_kv_stage) * 8);
                            mbarrier_arrive(pk_empty_addr + (dq_pk_stage) * 8);
                        }
                        dq_pk_stage += 1;
                        if (dq_pk_stage == 2) { dq_pk_stage = 0; _phase_pk_full ^= 1; }
                        dq_kv_stage += 1;
                        if (dq_kv_stage == 3) { dq_kv_stage = 0; _phase_kv_empty ^= 1; }
                    }
                    mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                    _phase_corr_sig_1 ^= 1;
                    float scale1 = sScale[warp % 4 * 32 + lane + BLOCK_M];
                    int _vote_1 = __any_sync(0xFFFFFFFF, scale1 < 1.0f);
                    if (_vote_1 != 0) {
                        #pragma unroll
                        for (int cr_col_1 = 0; cr_col_1 < HEAD_DIM / 16; cr_col_1++) {
                            int cr_addr_1 = taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(cr_col_1 * 16);
                            float _tmem_load_3[16];
                            tmem_ld_x16(&_tmem_load_3[0], cr_addr_1);
                            const float2 _scale2_21 = {scale1, scale1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_21);
                            tmem_st_x16_f32(cr_addr_1, _tmem_load_3);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr + 8);
                    mbarrier_arrive(corr_done_addr);
                    if (n_iter_2 < num_n_blocks_1 - 1) {
                        mbarrier_wait(pk_full_addr + (dq_pk_stage) * 8, _phase_pk_full);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_wait(kv_empty_addr + (dq_kv_stage) * 8, _phase_kv_empty);
                        int pk_base_1_2 = smem_pk_addr + dq_pk_stage * 9216;
                        int sf_base_2_2 = pk_base_1_2 + 8192;
                        int out_base_3_2 = smem_kv_addr + dq_kv_stage * 16384;
                        unsigned int vsf_words_4[8];
                        {
                            int vsf_row_2 = (dq_tok & -4) * 8;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_4[(0) + 3]))
                                : "r"(sf_base_2_2 + vsf_row_2));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vsf_words_4[(4) + 3]))
                                : "r"(sf_base_2_2 + vsf_row_2 + 16));
                        }
                        #pragma unroll
                        for (int chunk_wave_5 = 0; chunk_wave_5 < 4; chunk_wave_5++) {
                            int mapped_tok_5 = dq_tok;
                            int group_pair_5 = chunk_wave_5;
                            int pk_row_5 = pk_base_1_2 + mapped_tok_5 * 64;
                            int out_row_5 = out_base_3_2 + mapped_tok_5 * 128;
                            int row_swizzle_5 = mapped_tok_5 % 8;
                            unsigned int packed_words_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_words_5[(0) + 3]))
                                : "r"(pk_row_5 + group_pair_5 * 16));
                            unsigned int chunk_vsf_words_2[2];
                            #pragma unroll
                            for (int in_pair_5 = 0; in_pair_5 < 2; in_pair_5++) {
                                int group_5 = group_pair_5 * 2 + in_pair_5;
                                int packed_idx_5 = in_pair_5 * 2;
                                int vshift_2 = mapped_tok_5 % 4 * 8;
                                unsigned int sf_5 = 0;
                                {
                                    sf_5 = vsf_words_4[group_5] >> (unsigned int)vshift_2;
                                }
                                unsigned int encoded_5[4];
                                {
                                    uint32_t _qmul4_marker_22;
                                    // qmul4 patch marker 0x51A7E005
                                    asm volatile("fma.rn.f32 %0, %1, 0f51A7E005, %2;" : "=r"(_qmul4_marker_22) : "r"(packed_words_5[packed_idx_5]), "r"(sf_5));
                                    encoded_5[0] = _qmul4_marker_22;
                                }
                                {
                                    uint32_t _qmul4_marker_23;
                                    // qmul4 patch marker 0x51A7E006
                                    asm volatile("fma.rn.f32 %0, %1, 0f51A7E006, %2;" : "=r"(_qmul4_marker_23) : "r"(packed_words_5[packed_idx_5]), "r"(sf_5));
                                    encoded_5[1] = _qmul4_marker_23;
                                }
                                {
                                    uint32_t _qmul4_marker_24;
                                    // qmul4 patch marker 0x51A7E005
                                    asm volatile("fma.rn.f32 %0, %1, 0f51A7E005, %2;" : "=r"(_qmul4_marker_24) : "r"(packed_words_5[packed_idx_5 + 1]), "r"(sf_5));
                                    encoded_5[2] = _qmul4_marker_24;
                                }
                                {
                                    uint32_t _qmul4_marker_25;
                                    // qmul4 patch marker 0x51A7E006
                                    asm volatile("fma.rn.f32 %0, %1, 0f51A7E006, %2;" : "=r"(_qmul4_marker_25) : "r"(packed_words_5[packed_idx_5 + 1]), "r"(sf_5));
                                    encoded_5[3] = _qmul4_marker_25;
                                }
                                int out_group_5 = out_row_5 + (group_5 ^ row_swizzle_5) * 16;
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_5), "r"((encoded_5[0])));
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_5 + 4), "r"((encoded_5[1])));
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_5 + 8), "r"((encoded_5[2])));
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(out_group_5 + 12), "r"((encoded_5[3])));
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(kv_full_addr + (dq_kv_stage) * 8);
                            mbarrier_arrive(pk_empty_addr + (dq_pk_stage) * 8);
                        }
                        dq_pk_stage += 1;
                        if (dq_pk_stage == 2) { dq_pk_stage = 0; _phase_pk_full ^= 1; }
                        dq_kv_stage += 1;
                        if (dq_kv_stage == 3) { dq_kv_stage = 0; _phase_kv_empty ^= 1; }
                    }
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
                    int tmem_o_off = ((stage_1 == 0) ? TMEM_OUTPUT_0_OFFSET : TMEM_OUTPUT_1_OFFSET);
                    int s_off = stage_1 * BLOCK_M;
                    float final_sum = sScale[warp % 4 * 32 + lane + s_off + 2 * BLOCK_M];
                    float final_max = sScale[warp % 4 * 32 + lane + s_off + 4 * BLOCK_M];
                    float final_scale;
                    if (final_sum != 0.0f && final_sum == final_sum) {
                        float _rcp_0 = approx_rcp(final_sum);
                        final_scale = _rcp_0;
                        final_scale = final_scale * output_scale;
                    } else {
                        final_scale = 0.0f;
                    }
                    int q_tok0 = (m_block_1 * 2 + (unsigned int)stage_1) * (unsigned int)TOK_PER_STAGE;
                    if (q_tok0 < seqlen_q_bh_1) {
                        int row_valid = warp % 4 * 32 + lane < TOK_PER_STAGE * PACK_G && seqlen_q_bh_1 > q_tok0 + row_tok;
                        long long o_tok = tok_base + q_tok0 + row_tok;
                        long long o_row = o_tok * (long long)NUM_Q_HEADS + (long long)row_head;
                        #pragma unroll
                        for (int ce_col = 0; ce_col < HEAD_DIM / 16; ce_col++) {
                            int ce_addr = taddr + (unsigned int)tmem_o_off + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(ce_col * 16);
                            float _tmem_load_4[16];
                            tmem_ld_x16(&_tmem_load_4[0], ce_addr);
                            if (row_valid != 0) {
                                long long o_elem = o_row * (long long)HEAD_DIM + (long long)(ce_col * 16);
                                {
                                    const float2 _prescale2_26 = {final_scale, final_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[0])[_ps], _prescale2_26);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_4[0 + _ps] *= final_scale;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_4[0 + 1]), "f"(_tmem_load_4[0 + 0]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_4[0 + 3]), "f"(_tmem_load_4[0 + 2]));
                                        _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_4[0 + 5]), "f"(_tmem_load_4[0 + 4]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_4[0 + 7]), "f"(_tmem_load_4[0 + 6]));
                                        _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_4[0 + 9]), "f"(_tmem_load_4[0 + 8]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_4[0 + 11]), "f"(_tmem_load_4[0 + 10]));
                                        _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_4[0 + 13]), "f"(_tmem_load_4[0 + 12]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_4[0 + 15]), "f"(_tmem_load_4[0 + 14]));
                                        _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(O_ptr + o_elem) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 12) {
        { // mma_main
            unsigned int total_tiles_2 = NUM_M_BLOCKS * total_bh;
            unsigned int max_rounds_2 = total_tiles_2 + 1;
            {
                max_rounds_2 = 1;
            }
            unsigned int mma_kv_stage = 0;
            unsigned int mma_kv_phase = 0;
            unsigned int _phase_work_id_full_0_2 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_2_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_p_full_2_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_iter_2 = 0; tile_iter_2 < max_rounds_2; tile_iter_2++) {
                unsigned int tile_idx_2;
                {
                    tile_idx_2 = blockIdx.x;
                }
                if (tile_idx_2 >= total_tiles_2) {
                    break;
                }
                unsigned int m_block_2;
                unsigned int bh_2;
                {
                    unsigned int l2section_2 = L2_SWIZZLE * NUM_M_BLOCKS;
                    unsigned int section_2 = tile_idx_2 / l2section_2;
                    unsigned int l2_mod_2 = tile_idx_2 % l2section_2;
                    m_block_2 = l2_mod_2 / (unsigned int)L2_SWIZZLE;
                    bh_2 = section_2 * (unsigned int)L2_SWIZZLE + l2_mod_2 % (unsigned int)L2_SWIZZLE;
                    m_block_2 = (unsigned int)(NUM_M_BLOCKS - 1) - m_block_2;
                }
                int seqlen_kv_bh_2 = UNIFORM_KV_LEN;
                int seqlen_q_bh_2 = UNIFORM_Q_LEN;
                {
                    seqlen_kv_bh_2 = seq_lens_kv[bh_2];
                }
                {
                    seqlen_q_bh_2 = seq_lens_q[bh_2];
                }
                unsigned int num_n_blocks_2 = (seqlen_kv_bh_2 + BLOCK_N - 1) / BLOCK_N;
                {
                    int kv_shift_3 = seqlen_kv_bh_2 - seqlen_q_bh_2;
                    unsigned int max_n_2 = (m_block_2 * 2 + 2) * (unsigned int)TOK_PER_STAGE + (unsigned int)kv_shift_3;
                    if (max_n_2 < (unsigned int)seqlen_kv_bh_2) {
                        num_n_blocks_2 = (max_n_2 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                    }
                }
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(q_full_addr + 8, _phase_q_full_1);
                _phase_q_full_1 ^= 1;
                mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase);
                int _mma_a_lo_0 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024);
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
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores_0), "r"(0));
                elect_commit(s_full_addr);
                int _mma_a_lo_1 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024);
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
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_scores_1), "r"(0));
                elect_commit(s_full_addr + 8);
                elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                mma_kv_stage += 1;
                if (mma_kv_stage == 3) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                unsigned int first_pv = 1;
                #pragma unroll 1
                for (unsigned int n_iter_3 = 0; n_iter_3 < num_n_blocks_2 - 1; n_iter_3++) {
                    unsigned int v_stage = mma_kv_stage;
                    unsigned int v_phase = mma_kv_phase;
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 3) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
                    int first_pv_flag = first_pv;
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    int _mma_b_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024);
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
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2 + 16], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_2), "r"(tmem_softmax_0), "r"(((first_pv_flag) ? 0 : 1)));
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                    _phase_p_full_2_0 ^= 1;
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024);
                    mma_ts_step(tmem_output_0, tmem_softmax_0 + 24, _mma_b_lo_3 + 768, 0x40004040, 136380432, 1);
                    unsigned int k_stage = mma_kv_stage;
                    unsigned int k_phase = mma_kv_phase;
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 3) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
                    int _mma_a_lo_4 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
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
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_scores_0), "r"(0));
                    elect_commit(s_full_addr);
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024);
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
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2 + 16], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_5), "r"(tmem_softmax_1), "r"(((first_pv_flag) ? 0 : 1)));
                    mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                    _phase_p_full_2_1 ^= 1;
                    int _mma_b_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024);
                    mma_ts_step(tmem_output_1, tmem_softmax_1 + 24, _mma_b_lo_6 + 768, 0x40004040, 136380432, 1);
                    elect_commit(kv_empty_addr + (v_stage) * 8);
                    int _mma_a_lo_7 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
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
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_scores_1), "r"(0));
                    elect_commit(s_full_addr + 8);
                    elect_commit(kv_empty_addr + (k_stage) * 8);
                    first_pv = 0;
                }
                elect_commit(q_empty_addr);
                mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase);
                int first_pv_flag_1 = first_pv;
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                int _mma_b_lo_8 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 1024);
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
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2 + 16], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_8), "r"(tmem_softmax_0), "r"(((first_pv_flag_1) ? 0 : 1)));
                mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                _phase_p_full_2_0 ^= 1;
                int _mma_b_lo_9 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 1024);
                mma_ts_step(tmem_output_0, tmem_softmax_0 + 24, _mma_b_lo_9 + 768, 0x40004040, 136380432, 1);
                mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                _phase_p_full_1 ^= 1;
                int _mma_b_lo_10 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 1024);
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
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%2 + 16], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_10), "r"(tmem_softmax_1), "r"(((first_pv_flag_1) ? 0 : 1)));
                mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                _phase_p_full_2_1 ^= 1;
                int _mma_b_lo_11 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 1024);
                mma_ts_step(tmem_output_1, tmem_softmax_1 + 24, _mma_b_lo_11 + 768, 0x40004040, 136380432, 1);
                elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                mma_kv_stage += 1;
                if (mma_kv_stage == 3) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                elect_commit2(o_full_addr, o_full_addr + 8);
            }
        }
    // ---- Role: page_offsets ----
    } else if (warp == 13) {
        { // page_offsets_main
            unsigned int total_tiles_3 = NUM_M_BLOCKS * total_bh;
            unsigned int max_rounds_3 = total_tiles_3 + 1;
            {
                max_rounds_3 = 1;
            }
            unsigned int po_stage = 0;
            unsigned int _phase_work_id_full_0_3 = 0;
            unsigned int _phase_page_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_iter_3 = 0; tile_iter_3 < max_rounds_3; tile_iter_3++) {
                unsigned int tile_idx_3;
                {
                    tile_idx_3 = blockIdx.x;
                }
                if (tile_idx_3 >= total_tiles_3) {
                    break;
                }
                unsigned int m_block_3;
                unsigned int bh_3;
                {
                    unsigned int l2section_3 = L2_SWIZZLE * NUM_M_BLOCKS;
                    unsigned int section_3 = tile_idx_3 / l2section_3;
                    unsigned int l2_mod_3 = tile_idx_3 % l2section_3;
                    m_block_3 = l2_mod_3 / (unsigned int)L2_SWIZZLE;
                    bh_3 = section_3 * (unsigned int)L2_SWIZZLE + l2_mod_3 % (unsigned int)L2_SWIZZLE;
                    m_block_3 = (unsigned int)(NUM_M_BLOCKS - 1) - m_block_3;
                }
                int seqlen_kv_bh_3 = UNIFORM_KV_LEN;
                int seqlen_q_bh_3 = UNIFORM_Q_LEN;
                {
                    seqlen_kv_bh_3 = seq_lens_kv[bh_3];
                }
                {
                    seqlen_q_bh_3 = seq_lens_q[bh_3];
                }
                unsigned int num_n_blocks_3 = (seqlen_kv_bh_3 + BLOCK_N - 1) / BLOCK_N;
                {
                    int kv_shift_4 = seqlen_kv_bh_3 - seqlen_q_bh_3;
                    unsigned int max_n_3 = (m_block_3 * 2 + 2) * (unsigned int)TOK_PER_STAGE + (unsigned int)kv_shift_4;
                    if (max_n_3 < (unsigned int)seqlen_kv_bh_3) {
                        num_n_blocks_3 = (max_n_3 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                    }
                }
                unsigned int b_idx = bh_3 / (unsigned int)(NUM_Q_HEADS / PACK_G);
                int pt_base = b_idx * (unsigned int)page_row_stride;
                int last_slot = (seqlen_kv_bh_3 - 1) / PAGE_SIZE;
                #pragma unroll 1
                for (unsigned int ni = 0; ni < num_n_blocks_3; ni++) {
                    unsigned int n = num_n_blocks_3 - 1 - ni;
                    int base_tok = n * (unsigned int)BLOCK_N;
                    mbarrier_wait(page_empty_addr + (po_stage) * 8, _phase_page_empty);
                    int pg_idx = po_stage * 16;
                    int pg_lane = lane;
                    if (pg_lane < 8) {
                        int _min_0 = (((base_tok + pg_lane * PAGE_SIZE) / PAGE_SIZE) < (last_slot) ? ((base_tok + pg_lane * PAGE_SIZE) / PAGE_SIZE) : (last_slot));
                        int page_slot = _min_0;
                        int page_id = page_table_k[pt_base + page_slot];
                        smem_pages[pg_idx + pg_lane] = page_id;
                        smem_pages[pg_idx + 8 + pg_lane] = page_id;
                        mbarrier_arrive(page_full_addr + (po_stage) * 8);
                    }
                    po_stage += 1;
                    if (po_stage == 6) { po_stage = 0; _phase_page_empty ^= 1; }
                }
            }
        }
    // ---- Role: scheduler ----
    } else if (warp == 14) {
        { // scheduler_main
            unsigned int _phase_work_id_empty_0 = 1;
        }
    // ---- Role: load ----
    } else if (warp == 15) {
        { // load_main
            unsigned int total_tiles_4 = NUM_M_BLOCKS * total_bh;
            unsigned int max_rounds_4 = total_tiles_4 + 1;
            {
                max_rounds_4 = 1;
            }
            unsigned int load_pk_stage = 0;
            unsigned int load_po_stage = 0;
            unsigned int _phase_work_id_full_0_4 = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_page_full = 0;
            unsigned int _phase_pk_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_iter_4 = 0; tile_iter_4 < max_rounds_4; tile_iter_4++) {
                unsigned int tile_idx_4;
                {
                    tile_idx_4 = blockIdx.x;
                }
                if (tile_idx_4 >= total_tiles_4) {
                    break;
                }
                unsigned int m_block_4;
                unsigned int bh_4;
                {
                    unsigned int l2section_4 = L2_SWIZZLE * NUM_M_BLOCKS;
                    unsigned int section_4 = tile_idx_4 / l2section_4;
                    unsigned int l2_mod_4 = tile_idx_4 % l2section_4;
                    m_block_4 = l2_mod_4 / (unsigned int)L2_SWIZZLE;
                    bh_4 = section_4 * (unsigned int)L2_SWIZZLE + l2_mod_4 % (unsigned int)L2_SWIZZLE;
                    m_block_4 = (unsigned int)(NUM_M_BLOCKS - 1) - m_block_4;
                }
                int seqlen_kv_bh_4 = UNIFORM_KV_LEN;
                int seqlen_q_bh_4 = UNIFORM_Q_LEN;
                {
                    seqlen_kv_bh_4 = seq_lens_kv[bh_4];
                }
                {
                    seqlen_q_bh_4 = seq_lens_q[bh_4];
                }
                unsigned int num_n_blocks_4 = (seqlen_kv_bh_4 + BLOCK_N - 1) / BLOCK_N;
                {
                    int kv_shift_5 = seqlen_kv_bh_4 - seqlen_q_bh_4;
                    unsigned int max_n_4 = (m_block_4 * 2 + 2) * (unsigned int)TOK_PER_STAGE + (unsigned int)kv_shift_5;
                    if (max_n_4 < (unsigned int)seqlen_kv_bh_4) {
                        num_n_blocks_4 = (max_n_4 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                    }
                }
                int unit_1 = bh_4 % (unsigned int)(NUM_Q_HEADS / PACK_G);
                int kv_head = unit_1 * PACK_G / HEADS_PER_GROUP;
                int tok_q0 = (unsigned int)cu_seq_lens_q[bh_4] + m_block_4 * 2 * (unsigned int)TOK_PER_STAGE;
                int tok_q1 = tok_q0 + TOK_PER_STAGE;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, TOK_PER_STAGE * PACK_G * 128);
                    tma_5d_gmem2smem(smem_q0_addr, Q, 0, 0, tok_q0, 0, unit_1, q_full_addr);
                }
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + 8, TOK_PER_STAGE * PACK_G * 128);
                    tma_5d_gmem2smem(smem_q1_addr, Q, 0, 0, tok_q1, 0, unit_1, q_full_addr + 8);
                }
                #pragma unroll 1
                for (unsigned int ni_1 = 0; ni_1 < num_n_blocks_4; ni_1++) {
                    unsigned int n_1 = num_n_blocks_4 - 1 - ni_1;
                    int base_tok_1 = n_1 * (unsigned int)BLOCK_N;
                    int w0 = base_tok_1 % PAGE_SIZE;
                    int w1 = (base_tok_1 + 16) % PAGE_SIZE;
                    int w2 = (base_tok_1 + 32) % PAGE_SIZE;
                    int w3 = (base_tok_1 + 48) % PAGE_SIZE;
                    int w4 = (base_tok_1 + 64) % PAGE_SIZE;
                    int w5 = (base_tok_1 + 80) % PAGE_SIZE;
                    int w6 = (base_tok_1 + 96) % PAGE_SIZE;
                    int w7 = (base_tok_1 + 112) % PAGE_SIZE;
                    mbarrier_wait(page_full_addr + (load_po_stage) * 8, _phase_page_full);
                    int pg_idx_1 = load_po_stage * 16;
                    mbarrier_wait(pk_empty_addr + (load_pk_stage) * 8, _phase_pk_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(pk_full_addr + (load_pk_stage) * 8, 9216);
                        #pragma unroll
                        for (int chunk = 0; chunk < 8; chunk++) {
                            int pg_k = smem_pages[pg_idx_1];
                            int w_k = w0;
                            if (chunk == 1) {
                                pg_k = smem_pages[pg_idx_1 + 1];
                                w_k = w1;
                            } else if (chunk == 2) {
                                pg_k = smem_pages[pg_idx_1 + 2];
                                w_k = w2;
                            } else {
                                if (chunk == 3) {
                                    pg_k = smem_pages[pg_idx_1 + 3];
                                    w_k = w3;
                                } else if (chunk == 4) {
                                    pg_k = smem_pages[pg_idx_1 + 4];
                                    w_k = w4;
                                } else {
                                    if (chunk == 5) {
                                        pg_k = smem_pages[pg_idx_1 + 5];
                                        w_k = w5;
                                    } else if (chunk == 6) {
                                        pg_k = smem_pages[pg_idx_1 + 6];
                                        w_k = w6;
                                    } else {
                                        if (chunk == 7) {
                                            pg_k = smem_pages[pg_idx_1 + 7];
                                            w_k = w7;
                                        }
                                    }
                                }
                            }
                            tma_5d_gmem2smem(smem_pk_addr + load_pk_stage * 9216 + (unsigned int)(chunk * 1024), Kp, 0, w_k, 0, kv_head, pg_k, pk_full_addr + (load_pk_stage) * 8);
                            tma_3d_gmem2smem(smem_pk_addr + load_pk_stage * 9216 + (unsigned int)(8192 + chunk * 128), Ksf, w_k * 8, kv_head, pg_k, pk_full_addr + (load_pk_stage) * 8);
                        }
                    }
                    load_pk_stage += 1;
                    if (load_pk_stage == 2) { load_pk_stage = 0; _phase_pk_empty ^= 1; }
                    mbarrier_wait(pk_empty_addr + (load_pk_stage) * 8, _phase_pk_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(pk_full_addr + (load_pk_stage) * 8, 9216);
                        #pragma unroll
                        for (int chunk_1 = 0; chunk_1 < 8; chunk_1++) {
                            int pg_v = smem_pages[pg_idx_1 + 8];
                            int w_v = w0;
                            if (chunk_1 == 1) {
                                pg_v = smem_pages[pg_idx_1 + 9];
                                w_v = w1;
                            } else if (chunk_1 == 2) {
                                pg_v = smem_pages[pg_idx_1 + 10];
                                w_v = w2;
                            } else {
                                if (chunk_1 == 3) {
                                    pg_v = smem_pages[pg_idx_1 + 11];
                                    w_v = w3;
                                } else if (chunk_1 == 4) {
                                    pg_v = smem_pages[pg_idx_1 + 12];
                                    w_v = w4;
                                } else {
                                    if (chunk_1 == 5) {
                                        pg_v = smem_pages[pg_idx_1 + 13];
                                        w_v = w5;
                                    } else if (chunk_1 == 6) {
                                        pg_v = smem_pages[pg_idx_1 + 14];
                                        w_v = w6;
                                    } else {
                                        if (chunk_1 == 7) {
                                            pg_v = smem_pages[pg_idx_1 + 15];
                                            w_v = w7;
                                        }
                                    }
                                }
                            }
                            tma_5d_gmem2smem(smem_pk_addr + load_pk_stage * 9216 + (unsigned int)(chunk_1 * 1024), Vp, 0, w_v, 0, kv_head, pg_v, pk_full_addr + (load_pk_stage) * 8);
                            tma_3d_gmem2smem(smem_pk_addr + load_pk_stage * 9216 + (unsigned int)(8192 + chunk_1 * 128), Vsf, w_v * 8, kv_head, pg_v, pk_full_addr + (load_pk_stage) * 8);
                        }
                    }
                    load_pk_stage += 1;
                    if (load_pk_stage == 2) { load_pk_stage = 0; _phase_pk_empty ^= 1; }
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
