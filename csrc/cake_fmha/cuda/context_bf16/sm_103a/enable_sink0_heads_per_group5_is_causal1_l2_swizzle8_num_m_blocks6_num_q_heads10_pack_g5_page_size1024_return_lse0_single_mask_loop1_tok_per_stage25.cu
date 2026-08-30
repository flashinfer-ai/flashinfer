/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

typedef signed char        int8_t;
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

#define CAKE_FMHA_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_SCORES_0_OFFSET 0
#define TMEM_SOFTMAX_0_OFFSET 64
#define TMEM_SCORES_1_OFFSET 128
#define TMEM_SOFTMAX_1_OFFSET 192
#define TMEM_OUTPUT_0_OFFSET 256
#define TMEM_OUTPUT_1_OFFSET 384
#define NUM_Q_STAGES 2
#define NUM_KV_STAGES 3
#define NUM_PO_STAGES 6
#define NUM_ACC_STAGES 1
#define NUM_ORDER_P01_STAGES 1
#define SMEM_SSCALE_OFF 1024
#define SMEM_SSCALE_STAGE_BYTES 3072
#define SMEM_SSCALE_STRIDE 3072
#define SMEM_SMEM_Q0_OFF 4096
#define SMEM_SMEM_Q0_STAGE_BYTES 32768
#define SMEM_SMEM_Q0_STRIDE 32768
#define SMEM_SMEM_Q1_OFF 36864
#define SMEM_SMEM_Q1_STAGE_BYTES 32768
#define SMEM_SMEM_Q1_STRIDE 32768
#define SMEM_SMEM_KV_OFF 69632
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 69632
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_PAGES_OFF 167936
#define SMEM_SMEM_PAGES_STAGE_BYTES 384
#define SMEM_SMEM_PAGES_STRIDE 384
#define SMEM_WORK_ID_SLOT_OFF 168320
#define SMEM_WORK_ID_SLOT_STAGE_BYTES 4
#define SMEM_WORK_ID_SLOT_STRIDE 4
#define SMEM_TOTAL 168448
#define IS_CAUSAL 1
#define NUM_M_BLOCKS 6
#define NUM_Q_HEADS 10
#define HEADS_PER_GROUP 5
#define PACK_G 5
#define TOK_PER_STAGE 25
#define L2_SWIZZLE 8
#define STATIC_ONE_TILE 1
#define RETURN_LSE 0
#define ENABLE_SINK 0
#define UNIFORM_Q_LEN 257
#define UNIFORM_KV_LEN 1024
#define SINGLE_MASK_LOOP 1
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128
#define MMA_K 16
#define PAGE_SIZE 1024
#define KV_TOK_CHUNK 128

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

__global__ __launch_bounds__(512, 1) void
kernel_cake_fmha_context_bf16(CakeFmhaTensorMap const* Q, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V, __nv_bfloat16* __restrict__ O_ptr, float* __restrict__ LSE_ptr, float* __restrict__ sinks, int* __restrict__ page_table_k, int* __restrict__ page_table_v, int* __restrict__ seq_lens_q, int* __restrict__ seq_lens_kv, int* __restrict__ cu_seq_lens_q, float softmax_scale_log2, int total_bh, int page_row_stride, int num_ctas, unsigned int* __restrict__ dynamic_counter)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 40)
    #define s_full_addr (mbar_base + 64)
    #define p_full_addr (mbar_base + 80)
    #define p_full_2_addr (mbar_base + 96)
    #define corr_sig_addr (mbar_base + 112)
    #define corr_done_addr (mbar_base + 128)
    #define order_p01_0_addr (mbar_base + 144)
    #define order_p01_1_addr (mbar_base + 152)
    #define o_full_addr (mbar_base + 160)
    #define q_empty_addr (mbar_base + 176)
    #define page_full_addr (mbar_base + 184)
    #define page_empty_addr (mbar_base + 232)
    #define work_id_full_addr (mbar_base + 280)
    #define work_id_empty_addr (mbar_base + 288)

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
    __nv_bfloat16* smem_q0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4096);
    const int smem_q0_addr = smem + 4096;
    __nv_bfloat16* smem_q1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 36864);
    const int smem_q1_addr = smem + 36864;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 69632);
    const int smem_kv_addr = smem + 69632;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 69632);
    const int smem_v_addr = smem + 69632;
    int* smem_pages = reinterpret_cast<int*>(smem_raw + 167936);
    const int smem_pages_addr = smem + 167936;
    unsigned int* work_id_slot = reinterpret_cast<unsigned int*>(smem_raw + 168320);
    const int work_id_slot_addr = smem + 168320;

    // Mbarrier init (16 groups, 37 barriers)
    // Mbarriers at smem_raw[0..296)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'kv' ---
            // kv_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // kv_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 80, 256);
            mbarrier_init(smem + 88, 256);
            // p_full_2: 2 barriers, init_count=128
            mbarrier_init(smem + 96, 128);
            mbarrier_init(smem + 104, 128);
            // corr_sig: 2 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            mbarrier_init(smem + 120, 128);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            // order_p01_0: 1 barriers, init_count=128
            mbarrier_init(smem + 144, 128);
            // order_p01_1: 1 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // --- pipeline 'po' ---
            // page_full: 6 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // page_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // work_id_full: 1 barriers, init_count=1
            mbarrier_init(smem + 280, 1);
            // work_id_empty: 1 barriers, init_count=15
            mbarrier_init(smem + 288, 15);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 296);
    if (warp == 0) {
        int _tmem_hold = smem + 296;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores_0 = taddr;
    const int tmem_softmax_0 = taddr + 64;
    const int tmem_scores_1 = taddr + 128;
    const int tmem_softmax_1 = taddr + 192;
    const int tmem_output_0 = taddr + 256;
    const int tmem_output_1 = taddr + 384;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
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
                float row_max_val = -CAKE_FMHA_INF;
                float row_sum_val = 0.0f;
                unsigned int masked_loop_end = num_masked_iters;
                unsigned int unmasked_loop_start = num_masked_iters;
                {
                    masked_loop_end = 0;
                    unmasked_loop_start = 0;
                }
                #pragma unroll 1
                for (unsigned int n_iter = 0; n_iter < masked_loop_end; n_iter++) {
                    int n_block = num_n_blocks - 1 - n_iter;
                    mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                    _phase_s_full ^= 1;
                    int s_base = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                    int tail_valid = seqlen_kv_bh - n_block * BLOCK_N;
                    int valid_count = causal_row - n_block * BLOCK_N + 1;
                    float _tmem_load_0_max = -CAKE_FMHA_INF;
                    #pragma unroll 1
                    for (int score_chunk_0 = 0; score_chunk_0 < 4; score_chunk_0++) {
                        float _tmem_load_0[32];
                        int chunk_offset_1 = score_chunk_0 * 32;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(s_base + chunk_offset_1)
                            : "memory");
                        #pragma unroll
                        for (int score_lane_2 = 0; score_lane_2 < 32; score_lane_2++) {
                            if (tail_valid <= (chunk_offset_1 + score_lane_2)) _tmem_load_0[score_lane_2] = -CAKE_FMHA_INF;
                        }
                        #pragma unroll
                        for (int score_lane_3 = 0; score_lane_3 < 32; score_lane_3++) {
                            if (valid_count <= (chunk_offset_1 + score_lane_3)) _tmem_load_0[score_lane_3] = -CAKE_FMHA_INF;
                        }
                        float2 _stream_max2_4 = make_float2(-CAKE_FMHA_INF, -CAKE_FMHA_INF);
                        _stream_max2_4.x = max_noftz(_stream_max2_4.x, max_noftz(_tmem_load_0[0], _tmem_load_0[1]));
                        _stream_max2_4.y = max_noftz(_stream_max2_4.y, max_noftz(_tmem_load_0[2], _tmem_load_0[3]));
                        _stream_max2_4.x = max_noftz(_stream_max2_4.x, max_noftz(_tmem_load_0[4], _tmem_load_0[5]));
                        _stream_max2_4.y = max_noftz(_stream_max2_4.y, max_noftz(_tmem_load_0[6], _tmem_load_0[7]));
                        _stream_max2_4.x = max_noftz(_stream_max2_4.x, max_noftz(_tmem_load_0[8], _tmem_load_0[9]));
                        _stream_max2_4.y = max_noftz(_stream_max2_4.y, max_noftz(_tmem_load_0[10], _tmem_load_0[11]));
                        _stream_max2_4.x = max_noftz(_stream_max2_4.x, max_noftz(_tmem_load_0[12], _tmem_load_0[13]));
                        _stream_max2_4.y = max_noftz(_stream_max2_4.y, max_noftz(_tmem_load_0[14], _tmem_load_0[15]));
                        _stream_max2_4.x = max_noftz(_stream_max2_4.x, max_noftz(_tmem_load_0[16], _tmem_load_0[17]));
                        _stream_max2_4.y = max_noftz(_stream_max2_4.y, max_noftz(_tmem_load_0[18], _tmem_load_0[19]));
                        _stream_max2_4.x = max_noftz(_stream_max2_4.x, max_noftz(_tmem_load_0[20], _tmem_load_0[21]));
                        _stream_max2_4.y = max_noftz(_stream_max2_4.y, max_noftz(_tmem_load_0[22], _tmem_load_0[23]));
                        _stream_max2_4.x = max_noftz(_stream_max2_4.x, max_noftz(_tmem_load_0[24], _tmem_load_0[25]));
                        _stream_max2_4.y = max_noftz(_stream_max2_4.y, max_noftz(_tmem_load_0[26], _tmem_load_0[27]));
                        _stream_max2_4.x = max_noftz(_stream_max2_4.x, max_noftz(_tmem_load_0[28], _tmem_load_0[29]));
                        _stream_max2_4.y = max_noftz(_stream_max2_4.y, max_noftz(_tmem_load_0[30], _tmem_load_0[31]));
                        float _stream_chunk_max_5 = row_max_reduce(_stream_max2_4);
                        _tmem_load_0_max = max_noftz(_tmem_load_0_max, _stream_chunk_max_5);
                    }
                    float new_max = _tmem_load_0_max;
                    float _max_0 = max_noftz(new_max, row_max_val);
                    new_max = _max_0;
                    float new_max_scaled = ((new_max == -CAKE_FMHA_INF) ? 0.0f : new_max) * softmax_scale_log2;
                    float acc_scale;
                    float selected_max;
                    float _fma_0 = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max_scaled);
                    if (_fma_0 >= -8.0f) {
                        selected_max = row_max_val;
                        acc_scale = 1.0f;
                        new_max_scaled = ((row_max_val == -CAKE_FMHA_INF) ? 0.0f : row_max_val) * softmax_scale_log2;
                    } else {
                        selected_max = new_max;
                        if (row_max_val > -CAKE_FMHA_INF) {
                            float _exp2_0 = approx_exp2(_fma_0);
                            acc_scale = _exp2_0;
                        } else {
                            acc_scale = 1.0f;
                        }
                    }
                    row_max_val = selected_max;
                    sScale[warp % 4 * 32 + lane + scale_off] = acc_scale;
                    mbarrier_arrive(corr_sig_addr + (stage) * 8);
                    int p_base = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                    if (stage == 0) {
                        mbarrier_wait(order_p01_0_addr, order_p01_phase);
                    } else {
                        mbarrier_wait(order_p01_1_addr, order_p01_phase);
                    }
                    uint32_t _tmem_load_0_bf16_2[16];
                    float block_sum_6;
                    {
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(s_base + 96));
                        #pragma unroll
                        for (int score_lane_7 = 0; score_lane_7 < 32; score_lane_7++) {
                            if (tail_valid <= (96 + score_lane_7)) _tmem_load_0[score_lane_7] = -CAKE_FMHA_INF;
                        }
                        #pragma unroll
                        for (int score_lane_8 = 0; score_lane_8 < 32; score_lane_8++) {
                            if (valid_count <= (96 + score_lane_8)) _tmem_load_0[score_lane_8] = -CAKE_FMHA_INF;
                        }
                        const float2 _fma_b2_9 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_10 = {-new_max_scaled, -new_max_scaled};
                        float2 _fma_pair_11 = fma_f32x2(make_float2(_tmem_load_0[0], _tmem_load_0[1]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[0] = _fma_pair_11.x;
                        _tmem_load_0[1] = _fma_pair_11.y;
                        float2 _fma_pair_12 = fma_f32x2(make_float2(_tmem_load_0[2], _tmem_load_0[3]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[2] = _fma_pair_12.x;
                        _tmem_load_0[3] = _fma_pair_12.y;
                        float2 _fma_pair_13 = fma_f32x2(make_float2(_tmem_load_0[4], _tmem_load_0[5]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[4] = _fma_pair_13.x;
                        _tmem_load_0[5] = _fma_pair_13.y;
                        float2 _fma_pair_14 = fma_f32x2(make_float2(_tmem_load_0[6], _tmem_load_0[7]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[6] = _fma_pair_14.x;
                        _tmem_load_0[7] = _fma_pair_14.y;
                        float2 _fma_pair_15 = fma_f32x2(make_float2(_tmem_load_0[8], _tmem_load_0[9]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[8] = _fma_pair_15.x;
                        _tmem_load_0[9] = _fma_pair_15.y;
                        float2 _fma_pair_16 = fma_f32x2(make_float2(_tmem_load_0[10], _tmem_load_0[11]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[10] = _fma_pair_16.x;
                        _tmem_load_0[11] = _fma_pair_16.y;
                        float2 _fma_pair_17 = fma_f32x2(make_float2(_tmem_load_0[12], _tmem_load_0[13]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[12] = _fma_pair_17.x;
                        _tmem_load_0[13] = _fma_pair_17.y;
                        float2 _fma_pair_18 = fma_f32x2(make_float2(_tmem_load_0[14], _tmem_load_0[15]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[14] = _fma_pair_18.x;
                        _tmem_load_0[15] = _fma_pair_18.y;
                        float2 _fma_pair_19 = fma_f32x2(make_float2(_tmem_load_0[16], _tmem_load_0[17]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[16] = _fma_pair_19.x;
                        _tmem_load_0[17] = _fma_pair_19.y;
                        float2 _fma_pair_20 = fma_f32x2(make_float2(_tmem_load_0[18], _tmem_load_0[19]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[18] = _fma_pair_20.x;
                        _tmem_load_0[19] = _fma_pair_20.y;
                        float2 _fma_pair_21 = fma_f32x2(make_float2(_tmem_load_0[20], _tmem_load_0[21]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[20] = _fma_pair_21.x;
                        _tmem_load_0[21] = _fma_pair_21.y;
                        float2 _fma_pair_22 = fma_f32x2(make_float2(_tmem_load_0[22], _tmem_load_0[23]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[22] = _fma_pair_22.x;
                        _tmem_load_0[23] = _fma_pair_22.y;
                        float2 _fma_pair_23 = fma_f32x2(make_float2(_tmem_load_0[24], _tmem_load_0[25]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[24] = _fma_pair_23.x;
                        _tmem_load_0[25] = _fma_pair_23.y;
                        float2 _fma_pair_24 = fma_f32x2(make_float2(_tmem_load_0[26], _tmem_load_0[27]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[26] = _fma_pair_24.x;
                        _tmem_load_0[27] = _fma_pair_24.y;
                        float2 _fma_pair_25 = fma_f32x2(make_float2(_tmem_load_0[28], _tmem_load_0[29]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[28] = _fma_pair_25.x;
                        _tmem_load_0[29] = _fma_pair_25.y;
                        float2 _fma_pair_26 = fma_f32x2(make_float2(_tmem_load_0[30], _tmem_load_0[31]), _fma_b2_9, _fma_c2_10);
                        _tmem_load_0[30] = _fma_pair_26.x;
                        _tmem_load_0[31] = _fma_pair_26.y;
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                        }
                        float2 block_sum_6_pair = make_float2(0.0f, 0.0f);
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[4], _tmem_load_0[5]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[6], _tmem_load_0[7]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[8], _tmem_load_0[9]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[10], _tmem_load_0[11]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[12], _tmem_load_0[13]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[14], _tmem_load_0[15]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[16], _tmem_load_0[17]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[18], _tmem_load_0[19]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[20], _tmem_load_0[21]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[22], _tmem_load_0[23]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[24], _tmem_load_0[25]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[26], _tmem_load_0[27]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[28], _tmem_load_0[29]));
                        block_sum_6_pair = add_f32x2(block_sum_6_pair, make_float2(_tmem_load_0[30], _tmem_load_0[31]));
                        block_sum_6 = block_sum_6_pair.x + block_sum_6_pair.y;
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            _tmem_load_0_bf16_2[_lp] = *(uint32_t*)&_bf2;
                        }
                    }
                    #pragma unroll 1
                    for (int reverse_index_27 = 0; reverse_index_27 < 3; reverse_index_27++) {
                        float _tmem_load_0[32];
                        int p_chunk_28 = 2 - reverse_index_27;
                        int chunk_offset_29 = p_chunk_28 * 32;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(s_base + chunk_offset_29)
                            : "memory");
                        #pragma unroll
                        for (int score_lane_31 = 0; score_lane_31 < 32; score_lane_31++) {
                            if (tail_valid <= (chunk_offset_29 + score_lane_31)) _tmem_load_0[score_lane_31] = -CAKE_FMHA_INF;
                        }
                        #pragma unroll
                        for (int score_lane_32 = 0; score_lane_32 < 32; score_lane_32++) {
                            if (valid_count <= (chunk_offset_29 + score_lane_32)) _tmem_load_0[score_lane_32] = -CAKE_FMHA_INF;
                        }
                        const float2 _fma_b2_33 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_34 = {-new_max_scaled, -new_max_scaled};
                        float2 _fma_pair_35 = fma_f32x2(make_float2(_tmem_load_0[0], _tmem_load_0[1]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[0] = _fma_pair_35.x;
                        _tmem_load_0[1] = _fma_pair_35.y;
                        float2 _fma_pair_36 = fma_f32x2(make_float2(_tmem_load_0[2], _tmem_load_0[3]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[2] = _fma_pair_36.x;
                        _tmem_load_0[3] = _fma_pair_36.y;
                        float2 _fma_pair_37 = fma_f32x2(make_float2(_tmem_load_0[4], _tmem_load_0[5]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[4] = _fma_pair_37.x;
                        _tmem_load_0[5] = _fma_pair_37.y;
                        float2 _fma_pair_38 = fma_f32x2(make_float2(_tmem_load_0[6], _tmem_load_0[7]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[6] = _fma_pair_38.x;
                        _tmem_load_0[7] = _fma_pair_38.y;
                        float2 _fma_pair_39 = fma_f32x2(make_float2(_tmem_load_0[8], _tmem_load_0[9]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[8] = _fma_pair_39.x;
                        _tmem_load_0[9] = _fma_pair_39.y;
                        float2 _fma_pair_40 = fma_f32x2(make_float2(_tmem_load_0[10], _tmem_load_0[11]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[10] = _fma_pair_40.x;
                        _tmem_load_0[11] = _fma_pair_40.y;
                        float2 _fma_pair_41 = fma_f32x2(make_float2(_tmem_load_0[12], _tmem_load_0[13]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[12] = _fma_pair_41.x;
                        _tmem_load_0[13] = _fma_pair_41.y;
                        float2 _fma_pair_42 = fma_f32x2(make_float2(_tmem_load_0[14], _tmem_load_0[15]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[14] = _fma_pair_42.x;
                        _tmem_load_0[15] = _fma_pair_42.y;
                        float2 _fma_pair_43 = fma_f32x2(make_float2(_tmem_load_0[16], _tmem_load_0[17]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[16] = _fma_pair_43.x;
                        _tmem_load_0[17] = _fma_pair_43.y;
                        float2 _fma_pair_44 = fma_f32x2(make_float2(_tmem_load_0[18], _tmem_load_0[19]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[18] = _fma_pair_44.x;
                        _tmem_load_0[19] = _fma_pair_44.y;
                        float2 _fma_pair_45 = fma_f32x2(make_float2(_tmem_load_0[20], _tmem_load_0[21]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[20] = _fma_pair_45.x;
                        _tmem_load_0[21] = _fma_pair_45.y;
                        float2 _fma_pair_46 = fma_f32x2(make_float2(_tmem_load_0[22], _tmem_load_0[23]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[22] = _fma_pair_46.x;
                        _tmem_load_0[23] = _fma_pair_46.y;
                        float2 _fma_pair_47 = fma_f32x2(make_float2(_tmem_load_0[24], _tmem_load_0[25]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[24] = _fma_pair_47.x;
                        _tmem_load_0[25] = _fma_pair_47.y;
                        float2 _fma_pair_48 = fma_f32x2(make_float2(_tmem_load_0[26], _tmem_load_0[27]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[26] = _fma_pair_48.x;
                        _tmem_load_0[27] = _fma_pair_48.y;
                        float2 _fma_pair_49 = fma_f32x2(make_float2(_tmem_load_0[28], _tmem_load_0[29]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[28] = _fma_pair_49.x;
                        _tmem_load_0[29] = _fma_pair_49.y;
                        float2 _fma_pair_50 = fma_f32x2(make_float2(_tmem_load_0[30], _tmem_load_0[31]), _fma_b2_33, _fma_c2_34);
                        _tmem_load_0[30] = _fma_pair_50.x;
                        _tmem_load_0[31] = _fma_pair_50.y;
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                        }
                        float2 chunk_sum_51_pair = make_float2(0.0f, 0.0f);
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[4], _tmem_load_0[5]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[6], _tmem_load_0[7]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[8], _tmem_load_0[9]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[10], _tmem_load_0[11]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[12], _tmem_load_0[13]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[14], _tmem_load_0[15]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[16], _tmem_load_0[17]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[18], _tmem_load_0[19]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[20], _tmem_load_0[21]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[22], _tmem_load_0[23]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[24], _tmem_load_0[25]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[26], _tmem_load_0[27]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[28], _tmem_load_0[29]));
                        chunk_sum_51_pair = add_f32x2(chunk_sum_51_pair, make_float2(_tmem_load_0[30], _tmem_load_0[31]));
                        float chunk_sum_51 = chunk_sum_51_pair.x + chunk_sum_51_pair.y;
                        block_sum_6 += chunk_sum_51;
                        uint32_t _stream_packed_30[16];
                        __nv_bfloat162 _stream_bf2_52 = __float22bfloat162_rn(make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                        _stream_packed_30[0] = reinterpret_cast<uint32_t&>(_stream_bf2_52);
                        __nv_bfloat162 _stream_bf2_53 = __float22bfloat162_rn(make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                        _stream_packed_30[1] = reinterpret_cast<uint32_t&>(_stream_bf2_53);
                        __nv_bfloat162 _stream_bf2_54 = __float22bfloat162_rn(make_float2(_tmem_load_0[4], _tmem_load_0[5]));
                        _stream_packed_30[2] = reinterpret_cast<uint32_t&>(_stream_bf2_54);
                        __nv_bfloat162 _stream_bf2_55 = __float22bfloat162_rn(make_float2(_tmem_load_0[6], _tmem_load_0[7]));
                        _stream_packed_30[3] = reinterpret_cast<uint32_t&>(_stream_bf2_55);
                        __nv_bfloat162 _stream_bf2_56 = __float22bfloat162_rn(make_float2(_tmem_load_0[8], _tmem_load_0[9]));
                        _stream_packed_30[4] = reinterpret_cast<uint32_t&>(_stream_bf2_56);
                        __nv_bfloat162 _stream_bf2_57 = __float22bfloat162_rn(make_float2(_tmem_load_0[10], _tmem_load_0[11]));
                        _stream_packed_30[5] = reinterpret_cast<uint32_t&>(_stream_bf2_57);
                        __nv_bfloat162 _stream_bf2_58 = __float22bfloat162_rn(make_float2(_tmem_load_0[12], _tmem_load_0[13]));
                        _stream_packed_30[6] = reinterpret_cast<uint32_t&>(_stream_bf2_58);
                        __nv_bfloat162 _stream_bf2_59 = __float22bfloat162_rn(make_float2(_tmem_load_0[14], _tmem_load_0[15]));
                        _stream_packed_30[7] = reinterpret_cast<uint32_t&>(_stream_bf2_59);
                        __nv_bfloat162 _stream_bf2_60 = __float22bfloat162_rn(make_float2(_tmem_load_0[16], _tmem_load_0[17]));
                        _stream_packed_30[8] = reinterpret_cast<uint32_t&>(_stream_bf2_60);
                        __nv_bfloat162 _stream_bf2_61 = __float22bfloat162_rn(make_float2(_tmem_load_0[18], _tmem_load_0[19]));
                        _stream_packed_30[9] = reinterpret_cast<uint32_t&>(_stream_bf2_61);
                        __nv_bfloat162 _stream_bf2_62 = __float22bfloat162_rn(make_float2(_tmem_load_0[20], _tmem_load_0[21]));
                        _stream_packed_30[10] = reinterpret_cast<uint32_t&>(_stream_bf2_62);
                        __nv_bfloat162 _stream_bf2_63 = __float22bfloat162_rn(make_float2(_tmem_load_0[22], _tmem_load_0[23]));
                        _stream_packed_30[11] = reinterpret_cast<uint32_t&>(_stream_bf2_63);
                        __nv_bfloat162 _stream_bf2_64 = __float22bfloat162_rn(make_float2(_tmem_load_0[24], _tmem_load_0[25]));
                        _stream_packed_30[12] = reinterpret_cast<uint32_t&>(_stream_bf2_64);
                        __nv_bfloat162 _stream_bf2_65 = __float22bfloat162_rn(make_float2(_tmem_load_0[26], _tmem_load_0[27]));
                        _stream_packed_30[13] = reinterpret_cast<uint32_t&>(_stream_bf2_65);
                        __nv_bfloat162 _stream_bf2_66 = __float22bfloat162_rn(make_float2(_tmem_load_0[28], _tmem_load_0[29]));
                        _stream_packed_30[14] = reinterpret_cast<uint32_t&>(_stream_bf2_66);
                        __nv_bfloat162 _stream_bf2_67 = __float22bfloat162_rn(make_float2(_tmem_load_0[30], _tmem_load_0[31]));
                        _stream_packed_30[15] = reinterpret_cast<uint32_t&>(_stream_bf2_67);
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_base + p_chunk_28 * 16), "r"(_stream_packed_30[0]), "r"(_stream_packed_30[1]), "r"(_stream_packed_30[2]), "r"(_stream_packed_30[3]), "r"(_stream_packed_30[4]), "r"(_stream_packed_30[5]), "r"(_stream_packed_30[6]), "r"(_stream_packed_30[7]), "r"(_stream_packed_30[8]), "r"(_stream_packed_30[9]), "r"(_stream_packed_30[10]), "r"(_stream_packed_30[11]), "r"(_stream_packed_30[12]), "r"(_stream_packed_30[13]), "r"(_stream_packed_30[14]), "r"(_stream_packed_30[15]));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + (stage) * 8);
                    if (stage == 0) {
                        mbarrier_arrive(order_p01_1_addr);
                    } else {
                        mbarrier_arrive(order_p01_0_addr);
                    }
                    order_p01_phase ^= 1;
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base + 48), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_2_addr + (stage) * 8);
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                    float _tmem_load_0_sum = block_sum_6;
                    row_sum_val = row_sum_val * acc_scale + _tmem_load_0_sum;
                }
                #pragma unroll 1
                for (unsigned int n_iter_1 = unmasked_loop_start; n_iter_1 < num_n_blocks; n_iter_1++) {
                    int n_block_1 = num_n_blocks - 1 - n_iter_1;
                    mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                    _phase_s_full ^= 1;
                    int s_base_1 = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_1[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(s_base_1));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(s_base_1 + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[64]), "=f"(_tmem_load_1[65]), "=f"(_tmem_load_1[66]), "=f"(_tmem_load_1[67]), "=f"(_tmem_load_1[68]), "=f"(_tmem_load_1[69]), "=f"(_tmem_load_1[70]), "=f"(_tmem_load_1[71]), "=f"(_tmem_load_1[72]), "=f"(_tmem_load_1[73]), "=f"(_tmem_load_1[74]), "=f"(_tmem_load_1[75]), "=f"(_tmem_load_1[76]), "=f"(_tmem_load_1[77]), "=f"(_tmem_load_1[78]), "=f"(_tmem_load_1[79]), "=f"(_tmem_load_1[80]), "=f"(_tmem_load_1[81]), "=f"(_tmem_load_1[82]), "=f"(_tmem_load_1[83]), "=f"(_tmem_load_1[84]), "=f"(_tmem_load_1[85]), "=f"(_tmem_load_1[86]), "=f"(_tmem_load_1[87]), "=f"(_tmem_load_1[88]), "=f"(_tmem_load_1[89]), "=f"(_tmem_load_1[90]), "=f"(_tmem_load_1[91]), "=f"(_tmem_load_1[92]), "=f"(_tmem_load_1[93]), "=f"(_tmem_load_1[94]), "=f"(_tmem_load_1[95])
                        : "r"(s_base_1 + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[96]), "=f"(_tmem_load_1[97]), "=f"(_tmem_load_1[98]), "=f"(_tmem_load_1[99]), "=f"(_tmem_load_1[100]), "=f"(_tmem_load_1[101]), "=f"(_tmem_load_1[102]), "=f"(_tmem_load_1[103]), "=f"(_tmem_load_1[104]), "=f"(_tmem_load_1[105]), "=f"(_tmem_load_1[106]), "=f"(_tmem_load_1[107]), "=f"(_tmem_load_1[108]), "=f"(_tmem_load_1[109]), "=f"(_tmem_load_1[110]), "=f"(_tmem_load_1[111]), "=f"(_tmem_load_1[112]), "=f"(_tmem_load_1[113]), "=f"(_tmem_load_1[114]), "=f"(_tmem_load_1[115]), "=f"(_tmem_load_1[116]), "=f"(_tmem_load_1[117]), "=f"(_tmem_load_1[118]), "=f"(_tmem_load_1[119]), "=f"(_tmem_load_1[120]), "=f"(_tmem_load_1[121]), "=f"(_tmem_load_1[122]), "=f"(_tmem_load_1[123]), "=f"(_tmem_load_1[124]), "=f"(_tmem_load_1[125]), "=f"(_tmem_load_1[126]), "=f"(_tmem_load_1[127])
                        : "r"(s_base_1 + 96));
                    int tail_valid_1 = seqlen_kv_bh - n_block_1 * BLOCK_N;
                    {
                        int valid_count_1 = tail_valid_1;
                        if (num_masked_iters > n_iter_1) {
                            int causal_valid = causal_row - n_block_1 * BLOCK_N + 1;
                            if (causal_valid < valid_count_1) {
                                valid_count_1 = causal_valid;
                            }
                        }
                        if (valid_count_1 < BLOCK_N) {
                            uint32_t _slice_lo_mask_8;
                            {
                                int _lim_68 = valid_count_1;
                                if (_lim_68 <= 0) { _slice_lo_mask_8 = 0u; }
                                else if (_lim_68 >= 32) { _slice_lo_mask_8 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_8) : "r"(_lim_68));
                                }
                            }
                            if (!(_slice_lo_mask_8 & (1u << 0))) _tmem_load_1[0] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 1))) _tmem_load_1[1] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 2))) _tmem_load_1[2] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 3))) _tmem_load_1[3] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 4))) _tmem_load_1[4] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 5))) _tmem_load_1[5] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 6))) _tmem_load_1[6] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 7))) _tmem_load_1[7] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 8))) _tmem_load_1[8] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 9))) _tmem_load_1[9] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 10))) _tmem_load_1[10] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 11))) _tmem_load_1[11] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 12))) _tmem_load_1[12] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 13))) _tmem_load_1[13] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 14))) _tmem_load_1[14] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 15))) _tmem_load_1[15] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 16))) _tmem_load_1[16] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 17))) _tmem_load_1[17] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 18))) _tmem_load_1[18] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 19))) _tmem_load_1[19] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 20))) _tmem_load_1[20] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 21))) _tmem_load_1[21] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 22))) _tmem_load_1[22] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 23))) _tmem_load_1[23] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 24))) _tmem_load_1[24] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 25))) _tmem_load_1[25] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 26))) _tmem_load_1[26] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 27))) _tmem_load_1[27] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 28))) _tmem_load_1[28] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 29))) _tmem_load_1[29] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 30))) _tmem_load_1[30] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_8 & (1u << 31))) _tmem_load_1[31] = -CAKE_FMHA_INF;
                            uint32_t _slice_lo_mask_9;
                            {
                                int _lim_69 = valid_count_1 - 32;
                                if (_lim_69 <= 0) { _slice_lo_mask_9 = 0u; }
                                else if (_lim_69 >= 32) { _slice_lo_mask_9 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_9) : "r"(_lim_69));
                                }
                            }
                            if (!(_slice_lo_mask_9 & (1u << 0))) _tmem_load_1[32] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 1))) _tmem_load_1[33] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 2))) _tmem_load_1[34] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 3))) _tmem_load_1[35] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 4))) _tmem_load_1[36] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 5))) _tmem_load_1[37] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 6))) _tmem_load_1[38] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 7))) _tmem_load_1[39] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 8))) _tmem_load_1[40] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 9))) _tmem_load_1[41] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 10))) _tmem_load_1[42] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 11))) _tmem_load_1[43] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 12))) _tmem_load_1[44] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 13))) _tmem_load_1[45] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 14))) _tmem_load_1[46] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 15))) _tmem_load_1[47] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 16))) _tmem_load_1[48] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 17))) _tmem_load_1[49] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 18))) _tmem_load_1[50] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 19))) _tmem_load_1[51] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 20))) _tmem_load_1[52] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 21))) _tmem_load_1[53] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 22))) _tmem_load_1[54] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 23))) _tmem_load_1[55] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 24))) _tmem_load_1[56] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 25))) _tmem_load_1[57] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 26))) _tmem_load_1[58] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 27))) _tmem_load_1[59] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 28))) _tmem_load_1[60] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 29))) _tmem_load_1[61] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 30))) _tmem_load_1[62] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_9 & (1u << 31))) _tmem_load_1[63] = -CAKE_FMHA_INF;
                            uint32_t _slice_lo_mask_10;
                            {
                                int _lim_70 = valid_count_1 - 64;
                                if (_lim_70 <= 0) { _slice_lo_mask_10 = 0u; }
                                else if (_lim_70 >= 32) { _slice_lo_mask_10 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_10) : "r"(_lim_70));
                                }
                            }
                            if (!(_slice_lo_mask_10 & (1u << 0))) _tmem_load_1[64] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 1))) _tmem_load_1[65] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 2))) _tmem_load_1[66] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 3))) _tmem_load_1[67] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 4))) _tmem_load_1[68] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 5))) _tmem_load_1[69] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 6))) _tmem_load_1[70] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 7))) _tmem_load_1[71] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 8))) _tmem_load_1[72] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 9))) _tmem_load_1[73] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 10))) _tmem_load_1[74] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 11))) _tmem_load_1[75] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 12))) _tmem_load_1[76] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 13))) _tmem_load_1[77] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 14))) _tmem_load_1[78] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 15))) _tmem_load_1[79] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 16))) _tmem_load_1[80] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 17))) _tmem_load_1[81] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 18))) _tmem_load_1[82] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 19))) _tmem_load_1[83] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 20))) _tmem_load_1[84] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 21))) _tmem_load_1[85] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 22))) _tmem_load_1[86] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 23))) _tmem_load_1[87] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 24))) _tmem_load_1[88] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 25))) _tmem_load_1[89] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 26))) _tmem_load_1[90] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 27))) _tmem_load_1[91] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 28))) _tmem_load_1[92] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 29))) _tmem_load_1[93] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 30))) _tmem_load_1[94] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_10 & (1u << 31))) _tmem_load_1[95] = -CAKE_FMHA_INF;
                            uint32_t _slice_lo_mask_11;
                            {
                                int _lim_71 = valid_count_1 - 96;
                                if (_lim_71 <= 0) { _slice_lo_mask_11 = 0u; }
                                else if (_lim_71 >= 32) { _slice_lo_mask_11 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_11) : "r"(_lim_71));
                                }
                            }
                            if (!(_slice_lo_mask_11 & (1u << 0))) _tmem_load_1[96] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 1))) _tmem_load_1[97] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 2))) _tmem_load_1[98] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 3))) _tmem_load_1[99] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 4))) _tmem_load_1[100] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 5))) _tmem_load_1[101] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 6))) _tmem_load_1[102] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 7))) _tmem_load_1[103] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 8))) _tmem_load_1[104] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 9))) _tmem_load_1[105] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 10))) _tmem_load_1[106] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 11))) _tmem_load_1[107] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 12))) _tmem_load_1[108] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 13))) _tmem_load_1[109] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 14))) _tmem_load_1[110] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 15))) _tmem_load_1[111] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 16))) _tmem_load_1[112] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 17))) _tmem_load_1[113] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 18))) _tmem_load_1[114] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 19))) _tmem_load_1[115] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 20))) _tmem_load_1[116] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 21))) _tmem_load_1[117] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 22))) _tmem_load_1[118] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 23))) _tmem_load_1[119] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 24))) _tmem_load_1[120] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 25))) _tmem_load_1[121] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 26))) _tmem_load_1[122] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 27))) _tmem_load_1[123] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 28))) _tmem_load_1[124] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 29))) _tmem_load_1[125] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 30))) _tmem_load_1[126] = -CAKE_FMHA_INF;
                            if (!(_slice_lo_mask_11 & (1u << 31))) _tmem_load_1[127] = -CAKE_FMHA_INF;
                        }
                    }
                    float2 _reg_reduce_max2_72 = {-CAKE_FMHA_INF, -CAKE_FMHA_INF};
                    row_max_x32_accum(&_tmem_load_1[0], _reg_reduce_max2_72);
                    row_max_x32_accum(&_tmem_load_1[32], _reg_reduce_max2_72);
                    row_max_x32_accum(&_tmem_load_1[64], _reg_reduce_max2_72);
                    row_max_x32_accum(&_tmem_load_1[96], _reg_reduce_max2_72);
                    float _tmem_load_1_max = row_max_reduce(_reg_reduce_max2_72);
                    float new_max_1 = _tmem_load_1_max;
                    float _max_1 = max_noftz(new_max_1, row_max_val);
                    new_max_1 = _max_1;
                    float new_max_scaled_1 = ((new_max_1 == -CAKE_FMHA_INF) ? 0.0f : new_max_1) * softmax_scale_log2;
                    float acc_scale_1;
                    float selected_max_1;
                    float _fma_1 = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max_scaled_1);
                    if (_fma_1 >= -8.0f) {
                        selected_max_1 = row_max_val;
                        acc_scale_1 = 1.0f;
                        new_max_scaled_1 = ((row_max_val == -CAKE_FMHA_INF) ? 0.0f : row_max_val) * softmax_scale_log2;
                    } else {
                        selected_max_1 = new_max_1;
                        if (row_max_val > -CAKE_FMHA_INF) {
                            float _exp2_1 = approx_exp2(_fma_1);
                            acc_scale_1 = _exp2_1;
                        } else {
                            acc_scale_1 = 1.0f;
                        }
                    }
                    row_max_val = selected_max_1;
                    sScale[warp % 4 * 32 + lane + scale_off] = acc_scale_1;
                    mbarrier_arrive(corr_sig_addr + (stage) * 8);
                    const float2 _fma_b2_73 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_74 = {-new_max_scaled_1, -new_max_scaled_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_lf], _fma_b2_73, _fma_c2_74);
                    int p_base_1 = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                    if (stage == 0) {
                        mbarrier_wait(order_p01_0_addr, order_p01_phase);
                    } else {
                        mbarrier_wait(order_p01_1_addr, order_p01_phase);
                    }
                    {
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            _tmem_load_1[_le] = approx_exp2(_tmem_load_1[_le]);
                        }
                    }
                    uint32_t _tmem_load_1_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                        _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[15])));
                    {
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            _tmem_load_1[_le + 32] = approx_exp2(_tmem_load_1[_le + 32]);
                        }
                    }
                    uint32_t _tmem_load_1_bf16_0[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 32], _tmem_load_1[_lp*2+1 + 32]));
                        _tmem_load_1_bf16_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base_1 + 16), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[15])));
                    {
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            _tmem_load_1[_le + 64] = approx_exp2(_tmem_load_1[_le + 64]);
                        }
                    }
                    uint32_t _tmem_load_1_bf16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 64], _tmem_load_1[_lp*2+1 + 64]));
                        _tmem_load_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base_1 + 32), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + (stage) * 8);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_1[_le + 96] = approx_exp2(_tmem_load_1[_le + 96]);
                    }
                    if (stage == 0) {
                        mbarrier_arrive(order_p01_1_addr);
                    } else {
                        mbarrier_arrive(order_p01_0_addr);
                    }
                    order_p01_phase ^= 1;
                    {
                        uint32_t _tmem_load_1_bf16_2[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 96], _tmem_load_1[_lp*2+1 + 96]));
                            _tmem_load_1_bf16_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_base_1 + 48), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[15])));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_2_addr + (stage) * 8);
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                    float2 _reg_reduce_sum2_75 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_1[0], &_reg_reduce_sum2_75);
                    softmax_block_sum(&_tmem_load_1[32], &_reg_reduce_sum2_75);
                    softmax_block_sum(&_tmem_load_1[64], &_reg_reduce_sum2_75);
                    softmax_block_sum(&_tmem_load_1[96], &_reg_reduce_sum2_75);
                    float _tmem_load_1_sum = _reg_reduce_sum2_75.x + _reg_reduce_sum2_75.y;
                    row_sum_val = row_sum_val * acc_scale_1 + _tmem_load_1_sum;
                }
                sScale[warp % 4 * 32 + lane + scale_off + 2 * BLOCK_M] = row_sum_val;
                sScale[warp % 4 * 32 + lane + scale_off + 4 * BLOCK_M] = row_max_val;
                mbarrier_arrive(corr_sig_addr + (stage) * 8);
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            unsigned int total_tiles_1 = NUM_M_BLOCKS * total_bh;
            unsigned int max_rounds_1 = total_tiles_1 + 1;
            {
                max_rounds_1 = 1;
            }
            unsigned int _phase_work_id_full_0_1 = 0;
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
                            const float2 _scale2_0 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_0);
                            tmem_st_x16_f32(cr_addr, _tmem_load_2);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr);
                    mbarrier_arrive(corr_done_addr + 8);
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
                            const float2 _scale2_1 = {scale1, scale1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_1);
                            tmem_st_x16_f32(cr_addr_1, _tmem_load_3);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr + 8);
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
                    int tmem_o_off = ((stage_1 == 0) ? TMEM_OUTPUT_0_OFFSET : TMEM_OUTPUT_1_OFFSET);
                    int s_off = stage_1 * BLOCK_M;
                    float final_sum = sScale[warp % 4 * 32 + lane + s_off + 2 * BLOCK_M];
                    float final_max = sScale[warp % 4 * 32 + lane + s_off + 4 * BLOCK_M];
                    float final_scale;
                    if (final_sum != 0.0f && final_sum == final_sum) {
                        float _rcp_0 = approx_rcp(final_sum);
                        final_scale = _rcp_0;
                    } else {
                        final_scale = 0.0f;
                    }
                    int q_tok0 = (m_block_1 * 2 + (unsigned int)stage_1) * (unsigned int)TOK_PER_STAGE;
                    if (q_tok0 < seqlen_q_bh_1) {
                        int row_valid = warp % 4 * 32 + lane < TOK_PER_STAGE * PACK_G && seqlen_q_bh_1 > q_tok0 + row_tok;
                        long long o_tok = tok_base + q_tok0 + row_tok;
                        long long o_row = o_tok * (long long)NUM_Q_HEADS + (long long)row_head;
                        #pragma unroll
                        for (int ce_col = 0; ce_col < HEAD_DIM / 8; ce_col++) {
                            int ce_addr = taddr + (unsigned int)tmem_o_off + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(ce_col * 8);
                            float _tmem_load_4[8];
                            tmem_ld_x8(&_tmem_load_4[0], ce_addr);
                            if (row_valid != 0) {
                                long long o_elem = o_row * (long long)HEAD_DIM + (long long)(ce_col * 8);
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
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_4[0 + 0], _tmem_load_4[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_4[0 + 2], _tmem_load_4[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_tmem_load_4[0 + 4], _tmem_load_4[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_tmem_load_4[0 + 6], _tmem_load_4[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + o_elem))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
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
                int _mma_b_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
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
                    "mov.b32 id, 136316048;\n\t"
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
                elect_commit(s_full_addr);
                int _mma_a_lo_1 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
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
                    "mov.b32 id, 136316048;\n\t"
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
                    int _mma_b_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
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
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_2), "r"(tmem_softmax_0), "r"(((first_pv_flag) ? 0 : 1)));
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                    _phase_p_full_2_0 ^= 1;
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_3), "r"(tmem_softmax_0), "r"(1));
                    unsigned int k_stage = mma_kv_stage;
                    unsigned int k_phase = mma_kv_phase;
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 3) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
                    int _mma_a_lo_4 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
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
                    "mov.b32 id, 136316048;\n\t"
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_scores_0), "r"(0));
                    elect_commit(s_full_addr);
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
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
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_5), "r"(tmem_softmax_1), "r"(((first_pv_flag) ? 0 : 1)));
                    mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                    _phase_p_full_2_1 ^= 1;
                    int _mma_b_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_6), "r"(tmem_softmax_1), "r"(1));
                    elect_commit(kv_empty_addr + (v_stage) * 8);
                    int _mma_a_lo_7 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
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
                    "mov.b32 id, 136316048;\n\t"
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
                int _mma_b_lo_8 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
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
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_8), "r"(tmem_softmax_0), "r"(((first_pv_flag_1) ? 0 : 1)));
                mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                _phase_p_full_2_0 ^= 1;
                int _mma_b_lo_9 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_9), "r"(tmem_softmax_0), "r"(1));
                mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                _phase_p_full_1 ^= 1;
                int _mma_b_lo_10 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
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
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_10), "r"(tmem_softmax_1), "r"(((first_pv_flag_1) ? 0 : 1)));
                mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                _phase_p_full_2_1 ^= 1;
                int _mma_b_lo_11 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_11), "r"(tmem_softmax_1), "r"(1));
                elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                mma_kv_stage += 1;
                if (mma_kv_stage == 3) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                elect_commit2(o_full_addr, o_full_addr + 8);
            }
        }
    }
    // ---- Role: page_offsets ----
    if (warp == 13) {
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
                    if (elect_sync()) {
                        int pg_idx = po_stage * 16;
                        #pragma unroll
                        for (int chunk = 0; chunk < BLOCK_N / KV_TOK_CHUNK; chunk++) {
                            int _min_0 = (((base_tok + chunk * KV_TOK_CHUNK) / PAGE_SIZE) < (last_slot) ? ((base_tok + chunk * KV_TOK_CHUNK) / PAGE_SIZE) : (last_slot));
                            int slot = _min_0;
                            smem_pages[pg_idx + chunk] = page_table_k[pt_base + slot];
                            smem_pages[pg_idx + 8 + chunk] = page_table_v[pt_base + slot];
                        }
                        mbarrier_arrive(page_full_addr + (po_stage) * 8);
                    }
                    po_stage += 1;
                    if (po_stage == 6) { po_stage = 0; _phase_page_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 14) {
        { // scheduler_main
            unsigned int _phase_work_id_empty_0 = 1;
        }
    }
    // ---- Role: load ----
    if (warp == 15) {
        { // load_main
            unsigned int total_tiles_4 = NUM_M_BLOCKS * total_bh;
            unsigned int max_rounds_4 = total_tiles_4 + 1;
            {
                max_rounds_4 = 1;
            }
            unsigned int load_kv_stage = 0;
            unsigned int load_po_stage = 0;
            unsigned int _phase_work_id_full_0_4 = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_page_full = 0;
            unsigned int _phase_kv_empty = 1;
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
                    mbarrier_arrive_expect_tx(q_full_addr, TOK_PER_STAGE * PACK_G * 256);
                    tma_5d_gmem2smem(smem_q0_addr, Q, 0, 0, tok_q0, 0, unit_1, q_full_addr);
                    tma_5d_gmem2smem(smem_q0_addr + 16384, Q, 0, 0, tok_q0, 1, unit_1, q_full_addr);
                }
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + 8, TOK_PER_STAGE * PACK_G * 256);
                    tma_5d_gmem2smem(smem_q1_addr, Q, 0, 0, tok_q1, 0, unit_1, q_full_addr + 8);
                    tma_5d_gmem2smem(smem_q1_addr + 16384, Q, 0, 0, tok_q1, 1, unit_1, q_full_addr + 8);
                }
                #pragma unroll 1
                for (unsigned int ni_1 = 0; ni_1 < num_n_blocks_4; ni_1++) {
                    unsigned int n_1 = num_n_blocks_4 - 1 - ni_1;
                    int base_tok_1 = n_1 * (unsigned int)BLOCK_N;
                    mbarrier_wait(page_full_addr + (load_po_stage) * 8, _phase_page_full);
                    int pg_idx_1 = load_po_stage * 16;
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 32768);
                        #pragma unroll
                        for (int chunk_1 = 0; chunk_1 < BLOCK_N / KV_TOK_CHUNK; chunk_1++) {
                            int tok_off = (base_tok_1 + chunk_1 * KV_TOK_CHUNK) % PAGE_SIZE;
                            const int chunk_bytes = KV_TOK_CHUNK * 64 * 2;
                            tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 32768 + (unsigned int)(chunk_1 * chunk_bytes), K, 0, tok_off, 0, kv_head, smem_pages[pg_idx_1 + chunk_1], kv_full_addr + (load_kv_stage) * 8);
                            tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 32768 + (unsigned int)(16384 + chunk_1 * chunk_bytes), K, 0, tok_off, 1, kv_head, smem_pages[pg_idx_1 + chunk_1], kv_full_addr + (load_kv_stage) * 8);
                        }
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 3) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 32768);
                        #pragma unroll
                        for (int chunk_2 = 0; chunk_2 < BLOCK_N / KV_TOK_CHUNK; chunk_2++) {
                            int tok_off_1 = (base_tok_1 + chunk_2 * KV_TOK_CHUNK) % PAGE_SIZE;
                            const int chunk_bytes_1 = KV_TOK_CHUNK * 64 * 2;
                            tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 32768 + (unsigned int)(chunk_2 * chunk_bytes_1), V, 0, tok_off_1, 0, kv_head, smem_pages[pg_idx_1 + 8 + chunk_2], kv_full_addr + (load_kv_stage) * 8);
                            tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 32768 + (unsigned int)(16384 + chunk_2 * chunk_bytes_1), V, 0, tok_off_1, 1, kv_head, smem_pages[pg_idx_1 + 8 + chunk_2], kv_full_addr + (load_kv_stage) * 8);
                        }
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 3) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
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
