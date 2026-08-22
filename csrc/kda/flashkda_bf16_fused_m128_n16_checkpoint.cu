/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

// Frozen generated N16 checkpoint-TMA kernel; do not edit the generated body.
// clang-format off
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) FlashKDATensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) FlashKDATensorMapPack { FlashKDATensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

#define FLASH_KDA_INF CUDART_INF_F
#define TMEM_NCOLS 240
#define TMEM_TMEM_STATE_OFFSET 64
#define TMEM_TMEM_STATE_INP_OFFSET 0
#define TMEM_TMEM_U_ACC_OFFSET 224
#define TMEM_TMEM_U2_INP_OFFSET 224
#define TMEM_TMEM_U2_ACC_OFFSET 208
#define TMEM_TMEM_OUT_OFFSET 192
#define TMEM_TMEM_STATE_OUT_OFFSET 64
#define NUM_CHUNK_PIPE_STAGES 5
#define NUM_CHECKPOINT_PIPE_STAGES 2
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 4096
#define SMEM_SMEM_QD_STRIDE 21504
#define SMEM_SMEM_G_RAW_OFF 1024
#define SMEM_SMEM_G_RAW_STAGE_BYTES 4096
#define SMEM_SMEM_G_RAW_STRIDE 21504
#define SMEM_SMEM_G_RAW_ALL_OFF 1024
#define SMEM_SMEM_G_RAW_ALL_STAGE_BYTES 90112
#define SMEM_SMEM_G_RAW_ALL_STRIDE 90112
#define SMEM_SMEM_KD_OFF 5120
#define SMEM_SMEM_KD_STAGE_BYTES 4096
#define SMEM_SMEM_KD_STRIDE 21504
#define SMEM_SMEM_Q_RAW_PREFETCH_OFF 9216
#define SMEM_SMEM_Q_RAW_PREFETCH_STAGE_BYTES 4096
#define SMEM_SMEM_Q_RAW_PREFETCH_STRIDE 21504
#define SMEM_SMEM_FINAL_TRANS_OFF 9216
#define SMEM_SMEM_FINAL_TRANS_STAGE_BYTES 6144
#define SMEM_SMEM_FINAL_TRANS_STRIDE 21504
#define SMEM_SMEM_KR_TRANS_OFF 9216
#define SMEM_SMEM_KR_TRANS_STAGE_BYTES 4096
#define SMEM_SMEM_KR_TRANS_STRIDE 21504
#define SMEM_SMEM_MQK_TRANS_OFF 13312
#define SMEM_SMEM_MQK_TRANS_STAGE_BYTES 512
#define SMEM_SMEM_MQK_TRANS_STRIDE 21504
#define SMEM_SMEM_INV_OFF 15360
#define SMEM_SMEM_INV_STAGE_BYTES 512
#define SMEM_SMEM_INV_STRIDE 21504
#define SMEM_SMEM_V_OFF 16512
#define SMEM_SMEM_V_STAGE_BYTES 4096
#define SMEM_SMEM_V_STRIDE 21504
#define SMEM_SMEM_KI_OFF 9216
#define SMEM_SMEM_KI_STAGE_BYTES 4096
#define SMEM_SMEM_KI_STRIDE 21504
#define SMEM_SMEM_GATE_OFF 13312
#define SMEM_SMEM_GATE_STAGE_BYTES 8192
#define SMEM_SMEM_GATE_STRIDE 21504
#define SMEM_SMEM_BETA_RAW_OFF 21504
#define SMEM_SMEM_BETA_RAW_STAGE_BYTES 256
#define SMEM_SMEM_BETA_RAW_STRIDE 21504
#define SMEM_SMEM_INV_WORK_OFF 16512
#define SMEM_SMEM_INV_WORK_STAGE_BYTES 4096
#define SMEM_SMEM_INV_WORK_STRIDE 21504
#define SMEM_SMEM_OUT_OFF 108544
#define SMEM_SMEM_OUT_STAGE_BYTES 4096
#define SMEM_SMEM_OUT_STRIDE 4096
#define SMEM_SMEM_CHECKPOINT_OFF 117760
#define SMEM_SMEM_CHECKPOINT_STAGE_BYTES 32768
#define SMEM_SMEM_CHECKPOINT_STRIDE 32768
#define SMEM_SMEM_RESTORE_FACTOR_ALL_OFF 21504
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES 86532
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STRIDE 86532
#define SMEM_SMEM_GT_PREFIX_ALL_OFF 20992
#define SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES 86528
#define SMEM_SMEM_GT_PREFIX_ALL_STRIDE 86528
#define SMEM_SMEM_GT_ALL_OFF 15872
#define SMEM_SMEM_GT_ALL_STAGE_BYTES 86528
#define SMEM_SMEM_GT_ALL_STRIDE 86528
#define SMEM_SMEM_PREP_BETA_ALL_OFF 22020
#define SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES 86080
#define SMEM_SMEM_PREP_BETA_ALL_STRIDE 86080
#define SMEM_SMEM_GATE_RATE_ALL_OFF 22084
#define SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES 86020
#define SMEM_SMEM_GATE_RATE_ALL_STRIDE 86020
#define SMEM_SMEM_V_ALL_OFF 16512
#define SMEM_SMEM_V_ALL_STAGE_BYTES 90112
#define SMEM_SMEM_V_ALL_STRIDE 90112
#define SMEM_SMEM_GATE_ALL_OFF 13312
#define SMEM_SMEM_GATE_ALL_STAGE_BYTES 94208
#define SMEM_SMEM_GATE_ALL_STRIDE 94208
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_OFF 116736
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STAGE_BYTES 80
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STRIDE 80
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_OFF 116736
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_STAGE_BYTES 16
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_STRIDE 16
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_OFF 116752
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_STAGE_BYTES 4
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_STRIDE 4
#define SMEM_SMEM_WORK_ITEM_RESOLVED_OFF 116756
#define SMEM_SMEM_WORK_ITEM_RESOLVED_STAGE_BYTES 4
#define SMEM_SMEM_WORK_ITEM_RESOLVED_STRIDE 4
#define SMEM_TOTAL 183296
#define THREADS 1024
#define STORE_BACKWARD_TAPE 0
#define STORE_E_TAPE 1
#define SPLIT_WORK_ITEMS 0

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


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
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

__global__ __launch_bounds__(1024) void
kernel_flashkda_bf16_fused_m128(__nv_bfloat16* __restrict__ q, FlashKDATensorMap const* q_tma, __nv_bfloat16* __restrict__ k, FlashKDATensorMap const* k_tma, __nv_bfloat16* __restrict__ v, FlashKDATensorMap const* v_tma, __nv_bfloat16* __restrict__ g, FlashKDATensorMap const* g_tma, __nv_bfloat16* __restrict__ beta, FlashKDATensorMap const* beta_tma, float* __restrict__ A_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ seq_order, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ out, FlashKDATensorMap const* out_tma, __nv_bfloat16* __restrict__ final_state, int num_heads, int use_initial_state, int store_final_state, float scale, float lower_bound, unsigned long long state_indices_addr, unsigned long long state_checkpoints_addr, unsigned long long checkpoint_cu_starts_addr, long long beta_token_stride, long long state_slot_stride, int use_state_indices, int checkpoint_every_n_tokens, long long* __restrict__ cu_chunk_offsets, __nv_bfloat16* __restrict__ chunk_state, unsigned int* __restrict__ state_checkpoint_needed, __nv_bfloat16* __restrict__ tape_qd, __nv_bfloat16* __restrict__ tape_kd, __nv_bfloat16* __restrict__ tape_kr, __nv_bfloat16* __restrict__ tape_j, float* __restrict__ tape_restore_factor, __nv_bfloat16* __restrict__ tape_e, __nv_bfloat16* __restrict__ tape_x, __nv_bfloat16* __restrict__ tape_r, float* __restrict__ norm_inv_out, __nv_bfloat16* __restrict__ decay_out, float* __restrict__ beta_active_out, float* __restrict__ initial_state_f32, unsigned int* __restrict__ zero_workspace, int zero_words, int num_sequences, FlashKDATensorMap const* state_checkpoints_tma)
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
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(q_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(k_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(v_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(g_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(beta_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(out_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(state_checkpoints_tma)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_qd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qd_addr = smem + 1024;
    __nv_bfloat16* smem_g_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_g_raw_addr = smem + 1024;
    __nv_bfloat16* smem_g_raw_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_g_raw_all_addr = smem + 1024;
    __nv_bfloat16* smem_kd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 5120);
    const int smem_kd_addr = smem + 5120;
    __nv_bfloat16* smem_q_raw_prefetch = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_q_raw_prefetch_addr = smem + 9216;
    __nv_bfloat16* smem_final_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_final_trans_addr = smem + 9216;
    __nv_bfloat16* smem_kr_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kr_trans_addr = smem + 9216;
    __nv_bfloat16* smem_mqk_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 13312);
    const int smem_mqk_trans_addr = smem + 13312;
    __nv_bfloat16* smem_inv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 15360);
    const int smem_inv_addr = smem + 15360;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16512);
    const int smem_v_addr = smem + 16512;
    __nv_bfloat16* smem_ki = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_ki_addr = smem + 9216;
    float* smem_gate = reinterpret_cast<float*>(smem_raw + 13312);
    const int smem_gate_addr = smem + 13312;
    __nv_bfloat16* smem_beta_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int smem_beta_raw_addr = smem + 21504;
    __nv_bfloat16* smem_inv_work = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16512);
    const int smem_inv_work_addr = smem + 16512;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 108544);
    const int smem_out_addr = smem + 108544;
    __nv_bfloat16* smem_checkpoint = reinterpret_cast<__nv_bfloat16*>(smem_raw + 117760);
    const int smem_checkpoint_addr = smem + 117760;
    float* smem_restore_factor_all = reinterpret_cast<float*>(smem_raw + 21504);
    const int smem_restore_factor_all_addr = smem + 21504;
    float* smem_gt_prefix_all = reinterpret_cast<float*>(smem_raw + 20992);
    const int smem_gt_prefix_all_addr = smem + 20992;
    float* smem_gt_all = reinterpret_cast<float*>(smem_raw + 15872);
    const int smem_gt_all_addr = smem + 15872;
    float* smem_prep_beta_all = reinterpret_cast<float*>(smem_raw + 22020);
    const int smem_prep_beta_all_addr = smem + 22020;
    float* smem_gate_rate_all = reinterpret_cast<float*>(smem_raw + 22084);
    const int smem_gate_rate_all_addr = smem + 22084;
    __nv_bfloat16* smem_v_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16512);
    const int smem_v_all_addr = smem + 16512;
    float* smem_gate_all = reinterpret_cast<float*>(smem_raw + 13312);
    const int smem_gate_all_addr = smem + 13312;
    unsigned int* smem_state_checkpoint_needed = reinterpret_cast<unsigned int*>(smem_raw + 116736);
    const int smem_state_checkpoint_needed_addr = smem + 116736;
    float* smem_work_item_warp_max = reinterpret_cast<float*>(smem_raw + 116736);
    const int smem_work_item_warp_max_addr = smem + 116736;
    int* smem_work_item_compute_start = reinterpret_cast<int*>(smem_raw + 116752);
    const int smem_work_item_compute_start_addr = smem + 116752;
    unsigned int* smem_work_item_resolved = reinterpret_cast<unsigned int*>(smem_raw + 116756);
    const int smem_work_item_resolved_addr = smem + 116756;

    // Mbarrier init (22 groups, 92 barriers)
    // Mbarriers at smem_raw[0..736)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'chunk_pipe' ---
            // qk_full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // tape_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // tape_free: 5 barriers, init_count=2
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            // gate_raw_full: 5 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // qk_raw_full: 5 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // v_full: 5 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // v_free: 5 barriers, init_count=4
            mbarrier_init(smem + 240, 4);
            mbarrier_init(smem + 248, 4);
            mbarrier_init(smem + 256, 4);
            mbarrier_init(smem + 264, 4);
            mbarrier_init(smem + 272, 4);
            // smem_free: 5 barriers, init_count=4
            mbarrier_init(smem + 280, 4);
            mbarrier_init(smem + 288, 4);
            mbarrier_init(smem + 296, 4);
            mbarrier_init(smem + 304, 4);
            mbarrier_init(smem + 312, 4);
            // raw_inputs_free: 5 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            // state_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 360, 4);
            mbarrier_init(smem + 368, 4);
            mbarrier_init(smem + 376, 4);
            mbarrier_init(smem + 384, 4);
            mbarrier_init(smem + 392, 4);
            // old_out_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 400, 1);
            mbarrier_init(smem + 408, 1);
            mbarrier_init(smem + 416, 1);
            mbarrier_init(smem + 424, 1);
            mbarrier_init(smem + 432, 1);
            // u_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 440, 4);
            mbarrier_init(smem + 448, 4);
            mbarrier_init(smem + 456, 4);
            mbarrier_init(smem + 464, 4);
            mbarrier_init(smem + 472, 4);
            // u2_acc_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 480, 1);
            mbarrier_init(smem + 488, 1);
            mbarrier_init(smem + 496, 1);
            mbarrier_init(smem + 504, 1);
            mbarrier_init(smem + 512, 1);
            // u2_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 520, 4);
            mbarrier_init(smem + 528, 4);
            mbarrier_init(smem + 536, 4);
            mbarrier_init(smem + 544, 4);
            mbarrier_init(smem + 552, 4);
            // final_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 560, 1);
            mbarrier_init(smem + 568, 1);
            mbarrier_init(smem + 576, 1);
            mbarrier_init(smem + 584, 1);
            mbarrier_init(smem + 592, 1);
            // out_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 600, 1);
            // tmem_dealloc_ready: 1 barriers, init_count=2
            mbarrier_init(smem + 608, 2);
            // --- pipeline 'checkpoint_pipe' ---
            // checkpoint_ready: 2 barriers, init_count=4
            mbarrier_init(smem + 616, 4);
            mbarrier_init(smem + 624, 4);
            // checkpoint_free: 2 barriers, init_count=1
            mbarrier_init(smem + 632, 1);
            mbarrier_init(smem + 640, 1);
            // --- pipeline 'chunk_pipe' ---
            // prep_diag_ready: 5 barriers, init_count=2
            mbarrier_init(smem + 648, 2);
            mbarrier_init(smem + 656, 2);
            mbarrier_init(smem + 664, 2);
            mbarrier_init(smem + 672, 2);
            mbarrier_init(smem + 680, 2);
            // prep_inv16_ready: 5 barriers, init_count=2
            mbarrier_init(smem + 688, 2);
            mbarrier_init(smem + 696, 2);
            mbarrier_init(smem + 704, 2);
            mbarrier_init(smem + 712, 2);
            mbarrier_init(smem + 720, 2);
            // work_item_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 728, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 240 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 736);
    if (warp == 0) {
        int _tmem_hold = smem + 736;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define qk_full_addr (mbar_base + 0)
    #define tape_ready_addr (mbar_base + 40)
    #define tape_free_addr (mbar_base + 80)
    #define gate_raw_full_addr (mbar_base + 120)
    #define qk_raw_full_addr (mbar_base + 160)
    #define v_full_addr (mbar_base + 200)
    #define v_free_addr (mbar_base + 240)
    #define smem_free_addr (mbar_base + 280)
    #define raw_inputs_free_addr (mbar_base + 320)
    #define state_inp_ready_addr (mbar_base + 360)
    #define old_out_ready_addr (mbar_base + 400)
    #define u_inp_ready_addr (mbar_base + 440)
    #define u2_acc_ready_addr (mbar_base + 480)
    #define u2_inp_ready_addr (mbar_base + 520)
    #define final_ready_addr (mbar_base + 560)
    #define out_empty_addr (mbar_base + 600)
    #define tmem_dealloc_ready_addr (mbar_base + 608)
    #define checkpoint_ready_addr (mbar_base + 616)
    #define checkpoint_free_addr (mbar_base + 632)
    #define prep_diag_ready_addr (mbar_base + 648)
    #define prep_inv16_ready_addr (mbar_base + 688)
    #define work_item_ready_addr (mbar_base + 728)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state = taddr + 64;
    const int tmem_tmem_state_inp = taddr;
    const int tmem_tmem_u_acc = taddr + 224;
    const int tmem_tmem_u2_inp = taddr + 224;
    const int tmem_tmem_u2_acc = taddr + 208;
    const int tmem_tmem_out = taddr + 192;
    const int tmem_tmem_state_out = taddr + 64;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: compute ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // compute_main
            int task_idx = blockIdx.x;
            int warp_id_in_role = (warp - 0);
            int compute_local_warp = warp_id_in_role;
            int warp_in_wg = warp % 4;
            int state_row = warp_in_wg * 32 + lane;
            int split_compute_start = 0;
            int seq_len = (int)(cu_seqlens[seq_order[task_idx / num_heads] + 1] - cu_seqlens[seq_order[task_idx / num_heads]]);
            int num_chunks = (seq_len + 16 - 1) / 16;
            long long total_chunks = cu_chunk_offsets[num_sequences];
            long long fallback_head = total_chunks * (long long)num_heads + (long long)seq_order[task_idx / num_heads] * (long long)num_heads + (long long)(task_idx % num_heads);
            const int tmem_row_base = warp_in_wg * 32 << 16;
            long long state_base = (((long long)seq_order[task_idx / num_heads] * (long long)num_heads + (long long)(task_idx % num_heads)) * 128 + (long long)state_row) * 128;
            {
                int state_slot = seq_order[task_idx / num_heads];
                if (use_state_indices != 0) {
                    state_slot = reinterpret_cast<int*>(state_indices_addr)[seq_order[task_idx / num_heads]];
                }
                state_base = (long long)state_slot * state_slot_stride + ((long long)(task_idx % num_heads) * 128 + (long long)state_row) * 128;
            }
            #pragma unroll
            for (int state_col_block = 0; state_col_block < 4; state_col_block++) {
                float state_frag[32];
                state_frag[0] = 0.0f;
                state_frag[1] = 0.0f;
                state_frag[2] = 0.0f;
                state_frag[3] = 0.0f;
                state_frag[4] = 0.0f;
                state_frag[5] = 0.0f;
                state_frag[6] = 0.0f;
                state_frag[7] = 0.0f;
                state_frag[8] = 0.0f;
                state_frag[9] = 0.0f;
                state_frag[10] = 0.0f;
                state_frag[11] = 0.0f;
                state_frag[12] = 0.0f;
                state_frag[13] = 0.0f;
                state_frag[14] = 0.0f;
                state_frag[15] = 0.0f;
                state_frag[16] = 0.0f;
                state_frag[17] = 0.0f;
                state_frag[18] = 0.0f;
                state_frag[19] = 0.0f;
                state_frag[20] = 0.0f;
                state_frag[21] = 0.0f;
                state_frag[22] = 0.0f;
                state_frag[23] = 0.0f;
                state_frag[24] = 0.0f;
                state_frag[25] = 0.0f;
                state_frag[26] = 0.0f;
                state_frag[27] = 0.0f;
                state_frag[28] = 0.0f;
                state_frag[29] = 0.0f;
                state_frag[30] = 0.0f;
                state_frag[31] = 0.0f;
                if (use_initial_state != 0) {
                    {
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(initial_state + state_base + (long long)(state_col_block * 32));
                            uint4 _vld_0[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&state_frag[0 + _blk * 8 + _pair * 2])[0]), "=f"((&state_frag[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(initial_state + state_base + (long long)(state_col_block * 32) + 16);
                            uint4 _vld_1[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                _vld_1[_blk] = _vptr_1[_blk];
                                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&state_frag[16 + _blk * 8 + _pair * 2])[0]), "=f"((&state_frag[16 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_1[_pair]));
                                }
                            }
                        }
                    }
                }
                tmem_st_x32_f32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block * 32), state_frag);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            {
            }
            unsigned int compute_stage = 0;
            unsigned int checkpoint_stage_compute = 0;
            unsigned int _phase_qk_full = 0;
            unsigned int _phase_checkpoint_free = 1;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_old_out_ready = 0;
            unsigned int _phase_u2_acc_ready = 0;
            unsigned int _phase_final_ready = 0;
            #pragma unroll 1
            for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                int chunk_global_local = chunk_idx;
                int owned_chunk = chunk_global_local >= 0 && chunk_global_local < ((int)(cu_seqlens[seq_order[task_idx / num_heads] + 1] - cu_seqlens[seq_order[task_idx / num_heads]]) + 16 - 1) / 16;
                mbarrier_wait(qk_full_addr + (compute_stage) * 8, _phase_qk_full);
                int checkpoint_token_entering = chunk_idx * 16;
                int checkpoint_entering = checkpoint_every_n_tokens != 0 && checkpoint_token_entering % checkpoint_every_n_tokens == 0;
                if (checkpoint_entering != 0) {
                    mbarrier_wait(checkpoint_free_addr + (checkpoint_stage_compute) * 8, _phase_checkpoint_free);
                }
                #pragma unroll 1
                for (int state_col_block_1 = 0; state_col_block_1 < 4; state_col_block_1++) {
                    int state_addr = taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_1 * 32);
                    float _tmem_load_1[32];
                    tmem_ld_x32(&_tmem_load_1[0], state_addr);
                    uint32_t _tmem_load_1_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                        _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_1 * 16)), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[15]))
                        : "memory");
                    if (checkpoint_entering != 0) {
                        #pragma unroll
                        for (int checkpoint_vec = 0; checkpoint_vec < 4; checkpoint_vec++) {
                            int checkpoint_addr = (smem_checkpoint_addr + checkpoint_stage_compute * 32768 + (unsigned int)((state_col_block_1 * 32 + checkpoint_vec * 8) / 64 * 16384 + state_row * 128 + (state_col_block_1 * 32 + checkpoint_vec * 8) % 64 * 2 ^ ((state_col_block_1 * 32 + checkpoint_vec * 8) / 64 * 16384 + state_row * 128 + (state_col_block_1 * 32 + checkpoint_vec * 8) % 64 * 2 >> 7 & 7) << 4));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(checkpoint_addr), "r"(_tmem_load_1_bf16[checkpoint_vec * 4]), "r"(_tmem_load_1_bf16[checkpoint_vec * 4 + 1]), "r"(_tmem_load_1_bf16[checkpoint_vec * 4 + 2]), "r"(_tmem_load_1_bf16[checkpoint_vec * 4 + 3]) : "memory");
                        }
                    }
                    float _tmem_load_1_bf16_f32[32];
                    #pragma unroll
                    for (int _pair = 0; _pair < 16; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_tmem_load_1_bf16_f32[_pair * 2])[0]), "=f"((&_tmem_load_1_bf16_f32[_pair * 2])[1])
                            : "r"(_tmem_load_1_bf16[_pair]));
                    }
                    float state_scale[16];
                    #pragma unroll
                    for (int state_half = 0; state_half < 2; state_half++) {
                        #pragma unroll
                        for (int state_col = 0; state_col < 16; state_col++) {
                            state_scale[state_col] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_1 * 32) + (unsigned int)(state_half * 16) + (unsigned int)state_col];
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1_bf16_f32 + state_half * 16))[_ls], reinterpret_cast<const float2*>(state_scale)[_ls]);
                    }
                    tmem_st_x32_f32(state_addr, _tmem_load_1_bf16_f32);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (checkpoint_entering != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(checkpoint_ready_addr + (checkpoint_stage_compute) * 8);
                    }
                    checkpoint_stage_compute += 1;
                    if (checkpoint_stage_compute == 2) { checkpoint_stage_compute = 0; _phase_checkpoint_free ^= 1; }
                }
                if (elect_sync()) {
                    mbarrier_arrive(state_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(v_full_addr + (compute_stage) * 8, _phase_v_full);
                mbarrier_wait(old_out_ready_addr + (compute_stage) * 8, _phase_old_out_ready);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + 224 + (unsigned int)tmem_row_base);
                uint32_t _tmem_load_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _tmem_load_2_bf16_f32[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_tmem_load_2_bf16_f32[_pair * 2])[0]), "=f"((&_tmem_load_2_bf16_f32[_pair * 2])[1])
                        : "r"(_tmem_load_2_bf16[_pair]));
                }
                long long chunk_global_e = cu_chunk_offsets[seq_order[task_idx / num_heads]] + (long long)chunk_global_local;
                long long tape_ex_base = ((chunk_global_e * (long long)num_heads + (long long)(task_idx % num_heads)) * 128 + (long long)state_row) * 16;
                #pragma unroll
                for (int residual_half = 0; residual_half < 1; residual_half++) {
                    float residual_v[16];
                    float residual_beta[16];
                    #pragma unroll
                    for (int residual_col = 0; residual_col < 16; residual_col++) {
                        int token_col = residual_half * 16 + residual_col;
                        __nv_bfloat16 v_value = smem_v_all[compute_stage * 10752 + (unsigned int)(token_col * 128) + (unsigned int)state_row];
                        float _cvt_f32_72 = __bfloat162float(v_value);
                        residual_v[residual_col] = _cvt_f32_72;
                        residual_beta[residual_col] = smem_prep_beta_all[compute_stage * 5376 + (unsigned int)token_col];
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        sub_f32x2_inplace(&reinterpret_cast<float2*>(residual_v)[_ls], reinterpret_cast<const float2*>((_tmem_load_2_bf16_f32 + residual_half * 16))[_ls]);
                    if (STORE_BACKWARD_TAPE != 0 && STORE_E_TAPE != 0 && owned_chunk != 0) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(residual_v[0 + 0], residual_v[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(residual_v[0 + 2], residual_v[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(residual_v[0 + 4], residual_v[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(residual_v[0 + 6], residual_v[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_e + (tape_ex_base + (long long)(residual_half * 16))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(residual_v[8 + 0], residual_v[8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(residual_v[8 + 2], residual_v[8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(residual_v[8 + 4], residual_v[8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(residual_v[8 + 6], residual_v[8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_e + (tape_ex_base + (long long)(residual_half * 16) + 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                    uint32_t residual_v_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 0], residual_v[_lp*2+1 + 0]));
                        residual_v_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    float residual_v_bf16_f32[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&residual_v_bf16_f32[_pair * 2])[0]), "=f"((&residual_v_bf16_f32[_pair * 2])[1])
                            : "r"(residual_v_bf16[_pair]));
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(residual_v_bf16_f32)[_ls], reinterpret_cast<const float2*>(residual_beta)[_ls]);
                    if (STORE_BACKWARD_TAPE != 0 && owned_chunk != 0) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(residual_v_bf16_f32[0 + 0], residual_v_bf16_f32[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(residual_v_bf16_f32[0 + 2], residual_v_bf16_f32[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(residual_v_bf16_f32[0 + 4], residual_v_bf16_f32[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(residual_v_bf16_f32[0 + 6], residual_v_bf16_f32[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_x + (tape_ex_base + (long long)(residual_half * 16))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(residual_v_bf16_f32[8 + 0], residual_v_bf16_f32[8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(residual_v_bf16_f32[8 + 2], residual_v_bf16_f32[8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(residual_v_bf16_f32[8 + 4], residual_v_bf16_f32[8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(residual_v_bf16_f32[8 + 6], residual_v_bf16_f32[8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_x + (tape_ex_base + (long long)(residual_half * 16) + 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                    uint32_t residual_v_bf16_f32_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v_bf16_f32[_lp*2 + 0], residual_v_bf16_f32[_lp*2+1 + 0]));
                        residual_v_bf16_f32_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 224 + (unsigned int)tmem_row_base + (unsigned int)(residual_half * 8), (const uint32_t*)residual_v_bf16_f32_bf16);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(v_free_addr + (compute_stage) * 8);
                    mbarrier_arrive(u_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(u2_acc_ready_addr + (compute_stage) * 8, _phase_u2_acc_ready);
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + 208 + (unsigned int)tmem_row_base);
                if (STORE_BACKWARD_TAPE != 0 && owned_chunk != 0) {
                    long long chunk_global_r = cu_chunk_offsets[seq_order[task_idx / num_heads]] + (long long)chunk_global_local;
                    long long tape_r_base = ((chunk_global_r * (long long)num_heads + (long long)(task_idx % num_heads)) * 128 + (long long)state_row) * 16;
                    #pragma unroll
                    for (int tape_r_vec = 0; tape_r_vec < 4; tape_r_vec++) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_3[tape_r_vec * 8 + 0], _tmem_load_3[tape_r_vec * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_3[tape_r_vec * 8 + 2], _tmem_load_3[tape_r_vec * 8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_3[tape_r_vec * 8 + 4], _tmem_load_3[tape_r_vec * 8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_3[tape_r_vec * 8 + 6], _tmem_load_3[tape_r_vec * 8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_r + (tape_r_base + (long long)(tape_r_vec * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
                uint32_t _tmem_load_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                tmem_st_x8_u32(taddr + 224 + (unsigned int)tmem_row_base, (const uint32_t*)_tmem_load_3_bf16);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(u2_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(final_ready_addr + (compute_stage) * 8, _phase_final_ready);
                #pragma unroll
                for (int state_col_block_post = 0; state_col_block_post < 4; state_col_block_post++) {
                    float _tmem_load_4[16];
                    tmem_ld_x16(&_tmem_load_4[0], taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_post * 32));
                    float _tmem_load_5[8];
                    tmem_ld_x8(&_tmem_load_5[0], taddr + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_post * 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int old_state_packed_0[8];
                    old_state_packed_0[0] = __as_u32(_tmem_load_5[0]);
                    old_state_packed_0[1] = __as_u32(_tmem_load_5[1]);
                    old_state_packed_0[2] = __as_u32(_tmem_load_5[2]);
                    old_state_packed_0[3] = __as_u32(_tmem_load_5[3]);
                    old_state_packed_0[4] = __as_u32(_tmem_load_5[4]);
                    old_state_packed_0[5] = __as_u32(_tmem_load_5[5]);
                    old_state_packed_0[6] = __as_u32(_tmem_load_5[6]);
                    old_state_packed_0[7] = __as_u32(_tmem_load_5[7]);
                    float old_state_packed_0_f32[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&old_state_packed_0_f32[_pair * 2])[0]), "=f"((&old_state_packed_0_f32[_pair * 2])[1])
                            : "r"(old_state_packed_0[_pair]));
                    }
                    float post_state_scale_0[16];
                    post_state_scale_0[0] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32)];
                    post_state_scale_0[1] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 1];
                    post_state_scale_0[2] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 2];
                    post_state_scale_0[3] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 3];
                    post_state_scale_0[4] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 4];
                    post_state_scale_0[5] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 5];
                    post_state_scale_0[6] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 6];
                    post_state_scale_0[7] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 7];
                    post_state_scale_0[8] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 8];
                    post_state_scale_0[9] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 9];
                    post_state_scale_0[10] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 10];
                    post_state_scale_0[11] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 11];
                    post_state_scale_0[12] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 12];
                    post_state_scale_0[13] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 13];
                    post_state_scale_0[14] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 14];
                    post_state_scale_0[15] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 15];
                    #pragma unroll
                    for (int _lf = 0; _lf < 16; _lf++) {
                        old_state_packed_0_f32[_lf] = fmaf(old_state_packed_0_f32[_lf], post_state_scale_0[_lf], _tmem_load_4[_lf]);
                    }
                    uint32_t old_state_packed_0_f32_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(old_state_packed_0_f32[_lp*2 + 0], old_state_packed_0_f32[_lp*2+1 + 0]));
                        old_state_packed_0_f32_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&old_state_packed_0_f32[_pair * 2])[0]), "=f"((&old_state_packed_0_f32[_pair * 2])[1])
                            : "r"(old_state_packed_0_f32_bf16[_pair]));
                    }
                    tmem_st_x16_f32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_post * 32), old_state_packed_0_f32);
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_post * 32) + 16);
                    float _tmem_load_7[8];
                    tmem_ld_x8(&_tmem_load_7[0], taddr + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_post * 16) + 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int old_state_packed_1[8];
                    old_state_packed_1[0] = __as_u32(_tmem_load_7[0]);
                    old_state_packed_1[1] = __as_u32(_tmem_load_7[1]);
                    old_state_packed_1[2] = __as_u32(_tmem_load_7[2]);
                    old_state_packed_1[3] = __as_u32(_tmem_load_7[3]);
                    old_state_packed_1[4] = __as_u32(_tmem_load_7[4]);
                    old_state_packed_1[5] = __as_u32(_tmem_load_7[5]);
                    old_state_packed_1[6] = __as_u32(_tmem_load_7[6]);
                    old_state_packed_1[7] = __as_u32(_tmem_load_7[7]);
                    float old_state_packed_1_f32[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&old_state_packed_1_f32[_pair * 2])[0]), "=f"((&old_state_packed_1_f32[_pair * 2])[1])
                            : "r"(old_state_packed_1[_pair]));
                    }
                    float post_state_scale_1[16];
                    post_state_scale_1[0] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16];
                    post_state_scale_1[1] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 1];
                    post_state_scale_1[2] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 2];
                    post_state_scale_1[3] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 3];
                    post_state_scale_1[4] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 4];
                    post_state_scale_1[5] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 5];
                    post_state_scale_1[6] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 6];
                    post_state_scale_1[7] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 7];
                    post_state_scale_1[8] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 8];
                    post_state_scale_1[9] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 9];
                    post_state_scale_1[10] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 10];
                    post_state_scale_1[11] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 11];
                    post_state_scale_1[12] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 12];
                    post_state_scale_1[13] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 13];
                    post_state_scale_1[14] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 14];
                    post_state_scale_1[15] = smem_gt_all[compute_stage * 5376 + (unsigned int)(state_col_block_post * 32) + 16 + 15];
                    #pragma unroll
                    for (int _lf = 0; _lf < 16; _lf++) {
                        old_state_packed_1_f32[_lf] = fmaf(old_state_packed_1_f32[_lf], post_state_scale_1[_lf], _tmem_load_6[_lf]);
                    }
                    uint32_t old_state_packed_1_f32_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(old_state_packed_1_f32[_lp*2 + 0], old_state_packed_1_f32[_lp*2+1 + 0]));
                        old_state_packed_1_f32_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&old_state_packed_1_f32[_pair * 2])[0]), "=f"((&old_state_packed_1_f32[_pair * 2])[1])
                            : "r"(old_state_packed_1_f32_bf16[_pair]));
                    }
                    tmem_st_x16_f32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_post * 32) + 16, old_state_packed_1_f32);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(smem_free_addr + (compute_stage) * 8);
                }
                {
                    int checkpoint_token = (chunk_idx + 1) * 16;
                }
                compute_stage += 1;
                if (compute_stage == 5) { compute_stage = 0; _phase_qk_full ^= 1; _phase_v_full ^= 1; _phase_old_out_ready ^= 1; _phase_u2_acc_ready ^= 1; _phase_final_ready ^= 1; }
            }
            if (store_final_state != 0) {
                #pragma unroll
                for (int state_col_block_2 = 0; state_col_block_2 < 4; state_col_block_2++) {
                    float _tmem_load_9[32];
                    tmem_ld_x32(&_tmem_load_9[0], taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_2 * 32));
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(_tmem_load_9[0 + 0], _tmem_load_9[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(_tmem_load_9[0 + 2], _tmem_load_9[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(_tmem_load_9[0 + 4], _tmem_load_9[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(_tmem_load_9[0 + 6], _tmem_load_9[0 + 7]);
                        _pk[4] = __floats2bfloat162_rn(_tmem_load_9[0 + 8], _tmem_load_9[0 + 9]);
                        _pk[5] = __floats2bfloat162_rn(_tmem_load_9[0 + 10], _tmem_load_9[0 + 11]);
                        _pk[6] = __floats2bfloat162_rn(_tmem_load_9[0 + 12], _tmem_load_9[0 + 13]);
                        _pk[7] = __floats2bfloat162_rn(_tmem_load_9[0 + 14], _tmem_load_9[0 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base + (long long)(state_col_block_2 * 32))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base + (long long)(state_col_block_2 * 32))))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(_tmem_load_9[16 + 0], _tmem_load_9[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(_tmem_load_9[16 + 2], _tmem_load_9[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(_tmem_load_9[16 + 4], _tmem_load_9[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(_tmem_load_9[16 + 6], _tmem_load_9[16 + 7]);
                        _pk[4] = __floats2bfloat162_rn(_tmem_load_9[16 + 8], _tmem_load_9[16 + 9]);
                        _pk[5] = __floats2bfloat162_rn(_tmem_load_9[16 + 10], _tmem_load_9[16 + 11]);
                        _pk[6] = __floats2bfloat162_rn(_tmem_load_9[16 + 12], _tmem_load_9[16 + 13]);
                        _pk[7] = __floats2bfloat162_rn(_tmem_load_9[16 + 14], _tmem_load_9[16 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base + (long long)(state_col_block_2 * 32) + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base + (long long)(state_col_block_2 * 32) + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            if (compute_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // epilogue_main
            int task_idx_1 = blockIdx.x;
            int split_compute_start_1 = 0;
            unsigned int _phase_work_item_ready_0 = 0;
            int seq_len_1 = (int)(cu_seqlens[seq_order[task_idx_1 / num_heads] + 1] - cu_seqlens[seq_order[task_idx_1 / num_heads]]);
            int num_chunks_1 = (seq_len_1 + 16 - 1) / 16;
            int warp_id_in_role_1 = (warp - 4);
            int epilogue_local_warp = warp_id_in_role_1;
            int warp_in_wg_1 = warp % 4;
            const int tmem_row_base_1 = warp_in_wg_1 * 32 << 16;
            int state_row_1 = warp_in_wg_1 * 32 + lane;
            unsigned int epilogue_stage = 0;
            unsigned int output_stage = 0;
            unsigned int checkpoint_stage_epilogue = 0;
            int epilogue_chunks = num_chunks_1;
            unsigned int _phase_checkpoint_ready = 0;
            unsigned int _phase_final_ready_1 = 0;
            #pragma unroll 1
            for (int chunk_idx_1 = 0; chunk_idx_1 < epilogue_chunks; chunk_idx_1++) {
                int checkpoint_token_epilogue = chunk_idx_1 * 16;
                int checkpoint_entering_epilogue = checkpoint_every_n_tokens != 0 && checkpoint_token_epilogue % checkpoint_every_n_tokens == 0;
                if (checkpoint_entering_epilogue != 0) {
                    mbarrier_wait(checkpoint_ready_addr + (checkpoint_stage_epilogue) * 8, _phase_checkpoint_ready);
                    if (epilogue_local_warp == 0) {
                        long long checkpoint_idx_epilogue = reinterpret_cast<long long*>(checkpoint_cu_starts_addr)[seq_order[task_idx_1 / num_heads]] + (long long)(checkpoint_token_epilogue / checkpoint_every_n_tokens);
                        if (elect_sync()) {
                            #pragma unroll
                            for (int checkpoint_segment = 0; checkpoint_segment < 2; checkpoint_segment++) {
                                tma_store_4d(state_checkpoints_tma, checkpoint_segment * 64, 0, task_idx_1 % num_heads, checkpoint_idx_epilogue, smem_checkpoint_addr + checkpoint_stage_epilogue * 32768 + (unsigned int)(checkpoint_segment * 128 * 64 * 2));
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        if (elect_sync()) {
                            mbarrier_arrive(checkpoint_free_addr + (checkpoint_stage_epilogue) * 8);
                        }
                    }
                    checkpoint_stage_epilogue += 1;
                    if (checkpoint_stage_epilogue == 2) { checkpoint_stage_epilogue = 0; _phase_checkpoint_ready ^= 1; }
                }
                int chunk_is_full = ((seq_len_1 >= (chunk_idx_1 + 1) * 16) ? 1 : 0);
                if (chunk_is_full != 0) {
                    mbarrier_wait(final_ready_addr + (epilogue_stage) * 8, _phase_final_ready_1);
                    float _tmem_load_10[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7]))
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1)
                        : "memory");
                    float _tmem_load_11[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7]))
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1 + 1048576)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
                    if (epilogue_local_warp == 0) {
                        if (chunk_idx_1 >= 2) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    int out_stage_addr = smem_out_addr + output_stage * 4096;
                    #pragma unroll
                    for (int dim_half = 0; dim_half < 2; dim_half++) {
                        unsigned int out_packed[8];
                        if (dim_half == 0) {
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 0], _tmem_load_10[_lp*2+1 + 0]));
                                out_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                        } else {
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                                out_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                        }
                        #pragma unroll
                        for (int token_group = 0; token_group < 1; token_group++) {
                            int mtx_idx = lane / 8;
                            int row_addr = lane & 7;
                            int dim_base = epilogue_local_warp * 32 + dim_half * 16 + (mtx_idx & 1) * 8;
                            int token_base = token_group * 16 + mtx_idx / 2 * 8;
                            int token_addr = token_base + row_addr;
                            int token_pair = token_addr / 2;
                            int token_parity = token_addr & 1;
                            int raw_row = token_pair + dim_base / 64 * 8;
                            int raw_col = (dim_base & 63 ^ (token_pair & 3) << 4 ^ token_parity << 3) + token_parity * 64;
                            int stsm_offset = (raw_row * 128 + raw_col) * 2;
                            const int pack_base = token_group * 4;
                            uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(out_stage_addr + stsm_offset));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 1])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 2])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 3]))
                                : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            tma_store_4d(out_tma, 0, (int)(cu_seqlens[seq_order[task_idx_1 / num_heads]] + (long long)(chunk_idx_1 * 16)), task_idx_1 % num_heads, 0, smem_out_addr + output_stage * 4096);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    output_stage = output_stage ^ 1;
                } else {
                    mbarrier_wait(final_ready_addr + (epilogue_stage) * 8, _phase_final_ready_1);
                    float _tmem_load_12[16];
                    tmem_ld_x16(&_tmem_load_12[0], taddr + 192 + (unsigned int)tmem_row_base_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
                    #pragma unroll
                    for (int token_col_1 = 0; token_col_1 < 16; token_col_1++) {
                        long long out_token = cu_seqlens[seq_order[task_idx_1 / num_heads]] + (long long)(chunk_idx_1 * 16 + token_col_1);
                        if (out_token < cu_seqlens[seq_order[task_idx_1 / num_heads] + 1]) {
                            long long out_idx = (out_token * (long long)num_heads + (long long)(task_idx_1 % num_heads)) * 128 + (long long)state_row_1;
                            out[out_idx] = _tmem_load_12[token_col_1];
                        }
                    }
                }
                epilogue_stage += 1;
                if (epilogue_stage == 5) { epilogue_stage = 0; _phase_final_ready_1 ^= 1; }
            }
            {
                if (epilogue_local_warp == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (epilogue_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int task_idx_2 = blockIdx.x;
            int split_compute_start_2 = 0;
            unsigned int _phase_work_item_ready_0_1 = 0;
            int seq_len_2 = (int)(cu_seqlens[seq_order[task_idx_2 / num_heads] + 1] - cu_seqlens[seq_order[task_idx_2 / num_heads]]);
            int num_chunks_2 = (seq_len_2 + 16 - 1) / 16;
            unsigned int mma_stage = 0;
            unsigned int _phase_qk_full_1 = 0;
            unsigned int _phase_state_inp_ready = 0;
            unsigned int _phase_out_empty_0 = 1;
            unsigned int _phase_u_inp_ready = 0;
            unsigned int _phase_u2_inp_ready = 0;
            unsigned int _phase_final_ready_2 = 0;
            #pragma unroll 1
            for (int _chunk_idx = 0; _chunk_idx < num_chunks_2; _chunk_idx++) {
                mbarrier_wait(qk_full_addr + (mma_stage) * 8, _phase_qk_full_1);
                mbarrier_wait(state_inp_ready_addr + (mma_stage) * 8, _phase_state_inp_ready);
                {
                    mbarrier_wait(out_empty_addr, _phase_out_empty_0);
                    _phase_out_empty_0 ^= 1;
                }
                {
                    int _mma_b_lo_0 = make_warp_uniform((((smem_qd_addr) >> 4) & 0x3FFF) + (mma_stage) * 1344);
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
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_out), "r"(_mma_b_lo_0), "r"(tmem_tmem_state_inp), "r"(0));
                }
                int _mma_b_lo_1 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + (mma_stage) * 1344);
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
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_u_acc), "r"(_mma_b_lo_1), "r"(tmem_tmem_state_inp), "r"(0));
                elect_commit2(old_out_ready_addr + (mma_stage) * 8, raw_inputs_free_addr + (mma_stage) * 8);
                mbarrier_wait(u_inp_ready_addr + (mma_stage) * 8, _phase_u_inp_ready);
                int _mma_b_lo_2 = make_warp_uniform((((smem_inv_addr) >> 4) & 0x3FFF) + (mma_stage) * 1344);
                mma_ts_step(tmem_tmem_u2_acc, tmem_tmem_u2_inp, _mma_b_lo_2, 0xC0004010, 134481040, 0);
                elect_commit(u2_acc_ready_addr + (mma_stage) * 8);
                mbarrier_wait(u2_inp_ready_addr + (mma_stage) * 8, _phase_u2_inp_ready);
                int _mma_b_lo_3 = make_warp_uniform(((((smem_kr_trans_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_stage) * 1344);
                mma_ts_step(tmem_tmem_state, tmem_tmem_u2_inp, _mma_b_lo_3, 0x40004040, 136381584, 0);
                int _mma_b_lo_4 = make_warp_uniform(((((smem_mqk_trans_addr) >> 4) & 0x3FFF) | 0x200000) + (mma_stage) * 1344);
                mma_ts_step(tmem_tmem_out, tmem_tmem_u2_inp, _mma_b_lo_4, 0xC0004010, 134546576, 1);
                elect_commit(final_ready_addr + (mma_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 5) { mma_stage = 0; _phase_qk_full_1 ^= 1; _phase_state_inp_ready ^= 1; _phase_u_inp_ready ^= 1; _phase_u2_inp_ready ^= 1; _phase_final_ready_2 ^= 1; }
            }
            unsigned int _phase_tmem_dealloc_ready_0 = 0;
            mbarrier_wait(tmem_dealloc_ready_addr, _phase_tmem_dealloc_ready_0);
            _phase_tmem_dealloc_ready_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    }
    // ---- Role: tape ----
    if (warp >= 10 && warp <= 11) {
        { // tape_main
            unsigned int _phase_work_item_ready_0_2 = 0;
            unsigned int _phase_tape_ready = 0;
        }
    }
    // ---- Role: prep ----
    if (warp >= 12 && warp <= 31) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // prep_main
            int task_idx_3 = blockIdx.x;
            int split_compute_start_3 = 0;
            unsigned int _phase_work_item_ready_0_3 = 0;
            int seq_len_3 = (int)(cu_seqlens[seq_order[task_idx_3 / num_heads] + 1] - cu_seqlens[seq_order[task_idx_3 / num_heads]]);
            int num_chunks_3 = (seq_len_3 + 16 - 1) / 16;
            int instance_id = (warp - 12) / 4;
            int prep_instance = instance_id;
            int warp_id_in_role_2 = (warp - 12);
            int prep_local_warp = warp_id_in_role_2 - prep_instance * 4;
            int prep_tid = prep_local_warp * 32 + lane;
            int num_prep_iters = (num_chunks_3 + 4 - prep_instance) / 5;
            unsigned int prep_stage = (unsigned int)prep_instance;
            int gate_rate_stage_f32 = prep_instance * 5376;
            if (prep_tid == 0) {
                float _expf_0 = __expf(A_log[task_idx_3 % num_heads]);
                smem_gate_rate_all[gate_rate_stage_f32] = _expf_0;
            }
            asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
            unsigned int _phase_raw_inputs_free = 1;
            unsigned int _phase_gate_raw_full = 0;
            unsigned int _phase_smem_free = 1;
            unsigned int _phase_v_free = 1;
            unsigned int _phase_qk_raw_full = 0;
            unsigned int _phase_prep_diag_ready = 0;
            unsigned int _phase_prep_inv16_ready = 0;
            unsigned int _phase_tape_free = 1;
            #pragma unroll 1
            for (int prep_iter = 0; prep_iter < num_prep_iters; prep_iter++) {
                int chunk_idx_2 = prep_iter * 5 + prep_instance;
                int chunk_global_local_1 = chunk_idx_2;
                int owned_chunk_1 = chunk_global_local_1 >= 0 && chunk_global_local_1 < ((int)(cu_seqlens[seq_order[task_idx_3 / num_heads] + 1] - cu_seqlens[seq_order[task_idx_3 / num_heads]]) + 16 - 1) / 16;
                int stage_f32 = prep_stage * 5376;
                int stage_bf16 = prep_stage * 10752;
                int chunk_is_full_1 = ((seq_len_3 >= (chunk_idx_2 + 1) * 16) ? 1 : 0);
                float early_beta_value = 0.0f;
                float early_gate0 = 0.0f;
                if (chunk_is_full_1 != 0 || prep_iter != 0) {
                    mbarrier_wait(raw_inputs_free_addr + (prep_stage) * 8, _phase_raw_inputs_free);
                }
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(gate_raw_full_addr + (prep_stage) * 8, 4352);
                            tma_3d_gmem2smem(smem_g_raw_addr + prep_stage * 21504, g_tma, 0, task_idx_3 % num_heads, (int)(cu_seqlens[seq_order[task_idx_3 / num_heads]] + (long long)(chunk_idx_2 * 16)), gate_raw_full_addr + (prep_stage) * 8);
                            tma_2d_gmem2smem(smem_beta_raw_addr + prep_stage * 21504, beta_tma, task_idx_3 % num_heads / 8 * 8, (int)(cu_seqlens[seq_order[task_idx_3 / num_heads]] + (long long)(chunk_idx_2 * 16)), gate_raw_full_addr + (prep_stage) * 8);
                            mbarrier_arrive_expect_tx(qk_raw_full_addr + (prep_stage) * 8, 8192);
                            tma_4d_gmem2smem(smem_kd_addr + prep_stage * 21504, k_tma, 0, (int)(cu_seqlens[seq_order[task_idx_3 / num_heads]] + (long long)(chunk_idx_2 * 16)), task_idx_3 % num_heads, 0, qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                    mbarrier_wait(gate_raw_full_addr + (prep_stage) * 8, _phase_gate_raw_full);
                    if (prep_local_warp == 2 && lane < 16) {
                        unsigned int beta_raw_pair[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&beta_raw_pair[0])) : "r"(smem_beta_raw_addr + prep_stage * 21504 + (unsigned int)(lane * 16) + (unsigned int)(task_idx_3 % num_heads % 8 / 2 * 4)));
                        float beta_raw_pair_f32[2];
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&beta_raw_pair_f32[_pair * 2])[0]), "=f"((&beta_raw_pair_f32[_pair * 2])[1])
                                : "r"(beta_raw_pair[_pair]));
                        }
                        float beta_logit = beta_raw_pair_f32[0];
                        if (task_idx_3 % num_heads % 2 != 0) {
                            beta_logit = beta_raw_pair_f32[1];
                        }
                        float _tanh_approx_0;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(beta_logit * 0.5f));
                        early_beta_value = _tanh_approx_0 * 0.5f + 0.5f;
                    }
                    if (prep_tid < 128) {
                        float early_gate_rate = smem_gate_rate_all[stage_f32];
                        float early_gate_bias = dt_bias[task_idx_3 % num_heads * 128 + prep_tid];
                        __nv_bfloat16 early_gate_raw = smem_g_raw_all[stage_bf16 + prep_tid];
                        float _cvt_f32_0 = __bfloat162float(early_gate_raw);
                        float early_gate_arg = early_gate_rate * (_cvt_f32_0 + early_gate_bias);
                        float _tanh_approx_1;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(early_gate_arg * 0.5f));
                        float early_gate_sigmoid = _tanh_approx_1 * 0.5f + 0.5f;
                        early_gate0 = lower_bound * 1.4426950408889634f * early_gate_sigmoid;
                    }
                }
                mbarrier_wait(smem_free_addr + (prep_stage) * 8, _phase_smem_free);
                mbarrier_wait(v_free_addr + (prep_stage) * 8, _phase_v_free);
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            tma_4d_gmem2smem(smem_q_raw_prefetch_addr + prep_stage * 21504, q_tma, 0, (int)(cu_seqlens[seq_order[task_idx_3 / num_heads]] + (long long)(chunk_idx_2 * 16)), task_idx_3 % num_heads, 0, qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                }
                if (chunk_is_full_1 == 0) {
                    #pragma unroll
                    for (int gate_load_pass = 0; gate_load_pass < 2; gate_load_pass++) {
                        int gate_load_item = gate_load_pass * 128 + prep_tid;
                        int gate_load_row = gate_load_item / 16;
                        int gate_load_segment = gate_load_item % 16;
                        long long gate_load_token = cu_seqlens[seq_order[task_idx_3 / num_heads]] + (long long)(chunk_idx_2 * 16 + gate_load_row);
                        long long gate_load_base = (gate_load_token * (long long)num_heads + (long long)(task_idx_3 % num_heads)) * 128 + (long long)(gate_load_segment * 8);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(smem_g_raw_addr + prep_stage * 21504 + (unsigned int)(gate_load_item * 16)), "l"(g + gate_load_base), "r"((gate_load_token < cu_seqlens[seq_order[task_idx_3 / num_heads] + 1]) ? 16 : 0));
                    }
                }
                if (chunk_is_full_1 == 0) {
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                }
                if (prep_local_warp == 2 && lane < 16) {
                    long long beta_token = cu_seqlens[seq_order[task_idx_3 / num_heads]] + (long long)(chunk_idx_2 * 16 + lane);
                    float beta_value = early_beta_value;
                    if (chunk_is_full_1 == 0) {
                        if (beta_token < cu_seqlens[seq_order[task_idx_3 / num_heads] + 1]) {
                            float beta_logit_1 = (float)beta[beta_token * (long long)num_heads + (long long)(task_idx_3 % num_heads)];
                            {
                                beta_logit_1 = (float)beta[beta_token * beta_token_stride + (long long)(task_idx_3 % num_heads)];
                            }
                            float _tanh_approx_2;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(beta_logit_1 * 0.5f));
                            beta_value = _tanh_approx_2 * 0.5f + 0.5f;
                        }
                    }
                    __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(beta_value);
                    float _cvt_f32_1 = __bfloat162float(_cvt_bf16_0);
                    smem_prep_beta_all[stage_f32 + lane] = _cvt_f32_1;
                }
                if (prep_tid < 128) {
                    int gate_col = prep_tid;
                    float gate_rate = smem_gate_rate_all[stage_f32];
                    float gate_bias = dt_bias[task_idx_3 % num_heads * 128 + gate_col];
                    float prefix_log2 = 0.0f;
                    for (int gate_row = 0; gate_row < 16; gate_row++) {
                        long long gate_token = cu_seqlens[seq_order[task_idx_3 / num_heads]] + (long long)(chunk_idx_2 * 16 + gate_row);
                        float gate_log2 = 0.0f;
                        int gate_needs_compute = 1;
                        if (gate_row == 0) {
                            if (chunk_is_full_1 != 0) {
                                gate_log2 = early_gate0;
                                gate_needs_compute = 0;
                            }
                        }
                        if (gate_needs_compute != 0) {
                            if (gate_token < cu_seqlens[seq_order[task_idx_3 / num_heads] + 1]) {
                                __nv_bfloat16 gate_raw = smem_g_raw_all[stage_bf16 + gate_row * 128 + gate_col];
                                float _cvt_f32_2 = __bfloat162float(gate_raw);
                                float gate_arg = gate_rate * (_cvt_f32_2 + gate_bias);
                                float _tanh_approx_3;
                                asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(gate_arg * 0.5f));
                                float gate_sigmoid = _tanh_approx_3 * 0.5f + 0.5f;
                                gate_log2 = lower_bound * 1.4426950408889634f * gate_sigmoid;
                            }
                        }
                        prefix_log2 += gate_log2;
                        smem_gate_all[stage_f32 + gate_row * 128 + gate_col] = prefix_log2;
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                if (chunk_is_full_1 != 0) {
                    mbarrier_wait(qk_raw_full_addr + (prep_stage) * 8, _phase_qk_raw_full);
                }
                if (prep_tid < 128) {
                    float total_log2 = smem_gt_prefix_all[stage_f32 + prep_tid];
                    float restore_factor_value = smem_gate_all[stage_f32 + 1024 + prep_tid];
                    smem_restore_factor_all[stage_f32 + prep_tid] = restore_factor_value;
                }
                if (prep_tid == 0) {
                    float _exp2_0 = approx_exp2(lower_bound * 1.4426950408889634f * 8.0f);
                    smem_restore_factor_all[stage_f32 + 128] = _exp2_0;
                }
                #pragma unroll 1
                for (int work_pass = 0; work_pass < 2; work_pass++) {
                    int work_item = work_pass * 128 + prep_tid;
                    int row = work_item / 16;
                    int segment = work_item % 16;
                    long long token = cu_seqlens[seq_order[task_idx_3 / num_heads]] + (long long)(chunk_idx_2 * 16 + row);
                    int token_valid = ((token < cu_seqlens[seq_order[task_idx_3 / num_heads] + 1]) ? 1 : 0);
                    long long gmem_base = (token * (long long)num_heads + (long long)(task_idx_3 % num_heads)) * 128 + (long long)(segment * 8);
                    float q_raw_vec[8];
                    float k_raw_vec[8];
                    q_raw_vec[0] = 0.0f;
                    q_raw_vec[1] = 0.0f;
                    q_raw_vec[2] = 0.0f;
                    q_raw_vec[3] = 0.0f;
                    q_raw_vec[4] = 0.0f;
                    q_raw_vec[5] = 0.0f;
                    q_raw_vec[6] = 0.0f;
                    q_raw_vec[7] = 0.0f;
                    k_raw_vec[0] = 0.0f;
                    k_raw_vec[1] = 0.0f;
                    k_raw_vec[2] = 0.0f;
                    k_raw_vec[3] = 0.0f;
                    k_raw_vec[4] = 0.0f;
                    k_raw_vec[5] = 0.0f;
                    k_raw_vec[6] = 0.0f;
                    k_raw_vec[7] = 0.0f;
                    if (chunk_is_full_1 != 0) {
                        unsigned int packed[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                            : "r"((smem_q_raw_prefetch_addr + prep_stage * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_f32[_pair * 2])[0]), "=f"((&packed_f32[_pair * 2])[1])
                                : "r"(packed[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx = 0; value_idx < 8; value_idx++) {
                            q_raw_vec[value_idx] = packed_f32[value_idx];
                        }
                        unsigned int packed_0[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 3]))
                            : "r"((smem_kd_addr + prep_stage * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_0_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_0_f32[_pair * 2])[0]), "=f"((&packed_0_f32[_pair * 2])[1])
                                : "r"(packed_0[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_1 = 0; value_idx_1 < 8; value_idx_1++) {
                            k_raw_vec[value_idx_1] = packed_0_f32[value_idx_1];
                        }
                    } else if (token_valid != 0) {
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q + gmem_base);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&q_raw_vec[0 + _blk * 8 + _pair * 2])[0]), "=f"((&q_raw_vec[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(k + gmem_base);
                            uint4 _vld_1[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_1[_blk] = _vptr_1[_blk];
                                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&k_raw_vec[0 + _blk * 8 + _pair * 2])[0]), "=f"((&k_raw_vec[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_1[_pair]));
                                }
                            }
                        }
                    }
                    float q_sum = 0.0f;
                    float k_sum = 0.0f;
                    for (int elem_in_segment = 0; elem_in_segment < 8; elem_in_segment++) {
                        float q_raw = q_raw_vec[elem_in_segment];
                        float k_raw = k_raw_vec[elem_in_segment];
                        float _fma_0 = __fmaf_rn(q_raw, q_raw, q_sum);
                        q_sum = _fma_0;
                        float _fma_1 = __fmaf_rn(k_raw, k_raw, k_sum);
                        k_sum = _fma_1;
                    }
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 8);
                    q_sum += _shfl_xor_0;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 8);
                    k_sum += _shfl_xor_1;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 4);
                    q_sum += _shfl_xor_2;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 4);
                    k_sum += _shfl_xor_3;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 2);
                    q_sum += _shfl_xor_4;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 2);
                    k_sum += _shfl_xor_5;
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 1);
                    q_sum += _shfl_xor_6;
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 1);
                    k_sum += _shfl_xor_7;
                    float _rsqrt_0 = rsqrtf(q_sum + 1e-06f);
                    float q_inv = _rsqrt_0;
                    float _rsqrt_1 = rsqrtf(k_sum + 1e-06f);
                    float k_inv = _rsqrt_1;
                    const float2 _scale2_2 = {q_inv, q_inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(q_raw_vec)[_ls], _scale2_2);
                    const float2 _scale2_3 = {k_inv, k_inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(k_raw_vec)[_ls], _scale2_3);
                    float qd_vec[8];
                    float kd_vec[8];
                    float ki_vec[8];
                    for (int elem_in_segment_1 = 0; elem_in_segment_1 < 8; elem_in_segment_1++) {
                        int col = segment * 8 + elem_in_segment_1;
                        float prefix = smem_gate_all[stage_f32 + row * 128 + col];
                        float common_log2 = smem_gate_all[stage_f32 + 1024 + col];
                        float _exp2_1 = approx_exp2(prefix - common_log2);
                        float decay = _exp2_1;
                        qd_vec[elem_in_segment_1] = decay;
                        kd_vec[elem_in_segment_1] = decay;
                        ki_vec[elem_in_segment_1] = k_raw_vec[elem_in_segment_1] / decay;
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], reinterpret_cast<const float2*>(q_raw_vec)[_ls]);
                    const float2 _scale2_4 = {scale, scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], _scale2_4);
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(kd_vec)[_ls], reinterpret_cast<const float2*>(k_raw_vec)[_ls]);
                    __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(scale);
                    float _cvt_f32_3 = __bfloat162float(_cvt_bf16_1);
                    float _exp2_2 = approx_exp2(smem_gate_all[stage_f32 + row * 128 + segment * 8]);
                    __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_exp2_2);
                    float _cvt_f32_4 = __bfloat162float(_cvt_bf16_2);
                    float _exp2_3 = approx_exp2(-smem_gate_all[stage_f32 + row * 128 + segment * 8]);
                    __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_exp2_3);
                    float _cvt_f32_5 = __bfloat162float(_cvt_bf16_3);
                    __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(q_raw_vec[0]);
                    float _cvt_f32_6 = __bfloat162float(_cvt_bf16_4);
                    __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(k_raw_vec[0]);
                    float _cvt_f32_7 = __bfloat162float(_cvt_bf16_5);
                    __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(_cvt_f32_6 * _cvt_f32_4);
                    float _cvt_f32_8 = __bfloat162float(_cvt_bf16_6);
                    __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(_cvt_f32_8 * _cvt_f32_3);
                    float _cvt_f32_9 = __bfloat162float(_cvt_bf16_7);
                    qd_vec[0] = _cvt_f32_9;
                    __nv_bfloat16 _cvt_bf16_8 = __float2bfloat16(_cvt_f32_7 * _cvt_f32_4);
                    float _cvt_f32_10 = __bfloat162float(_cvt_bf16_8);
                    kd_vec[0] = _cvt_f32_10;
                    __nv_bfloat16 _cvt_bf16_9 = __float2bfloat16(_cvt_f32_7 * _cvt_f32_5);
                    float _cvt_f32_11 = __bfloat162float(_cvt_bf16_9);
                    ki_vec[0] = _cvt_f32_11;
                    float _exp2_4 = approx_exp2(smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 1)]);
                    __nv_bfloat16 _cvt_bf16_10 = __float2bfloat16(_exp2_4);
                    float _cvt_f32_12 = __bfloat162float(_cvt_bf16_10);
                    float _exp2_5 = approx_exp2(-smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 1)]);
                    __nv_bfloat16 _cvt_bf16_11 = __float2bfloat16(_exp2_5);
                    float _cvt_f32_13 = __bfloat162float(_cvt_bf16_11);
                    __nv_bfloat16 _cvt_bf16_12 = __float2bfloat16(q_raw_vec[1]);
                    float _cvt_f32_14 = __bfloat162float(_cvt_bf16_12);
                    __nv_bfloat16 _cvt_bf16_13 = __float2bfloat16(k_raw_vec[1]);
                    float _cvt_f32_15 = __bfloat162float(_cvt_bf16_13);
                    __nv_bfloat16 _cvt_bf16_14 = __float2bfloat16(_cvt_f32_14 * _cvt_f32_12);
                    float _cvt_f32_16 = __bfloat162float(_cvt_bf16_14);
                    __nv_bfloat16 _cvt_bf16_15 = __float2bfloat16(_cvt_f32_16 * _cvt_f32_3);
                    float _cvt_f32_17 = __bfloat162float(_cvt_bf16_15);
                    qd_vec[1] = _cvt_f32_17;
                    __nv_bfloat16 _cvt_bf16_16 = __float2bfloat16(_cvt_f32_15 * _cvt_f32_12);
                    float _cvt_f32_18 = __bfloat162float(_cvt_bf16_16);
                    kd_vec[1] = _cvt_f32_18;
                    __nv_bfloat16 _cvt_bf16_17 = __float2bfloat16(_cvt_f32_15 * _cvt_f32_13);
                    float _cvt_f32_19 = __bfloat162float(_cvt_bf16_17);
                    ki_vec[1] = _cvt_f32_19;
                    float _exp2_6 = approx_exp2(smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 2)]);
                    __nv_bfloat16 _cvt_bf16_18 = __float2bfloat16(_exp2_6);
                    float _cvt_f32_20 = __bfloat162float(_cvt_bf16_18);
                    float _exp2_7 = approx_exp2(-smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 2)]);
                    __nv_bfloat16 _cvt_bf16_19 = __float2bfloat16(_exp2_7);
                    float _cvt_f32_21 = __bfloat162float(_cvt_bf16_19);
                    __nv_bfloat16 _cvt_bf16_20 = __float2bfloat16(q_raw_vec[2]);
                    float _cvt_f32_22 = __bfloat162float(_cvt_bf16_20);
                    __nv_bfloat16 _cvt_bf16_21 = __float2bfloat16(k_raw_vec[2]);
                    float _cvt_f32_23 = __bfloat162float(_cvt_bf16_21);
                    __nv_bfloat16 _cvt_bf16_22 = __float2bfloat16(_cvt_f32_22 * _cvt_f32_20);
                    float _cvt_f32_24 = __bfloat162float(_cvt_bf16_22);
                    __nv_bfloat16 _cvt_bf16_23 = __float2bfloat16(_cvt_f32_24 * _cvt_f32_3);
                    float _cvt_f32_25 = __bfloat162float(_cvt_bf16_23);
                    qd_vec[2] = _cvt_f32_25;
                    __nv_bfloat16 _cvt_bf16_24 = __float2bfloat16(_cvt_f32_23 * _cvt_f32_20);
                    float _cvt_f32_26 = __bfloat162float(_cvt_bf16_24);
                    kd_vec[2] = _cvt_f32_26;
                    __nv_bfloat16 _cvt_bf16_25 = __float2bfloat16(_cvt_f32_23 * _cvt_f32_21);
                    float _cvt_f32_27 = __bfloat162float(_cvt_bf16_25);
                    ki_vec[2] = _cvt_f32_27;
                    float _exp2_8 = approx_exp2(smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 3)]);
                    __nv_bfloat16 _cvt_bf16_26 = __float2bfloat16(_exp2_8);
                    float _cvt_f32_28 = __bfloat162float(_cvt_bf16_26);
                    float _exp2_9 = approx_exp2(-smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 3)]);
                    __nv_bfloat16 _cvt_bf16_27 = __float2bfloat16(_exp2_9);
                    float _cvt_f32_29 = __bfloat162float(_cvt_bf16_27);
                    __nv_bfloat16 _cvt_bf16_28 = __float2bfloat16(q_raw_vec[3]);
                    float _cvt_f32_30 = __bfloat162float(_cvt_bf16_28);
                    __nv_bfloat16 _cvt_bf16_29 = __float2bfloat16(k_raw_vec[3]);
                    float _cvt_f32_31 = __bfloat162float(_cvt_bf16_29);
                    __nv_bfloat16 _cvt_bf16_30 = __float2bfloat16(_cvt_f32_30 * _cvt_f32_28);
                    float _cvt_f32_32 = __bfloat162float(_cvt_bf16_30);
                    __nv_bfloat16 _cvt_bf16_31 = __float2bfloat16(_cvt_f32_32 * _cvt_f32_3);
                    float _cvt_f32_33 = __bfloat162float(_cvt_bf16_31);
                    qd_vec[3] = _cvt_f32_33;
                    __nv_bfloat16 _cvt_bf16_32 = __float2bfloat16(_cvt_f32_31 * _cvt_f32_28);
                    float _cvt_f32_34 = __bfloat162float(_cvt_bf16_32);
                    kd_vec[3] = _cvt_f32_34;
                    __nv_bfloat16 _cvt_bf16_33 = __float2bfloat16(_cvt_f32_31 * _cvt_f32_29);
                    float _cvt_f32_35 = __bfloat162float(_cvt_bf16_33);
                    ki_vec[3] = _cvt_f32_35;
                    float _exp2_10 = approx_exp2(smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 4)]);
                    __nv_bfloat16 _cvt_bf16_34 = __float2bfloat16(_exp2_10);
                    float _cvt_f32_36 = __bfloat162float(_cvt_bf16_34);
                    float _exp2_11 = approx_exp2(-smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 4)]);
                    __nv_bfloat16 _cvt_bf16_35 = __float2bfloat16(_exp2_11);
                    float _cvt_f32_37 = __bfloat162float(_cvt_bf16_35);
                    __nv_bfloat16 _cvt_bf16_36 = __float2bfloat16(q_raw_vec[4]);
                    float _cvt_f32_38 = __bfloat162float(_cvt_bf16_36);
                    __nv_bfloat16 _cvt_bf16_37 = __float2bfloat16(k_raw_vec[4]);
                    float _cvt_f32_39 = __bfloat162float(_cvt_bf16_37);
                    __nv_bfloat16 _cvt_bf16_38 = __float2bfloat16(_cvt_f32_38 * _cvt_f32_36);
                    float _cvt_f32_40 = __bfloat162float(_cvt_bf16_38);
                    __nv_bfloat16 _cvt_bf16_39 = __float2bfloat16(_cvt_f32_40 * _cvt_f32_3);
                    float _cvt_f32_41 = __bfloat162float(_cvt_bf16_39);
                    qd_vec[4] = _cvt_f32_41;
                    __nv_bfloat16 _cvt_bf16_40 = __float2bfloat16(_cvt_f32_39 * _cvt_f32_36);
                    float _cvt_f32_42 = __bfloat162float(_cvt_bf16_40);
                    kd_vec[4] = _cvt_f32_42;
                    __nv_bfloat16 _cvt_bf16_41 = __float2bfloat16(_cvt_f32_39 * _cvt_f32_37);
                    float _cvt_f32_43 = __bfloat162float(_cvt_bf16_41);
                    ki_vec[4] = _cvt_f32_43;
                    float _exp2_12 = approx_exp2(smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 5)]);
                    __nv_bfloat16 _cvt_bf16_42 = __float2bfloat16(_exp2_12);
                    float _cvt_f32_44 = __bfloat162float(_cvt_bf16_42);
                    float _exp2_13 = approx_exp2(-smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 5)]);
                    __nv_bfloat16 _cvt_bf16_43 = __float2bfloat16(_exp2_13);
                    float _cvt_f32_45 = __bfloat162float(_cvt_bf16_43);
                    __nv_bfloat16 _cvt_bf16_44 = __float2bfloat16(q_raw_vec[5]);
                    float _cvt_f32_46 = __bfloat162float(_cvt_bf16_44);
                    __nv_bfloat16 _cvt_bf16_45 = __float2bfloat16(k_raw_vec[5]);
                    float _cvt_f32_47 = __bfloat162float(_cvt_bf16_45);
                    __nv_bfloat16 _cvt_bf16_46 = __float2bfloat16(_cvt_f32_46 * _cvt_f32_44);
                    float _cvt_f32_48 = __bfloat162float(_cvt_bf16_46);
                    __nv_bfloat16 _cvt_bf16_47 = __float2bfloat16(_cvt_f32_48 * _cvt_f32_3);
                    float _cvt_f32_49 = __bfloat162float(_cvt_bf16_47);
                    qd_vec[5] = _cvt_f32_49;
                    __nv_bfloat16 _cvt_bf16_48 = __float2bfloat16(_cvt_f32_47 * _cvt_f32_44);
                    float _cvt_f32_50 = __bfloat162float(_cvt_bf16_48);
                    kd_vec[5] = _cvt_f32_50;
                    __nv_bfloat16 _cvt_bf16_49 = __float2bfloat16(_cvt_f32_47 * _cvt_f32_45);
                    float _cvt_f32_51 = __bfloat162float(_cvt_bf16_49);
                    ki_vec[5] = _cvt_f32_51;
                    float _exp2_14 = approx_exp2(smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 6)]);
                    __nv_bfloat16 _cvt_bf16_50 = __float2bfloat16(_exp2_14);
                    float _cvt_f32_52 = __bfloat162float(_cvt_bf16_50);
                    float _exp2_15 = approx_exp2(-smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 6)]);
                    __nv_bfloat16 _cvt_bf16_51 = __float2bfloat16(_exp2_15);
                    float _cvt_f32_53 = __bfloat162float(_cvt_bf16_51);
                    __nv_bfloat16 _cvt_bf16_52 = __float2bfloat16(q_raw_vec[6]);
                    float _cvt_f32_54 = __bfloat162float(_cvt_bf16_52);
                    __nv_bfloat16 _cvt_bf16_53 = __float2bfloat16(k_raw_vec[6]);
                    float _cvt_f32_55 = __bfloat162float(_cvt_bf16_53);
                    __nv_bfloat16 _cvt_bf16_54 = __float2bfloat16(_cvt_f32_54 * _cvt_f32_52);
                    float _cvt_f32_56 = __bfloat162float(_cvt_bf16_54);
                    __nv_bfloat16 _cvt_bf16_55 = __float2bfloat16(_cvt_f32_56 * _cvt_f32_3);
                    float _cvt_f32_57 = __bfloat162float(_cvt_bf16_55);
                    qd_vec[6] = _cvt_f32_57;
                    __nv_bfloat16 _cvt_bf16_56 = __float2bfloat16(_cvt_f32_55 * _cvt_f32_52);
                    float _cvt_f32_58 = __bfloat162float(_cvt_bf16_56);
                    kd_vec[6] = _cvt_f32_58;
                    __nv_bfloat16 _cvt_bf16_57 = __float2bfloat16(_cvt_f32_55 * _cvt_f32_53);
                    float _cvt_f32_59 = __bfloat162float(_cvt_bf16_57);
                    ki_vec[6] = _cvt_f32_59;
                    float _exp2_16 = approx_exp2(smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 7)]);
                    __nv_bfloat16 _cvt_bf16_58 = __float2bfloat16(_exp2_16);
                    float _cvt_f32_60 = __bfloat162float(_cvt_bf16_58);
                    float _exp2_17 = approx_exp2(-smem_gate_all[stage_f32 + row * 128 + (segment * 8 + 7)]);
                    __nv_bfloat16 _cvt_bf16_59 = __float2bfloat16(_exp2_17);
                    float _cvt_f32_61 = __bfloat162float(_cvt_bf16_59);
                    __nv_bfloat16 _cvt_bf16_60 = __float2bfloat16(q_raw_vec[7]);
                    float _cvt_f32_62 = __bfloat162float(_cvt_bf16_60);
                    __nv_bfloat16 _cvt_bf16_61 = __float2bfloat16(k_raw_vec[7]);
                    float _cvt_f32_63 = __bfloat162float(_cvt_bf16_61);
                    __nv_bfloat16 _cvt_bf16_62 = __float2bfloat16(_cvt_f32_62 * _cvt_f32_60);
                    float _cvt_f32_64 = __bfloat162float(_cvt_bf16_62);
                    __nv_bfloat16 _cvt_bf16_63 = __float2bfloat16(_cvt_f32_64 * _cvt_f32_3);
                    float _cvt_f32_65 = __bfloat162float(_cvt_bf16_63);
                    qd_vec[7] = _cvt_f32_65;
                    __nv_bfloat16 _cvt_bf16_64 = __float2bfloat16(_cvt_f32_63 * _cvt_f32_60);
                    float _cvt_f32_66 = __bfloat162float(_cvt_bf16_64);
                    kd_vec[7] = _cvt_f32_66;
                    __nv_bfloat16 _cvt_bf16_65 = __float2bfloat16(_cvt_f32_63 * _cvt_f32_61);
                    float _cvt_f32_67 = __bfloat162float(_cvt_bf16_65);
                    ki_vec[7] = _cvt_f32_67;
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word = 0; word < 4; word++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + prep_stage * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word * 4)), "r"((packed_1[word])));
                    }
                    unsigned int packed_0_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_vec[_lp*2 + 0], kd_vec[_lp*2+1 + 0]));
                        packed_0_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_1 = 0; word_1 < 4; word_1++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_1 * 4)), "r"((packed_0_1[word_1])));
                    }
                    unsigned int packed_1_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_vec[_lp*2 + 0], ki_vec[_lp*2+1 + 0]));
                        packed_1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_2 = 0; word_2 < 4; word_2++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_ki_addr + prep_stage * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_2 * 4)), "r"((packed_1_1[word_2])));
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                unsigned int a_frag[4];
                unsigned int b_frag[4];
                float acc[8];
                {
                    if (prep_local_warp == 0) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)(((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)(((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)(((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        int row0 = lane / 4;
                        int row1 = row0 + 8;
                        int col0 = lane % 4 * 2;
                        float beta0 = smem_prep_beta_all[stage_f32 + row0];
                        float beta1 = smem_prep_beta_all[stage_f32 + row1];
                        float seed[8];
                        seed[0] = 0.0f;
                        seed[1] = 0.0f;
                        seed[2] = 0.0f;
                        seed[3] = 0.0f;
                        seed[4] = 0.0f;
                        seed[5] = 0.0f;
                        seed[6] = 0.0f;
                        seed[7] = 0.0f;
                        if (row0 > col0) {
                            seed[0] = acc[0] * beta0;
                        }
                        if (row0 > col0 + 1) {
                            seed[1] = acc[1] * beta0;
                        }
                        if (row1 > col0) {
                            seed[2] = acc[2] * beta1;
                        }
                        if (row1 > col0 + 1) {
                            seed[3] = acc[3] * beta1;
                        }
                        if (row0 > col0 + 8) {
                            seed[4] = acc[4] * beta0;
                        }
                        if (row0 > col0 + 9) {
                            seed[5] = acc[5] * beta0;
                        }
                        if (row1 > col0 + 8) {
                            seed[6] = acc[6] * beta1;
                        }
                        if (row1 > col0 + 9) {
                            seed[7] = acc[7] * beta1;
                        }
                        unsigned int seed_packed[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(seed[_lp*2 + 0], seed[_lp*2+1 + 0]));
                            seed_packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        int seed_lane_row = lane % 16;
                        int seed_lane_col = lane / 16 * 8;
                        int byte_off = (int)prep_stage * 21504 + seed_lane_row * 128 + seed_lane_col * 2;
                        int swizzled_off = byte_off ^ (byte_off >> 7 & 7) << 4;
                        int seed_addr = smem_inv_work_addr + (unsigned int)swizzled_off;
                        uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)seed_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[3]))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)(((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)(((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)(((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 21504 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        int row0_0 = lane / 4;
                        int row1_1 = row0_0 + 8;
                        int col0_2 = lane % 4 * 2;
                        float mqk[8];
                        mqk[0] = 0.0f;
                        mqk[1] = 0.0f;
                        mqk[2] = 0.0f;
                        mqk[3] = 0.0f;
                        mqk[4] = 0.0f;
                        mqk[5] = 0.0f;
                        mqk[6] = 0.0f;
                        mqk[7] = 0.0f;
                        if (row0_0 >= col0_2) {
                            mqk[0] = acc[0];
                        }
                        if (row0_0 >= col0_2 + 1) {
                            mqk[1] = acc[1];
                        }
                        if (row1_1 >= col0_2) {
                            mqk[2] = acc[2];
                        }
                        if (row1_1 >= col0_2 + 1) {
                            mqk[3] = acc[3];
                        }
                        if (row0_0 >= col0_2 + 8) {
                            mqk[4] = acc[4];
                        }
                        if (row0_0 >= col0_2 + 9) {
                            mqk[5] = acc[5];
                        }
                        if (row1_1 >= col0_2 + 8) {
                            mqk[6] = acc[6];
                        }
                        if (row1_1 >= col0_2 + 9) {
                            mqk[7] = acc[7];
                        }
                        unsigned int mqk_packed[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mqk[_lp*2 + 0], mqk[_lp*2+1 + 0]));
                            mqk_packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int publish_pair = 0; publish_pair < 2; publish_pair++) {
                            int publish_row = publish_pair * 8 + (lane & 7);
                            int publish_col = lane / 8 * 8;
                            uint32_t _stmatrix_addr_6 = static_cast<uint32_t>((unsigned long long)(smem_mqk_trans_addr + prep_stage * 21504 + (unsigned int)(publish_col / 16 * 512 + publish_row * 32 + publish_col % 16 * 2 ^ (publish_col / 16 * 512 + publish_row * 32 + publish_col % 16 * 2 >> 7 & 1) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                                :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2 + 1]))
                                : "memory");
                        }
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                long long tape_scaled_base = 0;
                if (prep_tid < 128) {
                    float total_log2_1 = smem_gt_prefix_all[stage_f32 + prep_tid];
                    float _exp2_18 = approx_exp2(total_log2_1);
                    smem_gt_all[stage_f32 + prep_tid] = _exp2_18;
                }
                {
                    if (prep_local_warp >= 2) {
                        int stage_f32_0 = prep_stage * 5376;
                        int restore_segment = lane & 15;
                        #pragma unroll 1
                        for (int restore_k_pass = 0; restore_k_pass < 4; restore_k_pass++) {
                            int restore_row = (prep_local_warp - 2) * 8 + restore_k_pass * 2 + (lane >> 4);
                            float restore_ki_values[8];
                            float restore_kr_values[8];
                            unsigned int packed_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 3]))
                                : "r"((smem_ki_addr + prep_stage * 21504 + (unsigned int)(restore_segment * 8 / 64 * 2048 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 2048 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_f32_1[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_f32_1[_pair * 2])[0]), "=f"((&packed_f32_1[_pair * 2])[1])
                                    : "r"(packed_2[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_2 = 0; value_idx_2 < 8; value_idx_2++) {
                                restore_ki_values[value_idx_2] = packed_f32_1[value_idx_2];
                            }
                            #pragma unroll
                            for (int restore_elem = 0; restore_elem < 8; restore_elem++) {
                                int restore_col = restore_segment * 8 + restore_elem;
                                __nv_bfloat16 _cvt_bf16_66 = __float2bfloat16(restore_ki_values[restore_elem]);
                                float _cvt_f32_68 = __bfloat162float(_cvt_bf16_66);
                                float restore_ki_carrier = _cvt_f32_68;
                                float restore_total_log2 = smem_gt_prefix_all[stage_f32_0 + restore_col];
                                float _exp2_19 = approx_exp2(restore_total_log2);
                                __nv_bfloat16 _cvt_bf16_67 = __float2bfloat16(_exp2_19);
                                float _cvt_f32_69 = __bfloat162float(_cvt_bf16_67);
                                float restore_total_carrier = _cvt_f32_69;
                                __nv_bfloat16 _cvt_bf16_68 = __float2bfloat16(restore_ki_carrier * restore_total_carrier);
                                float _cvt_f32_70 = __bfloat162float(_cvt_bf16_68);
                                restore_kr_values[restore_elem] = _cvt_f32_70;
                            }
                            unsigned int packed_0_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values[_lp*2 + 0], restore_kr_values[_lp*2+1 + 0]));
                                packed_0_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_3 = 0; word_3 < 4; word_3++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + prep_stage * 21504 + (unsigned int)(restore_segment * 8 / 64 * 2048 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 2048 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_3 * 4)), "r"((packed_0_2[word_3])));
                            }
                        }
                    }
                }
                if (prep_local_warp == 0) {
                    int inverse_row = lane;
                    int diag_block = inverse_row / 8;
                    int lane_in_diag = lane & 7;
                    float inv_row[8];
                    {
                        inv_row[0] = 0.0f;
                        inv_row[1] = 0.0f;
                        inv_row[2] = 0.0f;
                        inv_row[3] = 0.0f;
                        inv_row[4] = 0.0f;
                        inv_row[5] = 0.0f;
                        inv_row[6] = 0.0f;
                        inv_row[7] = 0.0f;
                        if (inverse_row < 16) {
                            unsigned int packed_3[4];
                            int byte_off_1 = (int)prep_stage * 21504 + inverse_row * 128 + diag_block * 8 * 2;
                            int swizzled_off_1 = byte_off_1 ^ (byte_off_1 >> 7 & 7) << 4;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 3]))
                                : "r"(smem_inv_work_addr + (unsigned int)swizzled_off_1));
                            float packed_f32_2[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_f32_2[_pair * 2])[0]), "=f"((&packed_f32_2[_pair * 2])[1])
                                    : "r"(packed_3[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_3 = 0; value_idx_3 < 8; value_idx_3++) {
                                inv_row[value_idx_3] = packed_f32_2[value_idx_3];
                            }
                        }
                    }
                    #pragma unroll
                    for (int diag_elem = 0; diag_elem < 8; diag_elem++) {
                        if (lane_in_diag == diag_elem) {
                            inv_row[diag_elem] = 1.0f;
                        }
                    }
                    int diag_group_base = lane - lane_in_diag;
                    #pragma unroll
                    for (int src_row = 0; src_row < 7; src_row++) {
                        float row_scale = -inv_row[src_row];
                        #pragma unroll
                        for (int prev_col = 0; prev_col < src_row; prev_col++) {
                            int pivot_lane = diag_group_base + src_row;
                            float _shfl_0 = __shfl_sync(0xFFFFFFFF, inv_row[prev_col], pivot_lane);
                            float pivot = _shfl_0;
                            if (lane_in_diag > src_row) {
                                float _fma_2 = __fmaf_rn(row_scale, pivot, inv_row[prev_col]);
                                inv_row[prev_col] = _fma_2;
                            }
                        }
                        if (lane_in_diag > src_row) {
                            inv_row[src_row] = row_scale;
                        }
                    }
                    {
                        if (inverse_row < 16) {
                            unsigned int packed_4[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv_row[_lp*2 + 0], inv_row[_lp*2+1 + 0]));
                                packed_4[_lp] = *(uint32_t*)&_bf2;
                            }
                            int byte_off_2 = (int)prep_stage * 21504 + inverse_row * 128 + diag_block * 8 * 2;
                            int swizzled_off_2 = byte_off_2 ^ (byte_off_2 >> 7 & 7) << 4;
                            #pragma unroll
                            for (int word_4 = 0; word_4 < 4; word_4++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_inv_work_addr + (unsigned int)swizzled_off_2 + (unsigned int)(word_4 * 4)), "r"((packed_4[word_4])));
                            }
                        }
                    }
                }
                if (prep_local_warp < 2) {
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(prep_diag_ready_addr + (prep_stage) * 8);
                    }
                    mbarrier_wait(prep_diag_ready_addr + (prep_stage) * 8, _phase_prep_diag_ready);
                }
                {
                    if (prep_local_warp < 2) {
                        if (prep_local_warp == 0) {
                            int lane_row = lane & 7;
                            int byte_off_3 = (int)prep_stage * 21504 + (8 + lane_row) * 128 + 16;
                            int swizzled_off_3 = byte_off_3 ^ (byte_off_3 >> 7 & 7) << 4;
                            int d_addr = smem_inv_work_addr + (unsigned int)swizzled_off_3;
                            int byte_off_0 = (int)prep_stage * 21504 + (8 + lane_row) * 128;
                            int swizzled_off_1_1 = byte_off_0 ^ (byte_off_0 >> 7 & 7) << 4;
                            int c_addr = smem_inv_work_addr + (unsigned int)swizzled_off_1_1;
                            int byte_off_2_1 = (int)prep_stage * 21504 + lane_row * 128;
                            int swizzled_off_3_1 = byte_off_2_1 ^ (byte_off_2_1 >> 7 & 7) << 4;
                            int a_addr = smem_inv_work_addr + (unsigned int)swizzled_off_3_1;
                            unsigned int d_frag[2];
                            unsigned int c_frag[1];
                            float dc_acc[4];
                            unsigned int dc_bf16[2];
                            unsigned int inv_a_frag[1];
                            float o_acc[4];
                            unsigned int o_bf16[2];
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                                : "=r"(d_frag[0])
                                : "r"(d_addr)
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                                : "=r"(d_frag[1])
                                : "r"(d_addr)
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                                : "=r"(c_frag[0])
                                : "r"(c_addr)
                                : "memory");
                            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                                : "=f"(dc_acc[0]), "=f"(dc_acc[1]), "=f"(dc_acc[2]), "=f"(dc_acc[3])
                                : "r"(d_frag[0]), "r"(d_frag[1]), "r"(c_frag[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            const float2 _scale2_7 = {-1.0f, -1.0f};
                            #pragma unroll
                            for (int _ls = 0; _ls < 2; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(dc_acc)[_ls], _scale2_7);
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dc_acc[_lp*2 + 0], dc_acc[_lp*2+1 + 0]));
                                dc_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                                : "=r"(inv_a_frag[0])
                                : "r"(a_addr)
                                : "memory");
                            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                                : "=f"(o_acc[0]), "=f"(o_acc[1]), "=f"(o_acc[2]), "=f"(o_acc[3])
                                : "r"(dc_bf16[0]), "r"(dc_bf16[1]), "r"(inv_a_frag[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_acc[_lp*2 + 0], o_acc[_lp*2+1 + 0]));
                                o_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            int byte_off_4 = (int)prep_stage * 21504 + (8 + lane_row) * 128;
                            int swizzled_off_5 = byte_off_4 ^ (byte_off_4 >> 7 & 7) << 4;
                            int o_addr = smem_inv_work_addr + (unsigned int)swizzled_off_5;
                            uint32_t _stmatrix_addr_8 = static_cast<uint32_t>((unsigned long long)o_addr);
                            asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                                :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&o_bf16[0]))
                                : "memory");
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(prep_inv16_ready_addr + (prep_stage) * 8);
                        }
                        mbarrier_wait(prep_inv16_ready_addr + (prep_stage) * 8, _phase_prep_inv16_ready);
                    }
                }
                {
                    if (prep_local_warp == 0) {
                        int lane_row_1 = lane % 16;
                        int lane_col = lane / 16 * 8;
                        int byte_off_5 = (int)prep_stage * 21504 + lane_row_1 * 128 + lane_col * 2;
                        int swizzled_off_4 = byte_off_5 ^ (byte_off_5 >> 7 & 7) << 4;
                        int inv16_addr = smem_inv_work_addr + (unsigned int)swizzled_off_4;
                        unsigned int inv16_frag[4];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv16_frag[0]), "=r"(inv16_frag[1]), "=r"(inv16_frag[2]), "=r"(inv16_frag[3])
                            : "r"(inv16_addr)
                            : "memory");
                        int inv16_publish_addr = (smem_inv_addr + prep_stage * 21504 + (unsigned int)(lane_col / 16 * 512 + lane_row_1 * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 512 + lane_row_1 * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_9 = static_cast<uint32_t>((unsigned long long)inv16_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&inv16_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv16_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv16_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv16_frag[3]))
                            : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                if (prep_local_warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(qk_full_addr + (prep_stage) * 8);
                    }
                }
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(v_full_addr + (prep_stage) * 8, 4096);
                            tma_3d_gmem2smem(smem_v_addr + prep_stage * 21504, v_tma, 0, task_idx_3 % num_heads, (int)(cu_seqlens[seq_order[task_idx_3 / num_heads]] + (long long)(chunk_idx_2 * 16)), v_full_addr + (prep_stage) * 8);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int v_load_iter = 0; v_load_iter < 2; v_load_iter++) {
                        int v_item = v_load_iter * 128 + prep_tid;
                        int row_1 = v_item / 16;
                        int segment_1 = v_item % 16;
                        long long token_1 = cu_seqlens[seq_order[task_idx_3 / num_heads]] + (long long)(chunk_idx_2 * 16 + row_1);
                        int token_valid_1 = ((token_1 < cu_seqlens[seq_order[task_idx_3 / num_heads] + 1]) ? 1 : 0);
                        long long v_src = (token_1 * (long long)num_heads + (long long)(task_idx_3 % num_heads)) * 128 + (long long)(segment_1 * 8);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(smem_v_addr + prep_stage * 21504 + (unsigned int)((row_1 * 128 + segment_1 * 8) * 2)), "l"(v + v_src), "r"((token_valid_1 != 0) ? 16 : 0));
                    }
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(v_full_addr + (prep_stage) * 8);
                        }
                    }
                }
                for (int _advance = 0; _advance < 5; _advance++) {
                    prep_stage += 1;
                    if (prep_stage == 5) { prep_stage = 0; _phase_raw_inputs_free ^= 1; _phase_gate_raw_full ^= 1; _phase_smem_free ^= 1; _phase_v_free ^= 1; _phase_qk_raw_full ^= 1; _phase_prep_diag_ready ^= 1; _phase_prep_inv16_ready ^= 1; _phase_tape_free ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

// clang-format on
