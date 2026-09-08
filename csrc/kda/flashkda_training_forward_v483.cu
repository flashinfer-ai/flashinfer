/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

// Frozen generated training-forward kernel; do not edit the generated body.
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
#define TMEM_NCOLS 272
#define TMEM_TMEM_STATE_OFFSET 0
#define TMEM_TMEM_STATE_INP_OFFSET 128
#define TMEM_TMEM_Q_STATE_OFFSET 192
#define TMEM_TMEM_STATE_K_OFFSET 224
#define TMEM_TMEM_U_ACC_OFFSET 240
#define TMEM_TMEM_Y_INP_OFFSET 256
#define TMEM_TMEM_U_INP_OFFSET 264
#define NUM_SCHED_PIPE_STAGES 8
#define NUM_RAW_PIPE_STAGES 5
#define NUM_RAW_BAR_PIPE_STAGES 6
#define NUM_DECAY_PIPE_STAGES 2
#define NUM_INTERMEDIATE_PIPE_STAGES 2
#define NUM_DIAG_PIPE_STAGES 4
#define NUM_STATE_PIPE_STAGES 1
#define NUM_O_PIPE_STAGES 2
#define NUM_CHECKPOINT_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 4096
#define SMEM_SMEM_Q_STRIDE 4096
#define SMEM_SMEM_K_OFF 21504
#define SMEM_SMEM_K_STAGE_BYTES 4096
#define SMEM_SMEM_K_STRIDE 4096
#define SMEM_SMEM_V_OFF 41984
#define SMEM_SMEM_V_STAGE_BYTES 4096
#define SMEM_SMEM_V_STRIDE 4096
#define SMEM_SMEM_G_OFF 62464
#define SMEM_SMEM_G_STAGE_BYTES 8192
#define SMEM_SMEM_G_STRIDE 8192
#define SMEM_SMEM_BETA_OFF 103424
#define SMEM_SMEM_BETA_STAGE_BYTES 32
#define SMEM_SMEM_BETA_STRIDE 32
#define SMEM_SMEM_BETA_ALL_OFF 103424
#define SMEM_SMEM_BETA_ALL_STAGE_BYTES 192
#define SMEM_SMEM_BETA_ALL_STRIDE 192
#define SMEM_SMEM_K_INV_OFF 104448
#define SMEM_SMEM_K_INV_STAGE_BYTES 4096
#define SMEM_SMEM_K_INV_STRIDE 4096
#define SMEM_SMEM_K_DECAY_OFF 112640
#define SMEM_SMEM_K_DECAY_STAGE_BYTES 4096
#define SMEM_SMEM_K_DECAY_STRIDE 4096
#define SMEM_SMEM_Q_DECAY_OFF 120832
#define SMEM_SMEM_Q_DECAY_STAGE_BYTES 4096
#define SMEM_SMEM_Q_DECAY_STRIDE 4096
#define SMEM_SMEM_K_RESTORE_OFF 129024
#define SMEM_SMEM_K_RESTORE_STAGE_BYTES 4096
#define SMEM_SMEM_K_RESTORE_STRIDE 4096
#define SMEM_SMEM_K_RESTORE_MN_OFF 129024
#define SMEM_SMEM_K_RESTORE_MN_STAGE_BYTES 4096
#define SMEM_SMEM_K_RESTORE_MN_STRIDE 4096
#define SMEM_SMEM_TINV_OFF 137216
#define SMEM_SMEM_TINV_STAGE_BYTES 512
#define SMEM_SMEM_TINV_STRIDE 1024
#define SMEM_SMEM_A_OFF 137728
#define SMEM_SMEM_A_STAGE_BYTES 512
#define SMEM_SMEM_A_STRIDE 1024
#define SMEM_SMEM_TINV_MN_OFF 137216
#define SMEM_SMEM_TINV_MN_STAGE_BYTES 512
#define SMEM_SMEM_TINV_MN_STRIDE 1024
#define SMEM_SMEM_A_MN_OFF 137728
#define SMEM_SMEM_A_MN_STAGE_BYTES 512
#define SMEM_SMEM_A_MN_STRIDE 1024
#define SMEM_SMEM_STATE_DIAG_OFF 139264
#define SMEM_SMEM_STATE_DIAG_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DIAG_STRIDE 512
#define SMEM_SMEM_O_OFF 155648
#define SMEM_SMEM_O_STAGE_BYTES 4096
#define SMEM_SMEM_O_STRIDE 4096
#define SMEM_SMEM_CHECKPOINT_OFF 163840
#define SMEM_SMEM_CHECKPOINT_STAGE_BYTES 32768
#define SMEM_SMEM_CHECKPOINT_STRIDE 32768
#define SMEM_TINV_SCRATCH_OFF 229376
#define SMEM_TINV_SCRATCH_STAGE_BYTES 512
#define SMEM_TINV_SCRATCH_STRIDE 512
#define SMEM_SCHED_SLOT_OFF 229888
#define SMEM_SCHED_SLOT_STAGE_BYTES 4
#define SMEM_SCHED_SLOT_STRIDE 4
#define SMEM_SMEM_Q_ALL_OFF 1024
#define SMEM_SMEM_Q_ALL_STAGE_BYTES 20480
#define SMEM_SMEM_Q_ALL_STRIDE 20480
#define SMEM_SMEM_K_ALL_OFF 21504
#define SMEM_SMEM_K_ALL_STAGE_BYTES 20480
#define SMEM_SMEM_K_ALL_STRIDE 20480
#define SMEM_SMEM_G_ALL_OFF 62464
#define SMEM_SMEM_G_ALL_STAGE_BYTES 40960
#define SMEM_SMEM_G_ALL_STRIDE 40960
#define SMEM_SMEM_V_ALL_OFF 41984
#define SMEM_SMEM_V_ALL_STAGE_BYTES 20480
#define SMEM_SMEM_V_ALL_STRIDE 20480
#define SMEM_SMEM_O_ALL_OFF 155648
#define SMEM_SMEM_O_ALL_STAGE_BYTES 8192
#define SMEM_SMEM_O_ALL_STRIDE 8192
#define SMEM_TOTAL 230016
#define THREADS 512
#define USE_INITIAL_STATE 1
#define STORE_FINAL_STATE 1
#define ENABLE_CHECKPOINTS 1
#define STORE_BETA_ACTIVE 1
#define G_INPUT_BF16 1

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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
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

__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_flashkda_forward_checkpoint_c16(unsigned int* __restrict__ dynamic_counter, FlashKDATensorMap const* q_tma, FlashKDATensorMap const* k_tma, FlashKDATensorMap const* v_tma, FlashKDATensorMap const* g_tma, __nv_bfloat16* __restrict__ g, FlashKDATensorMap const* out_tma, FlashKDATensorMap const* checkpoint_tma, __nv_bfloat16* __restrict__ beta, __nv_bfloat16* __restrict__ beta_active_out, float* __restrict__ A_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, long long* __restrict__ checkpoint_cu_starts, int* __restrict__ work_items, float* __restrict__ initial_state, __nv_bfloat16* __restrict__ final_state, int total_work_items, int uniform_work_items, int num_heads, int checkpoint_every_n_tokens, float scale, float lower_bound)
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
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(out_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(checkpoint_tma)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int smem_k_addr = smem + 21504;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_v_addr = smem + 41984;
    float* smem_g = reinterpret_cast<float*>(smem_raw + 62464);
    const int smem_g_addr = smem + 62464;
    __nv_bfloat16* smem_beta = reinterpret_cast<__nv_bfloat16*>(smem_raw + 103424);
    const int smem_beta_addr = smem + 103424;
    __nv_bfloat16* smem_beta_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 103424);
    const int smem_beta_all_addr = smem + 103424;
    __nv_bfloat16* smem_k_inv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 104448);
    const int smem_k_inv_addr = smem + 104448;
    __nv_bfloat16* smem_k_decay = reinterpret_cast<__nv_bfloat16*>(smem_raw + 112640);
    const int smem_k_decay_addr = smem + 112640;
    __nv_bfloat16* smem_q_decay = reinterpret_cast<__nv_bfloat16*>(smem_raw + 120832);
    const int smem_q_decay_addr = smem + 120832;
    __nv_bfloat16* smem_k_restore = reinterpret_cast<__nv_bfloat16*>(smem_raw + 129024);
    const int smem_k_restore_addr = smem + 129024;
    __nv_bfloat16* smem_k_restore_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 129024);
    const int smem_k_restore_mn_addr = smem + 129024;
    __nv_bfloat16* smem_tinv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137216);
    const int smem_tinv_addr = smem + 137216;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137728);
    const int smem_a_addr = smem + 137728;
    __nv_bfloat16* smem_tinv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137216);
    const int smem_tinv_mn_addr = smem + 137216;
    __nv_bfloat16* smem_a_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137728);
    const int smem_a_mn_addr = smem + 137728;
    __nv_bfloat16* smem_state_diag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 139264);
    const int smem_state_diag_addr = smem + 139264;
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 155648);
    const int smem_o_addr = smem + 155648;
    __nv_bfloat16* smem_checkpoint = reinterpret_cast<__nv_bfloat16*>(smem_raw + 163840);
    const int smem_checkpoint_addr = smem + 163840;
    __nv_bfloat16* tinv_scratch = reinterpret_cast<__nv_bfloat16*>(smem_raw + 229376);
    const int tinv_scratch_addr = smem + 229376;
    unsigned int* sched_slot = reinterpret_cast<unsigned int*>(smem_raw + 229888);
    const int sched_slot_addr = smem + 229888;
    __nv_bfloat16* smem_q_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_all_addr = smem + 1024;
    __nv_bfloat16* smem_k_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int smem_k_all_addr = smem + 21504;
    float* smem_g_all = reinterpret_cast<float*>(smem_raw + 62464);
    const int smem_g_all_addr = smem + 62464;
    __nv_bfloat16* smem_v_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_v_all_addr = smem + 41984;
    __nv_bfloat16* smem_o_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 155648);
    const int smem_o_all_addr = smem + 155648;

    // Mbarrier init (37 groups, 116 barriers)
    // Mbarriers at smem_raw[0..928)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'sched_pipe' ---
            // sched_ready: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // sched_done: 8 barriers, init_count=15
            mbarrier_init(smem + 64, 15);
            mbarrier_init(smem + 72, 15);
            mbarrier_init(smem + 80, 15);
            mbarrier_init(smem + 88, 15);
            mbarrier_init(smem + 96, 15);
            mbarrier_init(smem + 104, 15);
            mbarrier_init(smem + 112, 15);
            mbarrier_init(smem + 120, 15);
            // --- pipeline 'raw_bar_pipe' ---
            // q_ready: 6 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // k_ready: 6 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // v_ready: 6 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            // g_ready: 6 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // --- pipeline 'raw_pipe' ---
            // q_done: 5 barriers, init_count=128
            mbarrier_init(smem + 320, 128);
            mbarrier_init(smem + 328, 128);
            mbarrier_init(smem + 336, 128);
            mbarrier_init(smem + 344, 128);
            mbarrier_init(smem + 352, 128);
            // k_done: 5 barriers, init_count=128
            mbarrier_init(smem + 360, 128);
            mbarrier_init(smem + 368, 128);
            mbarrier_init(smem + 376, 128);
            mbarrier_init(smem + 384, 128);
            mbarrier_init(smem + 392, 128);
            // g_done: 5 barriers, init_count=128
            mbarrier_init(smem + 400, 128);
            mbarrier_init(smem + 408, 128);
            mbarrier_init(smem + 416, 128);
            mbarrier_init(smem + 424, 128);
            mbarrier_init(smem + 432, 128);
            // v_done: 5 barriers, init_count=128
            mbarrier_init(smem + 440, 128);
            mbarrier_init(smem + 448, 128);
            mbarrier_init(smem + 456, 128);
            mbarrier_init(smem + 464, 128);
            mbarrier_init(smem + 472, 128);
            // --- pipeline 'raw_bar_pipe' ---
            // beta_ready: 6 barriers, init_count=32
            mbarrier_init(smem + 480, 32);
            mbarrier_init(smem + 488, 32);
            mbarrier_init(smem + 496, 32);
            mbarrier_init(smem + 504, 32);
            mbarrier_init(smem + 512, 32);
            mbarrier_init(smem + 520, 32);
            // beta_done: 6 barriers, init_count=160
            mbarrier_init(smem + 528, 160);
            mbarrier_init(smem + 536, 160);
            mbarrier_init(smem + 544, 160);
            mbarrier_init(smem + 552, 160);
            mbarrier_init(smem + 560, 160);
            mbarrier_init(smem + 568, 160);
            // --- pipeline 'decay_pipe' ---
            // k_decay_inv_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 576, 128);
            mbarrier_init(smem + 584, 128);
            // --- pipeline 'diag_pipe' ---
            // qk_scale_ready: 4 barriers, init_count=128
            mbarrier_init(smem + 592, 128);
            mbarrier_init(smem + 600, 128);
            mbarrier_init(smem + 608, 128);
            mbarrier_init(smem + 616, 128);
            // --- pipeline 'decay_pipe' ---
            // decay_tcgen_done: 2 barriers, init_count=1
            mbarrier_init(smem + 624, 1);
            mbarrier_init(smem + 632, 1);
            // decay_super_done: 2 barriers, init_count=64
            mbarrier_init(smem + 640, 64);
            mbarrier_init(smem + 648, 64);
            // k_restore_done: 2 barriers, init_count=1
            mbarrier_init(smem + 656, 1);
            mbarrier_init(smem + 664, 1);
            // --- pipeline 'diag_pipe' ---
            // state_diag_done: 4 barriers, init_count=1
            mbarrier_init(smem + 672, 1);
            mbarrier_init(smem + 680, 1);
            mbarrier_init(smem + 688, 1);
            mbarrier_init(smem + 696, 1);
            // --- pipeline 'intermediate_pipe' ---
            // tinv_ready: 2 barriers, init_count=32
            mbarrier_init(smem + 704, 32);
            mbarrier_init(smem + 712, 32);
            // tinv_done: 2 barriers, init_count=1
            mbarrier_init(smem + 720, 1);
            mbarrier_init(smem + 728, 1);
            // a_ready: 2 barriers, init_count=32
            mbarrier_init(smem + 736, 32);
            mbarrier_init(smem + 744, 32);
            // a_done: 2 barriers, init_count=1
            mbarrier_init(smem + 752, 1);
            mbarrier_init(smem + 760, 1);
            // --- pipeline 'state_pipe' ---
            // state_inp_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 768, 128);
            // state_read_done: 1 barriers, init_count=128
            mbarrier_init(smem + 776, 128);
            // state_acc_done: 1 barriers, init_count=1
            mbarrier_init(smem + 784, 1);
            // state_k_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 792, 1);
            // y_inp_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 800, 128);
            // u_acc_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 808, 1);
            // u_inp_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 816, 128);
            // o_acc_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 824, 1);
            // --- pipeline 'o_pipe' ---
            // o_acc_done: 2 barriers, init_count=128
            mbarrier_init(smem + 832, 128);
            mbarrier_init(smem + 840, 128);
            // o_tma_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 848, 128);
            mbarrier_init(smem + 856, 128);
            // o_tma_done: 2 barriers, init_count=32
            mbarrier_init(smem + 864, 32);
            mbarrier_init(smem + 872, 32);
            // --- pipeline 'checkpoint_pipe' ---
            // checkpoint_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 880, 128);
            mbarrier_init(smem + 888, 128);
            // checkpoint_done: 2 barriers, init_count=32
            mbarrier_init(smem + 896, 32);
            mbarrier_init(smem + 904, 32);
            // consumers_done: 1 barriers, init_count=15
            mbarrier_init(smem + 912, 15);
            // cleanup_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 920, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 272 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 928);
    if (warp == 13) {
        int _tmem_hold = smem + 928;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define sched_ready_addr (mbar_base + 0)
    #define sched_done_addr (mbar_base + 64)
    #define q_ready_addr (mbar_base + 128)
    #define k_ready_addr (mbar_base + 176)
    #define v_ready_addr (mbar_base + 224)
    #define g_ready_addr (mbar_base + 272)
    #define q_done_addr (mbar_base + 320)
    #define k_done_addr (mbar_base + 360)
    #define g_done_addr (mbar_base + 400)
    #define v_done_addr (mbar_base + 440)
    #define beta_ready_addr (mbar_base + 480)
    #define beta_done_addr (mbar_base + 528)
    #define k_decay_inv_ready_addr (mbar_base + 576)
    #define qk_scale_ready_addr (mbar_base + 592)
    #define decay_tcgen_done_addr (mbar_base + 624)
    #define decay_super_done_addr (mbar_base + 640)
    #define k_restore_done_addr (mbar_base + 656)
    #define state_diag_done_addr (mbar_base + 672)
    #define tinv_ready_addr (mbar_base + 704)
    #define tinv_done_addr (mbar_base + 720)
    #define a_ready_addr (mbar_base + 736)
    #define a_done_addr (mbar_base + 752)
    #define state_inp_ready_addr (mbar_base + 768)
    #define state_read_done_addr (mbar_base + 776)
    #define state_acc_done_addr (mbar_base + 784)
    #define state_k_ready_addr (mbar_base + 792)
    #define y_inp_ready_addr (mbar_base + 800)
    #define u_acc_ready_addr (mbar_base + 808)
    #define u_inp_ready_addr (mbar_base + 816)
    #define o_acc_ready_addr (mbar_base + 824)
    #define o_acc_done_addr (mbar_base + 832)
    #define o_tma_ready_addr (mbar_base + 848)
    #define o_tma_done_addr (mbar_base + 864)
    #define checkpoint_ready_addr (mbar_base + 880)
    #define checkpoint_done_addr (mbar_base + 896)
    #define consumers_done_addr (mbar_base + 912)
    #define cleanup_ready_addr (mbar_base + 920)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state = taddr;
    const int tmem_tmem_state_inp = taddr + 128;
    const int tmem_tmem_q_state = taddr + 192;
    const int tmem_tmem_state_k = taddr + 224;
    const int tmem_tmem_u_acc = taddr + 240;
    const int tmem_tmem_y_inp = taddr + 256;
    const int tmem_tmem_u_inp = taddr + 264;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: cg0 ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // cg0_main
            unsigned int sched_stage_cg0 = 0;
            int cumulative_chunk_cg0 = 0;
            int instance_id = (warp - 0) / 4;
            int cg0_instance = instance_id;
            int warp_id_in_role = (warp - 0);
            int cg0_local_warp = warp_id_in_role - cg0_instance * 4;
            int cg0_tid = cg0_local_warp * 32 + lane;
            unsigned int _phase_sched_ready = 0;
            #pragma unroll 1
            for (int _ = 0; _ < total_work_items + 1; _++) {
                mbarrier_wait(sched_ready_addr + (sched_stage_cg0) * 8, _phase_sched_ready);
                unsigned int slot[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&slot[0])) : "r"(sched_slot_addr + sched_stage_cg0 * 4));
                unsigned int tile_cg0 = slot[0];
                if (elect_sync()) {
                    mbarrier_arrive(sched_done_addr + (sched_stage_cg0) * 8);
                }
                sched_stage_cg0 += 1;
                if (sched_stage_cg0 == 8) { sched_stage_cg0 = 0; _phase_sched_ready ^= 1; }
                if (tile_cg0 >= (unsigned int)total_work_items) {
                    break;
                }
                int item_base_cg0 = (int)tile_cg0 * 8;
                int _vec_load_2[4];
                {
                    int4 _iv4 = *reinterpret_cast<const int4*>(work_items + item_base_cg0 + 4);
                    _vec_load_2[0 + 0] = _iv4.x;
                    _vec_load_2[0 + 1] = _iv4.y;
                    _vec_load_2[0 + 2] = _iv4.z;
                    _vec_load_2[0 + 3] = _iv4.w;
                }
                int head_cg0 = work_items[item_base_cg0 + 1];
                int wend_cg0 = work_items[item_base_cg0 + 3];
                int cstart_cg0 = _vec_load_2[0];
                long long bos_cg0 = (long long)_vec_load_2[2];
                long long eos_cg0 = (long long)_vec_load_2[3];
                int chunks_cg0 = wend_cg0 - cstart_cg0;
                float _expf_0 = __expf(A_log[head_cg0]);
                float gate_rate_cg0 = _expf_0;
                float gate_bias_cg0 = dt_bias[head_cg0 * 128 + cg0_tid];
                asm volatile("barrier.sync 10, 256;" ::: "memory");
                int first_cumulative_cg0 = cumulative_chunk_cg0 + cg0_instance;
                int raw_stage_cg0 = first_cumulative_cg0 % 5;
                int raw_bar_stage_cg0 = first_cumulative_cg0 % 6;
                int decay_stage_cg0 = first_cumulative_cg0 % 2;
                int diag_stage_cg0 = first_cumulative_cg0 % 4;
                int raw_bar_phase_cg0 = first_cumulative_cg0 / 6 & 1;
                int decay_free_phase_cg0 = first_cumulative_cg0 / 2 + 1 & 1;
                int diag_free_phase_cg0 = first_cumulative_cg0 / 4 + 1 & 1;
                #pragma unroll 1
                for (int chunk_cg0 = cg0_instance; chunk_cg0 < chunks_cg0; chunk_cg0 += 2) {
                    int cumulative_cg0 = cumulative_chunk_cg0 + chunk_cg0;
                    int logical_chunk_cg0 = cstart_cg0 + chunk_cg0;
                    if (cg0_local_warp == 0) {
                        mbarrier_wait(beta_done_addr + (raw_bar_stage_cg0) * 8, (unsigned int)(cumulative_cg0 / 6 + 1 & 1));
                        if (lane < 16) {
                            long long beta_token_cg0 = bos_cg0 + (long long)logical_chunk_cg0 * 16 + (long long)lane;
                            float beta_value_cg0 = 0.0f;
                            if (beta_token_cg0 < eos_cg0) {
                                float beta_logit_cg0 = (float)beta[beta_token_cg0 * (long long)num_heads + (long long)head_cg0];
                                float _tanh_approx_0;
                                asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(beta_logit_cg0 * 0.5f));
                                beta_value_cg0 = _tanh_approx_0 * 0.5f + 0.5f;
                            }
                            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(beta_value_cg0);
                            __nv_bfloat16 beta_active_cg0 = _cvt_bf16_0;
                            smem_beta_all[(int)raw_bar_stage_cg0 * 16 + lane] = beta_active_cg0;
                            if (STORE_BETA_ACTIVE != 0 && beta_token_cg0 < eos_cg0) {
                                beta_active_out[beta_token_cg0 * (long long)num_heads + (long long)head_cg0] = beta_active_cg0;
                            }
                        }
                        mbarrier_arrive(beta_ready_addr + (raw_bar_stage_cg0) * 8);
                    }
                    mbarrier_wait(g_ready_addr + (raw_bar_stage_cg0) * 8, raw_bar_phase_cg0);
                    float gate_raw_cg0[16];
                    float gate_log_cg0[16];
                    float gate_prefix_regs_cg0[16];
                    #pragma unroll
                    for (int token_gate_cg0 = 0; token_gate_cg0 < 16; token_gate_cg0++) {
                        {
                            long long gate_token_cg0 = bos_cg0 + (long long)logical_chunk_cg0 * 16 + (long long)token_gate_cg0;
                            gate_raw_cg0[token_gate_cg0] = (float)g[(gate_token_cg0 * (long long)num_heads + (long long)head_cg0) * 128 + (long long)cg0_tid];
                        }
                    }
                    #pragma unroll
                    for (int gate_row_cg0 = 0; gate_row_cg0 < 16; gate_row_cg0++) {
                        float gate_arg_cg0 = gate_rate_cg0 * (gate_raw_cg0[gate_row_cg0] + gate_bias_cg0);
                        float _tanh_approx_1;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(gate_arg_cg0 * 0.5f));
                        float gate_value_cg0 = _tanh_approx_1 * 0.5f + 0.5f;
                        long long valid_gate_cg0 = bos_cg0 + (long long)logical_chunk_cg0 * 16 + (long long)gate_row_cg0;
                        gate_log_cg0[gate_row_cg0] = 0.0f;
                        if (valid_gate_cg0 < eos_cg0) {
                            gate_log_cg0[gate_row_cg0] = lower_bound * 1.4426950408889634f * gate_value_cg0;
                        }
                    }
                    float gate_prefix_cg0 = 0.0f;
                    #pragma unroll
                    for (int gate_pair_idx_cg0 = 0; gate_pair_idx_cg0 < 8; gate_pair_idx_cg0++) {
                        int gate_row0_cg0 = gate_pair_idx_cg0 * 2;
                        int gate_row1_cg0 = gate_row0_cg0 + 1;
                        float2 _f2_0 = make_float2(gate_prefix_cg0, gate_log_cg0[gate_row0_cg0]);
                        float2 _f2_1 = make_float2(gate_log_cg0[gate_row0_cg0], gate_log_cg0[gate_row1_cg0]);
                        float2 gate_pair_sum_cg0 = add_f32x2(_f2_0, _f2_1);
                        gate_prefix_regs_cg0[gate_row0_cg0] = gate_pair_sum_cg0.x;
                        gate_prefix_cg0 += gate_pair_sum_cg0.y;
                        gate_prefix_regs_cg0[gate_row1_cg0] = gate_prefix_cg0;
                    }
                    float gate_last_cg0 = 1.0f;
                    #pragma unroll
                    for (int token_gate_cg0_1 = 0; token_gate_cg0_1 < 16; token_gate_cg0_1++) {
                        float _exp2_0 = approx_exp2(gate_prefix_regs_cg0[token_gate_cg0_1]);
                        gate_last_cg0 = _exp2_0;
                        int segment = cg0_tid / 32;
                        int segment_col = cg0_tid - segment * 32;
                        int swizzled_col = segment_col ^ (token_gate_cg0_1 & 7) * 4;
                        smem_g_all[raw_stage_cg0 * 16 * 128 + segment * 16 * 32 + token_gate_cg0_1 * 32 + swizzled_col] = gate_last_cg0;
                    }
                    mbarrier_wait(state_diag_done_addr + (diag_stage_cg0) * 8, diag_free_phase_cg0);
                    int diag_block_cg0 = cg0_tid / 16;
                    int diag_coord_cg0 = cg0_tid - diag_block_cg0 * 16;
                    int diag_storage_stage_cg0 = (int)diag_stage_cg0 * 8 + diag_block_cg0;
                    if (cumulative_cg0 < 4) {
                        #pragma unroll
                        for (int diag_half_cg0 = 0; diag_half_cg0 < 2; diag_half_cg0++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_state_diag_addr + (unsigned int)(diag_storage_stage_cg0 * 512) + (unsigned int)(diag_half_cg0 * 8 / 16 * 512 + diag_coord_cg0 * 32 + diag_half_cg0 * 8 % 16 * 2 ^ (diag_half_cg0 * 8 / 16 * 512 + diag_coord_cg0 * 32 + diag_half_cg0 * 8 % 16 * 2 >> 7 & 1) << 4))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        }
                    }
                    {
                        __nv_bfloat16 _bval_0 = __float2bfloat16_rn(gate_last_cg0);
                        uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                        uint32_t _addr_0 = static_cast<uint32_t>((smem_state_diag_addr + (unsigned int)(diag_storage_stage_cg0 * 512) + (unsigned int)(diag_coord_cg0 / 16 * 512 + diag_coord_cg0 * 32 + diag_coord_cg0 % 16 * 2 ^ (diag_coord_cg0 / 16 * 512 + diag_coord_cg0 * 32 + diag_coord_cg0 % 16 * 2 >> 7 & 1) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                    }
                    if (cg0_instance == 0) {
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                    }
                    mbarrier_wait(q_ready_addr + (raw_bar_stage_cg0) * 8, raw_bar_phase_cg0);
                    mbarrier_wait(k_ready_addr + (raw_bar_stage_cg0) * 8, raw_bar_phase_cg0);
                    int decay_row_cg0 = cg0_local_warp * 4 + lane / 8;
                    int decay_lane_cg0 = lane & 7;
                    float q_values_cg0[16];
                    float k_values_cg0[16];
                    float2 _f2_2 = make_float2(0.0f, 0.0f);
                    float2 qk_sq_even_cg0 = _f2_2;
                    float2 _f2_3 = make_float2(0.0f, 0.0f);
                    float2 qk_sq_odd_cg0 = _f2_3;
                    #pragma unroll
                    for (int dim_half_cg0 = 0; dim_half_cg0 < 2; dim_half_cg0++) {
                        int dim_base_cg0 = dim_half_cg0 * 64 + decay_lane_cg0 * 8;
                        unsigned int q_words_cg0[4];
                        unsigned int k_words_cg0[4];
                        int segment_1 = dim_base_cg0 / 64;
                        int segment_col_1 = dim_base_cg0 - segment_1 * 64;
                        int swizzled_col_1 = segment_col_1 ^ (decay_row_cg0 & 7) * 8;
                        int raw_index_cg0 = raw_stage_cg0 * 16 * 128 + segment_1 * 16 * 64 + decay_row_cg0 * 64 + swizzled_col_1;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_words_cg0[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_cg0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_cg0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_cg0[(0) + 3]))
                            : "r"(smem_q_all_addr + (unsigned int)(raw_index_cg0 * 2)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_cg0[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_cg0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_cg0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_cg0[(0) + 3]))
                            : "r"(smem_k_all_addr + (unsigned int)(raw_index_cg0 * 2)));
                        float q_words_cg0_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&q_words_cg0_f32[_pair * 2])[0]), "=f"((&q_words_cg0_f32[_pair * 2])[1])
                                : "r"(q_words_cg0[_pair]));
                        }
                        float k_words_cg0_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&k_words_cg0_f32[_pair * 2])[0]), "=f"((&k_words_cg0_f32[_pair * 2])[1])
                                : "r"(k_words_cg0[_pair]));
                        }
                        #pragma unroll
                        for (int dim_local_cg0 = 0; dim_local_cg0 < 8; dim_local_cg0++) {
                            int reg_cg0 = dim_half_cg0 * 8 + dim_local_cg0;
                            q_values_cg0[reg_cg0] = q_words_cg0_f32[dim_local_cg0];
                            k_values_cg0[reg_cg0] = k_words_cg0_f32[dim_local_cg0];
                        }
                        #pragma unroll
                        for (int dim_pair_sq_cg0 = 0; dim_pair_sq_cg0 < 4; dim_pair_sq_cg0++) {
                            int even_reg_cg0 = dim_half_cg0 * 8 + dim_pair_sq_cg0 * 2;
                            int odd_reg_cg0 = even_reg_cg0 + 1;
                            float2 _f2_4 = make_float2(q_values_cg0[even_reg_cg0], k_values_cg0[even_reg_cg0]);
                            float2 qk_even_cg0 = _f2_4;
                            float2 _f2_5 = make_float2(q_values_cg0[odd_reg_cg0], k_values_cg0[odd_reg_cg0]);
                            float2 qk_odd_cg0 = _f2_5;
                            qk_sq_even_cg0 = fma_f32x2(qk_even_cg0, qk_even_cg0, qk_sq_even_cg0);
                            qk_sq_odd_cg0 = fma_f32x2(qk_odd_cg0, qk_odd_cg0, qk_sq_odd_cg0);
                        }
                    }
                    float q_sq_cg0 = qk_sq_even_cg0.x + qk_sq_odd_cg0.x;
                    float k_sq_cg0 = qk_sq_even_cg0.y + qk_sq_odd_cg0.y;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sq_cg0, 4);
                    q_sq_cg0 += _shfl_xor_0;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_sq_cg0, 4);
                    k_sq_cg0 += _shfl_xor_1;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sq_cg0, 2);
                    q_sq_cg0 += _shfl_xor_2;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sq_cg0, 2);
                    k_sq_cg0 += _shfl_xor_3;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, q_sq_cg0, 1);
                    q_sq_cg0 += _shfl_xor_4;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sq_cg0, 1);
                    k_sq_cg0 += _shfl_xor_5;
                    float _max_0 = max_noftz(q_sq_cg0, 1e-12f);
                    float _rsqrt_0 = rsqrtf(_max_0);
                    float q_inv_norm_cg0 = _rsqrt_0;
                    float _max_1 = max_noftz(k_sq_cg0, 1e-12f);
                    float _rsqrt_1 = rsqrtf(_max_1);
                    float k_inv_norm_cg0 = _rsqrt_1;
                    float exp_g_regs_cg0[16];
                    float exp_g_last_regs_cg0[16];
                    #pragma unroll
                    for (int dim_half_prefix_cg0 = 0; dim_half_prefix_cg0 < 2; dim_half_prefix_cg0++) {
                        int dim_base_prefix_cg0 = dim_half_prefix_cg0 * 64 + decay_lane_cg0 * 8;
                        #pragma unroll
                        for (int f32_group_cg0 = 0; f32_group_cg0 < 2; f32_group_cg0++) {
                            int f32_dim_base_cg0 = dim_base_prefix_cg0 + f32_group_cg0 * 4;
                            unsigned int exp_g_words_cg0[4];
                            unsigned int exp_g_last_words_cg0[4];
                            int segment_2 = f32_dim_base_cg0 / 32;
                            int segment_col_2 = f32_dim_base_cg0 - segment_2 * 32;
                            int swizzled_col_2 = segment_col_2 ^ (decay_row_cg0 & 7) * 4;
                            int exp_g_index_cg0 = raw_stage_cg0 * 16 * 128 + segment_2 * 16 * 32 + decay_row_cg0 * 32 + swizzled_col_2;
                            int segment_0 = f32_dim_base_cg0 / 32;
                            int segment_col_1_1 = f32_dim_base_cg0 - segment_0 * 32;
                            int swizzled_col_2_1 = segment_col_1_1 ^ 28;
                            int exp_g_last_index_cg0 = raw_stage_cg0 * 16 * 128 + segment_0 * 16 * 32 + 480 + swizzled_col_2_1;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&exp_g_words_cg0[0])), "=r"(*reinterpret_cast<uint32_t*>(&exp_g_words_cg0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&exp_g_words_cg0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&exp_g_words_cg0[(0) + 3]))
                                : "r"(smem_g_all_addr + (unsigned int)(exp_g_index_cg0 * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&exp_g_last_words_cg0[0])), "=r"(*reinterpret_cast<uint32_t*>(&exp_g_last_words_cg0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&exp_g_last_words_cg0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&exp_g_last_words_cg0[(0) + 3]))
                                : "r"(smem_g_all_addr + (unsigned int)(exp_g_last_index_cg0 * 4)));
                            #pragma unroll
                            for (int prefix_word_cg0 = 0; prefix_word_cg0 < 4; prefix_word_cg0++) {
                                int prefix_reg_cg0 = dim_half_prefix_cg0 * 8 + f32_group_cg0 * 4 + prefix_word_cg0;
                                exp_g_regs_cg0[prefix_reg_cg0] = __uint_as_float(exp_g_words_cg0[prefix_word_cg0]);
                                exp_g_last_regs_cg0[prefix_reg_cg0] = __uint_as_float(exp_g_last_words_cg0[prefix_word_cg0]);
                            }
                        }
                    }
                    __nv_bfloat162 k_inv_pairs_all_cg0[8];
                    float2 _f2_6 = make_float2(k_inv_norm_cg0, k_inv_norm_cg0);
                    float2 k_inv_norm_pair_cg0 = _f2_6;
                    #pragma unroll
                    for (int dim_half_k_cg0 = 0; dim_half_k_cg0 < 2; dim_half_k_cg0++) {
                        int dim_base_k_cg0 = dim_half_k_cg0 * 64 + decay_lane_cg0 * 8;
                        unsigned int k_decay_words_cg0[4];
                        #pragma unroll
                        for (int dim_pair_k_cg0 = 0; dim_pair_k_cg0 < 4; dim_pair_k_cg0++) {
                            int dim_local0_k_cg0 = dim_pair_k_cg0 * 2;
                            int dim_local1_k_cg0 = dim_local0_k_cg0 + 1;
                            int reg_k0_cg0 = dim_half_k_cg0 * 8 + dim_local0_k_cg0;
                            int reg_k1_cg0 = reg_k0_cg0 + 1;
                            float prefix_k0_cg0 = exp_g_regs_cg0[reg_k0_cg0];
                            float prefix_k1_cg0 = exp_g_regs_cg0[reg_k1_cg0];
                            float2 _f2_7 = make_float2(k_values_cg0[reg_k0_cg0], k_values_cg0[reg_k1_cg0]);
                            float2 k_norm_pair_cg0 = mul_f32x2(_f2_7, k_inv_norm_pair_cg0);
                            __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(k_norm_pair_cg0.x, k_norm_pair_cg0.y));
                            __nv_bfloat162 k_norm_bf16x2_cg0 = _bf16x2_0;
                            __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(prefix_k0_cg0, prefix_k1_cg0));
                            __nv_bfloat162 prefix_bf16x2_cg0 = _bf16x2_1;
                            float _rcp_0 = approx_rcp(prefix_k0_cg0);
                            float _rcp_1 = approx_rcp(prefix_k1_cg0);
                            __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_rcp_0, _rcp_1));
                            __nv_bfloat162 reciprocal_bf16x2_cg0 = _bf16x2_2;
                            __nv_bfloat162 k_inv_bf16x2_cg0 = k_norm_bf16x2_cg0 * reciprocal_bf16x2_cg0;
                            __nv_bfloat162 k_decay_bf16x2_cg0 = k_norm_bf16x2_cg0 * prefix_bf16x2_cg0;
                            int k_word_cg0 = dim_half_k_cg0 * 4 + dim_pair_k_cg0;
                            k_inv_pairs_all_cg0[k_word_cg0] = k_inv_bf16x2_cg0;
                            k_decay_words_cg0[dim_pair_k_cg0] = __as_u32(k_decay_bf16x2_cg0);
                        }
                        if (dim_half_k_cg0 == 0) {
                            mbarrier_wait(decay_super_done_addr + (decay_stage_cg0) * 8, decay_free_phase_cg0);
                            mbarrier_wait(decay_tcgen_done_addr + (decay_stage_cg0) * 8, decay_free_phase_cg0);
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_k_inv_addr + (unsigned int)(decay_stage_cg0 * 4096) + (unsigned int)(dim_base_k_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_k_cg0 % 64 * 2 ^ (dim_base_k_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_k_cg0 % 64 * 2 >> 7 & 7) << 4))), "r"(__as_u32(k_inv_pairs_all_cg0[dim_half_k_cg0 * 4])), "r"(__as_u32(k_inv_pairs_all_cg0[dim_half_k_cg0 * 4 + 1])), "r"(__as_u32(k_inv_pairs_all_cg0[dim_half_k_cg0 * 4 + 2])), "r"(__as_u32(k_inv_pairs_all_cg0[dim_half_k_cg0 * 4 + 3])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_k_decay_addr + (unsigned int)(decay_stage_cg0 * 4096) + (unsigned int)(dim_base_k_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_k_cg0 % 64 * 2 ^ (dim_base_k_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_k_cg0 % 64 * 2 >> 7 & 7) << 4))), "r"(k_decay_words_cg0[0]), "r"(k_decay_words_cg0[1]), "r"(k_decay_words_cg0[2]), "r"(k_decay_words_cg0[3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(k_decay_inv_ready_addr + (decay_stage_cg0) * 8);
                    mbarrier_arrive(q_done_addr + (raw_stage_cg0) * 8);
                    mbarrier_arrive(k_done_addr + (raw_stage_cg0) * 8);
                    mbarrier_arrive(g_done_addr + (raw_stage_cg0) * 8);
                    float2 _f2_8 = make_float2(q_inv_norm_cg0, q_inv_norm_cg0);
                    float2 q_inv_pair_cg0 = _f2_8;
                    #pragma unroll
                    for (int dim_half_q_cg0 = 0; dim_half_q_cg0 < 2; dim_half_q_cg0++) {
                        int dim_base_q_cg0 = dim_half_q_cg0 * 64 + decay_lane_cg0 * 8;
                        unsigned int q_decay_words_cg0[4];
                        #pragma unroll
                        for (int dim_pair_q_cg0 = 0; dim_pair_q_cg0 < 4; dim_pair_q_cg0++) {
                            int dim_local0_q_cg0 = dim_pair_q_cg0 * 2;
                            int dim_local1_q_cg0 = dim_local0_q_cg0 + 1;
                            int reg_q0_cg0 = dim_half_q_cg0 * 8 + dim_local0_q_cg0;
                            int reg_q1_cg0 = reg_q0_cg0 + 1;
                            float2 _f2_9 = make_float2(q_values_cg0[reg_q0_cg0], q_values_cg0[reg_q1_cg0]);
                            float2 q_norm_pair_cg0 = mul_f32x2(_f2_9, q_inv_pair_cg0);
                            __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(q_norm_pair_cg0.x, q_norm_pair_cg0.y));
                            __nv_bfloat162 q_norm_bf16x2_cg0 = _bf16x2_3;
                            __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(exp_g_regs_cg0[reg_q0_cg0], exp_g_regs_cg0[reg_q1_cg0]));
                            __nv_bfloat162 q_prefix_bf16x2_cg0 = _bf16x2_4;
                            __nv_bfloat162 q_decay_bf16x2_cg0 = q_norm_bf16x2_cg0 * q_prefix_bf16x2_cg0;
                            q_decay_words_cg0[dim_pair_q_cg0] = __as_u32(q_decay_bf16x2_cg0);
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_decay_addr + (unsigned int)(decay_stage_cg0 * 4096) + (unsigned int)(dim_base_q_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_q_cg0 % 64 * 2 ^ (dim_base_q_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_q_cg0 % 64 * 2 >> 7 & 7) << 4))), "r"(q_decay_words_cg0[0]), "r"(q_decay_words_cg0[1]), "r"(q_decay_words_cg0[2]), "r"(q_decay_words_cg0[3]) : "memory");
                    }
                    mbarrier_wait(k_restore_done_addr + (decay_stage_cg0) * 8, decay_free_phase_cg0);
                    #pragma unroll
                    for (int dim_half_restore_cg0 = 0; dim_half_restore_cg0 < 2; dim_half_restore_cg0++) {
                        int dim_base_restore_cg0 = dim_half_restore_cg0 * 64 + decay_lane_cg0 * 8;
                        unsigned int k_restore_words_cg0[4];
                        #pragma unroll
                        for (int dim_pair_restore_cg0 = 0; dim_pair_restore_cg0 < 4; dim_pair_restore_cg0++) {
                            int dim_local0_restore_cg0 = dim_pair_restore_cg0 * 2;
                            int dim_local1_restore_cg0 = dim_local0_restore_cg0 + 1;
                            int reg_restore0_cg0 = dim_half_restore_cg0 * 8 + dim_local0_restore_cg0;
                            int reg_restore1_cg0 = reg_restore0_cg0 + 1;
                            int restore_word_cg0 = dim_half_restore_cg0 * 4 + dim_pair_restore_cg0;
                            __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(exp_g_last_regs_cg0[reg_restore0_cg0], exp_g_last_regs_cg0[reg_restore1_cg0]));
                            __nv_bfloat162 k_restore_bf16x2_cg0 = k_inv_pairs_all_cg0[restore_word_cg0] * _bf16x2_5;
                            k_restore_words_cg0[dim_pair_restore_cg0] = __as_u32(k_restore_bf16x2_cg0);
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_k_restore_addr + (unsigned int)(decay_stage_cg0 * 4096) + (unsigned int)(dim_base_restore_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_restore_cg0 % 64 * 2 ^ (dim_base_restore_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_restore_cg0 % 64 * 2 >> 7 & 7) << 4))), "r"(k_restore_words_cg0[0]), "r"(k_restore_words_cg0[1]), "r"(k_restore_words_cg0[2]), "r"(k_restore_words_cg0[3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(qk_scale_ready_addr + (diag_stage_cg0) * 8);
                    raw_stage_cg0 += 2;
                    if (raw_stage_cg0 >= 5) {
                        raw_stage_cg0 -= 5;
                    }
                    raw_bar_stage_cg0 += 2;
                    if (raw_bar_stage_cg0 >= 6) {
                        raw_bar_stage_cg0 -= 6;
                        raw_bar_phase_cg0 ^= 1;
                    }
                    decay_free_phase_cg0 ^= 1;
                    diag_stage_cg0 += 2;
                    if (diag_stage_cg0 >= 4) {
                        diag_stage_cg0 -= 4;
                        diag_free_phase_cg0 ^= 1;
                    }
                }
                cumulative_chunk_cg0 += chunks_cg0;
            }
            if (elect_sync()) {
                mbarrier_arrive(consumers_done_addr);
            }
        }
    }
    // ---- Role: cg1 ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
        { // cg1_main
            unsigned int sched_stage_cg1 = 0;
            int cumulative_chunk_cg1 = 0;
            int warp_in_wg = warp % 4;
            int value_row_cg1 = warp_in_wg * 32 + lane;
            int value_dim_base_cg1 = warp_in_wg * 32;
            const int tmem_row_base_cg1 = warp_in_wg * 32 << 16;
            int ov_token_cg1 = lane / 16 * 8 + (lane & 7);
            int ov_col_cg1 = (lane / 8 & 1) * 8;
            unsigned int _phase_sched_ready_1 = 0;
            #pragma unroll 1
            for (int __1 = 0; __1 < total_work_items + 1; __1++) {
                mbarrier_wait(sched_ready_addr + (sched_stage_cg1) * 8, _phase_sched_ready_1);
                unsigned int slot_1[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&slot_1[0])) : "r"(sched_slot_addr + sched_stage_cg1 * 4));
                unsigned int tile_cg1 = slot_1[0];
                if (elect_sync()) {
                    mbarrier_arrive(sched_done_addr + (sched_stage_cg1) * 8);
                }
                sched_stage_cg1 += 1;
                if (sched_stage_cg1 == 8) { sched_stage_cg1 = 0; _phase_sched_ready_1 ^= 1; }
                if (tile_cg1 >= (unsigned int)total_work_items) {
                    break;
                }
                int item_base_cg1 = (int)tile_cg1 * 8;
                int _vec_load_4[4];
                {
                    int4 _iv4 = *reinterpret_cast<const int4*>(work_items + item_base_cg1);
                    _vec_load_4[0 + 0] = _iv4.x;
                    _vec_load_4[0 + 1] = _iv4.y;
                    _vec_load_4[0 + 2] = _iv4.z;
                    _vec_load_4[0 + 3] = _iv4.w;
                }
                int _vec_load_5[4];
                {
                    int4 _iv4 = *reinterpret_cast<const int4*>(work_items + item_base_cg1 + 4);
                    _vec_load_5[0 + 0] = _iv4.x;
                    _vec_load_5[0 + 1] = _iv4.y;
                    _vec_load_5[0 + 2] = _iv4.z;
                    _vec_load_5[0 + 3] = _iv4.w;
                }
                int seq_cg1 = _vec_load_4[0];
                int head_cg1 = _vec_load_4[1];
                int wstart_cg1 = _vec_load_4[2];
                int wend_cg1 = _vec_load_4[3];
                int cstart_cg1 = _vec_load_5[0];
                int chunks_cg1 = wend_cg1 - cstart_cg1;
                long long state_base_cg1 = (((long long)seq_cg1 * (long long)num_heads + (long long)head_cg1) * 128 + (long long)value_row_cg1) * 128;
                #pragma unroll
                for (int state_block_cg1 = 0; state_block_cg1 < 4; state_block_cg1++) {
                    float state_init_cg1[32];
                    state_init_cg1[0] = 0.0f;
                    state_init_cg1[1] = 0.0f;
                    state_init_cg1[2] = 0.0f;
                    state_init_cg1[3] = 0.0f;
                    state_init_cg1[4] = 0.0f;
                    state_init_cg1[5] = 0.0f;
                    state_init_cg1[6] = 0.0f;
                    state_init_cg1[7] = 0.0f;
                    state_init_cg1[8] = 0.0f;
                    state_init_cg1[9] = 0.0f;
                    state_init_cg1[10] = 0.0f;
                    state_init_cg1[11] = 0.0f;
                    state_init_cg1[12] = 0.0f;
                    state_init_cg1[13] = 0.0f;
                    state_init_cg1[14] = 0.0f;
                    state_init_cg1[15] = 0.0f;
                    state_init_cg1[16] = 0.0f;
                    state_init_cg1[17] = 0.0f;
                    state_init_cg1[18] = 0.0f;
                    state_init_cg1[19] = 0.0f;
                    state_init_cg1[20] = 0.0f;
                    state_init_cg1[21] = 0.0f;
                    state_init_cg1[22] = 0.0f;
                    state_init_cg1[23] = 0.0f;
                    state_init_cg1[24] = 0.0f;
                    state_init_cg1[25] = 0.0f;
                    state_init_cg1[26] = 0.0f;
                    state_init_cg1[27] = 0.0f;
                    state_init_cg1[28] = 0.0f;
                    state_init_cg1[29] = 0.0f;
                    state_init_cg1[30] = 0.0f;
                    state_init_cg1[31] = 0.0f;
                    if (USE_INITIAL_STATE != 0 && cstart_cg1 == 0) {
                        #pragma unroll
                        for (int state_vec_cg1 = 0; state_vec_cg1 < 4; state_vec_cg1++) {
                            {
                                unsigned _ldv8_0_0;
                                unsigned _ldv8_0_1;
                                unsigned _ldv8_0_2;
                                unsigned _ldv8_0_3;
                                unsigned _ldv8_0_4;
                                unsigned _ldv8_0_5;
                                unsigned _ldv8_0_6;
                                unsigned _ldv8_0_7;
                                asm volatile(
                                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(initial_state + (state_base_cg1 + (long long)(state_block_cg1 * 32) + (long long)(state_vec_cg1 * 8)))) : "memory");
                                state_init_cg1[state_vec_cg1 * 8 + 0] = __uint_as_float(_ldv8_0_0);
                                state_init_cg1[state_vec_cg1 * 8 + 1] = __uint_as_float(_ldv8_0_1);
                                state_init_cg1[state_vec_cg1 * 8 + 2] = __uint_as_float(_ldv8_0_2);
                                state_init_cg1[state_vec_cg1 * 8 + 3] = __uint_as_float(_ldv8_0_3);
                                state_init_cg1[state_vec_cg1 * 8 + 4] = __uint_as_float(_ldv8_0_4);
                                state_init_cg1[state_vec_cg1 * 8 + 5] = __uint_as_float(_ldv8_0_5);
                                state_init_cg1[state_vec_cg1 * 8 + 6] = __uint_as_float(_ldv8_0_6);
                                state_init_cg1[state_vec_cg1 * 8 + 7] = __uint_as_float(_ldv8_0_7);
                            }
                        }
                        #pragma unroll
                        for (int state_elem_cg1 = 0; state_elem_cg1 < 32; state_elem_cg1++) {
                            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(state_init_cg1[state_elem_cg1]);
                            float _cvt_f32_2 = __bfloat162float(_cvt_bf16_1);
                            state_init_cg1[state_elem_cg1] = _cvt_f32_2;
                        }
                    }
                    tmem_st_x32_f32(taddr + (unsigned int)tmem_row_base_cg1 + (unsigned int)(state_block_cg1 * 32), state_init_cg1);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (chunks_cg1 > 0) {
                    int cumulative_cg1 = cumulative_chunk_cg1;
                    unsigned int raw_stage_cg1 = (unsigned int)(cumulative_cg1 % 5);
                    unsigned int raw_bar_stage_cg1 = (unsigned int)(cumulative_cg1 % 6);
                    unsigned int o_stage_cg1 = (unsigned int)(cumulative_cg1 % 2);
                    unsigned int checkpoint_stage_cg1 = (unsigned int)(cumulative_cg1 % 2);
                    unsigned int state_phase_cg1 = (unsigned int)(cumulative_cg1 & 1);
                    float _tmem_load_0[128];
                    tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)tmem_row_base_cg1);
                    tmem_ld_x16(&_tmem_load_0[16], taddr + (unsigned int)tmem_row_base_cg1 + 16);
                    tmem_ld_x16(&_tmem_load_0[32], taddr + (unsigned int)tmem_row_base_cg1 + 32);
                    tmem_ld_x16(&_tmem_load_0[48], taddr + (unsigned int)tmem_row_base_cg1 + 48);
                    tmem_ld_x16(&_tmem_load_0[64], taddr + (unsigned int)tmem_row_base_cg1 + 64);
                    tmem_ld_x16(&_tmem_load_0[80], taddr + (unsigned int)tmem_row_base_cg1 + 80);
                    tmem_ld_x16(&_tmem_load_0[96], taddr + (unsigned int)tmem_row_base_cg1 + 96);
                    tmem_ld_x16(&_tmem_load_0[112], taddr + (unsigned int)tmem_row_base_cg1 + 112);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    {
                        mbarrier_wait(checkpoint_done_addr + (checkpoint_stage_cg1) * 8, (unsigned int)(cumulative_cg1 / 2 + 1 & 1));
                    }
                    unsigned int state_words_cg1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 ^ (value_row_cg1 * 128 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 16 ^ (value_row_cg1 * 128 + 16 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 16], _tmem_load_0[_lp*2+1 + 16]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 8, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 32 ^ (value_row_cg1 * 128 + 32 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 48 ^ (value_row_cg1 * 128 + 48 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 32], _tmem_load_0[_lp*2+1 + 32]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 16, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 64 ^ (value_row_cg1 * 128 + 64 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 80 ^ (value_row_cg1 * 128 + 80 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 48], _tmem_load_0[_lp*2+1 + 48]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 24, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 96 ^ (value_row_cg1 * 128 + 96 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 112 ^ (value_row_cg1 * 128 + 112 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 64], _tmem_load_0[_lp*2+1 + 64]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 32, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 ^ (16384 + value_row_cg1 * 128 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 16 ^ (16384 + value_row_cg1 * 128 + 16 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 80], _tmem_load_0[_lp*2+1 + 80]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 40, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 32 ^ (16384 + value_row_cg1 * 128 + 32 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 48 ^ (16384 + value_row_cg1 * 128 + 48 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 96], _tmem_load_0[_lp*2+1 + 96]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 48, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 64 ^ (16384 + value_row_cg1 * 128 + 64 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 80 ^ (16384 + value_row_cg1 * 128 + 80 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 112], _tmem_load_0[_lp*2+1 + 112]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 56, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 96 ^ (16384 + value_row_cg1 * 128 + 96 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 112 ^ (16384 + value_row_cg1 * 128 + 112 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(state_inp_ready_addr);
                    {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(state_read_done_addr);
                        mbarrier_arrive(checkpoint_ready_addr + (checkpoint_stage_cg1) * 8);
                    }
                    mbarrier_wait(v_ready_addr + (raw_bar_stage_cg1) * 8, (unsigned int)(cumulative_cg1 / 6 & 1));
                    mbarrier_wait(beta_ready_addr + (raw_bar_stage_cg1) * 8, (unsigned int)(cumulative_cg1 / 6 & 1));
                    mbarrier_wait(state_k_ready_addr, state_phase_cg1);
                    float _tmem_load_3[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7]))
                        : "r"(taddr + 224 + (unsigned int)tmem_row_base_cg1)
                        : "memory");
                    float _tmem_load_4[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7]))
                        : "r"(taddr + 224 + (unsigned int)tmem_row_base_cg1 + 1048576)
                        : "memory");
                    unsigned int raw_v_words_lo_cg1[4];
                    unsigned int raw_v_words_hi_cg1[4];
                    int segment_3 = (value_dim_base_cg1 + ov_col_cg1) / 64;
                    int segment_col_3 = value_dim_base_cg1 + ov_col_cg1 - segment_3 * 64;
                    int swizzled_col_3 = segment_col_3 ^ (ov_token_cg1 & 7) * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(raw_v_words_lo_cg1[0]), "=r"(raw_v_words_lo_cg1[1]), "=r"(raw_v_words_lo_cg1[2]), "=r"(raw_v_words_lo_cg1[3])
                        : "r"(smem_v_all_addr + (raw_stage_cg1 * 16 * 128 + (unsigned int)(segment_3 * 16 * 64) + (unsigned int)(ov_token_cg1 * 64) + (unsigned int)swizzled_col_3) * 2)
                        : "memory");
                    int segment_0_1 = (value_dim_base_cg1 + 16 + ov_col_cg1) / 64;
                    int segment_col_1_2 = value_dim_base_cg1 + 16 + ov_col_cg1 - segment_0_1 * 64;
                    int swizzled_col_2_2 = segment_col_1_2 ^ (ov_token_cg1 & 7) * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(raw_v_words_hi_cg1[0]), "=r"(raw_v_words_hi_cg1[1]), "=r"(raw_v_words_hi_cg1[2]), "=r"(raw_v_words_hi_cg1[3])
                        : "r"(smem_v_all_addr + (raw_stage_cg1 * 16 * 128 + (unsigned int)(segment_0_1 * 16 * 64) + (unsigned int)(ov_token_cg1 * 64) + (unsigned int)swizzled_col_2_2) * 2)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    __nv_bfloat162 beta_pairs_cg1[4];
                    #pragma unroll
                    for (int beta_reg_cg1 = 0; beta_reg_cg1 < 4; beta_reg_cg1++) {
                        int beta_packed_col_cg1 = beta_reg_cg1 / 2 * 4 + (lane & 3);
                        int beta_token0_cg1 = beta_packed_col_cg1 * 2;
                        int beta_token1_cg1 = beta_token0_cg1 + 1;
                        __nv_bfloat16 beta0_cg1 = smem_beta_all[(int)raw_bar_stage_cg1 * 16 + beta_token0_cg1];
                        __nv_bfloat16 beta1_cg1 = smem_beta_all[(int)raw_bar_stage_cg1 * 16 + beta_token1_cg1];
                        float _cvt_f32_3 = __bfloat162float(beta0_cg1);
                        float _cvt_f32_4 = __bfloat162float(beta1_cg1);
                        __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_cvt_f32_3, _cvt_f32_4));
                        beta_pairs_cg1[beta_reg_cg1] = _bf16x2_6;
                    }
                    unsigned int y_words_lo_cg1[4];
                    unsigned int y_words_hi_cg1[4];
                    #pragma unroll
                    for (int rhs_reg_cg1 = 0; rhs_reg_cg1 < 4; rhs_reg_cg1++) {
                        int rhs_raw_matrix_cg1 = rhs_reg_cg1;
                        int rhs_frag_pair_cg1 = rhs_reg_cg1 * 2;
                        __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_tmem_load_3[rhs_frag_pair_cg1], _tmem_load_3[rhs_frag_pair_cg1 + 1]));
                        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_tmem_load_4[rhs_frag_pair_cg1], _tmem_load_4[rhs_frag_pair_cg1 + 1]));
                        uint32_t _bf16x2_sub_0;
                        asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_sub_0) : "r"(raw_v_words_lo_cg1[rhs_raw_matrix_cg1]), "r"(__as_u32(_bf16x2_7)));
                        uint32_t _bf16x2_sub_1;
                        asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_sub_1) : "r"(raw_v_words_hi_cg1[rhs_raw_matrix_cg1]), "r"(__as_u32(_bf16x2_8)));
                        __nv_bfloat162 rhs_diff_pair_lo_cg1 = __as_bf16x2(_bf16x2_sub_0);
                        __nv_bfloat162 rhs_diff_pair_hi_cg1 = __as_bf16x2(_bf16x2_sub_1);
                        __nv_bfloat162 y_pair_lo_cg1 = beta_pairs_cg1[rhs_reg_cg1] * rhs_diff_pair_lo_cg1;
                        __nv_bfloat162 y_pair_hi_cg1 = beta_pairs_cg1[rhs_reg_cg1] * rhs_diff_pair_hi_cg1;
                        y_words_lo_cg1[rhs_reg_cg1] = __as_u32(y_pair_lo_cg1);
                        y_words_hi_cg1[rhs_reg_cg1] = __as_u32(y_pair_hi_cg1);
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x2.b32"
                        " [%0], {%1, %2, %3, %4};"
                        :: "r"(taddr + 256 + (unsigned int)tmem_row_base_cg1), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1[3]))
                        : "memory");
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x2.b32"
                        " [%0], {%1, %2, %3, %4};"
                        :: "r"(taddr + 256 + (unsigned int)tmem_row_base_cg1 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1[3]))
                        : "memory");
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(v_done_addr + (raw_stage_cg1) * 8);
                    mbarrier_arrive(beta_done_addr + (raw_bar_stage_cg1) * 8);
                    mbarrier_arrive(y_inp_ready_addr);
                    mbarrier_wait(u_acc_ready_addr, state_phase_cg1);
                    float _tmem_load_5[16];
                    tmem_ld_x16(&_tmem_load_5[0], taddr + 240 + (unsigned int)tmem_row_base_cg1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int u_words_cg1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                        u_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 264 + (unsigned int)tmem_row_base_cg1, (const uint32_t*)u_words_cg1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(u_inp_ready_addr);
                }
                #pragma unroll 1
                for (int chunk_cg1 = 1; chunk_cg1 < chunks_cg1; chunk_cg1++) {
                    int cumulative_cg1_1 = cumulative_chunk_cg1 + chunk_cg1;
                    unsigned int raw_stage_cg1_1 = (unsigned int)(cumulative_cg1_1 % 5);
                    unsigned int raw_bar_stage_cg1_1 = (unsigned int)(cumulative_cg1_1 % 6);
                    unsigned int o_stage_cg1_1 = (unsigned int)(cumulative_cg1_1 % 2);
                    unsigned int checkpoint_stage_cg1_1 = (unsigned int)(cumulative_cg1_1 % 2);
                    unsigned int state_phase_cg1_1 = (unsigned int)(cumulative_cg1_1 & 1);
                    {
                        mbarrier_wait(state_acc_done_addr, (unsigned int)(cumulative_cg1_1 - 1 & 1));
                    }
                    float _tmem_load_6[128];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)tmem_row_base_cg1);
                    tmem_ld_x16(&_tmem_load_6[16], taddr + (unsigned int)tmem_row_base_cg1 + 16);
                    tmem_ld_x16(&_tmem_load_6[32], taddr + (unsigned int)tmem_row_base_cg1 + 32);
                    tmem_ld_x16(&_tmem_load_6[48], taddr + (unsigned int)tmem_row_base_cg1 + 48);
                    tmem_ld_x16(&_tmem_load_6[64], taddr + (unsigned int)tmem_row_base_cg1 + 64);
                    tmem_ld_x16(&_tmem_load_6[80], taddr + (unsigned int)tmem_row_base_cg1 + 80);
                    tmem_ld_x16(&_tmem_load_6[96], taddr + (unsigned int)tmem_row_base_cg1 + 96);
                    tmem_ld_x16(&_tmem_load_6[112], taddr + (unsigned int)tmem_row_base_cg1 + 112);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    {
                        mbarrier_wait(checkpoint_done_addr + (checkpoint_stage_cg1_1) * 8, (unsigned int)(cumulative_cg1_1 / 2 + 1 & 1));
                    }
                    unsigned int state_words_cg1_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 ^ (value_row_cg1 * 128 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 16 ^ (value_row_cg1 * 128 + 16 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 16], _tmem_load_6[_lp*2+1 + 16]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 8, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 32 ^ (value_row_cg1 * 128 + 32 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 48 ^ (value_row_cg1 * 128 + 48 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 32], _tmem_load_6[_lp*2+1 + 32]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 16, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 64 ^ (value_row_cg1 * 128 + 64 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 80 ^ (value_row_cg1 * 128 + 80 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 48], _tmem_load_6[_lp*2+1 + 48]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 24, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 96 ^ (value_row_cg1 * 128 + 96 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 112 ^ (value_row_cg1 * 128 + 112 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 64], _tmem_load_6[_lp*2+1 + 64]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 32, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 ^ (16384 + value_row_cg1 * 128 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 16 ^ (16384 + value_row_cg1 * 128 + 16 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 80], _tmem_load_6[_lp*2+1 + 80]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 40, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 32 ^ (16384 + value_row_cg1 * 128 + 32 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 48 ^ (16384 + value_row_cg1 * 128 + 48 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 96], _tmem_load_6[_lp*2+1 + 96]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 48, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 64 ^ (16384 + value_row_cg1 * 128 + 64 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 80 ^ (16384 + value_row_cg1 * 128 + 80 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 112], _tmem_load_6[_lp*2+1 + 112]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 56, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 96 ^ (16384 + value_row_cg1 * 128 + 96 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 112 ^ (16384 + value_row_cg1 * 128 + 112 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(state_inp_ready_addr);
                    {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(state_read_done_addr);
                        mbarrier_arrive(checkpoint_ready_addr + (checkpoint_stage_cg1_1) * 8);
                    }
                    {
                        int previous_event_cg1 = cumulative_cg1_1 - 1;
                        unsigned int previous_o_stage_cg1 = (unsigned int)(previous_event_cg1 % 2);
                        mbarrier_wait(o_acc_ready_addr, (unsigned int)(previous_event_cg1 & 1));
                        mbarrier_wait(o_tma_done_addr + (previous_o_stage_cg1) * 8, (unsigned int)(previous_event_cg1 / 2 + 1 & 1));
                        int output_col_cg1 = 192 + (int)previous_o_stage_cg1 * 16;
                        float _tmem_load_7[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7]))
                            : "r"(taddr + (unsigned int)output_col_cg1 + (unsigned int)tmem_row_base_cg1)
                            : "memory");
                        float _tmem_load_8[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7]))
                            : "r"(taddr + (unsigned int)output_col_cg1 + (unsigned int)tmem_row_base_cg1 + 1048576)
                            : "memory");
                        const float2 _scale2_1 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_1);
                        const float2 _scale2_2 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_ls], _scale2_2);
                        uint32_t _tmem_load_7_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                            _tmem_load_7_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _tmem_load_8_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                            _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int segment_4 = (value_dim_base_cg1 + ov_col_cg1) / 64;
                        int segment_col_4 = value_dim_base_cg1 + ov_col_cg1 - segment_4 * 64;
                        int swizzled_col_4 = segment_col_4 ^ (ov_token_cg1 & 7) * 8;
                        int segment_0_2 = (value_dim_base_cg1 + 16 + ov_col_cg1) / 64;
                        int segment_col_1_3 = value_dim_base_cg1 + 16 + ov_col_cg1 - segment_0_2 * 64;
                        int swizzled_col_2_3 = segment_col_1_3 ^ (ov_token_cg1 & 7) * 8;
                        uint32_t _stmatrix_addr_3 = static_cast<uint32_t>((unsigned long long)(smem_o_all_addr + (unsigned int)(((int)previous_o_stage_cg1 * 16 * 128 + segment_4 * 16 * 64 + ov_token_cg1 * 64 + swizzled_col_4) * 2)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[3]))
                            : "memory");
                        uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((unsigned long long)(smem_o_all_addr + (unsigned int)(((int)previous_o_stage_cg1 * 16 * 128 + segment_0_2 * 16 * 64 + ov_token_cg1 * 64 + swizzled_col_2_3) * 2)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[3]))
                            : "memory");
                        mbarrier_arrive(o_acc_done_addr + (previous_o_stage_cg1) * 8);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(o_tma_ready_addr + (previous_o_stage_cg1) * 8);
                    }
                    mbarrier_wait(v_ready_addr + (raw_bar_stage_cg1_1) * 8, (unsigned int)(cumulative_cg1_1 / 6 & 1));
                    mbarrier_wait(beta_ready_addr + (raw_bar_stage_cg1_1) * 8, (unsigned int)(cumulative_cg1_1 / 6 & 1));
                    mbarrier_wait(state_k_ready_addr, state_phase_cg1_1);
                    float _tmem_load_9[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7]))
                        : "r"(taddr + 224 + (unsigned int)tmem_row_base_cg1)
                        : "memory");
                    float _tmem_load_10[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7]))
                        : "r"(taddr + 224 + (unsigned int)tmem_row_base_cg1 + 1048576)
                        : "memory");
                    unsigned int raw_v_words_lo_cg1_1[4];
                    unsigned int raw_v_words_hi_cg1_1[4];
                    int segment_5 = (value_dim_base_cg1 + ov_col_cg1) / 64;
                    int segment_col_5 = value_dim_base_cg1 + ov_col_cg1 - segment_5 * 64;
                    int swizzled_col_5 = segment_col_5 ^ (ov_token_cg1 & 7) * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(raw_v_words_lo_cg1_1[0]), "=r"(raw_v_words_lo_cg1_1[1]), "=r"(raw_v_words_lo_cg1_1[2]), "=r"(raw_v_words_lo_cg1_1[3])
                        : "r"(smem_v_all_addr + (raw_stage_cg1_1 * 16 * 128 + (unsigned int)(segment_5 * 16 * 64) + (unsigned int)(ov_token_cg1 * 64) + (unsigned int)swizzled_col_5) * 2)
                        : "memory");
                    int segment_0_3 = (value_dim_base_cg1 + 16 + ov_col_cg1) / 64;
                    int segment_col_1_4 = value_dim_base_cg1 + 16 + ov_col_cg1 - segment_0_3 * 64;
                    int swizzled_col_2_4 = segment_col_1_4 ^ (ov_token_cg1 & 7) * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(raw_v_words_hi_cg1_1[0]), "=r"(raw_v_words_hi_cg1_1[1]), "=r"(raw_v_words_hi_cg1_1[2]), "=r"(raw_v_words_hi_cg1_1[3])
                        : "r"(smem_v_all_addr + (raw_stage_cg1_1 * 16 * 128 + (unsigned int)(segment_0_3 * 16 * 64) + (unsigned int)(ov_token_cg1 * 64) + (unsigned int)swizzled_col_2_4) * 2)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    __nv_bfloat162 beta_pairs_cg1_1[4];
                    #pragma unroll
                    for (int beta_reg_cg1_1 = 0; beta_reg_cg1_1 < 4; beta_reg_cg1_1++) {
                        int beta_packed_col_cg1_1 = beta_reg_cg1_1 / 2 * 4 + (lane & 3);
                        int beta_token0_cg1_1 = beta_packed_col_cg1_1 * 2;
                        int beta_token1_cg1_1 = beta_token0_cg1_1 + 1;
                        __nv_bfloat16 beta0_cg1_1 = smem_beta_all[(int)raw_bar_stage_cg1_1 * 16 + beta_token0_cg1_1];
                        __nv_bfloat16 beta1_cg1_1 = smem_beta_all[(int)raw_bar_stage_cg1_1 * 16 + beta_token1_cg1_1];
                        float _cvt_f32_5 = __bfloat162float(beta0_cg1_1);
                        float _cvt_f32_6 = __bfloat162float(beta1_cg1_1);
                        __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_cvt_f32_5, _cvt_f32_6));
                        beta_pairs_cg1_1[beta_reg_cg1_1] = _bf16x2_9;
                    }
                    unsigned int y_words_lo_cg1_1[4];
                    unsigned int y_words_hi_cg1_1[4];
                    #pragma unroll
                    for (int rhs_reg_cg1_1 = 0; rhs_reg_cg1_1 < 4; rhs_reg_cg1_1++) {
                        int rhs_raw_matrix_cg1_1 = rhs_reg_cg1_1;
                        int rhs_frag_pair_cg1_1 = rhs_reg_cg1_1 * 2;
                        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_tmem_load_9[rhs_frag_pair_cg1_1], _tmem_load_9[rhs_frag_pair_cg1_1 + 1]));
                        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_tmem_load_10[rhs_frag_pair_cg1_1], _tmem_load_10[rhs_frag_pair_cg1_1 + 1]));
                        uint32_t _bf16x2_sub_2;
                        asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_sub_2) : "r"(raw_v_words_lo_cg1_1[rhs_raw_matrix_cg1_1]), "r"(__as_u32(_bf16x2_10)));
                        uint32_t _bf16x2_sub_3;
                        asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_sub_3) : "r"(raw_v_words_hi_cg1_1[rhs_raw_matrix_cg1_1]), "r"(__as_u32(_bf16x2_11)));
                        __nv_bfloat162 rhs_diff_pair_lo_cg1_1 = __as_bf16x2(_bf16x2_sub_2);
                        __nv_bfloat162 rhs_diff_pair_hi_cg1_1 = __as_bf16x2(_bf16x2_sub_3);
                        __nv_bfloat162 y_pair_lo_cg1_1 = beta_pairs_cg1_1[rhs_reg_cg1_1] * rhs_diff_pair_lo_cg1_1;
                        __nv_bfloat162 y_pair_hi_cg1_1 = beta_pairs_cg1_1[rhs_reg_cg1_1] * rhs_diff_pair_hi_cg1_1;
                        y_words_lo_cg1_1[rhs_reg_cg1_1] = __as_u32(y_pair_lo_cg1_1);
                        y_words_hi_cg1_1[rhs_reg_cg1_1] = __as_u32(y_pair_hi_cg1_1);
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x2.b32"
                        " [%0], {%1, %2, %3, %4};"
                        :: "r"(taddr + 256 + (unsigned int)tmem_row_base_cg1), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1_1[3]))
                        : "memory");
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x2.b32"
                        " [%0], {%1, %2, %3, %4};"
                        :: "r"(taddr + 256 + (unsigned int)tmem_row_base_cg1 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1_1[3]))
                        : "memory");
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(v_done_addr + (raw_stage_cg1_1) * 8);
                    mbarrier_arrive(beta_done_addr + (raw_bar_stage_cg1_1) * 8);
                    mbarrier_arrive(y_inp_ready_addr);
                    mbarrier_wait(u_acc_ready_addr, state_phase_cg1_1);
                    float _tmem_load_11[16];
                    tmem_ld_x16(&_tmem_load_11[0], taddr + 240 + (unsigned int)tmem_row_base_cg1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int u_words_cg1_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                        u_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 264 + (unsigned int)tmem_row_base_cg1, (const uint32_t*)u_words_cg1_1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(u_inp_ready_addr);
                }
                if (chunks_cg1 > 0) {
                    int final_event_cg1 = cumulative_chunk_cg1 + chunks_cg1 - 1;
                    unsigned int final_o_stage_cg1 = (unsigned int)(final_event_cg1 % 2);
                    mbarrier_wait(state_acc_done_addr, (unsigned int)(final_event_cg1 & 1));
                    mbarrier_wait(o_acc_ready_addr, (unsigned int)(final_event_cg1 & 1));
                    mbarrier_wait(o_tma_done_addr + (final_o_stage_cg1) * 8, (unsigned int)(final_event_cg1 / 2 + 1 & 1));
                    int output_col_cg1_1 = 192 + (int)final_o_stage_cg1 * 16;
                    float _tmem_load_12[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7]))
                        : "r"(taddr + (unsigned int)output_col_cg1_1 + (unsigned int)tmem_row_base_cg1)
                        : "memory");
                    float _tmem_load_13[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7]))
                        : "r"(taddr + (unsigned int)output_col_cg1_1 + (unsigned int)tmem_row_base_cg1 + 1048576)
                        : "memory");
                    const float2 _scale2_5 = {scale, scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_12)[_ls], _scale2_5);
                    const float2 _scale2_6 = {scale, scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_13)[_ls], _scale2_6);
                    uint32_t _tmem_load_12_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 0], _tmem_load_12[_lp*2+1 + 0]));
                        _tmem_load_12_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _tmem_load_13_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 0], _tmem_load_13[_lp*2+1 + 0]));
                        _tmem_load_13_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int segment_6 = (value_dim_base_cg1 + ov_col_cg1) / 64;
                    int segment_col_6 = value_dim_base_cg1 + ov_col_cg1 - segment_6 * 64;
                    int swizzled_col_6 = segment_col_6 ^ (ov_token_cg1 & 7) * 8;
                    int segment_0_4 = (value_dim_base_cg1 + 16 + ov_col_cg1) / 64;
                    int segment_col_1_5 = value_dim_base_cg1 + 16 + ov_col_cg1 - segment_0_4 * 64;
                    int swizzled_col_2_5 = segment_col_1_5 ^ (ov_token_cg1 & 7) * 8;
                    uint32_t _stmatrix_addr_7 = static_cast<uint32_t>((unsigned long long)(smem_o_all_addr + (unsigned int)(((int)final_o_stage_cg1 * 16 * 128 + segment_6 * 16 * 64 + ov_token_cg1 * 64 + swizzled_col_6) * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[3]))
                        : "memory");
                    uint32_t _stmatrix_addr_8 = static_cast<uint32_t>((unsigned long long)(smem_o_all_addr + (unsigned int)(((int)final_o_stage_cg1 * 16 * 128 + segment_0_4 * 16 * 64 + ov_token_cg1 * 64 + swizzled_col_2_5) * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[3]))
                        : "memory");
                    mbarrier_arrive(o_acc_done_addr + (final_o_stage_cg1) * 8);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(o_tma_ready_addr + (final_o_stage_cg1) * 8);
                    {
                        #pragma unroll
                        for (int final_block_cg1 = 0; final_block_cg1 < 4; final_block_cg1++) {
                            float _tmem_load_14[32];
                            tmem_ld_x32(&_tmem_load_14[0], taddr + (unsigned int)tmem_row_base_cg1 + (unsigned int)(final_block_cg1 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            {
                                __nv_bfloat162 _pk[8];
                                _pk[0] = __floats2bfloat162_rn(_tmem_load_14[0 + 0], _tmem_load_14[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_tmem_load_14[0 + 2], _tmem_load_14[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_tmem_load_14[0 + 4], _tmem_load_14[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_tmem_load_14[0 + 6], _tmem_load_14[0 + 7]);
                                _pk[4] = __floats2bfloat162_rn(_tmem_load_14[0 + 8], _tmem_load_14[0 + 9]);
                                _pk[5] = __floats2bfloat162_rn(_tmem_load_14[0 + 10], _tmem_load_14[0 + 11]);
                                _pk[6] = __floats2bfloat162_rn(_tmem_load_14[0 + 12], _tmem_load_14[0 + 13]);
                                _pk[7] = __floats2bfloat162_rn(_tmem_load_14[0 + 14], _tmem_load_14[0 + 15]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base_cg1 + (long long)(final_block_cg1 * 32))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base_cg1 + (long long)(final_block_cg1 * 32))))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                            }
                            {
                                __nv_bfloat162 _pk[8];
                                _pk[0] = __floats2bfloat162_rn(_tmem_load_14[16 + 0], _tmem_load_14[16 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_tmem_load_14[16 + 2], _tmem_load_14[16 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_tmem_load_14[16 + 4], _tmem_load_14[16 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_tmem_load_14[16 + 6], _tmem_load_14[16 + 7]);
                                _pk[4] = __floats2bfloat162_rn(_tmem_load_14[16 + 8], _tmem_load_14[16 + 9]);
                                _pk[5] = __floats2bfloat162_rn(_tmem_load_14[16 + 10], _tmem_load_14[16 + 11]);
                                _pk[6] = __floats2bfloat162_rn(_tmem_load_14[16 + 12], _tmem_load_14[16 + 13]);
                                _pk[7] = __floats2bfloat162_rn(_tmem_load_14[16 + 14], _tmem_load_14[16 + 15]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base_cg1 + (long long)(final_block_cg1 * 32) + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base_cg1 + (long long)(final_block_cg1 * 32) + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                            }
                        }
                    }
                }
                cumulative_chunk_cg1 += chunks_cg1;
            }
            if (elect_sync()) {
                mbarrier_arrive(consumers_done_addr);
            }
        }
    }
    // ---- Role: super_mma ----
    if (warp == 12) {
        { // super_mma_main
            unsigned int sched_stage_super = 0;
            int cumulative_chunk_super = 0;
            int lhs_row_super = lane % 8 + (lane / 8 & 1) * 8;
            int lhs_col_super = lane / 16 * 8;
            int rhs_row_super = lane % 8 + lane / 16 * 8;
            int rhs_col_super = (lane / 8 & 1) * 8;
            unsigned int _phase_sched_ready_2 = 0;
            #pragma unroll 1
            for (int __2 = 0; __2 < total_work_items + 1; __2++) {
                mbarrier_wait(sched_ready_addr + (sched_stage_super) * 8, _phase_sched_ready_2);
                unsigned int slot_2[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&slot_2[0])) : "r"(sched_slot_addr + sched_stage_super * 4));
                unsigned int tile_super = slot_2[0];
                if (elect_sync()) {
                    mbarrier_arrive(sched_done_addr + (sched_stage_super) * 8);
                }
                sched_stage_super += 1;
                if (sched_stage_super == 8) { sched_stage_super = 0; _phase_sched_ready_2 ^= 1; }
                if (tile_super >= (unsigned int)total_work_items) {
                    break;
                }
                int item_base_super = (int)tile_super * 8;
                int chunks_super = work_items[item_base_super + 3] - work_items[item_base_super + 4];
                #pragma unroll 1
                for (int chunk_super = 0; chunk_super < chunks_super; chunk_super++) {
                    int cumulative_super = cumulative_chunk_super + chunk_super;
                    unsigned int decay_stage_super = (unsigned int)(cumulative_super % 2);
                    unsigned int raw_bar_stage_super = (unsigned int)(cumulative_super % 6);
                    unsigned int intermediate_stage_super = (unsigned int)(cumulative_super % 2);
                    mbarrier_wait(k_decay_inv_ready_addr + (decay_stage_super) * 8, (unsigned int)(cumulative_super / 2 & 1));
                    mbarrier_wait(beta_ready_addr + (raw_bar_stage_super) * 8, (unsigned int)(cumulative_super / 6 & 1));
                    mbarrier_wait(tinv_done_addr + (intermediate_stage_super) * 8, (unsigned int)(cumulative_super / 2 + 1 & 1));
                    float kk_acc_super[8];
                    #pragma unroll
                    for (int k_block_super = 0; k_block_super < 8; k_block_super++) {
                        unsigned int a_frag_super[4];
                        unsigned int b_frag_super[4];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_super[0]), "=r"(a_frag_super[1]), "=r"(a_frag_super[2]), "=r"(a_frag_super[3])
                            : "r"((smem_k_decay_addr + decay_stage_super * 4096 + (unsigned int)((k_block_super * 16 + lhs_col_super) / 64 * 2048 + lhs_row_super * 128 + (k_block_super * 16 + lhs_col_super) % 64 * 2 ^ ((k_block_super * 16 + lhs_col_super) / 64 * 2048 + lhs_row_super * 128 + (k_block_super * 16 + lhs_col_super) % 64 * 2 >> 7 & 7) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_super[0]), "=r"(b_frag_super[1]), "=r"(b_frag_super[2]), "=r"(b_frag_super[3])
                            : "r"((smem_k_inv_addr + decay_stage_super * 4096 + (unsigned int)((k_block_super * 16 + rhs_col_super) / 64 * 2048 + rhs_row_super * 128 + (k_block_super * 16 + rhs_col_super) % 64 * 2 ^ ((k_block_super * 16 + rhs_col_super) / 64 * 2048 + rhs_row_super * 128 + (k_block_super * 16 + rhs_col_super) % 64 * 2 >> 7 & 7) << 4)))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(kk_acc_super[0]), "=f"(kk_acc_super[1]), "=f"(kk_acc_super[2]), "=f"(kk_acc_super[3])
                            : "r"(a_frag_super[0]), "r"(a_frag_super[1]), "r"(a_frag_super[2]), "r"(a_frag_super[3]), "r"(b_frag_super[0]), "r"(b_frag_super[1]), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[0])), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[1])), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[2])), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[3])));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(kk_acc_super[4]), "=f"(kk_acc_super[(4) + 1]), "=f"(kk_acc_super[(4) + 2]), "=f"(kk_acc_super[(4) + 3])
                            : "r"(a_frag_super[0]), "r"(a_frag_super[1]), "r"(a_frag_super[2]), "r"(a_frag_super[3]), "r"(b_frag_super[2]), "r"(b_frag_super[(2) + 1]), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[4])), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[(4) + 1])), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[(4) + 2])), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[(4) + 3])));
                    }
                    int beta_stage_base_super = (int)raw_bar_stage_super * 16;
                    __nv_bfloat16 beta_lo_bf_super = smem_beta_all[beta_stage_base_super + lane / 4];
                    __nv_bfloat16 beta_hi_bf_super = smem_beta_all[beta_stage_base_super + lane / 4 + 8];
                    float _cvt_f32_0 = __bfloat162float(beta_lo_bf_super);
                    float beta_lo_super = _cvt_f32_0;
                    float _cvt_f32_1 = __bfloat162float(beta_hi_bf_super);
                    float beta_hi_super = _cvt_f32_1;
                    float l_values_super[8];
                    float tinv_acc_super[8];
                    #pragma unroll
                    for (int accum_super = 0; accum_super < 8; accum_super++) {
                        int row_super = lane / 4 + accum_super % 4 / 2 * 8;
                        int col_super = accum_super / 4 * 8 + (lane & 3) * 2 + (accum_super & 1);
                        l_values_super[accum_super] = 0.0f;
                        if (row_super > col_super) {
                            float beta_scale_super = beta_lo_super;
                            if (accum_super % 4 >= 2) {
                                beta_scale_super = beta_hi_super;
                            }
                            l_values_super[accum_super] = kk_acc_super[accum_super] * beta_scale_super;
                        }
                        tinv_acc_super[accum_super] = -l_values_super[accum_super];
                        if (row_super == col_super) {
                            tinv_acc_super[accum_super] = 1.0f;
                        }
                    }
                    unsigned int lpow_words_super[4];
                    unsigned int lpow_trans_super[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(l_values_super[_lp*2 + 0], l_values_super[_lp*2+1 + 0]));
                        lpow_words_super[_lp] = *(uint32_t*)&_bf2;
                    }
                    int store_row_super = lane % 16;
                    int store_col_super = lane / 16 * 8;
                    int linear = store_row_super * 16 + store_col_super;
                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(tinv_scratch_addr + (unsigned int)((linear ^ (linear >> 6 & 1) * 8) * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[0])), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[1])), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[2])), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[3]))
                        : "memory");
                    __syncwarp();
                    int load_row_super = lane % 16;
                    #pragma unroll
                    for (int load_half_super = 0; load_half_super < 2; load_half_super++) {
                        int linear_0 = load_row_super * 16 + load_half_super * 8;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(lpow_trans_super[load_half_super * 2]), "=r"(lpow_trans_super[load_half_super * 2 + 1])
                            : "r"(tinv_scratch_addr + (unsigned int)((linear_0 ^ (linear_0 >> 6 & 1) * 8) * 2))
                            : "memory");
                    }
                    #pragma unroll
                    for (int neumann_super = 0; neumann_super < 3; neumann_super++) {
                        float square_acc_super[8];
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(square_acc_super[0]), "=f"(square_acc_super[1]), "=f"(square_acc_super[2]), "=f"(square_acc_super[3])
                            : "r"(lpow_words_super[0]), "r"(lpow_words_super[1]), "r"(lpow_words_super[2]), "r"(lpow_words_super[3]), "r"(lpow_trans_super[0]), "r"(lpow_trans_super[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(square_acc_super[4]), "=f"(square_acc_super[(4) + 1]), "=f"(square_acc_super[(4) + 2]), "=f"(square_acc_super[(4) + 3])
                            : "r"(lpow_words_super[0]), "r"(lpow_words_super[1]), "r"(lpow_words_super[2]), "r"(lpow_words_super[3]), "r"(lpow_trans_super[2]), "r"(lpow_trans_super[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(square_acc_super[_lp*2 + 0], square_acc_super[_lp*2+1 + 0]));
                            lpow_words_super[_lp] = *(uint32_t*)&_bf2;
                        }
                        int store_row_super_0 = lane % 16;
                        int store_col_super_1 = lane / 16 * 8;
                        int linear_2 = store_row_super_0 * 16 + store_col_super_1;
                        uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)(tinv_scratch_addr + (unsigned int)((linear_2 ^ (linear_2 >> 6 & 1) * 8) * 2)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[0])), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[1])), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[2])), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[3]))
                            : "memory");
                        __syncwarp();
                        int load_row_super_3 = lane % 16;
                        #pragma unroll
                        for (int load_half_super_1 = 0; load_half_super_1 < 2; load_half_super_1++) {
                            int linear_0_1 = load_row_super_3 * 16 + load_half_super_1 * 8;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                : "=r"(lpow_trans_super[load_half_super_1 * 2]), "=r"(lpow_trans_super[load_half_super_1 * 2 + 1])
                                : "r"(tinv_scratch_addr + (unsigned int)((linear_0_1 ^ (linear_0_1 >> 6 & 1) * 8) * 2))
                                : "memory");
                        }
                        unsigned int tinv_words_super[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tinv_acc_super[_lp*2 + 0], tinv_acc_super[_lp*2+1 + 0]));
                            tinv_words_super[_lp] = *(uint32_t*)&_bf2;
                        }
                        float update_acc_super[8];
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(update_acc_super[0]), "=f"(update_acc_super[1]), "=f"(update_acc_super[2]), "=f"(update_acc_super[3])
                            : "r"(tinv_words_super[0]), "r"(tinv_words_super[1]), "r"(tinv_words_super[2]), "r"(tinv_words_super[3]), "r"(lpow_trans_super[0]), "r"(lpow_trans_super[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(update_acc_super[4]), "=f"(update_acc_super[(4) + 1]), "=f"(update_acc_super[(4) + 2]), "=f"(update_acc_super[(4) + 3])
                            : "r"(tinv_words_super[0]), "r"(tinv_words_super[1]), "r"(tinv_words_super[2]), "r"(tinv_words_super[3]), "r"(lpow_trans_super[2]), "r"(lpow_trans_super[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        float tinv_words_super_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&tinv_words_super_f32[_pair * 2])[0]), "=f"((&tinv_words_super_f32[_pair * 2])[1])
                                : "r"(tinv_words_super[_pair]));
                        }
                        #pragma unroll
                        for (int update_super = 0; update_super < 8; update_super++) {
                            tinv_acc_super[update_super] = tinv_words_super_f32[update_super] + update_acc_super[update_super];
                        }
                    }
                    unsigned int tinv_publish_super[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tinv_acc_super[_lp*2 + 0], tinv_acc_super[_lp*2+1 + 0]));
                        tinv_publish_super[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_2 = static_cast<uint32_t>((unsigned long long)(smem_tinv_addr + intermediate_stage_super * 1024 + (unsigned int)(lane / 16 * 8 / 16 * 512 + lane % 16 * 32 + lane / 16 * 8 % 16 * 2 ^ (lane / 16 * 8 / 16 * 512 + lane % 16 * 32 + lane / 16 * 8 % 16 * 2 >> 7 & 1) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&tinv_publish_super[0])), "r"(*reinterpret_cast<const uint32_t*>(&tinv_publish_super[1])), "r"(*reinterpret_cast<const uint32_t*>(&tinv_publish_super[2])), "r"(*reinterpret_cast<const uint32_t*>(&tinv_publish_super[3]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(tinv_ready_addr + (intermediate_stage_super) * 8);
                    mbarrier_arrive(beta_done_addr + (raw_bar_stage_super) * 8);
                    mbarrier_arrive(decay_super_done_addr + (decay_stage_super) * 8);
                }
                cumulative_chunk_super += chunks_super;
            }
            if (elect_sync()) {
                mbarrier_arrive(consumers_done_addr);
            }
        }
    }
    // ---- Role: tcgen ----
    if (warp == 13) {
        { // tcgen_main
            float tmem_seed_tcgen[1];
            tmem_seed_tcgen[0] = 0.0f;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr), "r"(*reinterpret_cast<const uint32_t*>(&tmem_seed_tcgen[0]))
                : "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int sched_stage_tcgen = 0;
            int cumulative_chunk_tcgen = 0;
            unsigned int _phase_sched_ready_3 = 0;
            #pragma unroll 1
            for (int __3 = 0; __3 < total_work_items + 1; __3++) {
                mbarrier_wait(sched_ready_addr + (sched_stage_tcgen) * 8, _phase_sched_ready_3);
                unsigned int slot_3[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&slot_3[0])) : "r"(sched_slot_addr + sched_stage_tcgen * 4));
                unsigned int tile_tcgen = slot_3[0];
                if (elect_sync()) {
                    mbarrier_arrive(sched_done_addr + (sched_stage_tcgen) * 8);
                }
                sched_stage_tcgen += 1;
                if (sched_stage_tcgen == 8) { sched_stage_tcgen = 0; _phase_sched_ready_3 ^= 1; }
                if (tile_tcgen >= (unsigned int)total_work_items) {
                    break;
                }
                int item_base_tcgen = (int)tile_tcgen * 8;
                int chunks_tcgen = work_items[item_base_tcgen + 3] - work_items[item_base_tcgen + 4];
                #pragma unroll 1
                for (int chunk_tcgen = 0; chunk_tcgen < chunks_tcgen; chunk_tcgen++) {
                    int cumulative_tcgen = cumulative_chunk_tcgen + chunk_tcgen;
                    unsigned int state_phase_tcgen = (unsigned int)(cumulative_tcgen & 1);
                    unsigned int decay_stage_tcgen = (unsigned int)(cumulative_tcgen % 2);
                    unsigned int diag_stage_tcgen = (unsigned int)(cumulative_tcgen % 4);
                    unsigned int intermediate_stage_tcgen = (unsigned int)(cumulative_tcgen % 2);
                    unsigned int o_stage_tcgen = (unsigned int)(cumulative_tcgen % 2);
                    mbarrier_wait(k_decay_inv_ready_addr + (decay_stage_tcgen) * 8, (unsigned int)(cumulative_tcgen / 2 & 1));
                    mbarrier_wait(state_inp_ready_addr, state_phase_tcgen);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_k_decay_addr) >> 4) & 0x3FFF) + (decay_stage_tcgen) * 256);
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
                    :: "r"(tmem_tmem_state_k), "r"(_mma_b_lo_0), "r"(tmem_tmem_state_inp), "r"(0));
                    elect_commit(state_k_ready_addr);
                    mbarrier_wait(qk_scale_ready_addr + (diag_stage_tcgen) * 8, (unsigned int)(cumulative_tcgen / 4 & 1));
                    mbarrier_wait(o_acc_done_addr + (o_stage_tcgen) * 8, (unsigned int)(cumulative_tcgen / 2 + 1 & 1));
                    int _mma_b_lo_1 = make_warp_uniform((((smem_q_decay_addr) >> 4) & 0x3FFF) + (decay_stage_tcgen) * 256);
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
                    :: "r"((tmem_tmem_q_state + ((int)o_stage_tcgen * 16))), "r"(_mma_b_lo_1), "r"(tmem_tmem_state_inp), "r"(0));
                    elect_commit(decay_tcgen_done_addr + (decay_stage_tcgen) * 8);
                    {
                        mbarrier_wait(state_read_done_addr, state_phase_tcgen);
                    }
                    #pragma unroll
                    for (int diag_block_tcgen = 0; diag_block_tcgen < 8; diag_block_tcgen++) {
                        int _mma_b_lo_2 = make_warp_uniform((((smem_state_diag_addr) >> 4) & 0x3FFF) + ((int)diag_stage_tcgen * 8 + diag_block_tcgen) * 32);
                        mma_ts_step((tmem_tmem_state + (diag_block_tcgen * 16)), tmem_tmem_state_inp + diag_block_tcgen * 8, _mma_b_lo_2, 0xC0004010, 134481040, 0);
                    }
                    elect_commit(state_diag_done_addr + (diag_stage_tcgen) * 8);
                    mbarrier_wait(tinv_ready_addr + (intermediate_stage_tcgen) * 8, (unsigned int)(cumulative_tcgen / 2 & 1));
                    mbarrier_wait(y_inp_ready_addr, state_phase_tcgen);
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_tinv_mn_addr) >> 4) & 0x3FFF) | 0x200000) + (intermediate_stage_tcgen) * 64);
                    mma_ts_step(tmem_tmem_u_acc, tmem_tmem_y_inp, _mma_b_lo_3, 0xC0004010, 134546576, 0);
                    elect_commit2(tinv_done_addr + (intermediate_stage_tcgen) * 8, u_acc_ready_addr);
                    mbarrier_wait(u_inp_ready_addr, state_phase_tcgen);
                    int _mma_b_lo_4 = make_warp_uniform(((((smem_k_restore_mn_addr) >> 4) & 0x3FFF) | 0x800000) + (decay_stage_tcgen) * 256);
                    mma_ts_step(tmem_tmem_state, tmem_tmem_u_inp, _mma_b_lo_4, 0x40004040, 136381584, 1);
                    elect_commit2(k_restore_done_addr + (decay_stage_tcgen) * 8, state_acc_done_addr);
                    mbarrier_wait(a_ready_addr + (intermediate_stage_tcgen) * 8, (unsigned int)(cumulative_tcgen / 2 & 1));
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_a_mn_addr) >> 4) & 0x3FFF) | 0x200000) + (intermediate_stage_tcgen) * 64);
                    mma_ts_step((tmem_tmem_q_state + ((int)o_stage_tcgen * 16)), tmem_tmem_u_inp, _mma_b_lo_5, 0xC0004010, 134546576, 1);
                    elect_commit2(o_acc_ready_addr, a_done_addr + (intermediate_stage_tcgen) * 8);
                }
                cumulative_chunk_tcgen += chunks_tcgen;
            }
            if (elect_sync()) {
                mbarrier_arrive(consumers_done_addr);
            }
            unsigned int _phase_cleanup_ready_0 = 0;
            mbarrier_wait(cleanup_ready_addr, _phase_cleanup_ready_0);
            _phase_cleanup_ready_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: tma ----
    if (warp == 14) {
        { // tma_main
            unsigned int sched_stage_tma = 0;
            int cumulative_chunk_tma = 0;
            unsigned int _phase_sched_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int sched_iter_tma = 0; sched_iter_tma < total_work_items + 1; sched_iter_tma++) {
                    mbarrier_wait(sched_done_addr + (sched_stage_tma) * 8, _phase_sched_done);
                    unsigned int tile_tma = blockIdx.x;
                    if (uniform_work_items != 0) {
                        tile_tma = (unsigned int)blockIdx.x + (unsigned int)sched_iter_tma * (unsigned int)gridDim.x;
                    } else if (sched_iter_tma > 0) {
                        unsigned int _atomic_old_0 = atomicAdd(dynamic_counter, 1);
                        tile_tma = (unsigned int)gridDim.x + _atomic_old_0;
                    }
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sched_slot_addr + sched_stage_tma * 4), "r"(tile_tma));
                    mbarrier_arrive(sched_ready_addr + (sched_stage_tma) * 8);
                    sched_stage_tma += 1;
                    if (sched_stage_tma == 8) { sched_stage_tma = 0; _phase_sched_done ^= 1; }
                    if (tile_tma >= (unsigned int)total_work_items) {
                        break;
                    }
                    int item_base_tma = (int)tile_tma * 8;
                    int _vec_load_0[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(work_items + item_base_tma);
                        _vec_load_0[0 + 0] = _iv4.x;
                        _vec_load_0[0 + 1] = _iv4.y;
                        _vec_load_0[0 + 2] = _iv4.z;
                        _vec_load_0[0 + 3] = _iv4.w;
                    }
                    int _vec_load_1[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(work_items + item_base_tma + 4);
                        _vec_load_1[0 + 0] = _iv4.x;
                        _vec_load_1[0 + 1] = _iv4.y;
                        _vec_load_1[0 + 2] = _iv4.z;
                        _vec_load_1[0 + 3] = _iv4.w;
                    }
                    int head_tma = _vec_load_0[1];
                    int wend_tma = _vec_load_0[3];
                    int cstart_tma = _vec_load_1[0];
                    long long bos_tma = (long long)_vec_load_1[2];
                    int chunks_tma = wend_tma - cstart_tma;
                    #pragma unroll 1
                    for (int chunk_tma = 0; chunk_tma < chunks_tma; chunk_tma++) {
                        int cumulative_tma = cumulative_chunk_tma + chunk_tma;
                        unsigned int raw_stage_tma = (unsigned int)(cumulative_tma % 5);
                        unsigned int raw_bar_stage_tma = (unsigned int)(cumulative_tma % 6);
                        unsigned int raw_done_phase_tma = (unsigned int)(cumulative_tma / 5 + 1 & 1);
                        mbarrier_wait(q_done_addr + (raw_stage_tma) * 8, raw_done_phase_tma);
                        mbarrier_wait(k_done_addr + (raw_stage_tma) * 8, raw_done_phase_tma);
                        mbarrier_wait(v_done_addr + (raw_stage_tma) * 8, raw_done_phase_tma);
                        mbarrier_wait(g_done_addr + (raw_stage_tma) * 8, raw_done_phase_tma);
                        mbarrier_arrive_expect_tx(q_ready_addr + (raw_bar_stage_tma) * 8, 4096);
                        mbarrier_arrive_expect_tx(k_ready_addr + (raw_bar_stage_tma) * 8, 4096);
                        mbarrier_arrive_expect_tx(v_ready_addr + (raw_bar_stage_tma) * 8, 4096);
                        {
                            mbarrier_arrive(g_ready_addr + (raw_bar_stage_tma) * 8);
                        }
                        int logical_chunk_tma = cstart_tma + chunk_tma;
                        int token_tma = (int)(bos_tma + (long long)logical_chunk_tma * 16);
                        #pragma unroll
                        for (int segment_tma = 0; segment_tma < 2; segment_tma++) {
                            int segment_offset_tma = segment_tma * 16 * 64 * 2;
                            int segment_dim_tma = segment_tma * 64;
                            tma_3d_gmem2smem(smem_q_addr + raw_stage_tma * 4096 + (unsigned int)segment_offset_tma, q_tma, segment_dim_tma, head_tma, token_tma, q_ready_addr + (raw_bar_stage_tma) * 8);
                            tma_3d_gmem2smem(smem_k_addr + raw_stage_tma * 4096 + (unsigned int)segment_offset_tma, k_tma, segment_dim_tma, head_tma, token_tma, k_ready_addr + (raw_bar_stage_tma) * 8);
                            tma_3d_gmem2smem(smem_v_addr + raw_stage_tma * 4096 + (unsigned int)segment_offset_tma, v_tma, segment_dim_tma, head_tma, token_tma, v_ready_addr + (raw_bar_stage_tma) * 8);
                        }
                    }
                    cumulative_chunk_tma += chunks_tma;
                }
            }
            unsigned int _phase_consumers_done_0 = 0;
            mbarrier_wait(consumers_done_addr, _phase_consumers_done_0);
            _phase_consumers_done_0 ^= 1;
            if (elect_sync()) {
                mbarrier_arrive(cleanup_ready_addr);
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp == 15) {
        { // epilogue_main
            unsigned int sched_stage_epi = 0;
            int cumulative_chunk_epi = 0;
            int lhs_row_epi = lane % 8 + (lane / 8 & 1) * 8;
            int lhs_col_epi = lane / 16 * 8;
            int rhs_row_epi = lane % 8 + lane / 16 * 8;
            int rhs_col_epi = (lane / 8 & 1) * 8;
            unsigned int _phase_sched_ready_4 = 0;
            #pragma unroll 1
            for (int __4 = 0; __4 < total_work_items + 1; __4++) {
                mbarrier_wait(sched_ready_addr + (sched_stage_epi) * 8, _phase_sched_ready_4);
                unsigned int slot_4[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&slot_4[0])) : "r"(sched_slot_addr + sched_stage_epi * 4));
                unsigned int tile_epi = slot_4[0];
                if (elect_sync()) {
                    mbarrier_arrive(sched_done_addr + (sched_stage_epi) * 8);
                }
                sched_stage_epi += 1;
                if (sched_stage_epi == 8) { sched_stage_epi = 0; _phase_sched_ready_4 ^= 1; }
                if (tile_epi >= (unsigned int)total_work_items) {
                    break;
                }
                int item_base_epi = (int)tile_epi * 8;
                int _vec_load_3[4];
                {
                    int4 _iv4 = *reinterpret_cast<const int4*>(work_items + item_base_epi);
                    _vec_load_3[0 + 0] = _iv4.x;
                    _vec_load_3[0 + 1] = _iv4.y;
                    _vec_load_3[0 + 2] = _iv4.z;
                    _vec_load_3[0 + 3] = _iv4.w;
                }
                int seq_epi = _vec_load_3[0];
                int head_epi = _vec_load_3[1];
                int wstart_epi = _vec_load_3[2];
                int wend_epi = _vec_load_3[3];
                int cstart_epi = work_items[item_base_epi + 4];
                long long bos_epi = (long long)work_items[item_base_epi + 6];
                int chunks_epi = wend_epi - cstart_epi;
                long long checkpoint_base_epi = checkpoint_cu_starts[seq_epi];
                if (ENABLE_CHECKPOINTS != 0 && chunks_epi > 0 && wstart_epi == 0) {
                    int checkpoint_event_epi = cumulative_chunk_epi;
                    unsigned int checkpoint_stage_epi = (unsigned int)(checkpoint_event_epi % 2);
                    mbarrier_wait(checkpoint_ready_addr + (checkpoint_stage_epi) * 8, (unsigned int)(checkpoint_event_epi / 2 & 1));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int segment_checkpoint_epi = 0; segment_checkpoint_epi < 2; segment_checkpoint_epi++) {
                            tma_store_4d(checkpoint_tma, segment_checkpoint_epi * 64, 0, head_epi, checkpoint_base_epi, smem_checkpoint_addr + checkpoint_stage_epi * 32768 + (unsigned int)(segment_checkpoint_epi * 128 * 64 * 2));
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    mbarrier_arrive(checkpoint_done_addr + (checkpoint_stage_epi) * 8);
                }
                #pragma unroll 1
                for (int chunk_epi = 0; chunk_epi < chunks_epi; chunk_epi++) {
                    int cumulative_epi = cumulative_chunk_epi + chunk_epi;
                    int logical_chunk_epi = cstart_epi + chunk_epi;
                    unsigned int decay_stage_epi = (unsigned int)(cumulative_epi % 2);
                    unsigned int diag_stage_epi = (unsigned int)(cumulative_epi % 4);
                    unsigned int intermediate_stage_epi = (unsigned int)(cumulative_epi % 2);
                    mbarrier_wait(qk_scale_ready_addr + (diag_stage_epi) * 8, (unsigned int)(cumulative_epi / 4 & 1));
                    mbarrier_wait(a_done_addr + (intermediate_stage_epi) * 8, (unsigned int)(cumulative_epi / 2 + 1 & 1));
                    float a_acc_epi[8];
                    #pragma unroll
                    for (int k_block_epi = 0; k_block_epi < 8; k_block_epi++) {
                        unsigned int a_frag_epi[4];
                        unsigned int b_frag_epi[4];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_epi[0]), "=r"(a_frag_epi[1]), "=r"(a_frag_epi[2]), "=r"(a_frag_epi[3])
                            : "r"((smem_q_decay_addr + decay_stage_epi * 4096 + (unsigned int)((k_block_epi * 16 + lhs_col_epi) / 64 * 2048 + lhs_row_epi * 128 + (k_block_epi * 16 + lhs_col_epi) % 64 * 2 ^ ((k_block_epi * 16 + lhs_col_epi) / 64 * 2048 + lhs_row_epi * 128 + (k_block_epi * 16 + lhs_col_epi) % 64 * 2 >> 7 & 7) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_epi[0]), "=r"(b_frag_epi[1]), "=r"(b_frag_epi[2]), "=r"(b_frag_epi[3])
                            : "r"((smem_k_inv_addr + decay_stage_epi * 4096 + (unsigned int)((k_block_epi * 16 + rhs_col_epi) / 64 * 2048 + rhs_row_epi * 128 + (k_block_epi * 16 + rhs_col_epi) % 64 * 2 ^ ((k_block_epi * 16 + rhs_col_epi) / 64 * 2048 + rhs_row_epi * 128 + (k_block_epi * 16 + rhs_col_epi) % 64 * 2 >> 7 & 7) << 4)))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(a_acc_epi[0]), "=f"(a_acc_epi[1]), "=f"(a_acc_epi[2]), "=f"(a_acc_epi[3])
                            : "r"(a_frag_epi[0]), "r"(a_frag_epi[1]), "r"(a_frag_epi[2]), "r"(a_frag_epi[3]), "r"(b_frag_epi[0]), "r"(b_frag_epi[1]), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[0])), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[1])), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[2])), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[3])));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(a_acc_epi[4]), "=f"(a_acc_epi[(4) + 1]), "=f"(a_acc_epi[(4) + 2]), "=f"(a_acc_epi[(4) + 3])
                            : "r"(a_frag_epi[0]), "r"(a_frag_epi[1]), "r"(a_frag_epi[2]), "r"(a_frag_epi[3]), "r"(b_frag_epi[2]), "r"(b_frag_epi[(2) + 1]), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[4])), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[(4) + 1])), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[(4) + 2])), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[(4) + 3])));
                    }
                    float a_values_epi[8];
                    #pragma unroll
                    for (int accum_epi = 0; accum_epi < 8; accum_epi++) {
                        int row_epi = lane / 4 + accum_epi % 4 / 2 * 8;
                        int col_epi = accum_epi / 4 * 8 + (lane & 3) * 2 + (accum_epi & 1);
                        a_values_epi[accum_epi] = 0.0f;
                        if (row_epi >= col_epi) {
                            a_values_epi[accum_epi] = a_acc_epi[accum_epi];
                        }
                    }
                    unsigned int a_words_epi[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(a_values_epi[_lp*2 + 0], a_values_epi[_lp*2+1 + 0]));
                        a_words_epi[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(smem_a_addr + intermediate_stage_epi * 1024 + (unsigned int)(lane / 16 * 8 / 16 * 512 + lane % 16 * 32 + lane / 16 * 8 % 16 * 2 ^ (lane / 16 * 8 / 16 * 512 + lane % 16 * 32 + lane / 16 * 8 % 16 * 2 >> 7 & 1) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&a_words_epi[0])), "r"(*reinterpret_cast<const uint32_t*>(&a_words_epi[1])), "r"(*reinterpret_cast<const uint32_t*>(&a_words_epi[2])), "r"(*reinterpret_cast<const uint32_t*>(&a_words_epi[3]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(a_ready_addr + (intermediate_stage_epi) * 8);
                    mbarrier_arrive(decay_super_done_addr + (decay_stage_epi) * 8);
                    if (chunk_epi > 0) {
                        {
                            int checkpoint_event_loop_epi = cumulative_epi;
                            unsigned int checkpoint_stage_loop_epi = (unsigned int)(checkpoint_event_loop_epi % 2);
                            mbarrier_wait(checkpoint_ready_addr + (checkpoint_stage_loop_epi) * 8, (unsigned int)(checkpoint_event_loop_epi / 2 & 1));
                            if (elect_sync()) {
                                #pragma unroll
                                for (int checkpoint_segment_loop_epi = 0; checkpoint_segment_loop_epi < 2; checkpoint_segment_loop_epi++) {
                                    tma_store_4d(checkpoint_tma, checkpoint_segment_loop_epi * 64, 0, head_epi, checkpoint_base_epi + (long long)logical_chunk_epi, smem_checkpoint_addr + checkpoint_stage_loop_epi * 32768 + (unsigned int)(checkpoint_segment_loop_epi * 128 * 64 * 2));
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            mbarrier_arrive(checkpoint_done_addr + (checkpoint_stage_loop_epi) * 8);
                        }
                        int output_event_epi = cumulative_epi - 1;
                        unsigned int output_stage_epi = (unsigned int)(output_event_epi % 2);
                        mbarrier_wait(o_tma_ready_addr + (output_stage_epi) * 8, (unsigned int)(output_event_epi / 2 & 1));
                        if (elect_sync()) {
                            #pragma unroll
                            for (int output_segment_epi = 0; output_segment_epi < 2; output_segment_epi++) {
                                tma_store_3d(out_tma, output_segment_epi * 64, head_epi, (int)(bos_epi + (long long)(logical_chunk_epi - 1) * 16), smem_o_addr + output_stage_epi * 4096 + (unsigned int)(output_segment_epi * 16 * 64 * 2));
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive(o_tma_done_addr + (output_stage_epi) * 8);
                    }
                }
                if (chunks_epi > 0) {
                    int last_event_epi = cumulative_chunk_epi + chunks_epi - 1;
                    unsigned int last_o_stage_epi = (unsigned int)(last_event_epi % 2);
                    mbarrier_wait(o_tma_ready_addr + (last_o_stage_epi) * 8, (unsigned int)(last_event_epi / 2 & 1));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int last_segment_epi = 0; last_segment_epi < 2; last_segment_epi++) {
                            tma_store_3d(out_tma, last_segment_epi * 64, head_epi, (int)(bos_epi + (long long)(wend_epi - 1) * 16), smem_o_addr + last_o_stage_epi * 4096 + (unsigned int)(last_segment_epi * 16 * 64 * 2));
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    mbarrier_arrive(o_tma_done_addr + (last_o_stage_epi) * 8);
                }
                cumulative_chunk_epi += chunks_epi;
            }
            if (elect_sync()) {
                mbarrier_arrive(consumers_done_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"

// clang-format on
