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

// Frozen generated kernel export; do not edit by hand.
// Generated schedule 'flashkda_bf16_small_bh_m128'; module
// flashkda_bf16_small_bh_m128_73369168de.
// clang-format off

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
#define TMEM_NCOLS 256
#define TMEM_TMEM_STATE_OFFSET 64
#define TMEM_TMEM_STATE_INP_OFFSET 0
#define TMEM_TMEM_U_ACC_OFFSET 224
#define TMEM_TMEM_U2_INP_OFFSET 224
#define TMEM_TMEM_U2_ACC_OFFSET 0
#define TMEM_TMEM_OUT_OFFSET 192
#define TMEM_TMEM_STATE_OUT_OFFSET 64
#define NUM_CHUNK_PIPE_STAGES 5
#define SMEM_SMEM_PACKET_OFF 1024
#define SMEM_SMEM_PACKET_STAGE_BYTES 31488
#define SMEM_SMEM_PACKET_STRIDE 41984
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 8192
#define SMEM_SMEM_QD_STRIDE 41984
#define SMEM_SMEM_G_RAW_OFF 1024
#define SMEM_SMEM_G_RAW_STAGE_BYTES 8192
#define SMEM_SMEM_G_RAW_STRIDE 41984
#define SMEM_SMEM_G_RAW_ALL_OFF 1024
#define SMEM_SMEM_G_RAW_ALL_STAGE_BYTES 176128
#define SMEM_SMEM_G_RAW_ALL_STRIDE 176128
#define SMEM_SMEM_KD_OFF 9216
#define SMEM_SMEM_KD_STAGE_BYTES 8192
#define SMEM_SMEM_KD_STRIDE 41984
#define SMEM_SMEM_Q_RAW_PREFETCH_OFF 17408
#define SMEM_SMEM_Q_RAW_PREFETCH_STAGE_BYTES 8192
#define SMEM_SMEM_Q_RAW_PREFETCH_STRIDE 41984
#define SMEM_SMEM_FINAL_TRANS_OFF 17408
#define SMEM_SMEM_FINAL_TRANS_STAGE_BYTES 12288
#define SMEM_SMEM_FINAL_TRANS_STRIDE 41984
#define SMEM_SMEM_KR_TRANS_OFF 17408
#define SMEM_SMEM_KR_TRANS_STAGE_BYTES 8192
#define SMEM_SMEM_KR_TRANS_STRIDE 41984
#define SMEM_SMEM_MQK_TRANS_OFF 25600
#define SMEM_SMEM_MQK_TRANS_STAGE_BYTES 2048
#define SMEM_SMEM_MQK_TRANS_STRIDE 41984
#define SMEM_SMEM_INV_OFF 29696
#define SMEM_SMEM_INV_STAGE_BYTES 2048
#define SMEM_SMEM_INV_STRIDE 41984
#define SMEM_SMEM_V_OFF 32512
#define SMEM_SMEM_V_STAGE_BYTES 8192
#define SMEM_SMEM_V_STRIDE 41984
#define SMEM_SMEM_KI_OFF 17408
#define SMEM_SMEM_KI_STAGE_BYTES 8192
#define SMEM_SMEM_KI_STRIDE 41984
#define SMEM_SMEM_GATE_OFF 25600
#define SMEM_SMEM_GATE_STAGE_BYTES 16384
#define SMEM_SMEM_GATE_STRIDE 41984
#define SMEM_SMEM_BETA_RAW_OFF 41984
#define SMEM_SMEM_BETA_RAW_STAGE_BYTES 512
#define SMEM_SMEM_BETA_RAW_STRIDE 41984
#define SMEM_SMEM_INV_WORK_OFF 32512
#define SMEM_SMEM_INV_WORK_STAGE_BYTES 4096
#define SMEM_SMEM_INV_WORK_STRIDE 41984
#define SMEM_SMEM_OUT_OFF 210944
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_RESTORE_FACTOR_ALL_OFF 41984
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES 168452
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STRIDE 168452
#define SMEM_SMEM_GT_PREFIX_ALL_OFF 41472
#define SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES 168448
#define SMEM_SMEM_GT_PREFIX_ALL_STRIDE 168448
#define SMEM_SMEM_GT_ALL_OFF 31744
#define SMEM_SMEM_GT_ALL_STAGE_BYTES 168448
#define SMEM_SMEM_GT_ALL_STRIDE 168448
#define SMEM_SMEM_BETA_ALL_OFF 32256
#define SMEM_SMEM_BETA_ALL_STAGE_BYTES 168064
#define SMEM_SMEM_BETA_ALL_STRIDE 168064
#define SMEM_SMEM_PREP_BETA_ALL_OFF 42500
#define SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES 168064
#define SMEM_SMEM_PREP_BETA_ALL_STRIDE 168064
#define SMEM_SMEM_GATE_RATE_ALL_OFF 42628
#define SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES 167940
#define SMEM_SMEM_GATE_RATE_ALL_STRIDE 167940
#define SMEM_SMEM_V_ALL_OFF 32512
#define SMEM_SMEM_V_ALL_STAGE_BYTES 176128
#define SMEM_SMEM_V_ALL_STRIDE 176128
#define SMEM_SMEM_GATE_ALL_OFF 25600
#define SMEM_SMEM_GATE_ALL_STAGE_BYTES 184320
#define SMEM_SMEM_GATE_ALL_STRIDE 184320
#define SMEM_TOTAL 227328
#define THREADS 1024

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


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
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

__global__ __launch_bounds__(1024) void
// FLASHINFER INTEGRATION BEGIN: allow exact state alias
kernel_flashkda_bf16_small_bh_m128(__nv_bfloat16* __restrict__ q, FlashKDATensorMap const* q_tma, __nv_bfloat16* __restrict__ k, FlashKDATensorMap const* k_tma, __nv_bfloat16* __restrict__ v, FlashKDATensorMap const* v_tma, __nv_bfloat16* __restrict__ g, FlashKDATensorMap const* g_tma, __nv_bfloat16* __restrict__ beta, FlashKDATensorMap const* beta_tma, float* __restrict__ A_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ seq_order, __nv_bfloat16* initial_state, __nv_bfloat16* __restrict__ out, FlashKDATensorMap const* out_tma, __nv_bfloat16* final_state, int num_heads, int use_initial_state, int store_final_state, float scale, float lower_bound, unsigned long long state_indices_addr, unsigned long long state_checkpoints_addr, unsigned long long checkpoint_cu_starts_addr, long long beta_token_stride, long long state_slot_stride, int use_state_indices, int checkpoint_every_n_tokens, FlashKDATensorMap const* packet_workspace, unsigned int* __restrict__ packet_ready, unsigned int* __restrict__ packet_consumed, unsigned int* __restrict__ helper_done)
// FLASHINFER INTEGRATION END: allow exact state alias
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
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(packet_workspace)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_packet = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_packet_addr = smem + 1024;
    __nv_bfloat16* smem_qd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qd_addr = smem + 1024;
    __nv_bfloat16* smem_g_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_g_raw_addr = smem + 1024;
    __nv_bfloat16* smem_g_raw_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_g_raw_all_addr = smem + 1024;
    __nv_bfloat16* smem_kd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kd_addr = smem + 9216;
    __nv_bfloat16* smem_q_raw_prefetch = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_q_raw_prefetch_addr = smem + 17408;
    __nv_bfloat16* smem_final_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_final_trans_addr = smem + 17408;
    __nv_bfloat16* smem_kr_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_kr_trans_addr = smem + 17408;
    __nv_bfloat16* smem_mqk_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int smem_mqk_trans_addr = smem + 25600;
    __nv_bfloat16* smem_inv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 29696);
    const int smem_inv_addr = smem + 29696;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32512);
    const int smem_v_addr = smem + 32512;
    __nv_bfloat16* smem_ki = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_ki_addr = smem + 17408;
    float* smem_gate = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_gate_addr = smem + 25600;
    __nv_bfloat16* smem_beta_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_beta_raw_addr = smem + 41984;
    __nv_bfloat16* smem_inv_work = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32512);
    const int smem_inv_work_addr = smem + 32512;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 210944);
    const int smem_out_addr = smem + 210944;
    float* smem_restore_factor_all = reinterpret_cast<float*>(smem_raw + 41984);
    const int smem_restore_factor_all_addr = smem + 41984;
    float* smem_gt_prefix_all = reinterpret_cast<float*>(smem_raw + 41472);
    const int smem_gt_prefix_all_addr = smem + 41472;
    float* smem_gt_all = reinterpret_cast<float*>(smem_raw + 31744);
    const int smem_gt_all_addr = smem + 31744;
    float* smem_beta_all = reinterpret_cast<float*>(smem_raw + 32256);
    const int smem_beta_all_addr = smem + 32256;
    float* smem_prep_beta_all = reinterpret_cast<float*>(smem_raw + 42500);
    const int smem_prep_beta_all_addr = smem + 42500;
    float* smem_gate_rate_all = reinterpret_cast<float*>(smem_raw + 42628);
    const int smem_gate_rate_all_addr = smem + 42628;
    __nv_bfloat16* smem_v_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32512);
    const int smem_v_all_addr = smem + 32512;
    float* smem_gate_all = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_gate_all_addr = smem + 25600;

    // Mbarrier init (18 groups, 82 barriers)
    // Mbarriers at smem_raw[0..656)

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
            // gate_raw_full: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // qk_raw_full: 5 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // packet_full: 5 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // v_full: 5 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // v_free: 5 barriers, init_count=4
            mbarrier_init(smem + 200, 4);
            mbarrier_init(smem + 208, 4);
            mbarrier_init(smem + 216, 4);
            mbarrier_init(smem + 224, 4);
            mbarrier_init(smem + 232, 4);
            // smem_free: 5 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // raw_inputs_free: 5 barriers, init_count=1
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // state_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 320, 4);
            mbarrier_init(smem + 328, 4);
            mbarrier_init(smem + 336, 4);
            mbarrier_init(smem + 344, 4);
            mbarrier_init(smem + 352, 4);
            // old_out_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            mbarrier_init(smem + 384, 1);
            mbarrier_init(smem + 392, 1);
            // u_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 400, 4);
            mbarrier_init(smem + 408, 4);
            mbarrier_init(smem + 416, 4);
            mbarrier_init(smem + 424, 4);
            mbarrier_init(smem + 432, 4);
            // u2_acc_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 440, 1);
            mbarrier_init(smem + 448, 1);
            mbarrier_init(smem + 456, 1);
            mbarrier_init(smem + 464, 1);
            mbarrier_init(smem + 472, 1);
            // u2_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 480, 4);
            mbarrier_init(smem + 488, 4);
            mbarrier_init(smem + 496, 4);
            mbarrier_init(smem + 504, 4);
            mbarrier_init(smem + 512, 4);
            // final_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 520, 1);
            mbarrier_init(smem + 528, 1);
            mbarrier_init(smem + 536, 1);
            mbarrier_init(smem + 544, 1);
            mbarrier_init(smem + 552, 1);
            // out_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 560, 1);
            // tmem_dealloc_ready: 1 barriers, init_count=2
            mbarrier_init(smem + 568, 2);
            // prep_diag_ready: 5 barriers, init_count=2
            mbarrier_init(smem + 576, 2);
            mbarrier_init(smem + 584, 2);
            mbarrier_init(smem + 592, 2);
            mbarrier_init(smem + 600, 2);
            mbarrier_init(smem + 608, 2);
            // prep_inv16_ready: 5 barriers, init_count=2
            mbarrier_init(smem + 616, 2);
            mbarrier_init(smem + 624, 2);
            mbarrier_init(smem + 632, 2);
            mbarrier_init(smem + 640, 2);
            mbarrier_init(smem + 648, 2);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 656);
    if (warp == 0) {
        int _tmem_hold = smem + 656;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define qk_full_addr (mbar_base + 0)
    #define gate_raw_full_addr (mbar_base + 40)
    #define qk_raw_full_addr (mbar_base + 80)
    #define packet_full_addr (mbar_base + 120)
    #define v_full_addr (mbar_base + 160)
    #define v_free_addr (mbar_base + 200)
    #define smem_free_addr (mbar_base + 240)
    #define raw_inputs_free_addr (mbar_base + 280)
    #define state_inp_ready_addr (mbar_base + 320)
    #define old_out_ready_addr (mbar_base + 360)
    #define u_inp_ready_addr (mbar_base + 400)
    #define u2_acc_ready_addr (mbar_base + 440)
    #define u2_inp_ready_addr (mbar_base + 480)
    #define final_ready_addr (mbar_base + 520)
    #define out_empty_addr (mbar_base + 560)
    #define tmem_dealloc_ready_addr (mbar_base + 568)
    #define prep_diag_ready_addr (mbar_base + 576)
    #define prep_inv16_ready_addr (mbar_base + 616)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state = taddr + 64;
    const int tmem_tmem_state_inp = taddr;
    const int tmem_tmem_u_acc = taddr + 224;
    const int tmem_tmem_u2_inp = taddr + 224;
    const int tmem_tmem_u2_acc = taddr;
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
            int group_rank = blockIdx.x % 8;
            int task_idx = blockIdx.x / 8;
            int seq_idx = seq_order[task_idx / num_heads];
            int head_idx = task_idx % num_heads;
            long long bos = cu_seqlens[seq_idx];
            long long eos = cu_seqlens[seq_idx + 1];
            int seq_len = (int)(eos - bos);
            int num_chunks = (seq_len + 32 - 1) / 32;
            if (group_rank != 0) {
                num_chunks = 0;
            }
            int warp_in_wg = warp % 4;
            const int tmem_row_base = warp_in_wg * 32 << 16;
            int state_row = warp_in_wg * 32 + lane;
            int warp_id_in_role = (warp - 0);
            int compute_local_warp = warp_id_in_role;
            long long state_base = (((long long)seq_idx * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128;
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
                tmem_st_x32_f32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block * 32), state_frag);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int compute_stage = 0;
            unsigned int _phase_qk_full = 0;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_old_out_ready = 0;
            unsigned int _phase_u2_acc_ready = 0;
            unsigned int _phase_final_ready = 0;
            #pragma unroll 1
            for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                mbarrier_wait(qk_full_addr + (compute_stage) * 8, _phase_qk_full);
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
                    float state_scale[16];
                    #pragma unroll
                    for (int state_half = 0; state_half < 2; state_half++) {
                        #pragma unroll
                        for (int state_col = 0; state_col < 16; state_col++) {
                            state_scale[state_col] = smem_gt_all[compute_stage * 10496 + (unsigned int)(state_col_block_1 * 32) + (unsigned int)(state_half * 16) + (unsigned int)state_col];
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + state_half * 16))[_ls], reinterpret_cast<const float2*>(state_scale)[_ls]);
                    }
                    tmem_st_x32_f32(state_addr, _tmem_load_1);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(state_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(v_full_addr + (compute_stage) * 8, _phase_v_full);
                mbarrier_wait(old_out_ready_addr + (compute_stage) * 8, _phase_old_out_ready);
                float _tmem_load_2[32];
                tmem_ld_x32(&_tmem_load_2[0], taddr + 224 + (unsigned int)tmem_row_base);
                #pragma unroll
                for (int residual_half = 0; residual_half < 2; residual_half++) {
                    float residual_v[16];
                    float residual_beta[16];
                    #pragma unroll
                    for (int residual_col = 0; residual_col < 16; residual_col++) {
                        int token_col = residual_half * 16 + residual_col;
                        __nv_bfloat16 v_value = smem_v_all[compute_stage * 20992 + (unsigned int)(token_col * 128) + (unsigned int)state_row];
                        float _cvt_f32_2 = __bfloat162float(v_value);
                        residual_v[residual_col] = _cvt_f32_2;
                        residual_beta[residual_col] = smem_beta_all[compute_stage * 10496 + (unsigned int)token_col];
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        sub_f32x2_inplace(&reinterpret_cast<float2*>(residual_v)[_ls], reinterpret_cast<const float2*>((_tmem_load_2 + residual_half * 16))[_ls]);
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(residual_v)[_ls], reinterpret_cast<const float2*>(residual_beta)[_ls]);
                    uint32_t residual_v_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 0], residual_v[_lp*2+1 + 0]));
                        residual_v_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 224 + (unsigned int)tmem_row_base + (unsigned int)(residual_half * 8), (const uint32_t*)residual_v_bf16);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(v_free_addr + (compute_stage) * 8);
                    mbarrier_arrive(u_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(u2_acc_ready_addr + (compute_stage) * 8, _phase_u2_acc_ready);
                float _tmem_load_3[32];
                tmem_ld_x32(&_tmem_load_3[0], taddr + (unsigned int)tmem_row_base);
                uint32_t _tmem_load_3_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + 224 + (unsigned int)tmem_row_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[15]))
                    : "memory");
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(u2_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(final_ready_addr + (compute_stage) * 8, _phase_final_ready);
                compute_stage += 1;
                if (compute_stage == 5) { compute_stage = 0; _phase_qk_full ^= 1; _phase_v_full ^= 1; _phase_old_out_ready ^= 1; _phase_u2_acc_ready ^= 1; _phase_final_ready ^= 1; }
            }
            if (group_rank == 0 && store_final_state != 0) {
                #pragma unroll
                for (int state_col_block_2 = 0; state_col_block_2 < 4; state_col_block_2++) {
                    float _tmem_load_5[32];
                    tmem_ld_x32(&_tmem_load_5[0], taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_2 * 32));
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(_tmem_load_5[0 + 0], _tmem_load_5[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(_tmem_load_5[0 + 2], _tmem_load_5[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(_tmem_load_5[0 + 4], _tmem_load_5[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(_tmem_load_5[0 + 6], _tmem_load_5[0 + 7]);
                        _pk[4] = __floats2bfloat162_rn(_tmem_load_5[0 + 8], _tmem_load_5[0 + 9]);
                        _pk[5] = __floats2bfloat162_rn(_tmem_load_5[0 + 10], _tmem_load_5[0 + 11]);
                        _pk[6] = __floats2bfloat162_rn(_tmem_load_5[0 + 12], _tmem_load_5[0 + 13]);
                        _pk[7] = __floats2bfloat162_rn(_tmem_load_5[0 + 14], _tmem_load_5[0 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base + (long long)(state_col_block_2 * 32))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base + (long long)(state_col_block_2 * 32))))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(_tmem_load_5[16 + 0], _tmem_load_5[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(_tmem_load_5[16 + 2], _tmem_load_5[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(_tmem_load_5[16 + 4], _tmem_load_5[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(_tmem_load_5[16 + 6], _tmem_load_5[16 + 7]);
                        _pk[4] = __floats2bfloat162_rn(_tmem_load_5[16 + 8], _tmem_load_5[16 + 9]);
                        _pk[5] = __floats2bfloat162_rn(_tmem_load_5[16 + 10], _tmem_load_5[16 + 11]);
                        _pk[6] = __floats2bfloat162_rn(_tmem_load_5[16 + 12], _tmem_load_5[16 + 13]);
                        _pk[7] = __floats2bfloat162_rn(_tmem_load_5[16 + 14], _tmem_load_5[16 + 15]);
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
            int group_rank_1 = blockIdx.x % 8;
            int task_idx_1 = blockIdx.x / 8;
            int seq_idx_1 = seq_order[task_idx_1 / num_heads];
            int head_idx_1 = task_idx_1 % num_heads;
            long long bos_1 = cu_seqlens[seq_idx_1];
            long long eos_1 = cu_seqlens[seq_idx_1 + 1];
            int seq_len_1 = (int)(eos_1 - bos_1);
            int num_chunks_1 = (seq_len_1 + 32 - 1) / 32;
            if (group_rank_1 != 0) {
                num_chunks_1 = 0;
            }
            int warp_id_in_role_1 = (warp - 4);
            int epilogue_local_warp = warp_id_in_role_1;
            int warp_in_wg_1 = warp % 4;
            const int tmem_row_base_1 = warp_in_wg_1 * 32 << 16;
            int state_row_1 = warp_in_wg_1 * 32 + lane;
            unsigned int epilogue_stage = 0;
            unsigned int output_stage = 0;
            unsigned int _phase_final_ready_1 = 0;
            #pragma unroll 1
            for (int chunk_idx_1 = 0; chunk_idx_1 < num_chunks_1; chunk_idx_1++) {
                mbarrier_wait(final_ready_addr + (epilogue_stage) * 8, _phase_final_ready_1);
                int chunk_is_full = ((seq_len_1 >= (chunk_idx_1 + 1) * 32) ? 1 : 0);
                if (chunk_is_full != 0) {
                    float _tmem_load_6[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1)
                        : "memory");
                    float _tmem_load_7[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
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
                    int out_stage_addr = smem_out_addr + output_stage * 8192;
                    #pragma unroll
                    for (int dim_half = 0; dim_half < 2; dim_half++) {
                        unsigned int out_packed[8];
                        if (dim_half == 0) {
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                                out_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                        } else {
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                                out_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                        }
                        #pragma unroll
                        for (int token_group = 0; token_group < 2; token_group++) {
                            int mtx_idx = lane / 8;
                            int row_addr = lane & 7;
                            int dim_base = epilogue_local_warp * 32 + dim_half * 16 + (mtx_idx & 1) * 8;
                            int token_base = token_group * 16 + mtx_idx / 2 * 8;
                            int token_addr = token_base + row_addr;
                            int token_pair = token_addr / 2;
                            int token_parity = token_addr & 1;
                            int raw_row = token_pair + dim_base / 64 * 16;
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
                            tma_store_4d(out_tma, 0, (int)(bos_1 + (long long)(chunk_idx_1 * 32)), head_idx_1, 0, smem_out_addr + output_stage * 8192);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    output_stage = output_stage ^ 1;
                } else {
                    float _tmem_load_8[32];
                    tmem_ld_x32(&_tmem_load_8[0], taddr + 192 + (unsigned int)tmem_row_base_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
                    #pragma unroll
                    for (int token_col_1 = 0; token_col_1 < 32; token_col_1++) {
                        long long out_token = bos_1 + (long long)(chunk_idx_1 * 32 + token_col_1);
                        if (out_token < eos_1) {
                            long long out_idx = (out_token * (long long)num_heads + (long long)head_idx_1) * 128 + (long long)state_row_1;
                            out[out_idx] = _tmem_load_8[token_col_1];
                        }
                    }
                }
                epilogue_stage += 1;
                if (epilogue_stage == 5) { epilogue_stage = 0; _phase_final_ready_1 ^= 1; }
            }
            if (epilogue_local_warp == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
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
            int group_rank_2 = blockIdx.x % 8;
            int task_idx_2 = blockIdx.x / 8;
            int seq_idx_2 = seq_order[task_idx_2 / num_heads];
            long long bos_2 = cu_seqlens[seq_idx_2];
            long long eos_2 = cu_seqlens[seq_idx_2 + 1];
            int seq_len_2 = (int)(eos_2 - bos_2);
            int num_chunks_2 = (seq_len_2 + 32 - 1) / 32;
            if (group_rank_2 != 0) {
                num_chunks_2 = 0;
            }
            unsigned int mma_stage = 0;
            unsigned int _phase_qk_full_1 = 0;
            unsigned int _phase_state_inp_ready = 0;
            unsigned int _phase_out_empty_0 = 1;
            unsigned int _phase_u_inp_ready = 0;
            unsigned int _phase_u2_inp_ready = 0;
            #pragma unroll 1
            for (int _chunk_idx = 0; _chunk_idx < num_chunks_2; _chunk_idx++) {
                mbarrier_wait(qk_full_addr + (mma_stage) * 8, _phase_qk_full_1);
                mbarrier_wait(state_inp_ready_addr + (mma_stage) * 8, _phase_state_inp_ready);
                mbarrier_wait(out_empty_addr, _phase_out_empty_0);
                _phase_out_empty_0 ^= 1;
                int _mma_b_lo_0 = make_warp_uniform((((smem_qd_addr) >> 4) & 0x3FFF) + (mma_stage) * 2624);
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
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_out), "r"(_mma_b_lo_0), "r"(tmem_tmem_state_inp), "r"(0));
                int _mma_b_lo_1 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + (mma_stage) * 2624);
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
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_u_acc), "r"(_mma_b_lo_1), "r"(tmem_tmem_state_inp), "r"(0));
                elect_commit2(old_out_ready_addr + (mma_stage) * 8, raw_inputs_free_addr + (mma_stage) * 8);
                mbarrier_wait(u_inp_ready_addr + (mma_stage) * 8, _phase_u_inp_ready);
                int _mma_b_lo_2 = make_warp_uniform((((smem_inv_addr) >> 4) & 0x3FFF) + (mma_stage) * 2624);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_u2_acc), "r"(_mma_b_lo_2), "r"(tmem_tmem_u2_inp), "r"(0));
                elect_commit(u2_acc_ready_addr + (mma_stage) * 8);
                mbarrier_wait(u2_inp_ready_addr + (mma_stage) * 8, _phase_u2_inp_ready);
                int _mma_b_lo_3 = make_warp_uniform(((((smem_final_trans_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_stage) * 2624);
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
                    "mov.b32 id, 136905872;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_state_out), "r"(_mma_b_lo_3), "r"(tmem_tmem_u2_inp), "r"(1));
                elect_commit2(final_ready_addr + (mma_stage) * 8, smem_free_addr + (mma_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 5) { mma_stage = 0; _phase_qk_full_1 ^= 1; _phase_state_inp_ready ^= 1; _phase_u_inp_ready ^= 1; _phase_u2_inp_ready ^= 1; }
            }
            unsigned int _phase_tmem_dealloc_ready_0 = 0;
            mbarrier_wait(tmem_dealloc_ready_addr, _phase_tmem_dealloc_ready_0);
            _phase_tmem_dealloc_ready_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    }
    // ---- Role: prep ----
    if (warp >= 12 && warp <= 31) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // prep_main
            int group_rank_3 = blockIdx.x % 8;
            int task_idx_3 = blockIdx.x / 8;
            int seq_idx_3 = seq_order[task_idx_3 / num_heads];
            int head_idx_2 = task_idx_3 % num_heads;
            long long bos_3 = cu_seqlens[seq_idx_3];
            long long eos_3 = cu_seqlens[seq_idx_3 + 1];
            int seq_len_3 = (int)(eos_3 - bos_3);
            int num_chunks_3 = (seq_len_3 + 32 - 1) / 32;
            int instance_id = (warp - 12) / 4;
            int prep_instance = instance_id;
            int warp_id_in_role_2 = (warp - 12);
            int prep_local_warp = warp_id_in_role_2 - prep_instance * 4;
            int prep_tid = prep_local_warp * 32 + lane;
            int prep_all_tid = warp_id_in_role_2 * 32 + lane;
            int helper_instance = (group_rank_3 - 1) * 5 + prep_instance;
            int helper_active = (group_rank_3 + 7 - 1) / 7;
            int num_prep_iters = helper_active * (num_chunks_3 + 35 - 1 - helper_instance) / 35;
            unsigned int prep_stage = (unsigned int)prep_instance;
            int gate_rate_stage_f32 = prep_instance * 10496;
            if (prep_tid == 0) {
                float _expf_0 = __expf(A_log[head_idx_2]);
                smem_gate_rate_all[gate_rate_stage_f32] = _expf_0;
            }
            asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
            unsigned int _phase_gate_raw_full = 0;
            unsigned int _phase_qk_raw_full = 0;
            unsigned int _phase_prep_diag_ready = 0;
            unsigned int _phase_prep_inv16_ready = 0;
            #pragma unroll 1
            for (int prep_iter = 0; prep_iter < num_prep_iters; prep_iter++) {
                int chunk_idx_2 = prep_iter * 35 + helper_instance;
                int packet_slot = task_idx_3 * 35 + helper_instance;
                int stage_f32 = prep_stage * 10496;
                int stage_bf16 = prep_stage * 20992;
                int chunk_is_full_1 = ((seq_len_3 >= (chunk_idx_2 + 1) * 32) ? 1 : 0);
                float early_beta_value = 0.0f;
                float early_gate0 = 0.0f;
                if (prep_iter != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            {
                                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(packet_consumed) + (packet_slot);
                                while (true) {
                                    unsigned int _gca_v;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                    if (_gca_v >= (unsigned int)(prep_iter)) break;
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                }
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(gate_raw_full_addr + (prep_stage) * 8, 8704);
                            tma_3d_gmem2smem(smem_g_raw_addr + prep_stage * 41984, g_tma, 0, head_idx_2, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), gate_raw_full_addr + (prep_stage) * 8);
                            tma_2d_gmem2smem(smem_beta_raw_addr + prep_stage * 41984, beta_tma, head_idx_2 / 8 * 8, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), gate_raw_full_addr + (prep_stage) * 8);
                            mbarrier_arrive_expect_tx(qk_raw_full_addr + (prep_stage) * 8, 16384);
                            tma_4d_gmem2smem(smem_kd_addr + prep_stage * 41984, k_tma, 0, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), head_idx_2, 0, qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                    mbarrier_wait(gate_raw_full_addr + (prep_stage) * 8, _phase_gate_raw_full);
                    if (prep_local_warp == 2 && lane < 32) {
                        unsigned int beta_raw_pair[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&beta_raw_pair[0])) : "r"(smem_beta_raw_addr + prep_stage * 41984 + (unsigned int)(lane * 16) + (unsigned int)(head_idx_2 % 8 / 2 * 4)));
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
                        if (head_idx_2 % 2 != 0) {
                            beta_logit = beta_raw_pair_f32[1];
                        }
                        float _tanh_approx_0;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(beta_logit * 0.5f));
                        early_beta_value = _tanh_approx_0 * 0.5f + 0.5f;
                    }
                    if (prep_tid < 128) {
                        float early_gate_rate = smem_gate_rate_all[stage_f32];
                        float early_gate_bias = dt_bias[head_idx_2 * 128 + prep_tid];
                        __nv_bfloat16 early_gate_raw = smem_g_raw_all[stage_bf16 + prep_tid];
                        float _cvt_f32_0 = __bfloat162float(early_gate_raw);
                        float early_gate_arg = early_gate_rate * (_cvt_f32_0 + early_gate_bias);
                        float _tanh_approx_1;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(early_gate_arg * 0.5f));
                        float early_gate_sigmoid = _tanh_approx_1 * 0.5f + 0.5f;
                        early_gate0 = lower_bound * 1.4426950408889634f * early_gate_sigmoid;
                    }
                }
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            tma_4d_gmem2smem(smem_q_raw_prefetch_addr + prep_stage * 41984, q_tma, 0, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), head_idx_2, 0, qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                }
                if (chunk_is_full_1 == 0) {
                    #pragma unroll
                    for (int gate_load_pass = 0; gate_load_pass < 4; gate_load_pass++) {
                        int gate_load_item = gate_load_pass * 128 + prep_tid;
                        int gate_load_row = gate_load_item / 16;
                        int gate_load_segment = gate_load_item % 16;
                        long long gate_load_token = bos_3 + (long long)(chunk_idx_2 * 32 + gate_load_row);
                        long long gate_load_base = (gate_load_token * (long long)num_heads + (long long)head_idx_2) * 128 + (long long)(gate_load_segment * 8);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(smem_g_raw_addr + prep_stage * 41984 + (unsigned int)(gate_load_item * 16)), "l"(g + gate_load_base), "r"((gate_load_token < eos_3) ? 16 : 0));
                    }
                }
                if (chunk_is_full_1 == 0) {
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                }
                if (prep_local_warp == 2 && lane < 32) {
                    float beta_value = early_beta_value;
                    if (chunk_is_full_1 == 0) {
                        long long beta_token = bos_3 + (long long)(chunk_idx_2 * 32 + lane);
                        if (beta_token < eos_3) {
                            float beta_logit_1 = (float)beta[beta_token * (long long)num_heads + (long long)head_idx_2];
                            float _tanh_approx_2;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(beta_logit_1 * 0.5f));
                            beta_value = _tanh_approx_2 * 0.5f + 0.5f;
                        }
                    }
                    smem_prep_beta_all[stage_f32 + lane] = beta_value;
                }
                if (prep_tid < 128) {
                    int gate_col = prep_tid;
                    float gate_rate = smem_gate_rate_all[stage_f32];
                    float gate_bias = dt_bias[head_idx_2 * 128 + gate_col];
                    float prefix_log2 = 0.0f;
                    for (int gate_row = 0; gate_row < 32; gate_row++) {
                        long long gate_token = bos_3 + (long long)(chunk_idx_2 * 32 + gate_row);
                        float gate_log2 = 0.0f;
                        int gate_needs_compute = 1;
                        if (gate_row == 0) {
                            if (chunk_is_full_1 != 0) {
                                gate_log2 = early_gate0;
                                gate_needs_compute = 0;
                            }
                        }
                        if (gate_needs_compute != 0) {
                            if (gate_token < eos_3) {
                                __nv_bfloat16 gate_raw = smem_g_raw_all[stage_bf16 + gate_row * 128 + gate_col];
                                float _cvt_f32_1 = __bfloat162float(gate_raw);
                                float gate_arg = gate_rate * (_cvt_f32_1 + gate_bias);
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
                    float _exp2_0 = approx_exp2(total_log2 - lower_bound * 1.4426950408889634f * 16.0f);
                    smem_restore_factor_all[stage_f32 + prep_tid] = _exp2_0;
                }
                if (prep_tid == 0) {
                    float _exp2_1 = approx_exp2(lower_bound * 1.4426950408889634f * 16.0f);
                    smem_restore_factor_all[stage_f32 + 128] = _exp2_1;
                }
                #pragma unroll 1
                for (int work_pass = 0; work_pass < 4; work_pass++) {
                    int work_item = work_pass * 128 + prep_tid;
                    int row = work_item / 16;
                    int segment = work_item % 16;
                    long long token = bos_3 + (long long)(chunk_idx_2 * 32 + row);
                    int token_valid = ((token < eos_3) ? 1 : 0);
                    long long gmem_base = (token * (long long)num_heads + (long long)head_idx_2) * 128 + (long long)(segment * 8);
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
                            : "r"((smem_q_raw_prefetch_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
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
                            : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
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
                        float common_log2 = lower_bound * 1.4426950408889634f * 16.0f;
                        float _exp2_2 = approx_exp2(prefix - common_log2);
                        float decay = _exp2_2;
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
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word = 0; word < 4; word++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word * 4)), "r"((packed_1[word])));
                    }
                    unsigned int packed_0_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_vec[_lp*2 + 0], kd_vec[_lp*2+1 + 0]));
                        packed_0_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_1 = 0; word_1 < 4; word_1++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_1 * 4)), "r"((packed_0_1[word_1])));
                    }
                    unsigned int packed_1_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_vec[_lp*2 + 0], ki_vec[_lp*2+1 + 0]));
                        packed_1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_2 = 0; word_2 < 4; word_2++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_2 * 4)), "r"((packed_1_1[word_2])));
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                unsigned int a_frag[4];
                unsigned int b_frag[4];
                float acc[8];
                {
                    int pair_row_base = prep_local_warp / 2 * 16;
                    int pair_col_base = prep_local_warp % 2 * 16;
                    if (pair_row_base >= pair_col_base) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        int row0 = pair_row_base + lane / 4;
                        int row1 = row0 + 8;
                        int col0 = pair_col_base + lane % 4 * 2;
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
                        int byte_off = (int)prep_stage * 41984 + (pair_row_base + seed_lane_row) * 128 + (pair_col_base + seed_lane_col) * 2;
                        int swizzled_off = byte_off ^ (byte_off >> 7 & 7) << 4;
                        int seed_addr = smem_inv_work_addr + (unsigned int)swizzled_off;
                        uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)seed_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[3]))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    } else {
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        acc[4] = 0.0f;
                        acc[5] = 0.0f;
                        acc[6] = 0.0f;
                        acc[7] = 0.0f;
                    }
                    int row0_1 = pair_row_base + lane / 4;
                    int row1_1 = row0_1 + 8;
                    int col0_1 = pair_col_base + lane % 4 * 2;
                    float mqk[8];
                    mqk[0] = 0.0f;
                    mqk[1] = 0.0f;
                    mqk[2] = 0.0f;
                    mqk[3] = 0.0f;
                    mqk[4] = 0.0f;
                    mqk[5] = 0.0f;
                    mqk[6] = 0.0f;
                    mqk[7] = 0.0f;
                    if (row0_1 >= col0_1) {
                        mqk[0] = acc[0];
                    }
                    if (row0_1 >= col0_1 + 1) {
                        mqk[1] = acc[1];
                    }
                    if (row1_1 >= col0_1) {
                        mqk[2] = acc[2];
                    }
                    if (row1_1 >= col0_1 + 1) {
                        mqk[3] = acc[3];
                    }
                    if (row0_1 >= col0_1 + 8) {
                        mqk[4] = acc[4];
                    }
                    if (row0_1 >= col0_1 + 9) {
                        mqk[5] = acc[5];
                    }
                    if (row1_1 >= col0_1 + 8) {
                        mqk[6] = acc[6];
                    }
                    if (row1_1 >= col0_1 + 9) {
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
                        int publish_row = pair_col_base + publish_pair * 8 + (lane & 7);
                        int publish_col = 128 + pair_row_base + lane / 8 * 8;
                        uint32_t _stmatrix_addr_6 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + prep_stage * 41984 + (unsigned int)(publish_col / 64 * 4096 + publish_row * 128 + publish_col % 64 * 2 ^ (publish_col / 64 * 4096 + publish_row * 128 + publish_col % 64 * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2 + 1]))
                            : "memory");
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                if (prep_tid < 128) {
                    float total_log2_1 = smem_gt_prefix_all[stage_f32 + prep_tid];
                    float _exp2_3 = approx_exp2(total_log2_1);
                    smem_gt_all[stage_f32 + prep_tid] = _exp2_3;
                }
                if (prep_local_warp == 1) {
                    smem_beta_all[stage_f32 + lane] = smem_prep_beta_all[stage_f32 + lane];
                }
                {
                    if (prep_local_warp >= 2) {
                        int stage_f32_0 = prep_stage * 10496;
                        float restore_scale = smem_restore_factor_all[stage_f32_0 + 128];
                        float restore_factor[8];
                        int restore_segment = lane & 15;
                        #pragma unroll
                        for (int restore_elem = 0; restore_elem < 8; restore_elem++) {
                            int restore_col = restore_segment * 8 + restore_elem;
                            restore_factor[restore_elem] = smem_restore_factor_all[stage_f32_0 + restore_col];
                        }
                        #pragma unroll 1
                        for (int restore_pass = 0; restore_pass < 6; restore_pass++) {
                            int restore_row = 8 + (prep_local_warp - 2) * 12 + restore_pass * 2 + (lane >> 4);
                            float restore_qd_values[8];
                            float restore_kd_values[8];
                            float restore_ki_values[8];
                            unsigned int packed_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 3]))
                                : "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
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
                                restore_qd_values[value_idx_2] = packed_f32_1[value_idx_2];
                            }
                            unsigned int packed_0_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 3]))
                                : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_0_f32_1[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_0_f32_1[_pair * 2])[0]), "=f"((&packed_0_f32_1[_pair * 2])[1])
                                    : "r"(packed_0_2[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_3 = 0; value_idx_3 < 8; value_idx_3++) {
                                restore_kd_values[value_idx_3] = packed_0_f32_1[value_idx_3];
                            }
                            unsigned int packed_1_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 3]))
                                : "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_1_f32[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_1_f32[_pair * 2])[0]), "=f"((&packed_1_f32[_pair * 2])[1])
                                    : "r"(packed_1_2[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_4 = 0; value_idx_4 < 8; value_idx_4++) {
                                restore_ki_values[value_idx_4] = packed_1_f32[value_idx_4];
                            }
                            float restore_kr_values[8];
                            #pragma unroll
                            for (int restore_elem_1 = 0; restore_elem_1 < 8; restore_elem_1++) {
                                restore_kr_values[restore_elem_1] = restore_ki_values[restore_elem_1] * restore_factor[restore_elem_1];
                            }
                            const float2 _scale2_7 = {restore_scale, restore_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_qd_values)[_ls], _scale2_7);
                            const float2 _scale2_8 = {restore_scale, restore_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_kd_values)[_ls], _scale2_8);
                            unsigned int packed_2_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values[_lp*2 + 0], restore_qd_values[_lp*2+1 + 0]));
                                packed_2_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_3 = 0; word_3 < 4; word_3++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_3 * 4)), "r"((packed_2_1[word_3])));
                            }
                            unsigned int packed_3[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values[_lp*2 + 0], restore_kd_values[_lp*2+1 + 0]));
                                packed_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_4 = 0; word_4 < 4; word_4++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_4 * 4)), "r"((packed_3[word_4])));
                            }
                            unsigned int packed_4[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values[_lp*2 + 0], restore_kr_values[_lp*2+1 + 0]));
                                packed_4[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_5 = 0; word_5 < 4; word_5++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_5 * 4)), "r"((packed_4[word_5])));
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
                        unsigned int packed_5[4];
                        int byte_off_1 = (int)prep_stage * 41984 + inverse_row * 128 + diag_block * 8 * 2;
                        int swizzled_off_1 = byte_off_1 ^ (byte_off_1 >> 7 & 7) << 4;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 3]))
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
                                : "r"(packed_5[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_5 = 0; value_idx_5 < 8; value_idx_5++) {
                            inv_row[value_idx_5] = packed_f32_2[value_idx_5];
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
                        unsigned int packed_6[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv_row[_lp*2 + 0], inv_row[_lp*2+1 + 0]));
                            packed_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        int byte_off_2 = (int)prep_stage * 41984 + inverse_row * 128 + diag_block * 8 * 2;
                        int swizzled_off_2 = byte_off_2 ^ (byte_off_2 >> 7 & 7) << 4;
                        #pragma unroll
                        for (int word_6 = 0; word_6 < 4; word_6++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_inv_work_addr + (unsigned int)swizzled_off_2 + (unsigned int)(word_6 * 4)), "r"((packed_6[word_6])));
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
                        int lane_row = lane & 7;
                        int byte_off_3 = (int)prep_stage * 41984 + (prep_local_warp * 16 + 8 + lane_row) * 128 + (prep_local_warp * 16 + 8) * 2;
                        int swizzled_off_3 = byte_off_3 ^ (byte_off_3 >> 7 & 7) << 4;
                        int d_addr = smem_inv_work_addr + (unsigned int)swizzled_off_3;
                        int byte_off_0 = (int)prep_stage * 41984 + (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2;
                        int swizzled_off_1_1 = byte_off_0 ^ (byte_off_0 >> 7 & 7) << 4;
                        int c_addr = smem_inv_work_addr + (unsigned int)swizzled_off_1_1;
                        int byte_off_2_1 = (int)prep_stage * 41984 + (prep_local_warp * 16 + lane_row) * 128 + prep_local_warp * 16 * 2;
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
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(dc_acc[0]), "=f"(dc_acc[1]), "=f"(dc_acc[2]), "=f"(dc_acc[3])
                            : "r"(d_frag[0]), "r"(d_frag[1]), "r"(c_frag[0]));
                        const float2 _scale2_9 = {-1.0f, -1.0f};
                        #pragma unroll
                        for (int _ls = 0; _ls < 2; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(dc_acc)[_ls], _scale2_9);
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dc_acc[_lp*2 + 0], dc_acc[_lp*2+1 + 0]));
                            dc_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                            : "=r"(inv_a_frag[0])
                            : "r"(a_addr)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(o_acc[0]), "=f"(o_acc[1]), "=f"(o_acc[2]), "=f"(o_acc[3])
                            : "r"(dc_bf16[0]), "r"(dc_bf16[1]), "r"(inv_a_frag[0]));
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_acc[_lp*2 + 0], o_acc[_lp*2+1 + 0]));
                            o_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int byte_off_4 = (int)prep_stage * 41984 + (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2;
                        int swizzled_off_5 = byte_off_4 ^ (byte_off_4 >> 7 & 7) << 4;
                        int o_addr = smem_inv_work_addr + (unsigned int)swizzled_off_5;
                        uint32_t _stmatrix_addr_10 = static_cast<uint32_t>((unsigned long long)o_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                            :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&o_bf16[0]))
                            : "memory");
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
                        int byte_off_5 = (int)prep_stage * 41984 + (16 + lane_row_1) * 128 + (16 + lane_col) * 2;
                        int swizzled_off_4 = byte_off_5 ^ (byte_off_5 >> 7 & 7) << 4;
                        int d_addr_1 = smem_inv_work_addr + (unsigned int)swizzled_off_4;
                        int byte_off_0_1 = (int)prep_stage * 41984 + (16 + lane_row_1) * 128 + lane_col * 2;
                        int swizzled_off_1_2 = byte_off_0_1 ^ (byte_off_0_1 >> 7 & 7) << 4;
                        int c_addr_1 = smem_inv_work_addr + (unsigned int)swizzled_off_1_2;
                        int byte_off_2_2 = (int)prep_stage * 41984 + lane_row_1 * 128 + lane_col * 2;
                        int swizzled_off_3_2 = byte_off_2_2 ^ (byte_off_2_2 >> 7 & 7) << 4;
                        int a_addr_1 = smem_inv_work_addr + (unsigned int)swizzled_off_3_2;
                        unsigned int d32_frag[4];
                        unsigned int c32_frag[4];
                        float dc32_acc[8];
                        unsigned int dc32_bf16[4];
                        unsigned int a32_frag[4];
                        float o32_acc[8];
                        unsigned int o32_bf16[4];
                        unsigned int zero32_bf16[4];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(d32_frag[0]), "=r"(d32_frag[1]), "=r"(d32_frag[2]), "=r"(d32_frag[3])
                            : "r"(d_addr_1)
                            : "memory");
                        int d_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)((16 + lane_col) / 16 * 1024 + (16 + lane_row_1) * 32 + (16 + lane_col) % 16 * 2 ^ ((16 + lane_col) / 16 * 1024 + (16 + lane_row_1) * 32 + (16 + lane_col) % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_11 = static_cast<uint32_t>((unsigned long long)d_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[3]))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(c32_frag[0]), "=r"(c32_frag[1]), "=r"(c32_frag[2]), "=r"(c32_frag[3])
                            : "r"(c_addr_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(dc32_acc[0]), "=f"(dc32_acc[1]), "=f"(dc32_acc[2]), "=f"(dc32_acc[3])
                            : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]), "r"(c32_frag[0]), "r"(c32_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(dc32_acc[4]), "=f"(dc32_acc[(4) + 1]), "=f"(dc32_acc[(4) + 2]), "=f"(dc32_acc[(4) + 3])
                            : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]), "r"(c32_frag[2]), "r"(c32_frag[(2) + 1]));
                        const float2 _scale2_12 = {-1.0f, -1.0f};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(dc32_acc)[_ls], _scale2_12);
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dc32_acc[_lp*2 + 0], dc32_acc[_lp*2+1 + 0]));
                            dc32_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a32_frag[0]), "=r"(a32_frag[1]), "=r"(a32_frag[2]), "=r"(a32_frag[3])
                            : "r"(a_addr_1)
                            : "memory");
                        int a_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)(lane_col / 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_13 = static_cast<uint32_t>((unsigned long long)a_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[3]))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(o32_acc[0]), "=f"(o32_acc[1]), "=f"(o32_acc[2]), "=f"(o32_acc[3])
                            : "r"(dc32_bf16[0]), "r"(dc32_bf16[1]), "r"(dc32_bf16[2]), "r"(dc32_bf16[3]), "r"(a32_frag[0]), "r"(a32_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(o32_acc[4]), "=f"(o32_acc[(4) + 1]), "=f"(o32_acc[(4) + 2]), "=f"(o32_acc[(4) + 3])
                            : "r"(dc32_bf16[0]), "r"(dc32_bf16[1]), "r"(dc32_bf16[2]), "r"(dc32_bf16[3]), "r"(a32_frag[2]), "r"(a32_frag[(2) + 1]));
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o32_acc[_lp*2 + 0], o32_acc[_lp*2+1 + 0]));
                            o32_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int o_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)(lane_col / 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_14 = static_cast<uint32_t>((unsigned long long)o_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[3]))
                            : "memory");
                        #pragma unroll
                        for (int zero_word = 0; zero_word < 4; zero_word++) {
                            zero32_bf16[zero_word] = 0;
                        }
                        int zero_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)((16 + lane_col) / 16 * 1024 + lane_row_1 * 32 + (16 + lane_col) % 16 * 2 ^ ((16 + lane_col) / 16 * 1024 + lane_row_1 * 32 + (16 + lane_col) % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_15 = static_cast<uint32_t>((unsigned long long)zero_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[3]))
                            : "memory");
                    } else if (prep_local_warp == 1) {
                        int stage_f32_0_1 = prep_stage * 10496;
                        float restore_scale_1 = smem_restore_factor_all[stage_f32_0_1 + 128];
                        float restore_factor_1[8];
                        int restore_segment_1 = lane & 15;
                        #pragma unroll
                        for (int restore_elem_2 = 0; restore_elem_2 < 8; restore_elem_2++) {
                            int restore_col_1 = restore_segment_1 * 8 + restore_elem_2;
                            restore_factor_1[restore_elem_2] = smem_restore_factor_all[stage_f32_0_1 + restore_col_1];
                        }
                        #pragma unroll 1
                        for (int restore_pass_1 = 0; restore_pass_1 < 4; restore_pass_1++) {
                            int restore_row_1 = restore_pass_1 * 2 + (lane >> 4);
                            float restore_qd_values_1[8];
                            float restore_kd_values_1[8];
                            float restore_ki_values_1[8];
                            unsigned int packed_7[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_7[(0) + 3]))
                                : "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_f32_3[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_f32_3[_pair * 2])[0]), "=f"((&packed_f32_3[_pair * 2])[1])
                                    : "r"(packed_7[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_6 = 0; value_idx_6 < 8; value_idx_6++) {
                                restore_qd_values_1[value_idx_6] = packed_f32_3[value_idx_6];
                            }
                            unsigned int packed_0_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 3]))
                                : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_0_f32_2[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_0_f32_2[_pair * 2])[0]), "=f"((&packed_0_f32_2[_pair * 2])[1])
                                    : "r"(packed_0_3[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_7 = 0; value_idx_7 < 8; value_idx_7++) {
                                restore_kd_values_1[value_idx_7] = packed_0_f32_2[value_idx_7];
                            }
                            unsigned int packed_1_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 3]))
                                : "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_1_f32_1[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_1_f32_1[_pair * 2])[0]), "=f"((&packed_1_f32_1[_pair * 2])[1])
                                    : "r"(packed_1_3[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_8 = 0; value_idx_8 < 8; value_idx_8++) {
                                restore_ki_values_1[value_idx_8] = packed_1_f32_1[value_idx_8];
                            }
                            float restore_kr_values_1[8];
                            #pragma unroll
                            for (int restore_elem_3 = 0; restore_elem_3 < 8; restore_elem_3++) {
                                restore_kr_values_1[restore_elem_3] = restore_ki_values_1[restore_elem_3] * restore_factor_1[restore_elem_3];
                            }
                            const float2 _scale2_16 = {restore_scale_1, restore_scale_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_qd_values_1)[_ls], _scale2_16);
                            const float2 _scale2_17 = {restore_scale_1, restore_scale_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_kd_values_1)[_ls], _scale2_17);
                            unsigned int packed_2_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values_1[_lp*2 + 0], restore_qd_values_1[_lp*2+1 + 0]));
                                packed_2_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_7 = 0; word_7 < 4; word_7++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_7 * 4)), "r"((packed_2_2[word_7])));
                            }
                            unsigned int packed_3_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values_1[_lp*2 + 0], restore_kd_values_1[_lp*2+1 + 0]));
                                packed_3_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_8 = 0; word_8 < 4; word_8++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_8 * 4)), "r"((packed_3_1[word_8])));
                            }
                            unsigned int packed_4_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values_1[_lp*2 + 0], restore_kr_values_1[_lp*2+1 + 0]));
                                packed_4_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_9 = 0; word_9 < 4; word_9++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_9 * 4)), "r"((packed_4_1[word_9])));
                            }
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                if (prep_local_warp == 0) {
                    if (elect_sync()) {
                        tma_store_2d(packet_workspace, 0, packet_slot * 123, smem_packet_addr + prep_stage * 41984);
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    if (elect_sync()) {
                        __threadfence();
                    }
                }
                if (prep_local_warp == 0) {
                    if (elect_sync()) {
                        atomicAdd(reinterpret_cast<unsigned int*>(packet_ready) + (packet_slot), 1u);
                    }
                }
                for (int _advance = 0; _advance < 5; _advance++) {
                    prep_stage += 1;
                    if (prep_stage == 5) { prep_stage = 0; _phase_gate_raw_full ^= 1; _phase_qk_raw_full ^= 1; _phase_prep_diag_ready ^= 1; _phase_prep_inv16_ready ^= 1; }
                }
            }
            unsigned int _phase_raw_inputs_free = 1;
            unsigned int _phase_smem_free = 1;
            unsigned int _phase_v_free = 1;
            unsigned int _phase_packet_full = 0;
            if (group_rank_3 != 0) {
                if (num_prep_iters != 0) {
                    int packet_slot_1 = task_idx_3 * 35 + helper_instance;
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            {
                                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(packet_consumed) + (packet_slot_1);
                                while (true) {
                                    unsigned int _gca_v;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                    if (_gca_v >= (unsigned int)(num_prep_iters)) break;
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                }
                asm volatile("barrier.sync 15, 640;" ::: "memory");
                if (prep_all_tid == 0) {
                    atomicAdd(reinterpret_cast<unsigned int*>(helper_done) + (task_idx_3), 1u);
                }
            } else {
                unsigned int owner_stage = (unsigned int)prep_instance;
                int narrow_owner_waits = ((gridDim.x / 8 == 8) ? 1 : 0);
                int owner_iters = (num_chunks_3 + 5 - 1 - prep_instance) / 5;
                #pragma unroll 1
                for (int owner_iter = 0; owner_iter < owner_iters; owner_iter++) {
                    int owner_chunk = owner_iter * 5 + prep_instance;
                    int owner_slot_in_ring = owner_chunk % 35;
                    int owner_packet_slot = task_idx_3 * 35 + owner_slot_in_ring;
                    int owner_generation = owner_chunk / 35 + 1;
                    int owner_chunk_is_full = ((seq_len_3 >= (owner_chunk + 1) * 32) ? 1 : 0);
                    if (narrow_owner_waits != 0) {
                        if (prep_local_warp == 0) {
                            if (owner_chunk_is_full != 0 || owner_iter != 0) {
                                mbarrier_wait(raw_inputs_free_addr + (owner_stage) * 8, _phase_raw_inputs_free);
                            }
                            mbarrier_wait(smem_free_addr + (owner_stage) * 8, _phase_smem_free);
                            mbarrier_wait(v_free_addr + (owner_stage) * 8, _phase_v_free);
                        }
                    } else {
                        if (owner_chunk_is_full != 0 || owner_iter != 0) {
                            mbarrier_wait(raw_inputs_free_addr + (owner_stage) * 8, _phase_raw_inputs_free);
                        }
                        mbarrier_wait(smem_free_addr + (owner_stage) * 8, _phase_smem_free);
                        mbarrier_wait(v_free_addr + (owner_stage) * 8, _phase_v_free);
                    }
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            {
                                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(packet_ready) + (owner_packet_slot);
                                while (true) {
                                    unsigned int _gca_v;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                    if (_gca_v >= (unsigned int)(owner_generation)) break;
                                }
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive_expect_tx(packet_full_addr + (owner_stage) * 8, 31488);
                            tma_2d_gmem2smem(smem_packet_addr + owner_stage * 41984, packet_workspace, 0, owner_packet_slot * 123, packet_full_addr + (owner_stage) * 8);
                            mbarrier_arrive_expect_tx(v_full_addr + (owner_stage) * 8, 8192);
                            tma_3d_gmem2smem(smem_v_addr + owner_stage * 41984, v_tma, 0, head_idx_2, (int)(bos_3 + (long long)(owner_chunk * 32)), v_full_addr + (owner_stage) * 8);
                        }
                    }
                    if (narrow_owner_waits != 0) {
                        if (prep_local_warp == 0) {
                            mbarrier_wait(packet_full_addr + (owner_stage) * 8, _phase_packet_full);
                        }
                    } else {
                        mbarrier_wait(packet_full_addr + (owner_stage) * 8, _phase_packet_full);
                        asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    }
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(qk_full_addr + (owner_stage) * 8);
                            atomicAdd(reinterpret_cast<unsigned int*>(packet_consumed) + (owner_packet_slot), 1u);
                        }
                    }
                    for (int _advance_1 = 0; _advance_1 < 5; _advance_1++) {
                        owner_stage += 1;
                        if (owner_stage == 5) { owner_stage = 0; _phase_gate_raw_full ^= 1; _phase_qk_raw_full ^= 1; _phase_prep_diag_ready ^= 1; _phase_prep_inv16_ready ^= 1; _phase_raw_inputs_free ^= 1; _phase_smem_free ^= 1; _phase_v_free ^= 1; _phase_packet_full ^= 1; }
                    }
                }
                asm volatile("barrier.sync 15, 640;" ::: "memory");
                if (prep_all_tid == 0) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(helper_done) + (task_idx_3);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(7)) break;
                        }
                    }
                    #pragma unroll
                    for (int reset_slot = 0; reset_slot < 35; reset_slot++) {
                        int reset_index = task_idx_3 * 35 + reset_slot;
                        int reset_generation = (num_chunks_3 + 35 - 1 - reset_slot) / 35;
                        if (reset_generation != 0) {
                            uint32_t _atomic_cas_old_0 = atomicCAS(reinterpret_cast<unsigned int*>(&packet_ready[reset_index]), static_cast<unsigned int>(reset_generation), static_cast<unsigned int>(0));
                            uint32_t _atomic_cas_old_1 = atomicCAS(reinterpret_cast<unsigned int*>(&packet_consumed[reset_index]), static_cast<unsigned int>(reset_generation), static_cast<unsigned int>(0));
                        }
                    }
                    uint32_t _atomic_cas_old_2 = atomicCAS(reinterpret_cast<unsigned int*>(&helper_done[task_idx_3]), static_cast<unsigned int>(7), static_cast<unsigned int>(0));
                    __threadfence();
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

// clang-format on
