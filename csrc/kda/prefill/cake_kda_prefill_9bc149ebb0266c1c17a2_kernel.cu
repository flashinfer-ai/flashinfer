/*
 * Copyright (c) 2026 by FlashInfer team.
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
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_STATE_OFFSET 64
#define TMEM_TMEM_STATE_DECAY0_OFFSET 64
#define TMEM_TMEM_STATE_DECAY1_OFFSET 80
#define TMEM_TMEM_STATE_DECAY2_OFFSET 96
#define TMEM_TMEM_STATE_DECAY3_OFFSET 112
#define TMEM_TMEM_STATE_DECAY4_OFFSET 128
#define TMEM_TMEM_STATE_DECAY5_OFFSET 144
#define TMEM_TMEM_STATE_DECAY6_OFFSET 160
#define TMEM_TMEM_STATE_DECAY7_OFFSET 176
#define TMEM_TMEM_STATE_INP_OFFSET 0
#define TMEM_TMEM_STATE_INP_DECAY0_OFFSET 0
#define TMEM_TMEM_STATE_INP_DECAY1_OFFSET 8
#define TMEM_TMEM_STATE_INP_DECAY2_OFFSET 16
#define TMEM_TMEM_STATE_INP_DECAY3_OFFSET 24
#define TMEM_TMEM_STATE_INP_DECAY4_OFFSET 32
#define TMEM_TMEM_STATE_INP_DECAY5_OFFSET 40
#define TMEM_TMEM_STATE_INP_DECAY6_OFFSET 48
#define TMEM_TMEM_STATE_INP_DECAY7_OFFSET 56
#define TMEM_TMEM_U_ACC_OFFSET 224
#define TMEM_TMEM_U2_INP_OFFSET 224
#define TMEM_TMEM_U2_ACC_OFFSET 0
#define TMEM_TMEM_OUT_OFFSET 192
#define TMEM_TMEM_STATE_OUT_OFFSET 64
#define TMEM_TMEM_TURN_SET1_OFFSET 256
#define NUM_CHUNK_PIPE_STAGES 4
#define NUM_OUTPUT_PIPE_STAGES 2
#define NUM_CHECKPOINT_PIPE_STAGES 2
#define NUM_SNAPSHOT_PIPE_STAGES 4
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 8192
#define SMEM_SMEM_QD_STRIDE 41984
#define SMEM_SMEM_G_RAW_OFF 1024
#define SMEM_SMEM_G_RAW_STAGE_BYTES 8192
#define SMEM_SMEM_G_RAW_STRIDE 41984
#define SMEM_SMEM_G_RAW_ALL_OFF 1024
#define SMEM_SMEM_G_RAW_ALL_STAGE_BYTES 8192
#define SMEM_SMEM_G_RAW_ALL_STRIDE 41984
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
#define SMEM_SMEM_FINAL_MQK_SLAB_OFF 25600
#define SMEM_SMEM_FINAL_MQK_SLAB_STAGE_BYTES 4096
#define SMEM_SMEM_FINAL_MQK_SLAB_STRIDE 41984
#define SMEM_SMEM_INV_OFF 29696
#define SMEM_SMEM_INV_STAGE_BYTES 2048
#define SMEM_SMEM_INV_STRIDE 41984
#define SMEM_SMEM_V_OFF 32384
#define SMEM_SMEM_V_STAGE_BYTES 8192
#define SMEM_SMEM_V_STRIDE 41984
#define SMEM_SMEM_SHORT_N32_V_OFF 168960
#define SMEM_SMEM_SHORT_N32_V_STAGE_BYTES 32768
#define SMEM_SMEM_SHORT_N32_V_STRIDE 32768
#define SMEM_SMEM_KI_OFF 17408
#define SMEM_SMEM_KI_STAGE_BYTES 8192
#define SMEM_SMEM_KI_STRIDE 41984
#define SMEM_SMEM_GATE_OFF 25600
#define SMEM_SMEM_GATE_STAGE_BYTES 16384
#define SMEM_SMEM_GATE_STRIDE 41984
#define SMEM_SMEM_BETA_RAW_OFF 41984
#define SMEM_SMEM_BETA_RAW_STAGE_BYTES 512
#define SMEM_SMEM_BETA_RAW_STRIDE 41984
#define SMEM_SMEM_BETA_RAW_ALL_OFF 41984
#define SMEM_SMEM_BETA_RAW_ALL_STAGE_BYTES 512
#define SMEM_SMEM_BETA_RAW_ALL_STRIDE 41984
#define SMEM_SMEM_INV_WORK_OFF 32384
#define SMEM_SMEM_INV_WORK_STAGE_BYTES 4096
#define SMEM_SMEM_INV_WORK_STRIDE 41984
#define SMEM_SMEM_OUT_OFF 210944
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_V18_OFF 210944
#define SMEM_SMEM_V18_STAGE_BYTES 16384
#define SMEM_SMEM_V18_STRIDE 16384
#define SMEM_SMEM_V18_0_OFF 168960
#define SMEM_SMEM_V18_0_STAGE_BYTES 16384
#define SMEM_SMEM_V18_0_STRIDE 16384
#define SMEM_SMEM_V18_1_OFF 185344
#define SMEM_SMEM_V18_1_STAGE_BYTES 16384
#define SMEM_SMEM_V18_1_STRIDE 16384
#define SMEM_SMEM_CHECKPOINT_OFF 1024
#define SMEM_SMEM_CHECKPOINT_STAGE_BYTES 32768
#define SMEM_SMEM_CHECKPOINT_STRIDE 32768
#define SMEM_SMEM_RESTORE_FACTOR_ALL_OFF 41984
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES 516
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STRIDE 41984
#define SMEM_SMEM_V21_OFF 41984
#define SMEM_SMEM_V21_STAGE_BYTES 512
#define SMEM_SMEM_V21_STRIDE 41984
#define SMEM_SMEM_GT_PREFIX_OFF 41472
#define SMEM_SMEM_GT_PREFIX_STAGE_BYTES 512
#define SMEM_SMEM_GT_PREFIX_STRIDE 41984
#define SMEM_SMEM_GT_PREFIX_ALL_OFF 41472
#define SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES 168448
#define SMEM_SMEM_GT_PREFIX_ALL_STRIDE 168448
#define SMEM_SMEM_GT_ALL_OFF 31744
#define SMEM_SMEM_GT_ALL_STAGE_BYTES 512
#define SMEM_SMEM_GT_ALL_STRIDE 41984
#define SMEM_SMEM_V25_OFF 227920
#define SMEM_SMEM_V25_STAGE_BYTES 2560
#define SMEM_SMEM_V25_STRIDE 2560
#define SMEM_SMEM_PREP_BETA_ALL_OFF 42500
#define SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES 128
#define SMEM_SMEM_PREP_BETA_ALL_STRIDE 41984
#define SMEM_SMEM_PREP_BETA_BF16_ALL_OFF 42500
#define SMEM_SMEM_PREP_BETA_BF16_ALL_STAGE_BYTES 64
#define SMEM_SMEM_PREP_BETA_BF16_ALL_STRIDE 41984
#define SMEM_SMEM_PREP_BETA_U32_ALL_OFF 42500
#define SMEM_SMEM_PREP_BETA_U32_ALL_STAGE_BYTES 64
#define SMEM_SMEM_PREP_BETA_U32_ALL_STRIDE 41984
#define SMEM_SMEM_GATE_RATE_ALL_OFF 42628
#define SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES 4
#define SMEM_SMEM_GATE_RATE_ALL_STRIDE 41984
#define SMEM_SMEM_GATE_BIAS_ALL_OFF 227408
#define SMEM_SMEM_GATE_BIAS_ALL_STAGE_BYTES 512
#define SMEM_SMEM_GATE_BIAS_ALL_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG_RAW_OFF 1024
#define SMEM_SMEM_STATE_DECAY_DIAG_RAW_STAGE_BYTES 4096
#define SMEM_SMEM_STATE_DECAY_DIAG_RAW_STRIDE 4096
#define SMEM_SMEM_STATE_DECAY_DIAG0_OFF 1024
#define SMEM_SMEM_STATE_DECAY_DIAG0_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG0_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG1_OFF 1536
#define SMEM_SMEM_STATE_DECAY_DIAG1_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG1_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG2_OFF 2048
#define SMEM_SMEM_STATE_DECAY_DIAG2_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG2_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG3_OFF 2560
#define SMEM_SMEM_STATE_DECAY_DIAG3_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG3_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG4_OFF 3072
#define SMEM_SMEM_STATE_DECAY_DIAG4_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG4_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG5_OFF 3584
#define SMEM_SMEM_STATE_DECAY_DIAG5_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG5_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG6_OFF 4096
#define SMEM_SMEM_STATE_DECAY_DIAG6_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG6_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG7_OFF 4608
#define SMEM_SMEM_STATE_DECAY_DIAG7_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG7_STRIDE 512
#define SMEM_SMEM_V_ALL_OFF 32384
#define SMEM_SMEM_V_ALL_STAGE_BYTES 8192
#define SMEM_SMEM_V_ALL_STRIDE 41984
#define SMEM_SMEM_GATE_ALL_OFF 25600
#define SMEM_SMEM_GATE_ALL_STAGE_BYTES 16384
#define SMEM_SMEM_GATE_ALL_STRIDE 41984
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_OFF 227328
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STAGE_BYTES 80
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STRIDE 80
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_OFF 227328
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_STAGE_BYTES 16
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_STRIDE 16
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_OFF 227344
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_STAGE_BYTES 4
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_STRIDE 4
#define SMEM_SMEM_WORK_ITEM_RESOLVED_OFF 227348
#define SMEM_SMEM_WORK_ITEM_RESOLVED_STAGE_BYTES 4
#define SMEM_SMEM_WORK_ITEM_RESOLVED_STRIDE 4
#define SMEM_TOTAL 230528
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
        :: "r"(mbar_addr), "r"(count) : "memory");
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


__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
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


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}


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

__global__ __launch_bounds__(896) void
kernel_cake_kda_prefill_9bc149ebb0266c1c17a2(__nv_bfloat16* __restrict__ q, const __grid_constant__ CUtensorMap q_tma, __nv_bfloat16* __restrict__ k, const __grid_constant__ CUtensorMap k_tma, __nv_bfloat16* __restrict__ v, const __grid_constant__ CUtensorMap v_tma, __nv_bfloat16* __restrict__ g, const __grid_constant__ CUtensorMap g_tma, __nv_bfloat16* __restrict__ beta, const __grid_constant__ CUtensorMap beta_tma, float* __restrict__ A_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ seq_order, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ out, const __grid_constant__ CUtensorMap out_tma, __nv_bfloat16* __restrict__ final_state, int num_heads, int use_initial_state, int store_final_state, float scale, float lower_bound, unsigned long long state_indices_addr, unsigned long long state_checkpoints_addr, unsigned long long checkpoint_cu_starts_addr, long long beta_token_stride, long long state_slot_stride, int use_state_indices, int checkpoint_every_n_tokens, long long* __restrict__ cu_chunk_offsets, __nv_bfloat16* __restrict__ chunk_state, unsigned int* __restrict__ state_checkpoint_needed, __nv_bfloat16* __restrict__ tape_qd, __nv_bfloat16* __restrict__ tape_kd, __nv_bfloat16* __restrict__ tape_kr, __nv_bfloat16* __restrict__ tape_j, float* __restrict__ tape_restore_factor, __nv_bfloat16* __restrict__ tape_e, __nv_bfloat16* __restrict__ tape_x, __nv_bfloat16* __restrict__ tape_r, float* __restrict__ norm_inv_out, __nv_bfloat16* __restrict__ decay_out, float* __restrict__ beta_active_out, float* __restrict__ initial_state_f32, unsigned int* __restrict__ zero_workspace, int zero_words, int num_sequences, const __grid_constant__ CUtensorMap state_checkpoints_tma, float* __restrict__ final_state_f32, long long g_token_stride, long long q_stride_m3, long long k_stride_m3, long long v_stride_m3)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define qk_full_addr (mbar_base + 0)
    #define gate_raw_full_addr (mbar_base + 32)
    #define qk_raw_full_addr (mbar_base + 64)
    #define v_full_addr (mbar_base + 96)
    #define v_free_addr (mbar_base + 128)
    #define smem_free_addr (mbar_base + 160)
    #define raw_inputs_free_addr (mbar_base + 192)
    #define state_inp_ready_addr (mbar_base + 224)
    #define old_out_ready_addr (mbar_base + 256)
    #define u_inp_ready_addr (mbar_base + 288)
    #define u2_acc_ready_addr (mbar_base + 320)
    #define u2_inp_ready_addr (mbar_base + 352)
    #define final_ready_addr (mbar_base + 384)
    #define out_empty_addr (mbar_base + 416)
    #define tmem_dealloc_ready_addr (mbar_base + 424)
    #define checkpoint_ready_addr (mbar_base + 432)
    #define checkpoint_free_addr (mbar_base + 448)
    #define checkpoint_snapshot_done_addr (mbar_base + 464)
    #define turn_ready_addr (mbar_base + 496)
    #define turn_free_addr (mbar_base + 512)
    #define final_done_addr (mbar_base + 528)
    #define turn_cp_done_addr (mbar_base + 544)
    #define prep_diag_ready_addr (mbar_base + 552)
    #define prep_inv16_ready_addr (mbar_base + 584)
    #define work_item_ready_addr (mbar_base + 616)
    #define short_beta_ready_addr (mbar_base + 624)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
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
    __nv_bfloat16* smem_final_mqk_slab = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int smem_final_mqk_slab_addr = smem + 25600;
    __nv_bfloat16* smem_inv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 29696);
    const int smem_inv_addr = smem + 29696;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32384);
    const int smem_v_addr = smem + 32384;
    __nv_bfloat16* smem_short_n32_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 168960);
    const int smem_short_n32_v_addr = smem + 168960;
    __nv_bfloat16* smem_ki = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_ki_addr = smem + 17408;
    float* smem_gate = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_gate_addr = smem + 25600;
    __nv_bfloat16* smem_beta_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_beta_raw_addr = smem + 41984;
    __nv_bfloat16* smem_beta_raw_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_beta_raw_all_addr = smem + 41984;
    __nv_bfloat16* smem_inv_work = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32384);
    const int smem_inv_work_addr = smem + 32384;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 210944);
    const int smem_out_addr = smem + 210944;
    float* smem_v18 = reinterpret_cast<float*>(smem_raw + 210944);
    const int smem_v18_addr = smem + 210944;
    float* smem_v18_0 = reinterpret_cast<float*>(smem_raw + 168960);
    const int smem_v18_addr_0 = smem + 168960;
    float* smem_v18_1 = reinterpret_cast<float*>(smem_raw + 185344);
    const int smem_v18_addr_1 = smem + 185344;
    __nv_bfloat16* smem_checkpoint = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_checkpoint_addr = smem + 1024;
    float* smem_restore_factor_all = reinterpret_cast<float*>(smem_raw + 41984);
    const int smem_restore_factor_all_addr = smem + 41984;
    int* smem_v21 = reinterpret_cast<int*>(smem_raw + 41984);
    const int smem_v21_addr = smem + 41984;
    float* smem_gt_prefix = reinterpret_cast<float*>(smem_raw + 41472);
    const int smem_gt_prefix_addr = smem + 41472;
    float* smem_gt_prefix_all = reinterpret_cast<float*>(smem_raw + 41472);
    const int smem_gt_prefix_all_addr = smem + 41472;
    float* smem_gt_all = reinterpret_cast<float*>(smem_raw + 31744);
    const int smem_gt_all_addr = smem + 31744;
    int* smem_v25 = reinterpret_cast<int*>(smem_raw + 227920);
    const int smem_v25_addr = smem + 227920;
    float* smem_prep_beta_all = reinterpret_cast<float*>(smem_raw + 42500);
    const int smem_prep_beta_all_addr = smem + 42500;
    __nv_bfloat16* smem_prep_beta_bf16_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 42500);
    const int smem_prep_beta_bf16_all_addr = smem + 42500;
    unsigned int* smem_prep_beta_u32_all = reinterpret_cast<unsigned int*>(smem_raw + 42500);
    const int smem_prep_beta_u32_all_addr = smem + 42500;
    float* smem_gate_rate_all = reinterpret_cast<float*>(smem_raw + 42628);
    const int smem_gate_rate_all_addr = smem + 42628;
    float* smem_gate_bias_all = reinterpret_cast<float*>(smem_raw + 227408);
    const int smem_gate_bias_all_addr = smem + 227408;
    unsigned int* smem_state_decay_diag_raw = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_state_decay_diag_raw_addr = smem + 1024;
    __nv_bfloat16* smem_state_decay_diag0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_state_decay_diag0_addr = smem + 1024;
    __nv_bfloat16* smem_state_decay_diag1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1536);
    const int smem_state_decay_diag1_addr = smem + 1536;
    __nv_bfloat16* smem_state_decay_diag2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_state_decay_diag2_addr = smem + 2048;
    __nv_bfloat16* smem_state_decay_diag3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2560);
    const int smem_state_decay_diag3_addr = smem + 2560;
    __nv_bfloat16* smem_state_decay_diag4 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_state_decay_diag4_addr = smem + 3072;
    __nv_bfloat16* smem_state_decay_diag5 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3584);
    const int smem_state_decay_diag5_addr = smem + 3584;
    __nv_bfloat16* smem_state_decay_diag6 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4096);
    const int smem_state_decay_diag6_addr = smem + 4096;
    __nv_bfloat16* smem_state_decay_diag7 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4608);
    const int smem_state_decay_diag7_addr = smem + 4608;
    __nv_bfloat16* smem_v_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32384);
    const int smem_v_all_addr = smem + 32384;
    float* smem_gate_all = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_gate_all_addr = smem + 25600;
    unsigned int* smem_state_checkpoint_needed = reinterpret_cast<unsigned int*>(smem_raw + 227328);
    const int smem_state_checkpoint_needed_addr = smem + 227328;
    float* smem_work_item_warp_max = reinterpret_cast<float*>(smem_raw + 227328);
    const int smem_work_item_warp_max_addr = smem + 227328;
    int* smem_work_item_compute_start = reinterpret_cast<int*>(smem_raw + 227344);
    const int smem_work_item_compute_start_addr = smem + 227344;
    unsigned int* smem_work_item_resolved = reinterpret_cast<unsigned int*>(smem_raw + 227348);
    const int smem_work_item_resolved_addr = smem + 227348;

    // Mbarrier init (26 pipeline groups, 0 ordered-sequence groups, 82 barriers)
    // Mbarriers at smem_raw[0..656)

    if (warp == 10) {
        // --- pipeline 'chunk_pipe' ---
        // qk_full: 4 barriers, init_count=128
        // gate_raw_full: 4 barriers, init_count=1
        // qk_raw_full: 4 barriers, init_count=1
        // v_full: 4 barriers, init_count=1
        // v_free: 4 barriers, init_count=128
        // smem_free: 4 barriers, init_count=128
        // raw_inputs_free: 4 barriers, init_count=1
        // state_inp_ready: 4 barriers, init_count=4
        // old_out_ready: 4 barriers, init_count=1
        // u_inp_ready: 4 barriers, init_count=4
        // u2_acc_ready: 4 barriers, init_count=1
        // u2_inp_ready: 4 barriers, init_count=4
        // final_ready: 4 barriers, init_count=2
        // out_empty: 1 barriers, init_count=1
        // tmem_dealloc_ready: 1 barriers, init_count=2
        // --- pipeline 'checkpoint_pipe' ---
        // checkpoint_ready: 2 barriers, init_count=4
        // checkpoint_free: 2 barriers, init_count=1
        // --- pipeline 'snapshot_pipe' ---
        // checkpoint_snapshot_done: 4 barriers, init_count=4
        // turn_ready: 2 barriers, init_count=1
        // turn_free: 2 barriers, init_count=128
        // final_done: 2 barriers, init_count=128
        // turn_cp_done: 1 barriers, init_count=1
        // --- pipeline 'chunk_pipe' ---
        // prep_diag_ready: 4 barriers, init_count=64
        // prep_inv16_ready: 4 barriers, init_count=64
        // work_item_ready: 1 barriers, init_count=1
        // short_beta_ready: 4 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_10_0 = 4;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(28), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(24), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(16), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(4), "r"((uint32_t)(128)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_10_0);
        uint32_t _mbarrier_init_count_10_32 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(30), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(26), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(24), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(22), "r"((uint32_t)(2)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(21), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(20), "r"((uint32_t)(2)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(16), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(12), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(8), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(4), "r"((uint32_t)(1)));
        mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_10_32);
        uint32_t _mbarrier_init_count_10_64 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(13), "r"((uint32_t)(64)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(5), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(4), "r"((uint32_t)(128)));
        if (lane < 18) {
            mbarrier_init(smem + 512 + lane * 8, _mbarrier_init_count_10_64);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 656);
    if (warp == 0) {
        int _tmem_hold = smem + 656;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state = taddr + 64;
    const int tmem_tmem_state_decay0 = taddr + 64;
    const int tmem_tmem_state_decay1 = taddr + 80;
    const int tmem_tmem_state_decay2 = taddr + 96;
    const int tmem_tmem_state_decay3 = taddr + 112;
    const int tmem_tmem_state_decay4 = taddr + 128;
    const int tmem_tmem_state_decay5 = taddr + 144;
    const int tmem_tmem_state_decay6 = taddr + 160;
    const int tmem_tmem_state_decay7 = taddr + 176;
    const int tmem_tmem_state_inp = taddr;
    const int tmem_tmem_state_inp_decay0 = taddr;
    const int tmem_tmem_state_inp_decay1 = taddr + 8;
    const int tmem_tmem_state_inp_decay2 = taddr + 16;
    const int tmem_tmem_state_inp_decay3 = taddr + 24;
    const int tmem_tmem_state_inp_decay4 = taddr + 32;
    const int tmem_tmem_state_inp_decay5 = taddr + 40;
    const int tmem_tmem_state_inp_decay6 = taddr + 48;
    const int tmem_tmem_state_inp_decay7 = taddr + 56;
    const int tmem_tmem_u_acc = taddr + 224;
    const int tmem_tmem_u2_inp = taddr + 224;
    const int tmem_tmem_u2_acc = taddr;
    const int tmem_tmem_out = taddr + 192;
    const int tmem_tmem_state_out = taddr + 64;
    const int tmem_tmem_turn_set1 = taddr + 256;
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
    }

    // ---- Role: compute ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // compute_main
            int task_idx = blockIdx.x;
            int warp_id_in_role = warp - 0;
            int compute_local_warp = warp_id_in_role;
            int warp_in_wg = warp % 4;
            int state_row = warp_in_wg * 32 + lane;
            int bin_begin = seq_order[bid];
            int bin_end = seq_order[bid + 1];
            int bin_chunk_base = 0;
            unsigned int compute_checkpoint_stage = 0;
            unsigned int compute_snapshot_stage = 0;
            int compute_turn = 0;
            unsigned int _phase_checkpoint_snapshot_done = 0;
            #pragma unroll 1
            for (int task_pos = bin_begin; task_pos < bin_end; task_pos++) {
                int binned_task_idx = seq_order[task_pos];
                int split_compute_start = 0;
                int seq_idx = binned_task_idx / num_heads;
                int head_idx = binned_task_idx % num_heads;
                long long bos = cu_seqlens[seq_idx];
                long long eos = cu_seqlens[seq_idx + 1];
                int num_chunks = ((int)(eos - bos) + 32 - 1) / 32;
                int seq_len = (int)(eos - bos);
                int num_chunks_0 = (seq_len + 32 - 1) / 32;
                int num_pages = 0;
                long long total_chunks = cu_chunk_offsets[num_sequences];
                long long fallback_head = total_chunks * (long long)num_heads + (long long)seq_idx * (long long)num_heads + (long long)head_idx;
                const int tmem_row_base = (warp_in_wg * 32 << 16) + (compute_turn & 1) * 256;
                int _stage_turn_ready = (int)((unsigned int)compute_turn % 2);
                int _parity_turn_ready = (int)((unsigned int)compute_turn / 2 % 2);
                mbarrier_wait(turn_ready_addr + (_stage_turn_ready) * 8, _parity_turn_ready);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll 1
                for (int chunk_idx = 0; chunk_idx < num_chunks_0; chunk_idx++) {
                    int chunk_global_local = chunk_idx;
                    int owned_chunk = chunk_global_local >= 0 && chunk_global_local < num_chunks;
                    int checkpoint_token_entering = chunk_idx * 32;
                    int checkpoint_entering = checkpoint_every_n_tokens != 0 && checkpoint_token_entering % checkpoint_every_n_tokens == 0;
                    float state_panel0[32];
                    float state_panel1[32];
                    float state_panel2[32];
                    float state_panel3[32];
                    unsigned int state_packed0[16];
                    unsigned int state_packed1[16];
                    unsigned int state_packed2[16];
                    unsigned int state_packed3[16];
                    int _stage_qk_full = (int)((unsigned int)(bin_chunk_base + chunk_idx) % 4);
                    int _parity_qk_full = (int)((unsigned int)(bin_chunk_base + chunk_idx) / 4 % 2);
                    mbarrier_wait(qk_full_addr + (_stage_qk_full) * 8, _parity_qk_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int state_col_block = 0; state_col_block < 3; state_col_block++) {
                        int state_addr = taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block * 32);
                        float tmem_load[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(tmem_load[0]), "=f"(tmem_load[1]), "=f"(tmem_load[2]), "=f"(tmem_load[3]), "=f"(tmem_load[4]), "=f"(tmem_load[5]), "=f"(tmem_load[6]), "=f"(tmem_load[7]), "=f"(tmem_load[8]), "=f"(tmem_load[9]), "=f"(tmem_load[10]), "=f"(tmem_load[11]), "=f"(tmem_load[12]), "=f"(tmem_load[13]), "=f"(tmem_load[14]), "=f"(tmem_load[15]), "=f"(tmem_load[16]), "=f"(tmem_load[17]), "=f"(tmem_load[18]), "=f"(tmem_load[19]), "=f"(tmem_load[20]), "=f"(tmem_load[21]), "=f"(tmem_load[22]), "=f"(tmem_load[23]), "=f"(tmem_load[24]), "=f"(tmem_load[25]), "=f"(tmem_load[26]), "=f"(tmem_load[27]), "=f"(tmem_load[28]), "=f"(tmem_load[29]), "=f"(tmem_load[30]), "=f"(tmem_load[31])
                            : "r"(state_addr));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        unsigned int tmem_load_1_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load[_lp*2 + 0], tmem_load[_lp*2+1 + 0]));
                            tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + (unsigned int)tmem_row_base + (unsigned int)(state_col_block * 16)), "r"(tmem_load_1_bf16[0]), "r"(tmem_load_1_bf16[1]), "r"(tmem_load_1_bf16[2]), "r"(tmem_load_1_bf16[3]), "r"(tmem_load_1_bf16[4]), "r"(tmem_load_1_bf16[5]), "r"(tmem_load_1_bf16[6]), "r"(tmem_load_1_bf16[7]), "r"(tmem_load_1_bf16[8]), "r"(tmem_load_1_bf16[9]), "r"(tmem_load_1_bf16[10]), "r"(tmem_load_1_bf16[11]), "r"(tmem_load_1_bf16[12]), "r"(tmem_load_1_bf16[13]), "r"(tmem_load_1_bf16[14]), "r"(tmem_load_1_bf16[15]));
                        float state_scale[16];
                        #pragma unroll
                        for (int state_half = 0; state_half < 2; state_half++) {
                            #pragma unroll
                            for (int state_col = 0; state_col < 16; state_col++) {
                                state_scale[state_col] = smem_gt_all[(unsigned int)(bin_chunk_base + chunk_idx) % 4 * 10496 + (unsigned int)(state_col_block * 32) + (unsigned int)(state_half * 16) + (unsigned int)state_col];
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((tmem_load + state_half * 16))[_ls], reinterpret_cast<const float2*>(state_scale)[_ls]);
                        }
                        if (STORE_BACKWARD_TAPE == 0 && checkpoint_entering != 0 && chunk_idx != 0) {
                            mbarrier_wait(checkpoint_snapshot_done_addr + (compute_snapshot_stage) * 8, _phase_checkpoint_snapshot_done);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            compute_snapshot_stage += 1;
                            if (compute_snapshot_stage == 4) { compute_snapshot_stage = 0; _phase_checkpoint_snapshot_done ^= 1; }
                        }
                        tmem_st_x32_f32(state_addr, tmem_load);
                    }
                    int state_tail_addr = taddr + 64 + (unsigned int)tmem_row_base + 96;
                    float state_tail_frag[32];
                    unsigned int state_tail_packed[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(state_tail_frag[0]), "=f"(state_tail_frag[1]), "=f"(state_tail_frag[2]), "=f"(state_tail_frag[3]), "=f"(state_tail_frag[4]), "=f"(state_tail_frag[5]), "=f"(state_tail_frag[6]), "=f"(state_tail_frag[7]), "=f"(state_tail_frag[8]), "=f"(state_tail_frag[9]), "=f"(state_tail_frag[10]), "=f"(state_tail_frag[11]), "=f"(state_tail_frag[12]), "=f"(state_tail_frag[13]), "=f"(state_tail_frag[14]), "=f"(state_tail_frag[15]), "=f"(state_tail_frag[16]), "=f"(state_tail_frag[17]), "=f"(state_tail_frag[18]), "=f"(state_tail_frag[19]), "=f"(state_tail_frag[20]), "=f"(state_tail_frag[21]), "=f"(state_tail_frag[22]), "=f"(state_tail_frag[23]), "=f"(state_tail_frag[24]), "=f"(state_tail_frag[25]), "=f"(state_tail_frag[26]), "=f"(state_tail_frag[27]), "=f"(state_tail_frag[28]), "=f"(state_tail_frag[29]), "=f"(state_tail_frag[30]), "=f"(state_tail_frag[31])
                        : "r"(state_tail_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_tail_frag[_lp*2 + 0], state_tail_frag[_lp*2+1 + 0]));
                        state_tail_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + (unsigned int)tmem_row_base + 48), "r"(state_tail_packed[0]), "r"(state_tail_packed[1]), "r"(state_tail_packed[2]), "r"(state_tail_packed[3]), "r"(state_tail_packed[4]), "r"(state_tail_packed[5]), "r"(state_tail_packed[6]), "r"(state_tail_packed[7]), "r"(state_tail_packed[8]), "r"(state_tail_packed[9]), "r"(state_tail_packed[10]), "r"(state_tail_packed[11]), "r"(state_tail_packed[12]), "r"(state_tail_packed[13]), "r"(state_tail_packed[14]), "r"(state_tail_packed[15]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(state_inp_ready_addr + ((int)((unsigned int)(bin_chunk_base + chunk_idx) % 4)) * 8);
                    }
                    float state_tail_scale[16];
                    #pragma unroll
                    for (int state_half_1 = 0; state_half_1 < 2; state_half_1++) {
                        #pragma unroll
                        for (int state_col_1 = 0; state_col_1 < 16; state_col_1++) {
                            state_tail_scale[state_col_1] = smem_gt_all[(unsigned int)(bin_chunk_base + chunk_idx) % 4 * 10496 + 96 + (unsigned int)(state_half_1 * 16) + (unsigned int)state_col_1];
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((state_tail_frag + state_half_1 * 16))[_ls], reinterpret_cast<const float2*>(state_tail_scale)[_ls]);
                    }
                    if (STORE_BACKWARD_TAPE == 0 && checkpoint_entering != 0 && chunk_idx != 0) {
                        mbarrier_wait(checkpoint_snapshot_done_addr + (compute_snapshot_stage) * 8, _phase_checkpoint_snapshot_done);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        compute_snapshot_stage += 1;
                        if (compute_snapshot_stage == 4) { compute_snapshot_stage = 0; _phase_checkpoint_snapshot_done ^= 1; }
                    }
                    tmem_st_x32_f32(state_tail_addr, state_tail_frag);
                    int _stage_v_full = (int)((unsigned int)(bin_chunk_base + chunk_idx) % 4);
                    int _parity_v_full = (int)((unsigned int)(bin_chunk_base + chunk_idx) / 4 % 2);
                    mbarrier_wait(v_full_addr + (_stage_v_full) * 8, _parity_v_full);
                    unsigned int v_prefetch_bits[8];
                    int _stage_old_out_ready = (int)((unsigned int)(bin_chunk_base + chunk_idx) % 4);
                    int _parity_old_out_ready = (int)((unsigned int)(bin_chunk_base + chunk_idx) / 4 % 2);
                    mbarrier_wait(old_out_ready_addr + (_stage_old_out_ready) * 8, _parity_old_out_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float prediction[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(prediction[0]), "=f"(prediction[1]), "=f"(prediction[2]), "=f"(prediction[3]), "=f"(prediction[4]), "=f"(prediction[5]), "=f"(prediction[6]), "=f"(prediction[7]), "=f"(prediction[8]), "=f"(prediction[9]), "=f"(prediction[10]), "=f"(prediction[11]), "=f"(prediction[12]), "=f"(prediction[13]), "=f"(prediction[14]), "=f"(prediction[15]), "=f"(prediction[16]), "=f"(prediction[17]), "=f"(prediction[18]), "=f"(prediction[19]), "=f"(prediction[20]), "=f"(prediction[21]), "=f"(prediction[22]), "=f"(prediction[23]), "=f"(prediction[24]), "=f"(prediction[25]), "=f"(prediction[26]), "=f"(prediction[27]), "=f"(prediction[28]), "=f"(prediction[29]), "=f"(prediction[30]), "=f"(prediction[31])
                        : "r"(taddr + 224 + (unsigned int)tmem_row_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    long long chunk_global_e = cu_chunk_offsets[seq_idx] + (long long)chunk_global_local;
                    long long tape_ex_base = ((chunk_global_e * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 32;
                    #pragma unroll
                    for (int residual_half = 0; residual_half < 2; residual_half++) {
                        float residual_v[16];
                        float residual_beta[16];
                        #pragma unroll
                        for (int residual_col = 0; residual_col < 16; residual_col++) {
                            int token_col = residual_half * 16 + residual_col;
                            __nv_bfloat16 v_value = smem_v_all[(unsigned int)(bin_chunk_base + chunk_idx) % 4 * 20992 + (unsigned int)(token_col * 128) + (unsigned int)state_row];
                            residual_v[residual_col] = (float)v_value;
                            residual_beta[residual_col] = smem_prep_beta_all[(unsigned int)(bin_chunk_base + chunk_idx) % 4 * 10496 + (unsigned int)token_col];
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(residual_v)[_ls], reinterpret_cast<const float2*>((prediction + residual_half * 16))[_ls]);
                        if (STORE_BACKWARD_TAPE != 0 && STORE_E_TAPE != 0 && owned_chunk != 0) {
                            unsigned int _packed_3[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 0], residual_v[_lp*2+1 + 0]));
                                _packed_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 2], residual_v[_lp*2+1 + 2]));
                                _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 4], residual_v[_lp*2+1 + 4]));
                                _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 6], residual_v[_lp*2+1 + 6]));
                                _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(tape_e + (tape_ex_base + (long long)(residual_half * 16)))[0] = reinterpret_cast<int4*>(_packed_3)[0];
                            unsigned int _packed_4[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 8], residual_v[_lp*2+1 + 8]));
                                _packed_4[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 10], residual_v[_lp*2+1 + 10]));
                                _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 12], residual_v[_lp*2+1 + 12]));
                                _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 14], residual_v[_lp*2+1 + 14]));
                                _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(tape_e + (tape_ex_base + (long long)(residual_half * 16) + 8))[0] = reinterpret_cast<int4*>(_packed_4)[0];
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(residual_v)[_ls], reinterpret_cast<const float2*>(residual_beta)[_ls]);
                        if (STORE_BACKWARD_TAPE != 0 && owned_chunk != 0) {
                            unsigned int _packed_5[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 0], residual_v[_lp*2+1 + 0]));
                                _packed_5[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 2], residual_v[_lp*2+1 + 2]));
                                _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 4], residual_v[_lp*2+1 + 4]));
                                _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 6], residual_v[_lp*2+1 + 6]));
                                _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(tape_x + (tape_ex_base + (long long)(residual_half * 16)))[0] = reinterpret_cast<int4*>(_packed_5)[0];
                            unsigned int _packed_6[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 8], residual_v[_lp*2+1 + 8]));
                                _packed_6[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 10], residual_v[_lp*2+1 + 10]));
                                _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 12], residual_v[_lp*2+1 + 12]));
                                _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 14], residual_v[_lp*2+1 + 14]));
                                _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(tape_x + (tape_ex_base + (long long)(residual_half * 16) + 8))[0] = reinterpret_cast<int4*>(_packed_6)[0];
                        }
                        unsigned int residual_v_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 0], residual_v[_lp*2+1 + 0]));
                            residual_v_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        tmem_st_x8_u32(taddr + 224 + (unsigned int)tmem_row_base + (unsigned int)(residual_half * 8), (const uint32_t*)residual_v_bf16);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(v_free_addr + ((int)((unsigned int)(bin_chunk_base + chunk_idx) % 4)) * 8);
                    if (elect_sync()) {
                        mbarrier_arrive(u_inp_ready_addr + ((int)((unsigned int)(bin_chunk_base + chunk_idx) % 4)) * 8);
                    }
                    int _stage_u2_acc_ready = (int)((unsigned int)(bin_chunk_base + chunk_idx) % 4);
                    int _parity_u2_acc_ready = (int)((unsigned int)(bin_chunk_base + chunk_idx) / 4 % 2);
                    mbarrier_wait(u2_acc_ready_addr + (_stage_u2_acc_ready) * 8, _parity_u2_acc_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(tmem_load_1[0]), "=f"(tmem_load_1[1]), "=f"(tmem_load_1[2]), "=f"(tmem_load_1[3]), "=f"(tmem_load_1[4]), "=f"(tmem_load_1[5]), "=f"(tmem_load_1[6]), "=f"(tmem_load_1[7]), "=f"(tmem_load_1[8]), "=f"(tmem_load_1[9]), "=f"(tmem_load_1[10]), "=f"(tmem_load_1[11]), "=f"(tmem_load_1[12]), "=f"(tmem_load_1[13]), "=f"(tmem_load_1[14]), "=f"(tmem_load_1[15]), "=f"(tmem_load_1[16]), "=f"(tmem_load_1[17]), "=f"(tmem_load_1[18]), "=f"(tmem_load_1[19]), "=f"(tmem_load_1[20]), "=f"(tmem_load_1[21]), "=f"(tmem_load_1[22]), "=f"(tmem_load_1[23]), "=f"(tmem_load_1[24]), "=f"(tmem_load_1[25]), "=f"(tmem_load_1[26]), "=f"(tmem_load_1[27]), "=f"(tmem_load_1[28]), "=f"(tmem_load_1[29]), "=f"(tmem_load_1[30]), "=f"(tmem_load_1[31])
                        : "r"(taddr + (unsigned int)tmem_row_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (STORE_BACKWARD_TAPE != 0 && owned_chunk != 0) {
                        long long chunk_global_r = cu_chunk_offsets[seq_idx] + (long long)chunk_global_local;
                        long long tape_r_base = ((chunk_global_r * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 32;
                        #pragma unroll
                        for (int tape_r_vec = 0; tape_r_vec < 4; tape_r_vec++) {
                            unsigned int _packed_7[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((tmem_load_1 + tape_r_vec * 8)[_lp*2 + 0], (tmem_load_1 + tape_r_vec * 8)[_lp*2+1 + 0]));
                                _packed_7[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((tmem_load_1 + tape_r_vec * 8)[_lp*2 + 2], (tmem_load_1 + tape_r_vec * 8)[_lp*2+1 + 2]));
                                _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((tmem_load_1 + tape_r_vec * 8)[_lp*2 + 4], (tmem_load_1 + tape_r_vec * 8)[_lp*2+1 + 4]));
                                _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((tmem_load_1 + tape_r_vec * 8)[_lp*2 + 6], (tmem_load_1 + tape_r_vec * 8)[_lp*2+1 + 6]));
                                _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(tape_r + (tape_r_base + (long long)(tape_r_vec * 8)))[0] = reinterpret_cast<int4*>(_packed_7)[0];
                        }
                    }
                    unsigned int u2_packed[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_1[_lp*2 + 0], tmem_load_1[_lp*2+1 + 0]));
                        u2_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 224 + (unsigned int)tmem_row_base), "r"(u2_packed[0]), "r"(u2_packed[1]), "r"(u2_packed[2]), "r"(u2_packed[3]), "r"(u2_packed[4]), "r"(u2_packed[5]), "r"(u2_packed[6]), "r"(u2_packed[7]), "r"(u2_packed[8]), "r"(u2_packed[9]), "r"(u2_packed[10]), "r"(u2_packed[11]), "r"(u2_packed[12]), "r"(u2_packed[13]), "r"(u2_packed[14]), "r"(u2_packed[15]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(u2_inp_ready_addr + ((int)((unsigned int)(bin_chunk_base + chunk_idx) % 4)) * 8);
                    }
                    int _stage_final_ready = (int)((unsigned int)(bin_chunk_base + chunk_idx) % 4);
                    int _parity_final_ready = (int)((unsigned int)(bin_chunk_base + chunk_idx) / 4 % 2);
                    mbarrier_wait(final_ready_addr + (_stage_final_ready) * 8, _parity_final_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(smem_free_addr + ((int)((unsigned int)(bin_chunk_base + chunk_idx) % 4)) * 8);
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(turn_free_addr + ((int)((unsigned int)compute_turn % 2)) * 8);
                bin_chunk_base += num_chunks_0;
                compute_turn += 1;
                asm volatile("tcgen05.fence::before_thread_sync;");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            if (compute_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    // ---- Role: epilogue ----
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // epilogue_main
            int task_idx_1 = blockIdx.x;
            int bin_begin_1 = seq_order[bid];
            int bin_end_1 = seq_order[bid + 1];
            int bin_chunk_base_1 = 0;
            unsigned int epilogue_checkpoint_stage = 0;
            unsigned int epilogue_snapshot_stage = 0;
            int epilogue_turn = 0;
            unsigned int _phase_work_item_ready_0 = 0;
            #pragma unroll 1
            for (int task_pos_1 = bin_begin_1; task_pos_1 < bin_end_1; task_pos_1++) {
                int binned_task_idx_1 = seq_order[task_pos_1];
                int split_compute_start_1 = 0;
                int seq_idx_1 = binned_task_idx_1 / num_heads;
                int head_idx_1 = binned_task_idx_1 % num_heads;
                long long bos_1 = cu_seqlens[seq_idx_1];
                long long eos_1 = cu_seqlens[seq_idx_1 + 1];
                int num_chunks_1 = ((int)(eos_1 - bos_1) + 32 - 1) / 32;
                int seq_len_1 = (int)(eos_1 - bos_1);
                int num_chunks_0_1 = (seq_len_1 + 32 - 1) / 32;
                int warp_id_in_role_1 = warp - 4;
                int epilogue_local_warp = warp_id_in_role_1;
                int warp_in_wg_1 = warp % 4;
                const int tmem_row_base_1 = (warp_in_wg_1 * 32 << 16) + (epilogue_turn & 1) * 256;
                int state_row_1 = warp_in_wg_1 * 32 + lane;
                long long epilogue_checkpoint_row_start = 0;
                long long epilogue_checkpoint_single_token = 0;
                if (checkpoint_every_n_tokens != 0) {
                    epilogue_checkpoint_row_start = reinterpret_cast<long long*>(checkpoint_cu_starts_addr)[seq_idx_1];
                }
                int epilogue_chunks = num_chunks_0_1;
                if (epilogue_turn > 0) {
                    int turn_final_warp = warp_id_in_role_1;
                    int turn_final_row = warp_in_wg_1 * 32 + lane;
                    const int turn_final_tmem = (warp_in_wg_1 * 32 << 16) + (epilogue_turn - 1 & 1) * 256;
                    int _stage_turn_free = (int)((unsigned int)(epilogue_turn - 1) % 2);
                    int _parity_turn_free = (int)((unsigned int)(epilogue_turn - 1) / 2 % 2);
                    mbarrier_wait(turn_free_addr + (_stage_turn_free) * 8, _parity_turn_free);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (store_final_state != 0) {
                        int turn_final_task = seq_order[bin_begin_1 + (epilogue_turn - 1)];
                        int turn_final_seq = turn_final_task / num_heads;
                        int turn_final_head = turn_final_task % num_heads;
                        int turn_final_slot = turn_final_seq;
                        if (use_state_indices != 0) {
                            turn_final_slot = reinterpret_cast<int*>(state_indices_addr)[turn_final_seq];
                        }
                        long long turn_final_base = (long long)turn_final_slot * state_slot_stride + ((long long)turn_final_head * 128 + 0) * 128;
                        int turn_final_prow = turn_final_row / 8;
                        int turn_final_pcol = turn_final_row % 8 * 4;
                        if (turn_final_warp == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        #pragma unroll 1
                        for (int turn_final_block = 0; turn_final_block < 4; turn_final_block++) {
                            #pragma unroll 1
                            for (int turn_final_half = 0; turn_final_half < 2; turn_final_half++) {
                                float turn_final_frag[16];
                                tmem_ld_x16(&turn_final_frag[0], taddr + 64 + (unsigned int)turn_final_tmem + (unsigned int)((turn_final_block * 2 + turn_final_half) * 16));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int turn_final_vec = 0; turn_final_vec < 4; turn_final_vec++) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(turn_final_row * 128 + (turn_final_half * 4 + turn_final_vec) * 16 ^ (turn_final_row * 128 + (turn_final_half * 4 + turn_final_vec) * 16 >> 7 & 7) << 4))), "r"(__as_u32(turn_final_frag[turn_final_vec * 4])), "r"(__as_u32(turn_final_frag[turn_final_vec * 4 + 1])), "r"(__as_u32(turn_final_frag[turn_final_vec * 4 + 2])), "r"(__as_u32(turn_final_frag[turn_final_vec * 4 + 3])) : "memory");
                                }
                            }
                            asm volatile("barrier.sync 8, 128;" ::: "memory");
                            #pragma unroll
                            for (int turn_final_round = 0; turn_final_round < 8; turn_final_round++) {
                                float turn_final_words[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&turn_final_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&turn_final_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&turn_final_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&turn_final_words[(0) + 3]))
                                    : "r"((smem_v18_addr + (unsigned int)((turn_final_prow + turn_final_round * 16) * 128 + turn_final_pcol * 4 ^ ((turn_final_prow + turn_final_round * 16) * 128 + turn_final_pcol * 4 >> 7 & 7) << 4))));
                                {
                                    float4 _v4 = make_float4(turn_final_words[0 + 0], turn_final_words[0 + 1], turn_final_words[0 + 2], turn_final_words[0 + 3]);
                                    *reinterpret_cast<float4*>((final_state_f32 + (turn_final_base + (long long)(turn_final_prow + turn_final_round * 16) * 128 + (long long)(turn_final_block * 32) + (long long)turn_final_pcol)) + 0) = _v4;
                                }
                            }
                            asm volatile("barrier.sync 8, 128;" ::: "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(final_done_addr + ((int)((unsigned int)(epilogue_turn - 1) % 2)) * 8);
                }
                #pragma unroll 1
                for (int chunk_idx_1 = 0; chunk_idx_1 < epilogue_chunks; chunk_idx_1++) {
                    int checkpoint_token_epilogue = chunk_idx_1 * 32;
                    int checkpoint_entering_epilogue = checkpoint_every_n_tokens != 0 && checkpoint_token_epilogue % checkpoint_every_n_tokens == 0;
                    int chunk_is_full = ((seq_len_1 >= (chunk_idx_1 + 1) * 32) ? 1 : 0);
                    if (chunk_is_full != 0) {
                        int _stage_final_ready_1 = (int)((unsigned int)(bin_chunk_base_1 + chunk_idx_1) % 4);
                        int _parity_final_ready_1 = (int)((unsigned int)(bin_chunk_base_1 + chunk_idx_1) / 4 % 2);
                        mbarrier_wait(final_ready_addr + (_stage_final_ready_1) * 8, _parity_final_ready_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_3[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                            : "r"(taddr + 192 + (unsigned int)tmem_row_base_1));
                        float _tmem_load_4[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                            : "r"(taddr + 192 + (unsigned int)tmem_row_base_1 + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (epilogue_local_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(out_empty_addr);
                            }
                        }
                        if (epilogue_local_warp == 0) {
                            if (bin_chunk_base_1 + chunk_idx_1 >= 2) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        int out_stage_addr = smem_out_addr + (unsigned int)(bin_chunk_base_1 + chunk_idx_1) % 2 * 8192;
                        #pragma unroll
                        for (int dim_half = 0; dim_half < 2; dim_half++) {
                            unsigned int out_packed[8];
                            if (dim_half == 0) {
                                #pragma unroll
                                for (int _lp = 0; _lp < 8; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                                    out_packed[_lp] = *(uint32_t*)&_bf2;
                                }
                            } else {
                                #pragma unroll
                                for (int _lp = 0; _lp < 8; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
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
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (epilogue_local_warp == 0) {
                            if (elect_sync()) {
                                tma_store_4d((&out_tma), 0, (int)(bos_1 + (long long)(chunk_idx_1 * 32)), head_idx_1, 0, smem_out_addr + (unsigned int)(bin_chunk_base_1 + chunk_idx_1) % 2 * 8192);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        int snapshot_chunk = chunk_idx_1 + 1;
                        int snapshot_entering = checkpoint_every_n_tokens != 0 && snapshot_chunk * 32 % checkpoint_every_n_tokens == 0 && snapshot_chunk < num_chunks_0_1;
                        if (snapshot_entering != 0) {
                            long long snapshot_row_idx = epilogue_checkpoint_row_start + (long long)(snapshot_chunk * 32 / checkpoint_every_n_tokens);
                            int snapshot_outer = (int)(snapshot_row_idx * (long long)num_heads + (long long)head_idx_1);
                            #pragma unroll
                            for (int snapshot_panel = 0; snapshot_panel < 4; snapshot_panel++) {
                                float tmem_load_2[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(tmem_load_2[0]), "=f"(tmem_load_2[1]), "=f"(tmem_load_2[2]), "=f"(tmem_load_2[3]), "=f"(tmem_load_2[4]), "=f"(tmem_load_2[5]), "=f"(tmem_load_2[6]), "=f"(tmem_load_2[7]), "=f"(tmem_load_2[8]), "=f"(tmem_load_2[9]), "=f"(tmem_load_2[10]), "=f"(tmem_load_2[11]), "=f"(tmem_load_2[12]), "=f"(tmem_load_2[13]), "=f"(tmem_load_2[14]), "=f"(tmem_load_2[15]), "=f"(tmem_load_2[16]), "=f"(tmem_load_2[17]), "=f"(tmem_load_2[18]), "=f"(tmem_load_2[19]), "=f"(tmem_load_2[20]), "=f"(tmem_load_2[21]), "=f"(tmem_load_2[22]), "=f"(tmem_load_2[23]), "=f"(tmem_load_2[24]), "=f"(tmem_load_2[25]), "=f"(tmem_load_2[26]), "=f"(tmem_load_2[27]), "=f"(tmem_load_2[28]), "=f"(tmem_load_2[29]), "=f"(tmem_load_2[30]), "=f"(tmem_load_2[31])
                                    : "r"(taddr + 64 + (unsigned int)tmem_row_base_1 + (unsigned int)(snapshot_panel * 32)));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                if (elect_sync()) {
                                    asm volatile("tcgen05.fence::before_thread_sync;");
                                    mbarrier_arrive(checkpoint_snapshot_done_addr + (epilogue_snapshot_stage) * 8);
                                }
                                epilogue_snapshot_stage += 1;
                                if (epilogue_snapshot_stage == 4) { epilogue_snapshot_stage = 0; }
                                if (epilogue_local_warp == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile("barrier.sync 8, 128;" ::: "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 ^ (state_row_1 * 128 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[0])), "r"(__as_u32(tmem_load_2[1])), "r"(__as_u32(tmem_load_2[2])), "r"(__as_u32(tmem_load_2[3])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 16 ^ (state_row_1 * 128 + 16 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[4])), "r"(__as_u32(tmem_load_2[5])), "r"(__as_u32(tmem_load_2[6])), "r"(__as_u32(tmem_load_2[7])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 32 ^ (state_row_1 * 128 + 32 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[8])), "r"(__as_u32(tmem_load_2[9])), "r"(__as_u32(tmem_load_2[10])), "r"(__as_u32(tmem_load_2[11])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 48 ^ (state_row_1 * 128 + 48 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[12])), "r"(__as_u32(tmem_load_2[13])), "r"(__as_u32(tmem_load_2[14])), "r"(__as_u32(tmem_load_2[15])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 64 ^ (state_row_1 * 128 + 64 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[16])), "r"(__as_u32(tmem_load_2[17])), "r"(__as_u32(tmem_load_2[18])), "r"(__as_u32(tmem_load_2[19])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 80 ^ (state_row_1 * 128 + 80 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[20])), "r"(__as_u32(tmem_load_2[21])), "r"(__as_u32(tmem_load_2[22])), "r"(__as_u32(tmem_load_2[23])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 96 ^ (state_row_1 * 128 + 96 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[24])), "r"(__as_u32(tmem_load_2[25])), "r"(__as_u32(tmem_load_2[26])), "r"(__as_u32(tmem_load_2[27])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 112 ^ (state_row_1 * 128 + 112 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[28])), "r"(__as_u32(tmem_load_2[29])), "r"(__as_u32(tmem_load_2[30])), "r"(__as_u32(tmem_load_2[31])) : "memory");
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile("barrier.sync 8, 128;" ::: "memory");
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                if (epilogue_local_warp == 0) {
                                    if (elect_sync()) {
                                        tma_store_3d((&state_checkpoints_tma), snapshot_panel * 32, 0, snapshot_outer, smem_v18_addr);
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (epilogue_local_warp == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            asm volatile("barrier.sync 8, 128;" ::: "memory");
                        }
                    } else {
                        int _stage_final_ready_2 = (int)((unsigned int)(bin_chunk_base_1 + chunk_idx_1) % 4);
                        int _parity_final_ready_2 = (int)((unsigned int)(bin_chunk_base_1 + chunk_idx_1) / 4 % 2);
                        mbarrier_wait(final_ready_addr + (_stage_final_ready_2) * 8, _parity_final_ready_2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_6[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                            : "r"(taddr + 192 + (unsigned int)tmem_row_base_1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
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
                                out[out_idx] = _tmem_load_6[token_col_1];
                            }
                        }
                    }
                }
                bin_chunk_base_1 += num_chunks_0_1;
                epilogue_turn += 1;
            }
            if (bin_end_1 > bin_begin_1) {
                int warp_id_in_role_2 = warp - 4;
                int turn_final_warp_1 = warp_id_in_role_2;
                int warp_in_wg_2 = warp % 4;
                int turn_final_row_1 = warp_in_wg_2 * 32 + lane;
                const int turn_final_tmem_1 = (warp_in_wg_2 * 32 << 16) + (epilogue_turn - 1 & 1) * 256;
                int _stage_turn_free_1 = (int)((unsigned int)(epilogue_turn - 1) % 2);
                int _parity_turn_free_1 = (int)((unsigned int)(epilogue_turn - 1) / 2 % 2);
                mbarrier_wait(turn_free_addr + (_stage_turn_free_1) * 8, _parity_turn_free_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (store_final_state != 0) {
                    int turn_final_task_1 = seq_order[bin_begin_1 + (epilogue_turn - 1)];
                    int turn_final_seq_1 = turn_final_task_1 / num_heads;
                    int turn_final_head_1 = turn_final_task_1 % num_heads;
                    int turn_final_slot_1 = turn_final_seq_1;
                    if (use_state_indices != 0) {
                        turn_final_slot_1 = reinterpret_cast<int*>(state_indices_addr)[turn_final_seq_1];
                    }
                    long long turn_final_base_1 = (long long)turn_final_slot_1 * state_slot_stride + ((long long)turn_final_head_1 * 128 + 0) * 128;
                    int turn_final_prow_1 = turn_final_row_1 / 8;
                    int turn_final_pcol_1 = turn_final_row_1 % 8 * 4;
                    if (turn_final_warp_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    #pragma unroll 1
                    for (int turn_final_block_1 = 0; turn_final_block_1 < 4; turn_final_block_1++) {
                        #pragma unroll 1
                        for (int turn_final_half_1 = 0; turn_final_half_1 < 2; turn_final_half_1++) {
                            float turn_final_frag_1[16];
                            tmem_ld_x16(&turn_final_frag_1[0], taddr + 64 + (unsigned int)turn_final_tmem_1 + (unsigned int)((turn_final_block_1 * 2 + turn_final_half_1) * 16));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #pragma unroll
                            for (int turn_final_vec_1 = 0; turn_final_vec_1 < 4; turn_final_vec_1++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(turn_final_row_1 * 128 + (turn_final_half_1 * 4 + turn_final_vec_1) * 16 ^ (turn_final_row_1 * 128 + (turn_final_half_1 * 4 + turn_final_vec_1) * 16 >> 7 & 7) << 4))), "r"(__as_u32(turn_final_frag_1[turn_final_vec_1 * 4])), "r"(__as_u32(turn_final_frag_1[turn_final_vec_1 * 4 + 1])), "r"(__as_u32(turn_final_frag_1[turn_final_vec_1 * 4 + 2])), "r"(__as_u32(turn_final_frag_1[turn_final_vec_1 * 4 + 3])) : "memory");
                            }
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        #pragma unroll
                        for (int turn_final_round_1 = 0; turn_final_round_1 < 8; turn_final_round_1++) {
                            float turn_final_words_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&turn_final_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&turn_final_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&turn_final_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&turn_final_words_1[(0) + 3]))
                                : "r"((smem_v18_addr + (unsigned int)((turn_final_prow_1 + turn_final_round_1 * 16) * 128 + turn_final_pcol_1 * 4 ^ ((turn_final_prow_1 + turn_final_round_1 * 16) * 128 + turn_final_pcol_1 * 4 >> 7 & 7) << 4))));
                            {
                                float4 _v4 = make_float4(turn_final_words_1[0 + 0], turn_final_words_1[0 + 1], turn_final_words_1[0 + 2], turn_final_words_1[0 + 3]);
                                *reinterpret_cast<float4*>((final_state_f32 + (turn_final_base_1 + (long long)(turn_final_prow_1 + turn_final_round_1 * 16) * 128 + (long long)(turn_final_block_1 * 32) + (long long)turn_final_pcol_1)) + 0) = _v4;
                            }
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(final_done_addr + ((int)((unsigned int)(epilogue_turn - 1) % 2)) * 8);
            }
            int warp_id_in_role_3 = warp - 4;
            int epilogue_local_warp_1 = warp_id_in_role_3;
            {
                if (epilogue_local_warp_1 == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (epilogue_local_warp_1 == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    // ---- Role: beta_prefetch ----
    } else if (warp == 8) {
        { // beta_prefetch_main
            int bin_begin_2 = seq_order[bid];
            int bin_end_2 = seq_order[bid + 1];
            int beta_turn = 0;
            int turn_cp_phase = 0;
            #pragma unroll 1
            for (int task_pos_2 = bin_begin_2; task_pos_2 < bin_end_2; task_pos_2++) {
                if (beta_turn >= 2) {
                    int _stage_final_done = (int)((unsigned int)(beta_turn - 2) % 2);
                    int _parity_final_done = (int)((unsigned int)(beta_turn - 2) / 2 % 2);
                    mbarrier_wait(final_done_addr + (_stage_final_done) * 8, _parity_final_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                int turn_init_task = seq_order[bin_begin_2 + beta_turn];
                int turn_init_seq = turn_init_task / num_heads;
                int turn_init_head = turn_init_task % num_heads;
                int turn_init_slot = turn_init_seq;
                if (use_state_indices != 0) {
                    turn_init_slot = reinterpret_cast<int*>(state_indices_addr)[turn_init_seq];
                }
                long long turn_init_base = (long long)turn_init_slot * state_slot_stride + ((long long)turn_init_head * 128 + 0) * 128;
                int turn_init_outer = 0;
                if (checkpoint_every_n_tokens != 0) {
                    turn_init_outer = (int)(reinterpret_cast<long long*>(checkpoint_cu_starts_addr)[turn_init_seq] * (long long)num_heads + (long long)turn_init_head);
                }
                int turn_init_set = (beta_turn & 1) * 256;
                int turn_init_lrow = lane / 8;
                int turn_init_lcol = lane % 8 * 4;
                #pragma unroll 1
                for (int turn_init_half = 0; turn_init_half < 2; turn_init_half++) {
                    if (turn_cp_phase > 0) {
                        int _parity_turn_cp_done = (int)((unsigned int)turn_cp_phase % 2) ^ 1;
                        mbarrier_wait(turn_cp_done_addr, _parity_turn_cp_done);
                    }
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    __syncwarp();
                    #pragma unroll 1
                    for (int turn_init_it = 0; turn_init_it < 32; turn_init_it++) {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_v18_addr_0 + (unsigned int)((turn_init_lrow + turn_init_it * 4) * 128 + turn_init_lcol * 4 ^ ((turn_init_lrow + turn_init_it * 4) * 128 + turn_init_lcol * 4 >> 7 & 7) << 4))), "l"(initial_state_f32 + (turn_init_base + (long long)(turn_init_lrow + turn_init_it * 4) * 128 + (long long)((turn_init_half * 2 + 0) * 32) + (long long)turn_init_lcol)), "r"((use_initial_state != 0) ? 16 : 0));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"((smem_v18_addr_1 + (unsigned int)((turn_init_lrow + turn_init_it * 4) * 128 + turn_init_lcol * 4 ^ ((turn_init_lrow + turn_init_it * 4) * 128 + turn_init_lcol * 4 >> 7 & 7) << 4))), "l"(initial_state_f32 + (turn_init_base + (long long)(turn_init_lrow + turn_init_it * 4) * 128 + (long long)((turn_init_half * 2 + 1) * 32) + (long long)turn_init_lcol)), "r"((use_initial_state != 0) ? 16 : 0));
                    }
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (checkpoint_every_n_tokens != 0) {
                        if (elect_sync()) {
                            tma_store_3d((&state_checkpoints_tma), (turn_init_half * 2 + 0) * 32, 0, turn_init_outer, smem_v18_addr_0);
                            tma_store_3d((&state_checkpoints_tma), (turn_init_half * 2 + 1) * 32, 0, turn_init_outer, smem_v18_addr_1);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_v18_addr_0 + 0)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                                :: "r"((uint32_t)(taddr + (unsigned int)(turn_init_set + 64 + (turn_init_half * 2 + 0) * 32 + 0))), "l"(_tcgen05_cp_desc_0)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_v18_addr_0 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                                :: "r"((uint32_t)(taddr + (unsigned int)(turn_init_set + 64 + (turn_init_half * 2 + 0) * 32 + 8))), "l"(_tcgen05_cp_desc_1)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_v18_addr_0 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                                :: "r"((uint32_t)(taddr + (unsigned int)(turn_init_set + 64 + (turn_init_half * 2 + 0) * 32 + 16))), "l"(_tcgen05_cp_desc_2)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_v18_addr_0 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                                :: "r"((uint32_t)(taddr + (unsigned int)(turn_init_set + 64 + (turn_init_half * 2 + 0) * 32 + 24))), "l"(_tcgen05_cp_desc_3)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_v18_addr_1 + 0)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                                :: "r"((uint32_t)(taddr + (unsigned int)(turn_init_set + 64 + (turn_init_half * 2 + 1) * 32 + 0))), "l"(_tcgen05_cp_desc_4)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_v18_addr_1 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                                :: "r"((uint32_t)(taddr + (unsigned int)(turn_init_set + 64 + (turn_init_half * 2 + 1) * 32 + 8))), "l"(_tcgen05_cp_desc_5)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_v18_addr_1 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                                :: "r"((uint32_t)(taddr + (unsigned int)(turn_init_set + 64 + (turn_init_half * 2 + 1) * 32 + 16))), "l"(_tcgen05_cp_desc_6)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_v18_addr_1 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                                :: "r"((uint32_t)(taddr + (unsigned int)(turn_init_set + 64 + (turn_init_half * 2 + 1) * 32 + 24))), "l"(_tcgen05_cp_desc_7)
                                : "memory");
                        }
                        tcgen05_commit(turn_cp_done_addr);
                        if (turn_init_half == 1) {
                            tcgen05_commit(turn_ready_addr + ((int)((unsigned int)beta_turn % 2)) * 8);
                        }
                    }
                    turn_cp_phase += 1;
                }
                beta_turn += 1;
            }
            asm volatile("cp.async.bulk.wait_group 0;");
        }
    // ---- Role: aux_mma ----
    } else if (warp >= 10 && warp <= 11) {
        { // aux_mma_main
            unsigned int _phase_work_item_ready_0_1 = 0;
        }
    // ---- Role: mma ----
    } else if (warp == 9) {
        { // mma_main
            int task_idx_2 = blockIdx.x;
            int bin_begin_3 = seq_order[bid];
            int bin_end_3 = seq_order[bid + 1];
            int bin_chunk_base_2 = 0;
            unsigned int _phase_work_item_ready_0_2 = 0;
            unsigned int _phase_out_empty_0 = 1;
            #pragma unroll 1
            for (int task_pos_3 = bin_begin_3; task_pos_3 < bin_end_3; task_pos_3++) {
                int binned_task_idx_2 = seq_order[task_pos_3];
                int split_compute_start_2 = 0;
                int seq_idx_2 = binned_task_idx_2 / num_heads;
                int head_idx_2 = binned_task_idx_2 % num_heads;
                long long bos_2 = cu_seqlens[seq_idx_2];
                long long eos_2 = cu_seqlens[seq_idx_2 + 1];
                int num_chunks_2 = ((int)(eos_2 - bos_2) + 32 - 1) / 32;
                int seq_len_2 = (int)(eos_2 - bos_2);
                int num_chunks_0_2 = (seq_len_2 + 32 - 1) / 32;
                int turn_col = (task_pos_3 - bin_begin_3 & 1) * 256;
                #pragma unroll 1
                for (int _chunk_idx = 0; _chunk_idx < num_chunks_0_2; _chunk_idx++) {
                    int _stage_qk_full_1 = (int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4);
                    int _parity_qk_full_1 = (int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) / 4 % 2);
                    mbarrier_wait(qk_full_addr + (_stage_qk_full_1) * 8, _parity_qk_full_1);
                    {
                        mbarrier_wait(out_empty_addr, _phase_out_empty_0);
                        _phase_out_empty_0 ^= 1;
                    }
                    int _stage_state_inp_ready = (int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4);
                    int _parity_state_inp_ready = (int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) / 4 % 2);
                    mbarrier_wait(state_inp_ready_addr + (_stage_state_inp_ready) * 8, _parity_state_inp_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    {
                        int _mma_b_lo_0 = make_warp_uniform((((smem_qd_addr) >> 4) & 0x3FFF) + ((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4) * 2624);
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
                    "mov.b32 id, 134743184;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"((tmem_tmem_out + (turn_col))), "r"(_mma_b_lo_0), "r"(tmem_tmem_state_inp + turn_col), "r"(0) : "memory");
                    }
                    int _mma_b_lo_1 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + ((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4) * 2624);
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
                    "mov.b32 id, 134743184;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"((tmem_tmem_u_acc + (turn_col))), "r"(_mma_b_lo_1), "r"(tmem_tmem_state_inp + turn_col), "r"(0) : "memory");
                    elect_commit2(old_out_ready_addr + ((int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4)) * 8, raw_inputs_free_addr + ((int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4)) * 8);
                    int _stage_u_inp_ready = (int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4);
                    int _parity_u_inp_ready = (int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) / 4 % 2);
                    mbarrier_wait(u_inp_ready_addr + (_stage_u_inp_ready) * 8, _parity_u_inp_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_2 = make_warp_uniform((((smem_inv_addr) >> 4) & 0x3FFF) + ((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4) * 2624);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_u2_acc + (turn_col))), "r"(_mma_b_lo_2), "r"(tmem_tmem_u2_inp + turn_col), "r"(0) : "memory");
                    elect_commit(u2_acc_ready_addr + ((int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4)) * 8);
                    int _stage_u2_inp_ready = (int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4);
                    int _parity_u2_inp_ready = (int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) / 4 % 2);
                    mbarrier_wait(u2_inp_ready_addr + (_stage_u2_inp_ready) * 8, _parity_u2_inp_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_final_trans_addr) >> 4) & 0x3FFF) | 0x1000000) + ((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4) * 2624);
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
                    "mov.b32 id, 136905872;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_state_out + (turn_col))), "r"(_mma_b_lo_3), "r"(tmem_tmem_u2_inp + turn_col), "r"(1) : "memory");
                    elect_commit(final_ready_addr + ((int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4)) * 8);
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(final_ready_addr + ((int)((unsigned int)(bin_chunk_base_2 + _chunk_idx) % 4)) * 8);
                    }
                }
                bin_chunk_base_2 += num_chunks_0_2;
            }
            unsigned int _phase_tmem_dealloc_ready_0 = 0;
            mbarrier_wait(tmem_dealloc_ready_addr, _phase_tmem_dealloc_ready_0);
            _phase_tmem_dealloc_ready_0 ^= 1;
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    // ---- Role: prep ----
    } else if (warp >= 12 && warp <= 27) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
        { // prep_main
            int task_idx_3 = blockIdx.x;
            int bin_begin_4 = seq_order[bid];
            int bin_end_4 = seq_order[bid + 1];
            int bin_chunk_base_3 = 0;
            unsigned int _phase_work_item_ready_0_3 = 0;
            #pragma unroll 1
            for (int task_pos_4 = bin_begin_4; task_pos_4 < bin_end_4; task_pos_4++) {
                int binned_task_idx_3 = seq_order[task_pos_4];
                int split_compute_start_3 = 0;
                int seq_idx_3 = binned_task_idx_3 / num_heads;
                int head_idx_3 = binned_task_idx_3 % num_heads;
                long long bos_3 = cu_seqlens[seq_idx_3];
                long long eos_3 = cu_seqlens[seq_idx_3 + 1];
                int num_chunks_3 = ((int)(eos_3 - bos_3) + 32 - 1) / 32;
                int seq_len_3 = (int)(eos_3 - bos_3);
                int num_chunks_0_3 = (seq_len_3 + 32 - 1) / 32;
                int instance_id = (warp - 12) / 4;
                int prep_instance = instance_id;
                int warp_id_in_role_4 = warp - 12;
                int prep_local_warp = warp_id_in_role_4 - prep_instance * 4;
                int prep_tid = prep_local_warp * 32 + lane;
                int prep_start = (prep_instance - bin_chunk_base_3 % 4 + 4) % 4;
                int num_prep_iters = (num_chunks_0_3 + 4 - 1 - prep_start) / 4;
                int gate_rate_stage_f32 = prep_instance * 10496;
                int prep_global_tid = warp_id_in_role_4 * 32 + lane;
                float task_gate_rate = __expf(A_log[head_idx_3]);
                float task_gate_bias = dt_bias[head_idx_3 * 128 + prep_tid];
                int prep_ordinal = bin_chunk_base_3 + prep_start;
                #pragma unroll 1
                for (int prep_iter = 0; prep_iter < num_prep_iters; prep_iter++) {
                    int chunk_idx_2 = prep_iter * 4 + prep_start;
                    int chunk_global_local_1 = chunk_idx_2;
                    int owned_chunk_1 = chunk_global_local_1 >= 0 && chunk_global_local_1 < num_chunks_3;
                    int stage_f32 = (unsigned int)prep_ordinal % 4 * 10496;
                    int stage_bf16 = (unsigned int)prep_ordinal % 4 * 20992;
                    int chunk_is_full_1 = ((seq_len_3 >= (chunk_idx_2 + 1) * 32) ? 1 : 0);
                    float early_beta_value = 0.0f;
                    float early_gate0 = 0.0f;
                    int _stage_raw_inputs_free = (int)((unsigned int)prep_ordinal % 4);
                    int _parity_raw_inputs_free = (int)((unsigned int)prep_ordinal / 4 % 2) ^ 1;
                    mbarrier_wait(raw_inputs_free_addr + (_stage_raw_inputs_free) * 8, _parity_raw_inputs_free);
                    if (chunk_is_full_1 != 0) {
                        if (prep_local_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(gate_raw_full_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8, 8704);
                                tma_3d_gmem2smem(smem_g_raw_addr + (unsigned int)prep_ordinal % 4 * 41984, (&g_tma), 0, head_idx_3, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), gate_raw_full_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8);
                                tma_2d_gmem2smem(smem_beta_raw_addr + (unsigned int)prep_ordinal % 4 * 41984, (&beta_tma), head_idx_3 / 8 * 8, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), gate_raw_full_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8);
                                mbarrier_arrive_expect_tx(qk_raw_full_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8, 16384);
                                tma_4d_gmem2smem(smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984, (&k_tma), 0, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), head_idx_3, 0, qk_raw_full_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8);
                            }
                        }
                    } else if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(gate_raw_full_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8);
                            mbarrier_arrive(qk_raw_full_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8);
                        }
                    }
                    int _stage_gate_raw_full = (int)((unsigned int)prep_ordinal % 4);
                    int _parity_gate_raw_full = (int)((unsigned int)prep_ordinal / 4 % 2);
                    mbarrier_wait(gate_raw_full_addr + (_stage_gate_raw_full) * 8, _parity_gate_raw_full);
                    if (chunk_is_full_1 != 0) {
                        if (prep_local_warp == 2 && lane < 32) {
                            float beta_logit = 0.0f;
                            unsigned int beta_raw_pair[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&beta_raw_pair[0])) : "r"(smem_beta_raw_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(lane * 16) + (unsigned int)(head_idx_3 % 8 / 2 * 4)));
                            float _unpack_bf16_0[2];
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(beta_raw_pair[_pair]) << 16);
                                (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(beta_raw_pair[_pair]) & 0xffff0000u);
                            }
                            beta_logit = _unpack_bf16_0[0];
                            if (head_idx_3 % 2 != 0) {
                                beta_logit = _unpack_bf16_0[1];
                            }
                            float _tanh_0;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_0) : "f"(beta_logit * 0.5f));
                            early_beta_value = __fmaf_rn(_tanh_0, 0.5f, 0.5f);
                        }
                        if (prep_tid < 128) {
                            float early_gate_rate = task_gate_rate;
                            float early_gate_bias = task_gate_bias;
                            __nv_bfloat16 early_gate_raw = smem_g_raw_all[stage_bf16 + prep_tid];
                            float unbounded_biased = (float)early_gate_raw + early_gate_bias;
                            float unbounded_exponential = approx_exp2(unbounded_biased * 1.442695f);
                            float unbounded_series_log2 = __fmaf_rn(unbounded_exponential * unbounded_exponential, __fmaf_rn(unbounded_exponential, __fmaf_rn(unbounded_exponential, __fmaf_rn(unbounded_exponential, 0.2f, -0.25f), 0.33333334f), -0.5f), unbounded_exponential) * 1.442695f;
                            float unbounded_direct_log2;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(unbounded_direct_log2) : "f"(1.0f + unbounded_exponential));
                            float unbounded_softplus_log2 = ((unbounded_biased <= 20.0f) ? ((unbounded_biased < -4.0f) ? unbounded_series_log2 : unbounded_direct_log2) : unbounded_biased * 1.442695f);
                            early_gate0 = (-early_gate_rate) * unbounded_softplus_log2;
                        }
                    }
                    int _stage_smem_free = (int)((unsigned int)prep_ordinal % 4);
                    int _parity_smem_free = (int)((unsigned int)prep_ordinal / 4 % 2) ^ 1;
                    mbarrier_wait(smem_free_addr + (_stage_smem_free) * 8, _parity_smem_free);
                    int _stage_v_free = (int)((unsigned int)prep_ordinal % 4);
                    int _parity_v_free = (int)((unsigned int)prep_ordinal / 4 % 2) ^ 1;
                    mbarrier_wait(v_free_addr + (_stage_v_free) * 8, _parity_v_free);
                    if (chunk_is_full_1 != 0) {
                        if (prep_local_warp == 0) {
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_q_raw_prefetch_addr + (unsigned int)prep_ordinal % 4 * 41984, (&q_tma), 0, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), head_idx_3, 0, qk_raw_full_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8);
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
                            long long gate_load_base = gate_load_token * g_token_stride + (long long)head_idx_3 * 128 + (long long)(gate_load_segment * 8);
                            long long q_load_base = gate_load_token * q_stride_m3 + (long long)head_idx_3 * 128 + (long long)(gate_load_segment * 8);
                            long long k_load_base = gate_load_token * k_stride_m3 + (long long)head_idx_3 * 128 + (long long)(gate_load_segment * 8);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                :: "r"(smem_g_raw_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(gate_load_item * 16)), "l"(g + gate_load_base), "r"((gate_load_token < eos_3) ? 16 : 0));
                            int q_tail_addr = (smem_q_raw_prefetch_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(gate_load_segment * 8 / 64 * 4096 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 ^ (gate_load_segment * 8 / 64 * 4096 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 >> 7 & 7) << 4));
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                :: "r"(q_tail_addr), "l"(q + q_load_base), "r"((gate_load_token < eos_3) ? 16 : 0));
                            int k_tail_addr = (smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(gate_load_segment * 8 / 64 * 4096 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 ^ (gate_load_segment * 8 / 64 * 4096 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 >> 7 & 7) << 4));
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                :: "r"(k_tail_addr), "l"(k + k_load_base), "r"((gate_load_token < eos_3) ? 16 : 0));
                        }
                    }
                    if (chunk_is_full_1 == 0) {
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 0;");
                        asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    }
                    if (prep_local_warp == 2 && lane < 32) {
                        long long beta_token = bos_3 + (long long)(chunk_idx_2 * 32 + lane);
                        float beta_value = early_beta_value;
                        if (chunk_is_full_1 == 0) {
                            if (beta_token < eos_3) {
                                float beta_logit_1 = (float)beta[beta_token * (long long)num_heads + (long long)head_idx_3];
                                beta_logit_1 = (float)beta[beta_token * beta_token_stride + (long long)head_idx_3];
                                float _tanh_1;
                                asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_1) : "f"(beta_logit_1 * 0.5f));
                                beta_value = __fmaf_rn(_tanh_1, 0.5f, 0.5f);
                            }
                        }
                        smem_prep_beta_all[stage_f32 + lane] = (float)(__nv_bfloat16)beta_value;
                    }
                    float unbounded_total_log2 = 0.0f;
                    if (prep_tid < 128) {
                        int gate_col = prep_tid;
                        float gate_rate = task_gate_rate;
                        float gate_bias = task_gate_bias;
                        float prefix_log2 = 0.0f;
                        float2 scan_neg_rate_pair = make_float2(-gate_rate, -gate_rate);
                        float2 scan_bias_pair = make_float2(gate_bias, gate_bias);
                        long long first_gate_token = bos_3 + (long long)(chunk_idx_2 * 32);
                        __nv_bfloat16 first_gate_raw = smem_g_raw_all[stage_bf16 + gate_col];
                        float first_gate_log2 = early_gate0;
                        if (chunk_is_full_1 == 0) {
                            float unbounded_biased_1 = (float)first_gate_raw + gate_bias;
                            float unbounded_exponential_1 = approx_exp2(unbounded_biased_1 * 1.442695f);
                            float unbounded_series_log2_1 = __fmaf_rn(unbounded_exponential_1 * unbounded_exponential_1, __fmaf_rn(unbounded_exponential_1, __fmaf_rn(unbounded_exponential_1, __fmaf_rn(unbounded_exponential_1, 0.2f, -0.25f), 0.33333334f), -0.5f), unbounded_exponential_1) * 1.442695f;
                            float unbounded_direct_log2_1;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(unbounded_direct_log2_1) : "f"(1.0f + unbounded_exponential_1));
                            float unbounded_softplus_log2_1 = ((unbounded_biased_1 <= 20.0f) ? ((unbounded_biased_1 < -4.0f) ? unbounded_series_log2_1 : unbounded_direct_log2_1) : unbounded_biased_1 * 1.442695f);
                            float first_gate_value = (-gate_rate) * unbounded_softplus_log2_1;
                            first_gate_log2 = ((first_gate_token < eos_3) ? first_gate_value : 0.0f);
                        }
                        unbounded_total_log2 = unbounded_total_log2 + first_gate_log2;
                        prefix_log2 += max_noftz(first_gate_log2, -15.0f);
                        smem_gate_all[stage_f32 + gate_col] = prefix_log2;
                        for (int gate_pair = 0; gate_pair < 15; gate_pair++) {
                            int pair_row0 = gate_pair * 2 + 1;
                            int pair_row1 = pair_row0 + 1;
                            long long pair_token0 = bos_3 + (long long)(chunk_idx_2 * 32 + pair_row0);
                            long long pair_token1 = pair_token0 + 1;
                            __nv_bfloat16 pair_raw0 = smem_g_raw_all[stage_bf16 + pair_row0 * 128 + gate_col];
                            __nv_bfloat16 pair_raw1 = smem_g_raw_all[stage_bf16 + pair_row1 * 128 + gate_col];
                            float2 unbounded_biased_pair = add_f32x2(make_float2((float)pair_raw0, (float)pair_raw1), scan_bias_pair);
                            float2 _f32x2_mul_b_2 = make_float2(1.442695f, 1.442695f);
                            float2 scaled;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled) : "l"(*(const unsigned long long*)&unbounded_biased_pair), "l"(*(const unsigned long long*)&_f32x2_mul_b_2));
                            float _exp2_3 = approx_exp2(scaled.x);
                            float _exp2_4 = approx_exp2(scaled.y);
                            float2 _f2_0 = make_float2(_exp2_3, _exp2_4);
                            float2 _fma_0 = fma_f32x2_rn_ftz(_f2_0, fma_f32x2_rn_ftz(_f2_0, fma_f32x2_rn_ftz(_f2_0, make_float2(0.2f, 0.2f), make_float2(-0.25f, -0.25f)), make_float2(0.33333334f, 0.33333334f)), make_float2(-0.5f, -0.5f));
                            float2 squared;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&squared) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_0));
                            float2 _f32x2_mul_a_5 = fma_f32x2_rn_ftz(squared, _fma_0, _f2_0);
                            float2 _f32x2_mul_b_6 = make_float2(1.442695f, 1.442695f);
                            float2 series_log2;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&series_log2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_6));
                            float2 unbounded_one_plus_pair = add_f32x2(_f2_0, make_float2(1.0f, 1.0f));
                            float _log2_7;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_7) : "f"(unbounded_one_plus_pair.x));
                            float _log2_8;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_8) : "f"(unbounded_one_plus_pair.y));
                            float2 _f2_1 = make_float2(_log2_7, _log2_8);
                            float2 _f32x2_mul_b_9 = make_float2(((unbounded_biased_pair.x <= 20.0f) ? ((unbounded_biased_pair.x < -4.0f) ? series_log2.x : _f2_1.x) : scaled.x), ((unbounded_biased_pair.y <= 20.0f) ? ((unbounded_biased_pair.y < -4.0f) ? series_log2.y : _f2_1.y) : scaled.y));
                            float2 gate;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gate) : "l"(*(const unsigned long long*)&scan_neg_rate_pair), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
                            float pair_log0 = ((pair_token0 < eos_3) ? gate.x : 0.0f);
                            float pair_log1 = ((pair_token1 < eos_3) ? gate.y : 0.0f);
                            unbounded_total_log2 = unbounded_total_log2 + pair_log0;
                            prefix_log2 += max_noftz(pair_log0, -15.0f);
                            smem_gate_all[stage_f32 + pair_row0 * 128 + gate_col] = prefix_log2;
                            unbounded_total_log2 = unbounded_total_log2 + pair_log1;
                            prefix_log2 += max_noftz(pair_log1, -15.0f);
                            smem_gate_all[stage_f32 + pair_row1 * 128 + gate_col] = prefix_log2;
                        }
                        long long last_gate_token = bos_3 + (long long)(chunk_idx_2 * 32 + 31);
                        __nv_bfloat16 last_gate_raw = smem_g_raw_all[stage_bf16 + 3968 + gate_col];
                        float unbounded_biased_2 = (float)last_gate_raw + gate_bias;
                        float unbounded_exponential_2 = approx_exp2(unbounded_biased_2 * 1.442695f);
                        float unbounded_series_log2_2 = __fmaf_rn(unbounded_exponential_2 * unbounded_exponential_2, __fmaf_rn(unbounded_exponential_2, __fmaf_rn(unbounded_exponential_2, __fmaf_rn(unbounded_exponential_2, 0.2f, -0.25f), 0.33333334f), -0.5f), unbounded_exponential_2) * 1.442695f;
                        float unbounded_direct_log2_2;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(unbounded_direct_log2_2) : "f"(1.0f + unbounded_exponential_2));
                        float unbounded_softplus_log2_2 = ((unbounded_biased_2 <= 20.0f) ? ((unbounded_biased_2 < -4.0f) ? unbounded_series_log2_2 : unbounded_direct_log2_2) : unbounded_biased_2 * 1.442695f);
                        float last_gate_value = (-gate_rate) * unbounded_softplus_log2_2;
                        float last_gate_log2 = ((last_gate_token < eos_3) ? last_gate_value : 0.0f);
                        unbounded_total_log2 = unbounded_total_log2 + last_gate_log2;
                        prefix_log2 += max_noftz(last_gate_log2, -15.0f);
                        smem_gate_all[stage_f32 + 3968 + gate_col] = prefix_log2;
                        float unbounded_cross_anchor = (float)(int)smem_gate_all[stage_f32 + 1920 + gate_col];
                        float unbounded_tile0_half = (float)(int)((smem_gate_all[stage_f32 + gate_col] - smem_gate_all[stage_f32 + 1920 + gate_col]) * 0.5f);
                        float unbounded_tile1_half = fminf((float)(int)((smem_gate_all[stage_f32 + 1920 + gate_col] - smem_gate_all[stage_f32 + 2048 + gate_col] + (smem_gate_all[stage_f32 + 1920 + gate_col] - smem_gate_all[stage_f32 + 3968 + gate_col])) * 0.5f), 126.0f);
                        float unbounded_tile0_anchor = unbounded_cross_anchor + unbounded_tile0_half;
                        float unbounded_tile1_anchor = unbounded_cross_anchor - unbounded_tile1_half;
                        int anchor_pack_base = (unsigned int)prep_ordinal % 4 * 128;
                        int unbounded_tile0_int = (int)unbounded_tile0_anchor;
                        int unbounded_tile1_int = (int)unbounded_tile1_anchor;
                        smem_v25[anchor_pack_base + gate_col] = unbounded_tile0_int & 65535 | unbounded_tile1_int << 16;
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    float page_total_log2 = 0.0f;
                    int _stage_qk_raw_full = (int)((unsigned int)prep_ordinal % 4);
                    int _parity_qk_raw_full = (int)((unsigned int)prep_ordinal / 4 % 2);
                    mbarrier_wait(qk_raw_full_addr + (_stage_qk_raw_full) * 8, _parity_qk_raw_full);
                    if (prep_tid < 128) {
                        float total_log2 = smem_gt_prefix[stage_f32 + prep_tid];
                        float restore_factor_value = 1.0f;
                        float unbounded_cross_anchor_1 = (float)(int)smem_gate_all[stage_f32 + 1920 + prep_tid];
                        float unbounded_tile0_half_1 = (float)(int)((smem_gate_all[stage_f32 + prep_tid] - smem_gate_all[stage_f32 + 1920 + prep_tid]) * 0.5f);
                        float unbounded_tile1_half_1 = fminf((float)(int)((smem_gate_all[stage_f32 + 1920 + prep_tid] - smem_gate_all[stage_f32 + 2048 + prep_tid] + (smem_gate_all[stage_f32 + 1920 + prep_tid] - smem_gate_all[stage_f32 + 3968 + prep_tid])) * 0.5f), 126.0f);
                        float unbounded_tile0_anchor_1 = unbounded_cross_anchor_1 + unbounded_tile0_half_1;
                        float unbounded_tile1_anchor_1 = unbounded_cross_anchor_1 - unbounded_tile1_half_1;
                        int unbounded_cross_magnitude = ((-(int)unbounded_cross_anchor_1) > (0) ? (-(int)unbounded_cross_anchor_1) : (0));
                        int unbounded_row_scale_bits = (127 - (int)unbounded_tile1_half_1 << 7) + (unbounded_cross_magnitude & 127);
                        int unbounded_col_scale_bits = (127 - (int)unbounded_tile0_half_1 << 7) + (unbounded_cross_magnitude >> 7 & 127);
                        int unbounded_anchor_word = unbounded_row_scale_bits + (unbounded_col_scale_bits << 16);
                        smem_v21[stage_f32 + prep_tid] = unbounded_anchor_word;
                    }
                    if (prep_tid == 0) {
                        smem_restore_factor_all[stage_f32 + 128] = 1.0f;
                    }
                    float unbounded_anchor_lo[8];
                    float unbounded_anchor_hi[8];
                    unsigned int anchor_words[8];
                    int anchor_segment = prep_tid % 16;
                    int anchor_stage_base = (unsigned int)prep_ordinal % 4 * 128;
                    int anchor_addr = smem_v25_addr + (unsigned int)((anchor_stage_base + anchor_segment * 8) * 4);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&anchor_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&anchor_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&anchor_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&anchor_words[(0) + 3]))
                        : "r"(anchor_addr));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&anchor_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&anchor_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&anchor_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&anchor_words[(4) + 3]))
                        : "r"(anchor_addr + 16));
                    #pragma unroll
                    for (int anchor_elem = 0; anchor_elem < 8; anchor_elem++) {
                        int anchor_word = (int)anchor_words[anchor_elem];
                        int unbounded_anchor_lo_int = anchor_word << 16 >> 16;
                        unbounded_anchor_lo[anchor_elem] = __uint_as_float((unsigned int)(unbounded_anchor_lo_int + 1262485504)) - 1.2582912e+07f;
                        int unbounded_anchor_lo_int_0 = anchor_word >> 16;
                        unbounded_anchor_hi[anchor_elem] = __uint_as_float((unsigned int)(unbounded_anchor_lo_int_0 + 1262485504)) - 1.2582912e+07f;
                    }
                    #pragma unroll 1
                    for (int work_pass = 0; work_pass < 4; work_pass++) {
                        int work_item = work_pass * 128 + prep_tid;
                        int row = work_item / 16;
                        int segment = work_item % 16;
                        long long token = bos_3 + (long long)(chunk_idx_2 * 32 + row);
                        int token_valid = ((token < eos_3) ? 1 : 0);
                        long long gmem_base = (token * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)(segment * 8);
                        float q_raw_vec[8];
                        float k_raw_vec[8];
                        unsigned int packed[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                            : "r"((smem_q_raw_prefetch_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed[_pair]) << 16);
                            (&unpack_bf16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx = 0; value_idx < 8; value_idx++) {
                            q_raw_vec[value_idx] = unpack_bf16[value_idx];
                        }
                        unsigned int packed_0[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 3]))
                            : "r"((smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_0[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_0[_pair]) << 16);
                            (&unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_0[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx_1 = 0; value_idx_1 < 8; value_idx_1++) {
                            k_raw_vec[value_idx_1] = unpack_bf16_0[value_idx_1];
                        }
                        float2 q_sum_pair = make_float2(0.0f, 0.0f);
                        float2 k_sum_pair = make_float2(0.0f, 0.0f);
                        for (int elem_pair = 0; elem_pair < 4; elem_pair++) {
                            float2 q_raw_pair = make_float2(q_raw_vec[elem_pair * 2], q_raw_vec[elem_pair * 2 + 1]);
                            float2 k_raw_pair = make_float2(k_raw_vec[elem_pair * 2], k_raw_vec[elem_pair * 2 + 1]);
                            q_sum_pair = fma_f32x2_rn_ftz(q_raw_pair, q_raw_pair, q_sum_pair);
                            k_sum_pair = fma_f32x2_rn_ftz(k_raw_pair, k_raw_pair, k_sum_pair);
                        }
                        float q_sum = q_sum_pair.x + q_sum_pair.y;
                        float k_sum = k_sum_pair.x + k_sum_pair.y;
                        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 8);
                        q_sum += _shfl_xor_sync_10;
                        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 8);
                        k_sum += _shfl_xor_sync_11;
                        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 4);
                        q_sum += _shfl_xor_sync_12;
                        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 4);
                        k_sum += _shfl_xor_sync_13;
                        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 2);
                        q_sum += _shfl_xor_sync_14;
                        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 2);
                        k_sum += _shfl_xor_sync_15;
                        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 1);
                        q_sum += _shfl_xor_sync_16;
                        float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 1);
                        k_sum += _shfl_xor_sync_17;
                        float q_inv = rsqrtf(q_sum + 1e-06f);
                        float k_inv = rsqrtf(k_sum + 1e-06f);
                        const float2 _scale2_18 = {q_inv, q_inv};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(q_raw_vec)[_ls], _scale2_18);
                        const float2 _scale2_19 = {k_inv, k_inv};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(k_raw_vec)[_ls], _scale2_19);
                        q_raw_vec[0] = (float)(__nv_bfloat16)q_raw_vec[0];
                        k_raw_vec[0] = (float)(__nv_bfloat16)k_raw_vec[0];
                        q_raw_vec[1] = (float)(__nv_bfloat16)q_raw_vec[1];
                        k_raw_vec[1] = (float)(__nv_bfloat16)k_raw_vec[1];
                        q_raw_vec[2] = (float)(__nv_bfloat16)q_raw_vec[2];
                        k_raw_vec[2] = (float)(__nv_bfloat16)k_raw_vec[2];
                        q_raw_vec[3] = (float)(__nv_bfloat16)q_raw_vec[3];
                        k_raw_vec[3] = (float)(__nv_bfloat16)k_raw_vec[3];
                        q_raw_vec[4] = (float)(__nv_bfloat16)q_raw_vec[4];
                        k_raw_vec[4] = (float)(__nv_bfloat16)k_raw_vec[4];
                        q_raw_vec[5] = (float)(__nv_bfloat16)q_raw_vec[5];
                        k_raw_vec[5] = (float)(__nv_bfloat16)k_raw_vec[5];
                        q_raw_vec[6] = (float)(__nv_bfloat16)q_raw_vec[6];
                        k_raw_vec[6] = (float)(__nv_bfloat16)k_raw_vec[6];
                        q_raw_vec[7] = (float)(__nv_bfloat16)q_raw_vec[7];
                        k_raw_vec[7] = (float)(__nv_bfloat16)k_raw_vec[7];
                        float qd_vec[8];
                        float kd_vec[8];
                        float ki_vec[8];
                        for (int elem_in_segment = 0; elem_in_segment < 8; elem_in_segment++) {
                            int col = segment * 8 + elem_in_segment;
                            float prefix = smem_gate_all[stage_f32 + row * 128 + col];
                            float tile_anchor_log2 = unbounded_anchor_lo[elem_in_segment];
                            if (row >= 16) {
                                tile_anchor_log2 = unbounded_anchor_hi[elem_in_segment];
                            }
                            float decay = approx_exp2(prefix - tile_anchor_log2);
                            qd_vec[elem_in_segment] = decay;
                            kd_vec[elem_in_segment] = decay;
                            ki_vec[elem_in_segment] = k_raw_vec[elem_in_segment] / decay;
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], reinterpret_cast<const float2*>(q_raw_vec)[_ls]);
                        const float2 _scale2_20 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], _scale2_20);
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(kd_vec)[_ls], reinterpret_cast<const float2*>(k_raw_vec)[_ls]);
                        unsigned int packed_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                            packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word = 0; word < 4; word++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word * 4)), "r"(packed_1[word]));
                        }
                        unsigned int packed_2[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_vec[_lp*2 + 0], kd_vec[_lp*2+1 + 0]));
                            packed_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word_1 = 0; word_1 < 4; word_1++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_1 * 4)), "r"(packed_2[word_1]));
                        }
                        unsigned int packed_3[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_vec[_lp*2 + 0], ki_vec[_lp*2+1 + 0]));
                            packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word_2 = 0; word_2 < 4; word_2++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_2 * 4)), "r"(packed_3[word_2]));
                        }
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    unsigned int a_frag[4];
                    unsigned int b_frag[4];
                    float acc[8];
                    int pair_row_base = prep_local_warp / 2 * 16;
                    int pair_col_base = prep_local_warp % 2 * 16;
                    if (pair_row_base >= pair_col_base) {
                        int cross_scale_col = lane % 4 * 2;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0 = smem_v21[stage_f32 + cross_scale_col];
                            int cross_word_lo0_0 = smem_v21[stage_f32 + cross_scale_col + 1];
                            unsigned int cross_a_lo = (unsigned int)(cross_word_lo0 & 65408) + ((unsigned int)(cross_word_lo0_0 & 65408) << 16);
                            unsigned int cross_b_lo = (unsigned int)((cross_word_lo0 >> 16 & 65408) + (cross_word_lo0_0 & 4286578688));
                            int cross_word_lo0_1 = smem_v21[stage_f32 + cross_scale_col + 8];
                            int cross_word_lo0_2 = smem_v21[stage_f32 + cross_scale_col + 9];
                            unsigned int cross_a_lo_0 = (unsigned int)(cross_word_lo0_1 & 65408) + ((unsigned int)(cross_word_lo0_2 & 65408) << 16);
                            unsigned int cross_b_lo_0 = (unsigned int)((cross_word_lo0_1 >> 16 & 65408) + (cross_word_lo0_2 & 4286578688));
                            uint32_t bf16x2_mul;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul) : "r"(a_frag[0]), "r"(cross_a_lo));
                            a_frag[0] = bf16x2_mul;
                            uint32_t bf16x2_mul_1;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1) : "r"(a_frag[1]), "r"(cross_a_lo));
                            a_frag[1] = bf16x2_mul_1;
                            uint32_t bf16x2_mul_2;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2) : "r"(a_frag[2]), "r"(cross_a_lo_0));
                            a_frag[2] = bf16x2_mul_2;
                            uint32_t bf16x2_mul_3;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(a_frag[3]), "r"(cross_a_lo_0));
                            a_frag[3] = bf16x2_mul_3;
                            uint32_t bf16x2_mul_4;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(b_frag[0]), "r"(cross_b_lo));
                            b_frag[0] = bf16x2_mul_4;
                            uint32_t bf16x2_mul_5;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(b_frag[1]), "r"(cross_b_lo_0));
                            b_frag[1] = bf16x2_mul_5;
                            uint32_t bf16x2_mul_6;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(b_frag[2]), "r"(cross_b_lo));
                            b_frag[2] = bf16x2_mul_6;
                            uint32_t bf16x2_mul_7;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(b_frag[3]), "r"(cross_b_lo_0));
                            b_frag[3] = bf16x2_mul_7;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_3 = smem_v21[stage_f32 + 16 + cross_scale_col];
                            int cross_word_lo0_3_1 = smem_v21[stage_f32 + 16 + cross_scale_col + 1];
                            unsigned int cross_a_lo_1 = (unsigned int)(cross_word_lo0_3 & 65408) + ((unsigned int)(cross_word_lo0_3_1 & 65408) << 16);
                            unsigned int cross_b_lo_1 = (unsigned int)((cross_word_lo0_3 >> 16 & 65408) + (cross_word_lo0_3_1 & 4286578688));
                            int cross_word_lo0_4 = smem_v21[stage_f32 + 16 + cross_scale_col + 8];
                            int cross_word_lo0_5 = smem_v21[stage_f32 + 16 + cross_scale_col + 9];
                            unsigned int cross_a_lo_1_1 = (unsigned int)(cross_word_lo0_4 & 65408) + ((unsigned int)(cross_word_lo0_5 & 65408) << 16);
                            unsigned int cross_b_lo_1_1 = (unsigned int)((cross_word_lo0_4 >> 16 & 65408) + (cross_word_lo0_5 & 4286578688));
                            uint32_t bf16x2_mul_8;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(a_frag[0]), "r"(cross_a_lo_1));
                            a_frag[0] = bf16x2_mul_8;
                            uint32_t bf16x2_mul_1_1;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(a_frag[1]), "r"(cross_a_lo_1));
                            a_frag[1] = bf16x2_mul_1_1;
                            uint32_t bf16x2_mul_2_1;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_1) : "r"(a_frag[2]), "r"(cross_a_lo_1_1));
                            a_frag[2] = bf16x2_mul_2_1;
                            uint32_t bf16x2_mul_3_1;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_1) : "r"(a_frag[3]), "r"(cross_a_lo_1_1));
                            a_frag[3] = bf16x2_mul_3_1;
                            uint32_t bf16x2_mul_4_1;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_1) : "r"(b_frag[0]), "r"(cross_b_lo_1));
                            b_frag[0] = bf16x2_mul_4_1;
                            uint32_t bf16x2_mul_5_1;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_1) : "r"(b_frag[1]), "r"(cross_b_lo_1_1));
                            b_frag[1] = bf16x2_mul_5_1;
                            uint32_t bf16x2_mul_6_1;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_1) : "r"(b_frag[2]), "r"(cross_b_lo_1));
                            b_frag[2] = bf16x2_mul_6_1;
                            uint32_t bf16x2_mul_7_1;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_1) : "r"(b_frag[3]), "r"(cross_b_lo_1_1));
                            b_frag[3] = bf16x2_mul_7_1;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_6 = smem_v21[stage_f32 + 32 + cross_scale_col];
                            int cross_word_lo0_6_1 = smem_v21[stage_f32 + 32 + cross_scale_col + 1];
                            unsigned int cross_a_lo_2 = (unsigned int)(cross_word_lo0_6 & 65408) + ((unsigned int)(cross_word_lo0_6_1 & 65408) << 16);
                            unsigned int cross_b_lo_2 = (unsigned int)((cross_word_lo0_6 >> 16 & 65408) + (cross_word_lo0_6_1 & 4286578688));
                            int cross_word_lo0_7 = smem_v21[stage_f32 + 32 + cross_scale_col + 8];
                            int cross_word_lo0_8 = smem_v21[stage_f32 + 32 + cross_scale_col + 9];
                            unsigned int cross_a_lo_2_1 = (unsigned int)(cross_word_lo0_7 & 65408) + ((unsigned int)(cross_word_lo0_8 & 65408) << 16);
                            unsigned int cross_b_lo_2_1 = (unsigned int)((cross_word_lo0_7 >> 16 & 65408) + (cross_word_lo0_8 & 4286578688));
                            uint32_t bf16x2_mul_9;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(a_frag[0]), "r"(cross_a_lo_2));
                            a_frag[0] = bf16x2_mul_9;
                            uint32_t bf16x2_mul_1_2;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(a_frag[1]), "r"(cross_a_lo_2));
                            a_frag[1] = bf16x2_mul_1_2;
                            uint32_t bf16x2_mul_2_2;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_2) : "r"(a_frag[2]), "r"(cross_a_lo_2_1));
                            a_frag[2] = bf16x2_mul_2_2;
                            uint32_t bf16x2_mul_3_2;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_2) : "r"(a_frag[3]), "r"(cross_a_lo_2_1));
                            a_frag[3] = bf16x2_mul_3_2;
                            uint32_t bf16x2_mul_4_2;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_2) : "r"(b_frag[0]), "r"(cross_b_lo_2));
                            b_frag[0] = bf16x2_mul_4_2;
                            uint32_t bf16x2_mul_5_2;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_2) : "r"(b_frag[1]), "r"(cross_b_lo_2_1));
                            b_frag[1] = bf16x2_mul_5_2;
                            uint32_t bf16x2_mul_6_2;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_2) : "r"(b_frag[2]), "r"(cross_b_lo_2));
                            b_frag[2] = bf16x2_mul_6_2;
                            uint32_t bf16x2_mul_7_2;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_2) : "r"(b_frag[3]), "r"(cross_b_lo_2_1));
                            b_frag[3] = bf16x2_mul_7_2;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_9 = smem_v21[stage_f32 + 48 + cross_scale_col];
                            int cross_word_lo0_9_1 = smem_v21[stage_f32 + 48 + cross_scale_col + 1];
                            unsigned int cross_a_lo_3 = (unsigned int)(cross_word_lo0_9 & 65408) + ((unsigned int)(cross_word_lo0_9_1 & 65408) << 16);
                            unsigned int cross_b_lo_3 = (unsigned int)((cross_word_lo0_9 >> 16 & 65408) + (cross_word_lo0_9_1 & 4286578688));
                            int cross_word_lo0_10 = smem_v21[stage_f32 + 48 + cross_scale_col + 8];
                            int cross_word_lo0_11 = smem_v21[stage_f32 + 48 + cross_scale_col + 9];
                            unsigned int cross_a_lo_3_1 = (unsigned int)(cross_word_lo0_10 & 65408) + ((unsigned int)(cross_word_lo0_11 & 65408) << 16);
                            unsigned int cross_b_lo_3_1 = (unsigned int)((cross_word_lo0_10 >> 16 & 65408) + (cross_word_lo0_11 & 4286578688));
                            uint32_t bf16x2_mul_10;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(a_frag[0]), "r"(cross_a_lo_3));
                            a_frag[0] = bf16x2_mul_10;
                            uint32_t bf16x2_mul_1_3;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(a_frag[1]), "r"(cross_a_lo_3));
                            a_frag[1] = bf16x2_mul_1_3;
                            uint32_t bf16x2_mul_2_3;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_3) : "r"(a_frag[2]), "r"(cross_a_lo_3_1));
                            a_frag[2] = bf16x2_mul_2_3;
                            uint32_t bf16x2_mul_3_3;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_3) : "r"(a_frag[3]), "r"(cross_a_lo_3_1));
                            a_frag[3] = bf16x2_mul_3_3;
                            uint32_t bf16x2_mul_4_3;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_3) : "r"(b_frag[0]), "r"(cross_b_lo_3));
                            b_frag[0] = bf16x2_mul_4_3;
                            uint32_t bf16x2_mul_5_3;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_3) : "r"(b_frag[1]), "r"(cross_b_lo_3_1));
                            b_frag[1] = bf16x2_mul_5_3;
                            uint32_t bf16x2_mul_6_3;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_3) : "r"(b_frag[2]), "r"(cross_b_lo_3));
                            b_frag[2] = bf16x2_mul_6_3;
                            uint32_t bf16x2_mul_7_3;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_3) : "r"(b_frag[3]), "r"(cross_b_lo_3_1));
                            b_frag[3] = bf16x2_mul_7_3;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_12 = smem_v21[stage_f32 + 64 + cross_scale_col];
                            int cross_word_lo0_12_1 = smem_v21[stage_f32 + 64 + cross_scale_col + 1];
                            unsigned int cross_a_lo_4 = (unsigned int)(cross_word_lo0_12 & 65408) + ((unsigned int)(cross_word_lo0_12_1 & 65408) << 16);
                            unsigned int cross_b_lo_4 = (unsigned int)((cross_word_lo0_12 >> 16 & 65408) + (cross_word_lo0_12_1 & 4286578688));
                            int cross_word_lo0_13 = smem_v21[stage_f32 + 64 + cross_scale_col + 8];
                            int cross_word_lo0_14 = smem_v21[stage_f32 + 64 + cross_scale_col + 9];
                            unsigned int cross_a_lo_4_1 = (unsigned int)(cross_word_lo0_13 & 65408) + ((unsigned int)(cross_word_lo0_14 & 65408) << 16);
                            unsigned int cross_b_lo_4_1 = (unsigned int)((cross_word_lo0_13 >> 16 & 65408) + (cross_word_lo0_14 & 4286578688));
                            uint32_t bf16x2_mul_11;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(a_frag[0]), "r"(cross_a_lo_4));
                            a_frag[0] = bf16x2_mul_11;
                            uint32_t bf16x2_mul_1_4;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(a_frag[1]), "r"(cross_a_lo_4));
                            a_frag[1] = bf16x2_mul_1_4;
                            uint32_t bf16x2_mul_2_4;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_4) : "r"(a_frag[2]), "r"(cross_a_lo_4_1));
                            a_frag[2] = bf16x2_mul_2_4;
                            uint32_t bf16x2_mul_3_4;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_4) : "r"(a_frag[3]), "r"(cross_a_lo_4_1));
                            a_frag[3] = bf16x2_mul_3_4;
                            uint32_t bf16x2_mul_4_4;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_4) : "r"(b_frag[0]), "r"(cross_b_lo_4));
                            b_frag[0] = bf16x2_mul_4_4;
                            uint32_t bf16x2_mul_5_4;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_4) : "r"(b_frag[1]), "r"(cross_b_lo_4_1));
                            b_frag[1] = bf16x2_mul_5_4;
                            uint32_t bf16x2_mul_6_4;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_4) : "r"(b_frag[2]), "r"(cross_b_lo_4));
                            b_frag[2] = bf16x2_mul_6_4;
                            uint32_t bf16x2_mul_7_4;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_4) : "r"(b_frag[3]), "r"(cross_b_lo_4_1));
                            b_frag[3] = bf16x2_mul_7_4;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_15 = smem_v21[stage_f32 + 80 + cross_scale_col];
                            int cross_word_lo0_15_1 = smem_v21[stage_f32 + 80 + cross_scale_col + 1];
                            unsigned int cross_a_lo_5 = (unsigned int)(cross_word_lo0_15 & 65408) + ((unsigned int)(cross_word_lo0_15_1 & 65408) << 16);
                            unsigned int cross_b_lo_5 = (unsigned int)((cross_word_lo0_15 >> 16 & 65408) + (cross_word_lo0_15_1 & 4286578688));
                            int cross_word_lo0_16 = smem_v21[stage_f32 + 80 + cross_scale_col + 8];
                            int cross_word_lo0_17 = smem_v21[stage_f32 + 80 + cross_scale_col + 9];
                            unsigned int cross_a_lo_5_1 = (unsigned int)(cross_word_lo0_16 & 65408) + ((unsigned int)(cross_word_lo0_17 & 65408) << 16);
                            unsigned int cross_b_lo_5_1 = (unsigned int)((cross_word_lo0_16 >> 16 & 65408) + (cross_word_lo0_17 & 4286578688));
                            uint32_t bf16x2_mul_12;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(a_frag[0]), "r"(cross_a_lo_5));
                            a_frag[0] = bf16x2_mul_12;
                            uint32_t bf16x2_mul_1_5;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(a_frag[1]), "r"(cross_a_lo_5));
                            a_frag[1] = bf16x2_mul_1_5;
                            uint32_t bf16x2_mul_2_5;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_5) : "r"(a_frag[2]), "r"(cross_a_lo_5_1));
                            a_frag[2] = bf16x2_mul_2_5;
                            uint32_t bf16x2_mul_3_5;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_5) : "r"(a_frag[3]), "r"(cross_a_lo_5_1));
                            a_frag[3] = bf16x2_mul_3_5;
                            uint32_t bf16x2_mul_4_5;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_5) : "r"(b_frag[0]), "r"(cross_b_lo_5));
                            b_frag[0] = bf16x2_mul_4_5;
                            uint32_t bf16x2_mul_5_5;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_5) : "r"(b_frag[1]), "r"(cross_b_lo_5_1));
                            b_frag[1] = bf16x2_mul_5_5;
                            uint32_t bf16x2_mul_6_5;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_5) : "r"(b_frag[2]), "r"(cross_b_lo_5));
                            b_frag[2] = bf16x2_mul_6_5;
                            uint32_t bf16x2_mul_7_5;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_5) : "r"(b_frag[3]), "r"(cross_b_lo_5_1));
                            b_frag[3] = bf16x2_mul_7_5;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_18 = smem_v21[stage_f32 + 96 + cross_scale_col];
                            int cross_word_lo0_18_1 = smem_v21[stage_f32 + 96 + cross_scale_col + 1];
                            unsigned int cross_a_lo_6 = (unsigned int)(cross_word_lo0_18 & 65408) + ((unsigned int)(cross_word_lo0_18_1 & 65408) << 16);
                            unsigned int cross_b_lo_6 = (unsigned int)((cross_word_lo0_18 >> 16 & 65408) + (cross_word_lo0_18_1 & 4286578688));
                            int cross_word_lo0_19 = smem_v21[stage_f32 + 96 + cross_scale_col + 8];
                            int cross_word_lo0_20 = smem_v21[stage_f32 + 96 + cross_scale_col + 9];
                            unsigned int cross_a_lo_6_1 = (unsigned int)(cross_word_lo0_19 & 65408) + ((unsigned int)(cross_word_lo0_20 & 65408) << 16);
                            unsigned int cross_b_lo_6_1 = (unsigned int)((cross_word_lo0_19 >> 16 & 65408) + (cross_word_lo0_20 & 4286578688));
                            uint32_t bf16x2_mul_13;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(a_frag[0]), "r"(cross_a_lo_6));
                            a_frag[0] = bf16x2_mul_13;
                            uint32_t bf16x2_mul_1_6;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(a_frag[1]), "r"(cross_a_lo_6));
                            a_frag[1] = bf16x2_mul_1_6;
                            uint32_t bf16x2_mul_2_6;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_6) : "r"(a_frag[2]), "r"(cross_a_lo_6_1));
                            a_frag[2] = bf16x2_mul_2_6;
                            uint32_t bf16x2_mul_3_6;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_6) : "r"(a_frag[3]), "r"(cross_a_lo_6_1));
                            a_frag[3] = bf16x2_mul_3_6;
                            uint32_t bf16x2_mul_4_6;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_6) : "r"(b_frag[0]), "r"(cross_b_lo_6));
                            b_frag[0] = bf16x2_mul_4_6;
                            uint32_t bf16x2_mul_5_6;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_6) : "r"(b_frag[1]), "r"(cross_b_lo_6_1));
                            b_frag[1] = bf16x2_mul_5_6;
                            uint32_t bf16x2_mul_6_6;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_6) : "r"(b_frag[2]), "r"(cross_b_lo_6));
                            b_frag[2] = bf16x2_mul_6_6;
                            uint32_t bf16x2_mul_7_6;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_6) : "r"(b_frag[3]), "r"(cross_b_lo_6_1));
                            b_frag[3] = bf16x2_mul_7_6;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_21 = smem_v21[stage_f32 + 112 + cross_scale_col];
                            int cross_word_lo0_21_1 = smem_v21[stage_f32 + 112 + cross_scale_col + 1];
                            unsigned int cross_a_lo_7 = (unsigned int)(cross_word_lo0_21 & 65408) + ((unsigned int)(cross_word_lo0_21_1 & 65408) << 16);
                            unsigned int cross_b_lo_7 = (unsigned int)((cross_word_lo0_21 >> 16 & 65408) + (cross_word_lo0_21_1 & 4286578688));
                            int cross_word_lo0_22 = smem_v21[stage_f32 + 112 + cross_scale_col + 8];
                            int cross_word_lo0_23 = smem_v21[stage_f32 + 112 + cross_scale_col + 9];
                            unsigned int cross_a_lo_7_1 = (unsigned int)(cross_word_lo0_22 & 65408) + ((unsigned int)(cross_word_lo0_23 & 65408) << 16);
                            unsigned int cross_b_lo_7_1 = (unsigned int)((cross_word_lo0_22 >> 16 & 65408) + (cross_word_lo0_23 & 4286578688));
                            uint32_t bf16x2_mul_14;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(a_frag[0]), "r"(cross_a_lo_7));
                            a_frag[0] = bf16x2_mul_14;
                            uint32_t bf16x2_mul_1_7;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(a_frag[1]), "r"(cross_a_lo_7));
                            a_frag[1] = bf16x2_mul_1_7;
                            uint32_t bf16x2_mul_2_7;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_7) : "r"(a_frag[2]), "r"(cross_a_lo_7_1));
                            a_frag[2] = bf16x2_mul_2_7;
                            uint32_t bf16x2_mul_3_7;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_7) : "r"(a_frag[3]), "r"(cross_a_lo_7_1));
                            a_frag[3] = bf16x2_mul_3_7;
                            uint32_t bf16x2_mul_4_7;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_7) : "r"(b_frag[0]), "r"(cross_b_lo_7));
                            b_frag[0] = bf16x2_mul_4_7;
                            uint32_t bf16x2_mul_5_7;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_7) : "r"(b_frag[1]), "r"(cross_b_lo_7_1));
                            b_frag[1] = bf16x2_mul_5_7;
                            uint32_t bf16x2_mul_6_7;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_7) : "r"(b_frag[2]), "r"(cross_b_lo_7));
                            b_frag[2] = bf16x2_mul_6_7;
                            uint32_t bf16x2_mul_7_7;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_7) : "r"(b_frag[3]), "r"(cross_b_lo_7_1));
                            b_frag[3] = bf16x2_mul_7_7;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        int row0 = pair_row_base + lane / 4;
                        int row1 = row0 + 8;
                        int col0 = pair_col_base + lane % 4 * 2;
                        float beta0 = 0.0f;
                        float beta1 = 0.0f;
                        beta0 = smem_prep_beta_all[stage_f32 + row0];
                        beta1 = smem_prep_beta_all[stage_f32 + row1];
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
                        int byte_off = (int)((unsigned int)prep_ordinal % 4) * 41984 + (pair_row_base + seed_lane_row) * 128 + (pair_col_base + seed_lane_col) * 2;
                        int swizzled_off = byte_off ^ (byte_off >> 7 & 7) << 4;
                        int seed_addr = smem_inv_work_addr + (unsigned int)swizzled_off;
                        uint32_t _stmatrix_addr_21 = static_cast<uint32_t>((unsigned long long)seed_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_21), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[3]))
                            : "memory");
                    }
                    if (prep_local_warp == 1) {
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        acc[4] = 0.0f;
                        acc[5] = 0.0f;
                        acc[6] = 0.0f;
                        acc[7] = 0.0f;
                        int row0_1 = lane / 4;
                        int row1_1 = row0_1 + 8;
                        int col0_1 = 16 + lane % 4 * 2;
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
                            int publish_row = 16 + publish_pair * 8 + (lane & 7);
                            int publish_col = 128 + lane / 8 * 8;
                            uint32_t _stmatrix_addr_22 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(publish_col / 64 * 4096 + publish_row * 128 + publish_col % 64 * 2 ^ (publish_col / 64 * 4096 + publish_row * 128 + publish_col % 64 * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                                :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2 + 1]))
                                : "memory");
                        }
                        int cross_scale_col_1 = lane % 4 * 2;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        int cross_word_lo0_24 = smem_v21[stage_f32 + cross_scale_col_1];
                        int cross_word_lo0_24_1 = smem_v21[stage_f32 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_8 = (unsigned int)(cross_word_lo0_24 & 65408) + ((unsigned int)(cross_word_lo0_24_1 & 65408) << 16);
                        unsigned int cross_b_lo_8 = (unsigned int)((cross_word_lo0_24 >> 16 & 65408) + (cross_word_lo0_24_1 & 4286578688));
                        int cross_word_lo0_25 = smem_v21[stage_f32 + cross_scale_col_1 + 8];
                        int cross_word_lo0_26 = smem_v21[stage_f32 + cross_scale_col_1 + 9];
                        unsigned int cross_a_lo_8_1 = (unsigned int)(cross_word_lo0_25 & 65408) + ((unsigned int)(cross_word_lo0_26 & 65408) << 16);
                        unsigned int cross_b_lo_8_1 = (unsigned int)((cross_word_lo0_25 >> 16 & 65408) + (cross_word_lo0_26 & 4286578688));
                        uint32_t bf16x2_mul_15;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(a_frag[0]), "r"(cross_a_lo_8));
                        a_frag[0] = bf16x2_mul_15;
                        uint32_t bf16x2_mul_1_8;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(a_frag[1]), "r"(cross_a_lo_8));
                        a_frag[1] = bf16x2_mul_1_8;
                        uint32_t bf16x2_mul_2_8;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_8) : "r"(a_frag[2]), "r"(cross_a_lo_8_1));
                        a_frag[2] = bf16x2_mul_2_8;
                        uint32_t bf16x2_mul_3_8;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_8) : "r"(a_frag[3]), "r"(cross_a_lo_8_1));
                        a_frag[3] = bf16x2_mul_3_8;
                        uint32_t bf16x2_mul_4_8;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_8) : "r"(b_frag[0]), "r"(cross_b_lo_8));
                        b_frag[0] = bf16x2_mul_4_8;
                        uint32_t bf16x2_mul_5_8;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_8) : "r"(b_frag[1]), "r"(cross_b_lo_8_1));
                        b_frag[1] = bf16x2_mul_5_8;
                        uint32_t bf16x2_mul_6_8;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_8) : "r"(b_frag[2]), "r"(cross_b_lo_8));
                        b_frag[2] = bf16x2_mul_6_8;
                        uint32_t bf16x2_mul_7_8;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_8) : "r"(b_frag[3]), "r"(cross_b_lo_8_1));
                        b_frag[3] = bf16x2_mul_7_8;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        int cross_word_lo0_27 = smem_v21[stage_f32 + 16 + cross_scale_col_1];
                        int cross_word_lo0_28 = smem_v21[stage_f32 + 16 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_9 = (unsigned int)(cross_word_lo0_27 & 65408) + ((unsigned int)(cross_word_lo0_28 & 65408) << 16);
                        unsigned int cross_b_lo_9 = (unsigned int)((cross_word_lo0_27 >> 16 & 65408) + (cross_word_lo0_28 & 4286578688));
                        int cross_word_lo0_29 = smem_v21[stage_f32 + 16 + cross_scale_col_1 + 8];
                        int cross_word_lo0_30 = smem_v21[stage_f32 + 16 + cross_scale_col_1 + 9];
                        unsigned int cross_a_lo_10 = (unsigned int)(cross_word_lo0_29 & 65408) + ((unsigned int)(cross_word_lo0_30 & 65408) << 16);
                        unsigned int cross_b_lo_10 = (unsigned int)((cross_word_lo0_29 >> 16 & 65408) + (cross_word_lo0_30 & 4286578688));
                        uint32_t bf16x2_mul_0;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_0) : "r"(a_frag[0]), "r"(cross_a_lo_9));
                        a_frag[0] = bf16x2_mul_0;
                        uint32_t bf16x2_mul_1_0;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_0) : "r"(a_frag[1]), "r"(cross_a_lo_9));
                        a_frag[1] = bf16x2_mul_1_0;
                        uint32_t bf16x2_mul_2_0;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_0) : "r"(a_frag[2]), "r"(cross_a_lo_10));
                        a_frag[2] = bf16x2_mul_2_0;
                        uint32_t bf16x2_mul_3_0;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_0) : "r"(a_frag[3]), "r"(cross_a_lo_10));
                        a_frag[3] = bf16x2_mul_3_0;
                        uint32_t bf16x2_mul_4_0;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_0) : "r"(b_frag[0]), "r"(cross_b_lo_9));
                        b_frag[0] = bf16x2_mul_4_0;
                        uint32_t bf16x2_mul_5_0;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_0) : "r"(b_frag[1]), "r"(cross_b_lo_10));
                        b_frag[1] = bf16x2_mul_5_0;
                        uint32_t bf16x2_mul_6_0;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_0) : "r"(b_frag[2]), "r"(cross_b_lo_9));
                        b_frag[2] = bf16x2_mul_6_0;
                        uint32_t bf16x2_mul_7_0;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_0) : "r"(b_frag[3]), "r"(cross_b_lo_10));
                        b_frag[3] = bf16x2_mul_7_0;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        int cross_word_lo0_31 = smem_v21[stage_f32 + 32 + cross_scale_col_1];
                        int cross_word_lo0_32 = smem_v21[stage_f32 + 32 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_11 = (unsigned int)(cross_word_lo0_31 & 65408) + ((unsigned int)(cross_word_lo0_32 & 65408) << 16);
                        unsigned int cross_b_lo_11 = (unsigned int)((cross_word_lo0_31 >> 16 & 65408) + (cross_word_lo0_32 & 4286578688));
                        int cross_word_lo0_33 = smem_v21[stage_f32 + 32 + cross_scale_col_1 + 8];
                        int cross_word_lo0_34 = smem_v21[stage_f32 + 32 + cross_scale_col_1 + 9];
                        unsigned int cross_a_lo_12 = (unsigned int)(cross_word_lo0_33 & 65408) + ((unsigned int)(cross_word_lo0_34 & 65408) << 16);
                        unsigned int cross_b_lo_12 = (unsigned int)((cross_word_lo0_33 >> 16 & 65408) + (cross_word_lo0_34 & 4286578688));
                        uint32_t bf16x2_mul_8_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8_1) : "r"(a_frag[0]), "r"(cross_a_lo_11));
                        a_frag[0] = bf16x2_mul_8_1;
                        uint32_t bf16x2_mul_1_1_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1_1) : "r"(a_frag[1]), "r"(cross_a_lo_11));
                        a_frag[1] = bf16x2_mul_1_1_1;
                        uint32_t bf16x2_mul_2_1_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_1_1) : "r"(a_frag[2]), "r"(cross_a_lo_12));
                        a_frag[2] = bf16x2_mul_2_1_1;
                        uint32_t bf16x2_mul_3_1_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_1_1) : "r"(a_frag[3]), "r"(cross_a_lo_12));
                        a_frag[3] = bf16x2_mul_3_1_1;
                        uint32_t bf16x2_mul_4_1_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_1_1) : "r"(b_frag[0]), "r"(cross_b_lo_11));
                        b_frag[0] = bf16x2_mul_4_1_1;
                        uint32_t bf16x2_mul_5_1_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_1_1) : "r"(b_frag[1]), "r"(cross_b_lo_12));
                        b_frag[1] = bf16x2_mul_5_1_1;
                        uint32_t bf16x2_mul_6_1_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_1_1) : "r"(b_frag[2]), "r"(cross_b_lo_11));
                        b_frag[2] = bf16x2_mul_6_1_1;
                        uint32_t bf16x2_mul_7_1_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_1_1) : "r"(b_frag[3]), "r"(cross_b_lo_12));
                        b_frag[3] = bf16x2_mul_7_1_1;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        int cross_word_lo0_35 = smem_v21[stage_f32 + 48 + cross_scale_col_1];
                        int cross_word_lo0_36 = smem_v21[stage_f32 + 48 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_13 = (unsigned int)(cross_word_lo0_35 & 65408) + ((unsigned int)(cross_word_lo0_36 & 65408) << 16);
                        unsigned int cross_b_lo_13 = (unsigned int)((cross_word_lo0_35 >> 16 & 65408) + (cross_word_lo0_36 & 4286578688));
                        int cross_word_lo0_37 = smem_v21[stage_f32 + 48 + cross_scale_col_1 + 8];
                        int cross_word_lo0_38 = smem_v21[stage_f32 + 48 + cross_scale_col_1 + 9];
                        unsigned int cross_a_lo_14 = (unsigned int)(cross_word_lo0_37 & 65408) + ((unsigned int)(cross_word_lo0_38 & 65408) << 16);
                        unsigned int cross_b_lo_14 = (unsigned int)((cross_word_lo0_37 >> 16 & 65408) + (cross_word_lo0_38 & 4286578688));
                        uint32_t bf16x2_mul_9_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9_1) : "r"(a_frag[0]), "r"(cross_a_lo_13));
                        a_frag[0] = bf16x2_mul_9_1;
                        uint32_t bf16x2_mul_1_2_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2_1) : "r"(a_frag[1]), "r"(cross_a_lo_13));
                        a_frag[1] = bf16x2_mul_1_2_1;
                        uint32_t bf16x2_mul_2_2_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_2_1) : "r"(a_frag[2]), "r"(cross_a_lo_14));
                        a_frag[2] = bf16x2_mul_2_2_1;
                        uint32_t bf16x2_mul_3_2_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_2_1) : "r"(a_frag[3]), "r"(cross_a_lo_14));
                        a_frag[3] = bf16x2_mul_3_2_1;
                        uint32_t bf16x2_mul_4_2_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_2_1) : "r"(b_frag[0]), "r"(cross_b_lo_13));
                        b_frag[0] = bf16x2_mul_4_2_1;
                        uint32_t bf16x2_mul_5_2_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_2_1) : "r"(b_frag[1]), "r"(cross_b_lo_14));
                        b_frag[1] = bf16x2_mul_5_2_1;
                        uint32_t bf16x2_mul_6_2_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_2_1) : "r"(b_frag[2]), "r"(cross_b_lo_13));
                        b_frag[2] = bf16x2_mul_6_2_1;
                        uint32_t bf16x2_mul_7_2_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_2_1) : "r"(b_frag[3]), "r"(cross_b_lo_14));
                        b_frag[3] = bf16x2_mul_7_2_1;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                            : "memory");
                        int cross_word_lo0_39 = smem_v21[stage_f32 + 64 + cross_scale_col_1];
                        int cross_word_lo0_40 = smem_v21[stage_f32 + 64 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_15 = (unsigned int)(cross_word_lo0_39 & 65408) + ((unsigned int)(cross_word_lo0_40 & 65408) << 16);
                        unsigned int cross_b_lo_15 = (unsigned int)((cross_word_lo0_39 >> 16 & 65408) + (cross_word_lo0_40 & 4286578688));
                        int cross_word_lo0_41 = smem_v21[stage_f32 + 64 + cross_scale_col_1 + 8];
                        int cross_word_lo0_42 = smem_v21[stage_f32 + 64 + cross_scale_col_1 + 9];
                        unsigned int cross_a_lo_16 = (unsigned int)(cross_word_lo0_41 & 65408) + ((unsigned int)(cross_word_lo0_42 & 65408) << 16);
                        unsigned int cross_b_lo_16 = (unsigned int)((cross_word_lo0_41 >> 16 & 65408) + (cross_word_lo0_42 & 4286578688));
                        uint32_t bf16x2_mul_10_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10_1) : "r"(a_frag[0]), "r"(cross_a_lo_15));
                        a_frag[0] = bf16x2_mul_10_1;
                        uint32_t bf16x2_mul_1_3_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3_1) : "r"(a_frag[1]), "r"(cross_a_lo_15));
                        a_frag[1] = bf16x2_mul_1_3_1;
                        uint32_t bf16x2_mul_2_3_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_3_1) : "r"(a_frag[2]), "r"(cross_a_lo_16));
                        a_frag[2] = bf16x2_mul_2_3_1;
                        uint32_t bf16x2_mul_3_3_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_3_1) : "r"(a_frag[3]), "r"(cross_a_lo_16));
                        a_frag[3] = bf16x2_mul_3_3_1;
                        uint32_t bf16x2_mul_4_3_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_3_1) : "r"(b_frag[0]), "r"(cross_b_lo_15));
                        b_frag[0] = bf16x2_mul_4_3_1;
                        uint32_t bf16x2_mul_5_3_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_3_1) : "r"(b_frag[1]), "r"(cross_b_lo_16));
                        b_frag[1] = bf16x2_mul_5_3_1;
                        uint32_t bf16x2_mul_6_3_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_3_1) : "r"(b_frag[2]), "r"(cross_b_lo_15));
                        b_frag[2] = bf16x2_mul_6_3_1;
                        uint32_t bf16x2_mul_7_3_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_3_1) : "r"(b_frag[3]), "r"(cross_b_lo_16));
                        b_frag[3] = bf16x2_mul_7_3_1;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        int cross_word_lo0_43 = smem_v21[stage_f32 + 80 + cross_scale_col_1];
                        int cross_word_lo0_44 = smem_v21[stage_f32 + 80 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_17 = (unsigned int)(cross_word_lo0_43 & 65408) + ((unsigned int)(cross_word_lo0_44 & 65408) << 16);
                        unsigned int cross_b_lo_17 = (unsigned int)((cross_word_lo0_43 >> 16 & 65408) + (cross_word_lo0_44 & 4286578688));
                        int cross_word_lo0_45 = smem_v21[stage_f32 + 80 + cross_scale_col_1 + 8];
                        int cross_word_lo0_46 = smem_v21[stage_f32 + 80 + cross_scale_col_1 + 9];
                        unsigned int cross_a_lo_18 = (unsigned int)(cross_word_lo0_45 & 65408) + ((unsigned int)(cross_word_lo0_46 & 65408) << 16);
                        unsigned int cross_b_lo_18 = (unsigned int)((cross_word_lo0_45 >> 16 & 65408) + (cross_word_lo0_46 & 4286578688));
                        uint32_t bf16x2_mul_11_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11_1) : "r"(a_frag[0]), "r"(cross_a_lo_17));
                        a_frag[0] = bf16x2_mul_11_1;
                        uint32_t bf16x2_mul_1_4_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4_1) : "r"(a_frag[1]), "r"(cross_a_lo_17));
                        a_frag[1] = bf16x2_mul_1_4_1;
                        uint32_t bf16x2_mul_2_4_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_4_1) : "r"(a_frag[2]), "r"(cross_a_lo_18));
                        a_frag[2] = bf16x2_mul_2_4_1;
                        uint32_t bf16x2_mul_3_4_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_4_1) : "r"(a_frag[3]), "r"(cross_a_lo_18));
                        a_frag[3] = bf16x2_mul_3_4_1;
                        uint32_t bf16x2_mul_4_4_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_4_1) : "r"(b_frag[0]), "r"(cross_b_lo_17));
                        b_frag[0] = bf16x2_mul_4_4_1;
                        uint32_t bf16x2_mul_5_4_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_4_1) : "r"(b_frag[1]), "r"(cross_b_lo_18));
                        b_frag[1] = bf16x2_mul_5_4_1;
                        uint32_t bf16x2_mul_6_4_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_4_1) : "r"(b_frag[2]), "r"(cross_b_lo_17));
                        b_frag[2] = bf16x2_mul_6_4_1;
                        uint32_t bf16x2_mul_7_4_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_4_1) : "r"(b_frag[3]), "r"(cross_b_lo_18));
                        b_frag[3] = bf16x2_mul_7_4_1;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        int cross_word_lo0_47 = smem_v21[stage_f32 + 96 + cross_scale_col_1];
                        int cross_word_lo0_48 = smem_v21[stage_f32 + 96 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_19 = (unsigned int)(cross_word_lo0_47 & 65408) + ((unsigned int)(cross_word_lo0_48 & 65408) << 16);
                        unsigned int cross_b_lo_19 = (unsigned int)((cross_word_lo0_47 >> 16 & 65408) + (cross_word_lo0_48 & 4286578688));
                        int cross_word_lo0_49 = smem_v21[stage_f32 + 96 + cross_scale_col_1 + 8];
                        int cross_word_lo0_50 = smem_v21[stage_f32 + 96 + cross_scale_col_1 + 9];
                        unsigned int cross_a_lo_20 = (unsigned int)(cross_word_lo0_49 & 65408) + ((unsigned int)(cross_word_lo0_50 & 65408) << 16);
                        unsigned int cross_b_lo_20 = (unsigned int)((cross_word_lo0_49 >> 16 & 65408) + (cross_word_lo0_50 & 4286578688));
                        uint32_t bf16x2_mul_12_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12_1) : "r"(a_frag[0]), "r"(cross_a_lo_19));
                        a_frag[0] = bf16x2_mul_12_1;
                        uint32_t bf16x2_mul_1_5_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5_1) : "r"(a_frag[1]), "r"(cross_a_lo_19));
                        a_frag[1] = bf16x2_mul_1_5_1;
                        uint32_t bf16x2_mul_2_5_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_5_1) : "r"(a_frag[2]), "r"(cross_a_lo_20));
                        a_frag[2] = bf16x2_mul_2_5_1;
                        uint32_t bf16x2_mul_3_5_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_5_1) : "r"(a_frag[3]), "r"(cross_a_lo_20));
                        a_frag[3] = bf16x2_mul_3_5_1;
                        uint32_t bf16x2_mul_4_5_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_5_1) : "r"(b_frag[0]), "r"(cross_b_lo_19));
                        b_frag[0] = bf16x2_mul_4_5_1;
                        uint32_t bf16x2_mul_5_5_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_5_1) : "r"(b_frag[1]), "r"(cross_b_lo_20));
                        b_frag[1] = bf16x2_mul_5_5_1;
                        uint32_t bf16x2_mul_6_5_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_5_1) : "r"(b_frag[2]), "r"(cross_b_lo_19));
                        b_frag[2] = bf16x2_mul_6_5_1;
                        uint32_t bf16x2_mul_7_5_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_5_1) : "r"(b_frag[3]), "r"(cross_b_lo_20));
                        b_frag[3] = bf16x2_mul_7_5_1;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        int cross_word_lo0_51 = smem_v21[stage_f32 + 112 + cross_scale_col_1];
                        int cross_word_lo0_52 = smem_v21[stage_f32 + 112 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_21 = (unsigned int)(cross_word_lo0_51 & 65408) + ((unsigned int)(cross_word_lo0_52 & 65408) << 16);
                        unsigned int cross_b_lo_21 = (unsigned int)((cross_word_lo0_51 >> 16 & 65408) + (cross_word_lo0_52 & 4286578688));
                        int cross_word_lo0_53 = smem_v21[stage_f32 + 112 + cross_scale_col_1 + 8];
                        int cross_word_lo0_54 = smem_v21[stage_f32 + 112 + cross_scale_col_1 + 9];
                        unsigned int cross_a_lo_22 = (unsigned int)(cross_word_lo0_53 & 65408) + ((unsigned int)(cross_word_lo0_54 & 65408) << 16);
                        unsigned int cross_b_lo_22 = (unsigned int)((cross_word_lo0_53 >> 16 & 65408) + (cross_word_lo0_54 & 4286578688));
                        uint32_t bf16x2_mul_13_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13_1) : "r"(a_frag[0]), "r"(cross_a_lo_21));
                        a_frag[0] = bf16x2_mul_13_1;
                        uint32_t bf16x2_mul_1_6_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6_1) : "r"(a_frag[1]), "r"(cross_a_lo_21));
                        a_frag[1] = bf16x2_mul_1_6_1;
                        uint32_t bf16x2_mul_2_6_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2_6_1) : "r"(a_frag[2]), "r"(cross_a_lo_22));
                        a_frag[2] = bf16x2_mul_2_6_1;
                        uint32_t bf16x2_mul_3_6_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3_6_1) : "r"(a_frag[3]), "r"(cross_a_lo_22));
                        a_frag[3] = bf16x2_mul_3_6_1;
                        uint32_t bf16x2_mul_4_6_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4_6_1) : "r"(b_frag[0]), "r"(cross_b_lo_21));
                        b_frag[0] = bf16x2_mul_4_6_1;
                        uint32_t bf16x2_mul_5_6_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5_6_1) : "r"(b_frag[1]), "r"(cross_b_lo_22));
                        b_frag[1] = bf16x2_mul_5_6_1;
                        uint32_t bf16x2_mul_6_6_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6_6_1) : "r"(b_frag[2]), "r"(cross_b_lo_21));
                        b_frag[2] = bf16x2_mul_6_6_1;
                        uint32_t bf16x2_mul_7_6_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7_6_1) : "r"(b_frag[3]), "r"(cross_b_lo_22));
                        b_frag[3] = bf16x2_mul_7_6_1;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        int row0_0 = 16 + lane / 4;
                        int row1_0 = row0_0 + 8;
                        int col0_0 = lane % 4 * 2;
                        float mqk_0[8];
                        mqk_0[0] = 0.0f;
                        mqk_0[1] = 0.0f;
                        mqk_0[2] = 0.0f;
                        mqk_0[3] = 0.0f;
                        mqk_0[4] = 0.0f;
                        mqk_0[5] = 0.0f;
                        mqk_0[6] = 0.0f;
                        mqk_0[7] = 0.0f;
                        if (row0_0 >= col0_0) {
                            mqk_0[0] = acc[0];
                        }
                        if (row0_0 >= col0_0 + 1) {
                            mqk_0[1] = acc[1];
                        }
                        if (row1_0 >= col0_0) {
                            mqk_0[2] = acc[2];
                        }
                        if (row1_0 >= col0_0 + 1) {
                            mqk_0[3] = acc[3];
                        }
                        if (row0_0 >= col0_0 + 8) {
                            mqk_0[4] = acc[4];
                        }
                        if (row0_0 >= col0_0 + 9) {
                            mqk_0[5] = acc[5];
                        }
                        if (row1_0 >= col0_0 + 8) {
                            mqk_0[6] = acc[6];
                        }
                        if (row1_0 >= col0_0 + 9) {
                            mqk_0[7] = acc[7];
                        }
                        unsigned int mqk_packed_0[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mqk_0[_lp*2 + 0], mqk_0[_lp*2+1 + 0]));
                            mqk_packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int publish_pair_1 = 0; publish_pair_1 < 2; publish_pair_1++) {
                            int publish_row_1 = publish_pair_1 * 8 + (lane & 7);
                            int publish_col_1 = 144 + lane / 8 * 8;
                            uint32_t _stmatrix_addr_23 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(publish_col_1 / 64 * 4096 + publish_row_1 * 128 + publish_col_1 % 64 * 2 ^ (publish_col_1 / 64 * 4096 + publish_row_1 * 128 + publish_col_1 % 64 * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                                :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_0[publish_pair_1 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_0[publish_pair_1 * 2 + 1]))
                                : "memory");
                        }
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        int row0_1_1 = 16 + lane / 4;
                        int row1_1_1 = row0_1_1 + 8;
                        int col0_1_1 = 16 + lane % 4 * 2;
                        float mqk_1[8];
                        mqk_1[0] = 0.0f;
                        mqk_1[1] = 0.0f;
                        mqk_1[2] = 0.0f;
                        mqk_1[3] = 0.0f;
                        mqk_1[4] = 0.0f;
                        mqk_1[5] = 0.0f;
                        mqk_1[6] = 0.0f;
                        mqk_1[7] = 0.0f;
                        if (row0_1_1 >= col0_1_1) {
                            mqk_1[0] = acc[0];
                        }
                        if (row0_1_1 >= col0_1_1 + 1) {
                            mqk_1[1] = acc[1];
                        }
                        if (row1_1_1 >= col0_1_1) {
                            mqk_1[2] = acc[2];
                        }
                        if (row1_1_1 >= col0_1_1 + 1) {
                            mqk_1[3] = acc[3];
                        }
                        if (row0_1_1 >= col0_1_1 + 8) {
                            mqk_1[4] = acc[4];
                        }
                        if (row0_1_1 >= col0_1_1 + 9) {
                            mqk_1[5] = acc[5];
                        }
                        if (row1_1_1 >= col0_1_1 + 8) {
                            mqk_1[6] = acc[6];
                        }
                        if (row1_1_1 >= col0_1_1 + 9) {
                            mqk_1[7] = acc[7];
                        }
                        unsigned int mqk_packed_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mqk_1[_lp*2 + 0], mqk_1[_lp*2+1 + 0]));
                            mqk_packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int publish_pair_2 = 0; publish_pair_2 < 2; publish_pair_2++) {
                            int publish_row_2 = 16 + publish_pair_2 * 8 + (lane & 7);
                            int publish_col_2 = 144 + lane / 8 * 8;
                            uint32_t _stmatrix_addr_24 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(publish_col_2 / 64 * 4096 + publish_row_2 * 128 + publish_col_2 % 64 * 2 ^ (publish_col_2 / 64 * 4096 + publish_row_2 * 128 + publish_col_2 % 64 * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                                :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_1[publish_pair_2 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_1[publish_pair_2 * 2 + 1]))
                                : "memory");
                        }
                    } else if (prep_local_warp == 2) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        int row0_2 = lane / 4;
                        int row1_2 = row0_2 + 8;
                        int col0_2 = lane % 4 * 2;
                        float mqk_2[8];
                        mqk_2[0] = 0.0f;
                        mqk_2[1] = 0.0f;
                        mqk_2[2] = 0.0f;
                        mqk_2[3] = 0.0f;
                        mqk_2[4] = 0.0f;
                        mqk_2[5] = 0.0f;
                        mqk_2[6] = 0.0f;
                        mqk_2[7] = 0.0f;
                        if (row0_2 >= col0_2) {
                            mqk_2[0] = acc[0];
                        }
                        if (row0_2 >= col0_2 + 1) {
                            mqk_2[1] = acc[1];
                        }
                        if (row1_2 >= col0_2) {
                            mqk_2[2] = acc[2];
                        }
                        if (row1_2 >= col0_2 + 1) {
                            mqk_2[3] = acc[3];
                        }
                        if (row0_2 >= col0_2 + 8) {
                            mqk_2[4] = acc[4];
                        }
                        if (row0_2 >= col0_2 + 9) {
                            mqk_2[5] = acc[5];
                        }
                        if (row1_2 >= col0_2 + 8) {
                            mqk_2[6] = acc[6];
                        }
                        if (row1_2 >= col0_2 + 9) {
                            mqk_2[7] = acc[7];
                        }
                        unsigned int mqk_packed_2[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mqk_2[_lp*2 + 0], mqk_2[_lp*2+1 + 0]));
                            mqk_packed_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int publish_pair_3 = 0; publish_pair_3 < 2; publish_pair_3++) {
                            int publish_row_3 = publish_pair_3 * 8 + (lane & 7);
                            int publish_col_3 = 128 + lane / 8 * 8;
                            uint32_t _stmatrix_addr_25 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(publish_col_3 / 64 * 4096 + publish_row_3 * 128 + publish_col_3 % 64 * 2 ^ (publish_col_3 / 64 * 4096 + publish_row_3 * 128 + publish_col_3 % 64 * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                                :: "r"(_stmatrix_addr_25), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_2[publish_pair_3 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_2[publish_pair_3 * 2 + 1]))
                                : "memory");
                        }
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    long long tape_scaled_base = 0;
                    if (prep_tid < 128) {
                        float total_log2_1 = smem_gt_prefix[stage_f32 + prep_tid];
                        float _exp2_26 = approx_exp2(unbounded_total_log2);
                        smem_gt_all[stage_f32 + prep_tid] = _exp2_26;
                    }
                    int stage_f32_0 = (unsigned int)prep_ordinal % 4 * 10496;
                    int restore_segment = lane & 15;
                    int owner_tile = prep_local_warp / 2;
                    int owner_row_base = ((prep_local_warp == 0) ? 0 : ((prep_local_warp == 1) ? 6 : ((prep_local_warp == 2) ? 16 : 24)));
                    int owner_passes = ((prep_local_warp == 0) ? 3 : ((prep_local_warp == 1) ? 5 : 4));
                    float qk_factor[8];
                    float qk_tail[8];
                    float kr_factor[8];
                    float kr_tail[8];
                    #pragma unroll
                    for (int factor_elem = 0; factor_elem < 8; factor_elem++) {
                        int factor_col = restore_segment * 8 + factor_elem;
                        int unbounded_restore_word = smem_v21[stage_f32_0 + factor_col];
                        int unbounded_restore_anchor_int = -((unbounded_restore_word & 127) + ((unbounded_restore_word >> 16 & 127) << 7)) + (1 - 2 * owner_tile) * (127 - (unbounded_restore_word >> 23 - 16 * owner_tile & 255));
                        float unbounded_restore_anchor = (float)unbounded_restore_anchor_int;
                        float unbounded_kr_log2 = smem_gt_prefix[stage_f32_0 + factor_col] - unbounded_restore_anchor;
                        int _max_27 = ((unbounded_restore_anchor_int) > (-126) ? (unbounded_restore_anchor_int) : (-126));
                        int unbounded_qk_factor_bits = _max_27 + 127 << 23;
                        qk_factor[factor_elem] = __uint_as_float((unsigned int)unbounded_qk_factor_bits);
                        int _min_29 = ((unbounded_restore_anchor_int + 126) < (0) ? (unbounded_restore_anchor_int + 126) : (0));
                        int _max_28 = ((_min_29) > (-126) ? (_min_29) : (-126));
                        int unbounded_qk_factor_bits_0 = _max_28 + 127 << 23;
                        qk_tail[factor_elem] = __uint_as_float((unsigned int)unbounded_qk_factor_bits_0);
                        float _exp2_30 = approx_exp2(max_noftz(unbounded_kr_log2, -126.0f));
                        kr_factor[factor_elem] = _exp2_30;
                        float unbounded_restore_head;
                        asm("max.f32 %0, %1, %2;" : "=f"(unbounded_restore_head) : "f"(unbounded_kr_log2), "f"(-126.0f));
                        float _exp2_31 = approx_exp2(max_noftz(unbounded_kr_log2 - unbounded_restore_head, -126.0f));
                        kr_tail[factor_elem] = _exp2_31;
                    }
                    #pragma unroll 1
                    for (int restore_pass = 0; restore_pass < owner_passes; restore_pass++) {
                        int restore_row = owner_row_base + restore_pass * 2 + (lane >> 4);
                        float restore_qd_values[8];
                        float restore_kd_values[8];
                        float restore_ki_values[8];
                        float restore_kr_values[8];
                        unsigned int packed_4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 3]))
                            : "r"((smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_4[_pair]) << 16);
                            (&unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_4[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx_2 = 0; value_idx_2 < 8; value_idx_2++) {
                            restore_qd_values[value_idx_2] = unpack_bf16_1[value_idx_2];
                        }
                        unsigned int packed_4_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4_1[(0) + 3]))
                            : "r"((smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_1_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_4_1[_pair]) << 16);
                            (&unpack_bf16_1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_4_1[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx_3 = 0; value_idx_3 < 8; value_idx_3++) {
                            restore_kd_values[value_idx_3] = unpack_bf16_1_1[value_idx_3];
                        }
                        unsigned int packed_5[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 3]))
                            : "r"((smem_ki_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_2[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_5[_pair]) << 16);
                            (&unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_5[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx_4 = 0; value_idx_4 < 8; value_idx_4++) {
                            restore_ki_values[value_idx_4] = unpack_bf16_2[value_idx_4];
                        }
                        #pragma unroll
                        for (int restore_elem = 0; restore_elem < 8; restore_elem++) {
                            restore_qd_values[restore_elem] = restore_qd_values[restore_elem] * qk_factor[restore_elem];
                            restore_kd_values[restore_elem] = restore_kd_values[restore_elem] * qk_factor[restore_elem];
                            restore_kr_values[restore_elem] = restore_ki_values[restore_elem] * kr_factor[restore_elem];
                        }
                        #pragma unroll
                        for (int restore_elem_1 = 0; restore_elem_1 < 8; restore_elem_1++) {
                            restore_qd_values[restore_elem_1] = restore_qd_values[restore_elem_1] * qk_tail[restore_elem_1];
                            restore_kd_values[restore_elem_1] = restore_kd_values[restore_elem_1] * qk_tail[restore_elem_1];
                            restore_kr_values[restore_elem_1] = restore_kr_values[restore_elem_1] * kr_tail[restore_elem_1];
                        }
                        unsigned int packed_6[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values[_lp*2 + 0], restore_qd_values[_lp*2+1 + 0]));
                            packed_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word_3 = 0; word_3 < 4; word_3++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_3 * 4)), "r"(packed_6[word_3]));
                        }
                        unsigned int packed_7[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values[_lp*2 + 0], restore_kd_values[_lp*2+1 + 0]));
                            packed_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word_4 = 0; word_4 < 4; word_4++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_4 * 4)), "r"(packed_7[word_4]));
                        }
                        unsigned int packed_8[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values[_lp*2 + 0], restore_kr_values[_lp*2+1 + 0]));
                            packed_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word_5 = 0; word_5 < 4; word_5++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_5 * 4)), "r"(packed_8[word_5]));
                        }
                    }
                    if (prep_local_warp == 0) {
                        int inverse_row = lane;
                        int diag_block = inverse_row / 8;
                        int lane_in_diag = lane & 7;
                        float inv_row[8];
                        unsigned int packed_9[4];
                        int byte_off_1 = (int)((unsigned int)prep_ordinal % 4) * 41984 + inverse_row * 128 + diag_block * 8 * 2;
                        int swizzled_off_1 = byte_off_1 ^ (byte_off_1 >> 7 & 7) << 4;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_9[(0) + 3]))
                            : "r"(smem_inv_work_addr + (unsigned int)swizzled_off_1));
                        float _unpack_bf16_6[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_9[_pair]) << 16);
                            (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_9[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx_5 = 0; value_idx_5 < 8; value_idx_5++) {
                            inv_row[value_idx_5] = _unpack_bf16_6[value_idx_5];
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
                                float pivot = __shfl_sync(0xFFFFFFFF, inv_row[prev_col], pivot_lane);
                                if (lane_in_diag > src_row) {
                                    inv_row[prev_col] = __fmaf_rn(row_scale, pivot, inv_row[prev_col]);
                                }
                            }
                            if (lane_in_diag > src_row) {
                                inv_row[src_row] = row_scale;
                            }
                        }
                        unsigned int packed_0_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv_row[_lp*2 + 0], inv_row[_lp*2+1 + 0]));
                            packed_0_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        int byte_off_1_1 = (int)((unsigned int)prep_ordinal % 4) * 41984 + inverse_row * 128 + diag_block * 8 * 2;
                        int swizzled_off_2 = byte_off_1_1 ^ (byte_off_1_1 >> 7 & 7) << 4;
                        #pragma unroll
                        for (int word_6 = 0; word_6 < 4; word_6++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_inv_work_addr + (unsigned int)swizzled_off_2 + (unsigned int)(word_6 * 4)), "r"(packed_0_1[word_6]));
                        }
                    }
                    if (prep_local_warp < 2) {
                        __syncwarp();
                        mbarrier_arrive(prep_diag_ready_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8);
                        int _stage_prep_diag_ready = (int)((unsigned int)prep_ordinal % 4);
                        int _parity_prep_diag_ready = (int)((unsigned int)prep_ordinal / 4 % 2);
                        mbarrier_wait(prep_diag_ready_addr + (_stage_prep_diag_ready) * 8, _parity_prep_diag_ready);
                    }
                    if (prep_local_warp < 2) {
                        int lane_row = lane & 7;
                        int byte_off_2 = (int)((unsigned int)prep_ordinal % 4) * 41984 + (prep_local_warp * 16 + 8 + lane_row) * 128 + (prep_local_warp * 16 + 8) * 2;
                        int swizzled_off_3 = byte_off_2 ^ (byte_off_2 >> 7 & 7) << 4;
                        int d_addr = smem_inv_work_addr + (unsigned int)swizzled_off_3;
                        int byte_off_0 = (int)((unsigned int)prep_ordinal % 4) * 41984 + (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2;
                        int swizzled_off_0 = byte_off_0 ^ (byte_off_0 >> 7 & 7) << 4;
                        int c_addr = smem_inv_work_addr + (unsigned int)swizzled_off_0;
                        int byte_off_1_2 = (int)((unsigned int)prep_ordinal % 4) * 41984 + (prep_local_warp * 16 + lane_row) * 128 + prep_local_warp * 16 * 2;
                        int swizzled_off_1_1 = byte_off_1_2 ^ (byte_off_1_2 >> 7 & 7) << 4;
                        int a_addr = smem_inv_work_addr + (unsigned int)swizzled_off_1_1;
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
                        const float2 _scale2_32 = {-1.0f, -1.0f};
                        #pragma unroll
                        for (int _ls = 0; _ls < 2; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(dc_acc)[_ls], _scale2_32);
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
                        int byte_off_2_1 = (int)((unsigned int)prep_ordinal % 4) * 41984 + (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2;
                        int swizzled_off_2_1 = byte_off_2_1 ^ (byte_off_2_1 >> 7 & 7) << 4;
                        int o_addr = smem_inv_work_addr + (unsigned int)swizzled_off_2_1;
                        __syncwarp();
                        uint32_t _stmatrix_addr_33 = static_cast<uint32_t>((unsigned long long)o_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                            :: "r"(_stmatrix_addr_33), "r"(*reinterpret_cast<const uint32_t*>(&o_bf16[0]))
                            : "memory");
                        __syncwarp();
                        mbarrier_arrive(prep_inv16_ready_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8);
                        int _stage_prep_inv16_ready = (int)((unsigned int)prep_ordinal % 4);
                        int _parity_prep_inv16_ready = (int)((unsigned int)prep_ordinal / 4 % 2);
                        mbarrier_wait(prep_inv16_ready_addr + (_stage_prep_inv16_ready) * 8, _parity_prep_inv16_ready);
                    }
                    if (prep_local_warp == 0) {
                        int lane_row_1 = lane % 16;
                        int lane_col = lane / 16 * 8;
                        int byte_off_3 = (int)((unsigned int)prep_ordinal % 4) * 41984 + (16 + lane_row_1) * 128 + (16 + lane_col) * 2;
                        int swizzled_off_4 = byte_off_3 ^ (byte_off_3 >> 7 & 7) << 4;
                        int d_addr_1 = smem_inv_work_addr + (unsigned int)swizzled_off_4;
                        int byte_off_3_1 = (int)((unsigned int)prep_ordinal % 4) * 41984 + (16 + lane_row_1) * 128 + lane_col * 2;
                        int swizzled_off_3_1 = byte_off_3_1 ^ (byte_off_3_1 >> 7 & 7) << 4;
                        int c_addr_1 = smem_inv_work_addr + (unsigned int)swizzled_off_3_1;
                        int byte_off_4 = (int)((unsigned int)prep_ordinal % 4) * 41984 + lane_row_1 * 128 + lane_col * 2;
                        int swizzled_off_4_1 = byte_off_4 ^ (byte_off_4 >> 7 & 7) << 4;
                        int a_addr_1 = smem_inv_work_addr + (unsigned int)swizzled_off_4_1;
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
                        int d_publish_addr = (smem_inv_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((16 + lane_col) / 16 * 1024 + (16 + lane_row_1) * 32 + (16 + lane_col) % 16 * 2 ^ ((16 + lane_col) / 16 * 1024 + (16 + lane_row_1) * 32 + (16 + lane_col) % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_34 = static_cast<uint32_t>((unsigned long long)d_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[3]))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(c32_frag[0]), "=r"(c32_frag[1]), "=r"(c32_frag[2]), "=r"(c32_frag[3])
                            : "r"(c_addr_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(dc32_acc[0]), "=f"(dc32_acc[1]), "=f"(dc32_acc[2]), "=f"(dc32_acc[3])
                            : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]), "r"(c32_frag[0]), "r"(c32_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(dc32_acc[4]), "=f"(dc32_acc[(4) + 1]), "=f"(dc32_acc[(4) + 2]), "=f"(dc32_acc[(4) + 3])
                            : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]), "r"(c32_frag[2]), "r"(c32_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        const float2 _scale2_35 = {-1.0f, -1.0f};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(dc32_acc)[_ls], _scale2_35);
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dc32_acc[_lp*2 + 0], dc32_acc[_lp*2+1 + 0]));
                            dc32_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a32_frag[0]), "=r"(a32_frag[1]), "=r"(a32_frag[2]), "=r"(a32_frag[3])
                            : "r"(a_addr_1)
                            : "memory");
                        int a_publish_addr = (smem_inv_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(lane_col / 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_36 = static_cast<uint32_t>((unsigned long long)a_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_36), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[3]))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(o32_acc[0]), "=f"(o32_acc[1]), "=f"(o32_acc[2]), "=f"(o32_acc[3])
                            : "r"(dc32_bf16[0]), "r"(dc32_bf16[1]), "r"(dc32_bf16[2]), "r"(dc32_bf16[3]), "r"(a32_frag[0]), "r"(a32_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(o32_acc[4]), "=f"(o32_acc[(4) + 1]), "=f"(o32_acc[(4) + 2]), "=f"(o32_acc[(4) + 3])
                            : "r"(dc32_bf16[0]), "r"(dc32_bf16[1]), "r"(dc32_bf16[2]), "r"(dc32_bf16[3]), "r"(a32_frag[2]), "r"(a32_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o32_acc[_lp*2 + 0], o32_acc[_lp*2+1 + 0]));
                            o32_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int o_publish_addr = (smem_inv_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)(lane_col / 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_37 = static_cast<uint32_t>((unsigned long long)o_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_37), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[3]))
                            : "memory");
                        #pragma unroll
                        for (int zero_word = 0; zero_word < 4; zero_word++) {
                            zero32_bf16[zero_word] = 0;
                        }
                        int zero_publish_addr = (smem_inv_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((16 + lane_col) / 16 * 1024 + lane_row_1 * 32 + (16 + lane_col) % 16 * 2 ^ ((16 + lane_col) / 16 * 1024 + lane_row_1 * 32 + (16 + lane_col) % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_38 = static_cast<uint32_t>((unsigned long long)zero_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_38), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[3]))
                            : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    mbarrier_arrive(qk_full_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8);
                    if (chunk_is_full_1 != 0) {
                        if (prep_local_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(v_full_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8, 8192);
                                tma_3d_gmem2smem(smem_v_addr + (unsigned int)prep_ordinal % 4 * 41984, (&v_tma), 0, head_idx_3, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), v_full_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int v_load_iter = 0; v_load_iter < 4; v_load_iter++) {
                            int v_item = v_load_iter * 128 + prep_tid;
                            int row_1 = v_item / 16;
                            int segment_1 = v_item % 16;
                            long long token_1 = bos_3 + (long long)(chunk_idx_2 * 32 + row_1);
                            int token_valid_1 = ((token_1 < eos_3) ? 1 : 0);
                            long long v_src = token_1 * v_stride_m3 + (long long)head_idx_3 * 128 + (long long)(segment_1 * 8);
                            int v_dst = smem_v_addr + (unsigned int)prep_ordinal % 4 * 41984 + (unsigned int)((row_1 * 128 + segment_1 * 8) * 2);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                :: "r"(v_dst), "l"(v + v_src), "r"((token_valid_1 != 0) ? 16 : 0));
                        }
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 0;");
                        asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                        if (prep_local_warp == 0) {
                            if (elect_sync()) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                mbarrier_arrive(v_full_addr + ((int)((unsigned int)prep_ordinal % 4)) * 8);
                            }
                        }
                    }
                    prep_ordinal += 4;
                }
                bin_chunk_base_3 += num_chunks_0_3;
            }
        }
    }

    // Cleanup
}

} // extern "C"
