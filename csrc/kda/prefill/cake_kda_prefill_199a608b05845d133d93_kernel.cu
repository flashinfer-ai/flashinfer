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
#define TMEM_NCOLS 240
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
#define TMEM_TMEM_U2_ACC_OFFSET 208
#define TMEM_TMEM_OUT_OFFSET 192
#define TMEM_TMEM_STATE_OUT_OFFSET 64
#define NUM_CHUNK_PIPE_STAGES 5
#define NUM_OUTPUT_PIPE_STAGES 2
#define NUM_CHECKPOINT_PIPE_STAGES 2
#define NUM_SNAPSHOT_PIPE_STAGES 4
#define NUM_AUX_HANDOFF_PIPE_STAGES 10
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 4096
#define SMEM_SMEM_QD_STRIDE 21504
#define SMEM_SMEM_G_RAW_OFF 1024
#define SMEM_SMEM_G_RAW_STAGE_BYTES 4096
#define SMEM_SMEM_G_RAW_STRIDE 21504
#define SMEM_SMEM_G_RAW_ALL_OFF 1024
#define SMEM_SMEM_G_RAW_ALL_STAGE_BYTES 4096
#define SMEM_SMEM_G_RAW_ALL_STRIDE 21504
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
#define SMEM_SMEM_MQK_TRANS_OFF 129024
#define SMEM_SMEM_MQK_TRANS_STAGE_BYTES 512
#define SMEM_SMEM_MQK_TRANS_STRIDE 5120
#define SMEM_SMEM_FINAL_MQK_SLAB_OFF 13312
#define SMEM_SMEM_FINAL_MQK_SLAB_STAGE_BYTES 2048
#define SMEM_SMEM_FINAL_MQK_SLAB_STRIDE 21504
#define SMEM_SMEM_INV_OFF 129536
#define SMEM_SMEM_INV_STAGE_BYTES 512
#define SMEM_SMEM_INV_STRIDE 5120
#define SMEM_SMEM_V_OFF 108544
#define SMEM_SMEM_V_STAGE_BYTES 4096
#define SMEM_SMEM_V_STRIDE 4096
#define SMEM_SMEM_SHORT_N32_V_OFF 87040
#define SMEM_SMEM_SHORT_N32_V_STAGE_BYTES 16384
#define SMEM_SMEM_SHORT_N32_V_STRIDE 16384
#define SMEM_SMEM_KI_OFF 9216
#define SMEM_SMEM_KI_STAGE_BYTES 4096
#define SMEM_SMEM_KI_STRIDE 21504
#define SMEM_SMEM_GATE_OFF 13312
#define SMEM_SMEM_GATE_STAGE_BYTES 8192
#define SMEM_SMEM_GATE_STRIDE 21504
#define SMEM_SMEM_BETA_RAW_OFF 21504
#define SMEM_SMEM_BETA_RAW_STAGE_BYTES 256
#define SMEM_SMEM_BETA_RAW_STRIDE 21504
#define SMEM_SMEM_BETA_RAW_ALL_OFF 21504
#define SMEM_SMEM_BETA_RAW_ALL_STAGE_BYTES 256
#define SMEM_SMEM_BETA_RAW_ALL_STRIDE 21504
#define SMEM_SMEM_INV_WORK_OFF 130048
#define SMEM_SMEM_INV_WORK_STAGE_BYTES 4096
#define SMEM_SMEM_INV_WORK_STRIDE 5120
#define SMEM_SMEM_OUT_OFF 154624
#define SMEM_SMEM_OUT_STAGE_BYTES 4096
#define SMEM_SMEM_OUT_STRIDE 4096
#define SMEM_SMEM_V18_OFF 154624
#define SMEM_SMEM_V18_STAGE_BYTES 16384
#define SMEM_SMEM_V18_STRIDE 16384
#define SMEM_SMEM_CHECKPOINT_OFF 163840
#define SMEM_SMEM_CHECKPOINT_STAGE_BYTES 32768
#define SMEM_SMEM_CHECKPOINT_STRIDE 32768
#define SMEM_SMEM_FINAL_STATE_STAGE_PANEL_OFF 163840
#define SMEM_SMEM_FINAL_STATE_STAGE_PANEL_STAGE_BYTES 32768
#define SMEM_SMEM_FINAL_STATE_STAGE_PANEL_STRIDE 32768
#define SMEM_SMEM_RESTORE_FACTOR_ALL_OFF 21504
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES 516
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STRIDE 21504
#define SMEM_SMEM_GT_PREFIX_OFF 20992
#define SMEM_SMEM_GT_PREFIX_STAGE_BYTES 512
#define SMEM_SMEM_GT_PREFIX_STRIDE 21504
#define SMEM_SMEM_GT_PREFIX_ALL_OFF 20992
#define SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES 86528
#define SMEM_SMEM_GT_PREFIX_ALL_STRIDE 86528
#define SMEM_SMEM_GT_ALL_OFF 15872
#define SMEM_SMEM_GT_ALL_STAGE_BYTES 512
#define SMEM_SMEM_GT_ALL_STRIDE 21504
#define SMEM_SMEM_PREP_BETA_ALL_OFF 22020
#define SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES 64
#define SMEM_SMEM_PREP_BETA_ALL_STRIDE 21504
#define SMEM_SMEM_PREP_BETA_BF16_ALL_OFF 22020
#define SMEM_SMEM_PREP_BETA_BF16_ALL_STAGE_BYTES 32
#define SMEM_SMEM_PREP_BETA_BF16_ALL_STRIDE 21504
#define SMEM_SMEM_PREP_BETA_U32_ALL_OFF 22020
#define SMEM_SMEM_PREP_BETA_U32_ALL_STAGE_BYTES 32
#define SMEM_SMEM_PREP_BETA_U32_ALL_STRIDE 21504
#define SMEM_SMEM_GATE_RATE_ALL_OFF 22084
#define SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES 4
#define SMEM_SMEM_GATE_RATE_ALL_STRIDE 21504
#define SMEM_SMEM_GATE_BIAS_ALL_OFF 162896
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
#define SMEM_SMEM_V_ALL_OFF 108544
#define SMEM_SMEM_V_ALL_STAGE_BYTES 4096
#define SMEM_SMEM_V_ALL_STRIDE 4096
#define SMEM_SMEM_GATE_ALL_OFF 13312
#define SMEM_SMEM_GATE_ALL_STAGE_BYTES 8192
#define SMEM_SMEM_GATE_ALL_STRIDE 21504
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_OFF 162816
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STAGE_BYTES 80
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STRIDE 80
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_OFF 162816
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_STAGE_BYTES 16
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_STRIDE 16
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_OFF 162832
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_STAGE_BYTES 4
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_STRIDE 4
#define SMEM_SMEM_WORK_ITEM_RESOLVED_OFF 162836
#define SMEM_SMEM_WORK_ITEM_RESOLVED_STAGE_BYTES 4
#define SMEM_SMEM_WORK_ITEM_RESOLVED_STRIDE 4
#define SMEM_TOTAL 229376
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





__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
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

__device__ __forceinline__ unsigned int __as_u32(__half2 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_kda_prefill_199a608b05845d133d93(__nv_bfloat16* __restrict__ q, const __grid_constant__ CUtensorMap q_tma, __nv_bfloat16* __restrict__ k, const __grid_constant__ CUtensorMap k_tma, __nv_bfloat16* __restrict__ v, const __grid_constant__ CUtensorMap v_tma, __nv_bfloat16* __restrict__ g, const __grid_constant__ CUtensorMap g_tma, __nv_bfloat16* __restrict__ beta, const __grid_constant__ CUtensorMap beta_tma, float* __restrict__ A_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ seq_order, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ out, const __grid_constant__ CUtensorMap out_tma, __nv_bfloat16* __restrict__ final_state, int num_heads, int use_initial_state, int store_final_state, float scale, float lower_bound, unsigned long long state_indices_addr, unsigned long long state_checkpoints_addr, unsigned long long checkpoint_cu_starts_addr, long long beta_token_stride, long long state_slot_stride, int use_state_indices, int checkpoint_every_n_tokens, long long* __restrict__ cu_chunk_offsets, __nv_bfloat16* __restrict__ chunk_state, unsigned int* __restrict__ state_checkpoint_needed, __nv_bfloat16* __restrict__ tape_qd, __nv_bfloat16* __restrict__ tape_kd, __nv_bfloat16* __restrict__ tape_kr, __nv_bfloat16* __restrict__ tape_j, float* __restrict__ tape_restore_factor, __nv_bfloat16* __restrict__ tape_e, __nv_bfloat16* __restrict__ tape_x, __nv_bfloat16* __restrict__ tape_r, float* __restrict__ norm_inv_out, __nv_bfloat16* __restrict__ decay_out, float* __restrict__ beta_active_out, float* __restrict__ initial_state_f32, unsigned int* __restrict__ zero_workspace, int zero_words, int num_sequences, const __grid_constant__ CUtensorMap state_checkpoints_tma, float* __restrict__ final_state_f32, long long g_token_stride, long long q_stride_m3, long long k_stride_m3, long long v_stride_m3)
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
    #define gate_raw_full_addr (mbar_base + 40)
    #define qk_raw_full_addr (mbar_base + 80)
    #define v_full_addr (mbar_base + 120)
    #define v_free_addr (mbar_base + 160)
    #define smem_free_addr (mbar_base + 200)
    #define raw_inputs_free_addr (mbar_base + 240)
    #define state_inp_ready_addr (mbar_base + 280)
    #define state_inp_left_ready_addr (mbar_base + 320)
    #define old_out_ready_addr (mbar_base + 360)
    #define u_inp_ready_addr (mbar_base + 400)
    #define u2_acc_ready_addr (mbar_base + 440)
    #define u2_inp_ready_addr (mbar_base + 480)
    #define final_ready_addr (mbar_base + 520)
    #define out_empty_addr (mbar_base + 560)
    #define tmem_dealloc_ready_addr (mbar_base + 568)
    #define checkpoint_ready_addr (mbar_base + 576)
    #define checkpoint_free_addr (mbar_base + 592)
    #define checkpoint_snapshot_done_addr (mbar_base + 608)
    #define work_item_ready_addr (mbar_base + 640)
    #define aux_pairwise_inputs_ready_addr (mbar_base + 648)
    #define aux_pairwise_consumed_addr (mbar_base + 728)
    #define aux_inverse_ready_addr (mbar_base + 808)
    #define short_beta_ready_addr (mbar_base + 888)

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
    __nv_bfloat16* smem_kd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 5120);
    const int smem_kd_addr = smem + 5120;
    __nv_bfloat16* smem_q_raw_prefetch = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_q_raw_prefetch_addr = smem + 9216;
    __nv_bfloat16* smem_final_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_final_trans_addr = smem + 9216;
    __nv_bfloat16* smem_kr_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kr_trans_addr = smem + 9216;
    __nv_bfloat16* smem_mqk_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 129024);
    const int smem_mqk_trans_addr = smem + 129024;
    __nv_bfloat16* smem_final_mqk_slab = reinterpret_cast<__nv_bfloat16*>(smem_raw + 13312);
    const int smem_final_mqk_slab_addr = smem + 13312;
    __nv_bfloat16* smem_inv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 129536);
    const int smem_inv_addr = smem + 129536;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 108544);
    const int smem_v_addr = smem + 108544;
    __nv_bfloat16* smem_short_n32_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 87040);
    const int smem_short_n32_v_addr = smem + 87040;
    __nv_bfloat16* smem_ki = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_ki_addr = smem + 9216;
    float* smem_gate = reinterpret_cast<float*>(smem_raw + 13312);
    const int smem_gate_addr = smem + 13312;
    __nv_bfloat16* smem_beta_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int smem_beta_raw_addr = smem + 21504;
    __nv_bfloat16* smem_beta_raw_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int smem_beta_raw_all_addr = smem + 21504;
    __nv_bfloat16* smem_inv_work = reinterpret_cast<__nv_bfloat16*>(smem_raw + 130048);
    const int smem_inv_work_addr = smem + 130048;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 154624);
    const int smem_out_addr = smem + 154624;
    __nv_bfloat16* smem_v18 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 154624);
    const int smem_v18_addr = smem + 154624;
    __nv_bfloat16* smem_checkpoint = reinterpret_cast<__nv_bfloat16*>(smem_raw + 163840);
    const int smem_checkpoint_addr = smem + 163840;
    float* smem_final_state_stage_panel = reinterpret_cast<float*>(smem_raw + 163840);
    const int smem_final_state_stage_panel_addr = smem + 163840;
    float* smem_restore_factor_all = reinterpret_cast<float*>(smem_raw + 21504);
    const int smem_restore_factor_all_addr = smem + 21504;
    float* smem_gt_prefix = reinterpret_cast<float*>(smem_raw + 20992);
    const int smem_gt_prefix_addr = smem + 20992;
    float* smem_gt_prefix_all = reinterpret_cast<float*>(smem_raw + 20992);
    const int smem_gt_prefix_all_addr = smem + 20992;
    float* smem_gt_all = reinterpret_cast<float*>(smem_raw + 15872);
    const int smem_gt_all_addr = smem + 15872;
    float* smem_prep_beta_all = reinterpret_cast<float*>(smem_raw + 22020);
    const int smem_prep_beta_all_addr = smem + 22020;
    __nv_bfloat16* smem_prep_beta_bf16_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 22020);
    const int smem_prep_beta_bf16_all_addr = smem + 22020;
    unsigned int* smem_prep_beta_u32_all = reinterpret_cast<unsigned int*>(smem_raw + 22020);
    const int smem_prep_beta_u32_all_addr = smem + 22020;
    float* smem_gate_rate_all = reinterpret_cast<float*>(smem_raw + 22084);
    const int smem_gate_rate_all_addr = smem + 22084;
    float* smem_gate_bias_all = reinterpret_cast<float*>(smem_raw + 162896);
    const int smem_gate_bias_all_addr = smem + 162896;
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
    __nv_bfloat16* smem_v_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 108544);
    const int smem_v_all_addr = smem + 108544;
    float* smem_gate_all = reinterpret_cast<float*>(smem_raw + 13312);
    const int smem_gate_all_addr = smem + 13312;
    unsigned int* smem_state_checkpoint_needed = reinterpret_cast<unsigned int*>(smem_raw + 162816);
    const int smem_state_checkpoint_needed_addr = smem + 162816;
    float* smem_work_item_warp_max = reinterpret_cast<float*>(smem_raw + 162816);
    const int smem_work_item_warp_max_addr = smem + 162816;
    int* smem_work_item_compute_start = reinterpret_cast<int*>(smem_raw + 162832);
    const int smem_work_item_compute_start_addr = smem + 162832;
    unsigned int* smem_work_item_resolved = reinterpret_cast<unsigned int*>(smem_raw + 162836);
    const int smem_work_item_resolved_addr = smem + 162836;

    // Mbarrier init (24 pipeline groups, 0 ordered-sequence groups, 116 barriers)
    // Mbarriers at smem_raw[0..928)

    if (warp == 10) {
        // --- pipeline 'chunk_pipe' ---
        // qk_full: 5 barriers, init_count=128
        // gate_raw_full: 5 barriers, init_count=1
        // qk_raw_full: 5 barriers, init_count=1
        // v_full: 5 barriers, init_count=1
        // v_free: 5 barriers, init_count=128
        // smem_free: 5 barriers, init_count=128
        // raw_inputs_free: 5 barriers, init_count=1
        // state_inp_ready: 5 barriers, init_count=4
        // state_inp_left_ready: 5 barriers, init_count=4
        // old_out_ready: 5 barriers, init_count=1
        // u_inp_ready: 5 barriers, init_count=4
        // u2_acc_ready: 5 barriers, init_count=1
        // u2_inp_ready: 5 barriers, init_count=4
        // final_ready: 5 barriers, init_count=1
        // out_empty: 1 barriers, init_count=1
        // tmem_dealloc_ready: 1 barriers, init_count=2
        // --- pipeline 'checkpoint_pipe' ---
        // checkpoint_ready: 2 barriers, init_count=4
        // checkpoint_free: 2 barriers, init_count=1
        // --- pipeline 'snapshot_pipe' ---
        // checkpoint_snapshot_done: 4 barriers, init_count=4
        // work_item_ready: 1 barriers, init_count=1
        // --- pipeline 'aux_handoff_pipe' ---
        // aux_pairwise_inputs_ready: 10 barriers, init_count=128
        // aux_pairwise_consumed: 10 barriers, init_count=32
        // aux_inverse_ready: 10 barriers, init_count=32
        // --- pipeline 'chunk_pipe' ---
        // short_beta_ready: 5 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_10_0 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(30), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(20), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(5), "r"((uint32_t)(128)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_10_0);
        uint32_t _mbarrier_init_count_10_32 = 4;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(28), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(23), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(18), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(13), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(3), "r"((uint32_t)(1)));
        mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_10_32);
        uint32_t _mbarrier_init_count_10_64 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(27), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(17), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(16), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(12), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(10), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(8), "r"((uint32_t)(2)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(7), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(1), "r"((uint32_t)(4)));
        mbarrier_init(smem + 512 + lane * 8, _mbarrier_init_count_10_64);
        uint32_t _mbarrier_init_count_10_96 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_96) : "r"(lane), "n"(15), "r"((uint32_t)(32)));
        if (lane < 20) {
            mbarrier_init(smem + 768 + lane * 8, _mbarrier_init_count_10_96);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (256 columns, 240 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 928);
    if (warp == 0) {
        int _tmem_hold = smem + 928;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
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
    const int tmem_tmem_u2_acc = taddr + 208;
    const int tmem_tmem_out = taddr + 192;
    const int tmem_tmem_state_out = taddr + 64;
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
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
            unsigned int _phase_checkpoint_free = 1;
            #pragma unroll 1
            for (int task_pos = bin_begin; task_pos < bin_end; task_pos++) {
                int binned_task_idx = seq_order[task_pos];
                int split_compute_start = 0;
                int seq_idx = binned_task_idx / num_heads;
                int head_idx = binned_task_idx % num_heads;
                long long bos = cu_seqlens[seq_idx];
                long long eos = cu_seqlens[seq_idx + 1];
                int num_chunks = ((int)(eos - bos) + 16 - 1) / 16;
                int seq_len = (int)(eos - bos);
                int num_chunks_0 = (seq_len + 16 - 1) / 16;
                int num_pages = 0;
                long long total_chunks = cu_chunk_offsets[num_sequences];
                long long fallback_head = total_chunks * (long long)num_heads + (long long)seq_idx * (long long)num_heads + (long long)head_idx;
                const int tmem_row_base = warp_in_wg * 32 << 16;
                long long state_base = (((long long)seq_idx * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128;
                int state_slot = seq_idx;
                if (use_state_indices != 0) {
                    state_slot = reinterpret_cast<int*>(state_indices_addr)[seq_idx];
                }
                state_base = (long long)state_slot * state_slot_stride + ((long long)head_idx * 128 + (long long)state_row) * 128;
                long long checkpoint_row_start = 0;
                long long checkpoint_single_token = 0;
                if (checkpoint_every_n_tokens != 0) {
                    checkpoint_row_start = reinterpret_cast<long long*>(checkpoint_cu_starts_addr)[seq_idx];
                }
                long long initial_state_base = state_base;
                int initial_state_enabled = (int)(use_initial_state != 0);
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
                    if (initial_state_enabled != 0) {
                        float initial_values[8];
                        #pragma unroll
                        for (int initial_quarter = 0; initial_quarter < 4; initial_quarter++) {
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
                                    : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(initial_state_f32 + (initial_state_base + (long long)(state_col_block * 32) + (long long)(initial_quarter * 8)))) : "memory");
                                initial_values[0 + 0] = __uint_as_float(_ldv8_0_0);
                                initial_values[0 + 1] = __uint_as_float(_ldv8_0_1);
                                initial_values[0 + 2] = __uint_as_float(_ldv8_0_2);
                                initial_values[0 + 3] = __uint_as_float(_ldv8_0_3);
                                initial_values[0 + 4] = __uint_as_float(_ldv8_0_4);
                                initial_values[0 + 5] = __uint_as_float(_ldv8_0_5);
                                initial_values[0 + 6] = __uint_as_float(_ldv8_0_6);
                                initial_values[0 + 7] = __uint_as_float(_ldv8_0_7);
                            }
                            #pragma unroll
                            for (int initial_item = 0; initial_item < 8; initial_item++) {
                                state_frag[initial_quarter * 8 + initial_item] = initial_values[initial_item];
                            }
                        }
                    }
                    tmem_st_x32_f32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block * 32), state_frag);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                #pragma unroll 1
                for (int chunk_idx = 0; chunk_idx < num_chunks_0; chunk_idx++) {
                    int chunk_global_local = chunk_idx;
                    int owned_chunk = chunk_global_local >= 0 && chunk_global_local < num_chunks;
                    int checkpoint_token_entering = chunk_idx * 16;
                    int checkpoint_entering = checkpoint_every_n_tokens != 0 && checkpoint_token_entering % checkpoint_every_n_tokens == 0;
                    if (checkpoint_entering != 0) {
                        mbarrier_wait(checkpoint_free_addr + (compute_checkpoint_stage) * 8, _phase_checkpoint_free);
                    }
                    float state_panel0[32];
                    float state_panel1[32];
                    float state_panel2[32];
                    float state_panel3[32];
                    unsigned int state_packed0[16];
                    unsigned int state_packed1[16];
                    unsigned int state_packed2[16];
                    unsigned int state_packed3[16];
                    int _stage_qk_full = (int)((unsigned int)(bin_chunk_base + chunk_idx) % 5);
                    int _parity_qk_full = (int)((unsigned int)(bin_chunk_base + chunk_idx) / 5 % 2);
                    mbarrier_wait(qk_full_addr + (_stage_qk_full) * 8, _parity_qk_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int state_col_block_1 = 0; state_col_block_1 < 4; state_col_block_1++) {
                        int state_addr = taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_1 * 32);
                        float tmem_load[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(tmem_load[0]), "=f"(tmem_load[1]), "=f"(tmem_load[2]), "=f"(tmem_load[3]), "=f"(tmem_load[4]), "=f"(tmem_load[5]), "=f"(tmem_load[6]), "=f"(tmem_load[7]), "=f"(tmem_load[8]), "=f"(tmem_load[9]), "=f"(tmem_load[10]), "=f"(tmem_load[11]), "=f"(tmem_load[12]), "=f"(tmem_load[13]), "=f"(tmem_load[14]), "=f"(tmem_load[15]), "=f"(tmem_load[16]), "=f"(tmem_load[17]), "=f"(tmem_load[18]), "=f"(tmem_load[19]), "=f"(tmem_load[20]), "=f"(tmem_load[21]), "=f"(tmem_load[22]), "=f"(tmem_load[23]), "=f"(tmem_load[24]), "=f"(tmem_load[25]), "=f"(tmem_load[26]), "=f"(tmem_load[27]), "=f"(tmem_load[28]), "=f"(tmem_load[29]), "=f"(tmem_load[30]), "=f"(tmem_load[31])
                            : "r"(state_addr));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        unsigned int _tmem_load_0_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load[_lp*2 + 0], tmem_load[_lp*2+1 + 0]));
                            _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_1 * 16)), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]), "r"(_tmem_load_0_bf16[4]), "r"(_tmem_load_0_bf16[5]), "r"(_tmem_load_0_bf16[6]), "r"(_tmem_load_0_bf16[7]), "r"(_tmem_load_0_bf16[8]), "r"(_tmem_load_0_bf16[9]), "r"(_tmem_load_0_bf16[10]), "r"(_tmem_load_0_bf16[11]), "r"(_tmem_load_0_bf16[12]), "r"(_tmem_load_0_bf16[13]), "r"(_tmem_load_0_bf16[14]), "r"(_tmem_load_0_bf16[15]));
                        if (checkpoint_entering != 0) {
                            #pragma unroll
                            for (int checkpoint_vec = 0; checkpoint_vec < 4; checkpoint_vec++) {
                                int checkpoint_addr = (smem_checkpoint_addr + compute_checkpoint_stage * 32768 + (unsigned int)((state_col_block_1 * 32 + checkpoint_vec * 8) / 64 * 16384 + state_row * 128 + (state_col_block_1 * 32 + checkpoint_vec * 8) % 64 * 2 ^ ((state_col_block_1 * 32 + checkpoint_vec * 8) / 64 * 16384 + state_row * 128 + (state_col_block_1 * 32 + checkpoint_vec * 8) % 64 * 2 >> 7 & 7) << 4));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(checkpoint_addr), "r"(_tmem_load_0_bf16[checkpoint_vec * 4]), "r"(_tmem_load_0_bf16[checkpoint_vec * 4 + 1]), "r"(_tmem_load_0_bf16[checkpoint_vec * 4 + 2]), "r"(_tmem_load_0_bf16[checkpoint_vec * 4 + 3]) : "memory");
                            }
                        }
                        float state_scale[16];
                        #pragma unroll
                        for (int state_half = 0; state_half < 2; state_half++) {
                            #pragma unroll
                            for (int state_col = 0; state_col < 16; state_col++) {
                                state_scale[state_col] = smem_restore_factor_all[(unsigned int)(bin_chunk_base + chunk_idx) % 5 * 5376 + (unsigned int)(state_col_block_1 * 32) + (unsigned int)(state_half * 16) + (unsigned int)state_col];
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((tmem_load + state_half * 16))[_ls], reinterpret_cast<const float2*>(state_scale)[_ls]);
                        }
                        tmem_st_x32_f32(state_addr, tmem_load);
                        if (state_col_block_1 == 1) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            if (elect_sync()) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(state_inp_left_ready_addr + ((int)((unsigned int)(bin_chunk_base + chunk_idx) % 5)) * 8);
                            }
                        }
                    }
                    int state_tail_addr = taddr + 64 + (unsigned int)tmem_row_base + 96;
                    float state_tail_frag[32];
                    unsigned int state_tail_packed[16];
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (checkpoint_entering != 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(checkpoint_ready_addr + (compute_checkpoint_stage) * 8);
                        }
                    }
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(state_inp_ready_addr + ((int)((unsigned int)(bin_chunk_base + chunk_idx) % 5)) * 8);
                    }
                    if (checkpoint_entering != 0) {
                        compute_checkpoint_stage += 1;
                        if (compute_checkpoint_stage == 2) { compute_checkpoint_stage = 0; _phase_checkpoint_free ^= 1; }
                    }
                    int _stage_v_full = (int)((unsigned int)(bin_chunk_base + chunk_idx) % 5);
                    int _parity_v_full = (int)((unsigned int)(bin_chunk_base + chunk_idx) / 5 % 2);
                    mbarrier_wait(v_full_addr + (_stage_v_full) * 8, _parity_v_full);
                    unsigned int v_prefetch_bits[8];
                    {
                        int v_stage_addr = smem_v_addr + (unsigned int)(bin_chunk_base + chunk_idx) % 5 * 4096;
                        #pragma unroll
                        for (int residual_row_half = 0; residual_row_half < 2; residual_row_half++) {
                            #pragma unroll
                            for (int token_group = 0; token_group < 2; token_group++) {
                                const int v_prefetch_reg_base = residual_row_half * 4 + token_group * 2;
                                int v_ld_matrix = lane / 8 & 1;
                                int v_ld_token = token_group * 8 + (lane & 7);
                                int v_ld_panel = warp_in_wg / 2;
                                int v_ld_row = warp_in_wg % 2 * 32 + residual_row_half * 16 + v_ld_matrix * 8;
                                int v_ld_row_addr = v_stage_addr + v_ld_panel * 16 * 64 * 2 + v_ld_token * 64 * 2;
                                int v_ld_addr = (v_ld_row_addr + (v_ld_row * 2 ^ (v_ld_row_addr >> 7 & 7) << 4));
                                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                    : "=r"(v_prefetch_bits[v_prefetch_reg_base]), "=r"(v_prefetch_bits[v_prefetch_reg_base + 1])
                                    : "r"(v_ld_addr)
                                    : "memory");
                            }
                        }
                    }
                    int _stage_old_out_ready = (int)((unsigned int)(bin_chunk_base + chunk_idx) % 5);
                    int _parity_old_out_ready = (int)((unsigned int)(bin_chunk_base + chunk_idx) / 5 % 2);
                    mbarrier_wait(old_out_ready_addr + (_stage_old_out_ready) * 8, _parity_old_out_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float prediction[16];
                    tmem_ld_x16(&prediction[0], taddr + 224 + (unsigned int)tmem_row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    long long chunk_global_e = cu_chunk_offsets[seq_idx] + (long long)chunk_global_local;
                    long long tape_ex_base = ((chunk_global_e * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 16;
                    #pragma unroll
                    for (int residual_half = 0; residual_half < 1; residual_half++) {
                        float residual_v[16];
                        float residual_beta[16];
                        #pragma unroll
                        for (int residual_col = 0; residual_col < 16; residual_col++) {
                            int token_col = residual_half * 16 + residual_col;
                        }
                        {
                            #pragma unroll
                            for (int residual_row_half_1 = 0; residual_row_half_1 < 2; residual_row_half_1++) {
                                float _tmem_load_1[8];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7]))
                                    : "r"(taddr + 224 + (unsigned int)tmem_row_base + (unsigned int)(residual_row_half_1 * 1048576)));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                unsigned int _tmem_load_1_bf16[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                                    _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                unsigned int residual_packed[4];
                                #pragma unroll
                                for (int token_group_1 = 0; token_group_1 < 2; token_group_1++) {
                                    const int residual_pack_base = token_group_1 * 2;
                                    unsigned int beta_pair = 0;
                                    float beta_scale_lo = 0.0f;
                                    float beta_scale_hi = 0.0f;
                                    beta_pair = smem_prep_beta_u32_all[(unsigned int)(bin_chunk_base + chunk_idx) % 5 * 5376 + (unsigned int)(token_group_1 * 4) + (unsigned int)(lane & 3)];
                                    #pragma unroll
                                    for (int residual_matrix = 0; residual_matrix < 2; residual_matrix++) {
                                        const int residual_pair = residual_pack_base + residual_matrix;
                                        const int v_prefetch_pair = residual_row_half_1 * 4 + residual_pair;
                                        uint32_t _bf16x2_sub_0;
                                        asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_sub_0) : "r"(v_prefetch_bits[v_prefetch_pair]), "r"(_tmem_load_1_bf16[residual_pair]));
                                        unsigned int residual_pair_delta = _bf16x2_sub_0;
                                        uint32_t _bf16x2_mul_0;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(residual_pair_delta), "r"(beta_pair));
                                        residual_packed[residual_pair] = _bf16x2_mul_0;
                                    }
                                }
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x128b.x2.b32"
                                    " [%0], {%1, %2, %3, %4};"
                                    :: "r"(taddr + 224 + (unsigned int)tmem_row_base + (unsigned int)(residual_row_half_1 * 1048576)), "r"(*reinterpret_cast<const uint32_t*>(&residual_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&residual_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&residual_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&residual_packed[3])));
                            }
                        }
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
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(v_free_addr + ((int)((unsigned int)(bin_chunk_base + chunk_idx) % 5)) * 8);
                    if (elect_sync()) {
                        mbarrier_arrive(u_inp_ready_addr + ((int)((unsigned int)(bin_chunk_base + chunk_idx) % 5)) * 8);
                    }
                    int _stage_u2_acc_ready = (int)((unsigned int)(bin_chunk_base + chunk_idx) % 5);
                    int _parity_u2_acc_ready = (int)((unsigned int)(bin_chunk_base + chunk_idx) / 5 % 2);
                    mbarrier_wait(u2_acc_ready_addr + (_stage_u2_acc_ready) * 8, _parity_u2_acc_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float tmem_load_1[16];
                    tmem_ld_x16(&tmem_load_1[0], taddr + 208 + (unsigned int)tmem_row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (STORE_BACKWARD_TAPE != 0 && owned_chunk != 0) {
                        long long chunk_global_r = cu_chunk_offsets[seq_idx] + (long long)chunk_global_local;
                        long long tape_r_base = ((chunk_global_r * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 16;
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
                    unsigned int u2_packed[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_1[_lp*2 + 0], tmem_load_1[_lp*2+1 + 0]));
                        u2_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 224 + (unsigned int)tmem_row_base, (const uint32_t*)u2_packed);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(u2_inp_ready_addr + ((int)((unsigned int)(bin_chunk_base + chunk_idx) % 5)) * 8);
                    }
                    int _stage_final_ready = (int)((unsigned int)(bin_chunk_base + chunk_idx) % 5);
                    int _parity_final_ready = (int)((unsigned int)(bin_chunk_base + chunk_idx) / 5 % 2);
                    mbarrier_wait(final_ready_addr + (_stage_final_ready) * 8, _parity_final_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(smem_free_addr + ((int)((unsigned int)(bin_chunk_base + chunk_idx) % 5)) * 8);
                    }
                }
                mbarrier_wait(checkpoint_free_addr + (compute_checkpoint_stage) * 8, _phase_checkpoint_free);
                if (store_final_state != 0) {
                    long long final_panel_base = state_base - (long long)state_row * 128;
                    int final_panel_row = state_row / 8;
                    int final_panel_col = state_row % 8 * 4;
                    unsigned int final_stage_addr = smem_final_state_stage_panel_addr + compute_checkpoint_stage * 32768;
                    #pragma unroll
                    for (int state_col_block_2 = 0; state_col_block_2 < 4; state_col_block_2++) {
                        float tmem_load_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(tmem_load_2[0]), "=f"(tmem_load_2[1]), "=f"(tmem_load_2[2]), "=f"(tmem_load_2[3]), "=f"(tmem_load_2[4]), "=f"(tmem_load_2[5]), "=f"(tmem_load_2[6]), "=f"(tmem_load_2[7]), "=f"(tmem_load_2[8]), "=f"(tmem_load_2[9]), "=f"(tmem_load_2[10]), "=f"(tmem_load_2[11]), "=f"(tmem_load_2[12]), "=f"(tmem_load_2[13]), "=f"(tmem_load_2[14]), "=f"(tmem_load_2[15]), "=f"(tmem_load_2[16]), "=f"(tmem_load_2[17]), "=f"(tmem_load_2[18]), "=f"(tmem_load_2[19]), "=f"(tmem_load_2[20]), "=f"(tmem_load_2[21]), "=f"(tmem_load_2[22]), "=f"(tmem_load_2[23]), "=f"(tmem_load_2[24]), "=f"(tmem_load_2[25]), "=f"(tmem_load_2[26]), "=f"(tmem_load_2[27]), "=f"(tmem_load_2[28]), "=f"(tmem_load_2[29]), "=f"(tmem_load_2[30]), "=f"(tmem_load_2[31])
                            : "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_2 * 32)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        unsigned int final_half_addr = final_stage_addr + (unsigned int)(state_col_block_2 % 2 * 128 * 32 * 4);
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((final_half_addr + (unsigned int)(state_row * 128 ^ (state_row * 128 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[0])), "r"(__as_u32(tmem_load_2[1])), "r"(__as_u32(tmem_load_2[2])), "r"(__as_u32(tmem_load_2[3])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((final_half_addr + (unsigned int)(state_row * 128 + 16 ^ (state_row * 128 + 16 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[4])), "r"(__as_u32(tmem_load_2[5])), "r"(__as_u32(tmem_load_2[6])), "r"(__as_u32(tmem_load_2[7])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((final_half_addr + (unsigned int)(state_row * 128 + 32 ^ (state_row * 128 + 32 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[8])), "r"(__as_u32(tmem_load_2[9])), "r"(__as_u32(tmem_load_2[10])), "r"(__as_u32(tmem_load_2[11])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((final_half_addr + (unsigned int)(state_row * 128 + 48 ^ (state_row * 128 + 48 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[12])), "r"(__as_u32(tmem_load_2[13])), "r"(__as_u32(tmem_load_2[14])), "r"(__as_u32(tmem_load_2[15])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((final_half_addr + (unsigned int)(state_row * 128 + 64 ^ (state_row * 128 + 64 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[16])), "r"(__as_u32(tmem_load_2[17])), "r"(__as_u32(tmem_load_2[18])), "r"(__as_u32(tmem_load_2[19])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((final_half_addr + (unsigned int)(state_row * 128 + 80 ^ (state_row * 128 + 80 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[20])), "r"(__as_u32(tmem_load_2[21])), "r"(__as_u32(tmem_load_2[22])), "r"(__as_u32(tmem_load_2[23])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((final_half_addr + (unsigned int)(state_row * 128 + 96 ^ (state_row * 128 + 96 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[24])), "r"(__as_u32(tmem_load_2[25])), "r"(__as_u32(tmem_load_2[26])), "r"(__as_u32(tmem_load_2[27])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((final_half_addr + (unsigned int)(state_row * 128 + 112 ^ (state_row * 128 + 112 >> 7 & 7) << 4))), "r"(__as_u32(tmem_load_2[28])), "r"(__as_u32(tmem_load_2[29])), "r"(__as_u32(tmem_load_2[30])), "r"(__as_u32(tmem_load_2[31])) : "memory");
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        float final_words[32];
                        #pragma unroll
                        for (int store_round = 0; store_round < 8; store_round++) {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&final_words[store_round * 4])), "=r"(*reinterpret_cast<uint32_t*>(&final_words[(store_round * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&final_words[(store_round * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&final_words[(store_round * 4) + 3]))
                                : "r"((final_half_addr + (unsigned int)((final_panel_row + store_round * 16) * 128 + final_panel_col * 4 ^ ((final_panel_row + store_round * 16) * 128 + final_panel_col * 4 >> 7 & 7) << 4))));
                        }
                        #pragma unroll
                        for (int store_round_1 = 0; store_round_1 < 8; store_round_1++) {
                            {
                                float4 _v4 = make_float4(final_words[store_round_1 * 4 + 0], final_words[store_round_1 * 4 + 1], final_words[store_round_1 * 4 + 2], final_words[store_round_1 * 4 + 3]);
                                *reinterpret_cast<float4*>((final_state_f32 + (final_panel_base + (long long)(final_panel_row + store_round_1 * 16) * 128 + (long long)(state_col_block_2 * 32) + (long long)final_panel_col)) + 0) = _v4;
                            }
                        }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                }
                bin_chunk_base += num_chunks_0;
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
            unsigned int _phase_work_item_ready_0 = 0;
            unsigned int _phase_checkpoint_ready = 0;
            #pragma unroll 1
            for (int task_pos_1 = bin_begin_1; task_pos_1 < bin_end_1; task_pos_1++) {
                int binned_task_idx_1 = seq_order[task_pos_1];
                int split_compute_start_1 = 0;
                int seq_idx_1 = binned_task_idx_1 / num_heads;
                int head_idx_1 = binned_task_idx_1 % num_heads;
                long long bos_1 = cu_seqlens[seq_idx_1];
                long long eos_1 = cu_seqlens[seq_idx_1 + 1];
                int num_chunks_1 = ((int)(eos_1 - bos_1) + 16 - 1) / 16;
                int seq_len_1 = (int)(eos_1 - bos_1);
                int num_chunks_0_1 = (seq_len_1 + 16 - 1) / 16;
                int warp_id_in_role_1 = warp - 4;
                int epilogue_local_warp = warp_id_in_role_1;
                int warp_in_wg_1 = warp % 4;
                const int tmem_row_base_1 = warp_in_wg_1 * 32 << 16;
                int state_row_1 = warp_in_wg_1 * 32 + lane;
                long long epilogue_checkpoint_row_start = 0;
                long long epilogue_checkpoint_single_token = 0;
                if (checkpoint_every_n_tokens != 0) {
                    epilogue_checkpoint_row_start = reinterpret_cast<long long*>(checkpoint_cu_starts_addr)[seq_idx_1];
                }
                int epilogue_chunks = num_chunks_0_1;
                #pragma unroll 1
                for (int chunk_idx_1 = 0; chunk_idx_1 < epilogue_chunks; chunk_idx_1++) {
                    int checkpoint_token_epilogue = chunk_idx_1 * 16;
                    int checkpoint_entering_epilogue = checkpoint_every_n_tokens != 0 && checkpoint_token_epilogue % checkpoint_every_n_tokens == 0;
                    if (checkpoint_entering_epilogue != 0) {
                        mbarrier_wait(checkpoint_ready_addr + (epilogue_checkpoint_stage) * 8, _phase_checkpoint_ready);
                        if (epilogue_local_warp == 0) {
                            long long checkpoint_idx_epilogue = epilogue_checkpoint_row_start + (long long)(checkpoint_token_epilogue / checkpoint_every_n_tokens);
                            if (elect_sync()) {
                                #pragma unroll
                                for (int checkpoint_segment = 0; checkpoint_segment < 2; checkpoint_segment++) {
                                    tma_store_4d((&state_checkpoints_tma), checkpoint_segment * 64, 0, head_idx_1, (int)checkpoint_idx_epilogue, smem_checkpoint_addr + epilogue_checkpoint_stage * 32768 + (unsigned int)(checkpoint_segment * 128 * 64 * 2));
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            if (elect_sync()) {
                                mbarrier_arrive(checkpoint_free_addr + (epilogue_checkpoint_stage) * 8);
                            }
                        }
                        epilogue_checkpoint_stage += 1;
                        if (epilogue_checkpoint_stage == 2) { epilogue_checkpoint_stage = 0; _phase_checkpoint_ready ^= 1; }
                    }
                    int chunk_is_full = ((seq_len_1 >= (chunk_idx_1 + 1) * 16) ? 1 : 0);
                    if (chunk_is_full != 0) {
                        int _stage_final_ready_1 = (int)((unsigned int)(bin_chunk_base_1 + chunk_idx_1) % 5);
                        int _parity_final_ready_1 = (int)((unsigned int)(bin_chunk_base_1 + chunk_idx_1) / 5 % 2);
                        mbarrier_wait(final_ready_addr + (_stage_final_ready_1) * 8, _parity_final_ready_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_4[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7]))
                            : "r"(taddr + 192 + (unsigned int)tmem_row_base_1));
                        float _tmem_load_5[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7]))
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
                        int out_stage_addr = smem_out_addr + (unsigned int)(bin_chunk_base_1 + chunk_idx_1) % 2 * 4096;
                        #pragma unroll
                        for (int dim_half = 0; dim_half < 2; dim_half++) {
                            unsigned int out_packed[8];
                            if (dim_half == 0) {
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                                    out_packed[_lp] = *(uint32_t*)&_bf2;
                                }
                            } else {
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                                    out_packed[_lp] = *(uint32_t*)&_bf2;
                                }
                            }
                            #pragma unroll
                            for (int token_group_2 = 0; token_group_2 < 1; token_group_2++) {
                                int mtx_idx = lane / 8;
                                int row_addr = lane & 7;
                                int dim_base = epilogue_local_warp * 32 + dim_half * 16 + (mtx_idx & 1) * 8;
                                int token_base = token_group_2 * 16 + mtx_idx / 2 * 8;
                                int token_addr = token_base + row_addr;
                                int token_pair = token_addr / 2;
                                int token_parity = token_addr & 1;
                                int raw_row = token_pair + dim_base / 64 * 8;
                                int raw_col = (dim_base & 63 ^ (token_pair & 3) << 4 ^ token_parity << 3) + token_parity * 64;
                                int stsm_offset = (raw_row * 128 + raw_col) * 2;
                                const int pack_base = token_group_2 * 4;
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
                                tma_store_4d((&out_tma), 0, (int)(bos_1 + (long long)(chunk_idx_1 * 16)), head_idx_1, 0, smem_out_addr + (unsigned int)(bin_chunk_base_1 + chunk_idx_1) % 2 * 4096);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    } else {
                        int _stage_final_ready_2 = (int)((unsigned int)(bin_chunk_base_1 + chunk_idx_1) % 5);
                        int _parity_final_ready_2 = (int)((unsigned int)(bin_chunk_base_1 + chunk_idx_1) / 5 % 2);
                        mbarrier_wait(final_ready_addr + (_stage_final_ready_2) * 8, _parity_final_ready_2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_6[16];
                        tmem_ld_x16(&_tmem_load_6[0], taddr + 192 + (unsigned int)tmem_row_base_1);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (epilogue_local_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(out_empty_addr);
                            }
                        }
                        #pragma unroll
                        for (int token_col_1 = 0; token_col_1 < 16; token_col_1++) {
                            long long out_token = bos_1 + (long long)(chunk_idx_1 * 16 + token_col_1);
                            if (out_token < eos_1) {
                                long long out_idx = (out_token * (long long)num_heads + (long long)head_idx_1) * 128 + (long long)state_row_1;
                                out[out_idx] = _tmem_load_6[token_col_1];
                            }
                        }
                    }
                }
                bin_chunk_base_1 += num_chunks_0_1;
            }
            int warp_id_in_role_2 = warp - 4;
            int epilogue_local_warp_1 = warp_id_in_role_2;
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
        // idle — no tasks assigned
    // ---- Role: aux_mma ----
    } else if (warp >= 10 && warp <= 11) {
        { // aux_mma_main
            int instance_id = (warp - 10) / 1;
            int aux_instance = instance_id;
            int bin_begin_2 = seq_order[bid];
            int bin_end_2 = seq_order[bid + 1];
            int bin_chunk_base_2 = 0;
            #pragma unroll 1
            for (int task_pos_2 = bin_begin_2; task_pos_2 < bin_end_2; task_pos_2++) {
                int binned_task_idx_2 = seq_order[task_pos_2];
                int split_compute_start_2 = 0;
                int seq_idx_2 = binned_task_idx_2 / num_heads;
                int head_idx_2 = binned_task_idx_2 % num_heads;
                long long bos_2 = cu_seqlens[seq_idx_2];
                long long eos_2 = cu_seqlens[seq_idx_2 + 1];
                int num_chunks_2 = ((int)(eos_2 - bos_2) + 16 - 1) / 16;
                int seq_len_2 = (int)(eos_2 - bos_2);
                int num_chunks_0_2 = (seq_len_2 + 16 - 1) / 16;
                int aux_start = (aux_instance - bin_chunk_base_2 % 2 + 2) % 2;
                int num_aux_iters = (num_chunks_0_2 + 1 - aux_start) / 2;
                int aux_ordinal = bin_chunk_base_2 + aux_start;
                #pragma unroll 1
                for (int _aux_iter = 0; _aux_iter < num_aux_iters; _aux_iter++) {
                    int stage_f32 = (unsigned int)aux_ordinal % 5 * 5376;
                    int _stage_aux_pairwise_inputs_ready = (int)((unsigned int)aux_ordinal % 10);
                    int _parity_aux_pairwise_inputs_ready = (int)((unsigned int)aux_ordinal / 10 % 2);
                    mbarrier_wait(aux_pairwise_inputs_ready_addr + (_stage_aux_pairwise_inputs_ready) * 8, _parity_aux_pairwise_inputs_ready);
                    unsigned int a_frag[4];
                    unsigned int b_frag[4];
                    float acc[8];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    float inverse_values[8];
                    int row0 = lane / 4;
                    int row1 = row0 + 8;
                    int col0 = lane % 4 * 2;
                    float beta0 = 0.0f;
                    float beta1 = 0.0f;
                    {
                        __nv_bfloat16 beta0_bf16 = smem_prep_beta_bf16_all[stage_f32 * 2 + row0];
                        __nv_bfloat16 beta1_bf16 = smem_prep_beta_bf16_all[stage_f32 * 2 + row1];
                        beta0 = (float)beta0_bf16;
                        beta1 = (float)beta1_bf16;
                    }
                    float l_values[8];
                    l_values[0] = 0.0f;
                    l_values[1] = 0.0f;
                    l_values[2] = 0.0f;
                    l_values[3] = 0.0f;
                    l_values[4] = 0.0f;
                    l_values[5] = 0.0f;
                    l_values[6] = 0.0f;
                    l_values[7] = 0.0f;
                    if (row0 > col0) {
                        l_values[0] = (float)(__nv_bfloat16)(acc[0] * beta0);
                    }
                    if (row0 > col0 + 1) {
                        l_values[1] = (float)(__nv_bfloat16)(acc[1] * beta0);
                    }
                    if (row1 > col0) {
                        l_values[2] = (float)(__nv_bfloat16)(acc[2] * beta1);
                    }
                    if (row1 > col0 + 1) {
                        l_values[3] = (float)(__nv_bfloat16)(acc[3] * beta1);
                    }
                    if (row0 > col0 + 8) {
                        l_values[4] = (float)(__nv_bfloat16)(acc[4] * beta0);
                    }
                    if (row0 > col0 + 9) {
                        l_values[5] = (float)(__nv_bfloat16)(acc[5] * beta0);
                    }
                    if (row1 > col0 + 8) {
                        l_values[6] = (float)(__nv_bfloat16)(acc[6] * beta1);
                    }
                    if (row1 > col0 + 9) {
                        l_values[7] = (float)(__nv_bfloat16)(acc[7] * beta1);
                    }
                    unsigned int l_frag[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(l_values[_lp*2 + 0], l_values[_lp*2+1 + 0]));
                        l_frag[_lp] = *(uint32_t*)&_h2;
                    }
                    unsigned int allmma_d_frag[4];
                    allmma_d_frag[0] = 0;
                    allmma_d_frag[1] = 0;
                    allmma_d_frag[2] = 0;
                    allmma_d_frag[3] = 0;
                    allmma_d_frag[0] = l_frag[0];
                    allmma_d_frag[3] = l_frag[3];
                    float n_values[8];
                    n_values[0] = 0.0f;
                    n_values[1] = 0.0f;
                    n_values[2] = 0.0f;
                    n_values[3] = 0.0f;
                    n_values[4] = 0.0f;
                    n_values[5] = 0.0f;
                    n_values[6] = 0.0f;
                    n_values[7] = 0.0f;
                    n_values[0] = -l_values[0];
                    n_values[1] = -l_values[1];
                    n_values[6] = -l_values[6];
                    n_values[7] = -l_values[7];
                    if (row0 == col0) {
                        n_values[0] = n_values[0] + 1.0f;
                    }
                    if (row0 == col0 + 1) {
                        n_values[1] = n_values[1] + 1.0f;
                    }
                    if (row1 == col0) {
                        n_values[2] = n_values[2] + 1.0f;
                    }
                    if (row1 == col0 + 1) {
                        n_values[3] = n_values[3] + 1.0f;
                    }
                    if (row0 == col0 + 8) {
                        n_values[4] = n_values[4] + 1.0f;
                    }
                    if (row0 == col0 + 9) {
                        n_values[5] = n_values[5] + 1.0f;
                    }
                    if (row1 == col0 + 8) {
                        n_values[6] = n_values[6] + 1.0f;
                    }
                    if (row1 == col0 + 9) {
                        n_values[7] = n_values[7] + 1.0f;
                    }
                    unsigned int n_frag[4];
                    unsigned int low_word[1];
                    unsigned int high_word[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(n_values[_lp*2 + 0], n_values[_lp*2+1 + 0]));
                        low_word[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(n_values[_lp*2 + 6], n_values[_lp*2+1 + 6]));
                        high_word[_lp] = *(uint32_t*)&_h2;
                    }
                    n_frag[0] = 0;
                    n_frag[1] = 0;
                    n_frag[2] = 0;
                    n_frag[3] = 0;
                    n_frag[0] = low_word[0];
                    n_frag[3] = high_word[0];
                    float product[8];
                    unsigned int rhs_trans_frag[4];
                    #pragma unroll
                    for (int word = 0; word < 4; word++) {
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                            : "=r"(rhs_trans_frag[word])
                            : "r"(allmma_d_frag[word]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                        : "r"(allmma_d_frag[0]), "r"(allmma_d_frag[1]), "r"(allmma_d_frag[2]), "r"(allmma_d_frag[3]), "r"(rhs_trans_frag[0]), "r"(rhs_trans_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                        : "r"(allmma_d_frag[0]), "r"(allmma_d_frag[1]), "r"(allmma_d_frag[2]), "r"(allmma_d_frag[3]), "r"(rhs_trans_frag[2]), "r"(rhs_trans_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    unsigned int d2_frag[4];
                    unsigned int low_word_0[1];
                    unsigned int high_word_0[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(product[_lp*2 + 0], product[_lp*2+1 + 0]));
                        low_word_0[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(product[_lp*2 + 6], product[_lp*2+1 + 6]));
                        high_word_0[_lp] = *(uint32_t*)&_h2;
                    }
                    d2_frag[0] = 0;
                    d2_frag[1] = 0;
                    d2_frag[2] = 0;
                    d2_frag[3] = 0;
                    d2_frag[0] = low_word_0[0];
                    d2_frag[3] = high_word_0[0];
                    unsigned int rhs_trans_frag_0[4];
                    #pragma unroll
                    for (int word_1 = 0; word_1 < 4; word_1++) {
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                            : "=r"(rhs_trans_frag_0[word_1])
                            : "r"(d2_frag[word_1]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                        : "r"(n_frag[0]), "r"(n_frag[1]), "r"(n_frag[2]), "r"(n_frag[3]), "r"(rhs_trans_frag_0[0]), "r"(rhs_trans_frag_0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                        : "r"(n_frag[0]), "r"(n_frag[1]), "r"(n_frag[2]), "r"(n_frag[3]), "r"(rhs_trans_frag_0[2]), "r"(rhs_trans_frag_0[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    #pragma unroll
                    for (int value_idx = 0; value_idx < 8; value_idx++) {
                        n_values[value_idx] = n_values[value_idx] + product[value_idx];
                    }
                    unsigned int low_word_1[1];
                    unsigned int high_word_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(n_values[_lp*2 + 0], n_values[_lp*2+1 + 0]));
                        low_word_1[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(n_values[_lp*2 + 6], n_values[_lp*2+1 + 6]));
                        high_word_1[_lp] = *(uint32_t*)&_h2;
                    }
                    n_frag[0] = 0;
                    n_frag[1] = 0;
                    n_frag[2] = 0;
                    n_frag[3] = 0;
                    n_frag[0] = low_word_1[0];
                    n_frag[3] = high_word_1[0];
                    unsigned int rhs_trans_frag_1[4];
                    #pragma unroll
                    for (int word_2 = 0; word_2 < 4; word_2++) {
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                            : "=r"(rhs_trans_frag_1[word_2])
                            : "r"(d2_frag[word_2]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                        : "r"(d2_frag[0]), "r"(d2_frag[1]), "r"(d2_frag[2]), "r"(d2_frag[3]), "r"(rhs_trans_frag_1[0]), "r"(rhs_trans_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                        : "r"(d2_frag[0]), "r"(d2_frag[1]), "r"(d2_frag[2]), "r"(d2_frag[3]), "r"(rhs_trans_frag_1[2]), "r"(rhs_trans_frag_1[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    unsigned int d4_frag[4];
                    unsigned int low_word_2[1];
                    unsigned int high_word_2[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(product[_lp*2 + 0], product[_lp*2+1 + 0]));
                        low_word_2[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(product[_lp*2 + 6], product[_lp*2+1 + 6]));
                        high_word_2[_lp] = *(uint32_t*)&_h2;
                    }
                    d4_frag[0] = 0;
                    d4_frag[1] = 0;
                    d4_frag[2] = 0;
                    d4_frag[3] = 0;
                    d4_frag[0] = low_word_2[0];
                    d4_frag[3] = high_word_2[0];
                    unsigned int rhs_trans_frag_2[4];
                    #pragma unroll
                    for (int word_3 = 0; word_3 < 4; word_3++) {
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                            : "=r"(rhs_trans_frag_2[word_3])
                            : "r"(d4_frag[word_3]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                        : "r"(n_frag[0]), "r"(n_frag[1]), "r"(n_frag[2]), "r"(n_frag[3]), "r"(rhs_trans_frag_2[0]), "r"(rhs_trans_frag_2[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                        : "r"(n_frag[0]), "r"(n_frag[1]), "r"(n_frag[2]), "r"(n_frag[3]), "r"(rhs_trans_frag_2[2]), "r"(rhs_trans_frag_2[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    #pragma unroll
                    for (int value_idx_1 = 0; value_idx_1 < 8; value_idx_1++) {
                        n_values[value_idx_1] = n_values[value_idx_1] + product[value_idx_1];
                    }
                    unsigned int low_word_3[1];
                    unsigned int high_word_3[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(n_values[_lp*2 + 0], n_values[_lp*2+1 + 0]));
                        low_word_3[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(n_values[_lp*2 + 6], n_values[_lp*2+1 + 6]));
                        high_word_3[_lp] = *(uint32_t*)&_h2;
                    }
                    n_frag[0] = 0;
                    n_frag[1] = 0;
                    n_frag[2] = 0;
                    n_frag[3] = 0;
                    n_frag[0] = low_word_3[0];
                    n_frag[3] = high_word_3[0];
                    unsigned int binv_frag[4];
                    binv_frag[0] = 0;
                    binv_frag[1] = 0;
                    binv_frag[2] = 0;
                    binv_frag[3] = 0;
                    binv_frag[0] = n_frag[0];
                    binv_frag[3] = n_frag[3];
                    unsigned int a21_frag[4];
                    a21_frag[0] = 0;
                    a21_frag[1] = 0;
                    a21_frag[2] = 0;
                    a21_frag[3] = 0;
                    a21_frag[1] = l_frag[1];
                    unsigned int rhs_trans_frag_3[4];
                    #pragma unroll
                    for (int word_4 = 0; word_4 < 4; word_4++) {
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                            : "=r"(rhs_trans_frag_3[word_4])
                            : "r"(a21_frag[word_4]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                        : "r"(binv_frag[0]), "r"(binv_frag[1]), "r"(binv_frag[2]), "r"(binv_frag[3]), "r"(rhs_trans_frag_3[0]), "r"(rhs_trans_frag_3[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                        : "r"(binv_frag[0]), "r"(binv_frag[1]), "r"(binv_frag[2]), "r"(binv_frag[3]), "r"(rhs_trans_frag_3[2]), "r"(rhs_trans_frag_3[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    float correction_values[2];
                    correction_values[0] = -product[2];
                    correction_values[1] = -product[3];
                    unsigned int correction_word[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(correction_values[_lp*2 + 0], correction_values[_lp*2+1 + 0]));
                        correction_word[_lp] = *(uint32_t*)&_h2;
                    }
                    unsigned int correction_frag[4];
                    correction_frag[0] = 0;
                    correction_frag[1] = 0;
                    correction_frag[2] = 0;
                    correction_frag[3] = 0;
                    correction_frag[1] = correction_word[0];
                    unsigned int rhs_trans_frag_4[4];
                    #pragma unroll
                    for (int word_5 = 0; word_5 < 4; word_5++) {
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                            : "=r"(rhs_trans_frag_4[word_5])
                            : "r"(binv_frag[word_5]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                        : "r"(correction_frag[0]), "r"(correction_frag[1]), "r"(correction_frag[2]), "r"(correction_frag[3]), "r"(rhs_trans_frag_4[0]), "r"(rhs_trans_frag_4[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                        : "r"(correction_frag[0]), "r"(correction_frag[1]), "r"(correction_frag[2]), "r"(correction_frag[3]), "r"(rhs_trans_frag_4[2]), "r"(rhs_trans_frag_4[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    n_values[2] = product[2];
                    n_values[3] = product[3];
                    #pragma unroll
                    for (int value_idx_2 = 0; value_idx_2 < 8; value_idx_2++) {
                        inverse_values[value_idx_2] = n_values[value_idx_2];
                    }
                    unsigned int inverse_packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inverse_values[_lp*2 + 0], inverse_values[_lp*2+1 + 0]));
                        inverse_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    int inverse_lane_row = lane % 16;
                    int inverse_lane_col = lane / 16 * 8;
                    int byte_off = (int)((unsigned int)aux_ordinal % 5) * 5120 + inverse_lane_row * 128 + inverse_lane_col * 2;
                    int swizzled_off = byte_off ^ (byte_off >> 7 & 7) << 4;
                    int inverse_work_addr = smem_inv_work_addr + (unsigned int)swizzled_off;
                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)inverse_work_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&inverse_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&inverse_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&inverse_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&inverse_packed[3]))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)aux_ordinal % 5 * 21504 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    int row0_0 = lane / 4;
                    int row1_0 = row0_0 + 8;
                    int col0_0 = lane % 4 * 2;
                    float mqk[8];
                    mqk[0] = 0.0f;
                    mqk[1] = 0.0f;
                    mqk[2] = 0.0f;
                    mqk[3] = 0.0f;
                    mqk[4] = 0.0f;
                    mqk[5] = 0.0f;
                    mqk[6] = 0.0f;
                    mqk[7] = 0.0f;
                    if (row0_0 >= col0_0) {
                        mqk[0] = acc[0];
                    }
                    if (row0_0 >= col0_0 + 1) {
                        mqk[1] = acc[1];
                    }
                    if (row1_0 >= col0_0) {
                        mqk[2] = acc[2];
                    }
                    if (row1_0 >= col0_0 + 1) {
                        mqk[3] = acc[3];
                    }
                    if (row0_0 >= col0_0 + 8) {
                        mqk[4] = acc[4];
                    }
                    if (row0_0 >= col0_0 + 9) {
                        mqk[5] = acc[5];
                    }
                    if (row1_0 >= col0_0 + 8) {
                        mqk[6] = acc[6];
                    }
                    if (row1_0 >= col0_0 + 9) {
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
                        uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)(smem_mqk_trans_addr + (unsigned int)aux_ordinal % 5 * 5120 + (unsigned int)(publish_col / 16 * 512 + publish_row * 32 + publish_col % 16 * 2 ^ (publish_col / 16 * 512 + publish_row * 32 + publish_col % 16 * 2 >> 7 & 1) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2 + 1]))
                            : "memory");
                    }
                    __syncwarp();
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(aux_pairwise_consumed_addr + ((int)((unsigned int)aux_ordinal % 10)) * 8);
                    int lane_row = lane % 16;
                    int lane_col = lane / 16 * 8;
                    int byte_off_0 = (int)((unsigned int)aux_ordinal % 5) * 5120 + lane_row * 128 + lane_col * 2;
                    int swizzled_off_0 = byte_off_0 ^ (byte_off_0 >> 7 & 7) << 4;
                    int inv16_addr = smem_inv_work_addr + (unsigned int)swizzled_off_0;
                    unsigned int inv16_frag[4];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(inv16_frag[0]), "=r"(inv16_frag[1]), "=r"(inv16_frag[2]), "=r"(inv16_frag[3])
                        : "r"(inv16_addr)
                        : "memory");
                    int inv16_publish_addr = (smem_inv_addr + (unsigned int)aux_ordinal % 5 * 5120 + (unsigned int)(lane_col / 16 * 512 + lane_row * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 512 + lane_row * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_2 = static_cast<uint32_t>((unsigned long long)inv16_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&inv16_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv16_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv16_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv16_frag[3]))
                        : "memory");
                    __syncwarp();
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(aux_inverse_ready_addr + ((int)((unsigned int)aux_ordinal % 10)) * 8);
                    aux_ordinal += 2;
                }
                bin_chunk_base_2 += num_chunks_0_2;
            }
            unsigned int _phase_work_item_ready_0_1 = 0;
        }
    // ---- Role: mma ----
    } else if (warp == 9) {
        { // mma_main
            int task_idx_2 = blockIdx.x;
            int bin_begin_3 = seq_order[bid];
            int bin_end_3 = seq_order[bid + 1];
            int bin_chunk_base_3 = 0;
            unsigned int _phase_work_item_ready_0_2 = 0;
            unsigned int _phase_out_empty_0 = 1;
            #pragma unroll 1
            for (int task_pos_3 = bin_begin_3; task_pos_3 < bin_end_3; task_pos_3++) {
                int binned_task_idx_3 = seq_order[task_pos_3];
                int split_compute_start_3 = 0;
                int seq_idx_3 = binned_task_idx_3 / num_heads;
                int head_idx_3 = binned_task_idx_3 % num_heads;
                long long bos_3 = cu_seqlens[seq_idx_3];
                long long eos_3 = cu_seqlens[seq_idx_3 + 1];
                int num_chunks_3 = ((int)(eos_3 - bos_3) + 16 - 1) / 16;
                int seq_len_3 = (int)(eos_3 - bos_3);
                int num_chunks_0_3 = (seq_len_3 + 16 - 1) / 16;
                #pragma unroll 1
                for (int _chunk_idx = 0; _chunk_idx < num_chunks_0_3; _chunk_idx++) {
                    int _stage_qk_full_1 = (int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5);
                    int _parity_qk_full_1 = (int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) / 5 % 2);
                    mbarrier_wait(qk_full_addr + (_stage_qk_full_1) * 8, _parity_qk_full_1);
                    {
                        mbarrier_wait(out_empty_addr, _phase_out_empty_0);
                        _phase_out_empty_0 ^= 1;
                    }
                    int _stage_state_inp_left_ready = (int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5);
                    int _parity_state_inp_left_ready = (int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) / 5 % 2);
                    mbarrier_wait(state_inp_left_ready_addr + (_stage_state_inp_left_ready) * 8, _parity_state_inp_left_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_0 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + ((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5) * 1344);
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
                    "}\n"
                    :: "r"(tmem_tmem_u_acc), "r"(_mma_b_lo_0), "r"(tmem_tmem_state_inp), "r"(0) : "memory");
                    int _stage_state_inp_ready = (int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5);
                    int _parity_state_inp_ready = (int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) / 5 % 2);
                    mbarrier_wait(state_inp_ready_addr + (_stage_state_inp_ready) * 8, _parity_state_inp_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_1 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + ((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5) * 1344);
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
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
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
                    "}\n"
                    :: "r"(tmem_tmem_u_acc), "r"(_mma_b_lo_1), "r"(tmem_tmem_state_inp), "r"(1) : "memory");
                    {
                        int _mma_b_lo_2 = make_warp_uniform((((smem_qd_addr) >> 4) & 0x3FFF) + ((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5) * 1344);
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
                    :: "r"(tmem_tmem_out), "r"(_mma_b_lo_2), "r"(tmem_tmem_state_inp), "r"(0) : "memory");
                    }
                    elect_commit2(old_out_ready_addr + ((int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5)) * 8, raw_inputs_free_addr + ((int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5)) * 8);
                    int _stage_u_inp_ready = (int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5);
                    int _parity_u_inp_ready = (int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) / 5 % 2);
                    mbarrier_wait(u_inp_ready_addr + (_stage_u_inp_ready) * 8, _parity_u_inp_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_3 = make_warp_uniform((((smem_inv_addr) >> 4) & 0x3FFF) + ((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5) * 320);
                    mma_ts_step(tmem_tmem_u2_acc, tmem_tmem_u2_inp, _mma_b_lo_3, 0xC0004010, 134481040, 0);
                    elect_commit(u2_acc_ready_addr + ((int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5)) * 8);
                    int _stage_u2_inp_ready = (int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5);
                    int _parity_u2_inp_ready = (int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) / 5 % 2);
                    mbarrier_wait(u2_inp_ready_addr + (_stage_u2_inp_ready) * 8, _parity_u2_inp_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_4 = make_warp_uniform(((((smem_kr_trans_addr) >> 4) & 0x3FFF) | 0x800000) + ((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5) * 1344);
                    mma_ts_step(tmem_tmem_state, tmem_tmem_u2_inp, _mma_b_lo_4, 0x40004040, 136381584, 1);
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_mqk_trans_addr) >> 4) & 0x3FFF) | 0x200000) + ((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5) * 320);
                    mma_ts_step(tmem_tmem_out, tmem_tmem_u2_inp, _mma_b_lo_5, 0xC0004010, 134546576, 1);
                    elect_commit(final_ready_addr + ((int)((unsigned int)(bin_chunk_base_3 + _chunk_idx) % 5)) * 8);
                }
                bin_chunk_base_3 += num_chunks_0_3;
            }
            unsigned int _phase_tmem_dealloc_ready_0 = 0;
            mbarrier_wait(tmem_dealloc_ready_addr, _phase_tmem_dealloc_ready_0);
            _phase_tmem_dealloc_ready_0 ^= 1;
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        }
    // ---- Role: prep ----
    } else if (warp >= 12 && warp <= 31) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // prep_main
            int task_idx_3 = blockIdx.x;
            int bin_begin_4 = seq_order[bid];
            int bin_end_4 = seq_order[bid + 1];
            int bin_chunk_base_4 = 0;
            unsigned int _phase_work_item_ready_0_3 = 0;
            #pragma unroll 1
            for (int task_pos_4 = bin_begin_4; task_pos_4 < bin_end_4; task_pos_4++) {
                int binned_task_idx_4 = seq_order[task_pos_4];
                int split_compute_start_4 = 0;
                int seq_idx_4 = binned_task_idx_4 / num_heads;
                int head_idx_4 = binned_task_idx_4 % num_heads;
                long long bos_4 = cu_seqlens[seq_idx_4];
                long long eos_4 = cu_seqlens[seq_idx_4 + 1];
                int num_chunks_4 = ((int)(eos_4 - bos_4) + 16 - 1) / 16;
                int seq_len_4 = (int)(eos_4 - bos_4);
                int num_chunks_0_4 = (seq_len_4 + 16 - 1) / 16;
                int instance_id_1 = (warp - 12) / 4;
                int prep_instance = instance_id_1;
                int warp_id_in_role_3 = warp - 12;
                int prep_local_warp = warp_id_in_role_3 - prep_instance * 4;
                int prep_tid = prep_local_warp * 32 + lane;
                int prep_start = (prep_instance - bin_chunk_base_4 % 5 + 5) % 5;
                int num_prep_iters = (num_chunks_0_4 + 5 - 1 - prep_start) / 5;
                int gate_rate_stage_f32 = prep_instance * 5376;
                int prep_global_tid = warp_id_in_role_3 * 32 + lane;
                float task_gate_rate = __expf(A_log[head_idx_4]);
                float task_gate_bias = dt_bias[head_idx_4 * 128 + prep_tid];
                int prep_ordinal = bin_chunk_base_4 + prep_start;
                #pragma unroll 1
                for (int prep_iter = 0; prep_iter < num_prep_iters; prep_iter++) {
                    int chunk_idx_2 = prep_iter * 5 + prep_start;
                    int chunk_global_local_1 = chunk_idx_2;
                    int owned_chunk_1 = chunk_global_local_1 >= 0 && chunk_global_local_1 < num_chunks_4;
                    int stage_f32_1 = (unsigned int)prep_ordinal % 5 * 5376;
                    int stage_bf16 = (unsigned int)prep_ordinal % 5 * 10752;
                    int chunk_is_full_1 = ((seq_len_4 >= (chunk_idx_2 + 1) * 16) ? 1 : 0);
                    float early_beta_value = 0.0f;
                    float early_gate0 = 0.0f;
                    int _stage_raw_inputs_free = (int)((unsigned int)prep_ordinal % 5);
                    int _parity_raw_inputs_free = (int)((unsigned int)prep_ordinal / 5 % 2) ^ 1;
                    mbarrier_wait(raw_inputs_free_addr + (_stage_raw_inputs_free) * 8, _parity_raw_inputs_free);
                    if (chunk_is_full_1 != 0) {
                        if (prep_local_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(gate_raw_full_addr + ((int)((unsigned int)prep_ordinal % 5)) * 8, 4096);
                                tma_3d_gmem2smem(smem_g_raw_addr + (unsigned int)prep_ordinal % 5 * 21504, (&g_tma), 0, head_idx_4, (int)(bos_4 + (long long)(chunk_idx_2 * 16)), gate_raw_full_addr + ((int)((unsigned int)prep_ordinal % 5)) * 8);
                                mbarrier_arrive_expect_tx(qk_raw_full_addr + ((int)((unsigned int)prep_ordinal % 5)) * 8, 8192);
                                tma_4d_gmem2smem(smem_kd_addr + (unsigned int)prep_ordinal % 5 * 21504, (&k_tma), 0, (int)(bos_4 + (long long)(chunk_idx_2 * 16)), head_idx_4, 0, qk_raw_full_addr + ((int)((unsigned int)prep_ordinal % 5)) * 8);
                            }
                        }
                    } else if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(gate_raw_full_addr + ((int)((unsigned int)prep_ordinal % 5)) * 8);
                            mbarrier_arrive(qk_raw_full_addr + ((int)((unsigned int)prep_ordinal % 5)) * 8);
                        }
                    }
                    int _stage_gate_raw_full = (int)((unsigned int)prep_ordinal % 5);
                    int _parity_gate_raw_full = (int)((unsigned int)prep_ordinal / 5 % 2);
                    mbarrier_wait(gate_raw_full_addr + (_stage_gate_raw_full) * 8, _parity_gate_raw_full);
                    if (chunk_is_full_1 != 0) {
                        if (prep_local_warp == 2 && lane < 16) {
                            float beta_logit = 0.0f;
                            long long beta_token = bos_4 + (long long)(chunk_idx_2 * 16 + lane);
                            beta_logit = (float)beta[beta_token * beta_token_stride + (long long)head_idx_4];
                            float _tanh_0;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_0) : "f"(beta_logit * 0.5f));
                            early_beta_value = __fmaf_rn(_tanh_0, 0.5f, 0.5f);
                        }
                        if (prep_tid < 128) {
                            float early_gate_rate = task_gate_rate;
                            float early_gate_bias = task_gate_bias;
                            __nv_bfloat16 early_gate_raw = smem_g_raw_all[stage_bf16 + prep_tid];
                            float early_gate_arg = early_gate_rate * ((float)early_gate_raw + early_gate_bias);
                            float _tanh_1;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_1) : "f"(early_gate_arg * 0.5f));
                            float early_gate_sigmoid = __fmaf_rn(_tanh_1, 0.5f, 0.5f);
                            early_gate0 = lower_bound * 1.442695f * early_gate_sigmoid;
                        }
                    }
                    int _stage_smem_free = (int)((unsigned int)prep_ordinal % 5);
                    int _parity_smem_free = (int)((unsigned int)prep_ordinal / 5 % 2) ^ 1;
                    mbarrier_wait(smem_free_addr + (_stage_smem_free) * 8, _parity_smem_free);
                    int _stage_v_free = (int)((unsigned int)prep_ordinal % 5);
                    int _parity_v_free = (int)((unsigned int)prep_ordinal / 5 % 2) ^ 1;
                    mbarrier_wait(v_free_addr + (_stage_v_free) * 8, _parity_v_free);
                    if (chunk_is_full_1 != 0) {
                        if (prep_local_warp == 0) {
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_q_raw_prefetch_addr + (unsigned int)prep_ordinal % 5 * 21504, (&q_tma), 0, (int)(bos_4 + (long long)(chunk_idx_2 * 16)), head_idx_4, 0, qk_raw_full_addr + ((int)((unsigned int)prep_ordinal % 5)) * 8);
                            }
                        }
                    }
                    if (chunk_is_full_1 == 0) {
                        #pragma unroll
                        for (int gate_load_pass = 0; gate_load_pass < 2; gate_load_pass++) {
                            int gate_load_item = gate_load_pass * 128 + prep_tid;
                            int gate_load_row = gate_load_item / 16;
                            int gate_load_segment = gate_load_item % 16;
                            long long gate_load_token = bos_4 + (long long)(chunk_idx_2 * 16 + gate_load_row);
                            long long gate_load_base = gate_load_token * g_token_stride + (long long)head_idx_4 * 128 + (long long)(gate_load_segment * 8);
                            long long q_load_base = gate_load_token * q_stride_m3 + (long long)head_idx_4 * 128 + (long long)(gate_load_segment * 8);
                            long long k_load_base = gate_load_token * k_stride_m3 + (long long)head_idx_4 * 128 + (long long)(gate_load_segment * 8);
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                :: "r"(smem_g_raw_addr + (unsigned int)prep_ordinal % 5 * 21504 + (unsigned int)(gate_load_item * 16)), "l"(g + gate_load_base), "r"((gate_load_token < eos_4) ? 16 : 0));
                            int q_tail_addr = (smem_q_raw_prefetch_addr + (unsigned int)prep_ordinal % 5 * 21504 + (unsigned int)(gate_load_segment * 8 / 64 * 2048 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 ^ (gate_load_segment * 8 / 64 * 2048 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 >> 7 & 7) << 4));
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                :: "r"(q_tail_addr), "l"(q + q_load_base), "r"((gate_load_token < eos_4) ? 16 : 0));
                            int k_tail_addr = (smem_kd_addr + (unsigned int)prep_ordinal % 5 * 21504 + (unsigned int)(gate_load_segment * 8 / 64 * 2048 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 ^ (gate_load_segment * 8 / 64 * 2048 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 >> 7 & 7) << 4));
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                :: "r"(k_tail_addr), "l"(k + k_load_base), "r"((gate_load_token < eos_4) ? 16 : 0));
                        }
                    }
                    if (chunk_is_full_1 == 0) {
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 0;");
                        asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    }
                    if (prep_local_warp == 2 && lane < 16) {
                        long long beta_token_1 = bos_4 + (long long)(chunk_idx_2 * 16 + lane);
                        float beta_value = early_beta_value;
                        if (chunk_is_full_1 == 0) {
                            if (beta_token_1 < eos_4) {
                                float beta_logit_1 = (float)beta[beta_token_1 * (long long)num_heads + (long long)head_idx_4];
                                beta_logit_1 = (float)beta[beta_token_1 * beta_token_stride + (long long)head_idx_4];
                                float _tanh_2;
                                asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_2) : "f"(beta_logit_1 * 0.5f));
                                beta_value = __fmaf_rn(_tanh_2, 0.5f, 0.5f);
                            }
                        }
                        {
                            smem_prep_beta_bf16_all[stage_bf16 + lane] = (__nv_bfloat16)beta_value;
                        }
                    }
                    if (prep_tid < 128) {
                        int gate_col = prep_tid;
                        float gate_rate = task_gate_rate;
                        float gate_bias = task_gate_bias;
                        float prefix_log2 = 0.0f;
                        float2 scan_neg_rate_pair = make_float2(-gate_rate, -gate_rate);
                        float2 scan_bias_pair = make_float2(gate_bias, gate_bias);
                        long long first_gate_token = bos_4 + (long long)(chunk_idx_2 * 16);
                        __nv_bfloat16 first_gate_raw = smem_g_raw_all[stage_bf16 + gate_col];
                        float first_gate_log2 = early_gate0;
                        if (chunk_is_full_1 == 0) {
                            float gate_arg = gate_rate * ((float)first_gate_raw + gate_bias);
                            float _tanh_3;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_3) : "f"(gate_arg * 0.5f));
                            float gate_sigmoid = __fmaf_rn(_tanh_3, 0.5f, 0.5f);
                            float first_gate_value = lower_bound * 1.442695f * gate_sigmoid;
                            first_gate_log2 = ((first_gate_token < eos_4) ? first_gate_value : 0.0f);
                        }
                        prefix_log2 += first_gate_log2;
                        smem_gate_all[stage_f32_1 + gate_col] = prefix_log2;
                        for (int gate_pair = 0; gate_pair < 7; gate_pair++) {
                            int pair_row0 = gate_pair * 2 + 1;
                            int pair_row1 = pair_row0 + 1;
                            long long pair_token0 = bos_4 + (long long)(chunk_idx_2 * 16 + pair_row0);
                            long long pair_token1 = pair_token0 + 1;
                            __nv_bfloat16 pair_raw0 = smem_g_raw_all[stage_bf16 + pair_row0 * 128 + gate_col];
                            __nv_bfloat16 pair_raw1 = smem_g_raw_all[stage_bf16 + pair_row1 * 128 + gate_col];
                            float gate_arg_1 = gate_rate * ((float)pair_raw0 + gate_bias);
                            float _tanh_4;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_4) : "f"(gate_arg_1 * 0.5f));
                            float gate_sigmoid_1 = __fmaf_rn(_tanh_4, 0.5f, 0.5f);
                            float gate_arg_0 = gate_rate * ((float)pair_raw1 + gate_bias);
                            float _tanh_5;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_5) : "f"(gate_arg_0 * 0.5f));
                            float gate_sigmoid_0 = __fmaf_rn(_tanh_5, 0.5f, 0.5f);
                            float pair_log0 = ((pair_token0 < eos_4) ? lower_bound * 1.442695f * gate_sigmoid_1 : 0.0f);
                            float pair_log1 = ((pair_token1 < eos_4) ? lower_bound * 1.442695f * gate_sigmoid_0 : 0.0f);
                            prefix_log2 += pair_log0;
                            smem_gate_all[stage_f32_1 + pair_row0 * 128 + gate_col] = prefix_log2;
                            prefix_log2 += pair_log1;
                            smem_gate_all[stage_f32_1 + pair_row1 * 128 + gate_col] = prefix_log2;
                        }
                        long long last_gate_token = bos_4 + (long long)(chunk_idx_2 * 16 + 15);
                        __nv_bfloat16 last_gate_raw = smem_g_raw_all[stage_bf16 + 1920 + gate_col];
                        float gate_arg_2 = gate_rate * ((float)last_gate_raw + gate_bias);
                        float _tanh_6;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_6) : "f"(gate_arg_2 * 0.5f));
                        float gate_sigmoid_2 = __fmaf_rn(_tanh_6, 0.5f, 0.5f);
                        float last_gate_value = lower_bound * 1.442695f * gate_sigmoid_2;
                        float last_gate_log2 = ((last_gate_token < eos_4) ? last_gate_value : 0.0f);
                        prefix_log2 += last_gate_log2;
                        smem_gate_all[stage_f32_1 + 1920 + gate_col] = prefix_log2;
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    float page_total_log2 = 0.0f;
                    int _stage_qk_raw_full = (int)((unsigned int)prep_ordinal % 5);
                    int _parity_qk_raw_full = (int)((unsigned int)prep_ordinal / 5 % 2);
                    mbarrier_wait(qk_raw_full_addr + (_stage_qk_raw_full) * 8, _parity_qk_raw_full);
                    if (prep_tid < 128) {
                        float total_log2 = smem_gt_prefix[stage_f32_1 + prep_tid];
                        float restore_factor_value = approx_exp2(total_log2);
                        smem_restore_factor_all[stage_f32_1 + prep_tid] = restore_factor_value;
                    }
                    if (prep_tid == 0) {
                        float _exp2_7 = approx_exp2(lower_bound * 1.442695f * 8.0f);
                        smem_restore_factor_all[stage_f32_1 + 128] = _exp2_7;
                    }
                    #pragma unroll 1
                    for (int work_pass = 0; work_pass < 2; work_pass++) {
                        int work_item = work_pass * 128 + prep_tid;
                        int row = work_item / 16;
                        int segment = work_item % 16;
                        long long token = bos_4 + (long long)(chunk_idx_2 * 16 + row);
                        int token_valid = ((token < eos_4) ? 1 : 0);
                        long long gmem_base = (token * (long long)num_heads + (long long)head_idx_4) * 128 + (long long)(segment * 8);
                        float q_raw_vec[8];
                        float k_raw_vec[8];
                        unsigned int packed[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                            : "r"((smem_q_raw_prefetch_addr + (unsigned int)prep_ordinal % 5 * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed[_pair]) << 16);
                            (&unpack_bf16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx_3 = 0; value_idx_3 < 8; value_idx_3++) {
                            q_raw_vec[value_idx_3] = unpack_bf16[value_idx_3];
                        }
                        unsigned int packed_0[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 3]))
                            : "r"((smem_kd_addr + (unsigned int)prep_ordinal % 5 * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_0[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_0[_pair]) << 16);
                            (&unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_0[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx_4 = 0; value_idx_4 < 8; value_idx_4++) {
                            k_raw_vec[value_idx_4] = unpack_bf16_0[value_idx_4];
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
                        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 8);
                        q_sum += _shfl_xor_sync_8;
                        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 8);
                        k_sum += _shfl_xor_sync_9;
                        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 4);
                        q_sum += _shfl_xor_sync_10;
                        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 4);
                        k_sum += _shfl_xor_sync_11;
                        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 2);
                        q_sum += _shfl_xor_sync_12;
                        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 2);
                        k_sum += _shfl_xor_sync_13;
                        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 1);
                        q_sum += _shfl_xor_sync_14;
                        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 1);
                        k_sum += _shfl_xor_sync_15;
                        float q_inv = rsqrtf(q_sum + 1e-06f);
                        float k_inv = rsqrtf(k_sum + 1e-06f);
                        const float2 _scale2_16 = {q_inv, q_inv};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(q_raw_vec)[_ls], _scale2_16);
                        const float2 _scale2_17 = {k_inv, k_inv};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(k_raw_vec)[_ls], _scale2_17);
                        float qd_vec[8];
                        float kd_vec[8];
                        float ki_vec[8];
                        for (int elem_in_segment = 0; elem_in_segment < 8; elem_in_segment++) {
                            int col = segment * 8 + elem_in_segment;
                            float prefix = smem_gate_all[stage_f32_1 + row * 128 + col];
                            float common_log2 = smem_gate_all[stage_f32_1 + 1024 + col];
                            float decay = approx_exp2(prefix - common_log2);
                            qd_vec[elem_in_segment] = decay;
                            kd_vec[elem_in_segment] = decay;
                            ki_vec[elem_in_segment] = k_raw_vec[elem_in_segment] / decay;
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], reinterpret_cast<const float2*>(q_raw_vec)[_ls]);
                        const float2 _scale2_18 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], _scale2_18);
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(kd_vec)[_ls], reinterpret_cast<const float2*>(k_raw_vec)[_ls]);
                        float _cvt_f32_0 = (float)(__nv_bfloat16)scale;
                        float n16_decay_values[8];
                        float n16_inv_decay_values[8];
                        n16_decay_values[0] = smem_gate_all[stage_f32_1 + row * 128 + segment * 8];
                        n16_decay_values[1] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 1)];
                        n16_decay_values[2] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 2)];
                        n16_decay_values[3] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 3)];
                        n16_decay_values[4] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 4)];
                        n16_decay_values[5] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 5)];
                        n16_decay_values[6] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 6)];
                        n16_decay_values[7] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 7)];
                        unsigned int n16_decay_values_bf16[4];
                        #pragma unroll
                        for (int _fe = 0; _fe < 4; _fe++) {
                            n16_decay_values[0 + _fe*2] = approx_exp2(n16_decay_values[0 + _fe*2]);
                            n16_decay_values[0 + _fe*2 + 1] = approx_exp2(n16_decay_values[0 + _fe*2 + 1]);
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(n16_decay_values[0 + _fe*2], n16_decay_values[0 + _fe*2 + 1]));
                            n16_decay_values_bf16[0 + _fe] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&n16_decay_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(n16_decay_values_bf16[_pair]) << 16);
                            (&n16_decay_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(n16_decay_values_bf16[_pair]) & 0xffff0000u);
                        }
                        float _rcp_19 = approx_rcp(n16_decay_values[0]);
                        n16_inv_decay_values[0] = _rcp_19;
                        float _rcp_20 = approx_rcp(n16_decay_values[1]);
                        n16_inv_decay_values[1] = _rcp_20;
                        float _rcp_21 = approx_rcp(n16_decay_values[2]);
                        n16_inv_decay_values[2] = _rcp_21;
                        float _rcp_22 = approx_rcp(n16_decay_values[3]);
                        n16_inv_decay_values[3] = _rcp_22;
                        float _rcp_23 = approx_rcp(n16_decay_values[4]);
                        n16_inv_decay_values[4] = _rcp_23;
                        float _rcp_24 = approx_rcp(n16_decay_values[5]);
                        n16_inv_decay_values[5] = _rcp_24;
                        float _rcp_25 = approx_rcp(n16_decay_values[6]);
                        n16_inv_decay_values[6] = _rcp_25;
                        float _rcp_26 = approx_rcp(n16_decay_values[7]);
                        n16_inv_decay_values[7] = _rcp_26;
                        unsigned int q_raw_vec_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_raw_vec[_lp*2 + 0], q_raw_vec[_lp*2+1 + 0]));
                            q_raw_vec_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qd_vec[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(q_raw_vec_bf16[_pair]) << 16);
                            (&qd_vec[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(q_raw_vec_bf16[_pair]) & 0xffff0000u);
                        }
                        unsigned int k_raw_vec_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(k_raw_vec[_lp*2 + 0], k_raw_vec[_lp*2+1 + 0]));
                            k_raw_vec_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&kd_vec[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(k_raw_vec_bf16[_pair]) << 16);
                            (&kd_vec[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(k_raw_vec_bf16[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ki_vec[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(k_raw_vec_bf16[_pair]) << 16);
                            (&ki_vec[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(k_raw_vec_bf16[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], reinterpret_cast<const float2*>(n16_decay_values)[_ls]);
                        unsigned int qd_vec_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                            qd_vec_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qd_vec[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qd_vec_bf16[_pair]) << 16);
                            (&qd_vec[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qd_vec_bf16[_pair]) & 0xffff0000u);
                        }
                        const float2 _scale2_27 = {_cvt_f32_0, _cvt_f32_0};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], _scale2_27);
                        unsigned int qd_vec_bf16_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                            qd_vec_bf16_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&qd_vec[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qd_vec_bf16_1[_pair]) << 16);
                            (&qd_vec[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qd_vec_bf16_1[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(kd_vec)[_ls], reinterpret_cast<const float2*>(n16_decay_values)[_ls]);
                        unsigned int kd_vec_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_vec[_lp*2 + 0], kd_vec[_lp*2+1 + 0]));
                            kd_vec_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&kd_vec[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kd_vec_bf16[_pair]) << 16);
                            (&kd_vec[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kd_vec_bf16[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ki_vec)[_ls], reinterpret_cast<const float2*>(n16_inv_decay_values)[_ls]);
                        unsigned int ki_vec_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_vec[_lp*2 + 0], ki_vec[_lp*2+1 + 0]));
                            ki_vec_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&ki_vec[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ki_vec_bf16[_pair]) << 16);
                            (&ki_vec[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ki_vec_bf16[_pair]) & 0xffff0000u);
                        }
                        unsigned int packed_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                            packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word_6 = 0; word_6 < 4; word_6++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + (unsigned int)prep_ordinal % 5 * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_6 * 4)), "r"(packed_1[word_6]));
                        }
                        unsigned int packed_2[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_vec[_lp*2 + 0], kd_vec[_lp*2+1 + 0]));
                            packed_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word_7 = 0; word_7 < 4; word_7++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + (unsigned int)prep_ordinal % 5 * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_7 * 4)), "r"(packed_2[word_7]));
                        }
                        unsigned int packed_3[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_vec[_lp*2 + 0], ki_vec[_lp*2+1 + 0]));
                            packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word_8 = 0; word_8 < 4; word_8++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_ki_addr + (unsigned int)prep_ordinal % 5 * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_8 * 4)), "r"(packed_3[word_8]));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    mbarrier_arrive(aux_pairwise_inputs_ready_addr + ((int)((unsigned int)prep_ordinal % 10)) * 8);
                    unsigned int a_frag_1[4];
                    unsigned int b_frag_1[4];
                    float acc_1[8];
                    long long tape_scaled_base = 0;
                    int _stage_aux_pairwise_consumed = (int)((unsigned int)prep_ordinal % 10);
                    int _parity_aux_pairwise_consumed = (int)((unsigned int)prep_ordinal / 10 % 2);
                    mbarrier_wait(aux_pairwise_consumed_addr + (_stage_aux_pairwise_consumed) * 8, _parity_aux_pairwise_consumed);
                    int stage_f32_0 = (unsigned int)prep_ordinal % 5 * 5376;
                    int restore_segment = lane & 15;
                    #pragma unroll 1
                    for (int restore_k_pass = 0; restore_k_pass < 2; restore_k_pass++) {
                        int restore_row = prep_local_warp * 4 + restore_k_pass * 2 + (lane >> 4);
                        float restore_ki_values[8];
                        float restore_kr_values[8];
                        unsigned int packed_4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 3]))
                            : "r"((smem_ki_addr + (unsigned int)prep_ordinal % 5 * 21504 + (unsigned int)(restore_segment * 8 / 64 * 2048 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 2048 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_4[_pair]) << 16);
                            (&unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_4[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx_5 = 0; value_idx_5 < 8; value_idx_5++) {
                            restore_ki_values[value_idx_5] = unpack_bf16_1[value_idx_5];
                        }
                        #pragma unroll
                        for (int restore_elem = 0; restore_elem < 8; restore_elem++) {
                            int restore_col = restore_segment * 8 + restore_elem;
                            float restore_ki_carrier = (float)(__nv_bfloat16)restore_ki_values[restore_elem];
                            float restore_total = smem_restore_factor_all[stage_f32_0 + restore_col];
                            float restore_total_carrier = (float)(__nv_bfloat16)restore_total;
                            restore_kr_values[restore_elem] = (float)(__nv_bfloat16)(restore_ki_carrier * restore_total_carrier);
                        }
                        unsigned int packed_4_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values[_lp*2 + 0], restore_kr_values[_lp*2+1 + 0]));
                            packed_4_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word_9 = 0; word_9 < 4; word_9++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + (unsigned int)prep_ordinal % 5 * 21504 + (unsigned int)(restore_segment * 8 / 64 * 2048 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 2048 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_9 * 4)), "r"(packed_4_1[word_9]));
                        }
                    }
                    int _stage_aux_inverse_ready = (int)((unsigned int)prep_ordinal % 10);
                    int _parity_aux_inverse_ready = (int)((unsigned int)prep_ordinal / 10 % 2);
                    mbarrier_wait(aux_inverse_ready_addr + (_stage_aux_inverse_ready) * 8, _parity_aux_inverse_ready);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    mbarrier_arrive(qk_full_addr + ((int)((unsigned int)prep_ordinal % 5)) * 8);
                    if (chunk_is_full_1 != 0) {
                        if (prep_local_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(v_full_addr + ((int)((unsigned int)prep_ordinal % 5)) * 8, 4096);
                                int v_token = (int)(bos_4 + (long long)(chunk_idx_2 * 16));
                                tma_4d_gmem2smem(smem_v_addr + (unsigned int)prep_ordinal % 5 * 4096, (&v_tma), 0, head_idx_4, v_token, 0, v_full_addr + ((int)((unsigned int)prep_ordinal % 5)) * 8);
                                tma_4d_gmem2smem(smem_v_addr + (unsigned int)prep_ordinal % 5 * 4096 + 2048, (&v_tma), 0, head_idx_4, v_token, 1, v_full_addr + ((int)((unsigned int)prep_ordinal % 5)) * 8);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int v_load_iter = 0; v_load_iter < 2; v_load_iter++) {
                            int v_item = v_load_iter * 128 + prep_tid;
                            int row_1 = v_item / 16;
                            int segment_1 = v_item % 16;
                            long long token_1 = bos_4 + (long long)(chunk_idx_2 * 16 + row_1);
                            int token_valid_1 = ((token_1 < eos_4) ? 1 : 0);
                            long long v_src = token_1 * v_stride_m3 + (long long)head_idx_4 * 128 + (long long)(segment_1 * 8);
                            int v_dst = smem_v_addr + (unsigned int)prep_ordinal % 5 * 4096 + (unsigned int)((row_1 * 128 + segment_1 * 8) * 2);
                            int v_panel = segment_1 / 8;
                            int v_panel_segment = segment_1 % 8;
                            int v_row_addr = smem_v_addr + (unsigned int)prep_ordinal % 5 * 4096 + (unsigned int)(v_panel * 16 * 64 * 2) + (unsigned int)(row_1 * 64 * 2);
                            v_dst = (v_row_addr + (v_panel_segment * 8 * 2 ^ (v_row_addr >> 7 & 7) << 4));
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                :: "r"(v_dst), "l"(v + v_src), "r"((token_valid_1 != 0) ? 16 : 0));
                        }
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 0;");
                        asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                        if (prep_local_warp == 0) {
                            if (elect_sync()) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                mbarrier_arrive(v_full_addr + ((int)((unsigned int)prep_ordinal % 5)) * 8);
                            }
                        }
                    }
                    prep_ordinal += 5;
                }
                bin_chunk_base_4 += num_chunks_0_4;
            }
        }
    }

    // Cleanup
}

} // extern "C"
