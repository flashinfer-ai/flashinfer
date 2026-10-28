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
#define TMEM_NCOLS 256
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
#define NUM_CHUNK_PIPE_STAGES 5
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
#define SMEM_SMEM_CHECKPOINT_OFF 1024
#define SMEM_SMEM_CHECKPOINT_STAGE_BYTES 32768
#define SMEM_SMEM_CHECKPOINT_STRIDE 32768
#define SMEM_SMEM_RESTORE_FACTOR_ALL_OFF 41984
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES 516
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STRIDE 41984
#define SMEM_SMEM_GT_PREFIX_OFF 41472
#define SMEM_SMEM_GT_PREFIX_STAGE_BYTES 512
#define SMEM_SMEM_GT_PREFIX_STRIDE 41984
#define SMEM_SMEM_GT_PREFIX_ALL_OFF 41472
#define SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES 168448
#define SMEM_SMEM_GT_PREFIX_ALL_STRIDE 168448
#define SMEM_SMEM_GT_ALL_OFF 31744
#define SMEM_SMEM_GT_ALL_STAGE_BYTES 512
#define SMEM_SMEM_GT_ALL_STRIDE 41984
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
#define SMEM_TOTAL 227968
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



__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
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
kernel_cake_kda_prefill_ab6ca48560320943bbd5(__nv_bfloat16* __restrict__ q, const __grid_constant__ CUtensorMap q_tma, __nv_bfloat16* __restrict__ k, const __grid_constant__ CUtensorMap k_tma, __nv_bfloat16* __restrict__ v, const __grid_constant__ CUtensorMap v_tma, __nv_bfloat16* __restrict__ g, const __grid_constant__ CUtensorMap g_tma, __nv_bfloat16* __restrict__ beta, const __grid_constant__ CUtensorMap beta_tma, float* __restrict__ A_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ seq_order, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ out, const __grid_constant__ CUtensorMap out_tma, __nv_bfloat16* __restrict__ final_state, int num_heads, int use_initial_state, int store_final_state, float scale, float lower_bound, unsigned long long state_indices_addr, unsigned long long state_checkpoints_addr, unsigned long long checkpoint_cu_starts_addr, long long beta_token_stride, long long state_slot_stride, int use_state_indices, int checkpoint_every_n_tokens, long long* __restrict__ cu_chunk_offsets, __nv_bfloat16* __restrict__ chunk_state, unsigned int* __restrict__ state_checkpoint_needed, __nv_bfloat16* __restrict__ tape_qd, __nv_bfloat16* __restrict__ tape_kd, __nv_bfloat16* __restrict__ tape_kr, __nv_bfloat16* __restrict__ tape_j, float* __restrict__ tape_restore_factor, __nv_bfloat16* __restrict__ tape_e, __nv_bfloat16* __restrict__ tape_x, __nv_bfloat16* __restrict__ tape_r, float* __restrict__ norm_inv_out, __nv_bfloat16* __restrict__ decay_out, float* __restrict__ beta_active_out, float* __restrict__ initial_state_f32, unsigned int* __restrict__ zero_workspace, int zero_words, int num_sequences, const __grid_constant__ CUtensorMap state_checkpoints_tma, float* __restrict__ final_state_f32, long long g_token_stride, long long q_stride_m3, long long k_stride_m3, long long v_stride_m3)
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
    #define old_out_ready_addr (mbar_base + 320)
    #define u_inp_ready_addr (mbar_base + 360)
    #define u2_acc_ready_addr (mbar_base + 400)
    #define u2_inp_ready_addr (mbar_base + 440)
    #define final_ready_addr (mbar_base + 480)
    #define out_empty_addr (mbar_base + 520)
    #define tmem_dealloc_ready_addr (mbar_base + 528)
    #define checkpoint_ready_addr (mbar_base + 536)
    #define checkpoint_free_addr (mbar_base + 552)
    #define checkpoint_snapshot_done_addr (mbar_base + 568)
    #define work_item_ready_addr (mbar_base + 600)
    #define short_beta_ready_addr (mbar_base + 608)

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
    __nv_bfloat16* smem_v18 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 210944);
    const int smem_v18_addr = smem + 210944;
    __nv_bfloat16* smem_checkpoint = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_checkpoint_addr = smem + 1024;
    float* smem_restore_factor_all = reinterpret_cast<float*>(smem_raw + 41984);
    const int smem_restore_factor_all_addr = smem + 41984;
    float* smem_gt_prefix = reinterpret_cast<float*>(smem_raw + 41472);
    const int smem_gt_prefix_addr = smem + 41472;
    float* smem_gt_prefix_all = reinterpret_cast<float*>(smem_raw + 41472);
    const int smem_gt_prefix_all_addr = smem + 41472;
    float* smem_gt_all = reinterpret_cast<float*>(smem_raw + 31744);
    const int smem_gt_all_addr = smem + 31744;
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

    // Mbarrier init (20 pipeline groups, 0 ordered-sequence groups, 81 barriers)
    // Mbarriers at smem_raw[0..648)

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
        // old_out_ready: 5 barriers, init_count=1
        // u_inp_ready: 5 barriers, init_count=4
        // u2_acc_ready: 5 barriers, init_count=1
        // u2_inp_ready: 5 barriers, init_count=4
        // final_ready: 5 barriers, init_count=2
        // out_empty: 1 barriers, init_count=1
        // tmem_dealloc_ready: 1 barriers, init_count=2
        // --- pipeline 'checkpoint_pipe' ---
        // checkpoint_ready: 2 barriers, init_count=4
        // checkpoint_free: 2 barriers, init_count=1
        // --- pipeline 'snapshot_pipe' ---
        // checkpoint_snapshot_done: 4 barriers, init_count=4
        // work_item_ready: 1 barriers, init_count=1
        // --- pipeline 'chunk_pipe' ---
        // short_beta_ready: 5 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_10_0 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(30), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(20), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(5), "r"((uint32_t)(128)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_10_0);
        uint32_t _mbarrier_init_count_10_32 = 2;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(28), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(23), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(18), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(13), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(8), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(3), "r"((uint32_t)(1)));
        mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_10_32);
        uint32_t _mbarrier_init_count_10_64 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(11), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(7), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(5), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(3), "r"((uint32_t)(2)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(2), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(1), "r"((uint32_t)(2)));
        if (lane < 17) {
            mbarrier_init(smem + 512 + lane * 8, _mbarrier_init_count_10_64);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 648);
    if (warp == 0) {
        int _tmem_hold = smem + 648;
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
    const int tmem_tmem_u2_acc = taddr;
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
            int split_compute_start = 0;
            int seq_idx = seq_order[task_idx / num_heads];
            int head_idx = task_idx % num_heads;
            long long bos = cu_seqlens[seq_idx];
            long long eos = cu_seqlens[seq_idx + 1];
            int num_chunks = ((int)(eos - bos) + 32 - 1) / 32;
            int seq_len = (int)(eos - bos);
            int num_chunks_0 = (seq_len + 32 - 1) / 32;
            int num_pages = 0;
            long long total_chunks = cu_chunk_offsets[num_sequences];
            long long fallback_head = total_chunks * (long long)num_heads + (long long)seq_idx * (long long)num_heads + (long long)head_idx;
            const int tmem_row_base = warp_in_wg * 32 << 16;
            long long state_base = (((long long)seq_idx * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128;
            long long checkpoint_row_start = 0;
            long long checkpoint_single_token = 0;
            if (checkpoint_every_n_tokens != 0) {
                checkpoint_row_start = reinterpret_cast<long long*>(checkpoint_cu_starts_addr)[seq_idx];
            }
            long long initial_state_base = state_base;
            int initial_state_enabled = (int)(use_initial_state != 0);
            int affine_initial_slot = reinterpret_cast<int*>(state_indices_addr)[seq_idx];
            initial_state_enabled = (int)(initial_state_enabled != 0 && affine_initial_slot >= 0);
            initial_state_base = (long long)affine_initial_slot * state_slot_stride + ((long long)head_idx * 128 + (long long)state_row) * 128;
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
            if (checkpoint_every_n_tokens != 0) {
                long long checkpoint_base = ((checkpoint_row_start * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128;
                #pragma unroll
                for (int state_col_block_1 = 0; state_col_block_1 < 4; state_col_block_1++) {
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_1 * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int packed[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 2], _tmem_load_0[_lp*2+1 + 2]));
                        packed[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 4], _tmem_load_0[_lp*2+1 + 4]));
                        packed[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 6], _tmem_load_0[_lp*2+1 + 6]));
                        packed[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 8], _tmem_load_0[_lp*2+1 + 8]));
                        packed[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 10], _tmem_load_0[_lp*2+1 + 10]));
                        packed[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 12], _tmem_load_0[_lp*2+1 + 12]));
                        packed[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 14], _tmem_load_0[_lp*2+1 + 14]));
                        packed[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(state_checkpoints_addr) + (checkpoint_base + (long long)(state_col_block_1 * 32)))[0] = reinterpret_cast<int4*>(packed)[0];
                    reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(state_checkpoints_addr) + (checkpoint_base + (long long)(state_col_block_1 * 32) + 8))[0] = reinterpret_cast<int4*>(packed + (4))[0];
                    unsigned int packed_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 16], _tmem_load_0[_lp*2+1 + 16]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 18], _tmem_load_0[_lp*2+1 + 18]));
                        packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 20], _tmem_load_0[_lp*2+1 + 20]));
                        packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 22], _tmem_load_0[_lp*2+1 + 22]));
                        packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 24], _tmem_load_0[_lp*2+1 + 24]));
                        packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 26], _tmem_load_0[_lp*2+1 + 26]));
                        packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 28], _tmem_load_0[_lp*2+1 + 28]));
                        packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 30], _tmem_load_0[_lp*2+1 + 30]));
                        packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(state_checkpoints_addr) + (checkpoint_base + (long long)(state_col_block_1 * 32) + 16))[0] = reinterpret_cast<int4*>(packed_1)[0];
                    reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(state_checkpoints_addr) + (checkpoint_base + (long long)(state_col_block_1 * 32) + 16 + 8))[0] = reinterpret_cast<int4*>(packed_1 + (4))[0];
                }
            }
            unsigned int checkpoint_stage_compute = 0;
            unsigned int snapshot_stage_compute = 0;
            float _exp2_0 = approx_exp2(lower_bound * 1.442695f * 16.0f);
            unsigned int _phase_checkpoint_snapshot_done = 0;
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
                int _stage_qk_full = (int)((unsigned int)chunk_idx % 5);
                int _parity_qk_full = (int)((unsigned int)chunk_idx / 5 % 2);
                mbarrier_wait(qk_full_addr + (_stage_qk_full) * 8, _parity_qk_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll 1
                for (int state_col_block_2 = 0; state_col_block_2 < 3; state_col_block_2++) {
                    int state_addr = taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_2 * 32);
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
                        :: "r"(taddr + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_2 * 16)), "r"(tmem_load_1_bf16[0]), "r"(tmem_load_1_bf16[1]), "r"(tmem_load_1_bf16[2]), "r"(tmem_load_1_bf16[3]), "r"(tmem_load_1_bf16[4]), "r"(tmem_load_1_bf16[5]), "r"(tmem_load_1_bf16[6]), "r"(tmem_load_1_bf16[7]), "r"(tmem_load_1_bf16[8]), "r"(tmem_load_1_bf16[9]), "r"(tmem_load_1_bf16[10]), "r"(tmem_load_1_bf16[11]), "r"(tmem_load_1_bf16[12]), "r"(tmem_load_1_bf16[13]), "r"(tmem_load_1_bf16[14]), "r"(tmem_load_1_bf16[15]));
                    float state_scale[16];
                    #pragma unroll
                    for (int state_half = 0; state_half < 2; state_half++) {
                        #pragma unroll
                        for (int state_col = 0; state_col < 16; state_col++) {
                            state_scale[state_col] = smem_gt_all[(unsigned int)chunk_idx % 5 * 10496 + (unsigned int)(state_col_block_2 * 32) + (unsigned int)(state_half * 16) + (unsigned int)state_col];
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((tmem_load + state_half * 16))[_ls], reinterpret_cast<const float2*>(state_scale)[_ls]);
                    }
                    if (STORE_BACKWARD_TAPE == 0 && checkpoint_entering != 0 && chunk_idx != 0) {
                        mbarrier_wait(checkpoint_snapshot_done_addr + (snapshot_stage_compute) * 8, _phase_checkpoint_snapshot_done);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        snapshot_stage_compute += 1;
                        if (snapshot_stage_compute == 4) { snapshot_stage_compute = 0; _phase_checkpoint_snapshot_done ^= 1; }
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
                    mbarrier_arrive(state_inp_ready_addr + ((int)((unsigned int)chunk_idx % 5)) * 8);
                }
                float state_tail_scale[16];
                #pragma unroll
                for (int state_half_1 = 0; state_half_1 < 2; state_half_1++) {
                    #pragma unroll
                    for (int state_col_1 = 0; state_col_1 < 16; state_col_1++) {
                        state_tail_scale[state_col_1] = smem_gt_all[(unsigned int)chunk_idx % 5 * 10496 + 96 + (unsigned int)(state_half_1 * 16) + (unsigned int)state_col_1];
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((state_tail_frag + state_half_1 * 16))[_ls], reinterpret_cast<const float2*>(state_tail_scale)[_ls]);
                }
                if (STORE_BACKWARD_TAPE == 0 && checkpoint_entering != 0 && chunk_idx != 0) {
                    mbarrier_wait(checkpoint_snapshot_done_addr + (snapshot_stage_compute) * 8, _phase_checkpoint_snapshot_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    snapshot_stage_compute += 1;
                    if (snapshot_stage_compute == 4) { snapshot_stage_compute = 0; _phase_checkpoint_snapshot_done ^= 1; }
                }
                tmem_st_x32_f32(state_tail_addr, state_tail_frag);
                int _stage_v_full = (int)((unsigned int)chunk_idx % 5);
                int _parity_v_full = (int)((unsigned int)chunk_idx / 5 % 2);
                mbarrier_wait(v_full_addr + (_stage_v_full) * 8, _parity_v_full);
                unsigned int v_prefetch_bits[8];
                int _stage_old_out_ready = (int)((unsigned int)chunk_idx % 5);
                int _parity_old_out_ready = (int)((unsigned int)chunk_idx / 5 % 2);
                mbarrier_wait(old_out_ready_addr + (_stage_old_out_ready) * 8, _parity_old_out_ready);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float prediction[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(prediction[0]), "=f"(prediction[1]), "=f"(prediction[2]), "=f"(prediction[3]), "=f"(prediction[4]), "=f"(prediction[5]), "=f"(prediction[6]), "=f"(prediction[7]), "=f"(prediction[8]), "=f"(prediction[9]), "=f"(prediction[10]), "=f"(prediction[11]), "=f"(prediction[12]), "=f"(prediction[13]), "=f"(prediction[14]), "=f"(prediction[15]), "=f"(prediction[16]), "=f"(prediction[17]), "=f"(prediction[18]), "=f"(prediction[19]), "=f"(prediction[20]), "=f"(prediction[21]), "=f"(prediction[22]), "=f"(prediction[23]), "=f"(prediction[24]), "=f"(prediction[25]), "=f"(prediction[26]), "=f"(prediction[27]), "=f"(prediction[28]), "=f"(prediction[29]), "=f"(prediction[30]), "=f"(prediction[31])
                    : "r"(taddr + 224 + (unsigned int)tmem_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                const float2 _scale2_1 = {_exp2_0, _exp2_0};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(prediction)[_ls], _scale2_1);
                long long chunk_global_e = cu_chunk_offsets[seq_idx] + (long long)chunk_global_local;
                long long tape_ex_base = ((chunk_global_e * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 32;
                #pragma unroll
                for (int residual_half = 0; residual_half < 2; residual_half++) {
                    float residual_v[16];
                    float residual_beta[16];
                    #pragma unroll
                    for (int residual_col = 0; residual_col < 16; residual_col++) {
                        int token_col = residual_half * 16 + residual_col;
                        __nv_bfloat16 v_value = smem_v_all[(unsigned int)chunk_idx % 5 * 20992 + (unsigned int)(token_col * 128) + (unsigned int)state_row];
                        residual_v[residual_col] = (float)v_value;
                        residual_beta[residual_col] = smem_prep_beta_all[(unsigned int)chunk_idx % 5 * 10496 + (unsigned int)token_col];
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        sub_f32x2_inplace(&reinterpret_cast<float2*>(residual_v)[_ls], reinterpret_cast<const float2*>((prediction + residual_half * 16))[_ls]);
                    if (STORE_BACKWARD_TAPE != 0 && STORE_E_TAPE != 0 && owned_chunk != 0) {
                        unsigned int _packed_9[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 0], residual_v[_lp*2+1 + 0]));
                            _packed_9[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 2], residual_v[_lp*2+1 + 2]));
                            _packed_9[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 4], residual_v[_lp*2+1 + 4]));
                            _packed_9[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 6], residual_v[_lp*2+1 + 6]));
                            _packed_9[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(tape_e + (tape_ex_base + (long long)(residual_half * 16)))[0] = reinterpret_cast<int4*>(_packed_9)[0];
                        unsigned int _packed_10[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 8], residual_v[_lp*2+1 + 8]));
                            _packed_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 10], residual_v[_lp*2+1 + 10]));
                            _packed_10[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 12], residual_v[_lp*2+1 + 12]));
                            _packed_10[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 14], residual_v[_lp*2+1 + 14]));
                            _packed_10[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(tape_e + (tape_ex_base + (long long)(residual_half * 16) + 8))[0] = reinterpret_cast<int4*>(_packed_10)[0];
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(residual_v)[_ls], reinterpret_cast<const float2*>(residual_beta)[_ls]);
                    if (STORE_BACKWARD_TAPE != 0 && owned_chunk != 0) {
                        unsigned int _packed_11[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 0], residual_v[_lp*2+1 + 0]));
                            _packed_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 2], residual_v[_lp*2+1 + 2]));
                            _packed_11[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 4], residual_v[_lp*2+1 + 4]));
                            _packed_11[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 6], residual_v[_lp*2+1 + 6]));
                            _packed_11[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(tape_x + (tape_ex_base + (long long)(residual_half * 16)))[0] = reinterpret_cast<int4*>(_packed_11)[0];
                        unsigned int _packed_12[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 8], residual_v[_lp*2+1 + 8]));
                            _packed_12[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 10], residual_v[_lp*2+1 + 10]));
                            _packed_12[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 12], residual_v[_lp*2+1 + 12]));
                            _packed_12[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 14], residual_v[_lp*2+1 + 14]));
                            _packed_12[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(tape_x + (tape_ex_base + (long long)(residual_half * 16) + 8))[0] = reinterpret_cast<int4*>(_packed_12)[0];
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
                mbarrier_arrive(v_free_addr + ((int)((unsigned int)chunk_idx % 5)) * 8);
                if (elect_sync()) {
                    mbarrier_arrive(u_inp_ready_addr + ((int)((unsigned int)chunk_idx % 5)) * 8);
                }
                int _stage_u2_acc_ready = (int)((unsigned int)chunk_idx % 5);
                int _parity_u2_acc_ready = (int)((unsigned int)chunk_idx / 5 % 2);
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
                        unsigned int _packed_13[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((tmem_load_1 + tape_r_vec * 8)[_lp*2 + 0], (tmem_load_1 + tape_r_vec * 8)[_lp*2+1 + 0]));
                            _packed_13[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((tmem_load_1 + tape_r_vec * 8)[_lp*2 + 2], (tmem_load_1 + tape_r_vec * 8)[_lp*2+1 + 2]));
                            _packed_13[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((tmem_load_1 + tape_r_vec * 8)[_lp*2 + 4], (tmem_load_1 + tape_r_vec * 8)[_lp*2+1 + 4]));
                            _packed_13[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((tmem_load_1 + tape_r_vec * 8)[_lp*2 + 6], (tmem_load_1 + tape_r_vec * 8)[_lp*2+1 + 6]));
                            _packed_13[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(tape_r + (tape_r_base + (long long)(tape_r_vec * 8)))[0] = reinterpret_cast<int4*>(_packed_13)[0];
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
                    mbarrier_arrive(u2_inp_ready_addr + ((int)((unsigned int)chunk_idx % 5)) * 8);
                }
                int _stage_final_ready = (int)((unsigned int)chunk_idx % 5);
                int _parity_final_ready = (int)((unsigned int)chunk_idx / 5 % 2);
                mbarrier_wait(final_ready_addr + (_stage_final_ready) * 8, _parity_final_ready);
                asm volatile("tcgen05.fence::after_thread_sync;");
                {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(smem_free_addr + ((int)((unsigned int)chunk_idx % 5)) * 8);
                }
            }
            if (store_final_state != 0) {
                #pragma unroll
                for (int state_col_block_3 = 0; state_col_block_3 < 4; state_col_block_3++) {
                    float tmem_load_2[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(tmem_load_2[0]), "=f"(tmem_load_2[1]), "=f"(tmem_load_2[2]), "=f"(tmem_load_2[3]), "=f"(tmem_load_2[4]), "=f"(tmem_load_2[5]), "=f"(tmem_load_2[6]), "=f"(tmem_load_2[7]), "=f"(tmem_load_2[8]), "=f"(tmem_load_2[9]), "=f"(tmem_load_2[10]), "=f"(tmem_load_2[11]), "=f"(tmem_load_2[12]), "=f"(tmem_load_2[13]), "=f"(tmem_load_2[14]), "=f"(tmem_load_2[15]), "=f"(tmem_load_2[16]), "=f"(tmem_load_2[17]), "=f"(tmem_load_2[18]), "=f"(tmem_load_2[19]), "=f"(tmem_load_2[20]), "=f"(tmem_load_2[21]), "=f"(tmem_load_2[22]), "=f"(tmem_load_2[23]), "=f"(tmem_load_2[24]), "=f"(tmem_load_2[25]), "=f"(tmem_load_2[26]), "=f"(tmem_load_2[27]), "=f"(tmem_load_2[28]), "=f"(tmem_load_2[29]), "=f"(tmem_load_2[30]), "=f"(tmem_load_2[31])
                        : "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_3 * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int state_vec = 0; state_vec < 4; state_vec++) {
                        {
                            unsigned _stv8_2_0 = __float_as_uint(tmem_load_2[state_vec * 8 + 0]);
                            unsigned _stv8_2_1 = __float_as_uint(tmem_load_2[state_vec * 8 + 1]);
                            unsigned _stv8_2_2 = __float_as_uint(tmem_load_2[state_vec * 8 + 2]);
                            unsigned _stv8_2_3 = __float_as_uint(tmem_load_2[state_vec * 8 + 3]);
                            unsigned _stv8_2_4 = __float_as_uint(tmem_load_2[state_vec * 8 + 4]);
                            unsigned _stv8_2_5 = __float_as_uint(tmem_load_2[state_vec * 8 + 5]);
                            unsigned _stv8_2_6 = __float_as_uint(tmem_load_2[state_vec * 8 + 6]);
                            unsigned _stv8_2_7 = __float_as_uint(tmem_load_2[state_vec * 8 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(final_state_f32 + (state_base + (long long)(state_col_block_3 * 32) + (long long)(state_vec * 8)) + (0))), "r"(_stv8_2_0), "r"(_stv8_2_1), "r"(_stv8_2_2), "r"(_stv8_2_3), "r"(_stv8_2_4), "r"(_stv8_2_5), "r"(_stv8_2_6), "r"(_stv8_2_7) : "memory");
                        }
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            __threadfence();
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
            int split_compute_start_1 = 0;
            unsigned int _phase_work_item_ready_0 = 0;
            int seq_idx_1 = seq_order[task_idx_1 / num_heads];
            int head_idx_1 = task_idx_1 % num_heads;
            long long bos_1 = cu_seqlens[seq_idx_1];
            long long eos_1 = cu_seqlens[seq_idx_1 + 1];
            int num_chunks_1 = ((int)(eos_1 - bos_1) + 32 - 1) / 32;
            int seq_len_1 = (int)(eos_1 - bos_1);
            int num_chunks_0_1 = (seq_len_1 + 32 - 1) / 32;
            int warp_id_in_role_1 = warp - 4;
            int epilogue_local_warp = warp_id_in_role_1;
            int warp_in_wg_1 = warp % 4;
            const int tmem_row_base_1 = warp_in_wg_1 * 32 << 16;
            int state_row_1 = warp_in_wg_1 * 32 + lane;
            unsigned int checkpoint_stage_epilogue = 0;
            unsigned int snapshot_stage_epilogue = 0;
            long long epilogue_checkpoint_row_start = 0;
            long long epilogue_checkpoint_single_token = 0;
            if (checkpoint_every_n_tokens != 0) {
                epilogue_checkpoint_row_start = reinterpret_cast<long long*>(checkpoint_cu_starts_addr)[seq_idx_1];
            }
            int epilogue_chunks = num_chunks_0_1;
            #pragma unroll 1
            for (int chunk_idx_1 = 0; chunk_idx_1 < epilogue_chunks; chunk_idx_1++) {
                int checkpoint_token_epilogue = chunk_idx_1 * 32;
                int checkpoint_entering_epilogue = checkpoint_every_n_tokens != 0 && checkpoint_token_epilogue % checkpoint_every_n_tokens == 0;
                int chunk_is_full = ((seq_len_1 >= (chunk_idx_1 + 1) * 32) ? 1 : 0);
                if (chunk_is_full != 0) {
                    int _stage_final_ready_1 = (int)((unsigned int)chunk_idx_1 % 5);
                    int _parity_final_ready_1 = (int)((unsigned int)chunk_idx_1 / 5 % 2);
                    mbarrier_wait(final_ready_addr + (_stage_final_ready_1) * 8, _parity_final_ready_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_4[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1));
                    float _tmem_load_5[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
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
                        if (chunk_idx_1 >= 2) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    int out_stage_addr = smem_out_addr + (unsigned int)chunk_idx_1 % 2 * 8192;
                    #pragma unroll
                    for (int dim_half = 0; dim_half < 2; dim_half++) {
                        unsigned int out_packed[8];
                        if (dim_half == 0) {
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                                out_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                        } else {
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
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
                            tma_store_4d((&out_tma), 0, (int)(bos_1 + (long long)(chunk_idx_1 * 32)), head_idx_1, 0, smem_out_addr + (unsigned int)chunk_idx_1 % 2 * 8192);
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
                            float tmem_load_3[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(tmem_load_3[0]), "=f"(tmem_load_3[1]), "=f"(tmem_load_3[2]), "=f"(tmem_load_3[3]), "=f"(tmem_load_3[4]), "=f"(tmem_load_3[5]), "=f"(tmem_load_3[6]), "=f"(tmem_load_3[7]), "=f"(tmem_load_3[8]), "=f"(tmem_load_3[9]), "=f"(tmem_load_3[10]), "=f"(tmem_load_3[11]), "=f"(tmem_load_3[12]), "=f"(tmem_load_3[13]), "=f"(tmem_load_3[14]), "=f"(tmem_load_3[15]), "=f"(tmem_load_3[16]), "=f"(tmem_load_3[17]), "=f"(tmem_load_3[18]), "=f"(tmem_load_3[19]), "=f"(tmem_load_3[20]), "=f"(tmem_load_3[21]), "=f"(tmem_load_3[22]), "=f"(tmem_load_3[23]), "=f"(tmem_load_3[24]), "=f"(tmem_load_3[25]), "=f"(tmem_load_3[26]), "=f"(tmem_load_3[27]), "=f"(tmem_load_3[28]), "=f"(tmem_load_3[29]), "=f"(tmem_load_3[30]), "=f"(tmem_load_3[31])
                                : "r"(taddr + 64 + (unsigned int)tmem_row_base_1 + (unsigned int)(snapshot_panel * 32)));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (elect_sync()) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(checkpoint_snapshot_done_addr + (snapshot_stage_epilogue) * 8);
                            }
                            snapshot_stage_epilogue += 1;
                            if (snapshot_stage_epilogue == 4) { snapshot_stage_epilogue = 0; }
                            if (epilogue_local_warp == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile("barrier.sync 8, 128;" ::: "memory");
                            unsigned int tmem_load_6_bf16[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_3[_lp*2 + 0], tmem_load_3[_lp*2+1 + 0]));
                                tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 ^ (state_row_1 * 128 >> 7 & 7) << 4))), "r"(tmem_load_6_bf16[0]), "r"(tmem_load_6_bf16[1]), "r"(tmem_load_6_bf16[2]), "r"(tmem_load_6_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 16 ^ (state_row_1 * 128 + 16 >> 7 & 7) << 4))), "r"(tmem_load_6_bf16[4]), "r"(tmem_load_6_bf16[5]), "r"(tmem_load_6_bf16[6]), "r"(tmem_load_6_bf16[7]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 32 ^ (state_row_1 * 128 + 32 >> 7 & 7) << 4))), "r"(tmem_load_6_bf16[8]), "r"(tmem_load_6_bf16[9]), "r"(tmem_load_6_bf16[10]), "r"(tmem_load_6_bf16[11]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 48 ^ (state_row_1 * 128 + 48 >> 7 & 7) << 4))), "r"(tmem_load_6_bf16[12]), "r"(tmem_load_6_bf16[13]), "r"(tmem_load_6_bf16[14]), "r"(tmem_load_6_bf16[15]) : "memory");
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
                    int _stage_final_ready_2 = (int)((unsigned int)chunk_idx_1 % 5);
                    int _parity_final_ready_2 = (int)((unsigned int)chunk_idx_1 / 5 % 2);
                    mbarrier_wait(final_ready_addr + (_stage_final_ready_2) * 8, _parity_final_ready_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_7[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
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
                            out[out_idx] = _tmem_load_7[token_col_1];
                        }
                    }
                }
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
    // ---- Role: beta_prefetch ----
    } else if (warp == 8) {
        // idle — no tasks assigned
    // ---- Role: aux_mma ----
    } else if (warp >= 10 && warp <= 11) {
        { // aux_mma_main
            unsigned int _phase_work_item_ready_0_1 = 0;
        }
    // ---- Role: mma ----
    } else if (warp == 9) {
        { // mma_main
            int task_idx_2 = blockIdx.x;
            int split_compute_start_2 = 0;
            unsigned int _phase_work_item_ready_0_2 = 0;
            int seq_idx_2 = seq_order[task_idx_2 / num_heads];
            int head_idx_2 = task_idx_2 % num_heads;
            long long bos_2 = cu_seqlens[seq_idx_2];
            long long eos_2 = cu_seqlens[seq_idx_2 + 1];
            int num_chunks_2 = ((int)(eos_2 - bos_2) + 32 - 1) / 32;
            int seq_len_2 = (int)(eos_2 - bos_2);
            int num_chunks_0_2 = (seq_len_2 + 32 - 1) / 32;
            unsigned int _phase_out_empty_0 = 1;
            #pragma unroll 1
            for (int _chunk_idx = 0; _chunk_idx < num_chunks_0_2; _chunk_idx++) {
                int _stage_qk_full_1 = (int)((unsigned int)_chunk_idx % 5);
                int _parity_qk_full_1 = (int)((unsigned int)_chunk_idx / 5 % 2);
                mbarrier_wait(qk_full_addr + (_stage_qk_full_1) * 8, _parity_qk_full_1);
                {
                    mbarrier_wait(out_empty_addr, _phase_out_empty_0);
                    _phase_out_empty_0 ^= 1;
                }
                int _stage_state_inp_ready = (int)((unsigned int)_chunk_idx % 5);
                int _parity_state_inp_ready = (int)((unsigned int)_chunk_idx / 5 % 2);
                mbarrier_wait(state_inp_ready_addr + (_stage_state_inp_ready) * 8, _parity_state_inp_ready);
                asm volatile("tcgen05.fence::after_thread_sync;");
                {
                    int _mma_b_lo_0 = make_warp_uniform((((smem_qd_addr) >> 4) & 0x3FFF) + ((unsigned int)_chunk_idx % 5) * 2624);
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
                    :: "r"(tmem_tmem_out), "r"(_mma_b_lo_0), "r"(tmem_tmem_state_inp), "r"(0) : "memory");
                }
                int _mma_b_lo_1 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + ((unsigned int)_chunk_idx % 5) * 2624);
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
                    :: "r"(tmem_tmem_u_acc), "r"(_mma_b_lo_1), "r"(tmem_tmem_state_inp), "r"(0) : "memory");
                elect_commit2(old_out_ready_addr + ((int)((unsigned int)_chunk_idx % 5)) * 8, raw_inputs_free_addr + ((int)((unsigned int)_chunk_idx % 5)) * 8);
                int _stage_u_inp_ready = (int)((unsigned int)_chunk_idx % 5);
                int _parity_u_inp_ready = (int)((unsigned int)_chunk_idx / 5 % 2);
                mbarrier_wait(u_inp_ready_addr + (_stage_u_inp_ready) * 8, _parity_u_inp_ready);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_2 = make_warp_uniform((((smem_inv_addr) >> 4) & 0x3FFF) + ((unsigned int)_chunk_idx % 5) * 2624);
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
                    :: "r"(tmem_tmem_u2_acc), "r"(_mma_b_lo_2), "r"(tmem_tmem_u2_inp), "r"(0) : "memory");
                elect_commit(u2_acc_ready_addr + ((int)((unsigned int)_chunk_idx % 5)) * 8);
                int _stage_u2_inp_ready = (int)((unsigned int)_chunk_idx % 5);
                int _parity_u2_inp_ready = (int)((unsigned int)_chunk_idx / 5 % 2);
                mbarrier_wait(u2_inp_ready_addr + (_stage_u2_inp_ready) * 8, _parity_u2_inp_ready);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_3 = make_warp_uniform(((((smem_kr_trans_addr) >> 4) & 0x3FFF) | 0x1000000) + ((unsigned int)_chunk_idx % 5) * 2624);
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
                    "}\n"
                    :: "r"(tmem_tmem_state), "r"(_mma_b_lo_3), "r"(tmem_tmem_u2_inp), "r"(1) : "memory");
                int _mma_b_lo_4 = make_warp_uniform(((((smem_final_mqk_slab_addr) >> 4) & 0x3FFF) | 0x1000000) + ((unsigned int)_chunk_idx % 5) * 2624);
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
                    "mov.b32 id, 134808720;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_out), "r"(_mma_b_lo_4), "r"(tmem_tmem_u2_inp), "r"(1) : "memory");
                elect_commit(final_ready_addr + ((int)((unsigned int)_chunk_idx % 5)) * 8);
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(final_ready_addr + ((int)((unsigned int)_chunk_idx % 5)) * 8);
                }
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
            int split_compute_start_3 = 0;
            unsigned int _phase_work_item_ready_0_3 = 0;
            int seq_idx_3 = seq_order[task_idx_3 / num_heads];
            int head_idx_3 = task_idx_3 % num_heads;
            long long bos_3 = cu_seqlens[seq_idx_3];
            long long eos_3 = cu_seqlens[seq_idx_3 + 1];
            int num_chunks_3 = ((int)(eos_3 - bos_3) + 32 - 1) / 32;
            int seq_len_3 = (int)(eos_3 - bos_3);
            int num_chunks_0_3 = (seq_len_3 + 32 - 1) / 32;
            int instance_id = (warp - 12) / 4;
            int prep_instance = instance_id;
            int warp_id_in_role_2 = warp - 12;
            int prep_local_warp = warp_id_in_role_2 - prep_instance * 4;
            int prep_tid = prep_local_warp * 32 + lane;
            int num_prep_iters = (num_chunks_0_3 + 5 - 1 - prep_instance) / 5;
            int gate_rate_stage_f32 = prep_instance * 10496;
            int prep_global_tid = warp_id_in_role_2 * 32 + lane;
            if (prep_tid == 0) {
                float _exp_0 = __expf(A_log[head_idx_3]);
                smem_gate_rate_all[gate_rate_stage_f32] = _exp_0;
            }
            if (prep_global_tid < 128) {
                smem_gate_bias_all[prep_global_tid] = dt_bias[head_idx_3 * 128 + prep_global_tid];
            }
            asm volatile("barrier.sync 15, 640;" ::: "memory");
            if (prep_global_tid < 128) {
                smem_gate_bias_all[prep_global_tid] = smem_gate_bias_all[prep_global_tid] * smem_gate_rate_all[0];
            }
            asm volatile("barrier.sync 15, 640;" ::: "memory");
            #pragma unroll 1
            for (int prep_iter = 0; prep_iter < num_prep_iters; prep_iter++) {
                int chunk_idx_2 = prep_iter * 5 + prep_instance;
                int chunk_global_local_1 = chunk_idx_2;
                int owned_chunk_1 = chunk_global_local_1 >= 0 && chunk_global_local_1 < num_chunks_3;
                int stage_f32 = (unsigned int)chunk_idx_2 % 5 * 10496;
                int stage_bf16 = (unsigned int)chunk_idx_2 % 5 * 20992;
                int chunk_is_full_1 = ((seq_len_3 >= (chunk_idx_2 + 1) * 32) ? 1 : 0);
                float early_beta_value = 0.0f;
                float early_gate0 = 0.0f;
                int _parity_raw_inputs_free = (int)((unsigned int)prep_iter % 2) ^ 1;
                mbarrier_wait(raw_inputs_free_addr + ((unsigned int)chunk_idx_2 % 5) * 8, _parity_raw_inputs_free);
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(gate_raw_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8, 8192);
                            tma_3d_gmem2smem(smem_g_raw_addr + (unsigned int)chunk_idx_2 % 5 * 41984, (&g_tma), 0, head_idx_3, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), gate_raw_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8);
                            mbarrier_arrive_expect_tx(qk_raw_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8, 16384);
                            tma_4d_gmem2smem(smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984, (&k_tma), 0, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), head_idx_3, 0, qk_raw_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8);
                        }
                    }
                } else if (prep_local_warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(gate_raw_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8);
                        mbarrier_arrive(qk_raw_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8);
                    }
                }
                int _parity_gate_raw_full = (int)((unsigned int)prep_iter % 2);
                mbarrier_wait(gate_raw_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8, _parity_gate_raw_full);
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 2 && lane < 32) {
                        float beta_logit = 0.0f;
                        long long beta_token = bos_3 + (long long)(chunk_idx_2 * 32 + lane);
                        beta_logit = (float)beta[beta_token * beta_token_stride + (long long)head_idx_3];
                        float _tanh_1;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_1) : "f"(beta_logit * 0.5f));
                        early_beta_value = __fmaf_rn(_tanh_1, 0.5f, 0.5f);
                    }
                }
                int _parity_smem_free = (int)((unsigned int)prep_iter % 2) ^ 1;
                mbarrier_wait(smem_free_addr + ((unsigned int)chunk_idx_2 % 5) * 8, _parity_smem_free);
                int _parity_v_free = (int)((unsigned int)prep_iter % 2) ^ 1;
                mbarrier_wait(v_free_addr + ((unsigned int)chunk_idx_2 % 5) * 8, _parity_v_free);
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            tma_4d_gmem2smem(smem_q_raw_prefetch_addr + (unsigned int)chunk_idx_2 % 5 * 41984, (&q_tma), 0, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), head_idx_3, 0, qk_raw_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8);
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
                            :: "r"(smem_g_raw_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(gate_load_item * 16)), "l"(g + gate_load_base), "r"((gate_load_token < eos_3) ? 16 : 0));
                        int q_tail_addr = (smem_q_raw_prefetch_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(gate_load_segment * 8 / 64 * 4096 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 ^ (gate_load_segment * 8 / 64 * 4096 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(q_tail_addr), "l"(q + q_load_base), "r"((gate_load_token < eos_3) ? 16 : 0));
                        int k_tail_addr = (smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(gate_load_segment * 8 / 64 * 4096 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 ^ (gate_load_segment * 8 / 64 * 4096 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 >> 7 & 7) << 4));
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
                    long long beta_token_1 = bos_3 + (long long)(chunk_idx_2 * 32 + lane);
                    float beta_value = early_beta_value;
                    if (chunk_is_full_1 == 0) {
                        if (beta_token_1 < eos_3) {
                            float beta_logit_1 = (float)beta[beta_token_1 * (long long)num_heads + (long long)head_idx_3];
                            beta_logit_1 = (float)beta[beta_token_1 * beta_token_stride + (long long)head_idx_3];
                            float _tanh_2;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_2) : "f"(beta_logit_1 * 0.5f));
                            beta_value = __fmaf_rn(_tanh_2, 0.5f, 0.5f);
                        }
                    }
                    smem_prep_beta_all[stage_f32 + lane] = beta_value;
                }
                if (prep_tid < 128) {
                    int gate_col = prep_tid;
                    float gate_rate = smem_gate_rate_all[stage_f32];
                    float gate_bias = smem_gate_bias_all[gate_col];
                    float prefix_log2 = 0.0f;
                    float2 gate_rate_pair = make_float2(gate_rate, gate_rate);
                    float2 gate_bias_pair = make_float2(gate_bias, gate_bias);
                    float half_gate_scale = lower_bound * 0.7213475f;
                    float2 half_gate_scale_pair = make_float2(half_gate_scale, half_gate_scale);
                    if (chunk_is_full_1 != 0) {
                        for (int gate_row_pair = 0; gate_row_pair < 16; gate_row_pair++) {
                            const int gate_row0 = gate_row_pair * 2;
                            const int gate_row1 = gate_row0 + 1;
                            const int gate_row0_0 = gate_row_pair * 2;
                            const int gate_row1_1 = gate_row0_0 + 1;
                            __nv_bfloat16 gate_raw0 = smem_g_raw_all[stage_bf16 + gate_row0_0 * 128 + gate_col];
                            __nv_bfloat16 gate_raw1 = smem_g_raw_all[stage_bf16 + gate_row1_1 * 128 + gate_col];
                            float2 gate_raw_pair = make_float2((float)gate_raw0, (float)gate_raw1);
                            float2 gate_arg_pair = fma_f32x2_rn_ftz(gate_raw_pair, gate_rate_pair, gate_bias_pair);
                            float _tanh_3;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_3) : "f"(gate_arg_pair.x * 0.5f));
                            float _tanh_4;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_4) : "f"(gate_arg_pair.y * 0.5f));
                            float2 gate_tanh_pair = make_float2(_tanh_3, _tanh_4);
                            float2 gate_log_pair = fma_f32x2_rn_ftz(gate_tanh_pair, half_gate_scale_pair, half_gate_scale_pair);
                            float gate_log0 = gate_log_pair.x;
                            float gate_log1 = gate_log_pair.y;
                            prefix_log2 += gate_log0;
                            smem_gate_all[stage_f32 + gate_row0 * 128 + gate_col] = prefix_log2;
                            prefix_log2 += gate_log1;
                            smem_gate_all[stage_f32 + gate_row1 * 128 + gate_col] = prefix_log2;
                        }
                    } else {
                        for (int gate_row_pair_1 = 0; gate_row_pair_1 < 16; gate_row_pair_1++) {
                            const int gate_row0_1 = gate_row_pair_1 * 2;
                            const int gate_row1_2 = gate_row0_1 + 1;
                            const int gate_row0_0_1 = gate_row_pair_1 * 2;
                            const int gate_row1_1_1 = gate_row0_0_1 + 1;
                            __nv_bfloat16 gate_raw0_1 = smem_g_raw_all[stage_bf16 + gate_row0_0_1 * 128 + gate_col];
                            __nv_bfloat16 gate_raw1_1 = smem_g_raw_all[stage_bf16 + gate_row1_1_1 * 128 + gate_col];
                            float2 gate_raw_pair_1 = make_float2((float)gate_raw0_1, (float)gate_raw1_1);
                            float2 gate_arg_pair_1 = fma_f32x2_rn_ftz(gate_raw_pair_1, gate_rate_pair, gate_bias_pair);
                            float _tanh_5;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_5) : "f"(gate_arg_pair_1.x * 0.5f));
                            float _tanh_6;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_6) : "f"(gate_arg_pair_1.y * 0.5f));
                            float2 gate_tanh_pair_1 = make_float2(_tanh_5, _tanh_6);
                            float2 gate_log_pair_1 = fma_f32x2_rn_ftz(gate_tanh_pair_1, half_gate_scale_pair, half_gate_scale_pair);
                            float gate_log0_1 = gate_log_pair_1.x;
                            float gate_log1_1 = gate_log_pair_1.y;
                            long long gate_token0 = bos_3 + (long long)(chunk_idx_2 * 32 + gate_row0_0_1);
                            long long gate_token1 = gate_token0 + 1;
                            if (gate_token0 >= eos_3) {
                                gate_log0_1 = 0.0f;
                            }
                            if (gate_token1 >= eos_3) {
                                gate_log1_1 = 0.0f;
                            }
                            prefix_log2 += gate_log0_1;
                            smem_gate_all[stage_f32 + gate_row0_1 * 128 + gate_col] = prefix_log2;
                            prefix_log2 += gate_log1_1;
                            smem_gate_all[stage_f32 + gate_row1_2 * 128 + gate_col] = prefix_log2;
                        }
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                float page_total_log2 = 0.0f;
                int _parity_qk_raw_full = (int)((unsigned int)prep_iter % 2);
                mbarrier_wait(qk_raw_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8, _parity_qk_raw_full);
                if (prep_tid < 128) {
                    float total_log2 = smem_gt_prefix[stage_f32 + prep_tid];
                    float restore_factor_value = approx_exp2(__fmaf_rn(-(lower_bound * 1.442695f), 16.0f, total_log2));
                    smem_restore_factor_all[stage_f32 + prep_tid] = restore_factor_value;
                }
                if (prep_tid == 0) {
                    float _exp2_7 = approx_exp2(lower_bound * 1.442695f * 16.0f);
                    smem_restore_factor_all[stage_f32 + 128] = _exp2_7;
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
                    unsigned int packed_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 3]))
                        : "r"((smem_q_raw_prefetch_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                    float unpack_bf16[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&unpack_bf16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_2[_pair]) << 16);
                        (&unpack_bf16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_2[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int value_idx = 0; value_idx < 8; value_idx++) {
                        q_raw_vec[value_idx] = unpack_bf16[value_idx];
                    }
                    unsigned int packed_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 3]))
                        : "r"((smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
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
                    q_inv *= scale;
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
                        float prefix = smem_gate_all[stage_f32 + row * 128 + col];
                        float common_log2 = lower_bound * 1.442695f * 16.0f;
                        float decay = approx_exp2(prefix - common_log2);
                        qd_vec[elem_in_segment] = decay;
                        kd_vec[elem_in_segment] = decay;
                        ki_vec[elem_in_segment] = k_raw_vec[elem_in_segment] / decay;
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], reinterpret_cast<const float2*>(q_raw_vec)[_ls]);
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(kd_vec)[_ls], reinterpret_cast<const float2*>(k_raw_vec)[_ls]);
                    unsigned int packed_1_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                        packed_1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word = 0; word < 4; word++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word * 4)), "r"(packed_1_1[word]));
                    }
                    unsigned int packed_2_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_vec[_lp*2 + 0], kd_vec[_lp*2+1 + 0]));
                        packed_2_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_1 = 0; word_1 < 4; word_1++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_1 * 4)), "r"(packed_2_1[word_1]));
                    }
                    unsigned int packed_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_vec[_lp*2 + 0], ki_vec[_lp*2+1 + 0]));
                        packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_2 = 0; word_2 < 4; word_2++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_2 * 4)), "r"(packed_3[word_2]));
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                unsigned int a_frag[4];
                unsigned int b_frag[4];
                float acc[8];
                int pair_row_base = prep_local_warp / 2 * 16;
                int pair_col_base = prep_local_warp % 2 * 16;
                if (pair_row_base >= pair_col_base) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    if (pair_row_base == pair_col_base) {
                        float diagonal_inverse[8];
                        int row0 = lane / 4;
                        int row1 = row0 + 8;
                        int col0 = lane % 4 * 2;
                        float beta0 = 0.0f;
                        float beta1 = 0.0f;
                        beta0 = smem_prep_beta_all[stage_f32 + pair_row_base + row0];
                        beta1 = smem_prep_beta_all[stage_f32 + pair_row_base + row1];
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
                        for (int word_3 = 0; word_3 < 4; word_3++) {
                            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                                : "=r"(rhs_trans_frag[word_3])
                                : "r"(allmma_d_frag[word_3]));
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
                        for (int word_4 = 0; word_4 < 4; word_4++) {
                            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                                : "=r"(rhs_trans_frag_0[word_4])
                                : "r"(d2_frag[word_4]));
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                            : "r"(n_frag[0]), "r"(n_frag[1]), "r"(n_frag[2]), "r"(n_frag[3]), "r"(rhs_trans_frag_0[0]), "r"(rhs_trans_frag_0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                            : "r"(n_frag[0]), "r"(n_frag[1]), "r"(n_frag[2]), "r"(n_frag[3]), "r"(rhs_trans_frag_0[2]), "r"(rhs_trans_frag_0[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        #pragma unroll
                        for (int value_idx_2 = 0; value_idx_2 < 8; value_idx_2++) {
                            n_values[value_idx_2] = n_values[value_idx_2] + product[value_idx_2];
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
                        for (int word_5 = 0; word_5 < 4; word_5++) {
                            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                                : "=r"(rhs_trans_frag_1[word_5])
                                : "r"(d2_frag[word_5]));
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
                        for (int word_6 = 0; word_6 < 4; word_6++) {
                            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                                : "=r"(rhs_trans_frag_2[word_6])
                                : "r"(d4_frag[word_6]));
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                            : "r"(n_frag[0]), "r"(n_frag[1]), "r"(n_frag[2]), "r"(n_frag[3]), "r"(rhs_trans_frag_2[0]), "r"(rhs_trans_frag_2[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                            : "r"(n_frag[0]), "r"(n_frag[1]), "r"(n_frag[2]), "r"(n_frag[3]), "r"(rhs_trans_frag_2[2]), "r"(rhs_trans_frag_2[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        #pragma unroll
                        for (int value_idx_3 = 0; value_idx_3 < 8; value_idx_3++) {
                            n_values[value_idx_3] = n_values[value_idx_3] + product[value_idx_3];
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
                        for (int word_7 = 0; word_7 < 4; word_7++) {
                            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                                : "=r"(rhs_trans_frag_3[word_7])
                                : "r"(a21_frag[word_7]));
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
                        for (int word_8 = 0; word_8 < 4; word_8++) {
                            asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                                : "=r"(rhs_trans_frag_4[word_8])
                                : "r"(binv_frag[word_8]));
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
                        for (int value_idx_4 = 0; value_idx_4 < 8; value_idx_4++) {
                            diagonal_inverse[value_idx_4] = n_values[value_idx_4];
                        }
                        unsigned int inverse_store_packed[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(diagonal_inverse[_lp*2 + 0], diagonal_inverse[_lp*2+1 + 0]));
                            inverse_store_packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        int inverse_lane_row = lane % 16;
                        int inverse_lane_col = lane / 16 * 8;
                        int byte_off = (int)((unsigned int)chunk_idx_2 % 5) * 41984 + (pair_row_base + inverse_lane_row) * 128 + (pair_row_base + inverse_lane_col) * 2;
                        int swizzled_off = byte_off ^ (byte_off >> 7 & 7) << 4;
                        int inverse_addr = smem_inv_work_addr + (unsigned int)swizzled_off;
                        uint32_t _stmatrix_addr_18 = static_cast<uint32_t>((unsigned long long)inverse_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&inverse_store_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&inverse_store_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&inverse_store_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&inverse_store_packed[3]))
                            : "memory");
                    } else {
                        int row0_1 = pair_row_base + lane / 4;
                        int row1_1 = row0_1 + 8;
                        int col0_1 = pair_col_base + lane % 4 * 2;
                        float beta0_1 = 0.0f;
                        float beta1_1 = 0.0f;
                        beta0_1 = smem_prep_beta_all[stage_f32 + row0_1];
                        beta1_1 = smem_prep_beta_all[stage_f32 + row1_1];
                        float seed[8];
                        seed[0] = 0.0f;
                        seed[1] = 0.0f;
                        seed[2] = 0.0f;
                        seed[3] = 0.0f;
                        seed[4] = 0.0f;
                        seed[5] = 0.0f;
                        seed[6] = 0.0f;
                        seed[7] = 0.0f;
                        if (row0_1 > col0_1) {
                            seed[0] = acc[0] * beta0_1;
                        }
                        if (row0_1 > col0_1 + 1) {
                            seed[1] = acc[1] * beta0_1;
                        }
                        if (row1_1 > col0_1) {
                            seed[2] = acc[2] * beta1_1;
                        }
                        if (row1_1 > col0_1 + 1) {
                            seed[3] = acc[3] * beta1_1;
                        }
                        if (row0_1 > col0_1 + 8) {
                            seed[4] = acc[4] * beta0_1;
                        }
                        if (row0_1 > col0_1 + 9) {
                            seed[5] = acc[5] * beta0_1;
                        }
                        if (row1_1 > col0_1 + 8) {
                            seed[6] = acc[6] * beta1_1;
                        }
                        if (row1_1 > col0_1 + 9) {
                            seed[7] = acc[7] * beta1_1;
                        }
                        unsigned int seed_packed[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(seed[_lp*2 + 0], seed[_lp*2+1 + 0]));
                            seed_packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        int seed_lane_row = lane % 16;
                        int seed_lane_col = lane / 16 * 8;
                        int byte_off_1 = (int)((unsigned int)chunk_idx_2 % 5) * 41984 + (pair_row_base + seed_lane_row) * 128 + (pair_col_base + seed_lane_col) * 2;
                        int swizzled_off_1 = byte_off_1 ^ (byte_off_1 >> 7 & 7) << 4;
                        int seed_addr = smem_inv_work_addr + (unsigned int)swizzled_off_1;
                        uint32_t _stmatrix_addr_19 = static_cast<uint32_t>((unsigned long long)seed_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[3]))
                            : "memory");
                    }
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
                    int row0_2 = lane / 4;
                    int row1_2 = row0_2 + 8;
                    int col0_2 = 16 + lane % 4 * 2;
                    float mqk[8];
                    mqk[0] = 0.0f;
                    mqk[1] = 0.0f;
                    mqk[2] = 0.0f;
                    mqk[3] = 0.0f;
                    mqk[4] = 0.0f;
                    mqk[5] = 0.0f;
                    mqk[6] = 0.0f;
                    mqk[7] = 0.0f;
                    if (row0_2 >= col0_2) {
                        mqk[0] = acc[0];
                    }
                    if (row0_2 >= col0_2 + 1) {
                        mqk[1] = acc[1];
                    }
                    if (row1_2 >= col0_2) {
                        mqk[2] = acc[2];
                    }
                    if (row1_2 >= col0_2 + 1) {
                        mqk[3] = acc[3];
                    }
                    if (row0_2 >= col0_2 + 8) {
                        mqk[4] = acc[4];
                    }
                    if (row0_2 >= col0_2 + 9) {
                        mqk[5] = acc[5];
                    }
                    if (row1_2 >= col0_2 + 8) {
                        mqk[6] = acc[6];
                    }
                    if (row1_2 >= col0_2 + 9) {
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
                        uint32_t _stmatrix_addr_20 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(publish_col / 64 * 4096 + publish_row * 128 + publish_col % 64 * 2 ^ (publish_col / 64 * 4096 + publish_row * 128 + publish_col % 64 * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_20), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2 + 1]))
                            : "memory");
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
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
                        uint32_t _stmatrix_addr_21 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(publish_col_1 / 64 * 4096 + publish_row_1 * 128 + publish_col_1 % 64 * 2 ^ (publish_col_1 / 64 * 4096 + publish_row_1 * 128 + publish_col_1 % 64 * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_21), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_0[publish_pair_1 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_0[publish_pair_1 * 2 + 1]))
                            : "memory");
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
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
                        uint32_t _stmatrix_addr_22 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(publish_col_2 / 64 * 4096 + publish_row_2 * 128 + publish_col_2 % 64 * 2 ^ (publish_col_2 / 64 * 4096 + publish_row_2 * 128 + publish_col_2 % 64 * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_1[publish_pair_2 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_1[publish_pair_2 * 2 + 1]))
                            : "memory");
                    }
                } else if (prep_local_warp == 2) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    int row0_3 = lane / 4;
                    int row1_3 = row0_3 + 8;
                    int col0_3 = lane % 4 * 2;
                    float mqk_2[8];
                    mqk_2[0] = 0.0f;
                    mqk_2[1] = 0.0f;
                    mqk_2[2] = 0.0f;
                    mqk_2[3] = 0.0f;
                    mqk_2[4] = 0.0f;
                    mqk_2[5] = 0.0f;
                    mqk_2[6] = 0.0f;
                    mqk_2[7] = 0.0f;
                    if (row0_3 >= col0_3) {
                        mqk_2[0] = acc[0];
                    }
                    if (row0_3 >= col0_3 + 1) {
                        mqk_2[1] = acc[1];
                    }
                    if (row1_3 >= col0_3) {
                        mqk_2[2] = acc[2];
                    }
                    if (row1_3 >= col0_3 + 1) {
                        mqk_2[3] = acc[3];
                    }
                    if (row0_3 >= col0_3 + 8) {
                        mqk_2[4] = acc[4];
                    }
                    if (row0_3 >= col0_3 + 9) {
                        mqk_2[5] = acc[5];
                    }
                    if (row1_3 >= col0_3 + 8) {
                        mqk_2[6] = acc[6];
                    }
                    if (row1_3 >= col0_3 + 9) {
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
                        uint32_t _stmatrix_addr_23 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(publish_col_3 / 64 * 4096 + publish_row_3 * 128 + publish_col_3 % 64 * 2 ^ (publish_col_3 / 64 * 4096 + publish_row_3 * 128 + publish_col_3 % 64 * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_2[publish_pair_3 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_2[publish_pair_3 * 2 + 1]))
                            : "memory");
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                long long tape_scaled_base = 0;
                if (prep_tid < 128) {
                    float total_log2_1 = smem_gt_prefix[stage_f32 + prep_tid];
                    float _exp2_24 = approx_exp2(total_log2_1);
                    smem_gt_all[stage_f32 + prep_tid] = _exp2_24;
                }
                if (prep_local_warp >= 2) {
                    int stage_f32_0 = (unsigned int)chunk_idx_2 % 5 * 10496;
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
                        float restore_kr_values[8];
                        float restore_qd_values[8];
                        float restore_kd_values[8];
                        float restore_ki_values[8];
                        unsigned int packed_4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 3]))
                            : "r"((smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_4[_pair]) << 16);
                            (&unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_4[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx_5 = 0; value_idx_5 < 8; value_idx_5++) {
                            restore_qd_values[value_idx_5] = unpack_bf16_1[value_idx_5];
                        }
                        unsigned int packed_4_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4_1[(0) + 3]))
                            : "r"((smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_1_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_4_1[_pair]) << 16);
                            (&unpack_bf16_1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_4_1[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx_6 = 0; value_idx_6 < 8; value_idx_6++) {
                            restore_ki_values[value_idx_6] = unpack_bf16_1_1[value_idx_6];
                        }
                        if (STORE_BACKWARD_TAPE != 0 && owned_chunk_1 != 0) {
                            unsigned int packed_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 3]))
                                : "r"((smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float unpack_bf16_2[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_5[_pair]) << 16);
                                (&unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_5[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int value_idx_7 = 0; value_idx_7 < 8; value_idx_7++) {
                                restore_kd_values[value_idx_7] = unpack_bf16_2[value_idx_7];
                            }
                            long long tape_scaled_index = tape_scaled_base + (long long)restore_row * 128 + (long long)(restore_segment * 8);
                            unsigned int packed_6[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values[_lp*2 + 0], restore_qd_values[_lp*2+1 + 0]));
                                packed_6[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values[_lp*2 + 2], restore_qd_values[_lp*2+1 + 2]));
                                packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values[_lp*2 + 4], restore_qd_values[_lp*2+1 + 4]));
                                packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values[_lp*2 + 6], restore_qd_values[_lp*2+1 + 6]));
                                packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(tape_qd + tape_scaled_index)[0] = reinterpret_cast<int4*>(packed_6)[0];
                            unsigned int packed_1_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values[_lp*2 + 0], restore_kd_values[_lp*2+1 + 0]));
                                packed_1_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values[_lp*2 + 2], restore_kd_values[_lp*2+1 + 2]));
                                packed_1_2[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values[_lp*2 + 4], restore_kd_values[_lp*2+1 + 4]));
                                packed_1_2[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values[_lp*2 + 6], restore_kd_values[_lp*2+1 + 6]));
                                packed_1_2[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(tape_kd + tape_scaled_index)[0] = reinterpret_cast<int4*>(packed_1_2)[0];
                        }
                        const float2 _scale2_25 = {restore_scale, restore_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(restore_qd_values)[_ls], _scale2_25);
                        {
                            unsigned int packed_7[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values[_lp*2 + 0], restore_qd_values[_lp*2+1 + 0]));
                                packed_7[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_9 = 0; word_9 < 4; word_9++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_9 * 4)), "r"(packed_7[word_9]));
                            }
                        }
                        #pragma unroll
                        for (int restore_elem_1 = 0; restore_elem_1 < 8; restore_elem_1++) {
                            restore_kr_values[restore_elem_1] = restore_ki_values[restore_elem_1] * restore_factor[restore_elem_1];
                        }
                        unsigned int packed_8[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values[_lp*2 + 0], restore_kr_values[_lp*2+1 + 0]));
                            packed_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word_10 = 0; word_10 < 4; word_10++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_10 * 4)), "r"(packed_8[word_10]));
                        }
                    }
                } else if (prep_local_warp == 1) {
                    int stage_f32_0_1 = (unsigned int)chunk_idx_2 % 5 * 10496;
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
                        float restore_kr_values_1[8];
                        float restore_qd_values_1[8];
                        float restore_kd_values_1[8];
                        float restore_ki_values_1[8];
                        unsigned int packed_9[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_9[(0) + 3]))
                            : "r"((smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_3[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_9[_pair]) << 16);
                            (&unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_9[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx_8 = 0; value_idx_8 < 8; value_idx_8++) {
                            restore_qd_values_1[value_idx_8] = unpack_bf16_3[value_idx_8];
                        }
                        unsigned int packed_9_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_9_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_9_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_9_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_9_1[(0) + 3]))
                            : "r"((smem_ki_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_3_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_3_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_9_1[_pair]) << 16);
                            (&unpack_bf16_3_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_9_1[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int value_idx_9 = 0; value_idx_9 < 8; value_idx_9++) {
                            restore_ki_values_1[value_idx_9] = unpack_bf16_3_1[value_idx_9];
                        }
                        if (STORE_BACKWARD_TAPE != 0 && owned_chunk_1 != 0) {
                            unsigned int packed_10[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_10[(0) + 3]))
                                : "r"((smem_kd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float unpack_bf16_4[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_10[_pair]) << 16);
                                (&unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_10[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int value_idx_10 = 0; value_idx_10 < 8; value_idx_10++) {
                                restore_kd_values_1[value_idx_10] = unpack_bf16_4[value_idx_10];
                            }
                            long long tape_scaled_index_1 = tape_scaled_base + (long long)restore_row_1 * 128 + (long long)(restore_segment_1 * 8);
                            unsigned int packed_11[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values_1[_lp*2 + 0], restore_qd_values_1[_lp*2+1 + 0]));
                                packed_11[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values_1[_lp*2 + 2], restore_qd_values_1[_lp*2+1 + 2]));
                                packed_11[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values_1[_lp*2 + 4], restore_qd_values_1[_lp*2+1 + 4]));
                                packed_11[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values_1[_lp*2 + 6], restore_qd_values_1[_lp*2+1 + 6]));
                                packed_11[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(tape_qd + tape_scaled_index_1)[0] = reinterpret_cast<int4*>(packed_11)[0];
                            unsigned int packed_1_3[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values_1[_lp*2 + 0], restore_kd_values_1[_lp*2+1 + 0]));
                                packed_1_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values_1[_lp*2 + 2], restore_kd_values_1[_lp*2+1 + 2]));
                                packed_1_3[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values_1[_lp*2 + 4], restore_kd_values_1[_lp*2+1 + 4]));
                                packed_1_3[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values_1[_lp*2 + 6], restore_kd_values_1[_lp*2+1 + 6]));
                                packed_1_3[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(tape_kd + tape_scaled_index_1)[0] = reinterpret_cast<int4*>(packed_1_3)[0];
                        }
                        const float2 _scale2_26 = {restore_scale_1, restore_scale_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(restore_qd_values_1)[_ls], _scale2_26);
                        {
                            unsigned int packed_12[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values_1[_lp*2 + 0], restore_qd_values_1[_lp*2+1 + 0]));
                                packed_12[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_11 = 0; word_11 < 4; word_11++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_11 * 4)), "r"(packed_12[word_11]));
                            }
                        }
                        #pragma unroll
                        for (int restore_elem_3 = 0; restore_elem_3 < 8; restore_elem_3++) {
                            restore_kr_values_1[restore_elem_3] = restore_ki_values_1[restore_elem_3] * restore_factor_1[restore_elem_3];
                        }
                        unsigned int packed_13[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values_1[_lp*2 + 0], restore_kr_values_1[_lp*2+1 + 0]));
                            packed_13[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word_12 = 0; word_12 < 4; word_12++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_12 * 4)), "r"(packed_13[word_12]));
                        }
                    }
                }
                if (prep_local_warp == 0) {
                    int lane_row = lane % 16;
                    int lane_col = lane / 16 * 8;
                    int byte_off_2 = (int)((unsigned int)chunk_idx_2 % 5) * 41984 + (16 + lane_row) * 128 + (16 + lane_col) * 2;
                    int swizzled_off_2 = byte_off_2 ^ (byte_off_2 >> 7 & 7) << 4;
                    int d_addr = smem_inv_work_addr + (unsigned int)swizzled_off_2;
                    int byte_off_0 = (int)((unsigned int)chunk_idx_2 % 5) * 41984 + (16 + lane_row) * 128 + lane_col * 2;
                    int swizzled_off_0 = byte_off_0 ^ (byte_off_0 >> 7 & 7) << 4;
                    int c_addr = smem_inv_work_addr + (unsigned int)swizzled_off_0;
                    int byte_off_1_1 = (int)((unsigned int)chunk_idx_2 % 5) * 41984 + lane_row * 128 + lane_col * 2;
                    int swizzled_off_1_1 = byte_off_1_1 ^ (byte_off_1_1 >> 7 & 7) << 4;
                    int a_addr = smem_inv_work_addr + (unsigned int)swizzled_off_1_1;
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
                        : "r"(d_addr)
                        : "memory");
                    int d_publish_addr = (smem_inv_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((16 + lane_col) / 16 * 1024 + (16 + lane_row) * 32 + (16 + lane_col) % 16 * 2 ^ ((16 + lane_col) / 16 * 1024 + (16 + lane_row) * 32 + (16 + lane_col) % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_27 = static_cast<uint32_t>((unsigned long long)d_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_27), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[3]))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(c32_frag[0]), "=r"(c32_frag[1]), "=r"(c32_frag[2]), "=r"(c32_frag[3])
                        : "r"(c_addr)
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(dc32_acc[0]), "=f"(dc32_acc[1]), "=f"(dc32_acc[2]), "=f"(dc32_acc[3])
                        : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]), "r"(c32_frag[0]), "r"(c32_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(dc32_acc[4]), "=f"(dc32_acc[(4) + 1]), "=f"(dc32_acc[(4) + 2]), "=f"(dc32_acc[(4) + 3])
                        : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]), "r"(c32_frag[2]), "r"(c32_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    const float2 _scale2_28 = {-1.0f, -1.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(dc32_acc)[_ls], _scale2_28);
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dc32_acc[_lp*2 + 0], dc32_acc[_lp*2+1 + 0]));
                        dc32_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a32_frag[0]), "=r"(a32_frag[1]), "=r"(a32_frag[2]), "=r"(a32_frag[3])
                        : "r"(a_addr)
                        : "memory");
                    int a_publish_addr = (smem_inv_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(lane_col / 16 * 1024 + lane_row * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 1024 + lane_row * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_29 = static_cast<uint32_t>((unsigned long long)a_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[3]))
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
                    int o_publish_addr = (smem_inv_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)(lane_col / 16 * 1024 + (16 + lane_row) * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 1024 + (16 + lane_row) * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_30 = static_cast<uint32_t>((unsigned long long)o_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_30), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[3]))
                        : "memory");
                    #pragma unroll
                    for (int zero_word = 0; zero_word < 4; zero_word++) {
                        zero32_bf16[zero_word] = 0;
                    }
                    int zero_publish_addr = (smem_inv_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((16 + lane_col) / 16 * 1024 + lane_row * 32 + (16 + lane_col) % 16 * 2 ^ ((16 + lane_col) / 16 * 1024 + lane_row * 32 + (16 + lane_col) % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_31 = static_cast<uint32_t>((unsigned long long)zero_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_31), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[3]))
                        : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                mbarrier_arrive(qk_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8);
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(v_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8, 8192);
                            tma_3d_gmem2smem(smem_v_addr + (unsigned int)chunk_idx_2 % 5 * 41984, (&v_tma), 0, head_idx_3, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), v_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8);
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
                        int v_dst = smem_v_addr + (unsigned int)chunk_idx_2 % 5 * 41984 + (unsigned int)((row_1 * 128 + segment_1 * 8) * 2);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(v_dst), "l"(v + v_src), "r"((token_valid_1 != 0) ? 16 : 0));
                    }
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(v_full_addr + ((unsigned int)chunk_idx_2 % 5) * 8);
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
