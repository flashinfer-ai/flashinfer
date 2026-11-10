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
// Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
// DeepGEMM portions are licensed under MIT; see DEEPGEMM_NOTICE.txt in this directory.

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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_Q_PIPE_STAGES 1
#define NUM_KV_PIPE_STAGES 3
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 24576
#define SMEM_SMEM_Q_STRIDE 24576
#define SMEM_SMEM_W_OFF 25600
#define SMEM_SMEM_W_STAGE_BYTES 768
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_SMEM_KV_G0_OFF 26624
#define SMEM_SMEM_KV_G0_STAGE_BYTES 8192
#define SMEM_SMEM_KV_G0_STRIDE 8192
#define SMEM_SMEM_KV_G1_OFF 51200
#define SMEM_SMEM_KV_G1_STAGE_BYTES 8192
#define SMEM_SMEM_KV_G1_STRIDE 8192
#define SMEM_SMEM_SC_G0_OFF 75776
#define SMEM_SMEM_SC_G0_STAGE_BYTES 256
#define SMEM_SMEM_SC_G0_STRIDE 1024
#define SMEM_SMEM_SC_G1_OFF 78848
#define SMEM_SMEM_SC_G1_STAGE_BYTES 256
#define SMEM_SMEM_SC_G1_STRIDE 1024
#define SMEM_TOTAL 81920
#define THREADS 384
#define LAUNCH_MIN_BLOCKS 1

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






__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_sm120_paged_mqa_logits_4cc651b721a27f1983f7(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights, float* __restrict__ Logits, int* __restrict__ context_lens, int* __restrict__ block_table, int* __restrict__ schedule_meta, int logits_stride, int block_table_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define kv_full_g0_addr (mbar_base + 16)
    #define kv_full_g1_addr (mbar_base + 40)
    #define kv_empty_g0_addr (mbar_base + 64)
    #define kv_empty_g1_addr (mbar_base + 88)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    float* smem_w = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_w_addr = smem + 25600;
    uint8_t* smem_kv_g0 = reinterpret_cast<uint8_t*>(smem_raw + 26624);
    const int smem_kv_g0_addr = smem + 26624;
    uint8_t* smem_kv_g1 = reinterpret_cast<uint8_t*>(smem_raw + 51200);
    const int smem_kv_g1_addr = smem + 51200;
    float* smem_sc_g0 = reinterpret_cast<float*>(smem_raw + 75776);
    const int smem_sc_g0_addr = smem + 75776;
    float* smem_sc_g1 = reinterpret_cast<float*>(smem_raw + 78848);
    const int smem_sc_g1_addr = smem + 78848;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 14 barriers)
    // Mbarriers at smem_raw[0..112)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 8, 256);
            // --- pipeline 'kv_pipe' ---
            // kv_full_g0: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // kv_full_g1: 3 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // kv_empty_g0: 3 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            mbarrier_init(smem + 72, 128);
            mbarrier_init(smem + 80, 128);
            // kv_empty_g1: 3 barriers, init_count=128
            mbarrier_init(smem + 88, 128);
            mbarrier_init(smem + 96, 128);
            mbarrier_init(smem + 104, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: math_g0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // math_g0_main
            int w_in_g = warp;
            int lane_0 = lane;
            int quad = lane_0 / 4;
            int tq = lane_0 % 4;
            int a_row = w_in_g * 16 + (lane_0 & 7) + (lane_0 >> 3 & 1) * 8;
            int a_col = (lane_0 >> 4) * 16;
            int b_row_lane = lane_0 & 7;
            int b_col = (lane_0 >> 3 & 1) * 16;
            unsigned int kv_stage_m = 0;
            unsigned int a_frag[4];
            unsigned int b_frag[2];
            unsigned int a_tile[16];
            float acc[4];
            unsigned int fp8_unit_word_g0[1];
            fp8_unit_word_g0[0] = 2139062143;
            uint8_t fp8_unit_sfa_g0[1];
            uint8_t fp8_unit_sfb_g0[1];
            fp8_unit_sfa_g0[0] = fp8_unit_word_g0[0];
            fp8_unit_sfb_g0[0] = fp8_unit_word_g0[0];
            float w_atom[48];
            int sm = bid;
            int start_q = schedule_meta[sm * 2];
            int start_kv = schedule_meta[sm * 2 + 1] * 2;
            int end_q = schedule_meta[sm * 2 + 2];
            int end_kv = schedule_meta[sm * 2 + 3] * 2;
            int q_stop = ((end_kv > 0) ? end_q + 1 : end_q);
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full_g0 = 0;
            #pragma unroll 1
            for (int q_atom = start_q; q_atom < q_stop; q_atom++) {
                int q_idx = q_atom;
                int atom_in_q = 0;
                int tok = q_idx * 3 + atom_in_q * 3;
                int ctx_last = context_lens[q_idx * 3 + 3 - 1];
                int num_kv = (ctx_last + 64 - 1) / 64;
                int lo = ((q_atom == start_q) ? start_kv : 0);
                int hi = ((q_atom == end_q) ? end_kv : num_kv);
                mbarrier_wait(q_full_addr, _phase_q_full);
                int n_tok_m = 3;
                int ws_a_23_0 = 0 * 256 + (unsigned int)(tq * 2);
                w_atom[0] = smem_w[ws_a_23_0];
                w_atom[1] = smem_w[ws_a_23_0 + 1];
                int ws_a_23_1 = 0 * 256 + 8 + (unsigned int)(tq * 2);
                w_atom[2] = smem_w[ws_a_23_1];
                w_atom[3] = smem_w[ws_a_23_1 + 1];
                int ws_a_23_2 = 0 * 256 + 16 + (unsigned int)(tq * 2);
                w_atom[4] = smem_w[ws_a_23_2];
                w_atom[5] = smem_w[ws_a_23_2 + 1];
                int ws_a_23_3 = 0 * 256 + 24 + (unsigned int)(tq * 2);
                w_atom[6] = smem_w[ws_a_23_3];
                w_atom[7] = smem_w[ws_a_23_3 + 1];
                int ws_a_23_4 = 0 * 256 + 32 + (unsigned int)(tq * 2);
                w_atom[8] = smem_w[ws_a_23_4];
                w_atom[9] = smem_w[ws_a_23_4 + 1];
                int ws_a_23_5 = 0 * 256 + 40 + (unsigned int)(tq * 2);
                w_atom[10] = smem_w[ws_a_23_5];
                w_atom[11] = smem_w[ws_a_23_5 + 1];
                int ws_a_23_6 = 0 * 256 + 48 + (unsigned int)(tq * 2);
                w_atom[12] = smem_w[ws_a_23_6];
                w_atom[13] = smem_w[ws_a_23_6 + 1];
                int ws_a_23_7 = 0 * 256 + 56 + (unsigned int)(tq * 2);
                w_atom[14] = smem_w[ws_a_23_7];
                w_atom[15] = smem_w[ws_a_23_7 + 1];
                int ws_a_23_8 = 0 * 256 + 64 + (unsigned int)(tq * 2);
                w_atom[16] = smem_w[ws_a_23_8];
                w_atom[17] = smem_w[ws_a_23_8 + 1];
                int ws_a_23_9 = 0 * 256 + 64 + 8 + (unsigned int)(tq * 2);
                w_atom[18] = smem_w[ws_a_23_9];
                w_atom[19] = smem_w[ws_a_23_9 + 1];
                int ws_a_23_10 = 0 * 256 + 64 + 16 + (unsigned int)(tq * 2);
                w_atom[20] = smem_w[ws_a_23_10];
                w_atom[21] = smem_w[ws_a_23_10 + 1];
                int ws_a_23_11 = 0 * 256 + 64 + 24 + (unsigned int)(tq * 2);
                w_atom[22] = smem_w[ws_a_23_11];
                w_atom[23] = smem_w[ws_a_23_11 + 1];
                int ws_a_23_12 = 0 * 256 + 64 + 32 + (unsigned int)(tq * 2);
                w_atom[24] = smem_w[ws_a_23_12];
                w_atom[25] = smem_w[ws_a_23_12 + 1];
                int ws_a_23_13 = 0 * 256 + 64 + 40 + (unsigned int)(tq * 2);
                w_atom[26] = smem_w[ws_a_23_13];
                w_atom[27] = smem_w[ws_a_23_13 + 1];
                int ws_a_23_14 = 0 * 256 + 64 + 48 + (unsigned int)(tq * 2);
                w_atom[28] = smem_w[ws_a_23_14];
                w_atom[29] = smem_w[ws_a_23_14 + 1];
                int ws_a_23_15 = 0 * 256 + 64 + 56 + (unsigned int)(tq * 2);
                w_atom[30] = smem_w[ws_a_23_15];
                w_atom[31] = smem_w[ws_a_23_15 + 1];
                int ws_a_23_16 = 0 * 256 + 128 + (unsigned int)(tq * 2);
                w_atom[32] = smem_w[ws_a_23_16];
                w_atom[33] = smem_w[ws_a_23_16 + 1];
                int ws_a_23_17 = 0 * 256 + 128 + 8 + (unsigned int)(tq * 2);
                w_atom[34] = smem_w[ws_a_23_17];
                w_atom[35] = smem_w[ws_a_23_17 + 1];
                int ws_a_23_18 = 0 * 256 + 128 + 16 + (unsigned int)(tq * 2);
                w_atom[36] = smem_w[ws_a_23_18];
                w_atom[37] = smem_w[ws_a_23_18 + 1];
                int ws_a_23_19 = 0 * 256 + 128 + 24 + (unsigned int)(tq * 2);
                w_atom[38] = smem_w[ws_a_23_19];
                w_atom[39] = smem_w[ws_a_23_19 + 1];
                int ws_a_23_20 = 0 * 256 + 128 + 32 + (unsigned int)(tq * 2);
                w_atom[40] = smem_w[ws_a_23_20];
                w_atom[41] = smem_w[ws_a_23_20 + 1];
                int ws_a_23_21 = 0 * 256 + 128 + 40 + (unsigned int)(tq * 2);
                w_atom[42] = smem_w[ws_a_23_21];
                w_atom[43] = smem_w[ws_a_23_21 + 1];
                int ws_a_23_22 = 0 * 256 + 128 + 48 + (unsigned int)(tq * 2);
                w_atom[44] = smem_w[ws_a_23_22];
                w_atom[45] = smem_w[ws_a_23_22 + 1];
                int ws_a_23_23 = 0 * 256 + 128 + 56 + (unsigned int)(tq * 2);
                w_atom[46] = smem_w[ws_a_23_23];
                w_atom[47] = smem_w[ws_a_23_23 + 1];
                #pragma unroll 1
                for (int kv_idx = lo; kv_idx < hi; kv_idx += 2) {
                    mbarrier_wait(kv_full_g0_addr + (kv_stage_m) * 8, _phase_kv_full_g0);
                    int sc_idx = kv_stage_m * 256 + (unsigned int)(w_in_g * 16) + (unsigned int)quad;
                    float scale0 = smem_sc_g0[sc_idx];
                    float scale1 = smem_sc_g0[sc_idx + 8];
                    int kv_offset = tok * logits_stride + kv_idx * 64 + w_in_g * 16;
                    int a_cb = a_col;
                    unsigned int a_addr = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb ^ (a_row & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile[0]), "=r"(a_tile[1]), "=r"(a_tile[2]), "=r"(a_tile[3])
                        : "r"(a_addr)
                        : "memory");
                    int a_cb_0 = a_col + 32;
                    unsigned int a_addr_0 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_0 ^ (a_row & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile[4]), "=r"(a_tile[5]), "=r"(a_tile[6]), "=r"(a_tile[7])
                        : "r"(a_addr_0)
                        : "memory");
                    int a_cb_1 = a_col + 64;
                    unsigned int a_addr_1 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_1 ^ (a_row & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile[8]), "=r"(a_tile[9]), "=r"(a_tile[10]), "=r"(a_tile[11])
                        : "r"(a_addr_1)
                        : "memory");
                    int a_cb_2 = a_col + 96;
                    unsigned int a_addr_2 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_2 ^ (a_row & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile[12]), "=r"(a_tile[13]), "=r"(a_tile[14]), "=r"(a_tile[15])
                        : "r"(a_addr_2)
                        : "memory");
                    float partial0 = 0.0f;
                    float partial1 = 0.0f;
                    #pragma unroll
                    for (int nt = 0; nt < 8; nt++) {
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        int b_row = nt * 8 + b_row_lane;
                        int b_cb = b_col;
                        unsigned int b_addr = smem_q_addr + 0 * 24576 + (unsigned int)(b_row * 128) + (unsigned int)(b_cb ^ (b_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(b_addr)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_tile[0]), "r"(a_tile[1]), "r"(a_tile[2]), "r"(a_tile[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_0 = nt * 8 + b_row_lane;
                        int b_cb_0 = b_col + 32;
                        unsigned int b_addr_0 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_0 * 128) + (unsigned int)(b_cb_0 ^ (b_row_0 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(b_addr_0)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[4]), "r"(a_tile[(4) + 1]), "r"(a_tile[(4) + 2]), "r"(a_tile[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(fp8_unit_sfa_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_1 = nt * 8 + b_row_lane;
                        int b_cb_1 = b_col + 64;
                        unsigned int b_addr_1 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_1 * 128) + (unsigned int)(b_cb_1 ^ (b_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(b_addr_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[8]), "r"(a_tile[(8) + 1]), "r"(a_tile[(8) + 2]), "r"(a_tile[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(fp8_unit_sfa_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_2 = nt * 8 + b_row_lane;
                        int b_cb_2 = b_col + 96;
                        unsigned int b_addr_2 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_2 * 128) + (unsigned int)(b_cb_2 ^ (b_row_2 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(b_addr_2)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[12]), "r"(a_tile[(12) + 1]), "r"(a_tile[(12) + 2]), "r"(a_tile[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(fp8_unit_sfa_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        float w0 = w_atom[nt * 2];
                        float w1 = w_atom[nt * 2 + 1];
                        float r0 = fmaxf(acc[0], 0.0f);
                        float r1 = fmaxf(acc[1], 0.0f);
                        float r2 = fmaxf(acc[2], 0.0f);
                        float r3 = fmaxf(acc[3], 0.0f);
                        partial0 += __fmaf_rn(r0, w0, r1 * w1);
                        partial1 += __fmaf_rn(r2, w0, r3 * w1);
                    }
                    float v0 = partial0 * scale0;
                    float v1 = partial1 * scale1;
                    float _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, v0, 1);
                    v0 = v0 + _shfl_xor_sync_0;
                    float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, v1, 1);
                    v1 = v1 + _shfl_xor_sync_1;
                    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, v0, 2);
                    v0 = v0 + _shfl_xor_sync_2;
                    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, v1, 2);
                    v1 = v1 + _shfl_xor_sync_3;
                    int out0 = kv_offset + quad;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0)), "f"(v0), "r"((unsigned int)(tq == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0 + 8))), "f"(v1), "r"((unsigned int)(tq == 0)));
                    float partial0_0 = 0.0f;
                    float partial1_0 = 0.0f;
                    #pragma unroll
                    for (int nt_1 = 0; nt_1 < 8; nt_1++) {
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        int b_row_3 = 64 + nt_1 * 8 + b_row_lane;
                        int b_cb_3 = b_col;
                        unsigned int b_addr_3 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_3 * 128) + (unsigned int)(b_cb_3 ^ (b_row_3 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(b_addr_3)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_tile[0]), "r"(a_tile[1]), "r"(a_tile[2]), "r"(a_tile[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_3_1 = 64 + nt_1 * 8 + b_row_lane;
                        int b_cb_3_1 = b_col + 32;
                        unsigned int b_addr_3_1 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_3_1 * 128) + (unsigned int)(b_cb_3_1 ^ (b_row_3_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(b_addr_3_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[4]), "r"(a_tile[(4) + 1]), "r"(a_tile[(4) + 2]), "r"(a_tile[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(fp8_unit_sfa_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_4 = 64 + nt_1 * 8 + b_row_lane;
                        int b_cb_4 = b_col + 64;
                        unsigned int b_addr_4 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_4 * 128) + (unsigned int)(b_cb_4 ^ (b_row_4 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(b_addr_4)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[8]), "r"(a_tile[(8) + 1]), "r"(a_tile[(8) + 2]), "r"(a_tile[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(fp8_unit_sfa_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_5 = 64 + nt_1 * 8 + b_row_lane;
                        int b_cb_5 = b_col + 96;
                        unsigned int b_addr_5 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_5 * 128) + (unsigned int)(b_cb_5 ^ (b_row_5 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(b_addr_5)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[12]), "r"(a_tile[(12) + 1]), "r"(a_tile[(12) + 2]), "r"(a_tile[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(fp8_unit_sfa_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        float w0_1 = w_atom[(8 + nt_1) * 2];
                        float w1_1 = w_atom[(8 + nt_1) * 2 + 1];
                        float r0_1 = fmaxf(acc[0], 0.0f);
                        float r1_1 = fmaxf(acc[1], 0.0f);
                        float r2_1 = fmaxf(acc[2], 0.0f);
                        float r3_1 = fmaxf(acc[3], 0.0f);
                        partial0_0 += __fmaf_rn(r0_1, w0_1, r1_1 * w1_1);
                        partial1_0 += __fmaf_rn(r2_1, w0_1, r3_1 * w1_1);
                    }
                    float v0_0 = partial0_0 * scale0;
                    float v1_0 = partial1_0 * scale1;
                    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, v0_0, 1);
                    v0_0 = v0_0 + _shfl_xor_sync_4;
                    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, v1_0, 1);
                    v1_0 = v1_0 + _shfl_xor_sync_5;
                    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, v0_0, 2);
                    v0_0 = v0_0 + _shfl_xor_sync_6;
                    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, v1_0, 2);
                    v1_0 = v1_0 + _shfl_xor_sync_7;
                    int out0_0 = kv_offset + logits_stride + quad;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_0)), "f"(v0_0), "r"((unsigned int)(tq == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_0 + 8))), "f"(v1_0), "r"((unsigned int)(tq == 0)));
                    float partial0_1 = 0.0f;
                    float partial1_1 = 0.0f;
                    #pragma unroll
                    for (int nt_2 = 0; nt_2 < 8; nt_2++) {
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        int b_row_6 = 128 + nt_2 * 8 + b_row_lane;
                        int b_cb_6 = b_col;
                        unsigned int b_addr_6 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_6 * 128) + (unsigned int)(b_cb_6 ^ (b_row_6 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(b_addr_6)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_tile[0]), "r"(a_tile[1]), "r"(a_tile[2]), "r"(a_tile[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_6_1 = 128 + nt_2 * 8 + b_row_lane;
                        int b_cb_6_1 = b_col + 32;
                        unsigned int b_addr_6_1 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_6_1 * 128) + (unsigned int)(b_cb_6_1 ^ (b_row_6_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(b_addr_6_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[4]), "r"(a_tile[(4) + 1]), "r"(a_tile[(4) + 2]), "r"(a_tile[(4) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(fp8_unit_sfa_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_7 = 128 + nt_2 * 8 + b_row_lane;
                        int b_cb_7 = b_col + 64;
                        unsigned int b_addr_7 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_7 * 128) + (unsigned int)(b_cb_7 ^ (b_row_7 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(b_addr_7)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[8]), "r"(a_tile[(8) + 1]), "r"(a_tile[(8) + 2]), "r"(a_tile[(8) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(fp8_unit_sfa_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_8 = 128 + nt_2 * 8 + b_row_lane;
                        int b_cb_8 = b_col + 96;
                        unsigned int b_addr_8 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_8 * 128) + (unsigned int)(b_cb_8 ^ (b_row_8 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(b_addr_8)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_tile[12]), "r"(a_tile[(12) + 1]), "r"(a_tile[(12) + 2]), "r"(a_tile[(12) + 3]), "r"(b_frag[0]), "r"(b_frag[1]), "r"((uint32_t)(fp8_unit_sfa_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        float w0_2 = w_atom[(16 + nt_2) * 2];
                        float w1_2 = w_atom[(16 + nt_2) * 2 + 1];
                        float r0_2 = fmaxf(acc[0], 0.0f);
                        float r1_2 = fmaxf(acc[1], 0.0f);
                        float r2_2 = fmaxf(acc[2], 0.0f);
                        float r3_2 = fmaxf(acc[3], 0.0f);
                        partial0_1 += __fmaf_rn(r0_2, w0_2, r1_2 * w1_2);
                        partial1_1 += __fmaf_rn(r2_2, w0_2, r3_2 * w1_2);
                    }
                    float v0_1 = partial0_1 * scale0;
                    float v1_1 = partial1_1 * scale1;
                    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, v0_1, 1);
                    v0_1 = v0_1 + _shfl_xor_sync_8;
                    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 1);
                    v1_1 = v1_1 + _shfl_xor_sync_9;
                    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, v0_1, 2);
                    v0_1 = v0_1 + _shfl_xor_sync_10;
                    float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 2);
                    v1_1 = v1_1 + _shfl_xor_sync_11;
                    int out0_1 = kv_offset + 2 * logits_stride + quad;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_1)), "f"(v0_1), "r"((unsigned int)(tq == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_1 + 8))), "f"(v1_1), "r"((unsigned int)(tq == 0)));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_empty_g0_addr + (kv_stage_m) * 8);
                    kv_stage_m += 1;
                    if (kv_stage_m == 3) { kv_stage_m = 0; _phase_kv_full_g0 ^= 1; }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(q_empty_addr);
                _phase_q_full ^= 1;
            }
        }
    }
    // ---- Role: math_g1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // math_g1_main
            int w_in_g_1 = warp - 4;
            int lane_0_1 = lane;
            int quad_1 = lane_0_1 / 4;
            int tq_1 = lane_0_1 % 4;
            int a_row_1 = w_in_g_1 * 16 + (lane_0_1 & 7) + (lane_0_1 >> 3 & 1) * 8;
            int a_col_1 = (lane_0_1 >> 4) * 16;
            int b_row_lane_1 = lane_0_1 & 7;
            int b_col_1 = (lane_0_1 >> 3 & 1) * 16;
            unsigned int kv_stage_m_1 = 0;
            unsigned int a_frag_1[4];
            unsigned int b_frag_1[2];
            unsigned int a_tile_1[16];
            float acc_1[4];
            unsigned int fp8_unit_word_g0_1[1];
            fp8_unit_word_g0_1[0] = 2139062143;
            uint8_t fp8_unit_sfa_g0_1[1];
            uint8_t fp8_unit_sfb_g0_1[1];
            fp8_unit_sfa_g0_1[0] = fp8_unit_word_g0_1[0];
            fp8_unit_sfb_g0_1[0] = fp8_unit_word_g0_1[0];
            float w_atom_1[48];
            int sm_1 = bid;
            int start_q_1 = schedule_meta[sm_1 * 2];
            int start_kv_1 = schedule_meta[sm_1 * 2 + 1] * 2;
            int end_q_1 = schedule_meta[sm_1 * 2 + 2];
            int end_kv_1 = schedule_meta[sm_1 * 2 + 3] * 2;
            int q_stop_1 = ((end_kv_1 > 0) ? end_q_1 + 1 : end_q_1);
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full_g1 = 0;
            #pragma unroll 1
            for (int q_atom_1 = start_q_1; q_atom_1 < q_stop_1; q_atom_1++) {
                int q_idx_1 = q_atom_1;
                int atom_in_q_1 = 0;
                int tok_1 = q_idx_1 * 3 + atom_in_q_1 * 3;
                int ctx_last_1 = context_lens[q_idx_1 * 3 + 3 - 1];
                int num_kv_1 = (ctx_last_1 + 64 - 1) / 64;
                int lo_1 = ((q_atom_1 == start_q_1) ? start_kv_1 : 0);
                int hi_1 = ((q_atom_1 == end_q_1) ? end_kv_1 : num_kv_1);
                mbarrier_wait(q_full_addr, _phase_q_full_1);
                int n_tok_m_1 = 3;
                int ws_a_23_24 = 0 * 256 + (unsigned int)(tq_1 * 2);
                w_atom_1[0] = smem_w[ws_a_23_24];
                w_atom_1[1] = smem_w[ws_a_23_24 + 1];
                int ws_a_23_25 = 0 * 256 + 8 + (unsigned int)(tq_1 * 2);
                w_atom_1[2] = smem_w[ws_a_23_25];
                w_atom_1[3] = smem_w[ws_a_23_25 + 1];
                int ws_a_23_26 = 0 * 256 + 16 + (unsigned int)(tq_1 * 2);
                w_atom_1[4] = smem_w[ws_a_23_26];
                w_atom_1[5] = smem_w[ws_a_23_26 + 1];
                int ws_a_23_27 = 0 * 256 + 24 + (unsigned int)(tq_1 * 2);
                w_atom_1[6] = smem_w[ws_a_23_27];
                w_atom_1[7] = smem_w[ws_a_23_27 + 1];
                int ws_a_23_28 = 0 * 256 + 32 + (unsigned int)(tq_1 * 2);
                w_atom_1[8] = smem_w[ws_a_23_28];
                w_atom_1[9] = smem_w[ws_a_23_28 + 1];
                int ws_a_23_29 = 0 * 256 + 40 + (unsigned int)(tq_1 * 2);
                w_atom_1[10] = smem_w[ws_a_23_29];
                w_atom_1[11] = smem_w[ws_a_23_29 + 1];
                int ws_a_23_30 = 0 * 256 + 48 + (unsigned int)(tq_1 * 2);
                w_atom_1[12] = smem_w[ws_a_23_30];
                w_atom_1[13] = smem_w[ws_a_23_30 + 1];
                int ws_a_23_31 = 0 * 256 + 56 + (unsigned int)(tq_1 * 2);
                w_atom_1[14] = smem_w[ws_a_23_31];
                w_atom_1[15] = smem_w[ws_a_23_31 + 1];
                int ws_a_23_32 = 0 * 256 + 64 + (unsigned int)(tq_1 * 2);
                w_atom_1[16] = smem_w[ws_a_23_32];
                w_atom_1[17] = smem_w[ws_a_23_32 + 1];
                int ws_a_23_33 = 0 * 256 + 64 + 8 + (unsigned int)(tq_1 * 2);
                w_atom_1[18] = smem_w[ws_a_23_33];
                w_atom_1[19] = smem_w[ws_a_23_33 + 1];
                int ws_a_23_34 = 0 * 256 + 64 + 16 + (unsigned int)(tq_1 * 2);
                w_atom_1[20] = smem_w[ws_a_23_34];
                w_atom_1[21] = smem_w[ws_a_23_34 + 1];
                int ws_a_23_35 = 0 * 256 + 64 + 24 + (unsigned int)(tq_1 * 2);
                w_atom_1[22] = smem_w[ws_a_23_35];
                w_atom_1[23] = smem_w[ws_a_23_35 + 1];
                int ws_a_23_36 = 0 * 256 + 64 + 32 + (unsigned int)(tq_1 * 2);
                w_atom_1[24] = smem_w[ws_a_23_36];
                w_atom_1[25] = smem_w[ws_a_23_36 + 1];
                int ws_a_23_37 = 0 * 256 + 64 + 40 + (unsigned int)(tq_1 * 2);
                w_atom_1[26] = smem_w[ws_a_23_37];
                w_atom_1[27] = smem_w[ws_a_23_37 + 1];
                int ws_a_23_38 = 0 * 256 + 64 + 48 + (unsigned int)(tq_1 * 2);
                w_atom_1[28] = smem_w[ws_a_23_38];
                w_atom_1[29] = smem_w[ws_a_23_38 + 1];
                int ws_a_23_39 = 0 * 256 + 64 + 56 + (unsigned int)(tq_1 * 2);
                w_atom_1[30] = smem_w[ws_a_23_39];
                w_atom_1[31] = smem_w[ws_a_23_39 + 1];
                int ws_a_23_40 = 0 * 256 + 128 + (unsigned int)(tq_1 * 2);
                w_atom_1[32] = smem_w[ws_a_23_40];
                w_atom_1[33] = smem_w[ws_a_23_40 + 1];
                int ws_a_23_41 = 0 * 256 + 128 + 8 + (unsigned int)(tq_1 * 2);
                w_atom_1[34] = smem_w[ws_a_23_41];
                w_atom_1[35] = smem_w[ws_a_23_41 + 1];
                int ws_a_23_42 = 0 * 256 + 128 + 16 + (unsigned int)(tq_1 * 2);
                w_atom_1[36] = smem_w[ws_a_23_42];
                w_atom_1[37] = smem_w[ws_a_23_42 + 1];
                int ws_a_23_43 = 0 * 256 + 128 + 24 + (unsigned int)(tq_1 * 2);
                w_atom_1[38] = smem_w[ws_a_23_43];
                w_atom_1[39] = smem_w[ws_a_23_43 + 1];
                int ws_a_23_44 = 0 * 256 + 128 + 32 + (unsigned int)(tq_1 * 2);
                w_atom_1[40] = smem_w[ws_a_23_44];
                w_atom_1[41] = smem_w[ws_a_23_44 + 1];
                int ws_a_23_45 = 0 * 256 + 128 + 40 + (unsigned int)(tq_1 * 2);
                w_atom_1[42] = smem_w[ws_a_23_45];
                w_atom_1[43] = smem_w[ws_a_23_45 + 1];
                int ws_a_23_46 = 0 * 256 + 128 + 48 + (unsigned int)(tq_1 * 2);
                w_atom_1[44] = smem_w[ws_a_23_46];
                w_atom_1[45] = smem_w[ws_a_23_46 + 1];
                int ws_a_23_47 = 0 * 256 + 128 + 56 + (unsigned int)(tq_1 * 2);
                w_atom_1[46] = smem_w[ws_a_23_47];
                w_atom_1[47] = smem_w[ws_a_23_47 + 1];
                #pragma unroll 1
                for (int kv_idx_1 = lo_1; kv_idx_1 < hi_1; kv_idx_1 += 2) {
                    mbarrier_wait(kv_full_g1_addr + (kv_stage_m_1) * 8, _phase_kv_full_g1);
                    int sc_idx_1 = kv_stage_m_1 * 256 + (unsigned int)(w_in_g_1 * 16) + (unsigned int)quad_1;
                    float scale0_1 = smem_sc_g1[sc_idx_1];
                    float scale1_1 = smem_sc_g1[sc_idx_1 + 8];
                    int kv_offset_1 = tok_1 * logits_stride + (kv_idx_1 + 1) * 64 + w_in_g_1 * 16;
                    int a_cb_3 = a_col_1;
                    unsigned int a_addr_3 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_3 ^ (a_row_1 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile_1[0]), "=r"(a_tile_1[1]), "=r"(a_tile_1[2]), "=r"(a_tile_1[3])
                        : "r"(a_addr_3)
                        : "memory");
                    int a_cb_3_1 = a_col_1 + 32;
                    unsigned int a_addr_3_1 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_3_1 ^ (a_row_1 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile_1[4]), "=r"(a_tile_1[5]), "=r"(a_tile_1[6]), "=r"(a_tile_1[7])
                        : "r"(a_addr_3_1)
                        : "memory");
                    int a_cb_4 = a_col_1 + 64;
                    unsigned int a_addr_4 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_4 ^ (a_row_1 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile_1[8]), "=r"(a_tile_1[9]), "=r"(a_tile_1[10]), "=r"(a_tile_1[11])
                        : "r"(a_addr_4)
                        : "memory");
                    int a_cb_5 = a_col_1 + 96;
                    unsigned int a_addr_5 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_5 ^ (a_row_1 & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_tile_1[12]), "=r"(a_tile_1[13]), "=r"(a_tile_1[14]), "=r"(a_tile_1[15])
                        : "r"(a_addr_5)
                        : "memory");
                    float partial0_2 = 0.0f;
                    float partial1_2 = 0.0f;
                    #pragma unroll
                    for (int nt_3 = 0; nt_3 < 8; nt_3++) {
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int b_row_9 = nt_3 * 8 + b_row_lane_1;
                        int b_cb_9 = b_col_1;
                        unsigned int b_addr_9 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_9 * 128) + (unsigned int)(b_cb_9 ^ (b_row_9 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_9)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_tile_1[0]), "r"(a_tile_1[1]), "r"(a_tile_1[2]), "r"(a_tile_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_9_1 = nt_3 * 8 + b_row_lane_1;
                        int b_cb_9_1 = b_col_1 + 32;
                        unsigned int b_addr_9_1 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_9_1 * 128) + (unsigned int)(b_cb_9_1 ^ (b_row_9_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_9_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[4]), "r"(a_tile_1[(4) + 1]), "r"(a_tile_1[(4) + 2]), "r"(a_tile_1[(4) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_10 = nt_3 * 8 + b_row_lane_1;
                        int b_cb_10 = b_col_1 + 64;
                        unsigned int b_addr_10 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_10 * 128) + (unsigned int)(b_cb_10 ^ (b_row_10 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_10)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[8]), "r"(a_tile_1[(8) + 1]), "r"(a_tile_1[(8) + 2]), "r"(a_tile_1[(8) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_11 = nt_3 * 8 + b_row_lane_1;
                        int b_cb_11 = b_col_1 + 96;
                        unsigned int b_addr_11 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_11 * 128) + (unsigned int)(b_cb_11 ^ (b_row_11 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_11)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[12]), "r"(a_tile_1[(12) + 1]), "r"(a_tile_1[(12) + 2]), "r"(a_tile_1[(12) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        float w0_3 = w_atom_1[nt_3 * 2];
                        float w1_3 = w_atom_1[nt_3 * 2 + 1];
                        float r0_3 = fmaxf(acc_1[0], 0.0f);
                        float r1_3 = fmaxf(acc_1[1], 0.0f);
                        float r2_3 = fmaxf(acc_1[2], 0.0f);
                        float r3_3 = fmaxf(acc_1[3], 0.0f);
                        partial0_2 += __fmaf_rn(r0_3, w0_3, r1_3 * w1_3);
                        partial1_2 += __fmaf_rn(r2_3, w0_3, r3_3 * w1_3);
                    }
                    float v0_2 = partial0_2 * scale0_1;
                    float v1_2 = partial1_2 * scale1_1;
                    float _shfl_xor_sync_0_1 = __shfl_xor_sync(0xFFFFFFFF, v0_2, 1);
                    v0_2 = v0_2 + _shfl_xor_sync_0_1;
                    float _shfl_xor_sync_1_1 = __shfl_xor_sync(0xFFFFFFFF, v1_2, 1);
                    v1_2 = v1_2 + _shfl_xor_sync_1_1;
                    float _shfl_xor_sync_2_1 = __shfl_xor_sync(0xFFFFFFFF, v0_2, 2);
                    v0_2 = v0_2 + _shfl_xor_sync_2_1;
                    float _shfl_xor_sync_3_1 = __shfl_xor_sync(0xFFFFFFFF, v1_2, 2);
                    v1_2 = v1_2 + _shfl_xor_sync_3_1;
                    int out0_2 = kv_offset_1 + quad_1;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_2)), "f"(v0_2), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_2 + 8))), "f"(v1_2), "r"((unsigned int)(tq_1 == 0)));
                    float partial0_2_1 = 0.0f;
                    float partial1_2_1 = 0.0f;
                    #pragma unroll
                    for (int nt_4 = 0; nt_4 < 8; nt_4++) {
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int b_row_12 = 64 + nt_4 * 8 + b_row_lane_1;
                        int b_cb_12 = b_col_1;
                        unsigned int b_addr_12 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_12 * 128) + (unsigned int)(b_cb_12 ^ (b_row_12 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_12)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_tile_1[0]), "r"(a_tile_1[1]), "r"(a_tile_1[2]), "r"(a_tile_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_12_1 = 64 + nt_4 * 8 + b_row_lane_1;
                        int b_cb_12_1 = b_col_1 + 32;
                        unsigned int b_addr_12_1 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_12_1 * 128) + (unsigned int)(b_cb_12_1 ^ (b_row_12_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_12_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[4]), "r"(a_tile_1[(4) + 1]), "r"(a_tile_1[(4) + 2]), "r"(a_tile_1[(4) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_13 = 64 + nt_4 * 8 + b_row_lane_1;
                        int b_cb_13 = b_col_1 + 64;
                        unsigned int b_addr_13 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_13 * 128) + (unsigned int)(b_cb_13 ^ (b_row_13 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_13)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[8]), "r"(a_tile_1[(8) + 1]), "r"(a_tile_1[(8) + 2]), "r"(a_tile_1[(8) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_14 = 64 + nt_4 * 8 + b_row_lane_1;
                        int b_cb_14 = b_col_1 + 96;
                        unsigned int b_addr_14 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_14 * 128) + (unsigned int)(b_cb_14 ^ (b_row_14 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_14)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[12]), "r"(a_tile_1[(12) + 1]), "r"(a_tile_1[(12) + 2]), "r"(a_tile_1[(12) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        float w0_4 = w_atom_1[(8 + nt_4) * 2];
                        float w1_4 = w_atom_1[(8 + nt_4) * 2 + 1];
                        float r0_4 = fmaxf(acc_1[0], 0.0f);
                        float r1_4 = fmaxf(acc_1[1], 0.0f);
                        float r2_4 = fmaxf(acc_1[2], 0.0f);
                        float r3_4 = fmaxf(acc_1[3], 0.0f);
                        partial0_2_1 += __fmaf_rn(r0_4, w0_4, r1_4 * w1_4);
                        partial1_2_1 += __fmaf_rn(r2_4, w0_4, r3_4 * w1_4);
                    }
                    float v0_2_1 = partial0_2_1 * scale0_1;
                    float v1_2_1 = partial1_2_1 * scale1_1;
                    float _shfl_xor_sync_4_1 = __shfl_xor_sync(0xFFFFFFFF, v0_2_1, 1);
                    v0_2_1 = v0_2_1 + _shfl_xor_sync_4_1;
                    float _shfl_xor_sync_5_1 = __shfl_xor_sync(0xFFFFFFFF, v1_2_1, 1);
                    v1_2_1 = v1_2_1 + _shfl_xor_sync_5_1;
                    float _shfl_xor_sync_6_1 = __shfl_xor_sync(0xFFFFFFFF, v0_2_1, 2);
                    v0_2_1 = v0_2_1 + _shfl_xor_sync_6_1;
                    float _shfl_xor_sync_7_1 = __shfl_xor_sync(0xFFFFFFFF, v1_2_1, 2);
                    v1_2_1 = v1_2_1 + _shfl_xor_sync_7_1;
                    int out0_2_1 = kv_offset_1 + logits_stride + quad_1;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_2_1)), "f"(v0_2_1), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_2_1 + 8))), "f"(v1_2_1), "r"((unsigned int)(tq_1 == 0)));
                    float partial0_3 = 0.0f;
                    float partial1_3 = 0.0f;
                    #pragma unroll
                    for (int nt_5 = 0; nt_5 < 8; nt_5++) {
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int b_row_15 = 128 + nt_5 * 8 + b_row_lane_1;
                        int b_cb_15 = b_col_1;
                        unsigned int b_addr_15 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_15 * 128) + (unsigned int)(b_cb_15 ^ (b_row_15 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_15)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_tile_1[0]), "r"(a_tile_1[1]), "r"(a_tile_1[2]), "r"(a_tile_1[3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f), "r"((uint32_t)(fp8_unit_sfa_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_15_1 = 128 + nt_5 * 8 + b_row_lane_1;
                        int b_cb_15_1 = b_col_1 + 32;
                        unsigned int b_addr_15_1 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_15_1 * 128) + (unsigned int)(b_cb_15_1 ^ (b_row_15_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_15_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[4]), "r"(a_tile_1[(4) + 1]), "r"(a_tile_1[(4) + 2]), "r"(a_tile_1[(4) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_16 = 128 + nt_5 * 8 + b_row_lane_1;
                        int b_cb_16 = b_col_1 + 64;
                        unsigned int b_addr_16 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_16 * 128) + (unsigned int)(b_cb_16 ^ (b_row_16 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_16)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[8]), "r"(a_tile_1[(8) + 1]), "r"(a_tile_1[(8) + 2]), "r"(a_tile_1[(8) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        int b_row_17 = 128 + nt_5 * 8 + b_row_lane_1;
                        int b_cb_17 = b_col_1 + 96;
                        unsigned int b_addr_17 = smem_q_addr + 0 * 24576 + (unsigned int)(b_row_17 * 128) + (unsigned int)(b_cb_17 ^ (b_row_17 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                            : "r"(b_addr_17)
                            : "memory");
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_tile_1[12]), "r"(a_tile_1[(12) + 1]), "r"(a_tile_1[(12) + 2]), "r"(a_tile_1[(12) + 3]), "r"(b_frag_1[0]), "r"(b_frag_1[1]), "r"((uint32_t)(fp8_unit_sfa_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(fp8_unit_sfb_g0_1[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                        float w0_5 = w_atom_1[(16 + nt_5) * 2];
                        float w1_5 = w_atom_1[(16 + nt_5) * 2 + 1];
                        float r0_5 = fmaxf(acc_1[0], 0.0f);
                        float r1_5 = fmaxf(acc_1[1], 0.0f);
                        float r2_5 = fmaxf(acc_1[2], 0.0f);
                        float r3_5 = fmaxf(acc_1[3], 0.0f);
                        partial0_3 += __fmaf_rn(r0_5, w0_5, r1_5 * w1_5);
                        partial1_3 += __fmaf_rn(r2_5, w0_5, r3_5 * w1_5);
                    }
                    float v0_3 = partial0_3 * scale0_1;
                    float v1_3 = partial1_3 * scale1_1;
                    float _shfl_xor_sync_8_1 = __shfl_xor_sync(0xFFFFFFFF, v0_3, 1);
                    v0_3 = v0_3 + _shfl_xor_sync_8_1;
                    float _shfl_xor_sync_9_1 = __shfl_xor_sync(0xFFFFFFFF, v1_3, 1);
                    v1_3 = v1_3 + _shfl_xor_sync_9_1;
                    float _shfl_xor_sync_10_1 = __shfl_xor_sync(0xFFFFFFFF, v0_3, 2);
                    v0_3 = v0_3 + _shfl_xor_sync_10_1;
                    float _shfl_xor_sync_11_1 = __shfl_xor_sync(0xFFFFFFFF, v1_3, 2);
                    v1_3 = v1_3 + _shfl_xor_sync_11_1;
                    int out0_3 = kv_offset_1 + 2 * logits_stride + quad_1;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + out0_3)), "f"(v0_3), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(Logits + (out0_3 + 8))), "f"(v1_3), "r"((unsigned int)(tq_1 == 0)));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_empty_g1_addr + (kv_stage_m_1) * 8);
                    kv_stage_m_1 += 1;
                    if (kv_stage_m_1 == 3) { kv_stage_m_1 = 0; _phase_kv_full_g1 ^= 1; }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(q_empty_addr);
                _phase_q_full_1 ^= 1;
            }
        }
    }
    // ---- Role: tma_g0 ----
    if (warp == 8) {
        { // tma_g0_main
            unsigned int q_phase = 1;
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 1;
            int sm_2 = bid;
            int start_q_2 = schedule_meta[sm_2 * 2];
            int start_kv_2 = schedule_meta[sm_2 * 2 + 1] * 2;
            int end_q_2 = schedule_meta[sm_2 * 2 + 2];
            int end_kv_2 = schedule_meta[sm_2 * 2 + 3] * 2;
            int q_stop_2 = ((end_kv_2 > 0) ? end_q_2 + 1 : end_q_2);
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_kv_empty_g0 = 1;
            #pragma unroll 1
            for (int q_atom_2 = start_q_2; q_atom_2 < q_stop_2; q_atom_2++) {
                int q_idx_2 = q_atom_2;
                int atom_in_q_2 = 0;
                int tok_2 = q_idx_2 * 3 + atom_in_q_2 * 3;
                int ctx_last_2 = context_lens[q_idx_2 * 3 + 3 - 1];
                int kv_lo = ((q_atom_2 == start_q_2) ? start_kv_2 : 0);
                int bt_lane0 = lane;
                int bt_col0 = kv_lo + bt_lane0 * 2;
                int bt_raw = 0;
                if (bt_col0 < block_table_stride) {
                    bt_raw = block_table[q_idx_2 * block_table_stride + bt_col0];
                }
                mbarrier_wait(q_empty_addr, _phase_q_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 25344);
                    tma_2d_gmem2smem(smem_q_addr + 0 * 24576, (&Q), 0, tok_2 * 64, q_full_addr);
                    tma_2d_gmem2smem(smem_w_addr + 0 * 1024, (&Weights), 0, tok_2, q_full_addr);
                }
                _phase_q_empty ^= 1;
                int num_kv_2 = (ctx_last_2 + 64 - 1) / 64;
                int kv_hi = ((q_atom_2 == end_q_2) ? end_kv_2 : num_kv_2);
                int bt_row = q_idx_2 * block_table_stride;
                int bt_lane = lane;
                int bt_ptr = 0;
                int bt_cache = bt_raw;
                if (num_kv_2 <= kv_lo + bt_lane * 2) {
                    bt_cache = 0;
                }
                #pragma unroll 1
                for (int kv_idx_2 = kv_lo; kv_idx_2 < kv_hi; kv_idx_2 += 2) {
                    int tile = kv_idx_2;
                    if (bt_ptr == 32) {
                        bt_ptr = 0;
                        int bt_tile = tile + bt_lane * 2;
                        bt_cache = 0;
                        if (bt_tile < num_kv_2) {
                            bt_cache = block_table[bt_row + bt_tile];
                        }
                    }
                    int kv_block = __shfl_sync(0xFFFFFFFF, bt_cache, bt_ptr);
                    bt_ptr += 1;
                    int page_off = 0;
                    mbarrier_wait(kv_empty_g0_addr + (kv_stage) * 8, _phase_kv_empty_g0);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_g0_addr + (kv_stage) * 8, 8448);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_kv_g0_addr + kv_stage * 8192), "l"((&KV)), "r"(0), "r"(page_off), "r"(kv_block),
                               "r"(kv_full_g0_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(smem_sc_g0_addr + kv_stage * 1024), "l"((&KV_scales)), "r"(2048 + page_off), "r"(kv_block),
                               "r"(kv_full_g0_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; _phase_kv_empty_g0 ^= 1; }
                }
            }
        }
    }
    // ---- Role: tma_g1 ----
    if (warp == 9) {
        { // tma_g1_main
            unsigned int q_phase_1 = 1;
            unsigned int kv_stage_1 = 0;
            unsigned int kv_phase_1 = 1;
            int sm_3 = bid;
            int start_q_3 = schedule_meta[sm_3 * 2];
            int start_kv_3 = schedule_meta[sm_3 * 2 + 1] * 2;
            int end_q_3 = schedule_meta[sm_3 * 2 + 2];
            int end_kv_3 = schedule_meta[sm_3 * 2 + 3] * 2;
            int q_stop_3 = ((end_kv_3 > 0) ? end_q_3 + 1 : end_q_3);
            unsigned int _phase_kv_empty_g1 = 1;
            #pragma unroll 1
            for (int q_atom_3 = start_q_3; q_atom_3 < q_stop_3; q_atom_3++) {
                int q_idx_3 = q_atom_3;
                int atom_in_q_3 = 0;
                int tok_3 = q_idx_3 * 3 + atom_in_q_3 * 3;
                int ctx_last_3 = context_lens[q_idx_3 * 3 + 3 - 1];
                int kv_lo_1 = ((q_atom_3 == start_q_3) ? start_kv_3 : 0);
                int bt_lane0_1 = lane;
                int bt_col0_1 = kv_lo_1 + 1 + bt_lane0_1 * 2;
                int bt_raw_1 = 0;
                if (bt_col0_1 < block_table_stride) {
                    bt_raw_1 = block_table[q_idx_3 * block_table_stride + bt_col0_1];
                }
                int num_kv_3 = (ctx_last_3 + 64 - 1) / 64;
                int kv_hi_1 = ((q_atom_3 == end_q_3) ? end_kv_3 : num_kv_3);
                int bt_row_1 = q_idx_3 * block_table_stride;
                int bt_lane_1 = lane;
                int bt_ptr_1 = 0;
                int bt_cache_1 = bt_raw_1;
                if (num_kv_3 <= kv_lo_1 + 1 + bt_lane_1 * 2) {
                    bt_cache_1 = 0;
                }
                #pragma unroll 1
                for (int kv_idx_3 = kv_lo_1; kv_idx_3 < kv_hi_1; kv_idx_3 += 2) {
                    int tile_1 = kv_idx_3 + 1;
                    if (bt_ptr_1 == 32) {
                        bt_ptr_1 = 0;
                        int bt_tile_1 = tile_1 + bt_lane_1 * 2;
                        bt_cache_1 = 0;
                        if (bt_tile_1 < num_kv_3) {
                            bt_cache_1 = block_table[bt_row_1 + bt_tile_1];
                        }
                    }
                    int kv_block_1 = __shfl_sync(0xFFFFFFFF, bt_cache_1, bt_ptr_1);
                    bt_ptr_1 += 1;
                    int page_off_1 = 0;
                    mbarrier_wait(kv_empty_g1_addr + (kv_stage_1) * 8, _phase_kv_empty_g1);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_g1_addr + (kv_stage_1) * 8, 8448);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_kv_g1_addr + kv_stage_1 * 8192), "l"((&KV)), "r"(0), "r"(page_off_1), "r"(kv_block_1),
                               "r"(kv_full_g1_addr + (kv_stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(smem_sc_g1_addr + kv_stage_1 * 1024), "l"((&KV_scales)), "r"(2048 + page_off_1), "r"(kv_block_1),
                               "r"(kv_full_g1_addr + (kv_stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    kv_stage_1 += 1;
                    if (kv_stage_1 == 3) { kv_stage_1 = 0; _phase_kv_empty_g1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp >= 10 && warp <= 11) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
