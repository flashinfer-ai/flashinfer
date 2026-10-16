/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_deepgemm_sm120_paged_mqa_logits_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_Q_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 3
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_W_OFF 33792
#define SMEM_SMEM_W_STAGE_BYTES 512
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_SMEM_KV_G0_OFF 35840
#define SMEM_SMEM_KV_G0_STAGE_BYTES 8192
#define SMEM_SMEM_KV_G0_STRIDE 8192
#define SMEM_SMEM_KV_G1_OFF 60416
#define SMEM_SMEM_KV_G1_STAGE_BYTES 8192
#define SMEM_SMEM_KV_G1_STRIDE 8192
#define SMEM_SMEM_SC_G0_OFF 84992
#define SMEM_SMEM_SC_G0_STAGE_BYTES 256
#define SMEM_SMEM_SC_G0_STRIDE 1024
#define SMEM_SMEM_SC_G1_OFF 88064
#define SMEM_SMEM_SC_G1_STAGE_BYTES 256
#define SMEM_SMEM_SC_G1_STRIDE 1024
#define SMEM_TOTAL 91136
#define THREADS 384
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_sm120_paged_mqa_logits_de4c4dab3c64e1081042(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights, float* __restrict__ Logits, int* __restrict__ context_lens, int* __restrict__ block_table, int* __restrict__ schedule_meta, int logits_stride, int block_table_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_g0_addr (mbar_base + 32)
    #define kv_full_g1_addr (mbar_base + 56)
    #define kv_empty_g0_addr (mbar_base + 80)
    #define kv_empty_g1_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_Q_OFF);
    const int smem_q_addr = smem + SMEM_SMEM_Q_OFF;
    float* smem_w = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_W_OFF);
    const int smem_w_addr = smem + SMEM_SMEM_W_OFF;
    uint8_t* smem_kv_g0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KV_G0_OFF);
    const int smem_kv_g0_addr = smem + SMEM_SMEM_KV_G0_OFF;
    uint8_t* smem_kv_g1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KV_G1_OFF);
    const int smem_kv_g1_addr = smem + SMEM_SMEM_KV_G1_OFF;
    float* smem_sc_g0 = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_SC_G0_OFF);
    const int smem_sc_g0_addr = smem + SMEM_SMEM_SC_G0_OFF;
    float* smem_sc_g1 = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_SC_G1_OFF);
    const int smem_sc_g1_addr = smem + SMEM_SMEM_SC_G1_OFF;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // q_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 16, 256);
            mbarrier_init(smem + 24, 256);
            // kv_full_g0: 3 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // kv_full_g1: 3 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // kv_empty_g0: 3 barriers, init_count=128
            mbarrier_init(smem + 80, 128);
            mbarrier_init(smem + 88, 128);
            mbarrier_init(smem + 96, 128);
            // kv_empty_g1: 3 barriers, init_count=128
            mbarrier_init(smem + 104, 128);
            mbarrier_init(smem + 112, 128);
            mbarrier_init(smem + 120, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
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
            unsigned int q_stage_m = 0;
            unsigned int q_phase_m = 0;
            unsigned int kv_stage_m = 0;
            unsigned int kv_phase_m = 0;
            unsigned int a_frag[4];
            unsigned int b_frag[2];
            float acc[4];
            unsigned int b_atom[128];
            int sm = bid;
            int start_q = schedule_meta[sm * 2];
            int start_kv = schedule_meta[sm * 2 + 1] * 2;
            int end_q = schedule_meta[sm * 2 + 2];
            int end_kv = schedule_meta[sm * 2 + 3] * 2;
            int q_stop = ((end_kv > 0) ? end_q + 1 : end_q);
            #pragma unroll 1
            for (int q_atom = start_q; q_atom < q_stop; q_atom++) {
                int q_idx = q_atom;
                int atom_in_q = 0;
                int tok = q_idx * 4 + atom_in_q * 4;
                int ctx_last = context_lens[q_idx * 4 + 4 - 1];
                int num_kv = (ctx_last + 64 - 1) / 64;
                int lo = ((q_atom == start_q) ? start_kv : 0);
                int hi = ((q_atom == end_q) ? end_kv : num_kv);
                {
                    mbarrier_wait(q_full_addr + (q_stage_m) * 8, q_phase_m);
                }
                int b_row_h = b_row_lane;
                int b_cb_h = b_col;
                unsigned int b_addr_h = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h * 128) + (unsigned int)(b_cb_h ^ (b_row_h & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[0]), "=r"(b_atom[1])
                    : "r"(b_addr_h)
                    : "memory");
                int b_cb_h_0 = b_col + 32;
                unsigned int b_addr_h_1 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h * 128) + (unsigned int)(b_cb_h_0 ^ (b_row_h & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[2]), "=r"(b_atom[3])
                    : "r"(b_addr_h_1)
                    : "memory");
                int b_cb_h_2 = b_col + 64;
                unsigned int b_addr_h_3 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h * 128) + (unsigned int)(b_cb_h_2 ^ (b_row_h & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[4]), "=r"(b_atom[5])
                    : "r"(b_addr_h_3)
                    : "memory");
                int b_cb_h_4 = b_col + 96;
                unsigned int b_addr_h_5 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h * 128) + (unsigned int)(b_cb_h_4 ^ (b_row_h & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[6]), "=r"(b_atom[7])
                    : "r"(b_addr_h_5)
                    : "memory");
                int b_row_h_6 = 8 + b_row_lane;
                int b_cb_h_7 = b_col;
                unsigned int b_addr_h_8 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_6 * 128) + (unsigned int)(b_cb_h_7 ^ (b_row_h_6 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[8]), "=r"(b_atom[9])
                    : "r"(b_addr_h_8)
                    : "memory");
                int b_cb_h_9 = b_col + 32;
                unsigned int b_addr_h_10 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_6 * 128) + (unsigned int)(b_cb_h_9 ^ (b_row_h_6 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[10]), "=r"(b_atom[11])
                    : "r"(b_addr_h_10)
                    : "memory");
                int b_cb_h_11 = b_col + 64;
                unsigned int b_addr_h_12 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_6 * 128) + (unsigned int)(b_cb_h_11 ^ (b_row_h_6 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[12]), "=r"(b_atom[13])
                    : "r"(b_addr_h_12)
                    : "memory");
                int b_cb_h_13 = b_col + 96;
                unsigned int b_addr_h_14 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_6 * 128) + (unsigned int)(b_cb_h_13 ^ (b_row_h_6 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[14]), "=r"(b_atom[15])
                    : "r"(b_addr_h_14)
                    : "memory");
                int b_row_h_15 = 16 + b_row_lane;
                int b_cb_h_16 = b_col;
                unsigned int b_addr_h_17 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_15 * 128) + (unsigned int)(b_cb_h_16 ^ (b_row_h_15 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[16]), "=r"(b_atom[17])
                    : "r"(b_addr_h_17)
                    : "memory");
                int b_cb_h_18 = b_col + 32;
                unsigned int b_addr_h_19 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_15 * 128) + (unsigned int)(b_cb_h_18 ^ (b_row_h_15 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[18]), "=r"(b_atom[19])
                    : "r"(b_addr_h_19)
                    : "memory");
                int b_cb_h_20 = b_col + 64;
                unsigned int b_addr_h_21 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_15 * 128) + (unsigned int)(b_cb_h_20 ^ (b_row_h_15 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[20]), "=r"(b_atom[21])
                    : "r"(b_addr_h_21)
                    : "memory");
                int b_cb_h_22 = b_col + 96;
                unsigned int b_addr_h_23 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_15 * 128) + (unsigned int)(b_cb_h_22 ^ (b_row_h_15 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[22]), "=r"(b_atom[23])
                    : "r"(b_addr_h_23)
                    : "memory");
                int b_row_h_24 = 24 + b_row_lane;
                int b_cb_h_25 = b_col;
                unsigned int b_addr_h_26 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_24 * 128) + (unsigned int)(b_cb_h_25 ^ (b_row_h_24 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[24]), "=r"(b_atom[25])
                    : "r"(b_addr_h_26)
                    : "memory");
                int b_cb_h_27 = b_col + 32;
                unsigned int b_addr_h_28 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_24 * 128) + (unsigned int)(b_cb_h_27 ^ (b_row_h_24 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[26]), "=r"(b_atom[27])
                    : "r"(b_addr_h_28)
                    : "memory");
                int b_cb_h_29 = b_col + 64;
                unsigned int b_addr_h_30 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_24 * 128) + (unsigned int)(b_cb_h_29 ^ (b_row_h_24 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[28]), "=r"(b_atom[29])
                    : "r"(b_addr_h_30)
                    : "memory");
                int b_cb_h_31 = b_col + 96;
                unsigned int b_addr_h_32 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_24 * 128) + (unsigned int)(b_cb_h_31 ^ (b_row_h_24 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[30]), "=r"(b_atom[31])
                    : "r"(b_addr_h_32)
                    : "memory");
                int b_row_h_33 = 32 + b_row_lane;
                int b_cb_h_34 = b_col;
                unsigned int b_addr_h_35 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_33 * 128) + (unsigned int)(b_cb_h_34 ^ (b_row_h_33 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[32]), "=r"(b_atom[33])
                    : "r"(b_addr_h_35)
                    : "memory");
                int b_cb_h_36 = b_col + 32;
                unsigned int b_addr_h_37 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_33 * 128) + (unsigned int)(b_cb_h_36 ^ (b_row_h_33 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[34]), "=r"(b_atom[35])
                    : "r"(b_addr_h_37)
                    : "memory");
                int b_cb_h_38 = b_col + 64;
                unsigned int b_addr_h_39 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_33 * 128) + (unsigned int)(b_cb_h_38 ^ (b_row_h_33 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[36]), "=r"(b_atom[37])
                    : "r"(b_addr_h_39)
                    : "memory");
                int b_cb_h_40 = b_col + 96;
                unsigned int b_addr_h_41 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_33 * 128) + (unsigned int)(b_cb_h_40 ^ (b_row_h_33 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[38]), "=r"(b_atom[39])
                    : "r"(b_addr_h_41)
                    : "memory");
                int b_row_h_42 = 40 + b_row_lane;
                int b_cb_h_43 = b_col;
                unsigned int b_addr_h_44 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_42 * 128) + (unsigned int)(b_cb_h_43 ^ (b_row_h_42 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[40]), "=r"(b_atom[41])
                    : "r"(b_addr_h_44)
                    : "memory");
                int b_cb_h_45 = b_col + 32;
                unsigned int b_addr_h_46 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_42 * 128) + (unsigned int)(b_cb_h_45 ^ (b_row_h_42 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[42]), "=r"(b_atom[43])
                    : "r"(b_addr_h_46)
                    : "memory");
                int b_cb_h_47 = b_col + 64;
                unsigned int b_addr_h_48 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_42 * 128) + (unsigned int)(b_cb_h_47 ^ (b_row_h_42 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[44]), "=r"(b_atom[45])
                    : "r"(b_addr_h_48)
                    : "memory");
                int b_cb_h_49 = b_col + 96;
                unsigned int b_addr_h_50 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_42 * 128) + (unsigned int)(b_cb_h_49 ^ (b_row_h_42 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[46]), "=r"(b_atom[47])
                    : "r"(b_addr_h_50)
                    : "memory");
                int b_row_h_51 = 48 + b_row_lane;
                int b_cb_h_52 = b_col;
                unsigned int b_addr_h_53 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_51 * 128) + (unsigned int)(b_cb_h_52 ^ (b_row_h_51 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[48]), "=r"(b_atom[49])
                    : "r"(b_addr_h_53)
                    : "memory");
                int b_cb_h_54 = b_col + 32;
                unsigned int b_addr_h_55 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_51 * 128) + (unsigned int)(b_cb_h_54 ^ (b_row_h_51 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[50]), "=r"(b_atom[51])
                    : "r"(b_addr_h_55)
                    : "memory");
                int b_cb_h_56 = b_col + 64;
                unsigned int b_addr_h_57 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_51 * 128) + (unsigned int)(b_cb_h_56 ^ (b_row_h_51 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[52]), "=r"(b_atom[53])
                    : "r"(b_addr_h_57)
                    : "memory");
                int b_cb_h_58 = b_col + 96;
                unsigned int b_addr_h_59 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_51 * 128) + (unsigned int)(b_cb_h_58 ^ (b_row_h_51 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[54]), "=r"(b_atom[55])
                    : "r"(b_addr_h_59)
                    : "memory");
                int b_row_h_60 = 56 + b_row_lane;
                int b_cb_h_61 = b_col;
                unsigned int b_addr_h_62 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_60 * 128) + (unsigned int)(b_cb_h_61 ^ (b_row_h_60 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[56]), "=r"(b_atom[57])
                    : "r"(b_addr_h_62)
                    : "memory");
                int b_cb_h_63 = b_col + 32;
                unsigned int b_addr_h_64 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_60 * 128) + (unsigned int)(b_cb_h_63 ^ (b_row_h_60 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[58]), "=r"(b_atom[59])
                    : "r"(b_addr_h_64)
                    : "memory");
                int b_cb_h_65 = b_col + 64;
                unsigned int b_addr_h_66 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_60 * 128) + (unsigned int)(b_cb_h_65 ^ (b_row_h_60 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[60]), "=r"(b_atom[61])
                    : "r"(b_addr_h_66)
                    : "memory");
                int b_cb_h_67 = b_col + 96;
                unsigned int b_addr_h_68 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_60 * 128) + (unsigned int)(b_cb_h_67 ^ (b_row_h_60 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[62]), "=r"(b_atom[63])
                    : "r"(b_addr_h_68)
                    : "memory");
                int b_row_h_69 = 64 + b_row_lane;
                int b_cb_h_70 = b_col;
                unsigned int b_addr_h_71 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_69 * 128) + (unsigned int)(b_cb_h_70 ^ (b_row_h_69 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[64]), "=r"(b_atom[65])
                    : "r"(b_addr_h_71)
                    : "memory");
                int b_cb_h_72 = b_col + 32;
                unsigned int b_addr_h_73 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_69 * 128) + (unsigned int)(b_cb_h_72 ^ (b_row_h_69 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[66]), "=r"(b_atom[67])
                    : "r"(b_addr_h_73)
                    : "memory");
                int b_cb_h_74 = b_col + 64;
                unsigned int b_addr_h_75 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_69 * 128) + (unsigned int)(b_cb_h_74 ^ (b_row_h_69 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[68]), "=r"(b_atom[69])
                    : "r"(b_addr_h_75)
                    : "memory");
                int b_cb_h_76 = b_col + 96;
                unsigned int b_addr_h_77 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_69 * 128) + (unsigned int)(b_cb_h_76 ^ (b_row_h_69 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[70]), "=r"(b_atom[71])
                    : "r"(b_addr_h_77)
                    : "memory");
                int b_row_h_78 = 72 + b_row_lane;
                int b_cb_h_79 = b_col;
                unsigned int b_addr_h_80 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_78 * 128) + (unsigned int)(b_cb_h_79 ^ (b_row_h_78 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[72]), "=r"(b_atom[73])
                    : "r"(b_addr_h_80)
                    : "memory");
                int b_cb_h_81 = b_col + 32;
                unsigned int b_addr_h_82 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_78 * 128) + (unsigned int)(b_cb_h_81 ^ (b_row_h_78 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[74]), "=r"(b_atom[75])
                    : "r"(b_addr_h_82)
                    : "memory");
                int b_cb_h_83 = b_col + 64;
                unsigned int b_addr_h_84 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_78 * 128) + (unsigned int)(b_cb_h_83 ^ (b_row_h_78 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[76]), "=r"(b_atom[77])
                    : "r"(b_addr_h_84)
                    : "memory");
                int b_cb_h_85 = b_col + 96;
                unsigned int b_addr_h_86 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_78 * 128) + (unsigned int)(b_cb_h_85 ^ (b_row_h_78 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[78]), "=r"(b_atom[79])
                    : "r"(b_addr_h_86)
                    : "memory");
                int b_row_h_87 = 80 + b_row_lane;
                int b_cb_h_88 = b_col;
                unsigned int b_addr_h_89 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_87 * 128) + (unsigned int)(b_cb_h_88 ^ (b_row_h_87 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[80]), "=r"(b_atom[81])
                    : "r"(b_addr_h_89)
                    : "memory");
                int b_cb_h_90 = b_col + 32;
                unsigned int b_addr_h_91 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_87 * 128) + (unsigned int)(b_cb_h_90 ^ (b_row_h_87 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[82]), "=r"(b_atom[83])
                    : "r"(b_addr_h_91)
                    : "memory");
                int b_cb_h_92 = b_col + 64;
                unsigned int b_addr_h_93 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_87 * 128) + (unsigned int)(b_cb_h_92 ^ (b_row_h_87 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[84]), "=r"(b_atom[85])
                    : "r"(b_addr_h_93)
                    : "memory");
                int b_cb_h_94 = b_col + 96;
                unsigned int b_addr_h_95 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_87 * 128) + (unsigned int)(b_cb_h_94 ^ (b_row_h_87 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[86]), "=r"(b_atom[87])
                    : "r"(b_addr_h_95)
                    : "memory");
                int b_row_h_96 = 88 + b_row_lane;
                int b_cb_h_97 = b_col;
                unsigned int b_addr_h_98 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_96 * 128) + (unsigned int)(b_cb_h_97 ^ (b_row_h_96 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[88]), "=r"(b_atom[89])
                    : "r"(b_addr_h_98)
                    : "memory");
                int b_cb_h_99 = b_col + 32;
                unsigned int b_addr_h_100 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_96 * 128) + (unsigned int)(b_cb_h_99 ^ (b_row_h_96 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[90]), "=r"(b_atom[91])
                    : "r"(b_addr_h_100)
                    : "memory");
                int b_cb_h_101 = b_col + 64;
                unsigned int b_addr_h_102 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_96 * 128) + (unsigned int)(b_cb_h_101 ^ (b_row_h_96 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[92]), "=r"(b_atom[93])
                    : "r"(b_addr_h_102)
                    : "memory");
                int b_cb_h_103 = b_col + 96;
                unsigned int b_addr_h_104 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_96 * 128) + (unsigned int)(b_cb_h_103 ^ (b_row_h_96 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[94]), "=r"(b_atom[95])
                    : "r"(b_addr_h_104)
                    : "memory");
                int b_row_h_105 = 96 + b_row_lane;
                int b_cb_h_106 = b_col;
                unsigned int b_addr_h_107 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_105 * 128) + (unsigned int)(b_cb_h_106 ^ (b_row_h_105 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[96]), "=r"(b_atom[97])
                    : "r"(b_addr_h_107)
                    : "memory");
                int b_cb_h_108 = b_col + 32;
                unsigned int b_addr_h_109 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_105 * 128) + (unsigned int)(b_cb_h_108 ^ (b_row_h_105 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[98]), "=r"(b_atom[99])
                    : "r"(b_addr_h_109)
                    : "memory");
                int b_cb_h_110 = b_col + 64;
                unsigned int b_addr_h_111 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_105 * 128) + (unsigned int)(b_cb_h_110 ^ (b_row_h_105 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[100]), "=r"(b_atom[101])
                    : "r"(b_addr_h_111)
                    : "memory");
                int b_cb_h_112 = b_col + 96;
                unsigned int b_addr_h_113 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_105 * 128) + (unsigned int)(b_cb_h_112 ^ (b_row_h_105 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[102]), "=r"(b_atom[103])
                    : "r"(b_addr_h_113)
                    : "memory");
                int b_row_h_114 = 104 + b_row_lane;
                int b_cb_h_115 = b_col;
                unsigned int b_addr_h_116 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_114 * 128) + (unsigned int)(b_cb_h_115 ^ (b_row_h_114 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[104]), "=r"(b_atom[105])
                    : "r"(b_addr_h_116)
                    : "memory");
                int b_cb_h_117 = b_col + 32;
                unsigned int b_addr_h_118 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_114 * 128) + (unsigned int)(b_cb_h_117 ^ (b_row_h_114 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[106]), "=r"(b_atom[107])
                    : "r"(b_addr_h_118)
                    : "memory");
                int b_cb_h_119 = b_col + 64;
                unsigned int b_addr_h_120 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_114 * 128) + (unsigned int)(b_cb_h_119 ^ (b_row_h_114 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[108]), "=r"(b_atom[109])
                    : "r"(b_addr_h_120)
                    : "memory");
                int b_cb_h_121 = b_col + 96;
                unsigned int b_addr_h_122 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_114 * 128) + (unsigned int)(b_cb_h_121 ^ (b_row_h_114 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[110]), "=r"(b_atom[111])
                    : "r"(b_addr_h_122)
                    : "memory");
                int b_row_h_123 = 112 + b_row_lane;
                int b_cb_h_124 = b_col;
                unsigned int b_addr_h_125 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_123 * 128) + (unsigned int)(b_cb_h_124 ^ (b_row_h_123 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[112]), "=r"(b_atom[113])
                    : "r"(b_addr_h_125)
                    : "memory");
                int b_cb_h_126 = b_col + 32;
                unsigned int b_addr_h_127 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_123 * 128) + (unsigned int)(b_cb_h_126 ^ (b_row_h_123 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[114]), "=r"(b_atom[115])
                    : "r"(b_addr_h_127)
                    : "memory");
                int b_cb_h_128 = b_col + 64;
                unsigned int b_addr_h_129 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_123 * 128) + (unsigned int)(b_cb_h_128 ^ (b_row_h_123 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[116]), "=r"(b_atom[117])
                    : "r"(b_addr_h_129)
                    : "memory");
                int b_cb_h_130 = b_col + 96;
                unsigned int b_addr_h_131 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_123 * 128) + (unsigned int)(b_cb_h_130 ^ (b_row_h_123 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[118]), "=r"(b_atom[119])
                    : "r"(b_addr_h_131)
                    : "memory");
                int b_row_h_132 = 120 + b_row_lane;
                int b_cb_h_133 = b_col;
                unsigned int b_addr_h_134 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_132 * 128) + (unsigned int)(b_cb_h_133 ^ (b_row_h_132 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[120]), "=r"(b_atom[121])
                    : "r"(b_addr_h_134)
                    : "memory");
                int b_cb_h_135 = b_col + 32;
                unsigned int b_addr_h_136 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_132 * 128) + (unsigned int)(b_cb_h_135 ^ (b_row_h_132 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[122]), "=r"(b_atom[123])
                    : "r"(b_addr_h_136)
                    : "memory");
                int b_cb_h_137 = b_col + 64;
                unsigned int b_addr_h_138 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_132 * 128) + (unsigned int)(b_cb_h_137 ^ (b_row_h_132 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[124]), "=r"(b_atom[125])
                    : "r"(b_addr_h_138)
                    : "memory");
                int b_cb_h_139 = b_col + 96;
                unsigned int b_addr_h_140 = smem_q_addr + q_stage_m * 16384 + (unsigned int)(b_row_h_132 * 128) + (unsigned int)(b_cb_h_139 ^ (b_row_h_132 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[126]), "=r"(b_atom[127])
                    : "r"(b_addr_h_140)
                    : "memory");
                int n_tok_m = 4;
                #pragma unroll 1
                for (int kv_idx = lo; kv_idx < hi; kv_idx += 2) {
                    {
                        mbarrier_wait(kv_full_g0_addr + (kv_stage_m) * 8, kv_phase_m);
                    }
                    int sc_idx = kv_stage_m * 256 + (unsigned int)(w_in_g * 16) + (unsigned int)quad;
                    float scale0 = smem_sc_g0[sc_idx];
                    float scale1 = smem_sc_g0[sc_idx + 8];
                    int kv_offset = tok * logits_stride + kv_idx * 64 + w_in_g * 16;
                    float partial0 = 0.0f;
                    float partial1 = 0.0f;
                    #pragma unroll
                    for (int nt = 0; nt < 4; nt++) {
                        int b_row = nt * 8 + b_row_lane;
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        int a_cb = a_col;
                        unsigned int a_addr = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + nt * 4 * 2)[0]), "r"((b_atom + nt * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_0 = a_col + 32;
                        unsigned int a_addr_1 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_0 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + (nt * 4 + 1) * 2)[0]), "r"((b_atom + (nt * 4 + 1) * 2)[1]));
                        int a_cb_2 = a_col + 64;
                        unsigned int a_addr_3 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_2 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_3)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + (nt * 4 + 2) * 2)[0]), "r"((b_atom + (nt * 4 + 2) * 2)[1]));
                        int a_cb_4 = a_col + 96;
                        unsigned int a_addr_5 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_4 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_5)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + (nt * 4 + 3) * 2)[0]), "r"((b_atom + (nt * 4 + 3) * 2)[1]));
                        int w_idx = q_stage_m * 256 + (unsigned int)(nt * 8) + (unsigned int)(tq * 2);
                        float w0 = smem_w[w_idx];
                        float w1 = smem_w[w_idx + 1];
                        float _max_0 = max_noftz(acc[0], 0.0f);
                        float r0 = _max_0;
                        float _max_1 = max_noftz(acc[1], 0.0f);
                        float r1 = _max_1;
                        float _max_2 = max_noftz(acc[2], 0.0f);
                        float r2 = _max_2;
                        float _max_3 = max_noftz(acc[3], 0.0f);
                        float r3 = _max_3;
                        partial0 += r0 * w0 + r1 * w1;
                        partial1 += r2 * w0 + r3 * w1;
                    }
                    float v0 = partial0 * scale0;
                    float v1 = partial1 * scale1;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, v0, 1);
                    v0 = v0 + _shfl_xor_0;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, v1, 1);
                    v1 = v1 + _shfl_xor_1;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, v0, 2);
                    v0 = v0 + _shfl_xor_2;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, v1, 2);
                    v1 = v1 + _shfl_xor_3;
                    int out0 = kv_offset + quad;
                    *(reinterpret_cast<float*>(Logits + out0) + (0)) = v0;
                    *(reinterpret_cast<float*>(Logits + (out0 + 8)) + (0)) = v1;
                    float partial0_0 = 0.0f;
                    float partial1_1 = 0.0f;
                    #pragma unroll
                    for (int nt_1 = 0; nt_1 < 4; nt_1++) {
                        int b_row_1 = 32 + nt_1 * 8 + b_row_lane;
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        int a_cb_1 = a_col;
                        unsigned int a_addr_2 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_1 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_2)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + (4 + nt_1) * 4 * 2)[0]), "r"((b_atom + (4 + nt_1) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_0_1 = a_col + 32;
                        unsigned int a_addr_1_1 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_0_1 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_1_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + ((4 + nt_1) * 4 + 1) * 2)[0]), "r"((b_atom + ((4 + nt_1) * 4 + 1) * 2)[1]));
                        int a_cb_2_1 = a_col + 64;
                        unsigned int a_addr_3_1 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_2_1 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_3_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + ((4 + nt_1) * 4 + 2) * 2)[0]), "r"((b_atom + ((4 + nt_1) * 4 + 2) * 2)[1]));
                        int a_cb_4_1 = a_col + 96;
                        unsigned int a_addr_5_1 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_4_1 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_5_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + ((4 + nt_1) * 4 + 3) * 2)[0]), "r"((b_atom + ((4 + nt_1) * 4 + 3) * 2)[1]));
                        int w_idx_1 = q_stage_m * 256 + 32 + (unsigned int)(nt_1 * 8) + (unsigned int)(tq * 2);
                        float w0_1 = smem_w[w_idx_1];
                        float w1_1 = smem_w[w_idx_1 + 1];
                        float _max_4 = max_noftz(acc[0], 0.0f);
                        float r0_1 = _max_4;
                        float _max_5 = max_noftz(acc[1], 0.0f);
                        float r1_1 = _max_5;
                        float _max_6 = max_noftz(acc[2], 0.0f);
                        float r2_1 = _max_6;
                        float _max_7 = max_noftz(acc[3], 0.0f);
                        float r3_1 = _max_7;
                        partial0_0 += r0_1 * w0_1 + r1_1 * w1_1;
                        partial1_1 += r2_1 * w0_1 + r3_1 * w1_1;
                    }
                    float v0_2 = partial0_0 * scale0;
                    float v1_3 = partial1_1 * scale1;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, v0_2, 1);
                    v0_2 = v0_2 + _shfl_xor_4;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, v1_3, 1);
                    v1_3 = v1_3 + _shfl_xor_5;
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, v0_2, 2);
                    v0_2 = v0_2 + _shfl_xor_6;
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, v1_3, 2);
                    v1_3 = v1_3 + _shfl_xor_7;
                    int out0_4 = kv_offset + logits_stride + quad;
                    *(reinterpret_cast<float*>(Logits + out0_4) + (0)) = v0_2;
                    *(reinterpret_cast<float*>(Logits + (out0_4 + 8)) + (0)) = v1_3;
                    float partial0_5 = 0.0f;
                    float partial1_6 = 0.0f;
                    #pragma unroll
                    for (int nt_2 = 0; nt_2 < 4; nt_2++) {
                        int b_row_2 = 64 + nt_2 * 8 + b_row_lane;
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        int a_cb_3 = a_col;
                        unsigned int a_addr_4 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_3 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_4)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + (8 + nt_2) * 4 * 2)[0]), "r"((b_atom + (8 + nt_2) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_0_2 = a_col + 32;
                        unsigned int a_addr_1_2 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_0_2 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_1_2)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + ((8 + nt_2) * 4 + 1) * 2)[0]), "r"((b_atom + ((8 + nt_2) * 4 + 1) * 2)[1]));
                        int a_cb_2_2 = a_col + 64;
                        unsigned int a_addr_3_2 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_2_2 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_3_2)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + ((8 + nt_2) * 4 + 2) * 2)[0]), "r"((b_atom + ((8 + nt_2) * 4 + 2) * 2)[1]));
                        int a_cb_4_2 = a_col + 96;
                        unsigned int a_addr_5_2 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_4_2 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_5_2)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + ((8 + nt_2) * 4 + 3) * 2)[0]), "r"((b_atom + ((8 + nt_2) * 4 + 3) * 2)[1]));
                        int w_idx_2 = q_stage_m * 256 + 64 + (unsigned int)(nt_2 * 8) + (unsigned int)(tq * 2);
                        float w0_2 = smem_w[w_idx_2];
                        float w1_2 = smem_w[w_idx_2 + 1];
                        float _max_8 = max_noftz(acc[0], 0.0f);
                        float r0_2 = _max_8;
                        float _max_9 = max_noftz(acc[1], 0.0f);
                        float r1_2 = _max_9;
                        float _max_10 = max_noftz(acc[2], 0.0f);
                        float r2_2 = _max_10;
                        float _max_11 = max_noftz(acc[3], 0.0f);
                        float r3_2 = _max_11;
                        partial0_5 += r0_2 * w0_2 + r1_2 * w1_2;
                        partial1_6 += r2_2 * w0_2 + r3_2 * w1_2;
                    }
                    float v0_7 = partial0_5 * scale0;
                    float v1_8 = partial1_6 * scale1;
                    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, v0_7, 1);
                    v0_7 = v0_7 + _shfl_xor_8;
                    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, v1_8, 1);
                    v1_8 = v1_8 + _shfl_xor_9;
                    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, v0_7, 2);
                    v0_7 = v0_7 + _shfl_xor_10;
                    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, v1_8, 2);
                    v1_8 = v1_8 + _shfl_xor_11;
                    int out0_9 = kv_offset + 2 * logits_stride + quad;
                    *(reinterpret_cast<float*>(Logits + out0_9) + (0)) = v0_7;
                    *(reinterpret_cast<float*>(Logits + (out0_9 + 8)) + (0)) = v1_8;
                    float partial0_10 = 0.0f;
                    float partial1_11 = 0.0f;
                    #pragma unroll
                    for (int nt_3 = 0; nt_3 < 4; nt_3++) {
                        int b_row_3 = 96 + nt_3 * 8 + b_row_lane;
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        int a_cb_5 = a_col;
                        unsigned int a_addr_6 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_5 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_6)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + (12 + nt_3) * 4 * 2)[0]), "r"((b_atom + (12 + nt_3) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_0_3 = a_col + 32;
                        unsigned int a_addr_1_3 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_0_3 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_1_3)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + ((12 + nt_3) * 4 + 1) * 2)[0]), "r"((b_atom + ((12 + nt_3) * 4 + 1) * 2)[1]));
                        int a_cb_2_3 = a_col + 64;
                        unsigned int a_addr_3_3 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_2_3 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_3_3)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + ((12 + nt_3) * 4 + 2) * 2)[0]), "r"((b_atom + ((12 + nt_3) * 4 + 2) * 2)[1]));
                        int a_cb_4_3 = a_col + 96;
                        unsigned int a_addr_5_3 = smem_kv_g0_addr + kv_stage_m * 8192 + (unsigned int)(a_row * 128) + (unsigned int)(a_cb_4_3 ^ (a_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(a_addr_5_3)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"((b_atom + ((12 + nt_3) * 4 + 3) * 2)[0]), "r"((b_atom + ((12 + nt_3) * 4 + 3) * 2)[1]));
                        int w_idx_3 = q_stage_m * 256 + 96 + (unsigned int)(nt_3 * 8) + (unsigned int)(tq * 2);
                        float w0_3 = smem_w[w_idx_3];
                        float w1_3 = smem_w[w_idx_3 + 1];
                        float _max_12 = max_noftz(acc[0], 0.0f);
                        float r0_3 = _max_12;
                        float _max_13 = max_noftz(acc[1], 0.0f);
                        float r1_3 = _max_13;
                        float _max_14 = max_noftz(acc[2], 0.0f);
                        float r2_3 = _max_14;
                        float _max_15 = max_noftz(acc[3], 0.0f);
                        float r3_3 = _max_15;
                        partial0_10 += r0_3 * w0_3 + r1_3 * w1_3;
                        partial1_11 += r2_3 * w0_3 + r3_3 * w1_3;
                    }
                    float v0_12 = partial0_10 * scale0;
                    float v1_13 = partial1_11 * scale1;
                    float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, v0_12, 1);
                    v0_12 = v0_12 + _shfl_xor_12;
                    float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, v1_13, 1);
                    v1_13 = v1_13 + _shfl_xor_13;
                    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, v0_12, 2);
                    v0_12 = v0_12 + _shfl_xor_14;
                    float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, v1_13, 2);
                    v1_13 = v1_13 + _shfl_xor_15;
                    int out0_14 = kv_offset + 3 * logits_stride + quad;
                    *(reinterpret_cast<float*>(Logits + out0_14) + (0)) = v0_12;
                    *(reinterpret_cast<float*>(Logits + (out0_14 + 8)) + (0)) = v1_13;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_empty_g0_addr + (kv_stage_m) * 8);
                    kv_stage_m += 1;
                    if (kv_stage_m == 3) { kv_stage_m = 0; kv_phase_m ^= 1; }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(q_empty_addr + (q_stage_m) * 8);
                q_stage_m += 1;
                if (q_stage_m == 2) { q_stage_m = 0; q_phase_m ^= 1; }
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
            unsigned int q_stage_m_1 = 0;
            unsigned int q_phase_m_1 = 0;
            unsigned int kv_stage_m_1 = 0;
            unsigned int kv_phase_m_1 = 0;
            unsigned int a_frag_1[4];
            unsigned int b_frag_1[2];
            float acc_1[4];
            unsigned int b_atom_1[128];
            int sm_1 = bid;
            int start_q_1 = schedule_meta[sm_1 * 2];
            int start_kv_1 = schedule_meta[sm_1 * 2 + 1] * 2;
            int end_q_1 = schedule_meta[sm_1 * 2 + 2];
            int end_kv_1 = schedule_meta[sm_1 * 2 + 3] * 2;
            int q_stop_1 = ((end_kv_1 > 0) ? end_q_1 + 1 : end_q_1);
            #pragma unroll 1
            for (int q_atom_1 = start_q_1; q_atom_1 < q_stop_1; q_atom_1++) {
                int q_idx_1 = q_atom_1;
                int atom_in_q_1 = 0;
                int tok_1 = q_idx_1 * 4 + atom_in_q_1 * 4;
                int ctx_last_1 = context_lens[q_idx_1 * 4 + 4 - 1];
                int num_kv_1 = (ctx_last_1 + 64 - 1) / 64;
                int lo_1 = ((q_atom_1 == start_q_1) ? start_kv_1 : 0);
                int hi_1 = ((q_atom_1 == end_q_1) ? end_kv_1 : num_kv_1);
                {
                    mbarrier_wait(q_full_addr + (q_stage_m_1) * 8, q_phase_m_1);
                }
                int b_row_h_1 = b_row_lane_1;
                int b_cb_h_1 = b_col_1;
                unsigned int b_addr_h_2 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_1 * 128) + (unsigned int)(b_cb_h_1 ^ (b_row_h_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[0]), "=r"(b_atom_1[1])
                    : "r"(b_addr_h_2)
                    : "memory");
                int b_cb_h_0_1 = b_col_1 + 32;
                unsigned int b_addr_h_1_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_1 * 128) + (unsigned int)(b_cb_h_0_1 ^ (b_row_h_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[2]), "=r"(b_atom_1[3])
                    : "r"(b_addr_h_1_1)
                    : "memory");
                int b_cb_h_2_1 = b_col_1 + 64;
                unsigned int b_addr_h_3_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_1 * 128) + (unsigned int)(b_cb_h_2_1 ^ (b_row_h_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[4]), "=r"(b_atom_1[5])
                    : "r"(b_addr_h_3_1)
                    : "memory");
                int b_cb_h_4_1 = b_col_1 + 96;
                unsigned int b_addr_h_5_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_1 * 128) + (unsigned int)(b_cb_h_4_1 ^ (b_row_h_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[6]), "=r"(b_atom_1[7])
                    : "r"(b_addr_h_5_1)
                    : "memory");
                int b_row_h_6_1 = 8 + b_row_lane_1;
                int b_cb_h_7_1 = b_col_1;
                unsigned int b_addr_h_8_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_6_1 * 128) + (unsigned int)(b_cb_h_7_1 ^ (b_row_h_6_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[8]), "=r"(b_atom_1[9])
                    : "r"(b_addr_h_8_1)
                    : "memory");
                int b_cb_h_9_1 = b_col_1 + 32;
                unsigned int b_addr_h_10_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_6_1 * 128) + (unsigned int)(b_cb_h_9_1 ^ (b_row_h_6_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[10]), "=r"(b_atom_1[11])
                    : "r"(b_addr_h_10_1)
                    : "memory");
                int b_cb_h_11_1 = b_col_1 + 64;
                unsigned int b_addr_h_12_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_6_1 * 128) + (unsigned int)(b_cb_h_11_1 ^ (b_row_h_6_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[12]), "=r"(b_atom_1[13])
                    : "r"(b_addr_h_12_1)
                    : "memory");
                int b_cb_h_13_1 = b_col_1 + 96;
                unsigned int b_addr_h_14_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_6_1 * 128) + (unsigned int)(b_cb_h_13_1 ^ (b_row_h_6_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[14]), "=r"(b_atom_1[15])
                    : "r"(b_addr_h_14_1)
                    : "memory");
                int b_row_h_15_1 = 16 + b_row_lane_1;
                int b_cb_h_16_1 = b_col_1;
                unsigned int b_addr_h_17_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_15_1 * 128) + (unsigned int)(b_cb_h_16_1 ^ (b_row_h_15_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[16]), "=r"(b_atom_1[17])
                    : "r"(b_addr_h_17_1)
                    : "memory");
                int b_cb_h_18_1 = b_col_1 + 32;
                unsigned int b_addr_h_19_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_15_1 * 128) + (unsigned int)(b_cb_h_18_1 ^ (b_row_h_15_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[18]), "=r"(b_atom_1[19])
                    : "r"(b_addr_h_19_1)
                    : "memory");
                int b_cb_h_20_1 = b_col_1 + 64;
                unsigned int b_addr_h_21_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_15_1 * 128) + (unsigned int)(b_cb_h_20_1 ^ (b_row_h_15_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[20]), "=r"(b_atom_1[21])
                    : "r"(b_addr_h_21_1)
                    : "memory");
                int b_cb_h_22_1 = b_col_1 + 96;
                unsigned int b_addr_h_23_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_15_1 * 128) + (unsigned int)(b_cb_h_22_1 ^ (b_row_h_15_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[22]), "=r"(b_atom_1[23])
                    : "r"(b_addr_h_23_1)
                    : "memory");
                int b_row_h_24_1 = 24 + b_row_lane_1;
                int b_cb_h_25_1 = b_col_1;
                unsigned int b_addr_h_26_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_24_1 * 128) + (unsigned int)(b_cb_h_25_1 ^ (b_row_h_24_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[24]), "=r"(b_atom_1[25])
                    : "r"(b_addr_h_26_1)
                    : "memory");
                int b_cb_h_27_1 = b_col_1 + 32;
                unsigned int b_addr_h_28_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_24_1 * 128) + (unsigned int)(b_cb_h_27_1 ^ (b_row_h_24_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[26]), "=r"(b_atom_1[27])
                    : "r"(b_addr_h_28_1)
                    : "memory");
                int b_cb_h_29_1 = b_col_1 + 64;
                unsigned int b_addr_h_30_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_24_1 * 128) + (unsigned int)(b_cb_h_29_1 ^ (b_row_h_24_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[28]), "=r"(b_atom_1[29])
                    : "r"(b_addr_h_30_1)
                    : "memory");
                int b_cb_h_31_1 = b_col_1 + 96;
                unsigned int b_addr_h_32_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_24_1 * 128) + (unsigned int)(b_cb_h_31_1 ^ (b_row_h_24_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[30]), "=r"(b_atom_1[31])
                    : "r"(b_addr_h_32_1)
                    : "memory");
                int b_row_h_33_1 = 32 + b_row_lane_1;
                int b_cb_h_34_1 = b_col_1;
                unsigned int b_addr_h_35_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_33_1 * 128) + (unsigned int)(b_cb_h_34_1 ^ (b_row_h_33_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[32]), "=r"(b_atom_1[33])
                    : "r"(b_addr_h_35_1)
                    : "memory");
                int b_cb_h_36_1 = b_col_1 + 32;
                unsigned int b_addr_h_37_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_33_1 * 128) + (unsigned int)(b_cb_h_36_1 ^ (b_row_h_33_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[34]), "=r"(b_atom_1[35])
                    : "r"(b_addr_h_37_1)
                    : "memory");
                int b_cb_h_38_1 = b_col_1 + 64;
                unsigned int b_addr_h_39_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_33_1 * 128) + (unsigned int)(b_cb_h_38_1 ^ (b_row_h_33_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[36]), "=r"(b_atom_1[37])
                    : "r"(b_addr_h_39_1)
                    : "memory");
                int b_cb_h_40_1 = b_col_1 + 96;
                unsigned int b_addr_h_41_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_33_1 * 128) + (unsigned int)(b_cb_h_40_1 ^ (b_row_h_33_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[38]), "=r"(b_atom_1[39])
                    : "r"(b_addr_h_41_1)
                    : "memory");
                int b_row_h_42_1 = 40 + b_row_lane_1;
                int b_cb_h_43_1 = b_col_1;
                unsigned int b_addr_h_44_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_42_1 * 128) + (unsigned int)(b_cb_h_43_1 ^ (b_row_h_42_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[40]), "=r"(b_atom_1[41])
                    : "r"(b_addr_h_44_1)
                    : "memory");
                int b_cb_h_45_1 = b_col_1 + 32;
                unsigned int b_addr_h_46_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_42_1 * 128) + (unsigned int)(b_cb_h_45_1 ^ (b_row_h_42_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[42]), "=r"(b_atom_1[43])
                    : "r"(b_addr_h_46_1)
                    : "memory");
                int b_cb_h_47_1 = b_col_1 + 64;
                unsigned int b_addr_h_48_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_42_1 * 128) + (unsigned int)(b_cb_h_47_1 ^ (b_row_h_42_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[44]), "=r"(b_atom_1[45])
                    : "r"(b_addr_h_48_1)
                    : "memory");
                int b_cb_h_49_1 = b_col_1 + 96;
                unsigned int b_addr_h_50_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_42_1 * 128) + (unsigned int)(b_cb_h_49_1 ^ (b_row_h_42_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[46]), "=r"(b_atom_1[47])
                    : "r"(b_addr_h_50_1)
                    : "memory");
                int b_row_h_51_1 = 48 + b_row_lane_1;
                int b_cb_h_52_1 = b_col_1;
                unsigned int b_addr_h_53_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_51_1 * 128) + (unsigned int)(b_cb_h_52_1 ^ (b_row_h_51_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[48]), "=r"(b_atom_1[49])
                    : "r"(b_addr_h_53_1)
                    : "memory");
                int b_cb_h_54_1 = b_col_1 + 32;
                unsigned int b_addr_h_55_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_51_1 * 128) + (unsigned int)(b_cb_h_54_1 ^ (b_row_h_51_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[50]), "=r"(b_atom_1[51])
                    : "r"(b_addr_h_55_1)
                    : "memory");
                int b_cb_h_56_1 = b_col_1 + 64;
                unsigned int b_addr_h_57_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_51_1 * 128) + (unsigned int)(b_cb_h_56_1 ^ (b_row_h_51_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[52]), "=r"(b_atom_1[53])
                    : "r"(b_addr_h_57_1)
                    : "memory");
                int b_cb_h_58_1 = b_col_1 + 96;
                unsigned int b_addr_h_59_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_51_1 * 128) + (unsigned int)(b_cb_h_58_1 ^ (b_row_h_51_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[54]), "=r"(b_atom_1[55])
                    : "r"(b_addr_h_59_1)
                    : "memory");
                int b_row_h_60_1 = 56 + b_row_lane_1;
                int b_cb_h_61_1 = b_col_1;
                unsigned int b_addr_h_62_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_60_1 * 128) + (unsigned int)(b_cb_h_61_1 ^ (b_row_h_60_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[56]), "=r"(b_atom_1[57])
                    : "r"(b_addr_h_62_1)
                    : "memory");
                int b_cb_h_63_1 = b_col_1 + 32;
                unsigned int b_addr_h_64_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_60_1 * 128) + (unsigned int)(b_cb_h_63_1 ^ (b_row_h_60_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[58]), "=r"(b_atom_1[59])
                    : "r"(b_addr_h_64_1)
                    : "memory");
                int b_cb_h_65_1 = b_col_1 + 64;
                unsigned int b_addr_h_66_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_60_1 * 128) + (unsigned int)(b_cb_h_65_1 ^ (b_row_h_60_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[60]), "=r"(b_atom_1[61])
                    : "r"(b_addr_h_66_1)
                    : "memory");
                int b_cb_h_67_1 = b_col_1 + 96;
                unsigned int b_addr_h_68_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_60_1 * 128) + (unsigned int)(b_cb_h_67_1 ^ (b_row_h_60_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[62]), "=r"(b_atom_1[63])
                    : "r"(b_addr_h_68_1)
                    : "memory");
                int b_row_h_69_1 = 64 + b_row_lane_1;
                int b_cb_h_70_1 = b_col_1;
                unsigned int b_addr_h_71_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_69_1 * 128) + (unsigned int)(b_cb_h_70_1 ^ (b_row_h_69_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[64]), "=r"(b_atom_1[65])
                    : "r"(b_addr_h_71_1)
                    : "memory");
                int b_cb_h_72_1 = b_col_1 + 32;
                unsigned int b_addr_h_73_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_69_1 * 128) + (unsigned int)(b_cb_h_72_1 ^ (b_row_h_69_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[66]), "=r"(b_atom_1[67])
                    : "r"(b_addr_h_73_1)
                    : "memory");
                int b_cb_h_74_1 = b_col_1 + 64;
                unsigned int b_addr_h_75_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_69_1 * 128) + (unsigned int)(b_cb_h_74_1 ^ (b_row_h_69_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[68]), "=r"(b_atom_1[69])
                    : "r"(b_addr_h_75_1)
                    : "memory");
                int b_cb_h_76_1 = b_col_1 + 96;
                unsigned int b_addr_h_77_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_69_1 * 128) + (unsigned int)(b_cb_h_76_1 ^ (b_row_h_69_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[70]), "=r"(b_atom_1[71])
                    : "r"(b_addr_h_77_1)
                    : "memory");
                int b_row_h_78_1 = 72 + b_row_lane_1;
                int b_cb_h_79_1 = b_col_1;
                unsigned int b_addr_h_80_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_78_1 * 128) + (unsigned int)(b_cb_h_79_1 ^ (b_row_h_78_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[72]), "=r"(b_atom_1[73])
                    : "r"(b_addr_h_80_1)
                    : "memory");
                int b_cb_h_81_1 = b_col_1 + 32;
                unsigned int b_addr_h_82_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_78_1 * 128) + (unsigned int)(b_cb_h_81_1 ^ (b_row_h_78_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[74]), "=r"(b_atom_1[75])
                    : "r"(b_addr_h_82_1)
                    : "memory");
                int b_cb_h_83_1 = b_col_1 + 64;
                unsigned int b_addr_h_84_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_78_1 * 128) + (unsigned int)(b_cb_h_83_1 ^ (b_row_h_78_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[76]), "=r"(b_atom_1[77])
                    : "r"(b_addr_h_84_1)
                    : "memory");
                int b_cb_h_85_1 = b_col_1 + 96;
                unsigned int b_addr_h_86_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_78_1 * 128) + (unsigned int)(b_cb_h_85_1 ^ (b_row_h_78_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[78]), "=r"(b_atom_1[79])
                    : "r"(b_addr_h_86_1)
                    : "memory");
                int b_row_h_87_1 = 80 + b_row_lane_1;
                int b_cb_h_88_1 = b_col_1;
                unsigned int b_addr_h_89_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_87_1 * 128) + (unsigned int)(b_cb_h_88_1 ^ (b_row_h_87_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[80]), "=r"(b_atom_1[81])
                    : "r"(b_addr_h_89_1)
                    : "memory");
                int b_cb_h_90_1 = b_col_1 + 32;
                unsigned int b_addr_h_91_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_87_1 * 128) + (unsigned int)(b_cb_h_90_1 ^ (b_row_h_87_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[82]), "=r"(b_atom_1[83])
                    : "r"(b_addr_h_91_1)
                    : "memory");
                int b_cb_h_92_1 = b_col_1 + 64;
                unsigned int b_addr_h_93_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_87_1 * 128) + (unsigned int)(b_cb_h_92_1 ^ (b_row_h_87_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[84]), "=r"(b_atom_1[85])
                    : "r"(b_addr_h_93_1)
                    : "memory");
                int b_cb_h_94_1 = b_col_1 + 96;
                unsigned int b_addr_h_95_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_87_1 * 128) + (unsigned int)(b_cb_h_94_1 ^ (b_row_h_87_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[86]), "=r"(b_atom_1[87])
                    : "r"(b_addr_h_95_1)
                    : "memory");
                int b_row_h_96_1 = 88 + b_row_lane_1;
                int b_cb_h_97_1 = b_col_1;
                unsigned int b_addr_h_98_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_96_1 * 128) + (unsigned int)(b_cb_h_97_1 ^ (b_row_h_96_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[88]), "=r"(b_atom_1[89])
                    : "r"(b_addr_h_98_1)
                    : "memory");
                int b_cb_h_99_1 = b_col_1 + 32;
                unsigned int b_addr_h_100_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_96_1 * 128) + (unsigned int)(b_cb_h_99_1 ^ (b_row_h_96_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[90]), "=r"(b_atom_1[91])
                    : "r"(b_addr_h_100_1)
                    : "memory");
                int b_cb_h_101_1 = b_col_1 + 64;
                unsigned int b_addr_h_102_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_96_1 * 128) + (unsigned int)(b_cb_h_101_1 ^ (b_row_h_96_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[92]), "=r"(b_atom_1[93])
                    : "r"(b_addr_h_102_1)
                    : "memory");
                int b_cb_h_103_1 = b_col_1 + 96;
                unsigned int b_addr_h_104_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_96_1 * 128) + (unsigned int)(b_cb_h_103_1 ^ (b_row_h_96_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[94]), "=r"(b_atom_1[95])
                    : "r"(b_addr_h_104_1)
                    : "memory");
                int b_row_h_105_1 = 96 + b_row_lane_1;
                int b_cb_h_106_1 = b_col_1;
                unsigned int b_addr_h_107_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_105_1 * 128) + (unsigned int)(b_cb_h_106_1 ^ (b_row_h_105_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[96]), "=r"(b_atom_1[97])
                    : "r"(b_addr_h_107_1)
                    : "memory");
                int b_cb_h_108_1 = b_col_1 + 32;
                unsigned int b_addr_h_109_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_105_1 * 128) + (unsigned int)(b_cb_h_108_1 ^ (b_row_h_105_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[98]), "=r"(b_atom_1[99])
                    : "r"(b_addr_h_109_1)
                    : "memory");
                int b_cb_h_110_1 = b_col_1 + 64;
                unsigned int b_addr_h_111_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_105_1 * 128) + (unsigned int)(b_cb_h_110_1 ^ (b_row_h_105_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[100]), "=r"(b_atom_1[101])
                    : "r"(b_addr_h_111_1)
                    : "memory");
                int b_cb_h_112_1 = b_col_1 + 96;
                unsigned int b_addr_h_113_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_105_1 * 128) + (unsigned int)(b_cb_h_112_1 ^ (b_row_h_105_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[102]), "=r"(b_atom_1[103])
                    : "r"(b_addr_h_113_1)
                    : "memory");
                int b_row_h_114_1 = 104 + b_row_lane_1;
                int b_cb_h_115_1 = b_col_1;
                unsigned int b_addr_h_116_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_114_1 * 128) + (unsigned int)(b_cb_h_115_1 ^ (b_row_h_114_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[104]), "=r"(b_atom_1[105])
                    : "r"(b_addr_h_116_1)
                    : "memory");
                int b_cb_h_117_1 = b_col_1 + 32;
                unsigned int b_addr_h_118_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_114_1 * 128) + (unsigned int)(b_cb_h_117_1 ^ (b_row_h_114_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[106]), "=r"(b_atom_1[107])
                    : "r"(b_addr_h_118_1)
                    : "memory");
                int b_cb_h_119_1 = b_col_1 + 64;
                unsigned int b_addr_h_120_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_114_1 * 128) + (unsigned int)(b_cb_h_119_1 ^ (b_row_h_114_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[108]), "=r"(b_atom_1[109])
                    : "r"(b_addr_h_120_1)
                    : "memory");
                int b_cb_h_121_1 = b_col_1 + 96;
                unsigned int b_addr_h_122_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_114_1 * 128) + (unsigned int)(b_cb_h_121_1 ^ (b_row_h_114_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[110]), "=r"(b_atom_1[111])
                    : "r"(b_addr_h_122_1)
                    : "memory");
                int b_row_h_123_1 = 112 + b_row_lane_1;
                int b_cb_h_124_1 = b_col_1;
                unsigned int b_addr_h_125_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_123_1 * 128) + (unsigned int)(b_cb_h_124_1 ^ (b_row_h_123_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[112]), "=r"(b_atom_1[113])
                    : "r"(b_addr_h_125_1)
                    : "memory");
                int b_cb_h_126_1 = b_col_1 + 32;
                unsigned int b_addr_h_127_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_123_1 * 128) + (unsigned int)(b_cb_h_126_1 ^ (b_row_h_123_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[114]), "=r"(b_atom_1[115])
                    : "r"(b_addr_h_127_1)
                    : "memory");
                int b_cb_h_128_1 = b_col_1 + 64;
                unsigned int b_addr_h_129_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_123_1 * 128) + (unsigned int)(b_cb_h_128_1 ^ (b_row_h_123_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[116]), "=r"(b_atom_1[117])
                    : "r"(b_addr_h_129_1)
                    : "memory");
                int b_cb_h_130_1 = b_col_1 + 96;
                unsigned int b_addr_h_131_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_123_1 * 128) + (unsigned int)(b_cb_h_130_1 ^ (b_row_h_123_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[118]), "=r"(b_atom_1[119])
                    : "r"(b_addr_h_131_1)
                    : "memory");
                int b_row_h_132_1 = 120 + b_row_lane_1;
                int b_cb_h_133_1 = b_col_1;
                unsigned int b_addr_h_134_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_132_1 * 128) + (unsigned int)(b_cb_h_133_1 ^ (b_row_h_132_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[120]), "=r"(b_atom_1[121])
                    : "r"(b_addr_h_134_1)
                    : "memory");
                int b_cb_h_135_1 = b_col_1 + 32;
                unsigned int b_addr_h_136_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_132_1 * 128) + (unsigned int)(b_cb_h_135_1 ^ (b_row_h_132_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[122]), "=r"(b_atom_1[123])
                    : "r"(b_addr_h_136_1)
                    : "memory");
                int b_cb_h_137_1 = b_col_1 + 64;
                unsigned int b_addr_h_138_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_132_1 * 128) + (unsigned int)(b_cb_h_137_1 ^ (b_row_h_132_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[124]), "=r"(b_atom_1[125])
                    : "r"(b_addr_h_138_1)
                    : "memory");
                int b_cb_h_139_1 = b_col_1 + 96;
                unsigned int b_addr_h_140_1 = smem_q_addr + q_stage_m_1 * 16384 + (unsigned int)(b_row_h_132_1 * 128) + (unsigned int)(b_cb_h_139_1 ^ (b_row_h_132_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[126]), "=r"(b_atom_1[127])
                    : "r"(b_addr_h_140_1)
                    : "memory");
                int n_tok_m_1 = 4;
                #pragma unroll 1
                for (int kv_idx_1 = lo_1; kv_idx_1 < hi_1; kv_idx_1 += 2) {
                    {
                        mbarrier_wait(kv_full_g1_addr + (kv_stage_m_1) * 8, kv_phase_m_1);
                    }
                    int sc_idx_1 = kv_stage_m_1 * 256 + (unsigned int)(w_in_g_1 * 16) + (unsigned int)quad_1;
                    float scale0_1 = smem_sc_g1[sc_idx_1];
                    float scale1_1 = smem_sc_g1[sc_idx_1 + 8];
                    int kv_offset_1 = tok_1 * logits_stride + (kv_idx_1 + 1) * 64 + w_in_g_1 * 16;
                    float partial0_1 = 0.0f;
                    float partial1_2 = 0.0f;
                    #pragma unroll
                    for (int nt_4 = 0; nt_4 < 4; nt_4++) {
                        int b_row_4 = nt_4 * 8 + b_row_lane_1;
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int a_cb_6 = a_col_1;
                        unsigned int a_addr_7 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_6 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_7)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + nt_4 * 4 * 2)[0]), "r"((b_atom_1 + nt_4 * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_0_4 = a_col_1 + 32;
                        unsigned int a_addr_1_4 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_0_4 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_1_4)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + (nt_4 * 4 + 1) * 2)[0]), "r"((b_atom_1 + (nt_4 * 4 + 1) * 2)[1]));
                        int a_cb_2_4 = a_col_1 + 64;
                        unsigned int a_addr_3_4 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_2_4 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_3_4)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + (nt_4 * 4 + 2) * 2)[0]), "r"((b_atom_1 + (nt_4 * 4 + 2) * 2)[1]));
                        int a_cb_4_4 = a_col_1 + 96;
                        unsigned int a_addr_5_4 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_4_4 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_5_4)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + (nt_4 * 4 + 3) * 2)[0]), "r"((b_atom_1 + (nt_4 * 4 + 3) * 2)[1]));
                        int w_idx_4 = q_stage_m_1 * 256 + (unsigned int)(nt_4 * 8) + (unsigned int)(tq_1 * 2);
                        float w0_4 = smem_w[w_idx_4];
                        float w1_4 = smem_w[w_idx_4 + 1];
                        float _max_16 = max_noftz(acc_1[0], 0.0f);
                        float r0_4 = _max_16;
                        float _max_17 = max_noftz(acc_1[1], 0.0f);
                        float r1_4 = _max_17;
                        float _max_18 = max_noftz(acc_1[2], 0.0f);
                        float r2_4 = _max_18;
                        float _max_19 = max_noftz(acc_1[3], 0.0f);
                        float r3_4 = _max_19;
                        partial0_1 += r0_4 * w0_4 + r1_4 * w1_4;
                        partial1_2 += r2_4 * w0_4 + r3_4 * w1_4;
                    }
                    float v0_1 = partial0_1 * scale0_1;
                    float v1_1 = partial1_2 * scale1_1;
                    float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, v0_1, 1);
                    v0_1 = v0_1 + _shfl_xor_16;
                    float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 1);
                    v1_1 = v1_1 + _shfl_xor_17;
                    float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, v0_1, 2);
                    v0_1 = v0_1 + _shfl_xor_18;
                    float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 2);
                    v1_1 = v1_1 + _shfl_xor_19;
                    int out0_1 = kv_offset_1 + quad_1;
                    *(reinterpret_cast<float*>(Logits + out0_1) + (0)) = v0_1;
                    *(reinterpret_cast<float*>(Logits + (out0_1 + 8)) + (0)) = v1_1;
                    float partial0_0_1 = 0.0f;
                    float partial1_1_1 = 0.0f;
                    #pragma unroll
                    for (int nt_5 = 0; nt_5 < 4; nt_5++) {
                        int b_row_5 = 32 + nt_5 * 8 + b_row_lane_1;
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int a_cb_7 = a_col_1;
                        unsigned int a_addr_8 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_7 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_8)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + (4 + nt_5) * 4 * 2)[0]), "r"((b_atom_1 + (4 + nt_5) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_0_5 = a_col_1 + 32;
                        unsigned int a_addr_1_5 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_0_5 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_1_5)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + ((4 + nt_5) * 4 + 1) * 2)[0]), "r"((b_atom_1 + ((4 + nt_5) * 4 + 1) * 2)[1]));
                        int a_cb_2_5 = a_col_1 + 64;
                        unsigned int a_addr_3_5 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_2_5 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_3_5)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + ((4 + nt_5) * 4 + 2) * 2)[0]), "r"((b_atom_1 + ((4 + nt_5) * 4 + 2) * 2)[1]));
                        int a_cb_4_5 = a_col_1 + 96;
                        unsigned int a_addr_5_5 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_4_5 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_5_5)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + ((4 + nt_5) * 4 + 3) * 2)[0]), "r"((b_atom_1 + ((4 + nt_5) * 4 + 3) * 2)[1]));
                        int w_idx_5 = q_stage_m_1 * 256 + 32 + (unsigned int)(nt_5 * 8) + (unsigned int)(tq_1 * 2);
                        float w0_5 = smem_w[w_idx_5];
                        float w1_5 = smem_w[w_idx_5 + 1];
                        float _max_20 = max_noftz(acc_1[0], 0.0f);
                        float r0_5 = _max_20;
                        float _max_21 = max_noftz(acc_1[1], 0.0f);
                        float r1_5 = _max_21;
                        float _max_22 = max_noftz(acc_1[2], 0.0f);
                        float r2_5 = _max_22;
                        float _max_23 = max_noftz(acc_1[3], 0.0f);
                        float r3_5 = _max_23;
                        partial0_0_1 += r0_5 * w0_5 + r1_5 * w1_5;
                        partial1_1_1 += r2_5 * w0_5 + r3_5 * w1_5;
                    }
                    float v0_2_1 = partial0_0_1 * scale0_1;
                    float v1_3_1 = partial1_1_1 * scale1_1;
                    float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, v0_2_1, 1);
                    v0_2_1 = v0_2_1 + _shfl_xor_20;
                    float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, v1_3_1, 1);
                    v1_3_1 = v1_3_1 + _shfl_xor_21;
                    float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, v0_2_1, 2);
                    v0_2_1 = v0_2_1 + _shfl_xor_22;
                    float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, v1_3_1, 2);
                    v1_3_1 = v1_3_1 + _shfl_xor_23;
                    int out0_4_1 = kv_offset_1 + logits_stride + quad_1;
                    *(reinterpret_cast<float*>(Logits + out0_4_1) + (0)) = v0_2_1;
                    *(reinterpret_cast<float*>(Logits + (out0_4_1 + 8)) + (0)) = v1_3_1;
                    float partial0_5_1 = 0.0f;
                    float partial1_6_1 = 0.0f;
                    #pragma unroll
                    for (int nt_6 = 0; nt_6 < 4; nt_6++) {
                        int b_row_6 = 64 + nt_6 * 8 + b_row_lane_1;
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int a_cb_8 = a_col_1;
                        unsigned int a_addr_9 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_8 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_9)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + (8 + nt_6) * 4 * 2)[0]), "r"((b_atom_1 + (8 + nt_6) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_0_6 = a_col_1 + 32;
                        unsigned int a_addr_1_6 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_0_6 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_1_6)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + ((8 + nt_6) * 4 + 1) * 2)[0]), "r"((b_atom_1 + ((8 + nt_6) * 4 + 1) * 2)[1]));
                        int a_cb_2_6 = a_col_1 + 64;
                        unsigned int a_addr_3_6 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_2_6 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_3_6)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + ((8 + nt_6) * 4 + 2) * 2)[0]), "r"((b_atom_1 + ((8 + nt_6) * 4 + 2) * 2)[1]));
                        int a_cb_4_6 = a_col_1 + 96;
                        unsigned int a_addr_5_6 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_4_6 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_5_6)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + ((8 + nt_6) * 4 + 3) * 2)[0]), "r"((b_atom_1 + ((8 + nt_6) * 4 + 3) * 2)[1]));
                        int w_idx_6 = q_stage_m_1 * 256 + 64 + (unsigned int)(nt_6 * 8) + (unsigned int)(tq_1 * 2);
                        float w0_6 = smem_w[w_idx_6];
                        float w1_6 = smem_w[w_idx_6 + 1];
                        float _max_24 = max_noftz(acc_1[0], 0.0f);
                        float r0_6 = _max_24;
                        float _max_25 = max_noftz(acc_1[1], 0.0f);
                        float r1_6 = _max_25;
                        float _max_26 = max_noftz(acc_1[2], 0.0f);
                        float r2_6 = _max_26;
                        float _max_27 = max_noftz(acc_1[3], 0.0f);
                        float r3_6 = _max_27;
                        partial0_5_1 += r0_6 * w0_6 + r1_6 * w1_6;
                        partial1_6_1 += r2_6 * w0_6 + r3_6 * w1_6;
                    }
                    float v0_7_1 = partial0_5_1 * scale0_1;
                    float v1_8_1 = partial1_6_1 * scale1_1;
                    float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, v0_7_1, 1);
                    v0_7_1 = v0_7_1 + _shfl_xor_24;
                    float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, v1_8_1, 1);
                    v1_8_1 = v1_8_1 + _shfl_xor_25;
                    float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, v0_7_1, 2);
                    v0_7_1 = v0_7_1 + _shfl_xor_26;
                    float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, v1_8_1, 2);
                    v1_8_1 = v1_8_1 + _shfl_xor_27;
                    int out0_9_1 = kv_offset_1 + 2 * logits_stride + quad_1;
                    *(reinterpret_cast<float*>(Logits + out0_9_1) + (0)) = v0_7_1;
                    *(reinterpret_cast<float*>(Logits + (out0_9_1 + 8)) + (0)) = v1_8_1;
                    float partial0_10_1 = 0.0f;
                    float partial1_11_1 = 0.0f;
                    #pragma unroll
                    for (int nt_7 = 0; nt_7 < 4; nt_7++) {
                        int b_row_7 = 96 + nt_7 * 8 + b_row_lane_1;
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int a_cb_9 = a_col_1;
                        unsigned int a_addr_10 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_9 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_10)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + (12 + nt_7) * 4 * 2)[0]), "r"((b_atom_1 + (12 + nt_7) * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_0_7 = a_col_1 + 32;
                        unsigned int a_addr_1_7 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_0_7 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_1_7)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + ((12 + nt_7) * 4 + 1) * 2)[0]), "r"((b_atom_1 + ((12 + nt_7) * 4 + 1) * 2)[1]));
                        int a_cb_2_7 = a_col_1 + 64;
                        unsigned int a_addr_3_7 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_2_7 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_3_7)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + ((12 + nt_7) * 4 + 2) * 2)[0]), "r"((b_atom_1 + ((12 + nt_7) * 4 + 2) * 2)[1]));
                        int a_cb_4_7 = a_col_1 + 96;
                        unsigned int a_addr_5_7 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_4_7 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_5_7)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + ((12 + nt_7) * 4 + 3) * 2)[0]), "r"((b_atom_1 + ((12 + nt_7) * 4 + 3) * 2)[1]));
                        int w_idx_7 = q_stage_m_1 * 256 + 96 + (unsigned int)(nt_7 * 8) + (unsigned int)(tq_1 * 2);
                        float w0_7 = smem_w[w_idx_7];
                        float w1_7 = smem_w[w_idx_7 + 1];
                        float _max_28 = max_noftz(acc_1[0], 0.0f);
                        float r0_7 = _max_28;
                        float _max_29 = max_noftz(acc_1[1], 0.0f);
                        float r1_7 = _max_29;
                        float _max_30 = max_noftz(acc_1[2], 0.0f);
                        float r2_7 = _max_30;
                        float _max_31 = max_noftz(acc_1[3], 0.0f);
                        float r3_7 = _max_31;
                        partial0_10_1 += r0_7 * w0_7 + r1_7 * w1_7;
                        partial1_11_1 += r2_7 * w0_7 + r3_7 * w1_7;
                    }
                    float v0_12_1 = partial0_10_1 * scale0_1;
                    float v1_13_1 = partial1_11_1 * scale1_1;
                    float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, v0_12_1, 1);
                    v0_12_1 = v0_12_1 + _shfl_xor_28;
                    float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, v1_13_1, 1);
                    v1_13_1 = v1_13_1 + _shfl_xor_29;
                    float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, v0_12_1, 2);
                    v0_12_1 = v0_12_1 + _shfl_xor_30;
                    float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, v1_13_1, 2);
                    v1_13_1 = v1_13_1 + _shfl_xor_31;
                    int out0_14_1 = kv_offset_1 + 3 * logits_stride + quad_1;
                    *(reinterpret_cast<float*>(Logits + out0_14_1) + (0)) = v0_12_1;
                    *(reinterpret_cast<float*>(Logits + (out0_14_1 + 8)) + (0)) = v1_13_1;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_empty_g1_addr + (kv_stage_m_1) * 8);
                    kv_stage_m_1 += 1;
                    if (kv_stage_m_1 == 3) { kv_stage_m_1 = 0; kv_phase_m_1 ^= 1; }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(q_empty_addr + (q_stage_m_1) * 8);
                q_stage_m_1 += 1;
                if (q_stage_m_1 == 2) { q_stage_m_1 = 0; q_phase_m_1 ^= 1; }
            }
        }
    }
    // ---- Role: tma_g0 ----
    if (warp == 8) {
        { // tma_g0_main
            unsigned int q_stage = 0;
            unsigned int q_phase = 1;
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 1;
            int sm_2 = bid;
            int start_q_2 = schedule_meta[sm_2 * 2];
            int start_kv_2 = schedule_meta[sm_2 * 2 + 1] * 2;
            int end_q_2 = schedule_meta[sm_2 * 2 + 2];
            int end_kv_2 = schedule_meta[sm_2 * 2 + 3] * 2;
            int q_stop_2 = ((end_kv_2 > 0) ? end_q_2 + 1 : end_q_2);
            #pragma unroll 1
            for (int q_atom_2 = start_q_2; q_atom_2 < q_stop_2; q_atom_2++) {
                int q_idx_2 = q_atom_2;
                int atom_in_q_2 = 0;
                int tok_2 = q_idx_2 * 4 + atom_in_q_2 * 4;
                int ctx_last_2 = context_lens[q_idx_2 * 4 + 4 - 1];
                int num_kv_2 = (ctx_last_2 + 64 - 1) / 64;
                int lo_2 = ((q_atom_2 == start_q_2) ? start_kv_2 : 0);
                int hi_2 = ((q_atom_2 == end_q_2) ? end_kv_2 : num_kv_2);
                {
                    mbarrier_wait(q_empty_addr + (q_stage) * 8, q_phase);
                }
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + (q_stage) * 8, 16896);
                    tma_2d_gmem2smem(smem_q_addr + q_stage * 16384, (&Q), 0, tok_2 * 32, q_full_addr + (q_stage) * 8);
                    tma_2d_gmem2smem(smem_w_addr + q_stage * 1024, (&Weights), 0, tok_2, q_full_addr + (q_stage) * 8);
                }
                q_stage += 1;
                if (q_stage == 2) { q_stage = 0; q_phase ^= 1; }
                int bt_row = q_idx_2 * block_table_stride;
                int bt_lane = lane;
                int bt_ptr = 32;
                int bt_cache = 0;
                #pragma unroll 1
                for (int kv_idx_2 = lo_2; kv_idx_2 < hi_2; kv_idx_2 += 2) {
                    int tile = kv_idx_2;
                    if (bt_ptr == 32) {
                        bt_ptr = 0;
                        int bt_tile = tile + bt_lane * 2;
                        bt_cache = 0;
                        if (bt_tile < num_kv_2) {
                            bt_cache = block_table[bt_row + bt_tile / 2];
                        }
                    }
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, bt_cache, bt_ptr);
                    int kv_block = _shfl_0;
                    bt_ptr += 1;
                    int page_off = tile % 2 * 64;
                    {
                        mbarrier_wait(kv_empty_g0_addr + (kv_stage) * 8, kv_phase);
                    }
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_g0_addr + (kv_stage) * 8, 8448);
                        tma_3d_gmem2smem(smem_kv_g0_addr + kv_stage * 8192, (&KV), 0, page_off, kv_block, kv_full_g0_addr + (kv_stage) * 8);
                        tma_2d_gmem2smem(smem_sc_g0_addr + kv_stage * 1024, (&KV_scales), 4096 + page_off, kv_block, kv_full_g0_addr + (kv_stage) * 8);
                    }
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                }
            }
        }
    }
    // ---- Role: tma_g1 ----
    if (warp == 9) {
        { // tma_g1_main
            unsigned int q_stage_1 = 0;
            unsigned int q_phase_1 = 1;
            unsigned int kv_stage_1 = 0;
            unsigned int kv_phase_1 = 1;
            int sm_3 = bid;
            int start_q_3 = schedule_meta[sm_3 * 2];
            int start_kv_3 = schedule_meta[sm_3 * 2 + 1] * 2;
            int end_q_3 = schedule_meta[sm_3 * 2 + 2];
            int end_kv_3 = schedule_meta[sm_3 * 2 + 3] * 2;
            int q_stop_3 = ((end_kv_3 > 0) ? end_q_3 + 1 : end_q_3);
            #pragma unroll 1
            for (int q_atom_3 = start_q_3; q_atom_3 < q_stop_3; q_atom_3++) {
                int q_idx_3 = q_atom_3;
                int atom_in_q_3 = 0;
                int tok_3 = q_idx_3 * 4 + atom_in_q_3 * 4;
                int ctx_last_3 = context_lens[q_idx_3 * 4 + 4 - 1];
                int num_kv_3 = (ctx_last_3 + 64 - 1) / 64;
                int lo_3 = ((q_atom_3 == start_q_3) ? start_kv_3 : 0);
                int hi_3 = ((q_atom_3 == end_q_3) ? end_kv_3 : num_kv_3);
                int bt_row_1 = q_idx_3 * block_table_stride;
                int bt_lane_1 = lane;
                int bt_ptr_1 = 32;
                int bt_cache_1 = 0;
                #pragma unroll 1
                for (int kv_idx_3 = lo_3; kv_idx_3 < hi_3; kv_idx_3 += 2) {
                    int tile_1 = kv_idx_3 + 1;
                    if (bt_ptr_1 == 32) {
                        bt_ptr_1 = 0;
                        int bt_tile_1 = tile_1 + bt_lane_1 * 2;
                        bt_cache_1 = 0;
                        if (bt_tile_1 < num_kv_3) {
                            bt_cache_1 = block_table[bt_row_1 + bt_tile_1 / 2];
                        }
                    }
                    int _shfl_1 = __shfl_sync(0xFFFFFFFF, bt_cache_1, bt_ptr_1);
                    int kv_block_1 = _shfl_1;
                    bt_ptr_1 += 1;
                    int page_off_1 = tile_1 % 2 * 64;
                    {
                        mbarrier_wait(kv_empty_g1_addr + (kv_stage_1) * 8, kv_phase_1);
                    }
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_g1_addr + (kv_stage_1) * 8, 8448);
                        tma_3d_gmem2smem(smem_kv_g1_addr + kv_stage_1 * 8192, (&KV), 0, page_off_1, kv_block_1, kv_full_g1_addr + (kv_stage_1) * 8);
                        tma_2d_gmem2smem(smem_sc_g1_addr + kv_stage_1 * 1024, (&KV_scales), 4096 + page_off_1, kv_block_1, kv_full_g1_addr + (kv_stage_1) * 8);
                    }
                    kv_stage_1 += 1;
                    if (kv_stage_1 == 3) { kv_stage_1 = 0; kv_phase_1 ^= 1; }
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
