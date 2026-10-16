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
#define SMEM_SMEM_Q_STAGE_BYTES 4096
#define SMEM_SMEM_Q_STRIDE 4096
#define SMEM_SMEM_W_OFF 9216
#define SMEM_SMEM_W_STAGE_BYTES 128
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_SMEM_KV_G0_OFF 11264
#define SMEM_SMEM_KV_G0_STAGE_BYTES 8192
#define SMEM_SMEM_KV_G0_STRIDE 8192
#define SMEM_SMEM_KV_G1_OFF 35840
#define SMEM_SMEM_KV_G1_STAGE_BYTES 8192
#define SMEM_SMEM_KV_G1_STRIDE 8192
#define SMEM_SMEM_SC_G0_OFF 60416
#define SMEM_SMEM_SC_G0_STAGE_BYTES 256
#define SMEM_SMEM_SC_G0_STRIDE 1024
#define SMEM_SMEM_SC_G1_OFF 63488
#define SMEM_SMEM_SC_G1_STAGE_BYTES 256
#define SMEM_SMEM_SC_G1_STRIDE 1024
#define SMEM_TOTAL 66560
#define THREADS 384
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_sm120_paged_mqa_logits_a01d9cb34eb024910169(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights, float* __restrict__ Logits, int* __restrict__ context_lens, int* __restrict__ block_table, int* __restrict__ schedule_meta, int logits_stride, int block_table_stride)
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
            unsigned int b_atom[32];
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
                int tok = q_idx + atom_in_q;
                int ctx_last = context_lens[q_idx + 1 - 1];
                int num_kv = (ctx_last + 64 - 1) / 64;
                int lo = ((q_atom == start_q) ? start_kv : 0);
                int hi = ((q_atom == end_q) ? end_kv : num_kv);
                {
                    mbarrier_wait(q_full_addr + (q_stage_m) * 8, q_phase_m);
                }
                int b_row_h = b_row_lane;
                int b_cb_h = b_col;
                unsigned int b_addr_h = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h * 128) + (unsigned int)(b_cb_h ^ (b_row_h & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[0]), "=r"(b_atom[1])
                    : "r"(b_addr_h)
                    : "memory");
                int b_cb_h_0 = b_col + 32;
                unsigned int b_addr_h_1 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h * 128) + (unsigned int)(b_cb_h_0 ^ (b_row_h & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[2]), "=r"(b_atom[3])
                    : "r"(b_addr_h_1)
                    : "memory");
                int b_cb_h_2 = b_col + 64;
                unsigned int b_addr_h_3 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h * 128) + (unsigned int)(b_cb_h_2 ^ (b_row_h & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[4]), "=r"(b_atom[5])
                    : "r"(b_addr_h_3)
                    : "memory");
                int b_cb_h_4 = b_col + 96;
                unsigned int b_addr_h_5 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h * 128) + (unsigned int)(b_cb_h_4 ^ (b_row_h & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[6]), "=r"(b_atom[7])
                    : "r"(b_addr_h_5)
                    : "memory");
                int b_row_h_6 = 8 + b_row_lane;
                int b_cb_h_7 = b_col;
                unsigned int b_addr_h_8 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h_6 * 128) + (unsigned int)(b_cb_h_7 ^ (b_row_h_6 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[8]), "=r"(b_atom[9])
                    : "r"(b_addr_h_8)
                    : "memory");
                int b_cb_h_9 = b_col + 32;
                unsigned int b_addr_h_10 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h_6 * 128) + (unsigned int)(b_cb_h_9 ^ (b_row_h_6 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[10]), "=r"(b_atom[11])
                    : "r"(b_addr_h_10)
                    : "memory");
                int b_cb_h_11 = b_col + 64;
                unsigned int b_addr_h_12 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h_6 * 128) + (unsigned int)(b_cb_h_11 ^ (b_row_h_6 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[12]), "=r"(b_atom[13])
                    : "r"(b_addr_h_12)
                    : "memory");
                int b_cb_h_13 = b_col + 96;
                unsigned int b_addr_h_14 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h_6 * 128) + (unsigned int)(b_cb_h_13 ^ (b_row_h_6 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[14]), "=r"(b_atom[15])
                    : "r"(b_addr_h_14)
                    : "memory");
                int b_row_h_15 = 16 + b_row_lane;
                int b_cb_h_16 = b_col;
                unsigned int b_addr_h_17 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h_15 * 128) + (unsigned int)(b_cb_h_16 ^ (b_row_h_15 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[16]), "=r"(b_atom[17])
                    : "r"(b_addr_h_17)
                    : "memory");
                int b_cb_h_18 = b_col + 32;
                unsigned int b_addr_h_19 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h_15 * 128) + (unsigned int)(b_cb_h_18 ^ (b_row_h_15 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[18]), "=r"(b_atom[19])
                    : "r"(b_addr_h_19)
                    : "memory");
                int b_cb_h_20 = b_col + 64;
                unsigned int b_addr_h_21 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h_15 * 128) + (unsigned int)(b_cb_h_20 ^ (b_row_h_15 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[20]), "=r"(b_atom[21])
                    : "r"(b_addr_h_21)
                    : "memory");
                int b_cb_h_22 = b_col + 96;
                unsigned int b_addr_h_23 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h_15 * 128) + (unsigned int)(b_cb_h_22 ^ (b_row_h_15 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[22]), "=r"(b_atom[23])
                    : "r"(b_addr_h_23)
                    : "memory");
                int b_row_h_24 = 24 + b_row_lane;
                int b_cb_h_25 = b_col;
                unsigned int b_addr_h_26 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h_24 * 128) + (unsigned int)(b_cb_h_25 ^ (b_row_h_24 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[24]), "=r"(b_atom[25])
                    : "r"(b_addr_h_26)
                    : "memory");
                int b_cb_h_27 = b_col + 32;
                unsigned int b_addr_h_28 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h_24 * 128) + (unsigned int)(b_cb_h_27 ^ (b_row_h_24 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[26]), "=r"(b_atom[27])
                    : "r"(b_addr_h_28)
                    : "memory");
                int b_cb_h_29 = b_col + 64;
                unsigned int b_addr_h_30 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h_24 * 128) + (unsigned int)(b_cb_h_29 ^ (b_row_h_24 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[28]), "=r"(b_atom[29])
                    : "r"(b_addr_h_30)
                    : "memory");
                int b_cb_h_31 = b_col + 96;
                unsigned int b_addr_h_32 = smem_q_addr + q_stage_m * 4096 + (unsigned int)(b_row_h_24 * 128) + (unsigned int)(b_cb_h_31 ^ (b_row_h_24 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom[30]), "=r"(b_atom[31])
                    : "r"(b_addr_h_32)
                    : "memory");
                int n_tok_m = 1;
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
            unsigned int b_atom_1[32];
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
                int tok_1 = q_idx_1 + atom_in_q_1;
                int ctx_last_1 = context_lens[q_idx_1 + 1 - 1];
                int num_kv_1 = (ctx_last_1 + 64 - 1) / 64;
                int lo_1 = ((q_atom_1 == start_q_1) ? start_kv_1 : 0);
                int hi_1 = ((q_atom_1 == end_q_1) ? end_kv_1 : num_kv_1);
                {
                    mbarrier_wait(q_full_addr + (q_stage_m_1) * 8, q_phase_m_1);
                }
                int b_row_h_1 = b_row_lane_1;
                int b_cb_h_1 = b_col_1;
                unsigned int b_addr_h_2 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_1 * 128) + (unsigned int)(b_cb_h_1 ^ (b_row_h_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[0]), "=r"(b_atom_1[1])
                    : "r"(b_addr_h_2)
                    : "memory");
                int b_cb_h_0_1 = b_col_1 + 32;
                unsigned int b_addr_h_1_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_1 * 128) + (unsigned int)(b_cb_h_0_1 ^ (b_row_h_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[2]), "=r"(b_atom_1[3])
                    : "r"(b_addr_h_1_1)
                    : "memory");
                int b_cb_h_2_1 = b_col_1 + 64;
                unsigned int b_addr_h_3_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_1 * 128) + (unsigned int)(b_cb_h_2_1 ^ (b_row_h_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[4]), "=r"(b_atom_1[5])
                    : "r"(b_addr_h_3_1)
                    : "memory");
                int b_cb_h_4_1 = b_col_1 + 96;
                unsigned int b_addr_h_5_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_1 * 128) + (unsigned int)(b_cb_h_4_1 ^ (b_row_h_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[6]), "=r"(b_atom_1[7])
                    : "r"(b_addr_h_5_1)
                    : "memory");
                int b_row_h_6_1 = 8 + b_row_lane_1;
                int b_cb_h_7_1 = b_col_1;
                unsigned int b_addr_h_8_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_6_1 * 128) + (unsigned int)(b_cb_h_7_1 ^ (b_row_h_6_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[8]), "=r"(b_atom_1[9])
                    : "r"(b_addr_h_8_1)
                    : "memory");
                int b_cb_h_9_1 = b_col_1 + 32;
                unsigned int b_addr_h_10_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_6_1 * 128) + (unsigned int)(b_cb_h_9_1 ^ (b_row_h_6_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[10]), "=r"(b_atom_1[11])
                    : "r"(b_addr_h_10_1)
                    : "memory");
                int b_cb_h_11_1 = b_col_1 + 64;
                unsigned int b_addr_h_12_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_6_1 * 128) + (unsigned int)(b_cb_h_11_1 ^ (b_row_h_6_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[12]), "=r"(b_atom_1[13])
                    : "r"(b_addr_h_12_1)
                    : "memory");
                int b_cb_h_13_1 = b_col_1 + 96;
                unsigned int b_addr_h_14_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_6_1 * 128) + (unsigned int)(b_cb_h_13_1 ^ (b_row_h_6_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[14]), "=r"(b_atom_1[15])
                    : "r"(b_addr_h_14_1)
                    : "memory");
                int b_row_h_15_1 = 16 + b_row_lane_1;
                int b_cb_h_16_1 = b_col_1;
                unsigned int b_addr_h_17_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_15_1 * 128) + (unsigned int)(b_cb_h_16_1 ^ (b_row_h_15_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[16]), "=r"(b_atom_1[17])
                    : "r"(b_addr_h_17_1)
                    : "memory");
                int b_cb_h_18_1 = b_col_1 + 32;
                unsigned int b_addr_h_19_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_15_1 * 128) + (unsigned int)(b_cb_h_18_1 ^ (b_row_h_15_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[18]), "=r"(b_atom_1[19])
                    : "r"(b_addr_h_19_1)
                    : "memory");
                int b_cb_h_20_1 = b_col_1 + 64;
                unsigned int b_addr_h_21_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_15_1 * 128) + (unsigned int)(b_cb_h_20_1 ^ (b_row_h_15_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[20]), "=r"(b_atom_1[21])
                    : "r"(b_addr_h_21_1)
                    : "memory");
                int b_cb_h_22_1 = b_col_1 + 96;
                unsigned int b_addr_h_23_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_15_1 * 128) + (unsigned int)(b_cb_h_22_1 ^ (b_row_h_15_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[22]), "=r"(b_atom_1[23])
                    : "r"(b_addr_h_23_1)
                    : "memory");
                int b_row_h_24_1 = 24 + b_row_lane_1;
                int b_cb_h_25_1 = b_col_1;
                unsigned int b_addr_h_26_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_24_1 * 128) + (unsigned int)(b_cb_h_25_1 ^ (b_row_h_24_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[24]), "=r"(b_atom_1[25])
                    : "r"(b_addr_h_26_1)
                    : "memory");
                int b_cb_h_27_1 = b_col_1 + 32;
                unsigned int b_addr_h_28_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_24_1 * 128) + (unsigned int)(b_cb_h_27_1 ^ (b_row_h_24_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[26]), "=r"(b_atom_1[27])
                    : "r"(b_addr_h_28_1)
                    : "memory");
                int b_cb_h_29_1 = b_col_1 + 64;
                unsigned int b_addr_h_30_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_24_1 * 128) + (unsigned int)(b_cb_h_29_1 ^ (b_row_h_24_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[28]), "=r"(b_atom_1[29])
                    : "r"(b_addr_h_30_1)
                    : "memory");
                int b_cb_h_31_1 = b_col_1 + 96;
                unsigned int b_addr_h_32_1 = smem_q_addr + q_stage_m_1 * 4096 + (unsigned int)(b_row_h_24_1 * 128) + (unsigned int)(b_cb_h_31_1 ^ (b_row_h_24_1 & 7) << 4);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_atom_1[30]), "=r"(b_atom_1[31])
                    : "r"(b_addr_h_32_1)
                    : "memory");
                int n_tok_m_1 = 1;
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
                    float partial1_1 = 0.0f;
                    #pragma unroll
                    for (int nt_1 = 0; nt_1 < 4; nt_1++) {
                        int b_row_1 = nt_1 * 8 + b_row_lane_1;
                        acc_1[0] = 0.0f;
                        acc_1[1] = 0.0f;
                        acc_1[2] = 0.0f;
                        acc_1[3] = 0.0f;
                        int a_cb_1 = a_col_1;
                        unsigned int a_addr_2 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_1 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_2)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + nt_1 * 4 * 2)[0]), "r"((b_atom_1 + nt_1 * 4 * 2)[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        int a_cb_0_1 = a_col_1 + 32;
                        unsigned int a_addr_1_1 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_0_1 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_1_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + (nt_1 * 4 + 1) * 2)[0]), "r"((b_atom_1 + (nt_1 * 4 + 1) * 2)[1]));
                        int a_cb_2_1 = a_col_1 + 64;
                        unsigned int a_addr_3_1 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_2_1 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_3_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + (nt_1 * 4 + 2) * 2)[0]), "r"((b_atom_1 + (nt_1 * 4 + 2) * 2)[1]));
                        int a_cb_4_1 = a_col_1 + 96;
                        unsigned int a_addr_5_1 = smem_kv_g1_addr + kv_stage_m_1 * 8192 + (unsigned int)(a_row_1 * 128) + (unsigned int)(a_cb_4_1 ^ (a_row_1 & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]), "=r"(a_frag_1[3])
                            : "r"(a_addr_5_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                            : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]), "r"((b_atom_1 + (nt_1 * 4 + 3) * 2)[0]), "r"((b_atom_1 + (nt_1 * 4 + 3) * 2)[1]));
                        int w_idx_1 = q_stage_m_1 * 256 + (unsigned int)(nt_1 * 8) + (unsigned int)(tq_1 * 2);
                        float w0_1 = smem_w[w_idx_1];
                        float w1_1 = smem_w[w_idx_1 + 1];
                        float _max_4 = max_noftz(acc_1[0], 0.0f);
                        float r0_1 = _max_4;
                        float _max_5 = max_noftz(acc_1[1], 0.0f);
                        float r1_1 = _max_5;
                        float _max_6 = max_noftz(acc_1[2], 0.0f);
                        float r2_1 = _max_6;
                        float _max_7 = max_noftz(acc_1[3], 0.0f);
                        float r3_1 = _max_7;
                        partial0_1 += r0_1 * w0_1 + r1_1 * w1_1;
                        partial1_1 += r2_1 * w0_1 + r3_1 * w1_1;
                    }
                    float v0_1 = partial0_1 * scale0_1;
                    float v1_1 = partial1_1 * scale1_1;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, v0_1, 1);
                    v0_1 = v0_1 + _shfl_xor_4;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 1);
                    v1_1 = v1_1 + _shfl_xor_5;
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, v0_1, 2);
                    v0_1 = v0_1 + _shfl_xor_6;
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, v1_1, 2);
                    v1_1 = v1_1 + _shfl_xor_7;
                    int out0_1 = kv_offset_1 + quad_1;
                    *(reinterpret_cast<float*>(Logits + out0_1) + (0)) = v0_1;
                    *(reinterpret_cast<float*>(Logits + (out0_1 + 8)) + (0)) = v1_1;
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
                int tok_2 = q_idx_2 + atom_in_q_2;
                int ctx_last_2 = context_lens[q_idx_2 + 1 - 1];
                int num_kv_2 = (ctx_last_2 + 64 - 1) / 64;
                int lo_2 = ((q_atom_2 == start_q_2) ? start_kv_2 : 0);
                int hi_2 = ((q_atom_2 == end_q_2) ? end_kv_2 : num_kv_2);
                {
                    mbarrier_wait(q_empty_addr + (q_stage) * 8, q_phase);
                }
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + (q_stage) * 8, 4224);
                    tma_2d_gmem2smem(smem_q_addr + q_stage * 4096, (&Q), 0, tok_2 * 32, q_full_addr + (q_stage) * 8);
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
                int tok_3 = q_idx_3 + atom_in_q_3;
                int ctx_last_3 = context_lens[q_idx_3 + 1 - 1];
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
