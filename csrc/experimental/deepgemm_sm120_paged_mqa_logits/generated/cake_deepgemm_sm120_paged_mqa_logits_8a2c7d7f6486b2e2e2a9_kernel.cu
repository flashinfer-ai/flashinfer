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
#define NUM_Q_PIPE_STAGES 1
#define NUM_KV_PIPE_STAGES 3
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 32768
#define SMEM_SMEM_Q_STRIDE 32768
#define SMEM_SMEM_W_OFF 33792
#define SMEM_SMEM_W_STAGE_BYTES 1024
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_SMEM_KV_G0_OFF 34816
#define SMEM_SMEM_KV_G0_STAGE_BYTES 8192
#define SMEM_SMEM_KV_G0_STRIDE 8192
#define SMEM_SMEM_KV_G1_OFF 59392
#define SMEM_SMEM_KV_G1_STAGE_BYTES 8192
#define SMEM_SMEM_KV_G1_STRIDE 8192
#define SMEM_SMEM_SC_G0_OFF 83968
#define SMEM_SMEM_SC_G0_STAGE_BYTES 256
#define SMEM_SMEM_SC_G0_STRIDE 1024
#define SMEM_SMEM_SC_G1_OFF 87040
#define SMEM_SMEM_SC_G1_STAGE_BYTES 256
#define SMEM_SMEM_SC_G1_STRIDE 1024
#define SMEM_TOTAL 90112
#define THREADS 384
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__
__launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void kernel_cake_deepgemm_sm120_paged_mqa_logits_8a2c7d7f6486b2e2e2a9(
    const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV,
    const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights,
    float* __restrict__ Logits, int* __restrict__ context_lens, int* __restrict__ block_table,
    int* __restrict__ schedule_meta, int logits_stride, int block_table_stride) {
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

  // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 14 barriers)
  // Mbarriers at smem_raw[0..112)

  if (warp == 0) {
    uint32_t leader = elect_sync();
    if (leader) {
      // q_full: 1 barriers, init_count=1
      mbarrier_init(smem + 0, 1);
      // q_empty: 1 barriers, init_count=256
      mbarrier_init(smem + 8, 256);
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

  // ---- Ordered hardware-WG register redistribution ----
  // Dec phase frees registers before any WG attempts inc.
  if (warp >= 8 && warp <= 11) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
  }

  // ---- Role: math_g0 ----
  if (warp <= 3) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
    {  // math_g0_main
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
        mbarrier_wait(q_full_addr + (q_stage_m) * 8, q_phase_m);
        int n_tok_m = 4;
#pragma unroll 1
        for (int kv_idx = lo; kv_idx < hi; kv_idx += 2) {
          mbarrier_wait(kv_full_g0_addr + (kv_stage_m) * 8, kv_phase_m);
          int sc_idx = kv_stage_m * 256 + (unsigned int)(w_in_g * 16) + (unsigned int)quad;
          float scale0 = smem_sc_g0[sc_idx];
          float scale1 = smem_sc_g0[sc_idx + 8];
          int kv_offset = tok * logits_stride + kv_idx * 64 + w_in_g * 16;
          float partial0 = 0.0f;
          float partial1 = 0.0f;
#pragma unroll
          for (int nt = 0; nt < 8; nt++) {
            int b_row = nt * 8 + b_row_lane;
            acc[0] = 0.0f;
            acc[1] = 0.0f;
            acc[2] = 0.0f;
            acc[3] = 0.0f;
            int a_cb = a_col;
            unsigned int a_addr = smem_kv_g0_addr + kv_stage_m * 8192 +
                                  (unsigned int)(a_row * 128) +
                                  (unsigned int)(a_cb ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr)
                         : "memory");
            int b_cb = b_col;
            unsigned int b_addr = smem_q_addr + q_stage_m * 32768 + (unsigned int)(b_row * 128) +
                                  (unsigned int)(b_cb ^ (b_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            int a_cb_0 = a_col + 32;
            unsigned int a_addr_1 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                    (unsigned int)(a_row * 128) +
                                    (unsigned int)(a_cb_0 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_1)
                         : "memory");
            int b_cb_2 = b_col + 32;
            unsigned int b_addr_3 = smem_q_addr + q_stage_m * 32768 + (unsigned int)(b_row * 128) +
                                    (unsigned int)(b_cb_2 ^ (b_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_3)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]));
            int a_cb_4 = a_col + 64;
            unsigned int a_addr_5 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                    (unsigned int)(a_row * 128) +
                                    (unsigned int)(a_cb_4 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_5)
                         : "memory");
            int b_cb_6 = b_col + 64;
            unsigned int b_addr_7 = smem_q_addr + q_stage_m * 32768 + (unsigned int)(b_row * 128) +
                                    (unsigned int)(b_cb_6 ^ (b_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_7)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]));
            int a_cb_8 = a_col + 96;
            unsigned int a_addr_9 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                    (unsigned int)(a_row * 128) +
                                    (unsigned int)(a_cb_8 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_9)
                         : "memory");
            int b_cb_10 = b_col + 96;
            unsigned int b_addr_11 = smem_q_addr + q_stage_m * 32768 + (unsigned int)(b_row * 128) +
                                     (unsigned int)(b_cb_10 ^ (b_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_11)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]));
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
          for (int nt_1 = 0; nt_1 < 8; nt_1++) {
            int b_row_1 = 64 + nt_1 * 8 + b_row_lane;
            acc[0] = 0.0f;
            acc[1] = 0.0f;
            acc[2] = 0.0f;
            acc[3] = 0.0f;
            int a_cb_1 = a_col;
            unsigned int a_addr_2 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                    (unsigned int)(a_row * 128) +
                                    (unsigned int)(a_cb_1 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_2)
                         : "memory");
            int b_cb_1 = b_col;
            unsigned int b_addr_1 = smem_q_addr + q_stage_m * 32768 +
                                    (unsigned int)(b_row_1 * 128) +
                                    (unsigned int)(b_cb_1 ^ (b_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_1)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            int a_cb_0_1 = a_col + 32;
            unsigned int a_addr_1_1 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                      (unsigned int)(a_row * 128) +
                                      (unsigned int)(a_cb_0_1 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_1_1)
                         : "memory");
            int b_cb_2_1 = b_col + 32;
            unsigned int b_addr_3_1 = smem_q_addr + q_stage_m * 32768 +
                                      (unsigned int)(b_row_1 * 128) +
                                      (unsigned int)(b_cb_2_1 ^ (b_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_3_1)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]));
            int a_cb_4_1 = a_col + 64;
            unsigned int a_addr_5_1 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                      (unsigned int)(a_row * 128) +
                                      (unsigned int)(a_cb_4_1 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_5_1)
                         : "memory");
            int b_cb_6_1 = b_col + 64;
            unsigned int b_addr_7_1 = smem_q_addr + q_stage_m * 32768 +
                                      (unsigned int)(b_row_1 * 128) +
                                      (unsigned int)(b_cb_6_1 ^ (b_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_7_1)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]));
            int a_cb_8_1 = a_col + 96;
            unsigned int a_addr_9_1 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                      (unsigned int)(a_row * 128) +
                                      (unsigned int)(a_cb_8_1 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_9_1)
                         : "memory");
            int b_cb_10_1 = b_col + 96;
            unsigned int b_addr_11_1 = smem_q_addr + q_stage_m * 32768 +
                                       (unsigned int)(b_row_1 * 128) +
                                       (unsigned int)(b_cb_10_1 ^ (b_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_11_1)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]));
            int w_idx_1 = q_stage_m * 256 + 64 + (unsigned int)(nt_1 * 8) + (unsigned int)(tq * 2);
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
          for (int nt_2 = 0; nt_2 < 8; nt_2++) {
            int b_row_2 = 128 + nt_2 * 8 + b_row_lane;
            acc[0] = 0.0f;
            acc[1] = 0.0f;
            acc[2] = 0.0f;
            acc[3] = 0.0f;
            int a_cb_2 = a_col;
            unsigned int a_addr_3 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                    (unsigned int)(a_row * 128) +
                                    (unsigned int)(a_cb_2 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_3)
                         : "memory");
            int b_cb_3 = b_col;
            unsigned int b_addr_2 = smem_q_addr + q_stage_m * 32768 +
                                    (unsigned int)(b_row_2 * 128) +
                                    (unsigned int)(b_cb_3 ^ (b_row_2 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_2)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            int a_cb_0_2 = a_col + 32;
            unsigned int a_addr_1_2 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                      (unsigned int)(a_row * 128) +
                                      (unsigned int)(a_cb_0_2 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_1_2)
                         : "memory");
            int b_cb_2_2 = b_col + 32;
            unsigned int b_addr_3_2 = smem_q_addr + q_stage_m * 32768 +
                                      (unsigned int)(b_row_2 * 128) +
                                      (unsigned int)(b_cb_2_2 ^ (b_row_2 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_3_2)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]));
            int a_cb_4_2 = a_col + 64;
            unsigned int a_addr_5_2 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                      (unsigned int)(a_row * 128) +
                                      (unsigned int)(a_cb_4_2 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_5_2)
                         : "memory");
            int b_cb_6_2 = b_col + 64;
            unsigned int b_addr_7_2 = smem_q_addr + q_stage_m * 32768 +
                                      (unsigned int)(b_row_2 * 128) +
                                      (unsigned int)(b_cb_6_2 ^ (b_row_2 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_7_2)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]));
            int a_cb_8_2 = a_col + 96;
            unsigned int a_addr_9_2 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                      (unsigned int)(a_row * 128) +
                                      (unsigned int)(a_cb_8_2 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_9_2)
                         : "memory");
            int b_cb_10_2 = b_col + 96;
            unsigned int b_addr_11_2 = smem_q_addr + q_stage_m * 32768 +
                                       (unsigned int)(b_row_2 * 128) +
                                       (unsigned int)(b_cb_10_2 ^ (b_row_2 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_11_2)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]));
            int w_idx_2 = q_stage_m * 256 + 128 + (unsigned int)(nt_2 * 8) + (unsigned int)(tq * 2);
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
          for (int nt_3 = 0; nt_3 < 8; nt_3++) {
            int b_row_3 = 192 + nt_3 * 8 + b_row_lane;
            acc[0] = 0.0f;
            acc[1] = 0.0f;
            acc[2] = 0.0f;
            acc[3] = 0.0f;
            int a_cb_3 = a_col;
            unsigned int a_addr_4 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                    (unsigned int)(a_row * 128) +
                                    (unsigned int)(a_cb_3 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_4)
                         : "memory");
            int b_cb_4 = b_col;
            unsigned int b_addr_4 = smem_q_addr + q_stage_m * 32768 +
                                    (unsigned int)(b_row_3 * 128) +
                                    (unsigned int)(b_cb_4 ^ (b_row_3 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_4)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            int a_cb_0_3 = a_col + 32;
            unsigned int a_addr_1_3 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                      (unsigned int)(a_row * 128) +
                                      (unsigned int)(a_cb_0_3 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_1_3)
                         : "memory");
            int b_cb_2_3 = b_col + 32;
            unsigned int b_addr_3_3 = smem_q_addr + q_stage_m * 32768 +
                                      (unsigned int)(b_row_3 * 128) +
                                      (unsigned int)(b_cb_2_3 ^ (b_row_3 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_3_3)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]));
            int a_cb_4_3 = a_col + 64;
            unsigned int a_addr_5_3 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                      (unsigned int)(a_row * 128) +
                                      (unsigned int)(a_cb_4_3 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_5_3)
                         : "memory");
            int b_cb_6_3 = b_col + 64;
            unsigned int b_addr_7_3 = smem_q_addr + q_stage_m * 32768 +
                                      (unsigned int)(b_row_3 * 128) +
                                      (unsigned int)(b_cb_6_3 ^ (b_row_3 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_7_3)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]));
            int a_cb_8_3 = a_col + 96;
            unsigned int a_addr_9_3 = smem_kv_g0_addr + kv_stage_m * 8192 +
                                      (unsigned int)(a_row * 128) +
                                      (unsigned int)(a_cb_8_3 ^ (a_row & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(a_addr_9_3)
                         : "memory");
            int b_cb_10_3 = b_col + 96;
            unsigned int b_addr_11_3 = smem_q_addr + q_stage_m * 32768 +
                                       (unsigned int)(b_row_3 * 128) +
                                       (unsigned int)(b_cb_10_3 ^ (b_row_3 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag[0]), "=r"(b_frag[1])
                         : "r"(b_addr_11_3)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]),
                  "r"(b_frag[1]));
            int w_idx_3 = q_stage_m * 256 + 192 + (unsigned int)(nt_3 * 8) + (unsigned int)(tq * 2);
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
          if (kv_stage_m == 3) {
            kv_stage_m = 0;
            kv_phase_m ^= 1;
          }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(q_empty_addr + (q_stage_m) * 8);
        q_phase_m ^= 1;
      }
    }
  }
  // ---- Role: math_g1 ----
  if (warp >= 4 && warp <= 7) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
    {  // math_g1_main
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
        mbarrier_wait(q_full_addr + (q_stage_m_1) * 8, q_phase_m_1);
        int n_tok_m_1 = 4;
#pragma unroll 1
        for (int kv_idx_1 = lo_1; kv_idx_1 < hi_1; kv_idx_1 += 2) {
          mbarrier_wait(kv_full_g1_addr + (kv_stage_m_1) * 8, kv_phase_m_1);
          int sc_idx_1 = kv_stage_m_1 * 256 + (unsigned int)(w_in_g_1 * 16) + (unsigned int)quad_1;
          float scale0_1 = smem_sc_g1[sc_idx_1];
          float scale1_1 = smem_sc_g1[sc_idx_1 + 8];
          int kv_offset_1 = tok_1 * logits_stride + (kv_idx_1 + 1) * 64 + w_in_g_1 * 16;
          float partial0_1 = 0.0f;
          float partial1_2 = 0.0f;
#pragma unroll
          for (int nt_4 = 0; nt_4 < 8; nt_4++) {
            int b_row_4 = nt_4 * 8 + b_row_lane_1;
            acc_1[0] = 0.0f;
            acc_1[1] = 0.0f;
            acc_1[2] = 0.0f;
            acc_1[3] = 0.0f;
            int a_cb_5 = a_col_1;
            unsigned int a_addr_6 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                    (unsigned int)(a_row_1 * 128) +
                                    (unsigned int)(a_cb_5 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_6)
                         : "memory");
            int b_cb_5 = b_col_1;
            unsigned int b_addr_5 = smem_q_addr + q_stage_m_1 * 32768 +
                                    (unsigned int)(b_row_4 * 128) +
                                    (unsigned int)(b_cb_5 ^ (b_row_4 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_5)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            int a_cb_0_4 = a_col_1 + 32;
            unsigned int a_addr_1_4 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                      (unsigned int)(a_row_1 * 128) +
                                      (unsigned int)(a_cb_0_4 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_1_4)
                         : "memory");
            int b_cb_2_4 = b_col_1 + 32;
            unsigned int b_addr_3_4 = smem_q_addr + q_stage_m_1 * 32768 +
                                      (unsigned int)(b_row_4 * 128) +
                                      (unsigned int)(b_cb_2_4 ^ (b_row_4 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_3_4)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]));
            int a_cb_4_4 = a_col_1 + 64;
            unsigned int a_addr_5_4 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                      (unsigned int)(a_row_1 * 128) +
                                      (unsigned int)(a_cb_4_4 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_5_4)
                         : "memory");
            int b_cb_6_4 = b_col_1 + 64;
            unsigned int b_addr_7_4 = smem_q_addr + q_stage_m_1 * 32768 +
                                      (unsigned int)(b_row_4 * 128) +
                                      (unsigned int)(b_cb_6_4 ^ (b_row_4 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_7_4)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]));
            int a_cb_8_4 = a_col_1 + 96;
            unsigned int a_addr_9_4 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                      (unsigned int)(a_row_1 * 128) +
                                      (unsigned int)(a_cb_8_4 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_9_4)
                         : "memory");
            int b_cb_10_4 = b_col_1 + 96;
            unsigned int b_addr_11_4 = smem_q_addr + q_stage_m_1 * 32768 +
                                       (unsigned int)(b_row_4 * 128) +
                                       (unsigned int)(b_cb_10_4 ^ (b_row_4 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_11_4)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]));
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
          for (int nt_5 = 0; nt_5 < 8; nt_5++) {
            int b_row_5 = 64 + nt_5 * 8 + b_row_lane_1;
            acc_1[0] = 0.0f;
            acc_1[1] = 0.0f;
            acc_1[2] = 0.0f;
            acc_1[3] = 0.0f;
            int a_cb_6 = a_col_1;
            unsigned int a_addr_7 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                    (unsigned int)(a_row_1 * 128) +
                                    (unsigned int)(a_cb_6 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_7)
                         : "memory");
            int b_cb_7 = b_col_1;
            unsigned int b_addr_6 = smem_q_addr + q_stage_m_1 * 32768 +
                                    (unsigned int)(b_row_5 * 128) +
                                    (unsigned int)(b_cb_7 ^ (b_row_5 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_6)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            int a_cb_0_5 = a_col_1 + 32;
            unsigned int a_addr_1_5 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                      (unsigned int)(a_row_1 * 128) +
                                      (unsigned int)(a_cb_0_5 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_1_5)
                         : "memory");
            int b_cb_2_5 = b_col_1 + 32;
            unsigned int b_addr_3_5 = smem_q_addr + q_stage_m_1 * 32768 +
                                      (unsigned int)(b_row_5 * 128) +
                                      (unsigned int)(b_cb_2_5 ^ (b_row_5 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_3_5)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]));
            int a_cb_4_5 = a_col_1 + 64;
            unsigned int a_addr_5_5 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                      (unsigned int)(a_row_1 * 128) +
                                      (unsigned int)(a_cb_4_5 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_5_5)
                         : "memory");
            int b_cb_6_5 = b_col_1 + 64;
            unsigned int b_addr_7_5 = smem_q_addr + q_stage_m_1 * 32768 +
                                      (unsigned int)(b_row_5 * 128) +
                                      (unsigned int)(b_cb_6_5 ^ (b_row_5 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_7_5)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]));
            int a_cb_8_5 = a_col_1 + 96;
            unsigned int a_addr_9_5 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                      (unsigned int)(a_row_1 * 128) +
                                      (unsigned int)(a_cb_8_5 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_9_5)
                         : "memory");
            int b_cb_10_5 = b_col_1 + 96;
            unsigned int b_addr_11_5 = smem_q_addr + q_stage_m_1 * 32768 +
                                       (unsigned int)(b_row_5 * 128) +
                                       (unsigned int)(b_cb_10_5 ^ (b_row_5 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_11_5)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]));
            int w_idx_5 =
                q_stage_m_1 * 256 + 64 + (unsigned int)(nt_5 * 8) + (unsigned int)(tq_1 * 2);
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
          for (int nt_6 = 0; nt_6 < 8; nt_6++) {
            int b_row_6 = 128 + nt_6 * 8 + b_row_lane_1;
            acc_1[0] = 0.0f;
            acc_1[1] = 0.0f;
            acc_1[2] = 0.0f;
            acc_1[3] = 0.0f;
            int a_cb_7 = a_col_1;
            unsigned int a_addr_8 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                    (unsigned int)(a_row_1 * 128) +
                                    (unsigned int)(a_cb_7 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_8)
                         : "memory");
            int b_cb_8 = b_col_1;
            unsigned int b_addr_8 = smem_q_addr + q_stage_m_1 * 32768 +
                                    (unsigned int)(b_row_6 * 128) +
                                    (unsigned int)(b_cb_8 ^ (b_row_6 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_8)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            int a_cb_0_6 = a_col_1 + 32;
            unsigned int a_addr_1_6 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                      (unsigned int)(a_row_1 * 128) +
                                      (unsigned int)(a_cb_0_6 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_1_6)
                         : "memory");
            int b_cb_2_6 = b_col_1 + 32;
            unsigned int b_addr_3_6 = smem_q_addr + q_stage_m_1 * 32768 +
                                      (unsigned int)(b_row_6 * 128) +
                                      (unsigned int)(b_cb_2_6 ^ (b_row_6 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_3_6)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]));
            int a_cb_4_6 = a_col_1 + 64;
            unsigned int a_addr_5_6 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                      (unsigned int)(a_row_1 * 128) +
                                      (unsigned int)(a_cb_4_6 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_5_6)
                         : "memory");
            int b_cb_6_6 = b_col_1 + 64;
            unsigned int b_addr_7_6 = smem_q_addr + q_stage_m_1 * 32768 +
                                      (unsigned int)(b_row_6 * 128) +
                                      (unsigned int)(b_cb_6_6 ^ (b_row_6 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_7_6)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]));
            int a_cb_8_6 = a_col_1 + 96;
            unsigned int a_addr_9_6 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                      (unsigned int)(a_row_1 * 128) +
                                      (unsigned int)(a_cb_8_6 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_9_6)
                         : "memory");
            int b_cb_10_6 = b_col_1 + 96;
            unsigned int b_addr_11_6 = smem_q_addr + q_stage_m_1 * 32768 +
                                       (unsigned int)(b_row_6 * 128) +
                                       (unsigned int)(b_cb_10_6 ^ (b_row_6 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_11_6)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]));
            int w_idx_6 =
                q_stage_m_1 * 256 + 128 + (unsigned int)(nt_6 * 8) + (unsigned int)(tq_1 * 2);
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
          for (int nt_7 = 0; nt_7 < 8; nt_7++) {
            int b_row_7 = 192 + nt_7 * 8 + b_row_lane_1;
            acc_1[0] = 0.0f;
            acc_1[1] = 0.0f;
            acc_1[2] = 0.0f;
            acc_1[3] = 0.0f;
            int a_cb_9 = a_col_1;
            unsigned int a_addr_10 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                     (unsigned int)(a_row_1 * 128) +
                                     (unsigned int)(a_cb_9 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_10)
                         : "memory");
            int b_cb_9 = b_col_1;
            unsigned int b_addr_9 = smem_q_addr + q_stage_m_1 * 32768 +
                                    (unsigned int)(b_row_7 * 128) +
                                    (unsigned int)(b_cb_9 ^ (b_row_7 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_9)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc_1[0]), "=f"(acc_1[1]), "=f"(acc_1[2]), "=f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            int a_cb_0_7 = a_col_1 + 32;
            unsigned int a_addr_1_7 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                      (unsigned int)(a_row_1 * 128) +
                                      (unsigned int)(a_cb_0_7 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_1_7)
                         : "memory");
            int b_cb_2_7 = b_col_1 + 32;
            unsigned int b_addr_3_7 = smem_q_addr + q_stage_m_1 * 32768 +
                                      (unsigned int)(b_row_7 * 128) +
                                      (unsigned int)(b_cb_2_7 ^ (b_row_7 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_3_7)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]));
            int a_cb_4_7 = a_col_1 + 64;
            unsigned int a_addr_5_7 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                      (unsigned int)(a_row_1 * 128) +
                                      (unsigned int)(a_cb_4_7 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_5_7)
                         : "memory");
            int b_cb_6_7 = b_col_1 + 64;
            unsigned int b_addr_7_7 = smem_q_addr + q_stage_m_1 * 32768 +
                                      (unsigned int)(b_row_7 * 128) +
                                      (unsigned int)(b_cb_6_7 ^ (b_row_7 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_7_7)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]));
            int a_cb_8_7 = a_col_1 + 96;
            unsigned int a_addr_9_7 = smem_kv_g1_addr + kv_stage_m_1 * 8192 +
                                      (unsigned int)(a_row_1 * 128) +
                                      (unsigned int)(a_cb_8_7 ^ (a_row_1 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag_1[0]), "=r"(a_frag_1[1]), "=r"(a_frag_1[2]),
                           "=r"(a_frag_1[3])
                         : "r"(a_addr_9_7)
                         : "memory");
            int b_cb_10_7 = b_col_1 + 96;
            unsigned int b_addr_11_7 = smem_q_addr + q_stage_m_1 * 32768 +
                                       (unsigned int)(b_row_7 * 128) +
                                       (unsigned int)(b_cb_10_7 ^ (b_row_7 & 7) << 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                         : "=r"(b_frag_1[0]), "=r"(b_frag_1[1])
                         : "r"(b_addr_11_7)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3])
                : "r"(a_frag_1[0]), "r"(a_frag_1[1]), "r"(a_frag_1[2]), "r"(a_frag_1[3]),
                  "r"(b_frag_1[0]), "r"(b_frag_1[1]));
            int w_idx_7 =
                q_stage_m_1 * 256 + 192 + (unsigned int)(nt_7 * 8) + (unsigned int)(tq_1 * 2);
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
          if (kv_stage_m_1 == 3) {
            kv_stage_m_1 = 0;
            kv_phase_m_1 ^= 1;
          }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(q_empty_addr + (q_stage_m_1) * 8);
        q_phase_m_1 ^= 1;
      }
    }
  }
  // ---- Role: tma_g0 ----
  if (warp == 8) {
    {  // tma_g0_main
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
        mbarrier_wait(q_empty_addr + (q_stage) * 8, q_phase);
        if (elect_sync()) {
          mbarrier_arrive_expect_tx(q_full_addr + (q_stage) * 8, 33792);
          tma_2d_gmem2smem(smem_q_addr + q_stage * 32768, (&Q), 0, tok_2 * 64,
                           q_full_addr + (q_stage) * 8);
          tma_2d_gmem2smem(smem_w_addr + q_stage * 1024, (&Weights), 0, tok_2,
                           q_full_addr + (q_stage) * 8);
        }
        q_phase ^= 1;
        int bt_row = q_idx_2 * block_table_stride;
#pragma unroll 1
        for (int kv_idx_2 = lo_2; kv_idx_2 < hi_2; kv_idx_2 += 2) {
          int tile = kv_idx_2;
          int kv_block = 0;
          if (tile < num_kv_2) {
            kv_block = block_table[bt_row + tile];
          }
          int page_off = 0;
          mbarrier_wait(kv_empty_g0_addr + (kv_stage) * 8, kv_phase);
          if (elect_sync()) {
            mbarrier_arrive_expect_tx(kv_full_g0_addr + (kv_stage) * 8, 8448);
            tma_3d_gmem2smem(smem_kv_g0_addr + kv_stage * 8192, (&KV), 0, page_off, kv_block,
                             kv_full_g0_addr + (kv_stage) * 8);
            tma_2d_gmem2smem(smem_sc_g0_addr + kv_stage * 1024, (&KV_scales), 2048 + page_off,
                             kv_block, kv_full_g0_addr + (kv_stage) * 8);
          }
          kv_stage += 1;
          if (kv_stage == 3) {
            kv_stage = 0;
            kv_phase ^= 1;
          }
        }
      }
    }
  }
  // ---- Role: tma_g1 ----
  if (warp == 9) {
    {  // tma_g1_main
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
#pragma unroll 1
        for (int kv_idx_3 = lo_3; kv_idx_3 < hi_3; kv_idx_3 += 2) {
          int tile_1 = kv_idx_3 + 1;
          int kv_block_1 = 0;
          if (tile_1 < num_kv_3) {
            kv_block_1 = block_table[bt_row_1 + tile_1];
          }
          int page_off_1 = 0;
          mbarrier_wait(kv_empty_g1_addr + (kv_stage_1) * 8, kv_phase_1);
          if (elect_sync()) {
            mbarrier_arrive_expect_tx(kv_full_g1_addr + (kv_stage_1) * 8, 8448);
            tma_3d_gmem2smem(smem_kv_g1_addr + kv_stage_1 * 8192, (&KV), 0, page_off_1, kv_block_1,
                             kv_full_g1_addr + (kv_stage_1) * 8);
            tma_2d_gmem2smem(smem_sc_g1_addr + kv_stage_1 * 1024, (&KV_scales), 2048 + page_off_1,
                             kv_block_1, kv_full_g1_addr + (kv_stage_1) * 8);
          }
          kv_stage_1 += 1;
          if (kv_stage_1 == 3) {
            kv_stage_1 = 0;
            kv_phase_1 ^= 1;
          }
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

}  // extern "C"
