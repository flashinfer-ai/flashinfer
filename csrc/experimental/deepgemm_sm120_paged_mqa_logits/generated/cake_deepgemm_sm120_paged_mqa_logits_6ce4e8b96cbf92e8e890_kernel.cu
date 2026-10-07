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

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int int32_t;
typedef short int int16_t;
struct __align__(64) CakeTensorMap64 {
  uint64_t opaque[16];
};
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) {
  uint64_t opaque[16];
} CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;" : "=r"(result) : "r"(x));
  return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_PREFIX_OFF 0
#define SMEM_PREFIX_STAGE_BYTES 16384
#define SMEM_PREFIX_STRIDE 16384
#define SMEM_TOTAL 16384
#define THREADS 32

#include <math_constants.h>

extern "C" {

__global__
__launch_bounds__(32) void kernel_cake_deepgemm_sm120_paged_mqa_logits_6ce4e8b96cbf92e8e890(
    int* __restrict__ context_lens, int* __restrict__ schedule_meta, int batch_size, int next_n,
    int num_next_n_atoms, int split_kv, int num_sms) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  int* prefix = reinterpret_cast<int*>(smem_raw + 0);
  const int prefix_addr = smem + 0;

  // Kernel post-init ops
  asm volatile("griddepcontrol.wait;" ::: "memory");

  // === Task calls (dependency order) ===
  int lane_0 = lane;
  int carry = 0;
#pragma unroll 1
  for (int base = 0; base < batch_size; base += 32) {
    int qb = base + lane_0;
    int qr = ((qb < batch_size) ? qb : batch_size - 1);
    int ctx = context_lens[qr * next_n + next_n - 1];
    int x0 = ((qb < batch_size) ? (ctx + split_kv - 1) / split_kv : 0);
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, x0, 1, 32);
    int y1 = _shfl_up_0;
    int x1 = ((lane_0 >= 1) ? x0 + y1 : x0);
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, x1, 2, 32);
    int y2 = _shfl_up_1;
    int x2 = ((lane_0 >= 2) ? x1 + y2 : x1);
    int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, x2, 4, 32);
    int y3 = _shfl_up_2;
    int x3 = ((lane_0 >= 4) ? x2 + y3 : x2);
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, x3, 8, 32);
    int y4 = _shfl_up_3;
    int x4 = ((lane_0 >= 8) ? x3 + y4 : x3);
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, x4, 16, 32);
    int y5 = _shfl_up_4;
    int x5 = ((lane_0 >= 16) ? x4 + y5 : x4);
    int _shfl_0 = __shfl_sync(0xFFFFFFFF, x5, 31);
    int chunk_total = _shfl_0;
    int x_final = x5 + carry;
    if (qb < batch_size) {
      prefix[qb] = x_final;
    }
    carry += chunk_total;
  }
  __syncwarp();
  int total_segs = carry;
  int total = total_segs * num_next_n_atoms;
  int sentinel = batch_size * num_next_n_atoms;
  if (total == 0) {
#pragma unroll 1
    for (int sm0 = lane_0; sm0 < num_sms + 1; sm0 += 32) {
      *(reinterpret_cast<int*>(schedule_meta + (sm0 * 2)) + (0)) = sentinel;
      *(reinterpret_cast<int*>(schedule_meta + (sm0 * 2 + 1)) + (0)) = 0;
    }
  } else {
    int qd = total / num_sms;
    int rd = total % num_sms;
    int pivot = num_sms - rd;
#pragma unroll 1
    for (int sm = lane_0; sm < num_sms; sm += 32) {
      int extra = ((pivot < sm) ? sm - pivot : 0);
      int seg_starts = sm * qd + extra;
      int lo = 0;
      int hi = batch_size;
#pragma unroll
      for (int _ = 0; _ < 13; _++) {
        int mid = (lo + hi) / 2;
        int midr = ((mid < batch_size) ? mid : batch_size - 1);
        int le = ((seg_starts >= prefix[midr] * num_next_n_atoms) ? 1 : 0);
        int inb = ((mid < batch_size) ? 1 : 0);
        int go = le * inb;
        lo = ((go != 0) ? mid + 1 : lo);
        hi = ((go == 0) ? mid : hi);
      }
      int q_idx = ((lo < batch_size) ? lo : batch_size - 1);
      int prev_cum = ((q_idx > 0) ? prefix[q_idx - 1] : 0);
      int cur_cum = prefix[q_idx];
      int offset_in_q = seg_starts - prev_cum * num_next_n_atoms;
      int num_segs_q = cur_cum - prev_cum;
      int atom_idx = ((num_segs_q > 0) ? offset_in_q / num_segs_q : 0);
      int kv_split_idx = ((num_segs_q > 0) ? offset_in_q % num_segs_q : 0);
      int q_atom_idx = q_idx * num_next_n_atoms + atom_idx;
      *(reinterpret_cast<int*>(schedule_meta + (sm * 2)) + (0)) = q_atom_idx;
      *(reinterpret_cast<int*>(schedule_meta + (sm * 2 + 1)) + (0)) = kv_split_idx;
    }
    if (lane_0 == 0) {
      *(reinterpret_cast<int*>(schedule_meta + (num_sms * 2)) + (0)) = sentinel;
      *(reinterpret_cast<int*>(schedule_meta + (num_sms * 2 + 1)) + (0)) = 0;
    }
  }
}

}  // extern "C"
