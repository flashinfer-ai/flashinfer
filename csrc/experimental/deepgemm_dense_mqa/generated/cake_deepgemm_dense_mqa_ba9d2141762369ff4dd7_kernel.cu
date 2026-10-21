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
#include "cake_deepgemm_dense_mqa_device_common.cuh"
// The common preamble's fallback helper this module's helper library does not provide; the shared
// header carries another member's library definition of the name (kept translation-unit-local).
__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}


#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 192

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_deepgemm_dense_mqa_ba9d2141762369ff4dd7(int* __restrict__ context_lens, int* __restrict__ schedule_meta, int batch_size, int next_n, int num_next_n_atoms, int split_kv, int num_sms, int is_context_lens_2d)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // === Task calls (dependency order) ===
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    int lane_0 = lane;
    int nseg = 0;
    if (lane_0 < batch_size) {
        int lens_idx = ((is_context_lens_2d != 0) ? lane_0 * next_n + next_n - 1 : lane_0);
        int ctx = context_lens[lens_idx];
        nseg = (ctx + split_kv - 1) / split_kv;
    }
    int incl = nseg;
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, incl, 1, 32);
    int up = _shfl_up_0;
    if (lane_0 >= 1) {
        incl += up;
    }
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, incl, 2, 32);
    int up_1 = _shfl_up_1;
    if (lane_0 >= 2) {
        incl += up_1;
    }
    int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, incl, 4, 32);
    int up_2 = _shfl_up_2;
    if (lane_0 >= 4) {
        incl += up_2;
    }
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, incl, 8, 32);
    int up_3 = _shfl_up_3;
    if (lane_0 >= 8) {
        incl += up_3;
    }
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, incl, 16, 32);
    int up_4 = _shfl_up_4;
    if (lane_0 >= 16) {
        incl += up_4;
    }
    int _shfl_0 = __shfl_sync(0xFFFFFFFF, incl, 31);
    int total_segs = _shfl_0;
    int sm = tid;
    int sm_c = ((sm <= num_sms) ? sm : num_sms);
    int total = total_segs * num_next_n_atoms;
    int qd = total / num_sms;
    int rd = total % num_sms;
    int min_sm_rd = ((sm_c < rd) ? sm_c : rd);
    int seg_starts = sm_c * qd + min_sm_rd;
    int lo = 0;
    int hi = batch_size;
    #pragma unroll
    for (int _ = 0; _ < 6; _++) {
        int mid = (lo + hi) / 2;
        int midr = ((mid < batch_size) ? mid : batch_size - 1);
        int _shfl_1 = __shfl_sync(0xFFFFFFFF, incl, midr);
        int pm = _shfl_1;
        int le = ((seg_starts >= pm * num_next_n_atoms) ? 1 : 0);
        int inb = ((mid < batch_size) ? 1 : 0);
        int go = le * inb;
        lo = ((go != 0) ? mid + 1 : lo);
        hi = ((go == 0) ? mid : hi);
    }
    int q_idx_sm = lo;
    int prev_idx = ((q_idx_sm > 0) ? q_idx_sm - 1 : 0);
    int _shfl_2 = __shfl_sync(0xFFFFFFFF, incl, prev_idx);
    int prev_raw = _shfl_2;
    int prev_cum = ((q_idx_sm > 0) ? prev_raw : 0);
    int cur_idx = ((q_idx_sm < batch_size) ? q_idx_sm : batch_size - 1);
    int _shfl_3 = __shfl_sync(0xFFFFFFFF, incl, cur_idx);
    int cur_raw = _shfl_3;
    int cur_cum = ((q_idx_sm < batch_size) ? cur_raw : total_segs);
    int offset_in_q = seg_starts - prev_cum * num_next_n_atoms;
    int num_segs_q = cur_cum - prev_cum;
    int atom_idx = ((num_segs_q > 0) ? offset_in_q / num_segs_q : 0);
    int kv_split_idx = ((num_segs_q > 0) ? offset_in_q % num_segs_q : 0);
    int q_atom_idx = q_idx_sm * num_next_n_atoms + atom_idx;
    if (sm <= num_sms) {
        *(reinterpret_cast<int*>(schedule_meta + (sm * 2)) + (0)) = q_atom_idx;
        *(reinterpret_cast<int*>(schedule_meta + (sm * 2 + 1)) + (0)) = kv_split_idx;
    }
}

} // extern "C"
