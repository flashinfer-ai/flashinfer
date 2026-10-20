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
#define SMEM_PREFIX_OFF 0
#define SMEM_PREFIX_STAGE_BYTES 2048
#define SMEM_PREFIX_STRIDE 2048
#define SMEM_WARP_SUMS_OFF 2048
#define SMEM_WARP_SUMS_STAGE_BYTES 64
#define SMEM_WARP_SUMS_STRIDE 64
#define SMEM_TOTAL 2176
#define THREADS 512

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_deepgemm_dense_mqa_787fd338dfafacec4277(int* __restrict__ context_lens, int* __restrict__ schedule_meta, int batch_size, int next_n, int num_next_n_atoms, int split_kv, int num_sms, int is_context_lens_2d)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int* prefix = reinterpret_cast<int*>(smem_raw + SMEM_PREFIX_OFF);
    const int prefix_addr = smem + SMEM_PREFIX_OFF;
    int* warp_sums = reinterpret_cast<int*>(smem_raw + SMEM_WARP_SUMS_OFF);
    const int warp_sums_addr = smem + SMEM_WARP_SUMS_OFF;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // === Task calls (dependency order) ===
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    int tid_0 = tid;
    int lane_1 = lane;
    int warp_2 = warp;
    int nseg = 0;
    if (tid_0 < batch_size) {
        int lens_idx = ((is_context_lens_2d != 0) ? tid_0 * next_n + next_n - 1 : tid_0);
        int ctx = context_lens[lens_idx];
        nseg = (ctx + split_kv - 1) / split_kv;
    }
    int thread_total = nseg;
    int lane_sum = thread_total;
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 1, 32);
    int up = _shfl_up_0;
    if (lane_1 >= 1) {
        lane_sum += up;
    }
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 2, 32);
    int up_3 = _shfl_up_1;
    if (lane_1 >= 2) {
        lane_sum += up_3;
    }
    int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 4, 32);
    int up_4 = _shfl_up_2;
    if (lane_1 >= 4) {
        lane_sum += up_4;
    }
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 8, 32);
    int up_5 = _shfl_up_3;
    if (lane_1 >= 8) {
        lane_sum += up_5;
    }
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 16, 32);
    int up_6 = _shfl_up_4;
    if (lane_1 >= 16) {
        lane_sum += up_6;
    }
    if (lane_1 == 31) {
        warp_sums[warp_2] = lane_sum;
    }
    __syncthreads();
    int warp_total = ((lane_1 < 16) ? warp_sums[lane_1] : 0);
    int warp_sum = warp_total;
    int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 1, 32);
    int up2 = _shfl_up_5;
    if (lane_1 >= 1) {
        warp_sum += up2;
    }
    int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 2, 32);
    int up2_7 = _shfl_up_6;
    if (lane_1 >= 2) {
        warp_sum += up2_7;
    }
    int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 4, 32);
    int up2_8 = _shfl_up_7;
    if (lane_1 >= 4) {
        warp_sum += up2_8;
    }
    int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 8, 32);
    int up2_9 = _shfl_up_8;
    if (lane_1 >= 8) {
        warp_sum += up2_9;
    }
    int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 16, 32);
    int up2_10 = _shfl_up_9;
    if (lane_1 >= 16) {
        warp_sum += up2_10;
    }
    int _shfl_0 = __shfl_sync(0xFFFFFFFF, warp_sum, 15);
    int total_segs = _shfl_0;
    int _shfl_1 = __shfl_sync(0xFFFFFFFF, warp_sum - warp_total, warp_2);
    int preceding = _shfl_1;
    if (tid_0 < batch_size) {
        prefix[tid_0] = lane_sum + preceding;
    }
    __syncthreads();
    int total = total_segs * num_next_n_atoms;
    int qd = total / num_sms;
    int rd = total % num_sms;
    #pragma unroll 1
    for (int sm = tid_0; sm < num_sms + 1; sm += 512) {
        int min_sm_rd = ((rd > sm) ? sm : rd);
        int seg_starts = sm * qd + min_sm_rd;
        int lo = 0;
        int hi = batch_size;
        #pragma unroll
        for (int _ = 0; _ < 10; _++) {
            int mid = (lo + hi) / 2;
            int midr = ((mid < batch_size) ? mid : batch_size - 1);
            int le = ((seg_starts >= prefix[midr] * num_next_n_atoms) ? 1 : 0);
            int inb = ((mid < batch_size) ? 1 : 0);
            int go = le * inb;
            lo = ((go != 0) ? mid + 1 : lo);
            hi = ((go == 0) ? mid : hi);
        }
        int q_idx_sm = lo;
        int prev_cum = ((q_idx_sm > 0) ? prefix[q_idx_sm - 1] : 0);
        int cur_cum = ((q_idx_sm < batch_size) ? prefix[q_idx_sm] : total_segs);
        int offset_in_q = seg_starts - prev_cum * num_next_n_atoms;
        int num_segs_q = cur_cum - prev_cum;
        int atom_idx = ((num_segs_q > 0) ? offset_in_q / num_segs_q : 0);
        int kv_split_idx = ((num_segs_q > 0) ? offset_in_q % num_segs_q : 0);
        int q_atom_idx = q_idx_sm * num_next_n_atoms + atom_idx;
        *(reinterpret_cast<int*>(schedule_meta + (sm * 2)) + (0)) = q_atom_idx;
        *(reinterpret_cast<int*>(schedule_meta + (sm * 2 + 1)) + (0)) = kv_split_idx;
    }
}

} // extern "C"
