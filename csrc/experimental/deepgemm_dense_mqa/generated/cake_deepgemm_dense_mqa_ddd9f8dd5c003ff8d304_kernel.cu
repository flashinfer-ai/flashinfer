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
#define SMEM_PREFIX_STAGE_BYTES 768
#define SMEM_PREFIX_STRIDE 768
#define SMEM_TOTAL 768
#define THREADS 192

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_deepgemm_dense_mqa_ddd9f8dd5c003ff8d304(int* __restrict__ context_lens, int* __restrict__ schedule_meta, int batch_size, int next_n, int num_next_n_atoms, int split_kv, int num_sms, int is_context_lens_2d)
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
    int nseg = 0;
    if (tid_0 < batch_size) {
        int lens_idx = ((is_context_lens_2d != 0) ? tid_0 * next_n + next_n - 1 : tid_0);
        int ctx = context_lens[lens_idx];
        nseg = (ctx + split_kv - 1) / split_kv;
    }
    int incl = nseg;
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, incl, 1, 32);
    int up = _shfl_up_0;
    if (lane_1 >= 1) {
        incl += up;
    }
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, incl, 2, 32);
    int up_2 = _shfl_up_1;
    if (lane_1 >= 2) {
        incl += up_2;
    }
    int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, incl, 4, 32);
    int up_3 = _shfl_up_2;
    if (lane_1 >= 4) {
        incl += up_3;
    }
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, incl, 8, 32);
    int up_4 = _shfl_up_3;
    if (lane_1 >= 8) {
        incl += up_4;
    }
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, incl, 16, 32);
    int up_5 = _shfl_up_4;
    if (lane_1 >= 16) {
        incl += up_5;
    }
    prefix[tid_0] = incl;
    __syncthreads();
    int t0 = prefix[31];
    int t1 = prefix[63];
    int t2 = prefix[95];
    int t3 = prefix[127];
    int t4 = prefix[159];
    int t5 = prefix[191];
    int e1 = t0;
    int e2 = e1 + t1;
    int e3 = e2 + t2;
    int e4 = e3 + t3;
    int e5 = e4 + t4;
    int total_segs = e5 + t5;
    int sm = tid_0;
    int sm_c = ((sm <= num_sms) ? sm : num_sms);
    int total = total_segs * num_next_n_atoms;
    int qd = total / num_sms;
    int rd = total % num_sms;
    int min_sm_rd = ((sm_c < rd) ? sm_c : rd);
    int seg_starts = sm_c * qd + min_sm_rd;
    int sent = ((total <= seg_starts) ? 1 : 0);
    int l0 = ((seg_starts >= e1 * num_next_n_atoms) ? 1 : 0);
    int l1 = ((seg_starts >= e2 * num_next_n_atoms) ? 1 : 0);
    int l2 = ((seg_starts >= e3 * num_next_n_atoms) ? 1 : 0);
    int l3 = ((seg_starts >= e4 * num_next_n_atoms) ? 1 : 0);
    int l4 = ((seg_starts >= e5 * num_next_n_atoms) ? 1 : 0);
    int seg = l0 + l1 + l2 + l3 + l4;
    int o0 = ((l0 != 0) ? t0 : 0);
    int o1 = ((l1 != 0) ? t1 : 0);
    int o2 = ((l2 != 0) ? t2 : 0);
    int o3 = ((l3 != 0) ? t3 : 0);
    int o4 = ((l4 != 0) ? t4 : 0);
    int off_seg = o0 + o1 + o2 + o3 + o4;
    int seg_base = seg * 32;
    int rhs = seg_starts - off_seg * num_next_n_atoms;
    int q = seg_base;
    int pm = prefix[q + 16 - 1];
    int le = ((rhs >= pm * num_next_n_atoms) ? 1 : 0);
    q = ((le != 0) ? q + 16 : q);
    int pm_6 = prefix[q + 8 - 1];
    int le_7 = ((rhs >= pm_6 * num_next_n_atoms) ? 1 : 0);
    q = ((le_7 != 0) ? q + 8 : q);
    int pm_8 = prefix[q + 4 - 1];
    int le_9 = ((rhs >= pm_8 * num_next_n_atoms) ? 1 : 0);
    q = ((le_9 != 0) ? q + 4 : q);
    int pm_10 = prefix[q + 2 - 1];
    int le_11 = ((rhs >= pm_10 * num_next_n_atoms) ? 1 : 0);
    q = ((le_11 != 0) ? q + 2 : q);
    int pm_12 = prefix[q + 1 - 1];
    int le_13 = ((rhs >= pm_12 * num_next_n_atoms) ? 1 : 0);
    q = ((le_13 != 0) ? q + 1 : q);
    int q_idx_sm = ((sent != 0) ? batch_size : q);
    int prev_idx = ((q_idx_sm > 0) ? q_idx_sm - 1 : 0);
    int prev_raw = prefix[prev_idx];
    int prev_loc = ((q > seg_base) ? off_seg + prev_raw : off_seg);
    int prev_cum = ((sent != 0) ? total_segs : prev_loc);
    int cur_idx = ((q_idx_sm < batch_size) ? q_idx_sm : batch_size - 1);
    int cur_loc = prefix[cur_idx];
    int cur_raw = off_seg + cur_loc;
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
