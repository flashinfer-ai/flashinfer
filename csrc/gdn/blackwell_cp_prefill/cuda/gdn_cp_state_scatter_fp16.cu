// Copyright (c) 2026 by FlashInfer team.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// clang-format off
#include "gdn_cp_common.cuh"

#define GDN_CP_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256



extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_flashinfer_blackwell_gdn_cp_prefill_state_scatter_fp16_v1(float* __restrict__ packed, int* __restrict__ state_indices, __half* __restrict__ output, long long pool_stride0, long long pool_stride1, long long pool_stride2, long long pool_stride3, int num_heads, long long total_values, int use_indices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    long long linear = (long long)blockIdx.x * 256 + (long long)tid;
    if (linear < total_values) {
        long long values_per_seq = (long long)num_heads * 128 * 128;
        int seq_idx = (int)(linear / values_per_seq);
        long long inner = linear % values_per_seq;
        long long head_idx = inner / 16384;
        long long matrix_inner = inner % 16384;
        long long row_idx = matrix_inner / 128;
        long long col_idx = matrix_inner % 128;
        int pool_row = seq_idx;
        if (use_indices != 0) {
            pool_row = state_indices[seq_idx];
        }
        long long output_index = (long long)pool_row * pool_stride0 + head_idx * pool_stride1 + row_idx * pool_stride2 + col_idx * pool_stride3;
        output[output_index] = packed[linear];
    }
}

} // extern "C"

#undef GDN_CP_INF
#undef NUM_MAIN_STAGES
#undef THREADS
// clang-format on
