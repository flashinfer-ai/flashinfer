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
#include "cake_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SHARED_STATE_OFF 0
#define SMEM_SHARED_STATE_STAGE_BYTES 2048
#define SMEM_SHARED_STATE_STRIDE 2048
#define SMEM_TOTAL 2048
#define THREADS 128



extern "C" {

__global__ __launch_bounds__(128, 2) void
kernel_flashinfer_blackwell_gdn_cp_prefill_fixup_v1(float* __restrict__ local_transfer, float* __restrict__ local_state, float* __restrict__ initial_state, float* __restrict__ initial_state_workspace, float* __restrict__ fixed_state, float* __restrict__ output_state, long long* __restrict__ cu_seqlens, int chunk_len, int total_cp_chunks, int num_seqs, int num_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* shared_state = reinterpret_cast<float*>(smem_raw + 0);
    const int shared_state_addr = smem + 0;

    // === Task calls (dependency order) ===
    int row_cta_idx = blockIdx.x % 32;
    int head_seq_idx = blockIdx.x / 32;
    int head_idx = head_seq_idx % num_heads;
    int seq_idx = head_seq_idx / num_heads;
    int col = tid;
    int row_base = row_cta_idx * 4;
    int seq_start = (int)cu_seqlens[seq_idx];
    int seq_end = (int)cu_seqlens[seq_idx + 1];
    int seq_len = seq_end - seq_start;
    int num_chunks = (seq_len + chunk_len - 1) / chunk_len;
    int bounded_prefix = seq_idx;
    if (seq_start < bounded_prefix) {
        bounded_prefix = seq_start;
    }
    int chunk_start = bounded_prefix + (seq_start - bounded_prefix) / chunk_len;
    int gap_start = chunk_start + num_chunks;
    int gap_end = total_cp_chunks;
    if (seq_idx + 1 < num_seqs) {
        int next_bounded_prefix = seq_idx + 1;
        if (seq_end < next_bounded_prefix) {
            next_bounded_prefix = seq_end;
        }
        gap_end = next_bounded_prefix + (seq_end - next_bounded_prefix) / chunk_len;
    }
    int chunk_head_linear = chunk_start * num_heads + head_idx;
    int state_head_linear = seq_idx * num_heads + head_idx;
    long long chunk_head_base = (long long)chunk_head_linear * 16384;
    long long state_head_base = (long long)state_head_linear * 16384;
    __syncthreads();
    float r_acc[4];
    float r_acc_next[4];
    float r_m[16];
    float r_m_next[16];
    if (num_chunks > 0) {
        #pragma unroll
        for (int row_in_cta = 0; row_in_cta < 4; row_in_cta++) {
            int row = row_base + row_in_cta;
            long long state_index = state_head_base + (long long)(row * 128) + (long long)col;
            float value = initial_state[state_index];
            shared_state[row_in_cta * 128 + col] = value;
            initial_state_workspace[state_index] = value;
        }
        __syncthreads();
        #pragma unroll
        for (int row_in_cta_1 = 0; row_in_cta_1 < 4; row_in_cta_1++) {
            long long local_index = chunk_head_base + (long long)((row_base + row_in_cta_1) * 128) + (long long)col;
            r_acc[row_in_cta_1] = local_state[local_index];
        }
        #pragma unroll
        for (int frag_col = 0; frag_col < 16; frag_col++) {
            long long transfer_index = chunk_head_base + (long long)(frag_col * 128) + (long long)col;
            r_m[frag_col] = local_transfer[transfer_index];
        }
    }
    #pragma unroll 1
    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int next_chunk_idx = chunk_idx + 1;
        int current_chunk_head_linear = (chunk_start + chunk_idx) * num_heads + head_idx;
        long long current_chunk_head_base = (long long)current_chunk_head_linear * 16384;
        #pragma unroll
        for (int iter_k = 0; iter_k < 7; iter_k++) {
            int next_k_base = (iter_k + 1) * 16;
            #pragma unroll
            for (int frag_col_1 = 0; frag_col_1 < 16; frag_col_1++) {
                long long transfer_index_1 = current_chunk_head_base + (long long)((next_k_base + frag_col_1) * 128) + (long long)col;
                r_m_next[frag_col_1] = local_transfer[transfer_index_1];
            }
            int current_k_base = iter_k * 16;
            #pragma unroll
            for (int row_in_cta_2 = 0; row_in_cta_2 < 4; row_in_cta_2++) {
                #pragma unroll
                for (int frag_col_2 = 0; frag_col_2 < 16; frag_col_2++) {
                    r_acc[row_in_cta_2] = r_acc[row_in_cta_2] + shared_state[row_in_cta_2 * 128 + current_k_base + frag_col_2] * r_m[frag_col_2];
                }
            }
            #pragma unroll
            for (int frag_col_3 = 0; frag_col_3 < 16; frag_col_3++) {
                r_m[frag_col_3] = r_m_next[frag_col_3];
            }
        }
        if (next_chunk_idx < num_chunks) {
            int next_chunk_head_linear = (chunk_start + next_chunk_idx) * num_heads + head_idx;
            long long next_chunk_head_base = (long long)next_chunk_head_linear * 16384;
            #pragma unroll
            for (int row_in_cta_3 = 0; row_in_cta_3 < 4; row_in_cta_3++) {
                long long local_index_1 = next_chunk_head_base + (long long)((row_base + row_in_cta_3) * 128) + (long long)col;
                r_acc_next[row_in_cta_3] = local_state[local_index_1];
            }
            #pragma unroll
            for (int frag_col_4 = 0; frag_col_4 < 16; frag_col_4++) {
                long long transfer_index_2 = next_chunk_head_base + (long long)(frag_col_4 * 128) + (long long)col;
                r_m_next[frag_col_4] = local_transfer[transfer_index_2];
            }
        }
        int last_k_base = 112;
        #pragma unroll
        for (int row_in_cta_4 = 0; row_in_cta_4 < 4; row_in_cta_4++) {
            #pragma unroll
            for (int frag_col_5 = 0; frag_col_5 < 16; frag_col_5++) {
                r_acc[row_in_cta_4] = r_acc[row_in_cta_4] + shared_state[row_in_cta_4 * 128 + last_k_base + frag_col_5] * r_m[frag_col_5];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int row_in_cta_5 = 0; row_in_cta_5 < 4; row_in_cta_5++) {
            float value_1 = r_acc[row_in_cta_5];
            shared_state[row_in_cta_5 * 128 + col] = value_1;
            long long fixed_index = current_chunk_head_base + (long long)((row_base + row_in_cta_5) * 128) + (long long)col;
            fixed_state[fixed_index] = value_1;
        }
        __syncthreads();
        if (next_chunk_idx < num_chunks) {
            #pragma unroll
            for (int row_in_cta_6 = 0; row_in_cta_6 < 4; row_in_cta_6++) {
                r_acc[row_in_cta_6] = r_acc_next[row_in_cta_6];
            }
            #pragma unroll
            for (int frag_col_6 = 0; frag_col_6 < 16; frag_col_6++) {
                r_m[frag_col_6] = r_m_next[frag_col_6];
            }
        }
    }
    if (num_chunks > 0) {
        #pragma unroll
        for (int row_in_cta_7 = 0; row_in_cta_7 < 4; row_in_cta_7++) {
            long long output_index = state_head_base + (long long)((row_base + row_in_cta_7) * 128) + (long long)col;
            output_state[output_index] = shared_state[row_in_cta_7 * 128 + col];
        }
    } else {
        #pragma unroll
        for (int row_in_cta_8 = 0; row_in_cta_8 < 4; row_in_cta_8++) {
            long long output_index_1 = state_head_base + (long long)((row_base + row_in_cta_8) * 128) + (long long)col;
            output_state[output_index_1] = initial_state[output_index_1];
        }
    }
    #pragma unroll 1
    for (int gap_idx = gap_start; gap_idx < gap_end; gap_idx++) {
        int gap_head_linear = gap_idx * num_heads + head_idx;
        long long gap_head_base = (long long)gap_head_linear * 16384;
        #pragma unroll
        for (int row_in_cta_9 = 0; row_in_cta_9 < 4; row_in_cta_9++) {
            long long fixed_index_1 = gap_head_base + (long long)((row_base + row_in_cta_9) * 128) + (long long)col;
            fixed_state[fixed_index_1] = 0.0f;
        }
    }
}

} // extern "C"

#undef CAKE_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SHARED_STATE_OFF
#undef SMEM_SHARED_STATE_STAGE_BYTES
#undef SMEM_SHARED_STATE_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef shared_state_addr
// clang-format on
