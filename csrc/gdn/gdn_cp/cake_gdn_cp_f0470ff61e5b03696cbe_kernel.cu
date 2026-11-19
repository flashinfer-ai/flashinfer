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

#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_cp_f0470ff61e5b03696cbe(float* __restrict__ local_transfer, float* __restrict__ local_state, float* initial_state, float* __restrict__ initial_state_workspace, float* __restrict__ fixed_state, float* output_state, long long* __restrict__ cu_seqlens, int chunk_len, int total_cp_chunks, int num_seqs, int num_heads, int* __restrict__ state_indices, int* __restrict__ output_state_indices, int* __restrict__ checkpoint_state_indices, int* __restrict__ checkpoint_cu_starts, int* __restrict__ prefix_lens, long long state_stride0, int page_size)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
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

    // Prologue
    float* shared_state = reinterpret_cast<float*>(smem_raw + 0);
    const int shared_state_addr = smem + 0;

    // === Task calls (declaration order) ===
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
    long long chunk_head_base = (long long)chunk_head_linear * 16384ll;
    long long state_head_base = (long long)state_head_linear * 16384ll;
    int prefix_len = prefix_lens[seq_idx];
    bool aligned = (prefix_len & 63) == 0;
    int read_row = state_indices[seq_idx];
    int final_row = output_state_indices[seq_idx];
    int checkpoint_start = checkpoint_cu_starts[seq_idx];
    long long pool_head_offset = (long long)head_idx * 16384ll;
    float r_initial[4];
    #pragma unroll
    for (int row_in_cta = 0; row_in_cta < 4; row_in_cta++) {
        r_initial[row_in_cta] = 0.0f;
        if (read_row >= 0) {
            r_initial[row_in_cta] = initial_state[(long long)read_row * state_stride0 + pool_head_offset + (long long)((row_base + row_in_cta) * 128) + (long long)col];
        }
    }
    __syncthreads();
    float r_acc[4];
    float r_acc_next[4];
    float r_ring[64];
    if (num_chunks > 0) {
        #pragma unroll
        for (int row_in_cta_1 = 0; row_in_cta_1 < 4; row_in_cta_1++) {
            int row = row_base + row_in_cta_1;
            long long state_index = state_head_base + (long long)(row * 128) + (long long)col;
            float value = r_initial[row_in_cta_1];
            shared_state[row_in_cta_1 * 128 + col] = value;
            initial_state_workspace[state_index] = value;
        }
        __syncthreads();
        #pragma unroll
        for (int row_in_cta_2 = 0; row_in_cta_2 < 4; row_in_cta_2++) {
            long long local_index = chunk_head_base + (long long)((row_base + row_in_cta_2) * 128) + (long long)col;
            r_acc[row_in_cta_2] = local_state[local_index];
        }
        #pragma unroll
        for (int pre_k = 0; pre_k < 3; pre_k++) {
            int pre_k_base = pre_k * 16;
            #pragma unroll
            for (int frag_col = 0; frag_col < 16; frag_col++) {
                long long transfer_index = chunk_head_base + (long long)((pre_k_base + frag_col) * 128) + (long long)col;
                r_ring[pre_k_base + frag_col] = local_transfer[transfer_index];
            }
        }
    }
    #pragma unroll 1
    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int next_chunk_idx = chunk_idx + 1;
        int current_chunk_head_linear = (chunk_start + chunk_idx) * num_heads + head_idx;
        long long current_chunk_head_base = (long long)current_chunk_head_linear * 16384ll;
        int next_chunk_head_linear = (chunk_start + next_chunk_idx) * num_heads + head_idx;
        long long next_chunk_head_base = (long long)next_chunk_head_linear * 16384ll;
        #pragma unroll
        for (int iter_k = 0; iter_k < 5; iter_k++) {
            int load_k_base = (iter_k + 3) * 16;
            int load_slot = (iter_k + 3) % 4 * 16;
            #pragma unroll
            for (int frag_col_1 = 0; frag_col_1 < 16; frag_col_1++) {
                long long transfer_index_1 = current_chunk_head_base + (long long)((load_k_base + frag_col_1) * 128) + (long long)col;
                r_ring[load_slot + frag_col_1] = local_transfer[transfer_index_1];
            }
            int current_k_base = iter_k * 16;
            int use_slot = iter_k % 4 * 16;
            #pragma unroll
            for (int row_in_cta_3 = 0; row_in_cta_3 < 4; row_in_cta_3++) {
                #pragma unroll
                for (int frag_col_2 = 0; frag_col_2 < 16; frag_col_2++) {
                    r_acc[row_in_cta_3] = r_acc[row_in_cta_3] + shared_state[row_in_cta_3 * 128 + current_k_base + frag_col_2] * r_ring[use_slot + frag_col_2];
                }
            }
        }
        if (next_chunk_idx < num_chunks) {
            #pragma unroll
            for (int row_in_cta_4 = 0; row_in_cta_4 < 4; row_in_cta_4++) {
                long long local_index_1 = next_chunk_head_base + (long long)((row_base + row_in_cta_4) * 128) + (long long)col;
                r_acc_next[row_in_cta_4] = local_state[local_index_1];
            }
        }
        #pragma unroll
        for (int tail_k = 5; tail_k < 8; tail_k++) {
            if (next_chunk_idx < num_chunks) {
                int next_k_base = (tail_k - 5) * 16;
                int next_slot = (tail_k + 3) % 4 * 16;
                #pragma unroll
                for (int frag_col_3 = 0; frag_col_3 < 16; frag_col_3++) {
                    long long transfer_index_2 = next_chunk_head_base + (long long)((next_k_base + frag_col_3) * 128) + (long long)col;
                    r_ring[next_slot + frag_col_3] = local_transfer[transfer_index_2];
                }
            }
            int tail_k_base = tail_k * 16;
            int tail_slot = tail_k % 4 * 16;
            #pragma unroll
            for (int row_in_cta_5 = 0; row_in_cta_5 < 4; row_in_cta_5++) {
                #pragma unroll
                for (int frag_col_4 = 0; frag_col_4 < 16; frag_col_4++) {
                    r_acc[row_in_cta_5] = r_acc[row_in_cta_5] + shared_state[row_in_cta_5 * 128 + tail_k_base + frag_col_4] * r_ring[tail_slot + frag_col_4];
                }
            }
        }
        int page_end = prefix_len + next_chunk_idx * chunk_len;
        if (page_end > prefix_len + seq_len) {
            page_end = prefix_len + seq_len;
        }
        int checkpoint_row = -1;
        if (aligned && page_end % page_size == 0) {
            checkpoint_row = checkpoint_state_indices[checkpoint_start + page_end / page_size - prefix_len / page_size - 1];
        }
        long long checkpoint_base = (long long)checkpoint_row * state_stride0 + pool_head_offset;
        __syncthreads();
        #pragma unroll
        for (int row_in_cta_6 = 0; row_in_cta_6 < 4; row_in_cta_6++) {
            float value_1 = r_acc[row_in_cta_6];
            shared_state[row_in_cta_6 * 128 + col] = value_1;
            long long fixed_index = current_chunk_head_base + (long long)((row_base + row_in_cta_6) * 128) + (long long)col;
            fixed_state[fixed_index] = value_1;
            if (checkpoint_row >= 0) {
                output_state[checkpoint_base + (long long)((row_base + row_in_cta_6) * 128) + (long long)col] = value_1;
            }
        }
        __syncthreads();
        if (next_chunk_idx < num_chunks) {
            #pragma unroll
            for (int row_in_cta_7 = 0; row_in_cta_7 < 4; row_in_cta_7++) {
                r_acc[row_in_cta_7] = r_acc_next[row_in_cta_7];
            }
        }
    }
    if (num_chunks > 0 && final_row >= 0) {
        long long final_base = (long long)final_row * state_stride0 + pool_head_offset;
        #pragma unroll
        for (int row_in_cta_8 = 0; row_in_cta_8 < 4; row_in_cta_8++) {
            float final_value = shared_state[row_in_cta_8 * 128 + col];
            if ((prefix_len & 63) != 0) {
                final_value = __uint_as_float(2143289344);
            }
            output_state[final_base + (long long)((row_base + row_in_cta_8) * 128) + (long long)col] = final_value;
        }
    }
    if (num_chunks > 0 && (prefix_len & 63) != 0) {
        int crossed = (prefix_len + seq_len) / page_size - prefix_len / page_size;
        #pragma unroll 1
        for (int crossed_idx = 0; crossed_idx < crossed; crossed_idx++) {
            int nan_row = checkpoint_state_indices[checkpoint_start + crossed_idx];
            if (nan_row >= 0) {
                long long nan_base = (long long)nan_row * state_stride0 + pool_head_offset;
                #pragma unroll
                for (int row_in_cta_9 = 0; row_in_cta_9 < 4; row_in_cta_9++) {
                    output_state[nan_base + (long long)((row_base + row_in_cta_9) * 128) + (long long)col] = __uint_as_float(2143289344);
                }
            }
        }
    }
    #pragma unroll 1
    for (int gap_idx = gap_start; gap_idx < gap_end; gap_idx++) {
        int gap_head_linear = gap_idx * num_heads + head_idx;
        long long gap_head_base = (long long)gap_head_linear * 16384ll;
        #pragma unroll
        for (int row_in_cta_10 = 0; row_in_cta_10 < 4; row_in_cta_10++) {
            long long fixed_index_1 = gap_head_base + (long long)((row_base + row_in_cta_10) * 128) + (long long)col;
            fixed_state[fixed_index_1] = 0.0f;
        }
    }
}

} // extern "C"
