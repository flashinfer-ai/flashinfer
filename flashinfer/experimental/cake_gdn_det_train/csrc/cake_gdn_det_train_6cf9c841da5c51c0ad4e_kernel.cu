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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_gdn_det_train_6cf9c841da5c51c0ad4e(int* __restrict__ cu_seqlens, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seq_chunk_start, int* __restrict__ fwd_seg_chunk_start, int* __restrict__ fwd_seg_seq, int* __restrict__ fwd_seq_seg_start, int* __restrict__ bwd_seg_chunk_start, int* __restrict__ bwd_seg_seq, int* __restrict__ bwd_seq_seg_start, int num_seqs, int num_chunks_max, int total_tokens, int fwd_segment_len, int bwd_segment_len, int fwd_num_segments_max, int bwd_num_segments_max)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int tid_0 = tid;
    if (tid_0 == 0) {
        int acc = 0;
        seq_chunk_start[0] = 0;
        #pragma unroll 1
        for (int s0 = 0; s0 < num_seqs; s0++) {
            int seq_len = cu_seqlens[s0 + 1] - cu_seqlens[s0];
            acc = acc + (seq_len + 63) / 64;
            seq_chunk_start[s0 + 1] = acc;
        }
        int total = 0;
        fwd_seq_seg_start[0] = 0;
        fwd_seg_chunk_start[0] = 0;
        #pragma unroll 1
        for (int s0_1 = 0; s0_1 < num_seqs; s0_1++) {
            int c0 = seq_chunk_start[s0_1];
            int nc = seq_chunk_start[s0_1 + 1] - c0;
            int count = 1;
            if (fwd_segment_len > 0) {
                if (nc > fwd_segment_len) {
                    count = (nc + fwd_segment_len - 1) / fwd_segment_len;
                }
            }
            #pragma unroll 1
            for (int s1 = 0; s1 < count; s1++) {
                fwd_seg_chunk_start[total + s1 + 1] = c0 + (s1 + 1) * nc / count;
                fwd_seg_seq[total + s1] = s0_1;
            }
            total = total + count;
            fwd_seq_seg_start[s0_1 + 1] = total;
        }
        int total_chunks = seq_chunk_start[num_seqs];
        #pragma unroll 1
        for (int pad = 0; pad < fwd_num_segments_max - total; pad++) {
            fwd_seg_chunk_start[total + pad + 1] = total_chunks;
            fwd_seg_seq[total + pad] = -1;
        }
        int total_0 = 0;
        bwd_seq_seg_start[0] = 0;
        bwd_seg_chunk_start[0] = 0;
        #pragma unroll 1
        for (int s0_2 = 0; s0_2 < num_seqs; s0_2++) {
            int c0_1 = seq_chunk_start[s0_2];
            int nc_1 = seq_chunk_start[s0_2 + 1] - c0_1;
            int count_1 = 1;
            if (bwd_segment_len > 0) {
                if (nc_1 > bwd_segment_len) {
                    count_1 = (nc_1 + bwd_segment_len - 1) / bwd_segment_len;
                }
            }
            #pragma unroll 1
            for (int s1_1 = 0; s1_1 < count_1; s1_1++) {
                bwd_seg_chunk_start[total_0 + s1_1 + 1] = c0_1 + (s1_1 + 1) * nc_1 / count_1;
                bwd_seg_seq[total_0 + s1_1] = s0_2;
            }
            total_0 = total_0 + count_1;
            bwd_seq_seg_start[s0_2 + 1] = total_0;
        }
        int total_chunks_1 = seq_chunk_start[num_seqs];
        #pragma unroll 1
        for (int pad_1 = 0; pad_1 < bwd_num_segments_max - total_0; pad_1++) {
            bwd_seg_chunk_start[total_0 + pad_1 + 1] = total_chunks_1;
            bwd_seg_seq[total_0 + pad_1] = -1;
        }
    }
    __syncthreads();
    int total_1 = seq_chunk_start[num_seqs];
    #pragma unroll 1
    for (int it = 0; it < (num_seqs + 127) / 128; it++) {
        int seq = it * 128 + tid_0;
        if (seq < num_seqs) {
            int c0_2 = seq_chunk_start[seq];
            int n_chunks = seq_chunk_start[seq + 1] - c0_2;
            int seq_tok0 = cu_seqlens[seq];
            int seq_end = cu_seqlens[seq + 1];
            #pragma unroll 1
            for (int j = 0; j < n_chunks; j++) {
                int start = seq_tok0 + j * 64;
                chunk_start[c0_2 + j] = start;
                int _min_0 = ((seq_end - start) < (64) ? (seq_end - start) : (64));
                chunk_len[c0_2 + j] = _min_0;
            }
        }
    }
    #pragma unroll 1
    for (int it2 = 0; it2 < (num_chunks_max + 127) / 128; it2++) {
        int pad_2 = total_1 + it2 * 128 + tid_0;
        if (pad_2 < num_chunks_max) {
            chunk_start[pad_2] = total_tokens;
            chunk_len[pad_2] = 0;
        }
    }
}

} // extern "C"
