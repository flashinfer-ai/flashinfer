/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include <cuda_fp16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_factorized_persistent_packed_varlen_metadata(int* __restrict__ seq_idx_i32, long long* __restrict__ seq_idx_i64, int* __restrict__ chunk_indices, int* __restrict__ chunk_offsets, int* __restrict__ segment_starts, int* __restrict__ segment_lengths, int* __restrict__ sequence_offsets, int num_segments, int num_sequences, int seqlen, int seq_idx_int64, int* __restrict__ preprocess_status)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int segment = bid * 256 + tid;
    if (segment < num_segments) {
        int start = chunk_indices[segment] * 128 + chunk_offsets[segment];
        int end = seqlen;
        if (segment + 1 < num_segments) {
            end = chunk_indices[segment + 1] * 128 + chunk_offsets[segment + 1];
        }
        segment_starts[segment] = start;
        segment_lengths[segment] = end - start;
        int sequence = 0;
        if (seq_idx_int64 != 0) {
            long long raw_sequence = seq_idx_i64[start];
            sequence = (int)raw_sequence;
            if (raw_sequence < 0) {
                sequence = -1;
            }
            if (raw_sequence >= (long long)num_sequences) {
                sequence = num_sequences;
            }
        } else {
            sequence = seq_idx_i32[start];
        }
        int _max_0 = ((sequence) > (-1) ? (sequence) : (-1));
        int _min_0 = ((_max_0) < (num_sequences) ? (_max_0) : (num_sequences));
        sequence = _min_0;
        int previous_sequence = -1;
        if (segment != 0) {
            int previous_start = chunk_indices[segment - 1] * 128 + chunk_offsets[segment - 1];
            if (seq_idx_int64 != 0) {
                long long raw_previous = seq_idx_i64[previous_start];
                previous_sequence = (int)raw_previous;
                if (raw_previous < 0) {
                    previous_sequence = -1;
                }
                if (raw_previous >= (long long)num_sequences) {
                    previous_sequence = num_sequences;
                }
            } else {
                previous_sequence = seq_idx_i32[previous_start];
            }
            int _max_1 = ((previous_sequence) > (-1) ? (previous_sequence) : (-1));
            int _min_1 = ((_max_1) < (num_sequences) ? (_max_1) : (num_sequences));
            previous_sequence = _min_1;
        }
        int flagged = 0;
        if (sequence < 0) {
            flagged = 1;
        }
        if (sequence >= num_sequences) {
            flagged = 1;
        }
        if (sequence < previous_sequence) {
            flagged = 1;
        }
        if (flagged != 0) {
            preprocess_status[0] = 1;
        }
        #pragma unroll 1
        for (int opened_id = previous_sequence + 1; opened_id < sequence + 1; opened_id++) {
            sequence_offsets[opened_id] = segment;
        }
        if (segment == num_segments - 1) {
            #pragma unroll 1
            for (int closed_id = sequence + 1; closed_id < num_sequences + 1; closed_id++) {
                sequence_offsets[closed_id] = num_segments;
            }
        }
    }
}

} // extern "C"

