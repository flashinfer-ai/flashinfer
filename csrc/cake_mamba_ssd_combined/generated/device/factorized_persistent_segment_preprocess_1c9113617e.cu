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
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_factorized_persistent_segment_preprocess(float* __restrict__ dt, float* __restrict__ A, float* __restrict__ dt_bias, int* __restrict__ segment_starts, int* __restrict__ segment_lengths, int* __restrict__ chunk_indices, int* __restrict__ chunk_offsets, __half* __restrict__ delta, float* __restrict__ cumsum, int num_segments, int nheads, int seqlen, int direct_varlen_metadata, int dt_softplus, float dt_min, float dt_max, int* __restrict__ seq_idx_i32, long long* __restrict__ seq_idx_i64, int seq_idx_int64, int* __restrict__ seq_chunk_cumsum, int num_sequences, int write_seq_chunk_cumsum)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int tile = bid * 128 + tid;
    int total_tiles = num_segments * nheads;
    if (tile < total_tiles) {
        int segment = tile / nheads;
        int head = tile % nheads;
        int start = 0;
        int length = 128;
        int physical_start = 0;
        int segment_offset = 0;
        if (direct_varlen_metadata != 0) {
            start = chunk_indices[segment] * 128 + chunk_offsets[segment];
            int end = seqlen;
            if (segment + 1 < num_segments) {
                end = chunk_indices[segment + 1] * 128 + chunk_offsets[segment + 1];
            }
            length = end - start;
            physical_start = start / 128 * 128;
            segment_offset = start - physical_start;
        } else {
            start = segment_starts[segment];
            length = segment_lengths[segment];
            physical_start = start;
        }
        if (write_seq_chunk_cumsum != 0 && head == 0) {
            int sequence = 0;
            if (seq_idx_int64 != 0) {
                sequence = (int)seq_idx_i64[start];
            } else {
                sequence = seq_idx_i32[start];
            }
            if (segment == 0) {
                seq_chunk_cumsum[sequence] = 0;
            } else {
                int previous_start = chunk_indices[segment - 1] * 128 + chunk_offsets[segment - 1];
                int previous_sequence = 0;
                if (seq_idx_int64 != 0) {
                    previous_sequence = (int)seq_idx_i64[previous_start];
                } else {
                    previous_sequence = seq_idx_i32[previous_start];
                }
                if (sequence != previous_sequence) {
                    seq_chunk_cumsum[sequence] = segment;
                }
            }
            if (segment == num_segments - 1) {
                seq_chunk_cumsum[num_sequences] = num_segments;
            }
        }
        float a_value = A[head];
        float bias_value = dt_bias[head];
        float running = 0.0f;
        float segment_base = 0.0f;
        float dt_all[128];
        #pragma unroll
        for (int token = 0; token < 128; token++) {
            float raw_dt = 0.0f;
            if (token < segment_offset + length) {
                raw_dt = dt[(physical_start + token) * nheads + head];
            }
            dt_all[token] = raw_dt;
        }
        float group_dt[16];
        float group_product[16];
        float scan_offset_1[16];
        float scan_offset_2[16];
        float scan_offset_4[16];
        float group_scan[16];
        float group_dt_4[4];
        float group_product_4[4];
        float group_scan_4[4];
        if (nheads >= 16) {
            #pragma unroll
            for (int group_start = 0; group_start < 128; group_start += 16) {
                #pragma unroll
                for (int local = 0; local < 16; local++) {
                    int physical_token = group_start + local;
                    float transformed = 0.0f;
                    if (physical_token < segment_offset + length) {
                        float biased = dt_all[group_start + local] + bias_value;
                        transformed = biased;
                        if (dt_softplus != 0) {
                            if (biased <= 20.0f) {
                                float _exp2_0 = approx_exp2(biased * 1.4426950408889634f);
                                float _log_0 = logf(_exp2_0 + 1.0f);
                                transformed = _log_0;
                            }
                        }
                        if (transformed < dt_min) {
                            transformed = dt_min;
                        }
                        if (transformed > dt_max) {
                            transformed = dt_max;
                        }
                    }
                    group_dt[local] = transformed;
                    group_product[local] = transformed * a_value;
                    if (physical_token >= segment_offset && physical_token < segment_offset + length) {
                        int local_token = physical_token - segment_offset;
                        float _max_0 = max_noftz(transformed, -65504.0f);
                        float _min_0 = fminf(_max_0, 65504.0f);
                        delta[tile * 128 + local_token] = (__half)_min_0;
                    }
                }
                scan_offset_1[0] = group_product[0];
                #pragma unroll
                for (int local_1 = 1; local_1 < 16; local_1++) {
                    float _fma_0 = __fmaf_rn(group_dt[local_1], a_value, group_product[local_1 - 1]);
                    scan_offset_1[local_1] = _fma_0;
                }
                #pragma unroll
                for (int local_2 = 0; local_2 < 16; local_2++) {
                    if (local_2 < 2) {
                        scan_offset_2[local_2] = scan_offset_1[local_2];
                    } else {
                        scan_offset_2[local_2] = scan_offset_1[local_2 - 2] + scan_offset_1[local_2];
                    }
                }
                #pragma unroll
                for (int local_3 = 0; local_3 < 16; local_3++) {
                    if (local_3 < 4) {
                        scan_offset_4[local_3] = scan_offset_2[local_3];
                    } else {
                        scan_offset_4[local_3] = scan_offset_2[local_3 - 4] + scan_offset_2[local_3];
                    }
                }
                #pragma unroll
                for (int local_4 = 0; local_4 < 16; local_4++) {
                    if (local_4 < 8) {
                        group_scan[local_4] = scan_offset_4[local_4];
                    } else {
                        group_scan[local_4] = scan_offset_4[local_4 - 8] + scan_offset_4[local_4];
                    }
                }
                #pragma unroll
                for (int local_5 = 0; local_5 < 16; local_5++) {
                    int physical_token_1 = group_start + local_5;
                    float physical_cumsum = group_scan[local_5];
                    if (group_start != 0) {
                        physical_cumsum += running;
                    }
                    if (physical_token_1 == segment_offset - 1) {
                        segment_base = physical_cumsum;
                    }
                    if (physical_token_1 >= segment_offset && physical_token_1 < segment_offset + length) {
                        int local_token_1 = physical_token_1 - segment_offset;
                        cumsum[tile * 128 + local_token_1] = physical_cumsum - segment_base;
                    }
                }
                if (group_start == 0) {
                    running = group_scan[15];
                } else {
                    running += group_scan[15];
                }
            }
        } else {
            #pragma unroll
            for (int group_start_1 = 0; group_start_1 < 128; group_start_1 += 4) {
                #pragma unroll
                for (int local_6 = 0; local_6 < 4; local_6++) {
                    int physical_token_2 = group_start_1 + local_6;
                    float transformed_1 = 0.0f;
                    if (physical_token_2 < segment_offset + length) {
                        float biased_1 = dt_all[group_start_1 + local_6] + bias_value;
                        transformed_1 = biased_1;
                        if (dt_softplus != 0) {
                            if (biased_1 <= 20.0f) {
                                float _exp2_1 = approx_exp2(biased_1 * 1.4426950408889634f);
                                float _log_1 = logf(_exp2_1 + 1.0f);
                                transformed_1 = _log_1;
                            }
                        }
                        if (transformed_1 < dt_min) {
                            transformed_1 = dt_min;
                        }
                        if (transformed_1 > dt_max) {
                            transformed_1 = dt_max;
                        }
                    }
                    group_dt_4[local_6] = transformed_1;
                    group_product_4[local_6] = transformed_1 * a_value;
                    if (physical_token_2 >= segment_offset && physical_token_2 < segment_offset + length) {
                        int local_token_2 = physical_token_2 - segment_offset;
                        float _max_1 = max_noftz(transformed_1, -65504.0f);
                        float _min_1 = fminf(_max_1, 65504.0f);
                        delta[tile * 128 + local_token_2] = (__half)_min_1;
                    }
                }
                group_scan_4[0] = group_product_4[0];
                float _fma_1 = __fmaf_rn(group_dt_4[1], a_value, group_product_4[0]);
                group_scan_4[1] = _fma_1;
                float _fma_2 = __fmaf_rn(group_dt_4[2], a_value, group_product_4[1]);
                float group_pair_12 = _fma_2;
                group_scan_4[2] = group_pair_12 + group_product_4[0];
                float _fma_3 = __fmaf_rn(group_dt_4[3], a_value, group_product_4[2]);
                float group_pair_23 = _fma_3;
                group_scan_4[3] = group_pair_23 + group_scan_4[1];
                #pragma unroll
                for (int local_7 = 0; local_7 < 4; local_7++) {
                    int physical_token_3 = group_start_1 + local_7;
                    float physical_cumsum_1 = group_scan_4[local_7];
                    if (group_start_1 != 0) {
                        physical_cumsum_1 += running;
                    }
                    if (physical_token_3 == segment_offset - 1) {
                        segment_base = physical_cumsum_1;
                    }
                    if (physical_token_3 >= segment_offset && physical_token_3 < segment_offset + length) {
                        int local_token_3 = physical_token_3 - segment_offset;
                        cumsum[tile * 128 + local_token_3] = physical_cumsum_1 - segment_base;
                    }
                }
                if (group_start_1 == 0) {
                    running = group_scan_4[3];
                } else {
                    running += group_scan_4[3];
                }
            }
        }
    }
}

} // extern "C"

