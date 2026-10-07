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
#define SMEM_SMEM_SEQUENCE_EXCL_OFF 99328
#define SMEM_SMEM_SEQUENCE_EXCL_STAGE_BYTES 516
#define SMEM_SMEM_SEQUENCE_EXCL_STRIDE 516
#define SMEM_SMEM_WARP_TOTALS_OFF 99856
#define SMEM_SMEM_WARP_TOTALS_STAGE_BYTES 16
#define SMEM_SMEM_WARP_TOTALS_STRIDE 16
#define SMEM_SMEM_CUMSUM_OFF 0
#define SMEM_SMEM_CUMSUM_STAGE_BYTES 66048
#define SMEM_SMEM_CUMSUM_STRIDE 66048
#define SMEM_SMEM_DELTA_OFF 66048
#define SMEM_SMEM_DELTA_STAGE_BYTES 33280
#define SMEM_SMEM_DELTA_STRIDE 33280
#define SMEM_TOTAL 99968
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
kernel_factorized_persistent_segment_preprocess(float* __restrict__ dt, float* __restrict__ A, float* __restrict__ dt_bias, int* __restrict__ segment_starts, int* __restrict__ segment_lengths, int* __restrict__ chunk_indices, int* __restrict__ chunk_offsets, __half* __restrict__ delta, float* __restrict__ cumsum, int num_segments, int nheads, int seqlen, int direct_varlen_metadata, int dt_softplus, float dt_min, float dt_max, int* __restrict__ seq_idx_i32, long long* __restrict__ seq_idx_i64, int seq_idx_int64, int* __restrict__ seq_chunk_cumsum, int num_sequences, int write_seq_chunk_cumsum, int* __restrict__ cu_seqlens, int* __restrict__ checkpoint_token_indices, int metadata_from_cu_seqlens, int checkpoint_state_count, int* __restrict__ preprocess_status)
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
    int* smem_sequence_excl = reinterpret_cast<int*>(smem_raw + 99328);
    const int smem_sequence_excl_addr = smem + 99328;
    int* smem_warp_totals = reinterpret_cast<int*>(smem_raw + 99856);
    const int smem_warp_totals_addr = smem + 99856;
    float* smem_cumsum = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_cumsum_addr = smem + 0;
    __half* smem_delta = reinterpret_cast<__half*>(smem_raw + 66048);
    const int smem_delta_addr = smem + 66048;

    // === Task calls (dependency order) ===
    int tile = bid * 128 + tid;
    int segment = tile / nheads;
    int head = tile % nheads;
    int segment_count = num_segments;
    int derived_start = 0;
    int derived_end = 0;
    int slot_base_cumsum = tid * 129;
    int slot_base_delta = tid * 130;
    __half zero_half = (__half)0.0f;
    if (metadata_from_cu_seqlens != 0) {
        int segment_bound = num_segments;
        int warp_1 = tid / 32;
        int lane_0 = lane;
        int flagged_cu = 0;
        int carry = 0;
        int found = 0;
        #pragma unroll 1
        for (int block_base = 0; block_base < num_sequences; block_base += 128) {
            int sequence_slot = block_base + tid;
            int count = 0;
            if (sequence_slot < num_sequences) {
                int raw_lo = cu_seqlens[sequence_slot];
                int raw_hi = cu_seqlens[sequence_slot + 1];
                if (sequence_slot == 0) {
                    if (raw_lo != 0) {
                        flagged_cu = 1;
                    }
                }
                if (sequence_slot == num_sequences - 1) {
                    if (raw_hi != seqlen) {
                        flagged_cu = 1;
                    }
                }
                if (raw_hi < raw_lo) {
                    flagged_cu = 1;
                }
                if (raw_lo < 0) {
                    flagged_cu = 1;
                }
                if (raw_hi > seqlen) {
                    flagged_cu = 1;
                }
                int _max_0 = ((raw_lo) > (0) ? (raw_lo) : (0));
                int _min_0 = ((_max_0) < (seqlen) ? (_max_0) : (seqlen));
                int lo = _min_0;
                int _max_1 = ((raw_hi) > (0) ? (raw_hi) : (0));
                int _min_1 = ((_max_1) < (seqlen) ? (_max_1) : (seqlen));
                int hi = _min_1;
                if (hi > lo) {
                    count = (hi + 127) / 128 - lo / 128;
                    if (checkpoint_state_count > 0) {
                        int checkpoint = checkpoint_token_indices[sequence_slot];
                        if (checkpoint > lo && checkpoint < hi && checkpoint % 128 != 0) {
                            count = count + 1;
                        }
                    }
                }
            }
            int inclusive = count;
            int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
            int shifted = _shfl_up_0;
            if (lane_0 >= 1) {
                inclusive = inclusive + shifted;
            }
            int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
            shifted = _shfl_up_1;
            if (lane_0 >= 2) {
                inclusive = inclusive + shifted;
            }
            int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
            shifted = _shfl_up_2;
            if (lane_0 >= 4) {
                inclusive = inclusive + shifted;
            }
            int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
            shifted = _shfl_up_3;
            if (lane_0 >= 8) {
                inclusive = inclusive + shifted;
            }
            int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
            shifted = _shfl_up_4;
            if (lane_0 >= 16) {
                inclusive = inclusive + shifted;
            }
            if (lane_0 == 31) {
                smem_warp_totals[warp_1] = inclusive;
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            int warp_prefix = 0;
            int block_total = 0;
            #pragma unroll
            for (int other_warp = 0; other_warp < 4; other_warp++) {
                int other_total = smem_warp_totals[other_warp];
                block_total = block_total + other_total;
                if (warp_1 > other_warp) {
                    warp_prefix = warp_prefix + other_total;
                }
            }
            int exclusive = carry + warp_prefix + inclusive - count;
            smem_sequence_excl[tid] = exclusive;
            if (tid == 0) {
                smem_sequence_excl[128] = carry + block_total;
            }
            if (bid == 0) {
                if (sequence_slot < num_sequences) {
                    int _min_2 = ((exclusive) < (segment_bound) ? (exclusive) : (segment_bound));
                    seq_chunk_cumsum[sequence_slot] = _min_2;
                }
                if (tid == 0) {
                    if (block_base + 128 >= num_sequences) {
                        int _min_3 = ((carry + block_total) < (segment_bound) ? (carry + block_total) : (segment_bound));
                        seq_chunk_cumsum[num_sequences] = _min_3;
                    }
                }
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (found == 0) {
                if (segment >= carry && segment < carry + block_total) {
                    int owner_slot = 0;
                    int low = 0;
                    int high = 127;
                    #pragma unroll
                    for (int probe = 0; probe < 7; probe++) {
                        int middle = (low + high + 1) / 2;
                        if (segment >= smem_sequence_excl[middle]) {
                            low = middle;
                        } else {
                            high = middle - 1;
                        }
                    }
                    owner_slot = low;
                    int owner = block_base + owner_slot;
                    int _max_2 = ((cu_seqlens[owner]) > (0) ? (cu_seqlens[owner]) : (0));
                    int _min_4 = ((_max_2) < (seqlen) ? (_max_2) : (seqlen));
                    int owner_lo = _min_4;
                    int _max_3 = ((cu_seqlens[owner + 1]) > (0) ? (cu_seqlens[owner + 1]) : (0));
                    int _min_5 = ((_max_3) < (seqlen) ? (_max_3) : (seqlen));
                    int owner_hi = _min_5;
                    int owner_count = (owner_hi + 127) / 128 - owner_lo / 128;
                    int owner_first_chunk = owner_lo / 128;
                    int owner_checkpoint = -1;
                    int checkpoint_rank = 0;
                    if (checkpoint_state_count > 0) {
                        int candidate_checkpoint = checkpoint_token_indices[owner];
                        if (candidate_checkpoint > owner_lo && candidate_checkpoint < owner_hi && candidate_checkpoint % 128 != 0) {
                            owner_checkpoint = candidate_checkpoint;
                            owner_count = owner_count + 1;
                            checkpoint_rank = candidate_checkpoint / 128 - owner_first_chunk + 1;
                        }
                    }
                    int ordinal = segment - smem_sequence_excl[owner_slot];
                    derived_start = (owner_first_chunk + ordinal) * 128;
                    if (ordinal == 0) {
                        derived_start = owner_lo;
                    }
                    if (owner_checkpoint >= 0) {
                        if (ordinal == checkpoint_rank) {
                            derived_start = owner_checkpoint;
                        }
                        if (ordinal > checkpoint_rank) {
                            derived_start = (owner_first_chunk + ordinal - 1) * 128;
                        }
                    }
                    int next_ordinal = ordinal + 1;
                    derived_end = owner_hi;
                    if (next_ordinal < owner_count) {
                        derived_end = (owner_first_chunk + next_ordinal) * 128;
                        if (owner_checkpoint >= 0) {
                            if (next_ordinal == checkpoint_rank) {
                                derived_end = owner_checkpoint;
                            }
                            if (next_ordinal > checkpoint_rank) {
                                derived_end = (owner_first_chunk + next_ordinal - 1) * 128;
                            }
                        }
                    }
                    found = 1;
                }
            }
            carry = carry + block_total;
            asm volatile("barrier.sync 8, 128;" ::: "memory");
        }
        segment_count = carry;
        if (segment_count > segment_bound) {
            flagged_cu = 1;
            segment_count = segment_bound;
        }
        if (flagged_cu != 0) {
            preprocess_status[0] = 1;
        }
        if (tile == 0) {
            if (segment_count == 0) {
                chunk_indices[0] = -1;
            }
        }
    }
    int _shfl_0 = __shfl_sync(0xFFFFFFFF, segment_count, 0);
    segment_count = _shfl_0;
    int total_tiles = segment_count * nheads;
    if (tile < total_tiles) {
        int start = 0;
        int length = 128;
        int physical_start = 0;
        int segment_offset = 0;
        if (metadata_from_cu_seqlens != 0) {
            start = derived_start;
            length = derived_end - derived_start;
            physical_start = start / 128 * 128;
            segment_offset = start - physical_start;
            if (head == 0) {
                chunk_indices[segment] = start / 128;
                chunk_offsets[segment] = segment_offset;
                if (segment == segment_count - 1) {
                    chunk_indices[segment_count] = -1;
                }
            }
        } else if (direct_varlen_metadata != 0) {
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
        if (write_seq_chunk_cumsum != 0) {
            if (head == 0) {
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
                int _max_4 = ((sequence) > (-1) ? (sequence) : (-1));
                int _min_6 = ((_max_4) < (num_sequences) ? (_max_4) : (num_sequences));
                sequence = _min_6;
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
                    int _max_5 = ((previous_sequence) > (-1) ? (previous_sequence) : (-1));
                    int _min_7 = ((_max_5) < (num_sequences) ? (_max_5) : (num_sequences));
                    previous_sequence = _min_7;
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
                    seq_chunk_cumsum[opened_id] = segment;
                }
                if (segment == num_segments - 1) {
                    #pragma unroll 1
                    for (int closed_id = sequence + 1; closed_id < num_sequences + 1; closed_id++) {
                        seq_chunk_cumsum[closed_id] = num_segments;
                    }
                }
            }
        }
        float a_value = A[head];
        float bias_value = dt_bias[head];
        float running = 0.0f;
        float segment_base = 0.0f;
        int last_token = segment_offset + length - 1;
        float group_raw[16];
        float next_raw[16];
        float after_raw[16];
        float group_dt[16];
        float group_product[16];
        float scan_offset_1[16];
        float scan_offset_2[16];
        float scan_offset_4[16];
        float group_scan[16];
        float group_dt_4[4];
        float group_product_4[4];
        float group_scan_4[4];
        #pragma unroll
        for (int local = 0; local < 16; local++) {
            int _min_8 = ((last_token) < (local) ? (last_token) : (local));
            next_raw[local] = dt[(physical_start + _min_8) * nheads + head];
            int _min_9 = ((last_token) < (16 + local) ? (last_token) : (16 + local));
            after_raw[local] = dt[(physical_start + _min_9) * nheads + head];
        }
        if (nheads >= 16) {
            #pragma unroll 1
            for (int group_start = 0; group_start < 128; group_start += 16) {
                #pragma unroll
                for (int local_1 = 0; local_1 < 16; local_1++) {
                    group_raw[local_1] = next_raw[local_1];
                    next_raw[local_1] = after_raw[local_1];
                }
                if (group_start + 32 < 128) {
                    #pragma unroll
                    for (int local_2 = 0; local_2 < 16; local_2++) {
                        int _min_10 = ((last_token) < (group_start + 32 + local_2) ? (last_token) : (group_start + 32 + local_2));
                        int load_token = _min_10;
                        after_raw[local_2] = dt[(physical_start + load_token) * nheads + head];
                    }
                }
                #pragma unroll
                for (int local_3 = 0; local_3 < 16; local_3++) {
                    int physical_token = group_start + local_3;
                    float transformed = 0.0f;
                    if (physical_token < segment_offset + length) {
                        float biased = group_raw[local_3] + bias_value;
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
                    group_dt[local_3] = transformed;
                    group_product[local_3] = transformed * a_value;
                    if (physical_token >= segment_offset && physical_token < segment_offset + length) {
                        int local_token = physical_token - segment_offset;
                        float _max_6 = max_noftz(transformed, -65504.0f);
                        float _min_11 = fminf(_max_6, 65504.0f);
                        smem_delta[slot_base_delta + local_token] = (__half)_min_11;
                    }
                }
                scan_offset_1[0] = group_product[0];
                #pragma unroll
                for (int local_4 = 1; local_4 < 16; local_4++) {
                    float _fma_0 = __fmaf_rn(group_dt[local_4], a_value, group_product[local_4 - 1]);
                    scan_offset_1[local_4] = _fma_0;
                }
                #pragma unroll
                for (int local_5 = 0; local_5 < 16; local_5++) {
                    if (local_5 < 2) {
                        scan_offset_2[local_5] = scan_offset_1[local_5];
                    } else {
                        scan_offset_2[local_5] = scan_offset_1[local_5 - 2] + scan_offset_1[local_5];
                    }
                }
                #pragma unroll
                for (int local_6 = 0; local_6 < 16; local_6++) {
                    if (local_6 < 4) {
                        scan_offset_4[local_6] = scan_offset_2[local_6];
                    } else {
                        scan_offset_4[local_6] = scan_offset_2[local_6 - 4] + scan_offset_2[local_6];
                    }
                }
                #pragma unroll
                for (int local_7 = 0; local_7 < 16; local_7++) {
                    if (local_7 < 8) {
                        group_scan[local_7] = scan_offset_4[local_7];
                    } else {
                        group_scan[local_7] = scan_offset_4[local_7 - 8] + scan_offset_4[local_7];
                    }
                }
                #pragma unroll
                for (int local_8 = 0; local_8 < 16; local_8++) {
                    int physical_token_1 = group_start + local_8;
                    float physical_cumsum = group_scan[local_8];
                    if (group_start != 0) {
                        physical_cumsum += running;
                    }
                    if (physical_token_1 == segment_offset - 1) {
                        segment_base = physical_cumsum;
                    }
                    if (physical_token_1 >= segment_offset && physical_token_1 < segment_offset + length) {
                        int local_token_1 = physical_token_1 - segment_offset;
                        smem_cumsum[slot_base_cumsum + local_token_1] = physical_cumsum - segment_base;
                    }
                }
                if (group_start == 0) {
                    running = group_scan[15];
                } else {
                    running += group_scan[15];
                }
            }
        } else {
            #pragma unroll 1
            for (int block_start = 0; block_start < 128; block_start += 16) {
                #pragma unroll
                for (int local_9 = 0; local_9 < 16; local_9++) {
                    group_raw[local_9] = next_raw[local_9];
                    next_raw[local_9] = after_raw[local_9];
                }
                if (block_start + 32 < 128) {
                    #pragma unroll
                    for (int local_10 = 0; local_10 < 16; local_10++) {
                        int _min_12 = ((last_token) < (block_start + 32 + local_10) ? (last_token) : (block_start + 32 + local_10));
                        int load_token_1 = _min_12;
                        after_raw[local_10] = dt[(physical_start + load_token_1) * nheads + head];
                    }
                }
                #pragma unroll
                for (int group = 0; group < 16; group += 4) {
                    int group_start_1 = block_start + group;
                    #pragma unroll
                    for (int local_11 = 0; local_11 < 4; local_11++) {
                        int physical_token_2 = group_start_1 + local_11;
                        float transformed_1 = 0.0f;
                        if (physical_token_2 < segment_offset + length) {
                            float biased_1 = group_raw[group + local_11] + bias_value;
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
                        group_dt_4[local_11] = transformed_1;
                        group_product_4[local_11] = transformed_1 * a_value;
                        if (physical_token_2 >= segment_offset && physical_token_2 < segment_offset + length) {
                            int local_token_2 = physical_token_2 - segment_offset;
                            float _max_7 = max_noftz(transformed_1, -65504.0f);
                            float _min_13 = fminf(_max_7, 65504.0f);
                            smem_delta[slot_base_delta + local_token_2] = (__half)_min_13;
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
                    for (int local_12 = 0; local_12 < 4; local_12++) {
                        int physical_token_3 = group_start_1 + local_12;
                        float physical_cumsum_1 = group_scan_4[local_12];
                        if (group_start_1 != 0) {
                            physical_cumsum_1 += running;
                        }
                        if (physical_token_3 == segment_offset - 1) {
                            segment_base = physical_cumsum_1;
                        }
                        if (physical_token_3 >= segment_offset && physical_token_3 < segment_offset + length) {
                            int local_token_3 = physical_token_3 - segment_offset;
                            smem_cumsum[slot_base_cumsum + local_token_3] = physical_cumsum_1 - segment_base;
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
        if (length < 128) {
            #pragma unroll 4
            for (int slot = 0; slot < 128; slot++) {
                if (length <= slot) {
                    smem_cumsum[slot_base_cumsum + slot] = 0.0f;
                    smem_delta[slot_base_delta + slot] = zero_half;
                }
            }
        }
    }
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    int first_tile = bid * 128;
    int column = tid;
    #pragma unroll 8
    for (int row = 0; row < 128; row++) {
        int row_tile = first_tile + row;
        if (row_tile < total_tiles) {
            float cumsum_value = smem_cumsum[row * 129 + column];
            __half delta_value = smem_delta[row * 130 + column];
            cumsum[row_tile * 128 + column] = cumsum_value;
            delta[row_tile * 128 + column] = delta_value;
        }
    }
}

} // extern "C"

