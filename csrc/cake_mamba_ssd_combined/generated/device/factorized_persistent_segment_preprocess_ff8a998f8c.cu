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
#define SMEM_SMEM_SEQUENCE_EXCL_OFF 0
#define SMEM_SMEM_SEQUENCE_EXCL_STAGE_BYTES 1028
#define SMEM_SMEM_SEQUENCE_EXCL_STRIDE 1028
#define SMEM_SMEM_WARP_TOTALS_OFF 1040
#define SMEM_SMEM_WARP_TOTALS_STAGE_BYTES 32
#define SMEM_SMEM_WARP_TOTALS_STRIDE 32
#define SMEM_SMEM_DT_STAGE_OFF 1072
#define SMEM_SMEM_DT_STAGE_STAGE_BYTES 4096
#define SMEM_SMEM_DT_STAGE_STRIDE 4096
#define SMEM_TOTAL 5248
#define THREADS 256

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

__global__ __launch_bounds__(THREADS) void
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
    int* smem_sequence_excl = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_SEQUENCE_EXCL_OFF);
    const int smem_sequence_excl_addr = smem + SMEM_SMEM_SEQUENCE_EXCL_OFF;
    int* smem_warp_totals = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_WARP_TOTALS_OFF);
    const int smem_warp_totals_addr = smem + SMEM_SMEM_WARP_TOTALS_OFF;
    float* smem_dt_stage = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_DT_STAGE_OFF);
    const int smem_dt_stage_addr = smem + SMEM_SMEM_DT_STAGE_OFF;

    // === Task calls (dependency order) ===
    int tile_warp = tid / 32;
    int tile_lane = lane;
    int tile = bid * 8 + tile_warp;
    int segment = tile / nheads;
    int head = tile % nheads;
    int segment_count = num_segments;
    int derived_start = 0;
    int derived_end = 0;
    __half zero_half = (__half)0.0f;
    if (metadata_from_cu_seqlens != 0) {
        int segment_bound = num_segments;
        int warp_1 = tid / 32;
        int lane_0 = lane;
        int flagged_cu = 0;
        int carry = 0;
        int found = 0;
        #pragma unroll 1
        for (int block_base = 0; block_base < num_sequences; block_base += 256) {
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
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            int warp_prefix = 0;
            int block_total = 0;
            #pragma unroll
            for (int other_warp = 0; other_warp < 8; other_warp++) {
                int other_total = smem_warp_totals[other_warp];
                block_total = block_total + other_total;
                if (warp_1 > other_warp) {
                    warp_prefix = warp_prefix + other_total;
                }
            }
            int exclusive = carry + warp_prefix + inclusive - count;
            smem_sequence_excl[tid] = exclusive;
            if (tid == 0) {
                smem_sequence_excl[256] = carry + block_total;
            }
            if (bid == 0) {
                if (sequence_slot < num_sequences) {
                    int _min_2 = ((exclusive) < (segment_bound) ? (exclusive) : (segment_bound));
                    seq_chunk_cumsum[sequence_slot] = _min_2;
                }
                if (tid == 0) {
                    if (block_base + 256 >= num_sequences) {
                        int _min_3 = ((carry + block_total) < (segment_bound) ? (carry + block_total) : (segment_bound));
                        seq_chunk_cumsum[num_sequences] = _min_3;
                    }
                }
            }
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            if (found == 0) {
                if (segment >= carry && segment < carry + block_total) {
                    int owner_slot = 0;
                    int low = 0;
                    int high = 255;
                    #pragma unroll
                    for (int probe = 0; probe < 8; probe++) {
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
            asm volatile("barrier.sync 8, 256;" ::: "memory");
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
            if (tile_lane == 0) {
                if (segment_count == 0) {
                    chunk_indices[0] = -1;
                }
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
                if (tile_lane == 0) {
                    chunk_indices[segment] = start / 128;
                    chunk_offsets[segment] = segment_offset;
                    if (segment == segment_count - 1) {
                        chunk_indices[segment_count] = -1;
                    }
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
            if (head == 0 && tile_lane == 0) {
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
        int segment_end = segment_offset + length;
        int last_token = segment_end - 1;
        int token_base = tile_lane * 4;
        float lane_raw[4];
        if (nheads % 8 == 0) {
            int stage_token = tid / 2;
            int stage_half = tid % 2;
            int head_base = head - tile_warp;
            int _min_8 = ((last_token) < (stage_token) ? (last_token) : (stage_token));
            float _vec_load_0[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dt + (physical_start + _min_8) * nheads + head_base + stage_half * 4);
                _vec_load_0[0 + 0] = _v4.x;
                _vec_load_0[0 + 1] = _v4.y;
                _vec_load_0[0 + 2] = _v4.z;
                _vec_load_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int quad = 0; quad < 4; quad++) {
                smem_dt_stage[(stage_half * 4 + quad) * 128 + stage_token] = _vec_load_0[quad];
            }
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lane_raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&lane_raw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_raw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_raw[(0) + 3]))
                : "r"(smem_dt_stage_addr + (unsigned int)((tile_warp * 128 + token_base) * 4)));
        } else {
            #pragma unroll
            for (int local = 0; local < 4; local++) {
                int _min_9 = ((last_token) < (token_base + local) ? (last_token) : (token_base + local));
                lane_raw[local] = dt[(physical_start + _min_9) * nheads + head];
            }
        }
        float lane_dt[4];
        float lane_product[4];
        #pragma unroll
        for (int local_1 = 0; local_1 < 4; local_1++) {
            int physical_token = token_base + local_1;
            float transformed = 0.0f;
            if (physical_token < segment_end) {
                float biased = lane_raw[local_1] + bias_value;
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
            lane_dt[local_1] = transformed;
            lane_product[local_1] = transformed * a_value;
        }
        float lane_scan[4];
        float running = 0.0f;
        int has_carry = 0;
        if (nheads >= 16) {
            int block_lane = tile_lane % 4;
            float _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, lane_product[3], 1, 4);
            float previous_product = _shfl_up_5;
            float scan_offset_1[4];
            scan_offset_1[0] = lane_product[0];
            if (block_lane >= 1) {
                float _fma_0 = __fmaf_rn(lane_dt[0], a_value, previous_product);
                scan_offset_1[0] = _fma_0;
            }
            #pragma unroll
            for (int local_2 = 1; local_2 < 4; local_2++) {
                float _fma_1 = __fmaf_rn(lane_dt[local_2], a_value, lane_product[local_2 - 1]);
                scan_offset_1[local_2] = _fma_1;
            }
            float _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, scan_offset_1[2], 1, 4);
            float previous_scan_1_2 = _shfl_up_6;
            float _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, scan_offset_1[3], 1, 4);
            float previous_scan_1_3 = _shfl_up_7;
            float scan_offset_2[4];
            scan_offset_2[0] = scan_offset_1[0];
            scan_offset_2[1] = scan_offset_1[1];
            if (block_lane >= 1) {
                scan_offset_2[0] = previous_scan_1_2 + scan_offset_1[0];
                scan_offset_2[1] = previous_scan_1_3 + scan_offset_1[1];
            }
            scan_offset_2[2] = scan_offset_1[0] + scan_offset_1[2];
            scan_offset_2[3] = scan_offset_1[1] + scan_offset_1[3];
            float scan_offset_4[4];
            #pragma unroll
            for (int local_3 = 0; local_3 < 4; local_3++) {
                float _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, scan_offset_2[local_3], 1, 4);
                float previous_scan_2 = _shfl_up_8;
                scan_offset_4[local_3] = scan_offset_2[local_3];
                if (block_lane >= 1) {
                    scan_offset_4[local_3] = previous_scan_2 + scan_offset_2[local_3];
                }
            }
            #pragma unroll
            for (int local_4 = 0; local_4 < 4; local_4++) {
                float _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, scan_offset_4[local_4], 2, 4);
                float previous_scan_4 = _shfl_up_9;
                lane_scan[local_4] = scan_offset_4[local_4];
                if (block_lane >= 2) {
                    lane_scan[local_4] = previous_scan_4 + scan_offset_4[local_4];
                }
            }
            int block_index = tile_lane / 4;
            #pragma unroll
            for (int block = 0; block < 8; block++) {
                float _shfl_1 = __shfl_sync(0xFFFFFFFF, lane_scan[3], block * 4 + 3);
                float block_total_1 = _shfl_1;
                if (block_index > block) {
                    if (block == 0) {
                        running = block_total_1;
                    } else {
                        running = running + block_total_1;
                    }
                }
            }
            if (block_index > 0) {
                has_carry = 1;
            }
        } else {
            lane_scan[0] = lane_product[0];
            float _fma_2 = __fmaf_rn(lane_dt[1], a_value, lane_product[0]);
            lane_scan[1] = _fma_2;
            float _fma_3 = __fmaf_rn(lane_dt[2], a_value, lane_product[1]);
            float group_pair_12 = _fma_3;
            lane_scan[2] = group_pair_12 + lane_product[0];
            float _fma_4 = __fmaf_rn(lane_dt[3], a_value, lane_product[2]);
            float group_pair_23 = _fma_4;
            lane_scan[3] = group_pair_23 + lane_scan[1];
            #pragma unroll
            for (int group = 0; group < 32; group++) {
                float _shfl_2 = __shfl_sync(0xFFFFFFFF, lane_scan[3], group);
                float group_total = _shfl_2;
                if (tile_lane > group) {
                    if (group == 0) {
                        running = group_total;
                    } else {
                        running = running + group_total;
                    }
                }
            }
            if (tile_lane > 0) {
                has_carry = 1;
            }
        }
        float physical_cumsum[4];
        #pragma unroll
        for (int local_5 = 0; local_5 < 4; local_5++) {
            float value = lane_scan[local_5];
            if (has_carry != 0) {
                value = value + running;
            }
            physical_cumsum[local_5] = value;
        }
        int _max_6 = ((segment_offset - 1) > (0) ? (segment_offset - 1) : (0));
        int base_token = _max_6;
        float base_candidate = physical_cumsum[0];
        #pragma unroll
        for (int local_6 = 1; local_6 < 4; local_6++) {
            if (base_token % 4 == local_6) {
                base_candidate = physical_cumsum[local_6];
            }
        }
        float _shfl_3 = __shfl_sync(0xFFFFFFFF, base_candidate, base_token / 4);
        float base_broadcast = _shfl_3;
        float segment_base = 0.0f;
        if (segment_offset > 0) {
            segment_base = base_broadcast;
        }
        int row_base = tile * 128;
        #pragma unroll
        for (int local_7 = 0; local_7 < 4; local_7++) {
            int physical_token_1 = token_base + local_7;
            if (physical_token_1 >= segment_offset && physical_token_1 < segment_end) {
                int local_token = physical_token_1 - segment_offset;
                cumsum[row_base + local_token] = physical_cumsum[local_7] - segment_base;
                float _max_7 = max_noftz(lane_dt[local_7], -65504.0f);
                float _min_10 = fminf(_max_7, 65504.0f);
                float bounded = _min_10;
                delta[row_base + local_token] = (__half)bounded;
            }
            if (physical_token_1 >= length) {
                cumsum[row_base + physical_token_1] = 0.0f;
                delta[row_base + physical_token_1] = zero_half;
            }
        }
    }
}

} // extern "C"

