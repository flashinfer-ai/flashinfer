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

extern "C" {

__global__ __launch_bounds__(256) void
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
    int* smem_sequence_excl = reinterpret_cast<int*>(smem_raw + 0);
    const int smem_sequence_excl_addr = smem + 0;
    int* smem_warp_totals = reinterpret_cast<int*>(smem_raw + 1040);
    const int smem_warp_totals_addr = smem + 1040;
    float* smem_dt_stage = reinterpret_cast<float*>(smem_raw + 1072);
    const int smem_dt_stage_addr = smem + 1072;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int tile_warp = tid / 32;
    int tile_lane = lane;
    int tile = bid * 8 + tile_warp;
    int segment = tile / nheads;
    int head = tile % nheads;
    int segment_count = num_segments;
    int derived_start = 0;
    int derived_end = 0;
    __half zero_half = (__half)0.0f;
    int spec_dt_on = 0;
    int spec_dt_token = 0;
    float spec_dt[4];
    int aligned_table = 0;
    if (metadata_from_cu_seqlens == 0 && segment * 128 < seqlen && num_segments * 128 < seqlen + 128) {
        aligned_table = 1;
    }
    if (nheads % 8 == 0 && aligned_table != 0) {
        spec_dt_on = 1;
        int _min_0 = ((segment * 128 + tid / 2) < (seqlen - 1) ? (segment * 128 + tid / 2) : (seqlen - 1));
        spec_dt_token = _min_0;
        {
            float4 _v4 = *reinterpret_cast<const float4*>(dt + spec_dt_token * nheads + (head - tile_warp) + tid % 2 * 4);
            spec_dt[0 + 0] = _v4.x;
            spec_dt[0 + 1] = _v4.y;
            spec_dt[0 + 2] = _v4.z;
            spec_dt[0 + 3] = _v4.w;
        }
    }
    int spec_seq_on = 0;
    long long spec_sequence = 0;
    long long spec_previous = 0;
    int previous_chunk_index = 0;
    int previous_chunk_offset = 0;
    if (write_seq_chunk_cumsum != 0 && head == 0 && tile_lane == 0 && metadata_from_cu_seqlens == 0) {
        if (segment != 0 && segment < num_segments) {
            previous_chunk_index = chunk_indices[segment - 1];
            previous_chunk_offset = chunk_offsets[segment - 1];
        }
        if (aligned_table != 0) {
            spec_seq_on = 1;
            if (seq_idx_int64 != 0) {
                spec_sequence = seq_idx_i64[segment * 128];
                if (segment != 0) {
                    spec_previous = seq_idx_i64[(segment - 1) * 128];
                }
            } else {
                spec_sequence = (long long)seq_idx_i32[segment * 128];
                if (segment != 0) {
                    spec_previous = (long long)seq_idx_i32[(segment - 1) * 128];
                }
            }
        }
    }
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
                int _max_2 = ((raw_lo) > (0) ? (raw_lo) : (0));
                int lo = ((_max_2) < (seqlen) ? (_max_2) : (seqlen));
                int _max_3 = ((raw_hi) > (0) ? (raw_hi) : (0));
                int hi = ((_max_3) < (seqlen) ? (_max_3) : (seqlen));
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
            {
                uint32_t inclusive_scan = (uint32_t)inclusive;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_scan, 1, 32); if (lane >= 1) { inclusive_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_scan, 2, 32); if (lane >= 2) { inclusive_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_scan, 4, 32); if (lane >= 4) { inclusive_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_scan, 8, 32); if (lane >= 8) { inclusive_scan += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_scan, 16, 32); if (lane >= 16) { inclusive_scan += peer; } }
                inclusive = (int)inclusive_scan;
            }
            if (lane_0 == 31) {
                smem_warp_totals[warp_1] = inclusive;
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
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
                    int _min_4 = ((exclusive) < (segment_bound) ? (exclusive) : (segment_bound));
                    seq_chunk_cumsum[sequence_slot] = _min_4;
                }
                if (tid == 0) {
                    if (block_base + 256 >= num_sequences) {
                        int _min_5 = ((carry + block_total) < (segment_bound) ? (carry + block_total) : (segment_bound));
                        seq_chunk_cumsum[num_sequences] = _min_5;
                    }
                }
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
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
                    int _max_6 = ((cu_seqlens[owner]) > (0) ? (cu_seqlens[owner]) : (0));
                    int owner_lo = ((_max_6) < (seqlen) ? (_max_6) : (seqlen));
                    int _max_7 = ((cu_seqlens[owner + 1]) > (0) ? (cu_seqlens[owner + 1]) : (0));
                    int owner_hi = ((_max_7) < (seqlen) ? (_max_7) : (seqlen));
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
            asm volatile("barrier.sync 1, 256;" ::: "memory");
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
    int _shfl_sync_8 = __shfl_sync(0xFFFFFFFF, segment_count, 0);
    segment_count = _shfl_sync_8;
    int total_tiles = segment_count * nheads;
    if (tile < total_tiles) {
        float a_value = A[head];
        float bias_value = dt_bias[head];
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
        int segment_end = segment_offset + length;
        int last_token = segment_end - 1;
        int token_base = tile_lane * 4;
        float lane_raw[4];
        if (nheads % 8 == 0) {
            int stage_token = tid / 2;
            int stage_half = tid % 2;
            int head_base = head - tile_warp;
            int _min_0_1 = ((last_token) < (stage_token) ? (last_token) : (stage_token));
            float _vec_load_0[4];
            if (spec_dt_on != 0 && physical_start + _min_0_1 == spec_dt_token) {
                #pragma unroll
                for (int quad = 0; quad < 4; quad++) {
                    _vec_load_0[quad] = spec_dt[quad];
                }
            } else {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dt + (physical_start + _min_0_1) * nheads + head_base + stage_half * 4);
                    _vec_load_0[0 + 0] = _v4.x;
                    _vec_load_0[0 + 1] = _v4.y;
                    _vec_load_0[0 + 2] = _v4.z;
                    _vec_load_0[0 + 3] = _v4.w;
                }
            }
            if (write_seq_chunk_cumsum != 0) {
                if (head == 0 && tile_lane == 0) {
                    int sequence = 0;
                    if (seq_idx_int64 != 0) {
                        long long raw_sequence = spec_sequence;
                        if (spec_seq_on == 0 || start != segment * 128) {
                            raw_sequence = seq_idx_i64[start];
                        }
                        sequence = (int)raw_sequence;
                        if (raw_sequence < 0) {
                            sequence = -1;
                        }
                        if (raw_sequence >= (long long)num_sequences) {
                            sequence = num_sequences;
                        }
                    } else {
                        sequence = (int)spec_sequence;
                        if (spec_seq_on == 0 || start != segment * 128) {
                            sequence = seq_idx_i32[start];
                        }
                    }
                    int _max_11 = ((sequence) > (-1) ? (sequence) : (-1));
                    int _min_10 = ((_max_11) < (num_sequences) ? (_max_11) : (num_sequences));
                    sequence = _min_10;
                    int previous_sequence = -1;
                    if (segment != 0) {
                        int previous_start = previous_chunk_index * 128 + previous_chunk_offset;
                        if (metadata_from_cu_seqlens != 0) {
                            previous_start = chunk_indices[segment - 1] * 128 + chunk_offsets[segment - 1];
                        }
                        if (seq_idx_int64 != 0) {
                            long long raw_previous = spec_previous;
                            if (spec_seq_on == 0 || previous_start != (segment - 1) * 128) {
                                raw_previous = seq_idx_i64[previous_start];
                            }
                            previous_sequence = (int)raw_previous;
                            if (raw_previous < 0) {
                                previous_sequence = -1;
                            }
                            if (raw_previous >= (long long)num_sequences) {
                                previous_sequence = num_sequences;
                            }
                        } else {
                            previous_sequence = (int)spec_previous;
                            if (spec_seq_on == 0 || previous_start != (segment - 1) * 128) {
                                previous_sequence = seq_idx_i32[previous_start];
                            }
                        }
                        int _max_13 = ((previous_sequence) > (-1) ? (previous_sequence) : (-1));
                        int _min_12 = ((_max_13) < (num_sequences) ? (_max_13) : (num_sequences));
                        previous_sequence = _min_12;
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
            #pragma unroll
            for (int quad_1 = 0; quad_1 < 4; quad_1++) {
                smem_dt_stage[(stage_half * 4 + quad_1) * 128 + stage_token] = _vec_load_0[quad_1];
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lane_raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&lane_raw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_raw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_raw[(0) + 3]))
                : "r"(smem_dt_stage_addr + (unsigned int)((tile_warp * 128 + token_base) * 4)));
        } else {
            #pragma unroll
            for (int local = 0; local < 4; local++) {
                int _min_14 = ((last_token) < (token_base + local) ? (last_token) : (token_base + local));
                lane_raw[local] = dt[(physical_start + _min_14) * nheads + head];
            }
            if (write_seq_chunk_cumsum != 0) {
                if (head == 0 && tile_lane == 0) {
                    int sequence_1 = 0;
                    if (seq_idx_int64 != 0) {
                        long long raw_sequence_1 = spec_sequence;
                        if (spec_seq_on == 0 || start != segment * 128) {
                            raw_sequence_1 = seq_idx_i64[start];
                        }
                        sequence_1 = (int)raw_sequence_1;
                        if (raw_sequence_1 < 0) {
                            sequence_1 = -1;
                        }
                        if (raw_sequence_1 >= (long long)num_sequences) {
                            sequence_1 = num_sequences;
                        }
                    } else {
                        sequence_1 = (int)spec_sequence;
                        if (spec_seq_on == 0 || start != segment * 128) {
                            sequence_1 = seq_idx_i32[start];
                        }
                    }
                    int _max_16 = ((sequence_1) > (-1) ? (sequence_1) : (-1));
                    int _min_15 = ((_max_16) < (num_sequences) ? (_max_16) : (num_sequences));
                    sequence_1 = _min_15;
                    int previous_sequence_1 = -1;
                    if (segment != 0) {
                        int previous_start_1 = previous_chunk_index * 128 + previous_chunk_offset;
                        if (metadata_from_cu_seqlens != 0) {
                            previous_start_1 = chunk_indices[segment - 1] * 128 + chunk_offsets[segment - 1];
                        }
                        if (seq_idx_int64 != 0) {
                            long long raw_previous_1 = spec_previous;
                            if (spec_seq_on == 0 || previous_start_1 != (segment - 1) * 128) {
                                raw_previous_1 = seq_idx_i64[previous_start_1];
                            }
                            previous_sequence_1 = (int)raw_previous_1;
                            if (raw_previous_1 < 0) {
                                previous_sequence_1 = -1;
                            }
                            if (raw_previous_1 >= (long long)num_sequences) {
                                previous_sequence_1 = num_sequences;
                            }
                        } else {
                            previous_sequence_1 = (int)spec_previous;
                            if (spec_seq_on == 0 || previous_start_1 != (segment - 1) * 128) {
                                previous_sequence_1 = seq_idx_i32[previous_start_1];
                            }
                        }
                        int _max_18 = ((previous_sequence_1) > (-1) ? (previous_sequence_1) : (-1));
                        int _min_17 = ((_max_18) < (num_sequences) ? (_max_18) : (num_sequences));
                        previous_sequence_1 = _min_17;
                    }
                    int flagged_1 = 0;
                    if (sequence_1 < 0) {
                        flagged_1 = 1;
                    }
                    if (sequence_1 >= num_sequences) {
                        flagged_1 = 1;
                    }
                    if (sequence_1 < previous_sequence_1) {
                        flagged_1 = 1;
                    }
                    if (flagged_1 != 0) {
                        preprocess_status[0] = 1;
                    }
                    #pragma unroll 1
                    for (int opened_id_1 = previous_sequence_1 + 1; opened_id_1 < sequence_1 + 1; opened_id_1++) {
                        seq_chunk_cumsum[opened_id_1] = segment;
                    }
                    if (segment == num_segments - 1) {
                        #pragma unroll 1
                        for (int closed_id_1 = sequence_1 + 1; closed_id_1 < num_sequences + 1; closed_id_1++) {
                            seq_chunk_cumsum[closed_id_1] = num_segments;
                        }
                    }
                }
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
                        float _exp2_20 = approx_exp2(biased * 1.442695f);
                        float _log_19 = logf(_exp2_20 + 1.0f);
                        transformed = _log_19;
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
            float previous_product = __shfl_up_sync(0xFFFFFFFF, lane_product[3], 1, 4);
            float scan_offset_1[4];
            scan_offset_1[0] = lane_product[0];
            if (block_lane >= 1) {
                scan_offset_1[0] = __fmaf_rn(lane_dt[0], a_value, previous_product);
            }
            #pragma unroll
            for (int local_2 = 1; local_2 < 4; local_2++) {
                scan_offset_1[local_2] = __fmaf_rn(lane_dt[local_2], a_value, lane_product[local_2 - 1]);
            }
            float previous_scan_1_2 = __shfl_up_sync(0xFFFFFFFF, scan_offset_1[2], 1, 4);
            float previous_scan_1_3 = __shfl_up_sync(0xFFFFFFFF, scan_offset_1[3], 1, 4);
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
                float previous_scan_2 = __shfl_up_sync(0xFFFFFFFF, scan_offset_2[local_3], 1, 4);
                scan_offset_4[local_3] = scan_offset_2[local_3];
                if (block_lane >= 1) {
                    scan_offset_4[local_3] = previous_scan_2 + scan_offset_2[local_3];
                }
            }
            #pragma unroll
            for (int local_4 = 0; local_4 < 4; local_4++) {
                float previous_scan_4 = __shfl_up_sync(0xFFFFFFFF, scan_offset_4[local_4], 2, 4);
                lane_scan[local_4] = scan_offset_4[local_4];
                if (block_lane >= 2) {
                    lane_scan[local_4] = previous_scan_4 + scan_offset_4[local_4];
                }
            }
            int block_index = tile_lane / 4;
            #pragma unroll
            for (int block = 0; block < 8; block++) {
                float block_total_1 = __shfl_sync(0xFFFFFFFF, lane_scan[3], block * 4 + 3);
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
            lane_scan[1] = __fmaf_rn(lane_dt[1], a_value, lane_product[0]);
            float group_pair_12 = __fmaf_rn(lane_dt[2], a_value, lane_product[1]);
            lane_scan[2] = group_pair_12 + lane_product[0];
            float group_pair_23 = __fmaf_rn(lane_dt[3], a_value, lane_product[2]);
            lane_scan[3] = group_pair_23 + lane_scan[1];
            #pragma unroll
            for (int group = 0; group < 32; group++) {
                float group_total = __shfl_sync(0xFFFFFFFF, lane_scan[3], group);
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
        int base_token = ((segment_offset - 1) > (0) ? (segment_offset - 1) : (0));
        float base_candidate = physical_cumsum[0];
        #pragma unroll
        for (int local_6 = 1; local_6 < 4; local_6++) {
            if (base_token % 4 == local_6) {
                base_candidate = physical_cumsum[local_6];
            }
        }
        float base_broadcast = __shfl_sync(0xFFFFFFFF, base_candidate, base_token / 4);
        float segment_base = 0.0f;
        if (segment_offset > 0) {
            segment_base = base_broadcast;
        }
        int row_base = tile * 128;
        float row_cumsum[4];
        __half row_delta[4];
        if (segment_offset == 0) {
            #pragma unroll
            for (int local_7 = 0; local_7 < 4; local_7++) {
                row_cumsum[local_7] = 0.0f;
                row_delta[local_7] = zero_half;
                if (token_base + local_7 < segment_end) {
                    row_cumsum[local_7] = physical_cumsum[local_7] - segment_base;
                    float bounded = fminf(fmaxf(lane_dt[local_7], -65504.0f), 65504.0f);
                    row_delta[local_7] = (__half)bounded;
                }
            }
            {
                float4 _v4 = make_float4(row_cumsum[0 + 0], row_cumsum[0 + 1], row_cumsum[0 + 2], row_cumsum[0 + 3]);
                *reinterpret_cast<float4*>(cumsum + row_base + token_base) = _v4;
            }
            reinterpret_cast<int2*>(delta + (row_base + token_base))[0] = reinterpret_cast<int2*>(row_delta)[0];
        } else if (segment_end <= 128) {
            #pragma unroll
            for (int local_8 = 0; local_8 < 4; local_8++) {
                int source_item = (local_8 + segment_offset) % 4;
                int source_shift = (local_8 + segment_offset) / 4;
                float item_cumsum = physical_cumsum[0];
                float item_dt = lane_dt[0];
                #pragma unroll
                for (int other = 1; other < 4; other++) {
                    if (source_item == other) {
                        item_cumsum = physical_cumsum[other];
                        item_dt = lane_dt[other];
                    }
                }
                float moved_cumsum = __shfl_down_sync(0xFFFFFFFF, item_cumsum, source_shift, 32);
                float moved_dt = __shfl_down_sync(0xFFFFFFFF, item_dt, source_shift, 32);
                row_cumsum[local_8] = 0.0f;
                row_delta[local_8] = zero_half;
                if (token_base + local_8 < length) {
                    row_cumsum[local_8] = moved_cumsum - segment_base;
                    float bounded_moved = fminf(fmaxf(moved_dt, -65504.0f), 65504.0f);
                    row_delta[local_8] = (__half)bounded_moved;
                }
            }
            {
                float4 _v4 = make_float4(row_cumsum[0 + 0], row_cumsum[0 + 1], row_cumsum[0 + 2], row_cumsum[0 + 3]);
                *reinterpret_cast<float4*>(cumsum + row_base + token_base) = _v4;
            }
            reinterpret_cast<int2*>(delta + (row_base + token_base))[0] = reinterpret_cast<int2*>(row_delta)[0];
        } else {
            #pragma unroll
            for (int local_9 = 0; local_9 < 4; local_9++) {
                int physical_token_1 = token_base + local_9;
                if (physical_token_1 >= segment_offset && physical_token_1 < segment_end) {
                    int local_token = physical_token_1 - segment_offset;
                    cumsum[row_base + local_token] = physical_cumsum[local_9] - segment_base;
                    float bounded_1 = fminf(fmaxf(lane_dt[local_9], -65504.0f), 65504.0f);
                    delta[row_base + local_token] = (__half)bounded_1;
                }
                if (physical_token_1 >= length) {
                    cumsum[row_base + physical_token_1] = 0.0f;
                    delta[row_base + physical_token_1] = zero_half;
                }
            }
        }
    }
}

} // extern "C"

