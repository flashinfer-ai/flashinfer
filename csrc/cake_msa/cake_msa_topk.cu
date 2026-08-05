/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// clang-format off
// Frozen CAKE-generated CUDA device kernel.
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeMsaTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeMsaTensorMapPack { CakeMsaTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CakeMsaGeneratedTensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_MSA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WARP_SCORES_OFF 0
#define SMEM_WARP_SCORES_STAGE_BYTES 32
#define SMEM_WARP_SCORES_STRIDE 32
#define SMEM_WARP_INDICES_OFF 32
#define SMEM_WARP_INDICES_STAGE_BYTES 32
#define SMEM_WARP_INDICES_STRIDE 32
#define SMEM_SELECTED_OFF 64
#define SMEM_SELECTED_STAGE_BYTES 64
#define SMEM_SELECTED_STRIDE 64
#define SMEM_TOTAL 128
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_msa_topk(float* __restrict__ max_score, int* __restrict__ output, int num_heads, int max_k_tiles, int total_q, int num_valid_pages, int force_begin_blocks, int force_end_blocks)
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
    float* warp_scores = reinterpret_cast<float*>(smem_raw + 0);
    const int warp_scores_addr = smem + 0;
    int* warp_indices = reinterpret_cast<int*>(smem_raw + 32);
    const int warp_indices_addr = smem + 32;
    int* selected = reinterpret_cast<int*>(smem_raw + 64);
    const int selected_addr = smem + 64;

    // === Task calls (dependency order) ===
    int tid_0 = tid;
    int lane_1 = lane;
    int warp_2 = warp;
    int row = bid;
    int query = row / num_heads;
    int head = row - query * num_heads;
    int forced = force_begin_blocks + force_end_blocks;
    if (tid_0 < 16) {
        int forced_index = -1;
        if (tid_0 < force_begin_blocks) {
            forced_index = tid_0;
        } else if (tid_0 < forced) {
            forced_index = num_valid_pages - force_end_blocks + (tid_0 - force_begin_blocks);
        }
        selected[tid_0] = forced_index;
    }
    __syncthreads();
    for (int slot = 0; slot < 16; slot++) {
        if (forced <= slot) {
            float local_score = -CAKE_MSA_INF;
            int local_index = max_k_tiles;
            for (int block = tid_0; block < num_valid_pages; block += 256) {
                int already_selected = 0;
                for (int previous = 0; previous < slot; previous++) {
                    already_selected = already_selected | (int)(block == selected[previous]);
                }
                int eligible = block >= force_begin_blocks && block < num_valid_pages - force_end_blocks && already_selected == 0;
                if (eligible != 0) {
                    int score_offset = (head * max_k_tiles + block) * total_q + query;
                    float score = max_score[score_offset];
                    if (score > local_score || score == local_score && local_index > block) {
                        local_score = score;
                        local_index = block;
                    }
                }
            }
            float _warp_reduce_0 = local_score;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
            int index_at_warp_score = max_k_tiles;
            if (local_score == _warp_reduce_0) {
                index_at_warp_score = local_index;
            }
            float _warp_reduce_1 = index_at_warp_score;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_1 = fminf(_warp_reduce_1, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset));
            if (lane_1 == 0) {
                warp_scores[warp_2] = _warp_reduce_0;
                warp_indices[warp_2] = _warp_reduce_1;
            }
            __syncthreads();
            if (warp_2 == 0) {
                float block_score = -CAKE_MSA_INF;
                int block_index = max_k_tiles;
                if (lane_1 < 8) {
                    block_score = warp_scores[lane_1];
                    block_index = warp_indices[lane_1];
                }
                float _warp_reduce_2 = block_score;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_2 = max_noftz(_warp_reduce_2, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset));
                int index_at_best = max_k_tiles;
                if (block_score == _warp_reduce_2) {
                    index_at_best = block_index;
                }
                float _warp_reduce_3 = index_at_best;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_3 = fminf(_warp_reduce_3, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset));
                if (lane_1 == 0) {
                    int selected_index = _warp_reduce_3;
                    if (_warp_reduce_3 >= (float)num_valid_pages) {
                        selected_index = -1;
                    }
                    selected[slot] = selected_index;
                }
            }
            __syncthreads();
        }
    }
    if (warp_2 == 0) {
        int value = max_k_tiles;
        if (lane_1 < 16) {
            int loaded = selected[lane_1];
            if (loaded >= 0) {
                value = loaded;
            }
        }
        int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_0 = ((value) < (_shfl_xor_0) ? (value) : (_shfl_xor_0));
        int lo = _min_0;
        int _max_0 = ((value) > (_shfl_xor_0) ? (value) : (_shfl_xor_0));
        int hi = _max_0;
        int ascending = (int)((lane_1 & 2) == 0);
        int lower_lane = (int)((lane_1 & 1) == 0);
        int take_low = (int)(ascending == lower_lane);
        value = ((take_low == 1) ? lo : hi);
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, value, 2);
        int _min_1 = ((value) < (_shfl_xor_1) ? (value) : (_shfl_xor_1));
        int lo_0 = _min_1;
        int _max_1 = ((value) > (_shfl_xor_1) ? (value) : (_shfl_xor_1));
        int hi_1 = _max_1;
        int ascending_2 = (int)((lane_1 & 4) == 0);
        int lower_lane_3 = (int)((lane_1 & 2) == 0);
        int take_low_4 = (int)(ascending_2 == lower_lane_3);
        value = ((take_low_4 == 1) ? lo_0 : hi_1);
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_2 = ((value) < (_shfl_xor_2) ? (value) : (_shfl_xor_2));
        int lo_5 = _min_2;
        int _max_2 = ((value) > (_shfl_xor_2) ? (value) : (_shfl_xor_2));
        int hi_6 = _max_2;
        int ascending_7 = (int)((lane_1 & 4) == 0);
        int lower_lane_8 = (int)((lane_1 & 1) == 0);
        int take_low_9 = (int)(ascending_7 == lower_lane_8);
        value = ((take_low_9 == 1) ? lo_5 : hi_6);
        int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, value, 4);
        int _min_3 = ((value) < (_shfl_xor_3) ? (value) : (_shfl_xor_3));
        int lo_10 = _min_3;
        int _max_3 = ((value) > (_shfl_xor_3) ? (value) : (_shfl_xor_3));
        int hi_11 = _max_3;
        int ascending_12 = (int)((lane_1 & 8) == 0);
        int lower_lane_13 = (int)((lane_1 & 4) == 0);
        int take_low_14 = (int)(ascending_12 == lower_lane_13);
        value = ((take_low_14 == 1) ? lo_10 : hi_11);
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, value, 2);
        int _min_4 = ((value) < (_shfl_xor_4) ? (value) : (_shfl_xor_4));
        int lo_15 = _min_4;
        int _max_4 = ((value) > (_shfl_xor_4) ? (value) : (_shfl_xor_4));
        int hi_16 = _max_4;
        int ascending_17 = (int)((lane_1 & 8) == 0);
        int lower_lane_18 = (int)((lane_1 & 2) == 0);
        int take_low_19 = (int)(ascending_17 == lower_lane_18);
        value = ((take_low_19 == 1) ? lo_15 : hi_16);
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_5 = ((value) < (_shfl_xor_5) ? (value) : (_shfl_xor_5));
        int lo_20 = _min_5;
        int _max_5 = ((value) > (_shfl_xor_5) ? (value) : (_shfl_xor_5));
        int hi_21 = _max_5;
        int ascending_22 = (int)((lane_1 & 8) == 0);
        int lower_lane_23 = (int)((lane_1 & 1) == 0);
        int take_low_24 = (int)(ascending_22 == lower_lane_23);
        value = ((take_low_24 == 1) ? lo_20 : hi_21);
        int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, value, 8);
        int _min_6 = ((value) < (_shfl_xor_6) ? (value) : (_shfl_xor_6));
        int lo_25 = _min_6;
        int _max_6 = ((value) > (_shfl_xor_6) ? (value) : (_shfl_xor_6));
        int hi_26 = _max_6;
        int ascending_27 = (int)((lane_1 & 16) == 0);
        int lower_lane_28 = (int)((lane_1 & 8) == 0);
        int take_low_29 = (int)(ascending_27 == lower_lane_28);
        value = ((take_low_29 == 1) ? lo_25 : hi_26);
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, value, 4);
        int _min_7 = ((value) < (_shfl_xor_7) ? (value) : (_shfl_xor_7));
        int lo_30 = _min_7;
        int _max_7 = ((value) > (_shfl_xor_7) ? (value) : (_shfl_xor_7));
        int hi_31 = _max_7;
        int ascending_32 = (int)((lane_1 & 16) == 0);
        int lower_lane_33 = (int)((lane_1 & 4) == 0);
        int take_low_34 = (int)(ascending_32 == lower_lane_33);
        value = ((take_low_34 == 1) ? lo_30 : hi_31);
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, value, 2);
        int _min_8 = ((value) < (_shfl_xor_8) ? (value) : (_shfl_xor_8));
        int lo_35 = _min_8;
        int _max_8 = ((value) > (_shfl_xor_8) ? (value) : (_shfl_xor_8));
        int hi_36 = _max_8;
        int ascending_37 = (int)((lane_1 & 16) == 0);
        int lower_lane_38 = (int)((lane_1 & 2) == 0);
        int take_low_39 = (int)(ascending_37 == lower_lane_38);
        value = ((take_low_39 == 1) ? lo_35 : hi_36);
        int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_9 = ((value) < (_shfl_xor_9) ? (value) : (_shfl_xor_9));
        int lo_40 = _min_9;
        int _max_9 = ((value) > (_shfl_xor_9) ? (value) : (_shfl_xor_9));
        int hi_41 = _max_9;
        int ascending_42 = (int)((lane_1 & 16) == 0);
        int lower_lane_43 = (int)((lane_1 & 1) == 0);
        int take_low_44 = (int)(ascending_42 == lower_lane_43);
        value = ((take_low_44 == 1) ? lo_40 : hi_41);
        if (lane_1 < 16) {
            int stored = value;
            if (value >= num_valid_pages) {
                stored = -1;
            }
            output[row * 16 + lane_1] = stored;
        }
    }
}

} // extern "C"
// clang-format on
