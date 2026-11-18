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
#define THREADS 32

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_mxfp4_situ_moe_1c0edcae325d6ffbe30f(int* __restrict__ tile_expert, int* __restrict__ alt_expert, int* __restrict__ base_active, int* __restrict__ alt_active, int* __restrict__ wide_list, int* __restrict__ wide_count, int* __restrict__ alt_wide_list, int* __restrict__ alt_wide_count, int* __restrict__ narrow_count, int* __restrict__ gate, int min_windows, int every_group_vote, int wide_list_vote, int distinct_threshold)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int base_v = base_active[0];
    int alt_v = alt_active[0];
    int wide_v = wide_count[0];
    int alt_wide_v = alt_wide_count[0];
    int narrow_v = narrow_count[0];
    int num_valid = base_active[0];
    int lane_s = lane;
    int _max_0 = ((num_valid / 64) > (1) ? (num_valid / 64) : (1));
    int stride_s = _max_0;
    int _max_1 = ((num_valid - 1) > (0) ? (num_valid - 1) : (0));
    int last_group = _max_1;
    int _min_0 = ((lane_s * stride_s) < (last_group) ? (lane_s * stride_s) : (last_group));
    int g0 = _min_0;
    int _min_1 = (((lane_s + 32) * stride_s) < (last_group) ? ((lane_s + 32) * stride_s) : (last_group));
    int g1 = _min_1;
    int e0 = tile_expert[g0];
    int e1 = tile_expert[g1];
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, e0, 1, 32);
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, e1, 1, 32);
    int _shfl_0 = __shfl_sync(0xFFFFFFFF, e0, 31);
    int is_lane0 = (int)(lane_s == 0);
    int e1_prev = _shfl_up_1 * (1 - is_lane0) + _shfl_0 * is_lane0;
    int not_lane0 = (int)(lane_s > 0);
    int same = not_lane0 & (int)(e0 == _shfl_up_0) | (int)(e1 == e1_prev);
    int transitions = (not_lane0 & (int)(e0 != _shfl_up_0)) + (int)(e1 != e1_prev);
    int _vote_0 = __any_sync(0xFFFFFFFF, same != 0);
    unsigned int _warp_redux_u32_0;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"((unsigned int)transitions));
    int distinct = (int)_warp_redux_u32_0 + 1;
    int mode_n = _vote_0 & (int)(distinct > distinct_threshold) | (int)(num_valid == 0);
    int num_valid_0 = alt_active[0];
    int lane_s_1 = lane;
    int _max_2 = ((num_valid_0 / 64) > (1) ? (num_valid_0 / 64) : (1));
    int stride_s_2 = _max_2;
    int _max_3 = ((num_valid_0 - 1) > (0) ? (num_valid_0 - 1) : (0));
    int last_group_3 = _max_3;
    int _min_2 = ((lane_s_1 * stride_s_2) < (last_group_3) ? (lane_s_1 * stride_s_2) : (last_group_3));
    int g0_4 = _min_2;
    int _min_3 = (((lane_s_1 + 32) * stride_s_2) < (last_group_3) ? ((lane_s_1 + 32) * stride_s_2) : (last_group_3));
    int g1_5 = _min_3;
    int e0_6 = alt_expert[g0_4];
    int e1_7 = alt_expert[g1_5];
    int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, e0_6, 1, 32);
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, e1_7, 1, 32);
    int _shfl_1 = __shfl_sync(0xFFFFFFFF, e0_6, 31);
    int is_lane0_8 = (int)(lane_s_1 == 0);
    int e1_prev_9 = _shfl_up_3 * (1 - is_lane0_8) + _shfl_1 * is_lane0_8;
    int not_lane0_10 = (int)(lane_s_1 > 0);
    int same_11 = not_lane0_10 & (int)(e0_6 == _shfl_up_2) | (int)(e1_7 == e1_prev_9);
    int transitions_12 = (not_lane0_10 & (int)(e0_6 != _shfl_up_2)) + (int)(e1_7 != e1_prev_9);
    int _vote_1 = __any_sync(0xFFFFFFFF, same_11 != 0);
    unsigned int _warp_redux_u32_1;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"((unsigned int)transitions_12));
    int distinct_13 = (int)_warp_redux_u32_1 + 1;
    int mode_n_14 = _vote_1 & (int)(distinct_13 > distinct_threshold) | (int)(num_valid_0 == 0);
    int num_valid_15 = wide_count[0];
    int lane_s_16 = lane;
    int _max_4 = ((num_valid_15 / 64) > (1) ? (num_valid_15 / 64) : (1));
    int stride_s_17 = _max_4;
    int _max_5 = ((num_valid_15 - 1) > (0) ? (num_valid_15 - 1) : (0));
    int last_group_18 = _max_5;
    int _min_4 = ((lane_s_16 * stride_s_17) < (last_group_18) ? (lane_s_16 * stride_s_17) : (last_group_18));
    int g0_19 = _min_4;
    int _min_5 = (((lane_s_16 + 32) * stride_s_17) < (last_group_18) ? ((lane_s_16 + 32) * stride_s_17) : (last_group_18));
    int g1_20 = _min_5;
    int has_valid = (int)(num_valid_15 > 0);
    int s0 = wide_list[g0_19] * has_valid;
    int s1 = wide_list[g1_20] * has_valid;
    int e0_21 = tile_expert[s0];
    int e1_22 = tile_expert[s1];
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, e0_21, 1, 32);
    int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, e1_22, 1, 32);
    int _shfl_2 = __shfl_sync(0xFFFFFFFF, e0_21, 31);
    int is_lane0_23 = (int)(lane_s_16 == 0);
    int e1_prev_24 = _shfl_up_5 * (1 - is_lane0_23) + _shfl_2 * is_lane0_23;
    int not_lane0_25 = (int)(lane_s_16 > 0);
    int same_26 = not_lane0_25 & (int)(e0_21 == _shfl_up_4) | (int)(e1_22 == e1_prev_24);
    int transitions_27 = (not_lane0_25 & (int)(e0_21 != _shfl_up_4)) + (int)(e1_22 != e1_prev_24);
    int _vote_2 = __any_sync(0xFFFFFFFF, same_26 != 0);
    unsigned int _warp_redux_u32_2;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"((unsigned int)transitions_27));
    int distinct_28 = (int)_warp_redux_u32_2 + 1;
    int mode_n_29 = _vote_2 & (int)(distinct_28 > distinct_threshold) | (int)(num_valid_15 == 0) | (int)(distinct_28 == 1);
    int num_valid_30 = alt_wide_count[0];
    int lane_s_31 = lane;
    int _max_6 = ((num_valid_30 / 64) > (1) ? (num_valid_30 / 64) : (1));
    int stride_s_32 = _max_6;
    int _max_7 = ((num_valid_30 - 1) > (0) ? (num_valid_30 - 1) : (0));
    int last_group_33 = _max_7;
    int _min_6 = ((lane_s_31 * stride_s_32) < (last_group_33) ? (lane_s_31 * stride_s_32) : (last_group_33));
    int g0_34 = _min_6;
    int _min_7 = (((lane_s_31 + 32) * stride_s_32) < (last_group_33) ? ((lane_s_31 + 32) * stride_s_32) : (last_group_33));
    int g1_35 = _min_7;
    int has_valid_36 = (int)(num_valid_30 > 0);
    int s0_37 = alt_wide_list[g0_34] * has_valid_36;
    int s1_38 = alt_wide_list[g1_35] * has_valid_36;
    int e0_39 = alt_expert[s0_37];
    int e1_40 = alt_expert[s1_38];
    int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, e0_39, 1, 32);
    int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, e1_40, 1, 32);
    int _shfl_3 = __shfl_sync(0xFFFFFFFF, e0_39, 31);
    int is_lane0_41 = (int)(lane_s_31 == 0);
    int e1_prev_42 = _shfl_up_7 * (1 - is_lane0_41) + _shfl_3 * is_lane0_41;
    int not_lane0_43 = (int)(lane_s_31 > 0);
    int same_44 = not_lane0_43 & (int)(e0_39 == _shfl_up_6) | (int)(e1_40 == e1_prev_42);
    int transitions_45 = (not_lane0_43 & (int)(e0_39 != _shfl_up_6)) + (int)(e1_40 != e1_prev_42);
    int _vote_3 = __any_sync(0xFFFFFFFF, same_44 != 0);
    unsigned int _warp_redux_u32_3;
    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_3) : "r"((unsigned int)transitions_45));
    int distinct_46 = (int)_warp_redux_u32_3 + 1;
    int mode_n_47 = _vote_3 & (int)(distinct_46 > distinct_threshold) | (int)(num_valid_30 == 0) | (int)(distinct_46 == 1);
    int every_alt = (int)(base_v == 0);
    int every_n = mode_n * (1 - every_alt) + mode_n_14 * every_alt;
    int wide_alt = (int)(wide_v == 0);
    int wide_n = mode_n_29 * (1 - wide_alt) + mode_n_47 * wide_alt;
    int windows_ok = (int)(narrow_v >= min_windows);
    int every_ok = (int)(every_group_vote < 0) | (int)(every_n == every_group_vote);
    int wide_ok = (int)(wide_list_vote < 0) | (int)(wide_n == wide_list_vote);
    int split = windows_ok & every_ok & wide_ok;
    int dense = 1 - split;
    if (elect_sync()) {
        gate[0] = split;
        gate[64] = narrow_v * split;
        gate[128] = wide_v * split;
        gate[192] = alt_wide_v * split;
        gate[256] = base_v * dense;
        gate[320] = alt_v * dense;
        gate[384] = every_n;
        gate[448] = wide_n;
    }
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

} // extern "C"
