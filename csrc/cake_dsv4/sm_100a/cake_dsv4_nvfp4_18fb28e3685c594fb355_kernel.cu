/*
 * Copyright (c) 2023 by FlashInfer team.
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

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_cake_dsv4_nvfp4_18fb28e3685c594fb355(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ lse_out, int num_heads, int num_splits, float lse_scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int stat_base = (query_idx * num_heads + head_idx) * num_splits;
    int last_split = num_splits - 1;
    float local_lse = -CAKE_INF;
    if (lane < num_splits) {
        local_lse = partial_lse[stat_base + lane];
    }
    float _warp_reduce_0 = local_lse;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
    float global_max = _warp_reduce_0;
    float local_weight = 0.0f;
    if (lane < num_splits && local_lse != -CAKE_INF) {
        float _exp2_0 = approx_exp2(local_lse - global_max);
        local_weight = _exp2_0;
    }
    float _warp_reduce_1 = local_weight;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    float global_sum = _warp_reduce_1;
    float normalized_weight = 0.0f;
    if (global_sum > 0.0f) {
        float _rcp_0 = approx_rcp(global_sum);
        normalized_weight = local_weight * _rcp_0;
    }
    int partial_base = stat_base * 512 + tid * 4;
    float acc[4];
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;
    for (int split_group = 0; split_group < num_splits; split_group += 4) {
        int _min_0 = ((split_group) < (last_split) ? (split_group) : (last_split));
        int split_k = _min_0;
        float _vec_load_0[4];
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + split_k * 512) + 0);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                (&_vec_load_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
            }
        }
        int _min_1 = ((split_group + 1) < (last_split) ? (split_group + 1) : (last_split));
        int split_k_0 = _min_1;
        float _vec_load_1[4];
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + split_k_0 * 512) + 0);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&_vec_load_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
        int _min_2 = ((split_group + 2) < (last_split) ? (split_group + 2) : (last_split));
        int split_k_1 = _min_2;
        float _vec_load_2[4];
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + split_k_1 * 512) + 0);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&_vec_load_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
        int _min_3 = ((split_group + 3) < (last_split) ? (split_group + 3) : (last_split));
        int split_k_2 = _min_3;
        float _vec_load_3[4];
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + split_k_2 * 512) + 0);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&_vec_load_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
        int _min_4 = ((split_group) < (last_split) ? (split_group) : (last_split));
        float _shfl_0;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(normalized_weight), "r"(_min_4));
        float weight_k = _shfl_0;
        if (last_split < split_group) {
            weight_k = 0.0f;
        }
        acc[0] = acc[0] + weight_k * _vec_load_0[0];
        acc[1] = acc[1] + weight_k * _vec_load_0[1];
        acc[2] = acc[2] + weight_k * _vec_load_0[2];
        acc[3] = acc[3] + weight_k * _vec_load_0[3];
        int _min_5 = ((split_group + 1) < (last_split) ? (split_group + 1) : (last_split));
        float _shfl_1;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_1) : "f"(normalized_weight), "r"(_min_5));
        float weight_k_3 = _shfl_1;
        if (last_split < split_group + 1) {
            weight_k_3 = 0.0f;
        }
        acc[0] = acc[0] + weight_k_3 * _vec_load_1[0];
        acc[1] = acc[1] + weight_k_3 * _vec_load_1[1];
        acc[2] = acc[2] + weight_k_3 * _vec_load_1[2];
        acc[3] = acc[3] + weight_k_3 * _vec_load_1[3];
        int _min_6 = ((split_group + 2) < (last_split) ? (split_group + 2) : (last_split));
        float _shfl_2;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_2) : "f"(normalized_weight), "r"(_min_6));
        float weight_k_4 = _shfl_2;
        if (last_split < split_group + 2) {
            weight_k_4 = 0.0f;
        }
        acc[0] = acc[0] + weight_k_4 * _vec_load_2[0];
        acc[1] = acc[1] + weight_k_4 * _vec_load_2[1];
        acc[2] = acc[2] + weight_k_4 * _vec_load_2[2];
        acc[3] = acc[3] + weight_k_4 * _vec_load_2[3];
        int _min_7 = ((split_group + 3) < (last_split) ? (split_group + 3) : (last_split));
        float _shfl_3;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_3) : "f"(normalized_weight), "r"(_min_7));
        float weight_k_5 = _shfl_3;
        if (last_split < split_group + 3) {
            weight_k_5 = 0.0f;
        }
        acc[0] = acc[0] + weight_k_5 * _vec_load_3[0];
        acc[1] = acc[1] + weight_k_5 * _vec_load_3[1];
        acc[2] = acc[2] + weight_k_5 * _vec_load_3[2];
        acc[3] = acc[3] + weight_k_5 * _vec_load_3[3];
    }
    int output_base = (query_idx * num_heads + head_idx) * 512 + tid * 4;
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O + output_base))[0]) = _pk2;
    }
    if (tid == 0) {
        float _log2_0;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(global_sum));
        lse_out[query_idx * num_heads + head_idx] = ((global_sum > 0.0f) ? (global_max + _log2_0) * lse_scale : -CAKE_INF);
    }
}

} // extern "C"
