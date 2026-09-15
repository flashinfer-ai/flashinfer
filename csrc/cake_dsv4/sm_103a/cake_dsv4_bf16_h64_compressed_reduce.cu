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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
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

__global__ __launch_bounds__(128) void
kernel_cake_dsv4_bf16_h64_compressed_reduce(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, int num_heads, int num_splits)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int stat_base = (query_idx * num_heads + head_idx) * num_splits;
    float local_lse = -CAKE_INF;
    if (lane < (unsigned int)num_splits) {
        local_lse = partial_lse[(unsigned int)stat_base + lane];
    }
    float _warp_reduce_0 = local_lse;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
    float global_max = _warp_reduce_0;
    float local_weight = 0.0f;
    if (lane < (unsigned int)num_splits) {
        float _exp2_0 = approx_exp2(local_lse - global_max);
        local_weight = ((local_lse == -CAKE_INF) ? 0.0f : _exp2_0);
    }
    float _warp_reduce_1 = local_weight;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    float global_sum = _warp_reduce_1;
    float _rcp_0 = approx_rcp(global_sum);
    float normalized_weight = ((global_sum > 0.0f) ? local_weight * _rcp_0 : 0.0f);
    int partial_base = stat_base * 512;
    int output_base = (query_idx * num_heads + head_idx) * 512;
    int d_base = tid * 4;
    float acc[4];
    #pragma unroll
    for (int elem = 0; elem < 4; elem++) {
        acc[elem] = 0.0f;
    }
    if (num_splits == 4) {
        float _vec_load_0[4];
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + d_base) + 0);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_0[0 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _pair * 2])[1])
                    : "r"(_vpairs_0[_pair]));
            }
        }
        float _vec_load_1[4];
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 512 + d_base) + 0);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_1[0 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _pair * 2])[1])
                    : "r"(_vpairs_1[_pair]));
            }
        }
        float _vec_load_2[4];
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 1024 + d_base) + 0);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_2[0 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _pair * 2])[1])
                    : "r"(_vpairs_2[_pair]));
            }
        }
        float _vec_load_3[4];
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 1536 + d_base) + 0);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_3[0 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _pair * 2])[1])
                    : "r"(_vpairs_3[_pair]));
            }
        }
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, normalized_weight, 0);
        float weight0 = _shfl_0;
        float _shfl_1 = __shfl_sync(0xFFFFFFFF, normalized_weight, 1);
        float weight1 = _shfl_1;
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, normalized_weight, 2);
        float weight2 = _shfl_2;
        float _shfl_3 = __shfl_sync(0xFFFFFFFF, normalized_weight, 3);
        float weight3 = _shfl_3;
        #pragma unroll
        for (int elem_1 = 0; elem_1 < 4; elem_1++) {
            acc[elem_1] = acc[elem_1] + weight0 * _vec_load_0[elem_1];
            acc[elem_1] = acc[elem_1] + weight1 * _vec_load_1[elem_1];
            acc[elem_1] = acc[elem_1] + weight2 * _vec_load_2[elem_1];
            acc[elem_1] = acc[elem_1] + weight3 * _vec_load_3[elem_1];
        }
    } else {
        #pragma unroll 5
        for (int split = 0; split < num_splits; split++) {
            float _vec_load_4[4];
            {
                uint2 _vld_4;
                _vld_4 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + split * 512 + d_base) + 0);
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_4[0 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                }
            }
            float _shfl_4 = __shfl_sync(0xFFFFFFFF, normalized_weight, split);
            float weight = _shfl_4;
            #pragma unroll
            for (int elem_2 = 0; elem_2 < 4; elem_2++) {
                acc[elem_2] = acc[elem_2] + weight * _vec_load_4[elem_2];
            }
        }
    }
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O + (output_base + d_base)))[0]) = _pk2;
    }
}

} // extern "C"
