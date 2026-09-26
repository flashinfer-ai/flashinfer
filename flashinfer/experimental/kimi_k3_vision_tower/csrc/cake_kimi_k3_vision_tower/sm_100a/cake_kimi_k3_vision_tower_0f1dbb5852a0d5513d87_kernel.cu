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

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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
#define HIDDEN 1024
#define VECS_PER_LANE 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_kimi_k3_vision_tower_0f1dbb5852a0d5513d87(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ norm_weight, int* __restrict__ merge_table, __nv_bfloat16* __restrict__ m_out, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int rec = bid * 4;
    int first_token = merge_table[rec];
    int row_stride = merge_table[rec + 1];
    int frame_stride = merge_table[rec + 2];
    int frames = merge_table[rec + 3];
    int dy = warp >> 1;
    int dx = warp & 1;
    int window_token = first_token + dy * row_stride + dx;
    float w_cache[32];
    float acc[32];
    #pragma unroll
    for (int i = 0; i < VECS_PER_LANE; i++) {
        int k = (lane + i * 32) * 8;
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(norm_weight + k + 0);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_0[_pair]));
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            w_cache[i * 8 + j] = _vec_load_0[j];
            acc[i * 8 + j] = 0.0f;
        }
    }
    #pragma unroll 1
    for (int frame = 0; frame < frames; frame++) {
        int token = window_token + frame * frame_stride;
        unsigned long long row_base = (unsigned long long)token * (unsigned long long)HIDDEN;
        float x_cache[32];
        float sum_sq = 0.0f;
        #pragma unroll
        for (int i_1 = 0; i_1 < VECS_PER_LANE; i_1++) {
            int k_1 = (lane + i_1 * 32) * 8;
            float _vec_load_1[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x + (row_base + (unsigned long long)k_1) + 0);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_1[_pair]));
                    }
                }
            }
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                float value = _vec_load_1[j_1];
                x_cache[i_1 * 8 + j_1] = value;
                sum_sq += value * value;
            }
        }
        float _warp_reduce_0 = sum_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        float total = _warp_reduce_0;
        float _rsqrt_0 = rsqrtf(total / (float)HIDDEN + eps);
        float rstd = _rsqrt_0;
        #pragma unroll
        for (int j_2 = 0; j_2 < 32; j_2++) {
            float scaled = x_cache[j_2] * rstd;
            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(scaled * w_cache[j_2]);
            float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
            acc[j_2] = acc[j_2] + _cvt_f32_0;
        }
    }
    float frames_f32 = (float)frames;
    float _fdiv_rn_0 = __fdiv_rn(1.0f, frames_f32);
    float inv_frames = _fdiv_rn_0;
    unsigned long long out_base = (unsigned long long)bid * (unsigned long long)(4 * HIDDEN) + (unsigned long long)warp * (unsigned long long)HIDDEN;
    #pragma unroll
    for (int i_2 = 0; i_2 < VECS_PER_LANE; i_2++) {
        int k_2 = (lane + i_2 * 32) * 8;
        float vals[8];
        #pragma unroll
        for (int j_3 = 0; j_3 < 8; j_3++) {
            vals[j_3] = acc[i_2 * 8 + j_3] * inv_frames;
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(m_out + (out_base + (unsigned long long)k_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
    }
}

} // extern "C"
