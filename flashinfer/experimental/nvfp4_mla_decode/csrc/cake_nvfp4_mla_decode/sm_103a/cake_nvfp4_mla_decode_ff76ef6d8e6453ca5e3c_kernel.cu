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
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

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

__global__ __launch_bounds__(256, 1) void
kernel_cake_nvfp4_mla_decode_ff76ef6d8e6453ca5e3c(__nv_bfloat16* __restrict__ o, float* __restrict__ lse, __nv_bfloat16* __restrict__ partial_o, float* __restrict__ partial_lse, int* __restrict__ row_splits, int total_q, int num_heads, int max_splits)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    float NEG_INF = -CUDART_INF_F;
    int q_idx = blockIdx.x;
    int h_idx = blockIdx.y * 8 + warp;
    int lane_0 = lane;
    if (h_idx < num_heads) {
        int row = q_idx * num_heads + h_idx;
        int used = row_splits[q_idx];
        int lse_base = row * max_splits;
        if (used > 1) {
            float local_lse[2];
            float lse_max = NEG_INF;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int sk = lane_0 + i * 32;
                float lv = ((sk < used) ? partial_lse[lse_base + sk] : NEG_INF);
                local_lse[i] = lv;
                float _max_0 = max_noftz(lse_max, lv);
                lse_max = _max_0;
            }
            float _warp_reduce_0 = lse_max;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
            lse_max = _warp_reduce_0;
            lse_max = ((lse_max != NEG_INF) ? lse_max : 0.0f);
            float sum_w = 0.0f;
            #pragma unroll
            for (int i_1 = 0; i_1 < 2; i_1++) {
                float _exp2_0 = approx_exp2(local_lse[i_1] - lse_max);
                sum_w = sum_w + _exp2_0;
            }
            float _warp_reduce_1 = sum_w;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
            sum_w = _warp_reduce_1;
            float _log2_0;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(sum_w));
            float global_lse2 = lse_max + _log2_0;
            if (lane_0 == 0) {
                *(reinterpret_cast<float*>(lse + row) + (0)) = global_lse2 * 0.6931471805599453f;
            }
            float acc[16];
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                acc[j] = 0.0f;
            }
            int split_stride = total_q * num_heads * 512;
            int o_base = row * 512 + lane_0 * 16;
            #pragma unroll 1
            for (int i_2 = 0; i_2 < used; i_2++) {
                float _exp2_1 = approx_exp2(partial_lse[lse_base + i_2] - global_lse2);
                float w_i = _exp2_1;
                #pragma unroll
                for (int j8 = 0; j8 < 16; j8 += 8) {
                    float _vec_load_0[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_o + (o_base + i_2 * split_stride + j8) + 0);
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
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        acc[j8 + j_1] = acc[j8 + j_1] + _vec_load_0[j_1] * w_i;
                    }
                }
            }
            {
                __nv_bfloat162 _pk[8];
                _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(acc[0 + 4], acc[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(acc[0 + 6], acc[0 + 7]);
                _pk[4] = __floats2bfloat162_rn(acc[0 + 8], acc[0 + 9]);
                _pk[5] = __floats2bfloat162_rn(acc[0 + 10], acc[0 + 11]);
                _pk[6] = __floats2bfloat162_rn(acc[0 + 12], acc[0 + 13]);
                _pk[7] = __floats2bfloat162_rn(acc[0 + 14], acc[0 + 15]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(o + (row * 512 + lane_0 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(o + (row * 512 + lane_0 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
            }
        }
    }
}

} // extern "C"
