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
#define LATENT 3584
#define ROWS_PER_CTA 4
#define VECS_PER_LANE 14
#define EARLY_TRIGGER 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_kimi_k3_latent_moe_431f1efb77b13ed587bf(__nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ y_out, int M, int num_partials, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    int row = bid * ROWS_PER_CTA + warp;
    int _min_0 = ((row) < (M - 1) ? (row) : (M - 1));
    int load_row = _min_0;
    unsigned long long row_base = (unsigned long long)load_row * (unsigned long long)LATENT;
    unsigned long long partial_stride = (unsigned long long)M * (unsigned long long)LATENT;
    float acc[VECS_PER_LANE * 8];
    #pragma unroll
    for (int i = 0; i < VECS_PER_LANE * 8; i++) {
        acc[i] = 0.0f;
    }
    #pragma unroll 1
    for (int p = 0; p < num_partials; p++) {
        unsigned long long src_base = (unsigned long long)p * partial_stride + row_base;
        #pragma unroll
        for (int i_1 = 0; i_1 < VECS_PER_LANE; i_1++) {
            int k = (lane + i_1 * 32) * 8;
            float _vec_load_0[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(routed + (src_base + (unsigned long long)k) + 0);
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
                acc[i_1 * 8 + j] = acc[i_1 * 8 + j] + _vec_load_0[j];
            }
        }
    }
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i_2 = 0; i_2 < VECS_PER_LANE * 8; i_2++) {
        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(acc[i_2]);
        float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
        float s = _cvt_f32_0;
        sum_sq += s * s;
    }
    float _warp_reduce_0 = sum_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    float total = _warp_reduce_0;
    float _rsqrt_0 = rsqrtf(total / (float)LATENT + eps);
    float rstd = _rsqrt_0;
    if (row < M) {
        unsigned long long out_base = (unsigned long long)row * (unsigned long long)LATENT;
        #pragma unroll
        for (int i_3 = 0; i_3 < VECS_PER_LANE; i_3++) {
            int k_1 = (lane + i_3 * 32) * 8;
            float _vec_load_1[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(norm_weight + k_1 + 0);
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
            float vals[8];
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(acc[i_3 * 8 + j_1]);
                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                float s2 = _cvt_f32_1;
                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(s2 * rstd);
                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                float normed = _cvt_f32_2;
                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_vec_load_1[j_1] * normed);
                float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                vals[j_1] = _cvt_f32_3;
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(y_out + (out_base + (unsigned long long)k_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"
