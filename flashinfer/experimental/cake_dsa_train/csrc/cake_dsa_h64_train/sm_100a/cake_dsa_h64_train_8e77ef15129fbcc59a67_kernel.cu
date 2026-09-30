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

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_cake_dsa_h64_train_8e77ef15129fbcc59a67(__nv_bfloat16* __restrict__ dout, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ o_lo, float* __restrict__ delta, int num_rows)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int row = blockIdx.x * 8 + warp;
    if (row < num_rows) {
        long long base = (long long)row * 512 + (long long)(lane * 16);
        float acc = 0.0f;
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(dout + base + 0);
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
        float _vec_load_1[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(out + base + 0);
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
        float _vec_load_2[8];
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(o_lo + base + 0);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_2[_pair]));
                }
            }
        }
        float _fma_0 = __fmaf_rn(_vec_load_0[0], _vec_load_1[0] + _vec_load_2[0], acc);
        acc = _fma_0;
        float _fma_1 = __fmaf_rn(_vec_load_0[1], _vec_load_1[1] + _vec_load_2[1], acc);
        acc = _fma_1;
        float _fma_2 = __fmaf_rn(_vec_load_0[2], _vec_load_1[2] + _vec_load_2[2], acc);
        acc = _fma_2;
        float _fma_3 = __fmaf_rn(_vec_load_0[3], _vec_load_1[3] + _vec_load_2[3], acc);
        acc = _fma_3;
        float _fma_4 = __fmaf_rn(_vec_load_0[4], _vec_load_1[4] + _vec_load_2[4], acc);
        acc = _fma_4;
        float _fma_5 = __fmaf_rn(_vec_load_0[5], _vec_load_1[5] + _vec_load_2[5], acc);
        acc = _fma_5;
        float _fma_6 = __fmaf_rn(_vec_load_0[6], _vec_load_1[6] + _vec_load_2[6], acc);
        acc = _fma_6;
        float _fma_7 = __fmaf_rn(_vec_load_0[7], _vec_load_1[7] + _vec_load_2[7], acc);
        acc = _fma_7;
        float _vec_load_3[8];
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(dout + (base + 8) + 0);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_3[_pair]));
                }
            }
        }
        float _vec_load_4[8];
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(out + (base + 8) + 0);
            uint4 _vld_4[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_4[_blk] = _vptr_4[_blk];
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                }
            }
        }
        float _vec_load_5[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(o_lo + (base + 8) + 0);
            uint4 _vld_5[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_5[_blk] = _vptr_5[_blk];
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
        float _fma_8 = __fmaf_rn(_vec_load_3[0], _vec_load_4[0] + _vec_load_5[0], acc);
        acc = _fma_8;
        float _fma_9 = __fmaf_rn(_vec_load_3[1], _vec_load_4[1] + _vec_load_5[1], acc);
        acc = _fma_9;
        float _fma_10 = __fmaf_rn(_vec_load_3[2], _vec_load_4[2] + _vec_load_5[2], acc);
        acc = _fma_10;
        float _fma_11 = __fmaf_rn(_vec_load_3[3], _vec_load_4[3] + _vec_load_5[3], acc);
        acc = _fma_11;
        float _fma_12 = __fmaf_rn(_vec_load_3[4], _vec_load_4[4] + _vec_load_5[4], acc);
        acc = _fma_12;
        float _fma_13 = __fmaf_rn(_vec_load_3[5], _vec_load_4[5] + _vec_load_5[5], acc);
        acc = _fma_13;
        float _fma_14 = __fmaf_rn(_vec_load_3[6], _vec_load_4[6] + _vec_load_5[6], acc);
        acc = _fma_14;
        float _fma_15 = __fmaf_rn(_vec_load_3[7], _vec_load_4[7] + _vec_load_5[7], acc);
        acc = _fma_15;
        float _warp_reduce_0 = acc;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        float total = _warp_reduce_0;
        if (lane == 0) {
            delta[row] = total;
        }
    }
}

} // extern "C"
