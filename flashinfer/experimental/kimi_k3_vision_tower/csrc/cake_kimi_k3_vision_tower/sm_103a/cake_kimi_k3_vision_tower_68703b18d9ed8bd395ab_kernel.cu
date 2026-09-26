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
#define WIDTH 7168
#define ROWS_PER_CTA 4
#define VECS_PER_LANE 14

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_kimi_k3_vision_tower_68703b18d9ed8bd395ab(__nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ weight, float* __restrict__ rowsumsq, int M, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // === Task calls (dependency order) ===
    int row = bid * ROWS_PER_CTA + warp;
    const int lane_0 = lane;
    int safe_row = ((row < M) ? row : M - 1);
    unsigned long long row_off = (unsigned long long)safe_row * (unsigned long long)WIDTH + (unsigned long long)(lane_0 * 16);
    float vals[VECS_PER_LANE * 16];
    float partial = 0.0f;
    #pragma unroll
    for (int i = 0; i < VECS_PER_LANE; i++) {
        float _vec_load_0[16];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(y + (row_off + (unsigned long long)(i * 512)) + 0);
            uint4 _vld_0[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
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
        for (int j = 0; j < 16; j++) {
            vals[i * 16 + j] = _vec_load_0[j];
            partial += _vec_load_0[j] * _vec_load_0[j];
        }
    }
    float _warp_reduce_0 = partial;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    float total = _warp_reduce_0;
    float _rsqrt_0 = rsqrtf(total * (1.0f / (float)WIDTH) + eps);
    float rstd = _rsqrt_0;
    if (row < M) {
        if (lane_0 == 0) {
            rowsumsq[row] = total;
        }
        #pragma unroll
        for (int i_1 = 0; i_1 < VECS_PER_LANE; i_1++) {
            float _vec_load_1[16];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(weight + (lane_0 * 16 + i_1 * 512) + 0);
                uint4 _vld_1[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
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
            float outv[16];
            #pragma unroll
            for (int j_1 = 0; j_1 < 16; j_1++) {
                outv[j_1] = vals[i_1 * 16 + j_1] * rstd * _vec_load_1[j_1];
            }
            {
                {
                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(outv[0 + 0], outv[0 + 1]);
                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(outv[0 + 2], outv[0 + 3]);
                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(outv[0 + 4], outv[0 + 5]);
                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(outv[0 + 6], outv[0 + 7]);
                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(outv[0 + 8], outv[0 + 9]);
                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(outv[0 + 10], outv[0 + 11]);
                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(outv[0 + 12], outv[0 + 13]);
                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(outv[0 + 14], outv[0 + 15]);
                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(&((__nv_bfloat16*)(y + (row_off + (unsigned long long)(i_1 * 512))))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                }
            }
        }
    }
}

} // extern "C"
