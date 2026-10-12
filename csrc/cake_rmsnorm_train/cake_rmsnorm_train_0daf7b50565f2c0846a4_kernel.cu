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
#define THREADS 256
#define ROWS_PER_CTA 8
#define GROUP_THREADS 32
#define LAUNCH_BOUNDS_THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) void
kernel_cake_rmsnorm_train_0daf7b50565f2c0846a4(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, long long x_stride, long long u_stride, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int warp_0 = warp;
    int group = tid / GROUP_THREADS;
    int gtid = tid - group * GROUP_THREADS;
    float h_f = 512.0f;
    int it = 0;
    #pragma unroll 1
    for (int row0 = bid * ROWS_PER_CTA; row0 < bid * ROWS_PER_CTA + 1; row0++) {
        int row = row0 + group;
        long long row64 = (long long)row;
        long long xbase = row64 * x_stride;
        long long obase = row64 * 512;
        float x_cache[16];
        float w_cache[16];
        float sum_sq = 0.0f;
        if (row < T) {
            #pragma unroll
            for (int v = 0; v < 2; v++) {
                float _vec_load_0[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + (gtid * 8 + v * (GROUP_THREADS * 8)) + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_vld_0[_blk].x), "=r"(_vld_0[_blk].y), "=r"(_vld_0[_blk].z), "=r"(_vld_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)) : "memory");
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
                    float wv = _vec_load_0[j];
                    w_cache[v * 8 + j] = wv;
                }
            }
            #pragma unroll
            for (int v_1 = 0; v_1 < 2; v_1++) {
                float _vec_load_1[8];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x + (xbase + (long long)(gtid * 8 + v_1 * (GROUP_THREADS * 8))) + 0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vld_1[_blk].x), "=r"(_vld_1[_blk].y), "=r"(_vld_1[_blk].z), "=r"(_vld_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
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
                    x_cache[v_1 * 8 + j_1] = value;
                }
            }
            #pragma unroll
            for (int v_2 = 0; v_2 < 2; v_2++) {
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    sum_sq += x_cache[v_2 * 8 + j_2] * x_cache[v_2 * 8 + j_2];
                }
            }
        }
        float _warp_reduce_0 = sum_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        sum_sq = _warp_reduce_0;
        float total = sum_sq;
        float mean = total * 0.001953125f;
        float _sqrt_0;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_0) : "f"(mean + eps));
        float _rcp_0 = __frcp_rn(_sqrt_0);
        float rstd = _rcp_0;
        if (row < T) {
            if (gtid == 0) {
                r[row] = rstd;
            }
            #pragma unroll
            for (int v_3 = 0; v_3 < 2; v_3++) {
                #pragma unroll
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    x_cache[v_3 * 8 + j_3] = x_cache[v_3 * 8 + j_3] * rstd * w_cache[v_3 * 8 + j_3];
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(x_cache[v_3 * 8 + 0], x_cache[v_3 * 8 + 1]);
                    _pk[1] = __floats2bfloat162_rn(x_cache[v_3 * 8 + 2], x_cache[v_3 * 8 + 3]);
                    _pk[2] = __floats2bfloat162_rn(x_cache[v_3 * 8 + 4], x_cache[v_3 * 8 + 5]);
                    _pk[3] = __floats2bfloat162_rn(x_cache[v_3 * 8 + 6], x_cache[v_3 * 8 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(y + (obase + (long long)(gtid * 8 + v_3 * (GROUP_THREADS * 8)))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        }
    }
}

} // extern "C"
