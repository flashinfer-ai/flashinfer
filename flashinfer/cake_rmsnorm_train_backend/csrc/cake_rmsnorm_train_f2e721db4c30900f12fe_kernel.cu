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
#define THREADS 256
#define ROWS_PER_CTA 8
#define GROUP_THREADS 32
#define LAUNCH_BOUNDS_THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) void
kernel_cake_rmsnorm_train_f2e721db4c30900f12fe(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, float* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int warp_0 = warp;
    int group = tid / GROUP_THREADS;
    int gtid = tid - group * GROUP_THREADS;
    float h_f = 256.0f;
    int it = 0;
    #pragma unroll 1
    for (int row0 = bid * ROWS_PER_CTA; row0 < bid * ROWS_PER_CTA + 1; row0++) {
        int row = row0 + group;
        long long row64 = (long long)row;
        long long xbase = row64 * x_stride_p0;
        long long obase = row64 * 256;
        float x_cache[8];
        float sum_sq = 0.0f;
        if (row < T) {
            #pragma unroll
            for (int v = 0; v < 1; v++) {
                float _vec_load_0[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (xbase + (long long)(gtid * 8 + v * (GROUP_THREADS * 8))) + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vld_0[_blk].x), "=r"(_vld_0[_blk].y), "=r"(_vld_0[_blk].z), "=r"(_vld_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                            (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float value = _vec_load_0[j];
                    x_cache[v * 8 + j] = value;
                }
            }
            #pragma unroll
            for (int v_1 = 0; v_1 < 1; v_1++) {
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    sum_sq = __fmaf_rn(x_cache[v_1 * 8 + j_1], x_cache[v_1 * 8 + j_1], sum_sq);
                }
            }
        }
        float _warp_reduce_1 = sum_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        sum_sq = _warp_reduce_1;
        float total = sum_sq;
        float mean = total * 0.00390625f;
        float _sqrt_2;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_2) : "f"(mean + eps));
        float rstd = __frcp_rn(_sqrt_2);
        if (row < T) {
            if (gtid == 0) {
                r[row] = rstd;
            }
            #pragma unroll
            for (int v_2 = 0; v_2 < 1; v_2++) {
                float _vec_load_1[8];
                {
                    unsigned _ldv8_3_0;
                    unsigned _ldv8_3_1;
                    unsigned _ldv8_3_2;
                    unsigned _ldv8_3_3;
                    unsigned _ldv8_3_4;
                    unsigned _ldv8_3_5;
                    unsigned _ldv8_3_6;
                    unsigned _ldv8_3_7;
                    asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(w + (gtid * 8 + v_2 * (GROUP_THREADS * 8)) + (0))) : "memory");
                    _vec_load_1[0 + 0] = __uint_as_float(_ldv8_3_0);
                    _vec_load_1[0 + 1] = __uint_as_float(_ldv8_3_1);
                    _vec_load_1[0 + 2] = __uint_as_float(_ldv8_3_2);
                    _vec_load_1[0 + 3] = __uint_as_float(_ldv8_3_3);
                    _vec_load_1[0 + 4] = __uint_as_float(_ldv8_3_4);
                    _vec_load_1[0 + 5] = __uint_as_float(_ldv8_3_5);
                    _vec_load_1[0 + 6] = __uint_as_float(_ldv8_3_6);
                    _vec_load_1[0 + 7] = __uint_as_float(_ldv8_3_7);
                }
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    x_cache[v_2 * 8 + j_2] = x_cache[v_2 * 8 + j_2] * rstd * _vec_load_1[j_2];
                }
                unsigned int packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + v_2 * 8)[_lp*2 + 0], (x_cache + v_2 * 8)[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + v_2 * 8)[_lp*2 + 2], (x_cache + v_2 * 8)[_lp*2+1 + 2]));
                    packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + v_2 * 8)[_lp*2 + 4], (x_cache + v_2 * 8)[_lp*2+1 + 4]));
                    packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + v_2 * 8)[_lp*2 + 6], (x_cache + v_2 * 8)[_lp*2+1 + 6]));
                    packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(y + (obase + (long long)(gtid * 8 + v_2 * (GROUP_THREADS * 8))))[0] = reinterpret_cast<int4*>(packed_1)[0];
            }
        }
    }
}

} // extern "C"
