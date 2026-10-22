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


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_causal_conv1d_train_17cd42fb05079bbbf840(__nv_bfloat16* __restrict__ x, float* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ hist, int* __restrict__ hist_row_off, float* __restrict__ dpre_head, int* __restrict__ head_row_off, int S, int C, int num_docs)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long C64 = (long long)C;
    int n_cb = (C + 1023) / 1024;
    int n_items = num_docs * n_cb;
    float wr[32];
    float w4[4];
    float win[32];
    float dyf[8];
    float dp[8];
    #pragma unroll 1
    for (int item = bid; item < n_items; item += num_bids) {
        int f = item / n_cb;
        int c0 = (item - f * n_cb) * 1024 + tid * 8;
        int start = cu[f];
        int n_head = ((cu[f + 1] - start) < (3) ? (cu[f + 1] - start) : (3));
        int wanted = 0;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            if (n_head > i) {
                #pragma unroll
                for (int s = 0; s < 3; s++) {
                    if (head_row_off[(f * 3 + i) * 3 + s] >= 0) {
                        wanted = 1;
                    }
                }
            }
        }
        if (wanted != 0 && c0 < C) {
            long long c64 = (long long)c0;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(w + (c64 + (long long)j) * 4);
                    wr[j * 4 + 0] = _v4.x;
                    wr[j * 4 + 1] = _v4.y;
                    wr[j * 4 + 2] = _v4.z;
                    wr[j * 4 + 3] = _v4.w;
                }
            }
            #pragma unroll
            for (int q = 0; q < 32; q++) {
                win[q] = 0.0f;
            }
            int hrow = hist_row_off[f * 3 + 2];
            if (hrow >= 0) {
                long long rowh = (long long)hrow * C64 + c64;
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(hist + rowh);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                            (&win[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                        }
                    }
                }
            }
            int hrow_0 = hist_row_off[f * 3 + 1];
            if (hrow_0 >= 0) {
                long long rowh_1 = (long long)hrow_0 * C64 + c64;
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(hist + rowh_1);
                    uint4 _vld_2[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                            (&win[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                        }
                    }
                }
            }
            int hrow_1 = hist_row_off[f * 3];
            if (hrow_1 >= 0) {
                long long rowh_2 = (long long)hrow_1 * C64 + c64;
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(hist + rowh_2);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                            (&win[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                        }
                    }
                }
            }
            if (n_head > 0) {
                long long tok = (long long)start * C64 + c64;
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(x + tok);
                    uint4 _vld_4[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_4[_blk] = _vptr_4[_blk];
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                            (&win[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(dy + tok);
                    uint4 _vld_5[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_5[_blk] = _vptr_5[_blk];
                        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&dyf[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                            (&dyf[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    #pragma unroll
                    for (int kk = 0; kk < 4; kk++) {
                        w4[kk] = wr[j_1 * 4 + kk];
                    }
                    float _fma_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(w4[1]), "f"(win[16 + j_1]), "f"(mul_f32_rn_noftz(w4[0], win[8 + j_1])));
                    float _fma_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(w4[2]), "f"(win[24 + j_1]), "f"(_fma_7));
                    float _fma_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(w4[3]), "f"(win[j_1]), "f"(_fma_6));
                    float e = (-_fma_0) * 1.4426950408889634f;
                    float _exp2_8 = approx_exp2(e);
                    float den = 1.0f + _exp2_8;
                    float sg = approx_rcp(den);
                    float _fma_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(_fma_0), "f"(1.0f - sg), "f"(1.0f));
                    dp[j_1] = dyf[j_1] * (sg * _fma_9);
                }
                int srow = head_row_off[f * 3 * 3];
                if (srow >= 0) {
                    long long rowd = (long long)srow * C64 + c64;
                    {
                        unsigned _stv8_10_0 = __float_as_uint(dp[0 + 0]);
                        unsigned _stv8_10_1 = __float_as_uint(dp[0 + 1]);
                        unsigned _stv8_10_2 = __float_as_uint(dp[0 + 2]);
                        unsigned _stv8_10_3 = __float_as_uint(dp[0 + 3]);
                        unsigned _stv8_10_4 = __float_as_uint(dp[0 + 4]);
                        unsigned _stv8_10_5 = __float_as_uint(dp[0 + 5]);
                        unsigned _stv8_10_6 = __float_as_uint(dp[0 + 6]);
                        unsigned _stv8_10_7 = __float_as_uint(dp[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dpre_head + (rowd))), "r"(_stv8_10_0), "r"(_stv8_10_1), "r"(_stv8_10_2), "r"(_stv8_10_3), "r"(_stv8_10_4), "r"(_stv8_10_5), "r"(_stv8_10_6), "r"(_stv8_10_7) : "memory");
                    }
                }
                int srow_0 = head_row_off[f * 3 * 3 + 1];
                if (srow_0 >= 0) {
                    long long rowd_1 = (long long)srow_0 * C64 + c64;
                    {
                        unsigned _stv8_11_0 = __float_as_uint(dp[0 + 0]);
                        unsigned _stv8_11_1 = __float_as_uint(dp[0 + 1]);
                        unsigned _stv8_11_2 = __float_as_uint(dp[0 + 2]);
                        unsigned _stv8_11_3 = __float_as_uint(dp[0 + 3]);
                        unsigned _stv8_11_4 = __float_as_uint(dp[0 + 4]);
                        unsigned _stv8_11_5 = __float_as_uint(dp[0 + 5]);
                        unsigned _stv8_11_6 = __float_as_uint(dp[0 + 6]);
                        unsigned _stv8_11_7 = __float_as_uint(dp[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dpre_head + (rowd_1))), "r"(_stv8_11_0), "r"(_stv8_11_1), "r"(_stv8_11_2), "r"(_stv8_11_3), "r"(_stv8_11_4), "r"(_stv8_11_5), "r"(_stv8_11_6), "r"(_stv8_11_7) : "memory");
                    }
                }
                int srow_1 = head_row_off[f * 3 * 3 + 2];
                if (srow_1 >= 0) {
                    long long rowd_2 = (long long)srow_1 * C64 + c64;
                    {
                        unsigned _stv8_12_0 = __float_as_uint(dp[0 + 0]);
                        unsigned _stv8_12_1 = __float_as_uint(dp[0 + 1]);
                        unsigned _stv8_12_2 = __float_as_uint(dp[0 + 2]);
                        unsigned _stv8_12_3 = __float_as_uint(dp[0 + 3]);
                        unsigned _stv8_12_4 = __float_as_uint(dp[0 + 4]);
                        unsigned _stv8_12_5 = __float_as_uint(dp[0 + 5]);
                        unsigned _stv8_12_6 = __float_as_uint(dp[0 + 6]);
                        unsigned _stv8_12_7 = __float_as_uint(dp[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dpre_head + (rowd_2))), "r"(_stv8_12_0), "r"(_stv8_12_1), "r"(_stv8_12_2), "r"(_stv8_12_3), "r"(_stv8_12_4), "r"(_stv8_12_5), "r"(_stv8_12_6), "r"(_stv8_12_7) : "memory");
                    }
                }
            }
            if (n_head > 1) {
                long long tok_1 = (long long)(start + 1) * C64 + c64;
                {
                    const uint4* _vptr_13 = reinterpret_cast<const uint4*>(x + tok_1);
                    uint4 _vld_13[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_13[_blk] = _vptr_13[_blk];
                        uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                            (&win[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_14 = reinterpret_cast<const uint4*>(dy + tok_1);
                    uint4 _vld_14[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_14[_blk] = _vptr_14[_blk];
                        uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&dyf[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
                            (&dyf[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    #pragma unroll
                    for (int kk_1 = 0; kk_1 < 4; kk_1++) {
                        w4[kk_1] = wr[j_2 * 4 + kk_1];
                    }
                    float _fma_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(w4[1]), "f"(win[24 + j_2]), "f"(mul_f32_rn_noftz(w4[0], win[16 + j_2])));
                    float _fma_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(w4[2]), "f"(win[j_2]), "f"(_fma_16));
                    float _fma_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(w4[3]), "f"(win[8 + j_2]), "f"(_fma_15));
                    float e_1 = (-_fma_1) * 1.4426950408889634f;
                    float _exp2_17 = approx_exp2(e_1);
                    float den_1 = 1.0f + _exp2_17;
                    float sg_1 = approx_rcp(den_1);
                    float _fma_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(_fma_1), "f"(1.0f - sg_1), "f"(1.0f));
                    dp[j_2] = dyf[j_2] * (sg_1 * _fma_18);
                }
                int srow_2 = head_row_off[(f * 3 + 1) * 3];
                if (srow_2 >= 0) {
                    long long rowd_3 = (long long)srow_2 * C64 + c64;
                    {
                        unsigned _stv8_19_0 = __float_as_uint(dp[0 + 0]);
                        unsigned _stv8_19_1 = __float_as_uint(dp[0 + 1]);
                        unsigned _stv8_19_2 = __float_as_uint(dp[0 + 2]);
                        unsigned _stv8_19_3 = __float_as_uint(dp[0 + 3]);
                        unsigned _stv8_19_4 = __float_as_uint(dp[0 + 4]);
                        unsigned _stv8_19_5 = __float_as_uint(dp[0 + 5]);
                        unsigned _stv8_19_6 = __float_as_uint(dp[0 + 6]);
                        unsigned _stv8_19_7 = __float_as_uint(dp[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dpre_head + (rowd_3))), "r"(_stv8_19_0), "r"(_stv8_19_1), "r"(_stv8_19_2), "r"(_stv8_19_3), "r"(_stv8_19_4), "r"(_stv8_19_5), "r"(_stv8_19_6), "r"(_stv8_19_7) : "memory");
                    }
                }
                int srow_0_1 = head_row_off[(f * 3 + 1) * 3 + 1];
                if (srow_0_1 >= 0) {
                    long long rowd_4 = (long long)srow_0_1 * C64 + c64;
                    {
                        unsigned _stv8_20_0 = __float_as_uint(dp[0 + 0]);
                        unsigned _stv8_20_1 = __float_as_uint(dp[0 + 1]);
                        unsigned _stv8_20_2 = __float_as_uint(dp[0 + 2]);
                        unsigned _stv8_20_3 = __float_as_uint(dp[0 + 3]);
                        unsigned _stv8_20_4 = __float_as_uint(dp[0 + 4]);
                        unsigned _stv8_20_5 = __float_as_uint(dp[0 + 5]);
                        unsigned _stv8_20_6 = __float_as_uint(dp[0 + 6]);
                        unsigned _stv8_20_7 = __float_as_uint(dp[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dpre_head + (rowd_4))), "r"(_stv8_20_0), "r"(_stv8_20_1), "r"(_stv8_20_2), "r"(_stv8_20_3), "r"(_stv8_20_4), "r"(_stv8_20_5), "r"(_stv8_20_6), "r"(_stv8_20_7) : "memory");
                    }
                }
                int srow_1_1 = head_row_off[(f * 3 + 1) * 3 + 2];
                if (srow_1_1 >= 0) {
                    long long rowd_5 = (long long)srow_1_1 * C64 + c64;
                    {
                        unsigned _stv8_21_0 = __float_as_uint(dp[0 + 0]);
                        unsigned _stv8_21_1 = __float_as_uint(dp[0 + 1]);
                        unsigned _stv8_21_2 = __float_as_uint(dp[0 + 2]);
                        unsigned _stv8_21_3 = __float_as_uint(dp[0 + 3]);
                        unsigned _stv8_21_4 = __float_as_uint(dp[0 + 4]);
                        unsigned _stv8_21_5 = __float_as_uint(dp[0 + 5]);
                        unsigned _stv8_21_6 = __float_as_uint(dp[0 + 6]);
                        unsigned _stv8_21_7 = __float_as_uint(dp[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dpre_head + (rowd_5))), "r"(_stv8_21_0), "r"(_stv8_21_1), "r"(_stv8_21_2), "r"(_stv8_21_3), "r"(_stv8_21_4), "r"(_stv8_21_5), "r"(_stv8_21_6), "r"(_stv8_21_7) : "memory");
                    }
                }
            }
            if (n_head > 2) {
                long long tok_2 = (long long)(start + 2) * C64 + c64;
                {
                    const uint4* _vptr_22 = reinterpret_cast<const uint4*>(x + tok_2);
                    uint4 _vld_22[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_22[_blk] = _vptr_22[_blk];
                        uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) << 16);
                            (&win[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_23 = reinterpret_cast<const uint4*>(dy + tok_2);
                    uint4 _vld_23[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_23[_blk] = _vptr_23[_blk];
                        uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&dyf[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) << 16);
                            (&dyf[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    #pragma unroll
                    for (int kk_2 = 0; kk_2 < 4; kk_2++) {
                        w4[kk_2] = wr[j_3 * 4 + kk_2];
                    }
                    float _fma_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(w4[1]), "f"(win[j_3]), "f"(mul_f32_rn_noftz(w4[0], win[24 + j_3])));
                    float _fma_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(w4[2]), "f"(win[8 + j_3]), "f"(_fma_25));
                    float _fma_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(w4[3]), "f"(win[16 + j_3]), "f"(_fma_24));
                    float e_2 = (-_fma_2) * 1.4426950408889634f;
                    float _exp2_26 = approx_exp2(e_2);
                    float den_2 = 1.0f + _exp2_26;
                    float sg_2 = approx_rcp(den_2);
                    float _fma_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(_fma_2), "f"(1.0f - sg_2), "f"(1.0f));
                    dp[j_3] = dyf[j_3] * (sg_2 * _fma_27);
                }
                int srow_3 = head_row_off[(f * 3 + 2) * 3];
                if (srow_3 >= 0) {
                    long long rowd_6 = (long long)srow_3 * C64 + c64;
                    {
                        unsigned _stv8_28_0 = __float_as_uint(dp[0 + 0]);
                        unsigned _stv8_28_1 = __float_as_uint(dp[0 + 1]);
                        unsigned _stv8_28_2 = __float_as_uint(dp[0 + 2]);
                        unsigned _stv8_28_3 = __float_as_uint(dp[0 + 3]);
                        unsigned _stv8_28_4 = __float_as_uint(dp[0 + 4]);
                        unsigned _stv8_28_5 = __float_as_uint(dp[0 + 5]);
                        unsigned _stv8_28_6 = __float_as_uint(dp[0 + 6]);
                        unsigned _stv8_28_7 = __float_as_uint(dp[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dpre_head + (rowd_6))), "r"(_stv8_28_0), "r"(_stv8_28_1), "r"(_stv8_28_2), "r"(_stv8_28_3), "r"(_stv8_28_4), "r"(_stv8_28_5), "r"(_stv8_28_6), "r"(_stv8_28_7) : "memory");
                    }
                }
                int srow_0_2 = head_row_off[(f * 3 + 2) * 3 + 1];
                if (srow_0_2 >= 0) {
                    long long rowd_7 = (long long)srow_0_2 * C64 + c64;
                    {
                        unsigned _stv8_29_0 = __float_as_uint(dp[0 + 0]);
                        unsigned _stv8_29_1 = __float_as_uint(dp[0 + 1]);
                        unsigned _stv8_29_2 = __float_as_uint(dp[0 + 2]);
                        unsigned _stv8_29_3 = __float_as_uint(dp[0 + 3]);
                        unsigned _stv8_29_4 = __float_as_uint(dp[0 + 4]);
                        unsigned _stv8_29_5 = __float_as_uint(dp[0 + 5]);
                        unsigned _stv8_29_6 = __float_as_uint(dp[0 + 6]);
                        unsigned _stv8_29_7 = __float_as_uint(dp[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dpre_head + (rowd_7))), "r"(_stv8_29_0), "r"(_stv8_29_1), "r"(_stv8_29_2), "r"(_stv8_29_3), "r"(_stv8_29_4), "r"(_stv8_29_5), "r"(_stv8_29_6), "r"(_stv8_29_7) : "memory");
                    }
                }
                int srow_1_2 = head_row_off[(f * 3 + 2) * 3 + 2];
                if (srow_1_2 >= 0) {
                    long long rowd_8 = (long long)srow_1_2 * C64 + c64;
                    {
                        unsigned _stv8_30_0 = __float_as_uint(dp[0 + 0]);
                        unsigned _stv8_30_1 = __float_as_uint(dp[0 + 1]);
                        unsigned _stv8_30_2 = __float_as_uint(dp[0 + 2]);
                        unsigned _stv8_30_3 = __float_as_uint(dp[0 + 3]);
                        unsigned _stv8_30_4 = __float_as_uint(dp[0 + 4]);
                        unsigned _stv8_30_5 = __float_as_uint(dp[0 + 5]);
                        unsigned _stv8_30_6 = __float_as_uint(dp[0 + 6]);
                        unsigned _stv8_30_7 = __float_as_uint(dp[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dpre_head + (rowd_8))), "r"(_stv8_30_0), "r"(_stv8_30_1), "r"(_stv8_30_2), "r"(_stv8_30_3), "r"(_stv8_30_4), "r"(_stv8_30_5), "r"(_stv8_30_6), "r"(_stv8_30_7) : "memory");
                    }
                }
            }
        }
    }
}

} // extern "C"
