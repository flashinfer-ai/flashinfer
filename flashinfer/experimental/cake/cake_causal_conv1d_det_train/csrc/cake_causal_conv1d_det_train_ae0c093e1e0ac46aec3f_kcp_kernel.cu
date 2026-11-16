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
kernel_cake_causal_conv1d_det_train_ae0c093e1e0ac46aec3f_kcp(__nv_bfloat16* __restrict__ x, float* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ hist, int* __restrict__ hist_row_off, float* __restrict__ dpre_tail, int* __restrict__ tail_row_off, int C, int num_docs)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long C64 = (long long)C;
    int n_cb = (C + 1023) / 1024;
    int n_items = num_docs * n_cb;
    float wr[32];
    float w4[4];
    float win[32];
    float dyf[8];
    float dp[24];
    float tailf[8];
    float acc[8];
    #pragma unroll 1
    for (int item = bid; item < n_items; item += num_bids) {
        int f = item / n_cb;
        int c0 = (item - f * n_cb) * 1024 + tid * 8;
        int start = cu[f];
        int end = cu[f + 1];
        int n_tail = ((end - start) < (3) ? (end - start) : (3));
        int wanted = 0;
        #pragma unroll
        for (int r = 0; r < 3; r++) {
            if (tail_row_off[f * 3 + r] >= 0) {
                wanted = 1;
            }
        }
        if (wanted != 0 && n_tail > 0 && c0 < C) {
            long long c64 = (long long)c0;
            int t_first = end - n_tail;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(w + (c64 + (long long)j) * 4ll);
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
            int tk_2_0 = t_first - 1;
            if (tk_2_0 >= start) {
                long long rowk_1_0 = (long long)tk_2_0 * C64 + c64;
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x + rowk_1_0);
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
            } else {
                int hrow_1_0 = hist_row_off[f * 3 + 2 - (start - t_first)];
                if (hrow_1_0 >= 0) {
                    long long rowh_1_0 = (long long)hrow_1_0 * C64 + c64;
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(hist + rowh_1_0);
                        uint4 _vld_2[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_2[_blk] = _vptr_2[_blk];
                            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                                (&win[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
            int tk_2_1 = t_first - 2;
            if (tk_2_1 >= start) {
                long long rowk_1_1 = (long long)tk_2_1 * C64 + c64;
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(x + rowk_1_1);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                            (&win[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                int hrow_1_1 = hist_row_off[f * 3 + 1 - (start - t_first)];
                if (hrow_1_1 >= 0) {
                    long long rowh_1_1 = (long long)hrow_1_1 * C64 + c64;
                    {
                        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(hist + rowh_1_1);
                        uint4 _vld_4[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_4[_blk] = _vptr_4[_blk];
                            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                (&win[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
            int tk_2_2 = t_first - 3;
            if (tk_2_2 >= start) {
                long long rowk_1_2 = (long long)tk_2_2 * C64 + c64;
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + rowk_1_2);
                    uint4 _vld_5[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_5[_blk] = _vptr_5[_blk];
                        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                            (&win[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                int hrow_1_2 = hist_row_off[f * 3 - (start - t_first)];
                if (hrow_1_2 >= 0) {
                    long long rowh_1_2 = (long long)hrow_1_2 * C64 + c64;
                    {
                        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(hist + rowh_1_2);
                        uint4 _vld_6[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_6[_blk] = _vptr_6[_blk];
                            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                (&win[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
            if (n_tail > 0) {
                long long tok_0 = (long long)t_first * C64 + c64;
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(x + tok_0);
                    uint4 _vld_7[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_7[_blk] = _vptr_7[_blk];
                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                            (&win[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(dy + tok_0);
                    uint4 _vld_8[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_8[_blk] = _vptr_8[_blk];
                        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&dyf[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                            (&dyf[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_0 = 0; j_0 < 8; j_0++) {
                    #pragma unroll
                    for (int kk_0 = 0; kk_0 < 4; kk_0++) {
                        w4[kk_0] = wr[j_0 * 4 + kk_0];
                    }
                    float _fma_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(w4[1]), "f"(win[16 + j_0]), "f"(mul_f32_rn_noftz(w4[0], win[8 + j_0])));
                    float _fma_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(w4[2]), "f"(win[24 + j_0]), "f"(_fma_10));
                    float fma;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(w4[3]), "f"(win[j_0]), "f"(_fma_9));
                    float e = -fma * 1.442695f;
                    float _exp2_11 = approx_exp2(e);
                    float den = 1.0f + _exp2_11;
                    float sg = approx_rcp(den);
                    float _fma_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(fma), "f"(1.0f - sg), "f"(1.0f));
                    dp[j_0] = dyf[j_0] * (sg * _fma_12);
                }
            }
            if (n_tail > 1) {
                long long tok_1 = (long long)(t_first + 1) * C64 + c64;
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
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    #pragma unroll
                    for (int kk_1 = 0; kk_1 < 4; kk_1++) {
                        w4[kk_1] = wr[j_1 * 4 + kk_1];
                    }
                    float _fma_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(w4[1]), "f"(win[24 + j_1]), "f"(mul_f32_rn_noftz(w4[0], win[16 + j_1])));
                    float _fma_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(w4[2]), "f"(win[j_1]), "f"(_fma_16));
                    float fma_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(w4[3]), "f"(win[8 + j_1]), "f"(_fma_15));
                    float e_1 = -fma_1 * 1.442695f;
                    float _exp2_17 = approx_exp2(e_1);
                    float den_1 = 1.0f + _exp2_17;
                    float sg_1 = approx_rcp(den_1);
                    float _fma_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(fma_1), "f"(1.0f - sg_1), "f"(1.0f));
                    dp[8 + j_1] = dyf[j_1] * (sg_1 * _fma_18);
                }
            }
            if (n_tail > 2) {
                long long tok_2 = (long long)(t_first + 2) * C64 + c64;
                {
                    const uint4* _vptr_19 = reinterpret_cast<const uint4*>(x + tok_2);
                    uint4 _vld_19[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_19[_blk] = _vptr_19[_blk];
                        uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                            (&win[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_20 = reinterpret_cast<const uint4*>(dy + tok_2);
                    uint4 _vld_20[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_20[_blk] = _vptr_20[_blk];
                        uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&dyf[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
                            (&dyf[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    #pragma unroll
                    for (int kk_2 = 0; kk_2 < 4; kk_2++) {
                        w4[kk_2] = wr[j_2 * 4 + kk_2];
                    }
                    float _fma_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(w4[1]), "f"(win[j_2]), "f"(mul_f32_rn_noftz(w4[0], win[24 + j_2])));
                    float _fma_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(w4[2]), "f"(win[8 + j_2]), "f"(_fma_22));
                    float fma_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(w4[3]), "f"(win[16 + j_2]), "f"(_fma_21));
                    float e_2 = -fma_2 * 1.442695f;
                    float _exp2_23 = approx_exp2(e_2);
                    float den_2 = 1.0f + _exp2_23;
                    float sg_2 = approx_rcp(den_2);
                    float _fma_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(fma_2), "f"(1.0f - sg_2), "f"(1.0f));
                    dp[16 + j_2] = dyf[j_2] * (sg_2 * _fma_24);
                }
            }
            if (n_tail > 0) {
                #pragma unroll
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    acc[j_3] = wr[j_3 * 4 + 3] * dp[j_3];
                }
                if (1 < n_tail) {
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 8; j_4++) {
                        float _fma_25;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(wr[j_4 * 4 + 3 - 1]), "f"(dp[8 + j_4]), "f"(acc[j_4]));
                        acc[j_4] = _fma_25;
                    }
                } else {
                    int trow = tail_row_off[f * 3 + (1 - n_tail)];
                    if (trow >= 0) {
                        long long rowr = (long long)trow * C64 + c64;
                        {
                            unsigned _ldv8_26_0;
                            unsigned _ldv8_26_1;
                            unsigned _ldv8_26_2;
                            unsigned _ldv8_26_3;
                            unsigned _ldv8_26_4;
                            unsigned _ldv8_26_5;
                            unsigned _ldv8_26_6;
                            unsigned _ldv8_26_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_26_0), "=r"(_ldv8_26_1), "=r"(_ldv8_26_2), "=r"(_ldv8_26_3), "=r"(_ldv8_26_4), "=r"(_ldv8_26_5), "=r"(_ldv8_26_6), "=r"(_ldv8_26_7) : "l"((const void*)(dpre_tail + (rowr))) : "memory");
                            tailf[0 + 0] = __uint_as_float(_ldv8_26_0);
                            tailf[0 + 1] = __uint_as_float(_ldv8_26_1);
                            tailf[0 + 2] = __uint_as_float(_ldv8_26_2);
                            tailf[0 + 3] = __uint_as_float(_ldv8_26_3);
                            tailf[0 + 4] = __uint_as_float(_ldv8_26_4);
                            tailf[0 + 5] = __uint_as_float(_ldv8_26_5);
                            tailf[0 + 6] = __uint_as_float(_ldv8_26_6);
                            tailf[0 + 7] = __uint_as_float(_ldv8_26_7);
                        }
                        #pragma unroll
                        for (int j_5 = 0; j_5 < 8; j_5++) {
                            float _fma_27;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(wr[j_5 * 4 + 3 - 1]), "f"(tailf[j_5]), "f"(acc[j_5]));
                            acc[j_5] = _fma_27;
                        }
                    }
                }
                if (2 < n_tail) {
                    #pragma unroll
                    for (int j_5_1 = 0; j_5_1 < 8; j_5_1++) {
                        float _fma_28;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(wr[j_5_1 * 4 + 3 - 2]), "f"(dp[16 + j_5_1]), "f"(acc[j_5_1]));
                        acc[j_5_1] = _fma_28;
                    }
                } else {
                    int trow_1 = tail_row_off[f * 3 + (2 - n_tail)];
                    if (trow_1 >= 0) {
                        long long rowr_1 = (long long)trow_1 * C64 + c64;
                        {
                            unsigned _ldv8_29_0;
                            unsigned _ldv8_29_1;
                            unsigned _ldv8_29_2;
                            unsigned _ldv8_29_3;
                            unsigned _ldv8_29_4;
                            unsigned _ldv8_29_5;
                            unsigned _ldv8_29_6;
                            unsigned _ldv8_29_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_29_0), "=r"(_ldv8_29_1), "=r"(_ldv8_29_2), "=r"(_ldv8_29_3), "=r"(_ldv8_29_4), "=r"(_ldv8_29_5), "=r"(_ldv8_29_6), "=r"(_ldv8_29_7) : "l"((const void*)(dpre_tail + (rowr_1))) : "memory");
                            tailf[0 + 0] = __uint_as_float(_ldv8_29_0);
                            tailf[0 + 1] = __uint_as_float(_ldv8_29_1);
                            tailf[0 + 2] = __uint_as_float(_ldv8_29_2);
                            tailf[0 + 3] = __uint_as_float(_ldv8_29_3);
                            tailf[0 + 4] = __uint_as_float(_ldv8_29_4);
                            tailf[0 + 5] = __uint_as_float(_ldv8_29_5);
                            tailf[0 + 6] = __uint_as_float(_ldv8_29_6);
                            tailf[0 + 7] = __uint_as_float(_ldv8_29_7);
                        }
                        #pragma unroll
                        for (int j_6 = 0; j_6 < 8; j_6++) {
                            float _fma_30;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(wr[j_6 * 4 + 3 - 2]), "f"(tailf[j_6]), "f"(acc[j_6]));
                            acc[j_6] = _fma_30;
                        }
                    }
                }
                int trow_2 = tail_row_off[f * 3 + (3 - n_tail)];
                if (trow_2 >= 0) {
                    long long rowr_2 = (long long)trow_2 * C64 + c64;
                    {
                        unsigned _ldv8_31_0;
                        unsigned _ldv8_31_1;
                        unsigned _ldv8_31_2;
                        unsigned _ldv8_31_3;
                        unsigned _ldv8_31_4;
                        unsigned _ldv8_31_5;
                        unsigned _ldv8_31_6;
                        unsigned _ldv8_31_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_31_0), "=r"(_ldv8_31_1), "=r"(_ldv8_31_2), "=r"(_ldv8_31_3), "=r"(_ldv8_31_4), "=r"(_ldv8_31_5), "=r"(_ldv8_31_6), "=r"(_ldv8_31_7) : "l"((const void*)(dpre_tail + (rowr_2))) : "memory");
                        tailf[0 + 0] = __uint_as_float(_ldv8_31_0);
                        tailf[0 + 1] = __uint_as_float(_ldv8_31_1);
                        tailf[0 + 2] = __uint_as_float(_ldv8_31_2);
                        tailf[0 + 3] = __uint_as_float(_ldv8_31_3);
                        tailf[0 + 4] = __uint_as_float(_ldv8_31_4);
                        tailf[0 + 5] = __uint_as_float(_ldv8_31_5);
                        tailf[0 + 6] = __uint_as_float(_ldv8_31_6);
                        tailf[0 + 7] = __uint_as_float(_ldv8_31_7);
                    }
                    #pragma unroll
                    for (int j_7 = 0; j_7 < 8; j_7++) {
                        float _fma_32;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(wr[j_7 * 4 + 3 - 3]), "f"(tailf[j_7]), "f"(acc[j_7]));
                        acc[j_7] = _fma_32;
                    }
                }
                unsigned int packed_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 2], acc[_lp*2+1 + 2]));
                    packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 4], acc[_lp*2+1 + 4]));
                    packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 6], acc[_lp*2+1 + 6]));
                    packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                long long row_0 = (long long)t_first * C64 + c64;
                reinterpret_cast<int4*>(dx + (row_0))[0] = reinterpret_cast<int4*>(packed_0)[0];
            }
            if (n_tail > 1) {
                #pragma unroll
                for (int j_6_1 = 0; j_6_1 < 8; j_6_1++) {
                    acc[j_6_1] = wr[j_6_1 * 4 + 3] * dp[8 + j_6_1];
                }
                if (2 < n_tail) {
                    #pragma unroll
                    for (int j_7_1 = 0; j_7_1 < 8; j_7_1++) {
                        float _fma_33;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(wr[j_7_1 * 4 + 3 - 1]), "f"(dp[16 + j_7_1]), "f"(acc[j_7_1]));
                        acc[j_7_1] = _fma_33;
                    }
                } else {
                    int trow_3 = tail_row_off[f * 3 + (2 - n_tail)];
                    if (trow_3 >= 0) {
                        long long rowr_3 = (long long)trow_3 * C64 + c64;
                        {
                            unsigned _ldv8_34_0;
                            unsigned _ldv8_34_1;
                            unsigned _ldv8_34_2;
                            unsigned _ldv8_34_3;
                            unsigned _ldv8_34_4;
                            unsigned _ldv8_34_5;
                            unsigned _ldv8_34_6;
                            unsigned _ldv8_34_7;
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_ldv8_34_0), "=r"(_ldv8_34_1), "=r"(_ldv8_34_2), "=r"(_ldv8_34_3), "=r"(_ldv8_34_4), "=r"(_ldv8_34_5), "=r"(_ldv8_34_6), "=r"(_ldv8_34_7) : "l"((const void*)(dpre_tail + (rowr_3))) : "memory");
                            tailf[0 + 0] = __uint_as_float(_ldv8_34_0);
                            tailf[0 + 1] = __uint_as_float(_ldv8_34_1);
                            tailf[0 + 2] = __uint_as_float(_ldv8_34_2);
                            tailf[0 + 3] = __uint_as_float(_ldv8_34_3);
                            tailf[0 + 4] = __uint_as_float(_ldv8_34_4);
                            tailf[0 + 5] = __uint_as_float(_ldv8_34_5);
                            tailf[0 + 6] = __uint_as_float(_ldv8_34_6);
                            tailf[0 + 7] = __uint_as_float(_ldv8_34_7);
                        }
                        #pragma unroll
                        for (int j_8 = 0; j_8 < 8; j_8++) {
                            float _fma_35;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(wr[j_8 * 4 + 3 - 1]), "f"(tailf[j_8]), "f"(acc[j_8]));
                            acc[j_8] = _fma_35;
                        }
                    }
                }
                int trow_4 = tail_row_off[f * 3 + (3 - n_tail)];
                if (trow_4 >= 0) {
                    long long rowr_4 = (long long)trow_4 * C64 + c64;
                    {
                        unsigned _ldv8_36_0;
                        unsigned _ldv8_36_1;
                        unsigned _ldv8_36_2;
                        unsigned _ldv8_36_3;
                        unsigned _ldv8_36_4;
                        unsigned _ldv8_36_5;
                        unsigned _ldv8_36_6;
                        unsigned _ldv8_36_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_36_0), "=r"(_ldv8_36_1), "=r"(_ldv8_36_2), "=r"(_ldv8_36_3), "=r"(_ldv8_36_4), "=r"(_ldv8_36_5), "=r"(_ldv8_36_6), "=r"(_ldv8_36_7) : "l"((const void*)(dpre_tail + (rowr_4))) : "memory");
                        tailf[0 + 0] = __uint_as_float(_ldv8_36_0);
                        tailf[0 + 1] = __uint_as_float(_ldv8_36_1);
                        tailf[0 + 2] = __uint_as_float(_ldv8_36_2);
                        tailf[0 + 3] = __uint_as_float(_ldv8_36_3);
                        tailf[0 + 4] = __uint_as_float(_ldv8_36_4);
                        tailf[0 + 5] = __uint_as_float(_ldv8_36_5);
                        tailf[0 + 6] = __uint_as_float(_ldv8_36_6);
                        tailf[0 + 7] = __uint_as_float(_ldv8_36_7);
                    }
                    #pragma unroll
                    for (int j_9 = 0; j_9 < 8; j_9++) {
                        float _fma_37;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(wr[j_9 * 4 + 3 - 2]), "f"(tailf[j_9]), "f"(acc[j_9]));
                        acc[j_9] = _fma_37;
                    }
                }
                int trow_0 = tail_row_off[f * 3 + (4 - n_tail)];
                if (trow_0 >= 0) {
                    long long rowr_5 = (long long)trow_0 * C64 + c64;
                    {
                        unsigned _ldv8_38_0;
                        unsigned _ldv8_38_1;
                        unsigned _ldv8_38_2;
                        unsigned _ldv8_38_3;
                        unsigned _ldv8_38_4;
                        unsigned _ldv8_38_5;
                        unsigned _ldv8_38_6;
                        unsigned _ldv8_38_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_38_0), "=r"(_ldv8_38_1), "=r"(_ldv8_38_2), "=r"(_ldv8_38_3), "=r"(_ldv8_38_4), "=r"(_ldv8_38_5), "=r"(_ldv8_38_6), "=r"(_ldv8_38_7) : "l"((const void*)(dpre_tail + (rowr_5))) : "memory");
                        tailf[0 + 0] = __uint_as_float(_ldv8_38_0);
                        tailf[0 + 1] = __uint_as_float(_ldv8_38_1);
                        tailf[0 + 2] = __uint_as_float(_ldv8_38_2);
                        tailf[0 + 3] = __uint_as_float(_ldv8_38_3);
                        tailf[0 + 4] = __uint_as_float(_ldv8_38_4);
                        tailf[0 + 5] = __uint_as_float(_ldv8_38_5);
                        tailf[0 + 6] = __uint_as_float(_ldv8_38_6);
                        tailf[0 + 7] = __uint_as_float(_ldv8_38_7);
                    }
                    #pragma unroll
                    for (int j_10 = 0; j_10 < 8; j_10++) {
                        float _fma_39;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(wr[j_10 * 4 + 3 - 3]), "f"(tailf[j_10]), "f"(acc[j_10]));
                        acc[j_10] = _fma_39;
                    }
                }
                unsigned int packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 2], acc[_lp*2+1 + 2]));
                    packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 4], acc[_lp*2+1 + 4]));
                    packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 6], acc[_lp*2+1 + 6]));
                    packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                long long row_1 = (long long)(t_first + 1) * C64 + c64;
                reinterpret_cast<int4*>(dx + (row_1))[0] = reinterpret_cast<int4*>(packed_1)[0];
            }
            if (n_tail > 2) {
                #pragma unroll
                for (int j_8_1 = 0; j_8_1 < 8; j_8_1++) {
                    acc[j_8_1] = wr[j_8_1 * 4 + 3] * dp[16 + j_8_1];
                }
                int trow_5 = tail_row_off[f * 3 + (3 - n_tail)];
                if (trow_5 >= 0) {
                    long long rowr_6 = (long long)trow_5 * C64 + c64;
                    {
                        unsigned _ldv8_40_0;
                        unsigned _ldv8_40_1;
                        unsigned _ldv8_40_2;
                        unsigned _ldv8_40_3;
                        unsigned _ldv8_40_4;
                        unsigned _ldv8_40_5;
                        unsigned _ldv8_40_6;
                        unsigned _ldv8_40_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_40_0), "=r"(_ldv8_40_1), "=r"(_ldv8_40_2), "=r"(_ldv8_40_3), "=r"(_ldv8_40_4), "=r"(_ldv8_40_5), "=r"(_ldv8_40_6), "=r"(_ldv8_40_7) : "l"((const void*)(dpre_tail + (rowr_6))) : "memory");
                        tailf[0 + 0] = __uint_as_float(_ldv8_40_0);
                        tailf[0 + 1] = __uint_as_float(_ldv8_40_1);
                        tailf[0 + 2] = __uint_as_float(_ldv8_40_2);
                        tailf[0 + 3] = __uint_as_float(_ldv8_40_3);
                        tailf[0 + 4] = __uint_as_float(_ldv8_40_4);
                        tailf[0 + 5] = __uint_as_float(_ldv8_40_5);
                        tailf[0 + 6] = __uint_as_float(_ldv8_40_6);
                        tailf[0 + 7] = __uint_as_float(_ldv8_40_7);
                    }
                    #pragma unroll
                    for (int j_11 = 0; j_11 < 8; j_11++) {
                        float _fma_41;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(wr[j_11 * 4 + 3 - 1]), "f"(tailf[j_11]), "f"(acc[j_11]));
                        acc[j_11] = _fma_41;
                    }
                }
                int trow_1_1 = tail_row_off[f * 3 + (4 - n_tail)];
                if (trow_1_1 >= 0) {
                    long long rowr_7 = (long long)trow_1_1 * C64 + c64;
                    {
                        unsigned _ldv8_42_0;
                        unsigned _ldv8_42_1;
                        unsigned _ldv8_42_2;
                        unsigned _ldv8_42_3;
                        unsigned _ldv8_42_4;
                        unsigned _ldv8_42_5;
                        unsigned _ldv8_42_6;
                        unsigned _ldv8_42_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_42_0), "=r"(_ldv8_42_1), "=r"(_ldv8_42_2), "=r"(_ldv8_42_3), "=r"(_ldv8_42_4), "=r"(_ldv8_42_5), "=r"(_ldv8_42_6), "=r"(_ldv8_42_7) : "l"((const void*)(dpre_tail + (rowr_7))) : "memory");
                        tailf[0 + 0] = __uint_as_float(_ldv8_42_0);
                        tailf[0 + 1] = __uint_as_float(_ldv8_42_1);
                        tailf[0 + 2] = __uint_as_float(_ldv8_42_2);
                        tailf[0 + 3] = __uint_as_float(_ldv8_42_3);
                        tailf[0 + 4] = __uint_as_float(_ldv8_42_4);
                        tailf[0 + 5] = __uint_as_float(_ldv8_42_5);
                        tailf[0 + 6] = __uint_as_float(_ldv8_42_6);
                        tailf[0 + 7] = __uint_as_float(_ldv8_42_7);
                    }
                    #pragma unroll
                    for (int j_12 = 0; j_12 < 8; j_12++) {
                        float _fma_43;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(wr[j_12 * 4 + 3 - 2]), "f"(tailf[j_12]), "f"(acc[j_12]));
                        acc[j_12] = _fma_43;
                    }
                }
                int trow_2_1 = tail_row_off[f * 3 + (5 - n_tail)];
                if (trow_2_1 >= 0) {
                    long long rowr_8 = (long long)trow_2_1 * C64 + c64;
                    {
                        unsigned _ldv8_44_0;
                        unsigned _ldv8_44_1;
                        unsigned _ldv8_44_2;
                        unsigned _ldv8_44_3;
                        unsigned _ldv8_44_4;
                        unsigned _ldv8_44_5;
                        unsigned _ldv8_44_6;
                        unsigned _ldv8_44_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_44_0), "=r"(_ldv8_44_1), "=r"(_ldv8_44_2), "=r"(_ldv8_44_3), "=r"(_ldv8_44_4), "=r"(_ldv8_44_5), "=r"(_ldv8_44_6), "=r"(_ldv8_44_7) : "l"((const void*)(dpre_tail + (rowr_8))) : "memory");
                        tailf[0 + 0] = __uint_as_float(_ldv8_44_0);
                        tailf[0 + 1] = __uint_as_float(_ldv8_44_1);
                        tailf[0 + 2] = __uint_as_float(_ldv8_44_2);
                        tailf[0 + 3] = __uint_as_float(_ldv8_44_3);
                        tailf[0 + 4] = __uint_as_float(_ldv8_44_4);
                        tailf[0 + 5] = __uint_as_float(_ldv8_44_5);
                        tailf[0 + 6] = __uint_as_float(_ldv8_44_6);
                        tailf[0 + 7] = __uint_as_float(_ldv8_44_7);
                    }
                    #pragma unroll
                    for (int j_13 = 0; j_13 < 8; j_13++) {
                        float _fma_45;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(wr[j_13 * 4 + 3 - 3]), "f"(tailf[j_13]), "f"(acc[j_13]));
                        acc[j_13] = _fma_45;
                    }
                }
                unsigned int packed_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 2], acc[_lp*2+1 + 2]));
                    packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 4], acc[_lp*2+1 + 4]));
                    packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 6], acc[_lp*2+1 + 6]));
                    packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                }
                long long row_2 = (long long)(t_first + 2) * C64 + c64;
                reinterpret_cast<int4*>(dx + (row_2))[0] = reinterpret_cast<int4*>(packed_2)[0];
            }
        }
    }
}

} // extern "C"
