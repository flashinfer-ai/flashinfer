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
kernel_cake_causal_conv1d_train_138311f76132280b1e94_kcp(__nv_bfloat16* __restrict__ x, float* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ hist, int* __restrict__ hist_row_off, int C, int num_docs)
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
    float out[8];
    #pragma unroll 1
    for (int item = bid; item < n_items; item += num_bids) {
        int f = item / n_cb;
        int c0 = (item - f * n_cb) * 1024 + tid * 8;
        int start = cu[f];
        int n_head = ((cu[f + 1] - start) < (3) ? (cu[f + 1] - start) : (3));
        int wanted = 0;
        #pragma unroll
        for (int r = 0; r < 3; r++) {
            if (hist_row_off[f * 3 + r] >= 0) {
                wanted = 1;
            }
        }
        if (wanted != 0 && n_head > 0 && c0 < C) {
            long long c64 = (long long)c0;
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
            int hrow_0 = hist_row_off[f * 3 + 2];
            if (hrow_0 >= 0) {
                long long rowh_0 = (long long)hrow_0 * C64 + c64;
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(hist + rowh_0);
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
            int hrow_1 = hist_row_off[f * 3 + 2 - 1];
            if (hrow_1 >= 0) {
                long long rowh_1 = (long long)hrow_1 * C64 + c64;
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
            int hrow_2 = hist_row_off[f * 3 + 2 - 2];
            if (hrow_2 >= 0) {
                long long rowh_2 = (long long)hrow_2 * C64 + c64;
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
                long long tok_0 = (long long)start * C64 + c64;
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(x + tok_0);
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
                #pragma unroll
                for (int j_0 = 0; j_0 < 8; j_0++) {
                    #pragma unroll
                    for (int kk_0 = 0; kk_0 < 4; kk_0++) {
                        w4[kk_0] = wr[j_0 * 4 + kk_0];
                    }
                    float _fma_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(w4[1]), "f"(win[16 + j_0]), "f"(mul_f32_rn_noftz(w4[0], win[8 + j_0])));
                    float _fma_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(w4[2]), "f"(win[24 + j_0]), "f"(_fma_6));
                    float _fma_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(w4[3]), "f"(win[j_0]), "f"(_fma_5));
                    float e = -_fma_0 * 1.442695f;
                    float _exp2_7 = approx_exp2(e);
                    float den = 1.0f + _exp2_7;
                    float _rcp_8 = approx_rcp(den);
                    out[j_0] = _fma_0 * _rcp_8;
                }
                unsigned int packed_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                    packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 2], out[_lp*2+1 + 2]));
                    packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 4], out[_lp*2+1 + 4]));
                    packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 6], out[_lp*2+1 + 6]));
                    packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(y + (tok_0))[0] = reinterpret_cast<int4*>(packed_0)[0];
            }
            if (n_head > 1) {
                long long tok_1 = (long long)(start + 1) * C64 + c64;
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(x + tok_1);
                    uint4 _vld_9[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_9[_blk] = _vptr_9[_blk];
                        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                            (&win[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    #pragma unroll
                    for (int kk_1 = 0; kk_1 < 4; kk_1++) {
                        w4[kk_1] = wr[j_1 * 4 + kk_1];
                    }
                    float _fma_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(w4[1]), "f"(win[24 + j_1]), "f"(mul_f32_rn_noftz(w4[0], win[16 + j_1])));
                    float _fma_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(w4[2]), "f"(win[j_1]), "f"(_fma_11));
                    float _fma_0_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0_1) : "f"(w4[3]), "f"(win[8 + j_1]), "f"(_fma_10));
                    float e_1 = -_fma_0_1 * 1.442695f;
                    float _exp2_12 = approx_exp2(e_1);
                    float den_1 = 1.0f + _exp2_12;
                    float _rcp_13 = approx_rcp(den_1);
                    out[j_1] = _fma_0_1 * _rcp_13;
                }
                unsigned int packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 2], out[_lp*2+1 + 2]));
                    packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 4], out[_lp*2+1 + 4]));
                    packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 6], out[_lp*2+1 + 6]));
                    packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(y + (tok_1))[0] = reinterpret_cast<int4*>(packed_1)[0];
            }
            if (n_head > 2) {
                long long tok_2 = (long long)(start + 2) * C64 + c64;
                {
                    const uint4* _vptr_14 = reinterpret_cast<const uint4*>(x + tok_2);
                    uint4 _vld_14[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_14[_blk] = _vptr_14[_blk];
                        uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
                            (&win[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
                        }
                    }
                }
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    #pragma unroll
                    for (int kk_2 = 0; kk_2 < 4; kk_2++) {
                        w4[kk_2] = wr[j_2 * 4 + kk_2];
                    }
                    float _fma_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(w4[1]), "f"(win[j_2]), "f"(mul_f32_rn_noftz(w4[0], win[24 + j_2])));
                    float _fma_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(w4[2]), "f"(win[8 + j_2]), "f"(_fma_16));
                    float _fma_0_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0_2) : "f"(w4[3]), "f"(win[16 + j_2]), "f"(_fma_15));
                    float e_2 = -_fma_0_2 * 1.442695f;
                    float _exp2_17 = approx_exp2(e_2);
                    float den_2 = 1.0f + _exp2_17;
                    float _rcp_18 = approx_rcp(den_2);
                    out[j_2] = _fma_0_2 * _rcp_18;
                }
                unsigned int packed_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                    packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 2], out[_lp*2+1 + 2]));
                    packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 4], out[_lp*2+1 + 4]));
                    packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 6], out[_lp*2+1 + 6]));
                    packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(y + (tok_2))[0] = reinterpret_cast<int4*>(packed_2)[0];
            }
        }
    }
}

} // extern "C"
