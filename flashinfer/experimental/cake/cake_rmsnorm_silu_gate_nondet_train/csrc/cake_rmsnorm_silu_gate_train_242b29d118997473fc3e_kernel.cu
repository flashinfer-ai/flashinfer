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

#define LAUNCH_MIN_BLOCKS 8

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


__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_train_242b29d118997473fc3e(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int col = (tid & 15) * 8;
    int n = bid * 8 + (tid >> 4);
    int src = ((n) < (rows - 1) ? (n) : (rows - 1));
    long long off = (long long)src * 128ll + (long long)col;
    unsigned int xw[4];
    unsigned int zw[4];
    {
        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(x + off) + 0);
        xw[0 + 0] = _uv4_0.x;
        xw[0 + 1] = _uv4_0.y;
        xw[0 + 2] = _uv4_0.z;
        xw[0 + 3] = _uv4_0.w;
    }
    {
        uint4 _uv4_1 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(z + off) + 0);
        zw[0 + 0] = _uv4_1.x;
        zw[0 + 1] = _uv4_1.y;
        zw[0 + 2] = _uv4_1.z;
        zw[0 + 3] = _uv4_1.w;
    }
    float wv[8];
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(w + col + 0);
        uint4 _vld_2[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_2[_blk] = _vptr_2[_blk];
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&wv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&wv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
    }
    float xs[8];
    float zs[8];
    xs[0] = __uint_as_float(xw[0] << 16);
    xs[1] = __uint_as_float(xw[0] & 4294901760u);
    zs[0] = __uint_as_float(zw[0] << 16);
    zs[1] = __uint_as_float(zw[0] & 4294901760u);
    xs[2] = __uint_as_float(xw[1] << 16);
    xs[3] = __uint_as_float(xw[1] & 4294901760u);
    zs[2] = __uint_as_float(zw[1] << 16);
    zs[3] = __uint_as_float(zw[1] & 4294901760u);
    xs[4] = __uint_as_float(xw[2] << 16);
    xs[5] = __uint_as_float(xw[2] & 4294901760u);
    zs[4] = __uint_as_float(zw[2] << 16);
    zs[5] = __uint_as_float(zw[2] & 4294901760u);
    xs[6] = __uint_as_float(xw[3] << 16);
    xs[7] = __uint_as_float(xw[3] & 4294901760u);
    zs[6] = __uint_as_float(zw[3] << 16);
    zs[7] = __uint_as_float(zw[3] & 4294901760u);
    float2 scale2 = make_float2(2.3283064e-10f, 2.3283064e-10f);
    float2 z2_0 = make_float2(zs[0], zs[1]);
    float2 _f32x2_mul_b_3 = make_float2(-1.442695f, -1.442695f);
    float2 t2_0;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_3));
    float _exp2_4 = approx_exp2(t2_0.x);
    float _exp2_5 = approx_exp2(t2_0.y);
    float2 d2_0 = fma_f32x2_rn_noftz(make_float2(_exp2_4, _exp2_5), scale2, scale2);
    float _rcp_6 = approx_rcp(d2_0.x);
    float _rcp_7 = approx_rcp(d2_0.y);
    float2 _f32x2_mul_b_8 = mul_f32x2_noftz(make_float2(_rcp_6, _rcp_7), scale2);
    float2 g2_0;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
    zs[0] = g2_0.x;
    zs[1] = g2_0.y;
    float2 z2_1 = make_float2(zs[2], zs[3]);
    float2 _f32x2_mul_b_9 = make_float2(-1.442695f, -1.442695f);
    float2 t2_1;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
    float _exp2_10 = approx_exp2(t2_1.x);
    float _exp2_11 = approx_exp2(t2_1.y);
    float2 d2_1 = fma_f32x2_rn_noftz(make_float2(_exp2_10, _exp2_11), scale2, scale2);
    float _rcp_12 = approx_rcp(d2_1.x);
    float _rcp_13 = approx_rcp(d2_1.y);
    float2 _f32x2_mul_b_14 = mul_f32x2_noftz(make_float2(_rcp_12, _rcp_13), scale2);
    float2 g2_1;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
    zs[2] = g2_1.x;
    zs[3] = g2_1.y;
    float2 z2_2 = make_float2(zs[4], zs[5]);
    float2 _f32x2_mul_b_15 = make_float2(-1.442695f, -1.442695f);
    float2 t2_2;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_15));
    float _exp2_16 = approx_exp2(t2_2.x);
    float _exp2_17 = approx_exp2(t2_2.y);
    float2 d2_2 = fma_f32x2_rn_noftz(make_float2(_exp2_16, _exp2_17), scale2, scale2);
    float _rcp_18 = approx_rcp(d2_2.x);
    float _rcp_19 = approx_rcp(d2_2.y);
    float2 _f32x2_mul_b_20 = mul_f32x2_noftz(make_float2(_rcp_18, _rcp_19), scale2);
    float2 g2_2;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
    zs[4] = g2_2.x;
    zs[5] = g2_2.y;
    float2 z2_3 = make_float2(zs[6], zs[7]);
    float2 _f32x2_mul_b_21 = make_float2(-1.442695f, -1.442695f);
    float2 t2_3;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
    float _exp2_22 = approx_exp2(t2_3.x);
    float _exp2_23 = approx_exp2(t2_3.y);
    float2 d2_3 = fma_f32x2_rn_noftz(make_float2(_exp2_22, _exp2_23), scale2, scale2);
    float _rcp_24 = approx_rcp(d2_3.x);
    float _rcp_25 = approx_rcp(d2_3.y);
    float2 _f32x2_mul_b_26 = mul_f32x2_noftz(make_float2(_rcp_24, _rcp_25), scale2);
    float2 g2_3;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
    zs[6] = g2_3.x;
    zs[7] = g2_3.y;
    float total = 0.0f;
    total = __fmaf_rn(xs[0], xs[0], total);
    total = __fmaf_rn(xs[1], xs[1], total);
    total = __fmaf_rn(xs[2], xs[2], total);
    total = __fmaf_rn(xs[3], xs[3], total);
    total = __fmaf_rn(xs[4], xs[4], total);
    total = __fmaf_rn(xs[5], xs[5], total);
    total = __fmaf_rn(xs[6], xs[6], total);
    total = __fmaf_rn(xs[7], xs[7], total);
    float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
    total = total + _shfl_xor_sync_27;
    float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
    total = total + _shfl_xor_sync_28;
    float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
    total = total + _shfl_xor_sync_29;
    float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
    total = total + _shfl_xor_sync_30;
    float rstd;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
    float2 rstd2 = make_float2(rstd, rstd);
    float2 _f32x2_mul_a_31 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[0], xs[1]), rstd2), make_float2(wv[0], wv[1]));
    float2 _f32x2_mul_b_32 = make_float2(zs[0], zs[1]);
    float2 y2_0;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
    xs[0] = y2_0.x;
    xs[1] = y2_0.y;
    float2 _f32x2_mul_a_33 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[2], xs[3]), rstd2), make_float2(wv[2], wv[3]));
    float2 _f32x2_mul_b_34 = make_float2(zs[2], zs[3]);
    float2 y2_1;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
    xs[2] = y2_1.x;
    xs[3] = y2_1.y;
    float2 _f32x2_mul_a_35 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[4], xs[5]), rstd2), make_float2(wv[4], wv[5]));
    float2 _f32x2_mul_b_36 = make_float2(zs[4], zs[5]);
    float2 y2_2;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
    xs[4] = y2_2.x;
    xs[5] = y2_2.y;
    float2 _f32x2_mul_a_37 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[6], xs[7]), rstd2), make_float2(wv[6], wv[7]));
    float2 _f32x2_mul_b_38 = make_float2(zs[6], zs[7]);
    float2 y2_3;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
    xs[6] = y2_3.x;
    xs[7] = y2_3.y;
    unsigned int packed[4];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
        packed[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
        packed[_lp + 1] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
        packed[_lp + 2] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
        packed[_lp + 3] = *(uint32_t*)&_bf2;
    }
    if (n < rows) {
        reinterpret_cast<int4*>(y + ((long long)n * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
    }

    // === Task calls (declaration order) ===
}

} // extern "C"
