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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

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

#define LAUNCH_MIN_BLOCKS 3

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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_train_ff8010718cbdf7cfa830(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, CakeFastDivmod heads, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int group = tid >> 3;
    int col = (tid & 7) * 16;
    float wv[16];
    {
        const void* _v8p_0 = (const void*)(w + col + (0));
        uint32_t _v8_0_0[8];
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_v8_0_0[0]), "=r"(_v8_0_0[1]), "=r"(_v8_0_0[2]), "=r"(_v8_0_0[3]), "=r"(_v8_0_0[4]), "=r"(_v8_0_0[5]), "=r"(_v8_0_0[6]), "=r"(_v8_0_0[7]) : "l"((const char*)_v8p_0 + 0) : "memory");
        (&wv[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) << 16);
        (&wv[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) & 0xffff0000u);
        (&wv[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) << 16);
        (&wv[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) & 0xffff0000u);
        (&wv[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) << 16);
        (&wv[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) & 0xffff0000u);
        (&wv[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) << 16);
        (&wv[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) & 0xffff0000u);
        (&wv[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) << 16);
        (&wv[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) & 0xffff0000u);
        (&wv[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) << 16);
        (&wv[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) & 0xffff0000u);
        (&wv[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) << 16);
        (&wv[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) & 0xffff0000u);
        (&wv[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) << 16);
        (&wv[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) & 0xffff0000u);
    }
    int nblocks = (rows + 31) / 32;
    unsigned int xw[8];
    unsigned int zw[8];
    int src = ((bid * 32 + group) < (rows - 1) ? (bid * 32 + group) : (rows - 1));
    uint32_t tq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src);
    uint32_t th = (uint32_t)((unsigned int)src) - tq * (uint32_t)(heads.divisor);
    long long xoff = (long long)tq * x_tok + (long long)th * x_head + (long long)col;
    long long zoff = (long long)tq * z_tok + (long long)th * z_head + (long long)col;
    {
        uint32_t _iv_1_0;
        uint32_t _iv_1_1;
        uint32_t _iv_1_2;
        uint32_t _iv_1_3;
        uint32_t _iv_1_4;
        uint32_t _iv_1_5;
        uint32_t _iv_1_6;
        uint32_t _iv_1_7;
        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff) + (0))) : "memory");
        xw[0 + 0] = (uint32_t)_iv_1_0;
        xw[0 + 1] = (uint32_t)_iv_1_1;
        xw[0 + 2] = (uint32_t)_iv_1_2;
        xw[0 + 3] = (uint32_t)_iv_1_3;
        xw[0 + 4] = (uint32_t)_iv_1_4;
        xw[0 + 5] = (uint32_t)_iv_1_5;
        xw[0 + 6] = (uint32_t)_iv_1_6;
        xw[0 + 7] = (uint32_t)_iv_1_7;
    }
    {
        uint32_t _iv_2_0;
        uint32_t _iv_2_1;
        uint32_t _iv_2_2;
        uint32_t _iv_2_3;
        uint32_t _iv_2_4;
        uint32_t _iv_2_5;
        uint32_t _iv_2_6;
        uint32_t _iv_2_7;
        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff) + (0))) : "memory");
        zw[0 + 0] = (uint32_t)_iv_2_0;
        zw[0 + 1] = (uint32_t)_iv_2_1;
        zw[0 + 2] = (uint32_t)_iv_2_2;
        zw[0 + 3] = (uint32_t)_iv_2_3;
        zw[0 + 4] = (uint32_t)_iv_2_4;
        zw[0 + 5] = (uint32_t)_iv_2_5;
        zw[0 + 6] = (uint32_t)_iv_2_6;
        zw[0 + 7] = (uint32_t)_iv_2_7;
    }
    int packed = 0;
    long long heads64 = (long long)((uint32_t)(heads.divisor));
    if (x_head == 128ll) {
        if (z_head == 128ll) {
            if (x_tok == heads64 * 128ll) {
                if (z_tok == heads64 * 128ll) {
                    packed = 1;
                }
            }
        }
    }
    if (packed != 0) {
        if (tid == 0) {
            if (bid + num_bids < nblocks) {
                int first = (bid + num_bids) * 32;
                int _min_3 = ((rows - first) < (32) ? (rows - first) : (32));
                int nbytes = _min_3 * 256;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)first * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)first * 128ll), "r"((uint32_t)(nbytes)) : "memory");
            }
        }
    }
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        float xs[16];
        float zs[16];
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
        xs[8] = __uint_as_float(xw[4] << 16);
        xs[9] = __uint_as_float(xw[4] & 4294901760u);
        zs[8] = __uint_as_float(zw[4] << 16);
        zs[9] = __uint_as_float(zw[4] & 4294901760u);
        xs[10] = __uint_as_float(xw[5] << 16);
        xs[11] = __uint_as_float(xw[5] & 4294901760u);
        zs[10] = __uint_as_float(zw[5] << 16);
        zs[11] = __uint_as_float(zw[5] & 4294901760u);
        xs[12] = __uint_as_float(xw[6] << 16);
        xs[13] = __uint_as_float(xw[6] & 4294901760u);
        zs[12] = __uint_as_float(zw[6] << 16);
        zs[13] = __uint_as_float(zw[6] & 4294901760u);
        xs[14] = __uint_as_float(xw[7] << 16);
        xs[15] = __uint_as_float(xw[7] & 4294901760u);
        zs[14] = __uint_as_float(zw[7] << 16);
        zs[15] = __uint_as_float(zw[7] & 4294901760u);
        if (blk + num_bids < nblocks) {
            int src_0 = (((blk + num_bids) * 32 + group) < (rows - 1) ? ((blk + num_bids) * 32 + group) : (rows - 1));
            uint32_t tq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_0);
            uint32_t th_0 = (uint32_t)((unsigned int)src_0) - tq_0 * (uint32_t)(heads.divisor);
            long long xoff_0 = (long long)tq_0 * x_tok + (long long)th_0 * x_head + (long long)col;
            long long zoff_0 = (long long)tq_0 * z_tok + (long long)th_0 * z_head + (long long)col;
            {
                uint32_t _iv_4_0;
                uint32_t _iv_4_1;
                uint32_t _iv_4_2;
                uint32_t _iv_4_3;
                uint32_t _iv_4_4;
                uint32_t _iv_4_5;
                uint32_t _iv_4_6;
                uint32_t _iv_4_7;
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff_0) + (0))) : "memory");
                xw[0 + 0] = (uint32_t)_iv_4_0;
                xw[0 + 1] = (uint32_t)_iv_4_1;
                xw[0 + 2] = (uint32_t)_iv_4_2;
                xw[0 + 3] = (uint32_t)_iv_4_3;
                xw[0 + 4] = (uint32_t)_iv_4_4;
                xw[0 + 5] = (uint32_t)_iv_4_5;
                xw[0 + 6] = (uint32_t)_iv_4_6;
                xw[0 + 7] = (uint32_t)_iv_4_7;
            }
            {
                uint32_t _iv_5_0;
                uint32_t _iv_5_1;
                uint32_t _iv_5_2;
                uint32_t _iv_5_3;
                uint32_t _iv_5_4;
                uint32_t _iv_5_5;
                uint32_t _iv_5_6;
                uint32_t _iv_5_7;
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff_0) + (0))) : "memory");
                zw[0 + 0] = (uint32_t)_iv_5_0;
                zw[0 + 1] = (uint32_t)_iv_5_1;
                zw[0 + 2] = (uint32_t)_iv_5_2;
                zw[0 + 3] = (uint32_t)_iv_5_3;
                zw[0 + 4] = (uint32_t)_iv_5_4;
                zw[0 + 5] = (uint32_t)_iv_5_5;
                zw[0 + 6] = (uint32_t)_iv_5_6;
                zw[0 + 7] = (uint32_t)_iv_5_7;
            }
        }
        if (packed != 0) {
            if (tid == 0) {
                if (blk + 2 * num_bids < nblocks) {
                    int first_1 = (blk + 2 * num_bids) * 32;
                    int _min_6 = ((rows - first_1) < (32) ? (rows - first_1) : (32));
                    int nbytes_1 = _min_6 * 256;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)first_1 * 128ll), "r"((uint32_t)(nbytes_1)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)first_1 * 128ll), "r"((uint32_t)(nbytes_1)) : "memory");
                }
            }
        }
        float total = 0.0f;
        total = __fmaf_rn(xs[0], xs[0], total);
        total = __fmaf_rn(xs[1], xs[1], total);
        total = __fmaf_rn(xs[2], xs[2], total);
        total = __fmaf_rn(xs[3], xs[3], total);
        total = __fmaf_rn(xs[4], xs[4], total);
        total = __fmaf_rn(xs[5], xs[5], total);
        total = __fmaf_rn(xs[6], xs[6], total);
        total = __fmaf_rn(xs[7], xs[7], total);
        total = __fmaf_rn(xs[8], xs[8], total);
        total = __fmaf_rn(xs[9], xs[9], total);
        total = __fmaf_rn(xs[10], xs[10], total);
        total = __fmaf_rn(xs[11], xs[11], total);
        total = __fmaf_rn(xs[12], xs[12], total);
        total = __fmaf_rn(xs[13], xs[13], total);
        total = __fmaf_rn(xs[14], xs[14], total);
        total = __fmaf_rn(xs[15], xs[15], total);
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
        total = total + _shfl_xor_sync_7;
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        total = total + _shfl_xor_sync_8;
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
        total = total + _shfl_xor_sync_9;
        float rstd;
        asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
        float2 scale2 = make_float2(2.3283064e-10f, 2.3283064e-10f);
        float2 z2_0 = make_float2(zs[0], zs[1]);
        float2 _f32x2_mul_b_10 = make_float2(-1.442695f, -1.442695f);
        float2 t2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
        float _exp2_11 = approx_exp2(t2_0.x);
        float _exp2_12 = approx_exp2(t2_0.y);
        float2 d2_0 = fma_f32x2_rn_noftz(make_float2(_exp2_11, _exp2_12), scale2, scale2);
        float _rcp_13 = approx_rcp(d2_0.x);
        float _rcp_14 = approx_rcp(d2_0.y);
        float2 _f32x2_mul_b_15 = mul_f32x2_noftz(make_float2(_rcp_13, _rcp_14), scale2);
        float2 g2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_15));
        zs[0] = g2_0.x;
        zs[1] = g2_0.y;
        float2 z2_1 = make_float2(zs[2], zs[3]);
        float2 _f32x2_mul_b_16 = make_float2(-1.442695f, -1.442695f);
        float2 t2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        float _exp2_17 = approx_exp2(t2_1.x);
        float _exp2_18 = approx_exp2(t2_1.y);
        float2 d2_1 = fma_f32x2_rn_noftz(make_float2(_exp2_17, _exp2_18), scale2, scale2);
        float _rcp_19 = approx_rcp(d2_1.x);
        float _rcp_20 = approx_rcp(d2_1.y);
        float2 _f32x2_mul_b_21 = mul_f32x2_noftz(make_float2(_rcp_19, _rcp_20), scale2);
        float2 g2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
        zs[2] = g2_1.x;
        zs[3] = g2_1.y;
        float2 z2_2 = make_float2(zs[4], zs[5]);
        float2 _f32x2_mul_b_22 = make_float2(-1.442695f, -1.442695f);
        float2 t2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        float _exp2_23 = approx_exp2(t2_2.x);
        float _exp2_24 = approx_exp2(t2_2.y);
        float2 d2_2 = fma_f32x2_rn_noftz(make_float2(_exp2_23, _exp2_24), scale2, scale2);
        float _rcp_25 = approx_rcp(d2_2.x);
        float _rcp_26 = approx_rcp(d2_2.y);
        float2 _f32x2_mul_b_27 = mul_f32x2_noftz(make_float2(_rcp_25, _rcp_26), scale2);
        float2 g2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
        zs[4] = g2_2.x;
        zs[5] = g2_2.y;
        float2 z2_3 = make_float2(zs[6], zs[7]);
        float2 _f32x2_mul_b_28 = make_float2(-1.442695f, -1.442695f);
        float2 t2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
        float _exp2_29 = approx_exp2(t2_3.x);
        float _exp2_30 = approx_exp2(t2_3.y);
        float2 d2_3 = fma_f32x2_rn_noftz(make_float2(_exp2_29, _exp2_30), scale2, scale2);
        float _rcp_31 = approx_rcp(d2_3.x);
        float _rcp_32 = approx_rcp(d2_3.y);
        float2 _f32x2_mul_b_33 = mul_f32x2_noftz(make_float2(_rcp_31, _rcp_32), scale2);
        float2 g2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
        zs[6] = g2_3.x;
        zs[7] = g2_3.y;
        float2 z2_4 = make_float2(zs[8], zs[9]);
        float2 _f32x2_mul_b_34 = make_float2(-1.442695f, -1.442695f);
        float2 t2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
        float _exp2_35 = approx_exp2(t2_4.x);
        float _exp2_36 = approx_exp2(t2_4.y);
        float2 d2_4 = fma_f32x2_rn_noftz(make_float2(_exp2_35, _exp2_36), scale2, scale2);
        float _rcp_37 = approx_rcp(d2_4.x);
        float _rcp_38 = approx_rcp(d2_4.y);
        float2 _f32x2_mul_b_39 = mul_f32x2_noftz(make_float2(_rcp_37, _rcp_38), scale2);
        float2 g2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
        zs[8] = g2_4.x;
        zs[9] = g2_4.y;
        float2 z2_5 = make_float2(zs[10], zs[11]);
        float2 _f32x2_mul_b_40 = make_float2(-1.442695f, -1.442695f);
        float2 t2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
        float _exp2_41 = approx_exp2(t2_5.x);
        float _exp2_42 = approx_exp2(t2_5.y);
        float2 d2_5 = fma_f32x2_rn_noftz(make_float2(_exp2_41, _exp2_42), scale2, scale2);
        float _rcp_43 = approx_rcp(d2_5.x);
        float _rcp_44 = approx_rcp(d2_5.y);
        float2 _f32x2_mul_b_45 = mul_f32x2_noftz(make_float2(_rcp_43, _rcp_44), scale2);
        float2 g2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
        zs[10] = g2_5.x;
        zs[11] = g2_5.y;
        float2 z2_6 = make_float2(zs[12], zs[13]);
        float2 _f32x2_mul_b_46 = make_float2(-1.442695f, -1.442695f);
        float2 t2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_46));
        float _exp2_47 = approx_exp2(t2_6.x);
        float _exp2_48 = approx_exp2(t2_6.y);
        float2 d2_6 = fma_f32x2_rn_noftz(make_float2(_exp2_47, _exp2_48), scale2, scale2);
        float _rcp_49 = approx_rcp(d2_6.x);
        float _rcp_50 = approx_rcp(d2_6.y);
        float2 _f32x2_mul_b_51 = mul_f32x2_noftz(make_float2(_rcp_49, _rcp_50), scale2);
        float2 g2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
        zs[12] = g2_6.x;
        zs[13] = g2_6.y;
        float2 z2_7 = make_float2(zs[14], zs[15]);
        float2 _f32x2_mul_b_52 = make_float2(-1.442695f, -1.442695f);
        float2 t2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_52));
        float _exp2_53 = approx_exp2(t2_7.x);
        float _exp2_54 = approx_exp2(t2_7.y);
        float2 d2_7 = fma_f32x2_rn_noftz(make_float2(_exp2_53, _exp2_54), scale2, scale2);
        float _rcp_55 = approx_rcp(d2_7.x);
        float _rcp_56 = approx_rcp(d2_7.y);
        float2 _f32x2_mul_b_57 = mul_f32x2_noftz(make_float2(_rcp_55, _rcp_56), scale2);
        float2 g2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_57));
        zs[14] = g2_7.x;
        zs[15] = g2_7.y;
        float2 rstd2 = make_float2(rstd, rstd);
        float2 _f32x2_mul_a_58 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[0], xs[1]), rstd2), make_float2(wv[0], wv[1]));
        float2 _f32x2_mul_b_59 = make_float2(zs[0], zs[1]);
        float2 y2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
        xs[0] = y2_0.x;
        xs[1] = y2_0.y;
        float2 _f32x2_mul_a_60 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[2], xs[3]), rstd2), make_float2(wv[2], wv[3]));
        float2 _f32x2_mul_b_61 = make_float2(zs[2], zs[3]);
        float2 y2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
        xs[2] = y2_1.x;
        xs[3] = y2_1.y;
        float2 _f32x2_mul_a_62 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[4], xs[5]), rstd2), make_float2(wv[4], wv[5]));
        float2 _f32x2_mul_b_63 = make_float2(zs[4], zs[5]);
        float2 y2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
        xs[4] = y2_2.x;
        xs[5] = y2_2.y;
        float2 _f32x2_mul_a_64 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[6], xs[7]), rstd2), make_float2(wv[6], wv[7]));
        float2 _f32x2_mul_b_65 = make_float2(zs[6], zs[7]);
        float2 y2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
        xs[6] = y2_3.x;
        xs[7] = y2_3.y;
        float2 _f32x2_mul_a_66 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[8], xs[9]), rstd2), make_float2(wv[8], wv[9]));
        float2 _f32x2_mul_b_67 = make_float2(zs[8], zs[9]);
        float2 y2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
        xs[8] = y2_4.x;
        xs[9] = y2_4.y;
        float2 _f32x2_mul_a_68 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[10], xs[11]), rstd2), make_float2(wv[10], wv[11]));
        float2 _f32x2_mul_b_69 = make_float2(zs[10], zs[11]);
        float2 y2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
        xs[10] = y2_5.x;
        xs[11] = y2_5.y;
        float2 _f32x2_mul_a_70 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[12], xs[13]), rstd2), make_float2(wv[12], wv[13]));
        float2 _f32x2_mul_b_71 = make_float2(zs[12], zs[13]);
        float2 y2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
        xs[12] = y2_6.x;
        xs[13] = y2_6.y;
        float2 _f32x2_mul_a_72 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[14], xs[15]), rstd2), make_float2(wv[14], wv[15]));
        float2 _f32x2_mul_b_73 = make_float2(zs[14], zs[15]);
        float2 y2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
        xs[14] = y2_7.x;
        xs[15] = y2_7.y;
        unsigned int packed_0[8];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
            packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
            packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
            packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
            packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 8], xs[_lp*2+1 + 8]));
            packed_0[_lp + 4] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 10], xs[_lp*2+1 + 10]));
            packed_0[_lp + 5] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 12], xs[_lp*2+1 + 12]));
            packed_0[_lp + 6] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 14], xs[_lp*2+1 + 14]));
            packed_0[_lp + 7] = *(uint32_t*)&_bf2;
        }
        if (blk * 32 + group < rows) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 32 + group) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_0)[0];
            reinterpret_cast<int4*>(y + ((long long)(blk * 32 + group) * 128ll + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed_0 + (4))[0];
        }
    }

    // === Task calls (declaration order) ===
}

} // extern "C"
