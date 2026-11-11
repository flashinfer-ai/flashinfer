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

#define LAUNCH_MIN_BLOCKS 6

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
kernel_cake_rmsnorm_silu_gate_train_e6e0bf4c758851cabc99(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, CakeFastDivmod heads, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps)
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
    int nblocks = (rows + 15) / 16;
    unsigned int xw[8];
    unsigned int zw[8];
    int src = ((bid * 16 + group) < (rows - 1) ? (bid * 16 + group) : (rows - 1));
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
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff) + (0))), "l"(0x12F0000000000000ULL) : "memory");
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
        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
            : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff) + (0))), "l"(0x12F0000000000000ULL) : "memory");
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
            int pblk = bid + num_bids;
            if (pblk < nblocks) {
                int pfirst = pblk * 16;
                int _min_3 = ((rows - pfirst) < (16) ? (rows - pfirst) : (16));
                int pbytes = _min_3 * 256;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)pfirst * 128ll), "r"((uint32_t)(pbytes)) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)pfirst * 128ll), "r"((uint32_t)(pbytes)) : "memory");
            }
            uint32_t _read_special_reg_4;
            asm("mov.u32 %0, %%nsmid;" : "=r"(_read_special_reg_4));
            int wblk = bid + 6 * (int)_read_special_reg_4;
            if (wblk < num_bids) {
                if (wblk < nblocks) {
                    int wfirst = wblk * 16;
                    int _min_5 = ((rows - wfirst) < (16) ? (rows - wfirst) : (16));
                    int wbytes = _min_5 * 256;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)wfirst * 128ll), "r"((uint32_t)(wbytes)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)wfirst * 128ll), "r"((uint32_t)(wbytes)) : "memory");
                }
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
            int src_0 = (((blk + num_bids) * 16 + group) < (rows - 1) ? ((blk + num_bids) * 16 + group) : (rows - 1));
            uint32_t tq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_0);
            uint32_t th_0 = (uint32_t)((unsigned int)src_0) - tq_0 * (uint32_t)(heads.divisor);
            long long xoff_0 = (long long)tq_0 * x_tok + (long long)th_0 * x_head + (long long)col;
            long long zoff_0 = (long long)tq_0 * z_tok + (long long)th_0 * z_head + (long long)col;
            {
                uint32_t _iv_6_0;
                uint32_t _iv_6_1;
                uint32_t _iv_6_2;
                uint32_t _iv_6_3;
                uint32_t _iv_6_4;
                uint32_t _iv_6_5;
                uint32_t _iv_6_6;
                uint32_t _iv_6_7;
                asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                    : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff_0) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                xw[0 + 0] = (uint32_t)_iv_6_0;
                xw[0 + 1] = (uint32_t)_iv_6_1;
                xw[0 + 2] = (uint32_t)_iv_6_2;
                xw[0 + 3] = (uint32_t)_iv_6_3;
                xw[0 + 4] = (uint32_t)_iv_6_4;
                xw[0 + 5] = (uint32_t)_iv_6_5;
                xw[0 + 6] = (uint32_t)_iv_6_6;
                xw[0 + 7] = (uint32_t)_iv_6_7;
            }
            {
                uint32_t _iv_7_0;
                uint32_t _iv_7_1;
                uint32_t _iv_7_2;
                uint32_t _iv_7_3;
                uint32_t _iv_7_4;
                uint32_t _iv_7_5;
                uint32_t _iv_7_6;
                uint32_t _iv_7_7;
                asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                    : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff_0) + (0))), "l"(0x12F0000000000000ULL) : "memory");
                zw[0 + 0] = (uint32_t)_iv_7_0;
                zw[0 + 1] = (uint32_t)_iv_7_1;
                zw[0 + 2] = (uint32_t)_iv_7_2;
                zw[0 + 3] = (uint32_t)_iv_7_3;
                zw[0 + 4] = (uint32_t)_iv_7_4;
                zw[0 + 5] = (uint32_t)_iv_7_5;
                zw[0 + 6] = (uint32_t)_iv_7_6;
                zw[0 + 7] = (uint32_t)_iv_7_7;
            }
        }
        if (packed != 0) {
            if (tid == 0) {
                int nblk = blk + 2 * num_bids;
                if (nblk < nblocks) {
                    int nfirst = nblk * 16;
                    int _min_8 = ((rows - nfirst) < (16) ? (rows - nfirst) : (16));
                    int nbytes = _min_8 * 256;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128ll), "r"((uint32_t)(nbytes)) : "memory");
                }
            }
        }
        float gs[16];
        float2 scale2 = make_float2(2.3283064e-10f, 2.3283064e-10f);
        float2 z2_0 = make_float2(zs[0], zs[1]);
        float2 _f32x2_mul_b_9 = make_float2(-1.442695f, -1.442695f);
        float2 t2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
        float _exp2_10 = approx_exp2(t2_0.x);
        float _exp2_11 = approx_exp2(t2_0.y);
        float2 d2_0 = fma_f32x2_rn_noftz(make_float2(_exp2_10, _exp2_11), scale2, scale2);
        float _rcp_12 = approx_rcp(d2_0.x);
        float _rcp_13 = approx_rcp(d2_0.y);
        float2 _f32x2_mul_b_14 = mul_f32x2_noftz(make_float2(_rcp_12, _rcp_13), scale2);
        float2 g2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
        gs[0] = g2_0.x;
        gs[1] = g2_0.y;
        float2 z2_1 = make_float2(zs[2], zs[3]);
        float2 _f32x2_mul_b_15 = make_float2(-1.442695f, -1.442695f);
        float2 t2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_15));
        float _exp2_16 = approx_exp2(t2_1.x);
        float _exp2_17 = approx_exp2(t2_1.y);
        float2 d2_1 = fma_f32x2_rn_noftz(make_float2(_exp2_16, _exp2_17), scale2, scale2);
        float _rcp_18 = approx_rcp(d2_1.x);
        float _rcp_19 = approx_rcp(d2_1.y);
        float2 _f32x2_mul_b_20 = mul_f32x2_noftz(make_float2(_rcp_18, _rcp_19), scale2);
        float2 g2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        gs[2] = g2_1.x;
        gs[3] = g2_1.y;
        float2 z2_2 = make_float2(zs[4], zs[5]);
        float2 _f32x2_mul_b_21 = make_float2(-1.442695f, -1.442695f);
        float2 t2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
        float _exp2_22 = approx_exp2(t2_2.x);
        float _exp2_23 = approx_exp2(t2_2.y);
        float2 d2_2 = fma_f32x2_rn_noftz(make_float2(_exp2_22, _exp2_23), scale2, scale2);
        float _rcp_24 = approx_rcp(d2_2.x);
        float _rcp_25 = approx_rcp(d2_2.y);
        float2 _f32x2_mul_b_26 = mul_f32x2_noftz(make_float2(_rcp_24, _rcp_25), scale2);
        float2 g2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
        gs[4] = g2_2.x;
        gs[5] = g2_2.y;
        float2 z2_3 = make_float2(zs[6], zs[7]);
        float2 _f32x2_mul_b_27 = make_float2(-1.442695f, -1.442695f);
        float2 t2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
        float _exp2_28 = approx_exp2(t2_3.x);
        float _exp2_29 = approx_exp2(t2_3.y);
        float2 d2_3 = fma_f32x2_rn_noftz(make_float2(_exp2_28, _exp2_29), scale2, scale2);
        float _rcp_30 = approx_rcp(d2_3.x);
        float _rcp_31 = approx_rcp(d2_3.y);
        float2 _f32x2_mul_b_32 = mul_f32x2_noftz(make_float2(_rcp_30, _rcp_31), scale2);
        float2 g2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
        gs[6] = g2_3.x;
        gs[7] = g2_3.y;
        float2 z2_4 = make_float2(zs[8], zs[9]);
        float2 _f32x2_mul_b_33 = make_float2(-1.442695f, -1.442695f);
        float2 t2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
        float _exp2_34 = approx_exp2(t2_4.x);
        float _exp2_35 = approx_exp2(t2_4.y);
        float2 d2_4 = fma_f32x2_rn_noftz(make_float2(_exp2_34, _exp2_35), scale2, scale2);
        float _rcp_36 = approx_rcp(d2_4.x);
        float _rcp_37 = approx_rcp(d2_4.y);
        float2 _f32x2_mul_b_38 = mul_f32x2_noftz(make_float2(_rcp_36, _rcp_37), scale2);
        float2 g2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
        gs[8] = g2_4.x;
        gs[9] = g2_4.y;
        float2 z2_5 = make_float2(zs[10], zs[11]);
        float2 _f32x2_mul_b_39 = make_float2(-1.442695f, -1.442695f);
        float2 t2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
        float _exp2_40 = approx_exp2(t2_5.x);
        float _exp2_41 = approx_exp2(t2_5.y);
        float2 d2_5 = fma_f32x2_rn_noftz(make_float2(_exp2_40, _exp2_41), scale2, scale2);
        float _rcp_42 = approx_rcp(d2_5.x);
        float _rcp_43 = approx_rcp(d2_5.y);
        float2 _f32x2_mul_b_44 = mul_f32x2_noftz(make_float2(_rcp_42, _rcp_43), scale2);
        float2 g2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_44));
        gs[10] = g2_5.x;
        gs[11] = g2_5.y;
        float2 z2_6 = make_float2(zs[12], zs[13]);
        float2 _f32x2_mul_b_45 = make_float2(-1.442695f, -1.442695f);
        float2 t2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
        float _exp2_46 = approx_exp2(t2_6.x);
        float _exp2_47 = approx_exp2(t2_6.y);
        float2 d2_6 = fma_f32x2_rn_noftz(make_float2(_exp2_46, _exp2_47), scale2, scale2);
        float _rcp_48 = approx_rcp(d2_6.x);
        float _rcp_49 = approx_rcp(d2_6.y);
        float2 _f32x2_mul_b_50 = mul_f32x2_noftz(make_float2(_rcp_48, _rcp_49), scale2);
        float2 g2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_50));
        gs[12] = g2_6.x;
        gs[13] = g2_6.y;
        float2 z2_7 = make_float2(zs[14], zs[15]);
        float2 _f32x2_mul_b_51 = make_float2(-1.442695f, -1.442695f);
        float2 t2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
        float _exp2_52 = approx_exp2(t2_7.x);
        float _exp2_53 = approx_exp2(t2_7.y);
        float2 d2_7 = fma_f32x2_rn_noftz(make_float2(_exp2_52, _exp2_53), scale2, scale2);
        float _rcp_54 = approx_rcp(d2_7.x);
        float _rcp_55 = approx_rcp(d2_7.y);
        float2 _f32x2_mul_b_56 = mul_f32x2_noftz(make_float2(_rcp_54, _rcp_55), scale2);
        float2 g2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_56));
        gs[14] = g2_7.x;
        gs[15] = g2_7.y;
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
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
        total = total + _shfl_xor_sync_57;
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        total = total + _shfl_xor_sync_58;
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
        total = total + _shfl_xor_sync_59;
        float rstd;
        asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
        float2 rstd2 = make_float2(rstd, rstd);
        float2 _f32x2_mul_a_60 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[0], xs[1]), rstd2), make_float2(wv[0], wv[1]));
        float2 _f32x2_mul_b_61 = make_float2(gs[0], gs[1]);
        float2 y2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
        xs[0] = y2_0.x;
        xs[1] = y2_0.y;
        float2 _f32x2_mul_a_62 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[2], xs[3]), rstd2), make_float2(wv[2], wv[3]));
        float2 _f32x2_mul_b_63 = make_float2(gs[2], gs[3]);
        float2 y2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
        xs[2] = y2_1.x;
        xs[3] = y2_1.y;
        float2 _f32x2_mul_a_64 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[4], xs[5]), rstd2), make_float2(wv[4], wv[5]));
        float2 _f32x2_mul_b_65 = make_float2(gs[4], gs[5]);
        float2 y2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
        xs[4] = y2_2.x;
        xs[5] = y2_2.y;
        float2 _f32x2_mul_a_66 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[6], xs[7]), rstd2), make_float2(wv[6], wv[7]));
        float2 _f32x2_mul_b_67 = make_float2(gs[6], gs[7]);
        float2 y2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
        xs[6] = y2_3.x;
        xs[7] = y2_3.y;
        float2 _f32x2_mul_a_68 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[8], xs[9]), rstd2), make_float2(wv[8], wv[9]));
        float2 _f32x2_mul_b_69 = make_float2(gs[8], gs[9]);
        float2 y2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
        xs[8] = y2_4.x;
        xs[9] = y2_4.y;
        float2 _f32x2_mul_a_70 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[10], xs[11]), rstd2), make_float2(wv[10], wv[11]));
        float2 _f32x2_mul_b_71 = make_float2(gs[10], gs[11]);
        float2 y2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
        xs[10] = y2_5.x;
        xs[11] = y2_5.y;
        float2 _f32x2_mul_a_72 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[12], xs[13]), rstd2), make_float2(wv[12], wv[13]));
        float2 _f32x2_mul_b_73 = make_float2(gs[12], gs[13]);
        float2 y2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
        xs[12] = y2_6.x;
        xs[13] = y2_6.y;
        float2 _f32x2_mul_a_74 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[14], xs[15]), rstd2), make_float2(wv[14], wv[15]));
        float2 _f32x2_mul_b_75 = make_float2(gs[14], gs[15]);
        float2 y2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
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
        if (blk * 16 + group < rows) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed_0)[0];
            reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128ll + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed_0 + (4))[0];
        }
    }

    // === Task calls (declaration order) ===
}

} // extern "C"
