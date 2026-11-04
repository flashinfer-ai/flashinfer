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


extern "C" __device__ float __nv_expf(float);


__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
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


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_train_87f0173e9583ad4fdd2a(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, CakeFastDivmod heads, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps)
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
    if (x_head == 128) {
        if (z_head == 128) {
            if (x_tok == heads64 * 128) {
                if (z_tok == heads64 * 128) {
                    packed = 1;
                }
            }
        }
    }
    if (packed != 0) {
        if (tid == 0) {
            int pblk_0 = bid + num_bids;
            if (pblk_0 < nblocks) {
                int pfirst_0 = pblk_0 * 16;
                int _min_3 = ((rows - pfirst_0) < (16) ? (rows - pfirst_0) : (16));
                int pbytes_0 = _min_3 * 256;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)pfirst_0 * 128), "r"((uint32_t)(pbytes_0)) : "memory");
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)pfirst_0 * 128), "r"((uint32_t)(pbytes_0)) : "memory");
            }
            uint32_t _read_special_reg_4;
            asm("mov.u32 %0, %%nsmid;" : "=r"(_read_special_reg_4));
            int wblk = bid + 6 * (int)_read_special_reg_4;
            if (wblk < num_bids) {
                if (wblk < nblocks) {
                    int wfirst = wblk * 16;
                    int _min_5 = ((rows - wfirst) < (16) ? (rows - wfirst) : (16));
                    int wbytes = _min_5 * 256;
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)wfirst * 128), "r"((uint32_t)(wbytes)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)wfirst * 128), "r"((uint32_t)(wbytes)) : "memory");
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
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff_0) + (0))) : "memory");
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
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff_0) + (0))) : "memory");
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
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(x + (long long)nfirst * 128), "r"((uint32_t)(nbytes)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(z + (long long)nfirst * 128), "r"((uint32_t)(nbytes)) : "memory");
                }
            }
        }
        float total = 0.0f;
        float chain_0 = 0.0f;
        chain_0 = __fmaf_rn(xs[0], xs[0], chain_0);
        chain_0 = __fmaf_rn(xs[1], xs[1], chain_0);
        chain_0 = __fmaf_rn(xs[2], xs[2], chain_0);
        chain_0 = __fmaf_rn(xs[3], xs[3], chain_0);
        chain_0 = __fmaf_rn(xs[4], xs[4], chain_0);
        chain_0 = __fmaf_rn(xs[5], xs[5], chain_0);
        chain_0 = __fmaf_rn(xs[6], xs[6], chain_0);
        chain_0 = __fmaf_rn(xs[7], xs[7], chain_0);
        chain_0 = __fmaf_rn(xs[8], xs[8], chain_0);
        chain_0 = __fmaf_rn(xs[9], xs[9], chain_0);
        chain_0 = __fmaf_rn(xs[10], xs[10], chain_0);
        chain_0 = __fmaf_rn(xs[11], xs[11], chain_0);
        chain_0 = __fmaf_rn(xs[12], xs[12], chain_0);
        chain_0 = __fmaf_rn(xs[13], xs[13], chain_0);
        chain_0 = __fmaf_rn(xs[14], xs[14], chain_0);
        chain_0 = __fmaf_rn(xs[15], xs[15], chain_0);
        total = chain_0;
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
        total = total + _shfl_xor_sync_9;
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        total = total + _shfl_xor_sync_10;
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
        total = total + _shfl_xor_sync_11;
        float _sqrt_12;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_12) : "f"(total * 0.0078125f + eps));
        float rstd = __frcp_rn(_sqrt_12);
        int fast = 0;
        float _neg_13;
        asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(zs[0]));
        float _neg_14;
        asm("neg.f32 %0, %1;" : "=f"(_neg_14) : "f"(zs[1]));
        float _neg_15;
        asm("neg.f32 %0, %1;" : "=f"(_neg_15) : "f"(zs[2]));
        float mz;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz) : "f"(_neg_13), "f"(_neg_14), "f"(_neg_15));
        float _abs_17;
        asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(zs[0]));
        float _neg_16;
        asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(_abs_17));
        float _abs_19;
        asm("abs.f32 %0, %1;" : "=f"(_abs_19) : "f"(zs[1]));
        float _neg_18;
        asm("neg.f32 %0, %1;" : "=f"(_neg_18) : "f"(_abs_19));
        float _abs_21;
        asm("abs.f32 %0, %1;" : "=f"(_abs_21) : "f"(zs[2]));
        float _neg_20;
        asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(_abs_21));
        float ma;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma) : "f"(_neg_16), "f"(_neg_18), "f"(_neg_20));
        float _neg_23;
        asm("neg.f32 %0, %1;" : "=f"(_neg_23) : "f"(zs[3]));
        float _neg_24;
        asm("neg.f32 %0, %1;" : "=f"(_neg_24) : "f"(zs[4]));
        float _max3_22;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_22) : "f"(mz), "f"(_neg_23), "f"(_neg_24));
        mz = _max3_22;
        float _abs_27;
        asm("abs.f32 %0, %1;" : "=f"(_abs_27) : "f"(zs[3]));
        float _neg_26;
        asm("neg.f32 %0, %1;" : "=f"(_neg_26) : "f"(_abs_27));
        float _abs_29;
        asm("abs.f32 %0, %1;" : "=f"(_abs_29) : "f"(zs[4]));
        float _neg_28;
        asm("neg.f32 %0, %1;" : "=f"(_neg_28) : "f"(_abs_29));
        float _max3_25;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_25) : "f"(ma), "f"(_neg_26), "f"(_neg_28));
        ma = _max3_25;
        float _neg_31;
        asm("neg.f32 %0, %1;" : "=f"(_neg_31) : "f"(zs[5]));
        float _neg_32;
        asm("neg.f32 %0, %1;" : "=f"(_neg_32) : "f"(zs[6]));
        float _max3_30;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_30) : "f"(mz), "f"(_neg_31), "f"(_neg_32));
        mz = _max3_30;
        float _abs_35;
        asm("abs.f32 %0, %1;" : "=f"(_abs_35) : "f"(zs[5]));
        float _neg_34;
        asm("neg.f32 %0, %1;" : "=f"(_neg_34) : "f"(_abs_35));
        float _abs_37;
        asm("abs.f32 %0, %1;" : "=f"(_abs_37) : "f"(zs[6]));
        float _neg_36;
        asm("neg.f32 %0, %1;" : "=f"(_neg_36) : "f"(_abs_37));
        float _max3_33;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_33) : "f"(ma), "f"(_neg_34), "f"(_neg_36));
        ma = _max3_33;
        float _neg_39;
        asm("neg.f32 %0, %1;" : "=f"(_neg_39) : "f"(zs[7]));
        float _neg_40;
        asm("neg.f32 %0, %1;" : "=f"(_neg_40) : "f"(zs[8]));
        float _max3_38;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_38) : "f"(mz), "f"(_neg_39), "f"(_neg_40));
        mz = _max3_38;
        float _abs_43;
        asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(zs[7]));
        float _neg_42;
        asm("neg.f32 %0, %1;" : "=f"(_neg_42) : "f"(_abs_43));
        float _abs_45;
        asm("abs.f32 %0, %1;" : "=f"(_abs_45) : "f"(zs[8]));
        float _neg_44;
        asm("neg.f32 %0, %1;" : "=f"(_neg_44) : "f"(_abs_45));
        float _max3_41;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_41) : "f"(ma), "f"(_neg_42), "f"(_neg_44));
        ma = _max3_41;
        float _neg_47;
        asm("neg.f32 %0, %1;" : "=f"(_neg_47) : "f"(zs[9]));
        float _neg_48;
        asm("neg.f32 %0, %1;" : "=f"(_neg_48) : "f"(zs[10]));
        float _max3_46;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_46) : "f"(mz), "f"(_neg_47), "f"(_neg_48));
        mz = _max3_46;
        float _abs_51;
        asm("abs.f32 %0, %1;" : "=f"(_abs_51) : "f"(zs[9]));
        float _neg_50;
        asm("neg.f32 %0, %1;" : "=f"(_neg_50) : "f"(_abs_51));
        float _abs_53;
        asm("abs.f32 %0, %1;" : "=f"(_abs_53) : "f"(zs[10]));
        float _neg_52;
        asm("neg.f32 %0, %1;" : "=f"(_neg_52) : "f"(_abs_53));
        float _max3_49;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_49) : "f"(ma), "f"(_neg_50), "f"(_neg_52));
        ma = _max3_49;
        float _neg_55;
        asm("neg.f32 %0, %1;" : "=f"(_neg_55) : "f"(zs[11]));
        float _neg_56;
        asm("neg.f32 %0, %1;" : "=f"(_neg_56) : "f"(zs[12]));
        float _max3_54;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_54) : "f"(mz), "f"(_neg_55), "f"(_neg_56));
        mz = _max3_54;
        float _abs_59;
        asm("abs.f32 %0, %1;" : "=f"(_abs_59) : "f"(zs[11]));
        float _neg_58;
        asm("neg.f32 %0, %1;" : "=f"(_neg_58) : "f"(_abs_59));
        float _abs_61;
        asm("abs.f32 %0, %1;" : "=f"(_abs_61) : "f"(zs[12]));
        float _neg_60;
        asm("neg.f32 %0, %1;" : "=f"(_neg_60) : "f"(_abs_61));
        float _max3_57;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_57) : "f"(ma), "f"(_neg_58), "f"(_neg_60));
        ma = _max3_57;
        float _neg_63;
        asm("neg.f32 %0, %1;" : "=f"(_neg_63) : "f"(zs[13]));
        float _neg_64;
        asm("neg.f32 %0, %1;" : "=f"(_neg_64) : "f"(zs[14]));
        float _max3_62;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_62) : "f"(mz), "f"(_neg_63), "f"(_neg_64));
        mz = _max3_62;
        float _abs_67;
        asm("abs.f32 %0, %1;" : "=f"(_abs_67) : "f"(zs[13]));
        float _neg_66;
        asm("neg.f32 %0, %1;" : "=f"(_neg_66) : "f"(_abs_67));
        float _abs_69;
        asm("abs.f32 %0, %1;" : "=f"(_abs_69) : "f"(zs[14]));
        float _neg_68;
        asm("neg.f32 %0, %1;" : "=f"(_neg_68) : "f"(_abs_69));
        float _max3_65;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_65) : "f"(ma), "f"(_neg_66), "f"(_neg_68));
        ma = _max3_65;
        float _neg_71;
        asm("neg.f32 %0, %1;" : "=f"(_neg_71) : "f"(zs[15]));
        float _max_70;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_70) : "f"(mz), "f"(_neg_71));
        mz = _max_70;
        float _abs_74;
        asm("abs.f32 %0, %1;" : "=f"(_abs_74) : "f"(zs[15]));
        float _neg_73;
        asm("neg.f32 %0, %1;" : "=f"(_neg_73) : "f"(_abs_74));
        float _max_72;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_72) : "f"(ma), "f"(_neg_73));
        ma = _max_72;
        float _abs_75;
        asm("abs.f32 %0, %1;" : "=f"(_abs_75) : "f"(zs[0]));
        float _abs_76;
        asm("abs.f32 %0, %1;" : "=f"(_abs_76) : "f"(zs[1]));
        float _abs_77;
        asm("abs.f32 %0, %1;" : "=f"(_abs_77) : "f"(zs[2]));
        float mp;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp) : "f"(_abs_75), "f"(_abs_76), "f"(_abs_77));
        float _abs_79;
        asm("abs.f32 %0, %1;" : "=f"(_abs_79) : "f"(zs[3]));
        float _abs_80;
        asm("abs.f32 %0, %1;" : "=f"(_abs_80) : "f"(zs[4]));
        float _max3_78;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_78) : "f"(mp), "f"(_abs_79), "f"(_abs_80));
        mp = _max3_78;
        float _abs_82;
        asm("abs.f32 %0, %1;" : "=f"(_abs_82) : "f"(zs[5]));
        float _abs_83;
        asm("abs.f32 %0, %1;" : "=f"(_abs_83) : "f"(zs[6]));
        float _max3_81;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_81) : "f"(mp), "f"(_abs_82), "f"(_abs_83));
        mp = _max3_81;
        float _abs_85;
        asm("abs.f32 %0, %1;" : "=f"(_abs_85) : "f"(zs[7]));
        float _abs_86;
        asm("abs.f32 %0, %1;" : "=f"(_abs_86) : "f"(zs[8]));
        float _max3_84;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_84) : "f"(mp), "f"(_abs_85), "f"(_abs_86));
        mp = _max3_84;
        float _abs_88;
        asm("abs.f32 %0, %1;" : "=f"(_abs_88) : "f"(zs[9]));
        float _abs_89;
        asm("abs.f32 %0, %1;" : "=f"(_abs_89) : "f"(zs[10]));
        float _max3_87;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_87) : "f"(mp), "f"(_abs_88), "f"(_abs_89));
        mp = _max3_87;
        float _abs_91;
        asm("abs.f32 %0, %1;" : "=f"(_abs_91) : "f"(zs[11]));
        float _abs_92;
        asm("abs.f32 %0, %1;" : "=f"(_abs_92) : "f"(zs[12]));
        float _max3_90;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_90) : "f"(mp), "f"(_abs_91), "f"(_abs_92));
        mp = _max3_90;
        float _abs_94;
        asm("abs.f32 %0, %1;" : "=f"(_abs_94) : "f"(zs[13]));
        float _abs_95;
        asm("abs.f32 %0, %1;" : "=f"(_abs_95) : "f"(zs[14]));
        float _max3_93;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_93) : "f"(mp), "f"(_abs_94), "f"(_abs_95));
        mp = _max3_93;
        float _abs_97;
        asm("abs.f32 %0, %1;" : "=f"(_abs_97) : "f"(zs[15]));
        float _max_96;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_96) : "f"(mp), "f"(_abs_97));
        mp = _max_96;
        if (mz <= 41.0f) {
            if (ma <= -1.1754944e-38f) {
                if (mp <= 3.4028235e+38f) {
                    fast = 1;
                }
            }
        }
        if (fast != 0) {
            float2 rstd2 = make_float2(rstd, rstd);
            float2 z2_0 = make_float2(zs[0], zs[1]);
            float ta_0;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_0) : "f"(zs[0]), "f"(-0.00572498f), "f"(0.5f));
            float tb_0;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_0) : "f"(zs[1]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_98;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(ta_0), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_99;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(tb_0), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_0 = make_float2(_fma_98, _fma_99);
            float2 r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_0));
            r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.925963e-08f, -1.925963e-08f), r2_0);
            float2 nscale_0 = make_float2(__uint_as_float(__as_u32(t2_0.x) << 23), __uint_as_float(__as_u32(t2_0.y) << 23));
            float _exp2_100 = approx_exp2(r2_0.x);
            float _exp2_101 = approx_exp2(r2_0.y);
            float2 nden_0 = fma_f32x2_rn_noftz(nscale_0, make_float2(_exp2_100, _exp2_101), make_float2(-1.0f, -1.0f));
            float _neg_103;
            asm("neg.f32 %0, %1;" : "=f"(_neg_103) : "f"(nden_0.x));
            float _rcp_102 = approx_rcp(_neg_103);
            float _neg_105;
            asm("neg.f32 %0, %1;" : "=f"(_neg_105) : "f"(nden_0.y));
            float _rcp_104 = approx_rcp(_neg_105);
            float2 r0_0 = make_float2(_rcp_102, _rcp_104);
            float2 r1_0 = fma_f32x2_rn_noftz(r0_0, fma_f32x2_rn_noftz(nden_0, r0_0, make_float2(1.0f, 1.0f)), r0_0);
            float2 q0_0 = fma_f32x2_rn_noftz(z2_0, r1_0, make_float2(0.0f, 0.0f));
            float2 gate2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, q0_0, z2_0), q0_0);
            float2 _f32x2_mul_a_106 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[0], xs[1]), rstd2), make_float2(wv[0], wv[1]));
            float2 y2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_106), "l"(*(const unsigned long long*)&gate2_0));
            xs[0] = y2_0.x;
            xs[1] = y2_0.y;
            float2 z2_1 = make_float2(zs[2], zs[3]);
            float ta_1;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_1) : "f"(zs[2]), "f"(-0.00572498f), "f"(0.5f));
            float tb_1;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_1) : "f"(zs[3]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_107;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(ta_1), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_108;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(tb_1), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_1 = make_float2(_fma_107, _fma_108);
            float2 r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_1));
            r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.925963e-08f, -1.925963e-08f), r2_1);
            float2 nscale_1 = make_float2(__uint_as_float(__as_u32(t2_1.x) << 23), __uint_as_float(__as_u32(t2_1.y) << 23));
            float _exp2_109 = approx_exp2(r2_1.x);
            float _exp2_110 = approx_exp2(r2_1.y);
            float2 nden_1 = fma_f32x2_rn_noftz(nscale_1, make_float2(_exp2_109, _exp2_110), make_float2(-1.0f, -1.0f));
            float _neg_112;
            asm("neg.f32 %0, %1;" : "=f"(_neg_112) : "f"(nden_1.x));
            float _rcp_111 = approx_rcp(_neg_112);
            float _neg_114;
            asm("neg.f32 %0, %1;" : "=f"(_neg_114) : "f"(nden_1.y));
            float _rcp_113 = approx_rcp(_neg_114);
            float2 r0_1 = make_float2(_rcp_111, _rcp_113);
            float2 r1_1 = fma_f32x2_rn_noftz(r0_1, fma_f32x2_rn_noftz(nden_1, r0_1, make_float2(1.0f, 1.0f)), r0_1);
            float2 q0_1 = fma_f32x2_rn_noftz(z2_1, r1_1, make_float2(0.0f, 0.0f));
            float2 gate2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, q0_1, z2_1), q0_1);
            float2 _f32x2_mul_a_115 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[2], xs[3]), rstd2), make_float2(wv[2], wv[3]));
            float2 y2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&gate2_1));
            xs[2] = y2_1.x;
            xs[3] = y2_1.y;
            float2 z2_2 = make_float2(zs[4], zs[5]);
            float ta_2;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_2) : "f"(zs[4]), "f"(-0.00572498f), "f"(0.5f));
            float tb_2;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_2) : "f"(zs[5]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_116;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(ta_2), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_117;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(tb_2), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_2 = make_float2(_fma_116, _fma_117);
            float2 r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_2));
            r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.925963e-08f, -1.925963e-08f), r2_2);
            float2 nscale_2 = make_float2(__uint_as_float(__as_u32(t2_2.x) << 23), __uint_as_float(__as_u32(t2_2.y) << 23));
            float _exp2_118 = approx_exp2(r2_2.x);
            float _exp2_119 = approx_exp2(r2_2.y);
            float2 nden_2 = fma_f32x2_rn_noftz(nscale_2, make_float2(_exp2_118, _exp2_119), make_float2(-1.0f, -1.0f));
            float _neg_121;
            asm("neg.f32 %0, %1;" : "=f"(_neg_121) : "f"(nden_2.x));
            float _rcp_120 = approx_rcp(_neg_121);
            float _neg_123;
            asm("neg.f32 %0, %1;" : "=f"(_neg_123) : "f"(nden_2.y));
            float _rcp_122 = approx_rcp(_neg_123);
            float2 r0_2 = make_float2(_rcp_120, _rcp_122);
            float2 r1_2 = fma_f32x2_rn_noftz(r0_2, fma_f32x2_rn_noftz(nden_2, r0_2, make_float2(1.0f, 1.0f)), r0_2);
            float2 q0_2 = fma_f32x2_rn_noftz(z2_2, r1_2, make_float2(0.0f, 0.0f));
            float2 gate2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, q0_2, z2_2), q0_2);
            float2 _f32x2_mul_a_124 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[4], xs[5]), rstd2), make_float2(wv[4], wv[5]));
            float2 y2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&gate2_2));
            xs[4] = y2_2.x;
            xs[5] = y2_2.y;
            float2 z2_3 = make_float2(zs[6], zs[7]);
            float ta_3;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_3) : "f"(zs[6]), "f"(-0.00572498f), "f"(0.5f));
            float tb_3;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_3) : "f"(zs[7]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_125;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(ta_3), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_126;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(tb_3), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_3 = make_float2(_fma_125, _fma_126);
            float2 r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_3));
            r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.925963e-08f, -1.925963e-08f), r2_3);
            float2 nscale_3 = make_float2(__uint_as_float(__as_u32(t2_3.x) << 23), __uint_as_float(__as_u32(t2_3.y) << 23));
            float _exp2_127 = approx_exp2(r2_3.x);
            float _exp2_128 = approx_exp2(r2_3.y);
            float2 nden_3 = fma_f32x2_rn_noftz(nscale_3, make_float2(_exp2_127, _exp2_128), make_float2(-1.0f, -1.0f));
            float _neg_130;
            asm("neg.f32 %0, %1;" : "=f"(_neg_130) : "f"(nden_3.x));
            float _rcp_129 = approx_rcp(_neg_130);
            float _neg_132;
            asm("neg.f32 %0, %1;" : "=f"(_neg_132) : "f"(nden_3.y));
            float _rcp_131 = approx_rcp(_neg_132);
            float2 r0_3 = make_float2(_rcp_129, _rcp_131);
            float2 r1_3 = fma_f32x2_rn_noftz(r0_3, fma_f32x2_rn_noftz(nden_3, r0_3, make_float2(1.0f, 1.0f)), r0_3);
            float2 q0_3 = fma_f32x2_rn_noftz(z2_3, r1_3, make_float2(0.0f, 0.0f));
            float2 gate2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, q0_3, z2_3), q0_3);
            float2 _f32x2_mul_a_133 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[6], xs[7]), rstd2), make_float2(wv[6], wv[7]));
            float2 y2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&gate2_3));
            xs[6] = y2_3.x;
            xs[7] = y2_3.y;
            float2 z2_4 = make_float2(zs[8], zs[9]);
            float ta_4;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_4) : "f"(zs[8]), "f"(-0.00572498f), "f"(0.5f));
            float tb_4;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_4) : "f"(zs[9]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_134;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(ta_4), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_135;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(tb_4), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_4 = make_float2(_fma_134, _fma_135);
            float2 r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_4));
            r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.925963e-08f, -1.925963e-08f), r2_4);
            float2 nscale_4 = make_float2(__uint_as_float(__as_u32(t2_4.x) << 23), __uint_as_float(__as_u32(t2_4.y) << 23));
            float _exp2_136 = approx_exp2(r2_4.x);
            float _exp2_137 = approx_exp2(r2_4.y);
            float2 nden_4 = fma_f32x2_rn_noftz(nscale_4, make_float2(_exp2_136, _exp2_137), make_float2(-1.0f, -1.0f));
            float _neg_139;
            asm("neg.f32 %0, %1;" : "=f"(_neg_139) : "f"(nden_4.x));
            float _rcp_138 = approx_rcp(_neg_139);
            float _neg_141;
            asm("neg.f32 %0, %1;" : "=f"(_neg_141) : "f"(nden_4.y));
            float _rcp_140 = approx_rcp(_neg_141);
            float2 r0_4 = make_float2(_rcp_138, _rcp_140);
            float2 r1_4 = fma_f32x2_rn_noftz(r0_4, fma_f32x2_rn_noftz(nden_4, r0_4, make_float2(1.0f, 1.0f)), r0_4);
            float2 q0_4 = fma_f32x2_rn_noftz(z2_4, r1_4, make_float2(0.0f, 0.0f));
            float2 gate2_4 = fma_f32x2_rn_noftz(r1_4, fma_f32x2_rn_noftz(nden_4, q0_4, z2_4), q0_4);
            float2 _f32x2_mul_a_142 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[8], xs[9]), rstd2), make_float2(wv[8], wv[9]));
            float2 y2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_142), "l"(*(const unsigned long long*)&gate2_4));
            xs[8] = y2_4.x;
            xs[9] = y2_4.y;
            float2 z2_5 = make_float2(zs[10], zs[11]);
            float ta_5;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_5) : "f"(zs[10]), "f"(-0.00572498f), "f"(0.5f));
            float tb_5;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_5) : "f"(zs[11]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_143;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(ta_5), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_144;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(tb_5), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_5 = make_float2(_fma_143, _fma_144);
            float2 r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_5));
            r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.925963e-08f, -1.925963e-08f), r2_5);
            float2 nscale_5 = make_float2(__uint_as_float(__as_u32(t2_5.x) << 23), __uint_as_float(__as_u32(t2_5.y) << 23));
            float _exp2_145 = approx_exp2(r2_5.x);
            float _exp2_146 = approx_exp2(r2_5.y);
            float2 nden_5 = fma_f32x2_rn_noftz(nscale_5, make_float2(_exp2_145, _exp2_146), make_float2(-1.0f, -1.0f));
            float _neg_148;
            asm("neg.f32 %0, %1;" : "=f"(_neg_148) : "f"(nden_5.x));
            float _rcp_147 = approx_rcp(_neg_148);
            float _neg_150;
            asm("neg.f32 %0, %1;" : "=f"(_neg_150) : "f"(nden_5.y));
            float _rcp_149 = approx_rcp(_neg_150);
            float2 r0_5 = make_float2(_rcp_147, _rcp_149);
            float2 r1_5 = fma_f32x2_rn_noftz(r0_5, fma_f32x2_rn_noftz(nden_5, r0_5, make_float2(1.0f, 1.0f)), r0_5);
            float2 q0_5 = fma_f32x2_rn_noftz(z2_5, r1_5, make_float2(0.0f, 0.0f));
            float2 gate2_5 = fma_f32x2_rn_noftz(r1_5, fma_f32x2_rn_noftz(nden_5, q0_5, z2_5), q0_5);
            float2 _f32x2_mul_a_151 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[10], xs[11]), rstd2), make_float2(wv[10], wv[11]));
            float2 y2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_151), "l"(*(const unsigned long long*)&gate2_5));
            xs[10] = y2_5.x;
            xs[11] = y2_5.y;
            float2 z2_6 = make_float2(zs[12], zs[13]);
            float ta_6;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_6) : "f"(zs[12]), "f"(-0.00572498f), "f"(0.5f));
            float tb_6;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_6) : "f"(zs[13]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_152;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(ta_6), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_153;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(tb_6), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_6 = make_float2(_fma_152, _fma_153);
            float2 r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_6));
            r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.925963e-08f, -1.925963e-08f), r2_6);
            float2 nscale_6 = make_float2(__uint_as_float(__as_u32(t2_6.x) << 23), __uint_as_float(__as_u32(t2_6.y) << 23));
            float _exp2_154 = approx_exp2(r2_6.x);
            float _exp2_155 = approx_exp2(r2_6.y);
            float2 nden_6 = fma_f32x2_rn_noftz(nscale_6, make_float2(_exp2_154, _exp2_155), make_float2(-1.0f, -1.0f));
            float _neg_157;
            asm("neg.f32 %0, %1;" : "=f"(_neg_157) : "f"(nden_6.x));
            float _rcp_156 = approx_rcp(_neg_157);
            float _neg_159;
            asm("neg.f32 %0, %1;" : "=f"(_neg_159) : "f"(nden_6.y));
            float _rcp_158 = approx_rcp(_neg_159);
            float2 r0_6 = make_float2(_rcp_156, _rcp_158);
            float2 r1_6 = fma_f32x2_rn_noftz(r0_6, fma_f32x2_rn_noftz(nden_6, r0_6, make_float2(1.0f, 1.0f)), r0_6);
            float2 q0_6 = fma_f32x2_rn_noftz(z2_6, r1_6, make_float2(0.0f, 0.0f));
            float2 gate2_6 = fma_f32x2_rn_noftz(r1_6, fma_f32x2_rn_noftz(nden_6, q0_6, z2_6), q0_6);
            float2 _f32x2_mul_a_160 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[12], xs[13]), rstd2), make_float2(wv[12], wv[13]));
            float2 y2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_160), "l"(*(const unsigned long long*)&gate2_6));
            xs[12] = y2_6.x;
            xs[13] = y2_6.y;
            float2 z2_7 = make_float2(zs[14], zs[15]);
            float ta_7;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_7) : "f"(zs[14]), "f"(-0.00572498f), "f"(0.5f));
            float tb_7;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_7) : "f"(zs[15]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_161;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(ta_7), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_162;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(tb_7), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_7 = make_float2(_fma_161, _fma_162);
            float2 r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_7));
            r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.925963e-08f, -1.925963e-08f), r2_7);
            float2 nscale_7 = make_float2(__uint_as_float(__as_u32(t2_7.x) << 23), __uint_as_float(__as_u32(t2_7.y) << 23));
            float _exp2_163 = approx_exp2(r2_7.x);
            float _exp2_164 = approx_exp2(r2_7.y);
            float2 nden_7 = fma_f32x2_rn_noftz(nscale_7, make_float2(_exp2_163, _exp2_164), make_float2(-1.0f, -1.0f));
            float _neg_166;
            asm("neg.f32 %0, %1;" : "=f"(_neg_166) : "f"(nden_7.x));
            float _rcp_165 = approx_rcp(_neg_166);
            float _neg_168;
            asm("neg.f32 %0, %1;" : "=f"(_neg_168) : "f"(nden_7.y));
            float _rcp_167 = approx_rcp(_neg_168);
            float2 r0_7 = make_float2(_rcp_165, _rcp_167);
            float2 r1_7 = fma_f32x2_rn_noftz(r0_7, fma_f32x2_rn_noftz(nden_7, r0_7, make_float2(1.0f, 1.0f)), r0_7);
            float2 q0_7 = fma_f32x2_rn_noftz(z2_7, r1_7, make_float2(0.0f, 0.0f));
            float2 gate2_7 = fma_f32x2_rn_noftz(r1_7, fma_f32x2_rn_noftz(nden_7, q0_7, z2_7), q0_7);
            float2 _f32x2_mul_a_169 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[14], xs[15]), rstd2), make_float2(wv[14], wv[15]));
            float2 y2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&gate2_7));
            xs[14] = y2_7.x;
            xs[15] = y2_7.y;
        } else {
            float _exp_170 = __nv_expf(zs[0] * -1.0f);
            float gate_0 = __fdiv_rn(zs[0], 1.0f + _exp_170);
            xs[0] = xs[0] * rstd * wv[0] * gate_0;
            float _exp_171 = __nv_expf(zs[1] * -1.0f);
            float gate_1 = __fdiv_rn(zs[1], 1.0f + _exp_171);
            xs[1] = xs[1] * rstd * wv[1] * gate_1;
            float _exp_172 = __nv_expf(zs[2] * -1.0f);
            float gate_2 = __fdiv_rn(zs[2], 1.0f + _exp_172);
            xs[2] = xs[2] * rstd * wv[2] * gate_2;
            float _exp_173 = __nv_expf(zs[3] * -1.0f);
            float gate_3 = __fdiv_rn(zs[3], 1.0f + _exp_173);
            xs[3] = xs[3] * rstd * wv[3] * gate_3;
            float _exp_174 = __nv_expf(zs[4] * -1.0f);
            float gate_4 = __fdiv_rn(zs[4], 1.0f + _exp_174);
            xs[4] = xs[4] * rstd * wv[4] * gate_4;
            float _exp_175 = __nv_expf(zs[5] * -1.0f);
            float gate_5 = __fdiv_rn(zs[5], 1.0f + _exp_175);
            xs[5] = xs[5] * rstd * wv[5] * gate_5;
            float _exp_176 = __nv_expf(zs[6] * -1.0f);
            float gate_6 = __fdiv_rn(zs[6], 1.0f + _exp_176);
            xs[6] = xs[6] * rstd * wv[6] * gate_6;
            float _exp_177 = __nv_expf(zs[7] * -1.0f);
            float gate_7 = __fdiv_rn(zs[7], 1.0f + _exp_177);
            xs[7] = xs[7] * rstd * wv[7] * gate_7;
            float _exp_178 = __nv_expf(zs[8] * -1.0f);
            float gate_8 = __fdiv_rn(zs[8], 1.0f + _exp_178);
            xs[8] = xs[8] * rstd * wv[8] * gate_8;
            float _exp_179 = __nv_expf(zs[9] * -1.0f);
            float gate_9 = __fdiv_rn(zs[9], 1.0f + _exp_179);
            xs[9] = xs[9] * rstd * wv[9] * gate_9;
            float _exp_180 = __nv_expf(zs[10] * -1.0f);
            float gate_10 = __fdiv_rn(zs[10], 1.0f + _exp_180);
            xs[10] = xs[10] * rstd * wv[10] * gate_10;
            float _exp_181 = __nv_expf(zs[11] * -1.0f);
            float gate_11 = __fdiv_rn(zs[11], 1.0f + _exp_181);
            xs[11] = xs[11] * rstd * wv[11] * gate_11;
            float _exp_182 = __nv_expf(zs[12] * -1.0f);
            float gate_12 = __fdiv_rn(zs[12], 1.0f + _exp_182);
            xs[12] = xs[12] * rstd * wv[12] * gate_12;
            float _exp_183 = __nv_expf(zs[13] * -1.0f);
            float gate_13 = __fdiv_rn(zs[13], 1.0f + _exp_183);
            xs[13] = xs[13] * rstd * wv[13] * gate_13;
            float _exp_184 = __nv_expf(zs[14] * -1.0f);
            float gate_14 = __fdiv_rn(zs[14], 1.0f + _exp_184);
            xs[14] = xs[14] * rstd * wv[14] * gate_14;
            float _exp_185 = __nv_expf(zs[15] * -1.0f);
            float gate_15 = __fdiv_rn(zs[15], 1.0f + _exp_185);
            xs[15] = xs[15] * rstd * wv[15] * gate_15;
        }
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
            reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128 + (long long)col))[0] = reinterpret_cast<int4*>(packed_0)[0];
            reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128 + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed_0 + (4))[0];
        }
    }

    // === Task calls (declaration order) ===
}

} // extern "C"
