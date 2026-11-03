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

#define CAKE_INF CUDART_INF_F
#define THREADS 128
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
kernel_cake_rmsnorm_silu_gate_train_8bd481246a4b97382250(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, CakeFastDivmod heads, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps)
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
                uint32_t _iv_3_0;
                uint32_t _iv_3_1;
                uint32_t _iv_3_2;
                uint32_t _iv_3_3;
                uint32_t _iv_3_4;
                uint32_t _iv_3_5;
                uint32_t _iv_3_6;
                uint32_t _iv_3_7;
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(x + xoff_0) + (0))) : "memory");
                xw[0 + 0] = (uint32_t)_iv_3_0;
                xw[0 + 1] = (uint32_t)_iv_3_1;
                xw[0 + 2] = (uint32_t)_iv_3_2;
                xw[0 + 3] = (uint32_t)_iv_3_3;
                xw[0 + 4] = (uint32_t)_iv_3_4;
                xw[0 + 5] = (uint32_t)_iv_3_5;
                xw[0 + 6] = (uint32_t)_iv_3_6;
                xw[0 + 7] = (uint32_t)_iv_3_7;
            }
            {
                uint32_t _iv_4_0;
                uint32_t _iv_4_1;
                uint32_t _iv_4_2;
                uint32_t _iv_4_3;
                uint32_t _iv_4_4;
                uint32_t _iv_4_5;
                uint32_t _iv_4_6;
                uint32_t _iv_4_7;
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(reinterpret_cast<unsigned int*>(z + zoff_0) + (0))) : "memory");
                zw[0 + 0] = (uint32_t)_iv_4_0;
                zw[0 + 1] = (uint32_t)_iv_4_1;
                zw[0 + 2] = (uint32_t)_iv_4_2;
                zw[0 + 3] = (uint32_t)_iv_4_3;
                zw[0 + 4] = (uint32_t)_iv_4_4;
                zw[0 + 5] = (uint32_t)_iv_4_5;
                zw[0 + 6] = (uint32_t)_iv_4_6;
                zw[0 + 7] = (uint32_t)_iv_4_7;
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
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
        total = total + _shfl_xor_sync_5;
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        total = total + _shfl_xor_sync_6;
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
        total = total + _shfl_xor_sync_7;
        float _sqrt_8;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_8) : "f"(total * 0.0078125f + eps));
        float rstd = __frcp_rn(_sqrt_8);
        int fast = 0;
        float _neg_9;
        asm("neg.f32 %0, %1;" : "=f"(_neg_9) : "f"(zs[0]));
        float _neg_10;
        asm("neg.f32 %0, %1;" : "=f"(_neg_10) : "f"(zs[1]));
        float _neg_11;
        asm("neg.f32 %0, %1;" : "=f"(_neg_11) : "f"(zs[2]));
        float mz;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz) : "f"(_neg_9), "f"(_neg_10), "f"(_neg_11));
        float _abs_13;
        asm("abs.f32 %0, %1;" : "=f"(_abs_13) : "f"(zs[0]));
        float _neg_12;
        asm("neg.f32 %0, %1;" : "=f"(_neg_12) : "f"(_abs_13));
        float _abs_15;
        asm("abs.f32 %0, %1;" : "=f"(_abs_15) : "f"(zs[1]));
        float _neg_14;
        asm("neg.f32 %0, %1;" : "=f"(_neg_14) : "f"(_abs_15));
        float _abs_17;
        asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(zs[2]));
        float _neg_16;
        asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(_abs_17));
        float ma;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma) : "f"(_neg_12), "f"(_neg_14), "f"(_neg_16));
        float _neg_19;
        asm("neg.f32 %0, %1;" : "=f"(_neg_19) : "f"(zs[3]));
        float _neg_20;
        asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(zs[4]));
        float _max3_18;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_18) : "f"(mz), "f"(_neg_19), "f"(_neg_20));
        mz = _max3_18;
        float _abs_23;
        asm("abs.f32 %0, %1;" : "=f"(_abs_23) : "f"(zs[3]));
        float _neg_22;
        asm("neg.f32 %0, %1;" : "=f"(_neg_22) : "f"(_abs_23));
        float _abs_25;
        asm("abs.f32 %0, %1;" : "=f"(_abs_25) : "f"(zs[4]));
        float _neg_24;
        asm("neg.f32 %0, %1;" : "=f"(_neg_24) : "f"(_abs_25));
        float _max3_21;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_21) : "f"(ma), "f"(_neg_22), "f"(_neg_24));
        ma = _max3_21;
        float _neg_27;
        asm("neg.f32 %0, %1;" : "=f"(_neg_27) : "f"(zs[5]));
        float _neg_28;
        asm("neg.f32 %0, %1;" : "=f"(_neg_28) : "f"(zs[6]));
        float _max3_26;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_26) : "f"(mz), "f"(_neg_27), "f"(_neg_28));
        mz = _max3_26;
        float _abs_31;
        asm("abs.f32 %0, %1;" : "=f"(_abs_31) : "f"(zs[5]));
        float _neg_30;
        asm("neg.f32 %0, %1;" : "=f"(_neg_30) : "f"(_abs_31));
        float _abs_33;
        asm("abs.f32 %0, %1;" : "=f"(_abs_33) : "f"(zs[6]));
        float _neg_32;
        asm("neg.f32 %0, %1;" : "=f"(_neg_32) : "f"(_abs_33));
        float _max3_29;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_29) : "f"(ma), "f"(_neg_30), "f"(_neg_32));
        ma = _max3_29;
        float _neg_35;
        asm("neg.f32 %0, %1;" : "=f"(_neg_35) : "f"(zs[7]));
        float _neg_36;
        asm("neg.f32 %0, %1;" : "=f"(_neg_36) : "f"(zs[8]));
        float _max3_34;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_34) : "f"(mz), "f"(_neg_35), "f"(_neg_36));
        mz = _max3_34;
        float _abs_39;
        asm("abs.f32 %0, %1;" : "=f"(_abs_39) : "f"(zs[7]));
        float _neg_38;
        asm("neg.f32 %0, %1;" : "=f"(_neg_38) : "f"(_abs_39));
        float _abs_41;
        asm("abs.f32 %0, %1;" : "=f"(_abs_41) : "f"(zs[8]));
        float _neg_40;
        asm("neg.f32 %0, %1;" : "=f"(_neg_40) : "f"(_abs_41));
        float _max3_37;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_37) : "f"(ma), "f"(_neg_38), "f"(_neg_40));
        ma = _max3_37;
        float _neg_43;
        asm("neg.f32 %0, %1;" : "=f"(_neg_43) : "f"(zs[9]));
        float _neg_44;
        asm("neg.f32 %0, %1;" : "=f"(_neg_44) : "f"(zs[10]));
        float _max3_42;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_42) : "f"(mz), "f"(_neg_43), "f"(_neg_44));
        mz = _max3_42;
        float _abs_47;
        asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(zs[9]));
        float _neg_46;
        asm("neg.f32 %0, %1;" : "=f"(_neg_46) : "f"(_abs_47));
        float _abs_49;
        asm("abs.f32 %0, %1;" : "=f"(_abs_49) : "f"(zs[10]));
        float _neg_48;
        asm("neg.f32 %0, %1;" : "=f"(_neg_48) : "f"(_abs_49));
        float _max3_45;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_45) : "f"(ma), "f"(_neg_46), "f"(_neg_48));
        ma = _max3_45;
        float _neg_51;
        asm("neg.f32 %0, %1;" : "=f"(_neg_51) : "f"(zs[11]));
        float _neg_52;
        asm("neg.f32 %0, %1;" : "=f"(_neg_52) : "f"(zs[12]));
        float _max3_50;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_50) : "f"(mz), "f"(_neg_51), "f"(_neg_52));
        mz = _max3_50;
        float _abs_55;
        asm("abs.f32 %0, %1;" : "=f"(_abs_55) : "f"(zs[11]));
        float _neg_54;
        asm("neg.f32 %0, %1;" : "=f"(_neg_54) : "f"(_abs_55));
        float _abs_57;
        asm("abs.f32 %0, %1;" : "=f"(_abs_57) : "f"(zs[12]));
        float _neg_56;
        asm("neg.f32 %0, %1;" : "=f"(_neg_56) : "f"(_abs_57));
        float _max3_53;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_53) : "f"(ma), "f"(_neg_54), "f"(_neg_56));
        ma = _max3_53;
        float _neg_59;
        asm("neg.f32 %0, %1;" : "=f"(_neg_59) : "f"(zs[13]));
        float _neg_60;
        asm("neg.f32 %0, %1;" : "=f"(_neg_60) : "f"(zs[14]));
        float _max3_58;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_58) : "f"(mz), "f"(_neg_59), "f"(_neg_60));
        mz = _max3_58;
        float _abs_63;
        asm("abs.f32 %0, %1;" : "=f"(_abs_63) : "f"(zs[13]));
        float _neg_62;
        asm("neg.f32 %0, %1;" : "=f"(_neg_62) : "f"(_abs_63));
        float _abs_65;
        asm("abs.f32 %0, %1;" : "=f"(_abs_65) : "f"(zs[14]));
        float _neg_64;
        asm("neg.f32 %0, %1;" : "=f"(_neg_64) : "f"(_abs_65));
        float _max3_61;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_61) : "f"(ma), "f"(_neg_62), "f"(_neg_64));
        ma = _max3_61;
        float _neg_67;
        asm("neg.f32 %0, %1;" : "=f"(_neg_67) : "f"(zs[15]));
        float _max_66;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_66) : "f"(mz), "f"(_neg_67));
        mz = _max_66;
        float _abs_70;
        asm("abs.f32 %0, %1;" : "=f"(_abs_70) : "f"(zs[15]));
        float _neg_69;
        asm("neg.f32 %0, %1;" : "=f"(_neg_69) : "f"(_abs_70));
        float _max_68;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_68) : "f"(ma), "f"(_neg_69));
        ma = _max_68;
        float _abs_71;
        asm("abs.f32 %0, %1;" : "=f"(_abs_71) : "f"(zs[0]));
        float _abs_72;
        asm("abs.f32 %0, %1;" : "=f"(_abs_72) : "f"(zs[1]));
        float _abs_73;
        asm("abs.f32 %0, %1;" : "=f"(_abs_73) : "f"(zs[2]));
        float mp;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp) : "f"(_abs_71), "f"(_abs_72), "f"(_abs_73));
        float _abs_75;
        asm("abs.f32 %0, %1;" : "=f"(_abs_75) : "f"(zs[3]));
        float _abs_76;
        asm("abs.f32 %0, %1;" : "=f"(_abs_76) : "f"(zs[4]));
        float _max3_74;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_74) : "f"(mp), "f"(_abs_75), "f"(_abs_76));
        mp = _max3_74;
        float _abs_78;
        asm("abs.f32 %0, %1;" : "=f"(_abs_78) : "f"(zs[5]));
        float _abs_79;
        asm("abs.f32 %0, %1;" : "=f"(_abs_79) : "f"(zs[6]));
        float _max3_77;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_77) : "f"(mp), "f"(_abs_78), "f"(_abs_79));
        mp = _max3_77;
        float _abs_81;
        asm("abs.f32 %0, %1;" : "=f"(_abs_81) : "f"(zs[7]));
        float _abs_82;
        asm("abs.f32 %0, %1;" : "=f"(_abs_82) : "f"(zs[8]));
        float _max3_80;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_80) : "f"(mp), "f"(_abs_81), "f"(_abs_82));
        mp = _max3_80;
        float _abs_84;
        asm("abs.f32 %0, %1;" : "=f"(_abs_84) : "f"(zs[9]));
        float _abs_85;
        asm("abs.f32 %0, %1;" : "=f"(_abs_85) : "f"(zs[10]));
        float _max3_83;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_83) : "f"(mp), "f"(_abs_84), "f"(_abs_85));
        mp = _max3_83;
        float _abs_87;
        asm("abs.f32 %0, %1;" : "=f"(_abs_87) : "f"(zs[11]));
        float _abs_88;
        asm("abs.f32 %0, %1;" : "=f"(_abs_88) : "f"(zs[12]));
        float _max3_86;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_86) : "f"(mp), "f"(_abs_87), "f"(_abs_88));
        mp = _max3_86;
        float _abs_90;
        asm("abs.f32 %0, %1;" : "=f"(_abs_90) : "f"(zs[13]));
        float _abs_91;
        asm("abs.f32 %0, %1;" : "=f"(_abs_91) : "f"(zs[14]));
        float _max3_89;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_89) : "f"(mp), "f"(_abs_90), "f"(_abs_91));
        mp = _max3_89;
        float _abs_93;
        asm("abs.f32 %0, %1;" : "=f"(_abs_93) : "f"(zs[15]));
        float _max_92;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_92) : "f"(mp), "f"(_abs_93));
        mp = _max_92;
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
            float _fma_94;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(ta_0), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_95;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(tb_0), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_0 = make_float2(_fma_94, _fma_95);
            float2 r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_0));
            r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.925963e-08f, -1.925963e-08f), r2_0);
            float2 nscale_0 = make_float2(__uint_as_float(__as_u32(t2_0.x) << 23), __uint_as_float(__as_u32(t2_0.y) << 23));
            float _exp2_96 = approx_exp2(r2_0.x);
            float _exp2_97 = approx_exp2(r2_0.y);
            float2 nden_0 = fma_f32x2_rn_noftz(nscale_0, make_float2(_exp2_96, _exp2_97), make_float2(-1.0f, -1.0f));
            float _neg_99;
            asm("neg.f32 %0, %1;" : "=f"(_neg_99) : "f"(nden_0.x));
            float _rcp_98 = approx_rcp(_neg_99);
            float _neg_101;
            asm("neg.f32 %0, %1;" : "=f"(_neg_101) : "f"(nden_0.y));
            float _rcp_100 = approx_rcp(_neg_101);
            float2 r0_0 = make_float2(_rcp_98, _rcp_100);
            float2 r1_0 = fma_f32x2_rn_noftz(r0_0, fma_f32x2_rn_noftz(nden_0, r0_0, make_float2(1.0f, 1.0f)), r0_0);
            float2 q0_0 = fma_f32x2_rn_noftz(z2_0, r1_0, make_float2(0.0f, 0.0f));
            float2 gate2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, q0_0, z2_0), q0_0);
            float2 _f32x2_mul_a_102 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[0], xs[1]), rstd2), make_float2(wv[0], wv[1]));
            float2 y2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_102), "l"(*(const unsigned long long*)&gate2_0));
            xs[0] = y2_0.x;
            xs[1] = y2_0.y;
            float2 z2_1 = make_float2(zs[2], zs[3]);
            float ta_1;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_1) : "f"(zs[2]), "f"(-0.00572498f), "f"(0.5f));
            float tb_1;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_1) : "f"(zs[3]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_103;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(ta_1), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_104;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(tb_1), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_1 = make_float2(_fma_103, _fma_104);
            float2 r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_1));
            r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.925963e-08f, -1.925963e-08f), r2_1);
            float2 nscale_1 = make_float2(__uint_as_float(__as_u32(t2_1.x) << 23), __uint_as_float(__as_u32(t2_1.y) << 23));
            float _exp2_105 = approx_exp2(r2_1.x);
            float _exp2_106 = approx_exp2(r2_1.y);
            float2 nden_1 = fma_f32x2_rn_noftz(nscale_1, make_float2(_exp2_105, _exp2_106), make_float2(-1.0f, -1.0f));
            float _neg_108;
            asm("neg.f32 %0, %1;" : "=f"(_neg_108) : "f"(nden_1.x));
            float _rcp_107 = approx_rcp(_neg_108);
            float _neg_110;
            asm("neg.f32 %0, %1;" : "=f"(_neg_110) : "f"(nden_1.y));
            float _rcp_109 = approx_rcp(_neg_110);
            float2 r0_1 = make_float2(_rcp_107, _rcp_109);
            float2 r1_1 = fma_f32x2_rn_noftz(r0_1, fma_f32x2_rn_noftz(nden_1, r0_1, make_float2(1.0f, 1.0f)), r0_1);
            float2 q0_1 = fma_f32x2_rn_noftz(z2_1, r1_1, make_float2(0.0f, 0.0f));
            float2 gate2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, q0_1, z2_1), q0_1);
            float2 _f32x2_mul_a_111 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[2], xs[3]), rstd2), make_float2(wv[2], wv[3]));
            float2 y2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&gate2_1));
            xs[2] = y2_1.x;
            xs[3] = y2_1.y;
            float2 z2_2 = make_float2(zs[4], zs[5]);
            float ta_2;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_2) : "f"(zs[4]), "f"(-0.00572498f), "f"(0.5f));
            float tb_2;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_2) : "f"(zs[5]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_112;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(ta_2), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_113;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(tb_2), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_2 = make_float2(_fma_112, _fma_113);
            float2 r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_2));
            r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.925963e-08f, -1.925963e-08f), r2_2);
            float2 nscale_2 = make_float2(__uint_as_float(__as_u32(t2_2.x) << 23), __uint_as_float(__as_u32(t2_2.y) << 23));
            float _exp2_114 = approx_exp2(r2_2.x);
            float _exp2_115 = approx_exp2(r2_2.y);
            float2 nden_2 = fma_f32x2_rn_noftz(nscale_2, make_float2(_exp2_114, _exp2_115), make_float2(-1.0f, -1.0f));
            float _neg_117;
            asm("neg.f32 %0, %1;" : "=f"(_neg_117) : "f"(nden_2.x));
            float _rcp_116 = approx_rcp(_neg_117);
            float _neg_119;
            asm("neg.f32 %0, %1;" : "=f"(_neg_119) : "f"(nden_2.y));
            float _rcp_118 = approx_rcp(_neg_119);
            float2 r0_2 = make_float2(_rcp_116, _rcp_118);
            float2 r1_2 = fma_f32x2_rn_noftz(r0_2, fma_f32x2_rn_noftz(nden_2, r0_2, make_float2(1.0f, 1.0f)), r0_2);
            float2 q0_2 = fma_f32x2_rn_noftz(z2_2, r1_2, make_float2(0.0f, 0.0f));
            float2 gate2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, q0_2, z2_2), q0_2);
            float2 _f32x2_mul_a_120 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[4], xs[5]), rstd2), make_float2(wv[4], wv[5]));
            float2 y2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&gate2_2));
            xs[4] = y2_2.x;
            xs[5] = y2_2.y;
            float2 z2_3 = make_float2(zs[6], zs[7]);
            float ta_3;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_3) : "f"(zs[6]), "f"(-0.00572498f), "f"(0.5f));
            float tb_3;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_3) : "f"(zs[7]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_121;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(ta_3), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_122;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(tb_3), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_3 = make_float2(_fma_121, _fma_122);
            float2 r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_3));
            r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.925963e-08f, -1.925963e-08f), r2_3);
            float2 nscale_3 = make_float2(__uint_as_float(__as_u32(t2_3.x) << 23), __uint_as_float(__as_u32(t2_3.y) << 23));
            float _exp2_123 = approx_exp2(r2_3.x);
            float _exp2_124 = approx_exp2(r2_3.y);
            float2 nden_3 = fma_f32x2_rn_noftz(nscale_3, make_float2(_exp2_123, _exp2_124), make_float2(-1.0f, -1.0f));
            float _neg_126;
            asm("neg.f32 %0, %1;" : "=f"(_neg_126) : "f"(nden_3.x));
            float _rcp_125 = approx_rcp(_neg_126);
            float _neg_128;
            asm("neg.f32 %0, %1;" : "=f"(_neg_128) : "f"(nden_3.y));
            float _rcp_127 = approx_rcp(_neg_128);
            float2 r0_3 = make_float2(_rcp_125, _rcp_127);
            float2 r1_3 = fma_f32x2_rn_noftz(r0_3, fma_f32x2_rn_noftz(nden_3, r0_3, make_float2(1.0f, 1.0f)), r0_3);
            float2 q0_3 = fma_f32x2_rn_noftz(z2_3, r1_3, make_float2(0.0f, 0.0f));
            float2 gate2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, q0_3, z2_3), q0_3);
            float2 _f32x2_mul_a_129 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[6], xs[7]), rstd2), make_float2(wv[6], wv[7]));
            float2 y2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_129), "l"(*(const unsigned long long*)&gate2_3));
            xs[6] = y2_3.x;
            xs[7] = y2_3.y;
            float2 z2_4 = make_float2(zs[8], zs[9]);
            float ta_4;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_4) : "f"(zs[8]), "f"(-0.00572498f), "f"(0.5f));
            float tb_4;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_4) : "f"(zs[9]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_130;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(ta_4), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_131;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(tb_4), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_4 = make_float2(_fma_130, _fma_131);
            float2 r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_4));
            r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.925963e-08f, -1.925963e-08f), r2_4);
            float2 nscale_4 = make_float2(__uint_as_float(__as_u32(t2_4.x) << 23), __uint_as_float(__as_u32(t2_4.y) << 23));
            float _exp2_132 = approx_exp2(r2_4.x);
            float _exp2_133 = approx_exp2(r2_4.y);
            float2 nden_4 = fma_f32x2_rn_noftz(nscale_4, make_float2(_exp2_132, _exp2_133), make_float2(-1.0f, -1.0f));
            float _neg_135;
            asm("neg.f32 %0, %1;" : "=f"(_neg_135) : "f"(nden_4.x));
            float _rcp_134 = approx_rcp(_neg_135);
            float _neg_137;
            asm("neg.f32 %0, %1;" : "=f"(_neg_137) : "f"(nden_4.y));
            float _rcp_136 = approx_rcp(_neg_137);
            float2 r0_4 = make_float2(_rcp_134, _rcp_136);
            float2 r1_4 = fma_f32x2_rn_noftz(r0_4, fma_f32x2_rn_noftz(nden_4, r0_4, make_float2(1.0f, 1.0f)), r0_4);
            float2 q0_4 = fma_f32x2_rn_noftz(z2_4, r1_4, make_float2(0.0f, 0.0f));
            float2 gate2_4 = fma_f32x2_rn_noftz(r1_4, fma_f32x2_rn_noftz(nden_4, q0_4, z2_4), q0_4);
            float2 _f32x2_mul_a_138 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[8], xs[9]), rstd2), make_float2(wv[8], wv[9]));
            float2 y2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&gate2_4));
            xs[8] = y2_4.x;
            xs[9] = y2_4.y;
            float2 z2_5 = make_float2(zs[10], zs[11]);
            float ta_5;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_5) : "f"(zs[10]), "f"(-0.00572498f), "f"(0.5f));
            float tb_5;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_5) : "f"(zs[11]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_139;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(ta_5), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_140;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(tb_5), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_5 = make_float2(_fma_139, _fma_140);
            float2 r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_5));
            r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.925963e-08f, -1.925963e-08f), r2_5);
            float2 nscale_5 = make_float2(__uint_as_float(__as_u32(t2_5.x) << 23), __uint_as_float(__as_u32(t2_5.y) << 23));
            float _exp2_141 = approx_exp2(r2_5.x);
            float _exp2_142 = approx_exp2(r2_5.y);
            float2 nden_5 = fma_f32x2_rn_noftz(nscale_5, make_float2(_exp2_141, _exp2_142), make_float2(-1.0f, -1.0f));
            float _neg_144;
            asm("neg.f32 %0, %1;" : "=f"(_neg_144) : "f"(nden_5.x));
            float _rcp_143 = approx_rcp(_neg_144);
            float _neg_146;
            asm("neg.f32 %0, %1;" : "=f"(_neg_146) : "f"(nden_5.y));
            float _rcp_145 = approx_rcp(_neg_146);
            float2 r0_5 = make_float2(_rcp_143, _rcp_145);
            float2 r1_5 = fma_f32x2_rn_noftz(r0_5, fma_f32x2_rn_noftz(nden_5, r0_5, make_float2(1.0f, 1.0f)), r0_5);
            float2 q0_5 = fma_f32x2_rn_noftz(z2_5, r1_5, make_float2(0.0f, 0.0f));
            float2 gate2_5 = fma_f32x2_rn_noftz(r1_5, fma_f32x2_rn_noftz(nden_5, q0_5, z2_5), q0_5);
            float2 _f32x2_mul_a_147 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[10], xs[11]), rstd2), make_float2(wv[10], wv[11]));
            float2 y2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_147), "l"(*(const unsigned long long*)&gate2_5));
            xs[10] = y2_5.x;
            xs[11] = y2_5.y;
            float2 z2_6 = make_float2(zs[12], zs[13]);
            float ta_6;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_6) : "f"(zs[12]), "f"(-0.00572498f), "f"(0.5f));
            float tb_6;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_6) : "f"(zs[13]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_148;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(ta_6), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_149;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(tb_6), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_6 = make_float2(_fma_148, _fma_149);
            float2 r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_6));
            r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.925963e-08f, -1.925963e-08f), r2_6);
            float2 nscale_6 = make_float2(__uint_as_float(__as_u32(t2_6.x) << 23), __uint_as_float(__as_u32(t2_6.y) << 23));
            float _exp2_150 = approx_exp2(r2_6.x);
            float _exp2_151 = approx_exp2(r2_6.y);
            float2 nden_6 = fma_f32x2_rn_noftz(nscale_6, make_float2(_exp2_150, _exp2_151), make_float2(-1.0f, -1.0f));
            float _neg_153;
            asm("neg.f32 %0, %1;" : "=f"(_neg_153) : "f"(nden_6.x));
            float _rcp_152 = approx_rcp(_neg_153);
            float _neg_155;
            asm("neg.f32 %0, %1;" : "=f"(_neg_155) : "f"(nden_6.y));
            float _rcp_154 = approx_rcp(_neg_155);
            float2 r0_6 = make_float2(_rcp_152, _rcp_154);
            float2 r1_6 = fma_f32x2_rn_noftz(r0_6, fma_f32x2_rn_noftz(nden_6, r0_6, make_float2(1.0f, 1.0f)), r0_6);
            float2 q0_6 = fma_f32x2_rn_noftz(z2_6, r1_6, make_float2(0.0f, 0.0f));
            float2 gate2_6 = fma_f32x2_rn_noftz(r1_6, fma_f32x2_rn_noftz(nden_6, q0_6, z2_6), q0_6);
            float2 _f32x2_mul_a_156 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[12], xs[13]), rstd2), make_float2(wv[12], wv[13]));
            float2 y2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&gate2_6));
            xs[12] = y2_6.x;
            xs[13] = y2_6.y;
            float2 z2_7 = make_float2(zs[14], zs[15]);
            float ta_7;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_7) : "f"(zs[14]), "f"(-0.00572498f), "f"(0.5f));
            float tb_7;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_7) : "f"(zs[15]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_157;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(ta_7), "f"(252.0f), "f"(1.2583169e+07f));
            float _fma_158;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(tb_7), "f"(252.0f), "f"(1.2583169e+07f));
            float2 t2_7 = make_float2(_fma_157, _fma_158);
            float2 r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_7));
            r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.925963e-08f, -1.925963e-08f), r2_7);
            float2 nscale_7 = make_float2(__uint_as_float(__as_u32(t2_7.x) << 23), __uint_as_float(__as_u32(t2_7.y) << 23));
            float _exp2_159 = approx_exp2(r2_7.x);
            float _exp2_160 = approx_exp2(r2_7.y);
            float2 nden_7 = fma_f32x2_rn_noftz(nscale_7, make_float2(_exp2_159, _exp2_160), make_float2(-1.0f, -1.0f));
            float _neg_162;
            asm("neg.f32 %0, %1;" : "=f"(_neg_162) : "f"(nden_7.x));
            float _rcp_161 = approx_rcp(_neg_162);
            float _neg_164;
            asm("neg.f32 %0, %1;" : "=f"(_neg_164) : "f"(nden_7.y));
            float _rcp_163 = approx_rcp(_neg_164);
            float2 r0_7 = make_float2(_rcp_161, _rcp_163);
            float2 r1_7 = fma_f32x2_rn_noftz(r0_7, fma_f32x2_rn_noftz(nden_7, r0_7, make_float2(1.0f, 1.0f)), r0_7);
            float2 q0_7 = fma_f32x2_rn_noftz(z2_7, r1_7, make_float2(0.0f, 0.0f));
            float2 gate2_7 = fma_f32x2_rn_noftz(r1_7, fma_f32x2_rn_noftz(nden_7, q0_7, z2_7), q0_7);
            float2 _f32x2_mul_a_165 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[14], xs[15]), rstd2), make_float2(wv[14], wv[15]));
            float2 y2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_165), "l"(*(const unsigned long long*)&gate2_7));
            xs[14] = y2_7.x;
            xs[15] = y2_7.y;
        } else {
            float _exp_166 = __nv_expf(zs[0] * -1.0f);
            float gate_0 = __fdiv_rn(zs[0], 1.0f + _exp_166);
            xs[0] = xs[0] * rstd * wv[0] * gate_0;
            float _exp_167 = __nv_expf(zs[1] * -1.0f);
            float gate_1 = __fdiv_rn(zs[1], 1.0f + _exp_167);
            xs[1] = xs[1] * rstd * wv[1] * gate_1;
            float _exp_168 = __nv_expf(zs[2] * -1.0f);
            float gate_2 = __fdiv_rn(zs[2], 1.0f + _exp_168);
            xs[2] = xs[2] * rstd * wv[2] * gate_2;
            float _exp_169 = __nv_expf(zs[3] * -1.0f);
            float gate_3 = __fdiv_rn(zs[3], 1.0f + _exp_169);
            xs[3] = xs[3] * rstd * wv[3] * gate_3;
            float _exp_170 = __nv_expf(zs[4] * -1.0f);
            float gate_4 = __fdiv_rn(zs[4], 1.0f + _exp_170);
            xs[4] = xs[4] * rstd * wv[4] * gate_4;
            float _exp_171 = __nv_expf(zs[5] * -1.0f);
            float gate_5 = __fdiv_rn(zs[5], 1.0f + _exp_171);
            xs[5] = xs[5] * rstd * wv[5] * gate_5;
            float _exp_172 = __nv_expf(zs[6] * -1.0f);
            float gate_6 = __fdiv_rn(zs[6], 1.0f + _exp_172);
            xs[6] = xs[6] * rstd * wv[6] * gate_6;
            float _exp_173 = __nv_expf(zs[7] * -1.0f);
            float gate_7 = __fdiv_rn(zs[7], 1.0f + _exp_173);
            xs[7] = xs[7] * rstd * wv[7] * gate_7;
            float _exp_174 = __nv_expf(zs[8] * -1.0f);
            float gate_8 = __fdiv_rn(zs[8], 1.0f + _exp_174);
            xs[8] = xs[8] * rstd * wv[8] * gate_8;
            float _exp_175 = __nv_expf(zs[9] * -1.0f);
            float gate_9 = __fdiv_rn(zs[9], 1.0f + _exp_175);
            xs[9] = xs[9] * rstd * wv[9] * gate_9;
            float _exp_176 = __nv_expf(zs[10] * -1.0f);
            float gate_10 = __fdiv_rn(zs[10], 1.0f + _exp_176);
            xs[10] = xs[10] * rstd * wv[10] * gate_10;
            float _exp_177 = __nv_expf(zs[11] * -1.0f);
            float gate_11 = __fdiv_rn(zs[11], 1.0f + _exp_177);
            xs[11] = xs[11] * rstd * wv[11] * gate_11;
            float _exp_178 = __nv_expf(zs[12] * -1.0f);
            float gate_12 = __fdiv_rn(zs[12], 1.0f + _exp_178);
            xs[12] = xs[12] * rstd * wv[12] * gate_12;
            float _exp_179 = __nv_expf(zs[13] * -1.0f);
            float gate_13 = __fdiv_rn(zs[13], 1.0f + _exp_179);
            xs[13] = xs[13] * rstd * wv[13] * gate_13;
            float _exp_180 = __nv_expf(zs[14] * -1.0f);
            float gate_14 = __fdiv_rn(zs[14], 1.0f + _exp_180);
            xs[14] = xs[14] * rstd * wv[14] * gate_14;
            float _exp_181 = __nv_expf(zs[15] * -1.0f);
            float gate_15 = __fdiv_rn(zs[15], 1.0f + _exp_181);
            xs[15] = xs[15] * rstd * wv[15] * gate_15;
        }
        unsigned int packed[8];
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
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 8], xs[_lp*2+1 + 8]));
            packed[_lp + 4] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 10], xs[_lp*2+1 + 10]));
            packed[_lp + 5] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 12], xs[_lp*2+1 + 12]));
            packed[_lp + 6] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 14], xs[_lp*2+1 + 14]));
            packed[_lp + 7] = *(uint32_t*)&_bf2;
        }
        if (blk * 16 + group < rows) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128 + (long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
            reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128 + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed + (4))[0];
        }
    }

    // === Task calls (declaration order) ===
}

} // extern "C"
