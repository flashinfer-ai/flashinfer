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

__global__ __launch_bounds__(128) void
kernel_cake_rmsnorm_silu_gate_train_3c1aaebdec47df0111e4(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, CakeFastDivmod heads, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int group = tid >> 3;
    int col = (tid & 7) * 16;
    float wv[16];
    {
        const void* _v8p_0 = (const void*)(w + (col + 0) + (0));
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
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row_0 = blk * 16 + 0 + group;
        int src_0 = ((row_0) < (rows - 1) ? (row_0) : (rows - 1));
        uint32_t tq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_0);
        uint32_t th_0 = (uint32_t)((unsigned int)src_0) - tq_0 * (uint32_t)(heads.divisor);
        long long xoff_0 = (long long)tq_0 * x_tok + (long long)th_0 * x_head + (long long)col;
        long long zoff_0 = (long long)tq_0 * z_tok + (long long)th_0 * z_head + (long long)col;
        float xs_0[16];
        float zs_0[16];
        {
            const void* _v8p_1 = (const void*)(x + (xoff_0 + 0) + (0));
            uint32_t _v8_1_0[8];
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_v8_1_0[0]), "=r"(_v8_1_0[1]), "=r"(_v8_1_0[2]), "=r"(_v8_1_0[3]), "=r"(_v8_1_0[4]), "=r"(_v8_1_0[5]), "=r"(_v8_1_0[6]), "=r"(_v8_1_0[7]) : "l"((const char*)_v8p_1 + 0) : "memory");
            (&xs_0[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[0]) << 16);
            (&xs_0[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[0]) & 0xffff0000u);
            (&xs_0[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[1]) << 16);
            (&xs_0[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[1]) & 0xffff0000u);
            (&xs_0[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[2]) << 16);
            (&xs_0[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[2]) & 0xffff0000u);
            (&xs_0[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[3]) << 16);
            (&xs_0[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[3]) & 0xffff0000u);
            (&xs_0[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[4]) << 16);
            (&xs_0[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[4]) & 0xffff0000u);
            (&xs_0[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[5]) << 16);
            (&xs_0[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[5]) & 0xffff0000u);
            (&xs_0[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[6]) << 16);
            (&xs_0[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[6]) & 0xffff0000u);
            (&xs_0[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[7]) << 16);
            (&xs_0[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[7]) & 0xffff0000u);
        }
        {
            const void* _v8p_2 = (const void*)(z + (zoff_0 + 0) + (0));
            uint32_t _v8_2_0[8];
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_v8_2_0[0]), "=r"(_v8_2_0[1]), "=r"(_v8_2_0[2]), "=r"(_v8_2_0[3]), "=r"(_v8_2_0[4]), "=r"(_v8_2_0[5]), "=r"(_v8_2_0[6]), "=r"(_v8_2_0[7]) : "l"((const char*)_v8p_2 + 0) : "memory");
            (&zs_0[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) << 16);
            (&zs_0[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) & 0xffff0000u);
            (&zs_0[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) << 16);
            (&zs_0[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) & 0xffff0000u);
            (&zs_0[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) << 16);
            (&zs_0[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) & 0xffff0000u);
            (&zs_0[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) << 16);
            (&zs_0[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) & 0xffff0000u);
            (&zs_0[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) << 16);
            (&zs_0[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) & 0xffff0000u);
            (&zs_0[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) << 16);
            (&zs_0[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) & 0xffff0000u);
            (&zs_0[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) << 16);
            (&zs_0[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) & 0xffff0000u);
            (&zs_0[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) << 16);
            (&zs_0[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) & 0xffff0000u);
        }
        float total_0 = 0.0f;
        total_0 = __fmaf_rn(xs_0[0], xs_0[0], total_0);
        total_0 = __fmaf_rn(xs_0[1], xs_0[1], total_0);
        total_0 = __fmaf_rn(xs_0[2], xs_0[2], total_0);
        total_0 = __fmaf_rn(xs_0[3], xs_0[3], total_0);
        total_0 = __fmaf_rn(xs_0[4], xs_0[4], total_0);
        total_0 = __fmaf_rn(xs_0[5], xs_0[5], total_0);
        total_0 = __fmaf_rn(xs_0[6], xs_0[6], total_0);
        total_0 = __fmaf_rn(xs_0[7], xs_0[7], total_0);
        total_0 = __fmaf_rn(xs_0[8], xs_0[8], total_0);
        total_0 = __fmaf_rn(xs_0[9], xs_0[9], total_0);
        total_0 = __fmaf_rn(xs_0[10], xs_0[10], total_0);
        total_0 = __fmaf_rn(xs_0[11], xs_0[11], total_0);
        total_0 = __fmaf_rn(xs_0[12], xs_0[12], total_0);
        total_0 = __fmaf_rn(xs_0[13], xs_0[13], total_0);
        total_0 = __fmaf_rn(xs_0[14], xs_0[14], total_0);
        total_0 = __fmaf_rn(xs_0[15], xs_0[15], total_0);
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, total_0, 1);
        total_0 = total_0 + _shfl_xor_sync_3;
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, total_0, 2);
        total_0 = total_0 + _shfl_xor_sync_4;
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, total_0, 4);
        total_0 = total_0 + _shfl_xor_sync_5;
        float _sqrt_6;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_6) : "f"(total_0 * 0.0078125f + eps));
        float rstd_0 = __frcp_rn(_sqrt_6);
        int fast_0 = 0;
        float _neg_7;
        asm("neg.f32 %0, %1;" : "=f"(_neg_7) : "f"(zs_0[0]));
        float _neg_8;
        asm("neg.f32 %0, %1;" : "=f"(_neg_8) : "f"(zs_0[1]));
        float _neg_9;
        asm("neg.f32 %0, %1;" : "=f"(_neg_9) : "f"(zs_0[2]));
        float mz_0;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz_0) : "f"(_neg_7), "f"(_neg_8), "f"(_neg_9));
        float _abs_11;
        asm("abs.f32 %0, %1;" : "=f"(_abs_11) : "f"(zs_0[0]));
        float _neg_10;
        asm("neg.f32 %0, %1;" : "=f"(_neg_10) : "f"(_abs_11));
        float _abs_13;
        asm("abs.f32 %0, %1;" : "=f"(_abs_13) : "f"(zs_0[1]));
        float _neg_12;
        asm("neg.f32 %0, %1;" : "=f"(_neg_12) : "f"(_abs_13));
        float _abs_15;
        asm("abs.f32 %0, %1;" : "=f"(_abs_15) : "f"(zs_0[2]));
        float _neg_14;
        asm("neg.f32 %0, %1;" : "=f"(_neg_14) : "f"(_abs_15));
        float ma_0;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma_0) : "f"(_neg_10), "f"(_neg_12), "f"(_neg_14));
        float _neg_17;
        asm("neg.f32 %0, %1;" : "=f"(_neg_17) : "f"(zs_0[3]));
        float _neg_18;
        asm("neg.f32 %0, %1;" : "=f"(_neg_18) : "f"(zs_0[4]));
        float _max3_16;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_16) : "f"(mz_0), "f"(_neg_17), "f"(_neg_18));
        mz_0 = _max3_16;
        float _abs_21;
        asm("abs.f32 %0, %1;" : "=f"(_abs_21) : "f"(zs_0[3]));
        float _neg_20;
        asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(_abs_21));
        float _abs_23;
        asm("abs.f32 %0, %1;" : "=f"(_abs_23) : "f"(zs_0[4]));
        float _neg_22;
        asm("neg.f32 %0, %1;" : "=f"(_neg_22) : "f"(_abs_23));
        float _max3_19;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_19) : "f"(ma_0), "f"(_neg_20), "f"(_neg_22));
        ma_0 = _max3_19;
        float _neg_25;
        asm("neg.f32 %0, %1;" : "=f"(_neg_25) : "f"(zs_0[5]));
        float _neg_26;
        asm("neg.f32 %0, %1;" : "=f"(_neg_26) : "f"(zs_0[6]));
        float _max3_24;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_24) : "f"(mz_0), "f"(_neg_25), "f"(_neg_26));
        mz_0 = _max3_24;
        float _abs_29;
        asm("abs.f32 %0, %1;" : "=f"(_abs_29) : "f"(zs_0[5]));
        float _neg_28;
        asm("neg.f32 %0, %1;" : "=f"(_neg_28) : "f"(_abs_29));
        float _abs_31;
        asm("abs.f32 %0, %1;" : "=f"(_abs_31) : "f"(zs_0[6]));
        float _neg_30;
        asm("neg.f32 %0, %1;" : "=f"(_neg_30) : "f"(_abs_31));
        float _max3_27;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_27) : "f"(ma_0), "f"(_neg_28), "f"(_neg_30));
        ma_0 = _max3_27;
        float _neg_33;
        asm("neg.f32 %0, %1;" : "=f"(_neg_33) : "f"(zs_0[7]));
        float _neg_34;
        asm("neg.f32 %0, %1;" : "=f"(_neg_34) : "f"(zs_0[8]));
        float _max3_32;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_32) : "f"(mz_0), "f"(_neg_33), "f"(_neg_34));
        mz_0 = _max3_32;
        float _abs_37;
        asm("abs.f32 %0, %1;" : "=f"(_abs_37) : "f"(zs_0[7]));
        float _neg_36;
        asm("neg.f32 %0, %1;" : "=f"(_neg_36) : "f"(_abs_37));
        float _abs_39;
        asm("abs.f32 %0, %1;" : "=f"(_abs_39) : "f"(zs_0[8]));
        float _neg_38;
        asm("neg.f32 %0, %1;" : "=f"(_neg_38) : "f"(_abs_39));
        float _max3_35;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_35) : "f"(ma_0), "f"(_neg_36), "f"(_neg_38));
        ma_0 = _max3_35;
        float _neg_41;
        asm("neg.f32 %0, %1;" : "=f"(_neg_41) : "f"(zs_0[9]));
        float _neg_42;
        asm("neg.f32 %0, %1;" : "=f"(_neg_42) : "f"(zs_0[10]));
        float _max3_40;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_40) : "f"(mz_0), "f"(_neg_41), "f"(_neg_42));
        mz_0 = _max3_40;
        float _abs_45;
        asm("abs.f32 %0, %1;" : "=f"(_abs_45) : "f"(zs_0[9]));
        float _neg_44;
        asm("neg.f32 %0, %1;" : "=f"(_neg_44) : "f"(_abs_45));
        float _abs_47;
        asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(zs_0[10]));
        float _neg_46;
        asm("neg.f32 %0, %1;" : "=f"(_neg_46) : "f"(_abs_47));
        float _max3_43;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_43) : "f"(ma_0), "f"(_neg_44), "f"(_neg_46));
        ma_0 = _max3_43;
        float _neg_49;
        asm("neg.f32 %0, %1;" : "=f"(_neg_49) : "f"(zs_0[11]));
        float _neg_50;
        asm("neg.f32 %0, %1;" : "=f"(_neg_50) : "f"(zs_0[12]));
        float _max3_48;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_48) : "f"(mz_0), "f"(_neg_49), "f"(_neg_50));
        mz_0 = _max3_48;
        float _abs_53;
        asm("abs.f32 %0, %1;" : "=f"(_abs_53) : "f"(zs_0[11]));
        float _neg_52;
        asm("neg.f32 %0, %1;" : "=f"(_neg_52) : "f"(_abs_53));
        float _abs_55;
        asm("abs.f32 %0, %1;" : "=f"(_abs_55) : "f"(zs_0[12]));
        float _neg_54;
        asm("neg.f32 %0, %1;" : "=f"(_neg_54) : "f"(_abs_55));
        float _max3_51;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_51) : "f"(ma_0), "f"(_neg_52), "f"(_neg_54));
        ma_0 = _max3_51;
        float _neg_57;
        asm("neg.f32 %0, %1;" : "=f"(_neg_57) : "f"(zs_0[13]));
        float _neg_58;
        asm("neg.f32 %0, %1;" : "=f"(_neg_58) : "f"(zs_0[14]));
        float _max3_56;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_56) : "f"(mz_0), "f"(_neg_57), "f"(_neg_58));
        mz_0 = _max3_56;
        float _abs_61;
        asm("abs.f32 %0, %1;" : "=f"(_abs_61) : "f"(zs_0[13]));
        float _neg_60;
        asm("neg.f32 %0, %1;" : "=f"(_neg_60) : "f"(_abs_61));
        float _abs_63;
        asm("abs.f32 %0, %1;" : "=f"(_abs_63) : "f"(zs_0[14]));
        float _neg_62;
        asm("neg.f32 %0, %1;" : "=f"(_neg_62) : "f"(_abs_63));
        float _max3_59;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_59) : "f"(ma_0), "f"(_neg_60), "f"(_neg_62));
        ma_0 = _max3_59;
        float _neg_65;
        asm("neg.f32 %0, %1;" : "=f"(_neg_65) : "f"(zs_0[15]));
        float _max_64;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_64) : "f"(mz_0), "f"(_neg_65));
        mz_0 = _max_64;
        float _abs_68;
        asm("abs.f32 %0, %1;" : "=f"(_abs_68) : "f"(zs_0[15]));
        float _neg_67;
        asm("neg.f32 %0, %1;" : "=f"(_neg_67) : "f"(_abs_68));
        float _max_66;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_66) : "f"(ma_0), "f"(_neg_67));
        ma_0 = _max_66;
        float _abs_69;
        asm("abs.f32 %0, %1;" : "=f"(_abs_69) : "f"(zs_0[0]));
        float _abs_70;
        asm("abs.f32 %0, %1;" : "=f"(_abs_70) : "f"(zs_0[1]));
        float _abs_71;
        asm("abs.f32 %0, %1;" : "=f"(_abs_71) : "f"(zs_0[2]));
        float mp_0;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp_0) : "f"(_abs_69), "f"(_abs_70), "f"(_abs_71));
        float _abs_73;
        asm("abs.f32 %0, %1;" : "=f"(_abs_73) : "f"(zs_0[3]));
        float _abs_74;
        asm("abs.f32 %0, %1;" : "=f"(_abs_74) : "f"(zs_0[4]));
        float _max3_72;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_72) : "f"(mp_0), "f"(_abs_73), "f"(_abs_74));
        mp_0 = _max3_72;
        float _abs_76;
        asm("abs.f32 %0, %1;" : "=f"(_abs_76) : "f"(zs_0[5]));
        float _abs_77;
        asm("abs.f32 %0, %1;" : "=f"(_abs_77) : "f"(zs_0[6]));
        float _max3_75;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_75) : "f"(mp_0), "f"(_abs_76), "f"(_abs_77));
        mp_0 = _max3_75;
        float _abs_79;
        asm("abs.f32 %0, %1;" : "=f"(_abs_79) : "f"(zs_0[7]));
        float _abs_80;
        asm("abs.f32 %0, %1;" : "=f"(_abs_80) : "f"(zs_0[8]));
        float _max3_78;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_78) : "f"(mp_0), "f"(_abs_79), "f"(_abs_80));
        mp_0 = _max3_78;
        float _abs_82;
        asm("abs.f32 %0, %1;" : "=f"(_abs_82) : "f"(zs_0[9]));
        float _abs_83;
        asm("abs.f32 %0, %1;" : "=f"(_abs_83) : "f"(zs_0[10]));
        float _max3_81;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_81) : "f"(mp_0), "f"(_abs_82), "f"(_abs_83));
        mp_0 = _max3_81;
        float _abs_85;
        asm("abs.f32 %0, %1;" : "=f"(_abs_85) : "f"(zs_0[11]));
        float _abs_86;
        asm("abs.f32 %0, %1;" : "=f"(_abs_86) : "f"(zs_0[12]));
        float _max3_84;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_84) : "f"(mp_0), "f"(_abs_85), "f"(_abs_86));
        mp_0 = _max3_84;
        float _abs_88;
        asm("abs.f32 %0, %1;" : "=f"(_abs_88) : "f"(zs_0[13]));
        float _abs_89;
        asm("abs.f32 %0, %1;" : "=f"(_abs_89) : "f"(zs_0[14]));
        float _max3_87;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_87) : "f"(mp_0), "f"(_abs_88), "f"(_abs_89));
        mp_0 = _max3_87;
        float _abs_91;
        asm("abs.f32 %0, %1;" : "=f"(_abs_91) : "f"(zs_0[15]));
        float _max_90;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_90) : "f"(mp_0), "f"(_abs_91));
        mp_0 = _max_90;
        if (mz_0 <= 41.0f) {
            if (ma_0 <= -1.1754944e-38f) {
                if (mp_0 <= 3.4028235e+38f) {
                    fast_0 = 1;
                }
            }
        }
        if (fast_0 != 0) {
            float2 rstd2_0 = make_float2(rstd_0, rstd_0);
            float2 z2_0 = make_float2(zs_0[0], zs_0[1]);
            float ta_0;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_0) : "f"(zs_0[0]), "f"(-0.00572498f), "f"(0.5f));
            float tb_0;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_0) : "f"(zs_0[1]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_92;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(ta_0), "f"(252.0f), "f"(1.2582913e+07f));
            float _fma_93;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(tb_0), "f"(252.0f), "f"(1.2582913e+07f));
            float2 t2_0 = make_float2(_fma_92, _fma_93);
            float2 r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_0));
            r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.925963e-08f, -1.925963e-08f), r2_0);
            float2 nscale_0 = make_float2(__uint_as_float((__as_u32(t2_0.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_0.y) << 23) + 2147483648u));
            float _exp2_94 = approx_exp2(r2_0.x);
            float _exp2_95 = approx_exp2(r2_0.y);
            float2 nden_0 = fma_f32x2_rn_noftz(nscale_0, make_float2(_exp2_94, _exp2_95), make_float2(-1.0f, -1.0f));
            float _neg_97;
            asm("neg.f32 %0, %1;" : "=f"(_neg_97) : "f"(nden_0.x));
            float _rcp_96 = approx_rcp(_neg_97);
            float _neg_99;
            asm("neg.f32 %0, %1;" : "=f"(_neg_99) : "f"(nden_0.y));
            float _rcp_98 = approx_rcp(_neg_99);
            float2 r0_0 = make_float2(_rcp_96, _rcp_98);
            float2 r1_0 = fma_f32x2_rn_noftz(r0_0, fma_f32x2_rn_noftz(nden_0, r0_0, make_float2(1.0f, 1.0f)), r0_0);
            float2 q0_0 = fma_f32x2_rn_noftz(z2_0, r1_0, make_float2(0.0f, 0.0f));
            float2 gate2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, q0_0, z2_0), q0_0);
            float2 _f32x2_mul_a_100 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[0], xs_0[1]), rstd2_0), make_float2(wv[0], wv[1]));
            float2 y2_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&gate2_0));
            xs_0[0] = y2_0.x;
            xs_0[1] = y2_0.y;
            float2 z2_1 = make_float2(zs_0[2], zs_0[3]);
            float ta_1;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_1) : "f"(zs_0[2]), "f"(-0.00572498f), "f"(0.5f));
            float tb_1;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_1) : "f"(zs_0[3]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_101;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(ta_1), "f"(252.0f), "f"(1.2582913e+07f));
            float _fma_102;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(tb_1), "f"(252.0f), "f"(1.2582913e+07f));
            float2 t2_1 = make_float2(_fma_101, _fma_102);
            float2 r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_1));
            r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.925963e-08f, -1.925963e-08f), r2_1);
            float2 nscale_1 = make_float2(__uint_as_float((__as_u32(t2_1.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_1.y) << 23) + 2147483648u));
            float _exp2_103 = approx_exp2(r2_1.x);
            float _exp2_104 = approx_exp2(r2_1.y);
            float2 nden_1 = fma_f32x2_rn_noftz(nscale_1, make_float2(_exp2_103, _exp2_104), make_float2(-1.0f, -1.0f));
            float _neg_106;
            asm("neg.f32 %0, %1;" : "=f"(_neg_106) : "f"(nden_1.x));
            float _rcp_105 = approx_rcp(_neg_106);
            float _neg_108;
            asm("neg.f32 %0, %1;" : "=f"(_neg_108) : "f"(nden_1.y));
            float _rcp_107 = approx_rcp(_neg_108);
            float2 r0_1 = make_float2(_rcp_105, _rcp_107);
            float2 r1_1 = fma_f32x2_rn_noftz(r0_1, fma_f32x2_rn_noftz(nden_1, r0_1, make_float2(1.0f, 1.0f)), r0_1);
            float2 q0_1 = fma_f32x2_rn_noftz(z2_1, r1_1, make_float2(0.0f, 0.0f));
            float2 gate2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, q0_1, z2_1), q0_1);
            float2 _f32x2_mul_a_109 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[2], xs_0[3]), rstd2_0), make_float2(wv[2], wv[3]));
            float2 y2_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&gate2_1));
            xs_0[2] = y2_1.x;
            xs_0[3] = y2_1.y;
            float2 z2_2 = make_float2(zs_0[4], zs_0[5]);
            float ta_2;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_2) : "f"(zs_0[4]), "f"(-0.00572498f), "f"(0.5f));
            float tb_2;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_2) : "f"(zs_0[5]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_110;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(ta_2), "f"(252.0f), "f"(1.2582913e+07f));
            float _fma_111;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(tb_2), "f"(252.0f), "f"(1.2582913e+07f));
            float2 t2_2 = make_float2(_fma_110, _fma_111);
            float2 r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_2));
            r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.925963e-08f, -1.925963e-08f), r2_2);
            float2 nscale_2 = make_float2(__uint_as_float((__as_u32(t2_2.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_2.y) << 23) + 2147483648u));
            float _exp2_112 = approx_exp2(r2_2.x);
            float _exp2_113 = approx_exp2(r2_2.y);
            float2 nden_2 = fma_f32x2_rn_noftz(nscale_2, make_float2(_exp2_112, _exp2_113), make_float2(-1.0f, -1.0f));
            float _neg_115;
            asm("neg.f32 %0, %1;" : "=f"(_neg_115) : "f"(nden_2.x));
            float _rcp_114 = approx_rcp(_neg_115);
            float _neg_117;
            asm("neg.f32 %0, %1;" : "=f"(_neg_117) : "f"(nden_2.y));
            float _rcp_116 = approx_rcp(_neg_117);
            float2 r0_2 = make_float2(_rcp_114, _rcp_116);
            float2 r1_2 = fma_f32x2_rn_noftz(r0_2, fma_f32x2_rn_noftz(nden_2, r0_2, make_float2(1.0f, 1.0f)), r0_2);
            float2 q0_2 = fma_f32x2_rn_noftz(z2_2, r1_2, make_float2(0.0f, 0.0f));
            float2 gate2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, q0_2, z2_2), q0_2);
            float2 _f32x2_mul_a_118 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[4], xs_0[5]), rstd2_0), make_float2(wv[4], wv[5]));
            float2 y2_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_118), "l"(*(const unsigned long long*)&gate2_2));
            xs_0[4] = y2_2.x;
            xs_0[5] = y2_2.y;
            float2 z2_3 = make_float2(zs_0[6], zs_0[7]);
            float ta_3;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_3) : "f"(zs_0[6]), "f"(-0.00572498f), "f"(0.5f));
            float tb_3;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_3) : "f"(zs_0[7]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_119;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(ta_3), "f"(252.0f), "f"(1.2582913e+07f));
            float _fma_120;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(tb_3), "f"(252.0f), "f"(1.2582913e+07f));
            float2 t2_3 = make_float2(_fma_119, _fma_120);
            float2 r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_3));
            r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.925963e-08f, -1.925963e-08f), r2_3);
            float2 nscale_3 = make_float2(__uint_as_float((__as_u32(t2_3.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_3.y) << 23) + 2147483648u));
            float _exp2_121 = approx_exp2(r2_3.x);
            float _exp2_122 = approx_exp2(r2_3.y);
            float2 nden_3 = fma_f32x2_rn_noftz(nscale_3, make_float2(_exp2_121, _exp2_122), make_float2(-1.0f, -1.0f));
            float _neg_124;
            asm("neg.f32 %0, %1;" : "=f"(_neg_124) : "f"(nden_3.x));
            float _rcp_123 = approx_rcp(_neg_124);
            float _neg_126;
            asm("neg.f32 %0, %1;" : "=f"(_neg_126) : "f"(nden_3.y));
            float _rcp_125 = approx_rcp(_neg_126);
            float2 r0_3 = make_float2(_rcp_123, _rcp_125);
            float2 r1_3 = fma_f32x2_rn_noftz(r0_3, fma_f32x2_rn_noftz(nden_3, r0_3, make_float2(1.0f, 1.0f)), r0_3);
            float2 q0_3 = fma_f32x2_rn_noftz(z2_3, r1_3, make_float2(0.0f, 0.0f));
            float2 gate2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, q0_3, z2_3), q0_3);
            float2 _f32x2_mul_a_127 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[6], xs_0[7]), rstd2_0), make_float2(wv[6], wv[7]));
            float2 y2_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_127), "l"(*(const unsigned long long*)&gate2_3));
            xs_0[6] = y2_3.x;
            xs_0[7] = y2_3.y;
            float2 z2_4 = make_float2(zs_0[8], zs_0[9]);
            float ta_4;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_4) : "f"(zs_0[8]), "f"(-0.00572498f), "f"(0.5f));
            float tb_4;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_4) : "f"(zs_0[9]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_128;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(ta_4), "f"(252.0f), "f"(1.2582913e+07f));
            float _fma_129;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(tb_4), "f"(252.0f), "f"(1.2582913e+07f));
            float2 t2_4 = make_float2(_fma_128, _fma_129);
            float2 r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_4));
            r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.925963e-08f, -1.925963e-08f), r2_4);
            float2 nscale_4 = make_float2(__uint_as_float((__as_u32(t2_4.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_4.y) << 23) + 2147483648u));
            float _exp2_130 = approx_exp2(r2_4.x);
            float _exp2_131 = approx_exp2(r2_4.y);
            float2 nden_4 = fma_f32x2_rn_noftz(nscale_4, make_float2(_exp2_130, _exp2_131), make_float2(-1.0f, -1.0f));
            float _neg_133;
            asm("neg.f32 %0, %1;" : "=f"(_neg_133) : "f"(nden_4.x));
            float _rcp_132 = approx_rcp(_neg_133);
            float _neg_135;
            asm("neg.f32 %0, %1;" : "=f"(_neg_135) : "f"(nden_4.y));
            float _rcp_134 = approx_rcp(_neg_135);
            float2 r0_4 = make_float2(_rcp_132, _rcp_134);
            float2 r1_4 = fma_f32x2_rn_noftz(r0_4, fma_f32x2_rn_noftz(nden_4, r0_4, make_float2(1.0f, 1.0f)), r0_4);
            float2 q0_4 = fma_f32x2_rn_noftz(z2_4, r1_4, make_float2(0.0f, 0.0f));
            float2 gate2_4 = fma_f32x2_rn_noftz(r1_4, fma_f32x2_rn_noftz(nden_4, q0_4, z2_4), q0_4);
            float2 _f32x2_mul_a_136 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[8], xs_0[9]), rstd2_0), make_float2(wv[8], wv[9]));
            float2 y2_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&gate2_4));
            xs_0[8] = y2_4.x;
            xs_0[9] = y2_4.y;
            float2 z2_5 = make_float2(zs_0[10], zs_0[11]);
            float ta_5;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_5) : "f"(zs_0[10]), "f"(-0.00572498f), "f"(0.5f));
            float tb_5;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_5) : "f"(zs_0[11]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_137;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(ta_5), "f"(252.0f), "f"(1.2582913e+07f));
            float _fma_138;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(tb_5), "f"(252.0f), "f"(1.2582913e+07f));
            float2 t2_5 = make_float2(_fma_137, _fma_138);
            float2 r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_5));
            r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.925963e-08f, -1.925963e-08f), r2_5);
            float2 nscale_5 = make_float2(__uint_as_float((__as_u32(t2_5.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_5.y) << 23) + 2147483648u));
            float _exp2_139 = approx_exp2(r2_5.x);
            float _exp2_140 = approx_exp2(r2_5.y);
            float2 nden_5 = fma_f32x2_rn_noftz(nscale_5, make_float2(_exp2_139, _exp2_140), make_float2(-1.0f, -1.0f));
            float _neg_142;
            asm("neg.f32 %0, %1;" : "=f"(_neg_142) : "f"(nden_5.x));
            float _rcp_141 = approx_rcp(_neg_142);
            float _neg_144;
            asm("neg.f32 %0, %1;" : "=f"(_neg_144) : "f"(nden_5.y));
            float _rcp_143 = approx_rcp(_neg_144);
            float2 r0_5 = make_float2(_rcp_141, _rcp_143);
            float2 r1_5 = fma_f32x2_rn_noftz(r0_5, fma_f32x2_rn_noftz(nden_5, r0_5, make_float2(1.0f, 1.0f)), r0_5);
            float2 q0_5 = fma_f32x2_rn_noftz(z2_5, r1_5, make_float2(0.0f, 0.0f));
            float2 gate2_5 = fma_f32x2_rn_noftz(r1_5, fma_f32x2_rn_noftz(nden_5, q0_5, z2_5), q0_5);
            float2 _f32x2_mul_a_145 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[10], xs_0[11]), rstd2_0), make_float2(wv[10], wv[11]));
            float2 y2_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_145), "l"(*(const unsigned long long*)&gate2_5));
            xs_0[10] = y2_5.x;
            xs_0[11] = y2_5.y;
            float2 z2_6 = make_float2(zs_0[12], zs_0[13]);
            float ta_6;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_6) : "f"(zs_0[12]), "f"(-0.00572498f), "f"(0.5f));
            float tb_6;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_6) : "f"(zs_0[13]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_146;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(ta_6), "f"(252.0f), "f"(1.2582913e+07f));
            float _fma_147;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(tb_6), "f"(252.0f), "f"(1.2582913e+07f));
            float2 t2_6 = make_float2(_fma_146, _fma_147);
            float2 r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_6));
            r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.925963e-08f, -1.925963e-08f), r2_6);
            float2 nscale_6 = make_float2(__uint_as_float((__as_u32(t2_6.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_6.y) << 23) + 2147483648u));
            float _exp2_148 = approx_exp2(r2_6.x);
            float _exp2_149 = approx_exp2(r2_6.y);
            float2 nden_6 = fma_f32x2_rn_noftz(nscale_6, make_float2(_exp2_148, _exp2_149), make_float2(-1.0f, -1.0f));
            float _neg_151;
            asm("neg.f32 %0, %1;" : "=f"(_neg_151) : "f"(nden_6.x));
            float _rcp_150 = approx_rcp(_neg_151);
            float _neg_153;
            asm("neg.f32 %0, %1;" : "=f"(_neg_153) : "f"(nden_6.y));
            float _rcp_152 = approx_rcp(_neg_153);
            float2 r0_6 = make_float2(_rcp_150, _rcp_152);
            float2 r1_6 = fma_f32x2_rn_noftz(r0_6, fma_f32x2_rn_noftz(nden_6, r0_6, make_float2(1.0f, 1.0f)), r0_6);
            float2 q0_6 = fma_f32x2_rn_noftz(z2_6, r1_6, make_float2(0.0f, 0.0f));
            float2 gate2_6 = fma_f32x2_rn_noftz(r1_6, fma_f32x2_rn_noftz(nden_6, q0_6, z2_6), q0_6);
            float2 _f32x2_mul_a_154 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[12], xs_0[13]), rstd2_0), make_float2(wv[12], wv[13]));
            float2 y2_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&gate2_6));
            xs_0[12] = y2_6.x;
            xs_0[13] = y2_6.y;
            float2 z2_7 = make_float2(zs_0[14], zs_0[15]);
            float ta_7;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_7) : "f"(zs_0[14]), "f"(-0.00572498f), "f"(0.5f));
            float tb_7;
            asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_7) : "f"(zs_0[15]), "f"(-0.00572498f), "f"(0.5f));
            float _fma_155;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(ta_7), "f"(252.0f), "f"(1.2582913e+07f));
            float _fma_156;
            asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(tb_7), "f"(252.0f), "f"(1.2582913e+07f));
            float2 t2_7 = make_float2(_fma_155, _fma_156);
            float2 r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_7));
            r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.925963e-08f, -1.925963e-08f), r2_7);
            float2 nscale_7 = make_float2(__uint_as_float((__as_u32(t2_7.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_7.y) << 23) + 2147483648u));
            float _exp2_157 = approx_exp2(r2_7.x);
            float _exp2_158 = approx_exp2(r2_7.y);
            float2 nden_7 = fma_f32x2_rn_noftz(nscale_7, make_float2(_exp2_157, _exp2_158), make_float2(-1.0f, -1.0f));
            float _neg_160;
            asm("neg.f32 %0, %1;" : "=f"(_neg_160) : "f"(nden_7.x));
            float _rcp_159 = approx_rcp(_neg_160);
            float _neg_162;
            asm("neg.f32 %0, %1;" : "=f"(_neg_162) : "f"(nden_7.y));
            float _rcp_161 = approx_rcp(_neg_162);
            float2 r0_7 = make_float2(_rcp_159, _rcp_161);
            float2 r1_7 = fma_f32x2_rn_noftz(r0_7, fma_f32x2_rn_noftz(nden_7, r0_7, make_float2(1.0f, 1.0f)), r0_7);
            float2 q0_7 = fma_f32x2_rn_noftz(z2_7, r1_7, make_float2(0.0f, 0.0f));
            float2 gate2_7 = fma_f32x2_rn_noftz(r1_7, fma_f32x2_rn_noftz(nden_7, q0_7, z2_7), q0_7);
            float2 _f32x2_mul_a_163 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[14], xs_0[15]), rstd2_0), make_float2(wv[14], wv[15]));
            float2 y2_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_163), "l"(*(const unsigned long long*)&gate2_7));
            xs_0[14] = y2_7.x;
            xs_0[15] = y2_7.y;
        } else {
            float _exp_164 = __nv_expf(zs_0[0] * -1.0f);
            float gate_0 = __fdiv_rn(zs_0[0], 1.0f + _exp_164);
            xs_0[0] = xs_0[0] * rstd_0 * wv[0] * gate_0;
            float _exp_165 = __nv_expf(zs_0[1] * -1.0f);
            float gate_1 = __fdiv_rn(zs_0[1], 1.0f + _exp_165);
            xs_0[1] = xs_0[1] * rstd_0 * wv[1] * gate_1;
            float _exp_166 = __nv_expf(zs_0[2] * -1.0f);
            float gate_2 = __fdiv_rn(zs_0[2], 1.0f + _exp_166);
            xs_0[2] = xs_0[2] * rstd_0 * wv[2] * gate_2;
            float _exp_167 = __nv_expf(zs_0[3] * -1.0f);
            float gate_3 = __fdiv_rn(zs_0[3], 1.0f + _exp_167);
            xs_0[3] = xs_0[3] * rstd_0 * wv[3] * gate_3;
            float _exp_168 = __nv_expf(zs_0[4] * -1.0f);
            float gate_4 = __fdiv_rn(zs_0[4], 1.0f + _exp_168);
            xs_0[4] = xs_0[4] * rstd_0 * wv[4] * gate_4;
            float _exp_169 = __nv_expf(zs_0[5] * -1.0f);
            float gate_5 = __fdiv_rn(zs_0[5], 1.0f + _exp_169);
            xs_0[5] = xs_0[5] * rstd_0 * wv[5] * gate_5;
            float _exp_170 = __nv_expf(zs_0[6] * -1.0f);
            float gate_6 = __fdiv_rn(zs_0[6], 1.0f + _exp_170);
            xs_0[6] = xs_0[6] * rstd_0 * wv[6] * gate_6;
            float _exp_171 = __nv_expf(zs_0[7] * -1.0f);
            float gate_7 = __fdiv_rn(zs_0[7], 1.0f + _exp_171);
            xs_0[7] = xs_0[7] * rstd_0 * wv[7] * gate_7;
            float _exp_172 = __nv_expf(zs_0[8] * -1.0f);
            float gate_8 = __fdiv_rn(zs_0[8], 1.0f + _exp_172);
            xs_0[8] = xs_0[8] * rstd_0 * wv[8] * gate_8;
            float _exp_173 = __nv_expf(zs_0[9] * -1.0f);
            float gate_9 = __fdiv_rn(zs_0[9], 1.0f + _exp_173);
            xs_0[9] = xs_0[9] * rstd_0 * wv[9] * gate_9;
            float _exp_174 = __nv_expf(zs_0[10] * -1.0f);
            float gate_10 = __fdiv_rn(zs_0[10], 1.0f + _exp_174);
            xs_0[10] = xs_0[10] * rstd_0 * wv[10] * gate_10;
            float _exp_175 = __nv_expf(zs_0[11] * -1.0f);
            float gate_11 = __fdiv_rn(zs_0[11], 1.0f + _exp_175);
            xs_0[11] = xs_0[11] * rstd_0 * wv[11] * gate_11;
            float _exp_176 = __nv_expf(zs_0[12] * -1.0f);
            float gate_12 = __fdiv_rn(zs_0[12], 1.0f + _exp_176);
            xs_0[12] = xs_0[12] * rstd_0 * wv[12] * gate_12;
            float _exp_177 = __nv_expf(zs_0[13] * -1.0f);
            float gate_13 = __fdiv_rn(zs_0[13], 1.0f + _exp_177);
            xs_0[13] = xs_0[13] * rstd_0 * wv[13] * gate_13;
            float _exp_178 = __nv_expf(zs_0[14] * -1.0f);
            float gate_14 = __fdiv_rn(zs_0[14], 1.0f + _exp_178);
            xs_0[14] = xs_0[14] * rstd_0 * wv[14] * gate_14;
            float _exp_179 = __nv_expf(zs_0[15] * -1.0f);
            float gate_15 = __fdiv_rn(zs_0[15], 1.0f + _exp_179);
            xs_0[15] = xs_0[15] * rstd_0 * wv[15] * gate_15;
        }
        unsigned int packed_0[8];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 0], xs_0[_lp*2+1 + 0]));
            packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 2], xs_0[_lp*2+1 + 2]));
            packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 4], xs_0[_lp*2+1 + 4]));
            packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 6], xs_0[_lp*2+1 + 6]));
            packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 8], xs_0[_lp*2+1 + 8]));
            packed_0[_lp + 4] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 10], xs_0[_lp*2+1 + 10]));
            packed_0[_lp + 5] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 12], xs_0[_lp*2+1 + 12]));
            packed_0[_lp + 6] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 14], xs_0[_lp*2+1 + 14]));
            packed_0[_lp + 7] = *(uint32_t*)&_bf2;
        }
        if (row_0 < rows) {
            reinterpret_cast<int4*>(y + ((long long)row_0 * 128 + (long long)(col + 0)))[0] = reinterpret_cast<int4*>(packed_0)[0];
            reinterpret_cast<int4*>(y + ((long long)row_0 * 128 + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed_0 + (4))[0];
        }
    }
}

} // extern "C"
