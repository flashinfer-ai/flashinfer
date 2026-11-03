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
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 4

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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_train_55ff8a9d0ba6d83285b5(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, CakeFastDivmod heads, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int group = tid >> 4;
    int col = (tid & 15) * 8;
    float wv[8];
    {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + col + 0);
        uint4 _vld_0[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_0[_blk] = _vptr_0[_blk];
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&wv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                (&wv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
            }
        }
    }
    int nblocks = (rows + 15) / 16;
    unsigned int xw[4];
    unsigned int zw[4];
    int src = ((bid * 16 + group) < (rows - 1) ? (bid * 16 + group) : (rows - 1));
    uint32_t tq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src);
    uint32_t th = (uint32_t)((unsigned int)src) - tq * (uint32_t)(heads.divisor);
    long long xoff = (long long)tq * x_tok + (long long)th * x_head + (long long)col;
    long long zoff = (long long)tq * z_tok + (long long)th * z_head + (long long)col;
    {
        uint4 _uv4_1 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(x + xoff) + 0);
        xw[0 + 0] = _uv4_1.x;
        xw[0 + 1] = _uv4_1.y;
        xw[0 + 2] = _uv4_1.z;
        xw[0 + 3] = _uv4_1.w;
    }
    {
        uint4 _uv4_2 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(z + zoff) + 0);
        zw[0 + 0] = _uv4_2.x;
        zw[0 + 1] = _uv4_2.y;
        zw[0 + 2] = _uv4_2.z;
        zw[0 + 3] = _uv4_2.w;
    }
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
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
        if (blk + num_bids < nblocks) {
            int src_0 = (((blk + num_bids) * 16 + group) < (rows - 1) ? ((blk + num_bids) * 16 + group) : (rows - 1));
            uint32_t tq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_0);
            uint32_t th_0 = (uint32_t)((unsigned int)src_0) - tq_0 * (uint32_t)(heads.divisor);
            long long xoff_0 = (long long)tq_0 * x_tok + (long long)th_0 * x_head + (long long)col;
            long long zoff_0 = (long long)tq_0 * z_tok + (long long)th_0 * z_head + (long long)col;
            {
                uint4 _uv4_3 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(x + xoff_0) + 0);
                xw[0 + 0] = _uv4_3.x;
                xw[0 + 1] = _uv4_3.y;
                xw[0 + 2] = _uv4_3.z;
                xw[0 + 3] = _uv4_3.w;
            }
            {
                uint4 _uv4_4 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(z + zoff_0) + 0);
                zw[0 + 0] = _uv4_4.x;
                zw[0 + 1] = _uv4_4.y;
                zw[0 + 2] = _uv4_4.z;
                zw[0 + 3] = _uv4_4.w;
            }
        }
        int fast = 0;
        float _neg_5;
        asm("neg.f32 %0, %1;" : "=f"(_neg_5) : "f"(zs[0]));
        float _neg_6;
        asm("neg.f32 %0, %1;" : "=f"(_neg_6) : "f"(zs[1]));
        float _neg_7;
        asm("neg.f32 %0, %1;" : "=f"(_neg_7) : "f"(zs[2]));
        float mz;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz) : "f"(_neg_5), "f"(_neg_6), "f"(_neg_7));
        float _abs_9;
        asm("abs.f32 %0, %1;" : "=f"(_abs_9) : "f"(zs[0]));
        float _neg_8;
        asm("neg.f32 %0, %1;" : "=f"(_neg_8) : "f"(_abs_9));
        float _abs_11;
        asm("abs.f32 %0, %1;" : "=f"(_abs_11) : "f"(zs[1]));
        float _neg_10;
        asm("neg.f32 %0, %1;" : "=f"(_neg_10) : "f"(_abs_11));
        float _abs_13;
        asm("abs.f32 %0, %1;" : "=f"(_abs_13) : "f"(zs[2]));
        float _neg_12;
        asm("neg.f32 %0, %1;" : "=f"(_neg_12) : "f"(_abs_13));
        float ma;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma) : "f"(_neg_8), "f"(_neg_10), "f"(_neg_12));
        float _neg_15;
        asm("neg.f32 %0, %1;" : "=f"(_neg_15) : "f"(zs[3]));
        float _neg_16;
        asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(zs[4]));
        float _max3_14;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_14) : "f"(mz), "f"(_neg_15), "f"(_neg_16));
        mz = _max3_14;
        float _abs_19;
        asm("abs.f32 %0, %1;" : "=f"(_abs_19) : "f"(zs[3]));
        float _neg_18;
        asm("neg.f32 %0, %1;" : "=f"(_neg_18) : "f"(_abs_19));
        float _abs_21;
        asm("abs.f32 %0, %1;" : "=f"(_abs_21) : "f"(zs[4]));
        float _neg_20;
        asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(_abs_21));
        float _max3_17;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_17) : "f"(ma), "f"(_neg_18), "f"(_neg_20));
        ma = _max3_17;
        float _neg_23;
        asm("neg.f32 %0, %1;" : "=f"(_neg_23) : "f"(zs[5]));
        float _neg_24;
        asm("neg.f32 %0, %1;" : "=f"(_neg_24) : "f"(zs[6]));
        float _max3_22;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_22) : "f"(mz), "f"(_neg_23), "f"(_neg_24));
        mz = _max3_22;
        float _abs_27;
        asm("abs.f32 %0, %1;" : "=f"(_abs_27) : "f"(zs[5]));
        float _neg_26;
        asm("neg.f32 %0, %1;" : "=f"(_neg_26) : "f"(_abs_27));
        float _abs_29;
        asm("abs.f32 %0, %1;" : "=f"(_abs_29) : "f"(zs[6]));
        float _neg_28;
        asm("neg.f32 %0, %1;" : "=f"(_neg_28) : "f"(_abs_29));
        float _max3_25;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_25) : "f"(ma), "f"(_neg_26), "f"(_neg_28));
        ma = _max3_25;
        float _neg_31;
        asm("neg.f32 %0, %1;" : "=f"(_neg_31) : "f"(zs[7]));
        float _max_30;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_30) : "f"(mz), "f"(_neg_31));
        mz = _max_30;
        float _abs_34;
        asm("abs.f32 %0, %1;" : "=f"(_abs_34) : "f"(zs[7]));
        float _neg_33;
        asm("neg.f32 %0, %1;" : "=f"(_neg_33) : "f"(_abs_34));
        float _max_32;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_32) : "f"(ma), "f"(_neg_33));
        ma = _max_32;
        float _abs_35;
        asm("abs.f32 %0, %1;" : "=f"(_abs_35) : "f"(zs[0]));
        float _abs_36;
        asm("abs.f32 %0, %1;" : "=f"(_abs_36) : "f"(zs[1]));
        float _abs_37;
        asm("abs.f32 %0, %1;" : "=f"(_abs_37) : "f"(zs[2]));
        float mp;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp) : "f"(_abs_35), "f"(_abs_36), "f"(_abs_37));
        float _abs_39;
        asm("abs.f32 %0, %1;" : "=f"(_abs_39) : "f"(zs[3]));
        float _abs_40;
        asm("abs.f32 %0, %1;" : "=f"(_abs_40) : "f"(zs[4]));
        float _max3_38;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_38) : "f"(mp), "f"(_abs_39), "f"(_abs_40));
        mp = _max3_38;
        float _abs_42;
        asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(zs[5]));
        float _abs_43;
        asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(zs[6]));
        float _max3_41;
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
        #error "Max3Nan requires PTX three-input max.f32 support on sm_100+"
        #endif
        asm("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_41) : "f"(mp), "f"(_abs_42), "f"(_abs_43));
        mp = _max3_41;
        float _abs_45;
        asm("abs.f32 %0, %1;" : "=f"(_abs_45) : "f"(zs[7]));
        float _max_44;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_44) : "f"(mp), "f"(_abs_45));
        mp = _max_44;
        if (mz <= 41.0f) {
            if (ma <= -1.1754944e-38f) {
                if (mp <= 3.4028235e+38f) {
                    fast = 1;
                }
            }
        }
        float gs[8];
        float2 z2_0 = make_float2(zs[0], zs[1]);
        float ta_0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_0) : "f"(zs[0]), "f"(-0.00572498f), "f"(0.5f));
        float tb_0;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_0) : "f"(zs[1]), "f"(-0.00572498f), "f"(0.5f));
        float _fma_46;
        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(ta_0), "f"(252.0f), "f"(1.2583169e+07f));
        float _fma_47;
        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(tb_0), "f"(252.0f), "f"(1.2583169e+07f));
        float2 t2_0 = make_float2(_fma_46, _fma_47);
        float2 r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_0));
        r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.925963e-08f, -1.925963e-08f), r2_0);
        float2 nscale_0 = make_float2(__uint_as_float(__as_u32(t2_0.x) << 23), __uint_as_float(__as_u32(t2_0.y) << 23));
        float _exp2_48 = approx_exp2(r2_0.x);
        float _exp2_49 = approx_exp2(r2_0.y);
        float2 nden_0 = fma_f32x2_rn_noftz(nscale_0, make_float2(_exp2_48, _exp2_49), make_float2(-1.0f, -1.0f));
        float _neg_51;
        asm("neg.f32 %0, %1;" : "=f"(_neg_51) : "f"(nden_0.x));
        float _rcp_50 = approx_rcp(_neg_51);
        float _neg_53;
        asm("neg.f32 %0, %1;" : "=f"(_neg_53) : "f"(nden_0.y));
        float _rcp_52 = approx_rcp(_neg_53);
        float2 r0_0 = make_float2(_rcp_50, _rcp_52);
        float2 r1_0 = fma_f32x2_rn_noftz(r0_0, fma_f32x2_rn_noftz(nden_0, r0_0, make_float2(1.0f, 1.0f)), r0_0);
        float2 q0_0 = fma_f32x2_rn_noftz(z2_0, r1_0, make_float2(0.0f, 0.0f));
        float2 gate2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, q0_0, z2_0), q0_0);
        gs[0] = gate2_0.x;
        gs[1] = gate2_0.y;
        float2 z2_1 = make_float2(zs[2], zs[3]);
        float ta_1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_1) : "f"(zs[2]), "f"(-0.00572498f), "f"(0.5f));
        float tb_1;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_1) : "f"(zs[3]), "f"(-0.00572498f), "f"(0.5f));
        float _fma_54;
        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(ta_1), "f"(252.0f), "f"(1.2583169e+07f));
        float _fma_55;
        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(tb_1), "f"(252.0f), "f"(1.2583169e+07f));
        float2 t2_1 = make_float2(_fma_54, _fma_55);
        float2 r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_1));
        r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.925963e-08f, -1.925963e-08f), r2_1);
        float2 nscale_1 = make_float2(__uint_as_float(__as_u32(t2_1.x) << 23), __uint_as_float(__as_u32(t2_1.y) << 23));
        float _exp2_56 = approx_exp2(r2_1.x);
        float _exp2_57 = approx_exp2(r2_1.y);
        float2 nden_1 = fma_f32x2_rn_noftz(nscale_1, make_float2(_exp2_56, _exp2_57), make_float2(-1.0f, -1.0f));
        float _neg_59;
        asm("neg.f32 %0, %1;" : "=f"(_neg_59) : "f"(nden_1.x));
        float _rcp_58 = approx_rcp(_neg_59);
        float _neg_61;
        asm("neg.f32 %0, %1;" : "=f"(_neg_61) : "f"(nden_1.y));
        float _rcp_60 = approx_rcp(_neg_61);
        float2 r0_1 = make_float2(_rcp_58, _rcp_60);
        float2 r1_1 = fma_f32x2_rn_noftz(r0_1, fma_f32x2_rn_noftz(nden_1, r0_1, make_float2(1.0f, 1.0f)), r0_1);
        float2 q0_1 = fma_f32x2_rn_noftz(z2_1, r1_1, make_float2(0.0f, 0.0f));
        float2 gate2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, q0_1, z2_1), q0_1);
        gs[2] = gate2_1.x;
        gs[3] = gate2_1.y;
        float2 z2_2 = make_float2(zs[4], zs[5]);
        float ta_2;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_2) : "f"(zs[4]), "f"(-0.00572498f), "f"(0.5f));
        float tb_2;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_2) : "f"(zs[5]), "f"(-0.00572498f), "f"(0.5f));
        float _fma_62;
        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(ta_2), "f"(252.0f), "f"(1.2583169e+07f));
        float _fma_63;
        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(tb_2), "f"(252.0f), "f"(1.2583169e+07f));
        float2 t2_2 = make_float2(_fma_62, _fma_63);
        float2 r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_2));
        r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.925963e-08f, -1.925963e-08f), r2_2);
        float2 nscale_2 = make_float2(__uint_as_float(__as_u32(t2_2.x) << 23), __uint_as_float(__as_u32(t2_2.y) << 23));
        float _exp2_64 = approx_exp2(r2_2.x);
        float _exp2_65 = approx_exp2(r2_2.y);
        float2 nden_2 = fma_f32x2_rn_noftz(nscale_2, make_float2(_exp2_64, _exp2_65), make_float2(-1.0f, -1.0f));
        float _neg_67;
        asm("neg.f32 %0, %1;" : "=f"(_neg_67) : "f"(nden_2.x));
        float _rcp_66 = approx_rcp(_neg_67);
        float _neg_69;
        asm("neg.f32 %0, %1;" : "=f"(_neg_69) : "f"(nden_2.y));
        float _rcp_68 = approx_rcp(_neg_69);
        float2 r0_2 = make_float2(_rcp_66, _rcp_68);
        float2 r1_2 = fma_f32x2_rn_noftz(r0_2, fma_f32x2_rn_noftz(nden_2, r0_2, make_float2(1.0f, 1.0f)), r0_2);
        float2 q0_2 = fma_f32x2_rn_noftz(z2_2, r1_2, make_float2(0.0f, 0.0f));
        float2 gate2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, q0_2, z2_2), q0_2);
        gs[4] = gate2_2.x;
        gs[5] = gate2_2.y;
        float2 z2_3 = make_float2(zs[6], zs[7]);
        float ta_3;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_3) : "f"(zs[6]), "f"(-0.00572498f), "f"(0.5f));
        float tb_3;
        asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_3) : "f"(zs[7]), "f"(-0.00572498f), "f"(0.5f));
        float _fma_70;
        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(ta_3), "f"(252.0f), "f"(1.2583169e+07f));
        float _fma_71;
        asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(tb_3), "f"(252.0f), "f"(1.2583169e+07f));
        float2 t2_3 = make_float2(_fma_70, _fma_71);
        float2 r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583295e+07f, 1.2583295e+07f), t2_3));
        r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.925963e-08f, -1.925963e-08f), r2_3);
        float2 nscale_3 = make_float2(__uint_as_float(__as_u32(t2_3.x) << 23), __uint_as_float(__as_u32(t2_3.y) << 23));
        float _exp2_72 = approx_exp2(r2_3.x);
        float _exp2_73 = approx_exp2(r2_3.y);
        float2 nden_3 = fma_f32x2_rn_noftz(nscale_3, make_float2(_exp2_72, _exp2_73), make_float2(-1.0f, -1.0f));
        float _neg_75;
        asm("neg.f32 %0, %1;" : "=f"(_neg_75) : "f"(nden_3.x));
        float _rcp_74 = approx_rcp(_neg_75);
        float _neg_77;
        asm("neg.f32 %0, %1;" : "=f"(_neg_77) : "f"(nden_3.y));
        float _rcp_76 = approx_rcp(_neg_77);
        float2 r0_3 = make_float2(_rcp_74, _rcp_76);
        float2 r1_3 = fma_f32x2_rn_noftz(r0_3, fma_f32x2_rn_noftz(nden_3, r0_3, make_float2(1.0f, 1.0f)), r0_3);
        float2 q0_3 = fma_f32x2_rn_noftz(z2_3, r1_3, make_float2(0.0f, 0.0f));
        float2 gate2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, q0_3, z2_3), q0_3);
        gs[6] = gate2_3.x;
        gs[7] = gate2_3.y;
        float total = 0.0f;
        float half = 0.0f;
        half = __fmaf_rn(xs[0], xs[0], half);
        half = __fmaf_rn(xs[1], xs[1], half);
        half = __fmaf_rn(xs[2], xs[2], half);
        half = __fmaf_rn(xs[3], xs[3], half);
        half = __fmaf_rn(xs[4], xs[4], half);
        half = __fmaf_rn(xs[5], xs[5], half);
        half = __fmaf_rn(xs[6], xs[6], half);
        half = __fmaf_rn(xs[7], xs[7], half);
        float chain = __shfl_xor_sync(0xFFFFFFFF, half, 1);
        chain = __fmaf_rn(xs[0], xs[0], chain);
        chain = __fmaf_rn(xs[1], xs[1], chain);
        chain = __fmaf_rn(xs[2], xs[2], chain);
        chain = __fmaf_rn(xs[3], xs[3], chain);
        chain = __fmaf_rn(xs[4], xs[4], chain);
        chain = __fmaf_rn(xs[5], xs[5], chain);
        chain = __fmaf_rn(xs[6], xs[6], chain);
        chain = __fmaf_rn(xs[7], xs[7], chain);
        float other = __shfl_xor_sync(0xFFFFFFFF, chain, 1);
        total = (((lane & 1) != 0) ? chain : other);
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        total = total + _shfl_xor_sync_78;
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
        total = total + _shfl_xor_sync_79;
        float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
        total = total + _shfl_xor_sync_80;
        float _sqrt_81;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_81) : "f"(total * 0.0078125f + eps));
        float rstd = __frcp_rn(_sqrt_81);
        if (fast == 0) {
            float _exp_83 = __nv_expf(zs[0] * -1.0f);
            float _fdiv_82 = __fdiv_rn(zs[0], 1.0f + _exp_83);
            gs[0] = _fdiv_82;
            float _exp_85 = __nv_expf(zs[1] * -1.0f);
            float _fdiv_84 = __fdiv_rn(zs[1], 1.0f + _exp_85);
            gs[1] = _fdiv_84;
            float _exp_87 = __nv_expf(zs[2] * -1.0f);
            float _fdiv_86 = __fdiv_rn(zs[2], 1.0f + _exp_87);
            gs[2] = _fdiv_86;
            float _exp_89 = __nv_expf(zs[3] * -1.0f);
            float _fdiv_88 = __fdiv_rn(zs[3], 1.0f + _exp_89);
            gs[3] = _fdiv_88;
            float _exp_91 = __nv_expf(zs[4] * -1.0f);
            float _fdiv_90 = __fdiv_rn(zs[4], 1.0f + _exp_91);
            gs[4] = _fdiv_90;
            float _exp_93 = __nv_expf(zs[5] * -1.0f);
            float _fdiv_92 = __fdiv_rn(zs[5], 1.0f + _exp_93);
            gs[5] = _fdiv_92;
            float _exp_95 = __nv_expf(zs[6] * -1.0f);
            float _fdiv_94 = __fdiv_rn(zs[6], 1.0f + _exp_95);
            gs[6] = _fdiv_94;
            float _exp_97 = __nv_expf(zs[7] * -1.0f);
            float _fdiv_96 = __fdiv_rn(zs[7], 1.0f + _exp_97);
            gs[7] = _fdiv_96;
        }
        float2 rstd2 = make_float2(rstd, rstd);
        float2 _f32x2_mul_a_98 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[0], xs[1]), rstd2), make_float2(wv[0], wv[1]));
        float2 _f32x2_mul_b_99 = make_float2(gs[0], gs[1]);
        float2 y2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_98), "l"(*(const unsigned long long*)&_f32x2_mul_b_99));
        xs[0] = y2_0.x;
        xs[1] = y2_0.y;
        float2 _f32x2_mul_a_100 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[2], xs[3]), rstd2), make_float2(wv[2], wv[3]));
        float2 _f32x2_mul_b_101 = make_float2(gs[2], gs[3]);
        float2 y2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_101));
        xs[2] = y2_1.x;
        xs[3] = y2_1.y;
        float2 _f32x2_mul_a_102 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[4], xs[5]), rstd2), make_float2(wv[4], wv[5]));
        float2 _f32x2_mul_b_103 = make_float2(gs[4], gs[5]);
        float2 y2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_102), "l"(*(const unsigned long long*)&_f32x2_mul_b_103));
        xs[4] = y2_2.x;
        xs[5] = y2_2.y;
        float2 _f32x2_mul_a_104 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs[6], xs[7]), rstd2), make_float2(wv[6], wv[7]));
        float2 _f32x2_mul_b_105 = make_float2(gs[6], gs[7]);
        float2 y2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_105));
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
        if (blk * 16 + group < rows) {
            reinterpret_cast<int4*>(y + ((long long)(blk * 16 + group) * 128 + (long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
        }
    }

    // === Task calls (declaration order) ===
}

} // extern "C"
