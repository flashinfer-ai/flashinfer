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

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2_noftz(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
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

__global__ __launch_bounds__(64) void
kernel_cake_rmsnorm_silu_gate_train_c7919260b39b69e4e646(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, int rows, CakeFastDivmod heads, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps)
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
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (col + 8) + 0);
        uint4 _vld_1[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_1[_blk] = _vptr_1[_blk];
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&wv[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&wv[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
    }
    int nblocks = (rows + 7) / 8;
    #pragma unroll 1
    for (int blk = bid; blk < nblocks; blk += num_bids) {
        int row_0 = blk * 8 + group;
        int src_0 = ((row_0) < (rows - 1) ? (row_0) : (rows - 1));
        uint32_t tq_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src_0), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src_0);
        uint32_t th_0 = (uint32_t)((unsigned int)src_0) - tq_0 * (uint32_t)(heads.divisor);
        long long xoff_0 = (long long)tq_0 * x_tok + (long long)th_0 * x_head + (long long)col;
        long long zoff_0 = (long long)tq_0 * z_tok + (long long)th_0 * z_head + (long long)col;
        float xs_0[16];
        float zs_0[16];
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x + xoff_0 + 0);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xs_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                    (&xs_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(z + zoff_0 + 0);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&zs_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                    (&zs_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(x + (xoff_0 + 8ll) + 0);
            uint4 _vld_4[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_4[_blk] = _vptr_4[_blk];
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&xs_0[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&xs_0[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(z + (zoff_0 + 8ll) + 0);
            uint4 _vld_5[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_5[_blk] = _vptr_5[_blk];
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&zs_0[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                    (&zs_0[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                }
            }
        }
        float2 z2_0 = make_float2(zs_0[0], zs_0[1]);
        float _abs_6;
        asm("abs.f32 %0, %1;" : "=f"(_abs_6) : "f"(z2_0.x));
        float _abs_7;
        asm("abs.f32 %0, %1;" : "=f"(_abs_7) : "f"(z2_0.y));
        float2 _f32x2_mul_a_8 = make_float2(_abs_6, _abs_7);
        float2 _f32x2_mul_b_9 = make_float2(-1.442695f, -1.442695f);
        float2 t2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
        float _exp2_10 = approx_exp2_noftz(t2_0.x);
        float _exp2_11 = approx_exp2_noftz(t2_0.y);
        float2 e2_0 = make_float2(_exp2_10, _exp2_11);
        float2 d2_0 = fma_f32x2_rn_noftz(e2_0, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_12 = approx_rcp(d2_0.x);
        float _rcp_13 = approx_rcp(d2_0.y);
        float2 r2_0 = make_float2(_rcp_12, _rcp_13);
        float2 er2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&er2_0) : "l"(*(const unsigned long long*)&e2_0), "l"(*(const unsigned long long*)&r2_0));
        float2 s2_0 = make_float2(((z2_0.x >= 0.0f) ? r2_0.x : er2_0.x), ((z2_0.y >= 0.0f) ? r2_0.y : er2_0.y));
        float2 g2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_0) : "l"(*(const unsigned long long*)&z2_0), "l"(*(const unsigned long long*)&s2_0));
        zs_0[0] = g2_0.x;
        zs_0[1] = g2_0.y;
        float2 z2_1 = make_float2(zs_0[2], zs_0[3]);
        float _abs_14;
        asm("abs.f32 %0, %1;" : "=f"(_abs_14) : "f"(z2_1.x));
        float _abs_15;
        asm("abs.f32 %0, %1;" : "=f"(_abs_15) : "f"(z2_1.y));
        float2 _f32x2_mul_a_16 = make_float2(_abs_14, _abs_15);
        float2 _f32x2_mul_b_17 = make_float2(-1.442695f, -1.442695f);
        float2 t2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
        float _exp2_18 = approx_exp2_noftz(t2_1.x);
        float _exp2_19 = approx_exp2_noftz(t2_1.y);
        float2 e2_1 = make_float2(_exp2_18, _exp2_19);
        float2 d2_1 = fma_f32x2_rn_noftz(e2_1, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_20 = approx_rcp(d2_1.x);
        float _rcp_21 = approx_rcp(d2_1.y);
        float2 r2_1 = make_float2(_rcp_20, _rcp_21);
        float2 er2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&er2_1) : "l"(*(const unsigned long long*)&e2_1), "l"(*(const unsigned long long*)&r2_1));
        float2 s2_1 = make_float2(((z2_1.x >= 0.0f) ? r2_1.x : er2_1.x), ((z2_1.y >= 0.0f) ? r2_1.y : er2_1.y));
        float2 g2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_1) : "l"(*(const unsigned long long*)&z2_1), "l"(*(const unsigned long long*)&s2_1));
        zs_0[2] = g2_1.x;
        zs_0[3] = g2_1.y;
        float2 z2_2 = make_float2(zs_0[4], zs_0[5]);
        float _abs_22;
        asm("abs.f32 %0, %1;" : "=f"(_abs_22) : "f"(z2_2.x));
        float _abs_23;
        asm("abs.f32 %0, %1;" : "=f"(_abs_23) : "f"(z2_2.y));
        float2 _f32x2_mul_a_24 = make_float2(_abs_22, _abs_23);
        float2 _f32x2_mul_b_25 = make_float2(-1.442695f, -1.442695f);
        float2 t2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_25));
        float _exp2_26 = approx_exp2_noftz(t2_2.x);
        float _exp2_27 = approx_exp2_noftz(t2_2.y);
        float2 e2_2 = make_float2(_exp2_26, _exp2_27);
        float2 d2_2 = fma_f32x2_rn_noftz(e2_2, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_28 = approx_rcp(d2_2.x);
        float _rcp_29 = approx_rcp(d2_2.y);
        float2 r2_2 = make_float2(_rcp_28, _rcp_29);
        float2 er2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&er2_2) : "l"(*(const unsigned long long*)&e2_2), "l"(*(const unsigned long long*)&r2_2));
        float2 s2_2 = make_float2(((z2_2.x >= 0.0f) ? r2_2.x : er2_2.x), ((z2_2.y >= 0.0f) ? r2_2.y : er2_2.y));
        float2 g2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_2) : "l"(*(const unsigned long long*)&z2_2), "l"(*(const unsigned long long*)&s2_2));
        zs_0[4] = g2_2.x;
        zs_0[5] = g2_2.y;
        float2 z2_3 = make_float2(zs_0[6], zs_0[7]);
        float _abs_30;
        asm("abs.f32 %0, %1;" : "=f"(_abs_30) : "f"(z2_3.x));
        float _abs_31;
        asm("abs.f32 %0, %1;" : "=f"(_abs_31) : "f"(z2_3.y));
        float2 _f32x2_mul_a_32 = make_float2(_abs_30, _abs_31);
        float2 _f32x2_mul_b_33 = make_float2(-1.442695f, -1.442695f);
        float2 t2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
        float _exp2_34 = approx_exp2_noftz(t2_3.x);
        float _exp2_35 = approx_exp2_noftz(t2_3.y);
        float2 e2_3 = make_float2(_exp2_34, _exp2_35);
        float2 d2_3 = fma_f32x2_rn_noftz(e2_3, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_36 = approx_rcp(d2_3.x);
        float _rcp_37 = approx_rcp(d2_3.y);
        float2 r2_3 = make_float2(_rcp_36, _rcp_37);
        float2 er2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&er2_3) : "l"(*(const unsigned long long*)&e2_3), "l"(*(const unsigned long long*)&r2_3));
        float2 s2_3 = make_float2(((z2_3.x >= 0.0f) ? r2_3.x : er2_3.x), ((z2_3.y >= 0.0f) ? r2_3.y : er2_3.y));
        float2 g2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_3) : "l"(*(const unsigned long long*)&z2_3), "l"(*(const unsigned long long*)&s2_3));
        zs_0[6] = g2_3.x;
        zs_0[7] = g2_3.y;
        float2 z2_4 = make_float2(zs_0[8], zs_0[9]);
        float _abs_38;
        asm("abs.f32 %0, %1;" : "=f"(_abs_38) : "f"(z2_4.x));
        float _abs_39;
        asm("abs.f32 %0, %1;" : "=f"(_abs_39) : "f"(z2_4.y));
        float2 _f32x2_mul_a_40 = make_float2(_abs_38, _abs_39);
        float2 _f32x2_mul_b_41 = make_float2(-1.442695f, -1.442695f);
        float2 t2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_41));
        float _exp2_42 = approx_exp2_noftz(t2_4.x);
        float _exp2_43 = approx_exp2_noftz(t2_4.y);
        float2 e2_4 = make_float2(_exp2_42, _exp2_43);
        float2 d2_4 = fma_f32x2_rn_noftz(e2_4, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_44 = approx_rcp(d2_4.x);
        float _rcp_45 = approx_rcp(d2_4.y);
        float2 r2_4 = make_float2(_rcp_44, _rcp_45);
        float2 er2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&er2_4) : "l"(*(const unsigned long long*)&e2_4), "l"(*(const unsigned long long*)&r2_4));
        float2 s2_4 = make_float2(((z2_4.x >= 0.0f) ? r2_4.x : er2_4.x), ((z2_4.y >= 0.0f) ? r2_4.y : er2_4.y));
        float2 g2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_4) : "l"(*(const unsigned long long*)&z2_4), "l"(*(const unsigned long long*)&s2_4));
        zs_0[8] = g2_4.x;
        zs_0[9] = g2_4.y;
        float2 z2_5 = make_float2(zs_0[10], zs_0[11]);
        float _abs_46;
        asm("abs.f32 %0, %1;" : "=f"(_abs_46) : "f"(z2_5.x));
        float _abs_47;
        asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(z2_5.y));
        float2 _f32x2_mul_a_48 = make_float2(_abs_46, _abs_47);
        float2 _f32x2_mul_b_49 = make_float2(-1.442695f, -1.442695f);
        float2 t2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
        float _exp2_50 = approx_exp2_noftz(t2_5.x);
        float _exp2_51 = approx_exp2_noftz(t2_5.y);
        float2 e2_5 = make_float2(_exp2_50, _exp2_51);
        float2 d2_5 = fma_f32x2_rn_noftz(e2_5, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_52 = approx_rcp(d2_5.x);
        float _rcp_53 = approx_rcp(d2_5.y);
        float2 r2_5 = make_float2(_rcp_52, _rcp_53);
        float2 er2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&er2_5) : "l"(*(const unsigned long long*)&e2_5), "l"(*(const unsigned long long*)&r2_5));
        float2 s2_5 = make_float2(((z2_5.x >= 0.0f) ? r2_5.x : er2_5.x), ((z2_5.y >= 0.0f) ? r2_5.y : er2_5.y));
        float2 g2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_5) : "l"(*(const unsigned long long*)&z2_5), "l"(*(const unsigned long long*)&s2_5));
        zs_0[10] = g2_5.x;
        zs_0[11] = g2_5.y;
        float2 z2_6 = make_float2(zs_0[12], zs_0[13]);
        float _abs_54;
        asm("abs.f32 %0, %1;" : "=f"(_abs_54) : "f"(z2_6.x));
        float _abs_55;
        asm("abs.f32 %0, %1;" : "=f"(_abs_55) : "f"(z2_6.y));
        float2 _f32x2_mul_a_56 = make_float2(_abs_54, _abs_55);
        float2 _f32x2_mul_b_57 = make_float2(-1.442695f, -1.442695f);
        float2 t2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_57));
        float _exp2_58 = approx_exp2_noftz(t2_6.x);
        float _exp2_59 = approx_exp2_noftz(t2_6.y);
        float2 e2_6 = make_float2(_exp2_58, _exp2_59);
        float2 d2_6 = fma_f32x2_rn_noftz(e2_6, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_60 = approx_rcp(d2_6.x);
        float _rcp_61 = approx_rcp(d2_6.y);
        float2 r2_6 = make_float2(_rcp_60, _rcp_61);
        float2 er2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&er2_6) : "l"(*(const unsigned long long*)&e2_6), "l"(*(const unsigned long long*)&r2_6));
        float2 s2_6 = make_float2(((z2_6.x >= 0.0f) ? r2_6.x : er2_6.x), ((z2_6.y >= 0.0f) ? r2_6.y : er2_6.y));
        float2 g2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_6) : "l"(*(const unsigned long long*)&z2_6), "l"(*(const unsigned long long*)&s2_6));
        zs_0[12] = g2_6.x;
        zs_0[13] = g2_6.y;
        float2 z2_7 = make_float2(zs_0[14], zs_0[15]);
        float _abs_62;
        asm("abs.f32 %0, %1;" : "=f"(_abs_62) : "f"(z2_7.x));
        float _abs_63;
        asm("abs.f32 %0, %1;" : "=f"(_abs_63) : "f"(z2_7.y));
        float2 _f32x2_mul_a_64 = make_float2(_abs_62, _abs_63);
        float2 _f32x2_mul_b_65 = make_float2(-1.442695f, -1.442695f);
        float2 t2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
        float _exp2_66 = approx_exp2_noftz(t2_7.x);
        float _exp2_67 = approx_exp2_noftz(t2_7.y);
        float2 e2_7 = make_float2(_exp2_66, _exp2_67);
        float2 d2_7 = fma_f32x2_rn_noftz(e2_7, make_float2(1.0f, 1.0f), make_float2(1.0f, 1.0f));
        float _rcp_68 = approx_rcp(d2_7.x);
        float _rcp_69 = approx_rcp(d2_7.y);
        float2 r2_7 = make_float2(_rcp_68, _rcp_69);
        float2 er2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&er2_7) : "l"(*(const unsigned long long*)&e2_7), "l"(*(const unsigned long long*)&r2_7));
        float2 s2_7 = make_float2(((z2_7.x >= 0.0f) ? r2_7.x : er2_7.x), ((z2_7.y >= 0.0f) ? r2_7.y : er2_7.y));
        float2 g2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_7) : "l"(*(const unsigned long long*)&z2_7), "l"(*(const unsigned long long*)&s2_7));
        zs_0[14] = g2_7.x;
        zs_0[15] = g2_7.y;
        float total = 0.0f;
        total = __fmaf_rn(xs_0[0], xs_0[0], total);
        total = __fmaf_rn(xs_0[1], xs_0[1], total);
        total = __fmaf_rn(xs_0[2], xs_0[2], total);
        total = __fmaf_rn(xs_0[3], xs_0[3], total);
        total = __fmaf_rn(xs_0[4], xs_0[4], total);
        total = __fmaf_rn(xs_0[5], xs_0[5], total);
        total = __fmaf_rn(xs_0[6], xs_0[6], total);
        total = __fmaf_rn(xs_0[7], xs_0[7], total);
        total = __fmaf_rn(xs_0[8], xs_0[8], total);
        total = __fmaf_rn(xs_0[9], xs_0[9], total);
        total = __fmaf_rn(xs_0[10], xs_0[10], total);
        total = __fmaf_rn(xs_0[11], xs_0[11], total);
        total = __fmaf_rn(xs_0[12], xs_0[12], total);
        total = __fmaf_rn(xs_0[13], xs_0[13], total);
        total = __fmaf_rn(xs_0[14], xs_0[14], total);
        total = __fmaf_rn(xs_0[15], xs_0[15], total);
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
        total = total + _shfl_xor_sync_70;
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
        total = total + _shfl_xor_sync_71;
        float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
        total = total + _shfl_xor_sync_72;
        float rstd;
        asm("rsqrt.approx.f32 %0, %1;" : "=f"(rstd) : "f"(total * 0.0078125f + eps));
        float2 rstd2 = make_float2(rstd, rstd);
        float2 _f32x2_mul_a_73 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[0], xs_0[1]), rstd2), make_float2(wv[0], wv[1]));
        float2 _f32x2_mul_b_74 = make_float2(zs_0[0], zs_0[1]);
        float2 y2_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&_f32x2_mul_b_74));
        xs_0[0] = y2_0.x;
        xs_0[1] = y2_0.y;
        float2 _f32x2_mul_a_75 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[2], xs_0[3]), rstd2), make_float2(wv[2], wv[3]));
        float2 _f32x2_mul_b_76 = make_float2(zs_0[2], zs_0[3]);
        float2 y2_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_75), "l"(*(const unsigned long long*)&_f32x2_mul_b_76));
        xs_0[2] = y2_1.x;
        xs_0[3] = y2_1.y;
        float2 _f32x2_mul_a_77 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[4], xs_0[5]), rstd2), make_float2(wv[4], wv[5]));
        float2 _f32x2_mul_b_78 = make_float2(zs_0[4], zs_0[5]);
        float2 y2_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&_f32x2_mul_b_78));
        xs_0[4] = y2_2.x;
        xs_0[5] = y2_2.y;
        float2 _f32x2_mul_a_79 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[6], xs_0[7]), rstd2), make_float2(wv[6], wv[7]));
        float2 _f32x2_mul_b_80 = make_float2(zs_0[6], zs_0[7]);
        float2 y2_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&_f32x2_mul_b_80));
        xs_0[6] = y2_3.x;
        xs_0[7] = y2_3.y;
        float2 _f32x2_mul_a_81 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[8], xs_0[9]), rstd2), make_float2(wv[8], wv[9]));
        float2 _f32x2_mul_b_82 = make_float2(zs_0[8], zs_0[9]);
        float2 y2_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
        xs_0[8] = y2_4.x;
        xs_0[9] = y2_4.y;
        float2 _f32x2_mul_a_83 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[10], xs_0[11]), rstd2), make_float2(wv[10], wv[11]));
        float2 _f32x2_mul_b_84 = make_float2(zs_0[10], zs_0[11]);
        float2 y2_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&_f32x2_mul_b_84));
        xs_0[10] = y2_5.x;
        xs_0[11] = y2_5.y;
        float2 _f32x2_mul_a_85 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[12], xs_0[13]), rstd2), make_float2(wv[12], wv[13]));
        float2 _f32x2_mul_b_86 = make_float2(zs_0[12], zs_0[13]);
        float2 y2_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
        xs_0[12] = y2_6.x;
        xs_0[13] = y2_6.y;
        float2 _f32x2_mul_a_87 = mul_f32x2_noftz(mul_f32x2_noftz(make_float2(xs_0[14], xs_0[15]), rstd2), make_float2(wv[14], wv[15]));
        float2 _f32x2_mul_b_88 = make_float2(zs_0[14], zs_0[15]);
        float2 y2_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&y2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
        xs_0[14] = y2_7.x;
        xs_0[15] = y2_7.y;
        unsigned int packed[8];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 0], xs_0[_lp*2+1 + 0]));
            packed[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 2], xs_0[_lp*2+1 + 2]));
            packed[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 4], xs_0[_lp*2+1 + 4]));
            packed[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 6], xs_0[_lp*2+1 + 6]));
            packed[_lp + 3] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 8], xs_0[_lp*2+1 + 8]));
            packed[_lp + 4] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 10], xs_0[_lp*2+1 + 10]));
            packed[_lp + 5] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 12], xs_0[_lp*2+1 + 12]));
            packed[_lp + 6] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_0[_lp*2 + 14], xs_0[_lp*2+1 + 14]));
            packed[_lp + 7] = *(uint32_t*)&_bf2;
        }
        if (row_0 < rows) {
            reinterpret_cast<int4*>(y + ((long long)row_0 * 128ll + (long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
            reinterpret_cast<int4*>(y + ((long long)row_0 * 128ll + (long long)(col + 8)))[0] = reinterpret_cast<int4*>(packed + (4))[0];
        }
    }
}

} // extern "C"
