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


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kda_train_8d492fe7b35fdfc0a7e3(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, float* __restrict__ rstd, __nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, int num_tokens, int num_heads, int q_tstride, int q_hstride, int k_tstride, int k_hstride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int h = blockIdx.y;
    int c = tid & 15;
    int t0 = blockIdx.x * 64 + (tid >> 4);
    float x4[32];
    int _min_0 = ((t0) < (num_tokens - 1) ? (t0) : (num_tokens - 1));
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(q + ((long long)_min_0 * (long long)q_tstride + (long long)h * (long long)q_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_1[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_1[_blk] = _vptr_1[_blk];
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&x4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
    }
    int _min_2 = ((t0 + 16) < (num_tokens - 1) ? (t0 + 16) : (num_tokens - 1));
    {
        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(q + ((long long)_min_2 * (long long)q_tstride + (long long)h * (long long)q_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_3[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_3[_blk] = _vptr_3[_blk];
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&x4[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
    }
    int _min_4 = ((t0 + 32) < (num_tokens - 1) ? (t0 + 32) : (num_tokens - 1));
    {
        const uint4* _vptr_5 = reinterpret_cast<const uint4*>(q + ((long long)_min_4 * (long long)q_tstride + (long long)h * (long long)q_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_5[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_5[_blk] = _vptr_5[_blk];
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                (&x4[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
            }
        }
    }
    int _min_6 = ((t0 + 48) < (num_tokens - 1) ? (t0 + 48) : (num_tokens - 1));
    {
        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(q + ((long long)_min_6 * (long long)q_tstride + (long long)h * (long long)q_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_7[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_7[_blk] = _vptr_7[_blk];
            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                (&x4[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
            }
        }
    }
    float ss[4];
    ss[0] = mul_f32_rn_noftz(x4[0], x4[0]);
    float _fma_8;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(x4[1]), "f"(x4[1]), "f"(ss[0]));
    ss[0] = _fma_8;
    float _fma_9;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(x4[2]), "f"(x4[2]), "f"(ss[0]));
    ss[0] = _fma_9;
    float _fma_10;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(x4[3]), "f"(x4[3]), "f"(ss[0]));
    ss[0] = _fma_10;
    float _fma_11;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(x4[4]), "f"(x4[4]), "f"(ss[0]));
    ss[0] = _fma_11;
    float _fma_12;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(x4[5]), "f"(x4[5]), "f"(ss[0]));
    ss[0] = _fma_12;
    float _fma_13;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(x4[6]), "f"(x4[6]), "f"(ss[0]));
    ss[0] = _fma_13;
    float _fma_14;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(x4[7]), "f"(x4[7]), "f"(ss[0]));
    ss[0] = _fma_14;
    ss[1] = mul_f32_rn_noftz(x4[8], x4[8]);
    float _fma_15;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(x4[9]), "f"(x4[9]), "f"(ss[1]));
    ss[1] = _fma_15;
    float _fma_16;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(x4[10]), "f"(x4[10]), "f"(ss[1]));
    ss[1] = _fma_16;
    float _fma_17;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(x4[11]), "f"(x4[11]), "f"(ss[1]));
    ss[1] = _fma_17;
    float _fma_18;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(x4[12]), "f"(x4[12]), "f"(ss[1]));
    ss[1] = _fma_18;
    float _fma_19;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(x4[13]), "f"(x4[13]), "f"(ss[1]));
    ss[1] = _fma_19;
    float _fma_20;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(x4[14]), "f"(x4[14]), "f"(ss[1]));
    ss[1] = _fma_20;
    float _fma_21;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(x4[15]), "f"(x4[15]), "f"(ss[1]));
    ss[1] = _fma_21;
    ss[2] = mul_f32_rn_noftz(x4[16], x4[16]);
    float _fma_22;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(x4[17]), "f"(x4[17]), "f"(ss[2]));
    ss[2] = _fma_22;
    float _fma_23;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(x4[18]), "f"(x4[18]), "f"(ss[2]));
    ss[2] = _fma_23;
    float _fma_24;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(x4[19]), "f"(x4[19]), "f"(ss[2]));
    ss[2] = _fma_24;
    float _fma_25;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(x4[20]), "f"(x4[20]), "f"(ss[2]));
    ss[2] = _fma_25;
    float _fma_26;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(x4[21]), "f"(x4[21]), "f"(ss[2]));
    ss[2] = _fma_26;
    float _fma_27;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(x4[22]), "f"(x4[22]), "f"(ss[2]));
    ss[2] = _fma_27;
    float _fma_28;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(x4[23]), "f"(x4[23]), "f"(ss[2]));
    ss[2] = _fma_28;
    ss[3] = mul_f32_rn_noftz(x4[24], x4[24]);
    float _fma_29;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(x4[25]), "f"(x4[25]), "f"(ss[3]));
    ss[3] = _fma_29;
    float _fma_30;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(x4[26]), "f"(x4[26]), "f"(ss[3]));
    ss[3] = _fma_30;
    float _fma_31;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(x4[27]), "f"(x4[27]), "f"(ss[3]));
    ss[3] = _fma_31;
    float _fma_32;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(x4[28]), "f"(x4[28]), "f"(ss[3]));
    ss[3] = _fma_32;
    float _fma_33;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(x4[29]), "f"(x4[29]), "f"(ss[3]));
    ss[3] = _fma_33;
    float _fma_34;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(x4[30]), "f"(x4[30]), "f"(ss[3]));
    ss[3] = _fma_34;
    float _fma_35;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(x4[31]), "f"(x4[31]), "f"(ss[3]));
    ss[3] = _fma_35;
    float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, ss[0], 8);
    float _add_36;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(ss[0]), "f"(_shfl_xor_sync_37));
    ss[0] = _add_36;
    float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, ss[1], 8);
    float _add_38;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(ss[1]), "f"(_shfl_xor_sync_39));
    ss[1] = _add_38;
    float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, ss[2], 8);
    float _add_40;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(ss[2]), "f"(_shfl_xor_sync_41));
    ss[2] = _add_40;
    float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, ss[3], 8);
    float _add_42;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(ss[3]), "f"(_shfl_xor_sync_43));
    ss[3] = _add_42;
    float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, ss[0], 4);
    float _add_44;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(ss[0]), "f"(_shfl_xor_sync_45));
    ss[0] = _add_44;
    float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, ss[1], 4);
    float _add_46;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(ss[1]), "f"(_shfl_xor_sync_47));
    ss[1] = _add_46;
    float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, ss[2], 4);
    float _add_48;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(ss[2]), "f"(_shfl_xor_sync_49));
    ss[2] = _add_48;
    float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, ss[3], 4);
    float _add_50;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(ss[3]), "f"(_shfl_xor_sync_51));
    ss[3] = _add_50;
    float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, ss[0], 2);
    float _add_52;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(ss[0]), "f"(_shfl_xor_sync_53));
    ss[0] = _add_52;
    float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, ss[1], 2);
    float _add_54;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(ss[1]), "f"(_shfl_xor_sync_55));
    ss[1] = _add_54;
    float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, ss[2], 2);
    float _add_56;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(ss[2]), "f"(_shfl_xor_sync_57));
    ss[2] = _add_56;
    float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, ss[3], 2);
    float _add_58;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(ss[3]), "f"(_shfl_xor_sync_59));
    ss[3] = _add_58;
    float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, ss[0], 1);
    float _add_60;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(ss[0]), "f"(_shfl_xor_sync_61));
    ss[0] = _add_60;
    float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, ss[1], 1);
    float _add_62;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(ss[1]), "f"(_shfl_xor_sync_63));
    ss[1] = _add_62;
    float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, ss[2], 1);
    float _add_64;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(ss[2]), "f"(_shfl_xor_sync_65));
    ss[2] = _add_64;
    float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, ss[3], 1);
    float _add_66;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(ss[3]), "f"(_shfl_xor_sync_67));
    ss[3] = _add_66;
    float r[4];
    float _add_70;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(ss[0]), "f"(1e-06f));
    float _sqrt_69;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_69) : "f"(_add_70));
    float _fdiv_68 = __fdiv_rn(1.0f, _sqrt_69);
    r[0] = _fdiv_68;
    float _add_73;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(ss[1]), "f"(1e-06f));
    float _sqrt_72;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_72) : "f"(_add_73));
    float _fdiv_71 = __fdiv_rn(1.0f, _sqrt_72);
    r[1] = _fdiv_71;
    float _add_76;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(ss[2]), "f"(1e-06f));
    float _sqrt_75;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_75) : "f"(_add_76));
    float _fdiv_74 = __fdiv_rn(1.0f, _sqrt_75);
    r[2] = _fdiv_74;
    float _add_79;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(ss[3]), "f"(1e-06f));
    float _sqrt_78;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_78) : "f"(_add_79));
    float _fdiv_77 = __fdiv_rn(1.0f, _sqrt_78);
    r[3] = _fdiv_77;
    if (t0 < num_tokens) {
        float y[8];
        y[0] = mul_f32_rn_noftz(x4[0], r[0]);
        y[1] = mul_f32_rn_noftz(x4[1], r[0]);
        y[2] = mul_f32_rn_noftz(x4[2], r[0]);
        y[3] = mul_f32_rn_noftz(x4[3], r[0]);
        y[4] = mul_f32_rn_noftz(x4[4], r[0]);
        y[5] = mul_f32_rn_noftz(x4[5], r[0]);
        y[6] = mul_f32_rn_noftz(x4[6], r[0]);
        y[7] = mul_f32_rn_noftz(x4[7], r[0]);
        unsigned int yb[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y[_lp*2 + 0], y[_lp*2+1 + 0]));
            yb[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(qn + (((long long)t0 * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb)[0];
        if (c == 0) {
            *(reinterpret_cast<float*>(rstd + ((long long)t0 * (long long)num_heads + (long long)h)) + (0)) = r[0];
        }
    }
    if (t0 + 16 < num_tokens) {
        float y_1[8];
        y_1[0] = mul_f32_rn_noftz(x4[8], r[1]);
        y_1[1] = mul_f32_rn_noftz(x4[9], r[1]);
        y_1[2] = mul_f32_rn_noftz(x4[10], r[1]);
        y_1[3] = mul_f32_rn_noftz(x4[11], r[1]);
        y_1[4] = mul_f32_rn_noftz(x4[12], r[1]);
        y_1[5] = mul_f32_rn_noftz(x4[13], r[1]);
        y_1[6] = mul_f32_rn_noftz(x4[14], r[1]);
        y_1[7] = mul_f32_rn_noftz(x4[15], r[1]);
        unsigned int yb_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_1[_lp*2 + 0], y_1[_lp*2+1 + 0]));
            yb_1[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(qn + (((long long)(t0 + 16) * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_1)[0];
        if (c == 0) {
            *(reinterpret_cast<float*>(rstd + ((long long)(t0 + 16) * (long long)num_heads + (long long)h)) + (0)) = r[1];
        }
    }
    if (t0 + 32 < num_tokens) {
        float y_2[8];
        y_2[0] = mul_f32_rn_noftz(x4[16], r[2]);
        y_2[1] = mul_f32_rn_noftz(x4[17], r[2]);
        y_2[2] = mul_f32_rn_noftz(x4[18], r[2]);
        y_2[3] = mul_f32_rn_noftz(x4[19], r[2]);
        y_2[4] = mul_f32_rn_noftz(x4[20], r[2]);
        y_2[5] = mul_f32_rn_noftz(x4[21], r[2]);
        y_2[6] = mul_f32_rn_noftz(x4[22], r[2]);
        y_2[7] = mul_f32_rn_noftz(x4[23], r[2]);
        unsigned int yb_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_2[_lp*2 + 0], y_2[_lp*2+1 + 0]));
            yb_2[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(qn + (((long long)(t0 + 32) * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_2)[0];
        if (c == 0) {
            *(reinterpret_cast<float*>(rstd + ((long long)(t0 + 32) * (long long)num_heads + (long long)h)) + (0)) = r[2];
        }
    }
    if (t0 + 48 < num_tokens) {
        float y_3[8];
        y_3[0] = mul_f32_rn_noftz(x4[24], r[3]);
        y_3[1] = mul_f32_rn_noftz(x4[25], r[3]);
        y_3[2] = mul_f32_rn_noftz(x4[26], r[3]);
        y_3[3] = mul_f32_rn_noftz(x4[27], r[3]);
        y_3[4] = mul_f32_rn_noftz(x4[28], r[3]);
        y_3[5] = mul_f32_rn_noftz(x4[29], r[3]);
        y_3[6] = mul_f32_rn_noftz(x4[30], r[3]);
        y_3[7] = mul_f32_rn_noftz(x4[31], r[3]);
        unsigned int yb_3[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_3[_lp*2 + 0], y_3[_lp*2+1 + 0]));
            yb_3[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(qn + (((long long)(t0 + 48) * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_3)[0];
        if (c == 0) {
            *(reinterpret_cast<float*>(rstd + ((long long)(t0 + 48) * (long long)num_heads + (long long)h)) + (0)) = r[3];
        }
    }
    float x4_0[32];
    int _min_80 = ((t0) < (num_tokens - 1) ? (t0) : (num_tokens - 1));
    {
        const uint4* _vptr_81 = reinterpret_cast<const uint4*>(k + ((long long)_min_80 * (long long)k_tstride + (long long)h * (long long)k_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_81[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_81[_blk] = _vptr_81[_blk];
            uint32_t* _vpairs_81 = reinterpret_cast<uint32_t*>(&_vld_81[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) << 16);
                (&x4_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) & 0xffff0000u);
            }
        }
    }
    int _min_82 = ((t0 + 16) < (num_tokens - 1) ? (t0 + 16) : (num_tokens - 1));
    {
        const uint4* _vptr_83 = reinterpret_cast<const uint4*>(k + ((long long)_min_82 * (long long)k_tstride + (long long)h * (long long)k_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_83[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_83[_blk] = _vptr_83[_blk];
            uint32_t* _vpairs_83 = reinterpret_cast<uint32_t*>(&_vld_83[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4_0[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) << 16);
                (&x4_0[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) & 0xffff0000u);
            }
        }
    }
    int _min_84 = ((t0 + 32) < (num_tokens - 1) ? (t0 + 32) : (num_tokens - 1));
    {
        const uint4* _vptr_85 = reinterpret_cast<const uint4*>(k + ((long long)_min_84 * (long long)k_tstride + (long long)h * (long long)k_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_85[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_85[_blk] = _vptr_85[_blk];
            uint32_t* _vpairs_85 = reinterpret_cast<uint32_t*>(&_vld_85[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4_0[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_85[_pair]) << 16);
                (&x4_0[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_85[_pair]) & 0xffff0000u);
            }
        }
    }
    int _min_86 = ((t0 + 48) < (num_tokens - 1) ? (t0 + 48) : (num_tokens - 1));
    {
        const uint4* _vptr_87 = reinterpret_cast<const uint4*>(k + ((long long)_min_86 * (long long)k_tstride + (long long)h * (long long)k_hstride + (long long)(c * 8)) + 0);
        uint4 _vld_87[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_87[_blk] = _vptr_87[_blk];
            uint32_t* _vpairs_87 = reinterpret_cast<uint32_t*>(&_vld_87[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&x4_0[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_87[_pair]) << 16);
                (&x4_0[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_87[_pair]) & 0xffff0000u);
            }
        }
    }
    float ss_0[4];
    ss_0[0] = mul_f32_rn_noftz(x4_0[0], x4_0[0]);
    float _fma_88;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(x4_0[1]), "f"(x4_0[1]), "f"(ss_0[0]));
    ss_0[0] = _fma_88;
    float _fma_89;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(x4_0[2]), "f"(x4_0[2]), "f"(ss_0[0]));
    ss_0[0] = _fma_89;
    float _fma_90;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(x4_0[3]), "f"(x4_0[3]), "f"(ss_0[0]));
    ss_0[0] = _fma_90;
    float _fma_91;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(x4_0[4]), "f"(x4_0[4]), "f"(ss_0[0]));
    ss_0[0] = _fma_91;
    float _fma_92;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(x4_0[5]), "f"(x4_0[5]), "f"(ss_0[0]));
    ss_0[0] = _fma_92;
    float _fma_93;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(x4_0[6]), "f"(x4_0[6]), "f"(ss_0[0]));
    ss_0[0] = _fma_93;
    float _fma_94;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(x4_0[7]), "f"(x4_0[7]), "f"(ss_0[0]));
    ss_0[0] = _fma_94;
    ss_0[1] = mul_f32_rn_noftz(x4_0[8], x4_0[8]);
    float _fma_95;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(x4_0[9]), "f"(x4_0[9]), "f"(ss_0[1]));
    ss_0[1] = _fma_95;
    float _fma_96;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(x4_0[10]), "f"(x4_0[10]), "f"(ss_0[1]));
    ss_0[1] = _fma_96;
    float _fma_97;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(x4_0[11]), "f"(x4_0[11]), "f"(ss_0[1]));
    ss_0[1] = _fma_97;
    float _fma_98;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(x4_0[12]), "f"(x4_0[12]), "f"(ss_0[1]));
    ss_0[1] = _fma_98;
    float _fma_99;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(x4_0[13]), "f"(x4_0[13]), "f"(ss_0[1]));
    ss_0[1] = _fma_99;
    float _fma_100;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(x4_0[14]), "f"(x4_0[14]), "f"(ss_0[1]));
    ss_0[1] = _fma_100;
    float _fma_101;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(x4_0[15]), "f"(x4_0[15]), "f"(ss_0[1]));
    ss_0[1] = _fma_101;
    ss_0[2] = mul_f32_rn_noftz(x4_0[16], x4_0[16]);
    float _fma_102;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(x4_0[17]), "f"(x4_0[17]), "f"(ss_0[2]));
    ss_0[2] = _fma_102;
    float _fma_103;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(x4_0[18]), "f"(x4_0[18]), "f"(ss_0[2]));
    ss_0[2] = _fma_103;
    float _fma_104;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(x4_0[19]), "f"(x4_0[19]), "f"(ss_0[2]));
    ss_0[2] = _fma_104;
    float _fma_105;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(x4_0[20]), "f"(x4_0[20]), "f"(ss_0[2]));
    ss_0[2] = _fma_105;
    float _fma_106;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(x4_0[21]), "f"(x4_0[21]), "f"(ss_0[2]));
    ss_0[2] = _fma_106;
    float _fma_107;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(x4_0[22]), "f"(x4_0[22]), "f"(ss_0[2]));
    ss_0[2] = _fma_107;
    float _fma_108;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(x4_0[23]), "f"(x4_0[23]), "f"(ss_0[2]));
    ss_0[2] = _fma_108;
    ss_0[3] = mul_f32_rn_noftz(x4_0[24], x4_0[24]);
    float _fma_109;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(x4_0[25]), "f"(x4_0[25]), "f"(ss_0[3]));
    ss_0[3] = _fma_109;
    float _fma_110;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(x4_0[26]), "f"(x4_0[26]), "f"(ss_0[3]));
    ss_0[3] = _fma_110;
    float _fma_111;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(x4_0[27]), "f"(x4_0[27]), "f"(ss_0[3]));
    ss_0[3] = _fma_111;
    float _fma_112;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(x4_0[28]), "f"(x4_0[28]), "f"(ss_0[3]));
    ss_0[3] = _fma_112;
    float _fma_113;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(x4_0[29]), "f"(x4_0[29]), "f"(ss_0[3]));
    ss_0[3] = _fma_113;
    float _fma_114;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(x4_0[30]), "f"(x4_0[30]), "f"(ss_0[3]));
    ss_0[3] = _fma_114;
    float _fma_115;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(x4_0[31]), "f"(x4_0[31]), "f"(ss_0[3]));
    ss_0[3] = _fma_115;
    float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, ss_0[0], 8);
    float _add_116;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(ss_0[0]), "f"(_shfl_xor_sync_117));
    ss_0[0] = _add_116;
    float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, ss_0[1], 8);
    float _add_118;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(ss_0[1]), "f"(_shfl_xor_sync_119));
    ss_0[1] = _add_118;
    float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, ss_0[2], 8);
    float _add_120;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(ss_0[2]), "f"(_shfl_xor_sync_121));
    ss_0[2] = _add_120;
    float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, ss_0[3], 8);
    float _add_122;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(ss_0[3]), "f"(_shfl_xor_sync_123));
    ss_0[3] = _add_122;
    float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, ss_0[0], 4);
    float _add_124;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(ss_0[0]), "f"(_shfl_xor_sync_125));
    ss_0[0] = _add_124;
    float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, ss_0[1], 4);
    float _add_126;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(ss_0[1]), "f"(_shfl_xor_sync_127));
    ss_0[1] = _add_126;
    float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, ss_0[2], 4);
    float _add_128;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(ss_0[2]), "f"(_shfl_xor_sync_129));
    ss_0[2] = _add_128;
    float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, ss_0[3], 4);
    float _add_130;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(ss_0[3]), "f"(_shfl_xor_sync_131));
    ss_0[3] = _add_130;
    float _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, ss_0[0], 2);
    float _add_132;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(ss_0[0]), "f"(_shfl_xor_sync_133));
    ss_0[0] = _add_132;
    float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, ss_0[1], 2);
    float _add_134;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(ss_0[1]), "f"(_shfl_xor_sync_135));
    ss_0[1] = _add_134;
    float _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, ss_0[2], 2);
    float _add_136;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(ss_0[2]), "f"(_shfl_xor_sync_137));
    ss_0[2] = _add_136;
    float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, ss_0[3], 2);
    float _add_138;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_138) : "f"(ss_0[3]), "f"(_shfl_xor_sync_139));
    ss_0[3] = _add_138;
    float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, ss_0[0], 1);
    float _add_140;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(ss_0[0]), "f"(_shfl_xor_sync_141));
    ss_0[0] = _add_140;
    float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, ss_0[1], 1);
    float _add_142;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(ss_0[1]), "f"(_shfl_xor_sync_143));
    ss_0[1] = _add_142;
    float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, ss_0[2], 1);
    float _add_144;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(ss_0[2]), "f"(_shfl_xor_sync_145));
    ss_0[2] = _add_144;
    float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, ss_0[3], 1);
    float _add_146;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(ss_0[3]), "f"(_shfl_xor_sync_147));
    ss_0[3] = _add_146;
    float r_0[4];
    float _add_150;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(ss_0[0]), "f"(1e-06f));
    float _sqrt_149;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_149) : "f"(_add_150));
    float _fdiv_148 = __fdiv_rn(1.0f, _sqrt_149);
    r_0[0] = _fdiv_148;
    float _add_153;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(ss_0[1]), "f"(1e-06f));
    float _sqrt_152;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_152) : "f"(_add_153));
    float _fdiv_151 = __fdiv_rn(1.0f, _sqrt_152);
    r_0[1] = _fdiv_151;
    float _add_156;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(ss_0[2]), "f"(1e-06f));
    float _sqrt_155;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_155) : "f"(_add_156));
    float _fdiv_154 = __fdiv_rn(1.0f, _sqrt_155);
    r_0[2] = _fdiv_154;
    float _add_159;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159) : "f"(ss_0[3]), "f"(1e-06f));
    float _sqrt_158;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_158) : "f"(_add_159));
    float _fdiv_157 = __fdiv_rn(1.0f, _sqrt_158);
    r_0[3] = _fdiv_157;
    if (t0 < num_tokens) {
        float y_4[8];
        y_4[0] = mul_f32_rn_noftz(x4_0[0], r_0[0]);
        y_4[1] = mul_f32_rn_noftz(x4_0[1], r_0[0]);
        y_4[2] = mul_f32_rn_noftz(x4_0[2], r_0[0]);
        y_4[3] = mul_f32_rn_noftz(x4_0[3], r_0[0]);
        y_4[4] = mul_f32_rn_noftz(x4_0[4], r_0[0]);
        y_4[5] = mul_f32_rn_noftz(x4_0[5], r_0[0]);
        y_4[6] = mul_f32_rn_noftz(x4_0[6], r_0[0]);
        y_4[7] = mul_f32_rn_noftz(x4_0[7], r_0[0]);
        unsigned int yb_4[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_4[_lp*2 + 0], y_4[_lp*2+1 + 0]));
            yb_4[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(kn + (((long long)t0 * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_4)[0];
        if (c == 0) {
            *(reinterpret_cast<float*>(rstd + ((long long)(num_tokens + t0) * (long long)num_heads + (long long)h)) + (0)) = r_0[0];
        }
    }
    if (t0 + 16 < num_tokens) {
        float y_5[8];
        y_5[0] = mul_f32_rn_noftz(x4_0[8], r_0[1]);
        y_5[1] = mul_f32_rn_noftz(x4_0[9], r_0[1]);
        y_5[2] = mul_f32_rn_noftz(x4_0[10], r_0[1]);
        y_5[3] = mul_f32_rn_noftz(x4_0[11], r_0[1]);
        y_5[4] = mul_f32_rn_noftz(x4_0[12], r_0[1]);
        y_5[5] = mul_f32_rn_noftz(x4_0[13], r_0[1]);
        y_5[6] = mul_f32_rn_noftz(x4_0[14], r_0[1]);
        y_5[7] = mul_f32_rn_noftz(x4_0[15], r_0[1]);
        unsigned int yb_5[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_5[_lp*2 + 0], y_5[_lp*2+1 + 0]));
            yb_5[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(kn + (((long long)(t0 + 16) * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_5)[0];
        if (c == 0) {
            *(reinterpret_cast<float*>(rstd + ((long long)(num_tokens + t0 + 16) * (long long)num_heads + (long long)h)) + (0)) = r_0[1];
        }
    }
    if (t0 + 32 < num_tokens) {
        float y_6[8];
        y_6[0] = mul_f32_rn_noftz(x4_0[16], r_0[2]);
        y_6[1] = mul_f32_rn_noftz(x4_0[17], r_0[2]);
        y_6[2] = mul_f32_rn_noftz(x4_0[18], r_0[2]);
        y_6[3] = mul_f32_rn_noftz(x4_0[19], r_0[2]);
        y_6[4] = mul_f32_rn_noftz(x4_0[20], r_0[2]);
        y_6[5] = mul_f32_rn_noftz(x4_0[21], r_0[2]);
        y_6[6] = mul_f32_rn_noftz(x4_0[22], r_0[2]);
        y_6[7] = mul_f32_rn_noftz(x4_0[23], r_0[2]);
        unsigned int yb_6[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_6[_lp*2 + 0], y_6[_lp*2+1 + 0]));
            yb_6[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(kn + (((long long)(t0 + 32) * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_6)[0];
        if (c == 0) {
            *(reinterpret_cast<float*>(rstd + ((long long)(num_tokens + t0 + 32) * (long long)num_heads + (long long)h)) + (0)) = r_0[2];
        }
    }
    if (t0 + 48 < num_tokens) {
        float y_7[8];
        y_7[0] = mul_f32_rn_noftz(x4_0[24], r_0[3]);
        y_7[1] = mul_f32_rn_noftz(x4_0[25], r_0[3]);
        y_7[2] = mul_f32_rn_noftz(x4_0[26], r_0[3]);
        y_7[3] = mul_f32_rn_noftz(x4_0[27], r_0[3]);
        y_7[4] = mul_f32_rn_noftz(x4_0[28], r_0[3]);
        y_7[5] = mul_f32_rn_noftz(x4_0[29], r_0[3]);
        y_7[6] = mul_f32_rn_noftz(x4_0[30], r_0[3]);
        y_7[7] = mul_f32_rn_noftz(x4_0[31], r_0[3]);
        unsigned int yb_7[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_7[_lp*2 + 0], y_7[_lp*2+1 + 0]));
            yb_7[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(kn + (((long long)(t0 + 48) * (long long)num_heads + (long long)h) * 128 + (long long)(c * 8)))[0] = reinterpret_cast<int4*>(yb_7)[0];
        if (c == 0) {
            *(reinterpret_cast<float*>(rstd + ((long long)(num_tokens + t0 + 48) * (long long)num_heads + (long long)h)) + (0)) = r_0[3];
        }
    }
}

} // extern "C"
