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
kernel_cake_kda_det_train_0d93d9b4441327d74a72(float* __restrict__ dq_hat, float* __restrict__ dk_hat, __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, float* __restrict__ rstd, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ dq, __nv_bfloat16* __restrict__ dk, int num_tokens, int num_heads, int q_tstride, int q_hstride, int k_tstride, int k_hstride, int dq_tstride, int dk_tstride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int bos = chunk_bos[blockIdx.x];
    int clen = chunk_len[blockIdx.x];
    int h = blockIdx.y;
    int lane_1 = tid & 31;
    int wid = tid >> 5;
    float nz = __uint_as_float((unsigned int)num_heads >> 31 | 2147483648u);
    #pragma unroll 1
    for (int j = 0; j < 8; j++) {
        int s = wid + 8 * j;
        if (s < clen) {
            int row = blockIdx.x * 64 + s;
            float g[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dq_hat + (((long long)row * (long long)num_heads + (long long)h) * 128 + (long long)(lane_1 * 4)) + 0);
                g[0 + 0] = _v4.x;
                g[0 + 1] = _v4.y;
                g[0 + 2] = _v4.z;
                g[0 + 3] = _v4.w;
            }
            float xv[4];
            {
                uint2 _vld_1;
                _vld_1 = *reinterpret_cast<const uint2*>(q + ((long long)(bos + s) * (long long)q_tstride + (long long)h * (long long)q_hstride + (long long)(lane_1 * 4)) + 0);
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&xv[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&xv[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
            float rv[1];
            {
                rv[0] = *reinterpret_cast<const float*>(rstd + ((long long)(bos + s) * (long long)num_heads + (long long)h) + 0);
            }
            float y[4];
            float p[4];
            y[0] = mul_f32_rn_noftz(xv[0], rv[0]);
            float _fma_2;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(g[0]), "f"(y[0]), "f"(nz));
            p[0] = _fma_2;
            y[1] = mul_f32_rn_noftz(xv[1], rv[0]);
            float _fma_3;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(g[1]), "f"(y[1]), "f"(nz));
            p[1] = _fma_3;
            y[2] = mul_f32_rn_noftz(xv[2], rv[0]);
            float _fma_4;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(g[2]), "f"(y[2]), "f"(nz));
            p[2] = _fma_4;
            y[3] = mul_f32_rn_noftz(xv[3], rv[0]);
            float _fma_5;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(g[3]), "f"(y[3]), "f"(nz));
            p[3] = _fma_5;
            float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, p[0], 4);
            float _add_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(p[0]), "f"(_shfl_xor_sync_7));
            p[0] = _add_6;
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, p[1], 4);
            float _add_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(p[1]), "f"(_shfl_xor_sync_9));
            p[1] = _add_8;
            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, p[2], 4);
            float _add_10;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(p[2]), "f"(_shfl_xor_sync_11));
            p[2] = _add_10;
            float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, p[3], 4);
            float _add_12;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(p[3]), "f"(_shfl_xor_sync_13));
            p[3] = _add_12;
            float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, p[0], 2);
            float _add_14;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(p[0]), "f"(_shfl_xor_sync_15));
            p[0] = _add_14;
            float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, p[1], 2);
            float _add_16;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(p[1]), "f"(_shfl_xor_sync_17));
            p[1] = _add_16;
            float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, p[2], 2);
            float _add_18;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(p[2]), "f"(_shfl_xor_sync_19));
            p[2] = _add_18;
            float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, p[3], 2);
            float _add_20;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(p[3]), "f"(_shfl_xor_sync_21));
            p[3] = _add_20;
            float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, p[0], 1);
            float _add_22;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(p[0]), "f"(_shfl_xor_sync_23));
            p[0] = _add_22;
            float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, p[1], 1);
            float _add_24;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(p[1]), "f"(_shfl_xor_sync_25));
            p[1] = _add_24;
            float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, p[2], 1);
            float _add_26;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(p[2]), "f"(_shfl_xor_sync_27));
            p[2] = _add_26;
            float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, p[3], 1);
            float _add_28;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(p[3]), "f"(_shfl_xor_sync_29));
            p[3] = _add_28;
            float _add_30;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(p[0]), "f"(p[2]));
            float _add_31;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(p[1]), "f"(p[3]));
            float sq;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(sq) : "f"(_add_30), "f"(_add_31));
            float _shfl_sync_34 = __shfl_sync(0xFFFFFFFF, sq, 0);
            float _shfl_sync_35 = __shfl_sync(0xFFFFFFFF, sq, 8);
            float _add_33;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(_shfl_sync_34), "f"(_shfl_sync_35));
            float _shfl_sync_36 = __shfl_sync(0xFFFFFFFF, sq, 16);
            float _add_32;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(_add_33), "f"(_shfl_sync_36));
            float _shfl_sync_37 = __shfl_sync(0xFFFFFFFF, sq, 24);
            float dot;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(dot) : "f"(_add_32), "f"(_shfl_sync_37));
            float o[4];
            float _fma_39;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(y[0]), "f"(dot), "f"(nz));
            float _sub_38;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(g[0]), "f"(_fma_39));
            o[0] = mul_f32_rn_noftz(rv[0], _sub_38);
            float _fma_41;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(y[1]), "f"(dot), "f"(nz));
            float _sub_40;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(g[1]), "f"(_fma_41));
            o[1] = mul_f32_rn_noftz(rv[0], _sub_40);
            float _fma_43;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(y[2]), "f"(dot), "f"(nz));
            float _sub_42;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(g[2]), "f"(_fma_43));
            o[2] = mul_f32_rn_noftz(rv[0], _sub_42);
            float _fma_45;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(y[3]), "f"(dot), "f"(nz));
            float _sub_44;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(g[3]), "f"(_fma_45));
            o[3] = mul_f32_rn_noftz(rv[0], _sub_44);
            unsigned int ob[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o[_lp*2 + 0], o[_lp*2+1 + 0]));
                ob[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dq + ((long long)(bos + s) * (long long)dq_tstride + (long long)(h * 128 + lane_1 * 4)))[0] = reinterpret_cast<int2*>(ob)[0];
            float g_0[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dk_hat + (((long long)row * (long long)num_heads + (long long)h) * 128 + (long long)(lane_1 * 4)) + 0);
                g_0[0 + 0] = _v4.x;
                g_0[0 + 1] = _v4.y;
                g_0[0 + 2] = _v4.z;
                g_0[0 + 3] = _v4.w;
            }
            float xv_0[4];
            {
                uint2 _vld_47;
                _vld_47 = *reinterpret_cast<const uint2*>(k + ((long long)(bos + s) * (long long)k_tstride + (long long)h * (long long)k_hstride + (long long)(lane_1 * 4)) + 0);
                uint32_t* _vpairs_47 = reinterpret_cast<uint32_t*>(&_vld_47);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&xv_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) << 16);
                    (&xv_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) & 0xffff0000u);
                }
            }
            float rv_0[1];
            {
                rv_0[0] = *reinterpret_cast<const float*>(rstd + ((long long)(num_tokens + (bos + s)) * (long long)num_heads + (long long)h) + 0);
            }
            float y_0[4];
            float p_0[4];
            y_0[0] = mul_f32_rn_noftz(xv_0[0], rv_0[0]);
            float _fma_48;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(g_0[0]), "f"(y_0[0]), "f"(nz));
            p_0[0] = _fma_48;
            y_0[1] = mul_f32_rn_noftz(xv_0[1], rv_0[0]);
            float _fma_49;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(g_0[1]), "f"(y_0[1]), "f"(nz));
            p_0[1] = _fma_49;
            y_0[2] = mul_f32_rn_noftz(xv_0[2], rv_0[0]);
            float _fma_50;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(g_0[2]), "f"(y_0[2]), "f"(nz));
            p_0[2] = _fma_50;
            y_0[3] = mul_f32_rn_noftz(xv_0[3], rv_0[0]);
            float _fma_51;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(g_0[3]), "f"(y_0[3]), "f"(nz));
            p_0[3] = _fma_51;
            float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, p_0[0], 4);
            float _add_52;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(p_0[0]), "f"(_shfl_xor_sync_53));
            p_0[0] = _add_52;
            float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, p_0[1], 4);
            float _add_54;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(p_0[1]), "f"(_shfl_xor_sync_55));
            p_0[1] = _add_54;
            float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, p_0[2], 4);
            float _add_56;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(p_0[2]), "f"(_shfl_xor_sync_57));
            p_0[2] = _add_56;
            float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, p_0[3], 4);
            float _add_58;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(p_0[3]), "f"(_shfl_xor_sync_59));
            p_0[3] = _add_58;
            float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, p_0[0], 2);
            float _add_60;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(p_0[0]), "f"(_shfl_xor_sync_61));
            p_0[0] = _add_60;
            float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, p_0[1], 2);
            float _add_62;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(p_0[1]), "f"(_shfl_xor_sync_63));
            p_0[1] = _add_62;
            float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, p_0[2], 2);
            float _add_64;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(p_0[2]), "f"(_shfl_xor_sync_65));
            p_0[2] = _add_64;
            float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, p_0[3], 2);
            float _add_66;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(p_0[3]), "f"(_shfl_xor_sync_67));
            p_0[3] = _add_66;
            float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, p_0[0], 1);
            float _add_68;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(p_0[0]), "f"(_shfl_xor_sync_69));
            p_0[0] = _add_68;
            float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, p_0[1], 1);
            float _add_70;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(p_0[1]), "f"(_shfl_xor_sync_71));
            p_0[1] = _add_70;
            float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, p_0[2], 1);
            float _add_72;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(p_0[2]), "f"(_shfl_xor_sync_73));
            p_0[2] = _add_72;
            float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, p_0[3], 1);
            float _add_74;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(p_0[3]), "f"(_shfl_xor_sync_75));
            p_0[3] = _add_74;
            float _add_76;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(p_0[0]), "f"(p_0[2]));
            float _add_77;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(p_0[1]), "f"(p_0[3]));
            float sq_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(sq_0) : "f"(_add_76), "f"(_add_77));
            float _shfl_sync_80 = __shfl_sync(0xFFFFFFFF, sq_0, 0);
            float _shfl_sync_81 = __shfl_sync(0xFFFFFFFF, sq_0, 8);
            float _add_79;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(_shfl_sync_80), "f"(_shfl_sync_81));
            float _shfl_sync_82 = __shfl_sync(0xFFFFFFFF, sq_0, 16);
            float _add_78;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(_add_79), "f"(_shfl_sync_82));
            float _shfl_sync_83 = __shfl_sync(0xFFFFFFFF, sq_0, 24);
            float dot_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(dot_0) : "f"(_add_78), "f"(_shfl_sync_83));
            float o_0[4];
            float _fma_85;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(y_0[0]), "f"(dot_0), "f"(nz));
            float _sub_84;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(g_0[0]), "f"(_fma_85));
            o_0[0] = mul_f32_rn_noftz(rv_0[0], _sub_84);
            float _fma_87;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(y_0[1]), "f"(dot_0), "f"(nz));
            float _sub_86;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(g_0[1]), "f"(_fma_87));
            o_0[1] = mul_f32_rn_noftz(rv_0[0], _sub_86);
            float _fma_89;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(y_0[2]), "f"(dot_0), "f"(nz));
            float _sub_88;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_88) : "f"(g_0[2]), "f"(_fma_89));
            o_0[2] = mul_f32_rn_noftz(rv_0[0], _sub_88);
            float _fma_91;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(y_0[3]), "f"(dot_0), "f"(nz));
            float _sub_90;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_90) : "f"(g_0[3]), "f"(_fma_91));
            o_0[3] = mul_f32_rn_noftz(rv_0[0], _sub_90);
            unsigned int ob_0[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_0[_lp*2 + 0], o_0[_lp*2+1 + 0]));
                ob_0[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dk + ((long long)(bos + s) * (long long)dk_tstride + (long long)(h * 128 + lane_1 * 4)))[0] = reinterpret_cast<int2*>(ob_0)[0];
        }
    }
}

} // extern "C"
