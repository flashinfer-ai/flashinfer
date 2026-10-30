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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

#include <math_constants.h>

extern "C" __device__ float __nv_expf(float);

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_mhc_train_fwd_coef__lp2_w8_adj(float* __restrict__ spart, float* __restrict__ qpart, float* __restrict__ scale, float* __restrict__ base, float* __restrict__ stats, int T, int p_k, int t_pad, float rms_eps, float hc_eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int lane_0 = (int)lane;
    int warp_1 = (int)warp;
    long long tp64 = (long long)t_pad;
    int n = ((lane_0 < 24) ? lane_0 : 23);
    long long n64 = (long long)n;
    long long lane64 = (long long)lane_0;
    float sc0 = scale[0];
    float sc1 = scale[1];
    float sc2 = scale[2];
    float sc = ((lane_0 < 4) ? sc0 : ((lane_0 < 8) ? sc1 : sc2));
    float bs = base[n64];
    int kq = ((lane_0 < p_k) ? lane_0 : 0);
    int tokr[2];
    long long tcr[2];
    float s_acc[2];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        tokr[j] = bid * 8 + (2 * warp_1 + j);
        tcr[j] = (long long)(((tokr[j] < T) ? tokr[j] : T - 1));
        s_acc[j] = 0.0f;
    }
    #pragma unroll 8
    for (int k = 0; k < p_k; k++) {
        long long k64 = (long long)k * tp64;
        long long plane = k64 + tcr[0];
        float _add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(s_acc[0]), "f"(spart[plane * 24 + n64]));
        s_acc[0] = _add_0;
        float _add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(s_acc[1]), "f"(spart[plane * 24 + (24 + n64)]));
        s_acc[1] = _add_1;
    }
    float qv_load0 = qpart[(long long)kq * tp64 + tcr[0]];
    float qv0 = ((lane_0 < p_k) ? qv_load0 : 0.0f);
    float _warp_reduce_4 = qv0;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
    float _fma_3;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(_warp_reduce_4), "f"(6.1035156e-05f), "f"(rms_eps));
    float _sqrt_2;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_2) : "f"(_fma_3));
    float r0 = __frcp_rn(_sqrt_2);
    float z0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(z0) : "f"(r0), "f"(s_acc[0]));
    float a0;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a0) : "f"(z0), "f"(sc), "f"(bs));
    float qv_load1 = qpart[(long long)kq * tp64 + tcr[1]];
    float qv1 = ((lane_0 < p_k) ? qv_load1 : 0.0f);
    float _warp_reduce_7 = qv1;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_7 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset);
    float _fma_6;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(_warp_reduce_7), "f"(6.1035156e-05f), "f"(rms_eps));
    float _sqrt_5;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_5) : "f"(_fma_6));
    float r1 = __frcp_rn(_sqrt_5);
    float z1;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(z1) : "f"(r1), "f"(s_acc[1]));
    float a1;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a1) : "f"(z1), "f"(sc), "f"(bs));
    int src_sig = ((lane_0 >= 24) ? lane_0 - 24 : lane_0);
    float _shfl_0 = __shfl_sync(0xFFFFFFFF, a1, src_sig);
    float a_sig = ((lane_0 < 24) ? a0 : _shfl_0);
    int src_logit = ((lane_0 < 8) ? lane_0 + 8 : lane_0 - 8);
    float _shfl_1 = __shfl_sync(0xFFFFFFFF, a1, src_logit);
    float logit = ((lane_0 >= 8 && lane_0 < 24) ? a0 : _shfl_1);
    float _sub_8;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(0.0f), "f"(fabsf(a_sig)));
    float _exp_0 = __nv_expf(_sub_8);
    float _add_0_1;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0_1) : "f"(1.0f), "f"(_exp_0));
    float _fdiv_rn_0 = __fdiv_rn(1.0f, _add_0_1);
    float _fdiv_rn_1 = __fdiv_rn(_exp_0, _add_0_1);
    float sig = ((a_sig >= 0.0f) ? _fdiv_rn_0 : _fdiv_rn_1);
    float sneg = ((a_sig >= 0.0f) ? _fdiv_rn_1 : _fdiv_rn_0);
    float _add_1_1;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1_1) : "f"(sig), "f"(hc_eps));
    float _mul_0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(2.0f), "f"(sig));
    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
    float _max_0 = fmaxf(logit, _shfl_xor_sync_9);
    float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, _max_0, 2);
    float _sub_10;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(logit), "f"(fmaxf(_max_0, _shfl_xor_sync_11)));
    float _exp_1 = __nv_expf(_sub_10);
    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
    float add;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(_exp_1), "f"(_shfl_xor_sync_12));
    float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, add, 2);
    float _add_14;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(add), "f"(_shfl_xor_sync_15));
    float _fdiv_13 = __fdiv_rn(_exp_1, _add_14);
    float _add_3;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(_fdiv_13), "f"(hc_eps));
    float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, _add_3, 4);
    float add_0;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_0) : "f"(_add_3), "f"(_shfl_xor_sync_16));
    float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, add_0, 24);
    float _add_18;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(add_0), "f"(_shfl_xor_sync_19));
    float _add_17;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(_add_18), "f"(hc_eps));
    float _fdiv_rn_2 = __fdiv_rn(_add_3, _add_17);
    float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 1);
    float add_1;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(_fdiv_rn_2), "f"(_shfl_xor_sync_20));
    float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, add_1, 2);
    float _add_22;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(add_1), "f"(_shfl_xor_sync_23));
    float _add_21;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(_add_22), "f"(hc_eps));
    float _fdiv_rn_3 = __fdiv_rn(_fdiv_rn_2, _add_21);
    float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 4);
    float add_2;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(_fdiv_rn_3), "f"(_shfl_xor_sync_24));
    float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, add_2, 24);
    float _add_26;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(add_2), "f"(_shfl_xor_sync_27));
    float _add_25;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(_add_26), "f"(hc_eps));
    float _fdiv_rn_4 = __fdiv_rn(_fdiv_rn_3, _add_25);
    float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 1);
    float add_3;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(_fdiv_rn_4), "f"(_shfl_xor_sync_28));
    float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, add_3, 2);
    float _add_30;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(add_3), "f"(_shfl_xor_sync_31));
    float _add_29;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(_add_30), "f"(hc_eps));
    float _fdiv_rn_5 = __fdiv_rn(_fdiv_rn_4, _add_29);
    float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 4);
    float add_4;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(_fdiv_rn_5), "f"(_shfl_xor_sync_32));
    float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, add_4, 24);
    float _add_34;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(add_4), "f"(_shfl_xor_sync_35));
    float _add_33;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(_add_34), "f"(hc_eps));
    float _fdiv_rn_6 = __fdiv_rn(_fdiv_rn_5, _add_33);
    float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 1);
    float add_5;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(_fdiv_rn_6), "f"(_shfl_xor_sync_36));
    float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, add_5, 2);
    float _add_38;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(add_5), "f"(_shfl_xor_sync_39));
    float _add_37;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(_add_38), "f"(hc_eps));
    float _fdiv_rn_7 = __fdiv_rn(_fdiv_rn_6, _add_37);
    float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 4);
    float add_6;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_6) : "f"(_fdiv_rn_7), "f"(_shfl_xor_sync_40));
    float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, add_6, 24);
    float _add_42;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(add_6), "f"(_shfl_xor_sync_43));
    float _add_41;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(_add_42), "f"(hc_eps));
    float _fdiv_rn_8 = __fdiv_rn(_fdiv_rn_7, _add_41);
    float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 1);
    float add_7;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(_fdiv_rn_8), "f"(_shfl_xor_sync_44));
    float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, add_7, 2);
    float _add_46;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(add_7), "f"(_shfl_xor_sync_47));
    float _add_45;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(_add_46), "f"(hc_eps));
    float _fdiv_rn_9 = __fdiv_rn(_fdiv_rn_8, _add_45);
    float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 4);
    float add_8;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(_fdiv_rn_9), "f"(_shfl_xor_sync_48));
    float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, add_8, 24);
    float _add_50;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(add_8), "f"(_shfl_xor_sync_51));
    float _add_49;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(_add_50), "f"(hc_eps));
    float _fdiv_rn_10 = __fdiv_rn(_fdiv_rn_9, _add_49);
    float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 1);
    float add_9;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_9) : "f"(_fdiv_rn_10), "f"(_shfl_xor_sync_52));
    float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, add_9, 2);
    float _add_54;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(add_9), "f"(_shfl_xor_sync_55));
    float _add_53;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(_add_54), "f"(hc_eps));
    float _fdiv_rn_11 = __fdiv_rn(_fdiv_rn_10, _add_53);
    float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 4);
    float add_10;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_10) : "f"(_fdiv_rn_11), "f"(_shfl_xor_sync_56));
    float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, add_10, 24);
    float _add_58;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(add_10), "f"(_shfl_xor_sync_59));
    float _add_57;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(_add_58), "f"(hc_eps));
    float _fdiv_rn_12 = __fdiv_rn(_fdiv_rn_11, _add_57);
    float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 1);
    float add_11;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_11) : "f"(_fdiv_rn_12), "f"(_shfl_xor_sync_60));
    float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, add_11, 2);
    float _add_62;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(add_11), "f"(_shfl_xor_sync_63));
    float _add_61;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(_add_62), "f"(hc_eps));
    float _fdiv_rn_13 = __fdiv_rn(_fdiv_rn_12, _add_61);
    float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 4);
    float add_12;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_12) : "f"(_fdiv_rn_13), "f"(_shfl_xor_sync_64));
    float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, add_12, 24);
    float _add_66;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(add_12), "f"(_shfl_xor_sync_67));
    float _add_65;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(_add_66), "f"(hc_eps));
    float _fdiv_rn_14 = __fdiv_rn(_fdiv_rn_13, _add_65);
    float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 1);
    float add_13;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_13) : "f"(_fdiv_rn_14), "f"(_shfl_xor_sync_68));
    float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, add_13, 2);
    float _add_70;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(add_13), "f"(_shfl_xor_sync_71));
    float _add_69;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(_add_70), "f"(hc_eps));
    float _fdiv_rn_15 = __fdiv_rn(_fdiv_rn_14, _add_69);
    float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 4);
    float add_14;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_14) : "f"(_fdiv_rn_15), "f"(_shfl_xor_sync_72));
    float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, add_14, 24);
    float _add_74;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(add_14), "f"(_shfl_xor_sync_75));
    float _add_73;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(_add_74), "f"(hc_eps));
    float _fdiv_rn_16 = __fdiv_rn(_fdiv_rn_15, _add_73);
    float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 1);
    float add_15;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_15) : "f"(_fdiv_rn_16), "f"(_shfl_xor_sync_76));
    float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, add_15, 2);
    float _add_78;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(add_15), "f"(_shfl_xor_sync_79));
    float _add_77;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(_add_78), "f"(hc_eps));
    float _fdiv_rn_17 = __fdiv_rn(_fdiv_rn_16, _add_77);
    float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 4);
    float add_16;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_16) : "f"(_fdiv_rn_17), "f"(_shfl_xor_sync_80));
    float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, add_16, 24);
    float _add_82;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(add_16), "f"(_shfl_xor_sync_83));
    float _add_81;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(_add_82), "f"(hc_eps));
    float _fdiv_rn_18 = __fdiv_rn(_fdiv_rn_17, _add_81);
    float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 1);
    float add_17;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_17) : "f"(_fdiv_rn_18), "f"(_shfl_xor_sync_84));
    float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, add_17, 2);
    float _add_86;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(add_17), "f"(_shfl_xor_sync_87));
    float _add_85;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(_add_86), "f"(hc_eps));
    float _fdiv_rn_19 = __fdiv_rn(_fdiv_rn_18, _add_85);
    float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 4);
    float add_18;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_18) : "f"(_fdiv_rn_19), "f"(_shfl_xor_sync_88));
    float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, add_18, 24);
    float _add_90;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_90) : "f"(add_18), "f"(_shfl_xor_sync_91));
    float _add_89;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(_add_90), "f"(hc_eps));
    float _fdiv_rn_20 = __fdiv_rn(_fdiv_rn_19, _add_89);
    float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 1);
    float add_19;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_19) : "f"(_fdiv_rn_20), "f"(_shfl_xor_sync_92));
    float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, add_19, 2);
    float _add_94;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(add_19), "f"(_shfl_xor_sync_95));
    float _add_93;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(_add_94), "f"(hc_eps));
    float _fdiv_rn_21 = __fdiv_rn(_fdiv_rn_20, _add_93);
    float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 4);
    float add_20;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_20) : "f"(_fdiv_rn_21), "f"(_shfl_xor_sync_96));
    float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, add_20, 24);
    float _add_98;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(add_20), "f"(_shfl_xor_sync_99));
    float _add_97;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(_add_98), "f"(hc_eps));
    float _fdiv_rn_22 = __fdiv_rn(_fdiv_rn_21, _add_97);
    float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 1);
    float add_21;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_21) : "f"(_fdiv_rn_22), "f"(_shfl_xor_sync_100));
    float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, add_21, 2);
    float _add_102;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(add_21), "f"(_shfl_xor_sync_103));
    float _add_101;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(_add_102), "f"(hc_eps));
    float _fdiv_rn_23 = __fdiv_rn(_fdiv_rn_22, _add_101);
    float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 4);
    float add_22;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_22) : "f"(_fdiv_rn_23), "f"(_shfl_xor_sync_104));
    float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, add_22, 24);
    float _add_106;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(add_22), "f"(_shfl_xor_sync_107));
    float _add_105;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_105) : "f"(_add_106), "f"(hc_eps));
    float _fdiv_rn_24 = __fdiv_rn(_fdiv_rn_23, _add_105);
    float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 1);
    float add_23;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_23) : "f"(_fdiv_rn_24), "f"(_shfl_xor_sync_108));
    float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, add_23, 2);
    float _add_110;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(add_23), "f"(_shfl_xor_sync_111));
    float _add_109;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(_add_110), "f"(hc_eps));
    float _fdiv_rn_25 = __fdiv_rn(_fdiv_rn_24, _add_109);
    float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 4);
    float add_24;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_24) : "f"(_fdiv_rn_25), "f"(_shfl_xor_sync_112));
    float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, add_24, 24);
    float _add_114;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(add_24), "f"(_shfl_xor_sync_115));
    float _add_113;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(_add_114), "f"(hc_eps));
    float _fdiv_rn_26 = __fdiv_rn(_fdiv_rn_25, _add_113);
    float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 1);
    float add_25;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_25) : "f"(_fdiv_rn_26), "f"(_shfl_xor_sync_116));
    float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, add_25, 2);
    float _add_118;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(add_25), "f"(_shfl_xor_sync_119));
    float _add_117;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(_add_118), "f"(hc_eps));
    float _fdiv_rn_27 = __fdiv_rn(_fdiv_rn_26, _add_117);
    float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 4);
    float add_26;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_26) : "f"(_fdiv_rn_27), "f"(_shfl_xor_sync_120));
    float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, add_26, 24);
    float _add_122;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(add_26), "f"(_shfl_xor_sync_123));
    float _add_121;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(_add_122), "f"(hc_eps));
    float _fdiv_rn_28 = __fdiv_rn(_fdiv_rn_27, _add_121);
    float _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 1);
    float add_27;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_27) : "f"(_fdiv_rn_28), "f"(_shfl_xor_sync_124));
    float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, add_27, 2);
    float _add_126;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(add_27), "f"(_shfl_xor_sync_127));
    float _add_125;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(_add_126), "f"(hc_eps));
    float _fdiv_rn_29 = __fdiv_rn(_fdiv_rn_28, _add_125);
    float _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 4);
    float add_28;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_28) : "f"(_fdiv_rn_29), "f"(_shfl_xor_sync_128));
    float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, add_28, 24);
    float _add_130;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(add_28), "f"(_shfl_xor_sync_131));
    float _add_129;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(_add_130), "f"(hc_eps));
    float _fdiv_rn_30 = __fdiv_rn(_fdiv_rn_29, _add_129);
    float _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 1);
    float add_29;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_29) : "f"(_fdiv_rn_30), "f"(_shfl_xor_sync_132));
    float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, add_29, 2);
    float _add_134;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(add_29), "f"(_shfl_xor_sync_135));
    float _add_133;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(_add_134), "f"(hc_eps));
    float _fdiv_rn_31 = __fdiv_rn(_fdiv_rn_30, _add_133);
    float _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 4);
    float add_30;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_30) : "f"(_fdiv_rn_31), "f"(_shfl_xor_sync_136));
    float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, add_30, 24);
    float _add_138;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_138) : "f"(add_30), "f"(_shfl_xor_sync_139));
    float _add_137;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(_add_138), "f"(hc_eps));
    float _fdiv_rn_32 = __fdiv_rn(_fdiv_rn_31, _add_137);
    float _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 1);
    float add_31;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_31) : "f"(_fdiv_rn_32), "f"(_shfl_xor_sync_140));
    float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, add_31, 2);
    float _add_142;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(add_31), "f"(_shfl_xor_sync_143));
    float _add_141;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(_add_142), "f"(hc_eps));
    float _fdiv_rn_33 = __fdiv_rn(_fdiv_rn_32, _add_141);
    float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 4);
    float add_32;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_32) : "f"(_fdiv_rn_33), "f"(_shfl_xor_sync_144));
    float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, add_32, 24);
    float _add_146;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(add_32), "f"(_shfl_xor_sync_147));
    float _add_145;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_145) : "f"(_add_146), "f"(hc_eps));
    float _fdiv_rn_34 = __fdiv_rn(_fdiv_rn_33, _add_145);
    float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 1);
    float add_33;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_33) : "f"(_fdiv_rn_34), "f"(_shfl_xor_sync_148));
    float _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, add_33, 2);
    float _add_150;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(add_33), "f"(_shfl_xor_sync_151));
    float _add_149;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149) : "f"(_add_150), "f"(hc_eps));
    float _fdiv_rn_35 = __fdiv_rn(_fdiv_rn_34, _add_149);
    float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 4);
    float add_34;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_34) : "f"(_fdiv_rn_35), "f"(_shfl_xor_sync_152));
    float _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, add_34, 24);
    float _add_154;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_154) : "f"(add_34), "f"(_shfl_xor_sync_155));
    float _add_153;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(_add_154), "f"(hc_eps));
    float _fdiv_rn_36 = __fdiv_rn(_fdiv_rn_35, _add_153);
    float _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 1);
    float add_35;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_35) : "f"(_fdiv_rn_36), "f"(_shfl_xor_sync_156));
    float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, add_35, 2);
    float _add_158;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_158) : "f"(add_35), "f"(_shfl_xor_sync_159));
    float _add_157;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(_add_158), "f"(hc_eps));
    float _fdiv_rn_37 = __fdiv_rn(_fdiv_rn_36, _add_157);
    float _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 4);
    float add_36;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_36) : "f"(_fdiv_rn_37), "f"(_shfl_xor_sync_160));
    float _shfl_xor_sync_163 = __shfl_xor_sync(0xFFFFFFFF, add_36, 24);
    float _add_162;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(add_36), "f"(_shfl_xor_sync_163));
    float _add_161;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(_add_162), "f"(hc_eps));
    float _fdiv_rn_38 = __fdiv_rn(_fdiv_rn_37, _add_161);
    float _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 1);
    float add_37;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_37) : "f"(_fdiv_rn_38), "f"(_shfl_xor_sync_164));
    float _shfl_xor_sync_167 = __shfl_xor_sync(0xFFFFFFFF, add_37, 2);
    float _add_166;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_166) : "f"(add_37), "f"(_shfl_xor_sync_167));
    float _add_165;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(_add_166), "f"(hc_eps));
    float _fdiv_rn_39 = __fdiv_rn(_fdiv_rn_38, _add_165);
    float _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 4);
    float add_38;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_38) : "f"(_fdiv_rn_39), "f"(_shfl_xor_sync_168));
    float _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, add_38, 24);
    float _add_170;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_170) : "f"(add_38), "f"(_shfl_xor_sync_171));
    float _add_169;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(_add_170), "f"(hc_eps));
    float _fdiv_rn_40 = __fdiv_rn(_fdiv_rn_39, _add_169);
    float pp = ((lane_0 < 4 || lane_0 >= 24 && lane_0 < 28) ? _add_1_1 : _mul_0);
    float coef0 = ((lane_0 < 8) ? pp : _fdiv_rn_40);
    float _shfl_2 = __shfl_sync(0xFFFFFFFF, pp, lane_0 + 24);
    int src_c = ((lane_0 < 16) ? lane_0 - 8 : lane_0 + 8);
    float _shfl_3 = __shfl_sync(0xFFFFFFFF, _fdiv_rn_40, src_c);
    float coef1 = ((lane_0 < 8) ? _shfl_2 : _shfl_3);
    float rr[2];
    float zz[2];
    float cc[2];
    rr[0] = r0;
    rr[1] = r1;
    zz[0] = z0;
    zz[1] = z1;
    cc[0] = coef0;
    cc[1] = coef1;
    #pragma unroll
    for (int j_1 = 0; j_1 < 2; j_1++) {
        int tok = tokr[j_1];
        long long tc64 = tcr[j_1];
        if (tok < T) {
            long long row = tc64 * 64;
            if (lane_0 == 0) {
                *(reinterpret_cast<float*>(stats + row) + (0)) = rr[j_1];
            }
            if (lane_0 < 24) {
                *(reinterpret_cast<float*>(stats + (row + 1 + lane64)) + (0)) = zz[j_1];
                *(reinterpret_cast<float*>(stats + (row + 25 + lane64)) + (0)) = cc[j_1];
            }
            if (lane_0 < 15) {
                *(reinterpret_cast<float*>(stats + (row + 49 + lane64)) + (0)) = 0.0f;
            }
        }
    }
}

} // extern "C"
