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
#define LAUNCH_MIN_BLOCKS 10

#include <math_constants.h>

extern "C" __device__ float __nv_expf(float);

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_mhc_train_fwd_coef__qe_mb10_tree(float* __restrict__ spart, float* __restrict__ qpart, float* __restrict__ scale, float* __restrict__ base, float* __restrict__ stats, int T, int p_k, int t_pad, float rms_eps, float hc_eps)
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
    float qe[2];
    #pragma unroll
    for (int j_1 = 0; j_1 < 2; j_1++) {
        qe[j_1] = qpart[(long long)kq * tp64 + tcr[j_1]];
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
    float qv_load0 = qe[0];
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
    float qv_load1 = qe[1];
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
    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
    float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, logit, 2);
    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, logit, 3);
    float _sub_9;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(logit), "f"(fmaxf(fmaxf(logit, _shfl_xor_sync_10), fmaxf(_shfl_xor_sync_11, _shfl_xor_sync_12))));
    float _exp_1 = __nv_expf(_sub_9);
    float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
    float _add_15;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(_exp_1), "f"(_shfl_xor_sync_16));
    float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 2);
    float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 3);
    float _add_17;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(_shfl_xor_sync_18), "f"(_shfl_xor_sync_19));
    float _add_14;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(_add_15), "f"(_add_17));
    float _fdiv_13 = __fdiv_rn(_exp_1, _add_14);
    float _add_2;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(_fdiv_13), "f"(hc_eps));
    float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 4);
    float _add_22;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(_add_2), "f"(_shfl_xor_sync_23));
    float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 24);
    float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 28);
    float _add_24;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(_shfl_xor_sync_25), "f"(_shfl_xor_sync_26));
    float _add_21;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(_add_22), "f"(_add_24));
    float _add_20;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(_add_21), "f"(hc_eps));
    float _fdiv_rn_2 = __fdiv_rn(_add_2, _add_20);
    float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 1);
    float _add_29;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(_fdiv_rn_2), "f"(_shfl_xor_sync_30));
    float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 2);
    float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 3);
    float _add_31;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(_shfl_xor_sync_32), "f"(_shfl_xor_sync_33));
    float _add_28;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(_add_29), "f"(_add_31));
    float _add_27;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(_add_28), "f"(hc_eps));
    float _fdiv_rn_3 = __fdiv_rn(_fdiv_rn_2, _add_27);
    float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 4);
    float _add_36;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(_fdiv_rn_3), "f"(_shfl_xor_sync_37));
    float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 24);
    float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 28);
    float _add_38;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(_shfl_xor_sync_39), "f"(_shfl_xor_sync_40));
    float _add_35;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(_add_36), "f"(_add_38));
    float _add_34;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(_add_35), "f"(hc_eps));
    float _fdiv_rn_4 = __fdiv_rn(_fdiv_rn_3, _add_34);
    float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 1);
    float _add_43;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(_fdiv_rn_4), "f"(_shfl_xor_sync_44));
    float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 2);
    float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 3);
    float _add_45;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(_shfl_xor_sync_46), "f"(_shfl_xor_sync_47));
    float _add_42;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(_add_43), "f"(_add_45));
    float _add_41;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(_add_42), "f"(hc_eps));
    float _fdiv_rn_5 = __fdiv_rn(_fdiv_rn_4, _add_41);
    float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 4);
    float _add_50;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(_fdiv_rn_5), "f"(_shfl_xor_sync_51));
    float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 24);
    float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 28);
    float _add_52;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(_shfl_xor_sync_53), "f"(_shfl_xor_sync_54));
    float _add_49;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(_add_50), "f"(_add_52));
    float _add_48;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(_add_49), "f"(hc_eps));
    float _fdiv_rn_6 = __fdiv_rn(_fdiv_rn_5, _add_48);
    float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 1);
    float _add_57;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(_fdiv_rn_6), "f"(_shfl_xor_sync_58));
    float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 2);
    float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 3);
    float _add_59;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(_shfl_xor_sync_60), "f"(_shfl_xor_sync_61));
    float _add_56;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(_add_57), "f"(_add_59));
    float _add_55;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(_add_56), "f"(hc_eps));
    float _fdiv_rn_7 = __fdiv_rn(_fdiv_rn_6, _add_55);
    float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 4);
    float _add_64;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(_fdiv_rn_7), "f"(_shfl_xor_sync_65));
    float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 24);
    float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 28);
    float _add_66;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(_shfl_xor_sync_67), "f"(_shfl_xor_sync_68));
    float _add_63;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(_add_64), "f"(_add_66));
    float _add_62;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(_add_63), "f"(hc_eps));
    float _fdiv_rn_8 = __fdiv_rn(_fdiv_rn_7, _add_62);
    float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 1);
    float _add_71;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(_fdiv_rn_8), "f"(_shfl_xor_sync_72));
    float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 2);
    float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 3);
    float _add_73;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(_shfl_xor_sync_74), "f"(_shfl_xor_sync_75));
    float _add_70;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(_add_71), "f"(_add_73));
    float _add_69;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(_add_70), "f"(hc_eps));
    float _fdiv_rn_9 = __fdiv_rn(_fdiv_rn_8, _add_69);
    float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 4);
    float _add_78;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(_fdiv_rn_9), "f"(_shfl_xor_sync_79));
    float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 24);
    float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 28);
    float _add_80;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(_shfl_xor_sync_81), "f"(_shfl_xor_sync_82));
    float _add_77;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(_add_78), "f"(_add_80));
    float _add_76;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(_add_77), "f"(hc_eps));
    float _fdiv_rn_10 = __fdiv_rn(_fdiv_rn_9, _add_76);
    float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 1);
    float _add_85;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(_fdiv_rn_10), "f"(_shfl_xor_sync_86));
    float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 2);
    float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 3);
    float _add_87;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(_shfl_xor_sync_88), "f"(_shfl_xor_sync_89));
    float _add_84;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(_add_85), "f"(_add_87));
    float _add_83;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(_add_84), "f"(hc_eps));
    float _fdiv_rn_11 = __fdiv_rn(_fdiv_rn_10, _add_83);
    float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 4);
    float _add_92;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(_fdiv_rn_11), "f"(_shfl_xor_sync_93));
    float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 24);
    float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 28);
    float _add_94;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(_shfl_xor_sync_95), "f"(_shfl_xor_sync_96));
    float _add_91;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(_add_92), "f"(_add_94));
    float _add_90;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_90) : "f"(_add_91), "f"(hc_eps));
    float _fdiv_rn_12 = __fdiv_rn(_fdiv_rn_11, _add_90);
    float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 1);
    float _add_99;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(_fdiv_rn_12), "f"(_shfl_xor_sync_100));
    float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 2);
    float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 3);
    float _add_101;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(_shfl_xor_sync_102), "f"(_shfl_xor_sync_103));
    float _add_98;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(_add_99), "f"(_add_101));
    float _add_97;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(_add_98), "f"(hc_eps));
    float _fdiv_rn_13 = __fdiv_rn(_fdiv_rn_12, _add_97);
    float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 4);
    float _add_106;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(_fdiv_rn_13), "f"(_shfl_xor_sync_107));
    float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 24);
    float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 28);
    float _add_108;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(_shfl_xor_sync_109), "f"(_shfl_xor_sync_110));
    float _add_105;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_105) : "f"(_add_106), "f"(_add_108));
    float _add_104;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(_add_105), "f"(hc_eps));
    float _fdiv_rn_14 = __fdiv_rn(_fdiv_rn_13, _add_104);
    float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 1);
    float _add_113;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(_fdiv_rn_14), "f"(_shfl_xor_sync_114));
    float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 2);
    float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 3);
    float _add_115;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(_shfl_xor_sync_116), "f"(_shfl_xor_sync_117));
    float _add_112;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(_add_113), "f"(_add_115));
    float _add_111;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(_add_112), "f"(hc_eps));
    float _fdiv_rn_15 = __fdiv_rn(_fdiv_rn_14, _add_111);
    float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 4);
    float _add_120;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(_fdiv_rn_15), "f"(_shfl_xor_sync_121));
    float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 24);
    float _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 28);
    float _add_122;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(_shfl_xor_sync_123), "f"(_shfl_xor_sync_124));
    float _add_119;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(_add_120), "f"(_add_122));
    float _add_118;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(_add_119), "f"(hc_eps));
    float _fdiv_rn_16 = __fdiv_rn(_fdiv_rn_15, _add_118);
    float _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 1);
    float _add_127;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(_fdiv_rn_16), "f"(_shfl_xor_sync_128));
    float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 2);
    float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 3);
    float _add_129;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(_shfl_xor_sync_130), "f"(_shfl_xor_sync_131));
    float _add_126;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(_add_127), "f"(_add_129));
    float _add_125;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(_add_126), "f"(hc_eps));
    float _fdiv_rn_17 = __fdiv_rn(_fdiv_rn_16, _add_125);
    float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 4);
    float _add_134;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(_fdiv_rn_17), "f"(_shfl_xor_sync_135));
    float _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 24);
    float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 28);
    float _add_136;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(_shfl_xor_sync_137), "f"(_shfl_xor_sync_138));
    float _add_133;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(_add_134), "f"(_add_136));
    float _add_132;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(_add_133), "f"(hc_eps));
    float _fdiv_rn_18 = __fdiv_rn(_fdiv_rn_17, _add_132);
    float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 1);
    float _add_141;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(_fdiv_rn_18), "f"(_shfl_xor_sync_142));
    float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 2);
    float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 3);
    float _add_143;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(_shfl_xor_sync_144), "f"(_shfl_xor_sync_145));
    float _add_140;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(_add_141), "f"(_add_143));
    float _add_139;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(_add_140), "f"(hc_eps));
    float _fdiv_rn_19 = __fdiv_rn(_fdiv_rn_18, _add_139);
    float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 4);
    float _add_148;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(_fdiv_rn_19), "f"(_shfl_xor_sync_149));
    float _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 24);
    float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 28);
    float _add_150;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(_shfl_xor_sync_151), "f"(_shfl_xor_sync_152));
    float _add_147;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(_add_148), "f"(_add_150));
    float _add_146;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(_add_147), "f"(hc_eps));
    float _fdiv_rn_20 = __fdiv_rn(_fdiv_rn_19, _add_146);
    float _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 1);
    float _add_155;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(_fdiv_rn_20), "f"(_shfl_xor_sync_156));
    float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 2);
    float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 3);
    float _add_157;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(_shfl_xor_sync_158), "f"(_shfl_xor_sync_159));
    float _add_154;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_154) : "f"(_add_155), "f"(_add_157));
    float _add_153;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(_add_154), "f"(hc_eps));
    float _fdiv_rn_21 = __fdiv_rn(_fdiv_rn_20, _add_153);
    float _shfl_xor_sync_163 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 4);
    float _add_162;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(_fdiv_rn_21), "f"(_shfl_xor_sync_163));
    float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 24);
    float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 28);
    float _add_164;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(_shfl_xor_sync_165), "f"(_shfl_xor_sync_166));
    float _add_161;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(_add_162), "f"(_add_164));
    float _add_160;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(_add_161), "f"(hc_eps));
    float _fdiv_rn_22 = __fdiv_rn(_fdiv_rn_21, _add_160);
    float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 1);
    float _add_169;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(_fdiv_rn_22), "f"(_shfl_xor_sync_170));
    float _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 2);
    float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 3);
    float _add_171;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171) : "f"(_shfl_xor_sync_172), "f"(_shfl_xor_sync_173));
    float _add_168;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(_add_169), "f"(_add_171));
    float _add_167;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(_add_168), "f"(hc_eps));
    float _fdiv_rn_23 = __fdiv_rn(_fdiv_rn_22, _add_167);
    float _shfl_xor_sync_177 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 4);
    float _add_176;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_176) : "f"(_fdiv_rn_23), "f"(_shfl_xor_sync_177));
    float _shfl_xor_sync_179 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 24);
    float _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 28);
    float _add_178;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_178) : "f"(_shfl_xor_sync_179), "f"(_shfl_xor_sync_180));
    float _add_175;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(_add_176), "f"(_add_178));
    float _add_174;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_174) : "f"(_add_175), "f"(hc_eps));
    float _fdiv_rn_24 = __fdiv_rn(_fdiv_rn_23, _add_174);
    float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 1);
    float _add_183;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(_fdiv_rn_24), "f"(_shfl_xor_sync_184));
    float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 2);
    float _shfl_xor_sync_187 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 3);
    float _add_185;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_185) : "f"(_shfl_xor_sync_186), "f"(_shfl_xor_sync_187));
    float _add_182;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_182) : "f"(_add_183), "f"(_add_185));
    float _add_181;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(_add_182), "f"(hc_eps));
    float _fdiv_rn_25 = __fdiv_rn(_fdiv_rn_24, _add_181);
    float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 4);
    float _add_190;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_190) : "f"(_fdiv_rn_25), "f"(_shfl_xor_sync_191));
    float _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 24);
    float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 28);
    float _add_192;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(_shfl_xor_sync_193), "f"(_shfl_xor_sync_194));
    float _add_189;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_189) : "f"(_add_190), "f"(_add_192));
    float _add_188;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(_add_189), "f"(hc_eps));
    float _fdiv_rn_26 = __fdiv_rn(_fdiv_rn_25, _add_188);
    float _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 1);
    float _add_197;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(_fdiv_rn_26), "f"(_shfl_xor_sync_198));
    float _shfl_xor_sync_200 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 2);
    float _shfl_xor_sync_201 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 3);
    float _add_199;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_199) : "f"(_shfl_xor_sync_200), "f"(_shfl_xor_sync_201));
    float _add_196;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_196) : "f"(_add_197), "f"(_add_199));
    float _add_195;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(_add_196), "f"(hc_eps));
    float _fdiv_rn_27 = __fdiv_rn(_fdiv_rn_26, _add_195);
    float _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 4);
    float _add_204;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204) : "f"(_fdiv_rn_27), "f"(_shfl_xor_sync_205));
    float _shfl_xor_sync_207 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 24);
    float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 28);
    float _add_206;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(_shfl_xor_sync_207), "f"(_shfl_xor_sync_208));
    float _add_203;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_203) : "f"(_add_204), "f"(_add_206));
    float _add_202;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(_add_203), "f"(hc_eps));
    float _fdiv_rn_28 = __fdiv_rn(_fdiv_rn_27, _add_202);
    float _shfl_xor_sync_212 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 1);
    float _add_211;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(_fdiv_rn_28), "f"(_shfl_xor_sync_212));
    float _shfl_xor_sync_214 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 2);
    float _shfl_xor_sync_215 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 3);
    float _add_213;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_213) : "f"(_shfl_xor_sync_214), "f"(_shfl_xor_sync_215));
    float _add_210;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_210) : "f"(_add_211), "f"(_add_213));
    float _add_209;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209) : "f"(_add_210), "f"(hc_eps));
    float _fdiv_rn_29 = __fdiv_rn(_fdiv_rn_28, _add_209);
    float _shfl_xor_sync_219 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 4);
    float _add_218;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_218) : "f"(_fdiv_rn_29), "f"(_shfl_xor_sync_219));
    float _shfl_xor_sync_221 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 24);
    float _shfl_xor_sync_222 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 28);
    float _add_220;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_220) : "f"(_shfl_xor_sync_221), "f"(_shfl_xor_sync_222));
    float _add_217;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_217) : "f"(_add_218), "f"(_add_220));
    float _add_216;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_216) : "f"(_add_217), "f"(hc_eps));
    float _fdiv_rn_30 = __fdiv_rn(_fdiv_rn_29, _add_216);
    float _shfl_xor_sync_226 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 1);
    float _add_225;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_225) : "f"(_fdiv_rn_30), "f"(_shfl_xor_sync_226));
    float _shfl_xor_sync_228 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 2);
    float _shfl_xor_sync_229 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 3);
    float _add_227;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_227) : "f"(_shfl_xor_sync_228), "f"(_shfl_xor_sync_229));
    float _add_224;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_224) : "f"(_add_225), "f"(_add_227));
    float _add_223;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_223) : "f"(_add_224), "f"(hc_eps));
    float _fdiv_rn_31 = __fdiv_rn(_fdiv_rn_30, _add_223);
    float _shfl_xor_sync_233 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 4);
    float _add_232;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_232) : "f"(_fdiv_rn_31), "f"(_shfl_xor_sync_233));
    float _shfl_xor_sync_235 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 24);
    float _shfl_xor_sync_236 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 28);
    float _add_234;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_234) : "f"(_shfl_xor_sync_235), "f"(_shfl_xor_sync_236));
    float _add_231;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_231) : "f"(_add_232), "f"(_add_234));
    float _add_230;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_230) : "f"(_add_231), "f"(hc_eps));
    float _fdiv_rn_32 = __fdiv_rn(_fdiv_rn_31, _add_230);
    float _shfl_xor_sync_240 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 1);
    float _add_239;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(_fdiv_rn_32), "f"(_shfl_xor_sync_240));
    float _shfl_xor_sync_242 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 2);
    float _shfl_xor_sync_243 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 3);
    float _add_241;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_241) : "f"(_shfl_xor_sync_242), "f"(_shfl_xor_sync_243));
    float _add_238;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_238) : "f"(_add_239), "f"(_add_241));
    float _add_237;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_237) : "f"(_add_238), "f"(hc_eps));
    float _fdiv_rn_33 = __fdiv_rn(_fdiv_rn_32, _add_237);
    float _shfl_xor_sync_247 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 4);
    float _add_246;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(_fdiv_rn_33), "f"(_shfl_xor_sync_247));
    float _shfl_xor_sync_249 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 24);
    float _shfl_xor_sync_250 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 28);
    float _add_248;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_248) : "f"(_shfl_xor_sync_249), "f"(_shfl_xor_sync_250));
    float _add_245;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_245) : "f"(_add_246), "f"(_add_248));
    float _add_244;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_244) : "f"(_add_245), "f"(hc_eps));
    float _fdiv_rn_34 = __fdiv_rn(_fdiv_rn_33, _add_244);
    float _shfl_xor_sync_254 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 1);
    float _add_253;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(_fdiv_rn_34), "f"(_shfl_xor_sync_254));
    float _shfl_xor_sync_256 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 2);
    float _shfl_xor_sync_257 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 3);
    float _add_255;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_255) : "f"(_shfl_xor_sync_256), "f"(_shfl_xor_sync_257));
    float _add_252;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_252) : "f"(_add_253), "f"(_add_255));
    float _add_251;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_251) : "f"(_add_252), "f"(hc_eps));
    float _fdiv_rn_35 = __fdiv_rn(_fdiv_rn_34, _add_251);
    float _shfl_xor_sync_261 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 4);
    float _add_260;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_260) : "f"(_fdiv_rn_35), "f"(_shfl_xor_sync_261));
    float _shfl_xor_sync_263 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 24);
    float _shfl_xor_sync_264 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 28);
    float _add_262;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_262) : "f"(_shfl_xor_sync_263), "f"(_shfl_xor_sync_264));
    float _add_259;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_259) : "f"(_add_260), "f"(_add_262));
    float _add_258;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_258) : "f"(_add_259), "f"(hc_eps));
    float _fdiv_rn_36 = __fdiv_rn(_fdiv_rn_35, _add_258);
    float _shfl_xor_sync_268 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 1);
    float _add_267;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_267) : "f"(_fdiv_rn_36), "f"(_shfl_xor_sync_268));
    float _shfl_xor_sync_270 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 2);
    float _shfl_xor_sync_271 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 3);
    float _add_269;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_269) : "f"(_shfl_xor_sync_270), "f"(_shfl_xor_sync_271));
    float _add_266;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_266) : "f"(_add_267), "f"(_add_269));
    float _add_265;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_265) : "f"(_add_266), "f"(hc_eps));
    float _fdiv_rn_37 = __fdiv_rn(_fdiv_rn_36, _add_265);
    float _shfl_xor_sync_275 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 4);
    float _add_274;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_274) : "f"(_fdiv_rn_37), "f"(_shfl_xor_sync_275));
    float _shfl_xor_sync_277 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 24);
    float _shfl_xor_sync_278 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 28);
    float _add_276;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_276) : "f"(_shfl_xor_sync_277), "f"(_shfl_xor_sync_278));
    float _add_273;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_273) : "f"(_add_274), "f"(_add_276));
    float _add_272;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_272) : "f"(_add_273), "f"(hc_eps));
    float _fdiv_rn_38 = __fdiv_rn(_fdiv_rn_37, _add_272);
    float _shfl_xor_sync_282 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 1);
    float _add_281;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_281) : "f"(_fdiv_rn_38), "f"(_shfl_xor_sync_282));
    float _shfl_xor_sync_284 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 2);
    float _shfl_xor_sync_285 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 3);
    float _add_283;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_283) : "f"(_shfl_xor_sync_284), "f"(_shfl_xor_sync_285));
    float _add_280;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_280) : "f"(_add_281), "f"(_add_283));
    float _add_279;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_279) : "f"(_add_280), "f"(hc_eps));
    float _fdiv_rn_39 = __fdiv_rn(_fdiv_rn_38, _add_279);
    float _shfl_xor_sync_289 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 4);
    float _add_288;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_288) : "f"(_fdiv_rn_39), "f"(_shfl_xor_sync_289));
    float _shfl_xor_sync_291 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 24);
    float _shfl_xor_sync_292 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 28);
    float _add_290;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_290) : "f"(_shfl_xor_sync_291), "f"(_shfl_xor_sync_292));
    float _add_287;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_287) : "f"(_add_288), "f"(_add_290));
    float _add_286;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_286) : "f"(_add_287), "f"(hc_eps));
    float _fdiv_rn_40 = __fdiv_rn(_fdiv_rn_39, _add_286);
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
    for (int j_2 = 0; j_2 < 2; j_2++) {
        int tok = tokr[j_2];
        long long tc64 = tcr[j_2];
        if (tok < T) {
            long long row = tc64 * 64;
            if (lane_0 == 0) {
                *(reinterpret_cast<float*>(stats + row) + (0)) = rr[j_2];
            }
            if (lane_0 < 24) {
                *(reinterpret_cast<float*>(stats + (row + 1 + lane64)) + (0)) = zz[j_2];
                *(reinterpret_cast<float*>(stats + (row + 25 + lane64)) + (0)) = cc[j_2];
            }
            if (lane_0 < 15) {
                *(reinterpret_cast<float*>(stats + (row + 49 + lane64)) + (0)) = 0.0f;
            }
        }
    }
}

} // extern "C"
