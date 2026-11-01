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
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 10

#include <math_constants.h>

extern "C" __device__ float __nv_expf(float);

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_mhc_det_train_fwd_coef__qe_mb10_tree__h5120(float* __restrict__ spart, float* __restrict__ qpart, float* __restrict__ scale, float* __restrict__ base, float* __restrict__ stats, int T, int p_k, int t_pad, float rms_eps, float hc_eps)
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
    float _warp_reduce_3 = qv0;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
    float _sqrt_2;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_2) : "f"((float)((double)_warp_reduce_3 * 4.8828125e-05 + (double)rms_eps)));
    float r0 = __frcp_rn(_sqrt_2);
    float z0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(z0) : "f"(r0), "f"(s_acc[0]));
    float a0;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a0) : "f"(z0), "f"(sc), "f"(bs));
    float qv_load1 = qe[1];
    float qv1 = ((lane_0 < p_k) ? qv_load1 : 0.0f);
    float _warp_reduce_5 = qv1;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
    float _sqrt_4;
    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_4) : "f"((float)((double)_warp_reduce_5 * 4.8828125e-05 + (double)rms_eps)));
    float r1 = __frcp_rn(_sqrt_4);
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
    float _sub_6;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(0.0f), "f"(fabsf(a_sig)));
    float _exp_0 = __nv_expf(_sub_6);
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
    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, logit, 2);
    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, logit, 3);
    float _sub_7;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(logit), "f"(fmaxf(fmaxf(logit, _shfl_xor_sync_8), fmaxf(_shfl_xor_sync_9, _shfl_xor_sync_10))));
    float _exp_1 = __nv_expf(_sub_7);
    float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
    float _add_13;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(_exp_1), "f"(_shfl_xor_sync_14));
    float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 2);
    float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 3);
    float _add_15;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(_shfl_xor_sync_16), "f"(_shfl_xor_sync_17));
    float _add_12;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(_add_13), "f"(_add_15));
    float _fdiv_11 = __fdiv_rn(_exp_1, _add_12);
    float _add_2;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(_fdiv_11), "f"(hc_eps));
    float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 4);
    float _add_20;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(_add_2), "f"(_shfl_xor_sync_21));
    float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 24);
    float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 28);
    float _add_22;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(_shfl_xor_sync_23), "f"(_shfl_xor_sync_24));
    float _add_19;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(_add_20), "f"(_add_22));
    float _add_18;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(_add_19), "f"(hc_eps));
    float _fdiv_rn_2 = __fdiv_rn(_add_2, _add_18);
    float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 1);
    float _add_27;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(_fdiv_rn_2), "f"(_shfl_xor_sync_28));
    float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 2);
    float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 3);
    float _add_29;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(_shfl_xor_sync_30), "f"(_shfl_xor_sync_31));
    float _add_26;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(_add_27), "f"(_add_29));
    float _add_25;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(_add_26), "f"(hc_eps));
    float _fdiv_rn_3 = __fdiv_rn(_fdiv_rn_2, _add_25);
    float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 4);
    float _add_34;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(_fdiv_rn_3), "f"(_shfl_xor_sync_35));
    float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 24);
    float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 28);
    float _add_36;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(_shfl_xor_sync_37), "f"(_shfl_xor_sync_38));
    float _add_33;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(_add_34), "f"(_add_36));
    float _add_32;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(_add_33), "f"(hc_eps));
    float _fdiv_rn_4 = __fdiv_rn(_fdiv_rn_3, _add_32);
    float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 1);
    float _add_41;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(_fdiv_rn_4), "f"(_shfl_xor_sync_42));
    float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 2);
    float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 3);
    float _add_43;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(_shfl_xor_sync_44), "f"(_shfl_xor_sync_45));
    float _add_40;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(_add_41), "f"(_add_43));
    float _add_39;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(_add_40), "f"(hc_eps));
    float _fdiv_rn_5 = __fdiv_rn(_fdiv_rn_4, _add_39);
    float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 4);
    float _add_48;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(_fdiv_rn_5), "f"(_shfl_xor_sync_49));
    float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 24);
    float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 28);
    float _add_50;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(_shfl_xor_sync_51), "f"(_shfl_xor_sync_52));
    float _add_47;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(_add_48), "f"(_add_50));
    float _add_46;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(_add_47), "f"(hc_eps));
    float _fdiv_rn_6 = __fdiv_rn(_fdiv_rn_5, _add_46);
    float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 1);
    float _add_55;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(_fdiv_rn_6), "f"(_shfl_xor_sync_56));
    float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 2);
    float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 3);
    float _add_57;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(_shfl_xor_sync_58), "f"(_shfl_xor_sync_59));
    float _add_54;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(_add_55), "f"(_add_57));
    float _add_53;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(_add_54), "f"(hc_eps));
    float _fdiv_rn_7 = __fdiv_rn(_fdiv_rn_6, _add_53);
    float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 4);
    float _add_62;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(_fdiv_rn_7), "f"(_shfl_xor_sync_63));
    float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 24);
    float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 28);
    float _add_64;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(_shfl_xor_sync_65), "f"(_shfl_xor_sync_66));
    float _add_61;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(_add_62), "f"(_add_64));
    float _add_60;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(_add_61), "f"(hc_eps));
    float _fdiv_rn_8 = __fdiv_rn(_fdiv_rn_7, _add_60);
    float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 1);
    float _add_69;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(_fdiv_rn_8), "f"(_shfl_xor_sync_70));
    float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 2);
    float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 3);
    float _add_71;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(_shfl_xor_sync_72), "f"(_shfl_xor_sync_73));
    float _add_68;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(_add_69), "f"(_add_71));
    float _add_67;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(_add_68), "f"(hc_eps));
    float _fdiv_rn_9 = __fdiv_rn(_fdiv_rn_8, _add_67);
    float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 4);
    float _add_76;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(_fdiv_rn_9), "f"(_shfl_xor_sync_77));
    float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 24);
    float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 28);
    float _add_78;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(_shfl_xor_sync_79), "f"(_shfl_xor_sync_80));
    float _add_75;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(_add_76), "f"(_add_78));
    float _add_74;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(_add_75), "f"(hc_eps));
    float _fdiv_rn_10 = __fdiv_rn(_fdiv_rn_9, _add_74);
    float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 1);
    float _add_83;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(_fdiv_rn_10), "f"(_shfl_xor_sync_84));
    float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 2);
    float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 3);
    float _add_85;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(_shfl_xor_sync_86), "f"(_shfl_xor_sync_87));
    float _add_82;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(_add_83), "f"(_add_85));
    float _add_81;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(_add_82), "f"(hc_eps));
    float _fdiv_rn_11 = __fdiv_rn(_fdiv_rn_10, _add_81);
    float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 4);
    float _add_90;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_90) : "f"(_fdiv_rn_11), "f"(_shfl_xor_sync_91));
    float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 24);
    float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 28);
    float _add_92;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(_shfl_xor_sync_93), "f"(_shfl_xor_sync_94));
    float _add_89;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(_add_90), "f"(_add_92));
    float _add_88;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(_add_89), "f"(hc_eps));
    float _fdiv_rn_12 = __fdiv_rn(_fdiv_rn_11, _add_88);
    float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 1);
    float _add_97;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(_fdiv_rn_12), "f"(_shfl_xor_sync_98));
    float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 2);
    float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 3);
    float _add_99;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(_shfl_xor_sync_100), "f"(_shfl_xor_sync_101));
    float _add_96;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(_add_97), "f"(_add_99));
    float _add_95;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(_add_96), "f"(hc_eps));
    float _fdiv_rn_13 = __fdiv_rn(_fdiv_rn_12, _add_95);
    float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 4);
    float _add_104;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(_fdiv_rn_13), "f"(_shfl_xor_sync_105));
    float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 24);
    float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 28);
    float _add_106;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(_shfl_xor_sync_107), "f"(_shfl_xor_sync_108));
    float _add_103;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(_add_104), "f"(_add_106));
    float _add_102;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(_add_103), "f"(hc_eps));
    float _fdiv_rn_14 = __fdiv_rn(_fdiv_rn_13, _add_102);
    float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 1);
    float _add_111;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(_fdiv_rn_14), "f"(_shfl_xor_sync_112));
    float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 2);
    float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 3);
    float _add_113;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(_shfl_xor_sync_114), "f"(_shfl_xor_sync_115));
    float _add_110;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(_add_111), "f"(_add_113));
    float _add_109;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(_add_110), "f"(hc_eps));
    float _fdiv_rn_15 = __fdiv_rn(_fdiv_rn_14, _add_109);
    float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 4);
    float _add_118;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(_fdiv_rn_15), "f"(_shfl_xor_sync_119));
    float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 24);
    float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 28);
    float _add_120;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(_shfl_xor_sync_121), "f"(_shfl_xor_sync_122));
    float _add_117;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(_add_118), "f"(_add_120));
    float _add_116;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(_add_117), "f"(hc_eps));
    float _fdiv_rn_16 = __fdiv_rn(_fdiv_rn_15, _add_116);
    float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 1);
    float _add_125;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(_fdiv_rn_16), "f"(_shfl_xor_sync_126));
    float _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 2);
    float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 3);
    float _add_127;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(_shfl_xor_sync_128), "f"(_shfl_xor_sync_129));
    float _add_124;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(_add_125), "f"(_add_127));
    float _add_123;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(_add_124), "f"(hc_eps));
    float _fdiv_rn_17 = __fdiv_rn(_fdiv_rn_16, _add_123);
    float _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 4);
    float _add_132;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(_fdiv_rn_17), "f"(_shfl_xor_sync_133));
    float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 24);
    float _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 28);
    float _add_134;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(_shfl_xor_sync_135), "f"(_shfl_xor_sync_136));
    float _add_131;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(_add_132), "f"(_add_134));
    float _add_130;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(_add_131), "f"(hc_eps));
    float _fdiv_rn_18 = __fdiv_rn(_fdiv_rn_17, _add_130);
    float _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 1);
    float _add_139;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(_fdiv_rn_18), "f"(_shfl_xor_sync_140));
    float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 2);
    float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 3);
    float _add_141;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(_shfl_xor_sync_142), "f"(_shfl_xor_sync_143));
    float _add_138;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_138) : "f"(_add_139), "f"(_add_141));
    float _add_137;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(_add_138), "f"(hc_eps));
    float _fdiv_rn_19 = __fdiv_rn(_fdiv_rn_18, _add_137);
    float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 4);
    float _add_146;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(_fdiv_rn_19), "f"(_shfl_xor_sync_147));
    float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 24);
    float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 28);
    float _add_148;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(_shfl_xor_sync_149), "f"(_shfl_xor_sync_150));
    float _add_145;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_145) : "f"(_add_146), "f"(_add_148));
    float _add_144;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(_add_145), "f"(hc_eps));
    float _fdiv_rn_20 = __fdiv_rn(_fdiv_rn_19, _add_144);
    float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 1);
    float _add_153;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(_fdiv_rn_20), "f"(_shfl_xor_sync_154));
    float _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 2);
    float _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 3);
    float _add_155;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(_shfl_xor_sync_156), "f"(_shfl_xor_sync_157));
    float _add_152;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(_add_153), "f"(_add_155));
    float _add_151;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(_add_152), "f"(hc_eps));
    float _fdiv_rn_21 = __fdiv_rn(_fdiv_rn_20, _add_151);
    float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 4);
    float _add_160;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(_fdiv_rn_21), "f"(_shfl_xor_sync_161));
    float _shfl_xor_sync_163 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 24);
    float _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 28);
    float _add_162;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(_shfl_xor_sync_163), "f"(_shfl_xor_sync_164));
    float _add_159;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159) : "f"(_add_160), "f"(_add_162));
    float _add_158;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_158) : "f"(_add_159), "f"(hc_eps));
    float _fdiv_rn_22 = __fdiv_rn(_fdiv_rn_21, _add_158);
    float _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 1);
    float _add_167;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(_fdiv_rn_22), "f"(_shfl_xor_sync_168));
    float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 2);
    float _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 3);
    float _add_169;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(_shfl_xor_sync_170), "f"(_shfl_xor_sync_171));
    float _add_166;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_166) : "f"(_add_167), "f"(_add_169));
    float _add_165;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(_add_166), "f"(hc_eps));
    float _fdiv_rn_23 = __fdiv_rn(_fdiv_rn_22, _add_165);
    float _shfl_xor_sync_175 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 4);
    float _add_174;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_174) : "f"(_fdiv_rn_23), "f"(_shfl_xor_sync_175));
    float _shfl_xor_sync_177 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 24);
    float _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 28);
    float _add_176;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_176) : "f"(_shfl_xor_sync_177), "f"(_shfl_xor_sync_178));
    float _add_173;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_173) : "f"(_add_174), "f"(_add_176));
    float _add_172;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(_add_173), "f"(hc_eps));
    float _fdiv_rn_24 = __fdiv_rn(_fdiv_rn_23, _add_172);
    float _shfl_xor_sync_182 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 1);
    float _add_181;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(_fdiv_rn_24), "f"(_shfl_xor_sync_182));
    float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 2);
    float _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 3);
    float _add_183;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(_shfl_xor_sync_184), "f"(_shfl_xor_sync_185));
    float _add_180;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_180) : "f"(_add_181), "f"(_add_183));
    float _add_179;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179) : "f"(_add_180), "f"(hc_eps));
    float _fdiv_rn_25 = __fdiv_rn(_fdiv_rn_24, _add_179);
    float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 4);
    float _add_188;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(_fdiv_rn_25), "f"(_shfl_xor_sync_189));
    float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 24);
    float _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 28);
    float _add_190;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_190) : "f"(_shfl_xor_sync_191), "f"(_shfl_xor_sync_192));
    float _add_187;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_187) : "f"(_add_188), "f"(_add_190));
    float _add_186;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(_add_187), "f"(hc_eps));
    float _fdiv_rn_26 = __fdiv_rn(_fdiv_rn_25, _add_186);
    float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 1);
    float _add_195;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(_fdiv_rn_26), "f"(_shfl_xor_sync_196));
    float _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 2);
    float _shfl_xor_sync_199 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 3);
    float _add_197;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(_shfl_xor_sync_198), "f"(_shfl_xor_sync_199));
    float _add_194;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_194) : "f"(_add_195), "f"(_add_197));
    float _add_193;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_193) : "f"(_add_194), "f"(hc_eps));
    float _fdiv_rn_27 = __fdiv_rn(_fdiv_rn_26, _add_193);
    float _shfl_xor_sync_203 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 4);
    float _add_202;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(_fdiv_rn_27), "f"(_shfl_xor_sync_203));
    float _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 24);
    float _shfl_xor_sync_206 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 28);
    float _add_204;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204) : "f"(_shfl_xor_sync_205), "f"(_shfl_xor_sync_206));
    float _add_201;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_201) : "f"(_add_202), "f"(_add_204));
    float _add_200;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_200) : "f"(_add_201), "f"(hc_eps));
    float _fdiv_rn_28 = __fdiv_rn(_fdiv_rn_27, _add_200);
    float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 1);
    float _add_209;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209) : "f"(_fdiv_rn_28), "f"(_shfl_xor_sync_210));
    float _shfl_xor_sync_212 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 2);
    float _shfl_xor_sync_213 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 3);
    float _add_211;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(_shfl_xor_sync_212), "f"(_shfl_xor_sync_213));
    float _add_208;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(_add_209), "f"(_add_211));
    float _add_207;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_207) : "f"(_add_208), "f"(hc_eps));
    float _fdiv_rn_29 = __fdiv_rn(_fdiv_rn_28, _add_207);
    float _shfl_xor_sync_217 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 4);
    float _add_216;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_216) : "f"(_fdiv_rn_29), "f"(_shfl_xor_sync_217));
    float _shfl_xor_sync_219 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 24);
    float _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 28);
    float _add_218;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_218) : "f"(_shfl_xor_sync_219), "f"(_shfl_xor_sync_220));
    float _add_215;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_215) : "f"(_add_216), "f"(_add_218));
    float _add_214;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_214) : "f"(_add_215), "f"(hc_eps));
    float _fdiv_rn_30 = __fdiv_rn(_fdiv_rn_29, _add_214);
    float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 1);
    float _add_223;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_223) : "f"(_fdiv_rn_30), "f"(_shfl_xor_sync_224));
    float _shfl_xor_sync_226 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 2);
    float _shfl_xor_sync_227 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 3);
    float _add_225;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_225) : "f"(_shfl_xor_sync_226), "f"(_shfl_xor_sync_227));
    float _add_222;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_222) : "f"(_add_223), "f"(_add_225));
    float _add_221;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_221) : "f"(_add_222), "f"(hc_eps));
    float _fdiv_rn_31 = __fdiv_rn(_fdiv_rn_30, _add_221);
    float _shfl_xor_sync_231 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 4);
    float _add_230;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_230) : "f"(_fdiv_rn_31), "f"(_shfl_xor_sync_231));
    float _shfl_xor_sync_233 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 24);
    float _shfl_xor_sync_234 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 28);
    float _add_232;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_232) : "f"(_shfl_xor_sync_233), "f"(_shfl_xor_sync_234));
    float _add_229;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_229) : "f"(_add_230), "f"(_add_232));
    float _add_228;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_228) : "f"(_add_229), "f"(hc_eps));
    float _fdiv_rn_32 = __fdiv_rn(_fdiv_rn_31, _add_228);
    float _shfl_xor_sync_238 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 1);
    float _add_237;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_237) : "f"(_fdiv_rn_32), "f"(_shfl_xor_sync_238));
    float _shfl_xor_sync_240 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 2);
    float _shfl_xor_sync_241 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 3);
    float _add_239;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(_shfl_xor_sync_240), "f"(_shfl_xor_sync_241));
    float _add_236;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_236) : "f"(_add_237), "f"(_add_239));
    float _add_235;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_235) : "f"(_add_236), "f"(hc_eps));
    float _fdiv_rn_33 = __fdiv_rn(_fdiv_rn_32, _add_235);
    float _shfl_xor_sync_245 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 4);
    float _add_244;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_244) : "f"(_fdiv_rn_33), "f"(_shfl_xor_sync_245));
    float _shfl_xor_sync_247 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 24);
    float _shfl_xor_sync_248 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 28);
    float _add_246;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(_shfl_xor_sync_247), "f"(_shfl_xor_sync_248));
    float _add_243;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_243) : "f"(_add_244), "f"(_add_246));
    float _add_242;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_242) : "f"(_add_243), "f"(hc_eps));
    float _fdiv_rn_34 = __fdiv_rn(_fdiv_rn_33, _add_242);
    float _shfl_xor_sync_252 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 1);
    float _add_251;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_251) : "f"(_fdiv_rn_34), "f"(_shfl_xor_sync_252));
    float _shfl_xor_sync_254 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 2);
    float _shfl_xor_sync_255 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 3);
    float _add_253;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(_shfl_xor_sync_254), "f"(_shfl_xor_sync_255));
    float _add_250;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_250) : "f"(_add_251), "f"(_add_253));
    float _add_249;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_249) : "f"(_add_250), "f"(hc_eps));
    float _fdiv_rn_35 = __fdiv_rn(_fdiv_rn_34, _add_249);
    float _shfl_xor_sync_259 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 4);
    float _add_258;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_258) : "f"(_fdiv_rn_35), "f"(_shfl_xor_sync_259));
    float _shfl_xor_sync_261 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 24);
    float _shfl_xor_sync_262 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 28);
    float _add_260;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_260) : "f"(_shfl_xor_sync_261), "f"(_shfl_xor_sync_262));
    float _add_257;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_257) : "f"(_add_258), "f"(_add_260));
    float _add_256;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_256) : "f"(_add_257), "f"(hc_eps));
    float _fdiv_rn_36 = __fdiv_rn(_fdiv_rn_35, _add_256);
    float _shfl_xor_sync_266 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 1);
    float _add_265;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_265) : "f"(_fdiv_rn_36), "f"(_shfl_xor_sync_266));
    float _shfl_xor_sync_268 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 2);
    float _shfl_xor_sync_269 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 3);
    float _add_267;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_267) : "f"(_shfl_xor_sync_268), "f"(_shfl_xor_sync_269));
    float _add_264;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_264) : "f"(_add_265), "f"(_add_267));
    float _add_263;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_263) : "f"(_add_264), "f"(hc_eps));
    float _fdiv_rn_37 = __fdiv_rn(_fdiv_rn_36, _add_263);
    float _shfl_xor_sync_273 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 4);
    float _add_272;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_272) : "f"(_fdiv_rn_37), "f"(_shfl_xor_sync_273));
    float _shfl_xor_sync_275 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 24);
    float _shfl_xor_sync_276 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 28);
    float _add_274;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_274) : "f"(_shfl_xor_sync_275), "f"(_shfl_xor_sync_276));
    float _add_271;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_271) : "f"(_add_272), "f"(_add_274));
    float _add_270;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_270) : "f"(_add_271), "f"(hc_eps));
    float _fdiv_rn_38 = __fdiv_rn(_fdiv_rn_37, _add_270);
    float _shfl_xor_sync_280 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 1);
    float _add_279;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_279) : "f"(_fdiv_rn_38), "f"(_shfl_xor_sync_280));
    float _shfl_xor_sync_282 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 2);
    float _shfl_xor_sync_283 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 3);
    float _add_281;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_281) : "f"(_shfl_xor_sync_282), "f"(_shfl_xor_sync_283));
    float _add_278;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_278) : "f"(_add_279), "f"(_add_281));
    float _add_277;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_277) : "f"(_add_278), "f"(hc_eps));
    float _fdiv_rn_39 = __fdiv_rn(_fdiv_rn_38, _add_277);
    float _shfl_xor_sync_287 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 4);
    float _add_286;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_286) : "f"(_fdiv_rn_39), "f"(_shfl_xor_sync_287));
    float _shfl_xor_sync_289 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 24);
    float _shfl_xor_sync_290 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 28);
    float _add_288;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_288) : "f"(_shfl_xor_sync_289), "f"(_shfl_xor_sync_290));
    float _add_285;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_285) : "f"(_add_286), "f"(_add_288));
    float _add_284;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_284) : "f"(_add_285), "f"(hc_eps));
    float _fdiv_rn_40 = __fdiv_rn(_fdiv_rn_39, _add_284);
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
