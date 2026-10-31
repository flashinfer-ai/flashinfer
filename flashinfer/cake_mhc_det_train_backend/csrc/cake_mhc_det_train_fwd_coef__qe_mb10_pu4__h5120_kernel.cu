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
kernel_cake_mhc_det_train_fwd_coef__qe_mb10_pu4__h5120(float* __restrict__ spart, float* __restrict__ qpart, float* __restrict__ scale, float* __restrict__ base, float* __restrict__ stats, int T, int p_k, int t_pad, float rms_eps, float hc_eps)
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
    #pragma unroll 4
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
    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
    float _max_0 = fmaxf(logit, _shfl_xor_sync_7);
    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, _max_0, 2);
    float _sub_8;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(logit), "f"(fmaxf(_max_0, _shfl_xor_sync_9)));
    float _exp_1 = __nv_expf(_sub_8);
    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
    float add;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(_exp_1), "f"(_shfl_xor_sync_10));
    float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, add, 2);
    float _add_12;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(add), "f"(_shfl_xor_sync_13));
    float _fdiv_11 = __fdiv_rn(_exp_1, _add_12);
    float _add_3;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(_fdiv_11), "f"(hc_eps));
    float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, _add_3, 4);
    float add_0;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_0) : "f"(_add_3), "f"(_shfl_xor_sync_14));
    float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, add_0, 24);
    float _add_16;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(add_0), "f"(_shfl_xor_sync_17));
    float _add_15;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(_add_16), "f"(hc_eps));
    float _fdiv_rn_2 = __fdiv_rn(_add_3, _add_15);
    float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 1);
    float add_1;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(_fdiv_rn_2), "f"(_shfl_xor_sync_18));
    float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, add_1, 2);
    float _add_20;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(add_1), "f"(_shfl_xor_sync_21));
    float _add_19;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(_add_20), "f"(hc_eps));
    float _fdiv_rn_3 = __fdiv_rn(_fdiv_rn_2, _add_19);
    float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 4);
    float add_2;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(_fdiv_rn_3), "f"(_shfl_xor_sync_22));
    float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, add_2, 24);
    float _add_24;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(add_2), "f"(_shfl_xor_sync_25));
    float _add_23;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(_add_24), "f"(hc_eps));
    float _fdiv_rn_4 = __fdiv_rn(_fdiv_rn_3, _add_23);
    float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 1);
    float add_3;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(_fdiv_rn_4), "f"(_shfl_xor_sync_26));
    float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, add_3, 2);
    float _add_28;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(add_3), "f"(_shfl_xor_sync_29));
    float _add_27;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(_add_28), "f"(hc_eps));
    float _fdiv_rn_5 = __fdiv_rn(_fdiv_rn_4, _add_27);
    float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 4);
    float add_4;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(_fdiv_rn_5), "f"(_shfl_xor_sync_30));
    float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, add_4, 24);
    float _add_32;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(add_4), "f"(_shfl_xor_sync_33));
    float _add_31;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(_add_32), "f"(hc_eps));
    float _fdiv_rn_6 = __fdiv_rn(_fdiv_rn_5, _add_31);
    float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 1);
    float add_5;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(_fdiv_rn_6), "f"(_shfl_xor_sync_34));
    float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, add_5, 2);
    float _add_36;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(add_5), "f"(_shfl_xor_sync_37));
    float _add_35;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(_add_36), "f"(hc_eps));
    float _fdiv_rn_7 = __fdiv_rn(_fdiv_rn_6, _add_35);
    float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 4);
    float add_6;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_6) : "f"(_fdiv_rn_7), "f"(_shfl_xor_sync_38));
    float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, add_6, 24);
    float _add_40;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(add_6), "f"(_shfl_xor_sync_41));
    float _add_39;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(_add_40), "f"(hc_eps));
    float _fdiv_rn_8 = __fdiv_rn(_fdiv_rn_7, _add_39);
    float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 1);
    float add_7;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(_fdiv_rn_8), "f"(_shfl_xor_sync_42));
    float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, add_7, 2);
    float _add_44;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(add_7), "f"(_shfl_xor_sync_45));
    float _add_43;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(_add_44), "f"(hc_eps));
    float _fdiv_rn_9 = __fdiv_rn(_fdiv_rn_8, _add_43);
    float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 4);
    float add_8;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(_fdiv_rn_9), "f"(_shfl_xor_sync_46));
    float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, add_8, 24);
    float _add_48;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(add_8), "f"(_shfl_xor_sync_49));
    float _add_47;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(_add_48), "f"(hc_eps));
    float _fdiv_rn_10 = __fdiv_rn(_fdiv_rn_9, _add_47);
    float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 1);
    float add_9;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_9) : "f"(_fdiv_rn_10), "f"(_shfl_xor_sync_50));
    float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, add_9, 2);
    float _add_52;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(add_9), "f"(_shfl_xor_sync_53));
    float _add_51;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(_add_52), "f"(hc_eps));
    float _fdiv_rn_11 = __fdiv_rn(_fdiv_rn_10, _add_51);
    float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 4);
    float add_10;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_10) : "f"(_fdiv_rn_11), "f"(_shfl_xor_sync_54));
    float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, add_10, 24);
    float _add_56;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(add_10), "f"(_shfl_xor_sync_57));
    float _add_55;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(_add_56), "f"(hc_eps));
    float _fdiv_rn_12 = __fdiv_rn(_fdiv_rn_11, _add_55);
    float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 1);
    float add_11;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_11) : "f"(_fdiv_rn_12), "f"(_shfl_xor_sync_58));
    float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, add_11, 2);
    float _add_60;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(add_11), "f"(_shfl_xor_sync_61));
    float _add_59;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(_add_60), "f"(hc_eps));
    float _fdiv_rn_13 = __fdiv_rn(_fdiv_rn_12, _add_59);
    float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 4);
    float add_12;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_12) : "f"(_fdiv_rn_13), "f"(_shfl_xor_sync_62));
    float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, add_12, 24);
    float _add_64;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(add_12), "f"(_shfl_xor_sync_65));
    float _add_63;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(_add_64), "f"(hc_eps));
    float _fdiv_rn_14 = __fdiv_rn(_fdiv_rn_13, _add_63);
    float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 1);
    float add_13;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_13) : "f"(_fdiv_rn_14), "f"(_shfl_xor_sync_66));
    float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, add_13, 2);
    float _add_68;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(add_13), "f"(_shfl_xor_sync_69));
    float _add_67;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(_add_68), "f"(hc_eps));
    float _fdiv_rn_15 = __fdiv_rn(_fdiv_rn_14, _add_67);
    float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 4);
    float add_14;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_14) : "f"(_fdiv_rn_15), "f"(_shfl_xor_sync_70));
    float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, add_14, 24);
    float _add_72;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(add_14), "f"(_shfl_xor_sync_73));
    float _add_71;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(_add_72), "f"(hc_eps));
    float _fdiv_rn_16 = __fdiv_rn(_fdiv_rn_15, _add_71);
    float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 1);
    float add_15;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_15) : "f"(_fdiv_rn_16), "f"(_shfl_xor_sync_74));
    float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, add_15, 2);
    float _add_76;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(add_15), "f"(_shfl_xor_sync_77));
    float _add_75;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(_add_76), "f"(hc_eps));
    float _fdiv_rn_17 = __fdiv_rn(_fdiv_rn_16, _add_75);
    float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 4);
    float add_16;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_16) : "f"(_fdiv_rn_17), "f"(_shfl_xor_sync_78));
    float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, add_16, 24);
    float _add_80;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(add_16), "f"(_shfl_xor_sync_81));
    float _add_79;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(_add_80), "f"(hc_eps));
    float _fdiv_rn_18 = __fdiv_rn(_fdiv_rn_17, _add_79);
    float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 1);
    float add_17;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_17) : "f"(_fdiv_rn_18), "f"(_shfl_xor_sync_82));
    float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, add_17, 2);
    float _add_84;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(add_17), "f"(_shfl_xor_sync_85));
    float _add_83;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(_add_84), "f"(hc_eps));
    float _fdiv_rn_19 = __fdiv_rn(_fdiv_rn_18, _add_83);
    float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 4);
    float add_18;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_18) : "f"(_fdiv_rn_19), "f"(_shfl_xor_sync_86));
    float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, add_18, 24);
    float _add_88;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(add_18), "f"(_shfl_xor_sync_89));
    float _add_87;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(_add_88), "f"(hc_eps));
    float _fdiv_rn_20 = __fdiv_rn(_fdiv_rn_19, _add_87);
    float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 1);
    float add_19;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_19) : "f"(_fdiv_rn_20), "f"(_shfl_xor_sync_90));
    float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, add_19, 2);
    float _add_92;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(add_19), "f"(_shfl_xor_sync_93));
    float _add_91;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(_add_92), "f"(hc_eps));
    float _fdiv_rn_21 = __fdiv_rn(_fdiv_rn_20, _add_91);
    float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 4);
    float add_20;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_20) : "f"(_fdiv_rn_21), "f"(_shfl_xor_sync_94));
    float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, add_20, 24);
    float _add_96;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(add_20), "f"(_shfl_xor_sync_97));
    float _add_95;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(_add_96), "f"(hc_eps));
    float _fdiv_rn_22 = __fdiv_rn(_fdiv_rn_21, _add_95);
    float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 1);
    float add_21;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_21) : "f"(_fdiv_rn_22), "f"(_shfl_xor_sync_98));
    float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, add_21, 2);
    float _add_100;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(add_21), "f"(_shfl_xor_sync_101));
    float _add_99;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(_add_100), "f"(hc_eps));
    float _fdiv_rn_23 = __fdiv_rn(_fdiv_rn_22, _add_99);
    float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 4);
    float add_22;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_22) : "f"(_fdiv_rn_23), "f"(_shfl_xor_sync_102));
    float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, add_22, 24);
    float _add_104;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(add_22), "f"(_shfl_xor_sync_105));
    float _add_103;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(_add_104), "f"(hc_eps));
    float _fdiv_rn_24 = __fdiv_rn(_fdiv_rn_23, _add_103);
    float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 1);
    float add_23;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_23) : "f"(_fdiv_rn_24), "f"(_shfl_xor_sync_106));
    float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, add_23, 2);
    float _add_108;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(add_23), "f"(_shfl_xor_sync_109));
    float _add_107;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(_add_108), "f"(hc_eps));
    float _fdiv_rn_25 = __fdiv_rn(_fdiv_rn_24, _add_107);
    float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 4);
    float add_24;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_24) : "f"(_fdiv_rn_25), "f"(_shfl_xor_sync_110));
    float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, add_24, 24);
    float _add_112;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(add_24), "f"(_shfl_xor_sync_113));
    float _add_111;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(_add_112), "f"(hc_eps));
    float _fdiv_rn_26 = __fdiv_rn(_fdiv_rn_25, _add_111);
    float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 1);
    float add_25;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_25) : "f"(_fdiv_rn_26), "f"(_shfl_xor_sync_114));
    float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, add_25, 2);
    float _add_116;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(add_25), "f"(_shfl_xor_sync_117));
    float _add_115;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(_add_116), "f"(hc_eps));
    float _fdiv_rn_27 = __fdiv_rn(_fdiv_rn_26, _add_115);
    float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 4);
    float add_26;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_26) : "f"(_fdiv_rn_27), "f"(_shfl_xor_sync_118));
    float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, add_26, 24);
    float _add_120;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(add_26), "f"(_shfl_xor_sync_121));
    float _add_119;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(_add_120), "f"(hc_eps));
    float _fdiv_rn_28 = __fdiv_rn(_fdiv_rn_27, _add_119);
    float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 1);
    float add_27;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_27) : "f"(_fdiv_rn_28), "f"(_shfl_xor_sync_122));
    float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, add_27, 2);
    float _add_124;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(add_27), "f"(_shfl_xor_sync_125));
    float _add_123;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(_add_124), "f"(hc_eps));
    float _fdiv_rn_29 = __fdiv_rn(_fdiv_rn_28, _add_123);
    float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 4);
    float add_28;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_28) : "f"(_fdiv_rn_29), "f"(_shfl_xor_sync_126));
    float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, add_28, 24);
    float _add_128;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(add_28), "f"(_shfl_xor_sync_129));
    float _add_127;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(_add_128), "f"(hc_eps));
    float _fdiv_rn_30 = __fdiv_rn(_fdiv_rn_29, _add_127);
    float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 1);
    float add_29;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_29) : "f"(_fdiv_rn_30), "f"(_shfl_xor_sync_130));
    float _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, add_29, 2);
    float _add_132;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(add_29), "f"(_shfl_xor_sync_133));
    float _add_131;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(_add_132), "f"(hc_eps));
    float _fdiv_rn_31 = __fdiv_rn(_fdiv_rn_30, _add_131);
    float _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 4);
    float add_30;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_30) : "f"(_fdiv_rn_31), "f"(_shfl_xor_sync_134));
    float _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, add_30, 24);
    float _add_136;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(add_30), "f"(_shfl_xor_sync_137));
    float _add_135;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_135) : "f"(_add_136), "f"(hc_eps));
    float _fdiv_rn_32 = __fdiv_rn(_fdiv_rn_31, _add_135);
    float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 1);
    float add_31;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_31) : "f"(_fdiv_rn_32), "f"(_shfl_xor_sync_138));
    float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, add_31, 2);
    float _add_140;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(add_31), "f"(_shfl_xor_sync_141));
    float _add_139;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(_add_140), "f"(hc_eps));
    float _fdiv_rn_33 = __fdiv_rn(_fdiv_rn_32, _add_139);
    float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 4);
    float add_32;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_32) : "f"(_fdiv_rn_33), "f"(_shfl_xor_sync_142));
    float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, add_32, 24);
    float _add_144;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(add_32), "f"(_shfl_xor_sync_145));
    float _add_143;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(_add_144), "f"(hc_eps));
    float _fdiv_rn_34 = __fdiv_rn(_fdiv_rn_33, _add_143);
    float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 1);
    float add_33;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_33) : "f"(_fdiv_rn_34), "f"(_shfl_xor_sync_146));
    float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, add_33, 2);
    float _add_148;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(add_33), "f"(_shfl_xor_sync_149));
    float _add_147;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(_add_148), "f"(hc_eps));
    float _fdiv_rn_35 = __fdiv_rn(_fdiv_rn_34, _add_147);
    float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 4);
    float add_34;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_34) : "f"(_fdiv_rn_35), "f"(_shfl_xor_sync_150));
    float _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, add_34, 24);
    float _add_152;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(add_34), "f"(_shfl_xor_sync_153));
    float _add_151;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(_add_152), "f"(hc_eps));
    float _fdiv_rn_36 = __fdiv_rn(_fdiv_rn_35, _add_151);
    float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 1);
    float add_35;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_35) : "f"(_fdiv_rn_36), "f"(_shfl_xor_sync_154));
    float _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, add_35, 2);
    float _add_156;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(add_35), "f"(_shfl_xor_sync_157));
    float _add_155;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(_add_156), "f"(hc_eps));
    float _fdiv_rn_37 = __fdiv_rn(_fdiv_rn_36, _add_155);
    float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 4);
    float add_36;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_36) : "f"(_fdiv_rn_37), "f"(_shfl_xor_sync_158));
    float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, add_36, 24);
    float _add_160;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(add_36), "f"(_shfl_xor_sync_161));
    float _add_159;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_159) : "f"(_add_160), "f"(hc_eps));
    float _fdiv_rn_38 = __fdiv_rn(_fdiv_rn_37, _add_159);
    float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 1);
    float add_37;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_37) : "f"(_fdiv_rn_38), "f"(_shfl_xor_sync_162));
    float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, add_37, 2);
    float _add_164;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(add_37), "f"(_shfl_xor_sync_165));
    float _add_163;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(_add_164), "f"(hc_eps));
    float _fdiv_rn_39 = __fdiv_rn(_fdiv_rn_38, _add_163);
    float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 4);
    float add_38;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_38) : "f"(_fdiv_rn_39), "f"(_shfl_xor_sync_166));
    float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, add_38, 24);
    float _add_168;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(add_38), "f"(_shfl_xor_sync_169));
    float _add_167;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(_add_168), "f"(hc_eps));
    float _fdiv_rn_40 = __fdiv_rn(_fdiv_rn_39, _add_167);
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
