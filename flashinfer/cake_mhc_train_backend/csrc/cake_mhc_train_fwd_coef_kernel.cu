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
#define THREADS 512

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


extern "C" __device__ float __nv_expf(float);

extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_mhc_train_fwd_coef(float* __restrict__ spart, float* __restrict__ qpart, float* __restrict__ scale, float* __restrict__ base, float* __restrict__ stats, int T, int p_k, int t_pad, float rms_eps, float hc_eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
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
    int tokr[1];
    long long tcr[1];
    float s_acc[1];
    #pragma unroll
    for (int j = 0; j < 1; j++) {
        tokr[j] = bid * 16 + (j * 16 + warp_1);
        tcr[j] = (long long)(((tokr[j] < T) ? tokr[j] : T - 1));
        s_acc[j] = 0.0f;
    }
    #pragma unroll 8
    for (int k = 0; k < p_k; k++) {
        long long k64 = (long long)k * tp64;
        #pragma unroll
        for (int j_1 = 0; j_1 < 1; j_1++) {
            long long plane = k64 + tcr[j_1];
            float _add_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(s_acc[j_1]), "f"(spart[plane * 24 + n64]));
            s_acc[j_1] = _add_0;
        }
    }
    #pragma unroll
    for (int j_2 = 0; j_2 < 1; j_2++) {
        int tok = tokr[j_2];
        long long tc64 = tcr[j_2];
        float qv_load = qpart[(long long)kq * tp64 + tc64];
        float qv = ((lane_0 < p_k) ? qv_load : 0.0f);
        float _warp_reduce_0 = qv;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(_warp_reduce_0), "f"(6.103515625e-05f), "f"(rms_eps));
        float _sqrt_0;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_0) : "f"(_fma_0));
        float _rcp_0 = __frcp_rn(_sqrt_0);
        float r = _rcp_0;
        float _mul_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(r), "f"(s_acc[j_2]));
        float z = _mul_0;
        float _fma_1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(z), "f"(sc), "f"(bs));
        float a_map = _fma_1;
        float a = ((lane_0 < 24) ? a_map : 0.0f);
        float _fabs_0;
        asm("abs.f32 %0, %1;" : "=f"(_fabs_0) : "f"(a));
        float _sub_0;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(0.0f), "f"(_fabs_0));
        float _exp_0 = __nv_expf(_sub_0);
        float _add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(1.0f), "f"(_exp_0));
        float _fdiv_rn_0 = __fdiv_rn(1.0f, _add_1);
        float _fdiv_rn_1 = __fdiv_rn(_exp_0, _add_1);
        float sig = ((a >= 0.0f) ? _fdiv_rn_0 : _fdiv_rn_1);
        float sneg = ((a >= 0.0f) ? _fdiv_rn_1 : _fdiv_rn_0);
        float _add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(sig), "f"(hc_eps));
        float _mul_1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(2.0f), "f"(sig));
        float logit = ((lane_0 >= 8 && lane_0 < 24) ? a : 0.0f);
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
        float _max_0 = max_noftz(logit, _shfl_xor_0);
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, _max_0, 2);
        float _max_1 = max_noftz(_max_0, _shfl_xor_1);
        float _sub_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(logit), "f"(_max_1));
        float _exp_1 = __nv_expf(_sub_1);
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
        float _add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(_exp_1), "f"(_shfl_xor_2));
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, _add_3, 2);
        float _add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(_add_3), "f"(_shfl_xor_3));
        float _fdiv_rn_2 = __fdiv_rn(_exp_1, _add_4);
        float _add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(_fdiv_rn_2), "f"(hc_eps));
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, _add_5, 4);
        float _add_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(_add_5), "f"(_shfl_xor_4));
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, _add_6, 24);
        float _add_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(_add_6), "f"(_shfl_xor_5));
        float _add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(_add_7), "f"(hc_eps));
        float _fdiv_rn_3 = __fdiv_rn(_add_5, _add_8);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 1);
        float _add_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(_fdiv_rn_3), "f"(_shfl_xor_6));
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, _add_9, 2);
        float _add_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(_add_9), "f"(_shfl_xor_7));
        float _add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(_add_10), "f"(hc_eps));
        float _fdiv_rn_4 = __fdiv_rn(_fdiv_rn_3, _add_11);
        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 4);
        float _add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(_fdiv_rn_4), "f"(_shfl_xor_8));
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, _add_12, 24);
        float _add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(_add_12), "f"(_shfl_xor_9));
        float _add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(_add_13), "f"(hc_eps));
        float _fdiv_rn_5 = __fdiv_rn(_fdiv_rn_4, _add_14);
        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 1);
        float _add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(_fdiv_rn_5), "f"(_shfl_xor_10));
        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, _add_15, 2);
        float _add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(_add_15), "f"(_shfl_xor_11));
        float _add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(_add_16), "f"(hc_eps));
        float _fdiv_rn_6 = __fdiv_rn(_fdiv_rn_5, _add_17);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 4);
        float _add_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(_fdiv_rn_6), "f"(_shfl_xor_12));
        float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, _add_18, 24);
        float _add_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(_add_18), "f"(_shfl_xor_13));
        float _add_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(_add_19), "f"(hc_eps));
        float _fdiv_rn_7 = __fdiv_rn(_fdiv_rn_6, _add_20);
        float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 1);
        float _add_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(_fdiv_rn_7), "f"(_shfl_xor_14));
        float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, _add_21, 2);
        float _add_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(_add_21), "f"(_shfl_xor_15));
        float _add_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(_add_22), "f"(hc_eps));
        float _fdiv_rn_8 = __fdiv_rn(_fdiv_rn_7, _add_23);
        float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 4);
        float _add_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(_fdiv_rn_8), "f"(_shfl_xor_16));
        float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, _add_24, 24);
        float _add_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(_add_24), "f"(_shfl_xor_17));
        float _add_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(_add_25), "f"(hc_eps));
        float _fdiv_rn_9 = __fdiv_rn(_fdiv_rn_8, _add_26);
        float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 1);
        float _add_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(_fdiv_rn_9), "f"(_shfl_xor_18));
        float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, _add_27, 2);
        float _add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(_add_27), "f"(_shfl_xor_19));
        float _add_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(_add_28), "f"(hc_eps));
        float _fdiv_rn_10 = __fdiv_rn(_fdiv_rn_9, _add_29);
        float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 4);
        float _add_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(_fdiv_rn_10), "f"(_shfl_xor_20));
        float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, _add_30, 24);
        float _add_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(_add_30), "f"(_shfl_xor_21));
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(_add_31), "f"(hc_eps));
        float _fdiv_rn_11 = __fdiv_rn(_fdiv_rn_10, _add_32);
        float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 1);
        float _add_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(_fdiv_rn_11), "f"(_shfl_xor_22));
        float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, _add_33, 2);
        float _add_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(_add_33), "f"(_shfl_xor_23));
        float _add_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(_add_34), "f"(hc_eps));
        float _fdiv_rn_12 = __fdiv_rn(_fdiv_rn_11, _add_35);
        float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 4);
        float _add_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(_fdiv_rn_12), "f"(_shfl_xor_24));
        float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, _add_36, 24);
        float _add_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(_add_36), "f"(_shfl_xor_25));
        float _add_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(_add_37), "f"(hc_eps));
        float _fdiv_rn_13 = __fdiv_rn(_fdiv_rn_12, _add_38);
        float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 1);
        float _add_39;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(_fdiv_rn_13), "f"(_shfl_xor_26));
        float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, _add_39, 2);
        float _add_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(_add_39), "f"(_shfl_xor_27));
        float _add_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(_add_40), "f"(hc_eps));
        float _fdiv_rn_14 = __fdiv_rn(_fdiv_rn_13, _add_41);
        float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 4);
        float _add_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(_fdiv_rn_14), "f"(_shfl_xor_28));
        float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, _add_42, 24);
        float _add_43;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(_add_42), "f"(_shfl_xor_29));
        float _add_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(_add_43), "f"(hc_eps));
        float _fdiv_rn_15 = __fdiv_rn(_fdiv_rn_14, _add_44);
        float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 1);
        float _add_45;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(_fdiv_rn_15), "f"(_shfl_xor_30));
        float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, _add_45, 2);
        float _add_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(_add_45), "f"(_shfl_xor_31));
        float _add_47;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(_add_46), "f"(hc_eps));
        float _fdiv_rn_16 = __fdiv_rn(_fdiv_rn_15, _add_47);
        float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 4);
        float _add_48;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(_fdiv_rn_16), "f"(_shfl_xor_32));
        float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, _add_48, 24);
        float _add_49;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(_add_48), "f"(_shfl_xor_33));
        float _add_50;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(_add_49), "f"(hc_eps));
        float _fdiv_rn_17 = __fdiv_rn(_fdiv_rn_16, _add_50);
        float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 1);
        float _add_51;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(_fdiv_rn_17), "f"(_shfl_xor_34));
        float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, _add_51, 2);
        float _add_52;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(_add_51), "f"(_shfl_xor_35));
        float _add_53;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(_add_52), "f"(hc_eps));
        float _fdiv_rn_18 = __fdiv_rn(_fdiv_rn_17, _add_53);
        float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 4);
        float _add_54;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(_fdiv_rn_18), "f"(_shfl_xor_36));
        float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, _add_54, 24);
        float _add_55;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(_add_54), "f"(_shfl_xor_37));
        float _add_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(_add_55), "f"(hc_eps));
        float _fdiv_rn_19 = __fdiv_rn(_fdiv_rn_18, _add_56);
        float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 1);
        float _add_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(_fdiv_rn_19), "f"(_shfl_xor_38));
        float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, _add_57, 2);
        float _add_58;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(_add_57), "f"(_shfl_xor_39));
        float _add_59;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(_add_58), "f"(hc_eps));
        float _fdiv_rn_20 = __fdiv_rn(_fdiv_rn_19, _add_59);
        float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 4);
        float _add_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(_fdiv_rn_20), "f"(_shfl_xor_40));
        float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, _add_60, 24);
        float _add_61;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(_add_60), "f"(_shfl_xor_41));
        float _add_62;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(_add_61), "f"(hc_eps));
        float _fdiv_rn_21 = __fdiv_rn(_fdiv_rn_20, _add_62);
        float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 1);
        float _add_63;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(_fdiv_rn_21), "f"(_shfl_xor_42));
        float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, _add_63, 2);
        float _add_64;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(_add_63), "f"(_shfl_xor_43));
        float _add_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(_add_64), "f"(hc_eps));
        float _fdiv_rn_22 = __fdiv_rn(_fdiv_rn_21, _add_65);
        float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 4);
        float _add_66;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(_fdiv_rn_22), "f"(_shfl_xor_44));
        float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, _add_66, 24);
        float _add_67;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(_add_66), "f"(_shfl_xor_45));
        float _add_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(_add_67), "f"(hc_eps));
        float _fdiv_rn_23 = __fdiv_rn(_fdiv_rn_22, _add_68);
        float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 1);
        float _add_69;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(_fdiv_rn_23), "f"(_shfl_xor_46));
        float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, _add_69, 2);
        float _add_70;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(_add_69), "f"(_shfl_xor_47));
        float _add_71;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(_add_70), "f"(hc_eps));
        float _fdiv_rn_24 = __fdiv_rn(_fdiv_rn_23, _add_71);
        float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 4);
        float _add_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(_fdiv_rn_24), "f"(_shfl_xor_48));
        float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, _add_72, 24);
        float _add_73;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(_add_72), "f"(_shfl_xor_49));
        float _add_74;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(_add_73), "f"(hc_eps));
        float _fdiv_rn_25 = __fdiv_rn(_fdiv_rn_24, _add_74);
        float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 1);
        float _add_75;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(_fdiv_rn_25), "f"(_shfl_xor_50));
        float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, _add_75, 2);
        float _add_76;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(_add_75), "f"(_shfl_xor_51));
        float _add_77;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(_add_76), "f"(hc_eps));
        float _fdiv_rn_26 = __fdiv_rn(_fdiv_rn_25, _add_77);
        float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 4);
        float _add_78;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(_fdiv_rn_26), "f"(_shfl_xor_52));
        float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, _add_78, 24);
        float _add_79;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(_add_78), "f"(_shfl_xor_53));
        float _add_80;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(_add_79), "f"(hc_eps));
        float _fdiv_rn_27 = __fdiv_rn(_fdiv_rn_26, _add_80);
        float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 1);
        float _add_81;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(_fdiv_rn_27), "f"(_shfl_xor_54));
        float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, _add_81, 2);
        float _add_82;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(_add_81), "f"(_shfl_xor_55));
        float _add_83;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(_add_82), "f"(hc_eps));
        float _fdiv_rn_28 = __fdiv_rn(_fdiv_rn_27, _add_83);
        float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 4);
        float _add_84;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(_fdiv_rn_28), "f"(_shfl_xor_56));
        float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, _add_84, 24);
        float _add_85;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(_add_84), "f"(_shfl_xor_57));
        float _add_86;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(_add_85), "f"(hc_eps));
        float _fdiv_rn_29 = __fdiv_rn(_fdiv_rn_28, _add_86);
        float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 1);
        float _add_87;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(_fdiv_rn_29), "f"(_shfl_xor_58));
        float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, _add_87, 2);
        float _add_88;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(_add_87), "f"(_shfl_xor_59));
        float _add_89;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(_add_88), "f"(hc_eps));
        float _fdiv_rn_30 = __fdiv_rn(_fdiv_rn_29, _add_89);
        float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 4);
        float _add_90;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_90) : "f"(_fdiv_rn_30), "f"(_shfl_xor_60));
        float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, _add_90, 24);
        float _add_91;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(_add_90), "f"(_shfl_xor_61));
        float _add_92;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(_add_91), "f"(hc_eps));
        float _fdiv_rn_31 = __fdiv_rn(_fdiv_rn_30, _add_92);
        float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 1);
        float _add_93;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(_fdiv_rn_31), "f"(_shfl_xor_62));
        float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, _add_93, 2);
        float _add_94;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(_add_93), "f"(_shfl_xor_63));
        float _add_95;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(_add_94), "f"(hc_eps));
        float _fdiv_rn_32 = __fdiv_rn(_fdiv_rn_31, _add_95);
        float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 4);
        float _add_96;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(_fdiv_rn_32), "f"(_shfl_xor_64));
        float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, _add_96, 24);
        float _add_97;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(_add_96), "f"(_shfl_xor_65));
        float _add_98;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(_add_97), "f"(hc_eps));
        float _fdiv_rn_33 = __fdiv_rn(_fdiv_rn_32, _add_98);
        float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 1);
        float _add_99;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(_fdiv_rn_33), "f"(_shfl_xor_66));
        float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, _add_99, 2);
        float _add_100;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(_add_99), "f"(_shfl_xor_67));
        float _add_101;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(_add_100), "f"(hc_eps));
        float _fdiv_rn_34 = __fdiv_rn(_fdiv_rn_33, _add_101);
        float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 4);
        float _add_102;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(_fdiv_rn_34), "f"(_shfl_xor_68));
        float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, _add_102, 24);
        float _add_103;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(_add_102), "f"(_shfl_xor_69));
        float _add_104;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(_add_103), "f"(hc_eps));
        float _fdiv_rn_35 = __fdiv_rn(_fdiv_rn_34, _add_104);
        float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 1);
        float _add_105;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_105) : "f"(_fdiv_rn_35), "f"(_shfl_xor_70));
        float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, _add_105, 2);
        float _add_106;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(_add_105), "f"(_shfl_xor_71));
        float _add_107;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(_add_106), "f"(hc_eps));
        float _fdiv_rn_36 = __fdiv_rn(_fdiv_rn_35, _add_107);
        float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 4);
        float _add_108;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(_fdiv_rn_36), "f"(_shfl_xor_72));
        float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, _add_108, 24);
        float _add_109;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(_add_108), "f"(_shfl_xor_73));
        float _add_110;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(_add_109), "f"(hc_eps));
        float _fdiv_rn_37 = __fdiv_rn(_fdiv_rn_36, _add_110);
        float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 1);
        float _add_111;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(_fdiv_rn_37), "f"(_shfl_xor_74));
        float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, _add_111, 2);
        float _add_112;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(_add_111), "f"(_shfl_xor_75));
        float _add_113;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(_add_112), "f"(hc_eps));
        float _fdiv_rn_38 = __fdiv_rn(_fdiv_rn_37, _add_113);
        float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 4);
        float _add_114;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(_fdiv_rn_38), "f"(_shfl_xor_76));
        float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, _add_114, 24);
        float _add_115;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(_add_114), "f"(_shfl_xor_77));
        float _add_116;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(_add_115), "f"(hc_eps));
        float _fdiv_rn_39 = __fdiv_rn(_fdiv_rn_38, _add_116);
        float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 1);
        float _add_117;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(_fdiv_rn_39), "f"(_shfl_xor_78));
        float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, _add_117, 2);
        float _add_118;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(_add_117), "f"(_shfl_xor_79));
        float _add_119;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(_add_118), "f"(hc_eps));
        float _fdiv_rn_40 = __fdiv_rn(_fdiv_rn_39, _add_119);
        float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_40, 4);
        float _add_120;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(_fdiv_rn_40), "f"(_shfl_xor_80));
        float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, _add_120, 24);
        float _add_121;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(_add_120), "f"(_shfl_xor_81));
        float _add_122;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(_add_121), "f"(hc_eps));
        float _fdiv_rn_41 = __fdiv_rn(_fdiv_rn_40, _add_122);
        float coef = ((lane_0 < 4) ? _add_2 : ((lane_0 < 8) ? _mul_1 : _fdiv_rn_41));
        if (tok < T) {
            long long row = tc64 * 64;
            if (lane_0 == 0) {
                *(reinterpret_cast<float*>(stats + row) + (0)) = r;
            }
            if (lane_0 < 24) {
                *(reinterpret_cast<float*>(stats + (row + 1 + lane64)) + (0)) = z;
                *(reinterpret_cast<float*>(stats + (row + 25 + lane64)) + (0)) = coef;
            }
            if (lane_0 < 15) {
                *(reinterpret_cast<float*>(stats + (row + 49 + lane64)) + (0)) = 0.0f;
            }
        }
    }
}

} // extern "C"
