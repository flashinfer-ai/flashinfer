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
kernel_cake_mhc_train_fwd_coef__rt64(float* __restrict__ spart, float* __restrict__ qpart, float* __restrict__ scale, float* __restrict__ base, float* __restrict__ stats, int T, int p_k, int t_pad, float rms_eps, float hc_eps, int sinkhorn_iters)
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
    int n_pairs = sinkhorn_iters - 1;
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
        long long plane = (long long)k * tp64 + tcr[0];
        float _add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(s_acc[0]), "f"(spart[plane * 24 + n64]));
        s_acc[0] = _add_0;
        float _add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(s_acc[1]), "f"(spart[plane * 24 + (24 + n64)]));
        s_acc[1] = _add_1;
    }
    float qv0 = ((lane_0 < p_k) ? qe[0] : 0.0f);
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
    float qv1 = ((lane_0 < p_k) ? qe[1] : 0.0f);
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
    float c = __fdiv_rn(_add_2, _add_20);
    #pragma unroll 1
    for (int _p = 0; _p < n_pairs; _p++) {
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, c, 1);
        float _add_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(c), "f"(_shfl_xor_sync_31));
        float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, c, 2);
        float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, c, 3);
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(_shfl_xor_sync_33), "f"(_shfl_xor_sync_34));
        float _add_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(_add_30), "f"(_add_32));
        float _add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(_add_29), "f"(hc_eps));
        float _fdiv_27 = __fdiv_rn(c, _add_28);
        c = _fdiv_27;
        float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, c, 4);
        float _add_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(c), "f"(_shfl_xor_sync_39));
        float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, c, 24);
        float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, c, 28);
        float _add_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(_shfl_xor_sync_41), "f"(_shfl_xor_sync_42));
        float _add_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(_add_38), "f"(_add_40));
        float _add_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(_add_37), "f"(hc_eps));
        float _fdiv_35 = __fdiv_rn(c, _add_36);
        c = _fdiv_35;
    }
    float pp = ((lane_0 < 4 || lane_0 >= 24 && lane_0 < 28) ? _add_1_1 : _mul_0);
    float coef0 = ((lane_0 < 8) ? pp : c);
    float _shfl_2 = __shfl_sync(0xFFFFFFFF, pp, lane_0 + 24);
    int src_c = ((lane_0 < 16) ? lane_0 - 8 : lane_0 + 8);
    float _shfl_3 = __shfl_sync(0xFFFFFFFF, c, src_c);
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
