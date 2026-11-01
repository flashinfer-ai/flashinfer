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
#define SMEM_SVIEW_OFF 0
#define SMEM_SVIEW_STAGE_BYTES 6144
#define SMEM_SVIEW_STRIDE 6144
#define SMEM_QVIEW_OFF 6144
#define SMEM_QVIEW_STAGE_BYTES 256
#define SMEM_QVIEW_STRIDE 256
#define SMEM_TOTAL 6400
#define THREADS 128

#include <math_constants.h>

extern "C" __device__ float __nv_expf(float);

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_mhc_train_fwd_coef__st8_lp2__h5120(float* __restrict__ spart, float* __restrict__ qpart, float* __restrict__ scale, float* __restrict__ base, float* __restrict__ stats, int T, int p_k, int t_pad, float rms_eps, float hc_eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* sview = reinterpret_cast<float*>(smem_raw + 0);
    const int sview_addr = smem + 0;
    float* qview = reinterpret_cast<float*>(smem_raw + 6144);
    const int qview_addr = smem + 6144;

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
    int tid_2 = (int)tid;
    int n_groups = (T + 7) / 8;
    int nf4 = 48 * p_k;
    int nq4 = 2 * p_k;
    int tin[2];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        tin[j] = 2 * warp_1 + j;
    }
    float vbuf[12];
    float qv4[4];
    float qvl[2];
    int g0 = (int)bid;
    if (p_k <= 8) {
        long long sl_row_a = (long long)g0 * 8;
        int sl_i_a0 = ((nf4 > tid_2) ? tid_2 : nf4 - 1);
        int sl_plane_a0 = sl_i_a0 / 48;
        int sl_within_a0 = sl_i_a0 - sl_plane_a0 * 48;
        long long sl_src_a0 = ((long long)sl_plane_a0 * tp64 + sl_row_a) * 24 + (long long)(sl_within_a0 * 4);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(spart + sl_src_a0 + 0);
            vbuf[0 + 0] = _v4.x;
            vbuf[0 + 1] = _v4.y;
            vbuf[0 + 2] = _v4.z;
            vbuf[0 + 3] = _v4.w;
        }
        int sl_i_a1 = ((nf4 > tid_2 + 128) ? tid_2 + 128 : nf4 - 1);
        int sl_plane_a1 = sl_i_a1 / 48;
        int sl_within_a1 = sl_i_a1 - sl_plane_a1 * 48;
        long long sl_src_a1 = ((long long)sl_plane_a1 * tp64 + sl_row_a) * 24 + (long long)(sl_within_a1 * 4);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(spart + sl_src_a1 + 0);
            vbuf[4 + 0] = _v4.x;
            vbuf[4 + 1] = _v4.y;
            vbuf[4 + 2] = _v4.z;
            vbuf[4 + 3] = _v4.w;
        }
        int sl_i_a2 = ((nf4 > tid_2 + 256) ? tid_2 + 256 : nf4 - 1);
        int sl_plane_a2 = sl_i_a2 / 48;
        int sl_within_a2 = sl_i_a2 - sl_plane_a2 * 48;
        long long sl_src_a2 = ((long long)sl_plane_a2 * tp64 + sl_row_a) * 24 + (long long)(sl_within_a2 * 4);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(spart + sl_src_a2 + 0);
            vbuf[8 + 0] = _v4.x;
            vbuf[8 + 1] = _v4.y;
            vbuf[8 + 2] = _v4.z;
            vbuf[8 + 3] = _v4.w;
        }
        int sl_iq_a = ((tid_2 < nq4) ? tid_2 : nq4 - 1);
        int sl_kq__a = sl_iq_a / 2;
        int sl_t4_a = sl_iq_a - sl_kq__a * 2;
        long long sl_srcq_a = (long long)sl_kq__a * tp64 + sl_row_a + (long long)(sl_t4_a * 4);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(qpart + sl_srcq_a + 0);
            qv4[0 + 0] = _v4.x;
            qv4[0 + 1] = _v4.y;
            qv4[0 + 2] = _v4.z;
            qv4[0 + 3] = _v4.w;
        }
    }
    int it[1];
    it[0] = 0;
    int g_stop = g0 + 1;
    #pragma unroll 1
    for (int g = g0; g < g_stop; g++) {
        int boff = 0;
        int qoff = 0;
        #pragma unroll
        for (int j_1 = 0; j_1 < 2; j_1++) {
            tokr[j_1] = g * 8 + tin[j_1];
            tcr[j_1] = (long long)(((tokr[j_1] < T) ? tokr[j_1] : T - 1));
            s_acc[j_1] = 0.0f;
        }
        if (p_k <= 8) {
            #pragma unroll
            for (int m = 0; m < 3; m++) {
                int i = tid_2 + 128 * m;
                if (i < nf4) {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(sview_addr + (unsigned int)((boff + 4 * i) * 4)), "f"(vbuf[4 * m]), "f"(vbuf[4 * m + 1]), "f"(vbuf[4 * m + 2]), "f"(vbuf[4 * m + 3]) : "memory");
                }
            }
            if (tid_2 < nq4) {
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(qview_addr + (unsigned int)((qoff + 4 * tid_2) * 4)), "f"(qv4[0]), "f"(qv4[1]), "f"(qv4[2]), "f"(qv4[3]) : "memory");
            }
            __syncthreads();
            #pragma unroll 8
            for (int k = 0; k < p_k; k++) {
                #pragma unroll
                for (int j_2 = 0; j_2 < 2; j_2++) {
                    float _add_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(s_acc[j_2]), "f"(sview[boff + (k * 8 + tin[j_2]) * 24 + n]));
                    s_acc[j_2] = _add_4;
                }
            }
            #pragma unroll
            for (int j_3 = 0; j_3 < 2; j_3++) {
                float ql = qview[qoff + kq * 8 + tin[j_3]];
                qvl[j_3] = ((lane_0 < p_k) ? ql : 0.0f);
            }
        } else {
            #pragma unroll 8
            for (int k_1 = 0; k_1 < p_k; k_1++) {
                long long k64 = (long long)k_1 * tp64;
                long long plane = k64 + tcr[0];
                float _add_5;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(s_acc[0]), "f"(spart[plane * 24 + n64]));
                s_acc[0] = _add_5;
                float _add_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(s_acc[1]), "f"(spart[plane * 24 + (24 + n64)]));
                s_acc[1] = _add_6;
            }
            #pragma unroll
            for (int j_4 = 0; j_4 < 2; j_4++) {
                float ql_1 = qpart[(long long)kq * tp64 + tcr[j_4]];
                qvl[j_4] = ((lane_0 < p_k) ? ql_1 : 0.0f);
            }
        }
        float rr[2];
        float zz[2];
        float cc[2];
        float _warp_reduce_8 = qvl[0];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_8 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_8, offset);
        float _sqrt_7;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_7) : "f"((float)((double)_warp_reduce_8 * 4.8828125e-05 + (double)rms_eps)));
        float pc_r0 = __frcp_rn(_sqrt_7);
        float pc_z0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(pc_z0) : "f"(pc_r0), "f"(s_acc[0]));
        float pc_a0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pc_a0) : "f"(pc_z0), "f"(sc), "f"(bs));
        float _warp_reduce_10 = qvl[1];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_10 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_10, offset);
        float _sqrt_9;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_9) : "f"((float)((double)_warp_reduce_10 * 4.8828125e-05 + (double)rms_eps)));
        float pc_r1 = __frcp_rn(_sqrt_9);
        float pc_z1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(pc_z1) : "f"(pc_r1), "f"(s_acc[1]));
        float pc_a1;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pc_a1) : "f"(pc_z1), "f"(sc), "f"(bs));
        int pc_src_sig = ((lane_0 >= 24) ? lane_0 - 24 : lane_0);
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, pc_a1, pc_src_sig);
        float pc_a_sig = ((lane_0 < 24) ? pc_a0 : _shfl_0);
        int pc_src_logit = ((lane_0 < 8) ? lane_0 + 8 : lane_0 - 8);
        float _shfl_1 = __shfl_sync(0xFFFFFFFF, pc_a1, pc_src_logit);
        float pc_logit = ((lane_0 >= 8 && lane_0 < 24) ? pc_a0 : _shfl_1);
        float _sub_11;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(0.0f), "f"(fabsf(pc_a_sig)));
        float _exp_0 = __nv_expf(_sub_11);
        float _add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(1.0f), "f"(_exp_0));
        float _fdiv_rn_0 = __fdiv_rn(1.0f, _add_0);
        float _fdiv_rn_1 = __fdiv_rn(_exp_0, _add_0);
        float sig = ((pc_a_sig >= 0.0f) ? _fdiv_rn_0 : _fdiv_rn_1);
        float sneg = ((pc_a_sig >= 0.0f) ? _fdiv_rn_1 : _fdiv_rn_0);
        float _add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(sig), "f"(hc_eps));
        float _mul_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(2.0f), "f"(sig));
        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, pc_logit, 1);
        float _max_0 = fmaxf(pc_logit, _shfl_xor_sync_12);
        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, _max_0, 2);
        float _sub_13;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(pc_logit), "f"(fmaxf(_max_0, _shfl_xor_sync_14)));
        float _exp_1 = __nv_expf(_sub_13);
        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
        float add;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(_exp_1), "f"(_shfl_xor_sync_15));
        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, add, 2);
        float _add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(add), "f"(_shfl_xor_sync_18));
        float _fdiv_16 = __fdiv_rn(_exp_1, _add_17);
        float _add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(_fdiv_16), "f"(hc_eps));
        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, _add_3, 4);
        float add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_0) : "f"(_add_3), "f"(_shfl_xor_sync_19));
        float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, add_0, 24);
        float _add_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(add_0), "f"(_shfl_xor_sync_22));
        float _add_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(_add_21), "f"(hc_eps));
        float _fdiv_rn_2 = __fdiv_rn(_add_3, _add_20);
        float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 1);
        float add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(_fdiv_rn_2), "f"(_shfl_xor_sync_23));
        float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, add_1, 2);
        float _add_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(add_1), "f"(_shfl_xor_sync_26));
        float _add_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(_add_25), "f"(hc_eps));
        float _fdiv_rn_3 = __fdiv_rn(_fdiv_rn_2, _add_24);
        float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 4);
        float add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(_fdiv_rn_3), "f"(_shfl_xor_sync_27));
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, add_2, 24);
        float _add_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(add_2), "f"(_shfl_xor_sync_30));
        float _add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(_add_29), "f"(hc_eps));
        float _fdiv_rn_4 = __fdiv_rn(_fdiv_rn_3, _add_28);
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 1);
        float add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(_fdiv_rn_4), "f"(_shfl_xor_sync_31));
        float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, add_3, 2);
        float _add_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(add_3), "f"(_shfl_xor_sync_34));
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(_add_33), "f"(hc_eps));
        float _fdiv_rn_5 = __fdiv_rn(_fdiv_rn_4, _add_32);
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 4);
        float add_4;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(_fdiv_rn_5), "f"(_shfl_xor_sync_35));
        float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, add_4, 24);
        float _add_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(add_4), "f"(_shfl_xor_sync_38));
        float _add_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(_add_37), "f"(hc_eps));
        float _fdiv_rn_6 = __fdiv_rn(_fdiv_rn_5, _add_36);
        float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 1);
        float add_5;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(_fdiv_rn_6), "f"(_shfl_xor_sync_39));
        float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, add_5, 2);
        float _add_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(add_5), "f"(_shfl_xor_sync_42));
        float _add_40;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(_add_41), "f"(hc_eps));
        float _fdiv_rn_7 = __fdiv_rn(_fdiv_rn_6, _add_40);
        float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 4);
        float add_6;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_6) : "f"(_fdiv_rn_7), "f"(_shfl_xor_sync_43));
        float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, add_6, 24);
        float _add_45;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(add_6), "f"(_shfl_xor_sync_46));
        float _add_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(_add_45), "f"(hc_eps));
        float _fdiv_rn_8 = __fdiv_rn(_fdiv_rn_7, _add_44);
        float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 1);
        float add_7;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(_fdiv_rn_8), "f"(_shfl_xor_sync_47));
        float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, add_7, 2);
        float _add_49;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(add_7), "f"(_shfl_xor_sync_50));
        float _add_48;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(_add_49), "f"(hc_eps));
        float _fdiv_rn_9 = __fdiv_rn(_fdiv_rn_8, _add_48);
        float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 4);
        float add_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(_fdiv_rn_9), "f"(_shfl_xor_sync_51));
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, add_8, 24);
        float _add_53;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(add_8), "f"(_shfl_xor_sync_54));
        float _add_52;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(_add_53), "f"(hc_eps));
        float _fdiv_rn_10 = __fdiv_rn(_fdiv_rn_9, _add_52);
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 1);
        float add_9;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_9) : "f"(_fdiv_rn_10), "f"(_shfl_xor_sync_55));
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, add_9, 2);
        float _add_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(add_9), "f"(_shfl_xor_sync_58));
        float _add_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(_add_57), "f"(hc_eps));
        float _fdiv_rn_11 = __fdiv_rn(_fdiv_rn_10, _add_56);
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 4);
        float add_10;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_10) : "f"(_fdiv_rn_11), "f"(_shfl_xor_sync_59));
        float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, add_10, 24);
        float _add_61;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(add_10), "f"(_shfl_xor_sync_62));
        float _add_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(_add_61), "f"(hc_eps));
        float _fdiv_rn_12 = __fdiv_rn(_fdiv_rn_11, _add_60);
        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 1);
        float add_11;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_11) : "f"(_fdiv_rn_12), "f"(_shfl_xor_sync_63));
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, add_11, 2);
        float _add_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(add_11), "f"(_shfl_xor_sync_66));
        float _add_64;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(_add_65), "f"(hc_eps));
        float _fdiv_rn_13 = __fdiv_rn(_fdiv_rn_12, _add_64);
        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 4);
        float add_12;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_12) : "f"(_fdiv_rn_13), "f"(_shfl_xor_sync_67));
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, add_12, 24);
        float _add_69;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(add_12), "f"(_shfl_xor_sync_70));
        float _add_68;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(_add_69), "f"(hc_eps));
        float _fdiv_rn_14 = __fdiv_rn(_fdiv_rn_13, _add_68);
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 1);
        float add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_13) : "f"(_fdiv_rn_14), "f"(_shfl_xor_sync_71));
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, add_13, 2);
        float _add_73;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(add_13), "f"(_shfl_xor_sync_74));
        float _add_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(_add_73), "f"(hc_eps));
        float _fdiv_rn_15 = __fdiv_rn(_fdiv_rn_14, _add_72);
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 4);
        float add_14;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_14) : "f"(_fdiv_rn_15), "f"(_shfl_xor_sync_75));
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, add_14, 24);
        float _add_77;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(add_14), "f"(_shfl_xor_sync_78));
        float _add_76;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(_add_77), "f"(hc_eps));
        float _fdiv_rn_16 = __fdiv_rn(_fdiv_rn_15, _add_76);
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 1);
        float add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_15) : "f"(_fdiv_rn_16), "f"(_shfl_xor_sync_79));
        float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, add_15, 2);
        float _add_81;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(add_15), "f"(_shfl_xor_sync_82));
        float _add_80;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(_add_81), "f"(hc_eps));
        float _fdiv_rn_17 = __fdiv_rn(_fdiv_rn_16, _add_80);
        float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 4);
        float add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_16) : "f"(_fdiv_rn_17), "f"(_shfl_xor_sync_83));
        float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, add_16, 24);
        float _add_85;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(add_16), "f"(_shfl_xor_sync_86));
        float _add_84;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(_add_85), "f"(hc_eps));
        float _fdiv_rn_18 = __fdiv_rn(_fdiv_rn_17, _add_84);
        float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 1);
        float add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_17) : "f"(_fdiv_rn_18), "f"(_shfl_xor_sync_87));
        float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, add_17, 2);
        float _add_89;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(add_17), "f"(_shfl_xor_sync_90));
        float _add_88;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(_add_89), "f"(hc_eps));
        float _fdiv_rn_19 = __fdiv_rn(_fdiv_rn_18, _add_88);
        float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 4);
        float add_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_18) : "f"(_fdiv_rn_19), "f"(_shfl_xor_sync_91));
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, add_18, 24);
        float _add_93;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(add_18), "f"(_shfl_xor_sync_94));
        float _add_92;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(_add_93), "f"(hc_eps));
        float _fdiv_rn_20 = __fdiv_rn(_fdiv_rn_19, _add_92);
        float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 1);
        float add_19;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_19) : "f"(_fdiv_rn_20), "f"(_shfl_xor_sync_95));
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, add_19, 2);
        float _add_97;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(add_19), "f"(_shfl_xor_sync_98));
        float _add_96;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(_add_97), "f"(hc_eps));
        float _fdiv_rn_21 = __fdiv_rn(_fdiv_rn_20, _add_96);
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 4);
        float add_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_20) : "f"(_fdiv_rn_21), "f"(_shfl_xor_sync_99));
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, add_20, 24);
        float _add_101;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(add_20), "f"(_shfl_xor_sync_102));
        float _add_100;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(_add_101), "f"(hc_eps));
        float _fdiv_rn_22 = __fdiv_rn(_fdiv_rn_21, _add_100);
        float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 1);
        float add_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_21) : "f"(_fdiv_rn_22), "f"(_shfl_xor_sync_103));
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, add_21, 2);
        float _add_105;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_105) : "f"(add_21), "f"(_shfl_xor_sync_106));
        float _add_104;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(_add_105), "f"(hc_eps));
        float _fdiv_rn_23 = __fdiv_rn(_fdiv_rn_22, _add_104);
        float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 4);
        float add_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_22) : "f"(_fdiv_rn_23), "f"(_shfl_xor_sync_107));
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, add_22, 24);
        float _add_109;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(add_22), "f"(_shfl_xor_sync_110));
        float _add_108;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(_add_109), "f"(hc_eps));
        float _fdiv_rn_24 = __fdiv_rn(_fdiv_rn_23, _add_108);
        float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 1);
        float add_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_23) : "f"(_fdiv_rn_24), "f"(_shfl_xor_sync_111));
        float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, add_23, 2);
        float _add_113;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(add_23), "f"(_shfl_xor_sync_114));
        float _add_112;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(_add_113), "f"(hc_eps));
        float _fdiv_rn_25 = __fdiv_rn(_fdiv_rn_24, _add_112);
        float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 4);
        float add_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_24) : "f"(_fdiv_rn_25), "f"(_shfl_xor_sync_115));
        float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, add_24, 24);
        float _add_117;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(add_24), "f"(_shfl_xor_sync_118));
        float _add_116;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(_add_117), "f"(hc_eps));
        float _fdiv_rn_26 = __fdiv_rn(_fdiv_rn_25, _add_116);
        float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 1);
        float add_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_25) : "f"(_fdiv_rn_26), "f"(_shfl_xor_sync_119));
        float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, add_25, 2);
        float _add_121;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(add_25), "f"(_shfl_xor_sync_122));
        float _add_120;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(_add_121), "f"(hc_eps));
        float _fdiv_rn_27 = __fdiv_rn(_fdiv_rn_26, _add_120);
        float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 4);
        float add_26;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_26) : "f"(_fdiv_rn_27), "f"(_shfl_xor_sync_123));
        float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, add_26, 24);
        float _add_125;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(add_26), "f"(_shfl_xor_sync_126));
        float _add_124;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(_add_125), "f"(hc_eps));
        float _fdiv_rn_28 = __fdiv_rn(_fdiv_rn_27, _add_124);
        float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 1);
        float add_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_27) : "f"(_fdiv_rn_28), "f"(_shfl_xor_sync_127));
        float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, add_27, 2);
        float _add_129;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(add_27), "f"(_shfl_xor_sync_130));
        float _add_128;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(_add_129), "f"(hc_eps));
        float _fdiv_rn_29 = __fdiv_rn(_fdiv_rn_28, _add_128);
        float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 4);
        float add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_28) : "f"(_fdiv_rn_29), "f"(_shfl_xor_sync_131));
        float _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, add_28, 24);
        float _add_133;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(add_28), "f"(_shfl_xor_sync_134));
        float _add_132;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(_add_133), "f"(hc_eps));
        float _fdiv_rn_30 = __fdiv_rn(_fdiv_rn_29, _add_132);
        float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 1);
        float add_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_29) : "f"(_fdiv_rn_30), "f"(_shfl_xor_sync_135));
        float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, add_29, 2);
        float _add_137;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(add_29), "f"(_shfl_xor_sync_138));
        float _add_136;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(_add_137), "f"(hc_eps));
        float _fdiv_rn_31 = __fdiv_rn(_fdiv_rn_30, _add_136);
        float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 4);
        float add_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_30) : "f"(_fdiv_rn_31), "f"(_shfl_xor_sync_139));
        float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, add_30, 24);
        float _add_141;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(add_30), "f"(_shfl_xor_sync_142));
        float _add_140;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(_add_141), "f"(hc_eps));
        float _fdiv_rn_32 = __fdiv_rn(_fdiv_rn_31, _add_140);
        float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 1);
        float add_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_31) : "f"(_fdiv_rn_32), "f"(_shfl_xor_sync_143));
        float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, add_31, 2);
        float _add_145;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_145) : "f"(add_31), "f"(_shfl_xor_sync_146));
        float _add_144;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(_add_145), "f"(hc_eps));
        float _fdiv_rn_33 = __fdiv_rn(_fdiv_rn_32, _add_144);
        float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 4);
        float add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_32) : "f"(_fdiv_rn_33), "f"(_shfl_xor_sync_147));
        float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, add_32, 24);
        float _add_149;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149) : "f"(add_32), "f"(_shfl_xor_sync_150));
        float _add_148;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(_add_149), "f"(hc_eps));
        float _fdiv_rn_34 = __fdiv_rn(_fdiv_rn_33, _add_148);
        float _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 1);
        float add_33;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_33) : "f"(_fdiv_rn_34), "f"(_shfl_xor_sync_151));
        float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, add_33, 2);
        float _add_153;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(add_33), "f"(_shfl_xor_sync_154));
        float _add_152;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(_add_153), "f"(hc_eps));
        float _fdiv_rn_35 = __fdiv_rn(_fdiv_rn_34, _add_152);
        float _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 4);
        float add_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_34) : "f"(_fdiv_rn_35), "f"(_shfl_xor_sync_155));
        float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, add_34, 24);
        float _add_157;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(add_34), "f"(_shfl_xor_sync_158));
        float _add_156;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(_add_157), "f"(hc_eps));
        float _fdiv_rn_36 = __fdiv_rn(_fdiv_rn_35, _add_156);
        float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 1);
        float add_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_35) : "f"(_fdiv_rn_36), "f"(_shfl_xor_sync_159));
        float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, add_35, 2);
        float _add_161;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(add_35), "f"(_shfl_xor_sync_162));
        float _add_160;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(_add_161), "f"(hc_eps));
        float _fdiv_rn_37 = __fdiv_rn(_fdiv_rn_36, _add_160);
        float _shfl_xor_sync_163 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 4);
        float add_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_36) : "f"(_fdiv_rn_37), "f"(_shfl_xor_sync_163));
        float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, add_36, 24);
        float _add_165;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(add_36), "f"(_shfl_xor_sync_166));
        float _add_164;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(_add_165), "f"(hc_eps));
        float _fdiv_rn_38 = __fdiv_rn(_fdiv_rn_37, _add_164);
        float _shfl_xor_sync_167 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 1);
        float add_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_37) : "f"(_fdiv_rn_38), "f"(_shfl_xor_sync_167));
        float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, add_37, 2);
        float _add_169;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(add_37), "f"(_shfl_xor_sync_170));
        float _add_168;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(_add_169), "f"(hc_eps));
        float _fdiv_rn_39 = __fdiv_rn(_fdiv_rn_38, _add_168);
        float _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 4);
        float add_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_38) : "f"(_fdiv_rn_39), "f"(_shfl_xor_sync_171));
        float _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, add_38, 24);
        float _add_173;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_173) : "f"(add_38), "f"(_shfl_xor_sync_174));
        float _add_172;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(_add_173), "f"(hc_eps));
        float _fdiv_rn_40 = __fdiv_rn(_fdiv_rn_39, _add_172);
        float pc_pp = ((lane_0 < 4 || lane_0 >= 24 && lane_0 < 28) ? _add_1 : _mul_0);
        float pc_coef0 = ((lane_0 < 8) ? pc_pp : _fdiv_rn_40);
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, pc_pp, lane_0 + 24);
        int pc_src_c = ((lane_0 < 16) ? lane_0 - 8 : lane_0 + 8);
        float _shfl_3 = __shfl_sync(0xFFFFFFFF, _fdiv_rn_40, pc_src_c);
        float pc_coef1 = ((lane_0 < 8) ? _shfl_2 : _shfl_3);
        rr[0] = pc_r0;
        rr[1] = pc_r1;
        zz[0] = pc_z0;
        zz[1] = pc_z1;
        cc[0] = pc_coef0;
        cc[1] = pc_coef1;
        #pragma unroll
        for (int j_5 = 0; j_5 < 2; j_5++) {
            int tok = tokr[j_5];
            long long tc64 = tcr[j_5];
            if (tok < T) {
                long long row = tc64 * 64;
                if (lane_0 == 0) {
                    *(reinterpret_cast<float*>(stats + row) + (0)) = rr[j_5];
                }
                if (lane_0 < 24) {
                    *(reinterpret_cast<float*>(stats + (row + 1 + lane64)) + (0)) = zz[j_5];
                    *(reinterpret_cast<float*>(stats + (row + 25 + lane64)) + (0)) = cc[j_5];
                }
                if (lane_0 < 15) {
                    *(reinterpret_cast<float*>(stats + (row + 49 + lane64)) + (0)) = 0.0f;
                }
            }
        }
        it[0] = it[0] + 1;
    }
}

} // extern "C"
