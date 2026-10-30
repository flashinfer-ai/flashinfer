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
kernel_cake_mhc_train_fwd_coef__st16_w4_tree(float* __restrict__ spart, float* __restrict__ qpart, float* __restrict__ scale, float* __restrict__ base, float* __restrict__ stats, int T, int p_k, int t_pad, float rms_eps, float hc_eps)
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
    int tokr[1];
    long long tcr[1];
    float s_acc[1];
    int tid_2 = (int)tid;
    int n_groups = (T + 3) / 4;
    int nf4 = 24 * p_k;
    int nq4 = p_k;
    int tin[1];
    #pragma unroll
    for (int j = 0; j < 1; j++) {
        tin[j] = j * 4 + warp_1;
    }
    float vbuf[12];
    float qv4[4];
    float qvl[1];
    int g0 = (int)bid;
    if (p_k <= 16) {
        long long sl_row_a = (long long)g0 * 4;
        int sl_i_a0 = ((nf4 > tid_2) ? tid_2 : nf4 - 1);
        int sl_plane_a0 = sl_i_a0 / 24;
        int sl_within_a0 = sl_i_a0 - sl_plane_a0 * 24;
        long long sl_src_a0 = ((long long)sl_plane_a0 * tp64 + sl_row_a) * 24 + (long long)(sl_within_a0 * 4);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(spart + sl_src_a0 + 0);
            vbuf[0 + 0] = _v4.x;
            vbuf[0 + 1] = _v4.y;
            vbuf[0 + 2] = _v4.z;
            vbuf[0 + 3] = _v4.w;
        }
        int sl_i_a1 = ((nf4 > tid_2 + 128) ? tid_2 + 128 : nf4 - 1);
        int sl_plane_a1 = sl_i_a1 / 24;
        int sl_within_a1 = sl_i_a1 - sl_plane_a1 * 24;
        long long sl_src_a1 = ((long long)sl_plane_a1 * tp64 + sl_row_a) * 24 + (long long)(sl_within_a1 * 4);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(spart + sl_src_a1 + 0);
            vbuf[4 + 0] = _v4.x;
            vbuf[4 + 1] = _v4.y;
            vbuf[4 + 2] = _v4.z;
            vbuf[4 + 3] = _v4.w;
        }
        int sl_i_a2 = ((nf4 > tid_2 + 256) ? tid_2 + 256 : nf4 - 1);
        int sl_plane_a2 = sl_i_a2 / 24;
        int sl_within_a2 = sl_i_a2 - sl_plane_a2 * 24;
        long long sl_src_a2 = ((long long)sl_plane_a2 * tp64 + sl_row_a) * 24 + (long long)(sl_within_a2 * 4);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(spart + sl_src_a2 + 0);
            vbuf[8 + 0] = _v4.x;
            vbuf[8 + 1] = _v4.y;
            vbuf[8 + 2] = _v4.z;
            vbuf[8 + 3] = _v4.w;
        }
        int sl_iq_a = ((tid_2 < nq4) ? tid_2 : nq4 - 1);
        int sl_kq__a = sl_iq_a;
        int sl_t4_a = sl_iq_a - sl_kq__a;
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
        for (int j_1 = 0; j_1 < 1; j_1++) {
            tokr[j_1] = g * 4 + tin[j_1];
            tcr[j_1] = (long long)(((tokr[j_1] < T) ? tokr[j_1] : T - 1));
            s_acc[j_1] = 0.0f;
        }
        if (p_k <= 16) {
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
            #pragma unroll 16
            for (int k = 0; k < p_k; k++) {
                #pragma unroll
                for (int j_2 = 0; j_2 < 1; j_2++) {
                    float _add_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(s_acc[j_2]), "f"(sview[boff + (k * 4 + tin[j_2]) * 24 + n]));
                    s_acc[j_2] = _add_4;
                }
            }
            #pragma unroll
            for (int j_3 = 0; j_3 < 1; j_3++) {
                float ql = qview[qoff + kq * 4 + tin[j_3]];
                qvl[j_3] = ((lane_0 < p_k) ? ql : 0.0f);
            }
        } else {
            #pragma unroll 8
            for (int k_1 = 0; k_1 < p_k; k_1++) {
                long long k64 = (long long)k_1 * tp64;
                #pragma unroll
                for (int j_4 = 0; j_4 < 1; j_4++) {
                    long long plane = k64 + tcr[j_4];
                    float _add_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(s_acc[j_4]), "f"(spart[plane * 24 + n64]));
                    s_acc[j_4] = _add_5;
                }
            }
            #pragma unroll
            for (int j_5 = 0; j_5 < 1; j_5++) {
                float ql_1 = qpart[(long long)kq * tp64 + tcr[j_5]];
                qvl[j_5] = ((lane_0 < p_k) ? ql_1 : 0.0f);
            }
        }
        float rr[1];
        float zz[1];
        float cc[1];
        float _warp_reduce_8 = qvl[0];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_8 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_8, offset);
        float _fma_7;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(_warp_reduce_8), "f"(6.1035156e-05f), "f"(rms_eps));
        float _sqrt_6;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_6) : "f"(_fma_7));
        float sc_r = __frcp_rn(_sqrt_6);
        float sc_z;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(sc_z) : "f"(sc_r), "f"(s_acc[0]));
        float sc_a_map;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sc_a_map) : "f"(sc_z), "f"(sc), "f"(bs));
        float sc_a = ((lane_0 < 24) ? sc_a_map : 0.0f);
        float _sub_9;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(0.0f), "f"(fabsf(sc_a)));
        float _exp_0 = __nv_expf(_sub_9);
        float _add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(1.0f), "f"(_exp_0));
        float _fdiv_rn_0 = __fdiv_rn(1.0f, _add_0);
        float _fdiv_rn_1 = __fdiv_rn(_exp_0, _add_0);
        float sig = ((sc_a >= 0.0f) ? _fdiv_rn_0 : _fdiv_rn_1);
        float sneg = ((sc_a >= 0.0f) ? _fdiv_rn_1 : _fdiv_rn_0);
        float _add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(sig), "f"(hc_eps));
        float _mul_0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(2.0f), "f"(sig));
        float logit = ((lane_0 >= 8 && lane_0 < 24) ? sc_a : 0.0f);
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, logit, 2);
        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, logit, 3);
        float _sub_10;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(logit), "f"(fmaxf(fmaxf(logit, _shfl_xor_sync_11), fmaxf(_shfl_xor_sync_12, _shfl_xor_sync_13))));
        float _exp_1 = __nv_expf(_sub_10);
        float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
        float _add_16;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(_exp_1), "f"(_shfl_xor_sync_17));
        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 2);
        float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 3);
        float _add_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(_shfl_xor_sync_19), "f"(_shfl_xor_sync_20));
        float _add_15;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(_add_16), "f"(_add_18));
        float _fdiv_14 = __fdiv_rn(_exp_1, _add_15);
        float _add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(_fdiv_14), "f"(hc_eps));
        float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 4);
        float _add_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(_add_2), "f"(_shfl_xor_sync_24));
        float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 24);
        float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 28);
        float _add_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(_shfl_xor_sync_26), "f"(_shfl_xor_sync_27));
        float _add_22;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(_add_23), "f"(_add_25));
        float _add_21;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(_add_22), "f"(hc_eps));
        float _fdiv_rn_2 = __fdiv_rn(_add_2, _add_21);
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 1);
        float _add_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(_fdiv_rn_2), "f"(_shfl_xor_sync_31));
        float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 2);
        float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 3);
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(_shfl_xor_sync_33), "f"(_shfl_xor_sync_34));
        float _add_29;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(_add_30), "f"(_add_32));
        float _add_28;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(_add_29), "f"(hc_eps));
        float _fdiv_rn_3 = __fdiv_rn(_fdiv_rn_2, _add_28);
        float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 4);
        float _add_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(_fdiv_rn_3), "f"(_shfl_xor_sync_38));
        float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 24);
        float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 28);
        float _add_39;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(_shfl_xor_sync_40), "f"(_shfl_xor_sync_41));
        float _add_36;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(_add_37), "f"(_add_39));
        float _add_35;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(_add_36), "f"(hc_eps));
        float _fdiv_rn_4 = __fdiv_rn(_fdiv_rn_3, _add_35);
        float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 1);
        float _add_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(_fdiv_rn_4), "f"(_shfl_xor_sync_45));
        float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 2);
        float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 3);
        float _add_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(_shfl_xor_sync_47), "f"(_shfl_xor_sync_48));
        float _add_43;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(_add_44), "f"(_add_46));
        float _add_42;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(_add_43), "f"(hc_eps));
        float _fdiv_rn_5 = __fdiv_rn(_fdiv_rn_4, _add_42);
        float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 4);
        float _add_51;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(_fdiv_rn_5), "f"(_shfl_xor_sync_52));
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 24);
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 28);
        float _add_53;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(_shfl_xor_sync_54), "f"(_shfl_xor_sync_55));
        float _add_50;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(_add_51), "f"(_add_53));
        float _add_49;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(_add_50), "f"(hc_eps));
        float _fdiv_rn_6 = __fdiv_rn(_fdiv_rn_5, _add_49);
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 1);
        float _add_58;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(_fdiv_rn_6), "f"(_shfl_xor_sync_59));
        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 2);
        float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 3);
        float _add_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(_shfl_xor_sync_61), "f"(_shfl_xor_sync_62));
        float _add_57;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(_add_58), "f"(_add_60));
        float _add_56;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(_add_57), "f"(hc_eps));
        float _fdiv_rn_7 = __fdiv_rn(_fdiv_rn_6, _add_56);
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 4);
        float _add_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(_fdiv_rn_7), "f"(_shfl_xor_sync_66));
        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 24);
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 28);
        float _add_67;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(_shfl_xor_sync_68), "f"(_shfl_xor_sync_69));
        float _add_64;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(_add_65), "f"(_add_67));
        float _add_63;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(_add_64), "f"(hc_eps));
        float _fdiv_rn_8 = __fdiv_rn(_fdiv_rn_7, _add_63);
        float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 1);
        float _add_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(_fdiv_rn_8), "f"(_shfl_xor_sync_73));
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 2);
        float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 3);
        float _add_74;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(_shfl_xor_sync_75), "f"(_shfl_xor_sync_76));
        float _add_71;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(_add_72), "f"(_add_74));
        float _add_70;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(_add_71), "f"(hc_eps));
        float _fdiv_rn_9 = __fdiv_rn(_fdiv_rn_8, _add_70);
        float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 4);
        float _add_79;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(_fdiv_rn_9), "f"(_shfl_xor_sync_80));
        float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 24);
        float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 28);
        float _add_81;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(_shfl_xor_sync_82), "f"(_shfl_xor_sync_83));
        float _add_78;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(_add_79), "f"(_add_81));
        float _add_77;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(_add_78), "f"(hc_eps));
        float _fdiv_rn_10 = __fdiv_rn(_fdiv_rn_9, _add_77);
        float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 1);
        float _add_86;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(_fdiv_rn_10), "f"(_shfl_xor_sync_87));
        float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 2);
        float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 3);
        float _add_88;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(_shfl_xor_sync_89), "f"(_shfl_xor_sync_90));
        float _add_85;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(_add_86), "f"(_add_88));
        float _add_84;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(_add_85), "f"(hc_eps));
        float _fdiv_rn_11 = __fdiv_rn(_fdiv_rn_10, _add_84);
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 4);
        float _add_93;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(_fdiv_rn_11), "f"(_shfl_xor_sync_94));
        float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 24);
        float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 28);
        float _add_95;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(_shfl_xor_sync_96), "f"(_shfl_xor_sync_97));
        float _add_92;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(_add_93), "f"(_add_95));
        float _add_91;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(_add_92), "f"(hc_eps));
        float _fdiv_rn_12 = __fdiv_rn(_fdiv_rn_11, _add_91);
        float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 1);
        float _add_100;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(_fdiv_rn_12), "f"(_shfl_xor_sync_101));
        float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 2);
        float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 3);
        float _add_102;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(_shfl_xor_sync_103), "f"(_shfl_xor_sync_104));
        float _add_99;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(_add_100), "f"(_add_102));
        float _add_98;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(_add_99), "f"(hc_eps));
        float _fdiv_rn_13 = __fdiv_rn(_fdiv_rn_12, _add_98);
        float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 4);
        float _add_107;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(_fdiv_rn_13), "f"(_shfl_xor_sync_108));
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 24);
        float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 28);
        float _add_109;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(_shfl_xor_sync_110), "f"(_shfl_xor_sync_111));
        float _add_106;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(_add_107), "f"(_add_109));
        float _add_105;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_105) : "f"(_add_106), "f"(hc_eps));
        float _fdiv_rn_14 = __fdiv_rn(_fdiv_rn_13, _add_105);
        float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 1);
        float _add_114;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(_fdiv_rn_14), "f"(_shfl_xor_sync_115));
        float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 2);
        float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 3);
        float _add_116;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(_shfl_xor_sync_117), "f"(_shfl_xor_sync_118));
        float _add_113;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(_add_114), "f"(_add_116));
        float _add_112;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(_add_113), "f"(hc_eps));
        float _fdiv_rn_15 = __fdiv_rn(_fdiv_rn_14, _add_112);
        float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 4);
        float _add_121;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(_fdiv_rn_15), "f"(_shfl_xor_sync_122));
        float _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 24);
        float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 28);
        float _add_123;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(_shfl_xor_sync_124), "f"(_shfl_xor_sync_125));
        float _add_120;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(_add_121), "f"(_add_123));
        float _add_119;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(_add_120), "f"(hc_eps));
        float _fdiv_rn_16 = __fdiv_rn(_fdiv_rn_15, _add_119);
        float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 1);
        float _add_128;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(_fdiv_rn_16), "f"(_shfl_xor_sync_129));
        float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 2);
        float _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 3);
        float _add_130;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(_shfl_xor_sync_131), "f"(_shfl_xor_sync_132));
        float _add_127;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(_add_128), "f"(_add_130));
        float _add_126;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(_add_127), "f"(hc_eps));
        float _fdiv_rn_17 = __fdiv_rn(_fdiv_rn_16, _add_126);
        float _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 4);
        float _add_135;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_135) : "f"(_fdiv_rn_17), "f"(_shfl_xor_sync_136));
        float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 24);
        float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 28);
        float _add_137;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(_shfl_xor_sync_138), "f"(_shfl_xor_sync_139));
        float _add_134;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(_add_135), "f"(_add_137));
        float _add_133;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(_add_134), "f"(hc_eps));
        float _fdiv_rn_18 = __fdiv_rn(_fdiv_rn_17, _add_133);
        float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 1);
        float _add_142;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(_fdiv_rn_18), "f"(_shfl_xor_sync_143));
        float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 2);
        float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 3);
        float _add_144;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(_shfl_xor_sync_145), "f"(_shfl_xor_sync_146));
        float _add_141;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_141) : "f"(_add_142), "f"(_add_144));
        float _add_140;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(_add_141), "f"(hc_eps));
        float _fdiv_rn_19 = __fdiv_rn(_fdiv_rn_18, _add_140);
        float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 4);
        float _add_149;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149) : "f"(_fdiv_rn_19), "f"(_shfl_xor_sync_150));
        float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 24);
        float _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 28);
        float _add_151;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(_shfl_xor_sync_152), "f"(_shfl_xor_sync_153));
        float _add_148;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(_add_149), "f"(_add_151));
        float _add_147;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(_add_148), "f"(hc_eps));
        float _fdiv_rn_20 = __fdiv_rn(_fdiv_rn_19, _add_147);
        float _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 1);
        float _add_156;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(_fdiv_rn_20), "f"(_shfl_xor_sync_157));
        float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 2);
        float _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 3);
        float _add_158;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_158) : "f"(_shfl_xor_sync_159), "f"(_shfl_xor_sync_160));
        float _add_155;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(_add_156), "f"(_add_158));
        float _add_154;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_154) : "f"(_add_155), "f"(hc_eps));
        float _fdiv_rn_21 = __fdiv_rn(_fdiv_rn_20, _add_154);
        float _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 4);
        float _add_163;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(_fdiv_rn_21), "f"(_shfl_xor_sync_164));
        float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 24);
        float _shfl_xor_sync_167 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 28);
        float _add_165;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(_shfl_xor_sync_166), "f"(_shfl_xor_sync_167));
        float _add_162;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(_add_163), "f"(_add_165));
        float _add_161;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(_add_162), "f"(hc_eps));
        float _fdiv_rn_22 = __fdiv_rn(_fdiv_rn_21, _add_161);
        float _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 1);
        float _add_170;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_170) : "f"(_fdiv_rn_22), "f"(_shfl_xor_sync_171));
        float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 2);
        float _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 3);
        float _add_172;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(_shfl_xor_sync_173), "f"(_shfl_xor_sync_174));
        float _add_169;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_169) : "f"(_add_170), "f"(_add_172));
        float _add_168;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(_add_169), "f"(hc_eps));
        float _fdiv_rn_23 = __fdiv_rn(_fdiv_rn_22, _add_168);
        float _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 4);
        float _add_177;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_177) : "f"(_fdiv_rn_23), "f"(_shfl_xor_sync_178));
        float _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 24);
        float _shfl_xor_sync_181 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 28);
        float _add_179;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179) : "f"(_shfl_xor_sync_180), "f"(_shfl_xor_sync_181));
        float _add_176;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_176) : "f"(_add_177), "f"(_add_179));
        float _add_175;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_175) : "f"(_add_176), "f"(hc_eps));
        float _fdiv_rn_24 = __fdiv_rn(_fdiv_rn_23, _add_175);
        float _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 1);
        float _add_184;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_184) : "f"(_fdiv_rn_24), "f"(_shfl_xor_sync_185));
        float _shfl_xor_sync_187 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 2);
        float _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 3);
        float _add_186;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(_shfl_xor_sync_187), "f"(_shfl_xor_sync_188));
        float _add_183;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(_add_184), "f"(_add_186));
        float _add_182;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_182) : "f"(_add_183), "f"(hc_eps));
        float _fdiv_rn_25 = __fdiv_rn(_fdiv_rn_24, _add_182);
        float _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 4);
        float _add_191;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191) : "f"(_fdiv_rn_25), "f"(_shfl_xor_sync_192));
        float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 24);
        float _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 28);
        float _add_193;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_193) : "f"(_shfl_xor_sync_194), "f"(_shfl_xor_sync_195));
        float _add_190;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_190) : "f"(_add_191), "f"(_add_193));
        float _add_189;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_189) : "f"(_add_190), "f"(hc_eps));
        float _fdiv_rn_26 = __fdiv_rn(_fdiv_rn_25, _add_189);
        float _shfl_xor_sync_199 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 1);
        float _add_198;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(_fdiv_rn_26), "f"(_shfl_xor_sync_199));
        float _shfl_xor_sync_201 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 2);
        float _shfl_xor_sync_202 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 3);
        float _add_200;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_200) : "f"(_shfl_xor_sync_201), "f"(_shfl_xor_sync_202));
        float _add_197;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(_add_198), "f"(_add_200));
        float _add_196;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_196) : "f"(_add_197), "f"(hc_eps));
        float _fdiv_rn_27 = __fdiv_rn(_fdiv_rn_26, _add_196);
        float _shfl_xor_sync_206 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 4);
        float _add_205;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(_fdiv_rn_27), "f"(_shfl_xor_sync_206));
        float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 24);
        float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 28);
        float _add_207;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_207) : "f"(_shfl_xor_sync_208), "f"(_shfl_xor_sync_209));
        float _add_204;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_204) : "f"(_add_205), "f"(_add_207));
        float _add_203;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_203) : "f"(_add_204), "f"(hc_eps));
        float _fdiv_rn_28 = __fdiv_rn(_fdiv_rn_27, _add_203);
        float _shfl_xor_sync_213 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 1);
        float _add_212;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_212) : "f"(_fdiv_rn_28), "f"(_shfl_xor_sync_213));
        float _shfl_xor_sync_215 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 2);
        float _shfl_xor_sync_216 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 3);
        float _add_214;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_214) : "f"(_shfl_xor_sync_215), "f"(_shfl_xor_sync_216));
        float _add_211;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_211) : "f"(_add_212), "f"(_add_214));
        float _add_210;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_210) : "f"(_add_211), "f"(hc_eps));
        float _fdiv_rn_29 = __fdiv_rn(_fdiv_rn_28, _add_210);
        float _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 4);
        float _add_219;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_219) : "f"(_fdiv_rn_29), "f"(_shfl_xor_sync_220));
        float _shfl_xor_sync_222 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 24);
        float _shfl_xor_sync_223 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 28);
        float _add_221;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_221) : "f"(_shfl_xor_sync_222), "f"(_shfl_xor_sync_223));
        float _add_218;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_218) : "f"(_add_219), "f"(_add_221));
        float _add_217;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_217) : "f"(_add_218), "f"(hc_eps));
        float _fdiv_rn_30 = __fdiv_rn(_fdiv_rn_29, _add_217);
        float _shfl_xor_sync_227 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 1);
        float _add_226;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_226) : "f"(_fdiv_rn_30), "f"(_shfl_xor_sync_227));
        float _shfl_xor_sync_229 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 2);
        float _shfl_xor_sync_230 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 3);
        float _add_228;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_228) : "f"(_shfl_xor_sync_229), "f"(_shfl_xor_sync_230));
        float _add_225;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_225) : "f"(_add_226), "f"(_add_228));
        float _add_224;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_224) : "f"(_add_225), "f"(hc_eps));
        float _fdiv_rn_31 = __fdiv_rn(_fdiv_rn_30, _add_224);
        float _shfl_xor_sync_234 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 4);
        float _add_233;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_233) : "f"(_fdiv_rn_31), "f"(_shfl_xor_sync_234));
        float _shfl_xor_sync_236 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 24);
        float _shfl_xor_sync_237 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 28);
        float _add_235;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_235) : "f"(_shfl_xor_sync_236), "f"(_shfl_xor_sync_237));
        float _add_232;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_232) : "f"(_add_233), "f"(_add_235));
        float _add_231;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_231) : "f"(_add_232), "f"(hc_eps));
        float _fdiv_rn_32 = __fdiv_rn(_fdiv_rn_31, _add_231);
        float _shfl_xor_sync_241 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 1);
        float _add_240;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_240) : "f"(_fdiv_rn_32), "f"(_shfl_xor_sync_241));
        float _shfl_xor_sync_243 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 2);
        float _shfl_xor_sync_244 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 3);
        float _add_242;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_242) : "f"(_shfl_xor_sync_243), "f"(_shfl_xor_sync_244));
        float _add_239;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(_add_240), "f"(_add_242));
        float _add_238;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_238) : "f"(_add_239), "f"(hc_eps));
        float _fdiv_rn_33 = __fdiv_rn(_fdiv_rn_32, _add_238);
        float _shfl_xor_sync_248 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 4);
        float _add_247;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_247) : "f"(_fdiv_rn_33), "f"(_shfl_xor_sync_248));
        float _shfl_xor_sync_250 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 24);
        float _shfl_xor_sync_251 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 28);
        float _add_249;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_249) : "f"(_shfl_xor_sync_250), "f"(_shfl_xor_sync_251));
        float _add_246;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(_add_247), "f"(_add_249));
        float _add_245;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_245) : "f"(_add_246), "f"(hc_eps));
        float _fdiv_rn_34 = __fdiv_rn(_fdiv_rn_33, _add_245);
        float _shfl_xor_sync_255 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 1);
        float _add_254;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_254) : "f"(_fdiv_rn_34), "f"(_shfl_xor_sync_255));
        float _shfl_xor_sync_257 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 2);
        float _shfl_xor_sync_258 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 3);
        float _add_256;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_256) : "f"(_shfl_xor_sync_257), "f"(_shfl_xor_sync_258));
        float _add_253;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_253) : "f"(_add_254), "f"(_add_256));
        float _add_252;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_252) : "f"(_add_253), "f"(hc_eps));
        float _fdiv_rn_35 = __fdiv_rn(_fdiv_rn_34, _add_252);
        float _shfl_xor_sync_262 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 4);
        float _add_261;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_261) : "f"(_fdiv_rn_35), "f"(_shfl_xor_sync_262));
        float _shfl_xor_sync_264 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 24);
        float _shfl_xor_sync_265 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 28);
        float _add_263;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_263) : "f"(_shfl_xor_sync_264), "f"(_shfl_xor_sync_265));
        float _add_260;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_260) : "f"(_add_261), "f"(_add_263));
        float _add_259;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_259) : "f"(_add_260), "f"(hc_eps));
        float _fdiv_rn_36 = __fdiv_rn(_fdiv_rn_35, _add_259);
        float _shfl_xor_sync_269 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 1);
        float _add_268;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_268) : "f"(_fdiv_rn_36), "f"(_shfl_xor_sync_269));
        float _shfl_xor_sync_271 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 2);
        float _shfl_xor_sync_272 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 3);
        float _add_270;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_270) : "f"(_shfl_xor_sync_271), "f"(_shfl_xor_sync_272));
        float _add_267;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_267) : "f"(_add_268), "f"(_add_270));
        float _add_266;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_266) : "f"(_add_267), "f"(hc_eps));
        float _fdiv_rn_37 = __fdiv_rn(_fdiv_rn_36, _add_266);
        float _shfl_xor_sync_276 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 4);
        float _add_275;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_275) : "f"(_fdiv_rn_37), "f"(_shfl_xor_sync_276));
        float _shfl_xor_sync_278 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 24);
        float _shfl_xor_sync_279 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 28);
        float _add_277;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_277) : "f"(_shfl_xor_sync_278), "f"(_shfl_xor_sync_279));
        float _add_274;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_274) : "f"(_add_275), "f"(_add_277));
        float _add_273;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_273) : "f"(_add_274), "f"(hc_eps));
        float _fdiv_rn_38 = __fdiv_rn(_fdiv_rn_37, _add_273);
        float _shfl_xor_sync_283 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 1);
        float _add_282;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_282) : "f"(_fdiv_rn_38), "f"(_shfl_xor_sync_283));
        float _shfl_xor_sync_285 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 2);
        float _shfl_xor_sync_286 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 3);
        float _add_284;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_284) : "f"(_shfl_xor_sync_285), "f"(_shfl_xor_sync_286));
        float _add_281;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_281) : "f"(_add_282), "f"(_add_284));
        float _add_280;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_280) : "f"(_add_281), "f"(hc_eps));
        float _fdiv_rn_39 = __fdiv_rn(_fdiv_rn_38, _add_280);
        float _shfl_xor_sync_287 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 4);
        float _add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(_fdiv_rn_39), "f"(_shfl_xor_sync_287));
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 24);
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 28);
        float _add_291;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_291) : "f"(_shfl_xor_0), "f"(_shfl_xor_1));
        float _add_290;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_290) : "f"(_add_3), "f"(_add_291));
        float _add_289;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_289) : "f"(_add_290), "f"(hc_eps));
        float _fdiv_288 = __fdiv_rn(_fdiv_rn_39, _add_289);
        float sc_coef = ((lane_0 < 4) ? _add_1 : ((lane_0 < 8) ? _mul_0 : _fdiv_288));
        rr[0] = sc_r;
        zz[0] = sc_z;
        cc[0] = sc_coef;
        #pragma unroll
        for (int j_6 = 0; j_6 < 1; j_6++) {
            int tok = tokr[j_6];
            long long tc64 = tcr[j_6];
            if (tok < T) {
                long long row = tc64 * 64;
                if (lane_0 == 0) {
                    *(reinterpret_cast<float*>(stats + row) + (0)) = rr[j_6];
                }
                if (lane_0 < 24) {
                    *(reinterpret_cast<float*>(stats + (row + 1 + lane64)) + (0)) = zz[j_6];
                    *(reinterpret_cast<float*>(stats + (row + 25 + lane64)) + (0)) = cc[j_6];
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
