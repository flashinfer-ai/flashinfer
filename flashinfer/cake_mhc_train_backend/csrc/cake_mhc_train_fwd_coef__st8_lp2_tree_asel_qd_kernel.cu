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
kernel_cake_mhc_train_fwd_coef__st8_lp2_tree_asel_qd(float* __restrict__ spart, float* __restrict__ qpart, float* __restrict__ scale, float* __restrict__ base, float* __restrict__ stats, int T, int p_k, int t_pad, float rms_eps, float hc_eps)
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
    float qdn[2];
    float qdc[2];
    float rpre[2];
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
        #pragma unroll
        for (int j_2 = 0; j_2 < 2; j_2++) {
            qdc[j_2] = qpart[(long long)kq * tp64 + tcr[j_2]];
        }
        if (p_k <= 8) {
            long long sa_row = (long long)g * 8;
            int sa_i0_1_0 = tid_2;
            int sa_plane0_1_0 = sa_i0_1_0 / 48;
            int sa_within0_1_0 = sa_i0_1_0 - sa_plane0_1_0 * 48;
            long long sa_src0_1_0 = ((long long)sa_plane0_1_0 * tp64 + sa_row) * 24 + (long long)(sa_within0_1_0 * 4);
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %0, 0;\n\t"
                "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
                "}"
                :: "r"((sa_i0_1_0 < nf4) ? 1 : 0), "r"(sview_addr + (unsigned int)((boff + 4 * sa_i0_1_0) * 4)), "l"(spart + sa_src0_1_0), "l"(0x14F0000000000000ULL));
            int sa_i0_1_1 = tid_2 + 128;
            int sa_plane0_1_1 = sa_i0_1_1 / 48;
            int sa_within0_1_1 = sa_i0_1_1 - sa_plane0_1_1 * 48;
            long long sa_src0_1_1 = ((long long)sa_plane0_1_1 * tp64 + sa_row) * 24 + (long long)(sa_within0_1_1 * 4);
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %0, 0;\n\t"
                "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
                "}"
                :: "r"((sa_i0_1_1 < nf4) ? 1 : 0), "r"(sview_addr + (unsigned int)((boff + 4 * sa_i0_1_1) * 4)), "l"(spart + sa_src0_1_1), "l"(0x14F0000000000000ULL));
            int sa_i0_1_2 = tid_2 + 256;
            int sa_plane0_1_2 = sa_i0_1_2 / 48;
            int sa_within0_1_2 = sa_i0_1_2 - sa_plane0_1_2 * 48;
            long long sa_src0_1_2 = ((long long)sa_plane0_1_2 * tp64 + sa_row) * 24 + (long long)(sa_within0_1_2 * 4);
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %0, 0;\n\t"
                "@p cp.async.cg.shared::cta.global.L2::cache_hint [%1], [%2], 16, %3;\n\t"
                "}"
                :: "r"((sa_i0_1_2 < nf4) ? 1 : 0), "r"(sview_addr + (unsigned int)((boff + 4 * sa_i0_1_2) * 4)), "l"(spart + sa_src0_1_2), "l"(0x14F0000000000000ULL));
            asm volatile("cp.async.commit_group;");
            #pragma unroll
            for (int j_3 = 0; j_3 < 2; j_3++) {
                float qvd = ((lane_0 < p_k) ? qdc[j_3] : 0.0f);
                float _warp_reduce_3 = qvd;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
                float _fma_2;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(_warp_reduce_3), "f"(6.1035156e-05f), "f"(rms_eps));
                float _sqrt_1;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_1) : "f"(_fma_2));
                float _rcp_0 = __frcp_rn(_sqrt_1);
                rpre[j_3] = _rcp_0;
            }
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            #pragma unroll 8
            for (int k = 0; k < p_k; k++) {
                #pragma unroll
                for (int j_4 = 0; j_4 < 2; j_4++) {
                    float _add_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(s_acc[j_4]), "f"(sview[boff + (k * 8 + tin[j_4]) * 24 + n]));
                    s_acc[j_4] = _add_4;
                }
            }
            #pragma unroll
            for (int j_5 = 0; j_5 < 2; j_5++) {
                qvl[j_5] = ((lane_0 < p_k) ? qdc[j_5] : 0.0f);
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
            for (int j_6 = 0; j_6 < 2; j_6++) {
                float ql = qpart[(long long)kq * tp64 + tcr[j_6]];
                qvl[j_6] = ((lane_0 < p_k) ? ql : 0.0f);
                float _warp_reduce_10 = qvl[j_6];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_10 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_10, offset);
                float _fma_9;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(_warp_reduce_10), "f"(6.1035156e-05f), "f"(rms_eps));
                float _sqrt_8;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_8) : "f"(_fma_9));
                float _rcp_7 = __frcp_rn(_sqrt_8);
                rpre[j_6] = _rcp_7;
            }
        }
        float rr[2];
        float zz[2];
        float cc[2];
        float pc_z0;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(pc_z0) : "f"(rpre[0]), "f"(s_acc[0]));
        float pc_a0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(pc_a0) : "f"(pc_z0), "f"(sc), "f"(bs));
        float pc_z1;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(pc_z1) : "f"(rpre[1]), "f"(s_acc[1]));
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
        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, pc_logit, 1);
        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, pc_logit, 2);
        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, pc_logit, 3);
        float _sub_12;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(pc_logit), "f"(fmaxf(fmaxf(pc_logit, _shfl_xor_sync_13), fmaxf(_shfl_xor_sync_14, _shfl_xor_sync_15))));
        float _exp_1 = __nv_expf(_sub_12);
        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
        float _add_18;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(_exp_1), "f"(_shfl_xor_sync_19));
        float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 2);
        float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 3);
        float _add_20;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(_shfl_xor_sync_21), "f"(_shfl_xor_sync_22));
        float _add_17;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(_add_18), "f"(_add_20));
        float _fdiv_16 = __fdiv_rn(_exp_1, _add_17);
        float _add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(_fdiv_16), "f"(hc_eps));
        float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 4);
        float _add_25;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(_add_2), "f"(_shfl_xor_sync_26));
        float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 24);
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, _add_2, 28);
        float _add_27;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(_shfl_xor_sync_28), "f"(_shfl_xor_sync_29));
        float _add_24;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(_add_25), "f"(_add_27));
        float _add_23;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(_add_24), "f"(hc_eps));
        float _fdiv_rn_2 = __fdiv_rn(_add_2, _add_23);
        float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 1);
        float _add_32;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(_fdiv_rn_2), "f"(_shfl_xor_sync_33));
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 2);
        float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_2, 3);
        float _add_34;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(_shfl_xor_sync_35), "f"(_shfl_xor_sync_36));
        float _add_31;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(_add_32), "f"(_add_34));
        float _add_30;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(_add_31), "f"(hc_eps));
        float _fdiv_rn_3 = __fdiv_rn(_fdiv_rn_2, _add_30);
        float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 4);
        float _add_39;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(_fdiv_rn_3), "f"(_shfl_xor_sync_40));
        float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 24);
        float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_3, 28);
        float _add_41;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(_shfl_xor_sync_42), "f"(_shfl_xor_sync_43));
        float _add_38;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(_add_39), "f"(_add_41));
        float _add_37;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(_add_38), "f"(hc_eps));
        float _fdiv_rn_4 = __fdiv_rn(_fdiv_rn_3, _add_37);
        float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 1);
        float _add_46;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(_fdiv_rn_4), "f"(_shfl_xor_sync_47));
        float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 2);
        float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_4, 3);
        float _add_48;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(_shfl_xor_sync_49), "f"(_shfl_xor_sync_50));
        float _add_45;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(_add_46), "f"(_add_48));
        float _add_44;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(_add_45), "f"(hc_eps));
        float _fdiv_rn_5 = __fdiv_rn(_fdiv_rn_4, _add_44);
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 4);
        float _add_53;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(_fdiv_rn_5), "f"(_shfl_xor_sync_54));
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 24);
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_5, 28);
        float _add_55;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(_shfl_xor_sync_56), "f"(_shfl_xor_sync_57));
        float _add_52;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(_add_53), "f"(_add_55));
        float _add_51;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(_add_52), "f"(hc_eps));
        float _fdiv_rn_6 = __fdiv_rn(_fdiv_rn_5, _add_51);
        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 1);
        float _add_60;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(_fdiv_rn_6), "f"(_shfl_xor_sync_61));
        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 2);
        float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_6, 3);
        float _add_62;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(_shfl_xor_sync_63), "f"(_shfl_xor_sync_64));
        float _add_59;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(_add_60), "f"(_add_62));
        float _add_58;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(_add_59), "f"(hc_eps));
        float _fdiv_rn_7 = __fdiv_rn(_fdiv_rn_6, _add_58);
        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 4);
        float _add_67;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(_fdiv_rn_7), "f"(_shfl_xor_sync_68));
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 24);
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_7, 28);
        float _add_69;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(_shfl_xor_sync_70), "f"(_shfl_xor_sync_71));
        float _add_66;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(_add_67), "f"(_add_69));
        float _add_65;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(_add_66), "f"(hc_eps));
        float _fdiv_rn_8 = __fdiv_rn(_fdiv_rn_7, _add_65);
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 1);
        float _add_74;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(_fdiv_rn_8), "f"(_shfl_xor_sync_75));
        float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 2);
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_8, 3);
        float _add_76;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(_shfl_xor_sync_77), "f"(_shfl_xor_sync_78));
        float _add_73;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(_add_74), "f"(_add_76));
        float _add_72;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(_add_73), "f"(hc_eps));
        float _fdiv_rn_9 = __fdiv_rn(_fdiv_rn_8, _add_72);
        float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 4);
        float _add_81;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(_fdiv_rn_9), "f"(_shfl_xor_sync_82));
        float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 24);
        float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_9, 28);
        float _add_83;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(_shfl_xor_sync_84), "f"(_shfl_xor_sync_85));
        float _add_80;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(_add_81), "f"(_add_83));
        float _add_79;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(_add_80), "f"(hc_eps));
        float _fdiv_rn_10 = __fdiv_rn(_fdiv_rn_9, _add_79);
        float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 1);
        float _add_88;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(_fdiv_rn_10), "f"(_shfl_xor_sync_89));
        float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 2);
        float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_10, 3);
        float _add_90;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_90) : "f"(_shfl_xor_sync_91), "f"(_shfl_xor_sync_92));
        float _add_87;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(_add_88), "f"(_add_90));
        float _add_86;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(_add_87), "f"(hc_eps));
        float _fdiv_rn_11 = __fdiv_rn(_fdiv_rn_10, _add_86);
        float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 4);
        float _add_95;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(_fdiv_rn_11), "f"(_shfl_xor_sync_96));
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 24);
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_11, 28);
        float _add_97;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(_shfl_xor_sync_98), "f"(_shfl_xor_sync_99));
        float _add_94;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(_add_95), "f"(_add_97));
        float _add_93;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(_add_94), "f"(hc_eps));
        float _fdiv_rn_12 = __fdiv_rn(_fdiv_rn_11, _add_93);
        float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 1);
        float _add_102;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(_fdiv_rn_12), "f"(_shfl_xor_sync_103));
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 2);
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_12, 3);
        float _add_104;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(_shfl_xor_sync_105), "f"(_shfl_xor_sync_106));
        float _add_101;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(_add_102), "f"(_add_104));
        float _add_100;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(_add_101), "f"(hc_eps));
        float _fdiv_rn_13 = __fdiv_rn(_fdiv_rn_12, _add_100);
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 4);
        float _add_109;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(_fdiv_rn_13), "f"(_shfl_xor_sync_110));
        float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 24);
        float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_13, 28);
        float _add_111;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(_shfl_xor_sync_112), "f"(_shfl_xor_sync_113));
        float _add_108;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(_add_109), "f"(_add_111));
        float _add_107;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(_add_108), "f"(hc_eps));
        float _fdiv_rn_14 = __fdiv_rn(_fdiv_rn_13, _add_107);
        float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 1);
        float _add_116;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(_fdiv_rn_14), "f"(_shfl_xor_sync_117));
        float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 2);
        float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_14, 3);
        float _add_118;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(_shfl_xor_sync_119), "f"(_shfl_xor_sync_120));
        float _add_115;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(_add_116), "f"(_add_118));
        float _add_114;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(_add_115), "f"(hc_eps));
        float _fdiv_rn_15 = __fdiv_rn(_fdiv_rn_14, _add_114);
        float _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 4);
        float _add_123;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(_fdiv_rn_15), "f"(_shfl_xor_sync_124));
        float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 24);
        float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_15, 28);
        float _add_125;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(_shfl_xor_sync_126), "f"(_shfl_xor_sync_127));
        float _add_122;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(_add_123), "f"(_add_125));
        float _add_121;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(_add_122), "f"(hc_eps));
        float _fdiv_rn_16 = __fdiv_rn(_fdiv_rn_15, _add_121);
        float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 1);
        float _add_130;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(_fdiv_rn_16), "f"(_shfl_xor_sync_131));
        float _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 2);
        float _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_16, 3);
        float _add_132;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(_shfl_xor_sync_133), "f"(_shfl_xor_sync_134));
        float _add_129;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(_add_130), "f"(_add_132));
        float _add_128;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(_add_129), "f"(hc_eps));
        float _fdiv_rn_17 = __fdiv_rn(_fdiv_rn_16, _add_128);
        float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 4);
        float _add_137;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(_fdiv_rn_17), "f"(_shfl_xor_sync_138));
        float _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 24);
        float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_17, 28);
        float _add_139;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(_shfl_xor_sync_140), "f"(_shfl_xor_sync_141));
        float _add_136;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(_add_137), "f"(_add_139));
        float _add_135;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_135) : "f"(_add_136), "f"(hc_eps));
        float _fdiv_rn_18 = __fdiv_rn(_fdiv_rn_17, _add_135);
        float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 1);
        float _add_144;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(_fdiv_rn_18), "f"(_shfl_xor_sync_145));
        float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 2);
        float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_18, 3);
        float _add_146;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(_shfl_xor_sync_147), "f"(_shfl_xor_sync_148));
        float _add_143;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(_add_144), "f"(_add_146));
        float _add_142;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(_add_143), "f"(hc_eps));
        float _fdiv_rn_19 = __fdiv_rn(_fdiv_rn_18, _add_142);
        float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 4);
        float _add_151;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(_fdiv_rn_19), "f"(_shfl_xor_sync_152));
        float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 24);
        float _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_19, 28);
        float _add_153;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(_shfl_xor_sync_154), "f"(_shfl_xor_sync_155));
        float _add_150;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(_add_151), "f"(_add_153));
        float _add_149;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_149) : "f"(_add_150), "f"(hc_eps));
        float _fdiv_rn_20 = __fdiv_rn(_fdiv_rn_19, _add_149);
        float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 1);
        float _add_158;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_158) : "f"(_fdiv_rn_20), "f"(_shfl_xor_sync_159));
        float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 2);
        float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_20, 3);
        float _add_160;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(_shfl_xor_sync_161), "f"(_shfl_xor_sync_162));
        float _add_157;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(_add_158), "f"(_add_160));
        float _add_156;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(_add_157), "f"(hc_eps));
        float _fdiv_rn_21 = __fdiv_rn(_fdiv_rn_20, _add_156);
        float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 4);
        float _add_165;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(_fdiv_rn_21), "f"(_shfl_xor_sync_166));
        float _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 24);
        float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_21, 28);
        float _add_167;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(_shfl_xor_sync_168), "f"(_shfl_xor_sync_169));
        float _add_164;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(_add_165), "f"(_add_167));
        float _add_163;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(_add_164), "f"(hc_eps));
        float _fdiv_rn_22 = __fdiv_rn(_fdiv_rn_21, _add_163);
        float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 1);
        float _add_172;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(_fdiv_rn_22), "f"(_shfl_xor_sync_173));
        float _shfl_xor_sync_175 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 2);
        float _shfl_xor_sync_176 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_22, 3);
        float _add_174;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_174) : "f"(_shfl_xor_sync_175), "f"(_shfl_xor_sync_176));
        float _add_171;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_171) : "f"(_add_172), "f"(_add_174));
        float _add_170;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_170) : "f"(_add_171), "f"(hc_eps));
        float _fdiv_rn_23 = __fdiv_rn(_fdiv_rn_22, _add_170);
        float _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 4);
        float _add_179;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_179) : "f"(_fdiv_rn_23), "f"(_shfl_xor_sync_180));
        float _shfl_xor_sync_182 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 24);
        float _shfl_xor_sync_183 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_23, 28);
        float _add_181;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_181) : "f"(_shfl_xor_sync_182), "f"(_shfl_xor_sync_183));
        float _add_178;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_178) : "f"(_add_179), "f"(_add_181));
        float _add_177;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_177) : "f"(_add_178), "f"(hc_eps));
        float _fdiv_rn_24 = __fdiv_rn(_fdiv_rn_23, _add_177);
        float _shfl_xor_sync_187 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 1);
        float _add_186;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(_fdiv_rn_24), "f"(_shfl_xor_sync_187));
        float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 2);
        float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_24, 3);
        float _add_188;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(_shfl_xor_sync_189), "f"(_shfl_xor_sync_190));
        float _add_185;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_185) : "f"(_add_186), "f"(_add_188));
        float _add_184;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_184) : "f"(_add_185), "f"(hc_eps));
        float _fdiv_rn_25 = __fdiv_rn(_fdiv_rn_24, _add_184);
        float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 4);
        float _add_193;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_193) : "f"(_fdiv_rn_25), "f"(_shfl_xor_sync_194));
        float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 24);
        float _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_25, 28);
        float _add_195;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(_shfl_xor_sync_196), "f"(_shfl_xor_sync_197));
        float _add_192;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(_add_193), "f"(_add_195));
        float _add_191;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_191) : "f"(_add_192), "f"(hc_eps));
        float _fdiv_rn_26 = __fdiv_rn(_fdiv_rn_25, _add_191);
        float _shfl_xor_sync_201 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 1);
        float _add_200;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_200) : "f"(_fdiv_rn_26), "f"(_shfl_xor_sync_201));
        float _shfl_xor_sync_203 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 2);
        float _shfl_xor_sync_204 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_26, 3);
        float _add_202;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(_shfl_xor_sync_203), "f"(_shfl_xor_sync_204));
        float _add_199;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_199) : "f"(_add_200), "f"(_add_202));
        float _add_198;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(_add_199), "f"(hc_eps));
        float _fdiv_rn_27 = __fdiv_rn(_fdiv_rn_26, _add_198);
        float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 4);
        float _add_207;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_207) : "f"(_fdiv_rn_27), "f"(_shfl_xor_sync_208));
        float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 24);
        float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_27, 28);
        float _add_209;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209) : "f"(_shfl_xor_sync_210), "f"(_shfl_xor_sync_211));
        float _add_206;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(_add_207), "f"(_add_209));
        float _add_205;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_205) : "f"(_add_206), "f"(hc_eps));
        float _fdiv_rn_28 = __fdiv_rn(_fdiv_rn_27, _add_205);
        float _shfl_xor_sync_215 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 1);
        float _add_214;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_214) : "f"(_fdiv_rn_28), "f"(_shfl_xor_sync_215));
        float _shfl_xor_sync_217 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 2);
        float _shfl_xor_sync_218 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_28, 3);
        float _add_216;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_216) : "f"(_shfl_xor_sync_217), "f"(_shfl_xor_sync_218));
        float _add_213;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_213) : "f"(_add_214), "f"(_add_216));
        float _add_212;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_212) : "f"(_add_213), "f"(hc_eps));
        float _fdiv_rn_29 = __fdiv_rn(_fdiv_rn_28, _add_212);
        float _shfl_xor_sync_222 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 4);
        float _add_221;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_221) : "f"(_fdiv_rn_29), "f"(_shfl_xor_sync_222));
        float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 24);
        float _shfl_xor_sync_225 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_29, 28);
        float _add_223;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_223) : "f"(_shfl_xor_sync_224), "f"(_shfl_xor_sync_225));
        float _add_220;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_220) : "f"(_add_221), "f"(_add_223));
        float _add_219;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_219) : "f"(_add_220), "f"(hc_eps));
        float _fdiv_rn_30 = __fdiv_rn(_fdiv_rn_29, _add_219);
        float _shfl_xor_sync_229 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 1);
        float _add_228;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_228) : "f"(_fdiv_rn_30), "f"(_shfl_xor_sync_229));
        float _shfl_xor_sync_231 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 2);
        float _shfl_xor_sync_232 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_30, 3);
        float _add_230;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_230) : "f"(_shfl_xor_sync_231), "f"(_shfl_xor_sync_232));
        float _add_227;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_227) : "f"(_add_228), "f"(_add_230));
        float _add_226;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_226) : "f"(_add_227), "f"(hc_eps));
        float _fdiv_rn_31 = __fdiv_rn(_fdiv_rn_30, _add_226);
        float _shfl_xor_sync_236 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 4);
        float _add_235;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_235) : "f"(_fdiv_rn_31), "f"(_shfl_xor_sync_236));
        float _shfl_xor_sync_238 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 24);
        float _shfl_xor_sync_239 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_31, 28);
        float _add_237;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_237) : "f"(_shfl_xor_sync_238), "f"(_shfl_xor_sync_239));
        float _add_234;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_234) : "f"(_add_235), "f"(_add_237));
        float _add_233;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_233) : "f"(_add_234), "f"(hc_eps));
        float _fdiv_rn_32 = __fdiv_rn(_fdiv_rn_31, _add_233);
        float _shfl_xor_sync_243 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 1);
        float _add_242;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_242) : "f"(_fdiv_rn_32), "f"(_shfl_xor_sync_243));
        float _shfl_xor_sync_245 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 2);
        float _shfl_xor_sync_246 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_32, 3);
        float _add_244;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_244) : "f"(_shfl_xor_sync_245), "f"(_shfl_xor_sync_246));
        float _add_241;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_241) : "f"(_add_242), "f"(_add_244));
        float _add_240;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_240) : "f"(_add_241), "f"(hc_eps));
        float _fdiv_rn_33 = __fdiv_rn(_fdiv_rn_32, _add_240);
        float _shfl_xor_sync_250 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 4);
        float _add_249;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_249) : "f"(_fdiv_rn_33), "f"(_shfl_xor_sync_250));
        float _shfl_xor_sync_252 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 24);
        float _shfl_xor_sync_253 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_33, 28);
        float _add_251;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_251) : "f"(_shfl_xor_sync_252), "f"(_shfl_xor_sync_253));
        float _add_248;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_248) : "f"(_add_249), "f"(_add_251));
        float _add_247;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_247) : "f"(_add_248), "f"(hc_eps));
        float _fdiv_rn_34 = __fdiv_rn(_fdiv_rn_33, _add_247);
        float _shfl_xor_sync_257 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 1);
        float _add_256;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_256) : "f"(_fdiv_rn_34), "f"(_shfl_xor_sync_257));
        float _shfl_xor_sync_259 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 2);
        float _shfl_xor_sync_260 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_34, 3);
        float _add_258;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_258) : "f"(_shfl_xor_sync_259), "f"(_shfl_xor_sync_260));
        float _add_255;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_255) : "f"(_add_256), "f"(_add_258));
        float _add_254;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_254) : "f"(_add_255), "f"(hc_eps));
        float _fdiv_rn_35 = __fdiv_rn(_fdiv_rn_34, _add_254);
        float _shfl_xor_sync_264 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 4);
        float _add_263;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_263) : "f"(_fdiv_rn_35), "f"(_shfl_xor_sync_264));
        float _shfl_xor_sync_266 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 24);
        float _shfl_xor_sync_267 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_35, 28);
        float _add_265;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_265) : "f"(_shfl_xor_sync_266), "f"(_shfl_xor_sync_267));
        float _add_262;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_262) : "f"(_add_263), "f"(_add_265));
        float _add_261;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_261) : "f"(_add_262), "f"(hc_eps));
        float _fdiv_rn_36 = __fdiv_rn(_fdiv_rn_35, _add_261);
        float _shfl_xor_sync_271 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 1);
        float _add_270;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_270) : "f"(_fdiv_rn_36), "f"(_shfl_xor_sync_271));
        float _shfl_xor_sync_273 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 2);
        float _shfl_xor_sync_274 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_36, 3);
        float _add_272;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_272) : "f"(_shfl_xor_sync_273), "f"(_shfl_xor_sync_274));
        float _add_269;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_269) : "f"(_add_270), "f"(_add_272));
        float _add_268;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_268) : "f"(_add_269), "f"(hc_eps));
        float _fdiv_rn_37 = __fdiv_rn(_fdiv_rn_36, _add_268);
        float _shfl_xor_sync_278 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 4);
        float _add_277;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_277) : "f"(_fdiv_rn_37), "f"(_shfl_xor_sync_278));
        float _shfl_xor_sync_280 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 24);
        float _shfl_xor_sync_281 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_37, 28);
        float _add_279;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_279) : "f"(_shfl_xor_sync_280), "f"(_shfl_xor_sync_281));
        float _add_276;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_276) : "f"(_add_277), "f"(_add_279));
        float _add_275;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_275) : "f"(_add_276), "f"(hc_eps));
        float _fdiv_rn_38 = __fdiv_rn(_fdiv_rn_37, _add_275);
        float _shfl_xor_sync_285 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 1);
        float _add_284;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_284) : "f"(_fdiv_rn_38), "f"(_shfl_xor_sync_285));
        float _shfl_xor_sync_287 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 2);
        float _shfl_xor_sync_288 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_38, 3);
        float _add_286;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_286) : "f"(_shfl_xor_sync_287), "f"(_shfl_xor_sync_288));
        float _add_283;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_283) : "f"(_add_284), "f"(_add_286));
        float _add_282;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_282) : "f"(_add_283), "f"(hc_eps));
        float _fdiv_rn_39 = __fdiv_rn(_fdiv_rn_38, _add_282);
        float _shfl_xor_sync_292 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 4);
        float _add_291;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_291) : "f"(_fdiv_rn_39), "f"(_shfl_xor_sync_292));
        float _shfl_xor_sync_294 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 24);
        float _shfl_xor_sync_295 = __shfl_xor_sync(0xFFFFFFFF, _fdiv_rn_39, 28);
        float _add_293;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_293) : "f"(_shfl_xor_sync_294), "f"(_shfl_xor_sync_295));
        float _add_290;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_290) : "f"(_add_291), "f"(_add_293));
        float _add_289;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_289) : "f"(_add_290), "f"(hc_eps));
        float _fdiv_rn_40 = __fdiv_rn(_fdiv_rn_39, _add_289);
        float pc_pp = ((lane_0 < 4 || lane_0 >= 24 && lane_0 < 28) ? _add_1 : _mul_0);
        float pc_coef0 = ((lane_0 < 8) ? pc_pp : _fdiv_rn_40);
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, pc_pp, lane_0 + 24);
        int pc_src_c = ((lane_0 < 16) ? lane_0 - 8 : lane_0 + 8);
        float _shfl_3 = __shfl_sync(0xFFFFFFFF, _fdiv_rn_40, pc_src_c);
        float pc_coef1 = ((lane_0 < 8) ? _shfl_2 : _shfl_3);
        rr[0] = rpre[0];
        rr[1] = rpre[1];
        zz[0] = pc_z0;
        zz[1] = pc_z1;
        cc[0] = pc_coef0;
        cc[1] = pc_coef1;
        #pragma unroll
        for (int j_7 = 0; j_7 < 2; j_7++) {
            int tok = tokr[j_7];
            long long tc64 = tcr[j_7];
            if (tok < T) {
                long long row = tc64 * 64;
                if (lane_0 == 0) {
                    *(reinterpret_cast<float*>(stats + row) + (0)) = rr[j_7];
                }
                if (lane_0 < 24) {
                    *(reinterpret_cast<float*>(stats + (row + 1 + lane64)) + (0)) = zz[j_7];
                    *(reinterpret_cast<float*>(stats + (row + 25 + lane64)) + (0)) = cc[j_7];
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
