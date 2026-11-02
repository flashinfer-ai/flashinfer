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
#define SMEM_VW_OFF 0
#define SMEM_VW_STAGE_BYTES 64
#define SMEM_VW_STRIDE 64
#define SMEM_TOTAL 128
#define THREADS 256

#include <math_constants.h>


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mhc_det_train_bwd_reduce__wpu4_spred_trows(float* __restrict__ dfn_part, float* __restrict__ param_partial, float* __restrict__ dfn, float* __restrict__ dscale, float* __restrict__ dbase, int n_parts, int n_rows, int accumulate)
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
    float* vw = reinterpret_cast<float*>(smem_raw + 0);
    const int vw_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int b = bid;
    int tid_0 = tid;
    if (b >= 27) {
        long long e0 = (long long)((b - 27) * 256 + tid_0) * 4;
        float out[4];
        double a[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            a[k] = 0.0;
        }
        float _vec_load_0[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(dfn_part + e0 + 0);
            _vec_load_0[0 + 0] = _v4.x;
            _vec_load_0[0 + 1] = _v4.y;
            _vec_load_0[0 + 2] = _v4.z;
            _vec_load_0[0 + 3] = _v4.w;
        }
        #pragma unroll
        for (int k_1 = 0; k_1 < 4; k_1++) {
            a[k_1] = ((_vec_load_0[k_1] != 0.0f) ? (double)_vec_load_0[k_1] : 0.0);
        }
        #pragma unroll 4
        for (int p = 1; p < n_parts; p++) {
            float vec_load[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn_part + ((long long)p * 393216 + e0) + 0);
                vec_load[0 + 0] = _v4.x;
                vec_load[0 + 1] = _v4.y;
                vec_load[0 + 2] = _v4.z;
                vec_load[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_2 = 0; k_2 < 4; k_2++) {
                a[k_2] = a[k_2] + (double)vec_load[k_2];
            }
        }
        if (accumulate != 0) {
            float _vec_load_2[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn + e0 + 0);
                _vec_load_2[0 + 0] = _v4.x;
                _vec_load_2[0 + 1] = _v4.y;
                _vec_load_2[0 + 2] = _v4.z;
                _vec_load_2[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_3 = 0; k_3 < 4; k_3++) {
                a[k_3] = a[k_3] + (double)_vec_load_2[k_3];
            }
        }
        #pragma unroll
        for (int k_4 = 0; k_4 < 4; k_4++) {
            out[k_4] = (float)a[k_4];
        }
        {
            float4 _v4 = make_float4(out[0 + 0], out[0 + 1], out[0 + 2], out[0 + 3]);
            *reinterpret_cast<float4*>((dfn + e0) + 0) = _v4;
        }
    } else {
        int lane_1 = tid_0 & 31;
        int wrp = tid_0 >> 5;
        float hh[1];
        float ll[1];
        if (b >= 24) {
            #pragma unroll
            for (int k_5 = 0; k_5 < 1; k_5++) {
                hh[k_5] = 0.0f;
                ll[k_5] = 0.0f;
            }
            int nj = n_rows + 255 >> 8;
            if (nj <= 2) {
                int last_row = n_rows - 1;
                long long cbase = (long long)b * (long long)n_rows;
                #pragma unroll
                for (int k_6 = 0; k_6 < 1; k_6++) {
                    #pragma unroll
                    for (int u = 0; u < 2; u++) {
                        int row = u * 256 + (k_6 * 256 + tid_0);
                        int rowc = ((row < n_rows) ? row : last_row);
                        float xr = param_partial[cbase + (long long)rowc];
                        float add;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(hh[k_6]), "f"(((row < n_rows) ? xr : 0.0f)));
                        float sub;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub) : "f"(add), "f"(hh[k_6]));
                        float _sub_6;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(add), "f"(sub));
                        float _sub_5;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(hh[k_6]), "f"(_sub_6));
                        float _sub_7;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(((row < n_rows) ? xr : 0.0f)), "f"(sub));
                        float _add_4;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(_sub_5), "f"(_sub_7));
                        float _add_3;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(ll[k_6]), "f"(_add_4));
                        ll[k_6] = _add_3;
                        hh[k_6] = add;
                        float xq = param_partial[cbase + (long long)(3 * n_rows + rowc)];
                        float _add_8;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(ll[k_6]), "f"(((row < n_rows) ? xq : 0.0f)));
                        ll[k_6] = _add_8;
                    }
                }
            } else if (nj <= 8) {
                int last_row_1 = n_rows - 1;
                long long cbase_1 = (long long)b * (long long)n_rows;
                #pragma unroll
                for (int k_7 = 0; k_7 < 1; k_7++) {
                    #pragma unroll
                    for (int u_1 = 0; u_1 < 8; u_1++) {
                        int row_1 = u_1 * 256 + (k_7 * 256 + tid_0);
                        int rowc_1 = ((row_1 < n_rows) ? row_1 : last_row_1);
                        float xr_1 = param_partial[cbase_1 + (long long)rowc_1];
                        float add_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(hh[k_7]), "f"(((row_1 < n_rows) ? xr_1 : 0.0f)));
                        float sub_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1) : "f"(add_1), "f"(hh[k_7]));
                        float _sub_12;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(add_1), "f"(sub_1));
                        float _sub_11;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(hh[k_7]), "f"(_sub_12));
                        float _sub_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(((row_1 < n_rows) ? xr_1 : 0.0f)), "f"(sub_1));
                        float _add_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(_sub_11), "f"(_sub_13));
                        float _add_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(ll[k_7]), "f"(_add_10));
                        ll[k_7] = _add_9;
                        hh[k_7] = add_1;
                        float xq_1 = param_partial[cbase_1 + (long long)(3 * n_rows + rowc_1)];
                        float _add_14;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(ll[k_7]), "f"(((row_1 < n_rows) ? xq_1 : 0.0f)));
                        ll[k_7] = _add_14;
                    }
                }
            } else {
                #pragma unroll 1
                for (int jb = 0; jb < nj; jb += 32) {
                    int last_row_2 = n_rows - 1;
                    long long cbase_2 = (long long)b * (long long)n_rows;
                    #pragma unroll
                    for (int k_8 = 0; k_8 < 1; k_8++) {
                        #pragma unroll
                        for (int u_2 = 0; u_2 < 32; u_2++) {
                            int row_2 = (jb + u_2) * 256 + (k_8 * 256 + tid_0);
                            int rowc_2 = ((row_2 < n_rows) ? row_2 : last_row_2);
                            float xr_2 = param_partial[cbase_2 + (long long)rowc_2];
                            float add_2;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(hh[k_8]), "f"(((row_2 < n_rows) ? xr_2 : 0.0f)));
                            float sub_2;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(add_2), "f"(hh[k_8]));
                            float _sub_18;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(add_2), "f"(sub_2));
                            float _sub_17;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(hh[k_8]), "f"(_sub_18));
                            float _sub_19;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(((row_2 < n_rows) ? xr_2 : 0.0f)), "f"(sub_2));
                            float _add_16;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(_sub_17), "f"(_sub_19));
                            float _add_15;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(ll[k_8]), "f"(_add_16));
                            ll[k_8] = _add_15;
                            hh[k_8] = add_2;
                            float xq_2 = param_partial[cbase_2 + (long long)(3 * n_rows + rowc_2)];
                            float _add_20;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(ll[k_8]), "f"(((row_2 < n_rows) ? xq_2 : 0.0f)));
                            ll[k_8] = _add_20;
                        }
                    }
                }
            }
        } else {
            #pragma unroll
            for (int k_9 = 0; k_9 < 1; k_9++) {
                hh[k_9] = 0.0f;
                ll[k_9] = 0.0f;
            }
            int nj_1 = n_rows + 255 >> 8;
            if (nj_1 <= 2) {
                int last_row_3 = n_rows - 1;
                long long cbase_3 = (long long)b * (long long)n_rows;
                #pragma unroll
                for (int k_10 = 0; k_10 < 1; k_10++) {
                    #pragma unroll
                    for (int u_3 = 0; u_3 < 2; u_3++) {
                        int row_3 = u_3 * 256 + (k_10 * 256 + tid_0);
                        int rowc_3 = ((row_3 < n_rows) ? row_3 : last_row_3);
                        float xr_3 = param_partial[cbase_3 + (long long)rowc_3];
                        float add_3;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(hh[k_10]), "f"(((row_3 < n_rows) ? xr_3 : 0.0f)));
                        float sub_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_3) : "f"(add_3), "f"(hh[k_10]));
                        float _sub_24;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(add_3), "f"(sub_3));
                        float _sub_23;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(hh[k_10]), "f"(_sub_24));
                        float _sub_25;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(((row_3 < n_rows) ? xr_3 : 0.0f)), "f"(sub_3));
                        float _add_22;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(_sub_23), "f"(_sub_25));
                        float _add_21;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(ll[k_10]), "f"(_add_22));
                        ll[k_10] = _add_21;
                        hh[k_10] = add_3;
                        float xq_3 = param_partial[cbase_3 + (long long)(32 * n_rows + rowc_3)];
                        float _add_26;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(ll[k_10]), "f"(((row_3 < n_rows) ? xq_3 : 0.0f)));
                        ll[k_10] = _add_26;
                    }
                }
            } else if (nj_1 <= 8) {
                int last_row_4 = n_rows - 1;
                long long cbase_4 = (long long)b * (long long)n_rows;
                #pragma unroll
                for (int k_11 = 0; k_11 < 1; k_11++) {
                    #pragma unroll
                    for (int u_4 = 0; u_4 < 8; u_4++) {
                        int row_4 = u_4 * 256 + (k_11 * 256 + tid_0);
                        int rowc_4 = ((row_4 < n_rows) ? row_4 : last_row_4);
                        float xr_4 = param_partial[cbase_4 + (long long)rowc_4];
                        float add_4;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(hh[k_11]), "f"(((row_4 < n_rows) ? xr_4 : 0.0f)));
                        float sub_4;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_4) : "f"(add_4), "f"(hh[k_11]));
                        float _sub_30;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(add_4), "f"(sub_4));
                        float _sub_29;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(hh[k_11]), "f"(_sub_30));
                        float _sub_31;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(((row_4 < n_rows) ? xr_4 : 0.0f)), "f"(sub_4));
                        float _add_28;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(_sub_29), "f"(_sub_31));
                        float _add_27;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(ll[k_11]), "f"(_add_28));
                        ll[k_11] = _add_27;
                        hh[k_11] = add_4;
                        float xq_4 = param_partial[cbase_4 + (long long)(32 * n_rows + rowc_4)];
                        float _add_32;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(ll[k_11]), "f"(((row_4 < n_rows) ? xq_4 : 0.0f)));
                        ll[k_11] = _add_32;
                    }
                }
            } else {
                #pragma unroll 1
                for (int jb_1 = 0; jb_1 < nj_1; jb_1 += 32) {
                    int last_row_5 = n_rows - 1;
                    long long cbase_5 = (long long)b * (long long)n_rows;
                    #pragma unroll
                    for (int k_12 = 0; k_12 < 1; k_12++) {
                        #pragma unroll
                        for (int u_5 = 0; u_5 < 32; u_5++) {
                            int row_5 = (jb_1 + u_5) * 256 + (k_12 * 256 + tid_0);
                            int rowc_5 = ((row_5 < n_rows) ? row_5 : last_row_5);
                            float xr_5 = param_partial[cbase_5 + (long long)rowc_5];
                            float add_5;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(hh[k_12]), "f"(((row_5 < n_rows) ? xr_5 : 0.0f)));
                            float sub_5;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_5) : "f"(add_5), "f"(hh[k_12]));
                            float _sub_36;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(add_5), "f"(sub_5));
                            float _sub_35;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(hh[k_12]), "f"(_sub_36));
                            float _sub_37;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(((row_5 < n_rows) ? xr_5 : 0.0f)), "f"(sub_5));
                            float _add_34;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(_sub_35), "f"(_sub_37));
                            float _add_33;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(ll[k_12]), "f"(_add_34));
                            ll[k_12] = _add_33;
                            hh[k_12] = add_5;
                            float xq_5 = param_partial[cbase_5 + (long long)(32 * n_rows + rowc_5)];
                            float _add_38;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(ll[k_12]), "f"(((row_5 < n_rows) ? xq_5 : 0.0f)));
                            ll[k_12] = _add_38;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int k_13 = 0; k_13 < 1; k_13++) {
            float oh = __shfl_xor_sync(0xFFFFFFFF, hh[k_13], 16);
            float ol = __shfl_xor_sync(0xFFFFFFFF, ll[k_13], 16);
            float n;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n) : "f"(hh[k_13]), "f"(oh));
            float bp;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp) : "f"(n), "f"(hh[k_13]));
            float ap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap) : "f"(n), "f"(bp));
            float _sub_39;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(hh[k_13]), "f"(ap));
            float _sub_40;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(oh), "f"(bp));
            float e;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(_sub_39), "f"(_sub_40));
            float _add_42;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(ll[k_13]), "f"(ol));
            float _add_41;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(_add_42), "f"(e));
            ll[k_13] = _add_41;
            hh[k_13] = n;
            float oh_0 = __shfl_xor_sync(0xFFFFFFFF, hh[k_13], 8);
            float ol_0 = __shfl_xor_sync(0xFFFFFFFF, ll[k_13], 8);
            float n_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_0) : "f"(hh[k_13]), "f"(oh_0));
            float bp_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_0) : "f"(n_0), "f"(hh[k_13]));
            float ap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_0) : "f"(n_0), "f"(bp_0));
            float _sub_43;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(hh[k_13]), "f"(ap_0));
            float _sub_44;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(oh_0), "f"(bp_0));
            float e_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_0) : "f"(_sub_43), "f"(_sub_44));
            float _add_46;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(ll[k_13]), "f"(ol_0));
            float _add_45;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(_add_46), "f"(e_0));
            ll[k_13] = _add_45;
            hh[k_13] = n_0;
            float oh_1 = __shfl_xor_sync(0xFFFFFFFF, hh[k_13], 4);
            float ol_1 = __shfl_xor_sync(0xFFFFFFFF, ll[k_13], 4);
            float n_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_1) : "f"(hh[k_13]), "f"(oh_1));
            float bp_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_1) : "f"(n_1), "f"(hh[k_13]));
            float ap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_1) : "f"(n_1), "f"(bp_1));
            float _sub_47;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(hh[k_13]), "f"(ap_1));
            float _sub_48;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(oh_1), "f"(bp_1));
            float e_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_1) : "f"(_sub_47), "f"(_sub_48));
            float _add_50;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(ll[k_13]), "f"(ol_1));
            float _add_49;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(_add_50), "f"(e_1));
            ll[k_13] = _add_49;
            hh[k_13] = n_1;
            float oh_2 = __shfl_xor_sync(0xFFFFFFFF, hh[k_13], 2);
            float ol_2 = __shfl_xor_sync(0xFFFFFFFF, ll[k_13], 2);
            float n_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_2) : "f"(hh[k_13]), "f"(oh_2));
            float bp_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_2) : "f"(n_2), "f"(hh[k_13]));
            float ap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_2) : "f"(n_2), "f"(bp_2));
            float _sub_51;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(hh[k_13]), "f"(ap_2));
            float _sub_52;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(oh_2), "f"(bp_2));
            float e_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_2) : "f"(_sub_51), "f"(_sub_52));
            float _add_54;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(ll[k_13]), "f"(ol_2));
            float _add_53;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(_add_54), "f"(e_2));
            ll[k_13] = _add_53;
            hh[k_13] = n_2;
            float oh_3 = __shfl_xor_sync(0xFFFFFFFF, hh[k_13], 1);
            float ol_3 = __shfl_xor_sync(0xFFFFFFFF, ll[k_13], 1);
            float n_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_3) : "f"(hh[k_13]), "f"(oh_3));
            float bp_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_3) : "f"(n_3), "f"(hh[k_13]));
            float ap_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_3) : "f"(n_3), "f"(bp_3));
            float _sub_55;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(hh[k_13]), "f"(ap_3));
            float _sub_56;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(oh_3), "f"(bp_3));
            float e_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_3) : "f"(_sub_55), "f"(_sub_56));
            float _add_58;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(ll[k_13]), "f"(ol_3));
            float _add_57;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(_add_58), "f"(e_3));
            ll[k_13] = _add_57;
            hh[k_13] = n_3;
            if (lane_1 == 0) {
                int vm = (k_13 * 8 + wrp) * 2;
                vw[vm] = hh[k_13];
                vw[vm + 1] = ll[k_13];
            }
        }
        __syncthreads();
        if (tid_0 == 0) {
            float fh[1];
            float fl[1];
            fh[0] = 0.0f;
            fl[0] = 0.0f;
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                float add_6;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_6) : "f"(fh[0]), "f"(vw[m * 2]));
                float sub_6;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_6) : "f"(add_6), "f"(fh[0]));
                float _sub_62;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(add_6), "f"(sub_6));
                float _sub_61;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(fh[0]), "f"(_sub_62));
                float _sub_63;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(vw[m * 2]), "f"(sub_6));
                float _add_60;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(_sub_61), "f"(_sub_63));
                float _add_59;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(fl[0]), "f"(_add_60));
                fl[0] = _add_59;
                fh[0] = add_6;
                float _add_64;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(fl[0]), "f"(vw[m * 2 + 1]));
                fl[0] = _add_64;
            }
            long long col64 = (long long)b;
            if (accumulate != 0) {
                if (b < 24) {
                    float cbv = dbase[col64];
                    float add_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(fh[0]), "f"(cbv));
                    float sub_7;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_7) : "f"(add_7), "f"(fh[0]));
                    float _sub_68;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(add_7), "f"(sub_7));
                    float _sub_67;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(fh[0]), "f"(_sub_68));
                    float _sub_69;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(cbv), "f"(sub_7));
                    float _add_66;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(_sub_67), "f"(_sub_69));
                    float _add_65;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(fl[0]), "f"(_add_66));
                    fl[0] = _add_65;
                    fh[0] = add_7;
                } else {
                    float csv = dscale[col64 - 24];
                    float add_8;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(fh[0]), "f"(csv));
                    float sub_8;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_8) : "f"(add_8), "f"(fh[0]));
                    float _sub_73;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_73) : "f"(add_8), "f"(sub_8));
                    float _sub_72;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(fh[0]), "f"(_sub_73));
                    float _sub_74;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(csv), "f"(sub_8));
                    float _add_71;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(_sub_72), "f"(_sub_74));
                    float _add_70;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(fl[0]), "f"(_add_71));
                    fl[0] = _add_70;
                    fh[0] = add_8;
                }
            }
            bool nonfin = (__as_u32(fh[0]) & 2139095040) == 2139095040;
            float _add_75;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(fh[0]), "f"(fl[0]));
            float res = ((nonfin) ? fh[0] : _add_75);
            if (b < 24) {
                *(reinterpret_cast<float*>(dbase + col64) + (0)) = res;
            } else {
                *(reinterpret_cast<float*>(dscale + (col64 - 24)) + (0)) = res;
            }
        }
    }
}

} // extern "C"
