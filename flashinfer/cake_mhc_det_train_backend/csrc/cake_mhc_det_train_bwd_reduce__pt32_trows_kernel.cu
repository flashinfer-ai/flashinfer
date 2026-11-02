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
#define THREADS 32

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

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(32) void
kernel_cake_mhc_det_train_bwd_reduce__pt32_trows(float* __restrict__ dfn_part, float* __restrict__ param_partial, float* __restrict__ dfn, float* __restrict__ dscale, float* __restrict__ dbase, int n_parts, int n_rows, int accumulate)
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
        long long e0 = (long long)((b - 27) * 32 + tid_0) * 4;
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
            unsigned int wu_s = __as_u32(_vec_load_0[k_1]);
            unsigned int wa_s = wu_s & 2147483647;
            unsigned int we_s = wa_s >> 23;
            unsigned int ws_s = __as_u32(mul_f32_rn_noftz(_vec_load_0[k_1], 1.6777216e+07f));
            unsigned int ww_s = ((we_s == 0) ? ws_s & 2147483647 : wa_s);
            unsigned int wj_s = ((we_s == 0) ? 914358272 : ((we_s == 255) ? 1879048192 : 939524096));
            unsigned int wg_s = wu_s & 2147483648u;
            unsigned int wm_s = wj_s + (ww_s >> 3);
            unsigned int wh_s = ((ww_s == 0) ? wg_s : wg_s | wm_s);
            unsigned int wp_s[2];
            wp_s[0] = ww_s << 29;
            wp_s[1] = wh_s;
            a[k_1] = ((_vec_load_0[k_1] != 0.0f) ? reinterpret_cast<double*>(wp_s)[0] : 0.0);
        }
        #pragma unroll 1
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
                unsigned int wu_s_1 = __as_u32(vec_load[k_2]);
                unsigned int wa_s_1 = wu_s_1 & 2147483647;
                unsigned int we_s_1 = wa_s_1 >> 23;
                unsigned int ws_s_1 = __as_u32(mul_f32_rn_noftz(vec_load[k_2], 1.6777216e+07f));
                unsigned int ww_s_1 = ((we_s_1 == 0) ? ws_s_1 & 2147483647 : wa_s_1);
                unsigned int wj_s_1 = ((we_s_1 == 0) ? 914358272 : ((we_s_1 == 255) ? 1879048192 : 939524096));
                unsigned int wg_s_1 = wu_s_1 & 2147483648u;
                unsigned int wm_s_1 = wj_s_1 + (ww_s_1 >> 3);
                unsigned int wh_s_1 = ((ww_s_1 == 0) ? wg_s_1 : wg_s_1 | wm_s_1);
                unsigned int wp_s_1[2];
                wp_s_1[0] = ww_s_1 << 29;
                wp_s_1[1] = wh_s_1;
                a[k_2] = a[k_2] + reinterpret_cast<double*>(wp_s_1)[0];
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
                unsigned int wu_s_2 = __as_u32(_vec_load_2[k_3]);
                unsigned int wa_s_2 = wu_s_2 & 2147483647;
                unsigned int we_s_2 = wa_s_2 >> 23;
                unsigned int ws_s_2 = __as_u32(mul_f32_rn_noftz(_vec_load_2[k_3], 1.6777216e+07f));
                unsigned int ww_s_2 = ((we_s_2 == 0) ? ws_s_2 & 2147483647 : wa_s_2);
                unsigned int wj_s_2 = ((we_s_2 == 0) ? 914358272 : ((we_s_2 == 255) ? 1879048192 : 939524096));
                unsigned int wg_s_2 = wu_s_2 & 2147483648u;
                unsigned int wm_s_2 = wj_s_2 + (ww_s_2 >> 3);
                unsigned int wh_s_2 = ((ww_s_2 == 0) ? wg_s_2 : wg_s_2 | wm_s_2);
                unsigned int wp_s_2[2];
                wp_s_2[0] = ww_s_2 << 29;
                wp_s_2[1] = wh_s_2;
                a[k_3] = a[k_3] + reinterpret_cast<double*>(wp_s_2)[0];
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
        float hh[8];
        float ll[8];
        if (b >= 24) {
            #pragma unroll
            for (int k_5 = 0; k_5 < 8; k_5++) {
                hh[k_5] = 0.0f;
                ll[k_5] = 0.0f;
            }
            int nj = n_rows + 255 >> 8;
            #pragma unroll 1
            for (int jb = 0; jb < nj; jb += 4) {
                int last_row = n_rows - 1;
                long long cbase = (long long)b * (long long)n_rows;
                #pragma unroll
                for (int k_6 = 0; k_6 < 8; k_6++) {
                    #pragma unroll
                    for (int u = 0; u < 4; u++) {
                        int row = (jb + u) * 256 + (k_6 * 32 + tid_0);
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
            }
        } else {
            #pragma unroll
            for (int k_7 = 0; k_7 < 8; k_7++) {
                hh[k_7] = 0.0f;
                ll[k_7] = 0.0f;
            }
            int nj_1 = n_rows + 255 >> 8;
            #pragma unroll 1
            for (int jb_1 = 0; jb_1 < nj_1; jb_1 += 4) {
                int last_row_1 = n_rows - 1;
                long long cbase_1 = (long long)b * (long long)n_rows;
                #pragma unroll
                for (int k_8 = 0; k_8 < 8; k_8++) {
                    #pragma unroll
                    for (int u_1 = 0; u_1 < 4; u_1++) {
                        int row_1 = (jb_1 + u_1) * 256 + (k_8 * 32 + tid_0);
                        int rowc_1 = ((row_1 < n_rows) ? row_1 : last_row_1);
                        float xr_1 = param_partial[cbase_1 + (long long)rowc_1];
                        float add_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(hh[k_8]), "f"(((row_1 < n_rows) ? xr_1 : 0.0f)));
                        float sub_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1) : "f"(add_1), "f"(hh[k_8]));
                        float _sub_12;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(add_1), "f"(sub_1));
                        float _sub_11;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(hh[k_8]), "f"(_sub_12));
                        float _sub_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(((row_1 < n_rows) ? xr_1 : 0.0f)), "f"(sub_1));
                        float _add_10;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(_sub_11), "f"(_sub_13));
                        float _add_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(ll[k_8]), "f"(_add_10));
                        ll[k_8] = _add_9;
                        hh[k_8] = add_1;
                        float xq_1 = param_partial[cbase_1 + (long long)(32 * n_rows + rowc_1)];
                        float _add_14;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(ll[k_8]), "f"(((row_1 < n_rows) ? xq_1 : 0.0f)));
                        ll[k_8] = _add_14;
                    }
                }
            }
        }
        #pragma unroll
        for (int k_9 = 0; k_9 < 8; k_9++) {
            float oh = __shfl_xor_sync(0xFFFFFFFF, hh[k_9], 16);
            float ol = __shfl_xor_sync(0xFFFFFFFF, ll[k_9], 16);
            float n;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n) : "f"(hh[k_9]), "f"(oh));
            float bp;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp) : "f"(n), "f"(hh[k_9]));
            float ap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap) : "f"(n), "f"(bp));
            float _sub_15;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(hh[k_9]), "f"(ap));
            float _sub_16;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(oh), "f"(bp));
            float e;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(_sub_15), "f"(_sub_16));
            float _add_18;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(ll[k_9]), "f"(ol));
            float _add_17;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(_add_18), "f"(e));
            ll[k_9] = _add_17;
            hh[k_9] = n;
            float oh_0 = __shfl_xor_sync(0xFFFFFFFF, hh[k_9], 8);
            float ol_0 = __shfl_xor_sync(0xFFFFFFFF, ll[k_9], 8);
            float n_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_0) : "f"(hh[k_9]), "f"(oh_0));
            float bp_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_0) : "f"(n_0), "f"(hh[k_9]));
            float ap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_0) : "f"(n_0), "f"(bp_0));
            float _sub_19;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(hh[k_9]), "f"(ap_0));
            float _sub_20;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(oh_0), "f"(bp_0));
            float e_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_0) : "f"(_sub_19), "f"(_sub_20));
            float _add_22;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(ll[k_9]), "f"(ol_0));
            float _add_21;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(_add_22), "f"(e_0));
            ll[k_9] = _add_21;
            hh[k_9] = n_0;
            float oh_1 = __shfl_xor_sync(0xFFFFFFFF, hh[k_9], 4);
            float ol_1 = __shfl_xor_sync(0xFFFFFFFF, ll[k_9], 4);
            float n_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_1) : "f"(hh[k_9]), "f"(oh_1));
            float bp_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_1) : "f"(n_1), "f"(hh[k_9]));
            float ap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_1) : "f"(n_1), "f"(bp_1));
            float _sub_23;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(hh[k_9]), "f"(ap_1));
            float _sub_24;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(oh_1), "f"(bp_1));
            float e_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_1) : "f"(_sub_23), "f"(_sub_24));
            float _add_26;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(ll[k_9]), "f"(ol_1));
            float _add_25;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(_add_26), "f"(e_1));
            ll[k_9] = _add_25;
            hh[k_9] = n_1;
            float oh_2 = __shfl_xor_sync(0xFFFFFFFF, hh[k_9], 2);
            float ol_2 = __shfl_xor_sync(0xFFFFFFFF, ll[k_9], 2);
            float n_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_2) : "f"(hh[k_9]), "f"(oh_2));
            float bp_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_2) : "f"(n_2), "f"(hh[k_9]));
            float ap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_2) : "f"(n_2), "f"(bp_2));
            float _sub_27;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(hh[k_9]), "f"(ap_2));
            float _sub_28;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(oh_2), "f"(bp_2));
            float e_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_2) : "f"(_sub_27), "f"(_sub_28));
            float _add_30;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(ll[k_9]), "f"(ol_2));
            float _add_29;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(_add_30), "f"(e_2));
            ll[k_9] = _add_29;
            hh[k_9] = n_2;
            float oh_3 = __shfl_xor_sync(0xFFFFFFFF, hh[k_9], 1);
            float ol_3 = __shfl_xor_sync(0xFFFFFFFF, ll[k_9], 1);
            float n_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_3) : "f"(hh[k_9]), "f"(oh_3));
            float bp_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_3) : "f"(n_3), "f"(hh[k_9]));
            float ap_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_3) : "f"(n_3), "f"(bp_3));
            float _sub_31;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(hh[k_9]), "f"(ap_3));
            float _sub_32;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(oh_3), "f"(bp_3));
            float e_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_3) : "f"(_sub_31), "f"(_sub_32));
            float _add_34;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(ll[k_9]), "f"(ol_3));
            float _add_33;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(_add_34), "f"(e_3));
            ll[k_9] = _add_33;
            hh[k_9] = n_3;
            if (lane_1 == 0) {
                int vm = (k_9 + wrp) * 2;
                vw[vm] = hh[k_9];
                vw[vm + 1] = ll[k_9];
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
                float add_2;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(fh[0]), "f"(vw[m * 2]));
                float sub_2;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(add_2), "f"(fh[0]));
                float _sub_38;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(add_2), "f"(sub_2));
                float _sub_37;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(fh[0]), "f"(_sub_38));
                float _sub_39;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(vw[m * 2]), "f"(sub_2));
                float _add_36;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(_sub_37), "f"(_sub_39));
                float _add_35;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(fl[0]), "f"(_add_36));
                fl[0] = _add_35;
                fh[0] = add_2;
                float _add_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(fl[0]), "f"(vw[m * 2 + 1]));
                fl[0] = _add_40;
            }
            long long col64 = (long long)b;
            if (accumulate != 0) {
                if (b < 24) {
                    float cbv = dbase[col64];
                    float add_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(fh[0]), "f"(cbv));
                    float sub_3;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_3) : "f"(add_3), "f"(fh[0]));
                    float _sub_44;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(add_3), "f"(sub_3));
                    float _sub_43;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(fh[0]), "f"(_sub_44));
                    float _sub_45;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(cbv), "f"(sub_3));
                    float _add_42;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(_sub_43), "f"(_sub_45));
                    float _add_41;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(fl[0]), "f"(_add_42));
                    fl[0] = _add_41;
                    fh[0] = add_3;
                } else {
                    float csv = dscale[col64 - 24];
                    float add_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(fh[0]), "f"(csv));
                    float sub_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_4) : "f"(add_4), "f"(fh[0]));
                    float _sub_49;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(add_4), "f"(sub_4));
                    float _sub_48;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(fh[0]), "f"(_sub_49));
                    float _sub_50;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(csv), "f"(sub_4));
                    float _add_47;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(_sub_48), "f"(_sub_50));
                    float _add_46;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(fl[0]), "f"(_add_47));
                    fl[0] = _add_46;
                    fh[0] = add_4;
                }
            }
            bool nonfin = (__as_u32(fh[0]) & 2139095040) == 2139095040;
            float _add_51;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(fh[0]), "f"(fl[0]));
            float res = ((nonfin) ? fh[0] : _add_51);
            if (b < 24) {
                *(reinterpret_cast<float*>(dbase + col64) + (0)) = res;
            } else {
                *(reinterpret_cast<float*>(dscale + (col64 - 24)) + (0)) = res;
            }
        }
    }
}

} // extern "C"
