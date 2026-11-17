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
kernel_cake_mhc_det_train_bwd_reduce__wpu4_fadd_trows__h5120(float* __restrict__ dfn_part, float* __restrict__ param_partial, float* __restrict__ dfn, float* __restrict__ dscale, float* __restrict__ dbase, int n_parts, int n_rows, int accumulate)
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

    // Prologue
    float* vw = reinterpret_cast<float*>(smem_raw + 0);
    const int vw_addr = smem + 0;

    // === Task calls (declaration order) ===
    int b = bid;
    int tid_0 = tid;
    if (b >= 27) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        long long e0 = (long long)((b - 27) * 256 + tid_0) * 4ll;
        float out[4];
        double a[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            a[k] = 0.0;
        }
        #pragma unroll 4
        for (int p = 0; p < n_parts; p++) {
            float vec_load[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn_part + ((long long)p * 491520ll + e0) + 0);
                vec_load[0 + 0] = _v4.x;
                vec_load[0 + 1] = _v4.y;
                vec_load[0 + 2] = _v4.z;
                vec_load[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_1 = 0; k_1 < 4; k_1++) {
                a[k_1] = a[k_1] + (double)vec_load[k_1];
            }
        }
        if (accumulate != 0) {
            float _vec_load_1[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn + e0 + 0);
                _vec_load_1[0 + 0] = _v4.x;
                _vec_load_1[0 + 1] = _v4.y;
                _vec_load_1[0 + 2] = _v4.z;
                _vec_load_1[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_2 = 0; k_2 < 4; k_2++) {
                a[k_2] = a[k_2] + (double)_vec_load_1[k_2];
            }
        }
        #pragma unroll
        for (int k_3 = 0; k_3 < 4; k_3++) {
            out[k_3] = (float)a[k_3];
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
            for (int k_4 = 0; k_4 < 1; k_4++) {
                hh[k_4] = 0.0f;
                ll[k_4] = 0.0f;
            }
            int nj = n_rows + 255 >> 8;
            if (nj <= 2) {
                int last_row = n_rows - 1;
                long long cbase = (long long)b * (long long)n_rows;
                #pragma unroll
                for (int k_5 = 0; k_5 < 1; k_5++) {
                    #pragma unroll
                    for (int u = 0; u < 2; u++) {
                        int row = u * 256 + (k_5 * 256 + tid_0);
                        int rowc = ((row < n_rows) ? row : last_row);
                        float xv[1];
                        {
                            uint32_t _scalar_bits_2;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_2) : "l"((const void*)(param_partial + (cbase + (long long)rowc) + (0))) : "memory");
                            xv[0] = __uint_as_float(_scalar_bits_2);
                        }
                        float xr = xv[0];
                        float add;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(hh[k_5]), "f"(((row < n_rows) ? xr : 0.0f)));
                        float sub;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub) : "f"(add), "f"(hh[k_5]));
                        float _sub_6;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(add), "f"(sub));
                        float _sub_5;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(hh[k_5]), "f"(_sub_6));
                        float _sub_7;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(((row < n_rows) ? xr : 0.0f)), "f"(sub));
                        float _add_4;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(_sub_5), "f"(_sub_7));
                        float _add_3;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(ll[k_5]), "f"(_add_4));
                        ll[k_5] = _add_3;
                        hh[k_5] = add;
                        float xw[1];
                        {
                            uint32_t _scalar_bits_8;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_8) : "l"((const void*)(param_partial + (cbase + (long long)(3 * n_rows + rowc)) + (0))) : "memory");
                            xw[0] = __uint_as_float(_scalar_bits_8);
                        }
                        float xq = xw[0];
                        float _add_9;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(ll[k_5]), "f"(((row < n_rows) ? xq : 0.0f)));
                        ll[k_5] = _add_9;
                    }
                }
            } else if (nj <= 8) {
                int last_row_1 = n_rows - 1;
                long long cbase_1 = (long long)b * (long long)n_rows;
                #pragma unroll
                for (int k_6 = 0; k_6 < 1; k_6++) {
                    #pragma unroll
                    for (int u_1 = 0; u_1 < 8; u_1++) {
                        int row_1 = u_1 * 256 + (k_6 * 256 + tid_0);
                        int rowc_1 = ((row_1 < n_rows) ? row_1 : last_row_1);
                        float xv_1[1];
                        {
                            uint32_t _scalar_bits_10;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_10) : "l"((const void*)(param_partial + (cbase_1 + (long long)rowc_1) + (0))) : "memory");
                            xv_1[0] = __uint_as_float(_scalar_bits_10);
                        }
                        float xr_1 = xv_1[0];
                        float add_1;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(hh[k_6]), "f"(((row_1 < n_rows) ? xr_1 : 0.0f)));
                        float sub_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1) : "f"(add_1), "f"(hh[k_6]));
                        float _sub_14;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(add_1), "f"(sub_1));
                        float _sub_13;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(hh[k_6]), "f"(_sub_14));
                        float _sub_15;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(((row_1 < n_rows) ? xr_1 : 0.0f)), "f"(sub_1));
                        float _add_12;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(_sub_13), "f"(_sub_15));
                        float _add_11;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(ll[k_6]), "f"(_add_12));
                        ll[k_6] = _add_11;
                        hh[k_6] = add_1;
                        float xw_1[1];
                        {
                            uint32_t _scalar_bits_16;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_16) : "l"((const void*)(param_partial + (cbase_1 + (long long)(3 * n_rows + rowc_1)) + (0))) : "memory");
                            xw_1[0] = __uint_as_float(_scalar_bits_16);
                        }
                        float xq_1 = xw_1[0];
                        float _add_17;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(ll[k_6]), "f"(((row_1 < n_rows) ? xq_1 : 0.0f)));
                        ll[k_6] = _add_17;
                    }
                }
            } else {
                #pragma unroll 1
                for (int jb = 0; jb < nj; jb += 32) {
                    int last_row_2 = n_rows - 1;
                    long long cbase_2 = (long long)b * (long long)n_rows;
                    #pragma unroll
                    for (int k_7 = 0; k_7 < 1; k_7++) {
                        #pragma unroll
                        for (int u_2 = 0; u_2 < 32; u_2++) {
                            int row_2 = (jb + u_2) * 256 + (k_7 * 256 + tid_0);
                            int rowc_2 = ((row_2 < n_rows) ? row_2 : last_row_2);
                            float xv_2[1];
                            {
                                uint32_t _scalar_bits_18;
                                asm volatile("ld.global.cg.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_18) : "l"((const void*)(param_partial + (cbase_2 + (long long)rowc_2) + (0))) : "memory");
                                xv_2[0] = __uint_as_float(_scalar_bits_18);
                            }
                            float xr_2 = xv_2[0];
                            float add_2;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(hh[k_7]), "f"(((row_2 < n_rows) ? xr_2 : 0.0f)));
                            float sub_2;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(add_2), "f"(hh[k_7]));
                            float _sub_22;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(add_2), "f"(sub_2));
                            float _sub_21;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(hh[k_7]), "f"(_sub_22));
                            float _sub_23;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(((row_2 < n_rows) ? xr_2 : 0.0f)), "f"(sub_2));
                            float _add_20;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(_sub_21), "f"(_sub_23));
                            float _add_19;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(ll[k_7]), "f"(_add_20));
                            ll[k_7] = _add_19;
                            hh[k_7] = add_2;
                            float xw_2[1];
                            {
                                uint32_t _scalar_bits_24;
                                asm volatile("ld.global.cg.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_24) : "l"((const void*)(param_partial + (cbase_2 + (long long)(3 * n_rows + rowc_2)) + (0))) : "memory");
                                xw_2[0] = __uint_as_float(_scalar_bits_24);
                            }
                            float xq_2 = xw_2[0];
                            float _add_25;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(ll[k_7]), "f"(((row_2 < n_rows) ? xq_2 : 0.0f)));
                            ll[k_7] = _add_25;
                        }
                    }
                }
            }
        } else {
            #pragma unroll
            for (int k_8 = 0; k_8 < 1; k_8++) {
                hh[k_8] = 0.0f;
                ll[k_8] = 0.0f;
            }
            int nj_1 = n_rows + 255 >> 8;
            if (nj_1 <= 2) {
                int last_row_3 = n_rows - 1;
                long long cbase_3 = (long long)b * (long long)n_rows;
                #pragma unroll
                for (int k_9 = 0; k_9 < 1; k_9++) {
                    #pragma unroll
                    for (int u_3 = 0; u_3 < 2; u_3++) {
                        int row_3 = u_3 * 256 + (k_9 * 256 + tid_0);
                        int rowc_3 = ((row_3 < n_rows) ? row_3 : last_row_3);
                        float xv_3[1];
                        {
                            uint32_t _scalar_bits_26;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_26) : "l"((const void*)(param_partial + (cbase_3 + (long long)rowc_3) + (0))) : "memory");
                            xv_3[0] = __uint_as_float(_scalar_bits_26);
                        }
                        float xr_3 = xv_3[0];
                        float add_3;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(hh[k_9]), "f"(((row_3 < n_rows) ? xr_3 : 0.0f)));
                        float sub_3;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_3) : "f"(add_3), "f"(hh[k_9]));
                        float _sub_30;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(add_3), "f"(sub_3));
                        float _sub_29;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(hh[k_9]), "f"(_sub_30));
                        float _sub_31;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(((row_3 < n_rows) ? xr_3 : 0.0f)), "f"(sub_3));
                        float _add_28;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(_sub_29), "f"(_sub_31));
                        float _add_27;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(ll[k_9]), "f"(_add_28));
                        ll[k_9] = _add_27;
                        hh[k_9] = add_3;
                        float xw_3[1];
                        {
                            uint32_t _scalar_bits_32;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_32) : "l"((const void*)(param_partial + (cbase_3 + (long long)(32 * n_rows + rowc_3)) + (0))) : "memory");
                            xw_3[0] = __uint_as_float(_scalar_bits_32);
                        }
                        float xq_3 = xw_3[0];
                        float _add_33;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(ll[k_9]), "f"(((row_3 < n_rows) ? xq_3 : 0.0f)));
                        ll[k_9] = _add_33;
                    }
                }
            } else if (nj_1 <= 8) {
                int last_row_4 = n_rows - 1;
                long long cbase_4 = (long long)b * (long long)n_rows;
                #pragma unroll
                for (int k_10 = 0; k_10 < 1; k_10++) {
                    #pragma unroll
                    for (int u_4 = 0; u_4 < 8; u_4++) {
                        int row_4 = u_4 * 256 + (k_10 * 256 + tid_0);
                        int rowc_4 = ((row_4 < n_rows) ? row_4 : last_row_4);
                        float xv_4[1];
                        {
                            uint32_t _scalar_bits_34;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_34) : "l"((const void*)(param_partial + (cbase_4 + (long long)rowc_4) + (0))) : "memory");
                            xv_4[0] = __uint_as_float(_scalar_bits_34);
                        }
                        float xr_4 = xv_4[0];
                        float add_4;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(hh[k_10]), "f"(((row_4 < n_rows) ? xr_4 : 0.0f)));
                        float sub_4;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_4) : "f"(add_4), "f"(hh[k_10]));
                        float _sub_38;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(add_4), "f"(sub_4));
                        float _sub_37;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(hh[k_10]), "f"(_sub_38));
                        float _sub_39;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(((row_4 < n_rows) ? xr_4 : 0.0f)), "f"(sub_4));
                        float _add_36;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(_sub_37), "f"(_sub_39));
                        float _add_35;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(ll[k_10]), "f"(_add_36));
                        ll[k_10] = _add_35;
                        hh[k_10] = add_4;
                        float xw_4[1];
                        {
                            uint32_t _scalar_bits_40;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_40) : "l"((const void*)(param_partial + (cbase_4 + (long long)(32 * n_rows + rowc_4)) + (0))) : "memory");
                            xw_4[0] = __uint_as_float(_scalar_bits_40);
                        }
                        float xq_4 = xw_4[0];
                        float _add_41;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(ll[k_10]), "f"(((row_4 < n_rows) ? xq_4 : 0.0f)));
                        ll[k_10] = _add_41;
                    }
                }
            } else {
                #pragma unroll 1
                for (int jb_1 = 0; jb_1 < nj_1; jb_1 += 32) {
                    int last_row_5 = n_rows - 1;
                    long long cbase_5 = (long long)b * (long long)n_rows;
                    #pragma unroll
                    for (int k_11 = 0; k_11 < 1; k_11++) {
                        #pragma unroll
                        for (int u_5 = 0; u_5 < 32; u_5++) {
                            int row_5 = (jb_1 + u_5) * 256 + (k_11 * 256 + tid_0);
                            int rowc_5 = ((row_5 < n_rows) ? row_5 : last_row_5);
                            float xv_5[1];
                            {
                                uint32_t _scalar_bits_42;
                                asm volatile("ld.global.cg.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_42) : "l"((const void*)(param_partial + (cbase_5 + (long long)rowc_5) + (0))) : "memory");
                                xv_5[0] = __uint_as_float(_scalar_bits_42);
                            }
                            float xr_5 = xv_5[0];
                            float add_5;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(hh[k_11]), "f"(((row_5 < n_rows) ? xr_5 : 0.0f)));
                            float sub_5;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_5) : "f"(add_5), "f"(hh[k_11]));
                            float _sub_46;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(add_5), "f"(sub_5));
                            float _sub_45;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(hh[k_11]), "f"(_sub_46));
                            float _sub_47;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(((row_5 < n_rows) ? xr_5 : 0.0f)), "f"(sub_5));
                            float _add_44;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(_sub_45), "f"(_sub_47));
                            float _add_43;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(ll[k_11]), "f"(_add_44));
                            ll[k_11] = _add_43;
                            hh[k_11] = add_5;
                            float xw_5[1];
                            {
                                uint32_t _scalar_bits_48;
                                asm volatile("ld.global.cg.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_48) : "l"((const void*)(param_partial + (cbase_5 + (long long)(32 * n_rows + rowc_5)) + (0))) : "memory");
                                xw_5[0] = __uint_as_float(_scalar_bits_48);
                            }
                            float xq_5 = xw_5[0];
                            float _add_49;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(ll[k_11]), "f"(((row_5 < n_rows) ? xq_5 : 0.0f)));
                            ll[k_11] = _add_49;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int k_12 = 0; k_12 < 1; k_12++) {
            float oh = __shfl_xor_sync(0xFFFFFFFF, hh[k_12], 16);
            float ol = __shfl_xor_sync(0xFFFFFFFF, ll[k_12], 16);
            float n;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n) : "f"(hh[k_12]), "f"(oh));
            float bp;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp) : "f"(n), "f"(hh[k_12]));
            float ap;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap) : "f"(n), "f"(bp));
            float _sub_50;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(hh[k_12]), "f"(ap));
            float _sub_51;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(oh), "f"(bp));
            float e;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e) : "f"(_sub_50), "f"(_sub_51));
            float _add_53;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(ll[k_12]), "f"(ol));
            float _add_52;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(_add_53), "f"(e));
            ll[k_12] = _add_52;
            hh[k_12] = n;
            float oh_0 = __shfl_xor_sync(0xFFFFFFFF, hh[k_12], 8);
            float ol_0 = __shfl_xor_sync(0xFFFFFFFF, ll[k_12], 8);
            float n_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_0) : "f"(hh[k_12]), "f"(oh_0));
            float bp_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_0) : "f"(n_0), "f"(hh[k_12]));
            float ap_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_0) : "f"(n_0), "f"(bp_0));
            float _sub_54;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(hh[k_12]), "f"(ap_0));
            float _sub_55;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(oh_0), "f"(bp_0));
            float e_0;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_0) : "f"(_sub_54), "f"(_sub_55));
            float _add_57;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(ll[k_12]), "f"(ol_0));
            float _add_56;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(_add_57), "f"(e_0));
            ll[k_12] = _add_56;
            hh[k_12] = n_0;
            float oh_1 = __shfl_xor_sync(0xFFFFFFFF, hh[k_12], 4);
            float ol_1 = __shfl_xor_sync(0xFFFFFFFF, ll[k_12], 4);
            float n_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_1) : "f"(hh[k_12]), "f"(oh_1));
            float bp_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_1) : "f"(n_1), "f"(hh[k_12]));
            float ap_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_1) : "f"(n_1), "f"(bp_1));
            float _sub_58;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(hh[k_12]), "f"(ap_1));
            float _sub_59;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(oh_1), "f"(bp_1));
            float e_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_1) : "f"(_sub_58), "f"(_sub_59));
            float _add_61;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(ll[k_12]), "f"(ol_1));
            float _add_60;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(_add_61), "f"(e_1));
            ll[k_12] = _add_60;
            hh[k_12] = n_1;
            float oh_2 = __shfl_xor_sync(0xFFFFFFFF, hh[k_12], 2);
            float ol_2 = __shfl_xor_sync(0xFFFFFFFF, ll[k_12], 2);
            float n_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_2) : "f"(hh[k_12]), "f"(oh_2));
            float bp_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_2) : "f"(n_2), "f"(hh[k_12]));
            float ap_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_2) : "f"(n_2), "f"(bp_2));
            float _sub_62;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(hh[k_12]), "f"(ap_2));
            float _sub_63;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(oh_2), "f"(bp_2));
            float e_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_2) : "f"(_sub_62), "f"(_sub_63));
            float _add_65;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(ll[k_12]), "f"(ol_2));
            float _add_64;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(_add_65), "f"(e_2));
            ll[k_12] = _add_64;
            hh[k_12] = n_2;
            float oh_3 = __shfl_xor_sync(0xFFFFFFFF, hh[k_12], 1);
            float ol_3 = __shfl_xor_sync(0xFFFFFFFF, ll[k_12], 1);
            float n_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(n_3) : "f"(hh[k_12]), "f"(oh_3));
            float bp_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(bp_3) : "f"(n_3), "f"(hh[k_12]));
            float ap_3;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(ap_3) : "f"(n_3), "f"(bp_3));
            float _sub_66;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(hh[k_12]), "f"(ap_3));
            float _sub_67;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(oh_3), "f"(bp_3));
            float e_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(e_3) : "f"(_sub_66), "f"(_sub_67));
            float _add_69;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(ll[k_12]), "f"(ol_3));
            float _add_68;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(_add_69), "f"(e_3));
            ll[k_12] = _add_68;
            hh[k_12] = n_3;
            if (lane_1 == 0) {
                int vm = (k_12 * 8 + wrp) * 2;
                vw[vm] = hh[k_12];
                vw[vm + 1] = ll[k_12];
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
                float _sub_73;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_73) : "f"(add_6), "f"(sub_6));
                float _sub_72;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(fh[0]), "f"(_sub_73));
                float _sub_74;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(vw[m * 2]), "f"(sub_6));
                float _add_71;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(_sub_72), "f"(_sub_74));
                float _add_70;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(fl[0]), "f"(_add_71));
                fl[0] = _add_70;
                fh[0] = add_6;
                float _add_75;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(fl[0]), "f"(vw[m * 2 + 1]));
                fl[0] = _add_75;
            }
            long long col64 = (long long)b;
            if (accumulate != 0) {
                if (b < 24) {
                    float cbv = dbase[col64];
                    float add_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(fh[0]), "f"(cbv));
                    float sub_7;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_7) : "f"(add_7), "f"(fh[0]));
                    float _sub_79;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(add_7), "f"(sub_7));
                    float _sub_78;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(fh[0]), "f"(_sub_79));
                    float _sub_80;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(cbv), "f"(sub_7));
                    float _add_77;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(_sub_78), "f"(_sub_80));
                    float _add_76;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(fl[0]), "f"(_add_77));
                    fl[0] = _add_76;
                    fh[0] = add_7;
                } else {
                    float csv = dscale[col64 - 24ll];
                    float add_8;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(fh[0]), "f"(csv));
                    float sub_8;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_8) : "f"(add_8), "f"(fh[0]));
                    float _sub_84;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(add_8), "f"(sub_8));
                    float _sub_83;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_83) : "f"(fh[0]), "f"(_sub_84));
                    float _sub_85;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(csv), "f"(sub_8));
                    float _add_82;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(_sub_83), "f"(_sub_85));
                    float _add_81;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(fl[0]), "f"(_add_82));
                    fl[0] = _add_81;
                    fh[0] = add_8;
                }
            }
            bool nonfin = (__as_u32(fh[0]) & 2139095040) == 2139095040;
            float _add_86;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(fh[0]), "f"(fl[0]));
            float res = ((nonfin) ? fh[0] : _add_86);
            if (b < 24) {
                *(reinterpret_cast<float*>(dbase + col64) + (0)) = res;
            } else {
                *(reinterpret_cast<float*>(dscale + (col64 - 24ll)) + (0)) = res;
            }
        }
    }
}

} // extern "C"
