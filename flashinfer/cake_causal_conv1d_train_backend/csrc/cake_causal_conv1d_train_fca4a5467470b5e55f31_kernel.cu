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
#define SMEM_RED_OFF 0
#define SMEM_RED_STAGE_BYTES 8192
#define SMEM_RED_STRIDE 8192
#define SMEM_TOTAL 8192
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>
#include <cooperative_groups.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_causal_conv1d_train_fca4a5467470b5e55f31(__nv_bfloat16* __restrict__ x, float* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, int S, int C, int num_docs, int search_steps, int ct_vec, int n_sub, int n_ch_tiles)
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
    float* red = reinterpret_cast<float*>(smem_raw + 0);
    const int red_addr = smem + 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long C64 = (long long)C;
    int ct = bid % n_ch_tiles;
    int rank = bid / n_ch_tiles;
    int g_ct = (num_bids - ct + n_ch_tiles - 1) / n_ch_tiles;
    int sub = tid / ct_vec;
    int v = tid - sub * ct_vec;
    int c0 = (ct * ct_vec + v) * 4;
    int streams = g_ct * n_sub;
    int L = (S + streams - 1) / streams;
    int t0 = (rank * n_sub + sub) * L;
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i] = 0.0f;
    }
    if (sub < n_sub && c0 < C && t0 < S) {
        int _min_0 = ((t0 + L) < (S) ? (t0 + L) : (S));
        int t_end = _min_0;
        int _min_1 = ((t_end + 3) < (S + 1) ? (t_end + 3) : (S + 1));
        int t_stop = _min_1;
        int _min_2 = ((t_stop) < (S) ? (t_stop) : (S));
        int t_lim = _min_2;
        long long c64 = (long long)c0;
        float wr[16];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(w + (c64 + (long long)j) * 4);
                wr[j * 4 + 0] = _v4.x;
                wr[j * 4 + 1] = _v4.y;
                wr[j * 4 + 2] = _v4.z;
                wr[j * 4 + 3] = _v4.w;
            }
        }
        int lo = 0;
        int hi = num_docs - 1;
        #pragma unroll 1
        for (int step = 0; step < search_steps; step++) {
            int mid = lo + hi + 1 >> 1;
            if (t0 >= cu[mid]) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        int doc_start = cu[lo];
        int doc_end = cu[lo + 1];
        float win[16];
        float pend[16];
        #pragma unroll
        for (int i_1 = 0; i_1 < 16; i_1++) {
            win[i_1] = 0.0f;
            pend[i_1] = 0.0f;
        }
        #pragma unroll
        for (int k = 0; k < 3; k++) {
            int tk = t0 - 3 + k;
            if (tk >= doc_start) {
                long long rowk = (long long)tk * C64 + c64;
                {
                    uint2 _vld_1;
                    _vld_1 = *reinterpret_cast<const uint2*>(x + rowk);
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&win[(k + 1) * 4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                        (&win[(k + 1) * 4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                    }
                }
            }
        }
        unsigned int pkx[8];
        unsigned int pkd[8];
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            int tu = t0 + u;
            if (tu < t_lim) {
                long long rowu = (long long)tu * C64 + c64;
                {
                    uint2 _vld_2;
                    asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                        : "=r"(_vld_2.x), "=r"(_vld_2.y) : "l"((const void*)(x + rowu)), "l"(0x12F0000000000000ULL) : "memory");
                    uint2* _vdst_2 = reinterpret_cast<uint2*>(&pkx[u * 2]);
                    *_vdst_2 = _vld_2;
                }
                {
                    uint2 _vld_3;
                    asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                        : "=r"(_vld_3.x), "=r"(_vld_3.y) : "l"((const void*)(dy + rowu)), "l"(0x12F0000000000000ULL) : "memory");
                    uint2* _vdst_3 = reinterpret_cast<uint2*>(&pkd[u * 2]);
                    *_vdst_3 = _vld_3;
                }
            }
        }
        float dyf[4];
        float dp[4];
        #pragma unroll 1
        for (int tb = t0; tb < t_stop; tb += 4) {
            #pragma unroll
            for (int u_1 = 0; u_1 < 4; u_1++) {
                int tcur = tb + u_1;
                if (tcur < t_stop) {
                    if (tcur < t_lim) {
                        #pragma unroll
                        for (int k_1 = 0; k_1 < 2; k_1++) {
                            win[u_1 % 4 * 4 + 2 * k_1] = __uint_as_float(pkx[u_1 * 2 + k_1] << 16);
                            win[u_1 % 4 * 4 + 2 * k_1 + 1] = __uint_as_float(pkx[u_1 * 2 + k_1] & 4294901760u);
                            dyf[2 * k_1] = __uint_as_float(pkd[u_1 * 2 + k_1] << 16);
                            dyf[2 * k_1 + 1] = __uint_as_float(pkd[u_1 * 2 + k_1] & 4294901760u);
                        }
                    }
                    if ((u_1 + 1) % 4 == 0) {
                        #pragma unroll
                        for (int g = 0; g < 4; g++) {
                            int tn = tb + (u_1 + 1 - 4 + g) + 4;
                            if (tn < t_lim) {
                                long long rown = (long long)tn * C64 + c64;
                                {
                                    uint2 _vld_4;
                                    asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                                        : "=r"(_vld_4.x), "=r"(_vld_4.y) : "l"((const void*)(x + rown)), "l"(0x12F0000000000000ULL) : "memory");
                                    uint2* _vdst_4 = reinterpret_cast<uint2*>(&pkx[(u_1 + 1 - 4 + g) * 2]);
                                    *_vdst_4 = _vld_4;
                                }
                                {
                                    uint2 _vld_5;
                                    asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                                        : "=r"(_vld_5.x), "=r"(_vld_5.y) : "l"((const void*)(dy + rown)), "l"(0x12F0000000000000ULL) : "memory");
                                    uint2* _vdst_5 = reinterpret_cast<uint2*>(&pkd[(u_1 + 1 - 4 + g) * 2]);
                                    *_vdst_5 = _vld_5;
                                }
                            }
                        }
                    }
                    if (tcur >= doc_end) {
                        #pragma unroll
                        for (int m = 1; m < 4; m++) {
                            int tm = tcur - m;
                            if (tm >= t0 && tm < t_end && tm >= doc_start) {
                                long long rowm = (long long)tm * C64 + c64;
                                {
                                    uint2 _pk2;
                                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                    _pk[0] = __floats2bfloat162_rn(pend[(u_1 % 4 + 4 - m) % 4 * 4 + 0], pend[(u_1 % 4 + 4 - m) % 4 * 4 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(pend[(u_1 % 4 + 4 - m) % 4 * 4 + 2], pend[(u_1 % 4 + 4 - m) % 4 * 4 + 3]);
                                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(dx))[rowm]) = _pk2;
                                }
                            }
                        }
                        if (tcur < S) {
                            int lo2 = 0;
                            int hi2 = num_docs - 1;
                            #pragma unroll 1
                            for (int step2 = 0; step2 < search_steps; step2++) {
                                int mid2 = lo2 + hi2 + 1 >> 1;
                                if (tcur >= cu[mid2]) {
                                    lo2 = mid2;
                                } else {
                                    hi2 = mid2 - 1;
                                }
                            }
                            doc_start = cu[lo2];
                            doc_end = cu[lo2 + 1];
                        } else {
                            doc_start = tcur + 4;
                            doc_end = tcur + 4;
                        }
                        #pragma unroll
                        for (int m_1 = 1; m_1 < 4; m_1++) {
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 4; j_1++) {
                                win[(u_1 % 4 + 4 - m_1) % 4 * 4 + j_1] = 0.0f;
                            }
                        }
                    }
                    if (tcur < S) {
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 4; j_2++) {
                            float _mul_0;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(wr[j_2 * 4]), "f"(win[(u_1 % 4 + 1) % 4 * 4 + j_2]));
                            float _mul_1;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(wr[j_2 * 4 + 1]), "f"(win[(u_1 % 4 + 2) % 4 * 4 + j_2]));
                            float _add_0;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(_mul_0), "f"(_mul_1));
                            float _mul_2;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(wr[j_2 * 4 + 2]), "f"(win[(u_1 % 4 + 3) % 4 * 4 + j_2]));
                            float _add_1;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(_add_0), "f"(_mul_2));
                            float _mul_3;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(wr[j_2 * 4 + 3]), "f"(win[u_1 % 4 * 4 + j_2]));
                            float _add_2;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(_add_1), "f"(_mul_3));
                            __nv_bfloat16 _cvt_bf16_0 = (__nv_bfloat16)_add_2;
                            float _cvt_f32_0 = (float)_cvt_bf16_0;
                            float p = _cvt_f32_0;
                            float e = (-p) * 1.4426950408889634f;
                            float _exp2_0 = approx_exp2(e);
                            float den = 1.0f + _exp2_0;
                            float _rcp_0 = approx_rcp(den);
                            float sg = _rcp_0;
                            float g_1 = sg * __fmaf_rn(p, 1.0f - sg, 1.0f);
                            dp[j_2] = dyf[j_2] * g_1;
                        }
                        if (tcur < t_end) {
                            #pragma unroll
                            for (int j_3 = 0; j_3 < 4; j_3++) {
                                #pragma unroll
                                for (int k_2 = 0; k_2 < 4; k_2++) {
                                    acc[j_3 * 4 + k_2] = __fmaf_rn(dp[j_3], win[(u_1 % 4 + 1 + k_2) % 4 * 4 + j_3], acc[j_3 * 4 + k_2]);
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 4; j_4++) {
                            pend[(u_1 % 4 + 1) % 4 * 4 + j_4] = __fmaf_rn(wr[j_4 * 4], dp[j_4], pend[(u_1 % 4 + 1) % 4 * 4 + j_4]);
                            pend[(u_1 % 4 + 2) % 4 * 4 + j_4] = __fmaf_rn(wr[j_4 * 4 + 1], dp[j_4], pend[(u_1 % 4 + 2) % 4 * 4 + j_4]);
                            pend[(u_1 % 4 + 3) % 4 * 4 + j_4] = __fmaf_rn(wr[j_4 * 4 + 2], dp[j_4], pend[(u_1 % 4 + 3) % 4 * 4 + j_4]);
                            pend[u_1 % 4 * 4 + j_4] = wr[j_4 * 4 + 3] * dp[j_4];
                        }
                        int t3 = tcur - 3;
                        if (t3 >= t0 && t3 >= doc_start) {
                            long long row3 = (long long)t3 * C64 + c64;
                            {
                                uint2 _pk2;
                                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                _pk[0] = __floats2bfloat162_rn(pend[(u_1 % 4 + 1) % 4 * 4 + 0], pend[(u_1 % 4 + 1) % 4 * 4 + 1]);
                                _pk[1] = __floats2bfloat162_rn(pend[(u_1 % 4 + 1) % 4 * 4 + 2], pend[(u_1 % 4 + 1) % 4 * 4 + 3]);
                                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(dx))[row3]) = _pk2;
                            }
                        }
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; i_2++) {
        red[tid * 16 + i_2] = acc[i_2];
    }
    __syncthreads();
    int c0_0 = (ct * ct_vec + v) * 4;
    if (sub == 0 && c0_0 < C) {
        float tot[16];
        #pragma unroll
        for (int i_3 = 0; i_3 < 16; i_3++) {
            tot[i_3] = red[v * 16 + i_3];
        }
        #pragma unroll 1
        for (int s = 1; s < n_sub; s++) {
            #pragma unroll
            for (int i_4 = 0; i_4 < 16; i_4++) {
                tot[i_4] = tot[i_4] + red[(s * ct_vec + v) * 16 + i_4];
            }
        }
        #pragma unroll
        for (int j_5 = 0; j_5 < 4; j_5++) {
            long long pidx = ((long long)rank * C64 + (long long)c0_0 + (long long)j_5) * 4;
            {
                float4 _v4 = make_float4(tot[j_5 * 4 + 0], tot[j_5 * 4 + 1], tot[j_5 * 4 + 2], tot[j_5 * 4 + 3]);
                *reinterpret_cast<float4*>(partial + pidx) = _v4;
            }
        }
    }
    cooperative_groups::this_grid().sync();
    int lane_1 = tid & 31;
    int wch = tid >> 5;
    int n_groups = (C + 4 - 1) / 4;
    #pragma unroll 1
    for (int cg = bid; cg < n_groups; cg += num_bids) {
        int ch = cg * 4 + wch;
        if (ch < C) {
            int ctc = ch / 4 / ct_vec;
            int G = (num_bids - ctc + n_ch_tiles - 1) / n_ch_tiles;
            int n_trips = G + 31 >> 5;
            float a4[4];
            #pragma unroll
            for (int e_1 = 0; e_1 < 4; e_1++) {
                a4[e_1] = 0.0f;
            }
            #pragma unroll 4
            for (int k_3 = 0; k_3 < n_trips; k_3++) {
                int r = lane_1 + (k_3 << 5);
                if (r < G) {
                    float _vec_load_0[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial + ((long long)r * C64 + (long long)ch) * 4);
                        _vec_load_0[0 + 0] = _v4.x;
                        _vec_load_0[0 + 1] = _v4.y;
                        _vec_load_0[0 + 2] = _v4.z;
                        _vec_load_0[0 + 3] = _v4.w;
                    }
                    #pragma unroll
                    for (int e_2 = 0; e_2 < 4; e_2++) {
                        a4[e_2] = a4[e_2] + _vec_load_0[e_2];
                    }
                }
            }
            #pragma unroll
            for (int e_3 = 0; e_3 < 4; e_3++) {
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, a4[e_3], 16);
                float other = _shfl_xor_0;
                a4[e_3] = a4[e_3] + other;
            }
            #pragma unroll
            for (int e_4 = 0; e_4 < 4; e_4++) {
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, a4[e_4], 8);
                float other_1 = _shfl_xor_1;
                a4[e_4] = a4[e_4] + other_1;
            }
            #pragma unroll
            for (int e_5 = 0; e_5 < 4; e_5++) {
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, a4[e_5], 4);
                float other_2 = _shfl_xor_2;
                a4[e_5] = a4[e_5] + other_2;
            }
            #pragma unroll
            for (int e_6 = 0; e_6 < 4; e_6++) {
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, a4[e_6], 2);
                float other_3 = _shfl_xor_3;
                a4[e_6] = a4[e_6] + other_3;
            }
            #pragma unroll
            for (int e_7 = 0; e_7 < 4; e_7++) {
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, a4[e_7], 1);
                float other_4 = _shfl_xor_4;
                a4[e_7] = a4[e_7] + other_4;
            }
            if (lane_1 == 0) {
                {
                    float4 _v4 = make_float4(a4[0 + 0], a4[0 + 1], a4[0 + 2], a4[0 + 3]);
                    *reinterpret_cast<float4*>(dw + (long long)ch * 4) = _v4;
                }
            }
        }
    }
}

} // extern "C"

