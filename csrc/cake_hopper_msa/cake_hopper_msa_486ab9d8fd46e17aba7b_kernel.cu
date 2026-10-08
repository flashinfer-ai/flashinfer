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
#define SMEM_QCOL_OFF 0
#define SMEM_QCOL_STAGE_BYTES 4
#define SMEM_QCOL_STRIDE 4
#define SMEM_FLAGW_OFF 4
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_PUB_OFF 20
#define SMEM_PUB_STAGE_BYTES 8704
#define SMEM_PUB_STRIDE 8704
#define SMEM_Q2_OFF 8724
#define SMEM_Q2_STAGE_BYTES 1024
#define SMEM_Q2_STRIDE 1024
#define SMEM_TOTAL 9856
#define THREADS 128

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

__global__ __launch_bounds__(128) void
kernel_cake_hopper_msa_486ab9d8fd46e17aba7b(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    unsigned int* qcol = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int qcol_addr = smem + 0;
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 4);
    const int flagw_addr = smem + 4;
    float* pub = reinterpret_cast<float*>(smem_raw + 20);
    const int pub_addr = smem + 20;
    float* q2 = reinterpret_cast<float*>(smem_raw + 8724);
    const int q2_addr = smem + 8724;

    // === Task calls (dependency order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1;
    int c = tid_1 - w;
    int whi = w / 32;
    int col = blockIdx.x + c;
    int head = blockIdx.y;
    if (tid_1 == 0) {
        flagw[0] = 0;
    }
    int lim = tiles;
    if (col >= total_q) {
        lim = 0;
    }
    if (col < total_q) {
        int nv = nvp[col];
        if (nv < lim) {
            lim = nv;
        }
        if (lim < 0) {
            lim = 0;
        }
    }
    long long nq64 = (long long)total_q;
    long long cbase = (long long)head * (long long)tiles * nq64 + (long long)col;
    float neg_inf = -CAKE_INF;
    float a[16];
    a[0] = neg_inf;
    a[1] = neg_inf;
    a[2] = neg_inf;
    a[3] = neg_inf;
    a[4] = neg_inf;
    a[5] = neg_inf;
    a[6] = neg_inf;
    a[7] = neg_inf;
    a[8] = neg_inf;
    a[9] = neg_inf;
    a[10] = neg_inf;
    a[11] = neg_inf;
    a[12] = neg_inf;
    a[13] = neg_inf;
    a[14] = neg_inf;
    a[15] = neg_inf;
    float rej = neg_inf;
    unsigned int cb[16];
    float kb[16];
    int t0 = w * 16;
    int t0_0 = t0;
    long long p = cbase + (long long)t0_0 * nq64;
    cb[0] = 4286578688;
    if (lim > t0_0) {
        cb[0] = S[p];
    }
    cb[1] = 4286578688;
    if (lim > t0_0 + 1) {
        cb[1] = S[p + nq64];
    }
    cb[2] = 4286578688;
    if (lim > t0_0 + 2) {
        cb[2] = S[p + 2 * nq64];
    }
    cb[3] = 4286578688;
    if (lim > t0_0 + 3) {
        cb[3] = S[p + 3 * nq64];
    }
    cb[4] = 4286578688;
    if (lim > t0_0 + 4) {
        cb[4] = S[p + 4 * nq64];
    }
    cb[5] = 4286578688;
    if (lim > t0_0 + 5) {
        cb[5] = S[p + 5 * nq64];
    }
    cb[6] = 4286578688;
    if (lim > t0_0 + 6) {
        cb[6] = S[p + 6 * nq64];
    }
    cb[7] = 4286578688;
    if (lim > t0_0 + 7) {
        cb[7] = S[p + 7 * nq64];
    }
    cb[8] = 4286578688;
    if (lim > t0_0 + 8) {
        cb[8] = S[p + 8 * nq64];
    }
    cb[9] = 4286578688;
    if (lim > t0_0 + 9) {
        cb[9] = S[p + 9 * nq64];
    }
    cb[10] = 4286578688;
    if (lim > t0_0 + 10) {
        cb[10] = S[p + 10 * nq64];
    }
    cb[11] = 4286578688;
    if (lim > t0_0 + 11) {
        cb[11] = S[p + 11 * nq64];
    }
    cb[12] = 4286578688;
    if (lim > t0_0 + 12) {
        cb[12] = S[p + 12 * nq64];
    }
    cb[13] = 4286578688;
    if (lim > t0_0 + 13) {
        cb[13] = S[p + 13 * nq64];
    }
    cb[14] = 4286578688;
    if (lim > t0_0 + 14) {
        cb[14] = S[p + 14 * nq64];
    }
    cb[15] = 4286578688;
    if (lim > t0_0 + 15) {
        cb[15] = S[p + 15 * nq64];
    }
    asm volatile("" ::: "memory");
    unsigned int nb[16];
    #pragma unroll 1
    for (int j = 0; j < num_chunks; j++) {
        int t0_1 = ((j + 1) * 128 + w) * 16;
        int t0_2 = t0_1;
        long long p_3 = cbase + (long long)t0_2 * nq64;
        nb[0] = 4286578688;
        if (lim > t0_2) {
            nb[0] = S[p_3];
        }
        nb[1] = 4286578688;
        if (lim > t0_2 + 1) {
            nb[1] = S[p_3 + nq64];
        }
        nb[2] = 4286578688;
        if (lim > t0_2 + 2) {
            nb[2] = S[p_3 + 2 * nq64];
        }
        nb[3] = 4286578688;
        if (lim > t0_2 + 3) {
            nb[3] = S[p_3 + 3 * nq64];
        }
        nb[4] = 4286578688;
        if (lim > t0_2 + 4) {
            nb[4] = S[p_3 + 4 * nq64];
        }
        nb[5] = 4286578688;
        if (lim > t0_2 + 5) {
            nb[5] = S[p_3 + 5 * nq64];
        }
        nb[6] = 4286578688;
        if (lim > t0_2 + 6) {
            nb[6] = S[p_3 + 6 * nq64];
        }
        nb[7] = 4286578688;
        if (lim > t0_2 + 7) {
            nb[7] = S[p_3 + 7 * nq64];
        }
        nb[8] = 4286578688;
        if (lim > t0_2 + 8) {
            nb[8] = S[p_3 + 8 * nq64];
        }
        nb[9] = 4286578688;
        if (lim > t0_2 + 9) {
            nb[9] = S[p_3 + 9 * nq64];
        }
        nb[10] = 4286578688;
        if (lim > t0_2 + 10) {
            nb[10] = S[p_3 + 10 * nq64];
        }
        nb[11] = 4286578688;
        if (lim > t0_2 + 11) {
            nb[11] = S[p_3 + 11 * nq64];
        }
        nb[12] = 4286578688;
        if (lim > t0_2 + 12) {
            nb[12] = S[p_3 + 12 * nq64];
        }
        nb[13] = 4286578688;
        if (lim > t0_2 + 13) {
            nb[13] = S[p_3 + 13 * nq64];
        }
        nb[14] = 4286578688;
        if (lim > t0_2 + 14) {
            nb[14] = S[p_3 + 14 * nq64];
        }
        nb[15] = 4286578688;
        if (lim > t0_2 + 15) {
            nb[15] = S[p_3 + 15 * nq64];
        }
        asm volatile("" ::: "memory");
        int t0_4 = (j * 128 + w) * 16;
        int t0_5 = t0_4;
        float sc = __uint_as_float(cb[0]);
        float _fmax_0 = fmaxf(sc, -1.7014118346046923e+38f);
        sc = _fmax_0;
        float _min_0 = fminf(sc, 1.7014118346046923e+38f);
        sc = _min_0;
        sc = sc;
        float sc_6 = sc;
        unsigned int key = __as_u32(sc_6) & 4294965248u | (unsigned int)t0_5;
        {
            int f = 0;
            if (t0_5 < fb || t0_5 >= lim - fe && lim > t0_5) {
                f = 1;
            }
            if (f != 0) {
                key = 2139092992 | (unsigned int)t0_5;
            }
        }
        kb[0] = __uint_as_float(key);
        float sc_7 = __uint_as_float(cb[1]);
        float _fmax_1 = fmaxf(sc_7, -1.7014118346046923e+38f);
        sc_7 = _fmax_1;
        float _min_1 = fminf(sc_7, 1.7014118346046923e+38f);
        sc_7 = _min_1;
        sc_7 = sc_7;
        float sc_8 = sc_7;
        unsigned int key_9 = __as_u32(sc_8) & 4294965248u | (unsigned int)(t0_5 + 1);
        {
            int f_1 = 0;
            if (t0_5 + 1 < fb || t0_5 + 1 >= lim - fe && lim > t0_5 + 1) {
                f_1 = 1;
            }
            if (f_1 != 0) {
                key_9 = 2139092992 | (unsigned int)(t0_5 + 1);
            }
        }
        kb[1] = __uint_as_float(key_9);
        float sc_10 = __uint_as_float(cb[2]);
        float _fmax_2 = fmaxf(sc_10, -1.7014118346046923e+38f);
        sc_10 = _fmax_2;
        float _min_2 = fminf(sc_10, 1.7014118346046923e+38f);
        sc_10 = _min_2;
        sc_10 = sc_10;
        float sc_11 = sc_10;
        unsigned int key_12 = __as_u32(sc_11) & 4294965248u | (unsigned int)(t0_5 + 2);
        {
            int f_2 = 0;
            if (t0_5 + 2 < fb || t0_5 + 2 >= lim - fe && lim > t0_5 + 2) {
                f_2 = 1;
            }
            if (f_2 != 0) {
                key_12 = 2139092992 | (unsigned int)(t0_5 + 2);
            }
        }
        kb[2] = __uint_as_float(key_12);
        float sc_13 = __uint_as_float(cb[3]);
        float _fmax_3 = fmaxf(sc_13, -1.7014118346046923e+38f);
        sc_13 = _fmax_3;
        float _min_3 = fminf(sc_13, 1.7014118346046923e+38f);
        sc_13 = _min_3;
        sc_13 = sc_13;
        float sc_14 = sc_13;
        unsigned int key_15 = __as_u32(sc_14) & 4294965248u | (unsigned int)(t0_5 + 3);
        {
            int f_3 = 0;
            if (t0_5 + 3 < fb || t0_5 + 3 >= lim - fe && lim > t0_5 + 3) {
                f_3 = 1;
            }
            if (f_3 != 0) {
                key_15 = 2139092992 | (unsigned int)(t0_5 + 3);
            }
        }
        kb[3] = __uint_as_float(key_15);
        float sc_16 = __uint_as_float(cb[4]);
        float _fmax_4 = fmaxf(sc_16, -1.7014118346046923e+38f);
        sc_16 = _fmax_4;
        float _min_4 = fminf(sc_16, 1.7014118346046923e+38f);
        sc_16 = _min_4;
        sc_16 = sc_16;
        float sc_17 = sc_16;
        unsigned int key_18 = __as_u32(sc_17) & 4294965248u | (unsigned int)(t0_5 + 4);
        {
            int f_4 = 0;
            if (t0_5 + 4 < fb || t0_5 + 4 >= lim - fe && lim > t0_5 + 4) {
                f_4 = 1;
            }
            if (f_4 != 0) {
                key_18 = 2139092992 | (unsigned int)(t0_5 + 4);
            }
        }
        kb[4] = __uint_as_float(key_18);
        float sc_19 = __uint_as_float(cb[5]);
        float _fmax_5 = fmaxf(sc_19, -1.7014118346046923e+38f);
        sc_19 = _fmax_5;
        float _min_5 = fminf(sc_19, 1.7014118346046923e+38f);
        sc_19 = _min_5;
        sc_19 = sc_19;
        float sc_20 = sc_19;
        unsigned int key_21 = __as_u32(sc_20) & 4294965248u | (unsigned int)(t0_5 + 5);
        {
            int f_5 = 0;
            if (t0_5 + 5 < fb || t0_5 + 5 >= lim - fe && lim > t0_5 + 5) {
                f_5 = 1;
            }
            if (f_5 != 0) {
                key_21 = 2139092992 | (unsigned int)(t0_5 + 5);
            }
        }
        kb[5] = __uint_as_float(key_21);
        float sc_22 = __uint_as_float(cb[6]);
        float _fmax_6 = fmaxf(sc_22, -1.7014118346046923e+38f);
        sc_22 = _fmax_6;
        float _min_6 = fminf(sc_22, 1.7014118346046923e+38f);
        sc_22 = _min_6;
        sc_22 = sc_22;
        float sc_23 = sc_22;
        unsigned int key_24 = __as_u32(sc_23) & 4294965248u | (unsigned int)(t0_5 + 6);
        {
            int f_6 = 0;
            if (t0_5 + 6 < fb || t0_5 + 6 >= lim - fe && lim > t0_5 + 6) {
                f_6 = 1;
            }
            if (f_6 != 0) {
                key_24 = 2139092992 | (unsigned int)(t0_5 + 6);
            }
        }
        kb[6] = __uint_as_float(key_24);
        float sc_25 = __uint_as_float(cb[7]);
        float _fmax_7 = fmaxf(sc_25, -1.7014118346046923e+38f);
        sc_25 = _fmax_7;
        float _min_7 = fminf(sc_25, 1.7014118346046923e+38f);
        sc_25 = _min_7;
        sc_25 = sc_25;
        float sc_26 = sc_25;
        unsigned int key_27 = __as_u32(sc_26) & 4294965248u | (unsigned int)(t0_5 + 7);
        {
            int f_7 = 0;
            if (t0_5 + 7 < fb || t0_5 + 7 >= lim - fe && lim > t0_5 + 7) {
                f_7 = 1;
            }
            if (f_7 != 0) {
                key_27 = 2139092992 | (unsigned int)(t0_5 + 7);
            }
        }
        kb[7] = __uint_as_float(key_27);
        float sc_28 = __uint_as_float(cb[8]);
        float _fmax_8 = fmaxf(sc_28, -1.7014118346046923e+38f);
        sc_28 = _fmax_8;
        float _min_8 = fminf(sc_28, 1.7014118346046923e+38f);
        sc_28 = _min_8;
        sc_28 = sc_28;
        float sc_29 = sc_28;
        unsigned int key_30 = __as_u32(sc_29) & 4294965248u | (unsigned int)(t0_5 + 8);
        {
            int f_8 = 0;
            if (t0_5 + 8 < fb || t0_5 + 8 >= lim - fe && lim > t0_5 + 8) {
                f_8 = 1;
            }
            if (f_8 != 0) {
                key_30 = 2139092992 | (unsigned int)(t0_5 + 8);
            }
        }
        kb[8] = __uint_as_float(key_30);
        float sc_31 = __uint_as_float(cb[9]);
        float _fmax_9 = fmaxf(sc_31, -1.7014118346046923e+38f);
        sc_31 = _fmax_9;
        float _min_9 = fminf(sc_31, 1.7014118346046923e+38f);
        sc_31 = _min_9;
        sc_31 = sc_31;
        float sc_32 = sc_31;
        unsigned int key_33 = __as_u32(sc_32) & 4294965248u | (unsigned int)(t0_5 + 9);
        {
            int f_9 = 0;
            if (t0_5 + 9 < fb || t0_5 + 9 >= lim - fe && lim > t0_5 + 9) {
                f_9 = 1;
            }
            if (f_9 != 0) {
                key_33 = 2139092992 | (unsigned int)(t0_5 + 9);
            }
        }
        kb[9] = __uint_as_float(key_33);
        float sc_34 = __uint_as_float(cb[10]);
        float _fmax_10 = fmaxf(sc_34, -1.7014118346046923e+38f);
        sc_34 = _fmax_10;
        float _min_10 = fminf(sc_34, 1.7014118346046923e+38f);
        sc_34 = _min_10;
        sc_34 = sc_34;
        float sc_35 = sc_34;
        unsigned int key_36 = __as_u32(sc_35) & 4294965248u | (unsigned int)(t0_5 + 10);
        {
            int f_10 = 0;
            if (t0_5 + 10 < fb || t0_5 + 10 >= lim - fe && lim > t0_5 + 10) {
                f_10 = 1;
            }
            if (f_10 != 0) {
                key_36 = 2139092992 | (unsigned int)(t0_5 + 10);
            }
        }
        kb[10] = __uint_as_float(key_36);
        float sc_37 = __uint_as_float(cb[11]);
        float _fmax_11 = fmaxf(sc_37, -1.7014118346046923e+38f);
        sc_37 = _fmax_11;
        float _min_11 = fminf(sc_37, 1.7014118346046923e+38f);
        sc_37 = _min_11;
        sc_37 = sc_37;
        float sc_38 = sc_37;
        unsigned int key_39 = __as_u32(sc_38) & 4294965248u | (unsigned int)(t0_5 + 11);
        {
            int f_11 = 0;
            if (t0_5 + 11 < fb || t0_5 + 11 >= lim - fe && lim > t0_5 + 11) {
                f_11 = 1;
            }
            if (f_11 != 0) {
                key_39 = 2139092992 | (unsigned int)(t0_5 + 11);
            }
        }
        kb[11] = __uint_as_float(key_39);
        float sc_40 = __uint_as_float(cb[12]);
        float _fmax_12 = fmaxf(sc_40, -1.7014118346046923e+38f);
        sc_40 = _fmax_12;
        float _min_12 = fminf(sc_40, 1.7014118346046923e+38f);
        sc_40 = _min_12;
        sc_40 = sc_40;
        float sc_41 = sc_40;
        unsigned int key_42 = __as_u32(sc_41) & 4294965248u | (unsigned int)(t0_5 + 12);
        {
            int f_12 = 0;
            if (t0_5 + 12 < fb || t0_5 + 12 >= lim - fe && lim > t0_5 + 12) {
                f_12 = 1;
            }
            if (f_12 != 0) {
                key_42 = 2139092992 | (unsigned int)(t0_5 + 12);
            }
        }
        kb[12] = __uint_as_float(key_42);
        float sc_43 = __uint_as_float(cb[13]);
        float _fmax_13 = fmaxf(sc_43, -1.7014118346046923e+38f);
        sc_43 = _fmax_13;
        float _min_13 = fminf(sc_43, 1.7014118346046923e+38f);
        sc_43 = _min_13;
        sc_43 = sc_43;
        float sc_44 = sc_43;
        unsigned int key_45 = __as_u32(sc_44) & 4294965248u | (unsigned int)(t0_5 + 13);
        {
            int f_13 = 0;
            if (t0_5 + 13 < fb || t0_5 + 13 >= lim - fe && lim > t0_5 + 13) {
                f_13 = 1;
            }
            if (f_13 != 0) {
                key_45 = 2139092992 | (unsigned int)(t0_5 + 13);
            }
        }
        kb[13] = __uint_as_float(key_45);
        float sc_46 = __uint_as_float(cb[14]);
        float _fmax_14 = fmaxf(sc_46, -1.7014118346046923e+38f);
        sc_46 = _fmax_14;
        float _min_14 = fminf(sc_46, 1.7014118346046923e+38f);
        sc_46 = _min_14;
        sc_46 = sc_46;
        float sc_47 = sc_46;
        unsigned int key_48 = __as_u32(sc_47) & 4294965248u | (unsigned int)(t0_5 + 14);
        {
            int f_14 = 0;
            if (t0_5 + 14 < fb || t0_5 + 14 >= lim - fe && lim > t0_5 + 14) {
                f_14 = 1;
            }
            if (f_14 != 0) {
                key_48 = 2139092992 | (unsigned int)(t0_5 + 14);
            }
        }
        kb[14] = __uint_as_float(key_48);
        float sc_49 = __uint_as_float(cb[15]);
        float _fmax_15 = fmaxf(sc_49, -1.7014118346046923e+38f);
        sc_49 = _fmax_15;
        float _min_15 = fminf(sc_49, 1.7014118346046923e+38f);
        sc_49 = _min_15;
        sc_49 = sc_49;
        float sc_50 = sc_49;
        unsigned int key_51 = __as_u32(sc_50) & 4294965248u | (unsigned int)(t0_5 + 15);
        {
            int f_15 = 0;
            if (t0_5 + 15 < fb || t0_5 + 15 >= lim - fe && lim > t0_5 + 15) {
                f_15 = 1;
            }
            if (f_15 != 0) {
                key_51 = 2139092992 | (unsigned int)(t0_5 + 15);
            }
        }
        kb[15] = __uint_as_float(key_51);
        float _fmax_16 = fmaxf(kb[0], kb[13]);
        float hi = _fmax_16;
        float _min_16 = fminf(kb[0], kb[13]);
        float lo = _min_16;
        kb[0] = hi;
        kb[13] = lo;
        float _fmax_17 = fmaxf(kb[1], kb[12]);
        float hi_52 = _fmax_17;
        float _min_17 = fminf(kb[1], kb[12]);
        float lo_53 = _min_17;
        kb[1] = hi_52;
        kb[12] = lo_53;
        float _fmax_18 = fmaxf(kb[2], kb[15]);
        float hi_54 = _fmax_18;
        float _min_18 = fminf(kb[2], kb[15]);
        float lo_55 = _min_18;
        kb[2] = hi_54;
        kb[15] = lo_55;
        float _fmax_19 = fmaxf(kb[3], kb[14]);
        float hi_56 = _fmax_19;
        float _min_19 = fminf(kb[3], kb[14]);
        float lo_57 = _min_19;
        kb[3] = hi_56;
        kb[14] = lo_57;
        float _fmax_20 = fmaxf(kb[4], kb[8]);
        float hi_58 = _fmax_20;
        float _min_20 = fminf(kb[4], kb[8]);
        float lo_59 = _min_20;
        kb[4] = hi_58;
        kb[8] = lo_59;
        float _fmax_21 = fmaxf(kb[5], kb[6]);
        float hi_60 = _fmax_21;
        float _min_21 = fminf(kb[5], kb[6]);
        float lo_61 = _min_21;
        kb[5] = hi_60;
        kb[6] = lo_61;
        float _fmax_22 = fmaxf(kb[7], kb[11]);
        float hi_62 = _fmax_22;
        float _min_22 = fminf(kb[7], kb[11]);
        float lo_63 = _min_22;
        kb[7] = hi_62;
        kb[11] = lo_63;
        float _fmax_23 = fmaxf(kb[9], kb[10]);
        float hi_64 = _fmax_23;
        float _min_23 = fminf(kb[9], kb[10]);
        float lo_65 = _min_23;
        kb[9] = hi_64;
        kb[10] = lo_65;
        float _fmax_24 = fmaxf(kb[0], kb[5]);
        float hi_66 = _fmax_24;
        float _min_24 = fminf(kb[0], kb[5]);
        float lo_67 = _min_24;
        kb[0] = hi_66;
        kb[5] = lo_67;
        float _fmax_25 = fmaxf(kb[1], kb[7]);
        float hi_68 = _fmax_25;
        float _min_25 = fminf(kb[1], kb[7]);
        float lo_69 = _min_25;
        kb[1] = hi_68;
        kb[7] = lo_69;
        float _fmax_26 = fmaxf(kb[2], kb[9]);
        float hi_70 = _fmax_26;
        float _min_26 = fminf(kb[2], kb[9]);
        float lo_71 = _min_26;
        kb[2] = hi_70;
        kb[9] = lo_71;
        float _fmax_27 = fmaxf(kb[3], kb[4]);
        float hi_72 = _fmax_27;
        float _min_27 = fminf(kb[3], kb[4]);
        float lo_73 = _min_27;
        kb[3] = hi_72;
        kb[4] = lo_73;
        float _fmax_28 = fmaxf(kb[6], kb[13]);
        float hi_74 = _fmax_28;
        float _min_28 = fminf(kb[6], kb[13]);
        float lo_75 = _min_28;
        kb[6] = hi_74;
        kb[13] = lo_75;
        float _fmax_29 = fmaxf(kb[8], kb[14]);
        float hi_76 = _fmax_29;
        float _min_29 = fminf(kb[8], kb[14]);
        float lo_77 = _min_29;
        kb[8] = hi_76;
        kb[14] = lo_77;
        float _fmax_30 = fmaxf(kb[10], kb[15]);
        float hi_78 = _fmax_30;
        float _min_30 = fminf(kb[10], kb[15]);
        float lo_79 = _min_30;
        kb[10] = hi_78;
        kb[15] = lo_79;
        float _fmax_31 = fmaxf(kb[11], kb[12]);
        float hi_80 = _fmax_31;
        float _min_31 = fminf(kb[11], kb[12]);
        float lo_81 = _min_31;
        kb[11] = hi_80;
        kb[12] = lo_81;
        float _fmax_32 = fmaxf(kb[0], kb[1]);
        float hi_82 = _fmax_32;
        float _min_32 = fminf(kb[0], kb[1]);
        float lo_83 = _min_32;
        kb[0] = hi_82;
        kb[1] = lo_83;
        float _fmax_33 = fmaxf(kb[2], kb[3]);
        float hi_84 = _fmax_33;
        float _min_33 = fminf(kb[2], kb[3]);
        float lo_85 = _min_33;
        kb[2] = hi_84;
        kb[3] = lo_85;
        float _fmax_34 = fmaxf(kb[4], kb[5]);
        float hi_86 = _fmax_34;
        float _min_34 = fminf(kb[4], kb[5]);
        float lo_87 = _min_34;
        kb[4] = hi_86;
        kb[5] = lo_87;
        float _fmax_35 = fmaxf(kb[6], kb[8]);
        float hi_88 = _fmax_35;
        float _min_35 = fminf(kb[6], kb[8]);
        float lo_89 = _min_35;
        kb[6] = hi_88;
        kb[8] = lo_89;
        float _fmax_36 = fmaxf(kb[7], kb[9]);
        float hi_90 = _fmax_36;
        float _min_36 = fminf(kb[7], kb[9]);
        float lo_91 = _min_36;
        kb[7] = hi_90;
        kb[9] = lo_91;
        float _fmax_37 = fmaxf(kb[10], kb[11]);
        float hi_92 = _fmax_37;
        float _min_37 = fminf(kb[10], kb[11]);
        float lo_93 = _min_37;
        kb[10] = hi_92;
        kb[11] = lo_93;
        float _fmax_38 = fmaxf(kb[12], kb[13]);
        float hi_94 = _fmax_38;
        float _min_38 = fminf(kb[12], kb[13]);
        float lo_95 = _min_38;
        kb[12] = hi_94;
        kb[13] = lo_95;
        float _fmax_39 = fmaxf(kb[14], kb[15]);
        float hi_96 = _fmax_39;
        float _min_39 = fminf(kb[14], kb[15]);
        float lo_97 = _min_39;
        kb[14] = hi_96;
        kb[15] = lo_97;
        float _fmax_40 = fmaxf(kb[0], kb[2]);
        float hi_98 = _fmax_40;
        float _min_40 = fminf(kb[0], kb[2]);
        float lo_99 = _min_40;
        kb[0] = hi_98;
        kb[2] = lo_99;
        float _fmax_41 = fmaxf(kb[1], kb[3]);
        float hi_100 = _fmax_41;
        float _min_41 = fminf(kb[1], kb[3]);
        float lo_101 = _min_41;
        kb[1] = hi_100;
        kb[3] = lo_101;
        float _fmax_42 = fmaxf(kb[4], kb[10]);
        float hi_102 = _fmax_42;
        float _min_42 = fminf(kb[4], kb[10]);
        float lo_103 = _min_42;
        kb[4] = hi_102;
        kb[10] = lo_103;
        float _fmax_43 = fmaxf(kb[5], kb[11]);
        float hi_104 = _fmax_43;
        float _min_43 = fminf(kb[5], kb[11]);
        float lo_105 = _min_43;
        kb[5] = hi_104;
        kb[11] = lo_105;
        float _fmax_44 = fmaxf(kb[6], kb[7]);
        float hi_106 = _fmax_44;
        float _min_44 = fminf(kb[6], kb[7]);
        float lo_107 = _min_44;
        kb[6] = hi_106;
        kb[7] = lo_107;
        float _fmax_45 = fmaxf(kb[8], kb[9]);
        float hi_108 = _fmax_45;
        float _min_45 = fminf(kb[8], kb[9]);
        float lo_109 = _min_45;
        kb[8] = hi_108;
        kb[9] = lo_109;
        float _fmax_46 = fmaxf(kb[12], kb[14]);
        float hi_110 = _fmax_46;
        float _min_46 = fminf(kb[12], kb[14]);
        float lo_111 = _min_46;
        kb[12] = hi_110;
        kb[14] = lo_111;
        float _fmax_47 = fmaxf(kb[13], kb[15]);
        float hi_112 = _fmax_47;
        float _min_47 = fminf(kb[13], kb[15]);
        float lo_113 = _min_47;
        kb[13] = hi_112;
        kb[15] = lo_113;
        float _fmax_48 = fmaxf(kb[1], kb[2]);
        float hi_114 = _fmax_48;
        float _min_48 = fminf(kb[1], kb[2]);
        float lo_115 = _min_48;
        kb[1] = hi_114;
        kb[2] = lo_115;
        float _fmax_49 = fmaxf(kb[3], kb[12]);
        float hi_116 = _fmax_49;
        float _min_49 = fminf(kb[3], kb[12]);
        float lo_117 = _min_49;
        kb[3] = hi_116;
        kb[12] = lo_117;
        float _fmax_50 = fmaxf(kb[4], kb[6]);
        float hi_118 = _fmax_50;
        float _min_50 = fminf(kb[4], kb[6]);
        float lo_119 = _min_50;
        kb[4] = hi_118;
        kb[6] = lo_119;
        float _fmax_51 = fmaxf(kb[5], kb[7]);
        float hi_120 = _fmax_51;
        float _min_51 = fminf(kb[5], kb[7]);
        float lo_121 = _min_51;
        kb[5] = hi_120;
        kb[7] = lo_121;
        float _fmax_52 = fmaxf(kb[8], kb[10]);
        float hi_122 = _fmax_52;
        float _min_52 = fminf(kb[8], kb[10]);
        float lo_123 = _min_52;
        kb[8] = hi_122;
        kb[10] = lo_123;
        float _fmax_53 = fmaxf(kb[9], kb[11]);
        float hi_124 = _fmax_53;
        float _min_53 = fminf(kb[9], kb[11]);
        float lo_125 = _min_53;
        kb[9] = hi_124;
        kb[11] = lo_125;
        float _fmax_54 = fmaxf(kb[13], kb[14]);
        float hi_126 = _fmax_54;
        float _min_54 = fminf(kb[13], kb[14]);
        float lo_127 = _min_54;
        kb[13] = hi_126;
        kb[14] = lo_127;
        float _fmax_55 = fmaxf(kb[1], kb[4]);
        float hi_128 = _fmax_55;
        float _min_55 = fminf(kb[1], kb[4]);
        float lo_129 = _min_55;
        kb[1] = hi_128;
        kb[4] = lo_129;
        float _fmax_56 = fmaxf(kb[2], kb[6]);
        float hi_130 = _fmax_56;
        float _min_56 = fminf(kb[2], kb[6]);
        float lo_131 = _min_56;
        kb[2] = hi_130;
        kb[6] = lo_131;
        float _fmax_57 = fmaxf(kb[5], kb[8]);
        float hi_132 = _fmax_57;
        float _min_57 = fminf(kb[5], kb[8]);
        float lo_133 = _min_57;
        kb[5] = hi_132;
        kb[8] = lo_133;
        float _fmax_58 = fmaxf(kb[7], kb[10]);
        float hi_134 = _fmax_58;
        float _min_58 = fminf(kb[7], kb[10]);
        float lo_135 = _min_58;
        kb[7] = hi_134;
        kb[10] = lo_135;
        float _fmax_59 = fmaxf(kb[9], kb[13]);
        float hi_136 = _fmax_59;
        float _min_59 = fminf(kb[9], kb[13]);
        float lo_137 = _min_59;
        kb[9] = hi_136;
        kb[13] = lo_137;
        float _fmax_60 = fmaxf(kb[11], kb[14]);
        float hi_138 = _fmax_60;
        float _min_60 = fminf(kb[11], kb[14]);
        float lo_139 = _min_60;
        kb[11] = hi_138;
        kb[14] = lo_139;
        float _fmax_61 = fmaxf(kb[2], kb[4]);
        float hi_140 = _fmax_61;
        float _min_61 = fminf(kb[2], kb[4]);
        float lo_141 = _min_61;
        kb[2] = hi_140;
        kb[4] = lo_141;
        float _fmax_62 = fmaxf(kb[3], kb[6]);
        float hi_142 = _fmax_62;
        float _min_62 = fminf(kb[3], kb[6]);
        float lo_143 = _min_62;
        kb[3] = hi_142;
        kb[6] = lo_143;
        float _fmax_63 = fmaxf(kb[9], kb[12]);
        float hi_144 = _fmax_63;
        float _min_63 = fminf(kb[9], kb[12]);
        float lo_145 = _min_63;
        kb[9] = hi_144;
        kb[12] = lo_145;
        float _fmax_64 = fmaxf(kb[11], kb[13]);
        float hi_146 = _fmax_64;
        float _min_64 = fminf(kb[11], kb[13]);
        float lo_147 = _min_64;
        kb[11] = hi_146;
        kb[13] = lo_147;
        float _fmax_65 = fmaxf(kb[3], kb[5]);
        float hi_148 = _fmax_65;
        float _min_65 = fminf(kb[3], kb[5]);
        float lo_149 = _min_65;
        kb[3] = hi_148;
        kb[5] = lo_149;
        float _fmax_66 = fmaxf(kb[6], kb[8]);
        float hi_150 = _fmax_66;
        float _min_66 = fminf(kb[6], kb[8]);
        float lo_151 = _min_66;
        kb[6] = hi_150;
        kb[8] = lo_151;
        float _fmax_67 = fmaxf(kb[7], kb[9]);
        float hi_152 = _fmax_67;
        float _min_67 = fminf(kb[7], kb[9]);
        float lo_153 = _min_67;
        kb[7] = hi_152;
        kb[9] = lo_153;
        float _fmax_68 = fmaxf(kb[10], kb[12]);
        float hi_154 = _fmax_68;
        float _min_68 = fminf(kb[10], kb[12]);
        float lo_155 = _min_68;
        kb[10] = hi_154;
        kb[12] = lo_155;
        float _fmax_69 = fmaxf(kb[3], kb[4]);
        float hi_156 = _fmax_69;
        float _min_69 = fminf(kb[3], kb[4]);
        float lo_157 = _min_69;
        kb[3] = hi_156;
        kb[4] = lo_157;
        float _fmax_70 = fmaxf(kb[5], kb[6]);
        float hi_158 = _fmax_70;
        float _min_70 = fminf(kb[5], kb[6]);
        float lo_159 = _min_70;
        kb[5] = hi_158;
        kb[6] = lo_159;
        float _fmax_71 = fmaxf(kb[7], kb[8]);
        float hi_160 = _fmax_71;
        float _min_71 = fminf(kb[7], kb[8]);
        float lo_161 = _min_71;
        kb[7] = hi_160;
        kb[8] = lo_161;
        float _fmax_72 = fmaxf(kb[9], kb[10]);
        float hi_162 = _fmax_72;
        float _min_72 = fminf(kb[9], kb[10]);
        float lo_163 = _min_72;
        kb[9] = hi_162;
        kb[10] = lo_163;
        float _fmax_73 = fmaxf(kb[11], kb[12]);
        float hi_164 = _fmax_73;
        float _min_73 = fminf(kb[11], kb[12]);
        float lo_165 = _min_73;
        kb[11] = hi_164;
        kb[12] = lo_165;
        float _fmax_74 = fmaxf(kb[6], kb[7]);
        float hi_166 = _fmax_74;
        float _min_74 = fminf(kb[6], kb[7]);
        float lo_167 = _min_74;
        kb[6] = hi_166;
        kb[7] = lo_167;
        float _fmax_75 = fmaxf(kb[8], kb[9]);
        float hi_168 = _fmax_75;
        float _min_75 = fminf(kb[8], kb[9]);
        float lo_169 = _min_75;
        kb[8] = hi_168;
        kb[9] = lo_169;
        float r = rej;
        float _fmax_76 = fmaxf(a[0], kb[15]);
        float hi_170 = _fmax_76;
        float _min_76 = fminf(a[0], kb[15]);
        float lo_171 = _min_76;
        a[0] = hi_170;
        float _fmax_77 = fmaxf(r, lo_171);
        r = _fmax_77;
        float _fmax_78 = fmaxf(a[1], kb[14]);
        float hi_172 = _fmax_78;
        float _min_77 = fminf(a[1], kb[14]);
        float lo_173 = _min_77;
        a[1] = hi_172;
        float _fmax_79 = fmaxf(r, lo_173);
        r = _fmax_79;
        float _fmax_80 = fmaxf(a[2], kb[13]);
        float hi_174 = _fmax_80;
        float _min_78 = fminf(a[2], kb[13]);
        float lo_175 = _min_78;
        a[2] = hi_174;
        float _fmax_81 = fmaxf(r, lo_175);
        r = _fmax_81;
        float _fmax_82 = fmaxf(a[3], kb[12]);
        float hi_176 = _fmax_82;
        float _min_79 = fminf(a[3], kb[12]);
        float lo_177 = _min_79;
        a[3] = hi_176;
        float _fmax_83 = fmaxf(r, lo_177);
        r = _fmax_83;
        float _fmax_84 = fmaxf(a[4], kb[11]);
        float hi_178 = _fmax_84;
        float _min_80 = fminf(a[4], kb[11]);
        float lo_179 = _min_80;
        a[4] = hi_178;
        float _fmax_85 = fmaxf(r, lo_179);
        r = _fmax_85;
        float _fmax_86 = fmaxf(a[5], kb[10]);
        float hi_180 = _fmax_86;
        float _min_81 = fminf(a[5], kb[10]);
        float lo_181 = _min_81;
        a[5] = hi_180;
        float _fmax_87 = fmaxf(r, lo_181);
        r = _fmax_87;
        float _fmax_88 = fmaxf(a[6], kb[9]);
        float hi_182 = _fmax_88;
        float _min_82 = fminf(a[6], kb[9]);
        float lo_183 = _min_82;
        a[6] = hi_182;
        float _fmax_89 = fmaxf(r, lo_183);
        r = _fmax_89;
        float _fmax_90 = fmaxf(a[7], kb[8]);
        float hi_184 = _fmax_90;
        float _min_83 = fminf(a[7], kb[8]);
        float lo_185 = _min_83;
        a[7] = hi_184;
        float _fmax_91 = fmaxf(r, lo_185);
        r = _fmax_91;
        float _fmax_92 = fmaxf(a[8], kb[7]);
        float hi_186 = _fmax_92;
        float _min_84 = fminf(a[8], kb[7]);
        float lo_187 = _min_84;
        a[8] = hi_186;
        float _fmax_93 = fmaxf(r, lo_187);
        r = _fmax_93;
        float _fmax_94 = fmaxf(a[9], kb[6]);
        float hi_188 = _fmax_94;
        float _min_85 = fminf(a[9], kb[6]);
        float lo_189 = _min_85;
        a[9] = hi_188;
        float _fmax_95 = fmaxf(r, lo_189);
        r = _fmax_95;
        float _fmax_96 = fmaxf(a[10], kb[5]);
        float hi_190 = _fmax_96;
        float _min_86 = fminf(a[10], kb[5]);
        float lo_191 = _min_86;
        a[10] = hi_190;
        float _fmax_97 = fmaxf(r, lo_191);
        r = _fmax_97;
        float _fmax_98 = fmaxf(a[11], kb[4]);
        float hi_192 = _fmax_98;
        float _min_87 = fminf(a[11], kb[4]);
        float lo_193 = _min_87;
        a[11] = hi_192;
        float _fmax_99 = fmaxf(r, lo_193);
        r = _fmax_99;
        float _fmax_100 = fmaxf(a[12], kb[3]);
        float hi_194 = _fmax_100;
        float _min_88 = fminf(a[12], kb[3]);
        float lo_195 = _min_88;
        a[12] = hi_194;
        float _fmax_101 = fmaxf(r, lo_195);
        r = _fmax_101;
        float _fmax_102 = fmaxf(a[13], kb[2]);
        float hi_196 = _fmax_102;
        float _min_89 = fminf(a[13], kb[2]);
        float lo_197 = _min_89;
        a[13] = hi_196;
        float _fmax_103 = fmaxf(r, lo_197);
        r = _fmax_103;
        float _fmax_104 = fmaxf(a[14], kb[1]);
        float hi_198 = _fmax_104;
        float _min_90 = fminf(a[14], kb[1]);
        float lo_199 = _min_90;
        a[14] = hi_198;
        float _fmax_105 = fmaxf(r, lo_199);
        r = _fmax_105;
        float _fmax_106 = fmaxf(a[15], kb[0]);
        float hi_200 = _fmax_106;
        float _min_91 = fminf(a[15], kb[0]);
        float lo_201 = _min_91;
        a[15] = hi_200;
        float _fmax_107 = fmaxf(r, lo_201);
        r = _fmax_107;
        float _fmax_108 = fmaxf(a[0], a[8]);
        float hi_202 = _fmax_108;
        float _min_92 = fminf(a[0], a[8]);
        float lo_203 = _min_92;
        a[0] = hi_202;
        a[8] = lo_203;
        float _fmax_109 = fmaxf(a[1], a[9]);
        float hi_204 = _fmax_109;
        float _min_93 = fminf(a[1], a[9]);
        float lo_205 = _min_93;
        a[1] = hi_204;
        a[9] = lo_205;
        float _fmax_110 = fmaxf(a[2], a[10]);
        float hi_206 = _fmax_110;
        float _min_94 = fminf(a[2], a[10]);
        float lo_207 = _min_94;
        a[2] = hi_206;
        a[10] = lo_207;
        float _fmax_111 = fmaxf(a[3], a[11]);
        float hi_208 = _fmax_111;
        float _min_95 = fminf(a[3], a[11]);
        float lo_209 = _min_95;
        a[3] = hi_208;
        a[11] = lo_209;
        float _fmax_112 = fmaxf(a[4], a[12]);
        float hi_210 = _fmax_112;
        float _min_96 = fminf(a[4], a[12]);
        float lo_211 = _min_96;
        a[4] = hi_210;
        a[12] = lo_211;
        float _fmax_113 = fmaxf(a[5], a[13]);
        float hi_212 = _fmax_113;
        float _min_97 = fminf(a[5], a[13]);
        float lo_213 = _min_97;
        a[5] = hi_212;
        a[13] = lo_213;
        float _fmax_114 = fmaxf(a[6], a[14]);
        float hi_214 = _fmax_114;
        float _min_98 = fminf(a[6], a[14]);
        float lo_215 = _min_98;
        a[6] = hi_214;
        a[14] = lo_215;
        float _fmax_115 = fmaxf(a[7], a[15]);
        float hi_216 = _fmax_115;
        float _min_99 = fminf(a[7], a[15]);
        float lo_217 = _min_99;
        a[7] = hi_216;
        a[15] = lo_217;
        float _fmax_116 = fmaxf(a[0], a[4]);
        float hi_218 = _fmax_116;
        float _min_100 = fminf(a[0], a[4]);
        float lo_219 = _min_100;
        a[0] = hi_218;
        a[4] = lo_219;
        float _fmax_117 = fmaxf(a[1], a[5]);
        float hi_220 = _fmax_117;
        float _min_101 = fminf(a[1], a[5]);
        float lo_221 = _min_101;
        a[1] = hi_220;
        a[5] = lo_221;
        float _fmax_118 = fmaxf(a[2], a[6]);
        float hi_222 = _fmax_118;
        float _min_102 = fminf(a[2], a[6]);
        float lo_223 = _min_102;
        a[2] = hi_222;
        a[6] = lo_223;
        float _fmax_119 = fmaxf(a[3], a[7]);
        float hi_224 = _fmax_119;
        float _min_103 = fminf(a[3], a[7]);
        float lo_225 = _min_103;
        a[3] = hi_224;
        a[7] = lo_225;
        float _fmax_120 = fmaxf(a[8], a[12]);
        float hi_226 = _fmax_120;
        float _min_104 = fminf(a[8], a[12]);
        float lo_227 = _min_104;
        a[8] = hi_226;
        a[12] = lo_227;
        float _fmax_121 = fmaxf(a[9], a[13]);
        float hi_228 = _fmax_121;
        float _min_105 = fminf(a[9], a[13]);
        float lo_229 = _min_105;
        a[9] = hi_228;
        a[13] = lo_229;
        float _fmax_122 = fmaxf(a[10], a[14]);
        float hi_230 = _fmax_122;
        float _min_106 = fminf(a[10], a[14]);
        float lo_231 = _min_106;
        a[10] = hi_230;
        a[14] = lo_231;
        float _fmax_123 = fmaxf(a[11], a[15]);
        float hi_232 = _fmax_123;
        float _min_107 = fminf(a[11], a[15]);
        float lo_233 = _min_107;
        a[11] = hi_232;
        a[15] = lo_233;
        float _fmax_124 = fmaxf(a[0], a[2]);
        float hi_234 = _fmax_124;
        float _min_108 = fminf(a[0], a[2]);
        float lo_235 = _min_108;
        a[0] = hi_234;
        a[2] = lo_235;
        float _fmax_125 = fmaxf(a[1], a[3]);
        float hi_236 = _fmax_125;
        float _min_109 = fminf(a[1], a[3]);
        float lo_237 = _min_109;
        a[1] = hi_236;
        a[3] = lo_237;
        float _fmax_126 = fmaxf(a[4], a[6]);
        float hi_238 = _fmax_126;
        float _min_110 = fminf(a[4], a[6]);
        float lo_239 = _min_110;
        a[4] = hi_238;
        a[6] = lo_239;
        float _fmax_127 = fmaxf(a[5], a[7]);
        float hi_240 = _fmax_127;
        float _min_111 = fminf(a[5], a[7]);
        float lo_241 = _min_111;
        a[5] = hi_240;
        a[7] = lo_241;
        float _fmax_128 = fmaxf(a[8], a[10]);
        float hi_242 = _fmax_128;
        float _min_112 = fminf(a[8], a[10]);
        float lo_243 = _min_112;
        a[8] = hi_242;
        a[10] = lo_243;
        float _fmax_129 = fmaxf(a[9], a[11]);
        float hi_244 = _fmax_129;
        float _min_113 = fminf(a[9], a[11]);
        float lo_245 = _min_113;
        a[9] = hi_244;
        a[11] = lo_245;
        float _fmax_130 = fmaxf(a[12], a[14]);
        float hi_246 = _fmax_130;
        float _min_114 = fminf(a[12], a[14]);
        float lo_247 = _min_114;
        a[12] = hi_246;
        a[14] = lo_247;
        float _fmax_131 = fmaxf(a[13], a[15]);
        float hi_248 = _fmax_131;
        float _min_115 = fminf(a[13], a[15]);
        float lo_249 = _min_115;
        a[13] = hi_248;
        a[15] = lo_249;
        float _fmax_132 = fmaxf(a[0], a[1]);
        float hi_250 = _fmax_132;
        float _min_116 = fminf(a[0], a[1]);
        float lo_251 = _min_116;
        a[0] = hi_250;
        a[1] = lo_251;
        float _fmax_133 = fmaxf(a[2], a[3]);
        float hi_252 = _fmax_133;
        float _min_117 = fminf(a[2], a[3]);
        float lo_253 = _min_117;
        a[2] = hi_252;
        a[3] = lo_253;
        float _fmax_134 = fmaxf(a[4], a[5]);
        float hi_254 = _fmax_134;
        float _min_118 = fminf(a[4], a[5]);
        float lo_255 = _min_118;
        a[4] = hi_254;
        a[5] = lo_255;
        float _fmax_135 = fmaxf(a[6], a[7]);
        float hi_256 = _fmax_135;
        float _min_119 = fminf(a[6], a[7]);
        float lo_257 = _min_119;
        a[6] = hi_256;
        a[7] = lo_257;
        float _fmax_136 = fmaxf(a[8], a[9]);
        float hi_258 = _fmax_136;
        float _min_120 = fminf(a[8], a[9]);
        float lo_259 = _min_120;
        a[8] = hi_258;
        a[9] = lo_259;
        float _fmax_137 = fmaxf(a[10], a[11]);
        float hi_260 = _fmax_137;
        float _min_121 = fminf(a[10], a[11]);
        float lo_261 = _min_121;
        a[10] = hi_260;
        a[11] = lo_261;
        float _fmax_138 = fmaxf(a[12], a[13]);
        float hi_262 = _fmax_138;
        float _min_122 = fminf(a[12], a[13]);
        float lo_263 = _min_122;
        a[12] = hi_262;
        a[13] = lo_263;
        float _fmax_139 = fmaxf(a[14], a[15]);
        float hi_264 = _fmax_139;
        float _min_123 = fminf(a[14], a[15]);
        float lo_265 = _min_123;
        a[14] = hi_264;
        a[15] = lo_265;
        rej = r;
        cb[0] = nb[0];
        cb[1] = nb[1];
        cb[2] = nb[2];
        cb[3] = nb[3];
        cb[4] = nb[4];
        cb[5] = nb[5];
        cb[6] = nb[6];
        cb[7] = nb[7];
        cb[8] = nb[8];
        cb[9] = nb[9];
        cb[10] = nb[10];
        cb[11] = nb[11];
        cb[12] = nb[12];
        cb[13] = nb[13];
        cb[14] = nb[14];
        cb[15] = nb[15];
    }
    int ln = tid_1 & 15;
    int g = tid_1 >> 4;
    int cg = g & 0;
    int sg = g;
    int lnr = 15 - ln;
    int up[4];
    up[0] = 0;
    if ((ln & 8) == 0) {
        up[0] = 1;
    }
    up[1] = 0;
    if ((ln & 4) == 0) {
        up[1] = 1;
    }
    up[2] = 0;
    if ((ln & 2) == 0) {
        up[2] = 1;
    }
    up[3] = 0;
    if ((ln & 1) == 0) {
        up[3] = 1;
    }
    float rr1[1];
    int pb = tid_1 * 17;
    pub[pb] = a[0];
    pub[pb + 1] = a[1];
    pub[pb + 2] = a[2];
    pub[pb + 3] = a[3];
    pub[pb + 4] = a[4];
    pub[pb + 5] = a[5];
    pub[pb + 6] = a[6];
    pub[pb + 7] = a[7];
    pub[pb + 8] = a[8];
    pub[pb + 9] = a[9];
    pub[pb + 10] = a[10];
    pub[pb + 11] = a[11];
    pub[pb + 12] = a[12];
    pub[pb + 13] = a[13];
    pub[pb + 14] = a[14];
    pub[pb + 15] = a[15];
    pub[pb + 16] = rej;
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    float r_1 = neg_inf;
    float V[8];
    int s0 = (sg * 16 + cg) * 17;
    int s1 = (sg * 16 + 8 + cg) * 17;
    float x0 = pub[s0 + ln];
    float y0 = pub[s1 + lnr];
    float _min_124 = fminf(x0, y0);
    float lo0 = _min_124;
    float _fmax_140 = fmaxf(r_1, lo0);
    r_1 = _fmax_140;
    float _fmax_141 = fmaxf(x0, y0);
    float hi0 = _fmax_141;
    float cur = hi0;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
    float pv = _shfl_xor_0;
    float _fmax_142 = fmaxf(cur, pv);
    float hi_1 = _fmax_142;
    float _min_125 = fminf(cur, pv);
    float lo_1 = _min_125;
    cur = ((up[0] != 0) ? hi_1 : lo_1);
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
    float pv_1 = _shfl_xor_1;
    float _fmax_143 = fmaxf(cur, pv_1);
    float hi_2 = _fmax_143;
    float _min_126 = fminf(cur, pv_1);
    float lo_3 = _min_126;
    cur = ((up[1] != 0) ? hi_2 : lo_3);
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
    float pv_4 = _shfl_xor_2;
    float _fmax_144 = fmaxf(cur, pv_4);
    float hi_5 = _fmax_144;
    float _min_127 = fminf(cur, pv_4);
    float lo_6 = _min_127;
    cur = ((up[2] != 0) ? hi_5 : lo_6);
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
    float pv_7 = _shfl_xor_3;
    float _fmax_145 = fmaxf(cur, pv_7);
    float hi_8 = _fmax_145;
    float _min_128 = fminf(cur, pv_7);
    float lo_9 = _min_128;
    cur = ((up[3] != 0) ? hi_8 : lo_9);
    V[0] = cur;
    int s0_10 = (sg * 16 + 1 + cg) * 17;
    int s1_11 = (sg * 16 + 1 + 8 + cg) * 17;
    float x0_12 = pub[s0_10 + ln];
    float y0_13 = pub[s1_11 + lnr];
    float _min_129 = fminf(x0_12, y0_13);
    float lo0_14 = _min_129;
    float _fmax_146 = fmaxf(r_1, lo0_14);
    r_1 = _fmax_146;
    float _fmax_147 = fmaxf(x0_12, y0_13);
    float hi0_15 = _fmax_147;
    float cur_16 = hi0_15;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 8);
    float pv_17 = _shfl_xor_4;
    float _fmax_148 = fmaxf(cur_16, pv_17);
    float hi_18 = _fmax_148;
    float _min_130 = fminf(cur_16, pv_17);
    float lo_19 = _min_130;
    cur_16 = ((up[0] != 0) ? hi_18 : lo_19);
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 4);
    float pv_20 = _shfl_xor_5;
    float _fmax_149 = fmaxf(cur_16, pv_20);
    float hi_21 = _fmax_149;
    float _min_131 = fminf(cur_16, pv_20);
    float lo_22 = _min_131;
    cur_16 = ((up[1] != 0) ? hi_21 : lo_22);
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 2);
    float pv_23 = _shfl_xor_6;
    float _fmax_150 = fmaxf(cur_16, pv_23);
    float hi_24 = _fmax_150;
    float _min_132 = fminf(cur_16, pv_23);
    float lo_25 = _min_132;
    cur_16 = ((up[2] != 0) ? hi_24 : lo_25);
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 1);
    float pv_26 = _shfl_xor_7;
    float _fmax_151 = fmaxf(cur_16, pv_26);
    float hi_27 = _fmax_151;
    float _min_133 = fminf(cur_16, pv_26);
    float lo_28 = _min_133;
    cur_16 = ((up[3] != 0) ? hi_27 : lo_28);
    V[1] = cur_16;
    int s0_29 = (sg * 16 + 2 + cg) * 17;
    int s1_30 = (sg * 16 + 2 + 8 + cg) * 17;
    float x0_31 = pub[s0_29 + ln];
    float y0_32 = pub[s1_30 + lnr];
    float _min_134 = fminf(x0_31, y0_32);
    float lo0_33 = _min_134;
    float _fmax_152 = fmaxf(r_1, lo0_33);
    r_1 = _fmax_152;
    float _fmax_153 = fmaxf(x0_31, y0_32);
    float hi0_34 = _fmax_153;
    float cur_35 = hi0_34;
    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, cur_35, 8);
    float pv_36 = _shfl_xor_8;
    float _fmax_154 = fmaxf(cur_35, pv_36);
    float hi_37 = _fmax_154;
    float _min_135 = fminf(cur_35, pv_36);
    float lo_38 = _min_135;
    cur_35 = ((up[0] != 0) ? hi_37 : lo_38);
    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, cur_35, 4);
    float pv_39 = _shfl_xor_9;
    float _fmax_155 = fmaxf(cur_35, pv_39);
    float hi_40 = _fmax_155;
    float _min_136 = fminf(cur_35, pv_39);
    float lo_41 = _min_136;
    cur_35 = ((up[1] != 0) ? hi_40 : lo_41);
    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, cur_35, 2);
    float pv_42 = _shfl_xor_10;
    float _fmax_156 = fmaxf(cur_35, pv_42);
    float hi_43 = _fmax_156;
    float _min_137 = fminf(cur_35, pv_42);
    float lo_44 = _min_137;
    cur_35 = ((up[2] != 0) ? hi_43 : lo_44);
    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, cur_35, 1);
    float pv_45 = _shfl_xor_11;
    float _fmax_157 = fmaxf(cur_35, pv_45);
    float hi_46 = _fmax_157;
    float _min_138 = fminf(cur_35, pv_45);
    float lo_47 = _min_138;
    cur_35 = ((up[3] != 0) ? hi_46 : lo_47);
    V[2] = cur_35;
    int s0_48 = (sg * 16 + 3 + cg) * 17;
    int s1_49 = (sg * 16 + 3 + 8 + cg) * 17;
    float x0_50 = pub[s0_48 + ln];
    float y0_51 = pub[s1_49 + lnr];
    float _min_139 = fminf(x0_50, y0_51);
    float lo0_52 = _min_139;
    float _fmax_158 = fmaxf(r_1, lo0_52);
    r_1 = _fmax_158;
    float _fmax_159 = fmaxf(x0_50, y0_51);
    float hi0_53 = _fmax_159;
    float cur_54 = hi0_53;
    float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, cur_54, 8);
    float pv_55 = _shfl_xor_12;
    float _fmax_160 = fmaxf(cur_54, pv_55);
    float hi_56_1 = _fmax_160;
    float _min_140 = fminf(cur_54, pv_55);
    float lo_57_1 = _min_140;
    cur_54 = ((up[0] != 0) ? hi_56_1 : lo_57_1);
    float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, cur_54, 4);
    float pv_58 = _shfl_xor_13;
    float _fmax_161 = fmaxf(cur_54, pv_58);
    float hi_59 = _fmax_161;
    float _min_141 = fminf(cur_54, pv_58);
    float lo_60 = _min_141;
    cur_54 = ((up[1] != 0) ? hi_59 : lo_60);
    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, cur_54, 2);
    float pv_61 = _shfl_xor_14;
    float _fmax_162 = fmaxf(cur_54, pv_61);
    float hi_62_1 = _fmax_162;
    float _min_142 = fminf(cur_54, pv_61);
    float lo_63_1 = _min_142;
    cur_54 = ((up[2] != 0) ? hi_62_1 : lo_63_1);
    float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, cur_54, 1);
    float pv_64 = _shfl_xor_15;
    float _fmax_163 = fmaxf(cur_54, pv_64);
    float hi_65 = _fmax_163;
    float _min_143 = fminf(cur_54, pv_64);
    float lo_66 = _min_143;
    cur_54 = ((up[3] != 0) ? hi_65 : lo_66);
    V[3] = cur_54;
    int s0_67 = (sg * 16 + 4 + cg) * 17;
    int s1_68 = (sg * 16 + 4 + 8 + cg) * 17;
    float x0_69 = pub[s0_67 + ln];
    float y0_70 = pub[s1_68 + lnr];
    float _min_144 = fminf(x0_69, y0_70);
    float lo0_71 = _min_144;
    float _fmax_164 = fmaxf(r_1, lo0_71);
    r_1 = _fmax_164;
    float _fmax_165 = fmaxf(x0_69, y0_70);
    float hi0_72 = _fmax_165;
    float cur_73 = hi0_72;
    float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, cur_73, 8);
    float pv_74 = _shfl_xor_16;
    float _fmax_166 = fmaxf(cur_73, pv_74);
    float hi_75 = _fmax_166;
    float _min_145 = fminf(cur_73, pv_74);
    float lo_76 = _min_145;
    cur_73 = ((up[0] != 0) ? hi_75 : lo_76);
    float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, cur_73, 4);
    float pv_77 = _shfl_xor_17;
    float _fmax_167 = fmaxf(cur_73, pv_77);
    float hi_78_1 = _fmax_167;
    float _min_146 = fminf(cur_73, pv_77);
    float lo_79_1 = _min_146;
    cur_73 = ((up[1] != 0) ? hi_78_1 : lo_79_1);
    float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, cur_73, 2);
    float pv_80 = _shfl_xor_18;
    float _fmax_168 = fmaxf(cur_73, pv_80);
    float hi_81 = _fmax_168;
    float _min_147 = fminf(cur_73, pv_80);
    float lo_82 = _min_147;
    cur_73 = ((up[2] != 0) ? hi_81 : lo_82);
    float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, cur_73, 1);
    float pv_83 = _shfl_xor_19;
    float _fmax_169 = fmaxf(cur_73, pv_83);
    float hi_84_1 = _fmax_169;
    float _min_148 = fminf(cur_73, pv_83);
    float lo_85_1 = _min_148;
    cur_73 = ((up[3] != 0) ? hi_84_1 : lo_85_1);
    V[4] = cur_73;
    int s0_86 = (sg * 16 + 5 + cg) * 17;
    int s1_87 = (sg * 16 + 5 + 8 + cg) * 17;
    float x0_88 = pub[s0_86 + ln];
    float y0_89 = pub[s1_87 + lnr];
    float _min_149 = fminf(x0_88, y0_89);
    float lo0_90 = _min_149;
    float _fmax_170 = fmaxf(r_1, lo0_90);
    r_1 = _fmax_170;
    float _fmax_171 = fmaxf(x0_88, y0_89);
    float hi0_91 = _fmax_171;
    float cur_92 = hi0_91;
    float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, cur_92, 8);
    float pv_93 = _shfl_xor_20;
    float _fmax_172 = fmaxf(cur_92, pv_93);
    float hi_94_1 = _fmax_172;
    float _min_150 = fminf(cur_92, pv_93);
    float lo_95_1 = _min_150;
    cur_92 = ((up[0] != 0) ? hi_94_1 : lo_95_1);
    float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, cur_92, 4);
    float pv_96 = _shfl_xor_21;
    float _fmax_173 = fmaxf(cur_92, pv_96);
    float hi_97 = _fmax_173;
    float _min_151 = fminf(cur_92, pv_96);
    float lo_98 = _min_151;
    cur_92 = ((up[1] != 0) ? hi_97 : lo_98);
    float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, cur_92, 2);
    float pv_99 = _shfl_xor_22;
    float _fmax_174 = fmaxf(cur_92, pv_99);
    float hi_100_1 = _fmax_174;
    float _min_152 = fminf(cur_92, pv_99);
    float lo_101_1 = _min_152;
    cur_92 = ((up[2] != 0) ? hi_100_1 : lo_101_1);
    float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, cur_92, 1);
    float pv_102 = _shfl_xor_23;
    float _fmax_175 = fmaxf(cur_92, pv_102);
    float hi_103 = _fmax_175;
    float _min_153 = fminf(cur_92, pv_102);
    float lo_104 = _min_153;
    cur_92 = ((up[3] != 0) ? hi_103 : lo_104);
    V[5] = cur_92;
    int s0_105 = (sg * 16 + 6 + cg) * 17;
    int s1_106 = (sg * 16 + 6 + 8 + cg) * 17;
    float x0_107 = pub[s0_105 + ln];
    float y0_108 = pub[s1_106 + lnr];
    float _min_154 = fminf(x0_107, y0_108);
    float lo0_109 = _min_154;
    float _fmax_176 = fmaxf(r_1, lo0_109);
    r_1 = _fmax_176;
    float _fmax_177 = fmaxf(x0_107, y0_108);
    float hi0_110 = _fmax_177;
    float cur_111 = hi0_110;
    float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, cur_111, 8);
    float pv_112 = _shfl_xor_24;
    float _fmax_178 = fmaxf(cur_111, pv_112);
    float hi_113 = _fmax_178;
    float _min_155 = fminf(cur_111, pv_112);
    float lo_114 = _min_155;
    cur_111 = ((up[0] != 0) ? hi_113 : lo_114);
    float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, cur_111, 4);
    float pv_115 = _shfl_xor_25;
    float _fmax_179 = fmaxf(cur_111, pv_115);
    float hi_116_1 = _fmax_179;
    float _min_156 = fminf(cur_111, pv_115);
    float lo_117_1 = _min_156;
    cur_111 = ((up[1] != 0) ? hi_116_1 : lo_117_1);
    float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, cur_111, 2);
    float pv_118 = _shfl_xor_26;
    float _fmax_180 = fmaxf(cur_111, pv_118);
    float hi_119 = _fmax_180;
    float _min_157 = fminf(cur_111, pv_118);
    float lo_120 = _min_157;
    cur_111 = ((up[2] != 0) ? hi_119 : lo_120);
    float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, cur_111, 1);
    float pv_121 = _shfl_xor_27;
    float _fmax_181 = fmaxf(cur_111, pv_121);
    float hi_122_1 = _fmax_181;
    float _min_158 = fminf(cur_111, pv_121);
    float lo_123_1 = _min_158;
    cur_111 = ((up[3] != 0) ? hi_122_1 : lo_123_1);
    V[6] = cur_111;
    int s0_124 = (sg * 16 + 7 + cg) * 17;
    int s1_125 = (sg * 16 + 7 + 8 + cg) * 17;
    float x0_126 = pub[s0_124 + ln];
    float y0_127 = pub[s1_125 + lnr];
    float _min_159 = fminf(x0_126, y0_127);
    float lo0_128 = _min_159;
    float _fmax_182 = fmaxf(r_1, lo0_128);
    r_1 = _fmax_182;
    float _fmax_183 = fmaxf(x0_126, y0_127);
    float hi0_129 = _fmax_183;
    float cur_130 = hi0_129;
    float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, cur_130, 8);
    float pv_131 = _shfl_xor_28;
    float _fmax_184 = fmaxf(cur_130, pv_131);
    float hi_132_1 = _fmax_184;
    float _min_160 = fminf(cur_130, pv_131);
    float lo_133_1 = _min_160;
    cur_130 = ((up[0] != 0) ? hi_132_1 : lo_133_1);
    float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, cur_130, 4);
    float pv_134 = _shfl_xor_29;
    float _fmax_185 = fmaxf(cur_130, pv_134);
    float hi_135 = _fmax_185;
    float _min_161 = fminf(cur_130, pv_134);
    float lo_136 = _min_161;
    cur_130 = ((up[1] != 0) ? hi_135 : lo_136);
    float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, cur_130, 2);
    float pv_137 = _shfl_xor_30;
    float _fmax_186 = fmaxf(cur_130, pv_137);
    float hi_138_1 = _fmax_186;
    float _min_162 = fminf(cur_130, pv_137);
    float lo_139_1 = _min_162;
    cur_130 = ((up[2] != 0) ? hi_138_1 : lo_139_1);
    float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, cur_130, 1);
    float pv_140 = _shfl_xor_31;
    float _fmax_187 = fmaxf(cur_130, pv_140);
    float hi_141 = _fmax_187;
    float _min_163 = fminf(cur_130, pv_140);
    float lo_142 = _min_163;
    cur_130 = ((up[3] != 0) ? hi_141 : lo_142);
    V[7] = cur_130;
    float rs = pub[(sg * 16 + ln + cg) * 17 + 16];
    float _fmax_188 = fmaxf(r_1, rs);
    r_1 = _fmax_188;
    float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
    float y1 = _shfl_xor_32;
    float _min_164 = fminf(V[0], y1);
    float lo1 = _min_164;
    float _fmax_189 = fmaxf(r_1, lo1);
    r_1 = _fmax_189;
    float _fmax_190 = fmaxf(V[0], y1);
    float hi1 = _fmax_190;
    float cur_143 = hi1;
    float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, cur_143, 8);
    float pv_144 = _shfl_xor_33;
    float _fmax_191 = fmaxf(cur_143, pv_144);
    float hi_145 = _fmax_191;
    float _min_165 = fminf(cur_143, pv_144);
    float lo_146 = _min_165;
    cur_143 = ((up[0] != 0) ? hi_145 : lo_146);
    float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, cur_143, 4);
    float pv_147 = _shfl_xor_34;
    float _fmax_192 = fmaxf(cur_143, pv_147);
    float hi_148_1 = _fmax_192;
    float _min_166 = fminf(cur_143, pv_147);
    float lo_149_1 = _min_166;
    cur_143 = ((up[1] != 0) ? hi_148_1 : lo_149_1);
    float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, cur_143, 2);
    float pv_150 = _shfl_xor_35;
    float _fmax_193 = fmaxf(cur_143, pv_150);
    float hi_151 = _fmax_193;
    float _min_167 = fminf(cur_143, pv_150);
    float lo_152 = _min_167;
    cur_143 = ((up[2] != 0) ? hi_151 : lo_152);
    float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, cur_143, 1);
    float pv_153 = _shfl_xor_36;
    float _fmax_194 = fmaxf(cur_143, pv_153);
    float hi_154_1 = _fmax_194;
    float _min_168 = fminf(cur_143, pv_153);
    float lo_155_1 = _min_168;
    cur_143 = ((up[3] != 0) ? hi_154_1 : lo_155_1);
    V[0] = cur_143;
    float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
    float y1_156 = _shfl_xor_37;
    float _min_169 = fminf(V[1], y1_156);
    float lo1_157 = _min_169;
    float _fmax_195 = fmaxf(r_1, lo1_157);
    r_1 = _fmax_195;
    float _fmax_196 = fmaxf(V[1], y1_156);
    float hi1_158 = _fmax_196;
    float cur_159 = hi1_158;
    float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, cur_159, 8);
    float pv_160 = _shfl_xor_38;
    float _fmax_197 = fmaxf(cur_159, pv_160);
    float hi_161 = _fmax_197;
    float _min_170 = fminf(cur_159, pv_160);
    float lo_162 = _min_170;
    cur_159 = ((up[0] != 0) ? hi_161 : lo_162);
    float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, cur_159, 4);
    float pv_163 = _shfl_xor_39;
    float _fmax_198 = fmaxf(cur_159, pv_163);
    float hi_164_1 = _fmax_198;
    float _min_171 = fminf(cur_159, pv_163);
    float lo_165_1 = _min_171;
    cur_159 = ((up[1] != 0) ? hi_164_1 : lo_165_1);
    float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, cur_159, 2);
    float pv_166 = _shfl_xor_40;
    float _fmax_199 = fmaxf(cur_159, pv_166);
    float hi_167 = _fmax_199;
    float _min_172 = fminf(cur_159, pv_166);
    float lo_168 = _min_172;
    cur_159 = ((up[2] != 0) ? hi_167 : lo_168);
    float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, cur_159, 1);
    float pv_169 = _shfl_xor_41;
    float _fmax_200 = fmaxf(cur_159, pv_169);
    float hi_170_1 = _fmax_200;
    float _min_173 = fminf(cur_159, pv_169);
    float lo_171_1 = _min_173;
    cur_159 = ((up[3] != 0) ? hi_170_1 : lo_171_1);
    V[1] = cur_159;
    float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
    float y1_172 = _shfl_xor_42;
    float _min_174 = fminf(V[2], y1_172);
    float lo1_173 = _min_174;
    float _fmax_201 = fmaxf(r_1, lo1_173);
    r_1 = _fmax_201;
    float _fmax_202 = fmaxf(V[2], y1_172);
    float hi1_174 = _fmax_202;
    float cur_175 = hi1_174;
    float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, cur_175, 8);
    float pv_176 = _shfl_xor_43;
    float _fmax_203 = fmaxf(cur_175, pv_176);
    float hi_177 = _fmax_203;
    float _min_175 = fminf(cur_175, pv_176);
    float lo_178 = _min_175;
    cur_175 = ((up[0] != 0) ? hi_177 : lo_178);
    float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, cur_175, 4);
    float pv_179 = _shfl_xor_44;
    float _fmax_204 = fmaxf(cur_175, pv_179);
    float hi_180_1 = _fmax_204;
    float _min_176 = fminf(cur_175, pv_179);
    float lo_181_1 = _min_176;
    cur_175 = ((up[1] != 0) ? hi_180_1 : lo_181_1);
    float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, cur_175, 2);
    float pv_182 = _shfl_xor_45;
    float _fmax_205 = fmaxf(cur_175, pv_182);
    float hi_183 = _fmax_205;
    float _min_177 = fminf(cur_175, pv_182);
    float lo_184 = _min_177;
    cur_175 = ((up[2] != 0) ? hi_183 : lo_184);
    float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, cur_175, 1);
    float pv_185 = _shfl_xor_46;
    float _fmax_206 = fmaxf(cur_175, pv_185);
    float hi_186_1 = _fmax_206;
    float _min_178 = fminf(cur_175, pv_185);
    float lo_187_1 = _min_178;
    cur_175 = ((up[3] != 0) ? hi_186_1 : lo_187_1);
    V[2] = cur_175;
    float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
    float y1_188 = _shfl_xor_47;
    float _min_179 = fminf(V[3], y1_188);
    float lo1_189 = _min_179;
    float _fmax_207 = fmaxf(r_1, lo1_189);
    r_1 = _fmax_207;
    float _fmax_208 = fmaxf(V[3], y1_188);
    float hi1_190 = _fmax_208;
    float cur_191 = hi1_190;
    float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, cur_191, 8);
    float pv_192 = _shfl_xor_48;
    float _fmax_209 = fmaxf(cur_191, pv_192);
    float hi_193 = _fmax_209;
    float _min_180 = fminf(cur_191, pv_192);
    float lo_194 = _min_180;
    cur_191 = ((up[0] != 0) ? hi_193 : lo_194);
    float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, cur_191, 4);
    float pv_195 = _shfl_xor_49;
    float _fmax_210 = fmaxf(cur_191, pv_195);
    float hi_196_1 = _fmax_210;
    float _min_181 = fminf(cur_191, pv_195);
    float lo_197_1 = _min_181;
    cur_191 = ((up[1] != 0) ? hi_196_1 : lo_197_1);
    float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, cur_191, 2);
    float pv_198 = _shfl_xor_50;
    float _fmax_211 = fmaxf(cur_191, pv_198);
    float hi_199 = _fmax_211;
    float _min_182 = fminf(cur_191, pv_198);
    float lo_200 = _min_182;
    cur_191 = ((up[2] != 0) ? hi_199 : lo_200);
    float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, cur_191, 1);
    float pv_201 = _shfl_xor_51;
    float _fmax_212 = fmaxf(cur_191, pv_201);
    float hi_202_1 = _fmax_212;
    float _min_183 = fminf(cur_191, pv_201);
    float lo_203_1 = _min_183;
    cur_191 = ((up[3] != 0) ? hi_202_1 : lo_203_1);
    V[3] = cur_191;
    float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
    float y1_204 = _shfl_xor_52;
    float _min_184 = fminf(V[0], y1_204);
    float lo1_205 = _min_184;
    float _fmax_213 = fmaxf(r_1, lo1_205);
    r_1 = _fmax_213;
    float _fmax_214 = fmaxf(V[0], y1_204);
    float hi1_206 = _fmax_214;
    float cur_207 = hi1_206;
    float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, cur_207, 8);
    float pv_208 = _shfl_xor_53;
    float _fmax_215 = fmaxf(cur_207, pv_208);
    float hi_209 = _fmax_215;
    float _min_185 = fminf(cur_207, pv_208);
    float lo_210 = _min_185;
    cur_207 = ((up[0] != 0) ? hi_209 : lo_210);
    float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, cur_207, 4);
    float pv_211 = _shfl_xor_54;
    float _fmax_216 = fmaxf(cur_207, pv_211);
    float hi_212_1 = _fmax_216;
    float _min_186 = fminf(cur_207, pv_211);
    float lo_213_1 = _min_186;
    cur_207 = ((up[1] != 0) ? hi_212_1 : lo_213_1);
    float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, cur_207, 2);
    float pv_214 = _shfl_xor_55;
    float _fmax_217 = fmaxf(cur_207, pv_214);
    float hi_215 = _fmax_217;
    float _min_187 = fminf(cur_207, pv_214);
    float lo_216 = _min_187;
    cur_207 = ((up[2] != 0) ? hi_215 : lo_216);
    float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, cur_207, 1);
    float pv_217 = _shfl_xor_56;
    float _fmax_218 = fmaxf(cur_207, pv_217);
    float hi_218_1 = _fmax_218;
    float _min_188 = fminf(cur_207, pv_217);
    float lo_219_1 = _min_188;
    cur_207 = ((up[3] != 0) ? hi_218_1 : lo_219_1);
    V[0] = cur_207;
    float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
    float y1_220 = _shfl_xor_57;
    float _min_189 = fminf(V[1], y1_220);
    float lo1_221 = _min_189;
    float _fmax_219 = fmaxf(r_1, lo1_221);
    r_1 = _fmax_219;
    float _fmax_220 = fmaxf(V[1], y1_220);
    float hi1_222 = _fmax_220;
    float cur_223 = hi1_222;
    float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 8);
    float pv_224 = _shfl_xor_58;
    float _fmax_221 = fmaxf(cur_223, pv_224);
    float hi_225 = _fmax_221;
    float _min_190 = fminf(cur_223, pv_224);
    float lo_226 = _min_190;
    cur_223 = ((up[0] != 0) ? hi_225 : lo_226);
    float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 4);
    float pv_227 = _shfl_xor_59;
    float _fmax_222 = fmaxf(cur_223, pv_227);
    float hi_228_1 = _fmax_222;
    float _min_191 = fminf(cur_223, pv_227);
    float lo_229_1 = _min_191;
    cur_223 = ((up[1] != 0) ? hi_228_1 : lo_229_1);
    float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 2);
    float pv_230 = _shfl_xor_60;
    float _fmax_223 = fmaxf(cur_223, pv_230);
    float hi_231 = _fmax_223;
    float _min_192 = fminf(cur_223, pv_230);
    float lo_232 = _min_192;
    cur_223 = ((up[2] != 0) ? hi_231 : lo_232);
    float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 1);
    float pv_233 = _shfl_xor_61;
    float _fmax_224 = fmaxf(cur_223, pv_233);
    float hi_234_1 = _fmax_224;
    float _min_193 = fminf(cur_223, pv_233);
    float lo_235_1 = _min_193;
    cur_223 = ((up[3] != 0) ? hi_234_1 : lo_235_1);
    V[1] = cur_223;
    float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
    float yl = _shfl_xor_62;
    float _min_194 = fminf(V[0], yl);
    float lol = _min_194;
    float _fmax_225 = fmaxf(r_1, lol);
    r_1 = _fmax_225;
    float _fmax_226 = fmaxf(V[0], yl);
    float hil = _fmax_226;
    float cur_236 = hil;
    float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, cur_236, 8);
    float pv_237 = _shfl_xor_63;
    float _fmax_227 = fmaxf(cur_236, pv_237);
    float hi_238_1 = _fmax_227;
    float _min_195 = fminf(cur_236, pv_237);
    float lo_239_1 = _min_195;
    cur_236 = ((up[0] != 0) ? hi_238_1 : lo_239_1);
    float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, cur_236, 4);
    float pv_240 = _shfl_xor_64;
    float _fmax_228 = fmaxf(cur_236, pv_240);
    float hi_241 = _fmax_228;
    float _min_196 = fminf(cur_236, pv_240);
    float lo_242 = _min_196;
    cur_236 = ((up[1] != 0) ? hi_241 : lo_242);
    float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, cur_236, 2);
    float pv_243 = _shfl_xor_65;
    float _fmax_229 = fmaxf(cur_236, pv_243);
    float hi_244_1 = _fmax_229;
    float _min_197 = fminf(cur_236, pv_243);
    float lo_245_1 = _min_197;
    cur_236 = ((up[2] != 0) ? hi_244_1 : lo_245_1);
    float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, cur_236, 1);
    float pv_246 = _shfl_xor_66;
    float _fmax_230 = fmaxf(cur_236, pv_246);
    float hi_247 = _fmax_230;
    float _min_198 = fminf(cur_236, pv_246);
    float lo_248 = _min_198;
    cur_236 = ((up[3] != 0) ? hi_247 : lo_248);
    V[0] = cur_236;
    float K = V[0];
    int qb = (sg + cg) * 32;
    q2[qb + ln] = K;
    q2[qb + 16 + ln] = r_1;
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    if (tid_1 < 32) {
        float r2 = neg_inf;
        float _fmax_231 = fmaxf(r2, q2[cg * 32 + 16 + ln]);
        r2 = _fmax_231;
        float _fmax_232 = fmaxf(r2, q2[(1 + cg) * 32 + 16 + ln]);
        r2 = _fmax_232;
        float _fmax_233 = fmaxf(r2, q2[(2 + cg) * 32 + 16 + ln]);
        r2 = _fmax_233;
        float _fmax_234 = fmaxf(r2, q2[(3 + cg) * 32 + 16 + ln]);
        r2 = _fmax_234;
        float _fmax_235 = fmaxf(r2, q2[(4 + cg) * 32 + 16 + ln]);
        r2 = _fmax_235;
        float _fmax_236 = fmaxf(r2, q2[(5 + cg) * 32 + 16 + ln]);
        r2 = _fmax_236;
        float _fmax_237 = fmaxf(r2, q2[(6 + cg) * 32 + 16 + ln]);
        r2 = _fmax_237;
        float _fmax_238 = fmaxf(r2, q2[(7 + cg) * 32 + 16 + ln]);
        r2 = _fmax_238;
        float V2[4];
        float x2 = q2[cg * 32 + ln];
        float y2 = q2[(4 + cg) * 32 + lnr];
        float _min_199 = fminf(x2, y2);
        float lo2 = _min_199;
        float _fmax_239 = fmaxf(r2, lo2);
        r2 = _fmax_239;
        float _fmax_240 = fmaxf(x2, y2);
        float hi2 = _fmax_240;
        float cur_0 = hi2;
        float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 8);
        float pv_2 = _shfl_xor_67;
        float _fmax_241 = fmaxf(cur_0, pv_2);
        float hi_3 = _fmax_241;
        float _min_200 = fminf(cur_0, pv_2);
        float lo_4 = _min_200;
        cur_0 = ((up[0] != 0) ? hi_3 : lo_4);
        float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 4);
        float pv_5 = _shfl_xor_68;
        float _fmax_242 = fmaxf(cur_0, pv_5);
        float hi_6 = _fmax_242;
        float _min_201 = fminf(cur_0, pv_5);
        float lo_7 = _min_201;
        cur_0 = ((up[1] != 0) ? hi_6 : lo_7);
        float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 2);
        float pv_8 = _shfl_xor_69;
        float _fmax_243 = fmaxf(cur_0, pv_8);
        float hi_9 = _fmax_243;
        float _min_202 = fminf(cur_0, pv_8);
        float lo_10 = _min_202;
        cur_0 = ((up[2] != 0) ? hi_9 : lo_10);
        float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 1);
        float pv_11 = _shfl_xor_70;
        float _fmax_244 = fmaxf(cur_0, pv_11);
        float hi_12 = _fmax_244;
        float _min_203 = fminf(cur_0, pv_11);
        float lo_13 = _min_203;
        cur_0 = ((up[3] != 0) ? hi_12 : lo_13);
        V2[0] = cur_0;
        float x2_14 = q2[(1 + cg) * 32 + ln];
        float y2_15 = q2[(5 + cg) * 32 + lnr];
        float _min_204 = fminf(x2_14, y2_15);
        float lo2_16 = _min_204;
        float _fmax_245 = fmaxf(r2, lo2_16);
        r2 = _fmax_245;
        float _fmax_246 = fmaxf(x2_14, y2_15);
        float hi2_17 = _fmax_246;
        float cur_18 = hi2_17;
        float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 8);
        float pv_19 = _shfl_xor_71;
        float _fmax_247 = fmaxf(cur_18, pv_19);
        float hi_20 = _fmax_247;
        float _min_205 = fminf(cur_18, pv_19);
        float lo_21 = _min_205;
        cur_18 = ((up[0] != 0) ? hi_20 : lo_21);
        float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 4);
        float pv_22 = _shfl_xor_72;
        float _fmax_248 = fmaxf(cur_18, pv_22);
        float hi_23 = _fmax_248;
        float _min_206 = fminf(cur_18, pv_22);
        float lo_24 = _min_206;
        cur_18 = ((up[1] != 0) ? hi_23 : lo_24);
        float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 2);
        float pv_25 = _shfl_xor_73;
        float _fmax_249 = fmaxf(cur_18, pv_25);
        float hi_26 = _fmax_249;
        float _min_207 = fminf(cur_18, pv_25);
        float lo_27 = _min_207;
        cur_18 = ((up[2] != 0) ? hi_26 : lo_27);
        float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 1);
        float pv_28 = _shfl_xor_74;
        float _fmax_250 = fmaxf(cur_18, pv_28);
        float hi_29 = _fmax_250;
        float _min_208 = fminf(cur_18, pv_28);
        float lo_30 = _min_208;
        cur_18 = ((up[3] != 0) ? hi_29 : lo_30);
        V2[1] = cur_18;
        float x2_31 = q2[(2 + cg) * 32 + ln];
        float y2_32 = q2[(6 + cg) * 32 + lnr];
        float _min_209 = fminf(x2_31, y2_32);
        float lo2_33 = _min_209;
        float _fmax_251 = fmaxf(r2, lo2_33);
        r2 = _fmax_251;
        float _fmax_252 = fmaxf(x2_31, y2_32);
        float hi2_34 = _fmax_252;
        float cur_36 = hi2_34;
        float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, cur_36, 8);
        float pv_37 = _shfl_xor_75;
        float _fmax_253 = fmaxf(cur_36, pv_37);
        float hi_38 = _fmax_253;
        float _min_210 = fminf(cur_36, pv_37);
        float lo_39 = _min_210;
        cur_36 = ((up[0] != 0) ? hi_38 : lo_39);
        float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, cur_36, 4);
        float pv_40 = _shfl_xor_76;
        float _fmax_254 = fmaxf(cur_36, pv_40);
        float hi_41 = _fmax_254;
        float _min_211 = fminf(cur_36, pv_40);
        float lo_42 = _min_211;
        cur_36 = ((up[1] != 0) ? hi_41 : lo_42);
        float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, cur_36, 2);
        float pv_43 = _shfl_xor_77;
        float _fmax_255 = fmaxf(cur_36, pv_43);
        float hi_44 = _fmax_255;
        float _min_212 = fminf(cur_36, pv_43);
        float lo_45 = _min_212;
        cur_36 = ((up[2] != 0) ? hi_44 : lo_45);
        float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, cur_36, 1);
        float pv_46 = _shfl_xor_78;
        float _fmax_256 = fmaxf(cur_36, pv_46);
        float hi_47 = _fmax_256;
        float _min_213 = fminf(cur_36, pv_46);
        float lo_48 = _min_213;
        cur_36 = ((up[3] != 0) ? hi_47 : lo_48);
        V2[2] = cur_36;
        float x2_49 = q2[(3 + cg) * 32 + ln];
        float y2_50 = q2[(7 + cg) * 32 + lnr];
        float _min_214 = fminf(x2_49, y2_50);
        float lo2_51 = _min_214;
        float _fmax_257 = fmaxf(r2, lo2_51);
        r2 = _fmax_257;
        float _fmax_258 = fmaxf(x2_49, y2_50);
        float hi2_52 = _fmax_258;
        float cur_53 = hi2_52;
        float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, cur_53, 8);
        float pv_54 = _shfl_xor_79;
        float _fmax_259 = fmaxf(cur_53, pv_54);
        float hi_55 = _fmax_259;
        float _min_215 = fminf(cur_53, pv_54);
        float lo_56 = _min_215;
        cur_53 = ((up[0] != 0) ? hi_55 : lo_56);
        float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, cur_53, 4);
        float pv_57 = _shfl_xor_80;
        float _fmax_260 = fmaxf(cur_53, pv_57);
        float hi_58_1 = _fmax_260;
        float _min_216 = fminf(cur_53, pv_57);
        float lo_59_1 = _min_216;
        cur_53 = ((up[1] != 0) ? hi_58_1 : lo_59_1);
        float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, cur_53, 2);
        float pv_60 = _shfl_xor_81;
        float _fmax_261 = fmaxf(cur_53, pv_60);
        float hi_61 = _fmax_261;
        float _min_217 = fminf(cur_53, pv_60);
        float lo_62 = _min_217;
        cur_53 = ((up[2] != 0) ? hi_61 : lo_62);
        float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, cur_53, 1);
        float pv_63 = _shfl_xor_82;
        float _fmax_262 = fmaxf(cur_53, pv_63);
        float hi_64_1 = _fmax_262;
        float _min_218 = fminf(cur_53, pv_63);
        float lo_65_1 = _min_218;
        cur_53 = ((up[3] != 0) ? hi_64_1 : lo_65_1);
        V2[3] = cur_53;
        float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 15);
        float y3 = _shfl_xor_83;
        float _min_219 = fminf(V2[0], y3);
        float lo3 = _min_219;
        float _fmax_263 = fmaxf(r2, lo3);
        r2 = _fmax_263;
        float _fmax_264 = fmaxf(V2[0], y3);
        float hi3 = _fmax_264;
        float cur_66 = hi3;
        float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 8);
        float pv_67 = _shfl_xor_84;
        float _fmax_265 = fmaxf(cur_66, pv_67);
        float hi_68_1 = _fmax_265;
        float _min_220 = fminf(cur_66, pv_67);
        float lo_69_1 = _min_220;
        cur_66 = ((up[0] != 0) ? hi_68_1 : lo_69_1);
        float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 4);
        float pv_70 = _shfl_xor_85;
        float _fmax_266 = fmaxf(cur_66, pv_70);
        float hi_71 = _fmax_266;
        float _min_221 = fminf(cur_66, pv_70);
        float lo_72 = _min_221;
        cur_66 = ((up[1] != 0) ? hi_71 : lo_72);
        float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 2);
        float pv_73 = _shfl_xor_86;
        float _fmax_267 = fmaxf(cur_66, pv_73);
        float hi_74_1 = _fmax_267;
        float _min_222 = fminf(cur_66, pv_73);
        float lo_75_1 = _min_222;
        cur_66 = ((up[2] != 0) ? hi_74_1 : lo_75_1);
        float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 1);
        float pv_76 = _shfl_xor_87;
        float _fmax_268 = fmaxf(cur_66, pv_76);
        float hi_77 = _fmax_268;
        float _min_223 = fminf(cur_66, pv_76);
        float lo_78 = _min_223;
        cur_66 = ((up[3] != 0) ? hi_77 : lo_78);
        V2[0] = cur_66;
        float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 15);
        float y3_79 = _shfl_xor_88;
        float _min_224 = fminf(V2[1], y3_79);
        float lo3_80 = _min_224;
        float _fmax_269 = fmaxf(r2, lo3_80);
        r2 = _fmax_269;
        float _fmax_270 = fmaxf(V2[1], y3_79);
        float hi3_81 = _fmax_270;
        float cur_82 = hi3_81;
        float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, cur_82, 8);
        float pv_84 = _shfl_xor_89;
        float _fmax_271 = fmaxf(cur_82, pv_84);
        float hi_85 = _fmax_271;
        float _min_225 = fminf(cur_82, pv_84);
        float lo_86 = _min_225;
        cur_82 = ((up[0] != 0) ? hi_85 : lo_86);
        float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, cur_82, 4);
        float pv_87 = _shfl_xor_90;
        float _fmax_272 = fmaxf(cur_82, pv_87);
        float hi_88_1 = _fmax_272;
        float _min_226 = fminf(cur_82, pv_87);
        float lo_89_1 = _min_226;
        cur_82 = ((up[1] != 0) ? hi_88_1 : lo_89_1);
        float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, cur_82, 2);
        float pv_90 = _shfl_xor_91;
        float _fmax_273 = fmaxf(cur_82, pv_90);
        float hi_91 = _fmax_273;
        float _min_227 = fminf(cur_82, pv_90);
        float lo_92 = _min_227;
        cur_82 = ((up[2] != 0) ? hi_91 : lo_92);
        float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, cur_82, 1);
        float pv_94 = _shfl_xor_92;
        float _fmax_274 = fmaxf(cur_82, pv_94);
        float hi_95 = _fmax_274;
        float _min_228 = fminf(cur_82, pv_94);
        float lo_96 = _min_228;
        cur_82 = ((up[3] != 0) ? hi_95 : lo_96);
        V2[1] = cur_82;
        float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 15);
        float yl2 = _shfl_xor_93;
        float _min_229 = fminf(V2[0], yl2);
        float lol2 = _min_229;
        float _fmax_275 = fmaxf(r2, lol2);
        r2 = _fmax_275;
        float _fmax_276 = fmaxf(V2[0], yl2);
        V2[0] = _fmax_276;
        K = V2[0];
        r_1 = r2;
    }
    rr1[0] = r_1;
    int ucol = blockIdx.x + cg;
    int commit = 0;
    if (g < 1 && ucol < total_q) {
        commit = 1;
    }
    if (tid_1 < 32) {
        float u16 = K;
        float cr = rr1[0];
        float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, u16, 8);
        float _min_230 = fminf(u16, _shfl_xor_94);
        u16 = _min_230;
        float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, cr, 8);
        float _fmax_277 = fmaxf(cr, _shfl_xor_95);
        cr = _fmax_277;
        float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, u16, 4);
        float _min_231 = fminf(u16, _shfl_xor_96);
        u16 = _min_231;
        float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, cr, 4);
        float _fmax_278 = fmaxf(cr, _shfl_xor_97);
        cr = _fmax_278;
        float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, u16, 2);
        float _min_232 = fminf(u16, _shfl_xor_98);
        u16 = _min_232;
        float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, cr, 2);
        float _fmax_279 = fmaxf(cr, _shfl_xor_99);
        cr = _fmax_279;
        float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, u16, 1);
        float _min_233 = fminf(u16, _shfl_xor_100);
        u16 = _min_233;
        float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, cr, 1);
        float _fmax_280 = fmaxf(cr, _shfl_xor_101);
        cr = _fmax_280;
        unsigned int u16b = __as_u32(u16);
        unsigned int c16 = u16b & 4294965248u;
        unsigned int crb = __as_u32(cr) & 4294965248u;
        unsigned int q = 4294967295;
        if (c16 == crb && u16b < 4278190080u && c16 != 2139092992 && commit != 0) {
            q = c16;
            if (ln == 0) {
                flagw[0] = 1;
            }
        }
        if (ln == 0 && g < 1) {
            qcol[cg] = q;
        }
    }
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int need2 = flagw[0];
    unsigned int qc = qcol[c];
    int flagged = 0;
    if (qc != 4294967295u) {
        flagged = 1;
    }
    unsigned int qg = qcol[cg];
    int gflag = 0;
    if (qg != 4294967295u) {
        gflag = 1;
    }
    float K2 = 0.0f;
    if (need2 != 0) {
        int lim2 = 0;
        if (flagged != 0) {
            lim2 = lim;
        }
        float a2[16];
        a2[0] = 0.0f;
        a2[1] = 0.0f;
        a2[2] = 0.0f;
        a2[3] = 0.0f;
        a2[4] = 0.0f;
        a2[5] = 0.0f;
        a2[6] = 0.0f;
        a2[7] = 0.0f;
        a2[8] = 0.0f;
        a2[9] = 0.0f;
        a2[10] = 0.0f;
        a2[11] = 0.0f;
        a2[12] = 0.0f;
        a2[13] = 0.0f;
        a2[14] = 0.0f;
        a2[15] = 0.0f;
        int _vote_0 = __any_sync(0xFFFFFFFF, flagged != 0);
        if (_vote_0 != 0) {
            unsigned int cb2[16];
            unsigned int nb2[16];
            float kb2[16];
            int t0_1_1 = w * 16;
            int t0_2_1 = t0_1_1;
            long long p_3_1 = cbase + (long long)t0_2_1 * nq64;
            cb2[0] = 4286578688;
            if (lim2 > t0_2_1) {
                cb2[0] = S[p_3_1];
            }
            cb2[1] = 4286578688;
            if (lim2 > t0_2_1 + 1) {
                cb2[1] = S[p_3_1 + nq64];
            }
            cb2[2] = 4286578688;
            if (lim2 > t0_2_1 + 2) {
                cb2[2] = S[p_3_1 + 2 * nq64];
            }
            cb2[3] = 4286578688;
            if (lim2 > t0_2_1 + 3) {
                cb2[3] = S[p_3_1 + 3 * nq64];
            }
            cb2[4] = 4286578688;
            if (lim2 > t0_2_1 + 4) {
                cb2[4] = S[p_3_1 + 4 * nq64];
            }
            cb2[5] = 4286578688;
            if (lim2 > t0_2_1 + 5) {
                cb2[5] = S[p_3_1 + 5 * nq64];
            }
            cb2[6] = 4286578688;
            if (lim2 > t0_2_1 + 6) {
                cb2[6] = S[p_3_1 + 6 * nq64];
            }
            cb2[7] = 4286578688;
            if (lim2 > t0_2_1 + 7) {
                cb2[7] = S[p_3_1 + 7 * nq64];
            }
            cb2[8] = 4286578688;
            if (lim2 > t0_2_1 + 8) {
                cb2[8] = S[p_3_1 + 8 * nq64];
            }
            cb2[9] = 4286578688;
            if (lim2 > t0_2_1 + 9) {
                cb2[9] = S[p_3_1 + 9 * nq64];
            }
            cb2[10] = 4286578688;
            if (lim2 > t0_2_1 + 10) {
                cb2[10] = S[p_3_1 + 10 * nq64];
            }
            cb2[11] = 4286578688;
            if (lim2 > t0_2_1 + 11) {
                cb2[11] = S[p_3_1 + 11 * nq64];
            }
            cb2[12] = 4286578688;
            if (lim2 > t0_2_1 + 12) {
                cb2[12] = S[p_3_1 + 12 * nq64];
            }
            cb2[13] = 4286578688;
            if (lim2 > t0_2_1 + 13) {
                cb2[13] = S[p_3_1 + 13 * nq64];
            }
            cb2[14] = 4286578688;
            if (lim2 > t0_2_1 + 14) {
                cb2[14] = S[p_3_1 + 14 * nq64];
            }
            cb2[15] = 4286578688;
            if (lim2 > t0_2_1 + 15) {
                cb2[15] = S[p_3_1 + 15 * nq64];
            }
            asm volatile("" ::: "memory");
            #pragma unroll 1
            for (int j_1 = 0; j_1 < num_chunks; j_1++) {
                int t0_3 = ((j_1 + 1) * 128 + w) * 16;
                int t0_4_1 = t0_3;
                long long p_5 = cbase + (long long)t0_4_1 * nq64;
                nb2[0] = 4286578688;
                if (lim2 > t0_4_1) {
                    nb2[0] = S[p_5];
                }
                nb2[1] = 4286578688;
                if (lim2 > t0_4_1 + 1) {
                    nb2[1] = S[p_5 + nq64];
                }
                nb2[2] = 4286578688;
                if (lim2 > t0_4_1 + 2) {
                    nb2[2] = S[p_5 + 2 * nq64];
                }
                nb2[3] = 4286578688;
                if (lim2 > t0_4_1 + 3) {
                    nb2[3] = S[p_5 + 3 * nq64];
                }
                nb2[4] = 4286578688;
                if (lim2 > t0_4_1 + 4) {
                    nb2[4] = S[p_5 + 4 * nq64];
                }
                nb2[5] = 4286578688;
                if (lim2 > t0_4_1 + 5) {
                    nb2[5] = S[p_5 + 5 * nq64];
                }
                nb2[6] = 4286578688;
                if (lim2 > t0_4_1 + 6) {
                    nb2[6] = S[p_5 + 6 * nq64];
                }
                nb2[7] = 4286578688;
                if (lim2 > t0_4_1 + 7) {
                    nb2[7] = S[p_5 + 7 * nq64];
                }
                nb2[8] = 4286578688;
                if (lim2 > t0_4_1 + 8) {
                    nb2[8] = S[p_5 + 8 * nq64];
                }
                nb2[9] = 4286578688;
                if (lim2 > t0_4_1 + 9) {
                    nb2[9] = S[p_5 + 9 * nq64];
                }
                nb2[10] = 4286578688;
                if (lim2 > t0_4_1 + 10) {
                    nb2[10] = S[p_5 + 10 * nq64];
                }
                nb2[11] = 4286578688;
                if (lim2 > t0_4_1 + 11) {
                    nb2[11] = S[p_5 + 11 * nq64];
                }
                nb2[12] = 4286578688;
                if (lim2 > t0_4_1 + 12) {
                    nb2[12] = S[p_5 + 12 * nq64];
                }
                nb2[13] = 4286578688;
                if (lim2 > t0_4_1 + 13) {
                    nb2[13] = S[p_5 + 13 * nq64];
                }
                nb2[14] = 4286578688;
                if (lim2 > t0_4_1 + 14) {
                    nb2[14] = S[p_5 + 14 * nq64];
                }
                nb2[15] = 4286578688;
                if (lim2 > t0_4_1 + 15) {
                    nb2[15] = S[p_5 + 15 * nq64];
                }
                asm volatile("" ::: "memory");
                int t0_6 = (j_1 * 128 + w) * 16;
                int t0_7 = t0_6;
                float qf = __uint_as_float(qc);
                float sc_1 = __uint_as_float(cb2[0]);
                float _fmax_281 = fmaxf(sc_1, -1.7014118346046923e+38f);
                sc_1 = _fmax_281;
                float _min_234 = fminf(sc_1, 1.7014118346046923e+38f);
                sc_1 = _min_234;
                sc_1 = sc_1;
                float sc_8_1 = sc_1;
                unsigned int u = __as_u32(sc_8_1);
                unsigned int cls = u & 4294965248u;
                int f_16 = 0;
                if (t0_7 < fb || t0_7 >= lim - fe && lim > t0_7) {
                    f_16 = 1;
                }
                if (f_16 != 0) {
                    cls = 2139092992;
                }
                unsigned int key_1 = 0;
                if (qf < __uint_as_float(cls) && cls < 4278190080u) {
                    key_1 = 1073741824 | (unsigned int)t0_7;
                }
                if (cls == qc) {
                    unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 2047) & 2047;
                    key_1 = 536870912 | lowb << 11 | (unsigned int)t0_7;
                }
                kb2[0] = __uint_as_float(key_1);
                float sc_9 = __uint_as_float(cb2[1]);
                float _fmax_282 = fmaxf(sc_9, -1.7014118346046923e+38f);
                sc_9 = _fmax_282;
                float _min_235 = fminf(sc_9, 1.7014118346046923e+38f);
                sc_9 = _min_235;
                sc_9 = sc_9;
                float sc_10_1 = sc_9;
                unsigned int u_11 = __as_u32(sc_10_1);
                unsigned int cls_12 = u_11 & 4294965248u;
                int f_13_1 = 0;
                if (t0_7 + 1 < fb || t0_7 + 1 >= lim - fe && lim > t0_7 + 1) {
                    f_13_1 = 1;
                }
                if (f_13_1 != 0) {
                    cls_12 = 2139092992;
                }
                unsigned int key_14 = 0;
                if (qf < __uint_as_float(cls_12) && cls_12 < 4278190080u) {
                    key_14 = 1073741824 | (unsigned int)(t0_7 + 1);
                }
                if (cls_12 == qc) {
                    unsigned int lowb_1 = (u_11 ^ (unsigned int)((int)u_11 >> 31) & 2047) & 2047;
                    key_14 = 536870912 | lowb_1 << 11 | (unsigned int)(t0_7 + 1);
                }
                kb2[1] = __uint_as_float(key_14);
                float sc_15 = __uint_as_float(cb2[2]);
                float _fmax_283 = fmaxf(sc_15, -1.7014118346046923e+38f);
                sc_15 = _fmax_283;
                float _min_236 = fminf(sc_15, 1.7014118346046923e+38f);
                sc_15 = _min_236;
                sc_15 = sc_15;
                float sc_16_1 = sc_15;
                unsigned int u_17 = __as_u32(sc_16_1);
                unsigned int cls_18 = u_17 & 4294965248u;
                int f_19 = 0;
                if (t0_7 + 2 < fb || t0_7 + 2 >= lim - fe && lim > t0_7 + 2) {
                    f_19 = 1;
                }
                if (f_19 != 0) {
                    cls_18 = 2139092992;
                }
                unsigned int key_20 = 0;
                if (qf < __uint_as_float(cls_18) && cls_18 < 4278190080u) {
                    key_20 = 1073741824 | (unsigned int)(t0_7 + 2);
                }
                if (cls_18 == qc) {
                    unsigned int lowb_2 = (u_17 ^ (unsigned int)((int)u_17 >> 31) & 2047) & 2047;
                    key_20 = 536870912 | lowb_2 << 11 | (unsigned int)(t0_7 + 2);
                }
                kb2[2] = __uint_as_float(key_20);
                float sc_21 = __uint_as_float(cb2[3]);
                float _fmax_284 = fmaxf(sc_21, -1.7014118346046923e+38f);
                sc_21 = _fmax_284;
                float _min_237 = fminf(sc_21, 1.7014118346046923e+38f);
                sc_21 = _min_237;
                sc_21 = sc_21;
                float sc_22_1 = sc_21;
                unsigned int u_23 = __as_u32(sc_22_1);
                unsigned int cls_24 = u_23 & 4294965248u;
                int f_25 = 0;
                if (t0_7 + 3 < fb || t0_7 + 3 >= lim - fe && lim > t0_7 + 3) {
                    f_25 = 1;
                }
                if (f_25 != 0) {
                    cls_24 = 2139092992;
                }
                unsigned int key_26 = 0;
                if (qf < __uint_as_float(cls_24) && cls_24 < 4278190080u) {
                    key_26 = 1073741824 | (unsigned int)(t0_7 + 3);
                }
                if (cls_24 == qc) {
                    unsigned int lowb_3 = (u_23 ^ (unsigned int)((int)u_23 >> 31) & 2047) & 2047;
                    key_26 = 536870912 | lowb_3 << 11 | (unsigned int)(t0_7 + 3);
                }
                kb2[3] = __uint_as_float(key_26);
                float sc_27 = __uint_as_float(cb2[4]);
                float _fmax_285 = fmaxf(sc_27, -1.7014118346046923e+38f);
                sc_27 = _fmax_285;
                float _min_238 = fminf(sc_27, 1.7014118346046923e+38f);
                sc_27 = _min_238;
                sc_27 = sc_27;
                float sc_28_1 = sc_27;
                unsigned int u_29 = __as_u32(sc_28_1);
                unsigned int cls_30 = u_29 & 4294965248u;
                int f_31 = 0;
                if (t0_7 + 4 < fb || t0_7 + 4 >= lim - fe && lim > t0_7 + 4) {
                    f_31 = 1;
                }
                if (f_31 != 0) {
                    cls_30 = 2139092992;
                }
                unsigned int key_32 = 0;
                if (qf < __uint_as_float(cls_30) && cls_30 < 4278190080u) {
                    key_32 = 1073741824 | (unsigned int)(t0_7 + 4);
                }
                if (cls_30 == qc) {
                    unsigned int lowb_4 = (u_29 ^ (unsigned int)((int)u_29 >> 31) & 2047) & 2047;
                    key_32 = 536870912 | lowb_4 << 11 | (unsigned int)(t0_7 + 4);
                }
                kb2[4] = __uint_as_float(key_32);
                float sc_33 = __uint_as_float(cb2[5]);
                float _fmax_286 = fmaxf(sc_33, -1.7014118346046923e+38f);
                sc_33 = _fmax_286;
                float _min_239 = fminf(sc_33, 1.7014118346046923e+38f);
                sc_33 = _min_239;
                sc_33 = sc_33;
                float sc_34_1 = sc_33;
                unsigned int u_35 = __as_u32(sc_34_1);
                unsigned int cls_36 = u_35 & 4294965248u;
                int f_37 = 0;
                if (t0_7 + 5 < fb || t0_7 + 5 >= lim - fe && lim > t0_7 + 5) {
                    f_37 = 1;
                }
                if (f_37 != 0) {
                    cls_36 = 2139092992;
                }
                unsigned int key_38 = 0;
                if (qf < __uint_as_float(cls_36) && cls_36 < 4278190080u) {
                    key_38 = 1073741824 | (unsigned int)(t0_7 + 5);
                }
                if (cls_36 == qc) {
                    unsigned int lowb_5 = (u_35 ^ (unsigned int)((int)u_35 >> 31) & 2047) & 2047;
                    key_38 = 536870912 | lowb_5 << 11 | (unsigned int)(t0_7 + 5);
                }
                kb2[5] = __uint_as_float(key_38);
                float sc_39 = __uint_as_float(cb2[6]);
                float _fmax_287 = fmaxf(sc_39, -1.7014118346046923e+38f);
                sc_39 = _fmax_287;
                float _min_240 = fminf(sc_39, 1.7014118346046923e+38f);
                sc_39 = _min_240;
                sc_39 = sc_39;
                float sc_40_1 = sc_39;
                unsigned int u_41 = __as_u32(sc_40_1);
                unsigned int cls_42 = u_41 & 4294965248u;
                int f_43 = 0;
                if (t0_7 + 6 < fb || t0_7 + 6 >= lim - fe && lim > t0_7 + 6) {
                    f_43 = 1;
                }
                if (f_43 != 0) {
                    cls_42 = 2139092992;
                }
                unsigned int key_44 = 0;
                if (qf < __uint_as_float(cls_42) && cls_42 < 4278190080u) {
                    key_44 = 1073741824 | (unsigned int)(t0_7 + 6);
                }
                if (cls_42 == qc) {
                    unsigned int lowb_6 = (u_41 ^ (unsigned int)((int)u_41 >> 31) & 2047) & 2047;
                    key_44 = 536870912 | lowb_6 << 11 | (unsigned int)(t0_7 + 6);
                }
                kb2[6] = __uint_as_float(key_44);
                float sc_45 = __uint_as_float(cb2[7]);
                float _fmax_288 = fmaxf(sc_45, -1.7014118346046923e+38f);
                sc_45 = _fmax_288;
                float _min_241 = fminf(sc_45, 1.7014118346046923e+38f);
                sc_45 = _min_241;
                sc_45 = sc_45;
                float sc_46_1 = sc_45;
                unsigned int u_47 = __as_u32(sc_46_1);
                unsigned int cls_48 = u_47 & 4294965248u;
                int f_49 = 0;
                if (t0_7 + 7 < fb || t0_7 + 7 >= lim - fe && lim > t0_7 + 7) {
                    f_49 = 1;
                }
                if (f_49 != 0) {
                    cls_48 = 2139092992;
                }
                unsigned int key_50 = 0;
                if (qf < __uint_as_float(cls_48) && cls_48 < 4278190080u) {
                    key_50 = 1073741824 | (unsigned int)(t0_7 + 7);
                }
                if (cls_48 == qc) {
                    unsigned int lowb_7 = (u_47 ^ (unsigned int)((int)u_47 >> 31) & 2047) & 2047;
                    key_50 = 536870912 | lowb_7 << 11 | (unsigned int)(t0_7 + 7);
                }
                kb2[7] = __uint_as_float(key_50);
                float sc_51 = __uint_as_float(cb2[8]);
                float _fmax_289 = fmaxf(sc_51, -1.7014118346046923e+38f);
                sc_51 = _fmax_289;
                float _min_242 = fminf(sc_51, 1.7014118346046923e+38f);
                sc_51 = _min_242;
                sc_51 = sc_51;
                float sc_52 = sc_51;
                unsigned int u_53 = __as_u32(sc_52);
                unsigned int cls_54 = u_53 & 4294965248u;
                int f_55 = 0;
                if (t0_7 + 8 < fb || t0_7 + 8 >= lim - fe && lim > t0_7 + 8) {
                    f_55 = 1;
                }
                if (f_55 != 0) {
                    cls_54 = 2139092992;
                }
                unsigned int key_56 = 0;
                if (qf < __uint_as_float(cls_54) && cls_54 < 4278190080u) {
                    key_56 = 1073741824 | (unsigned int)(t0_7 + 8);
                }
                if (cls_54 == qc) {
                    unsigned int lowb_8 = (u_53 ^ (unsigned int)((int)u_53 >> 31) & 2047) & 2047;
                    key_56 = 536870912 | lowb_8 << 11 | (unsigned int)(t0_7 + 8);
                }
                kb2[8] = __uint_as_float(key_56);
                float sc_57 = __uint_as_float(cb2[9]);
                float _fmax_290 = fmaxf(sc_57, -1.7014118346046923e+38f);
                sc_57 = _fmax_290;
                float _min_243 = fminf(sc_57, 1.7014118346046923e+38f);
                sc_57 = _min_243;
                sc_57 = sc_57;
                float sc_58 = sc_57;
                unsigned int u_59 = __as_u32(sc_58);
                unsigned int cls_60 = u_59 & 4294965248u;
                int f_61 = 0;
                if (t0_7 + 9 < fb || t0_7 + 9 >= lim - fe && lim > t0_7 + 9) {
                    f_61 = 1;
                }
                if (f_61 != 0) {
                    cls_60 = 2139092992;
                }
                unsigned int key_62 = 0;
                if (qf < __uint_as_float(cls_60) && cls_60 < 4278190080u) {
                    key_62 = 1073741824 | (unsigned int)(t0_7 + 9);
                }
                if (cls_60 == qc) {
                    unsigned int lowb_9 = (u_59 ^ (unsigned int)((int)u_59 >> 31) & 2047) & 2047;
                    key_62 = 536870912 | lowb_9 << 11 | (unsigned int)(t0_7 + 9);
                }
                kb2[9] = __uint_as_float(key_62);
                float sc_63 = __uint_as_float(cb2[10]);
                float _fmax_291 = fmaxf(sc_63, -1.7014118346046923e+38f);
                sc_63 = _fmax_291;
                float _min_244 = fminf(sc_63, 1.7014118346046923e+38f);
                sc_63 = _min_244;
                sc_63 = sc_63;
                float sc_64 = sc_63;
                unsigned int u_65 = __as_u32(sc_64);
                unsigned int cls_66 = u_65 & 4294965248u;
                int f_67 = 0;
                if (t0_7 + 10 < fb || t0_7 + 10 >= lim - fe && lim > t0_7 + 10) {
                    f_67 = 1;
                }
                if (f_67 != 0) {
                    cls_66 = 2139092992;
                }
                unsigned int key_68 = 0;
                if (qf < __uint_as_float(cls_66) && cls_66 < 4278190080u) {
                    key_68 = 1073741824 | (unsigned int)(t0_7 + 10);
                }
                if (cls_66 == qc) {
                    unsigned int lowb_10 = (u_65 ^ (unsigned int)((int)u_65 >> 31) & 2047) & 2047;
                    key_68 = 536870912 | lowb_10 << 11 | (unsigned int)(t0_7 + 10);
                }
                kb2[10] = __uint_as_float(key_68);
                float sc_69 = __uint_as_float(cb2[11]);
                float _fmax_292 = fmaxf(sc_69, -1.7014118346046923e+38f);
                sc_69 = _fmax_292;
                float _min_245 = fminf(sc_69, 1.7014118346046923e+38f);
                sc_69 = _min_245;
                sc_69 = sc_69;
                float sc_70 = sc_69;
                unsigned int u_71 = __as_u32(sc_70);
                unsigned int cls_72 = u_71 & 4294965248u;
                int f_73 = 0;
                if (t0_7 + 11 < fb || t0_7 + 11 >= lim - fe && lim > t0_7 + 11) {
                    f_73 = 1;
                }
                if (f_73 != 0) {
                    cls_72 = 2139092992;
                }
                unsigned int key_74 = 0;
                if (qf < __uint_as_float(cls_72) && cls_72 < 4278190080u) {
                    key_74 = 1073741824 | (unsigned int)(t0_7 + 11);
                }
                if (cls_72 == qc) {
                    unsigned int lowb_11 = (u_71 ^ (unsigned int)((int)u_71 >> 31) & 2047) & 2047;
                    key_74 = 536870912 | lowb_11 << 11 | (unsigned int)(t0_7 + 11);
                }
                kb2[11] = __uint_as_float(key_74);
                float sc_75 = __uint_as_float(cb2[12]);
                float _fmax_293 = fmaxf(sc_75, -1.7014118346046923e+38f);
                sc_75 = _fmax_293;
                float _min_246 = fminf(sc_75, 1.7014118346046923e+38f);
                sc_75 = _min_246;
                sc_75 = sc_75;
                float sc_76 = sc_75;
                unsigned int u_77 = __as_u32(sc_76);
                unsigned int cls_78 = u_77 & 4294965248u;
                int f_79 = 0;
                if (t0_7 + 12 < fb || t0_7 + 12 >= lim - fe && lim > t0_7 + 12) {
                    f_79 = 1;
                }
                if (f_79 != 0) {
                    cls_78 = 2139092992;
                }
                unsigned int key_80 = 0;
                if (qf < __uint_as_float(cls_78) && cls_78 < 4278190080u) {
                    key_80 = 1073741824 | (unsigned int)(t0_7 + 12);
                }
                if (cls_78 == qc) {
                    unsigned int lowb_12 = (u_77 ^ (unsigned int)((int)u_77 >> 31) & 2047) & 2047;
                    key_80 = 536870912 | lowb_12 << 11 | (unsigned int)(t0_7 + 12);
                }
                kb2[12] = __uint_as_float(key_80);
                float sc_81 = __uint_as_float(cb2[13]);
                float _fmax_294 = fmaxf(sc_81, -1.7014118346046923e+38f);
                sc_81 = _fmax_294;
                float _min_247 = fminf(sc_81, 1.7014118346046923e+38f);
                sc_81 = _min_247;
                sc_81 = sc_81;
                float sc_82 = sc_81;
                unsigned int u_83 = __as_u32(sc_82);
                unsigned int cls_84 = u_83 & 4294965248u;
                int f_85 = 0;
                if (t0_7 + 13 < fb || t0_7 + 13 >= lim - fe && lim > t0_7 + 13) {
                    f_85 = 1;
                }
                if (f_85 != 0) {
                    cls_84 = 2139092992;
                }
                unsigned int key_86 = 0;
                if (qf < __uint_as_float(cls_84) && cls_84 < 4278190080u) {
                    key_86 = 1073741824 | (unsigned int)(t0_7 + 13);
                }
                if (cls_84 == qc) {
                    unsigned int lowb_13 = (u_83 ^ (unsigned int)((int)u_83 >> 31) & 2047) & 2047;
                    key_86 = 536870912 | lowb_13 << 11 | (unsigned int)(t0_7 + 13);
                }
                kb2[13] = __uint_as_float(key_86);
                float sc_87 = __uint_as_float(cb2[14]);
                float _fmax_295 = fmaxf(sc_87, -1.7014118346046923e+38f);
                sc_87 = _fmax_295;
                float _min_248 = fminf(sc_87, 1.7014118346046923e+38f);
                sc_87 = _min_248;
                sc_87 = sc_87;
                float sc_88 = sc_87;
                unsigned int u_89 = __as_u32(sc_88);
                unsigned int cls_90 = u_89 & 4294965248u;
                int f_91 = 0;
                if (t0_7 + 14 < fb || t0_7 + 14 >= lim - fe && lim > t0_7 + 14) {
                    f_91 = 1;
                }
                if (f_91 != 0) {
                    cls_90 = 2139092992;
                }
                unsigned int key_92 = 0;
                if (qf < __uint_as_float(cls_90) && cls_90 < 4278190080u) {
                    key_92 = 1073741824 | (unsigned int)(t0_7 + 14);
                }
                if (cls_90 == qc) {
                    unsigned int lowb_14 = (u_89 ^ (unsigned int)((int)u_89 >> 31) & 2047) & 2047;
                    key_92 = 536870912 | lowb_14 << 11 | (unsigned int)(t0_7 + 14);
                }
                kb2[14] = __uint_as_float(key_92);
                float sc_93 = __uint_as_float(cb2[15]);
                float _fmax_296 = fmaxf(sc_93, -1.7014118346046923e+38f);
                sc_93 = _fmax_296;
                float _min_249 = fminf(sc_93, 1.7014118346046923e+38f);
                sc_93 = _min_249;
                sc_93 = sc_93;
                float sc_94 = sc_93;
                unsigned int u_95 = __as_u32(sc_94);
                unsigned int cls_96 = u_95 & 4294965248u;
                int f_97 = 0;
                if (t0_7 + 15 < fb || t0_7 + 15 >= lim - fe && lim > t0_7 + 15) {
                    f_97 = 1;
                }
                if (f_97 != 0) {
                    cls_96 = 2139092992;
                }
                unsigned int key_98 = 0;
                if (qf < __uint_as_float(cls_96) && cls_96 < 4278190080u) {
                    key_98 = 1073741824 | (unsigned int)(t0_7 + 15);
                }
                if (cls_96 == qc) {
                    unsigned int lowb_15 = (u_95 ^ (unsigned int)((int)u_95 >> 31) & 2047) & 2047;
                    key_98 = 536870912 | lowb_15 << 11 | (unsigned int)(t0_7 + 15);
                }
                kb2[15] = __uint_as_float(key_98);
                unsigned int nzc = 0;
                nzc = nzc | __as_u32(kb2[0]);
                nzc = nzc | __as_u32(kb2[1]);
                nzc = nzc | __as_u32(kb2[2]);
                nzc = nzc | __as_u32(kb2[3]);
                nzc = nzc | __as_u32(kb2[4]);
                nzc = nzc | __as_u32(kb2[5]);
                nzc = nzc | __as_u32(kb2[6]);
                nzc = nzc | __as_u32(kb2[7]);
                nzc = nzc | __as_u32(kb2[8]);
                nzc = nzc | __as_u32(kb2[9]);
                nzc = nzc | __as_u32(kb2[10]);
                nzc = nzc | __as_u32(kb2[11]);
                nzc = nzc | __as_u32(kb2[12]);
                nzc = nzc | __as_u32(kb2[13]);
                nzc = nzc | __as_u32(kb2[14]);
                nzc = nzc | __as_u32(kb2[15]);
                int _vote_1 = __any_sync(0xFFFFFFFF, nzc != 0);
                if (_vote_1 != 0) {
                    float _fmax_297 = fmaxf(kb2[0], kb2[13]);
                    float hi_0 = _fmax_297;
                    float _min_250 = fminf(kb2[0], kb2[13]);
                    float lo_1_1 = _min_250;
                    kb2[0] = hi_0;
                    kb2[13] = lo_1_1;
                    float _fmax_298 = fmaxf(kb2[1], kb2[12]);
                    float hi_3_1 = _fmax_298;
                    float _min_251 = fminf(kb2[1], kb2[12]);
                    float lo_4_1 = _min_251;
                    kb2[1] = hi_3_1;
                    kb2[12] = lo_4_1;
                    float _fmax_299 = fmaxf(kb2[2], kb2[15]);
                    float hi_6_1 = _fmax_299;
                    float _min_252 = fminf(kb2[2], kb2[15]);
                    float lo_7_1 = _min_252;
                    kb2[2] = hi_6_1;
                    kb2[15] = lo_7_1;
                    float _fmax_300 = fmaxf(kb2[3], kb2[14]);
                    float hi_9_1 = _fmax_300;
                    float _min_253 = fminf(kb2[3], kb2[14]);
                    float lo_10_1 = _min_253;
                    kb2[3] = hi_9_1;
                    kb2[14] = lo_10_1;
                    float _fmax_301 = fmaxf(kb2[4], kb2[8]);
                    float hi_11 = _fmax_301;
                    float _min_254 = fminf(kb2[4], kb2[8]);
                    float lo_12 = _min_254;
                    kb2[4] = hi_11;
                    kb2[8] = lo_12;
                    float _fmax_302 = fmaxf(kb2[5], kb2[6]);
                    float hi_13 = _fmax_302;
                    float _min_255 = fminf(kb2[5], kb2[6]);
                    float lo_14 = _min_255;
                    kb2[5] = hi_13;
                    kb2[6] = lo_14;
                    float _fmax_303 = fmaxf(kb2[7], kb2[11]);
                    float hi_15 = _fmax_303;
                    float _min_256 = fminf(kb2[7], kb2[11]);
                    float lo_16 = _min_256;
                    kb2[7] = hi_15;
                    kb2[11] = lo_16;
                    float _fmax_304 = fmaxf(kb2[9], kb2[10]);
                    float hi_17 = _fmax_304;
                    float _min_257 = fminf(kb2[9], kb2[10]);
                    float lo_18 = _min_257;
                    kb2[9] = hi_17;
                    kb2[10] = lo_18;
                    float _fmax_305 = fmaxf(kb2[0], kb2[5]);
                    float hi_19 = _fmax_305;
                    float _min_258 = fminf(kb2[0], kb2[5]);
                    float lo_20 = _min_258;
                    kb2[0] = hi_19;
                    kb2[5] = lo_20;
                    float _fmax_306 = fmaxf(kb2[1], kb2[7]);
                    float hi_22 = _fmax_306;
                    float _min_259 = fminf(kb2[1], kb2[7]);
                    float lo_23 = _min_259;
                    kb2[1] = hi_22;
                    kb2[7] = lo_23;
                    float _fmax_307 = fmaxf(kb2[2], kb2[9]);
                    float hi_25 = _fmax_307;
                    float _min_260 = fminf(kb2[2], kb2[9]);
                    float lo_26 = _min_260;
                    kb2[2] = hi_25;
                    kb2[9] = lo_26;
                    float _fmax_308 = fmaxf(kb2[3], kb2[4]);
                    float hi_28 = _fmax_308;
                    float _min_261 = fminf(kb2[3], kb2[4]);
                    float lo_29 = _min_261;
                    kb2[3] = hi_28;
                    kb2[4] = lo_29;
                    float _fmax_309 = fmaxf(kb2[6], kb2[13]);
                    float hi_30 = _fmax_309;
                    float _min_262 = fminf(kb2[6], kb2[13]);
                    float lo_31 = _min_262;
                    kb2[6] = hi_30;
                    kb2[13] = lo_31;
                    float _fmax_310 = fmaxf(kb2[8], kb2[14]);
                    float hi_32 = _fmax_310;
                    float _min_263 = fminf(kb2[8], kb2[14]);
                    float lo_33 = _min_263;
                    kb2[8] = hi_32;
                    kb2[14] = lo_33;
                    float _fmax_311 = fmaxf(kb2[10], kb2[15]);
                    float hi_34 = _fmax_311;
                    float _min_264 = fminf(kb2[10], kb2[15]);
                    float lo_35 = _min_264;
                    kb2[10] = hi_34;
                    kb2[15] = lo_35;
                    float _fmax_312 = fmaxf(kb2[11], kb2[12]);
                    float hi_36 = _fmax_312;
                    float _min_265 = fminf(kb2[11], kb2[12]);
                    float lo_37 = _min_265;
                    kb2[11] = hi_36;
                    kb2[12] = lo_37;
                    float _fmax_313 = fmaxf(kb2[0], kb2[1]);
                    float hi_38_1 = _fmax_313;
                    float _min_266 = fminf(kb2[0], kb2[1]);
                    float lo_39_1 = _min_266;
                    kb2[0] = hi_38_1;
                    kb2[1] = lo_39_1;
                    float _fmax_314 = fmaxf(kb2[2], kb2[3]);
                    float hi_41_1 = _fmax_314;
                    float _min_267 = fminf(kb2[2], kb2[3]);
                    float lo_42_1 = _min_267;
                    kb2[2] = hi_41_1;
                    kb2[3] = lo_42_1;
                    float _fmax_315 = fmaxf(kb2[4], kb2[5]);
                    float hi_44_1 = _fmax_315;
                    float _min_268 = fminf(kb2[4], kb2[5]);
                    float lo_45_1 = _min_268;
                    kb2[4] = hi_44_1;
                    kb2[5] = lo_45_1;
                    float _fmax_316 = fmaxf(kb2[6], kb2[8]);
                    float hi_47_1 = _fmax_316;
                    float _min_269 = fminf(kb2[6], kb2[8]);
                    float lo_48_1 = _min_269;
                    kb2[6] = hi_47_1;
                    kb2[8] = lo_48_1;
                    float _fmax_317 = fmaxf(kb2[7], kb2[9]);
                    float hi_49 = _fmax_317;
                    float _min_270 = fminf(kb2[7], kb2[9]);
                    float lo_50 = _min_270;
                    kb2[7] = hi_49;
                    kb2[9] = lo_50;
                    float _fmax_318 = fmaxf(kb2[10], kb2[11]);
                    float hi_51 = _fmax_318;
                    float _min_271 = fminf(kb2[10], kb2[11]);
                    float lo_52 = _min_271;
                    kb2[10] = hi_51;
                    kb2[11] = lo_52;
                    float _fmax_319 = fmaxf(kb2[12], kb2[13]);
                    float hi_53 = _fmax_319;
                    float _min_272 = fminf(kb2[12], kb2[13]);
                    float lo_54 = _min_272;
                    kb2[12] = hi_53;
                    kb2[13] = lo_54;
                    float _fmax_320 = fmaxf(kb2[14], kb2[15]);
                    float hi_55_1 = _fmax_320;
                    float _min_273 = fminf(kb2[14], kb2[15]);
                    float lo_56_1 = _min_273;
                    kb2[14] = hi_55_1;
                    kb2[15] = lo_56_1;
                    float _fmax_321 = fmaxf(kb2[0], kb2[2]);
                    float hi_57 = _fmax_321;
                    float _min_274 = fminf(kb2[0], kb2[2]);
                    float lo_58 = _min_274;
                    kb2[0] = hi_57;
                    kb2[2] = lo_58;
                    float _fmax_322 = fmaxf(kb2[1], kb2[3]);
                    float hi_60_1 = _fmax_322;
                    float _min_275 = fminf(kb2[1], kb2[3]);
                    float lo_61_1 = _min_275;
                    kb2[1] = hi_60_1;
                    kb2[3] = lo_61_1;
                    float _fmax_323 = fmaxf(kb2[4], kb2[10]);
                    float hi_63 = _fmax_323;
                    float _min_276 = fminf(kb2[4], kb2[10]);
                    float lo_64 = _min_276;
                    kb2[4] = hi_63;
                    kb2[10] = lo_64;
                    float _fmax_324 = fmaxf(kb2[5], kb2[11]);
                    float hi_66_1 = _fmax_324;
                    float _min_277 = fminf(kb2[5], kb2[11]);
                    float lo_67_1 = _min_277;
                    kb2[5] = hi_66_1;
                    kb2[11] = lo_67_1;
                    float _fmax_325 = fmaxf(kb2[6], kb2[7]);
                    float hi_68_2 = _fmax_325;
                    float _min_278 = fminf(kb2[6], kb2[7]);
                    float lo_69_2 = _min_278;
                    kb2[6] = hi_68_2;
                    kb2[7] = lo_69_2;
                    float _fmax_326 = fmaxf(kb2[8], kb2[9]);
                    float hi_70_1 = _fmax_326;
                    float _min_279 = fminf(kb2[8], kb2[9]);
                    float lo_71_1 = _min_279;
                    kb2[8] = hi_70_1;
                    kb2[9] = lo_71_1;
                    float _fmax_327 = fmaxf(kb2[12], kb2[14]);
                    float hi_72_1 = _fmax_327;
                    float _min_280 = fminf(kb2[12], kb2[14]);
                    float lo_73_1 = _min_280;
                    kb2[12] = hi_72_1;
                    kb2[14] = lo_73_1;
                    float _fmax_328 = fmaxf(kb2[13], kb2[15]);
                    float hi_74_2 = _fmax_328;
                    float _min_281 = fminf(kb2[13], kb2[15]);
                    float lo_75_2 = _min_281;
                    kb2[13] = hi_74_2;
                    kb2[15] = lo_75_2;
                    float _fmax_329 = fmaxf(kb2[1], kb2[2]);
                    float hi_76_1 = _fmax_329;
                    float _min_282 = fminf(kb2[1], kb2[2]);
                    float lo_77_1 = _min_282;
                    kb2[1] = hi_76_1;
                    kb2[2] = lo_77_1;
                    float _fmax_330 = fmaxf(kb2[3], kb2[12]);
                    float hi_79 = _fmax_330;
                    float _min_283 = fminf(kb2[3], kb2[12]);
                    float lo_80 = _min_283;
                    kb2[3] = hi_79;
                    kb2[12] = lo_80;
                    float _fmax_331 = fmaxf(kb2[4], kb2[6]);
                    float hi_82_1 = _fmax_331;
                    float _min_284 = fminf(kb2[4], kb2[6]);
                    float lo_83_1 = _min_284;
                    kb2[4] = hi_82_1;
                    kb2[6] = lo_83_1;
                    float _fmax_332 = fmaxf(kb2[5], kb2[7]);
                    float hi_85_1 = _fmax_332;
                    float _min_285 = fminf(kb2[5], kb2[7]);
                    float lo_86_1 = _min_285;
                    kb2[5] = hi_85_1;
                    kb2[7] = lo_86_1;
                    float _fmax_333 = fmaxf(kb2[8], kb2[10]);
                    float hi_87 = _fmax_333;
                    float _min_286 = fminf(kb2[8], kb2[10]);
                    float lo_88 = _min_286;
                    kb2[8] = hi_87;
                    kb2[10] = lo_88;
                    float _fmax_334 = fmaxf(kb2[9], kb2[11]);
                    float hi_89 = _fmax_334;
                    float _min_287 = fminf(kb2[9], kb2[11]);
                    float lo_90 = _min_287;
                    kb2[9] = hi_89;
                    kb2[11] = lo_90;
                    float _fmax_335 = fmaxf(kb2[13], kb2[14]);
                    float hi_91_1 = _fmax_335;
                    float _min_288 = fminf(kb2[13], kb2[14]);
                    float lo_92_1 = _min_288;
                    kb2[13] = hi_91_1;
                    kb2[14] = lo_92_1;
                    float _fmax_336 = fmaxf(kb2[1], kb2[4]);
                    float hi_93 = _fmax_336;
                    float _min_289 = fminf(kb2[1], kb2[4]);
                    float lo_94 = _min_289;
                    kb2[1] = hi_93;
                    kb2[4] = lo_94;
                    float _fmax_337 = fmaxf(kb2[2], kb2[6]);
                    float hi_95_1 = _fmax_337;
                    float _min_290 = fminf(kb2[2], kb2[6]);
                    float lo_96_1 = _min_290;
                    kb2[2] = hi_95_1;
                    kb2[6] = lo_96_1;
                    float _fmax_338 = fmaxf(kb2[5], kb2[8]);
                    float hi_98_1 = _fmax_338;
                    float _min_291 = fminf(kb2[5], kb2[8]);
                    float lo_99_1 = _min_291;
                    kb2[5] = hi_98_1;
                    kb2[8] = lo_99_1;
                    float _fmax_339 = fmaxf(kb2[7], kb2[10]);
                    float hi_101 = _fmax_339;
                    float _min_292 = fminf(kb2[7], kb2[10]);
                    float lo_102 = _min_292;
                    kb2[7] = hi_101;
                    kb2[10] = lo_102;
                    float _fmax_340 = fmaxf(kb2[9], kb2[13]);
                    float hi_104_1 = _fmax_340;
                    float _min_293 = fminf(kb2[9], kb2[13]);
                    float lo_105_1 = _min_293;
                    kb2[9] = hi_104_1;
                    kb2[13] = lo_105_1;
                    float _fmax_341 = fmaxf(kb2[11], kb2[14]);
                    float hi_106_1 = _fmax_341;
                    float _min_294 = fminf(kb2[11], kb2[14]);
                    float lo_107_1 = _min_294;
                    kb2[11] = hi_106_1;
                    kb2[14] = lo_107_1;
                    float _fmax_342 = fmaxf(kb2[2], kb2[4]);
                    float hi_108_1 = _fmax_342;
                    float _min_295 = fminf(kb2[2], kb2[4]);
                    float lo_109_1 = _min_295;
                    kb2[2] = hi_108_1;
                    kb2[4] = lo_109_1;
                    float _fmax_343 = fmaxf(kb2[3], kb2[6]);
                    float hi_110_1 = _fmax_343;
                    float _min_296 = fminf(kb2[3], kb2[6]);
                    float lo_111_1 = _min_296;
                    kb2[3] = hi_110_1;
                    kb2[6] = lo_111_1;
                    float _fmax_344 = fmaxf(kb2[9], kb2[12]);
                    float hi_112_1 = _fmax_344;
                    float _min_297 = fminf(kb2[9], kb2[12]);
                    float lo_113_1 = _min_297;
                    kb2[9] = hi_112_1;
                    kb2[12] = lo_113_1;
                    float _fmax_345 = fmaxf(kb2[11], kb2[13]);
                    float hi_114_1 = _fmax_345;
                    float _min_298 = fminf(kb2[11], kb2[13]);
                    float lo_115_1 = _min_298;
                    kb2[11] = hi_114_1;
                    kb2[13] = lo_115_1;
                    float _fmax_346 = fmaxf(kb2[3], kb2[5]);
                    float hi_117 = _fmax_346;
                    float _min_299 = fminf(kb2[3], kb2[5]);
                    float lo_118 = _min_299;
                    kb2[3] = hi_117;
                    kb2[5] = lo_118;
                    float _fmax_347 = fmaxf(kb2[6], kb2[8]);
                    float hi_120_1 = _fmax_347;
                    float _min_300 = fminf(kb2[6], kb2[8]);
                    float lo_121_1 = _min_300;
                    kb2[6] = hi_120_1;
                    kb2[8] = lo_121_1;
                    float _fmax_348 = fmaxf(kb2[7], kb2[9]);
                    float hi_123 = _fmax_348;
                    float _min_301 = fminf(kb2[7], kb2[9]);
                    float lo_124 = _min_301;
                    kb2[7] = hi_123;
                    kb2[9] = lo_124;
                    float _fmax_349 = fmaxf(kb2[10], kb2[12]);
                    float hi_125 = _fmax_349;
                    float _min_302 = fminf(kb2[10], kb2[12]);
                    float lo_126 = _min_302;
                    kb2[10] = hi_125;
                    kb2[12] = lo_126;
                    float _fmax_350 = fmaxf(kb2[3], kb2[4]);
                    float hi_127 = _fmax_350;
                    float _min_303 = fminf(kb2[3], kb2[4]);
                    float lo_128 = _min_303;
                    kb2[3] = hi_127;
                    kb2[4] = lo_128;
                    float _fmax_351 = fmaxf(kb2[5], kb2[6]);
                    float hi_129 = _fmax_351;
                    float _min_304 = fminf(kb2[5], kb2[6]);
                    float lo_130 = _min_304;
                    kb2[5] = hi_129;
                    kb2[6] = lo_130;
                    float _fmax_352 = fmaxf(kb2[7], kb2[8]);
                    float hi_131 = _fmax_352;
                    float _min_305 = fminf(kb2[7], kb2[8]);
                    float lo_132 = _min_305;
                    kb2[7] = hi_131;
                    kb2[8] = lo_132;
                    float _fmax_353 = fmaxf(kb2[9], kb2[10]);
                    float hi_133 = _fmax_353;
                    float _min_306 = fminf(kb2[9], kb2[10]);
                    float lo_134 = _min_306;
                    kb2[9] = hi_133;
                    kb2[10] = lo_134;
                    float _fmax_354 = fmaxf(kb2[11], kb2[12]);
                    float hi_136_1 = _fmax_354;
                    float _min_307 = fminf(kb2[11], kb2[12]);
                    float lo_137_1 = _min_307;
                    kb2[11] = hi_136_1;
                    kb2[12] = lo_137_1;
                    float _fmax_355 = fmaxf(kb2[6], kb2[7]);
                    float hi_139 = _fmax_355;
                    float _min_308 = fminf(kb2[6], kb2[7]);
                    float lo_140 = _min_308;
                    kb2[6] = hi_139;
                    kb2[7] = lo_140;
                    float _fmax_356 = fmaxf(kb2[8], kb2[9]);
                    float hi_142_1 = _fmax_356;
                    float _min_309 = fminf(kb2[8], kb2[9]);
                    float lo_143_1 = _min_309;
                    kb2[8] = hi_142_1;
                    kb2[9] = lo_143_1;
                    float _fmax_357 = fmaxf(a2[0], kb2[15]);
                    float hi_144_1 = _fmax_357;
                    a2[0] = hi_144_1;
                    float _fmax_358 = fmaxf(a2[1], kb2[14]);
                    float hi_146_1 = _fmax_358;
                    a2[1] = hi_146_1;
                    float _fmax_359 = fmaxf(a2[2], kb2[13]);
                    float hi_147 = _fmax_359;
                    a2[2] = hi_147;
                    float _fmax_360 = fmaxf(a2[3], kb2[12]);
                    float hi_149 = _fmax_360;
                    a2[3] = hi_149;
                    float _fmax_361 = fmaxf(a2[4], kb2[11]);
                    float hi_150_1 = _fmax_361;
                    a2[4] = hi_150_1;
                    float _fmax_362 = fmaxf(a2[5], kb2[10]);
                    float hi_152_1 = _fmax_362;
                    a2[5] = hi_152_1;
                    float _fmax_363 = fmaxf(a2[6], kb2[9]);
                    float hi_153 = _fmax_363;
                    a2[6] = hi_153;
                    float _fmax_364 = fmaxf(a2[7], kb2[8]);
                    float hi_155 = _fmax_364;
                    a2[7] = hi_155;
                    float _fmax_365 = fmaxf(a2[8], kb2[7]);
                    float hi_156_1 = _fmax_365;
                    a2[8] = hi_156_1;
                    float _fmax_366 = fmaxf(a2[9], kb2[6]);
                    float hi_157 = _fmax_366;
                    a2[9] = hi_157;
                    float _fmax_367 = fmaxf(a2[10], kb2[5]);
                    float hi_158_1 = _fmax_367;
                    a2[10] = hi_158_1;
                    float _fmax_368 = fmaxf(a2[11], kb2[4]);
                    float hi_159 = _fmax_368;
                    a2[11] = hi_159;
                    float _fmax_369 = fmaxf(a2[12], kb2[3]);
                    float hi_160_1 = _fmax_369;
                    a2[12] = hi_160_1;
                    float _fmax_370 = fmaxf(a2[13], kb2[2]);
                    float hi_162_1 = _fmax_370;
                    a2[13] = hi_162_1;
                    float _fmax_371 = fmaxf(a2[14], kb2[1]);
                    float hi_163 = _fmax_371;
                    a2[14] = hi_163;
                    float _fmax_372 = fmaxf(a2[15], kb2[0]);
                    float hi_165 = _fmax_372;
                    a2[15] = hi_165;
                    float _fmax_373 = fmaxf(a2[0], a2[8]);
                    float hi_166_1 = _fmax_373;
                    float _min_310 = fminf(a2[0], a2[8]);
                    float lo_167_1 = _min_310;
                    a2[0] = hi_166_1;
                    a2[8] = lo_167_1;
                    float _fmax_374 = fmaxf(a2[1], a2[9]);
                    float hi_168_1 = _fmax_374;
                    float _min_311 = fminf(a2[1], a2[9]);
                    float lo_169_1 = _min_311;
                    a2[1] = hi_168_1;
                    a2[9] = lo_169_1;
                    float _fmax_375 = fmaxf(a2[2], a2[10]);
                    float hi_171 = _fmax_375;
                    float _min_312 = fminf(a2[2], a2[10]);
                    float lo_172 = _min_312;
                    a2[2] = hi_171;
                    a2[10] = lo_172;
                    float _fmax_376 = fmaxf(a2[3], a2[11]);
                    float hi_173 = _fmax_376;
                    float _min_313 = fminf(a2[3], a2[11]);
                    float lo_174 = _min_313;
                    a2[3] = hi_173;
                    a2[11] = lo_174;
                    float _fmax_377 = fmaxf(a2[4], a2[12]);
                    float hi_175 = _fmax_377;
                    float _min_314 = fminf(a2[4], a2[12]);
                    float lo_176 = _min_314;
                    a2[4] = hi_175;
                    a2[12] = lo_176;
                    float _fmax_378 = fmaxf(a2[5], a2[13]);
                    float hi_178_1 = _fmax_378;
                    float _min_315 = fminf(a2[5], a2[13]);
                    float lo_179_1 = _min_315;
                    a2[5] = hi_178_1;
                    a2[13] = lo_179_1;
                    float _fmax_379 = fmaxf(a2[6], a2[14]);
                    float hi_181 = _fmax_379;
                    float _min_316 = fminf(a2[6], a2[14]);
                    float lo_182 = _min_316;
                    a2[6] = hi_181;
                    a2[14] = lo_182;
                    float _fmax_380 = fmaxf(a2[7], a2[15]);
                    float hi_184_1 = _fmax_380;
                    float _min_317 = fminf(a2[7], a2[15]);
                    float lo_185_1 = _min_317;
                    a2[7] = hi_184_1;
                    a2[15] = lo_185_1;
                    float _fmax_381 = fmaxf(a2[0], a2[4]);
                    float hi_187 = _fmax_381;
                    float _min_318 = fminf(a2[0], a2[4]);
                    float lo_188 = _min_318;
                    a2[0] = hi_187;
                    a2[4] = lo_188;
                    float _fmax_382 = fmaxf(a2[1], a2[5]);
                    float hi_189 = _fmax_382;
                    float _min_319 = fminf(a2[1], a2[5]);
                    float lo_190 = _min_319;
                    a2[1] = hi_189;
                    a2[5] = lo_190;
                    float _fmax_383 = fmaxf(a2[2], a2[6]);
                    float hi_191 = _fmax_383;
                    float _min_320 = fminf(a2[2], a2[6]);
                    float lo_192 = _min_320;
                    a2[2] = hi_191;
                    a2[6] = lo_192;
                    float _fmax_384 = fmaxf(a2[3], a2[7]);
                    float hi_194_1 = _fmax_384;
                    float _min_321 = fminf(a2[3], a2[7]);
                    float lo_195_1 = _min_321;
                    a2[3] = hi_194_1;
                    a2[7] = lo_195_1;
                    float _fmax_385 = fmaxf(a2[8], a2[12]);
                    float hi_197 = _fmax_385;
                    float _min_322 = fminf(a2[8], a2[12]);
                    float lo_198 = _min_322;
                    a2[8] = hi_197;
                    a2[12] = lo_198;
                    float _fmax_386 = fmaxf(a2[9], a2[13]);
                    float hi_200_1 = _fmax_386;
                    float _min_323 = fminf(a2[9], a2[13]);
                    float lo_201_1 = _min_323;
                    a2[9] = hi_200_1;
                    a2[13] = lo_201_1;
                    float _fmax_387 = fmaxf(a2[10], a2[14]);
                    float hi_203 = _fmax_387;
                    float _min_324 = fminf(a2[10], a2[14]);
                    float lo_204 = _min_324;
                    a2[10] = hi_203;
                    a2[14] = lo_204;
                    float _fmax_388 = fmaxf(a2[11], a2[15]);
                    float hi_205 = _fmax_388;
                    float _min_325 = fminf(a2[11], a2[15]);
                    float lo_206 = _min_325;
                    a2[11] = hi_205;
                    a2[15] = lo_206;
                    float _fmax_389 = fmaxf(a2[0], a2[2]);
                    float hi_207 = _fmax_389;
                    float _min_326 = fminf(a2[0], a2[2]);
                    float lo_208 = _min_326;
                    a2[0] = hi_207;
                    a2[2] = lo_208;
                    float _fmax_390 = fmaxf(a2[1], a2[3]);
                    float hi_210_1 = _fmax_390;
                    float _min_327 = fminf(a2[1], a2[3]);
                    float lo_211_1 = _min_327;
                    a2[1] = hi_210_1;
                    a2[3] = lo_211_1;
                    float _fmax_391 = fmaxf(a2[4], a2[6]);
                    float hi_213 = _fmax_391;
                    float _min_328 = fminf(a2[4], a2[6]);
                    float lo_214 = _min_328;
                    a2[4] = hi_213;
                    a2[6] = lo_214;
                    float _fmax_392 = fmaxf(a2[5], a2[7]);
                    float hi_216_1 = _fmax_392;
                    float _min_329 = fminf(a2[5], a2[7]);
                    float lo_217_1 = _min_329;
                    a2[5] = hi_216_1;
                    a2[7] = lo_217_1;
                    float _fmax_393 = fmaxf(a2[8], a2[10]);
                    float hi_219 = _fmax_393;
                    float _min_330 = fminf(a2[8], a2[10]);
                    float lo_220 = _min_330;
                    a2[8] = hi_219;
                    a2[10] = lo_220;
                    float _fmax_394 = fmaxf(a2[9], a2[11]);
                    float hi_221 = _fmax_394;
                    float _min_331 = fminf(a2[9], a2[11]);
                    float lo_222 = _min_331;
                    a2[9] = hi_221;
                    a2[11] = lo_222;
                    float _fmax_395 = fmaxf(a2[12], a2[14]);
                    float hi_223 = _fmax_395;
                    float _min_332 = fminf(a2[12], a2[14]);
                    float lo_224 = _min_332;
                    a2[12] = hi_223;
                    a2[14] = lo_224;
                    float _fmax_396 = fmaxf(a2[13], a2[15]);
                    float hi_226_1 = _fmax_396;
                    float _min_333 = fminf(a2[13], a2[15]);
                    float lo_227_1 = _min_333;
                    a2[13] = hi_226_1;
                    a2[15] = lo_227_1;
                    float _fmax_397 = fmaxf(a2[0], a2[1]);
                    float hi_229 = _fmax_397;
                    float _min_334 = fminf(a2[0], a2[1]);
                    float lo_230 = _min_334;
                    a2[0] = hi_229;
                    a2[1] = lo_230;
                    float _fmax_398 = fmaxf(a2[2], a2[3]);
                    float hi_232_1 = _fmax_398;
                    float _min_335 = fminf(a2[2], a2[3]);
                    float lo_233_1 = _min_335;
                    a2[2] = hi_232_1;
                    a2[3] = lo_233_1;
                    float _fmax_399 = fmaxf(a2[4], a2[5]);
                    float hi_235 = _fmax_399;
                    float _min_336 = fminf(a2[4], a2[5]);
                    float lo_236 = _min_336;
                    a2[4] = hi_235;
                    a2[5] = lo_236;
                    float _fmax_400 = fmaxf(a2[6], a2[7]);
                    float hi_237 = _fmax_400;
                    float _min_337 = fminf(a2[6], a2[7]);
                    float lo_238 = _min_337;
                    a2[6] = hi_237;
                    a2[7] = lo_238;
                    float _fmax_401 = fmaxf(a2[8], a2[9]);
                    float hi_239 = _fmax_401;
                    float _min_338 = fminf(a2[8], a2[9]);
                    float lo_240 = _min_338;
                    a2[8] = hi_239;
                    a2[9] = lo_240;
                    float _fmax_402 = fmaxf(a2[10], a2[11]);
                    float hi_242_1 = _fmax_402;
                    float _min_339 = fminf(a2[10], a2[11]);
                    float lo_243_1 = _min_339;
                    a2[10] = hi_242_1;
                    a2[11] = lo_243_1;
                    float _fmax_403 = fmaxf(a2[12], a2[13]);
                    float hi_245 = _fmax_403;
                    float _min_340 = fminf(a2[12], a2[13]);
                    float lo_246 = _min_340;
                    a2[12] = hi_245;
                    a2[13] = lo_246;
                    float _fmax_404 = fmaxf(a2[14], a2[15]);
                    float hi_248_1 = _fmax_404;
                    float _min_341 = fminf(a2[14], a2[15]);
                    float lo_249_1 = _min_341;
                    a2[14] = hi_248_1;
                    a2[15] = lo_249_1;
                }
                cb2[0] = nb2[0];
                cb2[1] = nb2[1];
                cb2[2] = nb2[2];
                cb2[3] = nb2[3];
                cb2[4] = nb2[4];
                cb2[5] = nb2[5];
                cb2[6] = nb2[6];
                cb2[7] = nb2[7];
                cb2[8] = nb2[8];
                cb2[9] = nb2[9];
                cb2[10] = nb2[10];
                cb2[11] = nb2[11];
                cb2[12] = nb2[12];
                cb2[13] = nb2[13];
                cb2[14] = nb2[14];
                cb2[15] = nb2[15];
            }
        }
        float rr2[1];
        int pb_0 = tid_1 * 17;
        pub[pb_0] = a2[0];
        pub[pb_0 + 1] = a2[1];
        pub[pb_0 + 2] = a2[2];
        pub[pb_0 + 3] = a2[3];
        pub[pb_0 + 4] = a2[4];
        pub[pb_0 + 5] = a2[5];
        pub[pb_0 + 6] = a2[6];
        pub[pb_0 + 7] = a2[7];
        pub[pb_0 + 8] = a2[8];
        pub[pb_0 + 9] = a2[9];
        pub[pb_0 + 10] = a2[10];
        pub[pb_0 + 11] = a2[11];
        pub[pb_0 + 12] = a2[12];
        pub[pb_0 + 13] = a2[13];
        pub[pb_0 + 14] = a2[14];
        pub[pb_0 + 15] = a2[15];
        pub[pb_0 + 16] = neg_inf;
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        float r_1_1 = neg_inf;
        float V_2[8];
        int s0_3 = (sg * 16 + cg) * 17;
        int s1_4 = (sg * 16 + 8 + cg) * 17;
        float x0_5 = pub[s0_3 + ln];
        float y0_6 = pub[s1_4 + lnr];
        float _min_342 = fminf(x0_5, y0_6);
        float lo0_7 = _min_342;
        float _fmax_405 = fmaxf(r_1_1, lo0_7);
        r_1_1 = _fmax_405;
        float _fmax_406 = fmaxf(x0_5, y0_6);
        float hi0_8 = _fmax_406;
        float cur_9 = hi0_8;
        float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 8);
        float pv_10 = _shfl_xor_102;
        float _fmax_407 = fmaxf(cur_9, pv_10);
        float hi_11_1 = _fmax_407;
        float _min_343 = fminf(cur_9, pv_10);
        float lo_12_1 = _min_343;
        cur_9 = ((up[0] != 0) ? hi_11_1 : lo_12_1);
        float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 4);
        float pv_13 = _shfl_xor_103;
        float _fmax_408 = fmaxf(cur_9, pv_13);
        float hi_14 = _fmax_408;
        float _min_344 = fminf(cur_9, pv_13);
        float lo_15 = _min_344;
        cur_9 = ((up[1] != 0) ? hi_14 : lo_15);
        float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 2);
        float pv_16 = _shfl_xor_104;
        float _fmax_409 = fmaxf(cur_9, pv_16);
        float hi_17_1 = _fmax_409;
        float _min_345 = fminf(cur_9, pv_16);
        float lo_18_1 = _min_345;
        cur_9 = ((up[2] != 0) ? hi_17_1 : lo_18_1);
        float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 1);
        float pv_19_1 = _shfl_xor_105;
        float _fmax_410 = fmaxf(cur_9, pv_19_1);
        float hi_20_1 = _fmax_410;
        float _min_346 = fminf(cur_9, pv_19_1);
        float lo_21_1 = _min_346;
        cur_9 = ((up[3] != 0) ? hi_20_1 : lo_21_1);
        V_2[0] = cur_9;
        int s0_22 = (sg * 16 + 1 + cg) * 17;
        int s1_23 = (sg * 16 + 1 + 8 + cg) * 17;
        float x0_24 = pub[s0_22 + ln];
        float y0_25 = pub[s1_23 + lnr];
        float _min_347 = fminf(x0_24, y0_25);
        float lo0_26 = _min_347;
        float _fmax_411 = fmaxf(r_1_1, lo0_26);
        r_1_1 = _fmax_411;
        float _fmax_412 = fmaxf(x0_24, y0_25);
        float hi0_27 = _fmax_412;
        float cur_28 = hi0_27;
        float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 8);
        float pv_29 = _shfl_xor_106;
        float _fmax_413 = fmaxf(cur_28, pv_29);
        float hi_30_1 = _fmax_413;
        float _min_348 = fminf(cur_28, pv_29);
        float lo_31_1 = _min_348;
        cur_28 = ((up[0] != 0) ? hi_30_1 : lo_31_1);
        float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 4);
        float pv_32 = _shfl_xor_107;
        float _fmax_414 = fmaxf(cur_28, pv_32);
        float hi_33 = _fmax_414;
        float _min_349 = fminf(cur_28, pv_32);
        float lo_34 = _min_349;
        cur_28 = ((up[1] != 0) ? hi_33 : lo_34);
        float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 2);
        float pv_35 = _shfl_xor_108;
        float _fmax_415 = fmaxf(cur_28, pv_35);
        float hi_36_1 = _fmax_415;
        float _min_350 = fminf(cur_28, pv_35);
        float lo_37_1 = _min_350;
        cur_28 = ((up[2] != 0) ? hi_36_1 : lo_37_1);
        float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 1);
        float pv_38 = _shfl_xor_109;
        float _fmax_416 = fmaxf(cur_28, pv_38);
        float hi_39 = _fmax_416;
        float _min_351 = fminf(cur_28, pv_38);
        float lo_40 = _min_351;
        cur_28 = ((up[3] != 0) ? hi_39 : lo_40);
        V_2[1] = cur_28;
        int s0_41 = (sg * 16 + 2 + cg) * 17;
        int s1_42 = (sg * 16 + 2 + 8 + cg) * 17;
        float x0_43 = pub[s0_41 + ln];
        float y0_44 = pub[s1_42 + lnr];
        float _min_352 = fminf(x0_43, y0_44);
        float lo0_45 = _min_352;
        float _fmax_417 = fmaxf(r_1_1, lo0_45);
        r_1_1 = _fmax_417;
        float _fmax_418 = fmaxf(x0_43, y0_44);
        float hi0_46 = _fmax_418;
        float cur_47 = hi0_46;
        float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 8);
        float pv_48 = _shfl_xor_110;
        float _fmax_419 = fmaxf(cur_47, pv_48);
        float hi_49_1 = _fmax_419;
        float _min_353 = fminf(cur_47, pv_48);
        float lo_50_1 = _min_353;
        cur_47 = ((up[0] != 0) ? hi_49_1 : lo_50_1);
        float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 4);
        float pv_51 = _shfl_xor_111;
        float _fmax_420 = fmaxf(cur_47, pv_51);
        float hi_52_1 = _fmax_420;
        float _min_354 = fminf(cur_47, pv_51);
        float lo_53_1 = _min_354;
        cur_47 = ((up[1] != 0) ? hi_52_1 : lo_53_1);
        float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 2);
        float pv_54_1 = _shfl_xor_112;
        float _fmax_421 = fmaxf(cur_47, pv_54_1);
        float hi_55_2 = _fmax_421;
        float _min_355 = fminf(cur_47, pv_54_1);
        float lo_56_2 = _min_355;
        cur_47 = ((up[2] != 0) ? hi_55_2 : lo_56_2);
        float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 1);
        float pv_57_1 = _shfl_xor_113;
        float _fmax_422 = fmaxf(cur_47, pv_57_1);
        float hi_58_2 = _fmax_422;
        float _min_356 = fminf(cur_47, pv_57_1);
        float lo_59_2 = _min_356;
        cur_47 = ((up[3] != 0) ? hi_58_2 : lo_59_2);
        V_2[2] = cur_47;
        int s0_60 = (sg * 16 + 3 + cg) * 17;
        int s1_61 = (sg * 16 + 3 + 8 + cg) * 17;
        float x0_62 = pub[s0_60 + ln];
        float y0_63 = pub[s1_61 + lnr];
        float _min_357 = fminf(x0_62, y0_63);
        float lo0_64 = _min_357;
        float _fmax_423 = fmaxf(r_1_1, lo0_64);
        r_1_1 = _fmax_423;
        float _fmax_424 = fmaxf(x0_62, y0_63);
        float hi0_65 = _fmax_424;
        float cur_66_1 = hi0_65;
        float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, cur_66_1, 8);
        float pv_67_1 = _shfl_xor_114;
        float _fmax_425 = fmaxf(cur_66_1, pv_67_1);
        float hi_68_3 = _fmax_425;
        float _min_358 = fminf(cur_66_1, pv_67_1);
        float lo_69_3 = _min_358;
        cur_66_1 = ((up[0] != 0) ? hi_68_3 : lo_69_3);
        float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, cur_66_1, 4);
        float pv_70_1 = _shfl_xor_115;
        float _fmax_426 = fmaxf(cur_66_1, pv_70_1);
        float hi_71_1 = _fmax_426;
        float _min_359 = fminf(cur_66_1, pv_70_1);
        float lo_72_1 = _min_359;
        cur_66_1 = ((up[1] != 0) ? hi_71_1 : lo_72_1);
        float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, cur_66_1, 2);
        float pv_73_1 = _shfl_xor_116;
        float _fmax_427 = fmaxf(cur_66_1, pv_73_1);
        float hi_74_3 = _fmax_427;
        float _min_360 = fminf(cur_66_1, pv_73_1);
        float lo_75_3 = _min_360;
        cur_66_1 = ((up[2] != 0) ? hi_74_3 : lo_75_3);
        float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, cur_66_1, 1);
        float pv_76_1 = _shfl_xor_117;
        float _fmax_428 = fmaxf(cur_66_1, pv_76_1);
        float hi_77_1 = _fmax_428;
        float _min_361 = fminf(cur_66_1, pv_76_1);
        float lo_78_1 = _min_361;
        cur_66_1 = ((up[3] != 0) ? hi_77_1 : lo_78_1);
        V_2[3] = cur_66_1;
        int s0_79 = (sg * 16 + 4 + cg) * 17;
        int s1_80 = (sg * 16 + 4 + 8 + cg) * 17;
        float x0_81 = pub[s0_79 + ln];
        float y0_82 = pub[s1_80 + lnr];
        float _min_362 = fminf(x0_81, y0_82);
        float lo0_83 = _min_362;
        float _fmax_429 = fmaxf(r_1_1, lo0_83);
        r_1_1 = _fmax_429;
        float _fmax_430 = fmaxf(x0_81, y0_82);
        float hi0_84 = _fmax_430;
        float cur_85 = hi0_84;
        float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, cur_85, 8);
        float pv_86 = _shfl_xor_118;
        float _fmax_431 = fmaxf(cur_85, pv_86);
        float hi_87_1 = _fmax_431;
        float _min_363 = fminf(cur_85, pv_86);
        float lo_88_1 = _min_363;
        cur_85 = ((up[0] != 0) ? hi_87_1 : lo_88_1);
        float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, cur_85, 4);
        float pv_89 = _shfl_xor_119;
        float _fmax_432 = fmaxf(cur_85, pv_89);
        float hi_90_1 = _fmax_432;
        float _min_364 = fminf(cur_85, pv_89);
        float lo_91_1 = _min_364;
        cur_85 = ((up[1] != 0) ? hi_90_1 : lo_91_1);
        float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, cur_85, 2);
        float pv_92 = _shfl_xor_120;
        float _fmax_433 = fmaxf(cur_85, pv_92);
        float hi_93_1 = _fmax_433;
        float _min_365 = fminf(cur_85, pv_92);
        float lo_94_1 = _min_365;
        cur_85 = ((up[2] != 0) ? hi_93_1 : lo_94_1);
        float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, cur_85, 1);
        float pv_95 = _shfl_xor_121;
        float _fmax_434 = fmaxf(cur_85, pv_95);
        float hi_96_1 = _fmax_434;
        float _min_366 = fminf(cur_85, pv_95);
        float lo_97_1 = _min_366;
        cur_85 = ((up[3] != 0) ? hi_96_1 : lo_97_1);
        V_2[4] = cur_85;
        int s0_98 = (sg * 16 + 5 + cg) * 17;
        int s1_99 = (sg * 16 + 5 + 8 + cg) * 17;
        float x0_100 = pub[s0_98 + ln];
        float y0_101 = pub[s1_99 + lnr];
        float _min_367 = fminf(x0_100, y0_101);
        float lo0_102 = _min_367;
        float _fmax_435 = fmaxf(r_1_1, lo0_102);
        r_1_1 = _fmax_435;
        float _fmax_436 = fmaxf(x0_100, y0_101);
        float hi0_103 = _fmax_436;
        float cur_104 = hi0_103;
        float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, cur_104, 8);
        float pv_105 = _shfl_xor_122;
        float _fmax_437 = fmaxf(cur_104, pv_105);
        float hi_106_2 = _fmax_437;
        float _min_368 = fminf(cur_104, pv_105);
        float lo_107_2 = _min_368;
        cur_104 = ((up[0] != 0) ? hi_106_2 : lo_107_2);
        float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, cur_104, 4);
        float pv_108 = _shfl_xor_123;
        float _fmax_438 = fmaxf(cur_104, pv_108);
        float hi_109 = _fmax_438;
        float _min_369 = fminf(cur_104, pv_108);
        float lo_110 = _min_369;
        cur_104 = ((up[1] != 0) ? hi_109 : lo_110);
        float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, cur_104, 2);
        float pv_111 = _shfl_xor_124;
        float _fmax_439 = fmaxf(cur_104, pv_111);
        float hi_112_2 = _fmax_439;
        float _min_370 = fminf(cur_104, pv_111);
        float lo_113_2 = _min_370;
        cur_104 = ((up[2] != 0) ? hi_112_2 : lo_113_2);
        float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, cur_104, 1);
        float pv_114 = _shfl_xor_125;
        float _fmax_440 = fmaxf(cur_104, pv_114);
        float hi_115 = _fmax_440;
        float _min_371 = fminf(cur_104, pv_114);
        float lo_116 = _min_371;
        cur_104 = ((up[3] != 0) ? hi_115 : lo_116);
        V_2[5] = cur_104;
        int s0_117 = (sg * 16 + 6 + cg) * 17;
        int s1_118 = (sg * 16 + 6 + 8 + cg) * 17;
        float x0_119 = pub[s0_117 + ln];
        float y0_120 = pub[s1_118 + lnr];
        float _min_372 = fminf(x0_119, y0_120);
        float lo0_121 = _min_372;
        float _fmax_441 = fmaxf(r_1_1, lo0_121);
        r_1_1 = _fmax_441;
        float _fmax_442 = fmaxf(x0_119, y0_120);
        float hi0_122 = _fmax_442;
        float cur_123 = hi0_122;
        float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, cur_123, 8);
        float pv_124 = _shfl_xor_126;
        float _fmax_443 = fmaxf(cur_123, pv_124);
        float hi_125_1 = _fmax_443;
        float _min_373 = fminf(cur_123, pv_124);
        float lo_126_1 = _min_373;
        cur_123 = ((up[0] != 0) ? hi_125_1 : lo_126_1);
        float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, cur_123, 4);
        float pv_127 = _shfl_xor_127;
        float _fmax_444 = fmaxf(cur_123, pv_127);
        float hi_128_1 = _fmax_444;
        float _min_374 = fminf(cur_123, pv_127);
        float lo_129_1 = _min_374;
        cur_123 = ((up[1] != 0) ? hi_128_1 : lo_129_1);
        float _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, cur_123, 2);
        float pv_130 = _shfl_xor_128;
        float _fmax_445 = fmaxf(cur_123, pv_130);
        float hi_131_1 = _fmax_445;
        float _min_375 = fminf(cur_123, pv_130);
        float lo_132_1 = _min_375;
        cur_123 = ((up[2] != 0) ? hi_131_1 : lo_132_1);
        float _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, cur_123, 1);
        float pv_133 = _shfl_xor_129;
        float _fmax_446 = fmaxf(cur_123, pv_133);
        float hi_134_1 = _fmax_446;
        float _min_376 = fminf(cur_123, pv_133);
        float lo_135_1 = _min_376;
        cur_123 = ((up[3] != 0) ? hi_134_1 : lo_135_1);
        V_2[6] = cur_123;
        int s0_136 = (sg * 16 + 7 + cg) * 17;
        int s1_137 = (sg * 16 + 7 + 8 + cg) * 17;
        float x0_138 = pub[s0_136 + ln];
        float y0_139 = pub[s1_137 + lnr];
        float _min_377 = fminf(x0_138, y0_139);
        float lo0_140 = _min_377;
        float _fmax_447 = fmaxf(r_1_1, lo0_140);
        r_1_1 = _fmax_447;
        float _fmax_448 = fmaxf(x0_138, y0_139);
        float hi0_141 = _fmax_448;
        float cur_142 = hi0_141;
        float _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 8);
        float pv_143 = _shfl_xor_130;
        float _fmax_449 = fmaxf(cur_142, pv_143);
        float hi_144_2 = _fmax_449;
        float _min_378 = fminf(cur_142, pv_143);
        float lo_145_1 = _min_378;
        cur_142 = ((up[0] != 0) ? hi_144_2 : lo_145_1);
        float _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 4);
        float pv_146 = _shfl_xor_131;
        float _fmax_450 = fmaxf(cur_142, pv_146);
        float hi_147_1 = _fmax_450;
        float _min_379 = fminf(cur_142, pv_146);
        float lo_148 = _min_379;
        cur_142 = ((up[1] != 0) ? hi_147_1 : lo_148);
        float _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 2);
        float pv_149 = _shfl_xor_132;
        float _fmax_451 = fmaxf(cur_142, pv_149);
        float hi_150_2 = _fmax_451;
        float _min_380 = fminf(cur_142, pv_149);
        float lo_151_1 = _min_380;
        cur_142 = ((up[2] != 0) ? hi_150_2 : lo_151_1);
        float _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 1);
        float pv_152 = _shfl_xor_133;
        float _fmax_452 = fmaxf(cur_142, pv_152);
        float hi_153_1 = _fmax_452;
        float _min_381 = fminf(cur_142, pv_152);
        float lo_154 = _min_381;
        cur_142 = ((up[3] != 0) ? hi_153_1 : lo_154);
        V_2[7] = cur_142;
        float rs_155 = pub[(sg * 16 + ln + cg) * 17 + 16];
        float _fmax_453 = fmaxf(r_1_1, rs_155);
        r_1_1 = _fmax_453;
        float _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, V_2[4], 15);
        float y1_157 = _shfl_xor_134;
        float _min_382 = fminf(V_2[0], y1_157);
        float lo1_158 = _min_382;
        float _fmax_454 = fmaxf(r_1_1, lo1_158);
        r_1_1 = _fmax_454;
        float _fmax_455 = fmaxf(V_2[0], y1_157);
        float hi1_159 = _fmax_455;
        float cur_160 = hi1_159;
        float _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, cur_160, 8);
        float pv_161 = _shfl_xor_135;
        float _fmax_456 = fmaxf(cur_160, pv_161);
        float hi_162_2 = _fmax_456;
        float _min_383 = fminf(cur_160, pv_161);
        float lo_163_1 = _min_383;
        cur_160 = ((up[0] != 0) ? hi_162_2 : lo_163_1);
        float _shfl_xor_136 = __shfl_xor_sync(0xFFFFFFFF, cur_160, 4);
        float pv_164 = _shfl_xor_136;
        float _fmax_457 = fmaxf(cur_160, pv_164);
        float hi_165_1 = _fmax_457;
        float _min_384 = fminf(cur_160, pv_164);
        float lo_166 = _min_384;
        cur_160 = ((up[1] != 0) ? hi_165_1 : lo_166);
        float _shfl_xor_137 = __shfl_xor_sync(0xFFFFFFFF, cur_160, 2);
        float pv_167 = _shfl_xor_137;
        float _fmax_458 = fmaxf(cur_160, pv_167);
        float hi_168_2 = _fmax_458;
        float _min_385 = fminf(cur_160, pv_167);
        float lo_169_2 = _min_385;
        cur_160 = ((up[2] != 0) ? hi_168_2 : lo_169_2);
        float _shfl_xor_138 = __shfl_xor_sync(0xFFFFFFFF, cur_160, 1);
        float pv_170 = _shfl_xor_138;
        float _fmax_459 = fmaxf(cur_160, pv_170);
        float hi_171_1 = _fmax_459;
        float _min_386 = fminf(cur_160, pv_170);
        float lo_172_1 = _min_386;
        cur_160 = ((up[3] != 0) ? hi_171_1 : lo_172_1);
        V_2[0] = cur_160;
        float _shfl_xor_139 = __shfl_xor_sync(0xFFFFFFFF, V_2[5], 15);
        float y1_173 = _shfl_xor_139;
        float _min_387 = fminf(V_2[1], y1_173);
        float lo1_174 = _min_387;
        float _fmax_460 = fmaxf(r_1_1, lo1_174);
        r_1_1 = _fmax_460;
        float _fmax_461 = fmaxf(V_2[1], y1_173);
        float hi1_175 = _fmax_461;
        float cur_176 = hi1_175;
        float _shfl_xor_140 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 8);
        float pv_177 = _shfl_xor_140;
        float _fmax_462 = fmaxf(cur_176, pv_177);
        float hi_178_2 = _fmax_462;
        float _min_388 = fminf(cur_176, pv_177);
        float lo_179_2 = _min_388;
        cur_176 = ((up[0] != 0) ? hi_178_2 : lo_179_2);
        float _shfl_xor_141 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 4);
        float pv_180 = _shfl_xor_141;
        float _fmax_463 = fmaxf(cur_176, pv_180);
        float hi_181_1 = _fmax_463;
        float _min_389 = fminf(cur_176, pv_180);
        float lo_182_1 = _min_389;
        cur_176 = ((up[1] != 0) ? hi_181_1 : lo_182_1);
        float _shfl_xor_142 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 2);
        float pv_183 = _shfl_xor_142;
        float _fmax_464 = fmaxf(cur_176, pv_183);
        float hi_184_2 = _fmax_464;
        float _min_390 = fminf(cur_176, pv_183);
        float lo_185_2 = _min_390;
        cur_176 = ((up[2] != 0) ? hi_184_2 : lo_185_2);
        float _shfl_xor_143 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 1);
        float pv_186 = _shfl_xor_143;
        float _fmax_465 = fmaxf(cur_176, pv_186);
        float hi_187_1 = _fmax_465;
        float _min_391 = fminf(cur_176, pv_186);
        float lo_188_1 = _min_391;
        cur_176 = ((up[3] != 0) ? hi_187_1 : lo_188_1);
        V_2[1] = cur_176;
        float _shfl_xor_144 = __shfl_xor_sync(0xFFFFFFFF, V_2[6], 15);
        float y1_189 = _shfl_xor_144;
        float _min_392 = fminf(V_2[2], y1_189);
        float lo1_190 = _min_392;
        float _fmax_466 = fmaxf(r_1_1, lo1_190);
        r_1_1 = _fmax_466;
        float _fmax_467 = fmaxf(V_2[2], y1_189);
        float hi1_191 = _fmax_467;
        float cur_192 = hi1_191;
        float _shfl_xor_145 = __shfl_xor_sync(0xFFFFFFFF, cur_192, 8);
        float pv_193 = _shfl_xor_145;
        float _fmax_468 = fmaxf(cur_192, pv_193);
        float hi_194_2 = _fmax_468;
        float _min_393 = fminf(cur_192, pv_193);
        float lo_195_2 = _min_393;
        cur_192 = ((up[0] != 0) ? hi_194_2 : lo_195_2);
        float _shfl_xor_146 = __shfl_xor_sync(0xFFFFFFFF, cur_192, 4);
        float pv_196 = _shfl_xor_146;
        float _fmax_469 = fmaxf(cur_192, pv_196);
        float hi_197_1 = _fmax_469;
        float _min_394 = fminf(cur_192, pv_196);
        float lo_198_1 = _min_394;
        cur_192 = ((up[1] != 0) ? hi_197_1 : lo_198_1);
        float _shfl_xor_147 = __shfl_xor_sync(0xFFFFFFFF, cur_192, 2);
        float pv_199 = _shfl_xor_147;
        float _fmax_470 = fmaxf(cur_192, pv_199);
        float hi_200_2 = _fmax_470;
        float _min_395 = fminf(cur_192, pv_199);
        float lo_201_2 = _min_395;
        cur_192 = ((up[2] != 0) ? hi_200_2 : lo_201_2);
        float _shfl_xor_148 = __shfl_xor_sync(0xFFFFFFFF, cur_192, 1);
        float pv_202 = _shfl_xor_148;
        float _fmax_471 = fmaxf(cur_192, pv_202);
        float hi_203_1 = _fmax_471;
        float _min_396 = fminf(cur_192, pv_202);
        float lo_204_1 = _min_396;
        cur_192 = ((up[3] != 0) ? hi_203_1 : lo_204_1);
        V_2[2] = cur_192;
        float _shfl_xor_149 = __shfl_xor_sync(0xFFFFFFFF, V_2[7], 15);
        float y1_205 = _shfl_xor_149;
        float _min_397 = fminf(V_2[3], y1_205);
        float lo1_206 = _min_397;
        float _fmax_472 = fmaxf(r_1_1, lo1_206);
        r_1_1 = _fmax_472;
        float _fmax_473 = fmaxf(V_2[3], y1_205);
        float hi1_207 = _fmax_473;
        float cur_208 = hi1_207;
        float _shfl_xor_150 = __shfl_xor_sync(0xFFFFFFFF, cur_208, 8);
        float pv_209 = _shfl_xor_150;
        float _fmax_474 = fmaxf(cur_208, pv_209);
        float hi_210_2 = _fmax_474;
        float _min_398 = fminf(cur_208, pv_209);
        float lo_211_2 = _min_398;
        cur_208 = ((up[0] != 0) ? hi_210_2 : lo_211_2);
        float _shfl_xor_151 = __shfl_xor_sync(0xFFFFFFFF, cur_208, 4);
        float pv_212 = _shfl_xor_151;
        float _fmax_475 = fmaxf(cur_208, pv_212);
        float hi_213_1 = _fmax_475;
        float _min_399 = fminf(cur_208, pv_212);
        float lo_214_1 = _min_399;
        cur_208 = ((up[1] != 0) ? hi_213_1 : lo_214_1);
        float _shfl_xor_152 = __shfl_xor_sync(0xFFFFFFFF, cur_208, 2);
        float pv_215 = _shfl_xor_152;
        float _fmax_476 = fmaxf(cur_208, pv_215);
        float hi_216_2 = _fmax_476;
        float _min_400 = fminf(cur_208, pv_215);
        float lo_217_2 = _min_400;
        cur_208 = ((up[2] != 0) ? hi_216_2 : lo_217_2);
        float _shfl_xor_153 = __shfl_xor_sync(0xFFFFFFFF, cur_208, 1);
        float pv_218 = _shfl_xor_153;
        float _fmax_477 = fmaxf(cur_208, pv_218);
        float hi_219_1 = _fmax_477;
        float _min_401 = fminf(cur_208, pv_218);
        float lo_220_1 = _min_401;
        cur_208 = ((up[3] != 0) ? hi_219_1 : lo_220_1);
        V_2[3] = cur_208;
        float _shfl_xor_154 = __shfl_xor_sync(0xFFFFFFFF, V_2[2], 15);
        float y1_221 = _shfl_xor_154;
        float _min_402 = fminf(V_2[0], y1_221);
        float lo1_222 = _min_402;
        float _fmax_478 = fmaxf(r_1_1, lo1_222);
        r_1_1 = _fmax_478;
        float _fmax_479 = fmaxf(V_2[0], y1_221);
        float hi1_223 = _fmax_479;
        float cur_224 = hi1_223;
        float _shfl_xor_155 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 8);
        float pv_225 = _shfl_xor_155;
        float _fmax_480 = fmaxf(cur_224, pv_225);
        float hi_226_2 = _fmax_480;
        float _min_403 = fminf(cur_224, pv_225);
        float lo_227_2 = _min_403;
        cur_224 = ((up[0] != 0) ? hi_226_2 : lo_227_2);
        float _shfl_xor_156 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 4);
        float pv_228 = _shfl_xor_156;
        float _fmax_481 = fmaxf(cur_224, pv_228);
        float hi_229_1 = _fmax_481;
        float _min_404 = fminf(cur_224, pv_228);
        float lo_230_1 = _min_404;
        cur_224 = ((up[1] != 0) ? hi_229_1 : lo_230_1);
        float _shfl_xor_157 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 2);
        float pv_231 = _shfl_xor_157;
        float _fmax_482 = fmaxf(cur_224, pv_231);
        float hi_232_2 = _fmax_482;
        float _min_405 = fminf(cur_224, pv_231);
        float lo_233_2 = _min_405;
        cur_224 = ((up[2] != 0) ? hi_232_2 : lo_233_2);
        float _shfl_xor_158 = __shfl_xor_sync(0xFFFFFFFF, cur_224, 1);
        float pv_234 = _shfl_xor_158;
        float _fmax_483 = fmaxf(cur_224, pv_234);
        float hi_235_1 = _fmax_483;
        float _min_406 = fminf(cur_224, pv_234);
        float lo_236_1 = _min_406;
        cur_224 = ((up[3] != 0) ? hi_235_1 : lo_236_1);
        V_2[0] = cur_224;
        float _shfl_xor_159 = __shfl_xor_sync(0xFFFFFFFF, V_2[3], 15);
        float y1_237 = _shfl_xor_159;
        float _min_407 = fminf(V_2[1], y1_237);
        float lo1_238 = _min_407;
        float _fmax_484 = fmaxf(r_1_1, lo1_238);
        r_1_1 = _fmax_484;
        float _fmax_485 = fmaxf(V_2[1], y1_237);
        float hi1_239 = _fmax_485;
        float cur_240 = hi1_239;
        float _shfl_xor_160 = __shfl_xor_sync(0xFFFFFFFF, cur_240, 8);
        float pv_241 = _shfl_xor_160;
        float _fmax_486 = fmaxf(cur_240, pv_241);
        float hi_242_2 = _fmax_486;
        float _min_408 = fminf(cur_240, pv_241);
        float lo_243_2 = _min_408;
        cur_240 = ((up[0] != 0) ? hi_242_2 : lo_243_2);
        float _shfl_xor_161 = __shfl_xor_sync(0xFFFFFFFF, cur_240, 4);
        float pv_244 = _shfl_xor_161;
        float _fmax_487 = fmaxf(cur_240, pv_244);
        float hi_245_1 = _fmax_487;
        float _min_409 = fminf(cur_240, pv_244);
        float lo_246_1 = _min_409;
        cur_240 = ((up[1] != 0) ? hi_245_1 : lo_246_1);
        float _shfl_xor_162 = __shfl_xor_sync(0xFFFFFFFF, cur_240, 2);
        float pv_247 = _shfl_xor_162;
        float _fmax_488 = fmaxf(cur_240, pv_247);
        float hi_248_2 = _fmax_488;
        float _min_410 = fminf(cur_240, pv_247);
        float lo_249_2 = _min_410;
        cur_240 = ((up[2] != 0) ? hi_248_2 : lo_249_2);
        float _shfl_xor_163 = __shfl_xor_sync(0xFFFFFFFF, cur_240, 1);
        float pv_250 = _shfl_xor_163;
        float _fmax_489 = fmaxf(cur_240, pv_250);
        float hi_251 = _fmax_489;
        float _min_411 = fminf(cur_240, pv_250);
        float lo_252 = _min_411;
        cur_240 = ((up[3] != 0) ? hi_251 : lo_252);
        V_2[1] = cur_240;
        float _shfl_xor_164 = __shfl_xor_sync(0xFFFFFFFF, V_2[1], 15);
        float yl_253 = _shfl_xor_164;
        float _min_412 = fminf(V_2[0], yl_253);
        float lol_254 = _min_412;
        float _fmax_490 = fmaxf(r_1_1, lol_254);
        r_1_1 = _fmax_490;
        float _fmax_491 = fmaxf(V_2[0], yl_253);
        float hil_255 = _fmax_491;
        float cur_256 = hil_255;
        float _shfl_xor_165 = __shfl_xor_sync(0xFFFFFFFF, cur_256, 8);
        float pv_257 = _shfl_xor_165;
        float _fmax_492 = fmaxf(cur_256, pv_257);
        float hi_258_1 = _fmax_492;
        float _min_413 = fminf(cur_256, pv_257);
        float lo_259_1 = _min_413;
        cur_256 = ((up[0] != 0) ? hi_258_1 : lo_259_1);
        float _shfl_xor_166 = __shfl_xor_sync(0xFFFFFFFF, cur_256, 4);
        float pv_260 = _shfl_xor_166;
        float _fmax_493 = fmaxf(cur_256, pv_260);
        float hi_261 = _fmax_493;
        float _min_414 = fminf(cur_256, pv_260);
        float lo_262 = _min_414;
        cur_256 = ((up[1] != 0) ? hi_261 : lo_262);
        float _shfl_xor_167 = __shfl_xor_sync(0xFFFFFFFF, cur_256, 2);
        float pv_263 = _shfl_xor_167;
        float _fmax_494 = fmaxf(cur_256, pv_263);
        float hi_264_1 = _fmax_494;
        float _min_415 = fminf(cur_256, pv_263);
        float lo_265_1 = _min_415;
        cur_256 = ((up[2] != 0) ? hi_264_1 : lo_265_1);
        float _shfl_xor_168 = __shfl_xor_sync(0xFFFFFFFF, cur_256, 1);
        float pv_266 = _shfl_xor_168;
        float _fmax_495 = fmaxf(cur_256, pv_266);
        float hi_267 = _fmax_495;
        float _min_416 = fminf(cur_256, pv_266);
        float lo_268 = _min_416;
        cur_256 = ((up[3] != 0) ? hi_267 : lo_268);
        V_2[0] = cur_256;
        float K_269 = V_2[0];
        int qb_270 = (sg + cg) * 32;
        q2[qb_270 + ln] = K_269;
        q2[qb_270 + 16 + ln] = r_1_1;
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        if (tid_1 < 32) {
            float r2_1 = neg_inf;
            float _fmax_496 = fmaxf(r2_1, q2[cg * 32 + 16 + ln]);
            r2_1 = _fmax_496;
            float _fmax_497 = fmaxf(r2_1, q2[(1 + cg) * 32 + 16 + ln]);
            r2_1 = _fmax_497;
            float _fmax_498 = fmaxf(r2_1, q2[(2 + cg) * 32 + 16 + ln]);
            r2_1 = _fmax_498;
            float _fmax_499 = fmaxf(r2_1, q2[(3 + cg) * 32 + 16 + ln]);
            r2_1 = _fmax_499;
            float _fmax_500 = fmaxf(r2_1, q2[(4 + cg) * 32 + 16 + ln]);
            r2_1 = _fmax_500;
            float _fmax_501 = fmaxf(r2_1, q2[(5 + cg) * 32 + 16 + ln]);
            r2_1 = _fmax_501;
            float _fmax_502 = fmaxf(r2_1, q2[(6 + cg) * 32 + 16 + ln]);
            r2_1 = _fmax_502;
            float _fmax_503 = fmaxf(r2_1, q2[(7 + cg) * 32 + 16 + ln]);
            r2_1 = _fmax_503;
            float V2_1[4];
            float x2_1 = q2[cg * 32 + ln];
            float y2_1 = q2[(4 + cg) * 32 + lnr];
            float _min_417 = fminf(x2_1, y2_1);
            float lo2_1 = _min_417;
            float _fmax_504 = fmaxf(r2_1, lo2_1);
            r2_1 = _fmax_504;
            float _fmax_505 = fmaxf(x2_1, y2_1);
            float hi2_1 = _fmax_505;
            float cur_0_1 = hi2_1;
            float _shfl_xor_169 = __shfl_xor_sync(0xFFFFFFFF, cur_0_1, 8);
            float pv_2_1 = _shfl_xor_169;
            float _fmax_506 = fmaxf(cur_0_1, pv_2_1);
            float hi_3_2 = _fmax_506;
            float _min_418 = fminf(cur_0_1, pv_2_1);
            float lo_4_2 = _min_418;
            cur_0_1 = ((up[0] != 0) ? hi_3_2 : lo_4_2);
            float _shfl_xor_170 = __shfl_xor_sync(0xFFFFFFFF, cur_0_1, 4);
            float pv_5_1 = _shfl_xor_170;
            float _fmax_507 = fmaxf(cur_0_1, pv_5_1);
            float hi_6_2 = _fmax_507;
            float _min_419 = fminf(cur_0_1, pv_5_1);
            float lo_7_2 = _min_419;
            cur_0_1 = ((up[1] != 0) ? hi_6_2 : lo_7_2);
            float _shfl_xor_171 = __shfl_xor_sync(0xFFFFFFFF, cur_0_1, 2);
            float pv_8_1 = _shfl_xor_171;
            float _fmax_508 = fmaxf(cur_0_1, pv_8_1);
            float hi_9_2 = _fmax_508;
            float _min_420 = fminf(cur_0_1, pv_8_1);
            float lo_10_2 = _min_420;
            cur_0_1 = ((up[2] != 0) ? hi_9_2 : lo_10_2);
            float _shfl_xor_172 = __shfl_xor_sync(0xFFFFFFFF, cur_0_1, 1);
            float pv_11_1 = _shfl_xor_172;
            float _fmax_509 = fmaxf(cur_0_1, pv_11_1);
            float hi_12_1 = _fmax_509;
            float _min_421 = fminf(cur_0_1, pv_11_1);
            float lo_13_1 = _min_421;
            cur_0_1 = ((up[3] != 0) ? hi_12_1 : lo_13_1);
            V2_1[0] = cur_0_1;
            float x2_14_1 = q2[(1 + cg) * 32 + ln];
            float y2_15_1 = q2[(5 + cg) * 32 + lnr];
            float _min_422 = fminf(x2_14_1, y2_15_1);
            float lo2_16_1 = _min_422;
            float _fmax_510 = fmaxf(r2_1, lo2_16_1);
            r2_1 = _fmax_510;
            float _fmax_511 = fmaxf(x2_14_1, y2_15_1);
            float hi2_17_1 = _fmax_511;
            float cur_18_1 = hi2_17_1;
            float _shfl_xor_173 = __shfl_xor_sync(0xFFFFFFFF, cur_18_1, 8);
            float pv_21 = _shfl_xor_173;
            float _fmax_512 = fmaxf(cur_18_1, pv_21);
            float hi_22_1 = _fmax_512;
            float _min_423 = fminf(cur_18_1, pv_21);
            float lo_23_1 = _min_423;
            cur_18_1 = ((up[0] != 0) ? hi_22_1 : lo_23_1);
            float _shfl_xor_174 = __shfl_xor_sync(0xFFFFFFFF, cur_18_1, 4);
            float pv_24 = _shfl_xor_174;
            float _fmax_513 = fmaxf(cur_18_1, pv_24);
            float hi_25_1 = _fmax_513;
            float _min_424 = fminf(cur_18_1, pv_24);
            float lo_26_1 = _min_424;
            cur_18_1 = ((up[1] != 0) ? hi_25_1 : lo_26_1);
            float _shfl_xor_175 = __shfl_xor_sync(0xFFFFFFFF, cur_18_1, 2);
            float pv_27 = _shfl_xor_175;
            float _fmax_514 = fmaxf(cur_18_1, pv_27);
            float hi_28_1 = _fmax_514;
            float _min_425 = fminf(cur_18_1, pv_27);
            float lo_29_1 = _min_425;
            cur_18_1 = ((up[2] != 0) ? hi_28_1 : lo_29_1);
            float _shfl_xor_176 = __shfl_xor_sync(0xFFFFFFFF, cur_18_1, 1);
            float pv_30 = _shfl_xor_176;
            float _fmax_515 = fmaxf(cur_18_1, pv_30);
            float hi_31 = _fmax_515;
            float _min_426 = fminf(cur_18_1, pv_30);
            float lo_32 = _min_426;
            cur_18_1 = ((up[3] != 0) ? hi_31 : lo_32);
            V2_1[1] = cur_18_1;
            float x2_33 = q2[(2 + cg) * 32 + ln];
            float y2_34 = q2[(6 + cg) * 32 + lnr];
            float _min_427 = fminf(x2_33, y2_34);
            float lo2_35 = _min_427;
            float _fmax_516 = fmaxf(r2_1, lo2_35);
            r2_1 = _fmax_516;
            float _fmax_517 = fmaxf(x2_33, y2_34);
            float hi2_36 = _fmax_517;
            float cur_37 = hi2_36;
            float _shfl_xor_177 = __shfl_xor_sync(0xFFFFFFFF, cur_37, 8);
            float pv_40_1 = _shfl_xor_177;
            float _fmax_518 = fmaxf(cur_37, pv_40_1);
            float hi_41_2 = _fmax_518;
            float _min_428 = fminf(cur_37, pv_40_1);
            float lo_42_2 = _min_428;
            cur_37 = ((up[0] != 0) ? hi_41_2 : lo_42_2);
            float _shfl_xor_178 = __shfl_xor_sync(0xFFFFFFFF, cur_37, 4);
            float pv_43_1 = _shfl_xor_178;
            float _fmax_519 = fmaxf(cur_37, pv_43_1);
            float hi_44_2 = _fmax_519;
            float _min_429 = fminf(cur_37, pv_43_1);
            float lo_45_2 = _min_429;
            cur_37 = ((up[1] != 0) ? hi_44_2 : lo_45_2);
            float _shfl_xor_179 = __shfl_xor_sync(0xFFFFFFFF, cur_37, 2);
            float pv_46_1 = _shfl_xor_179;
            float _fmax_520 = fmaxf(cur_37, pv_46_1);
            float hi_47_2 = _fmax_520;
            float _min_430 = fminf(cur_37, pv_46_1);
            float lo_48_2 = _min_430;
            cur_37 = ((up[2] != 0) ? hi_47_2 : lo_48_2);
            float _shfl_xor_180 = __shfl_xor_sync(0xFFFFFFFF, cur_37, 1);
            float pv_49 = _shfl_xor_180;
            float _fmax_521 = fmaxf(cur_37, pv_49);
            float hi_50 = _fmax_521;
            float _min_431 = fminf(cur_37, pv_49);
            float lo_51 = _min_431;
            cur_37 = ((up[3] != 0) ? hi_50 : lo_51);
            V2_1[2] = cur_37;
            float x2_52 = q2[(3 + cg) * 32 + ln];
            float y2_53 = q2[(7 + cg) * 32 + lnr];
            float _min_432 = fminf(x2_52, y2_53);
            float lo2_54 = _min_432;
            float _fmax_522 = fmaxf(r2_1, lo2_54);
            r2_1 = _fmax_522;
            float _fmax_523 = fmaxf(x2_52, y2_53);
            float hi2_55 = _fmax_523;
            float cur_56 = hi2_55;
            float _shfl_xor_181 = __shfl_xor_sync(0xFFFFFFFF, cur_56, 8);
            float pv_59 = _shfl_xor_181;
            float _fmax_524 = fmaxf(cur_56, pv_59);
            float hi_60_2 = _fmax_524;
            float _min_433 = fminf(cur_56, pv_59);
            float lo_61_2 = _min_433;
            cur_56 = ((up[0] != 0) ? hi_60_2 : lo_61_2);
            float _shfl_xor_182 = __shfl_xor_sync(0xFFFFFFFF, cur_56, 4);
            float pv_62 = _shfl_xor_182;
            float _fmax_525 = fmaxf(cur_56, pv_62);
            float hi_63_1 = _fmax_525;
            float _min_434 = fminf(cur_56, pv_62);
            float lo_64_1 = _min_434;
            cur_56 = ((up[1] != 0) ? hi_63_1 : lo_64_1);
            float _shfl_xor_183 = __shfl_xor_sync(0xFFFFFFFF, cur_56, 2);
            float pv_65 = _shfl_xor_183;
            float _fmax_526 = fmaxf(cur_56, pv_65);
            float hi_66_2 = _fmax_526;
            float _min_435 = fminf(cur_56, pv_65);
            float lo_67_2 = _min_435;
            cur_56 = ((up[2] != 0) ? hi_66_2 : lo_67_2);
            float _shfl_xor_184 = __shfl_xor_sync(0xFFFFFFFF, cur_56, 1);
            float pv_68 = _shfl_xor_184;
            float _fmax_527 = fmaxf(cur_56, pv_68);
            float hi_69 = _fmax_527;
            float _min_436 = fminf(cur_56, pv_68);
            float lo_70 = _min_436;
            cur_56 = ((up[3] != 0) ? hi_69 : lo_70);
            V2_1[3] = cur_56;
            float _shfl_xor_185 = __shfl_xor_sync(0xFFFFFFFF, V2_1[2], 15);
            float y3_1 = _shfl_xor_185;
            float _min_437 = fminf(V2_1[0], y3_1);
            float lo3_1 = _min_437;
            float _fmax_528 = fmaxf(r2_1, lo3_1);
            r2_1 = _fmax_528;
            float _fmax_529 = fmaxf(V2_1[0], y3_1);
            float hi3_1 = _fmax_529;
            float cur_71 = hi3_1;
            float _shfl_xor_186 = __shfl_xor_sync(0xFFFFFFFF, cur_71, 8);
            float pv_72 = _shfl_xor_186;
            float _fmax_530 = fmaxf(cur_71, pv_72);
            float hi_73 = _fmax_530;
            float _min_438 = fminf(cur_71, pv_72);
            float lo_74 = _min_438;
            cur_71 = ((up[0] != 0) ? hi_73 : lo_74);
            float _shfl_xor_187 = __shfl_xor_sync(0xFFFFFFFF, cur_71, 4);
            float pv_75 = _shfl_xor_187;
            float _fmax_531 = fmaxf(cur_71, pv_75);
            float hi_76_2 = _fmax_531;
            float _min_439 = fminf(cur_71, pv_75);
            float lo_77_2 = _min_439;
            cur_71 = ((up[1] != 0) ? hi_76_2 : lo_77_2);
            float _shfl_xor_188 = __shfl_xor_sync(0xFFFFFFFF, cur_71, 2);
            float pv_78 = _shfl_xor_188;
            float _fmax_532 = fmaxf(cur_71, pv_78);
            float hi_79_1 = _fmax_532;
            float _min_440 = fminf(cur_71, pv_78);
            float lo_80_1 = _min_440;
            cur_71 = ((up[2] != 0) ? hi_79_1 : lo_80_1);
            float _shfl_xor_189 = __shfl_xor_sync(0xFFFFFFFF, cur_71, 1);
            float pv_81 = _shfl_xor_189;
            float _fmax_533 = fmaxf(cur_71, pv_81);
            float hi_82_2 = _fmax_533;
            float _min_441 = fminf(cur_71, pv_81);
            float lo_83_2 = _min_441;
            cur_71 = ((up[3] != 0) ? hi_82_2 : lo_83_2);
            V2_1[0] = cur_71;
            float _shfl_xor_190 = __shfl_xor_sync(0xFFFFFFFF, V2_1[3], 15);
            float y3_84 = _shfl_xor_190;
            float _min_442 = fminf(V2_1[1], y3_84);
            float lo3_85 = _min_442;
            float _fmax_534 = fmaxf(r2_1, lo3_85);
            r2_1 = _fmax_534;
            float _fmax_535 = fmaxf(V2_1[1], y3_84);
            float hi3_86 = _fmax_535;
            float cur_87 = hi3_86;
            float _shfl_xor_191 = __shfl_xor_sync(0xFFFFFFFF, cur_87, 8);
            float pv_88 = _shfl_xor_191;
            float _fmax_536 = fmaxf(cur_87, pv_88);
            float hi_89_1 = _fmax_536;
            float _min_443 = fminf(cur_87, pv_88);
            float lo_90_1 = _min_443;
            cur_87 = ((up[0] != 0) ? hi_89_1 : lo_90_1);
            float _shfl_xor_192 = __shfl_xor_sync(0xFFFFFFFF, cur_87, 4);
            float pv_91 = _shfl_xor_192;
            float _fmax_537 = fmaxf(cur_87, pv_91);
            float hi_92_1 = _fmax_537;
            float _min_444 = fminf(cur_87, pv_91);
            float lo_93_1 = _min_444;
            cur_87 = ((up[1] != 0) ? hi_92_1 : lo_93_1);
            float _shfl_xor_193 = __shfl_xor_sync(0xFFFFFFFF, cur_87, 2);
            float pv_94_1 = _shfl_xor_193;
            float _fmax_538 = fmaxf(cur_87, pv_94_1);
            float hi_95_2 = _fmax_538;
            float _min_445 = fminf(cur_87, pv_94_1);
            float lo_96_2 = _min_445;
            cur_87 = ((up[2] != 0) ? hi_95_2 : lo_96_2);
            float _shfl_xor_194 = __shfl_xor_sync(0xFFFFFFFF, cur_87, 1);
            float pv_97 = _shfl_xor_194;
            float _fmax_539 = fmaxf(cur_87, pv_97);
            float hi_98_2 = _fmax_539;
            float _min_446 = fminf(cur_87, pv_97);
            float lo_99_2 = _min_446;
            cur_87 = ((up[3] != 0) ? hi_98_2 : lo_99_2);
            V2_1[1] = cur_87;
            float _shfl_xor_195 = __shfl_xor_sync(0xFFFFFFFF, V2_1[1], 15);
            float yl2_1 = _shfl_xor_195;
            float _min_447 = fminf(V2_1[0], yl2_1);
            float lol2_1 = _min_447;
            float _fmax_540 = fmaxf(r2_1, lol2_1);
            r2_1 = _fmax_540;
            float _fmax_541 = fmaxf(V2_1[0], yl2_1);
            V2_1[0] = _fmax_541;
            K_269 = V2_1[0];
            r_1_1 = r2_1;
        }
        rr2[0] = r_1_1;
        K2 = K_269;
    }
    if (tid_1 < 32) {
        unsigned int kk = __as_u32(K);
        unsigned int idx = 2048;
        if (kk < 4278190080u) {
            idx = kk & 2047;
        }
        if (gflag != 0) {
            unsigned int k2 = __as_u32(K2);
            idx = 2048;
            if (k2 != 0) {
                idx = k2 & 2047;
            }
        }
        unsigned int rkey = idx << 4 | (unsigned int)ln;
        int rank = 0;
        unsigned int _shfl_xor_196 = __shfl_xor_sync(0xFFFFFFFF, rkey, 1);
        unsigned int ox = _shfl_xor_196;
        if (ox < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_197 = __shfl_xor_sync(0xFFFFFFFF, rkey, 2);
        unsigned int ox_0 = _shfl_xor_197;
        if (ox_0 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_198 = __shfl_xor_sync(0xFFFFFFFF, rkey, 3);
        unsigned int ox_1 = _shfl_xor_198;
        if (ox_1 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_199 = __shfl_xor_sync(0xFFFFFFFF, rkey, 4);
        unsigned int ox_2 = _shfl_xor_199;
        if (ox_2 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_200 = __shfl_xor_sync(0xFFFFFFFF, rkey, 5);
        unsigned int ox_3 = _shfl_xor_200;
        if (ox_3 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_201 = __shfl_xor_sync(0xFFFFFFFF, rkey, 6);
        unsigned int ox_4 = _shfl_xor_201;
        if (ox_4 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_202 = __shfl_xor_sync(0xFFFFFFFF, rkey, 7);
        unsigned int ox_5 = _shfl_xor_202;
        if (ox_5 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_203 = __shfl_xor_sync(0xFFFFFFFF, rkey, 8);
        unsigned int ox_6 = _shfl_xor_203;
        if (ox_6 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_204 = __shfl_xor_sync(0xFFFFFFFF, rkey, 9);
        unsigned int ox_7 = _shfl_xor_204;
        if (ox_7 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_205 = __shfl_xor_sync(0xFFFFFFFF, rkey, 10);
        unsigned int ox_8 = _shfl_xor_205;
        if (ox_8 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_206 = __shfl_xor_sync(0xFFFFFFFF, rkey, 11);
        unsigned int ox_9 = _shfl_xor_206;
        if (ox_9 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_207 = __shfl_xor_sync(0xFFFFFFFF, rkey, 12);
        unsigned int ox_10 = _shfl_xor_207;
        if (ox_10 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_208 = __shfl_xor_sync(0xFFFFFFFF, rkey, 13);
        unsigned int ox_11 = _shfl_xor_208;
        if (ox_11 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_209 = __shfl_xor_sync(0xFFFFFFFF, rkey, 14);
        unsigned int ox_12 = _shfl_xor_209;
        if (ox_12 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_210 = __shfl_xor_sync(0xFFFFFFFF, rkey, 15);
        unsigned int ox_13 = _shfl_xor_210;
        if (ox_13 < rkey) {
            rank = rank + 1;
        }
        int val = -1;
        if (idx <= 2047) {
            val = (int)idx;
        }
        if (commit != 0) {
            long long obase = ((long long)ucol * (long long)num_heads + (long long)head) * 16;
            out[obase + (long long)rank] = val;
        }
    }
}

} // extern "C"
