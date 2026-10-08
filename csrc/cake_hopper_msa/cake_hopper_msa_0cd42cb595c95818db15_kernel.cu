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
#define SMEM_QCOL_STAGE_BYTES 32
#define SMEM_QCOL_STRIDE 32
#define SMEM_FLAGW_OFF 32
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_PUB_OFF 48
#define SMEM_PUB_STAGE_BYTES 17408
#define SMEM_PUB_STRIDE 17408
#define SMEM_Q2_OFF 17456
#define SMEM_Q2_STAGE_BYTES 2048
#define SMEM_Q2_STRIDE 2048
#define SMEM_TOTAL 19584
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
kernel_cake_hopper_msa_0cd42cb595c95818db15(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 32);
    const int flagw_addr = smem + 32;
    float* pub = reinterpret_cast<float*>(smem_raw + 48);
    const int pub_addr = smem + 48;
    float* q2 = reinterpret_cast<float*>(smem_raw + 17456);
    const int q2_addr = smem + 17456;

    // === Task calls (dependency order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1 / 8;
    int c = tid_1 - w * 8;
    int whi = w / 4;
    int col = blockIdx.x * 8 + c;
    int head = blockIdx.y;
    if (tid_1 == 0) {
        flagw[0] = 0;
    }
    int lim = tiles;
    if (col >= total_q) {
        lim = 0;
    }
    int lim_full = lim;
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
    if (lim_full > t0_0) {
        cb[0] = S[p];
    }
    cb[1] = 4286578688;
    if (lim_full > t0_0 + 1) {
        cb[1] = S[p + nq64];
    }
    cb[2] = 4286578688;
    if (lim_full > t0_0 + 2) {
        cb[2] = S[p + 2 * nq64];
    }
    cb[3] = 4286578688;
    if (lim_full > t0_0 + 3) {
        cb[3] = S[p + 3 * nq64];
    }
    cb[4] = 4286578688;
    if (lim_full > t0_0 + 4) {
        cb[4] = S[p + 4 * nq64];
    }
    cb[5] = 4286578688;
    if (lim_full > t0_0 + 5) {
        cb[5] = S[p + 5 * nq64];
    }
    cb[6] = 4286578688;
    if (lim_full > t0_0 + 6) {
        cb[6] = S[p + 6 * nq64];
    }
    cb[7] = 4286578688;
    if (lim_full > t0_0 + 7) {
        cb[7] = S[p + 7 * nq64];
    }
    cb[8] = 4286578688;
    if (lim_full > t0_0 + 8) {
        cb[8] = S[p + 8 * nq64];
    }
    cb[9] = 4286578688;
    if (lim_full > t0_0 + 9) {
        cb[9] = S[p + 9 * nq64];
    }
    cb[10] = 4286578688;
    if (lim_full > t0_0 + 10) {
        cb[10] = S[p + 10 * nq64];
    }
    cb[11] = 4286578688;
    if (lim_full > t0_0 + 11) {
        cb[11] = S[p + 11 * nq64];
    }
    cb[12] = 4286578688;
    if (lim_full > t0_0 + 12) {
        cb[12] = S[p + 12 * nq64];
    }
    cb[13] = 4286578688;
    if (lim_full > t0_0 + 13) {
        cb[13] = S[p + 13 * nq64];
    }
    cb[14] = 4286578688;
    if (lim_full > t0_0 + 14) {
        cb[14] = S[p + 14 * nq64];
    }
    cb[15] = 4286578688;
    if (lim_full > t0_0 + 15) {
        cb[15] = S[p + 15 * nq64];
    }
    asm volatile("" ::: "memory");
    int t0_1 = w * 16;
    int t00 = t0_1;
    if (lim <= t00) {
        cb[0] = 4286578688;
    }
    if (lim <= t00 + 1) {
        cb[1] = 4286578688;
    }
    if (lim <= t00 + 2) {
        cb[2] = 4286578688;
    }
    if (lim <= t00 + 3) {
        cb[3] = 4286578688;
    }
    if (lim <= t00 + 4) {
        cb[4] = 4286578688;
    }
    if (lim <= t00 + 5) {
        cb[5] = 4286578688;
    }
    if (lim <= t00 + 6) {
        cb[6] = 4286578688;
    }
    if (lim <= t00 + 7) {
        cb[7] = 4286578688;
    }
    if (lim <= t00 + 8) {
        cb[8] = 4286578688;
    }
    if (lim <= t00 + 9) {
        cb[9] = 4286578688;
    }
    if (lim <= t00 + 10) {
        cb[10] = 4286578688;
    }
    if (lim <= t00 + 11) {
        cb[11] = 4286578688;
    }
    if (lim <= t00 + 12) {
        cb[12] = 4286578688;
    }
    if (lim <= t00 + 13) {
        cb[13] = 4286578688;
    }
    if (lim <= t00 + 14) {
        cb[14] = 4286578688;
    }
    if (lim <= t00 + 15) {
        cb[15] = 4286578688;
    }
    int t0_2 = w * 16;
    int t0_3 = t0_2;
    float sc = __uint_as_float(cb[0]);
    float _fmax_0 = fmaxf(sc, -1.7014118346046923e+38f);
    sc = _fmax_0;
    float _min_0 = fminf(sc, 1.7014118346046923e+38f);
    sc = _min_0;
    sc = sc;
    float sc_4 = sc;
    unsigned int key = __as_u32(sc_4) & 4294966784u | (unsigned int)t0_3;
    kb[0] = __uint_as_float(key);
    float sc_5 = __uint_as_float(cb[1]);
    float _fmax_1 = fmaxf(sc_5, -1.7014118346046923e+38f);
    sc_5 = _fmax_1;
    float _min_1 = fminf(sc_5, 1.7014118346046923e+38f);
    sc_5 = _min_1;
    sc_5 = sc_5;
    float sc_6 = sc_5;
    unsigned int key_7 = __as_u32(sc_6) & 4294966784u | (unsigned int)(t0_3 + 1);
    kb[1] = __uint_as_float(key_7);
    float sc_8 = __uint_as_float(cb[2]);
    float _fmax_2 = fmaxf(sc_8, -1.7014118346046923e+38f);
    sc_8 = _fmax_2;
    float _min_2 = fminf(sc_8, 1.7014118346046923e+38f);
    sc_8 = _min_2;
    sc_8 = sc_8;
    float sc_9 = sc_8;
    unsigned int key_10 = __as_u32(sc_9) & 4294966784u | (unsigned int)(t0_3 + 2);
    kb[2] = __uint_as_float(key_10);
    float sc_11 = __uint_as_float(cb[3]);
    float _fmax_3 = fmaxf(sc_11, -1.7014118346046923e+38f);
    sc_11 = _fmax_3;
    float _min_3 = fminf(sc_11, 1.7014118346046923e+38f);
    sc_11 = _min_3;
    sc_11 = sc_11;
    float sc_12 = sc_11;
    unsigned int key_13 = __as_u32(sc_12) & 4294966784u | (unsigned int)(t0_3 + 3);
    kb[3] = __uint_as_float(key_13);
    float sc_14 = __uint_as_float(cb[4]);
    float _fmax_4 = fmaxf(sc_14, -1.7014118346046923e+38f);
    sc_14 = _fmax_4;
    float _min_4 = fminf(sc_14, 1.7014118346046923e+38f);
    sc_14 = _min_4;
    sc_14 = sc_14;
    float sc_15 = sc_14;
    unsigned int key_16 = __as_u32(sc_15) & 4294966784u | (unsigned int)(t0_3 + 4);
    kb[4] = __uint_as_float(key_16);
    float sc_17 = __uint_as_float(cb[5]);
    float _fmax_5 = fmaxf(sc_17, -1.7014118346046923e+38f);
    sc_17 = _fmax_5;
    float _min_5 = fminf(sc_17, 1.7014118346046923e+38f);
    sc_17 = _min_5;
    sc_17 = sc_17;
    float sc_18 = sc_17;
    unsigned int key_19 = __as_u32(sc_18) & 4294966784u | (unsigned int)(t0_3 + 5);
    kb[5] = __uint_as_float(key_19);
    float sc_20 = __uint_as_float(cb[6]);
    float _fmax_6 = fmaxf(sc_20, -1.7014118346046923e+38f);
    sc_20 = _fmax_6;
    float _min_6 = fminf(sc_20, 1.7014118346046923e+38f);
    sc_20 = _min_6;
    sc_20 = sc_20;
    float sc_21 = sc_20;
    unsigned int key_22 = __as_u32(sc_21) & 4294966784u | (unsigned int)(t0_3 + 6);
    kb[6] = __uint_as_float(key_22);
    float sc_23 = __uint_as_float(cb[7]);
    float _fmax_7 = fmaxf(sc_23, -1.7014118346046923e+38f);
    sc_23 = _fmax_7;
    float _min_7 = fminf(sc_23, 1.7014118346046923e+38f);
    sc_23 = _min_7;
    sc_23 = sc_23;
    float sc_24 = sc_23;
    unsigned int key_25 = __as_u32(sc_24) & 4294966784u | (unsigned int)(t0_3 + 7);
    kb[7] = __uint_as_float(key_25);
    float sc_26 = __uint_as_float(cb[8]);
    float _fmax_8 = fmaxf(sc_26, -1.7014118346046923e+38f);
    sc_26 = _fmax_8;
    float _min_8 = fminf(sc_26, 1.7014118346046923e+38f);
    sc_26 = _min_8;
    sc_26 = sc_26;
    float sc_27 = sc_26;
    unsigned int key_28 = __as_u32(sc_27) & 4294966784u | (unsigned int)(t0_3 + 8);
    kb[8] = __uint_as_float(key_28);
    float sc_29 = __uint_as_float(cb[9]);
    float _fmax_9 = fmaxf(sc_29, -1.7014118346046923e+38f);
    sc_29 = _fmax_9;
    float _min_9 = fminf(sc_29, 1.7014118346046923e+38f);
    sc_29 = _min_9;
    sc_29 = sc_29;
    float sc_30 = sc_29;
    unsigned int key_31 = __as_u32(sc_30) & 4294966784u | (unsigned int)(t0_3 + 9);
    kb[9] = __uint_as_float(key_31);
    float sc_32 = __uint_as_float(cb[10]);
    float _fmax_10 = fmaxf(sc_32, -1.7014118346046923e+38f);
    sc_32 = _fmax_10;
    float _min_10 = fminf(sc_32, 1.7014118346046923e+38f);
    sc_32 = _min_10;
    sc_32 = sc_32;
    float sc_33 = sc_32;
    unsigned int key_34 = __as_u32(sc_33) & 4294966784u | (unsigned int)(t0_3 + 10);
    kb[10] = __uint_as_float(key_34);
    float sc_35 = __uint_as_float(cb[11]);
    float _fmax_11 = fmaxf(sc_35, -1.7014118346046923e+38f);
    sc_35 = _fmax_11;
    float _min_11 = fminf(sc_35, 1.7014118346046923e+38f);
    sc_35 = _min_11;
    sc_35 = sc_35;
    float sc_36 = sc_35;
    unsigned int key_37 = __as_u32(sc_36) & 4294966784u | (unsigned int)(t0_3 + 11);
    kb[11] = __uint_as_float(key_37);
    float sc_38 = __uint_as_float(cb[12]);
    float _fmax_12 = fmaxf(sc_38, -1.7014118346046923e+38f);
    sc_38 = _fmax_12;
    float _min_12 = fminf(sc_38, 1.7014118346046923e+38f);
    sc_38 = _min_12;
    sc_38 = sc_38;
    float sc_39 = sc_38;
    unsigned int key_40 = __as_u32(sc_39) & 4294966784u | (unsigned int)(t0_3 + 12);
    kb[12] = __uint_as_float(key_40);
    float sc_41 = __uint_as_float(cb[13]);
    float _fmax_13 = fmaxf(sc_41, -1.7014118346046923e+38f);
    sc_41 = _fmax_13;
    float _min_13 = fminf(sc_41, 1.7014118346046923e+38f);
    sc_41 = _min_13;
    sc_41 = sc_41;
    float sc_42 = sc_41;
    unsigned int key_43 = __as_u32(sc_42) & 4294966784u | (unsigned int)(t0_3 + 13);
    kb[13] = __uint_as_float(key_43);
    float sc_44 = __uint_as_float(cb[14]);
    float _fmax_14 = fmaxf(sc_44, -1.7014118346046923e+38f);
    sc_44 = _fmax_14;
    float _min_14 = fminf(sc_44, 1.7014118346046923e+38f);
    sc_44 = _min_14;
    sc_44 = sc_44;
    float sc_45 = sc_44;
    unsigned int key_46 = __as_u32(sc_45) & 4294966784u | (unsigned int)(t0_3 + 14);
    kb[14] = __uint_as_float(key_46);
    float sc_47 = __uint_as_float(cb[15]);
    float _fmax_15 = fmaxf(sc_47, -1.7014118346046923e+38f);
    sc_47 = _fmax_15;
    float _min_15 = fminf(sc_47, 1.7014118346046923e+38f);
    sc_47 = _min_15;
    sc_47 = sc_47;
    float sc_48 = sc_47;
    unsigned int key_49 = __as_u32(sc_48) & 4294966784u | (unsigned int)(t0_3 + 15);
    kb[15] = __uint_as_float(key_49);
    int f = 0;
    if (t0_3 < fb || t0_3 + 16 > lim - fe && t0_3 < lim) {
        f = 1;
    }
    int fch1 = f;
    if (fch1 != 0) {
        int f_0 = 0;
        if (t0_3 < fb || t0_3 >= lim - fe && lim > t0_3) {
            f_0 = 1;
        }
        if (f_0 != 0) {
            kb[0] = __uint_as_float(2139094528 | (unsigned int)t0_3);
        }
        int f_1 = 0;
        if (t0_3 + 1 < fb || t0_3 + 1 >= lim - fe && lim > t0_3 + 1) {
            f_1 = 1;
        }
        if (f_1 != 0) {
            kb[1] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 1));
        }
        int f_2 = 0;
        if (t0_3 + 2 < fb || t0_3 + 2 >= lim - fe && lim > t0_3 + 2) {
            f_2 = 1;
        }
        if (f_2 != 0) {
            kb[2] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 2));
        }
        int f_3 = 0;
        if (t0_3 + 3 < fb || t0_3 + 3 >= lim - fe && lim > t0_3 + 3) {
            f_3 = 1;
        }
        if (f_3 != 0) {
            kb[3] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 3));
        }
        int f_4 = 0;
        if (t0_3 + 4 < fb || t0_3 + 4 >= lim - fe && lim > t0_3 + 4) {
            f_4 = 1;
        }
        if (f_4 != 0) {
            kb[4] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 4));
        }
        int f_5 = 0;
        if (t0_3 + 5 < fb || t0_3 + 5 >= lim - fe && lim > t0_3 + 5) {
            f_5 = 1;
        }
        if (f_5 != 0) {
            kb[5] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 5));
        }
        int f_6 = 0;
        if (t0_3 + 6 < fb || t0_3 + 6 >= lim - fe && lim > t0_3 + 6) {
            f_6 = 1;
        }
        if (f_6 != 0) {
            kb[6] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 6));
        }
        int f_7 = 0;
        if (t0_3 + 7 < fb || t0_3 + 7 >= lim - fe && lim > t0_3 + 7) {
            f_7 = 1;
        }
        if (f_7 != 0) {
            kb[7] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 7));
        }
        int f_8 = 0;
        if (t0_3 + 8 < fb || t0_3 + 8 >= lim - fe && lim > t0_3 + 8) {
            f_8 = 1;
        }
        if (f_8 != 0) {
            kb[8] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 8));
        }
        int f_9 = 0;
        if (t0_3 + 9 < fb || t0_3 + 9 >= lim - fe && lim > t0_3 + 9) {
            f_9 = 1;
        }
        if (f_9 != 0) {
            kb[9] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 9));
        }
        int f_10 = 0;
        if (t0_3 + 10 < fb || t0_3 + 10 >= lim - fe && lim > t0_3 + 10) {
            f_10 = 1;
        }
        if (f_10 != 0) {
            kb[10] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 10));
        }
        int f_11 = 0;
        if (t0_3 + 11 < fb || t0_3 + 11 >= lim - fe && lim > t0_3 + 11) {
            f_11 = 1;
        }
        if (f_11 != 0) {
            kb[11] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 11));
        }
        int f_12 = 0;
        if (t0_3 + 12 < fb || t0_3 + 12 >= lim - fe && lim > t0_3 + 12) {
            f_12 = 1;
        }
        if (f_12 != 0) {
            kb[12] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 12));
        }
        int f_13 = 0;
        if (t0_3 + 13 < fb || t0_3 + 13 >= lim - fe && lim > t0_3 + 13) {
            f_13 = 1;
        }
        if (f_13 != 0) {
            kb[13] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 13));
        }
        int f_14 = 0;
        if (t0_3 + 14 < fb || t0_3 + 14 >= lim - fe && lim > t0_3 + 14) {
            f_14 = 1;
        }
        if (f_14 != 0) {
            kb[14] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 14));
        }
        int f_15 = 0;
        if (t0_3 + 15 < fb || t0_3 + 15 >= lim - fe && lim > t0_3 + 15) {
            f_15 = 1;
        }
        if (f_15 != 0) {
            kb[15] = __uint_as_float(2139094528 | (unsigned int)(t0_3 + 15));
        }
    }
    float _fmax_16 = fmaxf(kb[0], kb[13]);
    float hi = _fmax_16;
    float _min_16 = fminf(kb[0], kb[13]);
    float lo = _min_16;
    kb[0] = hi;
    kb[13] = lo;
    float _fmax_17 = fmaxf(kb[1], kb[12]);
    float hi_50 = _fmax_17;
    float _min_17 = fminf(kb[1], kb[12]);
    float lo_51 = _min_17;
    kb[1] = hi_50;
    kb[12] = lo_51;
    float _fmax_18 = fmaxf(kb[2], kb[15]);
    float hi_52 = _fmax_18;
    float _min_18 = fminf(kb[2], kb[15]);
    float lo_53 = _min_18;
    kb[2] = hi_52;
    kb[15] = lo_53;
    float _fmax_19 = fmaxf(kb[3], kb[14]);
    float hi_54 = _fmax_19;
    float _min_19 = fminf(kb[3], kb[14]);
    float lo_55 = _min_19;
    kb[3] = hi_54;
    kb[14] = lo_55;
    float _fmax_20 = fmaxf(kb[4], kb[8]);
    float hi_56 = _fmax_20;
    float _min_20 = fminf(kb[4], kb[8]);
    float lo_57 = _min_20;
    kb[4] = hi_56;
    kb[8] = lo_57;
    float _fmax_21 = fmaxf(kb[5], kb[6]);
    float hi_58 = _fmax_21;
    float _min_21 = fminf(kb[5], kb[6]);
    float lo_59 = _min_21;
    kb[5] = hi_58;
    kb[6] = lo_59;
    float _fmax_22 = fmaxf(kb[7], kb[11]);
    float hi_60 = _fmax_22;
    float _min_22 = fminf(kb[7], kb[11]);
    float lo_61 = _min_22;
    kb[7] = hi_60;
    kb[11] = lo_61;
    float _fmax_23 = fmaxf(kb[9], kb[10]);
    float hi_62 = _fmax_23;
    float _min_23 = fminf(kb[9], kb[10]);
    float lo_63 = _min_23;
    kb[9] = hi_62;
    kb[10] = lo_63;
    float _fmax_24 = fmaxf(kb[0], kb[5]);
    float hi_64 = _fmax_24;
    float _min_24 = fminf(kb[0], kb[5]);
    float lo_65 = _min_24;
    kb[0] = hi_64;
    kb[5] = lo_65;
    float _fmax_25 = fmaxf(kb[1], kb[7]);
    float hi_66 = _fmax_25;
    float _min_25 = fminf(kb[1], kb[7]);
    float lo_67 = _min_25;
    kb[1] = hi_66;
    kb[7] = lo_67;
    float _fmax_26 = fmaxf(kb[2], kb[9]);
    float hi_68 = _fmax_26;
    float _min_26 = fminf(kb[2], kb[9]);
    float lo_69 = _min_26;
    kb[2] = hi_68;
    kb[9] = lo_69;
    float _fmax_27 = fmaxf(kb[3], kb[4]);
    float hi_70 = _fmax_27;
    float _min_27 = fminf(kb[3], kb[4]);
    float lo_71 = _min_27;
    kb[3] = hi_70;
    kb[4] = lo_71;
    float _fmax_28 = fmaxf(kb[6], kb[13]);
    float hi_72 = _fmax_28;
    float _min_28 = fminf(kb[6], kb[13]);
    float lo_73 = _min_28;
    kb[6] = hi_72;
    kb[13] = lo_73;
    float _fmax_29 = fmaxf(kb[8], kb[14]);
    float hi_74 = _fmax_29;
    float _min_29 = fminf(kb[8], kb[14]);
    float lo_75 = _min_29;
    kb[8] = hi_74;
    kb[14] = lo_75;
    float _fmax_30 = fmaxf(kb[10], kb[15]);
    float hi_76 = _fmax_30;
    float _min_30 = fminf(kb[10], kb[15]);
    float lo_77 = _min_30;
    kb[10] = hi_76;
    kb[15] = lo_77;
    float _fmax_31 = fmaxf(kb[11], kb[12]);
    float hi_78 = _fmax_31;
    float _min_31 = fminf(kb[11], kb[12]);
    float lo_79 = _min_31;
    kb[11] = hi_78;
    kb[12] = lo_79;
    float _fmax_32 = fmaxf(kb[0], kb[1]);
    float hi_80 = _fmax_32;
    float _min_32 = fminf(kb[0], kb[1]);
    float lo_81 = _min_32;
    kb[0] = hi_80;
    kb[1] = lo_81;
    float _fmax_33 = fmaxf(kb[2], kb[3]);
    float hi_82 = _fmax_33;
    float _min_33 = fminf(kb[2], kb[3]);
    float lo_83 = _min_33;
    kb[2] = hi_82;
    kb[3] = lo_83;
    float _fmax_34 = fmaxf(kb[4], kb[5]);
    float hi_84 = _fmax_34;
    float _min_34 = fminf(kb[4], kb[5]);
    float lo_85 = _min_34;
    kb[4] = hi_84;
    kb[5] = lo_85;
    float _fmax_35 = fmaxf(kb[6], kb[8]);
    float hi_86 = _fmax_35;
    float _min_35 = fminf(kb[6], kb[8]);
    float lo_87 = _min_35;
    kb[6] = hi_86;
    kb[8] = lo_87;
    float _fmax_36 = fmaxf(kb[7], kb[9]);
    float hi_88 = _fmax_36;
    float _min_36 = fminf(kb[7], kb[9]);
    float lo_89 = _min_36;
    kb[7] = hi_88;
    kb[9] = lo_89;
    float _fmax_37 = fmaxf(kb[10], kb[11]);
    float hi_90 = _fmax_37;
    float _min_37 = fminf(kb[10], kb[11]);
    float lo_91 = _min_37;
    kb[10] = hi_90;
    kb[11] = lo_91;
    float _fmax_38 = fmaxf(kb[12], kb[13]);
    float hi_92 = _fmax_38;
    float _min_38 = fminf(kb[12], kb[13]);
    float lo_93 = _min_38;
    kb[12] = hi_92;
    kb[13] = lo_93;
    float _fmax_39 = fmaxf(kb[14], kb[15]);
    float hi_94 = _fmax_39;
    float _min_39 = fminf(kb[14], kb[15]);
    float lo_95 = _min_39;
    kb[14] = hi_94;
    kb[15] = lo_95;
    float _fmax_40 = fmaxf(kb[0], kb[2]);
    float hi_96 = _fmax_40;
    float _min_40 = fminf(kb[0], kb[2]);
    float lo_97 = _min_40;
    kb[0] = hi_96;
    kb[2] = lo_97;
    float _fmax_41 = fmaxf(kb[1], kb[3]);
    float hi_98 = _fmax_41;
    float _min_41 = fminf(kb[1], kb[3]);
    float lo_99 = _min_41;
    kb[1] = hi_98;
    kb[3] = lo_99;
    float _fmax_42 = fmaxf(kb[4], kb[10]);
    float hi_100 = _fmax_42;
    float _min_42 = fminf(kb[4], kb[10]);
    float lo_101 = _min_42;
    kb[4] = hi_100;
    kb[10] = lo_101;
    float _fmax_43 = fmaxf(kb[5], kb[11]);
    float hi_102 = _fmax_43;
    float _min_43 = fminf(kb[5], kb[11]);
    float lo_103 = _min_43;
    kb[5] = hi_102;
    kb[11] = lo_103;
    float _fmax_44 = fmaxf(kb[6], kb[7]);
    float hi_104 = _fmax_44;
    float _min_44 = fminf(kb[6], kb[7]);
    float lo_105 = _min_44;
    kb[6] = hi_104;
    kb[7] = lo_105;
    float _fmax_45 = fmaxf(kb[8], kb[9]);
    float hi_106 = _fmax_45;
    float _min_45 = fminf(kb[8], kb[9]);
    float lo_107 = _min_45;
    kb[8] = hi_106;
    kb[9] = lo_107;
    float _fmax_46 = fmaxf(kb[12], kb[14]);
    float hi_108 = _fmax_46;
    float _min_46 = fminf(kb[12], kb[14]);
    float lo_109 = _min_46;
    kb[12] = hi_108;
    kb[14] = lo_109;
    float _fmax_47 = fmaxf(kb[13], kb[15]);
    float hi_110 = _fmax_47;
    float _min_47 = fminf(kb[13], kb[15]);
    float lo_111 = _min_47;
    kb[13] = hi_110;
    kb[15] = lo_111;
    float _fmax_48 = fmaxf(kb[1], kb[2]);
    float hi_112 = _fmax_48;
    float _min_48 = fminf(kb[1], kb[2]);
    float lo_113 = _min_48;
    kb[1] = hi_112;
    kb[2] = lo_113;
    float _fmax_49 = fmaxf(kb[3], kb[12]);
    float hi_114 = _fmax_49;
    float _min_49 = fminf(kb[3], kb[12]);
    float lo_115 = _min_49;
    kb[3] = hi_114;
    kb[12] = lo_115;
    float _fmax_50 = fmaxf(kb[4], kb[6]);
    float hi_116 = _fmax_50;
    float _min_50 = fminf(kb[4], kb[6]);
    float lo_117 = _min_50;
    kb[4] = hi_116;
    kb[6] = lo_117;
    float _fmax_51 = fmaxf(kb[5], kb[7]);
    float hi_118 = _fmax_51;
    float _min_51 = fminf(kb[5], kb[7]);
    float lo_119 = _min_51;
    kb[5] = hi_118;
    kb[7] = lo_119;
    float _fmax_52 = fmaxf(kb[8], kb[10]);
    float hi_120 = _fmax_52;
    float _min_52 = fminf(kb[8], kb[10]);
    float lo_121 = _min_52;
    kb[8] = hi_120;
    kb[10] = lo_121;
    float _fmax_53 = fmaxf(kb[9], kb[11]);
    float hi_122 = _fmax_53;
    float _min_53 = fminf(kb[9], kb[11]);
    float lo_123 = _min_53;
    kb[9] = hi_122;
    kb[11] = lo_123;
    float _fmax_54 = fmaxf(kb[13], kb[14]);
    float hi_124 = _fmax_54;
    float _min_54 = fminf(kb[13], kb[14]);
    float lo_125 = _min_54;
    kb[13] = hi_124;
    kb[14] = lo_125;
    float _fmax_55 = fmaxf(kb[1], kb[4]);
    float hi_126 = _fmax_55;
    float _min_55 = fminf(kb[1], kb[4]);
    float lo_127 = _min_55;
    kb[1] = hi_126;
    kb[4] = lo_127;
    float _fmax_56 = fmaxf(kb[2], kb[6]);
    float hi_128 = _fmax_56;
    float _min_56 = fminf(kb[2], kb[6]);
    float lo_129 = _min_56;
    kb[2] = hi_128;
    kb[6] = lo_129;
    float _fmax_57 = fmaxf(kb[5], kb[8]);
    float hi_130 = _fmax_57;
    float _min_57 = fminf(kb[5], kb[8]);
    float lo_131 = _min_57;
    kb[5] = hi_130;
    kb[8] = lo_131;
    float _fmax_58 = fmaxf(kb[7], kb[10]);
    float hi_132 = _fmax_58;
    float _min_58 = fminf(kb[7], kb[10]);
    float lo_133 = _min_58;
    kb[7] = hi_132;
    kb[10] = lo_133;
    float _fmax_59 = fmaxf(kb[9], kb[13]);
    float hi_134 = _fmax_59;
    float _min_59 = fminf(kb[9], kb[13]);
    float lo_135 = _min_59;
    kb[9] = hi_134;
    kb[13] = lo_135;
    float _fmax_60 = fmaxf(kb[11], kb[14]);
    float hi_136 = _fmax_60;
    float _min_60 = fminf(kb[11], kb[14]);
    float lo_137 = _min_60;
    kb[11] = hi_136;
    kb[14] = lo_137;
    float _fmax_61 = fmaxf(kb[2], kb[4]);
    float hi_138 = _fmax_61;
    float _min_61 = fminf(kb[2], kb[4]);
    float lo_139 = _min_61;
    kb[2] = hi_138;
    kb[4] = lo_139;
    float _fmax_62 = fmaxf(kb[3], kb[6]);
    float hi_140 = _fmax_62;
    float _min_62 = fminf(kb[3], kb[6]);
    float lo_141 = _min_62;
    kb[3] = hi_140;
    kb[6] = lo_141;
    float _fmax_63 = fmaxf(kb[9], kb[12]);
    float hi_142 = _fmax_63;
    float _min_63 = fminf(kb[9], kb[12]);
    float lo_143 = _min_63;
    kb[9] = hi_142;
    kb[12] = lo_143;
    float _fmax_64 = fmaxf(kb[11], kb[13]);
    float hi_144 = _fmax_64;
    float _min_64 = fminf(kb[11], kb[13]);
    float lo_145 = _min_64;
    kb[11] = hi_144;
    kb[13] = lo_145;
    float _fmax_65 = fmaxf(kb[3], kb[5]);
    float hi_146 = _fmax_65;
    float _min_65 = fminf(kb[3], kb[5]);
    float lo_147 = _min_65;
    kb[3] = hi_146;
    kb[5] = lo_147;
    float _fmax_66 = fmaxf(kb[6], kb[8]);
    float hi_148 = _fmax_66;
    float _min_66 = fminf(kb[6], kb[8]);
    float lo_149 = _min_66;
    kb[6] = hi_148;
    kb[8] = lo_149;
    float _fmax_67 = fmaxf(kb[7], kb[9]);
    float hi_150 = _fmax_67;
    float _min_67 = fminf(kb[7], kb[9]);
    float lo_151 = _min_67;
    kb[7] = hi_150;
    kb[9] = lo_151;
    float _fmax_68 = fmaxf(kb[10], kb[12]);
    float hi_152 = _fmax_68;
    float _min_68 = fminf(kb[10], kb[12]);
    float lo_153 = _min_68;
    kb[10] = hi_152;
    kb[12] = lo_153;
    float _fmax_69 = fmaxf(kb[3], kb[4]);
    float hi_154 = _fmax_69;
    float _min_69 = fminf(kb[3], kb[4]);
    float lo_155 = _min_69;
    kb[3] = hi_154;
    kb[4] = lo_155;
    float _fmax_70 = fmaxf(kb[5], kb[6]);
    float hi_156 = _fmax_70;
    float _min_70 = fminf(kb[5], kb[6]);
    float lo_157 = _min_70;
    kb[5] = hi_156;
    kb[6] = lo_157;
    float _fmax_71 = fmaxf(kb[7], kb[8]);
    float hi_158 = _fmax_71;
    float _min_71 = fminf(kb[7], kb[8]);
    float lo_159 = _min_71;
    kb[7] = hi_158;
    kb[8] = lo_159;
    float _fmax_72 = fmaxf(kb[9], kb[10]);
    float hi_160 = _fmax_72;
    float _min_72 = fminf(kb[9], kb[10]);
    float lo_161 = _min_72;
    kb[9] = hi_160;
    kb[10] = lo_161;
    float _fmax_73 = fmaxf(kb[11], kb[12]);
    float hi_162 = _fmax_73;
    float _min_73 = fminf(kb[11], kb[12]);
    float lo_163 = _min_73;
    kb[11] = hi_162;
    kb[12] = lo_163;
    float _fmax_74 = fmaxf(kb[6], kb[7]);
    float hi_164 = _fmax_74;
    float _min_74 = fminf(kb[6], kb[7]);
    float lo_165 = _min_74;
    kb[6] = hi_164;
    kb[7] = lo_165;
    float _fmax_75 = fmaxf(kb[8], kb[9]);
    float hi_166 = _fmax_75;
    float _min_75 = fminf(kb[8], kb[9]);
    float lo_167 = _min_75;
    kb[8] = hi_166;
    kb[9] = lo_167;
    a[0] = kb[0];
    a[1] = kb[1];
    a[2] = kb[2];
    a[3] = kb[3];
    a[4] = kb[4];
    a[5] = kb[5];
    a[6] = kb[6];
    a[7] = kb[7];
    a[8] = kb[8];
    a[9] = kb[9];
    a[10] = kb[10];
    a[11] = kb[11];
    a[12] = kb[12];
    a[13] = kb[13];
    a[14] = kb[14];
    a[15] = kb[15];
    int ln = tid_1 & 15;
    int g = tid_1 >> 4;
    int cg = g & 7;
    int sg = g >> 3;
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
    asm volatile("barrier.sync 8, 256;" ::: "memory");
    float r = neg_inf;
    float V[8];
    int s0 = (sg * 16 * 8 + cg) * 17;
    int s1 = ((sg * 16 + 8) * 8 + cg) * 17;
    float x0 = pub[s0 + ln];
    float y0 = pub[s1 + lnr];
    float _min_76 = fminf(x0, y0);
    float lo0 = _min_76;
    float _fmax_76 = fmaxf(r, lo0);
    r = _fmax_76;
    float _fmax_77 = fmaxf(x0, y0);
    float hi0 = _fmax_77;
    float cur = hi0;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
    float pv = _shfl_xor_0;
    float _fmax_78 = fmaxf(cur, pv);
    float hi_168 = _fmax_78;
    float _min_77 = fminf(cur, pv);
    float lo_169 = _min_77;
    cur = ((up[0] != 0) ? hi_168 : lo_169);
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
    float pv_170 = _shfl_xor_1;
    float _fmax_79 = fmaxf(cur, pv_170);
    float hi_171 = _fmax_79;
    float _min_78 = fminf(cur, pv_170);
    float lo_172 = _min_78;
    cur = ((up[1] != 0) ? hi_171 : lo_172);
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
    float pv_173 = _shfl_xor_2;
    float _fmax_80 = fmaxf(cur, pv_173);
    float hi_174 = _fmax_80;
    float _min_79 = fminf(cur, pv_173);
    float lo_175 = _min_79;
    cur = ((up[2] != 0) ? hi_174 : lo_175);
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
    float pv_176 = _shfl_xor_3;
    float _fmax_81 = fmaxf(cur, pv_176);
    float hi_177 = _fmax_81;
    float _min_80 = fminf(cur, pv_176);
    float lo_178 = _min_80;
    cur = ((up[3] != 0) ? hi_177 : lo_178);
    V[0] = cur;
    int s0_179 = ((sg * 16 + 1) * 8 + cg) * 17;
    int s1_180 = ((sg * 16 + 1 + 8) * 8 + cg) * 17;
    float x0_181 = pub[s0_179 + ln];
    float y0_182 = pub[s1_180 + lnr];
    float _min_81 = fminf(x0_181, y0_182);
    float lo0_183 = _min_81;
    float _fmax_82 = fmaxf(r, lo0_183);
    r = _fmax_82;
    float _fmax_83 = fmaxf(x0_181, y0_182);
    float hi0_184 = _fmax_83;
    float cur_185 = hi0_184;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, cur_185, 8);
    float pv_186 = _shfl_xor_4;
    float _fmax_84 = fmaxf(cur_185, pv_186);
    float hi_187 = _fmax_84;
    float _min_82 = fminf(cur_185, pv_186);
    float lo_188 = _min_82;
    cur_185 = ((up[0] != 0) ? hi_187 : lo_188);
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, cur_185, 4);
    float pv_189 = _shfl_xor_5;
    float _fmax_85 = fmaxf(cur_185, pv_189);
    float hi_190 = _fmax_85;
    float _min_83 = fminf(cur_185, pv_189);
    float lo_191 = _min_83;
    cur_185 = ((up[1] != 0) ? hi_190 : lo_191);
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, cur_185, 2);
    float pv_192 = _shfl_xor_6;
    float _fmax_86 = fmaxf(cur_185, pv_192);
    float hi_193 = _fmax_86;
    float _min_84 = fminf(cur_185, pv_192);
    float lo_194 = _min_84;
    cur_185 = ((up[2] != 0) ? hi_193 : lo_194);
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, cur_185, 1);
    float pv_195 = _shfl_xor_7;
    float _fmax_87 = fmaxf(cur_185, pv_195);
    float hi_196 = _fmax_87;
    float _min_85 = fminf(cur_185, pv_195);
    float lo_197 = _min_85;
    cur_185 = ((up[3] != 0) ? hi_196 : lo_197);
    V[1] = cur_185;
    int s0_198 = ((sg * 16 + 2) * 8 + cg) * 17;
    int s1_199 = ((sg * 16 + 2 + 8) * 8 + cg) * 17;
    float x0_200 = pub[s0_198 + ln];
    float y0_201 = pub[s1_199 + lnr];
    float _min_86 = fminf(x0_200, y0_201);
    float lo0_202 = _min_86;
    float _fmax_88 = fmaxf(r, lo0_202);
    r = _fmax_88;
    float _fmax_89 = fmaxf(x0_200, y0_201);
    float hi0_203 = _fmax_89;
    float cur_204 = hi0_203;
    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, cur_204, 8);
    float pv_205 = _shfl_xor_8;
    float _fmax_90 = fmaxf(cur_204, pv_205);
    float hi_206 = _fmax_90;
    float _min_87 = fminf(cur_204, pv_205);
    float lo_207 = _min_87;
    cur_204 = ((up[0] != 0) ? hi_206 : lo_207);
    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, cur_204, 4);
    float pv_208 = _shfl_xor_9;
    float _fmax_91 = fmaxf(cur_204, pv_208);
    float hi_209 = _fmax_91;
    float _min_88 = fminf(cur_204, pv_208);
    float lo_210 = _min_88;
    cur_204 = ((up[1] != 0) ? hi_209 : lo_210);
    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, cur_204, 2);
    float pv_211 = _shfl_xor_10;
    float _fmax_92 = fmaxf(cur_204, pv_211);
    float hi_212 = _fmax_92;
    float _min_89 = fminf(cur_204, pv_211);
    float lo_213 = _min_89;
    cur_204 = ((up[2] != 0) ? hi_212 : lo_213);
    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, cur_204, 1);
    float pv_214 = _shfl_xor_11;
    float _fmax_93 = fmaxf(cur_204, pv_214);
    float hi_215 = _fmax_93;
    float _min_90 = fminf(cur_204, pv_214);
    float lo_216 = _min_90;
    cur_204 = ((up[3] != 0) ? hi_215 : lo_216);
    V[2] = cur_204;
    int s0_217 = ((sg * 16 + 3) * 8 + cg) * 17;
    int s1_218 = ((sg * 16 + 3 + 8) * 8 + cg) * 17;
    float x0_219 = pub[s0_217 + ln];
    float y0_220 = pub[s1_218 + lnr];
    float _min_91 = fminf(x0_219, y0_220);
    float lo0_221 = _min_91;
    float _fmax_94 = fmaxf(r, lo0_221);
    r = _fmax_94;
    float _fmax_95 = fmaxf(x0_219, y0_220);
    float hi0_222 = _fmax_95;
    float cur_223 = hi0_222;
    float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 8);
    float pv_224 = _shfl_xor_12;
    float _fmax_96 = fmaxf(cur_223, pv_224);
    float hi_225 = _fmax_96;
    float _min_92 = fminf(cur_223, pv_224);
    float lo_226 = _min_92;
    cur_223 = ((up[0] != 0) ? hi_225 : lo_226);
    float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 4);
    float pv_227 = _shfl_xor_13;
    float _fmax_97 = fmaxf(cur_223, pv_227);
    float hi_228 = _fmax_97;
    float _min_93 = fminf(cur_223, pv_227);
    float lo_229 = _min_93;
    cur_223 = ((up[1] != 0) ? hi_228 : lo_229);
    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 2);
    float pv_230 = _shfl_xor_14;
    float _fmax_98 = fmaxf(cur_223, pv_230);
    float hi_231 = _fmax_98;
    float _min_94 = fminf(cur_223, pv_230);
    float lo_232 = _min_94;
    cur_223 = ((up[2] != 0) ? hi_231 : lo_232);
    float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, cur_223, 1);
    float pv_233 = _shfl_xor_15;
    float _fmax_99 = fmaxf(cur_223, pv_233);
    float hi_234 = _fmax_99;
    float _min_95 = fminf(cur_223, pv_233);
    float lo_235 = _min_95;
    cur_223 = ((up[3] != 0) ? hi_234 : lo_235);
    V[3] = cur_223;
    int s0_236 = ((sg * 16 + 4) * 8 + cg) * 17;
    int s1_237 = ((sg * 16 + 4 + 8) * 8 + cg) * 17;
    float x0_238 = pub[s0_236 + ln];
    float y0_239 = pub[s1_237 + lnr];
    float _min_96 = fminf(x0_238, y0_239);
    float lo0_240 = _min_96;
    float _fmax_100 = fmaxf(r, lo0_240);
    r = _fmax_100;
    float _fmax_101 = fmaxf(x0_238, y0_239);
    float hi0_241 = _fmax_101;
    float cur_242 = hi0_241;
    float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, cur_242, 8);
    float pv_243 = _shfl_xor_16;
    float _fmax_102 = fmaxf(cur_242, pv_243);
    float hi_244 = _fmax_102;
    float _min_97 = fminf(cur_242, pv_243);
    float lo_245 = _min_97;
    cur_242 = ((up[0] != 0) ? hi_244 : lo_245);
    float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, cur_242, 4);
    float pv_246 = _shfl_xor_17;
    float _fmax_103 = fmaxf(cur_242, pv_246);
    float hi_247 = _fmax_103;
    float _min_98 = fminf(cur_242, pv_246);
    float lo_248 = _min_98;
    cur_242 = ((up[1] != 0) ? hi_247 : lo_248);
    float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, cur_242, 2);
    float pv_249 = _shfl_xor_18;
    float _fmax_104 = fmaxf(cur_242, pv_249);
    float hi_250 = _fmax_104;
    float _min_99 = fminf(cur_242, pv_249);
    float lo_251 = _min_99;
    cur_242 = ((up[2] != 0) ? hi_250 : lo_251);
    float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, cur_242, 1);
    float pv_252 = _shfl_xor_19;
    float _fmax_105 = fmaxf(cur_242, pv_252);
    float hi_253 = _fmax_105;
    float _min_100 = fminf(cur_242, pv_252);
    float lo_254 = _min_100;
    cur_242 = ((up[3] != 0) ? hi_253 : lo_254);
    V[4] = cur_242;
    int s0_255 = ((sg * 16 + 5) * 8 + cg) * 17;
    int s1_256 = ((sg * 16 + 5 + 8) * 8 + cg) * 17;
    float x0_257 = pub[s0_255 + ln];
    float y0_258 = pub[s1_256 + lnr];
    float _min_101 = fminf(x0_257, y0_258);
    float lo0_259 = _min_101;
    float _fmax_106 = fmaxf(r, lo0_259);
    r = _fmax_106;
    float _fmax_107 = fmaxf(x0_257, y0_258);
    float hi0_260 = _fmax_107;
    float cur_261 = hi0_260;
    float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, cur_261, 8);
    float pv_262 = _shfl_xor_20;
    float _fmax_108 = fmaxf(cur_261, pv_262);
    float hi_263 = _fmax_108;
    float _min_102 = fminf(cur_261, pv_262);
    float lo_264 = _min_102;
    cur_261 = ((up[0] != 0) ? hi_263 : lo_264);
    float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, cur_261, 4);
    float pv_265 = _shfl_xor_21;
    float _fmax_109 = fmaxf(cur_261, pv_265);
    float hi_266 = _fmax_109;
    float _min_103 = fminf(cur_261, pv_265);
    float lo_267 = _min_103;
    cur_261 = ((up[1] != 0) ? hi_266 : lo_267);
    float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, cur_261, 2);
    float pv_268 = _shfl_xor_22;
    float _fmax_110 = fmaxf(cur_261, pv_268);
    float hi_269 = _fmax_110;
    float _min_104 = fminf(cur_261, pv_268);
    float lo_270 = _min_104;
    cur_261 = ((up[2] != 0) ? hi_269 : lo_270);
    float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, cur_261, 1);
    float pv_271 = _shfl_xor_23;
    float _fmax_111 = fmaxf(cur_261, pv_271);
    float hi_272 = _fmax_111;
    float _min_105 = fminf(cur_261, pv_271);
    float lo_273 = _min_105;
    cur_261 = ((up[3] != 0) ? hi_272 : lo_273);
    V[5] = cur_261;
    int s0_274 = ((sg * 16 + 6) * 8 + cg) * 17;
    int s1_275 = ((sg * 16 + 6 + 8) * 8 + cg) * 17;
    float x0_276 = pub[s0_274 + ln];
    float y0_277 = pub[s1_275 + lnr];
    float _min_106 = fminf(x0_276, y0_277);
    float lo0_278 = _min_106;
    float _fmax_112 = fmaxf(r, lo0_278);
    r = _fmax_112;
    float _fmax_113 = fmaxf(x0_276, y0_277);
    float hi0_279 = _fmax_113;
    float cur_280 = hi0_279;
    float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, cur_280, 8);
    float pv_281 = _shfl_xor_24;
    float _fmax_114 = fmaxf(cur_280, pv_281);
    float hi_282 = _fmax_114;
    float _min_107 = fminf(cur_280, pv_281);
    float lo_283 = _min_107;
    cur_280 = ((up[0] != 0) ? hi_282 : lo_283);
    float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, cur_280, 4);
    float pv_284 = _shfl_xor_25;
    float _fmax_115 = fmaxf(cur_280, pv_284);
    float hi_285 = _fmax_115;
    float _min_108 = fminf(cur_280, pv_284);
    float lo_286 = _min_108;
    cur_280 = ((up[1] != 0) ? hi_285 : lo_286);
    float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, cur_280, 2);
    float pv_287 = _shfl_xor_26;
    float _fmax_116 = fmaxf(cur_280, pv_287);
    float hi_288 = _fmax_116;
    float _min_109 = fminf(cur_280, pv_287);
    float lo_289 = _min_109;
    cur_280 = ((up[2] != 0) ? hi_288 : lo_289);
    float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, cur_280, 1);
    float pv_290 = _shfl_xor_27;
    float _fmax_117 = fmaxf(cur_280, pv_290);
    float hi_291 = _fmax_117;
    float _min_110 = fminf(cur_280, pv_290);
    float lo_292 = _min_110;
    cur_280 = ((up[3] != 0) ? hi_291 : lo_292);
    V[6] = cur_280;
    int s0_293 = ((sg * 16 + 7) * 8 + cg) * 17;
    int s1_294 = ((sg * 16 + 7 + 8) * 8 + cg) * 17;
    float x0_295 = pub[s0_293 + ln];
    float y0_296 = pub[s1_294 + lnr];
    float _min_111 = fminf(x0_295, y0_296);
    float lo0_297 = _min_111;
    float _fmax_118 = fmaxf(r, lo0_297);
    r = _fmax_118;
    float _fmax_119 = fmaxf(x0_295, y0_296);
    float hi0_298 = _fmax_119;
    float cur_299 = hi0_298;
    float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, cur_299, 8);
    float pv_300 = _shfl_xor_28;
    float _fmax_120 = fmaxf(cur_299, pv_300);
    float hi_301 = _fmax_120;
    float _min_112 = fminf(cur_299, pv_300);
    float lo_302 = _min_112;
    cur_299 = ((up[0] != 0) ? hi_301 : lo_302);
    float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, cur_299, 4);
    float pv_303 = _shfl_xor_29;
    float _fmax_121 = fmaxf(cur_299, pv_303);
    float hi_304 = _fmax_121;
    float _min_113 = fminf(cur_299, pv_303);
    float lo_305 = _min_113;
    cur_299 = ((up[1] != 0) ? hi_304 : lo_305);
    float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, cur_299, 2);
    float pv_306 = _shfl_xor_30;
    float _fmax_122 = fmaxf(cur_299, pv_306);
    float hi_307 = _fmax_122;
    float _min_114 = fminf(cur_299, pv_306);
    float lo_308 = _min_114;
    cur_299 = ((up[2] != 0) ? hi_307 : lo_308);
    float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, cur_299, 1);
    float pv_309 = _shfl_xor_31;
    float _fmax_123 = fmaxf(cur_299, pv_309);
    float hi_310 = _fmax_123;
    float _min_115 = fminf(cur_299, pv_309);
    float lo_311 = _min_115;
    cur_299 = ((up[3] != 0) ? hi_310 : lo_311);
    V[7] = cur_299;
    float rs = pub[((sg * 16 + ln) * 8 + cg) * 17 + 16];
    float _fmax_124 = fmaxf(r, rs);
    r = _fmax_124;
    float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
    float y1 = _shfl_xor_32;
    float _min_116 = fminf(V[0], y1);
    float lo1 = _min_116;
    float _fmax_125 = fmaxf(r, lo1);
    r = _fmax_125;
    float _fmax_126 = fmaxf(V[0], y1);
    float hi1 = _fmax_126;
    float cur_312 = hi1;
    float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, cur_312, 8);
    float pv_313 = _shfl_xor_33;
    float _fmax_127 = fmaxf(cur_312, pv_313);
    float hi_314 = _fmax_127;
    float _min_117 = fminf(cur_312, pv_313);
    float lo_315 = _min_117;
    cur_312 = ((up[0] != 0) ? hi_314 : lo_315);
    float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, cur_312, 4);
    float pv_316 = _shfl_xor_34;
    float _fmax_128 = fmaxf(cur_312, pv_316);
    float hi_317 = _fmax_128;
    float _min_118 = fminf(cur_312, pv_316);
    float lo_318 = _min_118;
    cur_312 = ((up[1] != 0) ? hi_317 : lo_318);
    float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, cur_312, 2);
    float pv_319 = _shfl_xor_35;
    float _fmax_129 = fmaxf(cur_312, pv_319);
    float hi_320 = _fmax_129;
    float _min_119 = fminf(cur_312, pv_319);
    float lo_321 = _min_119;
    cur_312 = ((up[2] != 0) ? hi_320 : lo_321);
    float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, cur_312, 1);
    float pv_322 = _shfl_xor_36;
    float _fmax_130 = fmaxf(cur_312, pv_322);
    float hi_323 = _fmax_130;
    float _min_120 = fminf(cur_312, pv_322);
    float lo_324 = _min_120;
    cur_312 = ((up[3] != 0) ? hi_323 : lo_324);
    V[0] = cur_312;
    float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
    float y1_325 = _shfl_xor_37;
    float _min_121 = fminf(V[1], y1_325);
    float lo1_326 = _min_121;
    float _fmax_131 = fmaxf(r, lo1_326);
    r = _fmax_131;
    float _fmax_132 = fmaxf(V[1], y1_325);
    float hi1_327 = _fmax_132;
    float cur_328 = hi1_327;
    float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, cur_328, 8);
    float pv_329 = _shfl_xor_38;
    float _fmax_133 = fmaxf(cur_328, pv_329);
    float hi_330 = _fmax_133;
    float _min_122 = fminf(cur_328, pv_329);
    float lo_331 = _min_122;
    cur_328 = ((up[0] != 0) ? hi_330 : lo_331);
    float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, cur_328, 4);
    float pv_332 = _shfl_xor_39;
    float _fmax_134 = fmaxf(cur_328, pv_332);
    float hi_333 = _fmax_134;
    float _min_123 = fminf(cur_328, pv_332);
    float lo_334 = _min_123;
    cur_328 = ((up[1] != 0) ? hi_333 : lo_334);
    float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, cur_328, 2);
    float pv_335 = _shfl_xor_40;
    float _fmax_135 = fmaxf(cur_328, pv_335);
    float hi_336 = _fmax_135;
    float _min_124 = fminf(cur_328, pv_335);
    float lo_337 = _min_124;
    cur_328 = ((up[2] != 0) ? hi_336 : lo_337);
    float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, cur_328, 1);
    float pv_338 = _shfl_xor_41;
    float _fmax_136 = fmaxf(cur_328, pv_338);
    float hi_339 = _fmax_136;
    float _min_125 = fminf(cur_328, pv_338);
    float lo_340 = _min_125;
    cur_328 = ((up[3] != 0) ? hi_339 : lo_340);
    V[1] = cur_328;
    float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
    float y1_341 = _shfl_xor_42;
    float _min_126 = fminf(V[2], y1_341);
    float lo1_342 = _min_126;
    float _fmax_137 = fmaxf(r, lo1_342);
    r = _fmax_137;
    float _fmax_138 = fmaxf(V[2], y1_341);
    float hi1_343 = _fmax_138;
    float cur_344 = hi1_343;
    float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, cur_344, 8);
    float pv_345 = _shfl_xor_43;
    float _fmax_139 = fmaxf(cur_344, pv_345);
    float hi_346 = _fmax_139;
    float _min_127 = fminf(cur_344, pv_345);
    float lo_347 = _min_127;
    cur_344 = ((up[0] != 0) ? hi_346 : lo_347);
    float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, cur_344, 4);
    float pv_348 = _shfl_xor_44;
    float _fmax_140 = fmaxf(cur_344, pv_348);
    float hi_349 = _fmax_140;
    float _min_128 = fminf(cur_344, pv_348);
    float lo_350 = _min_128;
    cur_344 = ((up[1] != 0) ? hi_349 : lo_350);
    float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, cur_344, 2);
    float pv_351 = _shfl_xor_45;
    float _fmax_141 = fmaxf(cur_344, pv_351);
    float hi_352 = _fmax_141;
    float _min_129 = fminf(cur_344, pv_351);
    float lo_353 = _min_129;
    cur_344 = ((up[2] != 0) ? hi_352 : lo_353);
    float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, cur_344, 1);
    float pv_354 = _shfl_xor_46;
    float _fmax_142 = fmaxf(cur_344, pv_354);
    float hi_355 = _fmax_142;
    float _min_130 = fminf(cur_344, pv_354);
    float lo_356 = _min_130;
    cur_344 = ((up[3] != 0) ? hi_355 : lo_356);
    V[2] = cur_344;
    float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
    float y1_357 = _shfl_xor_47;
    float _min_131 = fminf(V[3], y1_357);
    float lo1_358 = _min_131;
    float _fmax_143 = fmaxf(r, lo1_358);
    r = _fmax_143;
    float _fmax_144 = fmaxf(V[3], y1_357);
    float hi1_359 = _fmax_144;
    float cur_360 = hi1_359;
    float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, cur_360, 8);
    float pv_361 = _shfl_xor_48;
    float _fmax_145 = fmaxf(cur_360, pv_361);
    float hi_362 = _fmax_145;
    float _min_132 = fminf(cur_360, pv_361);
    float lo_363 = _min_132;
    cur_360 = ((up[0] != 0) ? hi_362 : lo_363);
    float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, cur_360, 4);
    float pv_364 = _shfl_xor_49;
    float _fmax_146 = fmaxf(cur_360, pv_364);
    float hi_365 = _fmax_146;
    float _min_133 = fminf(cur_360, pv_364);
    float lo_366 = _min_133;
    cur_360 = ((up[1] != 0) ? hi_365 : lo_366);
    float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, cur_360, 2);
    float pv_367 = _shfl_xor_50;
    float _fmax_147 = fmaxf(cur_360, pv_367);
    float hi_368 = _fmax_147;
    float _min_134 = fminf(cur_360, pv_367);
    float lo_369 = _min_134;
    cur_360 = ((up[2] != 0) ? hi_368 : lo_369);
    float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, cur_360, 1);
    float pv_370 = _shfl_xor_51;
    float _fmax_148 = fmaxf(cur_360, pv_370);
    float hi_371 = _fmax_148;
    float _min_135 = fminf(cur_360, pv_370);
    float lo_372 = _min_135;
    cur_360 = ((up[3] != 0) ? hi_371 : lo_372);
    V[3] = cur_360;
    float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
    float y1_373 = _shfl_xor_52;
    float _min_136 = fminf(V[0], y1_373);
    float lo1_374 = _min_136;
    float _fmax_149 = fmaxf(r, lo1_374);
    r = _fmax_149;
    float _fmax_150 = fmaxf(V[0], y1_373);
    float hi1_375 = _fmax_150;
    float cur_376 = hi1_375;
    float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, cur_376, 8);
    float pv_377 = _shfl_xor_53;
    float _fmax_151 = fmaxf(cur_376, pv_377);
    float hi_378 = _fmax_151;
    float _min_137 = fminf(cur_376, pv_377);
    float lo_379 = _min_137;
    cur_376 = ((up[0] != 0) ? hi_378 : lo_379);
    float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, cur_376, 4);
    float pv_380 = _shfl_xor_54;
    float _fmax_152 = fmaxf(cur_376, pv_380);
    float hi_381 = _fmax_152;
    float _min_138 = fminf(cur_376, pv_380);
    float lo_382 = _min_138;
    cur_376 = ((up[1] != 0) ? hi_381 : lo_382);
    float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, cur_376, 2);
    float pv_383 = _shfl_xor_55;
    float _fmax_153 = fmaxf(cur_376, pv_383);
    float hi_384 = _fmax_153;
    float _min_139 = fminf(cur_376, pv_383);
    float lo_385 = _min_139;
    cur_376 = ((up[2] != 0) ? hi_384 : lo_385);
    float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, cur_376, 1);
    float pv_386 = _shfl_xor_56;
    float _fmax_154 = fmaxf(cur_376, pv_386);
    float hi_387 = _fmax_154;
    float _min_140 = fminf(cur_376, pv_386);
    float lo_388 = _min_140;
    cur_376 = ((up[3] != 0) ? hi_387 : lo_388);
    V[0] = cur_376;
    float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
    float y1_389 = _shfl_xor_57;
    float _min_141 = fminf(V[1], y1_389);
    float lo1_390 = _min_141;
    float _fmax_155 = fmaxf(r, lo1_390);
    r = _fmax_155;
    float _fmax_156 = fmaxf(V[1], y1_389);
    float hi1_391 = _fmax_156;
    float cur_392 = hi1_391;
    float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, cur_392, 8);
    float pv_393 = _shfl_xor_58;
    float _fmax_157 = fmaxf(cur_392, pv_393);
    float hi_394 = _fmax_157;
    float _min_142 = fminf(cur_392, pv_393);
    float lo_395 = _min_142;
    cur_392 = ((up[0] != 0) ? hi_394 : lo_395);
    float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, cur_392, 4);
    float pv_396 = _shfl_xor_59;
    float _fmax_158 = fmaxf(cur_392, pv_396);
    float hi_397 = _fmax_158;
    float _min_143 = fminf(cur_392, pv_396);
    float lo_398 = _min_143;
    cur_392 = ((up[1] != 0) ? hi_397 : lo_398);
    float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, cur_392, 2);
    float pv_399 = _shfl_xor_60;
    float _fmax_159 = fmaxf(cur_392, pv_399);
    float hi_400 = _fmax_159;
    float _min_144 = fminf(cur_392, pv_399);
    float lo_401 = _min_144;
    cur_392 = ((up[2] != 0) ? hi_400 : lo_401);
    float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, cur_392, 1);
    float pv_402 = _shfl_xor_61;
    float _fmax_160 = fmaxf(cur_392, pv_402);
    float hi_403 = _fmax_160;
    float _min_145 = fminf(cur_392, pv_402);
    float lo_404 = _min_145;
    cur_392 = ((up[3] != 0) ? hi_403 : lo_404);
    V[1] = cur_392;
    float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
    float yl = _shfl_xor_62;
    float _min_146 = fminf(V[0], yl);
    float lol = _min_146;
    float _fmax_161 = fmaxf(r, lol);
    r = _fmax_161;
    float _fmax_162 = fmaxf(V[0], yl);
    float hil = _fmax_162;
    float cur_405 = hil;
    float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, cur_405, 8);
    float pv_406 = _shfl_xor_63;
    float _fmax_163 = fmaxf(cur_405, pv_406);
    float hi_407 = _fmax_163;
    float _min_147 = fminf(cur_405, pv_406);
    float lo_408 = _min_147;
    cur_405 = ((up[0] != 0) ? hi_407 : lo_408);
    float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, cur_405, 4);
    float pv_409 = _shfl_xor_64;
    float _fmax_164 = fmaxf(cur_405, pv_409);
    float hi_410 = _fmax_164;
    float _min_148 = fminf(cur_405, pv_409);
    float lo_411 = _min_148;
    cur_405 = ((up[1] != 0) ? hi_410 : lo_411);
    float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, cur_405, 2);
    float pv_412 = _shfl_xor_65;
    float _fmax_165 = fmaxf(cur_405, pv_412);
    float hi_413 = _fmax_165;
    float _min_149 = fminf(cur_405, pv_412);
    float lo_414 = _min_149;
    cur_405 = ((up[2] != 0) ? hi_413 : lo_414);
    float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, cur_405, 1);
    float pv_415 = _shfl_xor_66;
    float _fmax_166 = fmaxf(cur_405, pv_415);
    float hi_416 = _fmax_166;
    float _min_150 = fminf(cur_405, pv_415);
    float lo_417 = _min_150;
    cur_405 = ((up[3] != 0) ? hi_416 : lo_417);
    V[0] = cur_405;
    float K = V[0];
    int qb = (sg * 8 + cg) * 32;
    q2[qb + ln] = K;
    q2[qb + 16 + ln] = r;
    asm volatile("barrier.sync 8, 256;" ::: "memory");
    if (tid_1 < 128) {
        float r2 = neg_inf;
        float _fmax_167 = fmaxf(r2, q2[cg * 32 + 16 + ln]);
        r2 = _fmax_167;
        float _fmax_168 = fmaxf(r2, q2[(8 + cg) * 32 + 16 + ln]);
        r2 = _fmax_168;
        float V2[1];
        float x2 = q2[cg * 32 + ln];
        float y2 = q2[(8 + cg) * 32 + lnr];
        float _min_151 = fminf(x2, y2);
        float lo2 = _min_151;
        float _fmax_169 = fmaxf(r2, lo2);
        r2 = _fmax_169;
        float _fmax_170 = fmaxf(x2, y2);
        float hi2 = _fmax_170;
        V2[0] = hi2;
        K = V2[0];
        r = r2;
    }
    rr1[0] = r;
    int ucol = blockIdx.x * 8 + cg;
    int commit = 0;
    if (g < 8 && ucol < total_q) {
        commit = 1;
    }
    if (tid_1 < 128) {
        float u16 = K;
        float cr = rr1[0];
        float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, u16, 8);
        float _min_152 = fminf(u16, _shfl_xor_67);
        u16 = _min_152;
        float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, cr, 8);
        float _fmax_171 = fmaxf(cr, _shfl_xor_68);
        cr = _fmax_171;
        float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, u16, 4);
        float _min_153 = fminf(u16, _shfl_xor_69);
        u16 = _min_153;
        float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, cr, 4);
        float _fmax_172 = fmaxf(cr, _shfl_xor_70);
        cr = _fmax_172;
        float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, u16, 2);
        float _min_154 = fminf(u16, _shfl_xor_71);
        u16 = _min_154;
        float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, cr, 2);
        float _fmax_173 = fmaxf(cr, _shfl_xor_72);
        cr = _fmax_173;
        float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, u16, 1);
        float _min_155 = fminf(u16, _shfl_xor_73);
        u16 = _min_155;
        float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, cr, 1);
        float _fmax_174 = fmaxf(cr, _shfl_xor_74);
        cr = _fmax_174;
        unsigned int u16b = __as_u32(u16);
        unsigned int c16 = u16b & 4294966784u;
        unsigned int crb = __as_u32(cr) & 4294966784u;
        unsigned int q = 4294967295;
        if (c16 == crb && u16b < 4278190080u && c16 != 2139094528 && commit != 0) {
            q = c16;
            if (ln == 0) {
                flagw[0] = 1;
            }
        }
        if (ln == 0 && g < 8) {
            qcol[cg] = q;
        }
    }
    asm volatile("barrier.sync 8, 256;" ::: "memory");
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
        if (flagged != 0) {
            float kb2o[16];
            int t0_4 = w * 16;
            int t0_5 = t0_4;
            float qf = __uint_as_float(qc);
            float sc_7 = __uint_as_float(cb[0]);
            float _fmax_175 = fmaxf(sc_7, -1.7014118346046923e+38f);
            sc_7 = _fmax_175;
            float _min_156 = fminf(sc_7, 1.7014118346046923e+38f);
            sc_7 = _min_156;
            sc_7 = sc_7;
            float sc_10 = sc_7;
            unsigned int u = __as_u32(sc_10);
            unsigned int cls = u & 4294966784u;
            int f_11_1 = 0;
            if (t0_5 < fb || t0_5 >= lim - fe && lim > t0_5) {
                f_11_1 = 1;
            }
            if (f_11_1 != 0) {
                cls = 2139094528;
            }
            unsigned int key_12 = 0;
            if (qf < __uint_as_float(cls) && cls < 4278190080u) {
                key_12 = 1073741824 | (unsigned int)t0_5;
            }
            if (cls == qc) {
                unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 511) & 511;
                key_12 = 536870912 | lowb << 9 | (unsigned int)t0_5;
            }
            kb2o[0] = __uint_as_float(key_12);
            float sc_13 = __uint_as_float(cb[1]);
            float _fmax_176 = fmaxf(sc_13, -1.7014118346046923e+38f);
            sc_13 = _fmax_176;
            float _min_157 = fminf(sc_13, 1.7014118346046923e+38f);
            sc_13 = _min_157;
            sc_13 = sc_13;
            float sc_16 = sc_13;
            unsigned int u_17 = __as_u32(sc_16);
            unsigned int cls_18 = u_17 & 4294966784u;
            int f_19 = 0;
            if (t0_5 + 1 < fb || t0_5 + 1 >= lim - fe && lim > t0_5 + 1) {
                f_19 = 1;
            }
            if (f_19 != 0) {
                cls_18 = 2139094528;
            }
            unsigned int key_20 = 0;
            if (qf < __uint_as_float(cls_18) && cls_18 < 4278190080u) {
                key_20 = 1073741824 | (unsigned int)(t0_5 + 1);
            }
            if (cls_18 == qc) {
                unsigned int lowb_1 = (u_17 ^ (unsigned int)((int)u_17 >> 31) & 511) & 511;
                key_20 = 536870912 | lowb_1 << 9 | (unsigned int)(t0_5 + 1);
            }
            kb2o[1] = __uint_as_float(key_20);
            float sc_22 = __uint_as_float(cb[2]);
            float _fmax_177 = fmaxf(sc_22, -1.7014118346046923e+38f);
            sc_22 = _fmax_177;
            float _min_158 = fminf(sc_22, 1.7014118346046923e+38f);
            sc_22 = _min_158;
            sc_22 = sc_22;
            float sc_25 = sc_22;
            unsigned int u_26 = __as_u32(sc_25);
            unsigned int cls_27 = u_26 & 4294966784u;
            int f_28 = 0;
            if (t0_5 + 2 < fb || t0_5 + 2 >= lim - fe && lim > t0_5 + 2) {
                f_28 = 1;
            }
            if (f_28 != 0) {
                cls_27 = 2139094528;
            }
            unsigned int key_29 = 0;
            if (qf < __uint_as_float(cls_27) && cls_27 < 4278190080u) {
                key_29 = 1073741824 | (unsigned int)(t0_5 + 2);
            }
            if (cls_27 == qc) {
                unsigned int lowb_2 = (u_26 ^ (unsigned int)((int)u_26 >> 31) & 511) & 511;
                key_29 = 536870912 | lowb_2 << 9 | (unsigned int)(t0_5 + 2);
            }
            kb2o[2] = __uint_as_float(key_29);
            float sc_31 = __uint_as_float(cb[3]);
            float _fmax_178 = fmaxf(sc_31, -1.7014118346046923e+38f);
            sc_31 = _fmax_178;
            float _min_159 = fminf(sc_31, 1.7014118346046923e+38f);
            sc_31 = _min_159;
            sc_31 = sc_31;
            float sc_34 = sc_31;
            unsigned int u_35 = __as_u32(sc_34);
            unsigned int cls_36 = u_35 & 4294966784u;
            int f_37 = 0;
            if (t0_5 + 3 < fb || t0_5 + 3 >= lim - fe && lim > t0_5 + 3) {
                f_37 = 1;
            }
            if (f_37 != 0) {
                cls_36 = 2139094528;
            }
            unsigned int key_38 = 0;
            if (qf < __uint_as_float(cls_36) && cls_36 < 4278190080u) {
                key_38 = 1073741824 | (unsigned int)(t0_5 + 3);
            }
            if (cls_36 == qc) {
                unsigned int lowb_3 = (u_35 ^ (unsigned int)((int)u_35 >> 31) & 511) & 511;
                key_38 = 536870912 | lowb_3 << 9 | (unsigned int)(t0_5 + 3);
            }
            kb2o[3] = __uint_as_float(key_38);
            float sc_40 = __uint_as_float(cb[4]);
            float _fmax_179 = fmaxf(sc_40, -1.7014118346046923e+38f);
            sc_40 = _fmax_179;
            float _min_160 = fminf(sc_40, 1.7014118346046923e+38f);
            sc_40 = _min_160;
            sc_40 = sc_40;
            float sc_43 = sc_40;
            unsigned int u_44 = __as_u32(sc_43);
            unsigned int cls_45 = u_44 & 4294966784u;
            int f_46 = 0;
            if (t0_5 + 4 < fb || t0_5 + 4 >= lim - fe && lim > t0_5 + 4) {
                f_46 = 1;
            }
            if (f_46 != 0) {
                cls_45 = 2139094528;
            }
            unsigned int key_47 = 0;
            if (qf < __uint_as_float(cls_45) && cls_45 < 4278190080u) {
                key_47 = 1073741824 | (unsigned int)(t0_5 + 4);
            }
            if (cls_45 == qc) {
                unsigned int lowb_4 = (u_44 ^ (unsigned int)((int)u_44 >> 31) & 511) & 511;
                key_47 = 536870912 | lowb_4 << 9 | (unsigned int)(t0_5 + 4);
            }
            kb2o[4] = __uint_as_float(key_47);
            float sc_49 = __uint_as_float(cb[5]);
            float _fmax_180 = fmaxf(sc_49, -1.7014118346046923e+38f);
            sc_49 = _fmax_180;
            float _min_161 = fminf(sc_49, 1.7014118346046923e+38f);
            sc_49 = _min_161;
            sc_49 = sc_49;
            float sc_50 = sc_49;
            unsigned int u_51 = __as_u32(sc_50);
            unsigned int cls_52 = u_51 & 4294966784u;
            int f_53 = 0;
            if (t0_5 + 5 < fb || t0_5 + 5 >= lim - fe && lim > t0_5 + 5) {
                f_53 = 1;
            }
            if (f_53 != 0) {
                cls_52 = 2139094528;
            }
            unsigned int key_54 = 0;
            if (qf < __uint_as_float(cls_52) && cls_52 < 4278190080u) {
                key_54 = 1073741824 | (unsigned int)(t0_5 + 5);
            }
            if (cls_52 == qc) {
                unsigned int lowb_5 = (u_51 ^ (unsigned int)((int)u_51 >> 31) & 511) & 511;
                key_54 = 536870912 | lowb_5 << 9 | (unsigned int)(t0_5 + 5);
            }
            kb2o[5] = __uint_as_float(key_54);
            float sc_55 = __uint_as_float(cb[6]);
            float _fmax_181 = fmaxf(sc_55, -1.7014118346046923e+38f);
            sc_55 = _fmax_181;
            float _min_162 = fminf(sc_55, 1.7014118346046923e+38f);
            sc_55 = _min_162;
            sc_55 = sc_55;
            float sc_56 = sc_55;
            unsigned int u_57 = __as_u32(sc_56);
            unsigned int cls_58 = u_57 & 4294966784u;
            int f_59 = 0;
            if (t0_5 + 6 < fb || t0_5 + 6 >= lim - fe && lim > t0_5 + 6) {
                f_59 = 1;
            }
            if (f_59 != 0) {
                cls_58 = 2139094528;
            }
            unsigned int key_60 = 0;
            if (qf < __uint_as_float(cls_58) && cls_58 < 4278190080u) {
                key_60 = 1073741824 | (unsigned int)(t0_5 + 6);
            }
            if (cls_58 == qc) {
                unsigned int lowb_6 = (u_57 ^ (unsigned int)((int)u_57 >> 31) & 511) & 511;
                key_60 = 536870912 | lowb_6 << 9 | (unsigned int)(t0_5 + 6);
            }
            kb2o[6] = __uint_as_float(key_60);
            float sc_61 = __uint_as_float(cb[7]);
            float _fmax_182 = fmaxf(sc_61, -1.7014118346046923e+38f);
            sc_61 = _fmax_182;
            float _min_163 = fminf(sc_61, 1.7014118346046923e+38f);
            sc_61 = _min_163;
            sc_61 = sc_61;
            float sc_62 = sc_61;
            unsigned int u_63 = __as_u32(sc_62);
            unsigned int cls_64 = u_63 & 4294966784u;
            int f_65 = 0;
            if (t0_5 + 7 < fb || t0_5 + 7 >= lim - fe && lim > t0_5 + 7) {
                f_65 = 1;
            }
            if (f_65 != 0) {
                cls_64 = 2139094528;
            }
            unsigned int key_66 = 0;
            if (qf < __uint_as_float(cls_64) && cls_64 < 4278190080u) {
                key_66 = 1073741824 | (unsigned int)(t0_5 + 7);
            }
            if (cls_64 == qc) {
                unsigned int lowb_7 = (u_63 ^ (unsigned int)((int)u_63 >> 31) & 511) & 511;
                key_66 = 536870912 | lowb_7 << 9 | (unsigned int)(t0_5 + 7);
            }
            kb2o[7] = __uint_as_float(key_66);
            float sc_67 = __uint_as_float(cb[8]);
            float _fmax_183 = fmaxf(sc_67, -1.7014118346046923e+38f);
            sc_67 = _fmax_183;
            float _min_164 = fminf(sc_67, 1.7014118346046923e+38f);
            sc_67 = _min_164;
            sc_67 = sc_67;
            float sc_68 = sc_67;
            unsigned int u_69 = __as_u32(sc_68);
            unsigned int cls_70 = u_69 & 4294966784u;
            int f_71 = 0;
            if (t0_5 + 8 < fb || t0_5 + 8 >= lim - fe && lim > t0_5 + 8) {
                f_71 = 1;
            }
            if (f_71 != 0) {
                cls_70 = 2139094528;
            }
            unsigned int key_72 = 0;
            if (qf < __uint_as_float(cls_70) && cls_70 < 4278190080u) {
                key_72 = 1073741824 | (unsigned int)(t0_5 + 8);
            }
            if (cls_70 == qc) {
                unsigned int lowb_8 = (u_69 ^ (unsigned int)((int)u_69 >> 31) & 511) & 511;
                key_72 = 536870912 | lowb_8 << 9 | (unsigned int)(t0_5 + 8);
            }
            kb2o[8] = __uint_as_float(key_72);
            float sc_73 = __uint_as_float(cb[9]);
            float _fmax_184 = fmaxf(sc_73, -1.7014118346046923e+38f);
            sc_73 = _fmax_184;
            float _min_165 = fminf(sc_73, 1.7014118346046923e+38f);
            sc_73 = _min_165;
            sc_73 = sc_73;
            float sc_74 = sc_73;
            unsigned int u_75 = __as_u32(sc_74);
            unsigned int cls_76 = u_75 & 4294966784u;
            int f_77 = 0;
            if (t0_5 + 9 < fb || t0_5 + 9 >= lim - fe && lim > t0_5 + 9) {
                f_77 = 1;
            }
            if (f_77 != 0) {
                cls_76 = 2139094528;
            }
            unsigned int key_78 = 0;
            if (qf < __uint_as_float(cls_76) && cls_76 < 4278190080u) {
                key_78 = 1073741824 | (unsigned int)(t0_5 + 9);
            }
            if (cls_76 == qc) {
                unsigned int lowb_9 = (u_75 ^ (unsigned int)((int)u_75 >> 31) & 511) & 511;
                key_78 = 536870912 | lowb_9 << 9 | (unsigned int)(t0_5 + 9);
            }
            kb2o[9] = __uint_as_float(key_78);
            float sc_79 = __uint_as_float(cb[10]);
            float _fmax_185 = fmaxf(sc_79, -1.7014118346046923e+38f);
            sc_79 = _fmax_185;
            float _min_166 = fminf(sc_79, 1.7014118346046923e+38f);
            sc_79 = _min_166;
            sc_79 = sc_79;
            float sc_80 = sc_79;
            unsigned int u_81 = __as_u32(sc_80);
            unsigned int cls_82 = u_81 & 4294966784u;
            int f_83 = 0;
            if (t0_5 + 10 < fb || t0_5 + 10 >= lim - fe && lim > t0_5 + 10) {
                f_83 = 1;
            }
            if (f_83 != 0) {
                cls_82 = 2139094528;
            }
            unsigned int key_84 = 0;
            if (qf < __uint_as_float(cls_82) && cls_82 < 4278190080u) {
                key_84 = 1073741824 | (unsigned int)(t0_5 + 10);
            }
            if (cls_82 == qc) {
                unsigned int lowb_10 = (u_81 ^ (unsigned int)((int)u_81 >> 31) & 511) & 511;
                key_84 = 536870912 | lowb_10 << 9 | (unsigned int)(t0_5 + 10);
            }
            kb2o[10] = __uint_as_float(key_84);
            float sc_85 = __uint_as_float(cb[11]);
            float _fmax_186 = fmaxf(sc_85, -1.7014118346046923e+38f);
            sc_85 = _fmax_186;
            float _min_167 = fminf(sc_85, 1.7014118346046923e+38f);
            sc_85 = _min_167;
            sc_85 = sc_85;
            float sc_86 = sc_85;
            unsigned int u_87 = __as_u32(sc_86);
            unsigned int cls_88 = u_87 & 4294966784u;
            int f_89 = 0;
            if (t0_5 + 11 < fb || t0_5 + 11 >= lim - fe && lim > t0_5 + 11) {
                f_89 = 1;
            }
            if (f_89 != 0) {
                cls_88 = 2139094528;
            }
            unsigned int key_90 = 0;
            if (qf < __uint_as_float(cls_88) && cls_88 < 4278190080u) {
                key_90 = 1073741824 | (unsigned int)(t0_5 + 11);
            }
            if (cls_88 == qc) {
                unsigned int lowb_11 = (u_87 ^ (unsigned int)((int)u_87 >> 31) & 511) & 511;
                key_90 = 536870912 | lowb_11 << 9 | (unsigned int)(t0_5 + 11);
            }
            kb2o[11] = __uint_as_float(key_90);
            float sc_91 = __uint_as_float(cb[12]);
            float _fmax_187 = fmaxf(sc_91, -1.7014118346046923e+38f);
            sc_91 = _fmax_187;
            float _min_168 = fminf(sc_91, 1.7014118346046923e+38f);
            sc_91 = _min_168;
            sc_91 = sc_91;
            float sc_92 = sc_91;
            unsigned int u_93 = __as_u32(sc_92);
            unsigned int cls_94 = u_93 & 4294966784u;
            int f_95 = 0;
            if (t0_5 + 12 < fb || t0_5 + 12 >= lim - fe && lim > t0_5 + 12) {
                f_95 = 1;
            }
            if (f_95 != 0) {
                cls_94 = 2139094528;
            }
            unsigned int key_96 = 0;
            if (qf < __uint_as_float(cls_94) && cls_94 < 4278190080u) {
                key_96 = 1073741824 | (unsigned int)(t0_5 + 12);
            }
            if (cls_94 == qc) {
                unsigned int lowb_12 = (u_93 ^ (unsigned int)((int)u_93 >> 31) & 511) & 511;
                key_96 = 536870912 | lowb_12 << 9 | (unsigned int)(t0_5 + 12);
            }
            kb2o[12] = __uint_as_float(key_96);
            float sc_97 = __uint_as_float(cb[13]);
            float _fmax_188 = fmaxf(sc_97, -1.7014118346046923e+38f);
            sc_97 = _fmax_188;
            float _min_169 = fminf(sc_97, 1.7014118346046923e+38f);
            sc_97 = _min_169;
            sc_97 = sc_97;
            float sc_98 = sc_97;
            unsigned int u_99 = __as_u32(sc_98);
            unsigned int cls_100 = u_99 & 4294966784u;
            int f_101 = 0;
            if (t0_5 + 13 < fb || t0_5 + 13 >= lim - fe && lim > t0_5 + 13) {
                f_101 = 1;
            }
            if (f_101 != 0) {
                cls_100 = 2139094528;
            }
            unsigned int key_102 = 0;
            if (qf < __uint_as_float(cls_100) && cls_100 < 4278190080u) {
                key_102 = 1073741824 | (unsigned int)(t0_5 + 13);
            }
            if (cls_100 == qc) {
                unsigned int lowb_13 = (u_99 ^ (unsigned int)((int)u_99 >> 31) & 511) & 511;
                key_102 = 536870912 | lowb_13 << 9 | (unsigned int)(t0_5 + 13);
            }
            kb2o[13] = __uint_as_float(key_102);
            float sc_103 = __uint_as_float(cb[14]);
            float _fmax_189 = fmaxf(sc_103, -1.7014118346046923e+38f);
            sc_103 = _fmax_189;
            float _min_170 = fminf(sc_103, 1.7014118346046923e+38f);
            sc_103 = _min_170;
            sc_103 = sc_103;
            float sc_104 = sc_103;
            unsigned int u_105 = __as_u32(sc_104);
            unsigned int cls_106 = u_105 & 4294966784u;
            int f_107 = 0;
            if (t0_5 + 14 < fb || t0_5 + 14 >= lim - fe && lim > t0_5 + 14) {
                f_107 = 1;
            }
            if (f_107 != 0) {
                cls_106 = 2139094528;
            }
            unsigned int key_108 = 0;
            if (qf < __uint_as_float(cls_106) && cls_106 < 4278190080u) {
                key_108 = 1073741824 | (unsigned int)(t0_5 + 14);
            }
            if (cls_106 == qc) {
                unsigned int lowb_14 = (u_105 ^ (unsigned int)((int)u_105 >> 31) & 511) & 511;
                key_108 = 536870912 | lowb_14 << 9 | (unsigned int)(t0_5 + 14);
            }
            kb2o[14] = __uint_as_float(key_108);
            float sc_109 = __uint_as_float(cb[15]);
            float _fmax_190 = fmaxf(sc_109, -1.7014118346046923e+38f);
            sc_109 = _fmax_190;
            float _min_171 = fminf(sc_109, 1.7014118346046923e+38f);
            sc_109 = _min_171;
            sc_109 = sc_109;
            float sc_110 = sc_109;
            unsigned int u_111 = __as_u32(sc_110);
            unsigned int cls_112 = u_111 & 4294966784u;
            int f_113 = 0;
            if (t0_5 + 15 < fb || t0_5 + 15 >= lim - fe && lim > t0_5 + 15) {
                f_113 = 1;
            }
            if (f_113 != 0) {
                cls_112 = 2139094528;
            }
            unsigned int key_114 = 0;
            if (qf < __uint_as_float(cls_112) && cls_112 < 4278190080u) {
                key_114 = 1073741824 | (unsigned int)(t0_5 + 15);
            }
            if (cls_112 == qc) {
                unsigned int lowb_15 = (u_111 ^ (unsigned int)((int)u_111 >> 31) & 511) & 511;
                key_114 = 536870912 | lowb_15 << 9 | (unsigned int)(t0_5 + 15);
            }
            kb2o[15] = __uint_as_float(key_114);
            float _fmax_191 = fmaxf(kb2o[0], kb2o[13]);
            float hi_115 = _fmax_191;
            float _min_172 = fminf(kb2o[0], kb2o[13]);
            float lo_116 = _min_172;
            kb2o[0] = hi_115;
            kb2o[13] = lo_116;
            float _fmax_192 = fmaxf(kb2o[1], kb2o[12]);
            float hi_117 = _fmax_192;
            float _min_173 = fminf(kb2o[1], kb2o[12]);
            float lo_118 = _min_173;
            kb2o[1] = hi_117;
            kb2o[12] = lo_118;
            float _fmax_193 = fmaxf(kb2o[2], kb2o[15]);
            float hi_119 = _fmax_193;
            float _min_174 = fminf(kb2o[2], kb2o[15]);
            float lo_120 = _min_174;
            kb2o[2] = hi_119;
            kb2o[15] = lo_120;
            float _fmax_194 = fmaxf(kb2o[3], kb2o[14]);
            float hi_121 = _fmax_194;
            float _min_175 = fminf(kb2o[3], kb2o[14]);
            float lo_122 = _min_175;
            kb2o[3] = hi_121;
            kb2o[14] = lo_122;
            float _fmax_195 = fmaxf(kb2o[4], kb2o[8]);
            float hi_123 = _fmax_195;
            float _min_176 = fminf(kb2o[4], kb2o[8]);
            float lo_124 = _min_176;
            kb2o[4] = hi_123;
            kb2o[8] = lo_124;
            float _fmax_196 = fmaxf(kb2o[5], kb2o[6]);
            float hi_125 = _fmax_196;
            float _min_177 = fminf(kb2o[5], kb2o[6]);
            float lo_126 = _min_177;
            kb2o[5] = hi_125;
            kb2o[6] = lo_126;
            float _fmax_197 = fmaxf(kb2o[7], kb2o[11]);
            float hi_127 = _fmax_197;
            float _min_178 = fminf(kb2o[7], kb2o[11]);
            float lo_128 = _min_178;
            kb2o[7] = hi_127;
            kb2o[11] = lo_128;
            float _fmax_198 = fmaxf(kb2o[9], kb2o[10]);
            float hi_129 = _fmax_198;
            float _min_179 = fminf(kb2o[9], kb2o[10]);
            float lo_130 = _min_179;
            kb2o[9] = hi_129;
            kb2o[10] = lo_130;
            float _fmax_199 = fmaxf(kb2o[0], kb2o[5]);
            float hi_131 = _fmax_199;
            float _min_180 = fminf(kb2o[0], kb2o[5]);
            float lo_132 = _min_180;
            kb2o[0] = hi_131;
            kb2o[5] = lo_132;
            float _fmax_200 = fmaxf(kb2o[1], kb2o[7]);
            float hi_133 = _fmax_200;
            float _min_181 = fminf(kb2o[1], kb2o[7]);
            float lo_134 = _min_181;
            kb2o[1] = hi_133;
            kb2o[7] = lo_134;
            float _fmax_201 = fmaxf(kb2o[2], kb2o[9]);
            float hi_135 = _fmax_201;
            float _min_182 = fminf(kb2o[2], kb2o[9]);
            float lo_136 = _min_182;
            kb2o[2] = hi_135;
            kb2o[9] = lo_136;
            float _fmax_202 = fmaxf(kb2o[3], kb2o[4]);
            float hi_137 = _fmax_202;
            float _min_183 = fminf(kb2o[3], kb2o[4]);
            float lo_138 = _min_183;
            kb2o[3] = hi_137;
            kb2o[4] = lo_138;
            float _fmax_203 = fmaxf(kb2o[6], kb2o[13]);
            float hi_139 = _fmax_203;
            float _min_184 = fminf(kb2o[6], kb2o[13]);
            float lo_140 = _min_184;
            kb2o[6] = hi_139;
            kb2o[13] = lo_140;
            float _fmax_204 = fmaxf(kb2o[8], kb2o[14]);
            float hi_141 = _fmax_204;
            float _min_185 = fminf(kb2o[8], kb2o[14]);
            float lo_142 = _min_185;
            kb2o[8] = hi_141;
            kb2o[14] = lo_142;
            float _fmax_205 = fmaxf(kb2o[10], kb2o[15]);
            float hi_143 = _fmax_205;
            float _min_186 = fminf(kb2o[10], kb2o[15]);
            float lo_144 = _min_186;
            kb2o[10] = hi_143;
            kb2o[15] = lo_144;
            float _fmax_206 = fmaxf(kb2o[11], kb2o[12]);
            float hi_145 = _fmax_206;
            float _min_187 = fminf(kb2o[11], kb2o[12]);
            float lo_146 = _min_187;
            kb2o[11] = hi_145;
            kb2o[12] = lo_146;
            float _fmax_207 = fmaxf(kb2o[0], kb2o[1]);
            float hi_147 = _fmax_207;
            float _min_188 = fminf(kb2o[0], kb2o[1]);
            float lo_148 = _min_188;
            kb2o[0] = hi_147;
            kb2o[1] = lo_148;
            float _fmax_208 = fmaxf(kb2o[2], kb2o[3]);
            float hi_149 = _fmax_208;
            float _min_189 = fminf(kb2o[2], kb2o[3]);
            float lo_150 = _min_189;
            kb2o[2] = hi_149;
            kb2o[3] = lo_150;
            float _fmax_209 = fmaxf(kb2o[4], kb2o[5]);
            float hi_151 = _fmax_209;
            float _min_190 = fminf(kb2o[4], kb2o[5]);
            float lo_152 = _min_190;
            kb2o[4] = hi_151;
            kb2o[5] = lo_152;
            float _fmax_210 = fmaxf(kb2o[6], kb2o[8]);
            float hi_153 = _fmax_210;
            float _min_191 = fminf(kb2o[6], kb2o[8]);
            float lo_154 = _min_191;
            kb2o[6] = hi_153;
            kb2o[8] = lo_154;
            float _fmax_211 = fmaxf(kb2o[7], kb2o[9]);
            float hi_155 = _fmax_211;
            float _min_192 = fminf(kb2o[7], kb2o[9]);
            float lo_156 = _min_192;
            kb2o[7] = hi_155;
            kb2o[9] = lo_156;
            float _fmax_212 = fmaxf(kb2o[10], kb2o[11]);
            float hi_157 = _fmax_212;
            float _min_193 = fminf(kb2o[10], kb2o[11]);
            float lo_158 = _min_193;
            kb2o[10] = hi_157;
            kb2o[11] = lo_158;
            float _fmax_213 = fmaxf(kb2o[12], kb2o[13]);
            float hi_159 = _fmax_213;
            float _min_194 = fminf(kb2o[12], kb2o[13]);
            float lo_160 = _min_194;
            kb2o[12] = hi_159;
            kb2o[13] = lo_160;
            float _fmax_214 = fmaxf(kb2o[14], kb2o[15]);
            float hi_161 = _fmax_214;
            float _min_195 = fminf(kb2o[14], kb2o[15]);
            float lo_162 = _min_195;
            kb2o[14] = hi_161;
            kb2o[15] = lo_162;
            float _fmax_215 = fmaxf(kb2o[0], kb2o[2]);
            float hi_163 = _fmax_215;
            float _min_196 = fminf(kb2o[0], kb2o[2]);
            float lo_164 = _min_196;
            kb2o[0] = hi_163;
            kb2o[2] = lo_164;
            float _fmax_216 = fmaxf(kb2o[1], kb2o[3]);
            float hi_165 = _fmax_216;
            float _min_197 = fminf(kb2o[1], kb2o[3]);
            float lo_166 = _min_197;
            kb2o[1] = hi_165;
            kb2o[3] = lo_166;
            float _fmax_217 = fmaxf(kb2o[4], kb2o[10]);
            float hi_167 = _fmax_217;
            float _min_198 = fminf(kb2o[4], kb2o[10]);
            float lo_168 = _min_198;
            kb2o[4] = hi_167;
            kb2o[10] = lo_168;
            float _fmax_218 = fmaxf(kb2o[5], kb2o[11]);
            float hi_169 = _fmax_218;
            float _min_199 = fminf(kb2o[5], kb2o[11]);
            float lo_170 = _min_199;
            kb2o[5] = hi_169;
            kb2o[11] = lo_170;
            float _fmax_219 = fmaxf(kb2o[6], kb2o[7]);
            float hi_172 = _fmax_219;
            float _min_200 = fminf(kb2o[6], kb2o[7]);
            float lo_173 = _min_200;
            kb2o[6] = hi_172;
            kb2o[7] = lo_173;
            float _fmax_220 = fmaxf(kb2o[8], kb2o[9]);
            float hi_175 = _fmax_220;
            float _min_201 = fminf(kb2o[8], kb2o[9]);
            float lo_176 = _min_201;
            kb2o[8] = hi_175;
            kb2o[9] = lo_176;
            float _fmax_221 = fmaxf(kb2o[12], kb2o[14]);
            float hi_178 = _fmax_221;
            float _min_202 = fminf(kb2o[12], kb2o[14]);
            float lo_179 = _min_202;
            kb2o[12] = hi_178;
            kb2o[14] = lo_179;
            float _fmax_222 = fmaxf(kb2o[13], kb2o[15]);
            float hi_180 = _fmax_222;
            float _min_203 = fminf(kb2o[13], kb2o[15]);
            float lo_181 = _min_203;
            kb2o[13] = hi_180;
            kb2o[15] = lo_181;
            float _fmax_223 = fmaxf(kb2o[1], kb2o[2]);
            float hi_182 = _fmax_223;
            float _min_204 = fminf(kb2o[1], kb2o[2]);
            float lo_183 = _min_204;
            kb2o[1] = hi_182;
            kb2o[2] = lo_183;
            float _fmax_224 = fmaxf(kb2o[3], kb2o[12]);
            float hi_184 = _fmax_224;
            float _min_205 = fminf(kb2o[3], kb2o[12]);
            float lo_185 = _min_205;
            kb2o[3] = hi_184;
            kb2o[12] = lo_185;
            float _fmax_225 = fmaxf(kb2o[4], kb2o[6]);
            float hi_186 = _fmax_225;
            float _min_206 = fminf(kb2o[4], kb2o[6]);
            float lo_187 = _min_206;
            kb2o[4] = hi_186;
            kb2o[6] = lo_187;
            float _fmax_226 = fmaxf(kb2o[5], kb2o[7]);
            float hi_188 = _fmax_226;
            float _min_207 = fminf(kb2o[5], kb2o[7]);
            float lo_189 = _min_207;
            kb2o[5] = hi_188;
            kb2o[7] = lo_189;
            float _fmax_227 = fmaxf(kb2o[8], kb2o[10]);
            float hi_191 = _fmax_227;
            float _min_208 = fminf(kb2o[8], kb2o[10]);
            float lo_192 = _min_208;
            kb2o[8] = hi_191;
            kb2o[10] = lo_192;
            float _fmax_228 = fmaxf(kb2o[9], kb2o[11]);
            float hi_194 = _fmax_228;
            float _min_209 = fminf(kb2o[9], kb2o[11]);
            float lo_195 = _min_209;
            kb2o[9] = hi_194;
            kb2o[11] = lo_195;
            float _fmax_229 = fmaxf(kb2o[13], kb2o[14]);
            float hi_197 = _fmax_229;
            float _min_210 = fminf(kb2o[13], kb2o[14]);
            float lo_198 = _min_210;
            kb2o[13] = hi_197;
            kb2o[14] = lo_198;
            float _fmax_230 = fmaxf(kb2o[1], kb2o[4]);
            float hi_199 = _fmax_230;
            float _min_211 = fminf(kb2o[1], kb2o[4]);
            float lo_200 = _min_211;
            kb2o[1] = hi_199;
            kb2o[4] = lo_200;
            float _fmax_231 = fmaxf(kb2o[2], kb2o[6]);
            float hi_201 = _fmax_231;
            float _min_212 = fminf(kb2o[2], kb2o[6]);
            float lo_202 = _min_212;
            kb2o[2] = hi_201;
            kb2o[6] = lo_202;
            float _fmax_232 = fmaxf(kb2o[5], kb2o[8]);
            float hi_203 = _fmax_232;
            float _min_213 = fminf(kb2o[5], kb2o[8]);
            float lo_204 = _min_213;
            kb2o[5] = hi_203;
            kb2o[8] = lo_204;
            float _fmax_233 = fmaxf(kb2o[7], kb2o[10]);
            float hi_205 = _fmax_233;
            float _min_214 = fminf(kb2o[7], kb2o[10]);
            float lo_206 = _min_214;
            kb2o[7] = hi_205;
            kb2o[10] = lo_206;
            float _fmax_234 = fmaxf(kb2o[9], kb2o[13]);
            float hi_207 = _fmax_234;
            float _min_215 = fminf(kb2o[9], kb2o[13]);
            float lo_208 = _min_215;
            kb2o[9] = hi_207;
            kb2o[13] = lo_208;
            float _fmax_235 = fmaxf(kb2o[11], kb2o[14]);
            float hi_210 = _fmax_235;
            float _min_216 = fminf(kb2o[11], kb2o[14]);
            float lo_211 = _min_216;
            kb2o[11] = hi_210;
            kb2o[14] = lo_211;
            float _fmax_236 = fmaxf(kb2o[2], kb2o[4]);
            float hi_213 = _fmax_236;
            float _min_217 = fminf(kb2o[2], kb2o[4]);
            float lo_214 = _min_217;
            kb2o[2] = hi_213;
            kb2o[4] = lo_214;
            float _fmax_237 = fmaxf(kb2o[3], kb2o[6]);
            float hi_216 = _fmax_237;
            float _min_218 = fminf(kb2o[3], kb2o[6]);
            float lo_217 = _min_218;
            kb2o[3] = hi_216;
            kb2o[6] = lo_217;
            float _fmax_238 = fmaxf(kb2o[9], kb2o[12]);
            float hi_218 = _fmax_238;
            float _min_219 = fminf(kb2o[9], kb2o[12]);
            float lo_219 = _min_219;
            kb2o[9] = hi_218;
            kb2o[12] = lo_219;
            float _fmax_239 = fmaxf(kb2o[11], kb2o[13]);
            float hi_220 = _fmax_239;
            float _min_220 = fminf(kb2o[11], kb2o[13]);
            float lo_221 = _min_220;
            kb2o[11] = hi_220;
            kb2o[13] = lo_221;
            float _fmax_240 = fmaxf(kb2o[3], kb2o[5]);
            float hi_222 = _fmax_240;
            float _min_221 = fminf(kb2o[3], kb2o[5]);
            float lo_223 = _min_221;
            kb2o[3] = hi_222;
            kb2o[5] = lo_223;
            float _fmax_241 = fmaxf(kb2o[6], kb2o[8]);
            float hi_224 = _fmax_241;
            float _min_222 = fminf(kb2o[6], kb2o[8]);
            float lo_225 = _min_222;
            kb2o[6] = hi_224;
            kb2o[8] = lo_225;
            float _fmax_242 = fmaxf(kb2o[7], kb2o[9]);
            float hi_226 = _fmax_242;
            float _min_223 = fminf(kb2o[7], kb2o[9]);
            float lo_227 = _min_223;
            kb2o[7] = hi_226;
            kb2o[9] = lo_227;
            float _fmax_243 = fmaxf(kb2o[10], kb2o[12]);
            float hi_229 = _fmax_243;
            float _min_224 = fminf(kb2o[10], kb2o[12]);
            float lo_230 = _min_224;
            kb2o[10] = hi_229;
            kb2o[12] = lo_230;
            float _fmax_244 = fmaxf(kb2o[3], kb2o[4]);
            float hi_232 = _fmax_244;
            float _min_225 = fminf(kb2o[3], kb2o[4]);
            float lo_233 = _min_225;
            kb2o[3] = hi_232;
            kb2o[4] = lo_233;
            float _fmax_245 = fmaxf(kb2o[5], kb2o[6]);
            float hi_235 = _fmax_245;
            float _min_226 = fminf(kb2o[5], kb2o[6]);
            float lo_236 = _min_226;
            kb2o[5] = hi_235;
            kb2o[6] = lo_236;
            float _fmax_246 = fmaxf(kb2o[7], kb2o[8]);
            float hi_237 = _fmax_246;
            float _min_227 = fminf(kb2o[7], kb2o[8]);
            float lo_238 = _min_227;
            kb2o[7] = hi_237;
            kb2o[8] = lo_238;
            float _fmax_247 = fmaxf(kb2o[9], kb2o[10]);
            float hi_239 = _fmax_247;
            float _min_228 = fminf(kb2o[9], kb2o[10]);
            float lo_240 = _min_228;
            kb2o[9] = hi_239;
            kb2o[10] = lo_240;
            float _fmax_248 = fmaxf(kb2o[11], kb2o[12]);
            float hi_241 = _fmax_248;
            float _min_229 = fminf(kb2o[11], kb2o[12]);
            float lo_242 = _min_229;
            kb2o[11] = hi_241;
            kb2o[12] = lo_242;
            float _fmax_249 = fmaxf(kb2o[6], kb2o[7]);
            float hi_243 = _fmax_249;
            float _min_230 = fminf(kb2o[6], kb2o[7]);
            float lo_244 = _min_230;
            kb2o[6] = hi_243;
            kb2o[7] = lo_244;
            float _fmax_250 = fmaxf(kb2o[8], kb2o[9]);
            float hi_245 = _fmax_250;
            float _min_231 = fminf(kb2o[8], kb2o[9]);
            float lo_246 = _min_231;
            kb2o[8] = hi_245;
            kb2o[9] = lo_246;
            a2[0] = kb2o[0];
            a2[1] = kb2o[1];
            a2[2] = kb2o[2];
            a2[3] = kb2o[3];
            a2[4] = kb2o[4];
            a2[5] = kb2o[5];
            a2[6] = kb2o[6];
            a2[7] = kb2o[7];
            a2[8] = kb2o[8];
            a2[9] = kb2o[9];
            a2[10] = kb2o[10];
            a2[11] = kb2o[11];
            a2[12] = kb2o[12];
            a2[13] = kb2o[13];
            a2[14] = kb2o[14];
            a2[15] = kb2o[15];
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
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        float r_1 = neg_inf;
        float V_2[8];
        int s0_3 = (sg * 16 * 8 + cg) * 17;
        int s1_4 = ((sg * 16 + 8) * 8 + cg) * 17;
        float x0_5 = pub[s0_3 + ln];
        float y0_6 = pub[s1_4 + lnr];
        float _min_232 = fminf(x0_5, y0_6);
        float lo0_7 = _min_232;
        float _fmax_251 = fmaxf(r_1, lo0_7);
        r_1 = _fmax_251;
        float _fmax_252 = fmaxf(x0_5, y0_6);
        float hi0_8 = _fmax_252;
        float cur_9 = hi0_8;
        float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 8);
        float pv_10 = _shfl_xor_75;
        float _fmax_253 = fmaxf(cur_9, pv_10);
        float hi_11 = _fmax_253;
        float _min_233 = fminf(cur_9, pv_10);
        float lo_12 = _min_233;
        cur_9 = ((up[0] != 0) ? hi_11 : lo_12);
        float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 4);
        float pv_13 = _shfl_xor_76;
        float _fmax_254 = fmaxf(cur_9, pv_13);
        float hi_14 = _fmax_254;
        float _min_234 = fminf(cur_9, pv_13);
        float lo_15 = _min_234;
        cur_9 = ((up[1] != 0) ? hi_14 : lo_15);
        float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 2);
        float pv_16 = _shfl_xor_77;
        float _fmax_255 = fmaxf(cur_9, pv_16);
        float hi_17 = _fmax_255;
        float _min_235 = fminf(cur_9, pv_16);
        float lo_18 = _min_235;
        cur_9 = ((up[2] != 0) ? hi_17 : lo_18);
        float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 1);
        float pv_19 = _shfl_xor_78;
        float _fmax_256 = fmaxf(cur_9, pv_19);
        float hi_20 = _fmax_256;
        float _min_236 = fminf(cur_9, pv_19);
        float lo_21 = _min_236;
        cur_9 = ((up[3] != 0) ? hi_20 : lo_21);
        V_2[0] = cur_9;
        int s0_22 = ((sg * 16 + 1) * 8 + cg) * 17;
        int s1_23 = ((sg * 16 + 1 + 8) * 8 + cg) * 17;
        float x0_24 = pub[s0_22 + ln];
        float y0_25 = pub[s1_23 + lnr];
        float _min_237 = fminf(x0_24, y0_25);
        float lo0_26 = _min_237;
        float _fmax_257 = fmaxf(r_1, lo0_26);
        r_1 = _fmax_257;
        float _fmax_258 = fmaxf(x0_24, y0_25);
        float hi0_27 = _fmax_258;
        float cur_28 = hi0_27;
        float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 8);
        float pv_29 = _shfl_xor_79;
        float _fmax_259 = fmaxf(cur_28, pv_29);
        float hi_30 = _fmax_259;
        float _min_238 = fminf(cur_28, pv_29);
        float lo_31 = _min_238;
        cur_28 = ((up[0] != 0) ? hi_30 : lo_31);
        float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 4);
        float pv_32 = _shfl_xor_80;
        float _fmax_260 = fmaxf(cur_28, pv_32);
        float hi_33 = _fmax_260;
        float _min_239 = fminf(cur_28, pv_32);
        float lo_34 = _min_239;
        cur_28 = ((up[1] != 0) ? hi_33 : lo_34);
        float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 2);
        float pv_35 = _shfl_xor_81;
        float _fmax_261 = fmaxf(cur_28, pv_35);
        float hi_36 = _fmax_261;
        float _min_240 = fminf(cur_28, pv_35);
        float lo_37 = _min_240;
        cur_28 = ((up[2] != 0) ? hi_36 : lo_37);
        float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 1);
        float pv_38 = _shfl_xor_82;
        float _fmax_262 = fmaxf(cur_28, pv_38);
        float hi_39 = _fmax_262;
        float _min_241 = fminf(cur_28, pv_38);
        float lo_40 = _min_241;
        cur_28 = ((up[3] != 0) ? hi_39 : lo_40);
        V_2[1] = cur_28;
        int s0_41 = ((sg * 16 + 2) * 8 + cg) * 17;
        int s1_42 = ((sg * 16 + 2 + 8) * 8 + cg) * 17;
        float x0_43 = pub[s0_41 + ln];
        float y0_44 = pub[s1_42 + lnr];
        float _min_242 = fminf(x0_43, y0_44);
        float lo0_45 = _min_242;
        float _fmax_263 = fmaxf(r_1, lo0_45);
        r_1 = _fmax_263;
        float _fmax_264 = fmaxf(x0_43, y0_44);
        float hi0_46 = _fmax_264;
        float cur_47 = hi0_46;
        float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 8);
        float pv_48 = _shfl_xor_83;
        float _fmax_265 = fmaxf(cur_47, pv_48);
        float hi_49 = _fmax_265;
        float _min_243 = fminf(cur_47, pv_48);
        float lo_50 = _min_243;
        cur_47 = ((up[0] != 0) ? hi_49 : lo_50);
        float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 4);
        float pv_51 = _shfl_xor_84;
        float _fmax_266 = fmaxf(cur_47, pv_51);
        float hi_53 = _fmax_266;
        float _min_244 = fminf(cur_47, pv_51);
        float lo_54 = _min_244;
        cur_47 = ((up[1] != 0) ? hi_53 : lo_54);
        float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 2);
        float pv_55 = _shfl_xor_85;
        float _fmax_267 = fmaxf(cur_47, pv_55);
        float hi_57 = _fmax_267;
        float _min_245 = fminf(cur_47, pv_55);
        float lo_58 = _min_245;
        cur_47 = ((up[2] != 0) ? hi_57 : lo_58);
        float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 1);
        float pv_59 = _shfl_xor_86;
        float _fmax_268 = fmaxf(cur_47, pv_59);
        float hi_61 = _fmax_268;
        float _min_246 = fminf(cur_47, pv_59);
        float lo_62 = _min_246;
        cur_47 = ((up[3] != 0) ? hi_61 : lo_62);
        V_2[2] = cur_47;
        int s0_63 = ((sg * 16 + 3) * 8 + cg) * 17;
        int s1_64 = ((sg * 16 + 3 + 8) * 8 + cg) * 17;
        float x0_65 = pub[s0_63 + ln];
        float y0_66 = pub[s1_64 + lnr];
        float _min_247 = fminf(x0_65, y0_66);
        float lo0_67 = _min_247;
        float _fmax_269 = fmaxf(r_1, lo0_67);
        r_1 = _fmax_269;
        float _fmax_270 = fmaxf(x0_65, y0_66);
        float hi0_68 = _fmax_270;
        float cur_69 = hi0_68;
        float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, cur_69, 8);
        float pv_70 = _shfl_xor_87;
        float _fmax_271 = fmaxf(cur_69, pv_70);
        float hi_71 = _fmax_271;
        float _min_248 = fminf(cur_69, pv_70);
        float lo_72 = _min_248;
        cur_69 = ((up[0] != 0) ? hi_71 : lo_72);
        float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, cur_69, 4);
        float pv_73 = _shfl_xor_88;
        float _fmax_272 = fmaxf(cur_69, pv_73);
        float hi_75 = _fmax_272;
        float _min_249 = fminf(cur_69, pv_73);
        float lo_76 = _min_249;
        cur_69 = ((up[1] != 0) ? hi_75 : lo_76);
        float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, cur_69, 2);
        float pv_77 = _shfl_xor_89;
        float _fmax_273 = fmaxf(cur_69, pv_77);
        float hi_79 = _fmax_273;
        float _min_250 = fminf(cur_69, pv_77);
        float lo_80 = _min_250;
        cur_69 = ((up[2] != 0) ? hi_79 : lo_80);
        float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, cur_69, 1);
        float pv_81 = _shfl_xor_90;
        float _fmax_274 = fmaxf(cur_69, pv_81);
        float hi_83 = _fmax_274;
        float _min_251 = fminf(cur_69, pv_81);
        float lo_84 = _min_251;
        cur_69 = ((up[3] != 0) ? hi_83 : lo_84);
        V_2[3] = cur_69;
        int s0_85 = ((sg * 16 + 4) * 8 + cg) * 17;
        int s1_86 = ((sg * 16 + 4 + 8) * 8 + cg) * 17;
        float x0_87 = pub[s0_85 + ln];
        float y0_88 = pub[s1_86 + lnr];
        float _min_252 = fminf(x0_87, y0_88);
        float lo0_89 = _min_252;
        float _fmax_275 = fmaxf(r_1, lo0_89);
        r_1 = _fmax_275;
        float _fmax_276 = fmaxf(x0_87, y0_88);
        float hi0_90 = _fmax_276;
        float cur_91 = hi0_90;
        float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, cur_91, 8);
        float pv_92 = _shfl_xor_91;
        float _fmax_277 = fmaxf(cur_91, pv_92);
        float hi_93 = _fmax_277;
        float _min_253 = fminf(cur_91, pv_92);
        float lo_94 = _min_253;
        cur_91 = ((up[0] != 0) ? hi_93 : lo_94);
        float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, cur_91, 4);
        float pv_95 = _shfl_xor_92;
        float _fmax_278 = fmaxf(cur_91, pv_95);
        float hi_97 = _fmax_278;
        float _min_254 = fminf(cur_91, pv_95);
        float lo_98 = _min_254;
        cur_91 = ((up[1] != 0) ? hi_97 : lo_98);
        float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, cur_91, 2);
        float pv_99 = _shfl_xor_93;
        float _fmax_279 = fmaxf(cur_91, pv_99);
        float hi_101 = _fmax_279;
        float _min_255 = fminf(cur_91, pv_99);
        float lo_102 = _min_255;
        cur_91 = ((up[2] != 0) ? hi_101 : lo_102);
        float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, cur_91, 1);
        float pv_103 = _shfl_xor_94;
        float _fmax_280 = fmaxf(cur_91, pv_103);
        float hi_105 = _fmax_280;
        float _min_256 = fminf(cur_91, pv_103);
        float lo_106 = _min_256;
        cur_91 = ((up[3] != 0) ? hi_105 : lo_106);
        V_2[4] = cur_91;
        int s0_107 = ((sg * 16 + 5) * 8 + cg) * 17;
        int s1_108 = ((sg * 16 + 5 + 8) * 8 + cg) * 17;
        float x0_109 = pub[s0_107 + ln];
        float y0_110 = pub[s1_108 + lnr];
        float _min_257 = fminf(x0_109, y0_110);
        float lo0_111 = _min_257;
        float _fmax_281 = fmaxf(r_1, lo0_111);
        r_1 = _fmax_281;
        float _fmax_282 = fmaxf(x0_109, y0_110);
        float hi0_112 = _fmax_282;
        float cur_113 = hi0_112;
        float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, cur_113, 8);
        float pv_114 = _shfl_xor_95;
        float _fmax_283 = fmaxf(cur_113, pv_114);
        float hi_115_1 = _fmax_283;
        float _min_258 = fminf(cur_113, pv_114);
        float lo_116_1 = _min_258;
        cur_113 = ((up[0] != 0) ? hi_115_1 : lo_116_1);
        float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, cur_113, 4);
        float pv_117 = _shfl_xor_96;
        float _fmax_284 = fmaxf(cur_113, pv_117);
        float hi_119_1 = _fmax_284;
        float _min_259 = fminf(cur_113, pv_117);
        float lo_120_1 = _min_259;
        cur_113 = ((up[1] != 0) ? hi_119_1 : lo_120_1);
        float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, cur_113, 2);
        float pv_121 = _shfl_xor_97;
        float _fmax_285 = fmaxf(cur_113, pv_121);
        float hi_123_1 = _fmax_285;
        float _min_260 = fminf(cur_113, pv_121);
        float lo_124_1 = _min_260;
        cur_113 = ((up[2] != 0) ? hi_123_1 : lo_124_1);
        float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, cur_113, 1);
        float pv_125 = _shfl_xor_98;
        float _fmax_286 = fmaxf(cur_113, pv_125);
        float hi_127_1 = _fmax_286;
        float _min_261 = fminf(cur_113, pv_125);
        float lo_128_1 = _min_261;
        cur_113 = ((up[3] != 0) ? hi_127_1 : lo_128_1);
        V_2[5] = cur_113;
        int s0_129 = ((sg * 16 + 6) * 8 + cg) * 17;
        int s1_130 = ((sg * 16 + 6 + 8) * 8 + cg) * 17;
        float x0_131 = pub[s0_129 + ln];
        float y0_132 = pub[s1_130 + lnr];
        float _min_262 = fminf(x0_131, y0_132);
        float lo0_133 = _min_262;
        float _fmax_287 = fmaxf(r_1, lo0_133);
        r_1 = _fmax_287;
        float _fmax_288 = fmaxf(x0_131, y0_132);
        float hi0_134 = _fmax_288;
        float cur_135 = hi0_134;
        float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, cur_135, 8);
        float pv_136 = _shfl_xor_99;
        float _fmax_289 = fmaxf(cur_135, pv_136);
        float hi_137_1 = _fmax_289;
        float _min_263 = fminf(cur_135, pv_136);
        float lo_138_1 = _min_263;
        cur_135 = ((up[0] != 0) ? hi_137_1 : lo_138_1);
        float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, cur_135, 4);
        float pv_139 = _shfl_xor_100;
        float _fmax_290 = fmaxf(cur_135, pv_139);
        float hi_141_1 = _fmax_290;
        float _min_264 = fminf(cur_135, pv_139);
        float lo_142_1 = _min_264;
        cur_135 = ((up[1] != 0) ? hi_141_1 : lo_142_1);
        float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, cur_135, 2);
        float pv_143 = _shfl_xor_101;
        float _fmax_291 = fmaxf(cur_135, pv_143);
        float hi_145_1 = _fmax_291;
        float _min_265 = fminf(cur_135, pv_143);
        float lo_146_1 = _min_265;
        cur_135 = ((up[2] != 0) ? hi_145_1 : lo_146_1);
        float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, cur_135, 1);
        float pv_147 = _shfl_xor_102;
        float _fmax_292 = fmaxf(cur_135, pv_147);
        float hi_149_1 = _fmax_292;
        float _min_266 = fminf(cur_135, pv_147);
        float lo_150_1 = _min_266;
        cur_135 = ((up[3] != 0) ? hi_149_1 : lo_150_1);
        V_2[6] = cur_135;
        int s0_151 = ((sg * 16 + 7) * 8 + cg) * 17;
        int s1_152 = ((sg * 16 + 7 + 8) * 8 + cg) * 17;
        float x0_153 = pub[s0_151 + ln];
        float y0_154 = pub[s1_152 + lnr];
        float _min_267 = fminf(x0_153, y0_154);
        float lo0_155 = _min_267;
        float _fmax_293 = fmaxf(r_1, lo0_155);
        r_1 = _fmax_293;
        float _fmax_294 = fmaxf(x0_153, y0_154);
        float hi0_156 = _fmax_294;
        float cur_157 = hi0_156;
        float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, cur_157, 8);
        float pv_158 = _shfl_xor_103;
        float _fmax_295 = fmaxf(cur_157, pv_158);
        float hi_159_1 = _fmax_295;
        float _min_268 = fminf(cur_157, pv_158);
        float lo_160_1 = _min_268;
        cur_157 = ((up[0] != 0) ? hi_159_1 : lo_160_1);
        float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, cur_157, 4);
        float pv_161 = _shfl_xor_104;
        float _fmax_296 = fmaxf(cur_157, pv_161);
        float hi_163_1 = _fmax_296;
        float _min_269 = fminf(cur_157, pv_161);
        float lo_164_1 = _min_269;
        cur_157 = ((up[1] != 0) ? hi_163_1 : lo_164_1);
        float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, cur_157, 2);
        float pv_165 = _shfl_xor_105;
        float _fmax_297 = fmaxf(cur_157, pv_165);
        float hi_167_1 = _fmax_297;
        float _min_270 = fminf(cur_157, pv_165);
        float lo_168_1 = _min_270;
        cur_157 = ((up[2] != 0) ? hi_167_1 : lo_168_1);
        float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, cur_157, 1);
        float pv_169 = _shfl_xor_106;
        float _fmax_298 = fmaxf(cur_157, pv_169);
        float hi_170 = _fmax_298;
        float _min_271 = fminf(cur_157, pv_169);
        float lo_171 = _min_271;
        cur_157 = ((up[3] != 0) ? hi_170 : lo_171);
        V_2[7] = cur_157;
        float rs_172 = pub[((sg * 16 + ln) * 8 + cg) * 17 + 16];
        float _fmax_299 = fmaxf(r_1, rs_172);
        r_1 = _fmax_299;
        float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, V_2[4], 15);
        float y1_173 = _shfl_xor_107;
        float _min_272 = fminf(V_2[0], y1_173);
        float lo1_174 = _min_272;
        float _fmax_300 = fmaxf(r_1, lo1_174);
        r_1 = _fmax_300;
        float _fmax_301 = fmaxf(V_2[0], y1_173);
        float hi1_175 = _fmax_301;
        float cur_176 = hi1_175;
        float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 8);
        float pv_177 = _shfl_xor_108;
        float _fmax_302 = fmaxf(cur_176, pv_177);
        float hi_178_1 = _fmax_302;
        float _min_273 = fminf(cur_176, pv_177);
        float lo_179_1 = _min_273;
        cur_176 = ((up[0] != 0) ? hi_178_1 : lo_179_1);
        float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 4);
        float pv_180 = _shfl_xor_109;
        float _fmax_303 = fmaxf(cur_176, pv_180);
        float hi_181 = _fmax_303;
        float _min_274 = fminf(cur_176, pv_180);
        float lo_182 = _min_274;
        cur_176 = ((up[1] != 0) ? hi_181 : lo_182);
        float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 2);
        float pv_183 = _shfl_xor_110;
        float _fmax_304 = fmaxf(cur_176, pv_183);
        float hi_184_1 = _fmax_304;
        float _min_275 = fminf(cur_176, pv_183);
        float lo_185_1 = _min_275;
        cur_176 = ((up[2] != 0) ? hi_184_1 : lo_185_1);
        float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, cur_176, 1);
        float pv_187 = _shfl_xor_111;
        float _fmax_305 = fmaxf(cur_176, pv_187);
        float hi_188_1 = _fmax_305;
        float _min_276 = fminf(cur_176, pv_187);
        float lo_189_1 = _min_276;
        cur_176 = ((up[3] != 0) ? hi_188_1 : lo_189_1);
        V_2[0] = cur_176;
        float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, V_2[5], 15);
        float y1_190 = _shfl_xor_112;
        float _min_277 = fminf(V_2[1], y1_190);
        float lo1_191 = _min_277;
        float _fmax_306 = fmaxf(r_1, lo1_191);
        r_1 = _fmax_306;
        float _fmax_307 = fmaxf(V_2[1], y1_190);
        float hi1_192 = _fmax_307;
        float cur_193 = hi1_192;
        float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, cur_193, 8);
        float pv_194 = _shfl_xor_113;
        float _fmax_308 = fmaxf(cur_193, pv_194);
        float hi_195 = _fmax_308;
        float _min_278 = fminf(cur_193, pv_194);
        float lo_196 = _min_278;
        cur_193 = ((up[0] != 0) ? hi_195 : lo_196);
        float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, cur_193, 4);
        float pv_197 = _shfl_xor_114;
        float _fmax_309 = fmaxf(cur_193, pv_197);
        float hi_198 = _fmax_309;
        float _min_279 = fminf(cur_193, pv_197);
        float lo_199 = _min_279;
        cur_193 = ((up[1] != 0) ? hi_198 : lo_199);
        float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, cur_193, 2);
        float pv_200 = _shfl_xor_115;
        float _fmax_310 = fmaxf(cur_193, pv_200);
        float hi_201_1 = _fmax_310;
        float _min_280 = fminf(cur_193, pv_200);
        float lo_202_1 = _min_280;
        cur_193 = ((up[2] != 0) ? hi_201_1 : lo_202_1);
        float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, cur_193, 1);
        float pv_203 = _shfl_xor_116;
        float _fmax_311 = fmaxf(cur_193, pv_203);
        float hi_204 = _fmax_311;
        float _min_281 = fminf(cur_193, pv_203);
        float lo_205 = _min_281;
        cur_193 = ((up[3] != 0) ? hi_204 : lo_205);
        V_2[1] = cur_193;
        float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, V_2[6], 15);
        float y1_206 = _shfl_xor_117;
        float _min_282 = fminf(V_2[2], y1_206);
        float lo1_207 = _min_282;
        float _fmax_312 = fmaxf(r_1, lo1_207);
        r_1 = _fmax_312;
        float _fmax_313 = fmaxf(V_2[2], y1_206);
        float hi1_208 = _fmax_313;
        float cur_209 = hi1_208;
        float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, cur_209, 8);
        float pv_210 = _shfl_xor_118;
        float _fmax_314 = fmaxf(cur_209, pv_210);
        float hi_211 = _fmax_314;
        float _min_283 = fminf(cur_209, pv_210);
        float lo_212 = _min_283;
        cur_209 = ((up[0] != 0) ? hi_211 : lo_212);
        float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, cur_209, 4);
        float pv_213 = _shfl_xor_119;
        float _fmax_315 = fmaxf(cur_209, pv_213);
        float hi_214 = _fmax_315;
        float _min_284 = fminf(cur_209, pv_213);
        float lo_215 = _min_284;
        cur_209 = ((up[1] != 0) ? hi_214 : lo_215);
        float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, cur_209, 2);
        float pv_216 = _shfl_xor_120;
        float _fmax_316 = fmaxf(cur_209, pv_216);
        float hi_217 = _fmax_316;
        float _min_285 = fminf(cur_209, pv_216);
        float lo_218 = _min_285;
        cur_209 = ((up[2] != 0) ? hi_217 : lo_218);
        float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, cur_209, 1);
        float pv_219 = _shfl_xor_121;
        float _fmax_317 = fmaxf(cur_209, pv_219);
        float hi_220_1 = _fmax_317;
        float _min_286 = fminf(cur_209, pv_219);
        float lo_221_1 = _min_286;
        cur_209 = ((up[3] != 0) ? hi_220_1 : lo_221_1);
        V_2[2] = cur_209;
        float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, V_2[7], 15);
        float y1_222 = _shfl_xor_122;
        float _min_287 = fminf(V_2[3], y1_222);
        float lo1_223 = _min_287;
        float _fmax_318 = fmaxf(r_1, lo1_223);
        r_1 = _fmax_318;
        float _fmax_319 = fmaxf(V_2[3], y1_222);
        float hi1_224 = _fmax_319;
        float cur_225 = hi1_224;
        float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, cur_225, 8);
        float pv_226 = _shfl_xor_123;
        float _fmax_320 = fmaxf(cur_225, pv_226);
        float hi_227 = _fmax_320;
        float _min_288 = fminf(cur_225, pv_226);
        float lo_228 = _min_288;
        cur_225 = ((up[0] != 0) ? hi_227 : lo_228);
        float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, cur_225, 4);
        float pv_229 = _shfl_xor_124;
        float _fmax_321 = fmaxf(cur_225, pv_229);
        float hi_230 = _fmax_321;
        float _min_289 = fminf(cur_225, pv_229);
        float lo_231 = _min_289;
        cur_225 = ((up[1] != 0) ? hi_230 : lo_231);
        float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, cur_225, 2);
        float pv_232 = _shfl_xor_125;
        float _fmax_322 = fmaxf(cur_225, pv_232);
        float hi_233 = _fmax_322;
        float _min_290 = fminf(cur_225, pv_232);
        float lo_234 = _min_290;
        cur_225 = ((up[2] != 0) ? hi_233 : lo_234);
        float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, cur_225, 1);
        float pv_235 = _shfl_xor_126;
        float _fmax_323 = fmaxf(cur_225, pv_235);
        float hi_236 = _fmax_323;
        float _min_291 = fminf(cur_225, pv_235);
        float lo_237 = _min_291;
        cur_225 = ((up[3] != 0) ? hi_236 : lo_237);
        V_2[3] = cur_225;
        float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, V_2[2], 15);
        float y1_238 = _shfl_xor_127;
        float _min_292 = fminf(V_2[0], y1_238);
        float lo1_239 = _min_292;
        float _fmax_324 = fmaxf(r_1, lo1_239);
        r_1 = _fmax_324;
        float _fmax_325 = fmaxf(V_2[0], y1_238);
        float hi1_240 = _fmax_325;
        float cur_241 = hi1_240;
        float _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, cur_241, 8);
        float pv_242 = _shfl_xor_128;
        float _fmax_326 = fmaxf(cur_241, pv_242);
        float hi_243_1 = _fmax_326;
        float _min_293 = fminf(cur_241, pv_242);
        float lo_244_1 = _min_293;
        cur_241 = ((up[0] != 0) ? hi_243_1 : lo_244_1);
        float _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, cur_241, 4);
        float pv_245 = _shfl_xor_129;
        float _fmax_327 = fmaxf(cur_241, pv_245);
        float hi_246 = _fmax_327;
        float _min_294 = fminf(cur_241, pv_245);
        float lo_247 = _min_294;
        cur_241 = ((up[1] != 0) ? hi_246 : lo_247);
        float _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, cur_241, 2);
        float pv_248 = _shfl_xor_130;
        float _fmax_328 = fmaxf(cur_241, pv_248);
        float hi_249 = _fmax_328;
        float _min_295 = fminf(cur_241, pv_248);
        float lo_250 = _min_295;
        cur_241 = ((up[2] != 0) ? hi_249 : lo_250);
        float _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, cur_241, 1);
        float pv_251 = _shfl_xor_131;
        float _fmax_329 = fmaxf(cur_241, pv_251);
        float hi_252 = _fmax_329;
        float _min_296 = fminf(cur_241, pv_251);
        float lo_253 = _min_296;
        cur_241 = ((up[3] != 0) ? hi_252 : lo_253);
        V_2[0] = cur_241;
        float _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, V_2[3], 15);
        float y1_254 = _shfl_xor_132;
        float _min_297 = fminf(V_2[1], y1_254);
        float lo1_255 = _min_297;
        float _fmax_330 = fmaxf(r_1, lo1_255);
        r_1 = _fmax_330;
        float _fmax_331 = fmaxf(V_2[1], y1_254);
        float hi1_256 = _fmax_331;
        float cur_257 = hi1_256;
        float _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, cur_257, 8);
        float pv_258 = _shfl_xor_133;
        float _fmax_332 = fmaxf(cur_257, pv_258);
        float hi_259 = _fmax_332;
        float _min_298 = fminf(cur_257, pv_258);
        float lo_260 = _min_298;
        cur_257 = ((up[0] != 0) ? hi_259 : lo_260);
        float _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, cur_257, 4);
        float pv_261 = _shfl_xor_134;
        float _fmax_333 = fmaxf(cur_257, pv_261);
        float hi_262 = _fmax_333;
        float _min_299 = fminf(cur_257, pv_261);
        float lo_263 = _min_299;
        cur_257 = ((up[1] != 0) ? hi_262 : lo_263);
        float _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, cur_257, 2);
        float pv_264 = _shfl_xor_135;
        float _fmax_334 = fmaxf(cur_257, pv_264);
        float hi_265 = _fmax_334;
        float _min_300 = fminf(cur_257, pv_264);
        float lo_266 = _min_300;
        cur_257 = ((up[2] != 0) ? hi_265 : lo_266);
        float _shfl_xor_136 = __shfl_xor_sync(0xFFFFFFFF, cur_257, 1);
        float pv_267 = _shfl_xor_136;
        float _fmax_335 = fmaxf(cur_257, pv_267);
        float hi_268 = _fmax_335;
        float _min_301 = fminf(cur_257, pv_267);
        float lo_269 = _min_301;
        cur_257 = ((up[3] != 0) ? hi_268 : lo_269);
        V_2[1] = cur_257;
        float _shfl_xor_137 = __shfl_xor_sync(0xFFFFFFFF, V_2[1], 15);
        float yl_270 = _shfl_xor_137;
        float _min_302 = fminf(V_2[0], yl_270);
        float lol_271 = _min_302;
        float _fmax_336 = fmaxf(r_1, lol_271);
        r_1 = _fmax_336;
        float _fmax_337 = fmaxf(V_2[0], yl_270);
        float hil_272 = _fmax_337;
        float cur_273 = hil_272;
        float _shfl_xor_138 = __shfl_xor_sync(0xFFFFFFFF, cur_273, 8);
        float pv_274 = _shfl_xor_138;
        float _fmax_338 = fmaxf(cur_273, pv_274);
        float hi_275 = _fmax_338;
        float _min_303 = fminf(cur_273, pv_274);
        float lo_276 = _min_303;
        cur_273 = ((up[0] != 0) ? hi_275 : lo_276);
        float _shfl_xor_139 = __shfl_xor_sync(0xFFFFFFFF, cur_273, 4);
        float pv_277 = _shfl_xor_139;
        float _fmax_339 = fmaxf(cur_273, pv_277);
        float hi_278 = _fmax_339;
        float _min_304 = fminf(cur_273, pv_277);
        float lo_279 = _min_304;
        cur_273 = ((up[1] != 0) ? hi_278 : lo_279);
        float _shfl_xor_140 = __shfl_xor_sync(0xFFFFFFFF, cur_273, 2);
        float pv_280 = _shfl_xor_140;
        float _fmax_340 = fmaxf(cur_273, pv_280);
        float hi_281 = _fmax_340;
        float _min_305 = fminf(cur_273, pv_280);
        float lo_282 = _min_305;
        cur_273 = ((up[2] != 0) ? hi_281 : lo_282);
        float _shfl_xor_141 = __shfl_xor_sync(0xFFFFFFFF, cur_273, 1);
        float pv_283 = _shfl_xor_141;
        float _fmax_341 = fmaxf(cur_273, pv_283);
        float hi_284 = _fmax_341;
        float _min_306 = fminf(cur_273, pv_283);
        float lo_285 = _min_306;
        cur_273 = ((up[3] != 0) ? hi_284 : lo_285);
        V_2[0] = cur_273;
        float K_286 = V_2[0];
        int qb_287 = (sg * 8 + cg) * 32;
        q2[qb_287 + ln] = K_286;
        q2[qb_287 + 16 + ln] = r_1;
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        if (tid_1 < 128) {
            float r2_1 = neg_inf;
            float _fmax_342 = fmaxf(r2_1, q2[cg * 32 + 16 + ln]);
            r2_1 = _fmax_342;
            float _fmax_343 = fmaxf(r2_1, q2[(8 + cg) * 32 + 16 + ln]);
            r2_1 = _fmax_343;
            float V2_1[1];
            float x2_1 = q2[cg * 32 + ln];
            float y2_1 = q2[(8 + cg) * 32 + lnr];
            float _min_307 = fminf(x2_1, y2_1);
            float lo2_1 = _min_307;
            float _fmax_344 = fmaxf(r2_1, lo2_1);
            r2_1 = _fmax_344;
            float _fmax_345 = fmaxf(x2_1, y2_1);
            float hi2_1 = _fmax_345;
            V2_1[0] = hi2_1;
            K_286 = V2_1[0];
            r_1 = r2_1;
        }
        rr2[0] = r_1;
        K2 = K_286;
    }
    if (tid_1 < 128) {
        unsigned int kk = __as_u32(K);
        unsigned int idx = 512;
        if (kk < 4278190080u) {
            idx = kk & 511;
        }
        if (gflag != 0) {
            unsigned int k2 = __as_u32(K2);
            idx = 512;
            if (k2 != 0) {
                idx = k2 & 511;
            }
        }
        unsigned int rkey = idx << 4 | (unsigned int)ln;
        int rank = 0;
        unsigned int _shfl_xor_142 = __shfl_xor_sync(0xFFFFFFFF, rkey, 1);
        unsigned int ox = _shfl_xor_142;
        if (ox < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_143 = __shfl_xor_sync(0xFFFFFFFF, rkey, 2);
        unsigned int ox_0 = _shfl_xor_143;
        if (ox_0 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_144 = __shfl_xor_sync(0xFFFFFFFF, rkey, 3);
        unsigned int ox_1 = _shfl_xor_144;
        if (ox_1 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_145 = __shfl_xor_sync(0xFFFFFFFF, rkey, 4);
        unsigned int ox_2 = _shfl_xor_145;
        if (ox_2 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_146 = __shfl_xor_sync(0xFFFFFFFF, rkey, 5);
        unsigned int ox_3 = _shfl_xor_146;
        if (ox_3 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_147 = __shfl_xor_sync(0xFFFFFFFF, rkey, 6);
        unsigned int ox_4 = _shfl_xor_147;
        if (ox_4 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_148 = __shfl_xor_sync(0xFFFFFFFF, rkey, 7);
        unsigned int ox_5 = _shfl_xor_148;
        if (ox_5 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_149 = __shfl_xor_sync(0xFFFFFFFF, rkey, 8);
        unsigned int ox_6 = _shfl_xor_149;
        if (ox_6 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_150 = __shfl_xor_sync(0xFFFFFFFF, rkey, 9);
        unsigned int ox_7 = _shfl_xor_150;
        if (ox_7 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_151 = __shfl_xor_sync(0xFFFFFFFF, rkey, 10);
        unsigned int ox_8 = _shfl_xor_151;
        if (ox_8 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_152 = __shfl_xor_sync(0xFFFFFFFF, rkey, 11);
        unsigned int ox_9 = _shfl_xor_152;
        if (ox_9 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_153 = __shfl_xor_sync(0xFFFFFFFF, rkey, 12);
        unsigned int ox_10 = _shfl_xor_153;
        if (ox_10 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_154 = __shfl_xor_sync(0xFFFFFFFF, rkey, 13);
        unsigned int ox_11 = _shfl_xor_154;
        if (ox_11 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_155 = __shfl_xor_sync(0xFFFFFFFF, rkey, 14);
        unsigned int ox_12 = _shfl_xor_155;
        if (ox_12 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_156 = __shfl_xor_sync(0xFFFFFFFF, rkey, 15);
        unsigned int ox_13 = _shfl_xor_156;
        if (ox_13 < rkey) {
            rank = rank + 1;
        }
        int val = -1;
        if (idx <= 511) {
            val = (int)idx;
        }
        if (commit != 0) {
            long long obase = ((long long)ucol * (long long)num_heads + (long long)head) * 16;
            out[obase + (long long)rank] = val;
        }
    }
}

} // extern "C"
