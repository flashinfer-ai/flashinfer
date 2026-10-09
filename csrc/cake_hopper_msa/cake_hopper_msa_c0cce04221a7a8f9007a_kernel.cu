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
#define SMEM_QCOL_STAGE_BYTES 64
#define SMEM_QCOL_STRIDE 64
#define SMEM_FLAGW_OFF 64
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_CCNT_OFF 17488
#define SMEM_CCNT_STAGE_BYTES 64
#define SMEM_CCNT_STRIDE 64
#define SMEM_LIMC_OFF 17552
#define SMEM_LIMC_STAGE_BYTES 64
#define SMEM_LIMC_STRIDE 64
#define SMEM_CBUF_OFF 17616
#define SMEM_CBUF_STAGE_BYTES 1024
#define SMEM_CBUF_STRIDE 1024
#define SMEM_PUB_OFF 80
#define SMEM_PUB_STAGE_BYTES 17408
#define SMEM_PUB_STRIDE 17408
#define SMEM_TOTAL 18688
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
kernel_cake_hopper_msa_c0cce04221a7a8f9007a(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 64);
    const int flagw_addr = smem + 64;
    unsigned int* ccnt = reinterpret_cast<unsigned int*>(smem_raw + 17488);
    const int ccnt_addr = smem + 17488;
    int* limc = reinterpret_cast<int*>(smem_raw + 17552);
    const int limc_addr = smem + 17552;
    unsigned int* cbuf = reinterpret_cast<unsigned int*>(smem_raw + 17616);
    const int cbuf_addr = smem + 17616;
    float* pub = reinterpret_cast<float*>(smem_raw + 80);
    const int pub_addr = smem + 80;

    // === Task calls (dependency order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1 / 16;
    int c = tid_1 - w * 16;
    int whi = w / 2;
    int col = blockIdx.x * 16 + c;
    int head = blockIdx.y;
    if (tid_1 == 0) {
        flagw[0] = 0;
    }
    if (tid_1 == 0) {
        flagw[1] = 0;
    }
    if (tid_1 < 16) {
        ccnt[tid_1] = 0;
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
    int R = num_chunks * 32;
    int tb = w * R;
    unsigned int cbt[32];
    float kbt[32];
    long long p = cbase + (long long)tb * nq64;
    cbt[0] = 4286578688;
    if (lim_full > tb) {
        cbt[0] = S[p];
    }
    cbt[1] = 4286578688;
    if (lim_full > tb + 1) {
        cbt[1] = S[p + nq64];
    }
    cbt[2] = 4286578688;
    if (lim_full > tb + 2) {
        cbt[2] = S[p + 2 * nq64];
    }
    cbt[3] = 4286578688;
    if (lim_full > tb + 3) {
        cbt[3] = S[p + 3 * nq64];
    }
    cbt[4] = 4286578688;
    if (lim_full > tb + 4) {
        cbt[4] = S[p + 4 * nq64];
    }
    cbt[5] = 4286578688;
    if (lim_full > tb + 5) {
        cbt[5] = S[p + 5 * nq64];
    }
    cbt[6] = 4286578688;
    if (lim_full > tb + 6) {
        cbt[6] = S[p + 6 * nq64];
    }
    cbt[7] = 4286578688;
    if (lim_full > tb + 7) {
        cbt[7] = S[p + 7 * nq64];
    }
    cbt[8] = 4286578688;
    if (lim_full > tb + 8) {
        cbt[8] = S[p + 8 * nq64];
    }
    cbt[9] = 4286578688;
    if (lim_full > tb + 9) {
        cbt[9] = S[p + 9 * nq64];
    }
    cbt[10] = 4286578688;
    if (lim_full > tb + 10) {
        cbt[10] = S[p + 10 * nq64];
    }
    cbt[11] = 4286578688;
    if (lim_full > tb + 11) {
        cbt[11] = S[p + 11 * nq64];
    }
    cbt[12] = 4286578688;
    if (lim_full > tb + 12) {
        cbt[12] = S[p + 12 * nq64];
    }
    cbt[13] = 4286578688;
    if (lim_full > tb + 13) {
        cbt[13] = S[p + 13 * nq64];
    }
    cbt[14] = 4286578688;
    if (lim_full > tb + 14) {
        cbt[14] = S[p + 14 * nq64];
    }
    cbt[15] = 4286578688;
    if (lim_full > tb + 15) {
        cbt[15] = S[p + 15 * nq64];
    }
    cbt[16] = 4286578688;
    if (lim_full > tb + 16) {
        cbt[16] = S[p + 16 * nq64];
    }
    cbt[17] = 4286578688;
    if (lim_full > tb + 17) {
        cbt[17] = S[p + 17 * nq64];
    }
    cbt[18] = 4286578688;
    if (lim_full > tb + 18) {
        cbt[18] = S[p + 18 * nq64];
    }
    cbt[19] = 4286578688;
    if (lim_full > tb + 19) {
        cbt[19] = S[p + 19 * nq64];
    }
    cbt[20] = 4286578688;
    if (lim_full > tb + 20) {
        cbt[20] = S[p + 20 * nq64];
    }
    cbt[21] = 4286578688;
    if (lim_full > tb + 21) {
        cbt[21] = S[p + 21 * nq64];
    }
    cbt[22] = 4286578688;
    if (lim_full > tb + 22) {
        cbt[22] = S[p + 22 * nq64];
    }
    cbt[23] = 4286578688;
    if (lim_full > tb + 23) {
        cbt[23] = S[p + 23 * nq64];
    }
    cbt[24] = 4286578688;
    if (lim_full > tb + 24) {
        cbt[24] = S[p + 24 * nq64];
    }
    cbt[25] = 4286578688;
    if (lim_full > tb + 25) {
        cbt[25] = S[p + 25 * nq64];
    }
    cbt[26] = 4286578688;
    if (lim_full > tb + 26) {
        cbt[26] = S[p + 26 * nq64];
    }
    cbt[27] = 4286578688;
    if (lim_full > tb + 27) {
        cbt[27] = S[p + 27 * nq64];
    }
    cbt[28] = 4286578688;
    if (lim_full > tb + 28) {
        cbt[28] = S[p + 28 * nq64];
    }
    cbt[29] = 4286578688;
    if (lim_full > tb + 29) {
        cbt[29] = S[p + 29 * nq64];
    }
    cbt[30] = 4286578688;
    if (lim_full > tb + 30) {
        cbt[30] = S[p + 30 * nq64];
    }
    cbt[31] = 4286578688;
    if (lim_full > tb + 31) {
        cbt[31] = S[p + 31 * nq64];
    }
    if (lim <= tb) {
        cbt[0] = 4286578688;
    }
    if (lim <= tb + 1) {
        cbt[1] = 4286578688;
    }
    if (lim <= tb + 2) {
        cbt[2] = 4286578688;
    }
    if (lim <= tb + 3) {
        cbt[3] = 4286578688;
    }
    if (lim <= tb + 4) {
        cbt[4] = 4286578688;
    }
    if (lim <= tb + 5) {
        cbt[5] = 4286578688;
    }
    if (lim <= tb + 6) {
        cbt[6] = 4286578688;
    }
    if (lim <= tb + 7) {
        cbt[7] = 4286578688;
    }
    if (lim <= tb + 8) {
        cbt[8] = 4286578688;
    }
    if (lim <= tb + 9) {
        cbt[9] = 4286578688;
    }
    if (lim <= tb + 10) {
        cbt[10] = 4286578688;
    }
    if (lim <= tb + 11) {
        cbt[11] = 4286578688;
    }
    if (lim <= tb + 12) {
        cbt[12] = 4286578688;
    }
    if (lim <= tb + 13) {
        cbt[13] = 4286578688;
    }
    if (lim <= tb + 14) {
        cbt[14] = 4286578688;
    }
    if (lim <= tb + 15) {
        cbt[15] = 4286578688;
    }
    if (lim <= tb + 16) {
        cbt[16] = 4286578688;
    }
    if (lim <= tb + 17) {
        cbt[17] = 4286578688;
    }
    if (lim <= tb + 18) {
        cbt[18] = 4286578688;
    }
    if (lim <= tb + 19) {
        cbt[19] = 4286578688;
    }
    if (lim <= tb + 20) {
        cbt[20] = 4286578688;
    }
    if (lim <= tb + 21) {
        cbt[21] = 4286578688;
    }
    if (lim <= tb + 22) {
        cbt[22] = 4286578688;
    }
    if (lim <= tb + 23) {
        cbt[23] = 4286578688;
    }
    if (lim <= tb + 24) {
        cbt[24] = 4286578688;
    }
    if (lim <= tb + 25) {
        cbt[25] = 4286578688;
    }
    if (lim <= tb + 26) {
        cbt[26] = 4286578688;
    }
    if (lim <= tb + 27) {
        cbt[27] = 4286578688;
    }
    if (lim <= tb + 28) {
        cbt[28] = 4286578688;
    }
    if (lim <= tb + 29) {
        cbt[29] = 4286578688;
    }
    if (lim <= tb + 30) {
        cbt[30] = 4286578688;
    }
    if (lim <= tb + 31) {
        cbt[31] = 4286578688;
    }
    float sc = __uint_as_float(cbt[0]);
    float _fmax_0 = fmaxf(sc, -1.7014118346046923e+38f);
    sc = _fmax_0;
    float _min_0 = fminf(sc, 1.7014118346046923e+38f);
    sc = _min_0;
    sc = sc;
    float sc1 = sc;
    unsigned int key = __as_u32(sc1) & 4294966784u | (unsigned int)tb;
    kbt[0] = __uint_as_float(key);
    float sc_0 = __uint_as_float(cbt[1]);
    float _fmax_1 = fmaxf(sc_0, -1.7014118346046923e+38f);
    sc_0 = _fmax_1;
    float _min_1 = fminf(sc_0, 1.7014118346046923e+38f);
    sc_0 = _min_1;
    sc_0 = sc_0;
    float sc1_1 = sc_0;
    unsigned int key_2 = __as_u32(sc1_1) & 4294966784u | (unsigned int)(tb + 1);
    kbt[1] = __uint_as_float(key_2);
    float sc_3 = __uint_as_float(cbt[2]);
    float _fmax_2 = fmaxf(sc_3, -1.7014118346046923e+38f);
    sc_3 = _fmax_2;
    float _min_2 = fminf(sc_3, 1.7014118346046923e+38f);
    sc_3 = _min_2;
    sc_3 = sc_3;
    float sc1_4 = sc_3;
    unsigned int key_5 = __as_u32(sc1_4) & 4294966784u | (unsigned int)(tb + 2);
    kbt[2] = __uint_as_float(key_5);
    float sc_6 = __uint_as_float(cbt[3]);
    float _fmax_3 = fmaxf(sc_6, -1.7014118346046923e+38f);
    sc_6 = _fmax_3;
    float _min_3 = fminf(sc_6, 1.7014118346046923e+38f);
    sc_6 = _min_3;
    sc_6 = sc_6;
    float sc1_7 = sc_6;
    unsigned int key_8 = __as_u32(sc1_7) & 4294966784u | (unsigned int)(tb + 3);
    kbt[3] = __uint_as_float(key_8);
    float sc_9 = __uint_as_float(cbt[4]);
    float _fmax_4 = fmaxf(sc_9, -1.7014118346046923e+38f);
    sc_9 = _fmax_4;
    float _min_4 = fminf(sc_9, 1.7014118346046923e+38f);
    sc_9 = _min_4;
    sc_9 = sc_9;
    float sc1_10 = sc_9;
    unsigned int key_11 = __as_u32(sc1_10) & 4294966784u | (unsigned int)(tb + 4);
    kbt[4] = __uint_as_float(key_11);
    float sc_12 = __uint_as_float(cbt[5]);
    float _fmax_5 = fmaxf(sc_12, -1.7014118346046923e+38f);
    sc_12 = _fmax_5;
    float _min_5 = fminf(sc_12, 1.7014118346046923e+38f);
    sc_12 = _min_5;
    sc_12 = sc_12;
    float sc1_13 = sc_12;
    unsigned int key_14 = __as_u32(sc1_13) & 4294966784u | (unsigned int)(tb + 5);
    kbt[5] = __uint_as_float(key_14);
    float sc_15 = __uint_as_float(cbt[6]);
    float _fmax_6 = fmaxf(sc_15, -1.7014118346046923e+38f);
    sc_15 = _fmax_6;
    float _min_6 = fminf(sc_15, 1.7014118346046923e+38f);
    sc_15 = _min_6;
    sc_15 = sc_15;
    float sc1_16 = sc_15;
    unsigned int key_17 = __as_u32(sc1_16) & 4294966784u | (unsigned int)(tb + 6);
    kbt[6] = __uint_as_float(key_17);
    float sc_18 = __uint_as_float(cbt[7]);
    float _fmax_7 = fmaxf(sc_18, -1.7014118346046923e+38f);
    sc_18 = _fmax_7;
    float _min_7 = fminf(sc_18, 1.7014118346046923e+38f);
    sc_18 = _min_7;
    sc_18 = sc_18;
    float sc1_19 = sc_18;
    unsigned int key_20 = __as_u32(sc1_19) & 4294966784u | (unsigned int)(tb + 7);
    kbt[7] = __uint_as_float(key_20);
    float sc_21 = __uint_as_float(cbt[8]);
    float _fmax_8 = fmaxf(sc_21, -1.7014118346046923e+38f);
    sc_21 = _fmax_8;
    float _min_8 = fminf(sc_21, 1.7014118346046923e+38f);
    sc_21 = _min_8;
    sc_21 = sc_21;
    float sc1_22 = sc_21;
    unsigned int key_23 = __as_u32(sc1_22) & 4294966784u | (unsigned int)(tb + 8);
    kbt[8] = __uint_as_float(key_23);
    float sc_24 = __uint_as_float(cbt[9]);
    float _fmax_9 = fmaxf(sc_24, -1.7014118346046923e+38f);
    sc_24 = _fmax_9;
    float _min_9 = fminf(sc_24, 1.7014118346046923e+38f);
    sc_24 = _min_9;
    sc_24 = sc_24;
    float sc1_25 = sc_24;
    unsigned int key_26 = __as_u32(sc1_25) & 4294966784u | (unsigned int)(tb + 9);
    kbt[9] = __uint_as_float(key_26);
    float sc_27 = __uint_as_float(cbt[10]);
    float _fmax_10 = fmaxf(sc_27, -1.7014118346046923e+38f);
    sc_27 = _fmax_10;
    float _min_10 = fminf(sc_27, 1.7014118346046923e+38f);
    sc_27 = _min_10;
    sc_27 = sc_27;
    float sc1_28 = sc_27;
    unsigned int key_29 = __as_u32(sc1_28) & 4294966784u | (unsigned int)(tb + 10);
    kbt[10] = __uint_as_float(key_29);
    float sc_30 = __uint_as_float(cbt[11]);
    float _fmax_11 = fmaxf(sc_30, -1.7014118346046923e+38f);
    sc_30 = _fmax_11;
    float _min_11 = fminf(sc_30, 1.7014118346046923e+38f);
    sc_30 = _min_11;
    sc_30 = sc_30;
    float sc1_31 = sc_30;
    unsigned int key_32 = __as_u32(sc1_31) & 4294966784u | (unsigned int)(tb + 11);
    kbt[11] = __uint_as_float(key_32);
    float sc_33 = __uint_as_float(cbt[12]);
    float _fmax_12 = fmaxf(sc_33, -1.7014118346046923e+38f);
    sc_33 = _fmax_12;
    float _min_12 = fminf(sc_33, 1.7014118346046923e+38f);
    sc_33 = _min_12;
    sc_33 = sc_33;
    float sc1_34 = sc_33;
    unsigned int key_35 = __as_u32(sc1_34) & 4294966784u | (unsigned int)(tb + 12);
    kbt[12] = __uint_as_float(key_35);
    float sc_36 = __uint_as_float(cbt[13]);
    float _fmax_13 = fmaxf(sc_36, -1.7014118346046923e+38f);
    sc_36 = _fmax_13;
    float _min_13 = fminf(sc_36, 1.7014118346046923e+38f);
    sc_36 = _min_13;
    sc_36 = sc_36;
    float sc1_37 = sc_36;
    unsigned int key_38 = __as_u32(sc1_37) & 4294966784u | (unsigned int)(tb + 13);
    kbt[13] = __uint_as_float(key_38);
    float sc_39 = __uint_as_float(cbt[14]);
    float _fmax_14 = fmaxf(sc_39, -1.7014118346046923e+38f);
    sc_39 = _fmax_14;
    float _min_14 = fminf(sc_39, 1.7014118346046923e+38f);
    sc_39 = _min_14;
    sc_39 = sc_39;
    float sc1_40 = sc_39;
    unsigned int key_41 = __as_u32(sc1_40) & 4294966784u | (unsigned int)(tb + 14);
    kbt[14] = __uint_as_float(key_41);
    float sc_42 = __uint_as_float(cbt[15]);
    float _fmax_15 = fmaxf(sc_42, -1.7014118346046923e+38f);
    sc_42 = _fmax_15;
    float _min_15 = fminf(sc_42, 1.7014118346046923e+38f);
    sc_42 = _min_15;
    sc_42 = sc_42;
    float sc1_43 = sc_42;
    unsigned int key_44 = __as_u32(sc1_43) & 4294966784u | (unsigned int)(tb + 15);
    kbt[15] = __uint_as_float(key_44);
    float sc_45 = __uint_as_float(cbt[16]);
    float _fmax_16 = fmaxf(sc_45, -1.7014118346046923e+38f);
    sc_45 = _fmax_16;
    float _min_16 = fminf(sc_45, 1.7014118346046923e+38f);
    sc_45 = _min_16;
    sc_45 = sc_45;
    float sc1_46 = sc_45;
    unsigned int key_47 = __as_u32(sc1_46) & 4294966784u | (unsigned int)(tb + 16);
    kbt[16] = __uint_as_float(key_47);
    float sc_48 = __uint_as_float(cbt[17]);
    float _fmax_17 = fmaxf(sc_48, -1.7014118346046923e+38f);
    sc_48 = _fmax_17;
    float _min_17 = fminf(sc_48, 1.7014118346046923e+38f);
    sc_48 = _min_17;
    sc_48 = sc_48;
    float sc1_49 = sc_48;
    unsigned int key_50 = __as_u32(sc1_49) & 4294966784u | (unsigned int)(tb + 17);
    kbt[17] = __uint_as_float(key_50);
    float sc_51 = __uint_as_float(cbt[18]);
    float _fmax_18 = fmaxf(sc_51, -1.7014118346046923e+38f);
    sc_51 = _fmax_18;
    float _min_18 = fminf(sc_51, 1.7014118346046923e+38f);
    sc_51 = _min_18;
    sc_51 = sc_51;
    float sc1_52 = sc_51;
    unsigned int key_53 = __as_u32(sc1_52) & 4294966784u | (unsigned int)(tb + 18);
    kbt[18] = __uint_as_float(key_53);
    float sc_54 = __uint_as_float(cbt[19]);
    float _fmax_19 = fmaxf(sc_54, -1.7014118346046923e+38f);
    sc_54 = _fmax_19;
    float _min_19 = fminf(sc_54, 1.7014118346046923e+38f);
    sc_54 = _min_19;
    sc_54 = sc_54;
    float sc1_55 = sc_54;
    unsigned int key_56 = __as_u32(sc1_55) & 4294966784u | (unsigned int)(tb + 19);
    kbt[19] = __uint_as_float(key_56);
    float sc_57 = __uint_as_float(cbt[20]);
    float _fmax_20 = fmaxf(sc_57, -1.7014118346046923e+38f);
    sc_57 = _fmax_20;
    float _min_20 = fminf(sc_57, 1.7014118346046923e+38f);
    sc_57 = _min_20;
    sc_57 = sc_57;
    float sc1_58 = sc_57;
    unsigned int key_59 = __as_u32(sc1_58) & 4294966784u | (unsigned int)(tb + 20);
    kbt[20] = __uint_as_float(key_59);
    float sc_60 = __uint_as_float(cbt[21]);
    float _fmax_21 = fmaxf(sc_60, -1.7014118346046923e+38f);
    sc_60 = _fmax_21;
    float _min_21 = fminf(sc_60, 1.7014118346046923e+38f);
    sc_60 = _min_21;
    sc_60 = sc_60;
    float sc1_61 = sc_60;
    unsigned int key_62 = __as_u32(sc1_61) & 4294966784u | (unsigned int)(tb + 21);
    kbt[21] = __uint_as_float(key_62);
    float sc_63 = __uint_as_float(cbt[22]);
    float _fmax_22 = fmaxf(sc_63, -1.7014118346046923e+38f);
    sc_63 = _fmax_22;
    float _min_22 = fminf(sc_63, 1.7014118346046923e+38f);
    sc_63 = _min_22;
    sc_63 = sc_63;
    float sc1_64 = sc_63;
    unsigned int key_65 = __as_u32(sc1_64) & 4294966784u | (unsigned int)(tb + 22);
    kbt[22] = __uint_as_float(key_65);
    float sc_66 = __uint_as_float(cbt[23]);
    float _fmax_23 = fmaxf(sc_66, -1.7014118346046923e+38f);
    sc_66 = _fmax_23;
    float _min_23 = fminf(sc_66, 1.7014118346046923e+38f);
    sc_66 = _min_23;
    sc_66 = sc_66;
    float sc1_67 = sc_66;
    unsigned int key_68 = __as_u32(sc1_67) & 4294966784u | (unsigned int)(tb + 23);
    kbt[23] = __uint_as_float(key_68);
    float sc_69 = __uint_as_float(cbt[24]);
    float _fmax_24 = fmaxf(sc_69, -1.7014118346046923e+38f);
    sc_69 = _fmax_24;
    float _min_24 = fminf(sc_69, 1.7014118346046923e+38f);
    sc_69 = _min_24;
    sc_69 = sc_69;
    float sc1_70 = sc_69;
    unsigned int key_71 = __as_u32(sc1_70) & 4294966784u | (unsigned int)(tb + 24);
    kbt[24] = __uint_as_float(key_71);
    float sc_72 = __uint_as_float(cbt[25]);
    float _fmax_25 = fmaxf(sc_72, -1.7014118346046923e+38f);
    sc_72 = _fmax_25;
    float _min_25 = fminf(sc_72, 1.7014118346046923e+38f);
    sc_72 = _min_25;
    sc_72 = sc_72;
    float sc1_73 = sc_72;
    unsigned int key_74 = __as_u32(sc1_73) & 4294966784u | (unsigned int)(tb + 25);
    kbt[25] = __uint_as_float(key_74);
    float sc_75 = __uint_as_float(cbt[26]);
    float _fmax_26 = fmaxf(sc_75, -1.7014118346046923e+38f);
    sc_75 = _fmax_26;
    float _min_26 = fminf(sc_75, 1.7014118346046923e+38f);
    sc_75 = _min_26;
    sc_75 = sc_75;
    float sc1_76 = sc_75;
    unsigned int key_77 = __as_u32(sc1_76) & 4294966784u | (unsigned int)(tb + 26);
    kbt[26] = __uint_as_float(key_77);
    float sc_78 = __uint_as_float(cbt[27]);
    float _fmax_27 = fmaxf(sc_78, -1.7014118346046923e+38f);
    sc_78 = _fmax_27;
    float _min_27 = fminf(sc_78, 1.7014118346046923e+38f);
    sc_78 = _min_27;
    sc_78 = sc_78;
    float sc1_79 = sc_78;
    unsigned int key_80 = __as_u32(sc1_79) & 4294966784u | (unsigned int)(tb + 27);
    kbt[27] = __uint_as_float(key_80);
    float sc_81 = __uint_as_float(cbt[28]);
    float _fmax_28 = fmaxf(sc_81, -1.7014118346046923e+38f);
    sc_81 = _fmax_28;
    float _min_28 = fminf(sc_81, 1.7014118346046923e+38f);
    sc_81 = _min_28;
    sc_81 = sc_81;
    float sc1_82 = sc_81;
    unsigned int key_83 = __as_u32(sc1_82) & 4294966784u | (unsigned int)(tb + 28);
    kbt[28] = __uint_as_float(key_83);
    float sc_84 = __uint_as_float(cbt[29]);
    float _fmax_29 = fmaxf(sc_84, -1.7014118346046923e+38f);
    sc_84 = _fmax_29;
    float _min_29 = fminf(sc_84, 1.7014118346046923e+38f);
    sc_84 = _min_29;
    sc_84 = sc_84;
    float sc1_85 = sc_84;
    unsigned int key_86 = __as_u32(sc1_85) & 4294966784u | (unsigned int)(tb + 29);
    kbt[29] = __uint_as_float(key_86);
    float sc_87 = __uint_as_float(cbt[30]);
    float _fmax_30 = fmaxf(sc_87, -1.7014118346046923e+38f);
    sc_87 = _fmax_30;
    float _min_30 = fminf(sc_87, 1.7014118346046923e+38f);
    sc_87 = _min_30;
    sc_87 = sc_87;
    float sc1_88 = sc_87;
    unsigned int key_89 = __as_u32(sc1_88) & 4294966784u | (unsigned int)(tb + 30);
    kbt[30] = __uint_as_float(key_89);
    float sc_90 = __uint_as_float(cbt[31]);
    float _fmax_31 = fmaxf(sc_90, -1.7014118346046923e+38f);
    sc_90 = _fmax_31;
    float _min_31 = fminf(sc_90, 1.7014118346046923e+38f);
    sc_90 = _min_31;
    sc_90 = sc_90;
    float sc1_91 = sc_90;
    unsigned int key_92 = __as_u32(sc1_91) & 4294966784u | (unsigned int)(tb + 31);
    kbt[31] = __uint_as_float(key_92);
    int f = 0;
    if ((tb < fb || tb + 32 > lim - fe) && tb < lim) {
        f = 1;
    }
    int fct = f;
    if (fct != 0) {
        int f_0 = 0;
        if ((tb < fb || tb >= lim - fe) && lim > tb) {
            f_0 = 1;
        }
        if (f_0 != 0) {
            kbt[0] = __uint_as_float(2139094528 | (unsigned int)tb);
        }
        int f_1 = 0;
        if ((tb + 1 < fb || tb + 1 >= lim - fe) && lim > tb + 1) {
            f_1 = 1;
        }
        if (f_1 != 0) {
            kbt[1] = __uint_as_float(2139094528 | (unsigned int)(tb + 1));
        }
        int f_2 = 0;
        if ((tb + 2 < fb || tb + 2 >= lim - fe) && lim > tb + 2) {
            f_2 = 1;
        }
        if (f_2 != 0) {
            kbt[2] = __uint_as_float(2139094528 | (unsigned int)(tb + 2));
        }
        int f_3 = 0;
        if ((tb + 3 < fb || tb + 3 >= lim - fe) && lim > tb + 3) {
            f_3 = 1;
        }
        if (f_3 != 0) {
            kbt[3] = __uint_as_float(2139094528 | (unsigned int)(tb + 3));
        }
        int f_4 = 0;
        if ((tb + 4 < fb || tb + 4 >= lim - fe) && lim > tb + 4) {
            f_4 = 1;
        }
        if (f_4 != 0) {
            kbt[4] = __uint_as_float(2139094528 | (unsigned int)(tb + 4));
        }
        int f_5 = 0;
        if ((tb + 5 < fb || tb + 5 >= lim - fe) && lim > tb + 5) {
            f_5 = 1;
        }
        if (f_5 != 0) {
            kbt[5] = __uint_as_float(2139094528 | (unsigned int)(tb + 5));
        }
        int f_6 = 0;
        if ((tb + 6 < fb || tb + 6 >= lim - fe) && lim > tb + 6) {
            f_6 = 1;
        }
        if (f_6 != 0) {
            kbt[6] = __uint_as_float(2139094528 | (unsigned int)(tb + 6));
        }
        int f_7 = 0;
        if ((tb + 7 < fb || tb + 7 >= lim - fe) && lim > tb + 7) {
            f_7 = 1;
        }
        if (f_7 != 0) {
            kbt[7] = __uint_as_float(2139094528 | (unsigned int)(tb + 7));
        }
        int f_8 = 0;
        if ((tb + 8 < fb || tb + 8 >= lim - fe) && lim > tb + 8) {
            f_8 = 1;
        }
        if (f_8 != 0) {
            kbt[8] = __uint_as_float(2139094528 | (unsigned int)(tb + 8));
        }
        int f_9 = 0;
        if ((tb + 9 < fb || tb + 9 >= lim - fe) && lim > tb + 9) {
            f_9 = 1;
        }
        if (f_9 != 0) {
            kbt[9] = __uint_as_float(2139094528 | (unsigned int)(tb + 9));
        }
        int f_10 = 0;
        if ((tb + 10 < fb || tb + 10 >= lim - fe) && lim > tb + 10) {
            f_10 = 1;
        }
        if (f_10 != 0) {
            kbt[10] = __uint_as_float(2139094528 | (unsigned int)(tb + 10));
        }
        int f_11 = 0;
        if ((tb + 11 < fb || tb + 11 >= lim - fe) && lim > tb + 11) {
            f_11 = 1;
        }
        if (f_11 != 0) {
            kbt[11] = __uint_as_float(2139094528 | (unsigned int)(tb + 11));
        }
        int f_12 = 0;
        if ((tb + 12 < fb || tb + 12 >= lim - fe) && lim > tb + 12) {
            f_12 = 1;
        }
        if (f_12 != 0) {
            kbt[12] = __uint_as_float(2139094528 | (unsigned int)(tb + 12));
        }
        int f_13 = 0;
        if ((tb + 13 < fb || tb + 13 >= lim - fe) && lim > tb + 13) {
            f_13 = 1;
        }
        if (f_13 != 0) {
            kbt[13] = __uint_as_float(2139094528 | (unsigned int)(tb + 13));
        }
        int f_14 = 0;
        if ((tb + 14 < fb || tb + 14 >= lim - fe) && lim > tb + 14) {
            f_14 = 1;
        }
        if (f_14 != 0) {
            kbt[14] = __uint_as_float(2139094528 | (unsigned int)(tb + 14));
        }
        int f_15 = 0;
        if ((tb + 15 < fb || tb + 15 >= lim - fe) && lim > tb + 15) {
            f_15 = 1;
        }
        if (f_15 != 0) {
            kbt[15] = __uint_as_float(2139094528 | (unsigned int)(tb + 15));
        }
        int f_16 = 0;
        if ((tb + 16 < fb || tb + 16 >= lim - fe) && lim > tb + 16) {
            f_16 = 1;
        }
        if (f_16 != 0) {
            kbt[16] = __uint_as_float(2139094528 | (unsigned int)(tb + 16));
        }
        int f_17 = 0;
        if ((tb + 17 < fb || tb + 17 >= lim - fe) && lim > tb + 17) {
            f_17 = 1;
        }
        if (f_17 != 0) {
            kbt[17] = __uint_as_float(2139094528 | (unsigned int)(tb + 17));
        }
        int f_18 = 0;
        if ((tb + 18 < fb || tb + 18 >= lim - fe) && lim > tb + 18) {
            f_18 = 1;
        }
        if (f_18 != 0) {
            kbt[18] = __uint_as_float(2139094528 | (unsigned int)(tb + 18));
        }
        int f_19 = 0;
        if ((tb + 19 < fb || tb + 19 >= lim - fe) && lim > tb + 19) {
            f_19 = 1;
        }
        if (f_19 != 0) {
            kbt[19] = __uint_as_float(2139094528 | (unsigned int)(tb + 19));
        }
        int f_20 = 0;
        if ((tb + 20 < fb || tb + 20 >= lim - fe) && lim > tb + 20) {
            f_20 = 1;
        }
        if (f_20 != 0) {
            kbt[20] = __uint_as_float(2139094528 | (unsigned int)(tb + 20));
        }
        int f_21 = 0;
        if ((tb + 21 < fb || tb + 21 >= lim - fe) && lim > tb + 21) {
            f_21 = 1;
        }
        if (f_21 != 0) {
            kbt[21] = __uint_as_float(2139094528 | (unsigned int)(tb + 21));
        }
        int f_22 = 0;
        if ((tb + 22 < fb || tb + 22 >= lim - fe) && lim > tb + 22) {
            f_22 = 1;
        }
        if (f_22 != 0) {
            kbt[22] = __uint_as_float(2139094528 | (unsigned int)(tb + 22));
        }
        int f_23 = 0;
        if ((tb + 23 < fb || tb + 23 >= lim - fe) && lim > tb + 23) {
            f_23 = 1;
        }
        if (f_23 != 0) {
            kbt[23] = __uint_as_float(2139094528 | (unsigned int)(tb + 23));
        }
        int f_24 = 0;
        if ((tb + 24 < fb || tb + 24 >= lim - fe) && lim > tb + 24) {
            f_24 = 1;
        }
        if (f_24 != 0) {
            kbt[24] = __uint_as_float(2139094528 | (unsigned int)(tb + 24));
        }
        int f_25 = 0;
        if ((tb + 25 < fb || tb + 25 >= lim - fe) && lim > tb + 25) {
            f_25 = 1;
        }
        if (f_25 != 0) {
            kbt[25] = __uint_as_float(2139094528 | (unsigned int)(tb + 25));
        }
        int f_26 = 0;
        if ((tb + 26 < fb || tb + 26 >= lim - fe) && lim > tb + 26) {
            f_26 = 1;
        }
        if (f_26 != 0) {
            kbt[26] = __uint_as_float(2139094528 | (unsigned int)(tb + 26));
        }
        int f_27 = 0;
        if ((tb + 27 < fb || tb + 27 >= lim - fe) && lim > tb + 27) {
            f_27 = 1;
        }
        if (f_27 != 0) {
            kbt[27] = __uint_as_float(2139094528 | (unsigned int)(tb + 27));
        }
        int f_28 = 0;
        if ((tb + 28 < fb || tb + 28 >= lim - fe) && lim > tb + 28) {
            f_28 = 1;
        }
        if (f_28 != 0) {
            kbt[28] = __uint_as_float(2139094528 | (unsigned int)(tb + 28));
        }
        int f_29 = 0;
        if ((tb + 29 < fb || tb + 29 >= lim - fe) && lim > tb + 29) {
            f_29 = 1;
        }
        if (f_29 != 0) {
            kbt[29] = __uint_as_float(2139094528 | (unsigned int)(tb + 29));
        }
        int f_30 = 0;
        if ((tb + 30 < fb || tb + 30 >= lim - fe) && lim > tb + 30) {
            f_30 = 1;
        }
        if (f_30 != 0) {
            kbt[30] = __uint_as_float(2139094528 | (unsigned int)(tb + 30));
        }
        int f_31 = 0;
        if ((tb + 31 < fb || tb + 31 >= lim - fe) && lim > tb + 31) {
            f_31 = 1;
        }
        if (f_31 != 0) {
            kbt[31] = __uint_as_float(2139094528 | (unsigned int)(tb + 31));
        }
    }
    float rr = rej;
    float _fmax_32 = fmaxf(kbt[0], kbt[13]);
    float hi = _fmax_32;
    float _min_32 = fminf(kbt[0], kbt[13]);
    float lo = _min_32;
    kbt[0] = hi;
    kbt[13] = lo;
    float _fmax_33 = fmaxf(kbt[1], kbt[12]);
    float hi_93 = _fmax_33;
    float _min_33 = fminf(kbt[1], kbt[12]);
    float lo_94 = _min_33;
    kbt[1] = hi_93;
    kbt[12] = lo_94;
    float _fmax_34 = fmaxf(kbt[2], kbt[15]);
    float hi_95 = _fmax_34;
    float _min_34 = fminf(kbt[2], kbt[15]);
    float lo_96 = _min_34;
    kbt[2] = hi_95;
    kbt[15] = lo_96;
    float _fmax_35 = fmaxf(kbt[3], kbt[14]);
    float hi_97 = _fmax_35;
    float _min_35 = fminf(kbt[3], kbt[14]);
    float lo_98 = _min_35;
    kbt[3] = hi_97;
    kbt[14] = lo_98;
    float _fmax_36 = fmaxf(kbt[4], kbt[8]);
    float hi_99 = _fmax_36;
    float _min_36 = fminf(kbt[4], kbt[8]);
    float lo_100 = _min_36;
    kbt[4] = hi_99;
    kbt[8] = lo_100;
    float _fmax_37 = fmaxf(kbt[5], kbt[6]);
    float hi_101 = _fmax_37;
    float _min_37 = fminf(kbt[5], kbt[6]);
    float lo_102 = _min_37;
    kbt[5] = hi_101;
    kbt[6] = lo_102;
    float _fmax_38 = fmaxf(kbt[7], kbt[11]);
    float hi_103 = _fmax_38;
    float _min_38 = fminf(kbt[7], kbt[11]);
    float lo_104 = _min_38;
    kbt[7] = hi_103;
    kbt[11] = lo_104;
    float _fmax_39 = fmaxf(kbt[9], kbt[10]);
    float hi_105 = _fmax_39;
    float _min_39 = fminf(kbt[9], kbt[10]);
    float lo_106 = _min_39;
    kbt[9] = hi_105;
    kbt[10] = lo_106;
    float _fmax_40 = fmaxf(kbt[0], kbt[5]);
    float hi_107 = _fmax_40;
    float _min_40 = fminf(kbt[0], kbt[5]);
    float lo_108 = _min_40;
    kbt[0] = hi_107;
    kbt[5] = lo_108;
    float _fmax_41 = fmaxf(kbt[1], kbt[7]);
    float hi_109 = _fmax_41;
    float _min_41 = fminf(kbt[1], kbt[7]);
    float lo_110 = _min_41;
    kbt[1] = hi_109;
    kbt[7] = lo_110;
    float _fmax_42 = fmaxf(kbt[2], kbt[9]);
    float hi_111 = _fmax_42;
    float _min_42 = fminf(kbt[2], kbt[9]);
    float lo_112 = _min_42;
    kbt[2] = hi_111;
    kbt[9] = lo_112;
    float _fmax_43 = fmaxf(kbt[3], kbt[4]);
    float hi_113 = _fmax_43;
    float _min_43 = fminf(kbt[3], kbt[4]);
    float lo_114 = _min_43;
    kbt[3] = hi_113;
    kbt[4] = lo_114;
    float _fmax_44 = fmaxf(kbt[6], kbt[13]);
    float hi_115 = _fmax_44;
    float _min_44 = fminf(kbt[6], kbt[13]);
    float lo_116 = _min_44;
    kbt[6] = hi_115;
    kbt[13] = lo_116;
    float _fmax_45 = fmaxf(kbt[8], kbt[14]);
    float hi_117 = _fmax_45;
    float _min_45 = fminf(kbt[8], kbt[14]);
    float lo_118 = _min_45;
    kbt[8] = hi_117;
    kbt[14] = lo_118;
    float _fmax_46 = fmaxf(kbt[10], kbt[15]);
    float hi_119 = _fmax_46;
    float _min_46 = fminf(kbt[10], kbt[15]);
    float lo_120 = _min_46;
    kbt[10] = hi_119;
    kbt[15] = lo_120;
    float _fmax_47 = fmaxf(kbt[11], kbt[12]);
    float hi_121 = _fmax_47;
    float _min_47 = fminf(kbt[11], kbt[12]);
    float lo_122 = _min_47;
    kbt[11] = hi_121;
    kbt[12] = lo_122;
    float _fmax_48 = fmaxf(kbt[0], kbt[1]);
    float hi_123 = _fmax_48;
    float _min_48 = fminf(kbt[0], kbt[1]);
    float lo_124 = _min_48;
    kbt[0] = hi_123;
    kbt[1] = lo_124;
    float _fmax_49 = fmaxf(kbt[2], kbt[3]);
    float hi_125 = _fmax_49;
    float _min_49 = fminf(kbt[2], kbt[3]);
    float lo_126 = _min_49;
    kbt[2] = hi_125;
    kbt[3] = lo_126;
    float _fmax_50 = fmaxf(kbt[4], kbt[5]);
    float hi_127 = _fmax_50;
    float _min_50 = fminf(kbt[4], kbt[5]);
    float lo_128 = _min_50;
    kbt[4] = hi_127;
    kbt[5] = lo_128;
    float _fmax_51 = fmaxf(kbt[6], kbt[8]);
    float hi_129 = _fmax_51;
    float _min_51 = fminf(kbt[6], kbt[8]);
    float lo_130 = _min_51;
    kbt[6] = hi_129;
    kbt[8] = lo_130;
    float _fmax_52 = fmaxf(kbt[7], kbt[9]);
    float hi_131 = _fmax_52;
    float _min_52 = fminf(kbt[7], kbt[9]);
    float lo_132 = _min_52;
    kbt[7] = hi_131;
    kbt[9] = lo_132;
    float _fmax_53 = fmaxf(kbt[10], kbt[11]);
    float hi_133 = _fmax_53;
    float _min_53 = fminf(kbt[10], kbt[11]);
    float lo_134 = _min_53;
    kbt[10] = hi_133;
    kbt[11] = lo_134;
    float _fmax_54 = fmaxf(kbt[12], kbt[13]);
    float hi_135 = _fmax_54;
    float _min_54 = fminf(kbt[12], kbt[13]);
    float lo_136 = _min_54;
    kbt[12] = hi_135;
    kbt[13] = lo_136;
    float _fmax_55 = fmaxf(kbt[14], kbt[15]);
    float hi_137 = _fmax_55;
    float _min_55 = fminf(kbt[14], kbt[15]);
    float lo_138 = _min_55;
    kbt[14] = hi_137;
    kbt[15] = lo_138;
    float _fmax_56 = fmaxf(kbt[0], kbt[2]);
    float hi_139 = _fmax_56;
    float _min_56 = fminf(kbt[0], kbt[2]);
    float lo_140 = _min_56;
    kbt[0] = hi_139;
    kbt[2] = lo_140;
    float _fmax_57 = fmaxf(kbt[1], kbt[3]);
    float hi_141 = _fmax_57;
    float _min_57 = fminf(kbt[1], kbt[3]);
    float lo_142 = _min_57;
    kbt[1] = hi_141;
    kbt[3] = lo_142;
    float _fmax_58 = fmaxf(kbt[4], kbt[10]);
    float hi_143 = _fmax_58;
    float _min_58 = fminf(kbt[4], kbt[10]);
    float lo_144 = _min_58;
    kbt[4] = hi_143;
    kbt[10] = lo_144;
    float _fmax_59 = fmaxf(kbt[5], kbt[11]);
    float hi_145 = _fmax_59;
    float _min_59 = fminf(kbt[5], kbt[11]);
    float lo_146 = _min_59;
    kbt[5] = hi_145;
    kbt[11] = lo_146;
    float _fmax_60 = fmaxf(kbt[6], kbt[7]);
    float hi_147 = _fmax_60;
    float _min_60 = fminf(kbt[6], kbt[7]);
    float lo_148 = _min_60;
    kbt[6] = hi_147;
    kbt[7] = lo_148;
    float _fmax_61 = fmaxf(kbt[8], kbt[9]);
    float hi_149 = _fmax_61;
    float _min_61 = fminf(kbt[8], kbt[9]);
    float lo_150 = _min_61;
    kbt[8] = hi_149;
    kbt[9] = lo_150;
    float _fmax_62 = fmaxf(kbt[12], kbt[14]);
    float hi_151 = _fmax_62;
    float _min_62 = fminf(kbt[12], kbt[14]);
    float lo_152 = _min_62;
    kbt[12] = hi_151;
    kbt[14] = lo_152;
    float _fmax_63 = fmaxf(kbt[13], kbt[15]);
    float hi_153 = _fmax_63;
    float _min_63 = fminf(kbt[13], kbt[15]);
    float lo_154 = _min_63;
    kbt[13] = hi_153;
    kbt[15] = lo_154;
    float _fmax_64 = fmaxf(kbt[1], kbt[2]);
    float hi_155 = _fmax_64;
    float _min_64 = fminf(kbt[1], kbt[2]);
    float lo_156 = _min_64;
    kbt[1] = hi_155;
    kbt[2] = lo_156;
    float _fmax_65 = fmaxf(kbt[3], kbt[12]);
    float hi_157 = _fmax_65;
    float _min_65 = fminf(kbt[3], kbt[12]);
    float lo_158 = _min_65;
    kbt[3] = hi_157;
    kbt[12] = lo_158;
    float _fmax_66 = fmaxf(kbt[4], kbt[6]);
    float hi_159 = _fmax_66;
    float _min_66 = fminf(kbt[4], kbt[6]);
    float lo_160 = _min_66;
    kbt[4] = hi_159;
    kbt[6] = lo_160;
    float _fmax_67 = fmaxf(kbt[5], kbt[7]);
    float hi_161 = _fmax_67;
    float _min_67 = fminf(kbt[5], kbt[7]);
    float lo_162 = _min_67;
    kbt[5] = hi_161;
    kbt[7] = lo_162;
    float _fmax_68 = fmaxf(kbt[8], kbt[10]);
    float hi_163 = _fmax_68;
    float _min_68 = fminf(kbt[8], kbt[10]);
    float lo_164 = _min_68;
    kbt[8] = hi_163;
    kbt[10] = lo_164;
    float _fmax_69 = fmaxf(kbt[9], kbt[11]);
    float hi_165 = _fmax_69;
    float _min_69 = fminf(kbt[9], kbt[11]);
    float lo_166 = _min_69;
    kbt[9] = hi_165;
    kbt[11] = lo_166;
    float _fmax_70 = fmaxf(kbt[13], kbt[14]);
    float hi_167 = _fmax_70;
    float _min_70 = fminf(kbt[13], kbt[14]);
    float lo_168 = _min_70;
    kbt[13] = hi_167;
    kbt[14] = lo_168;
    float _fmax_71 = fmaxf(kbt[1], kbt[4]);
    float hi_169 = _fmax_71;
    float _min_71 = fminf(kbt[1], kbt[4]);
    float lo_170 = _min_71;
    kbt[1] = hi_169;
    kbt[4] = lo_170;
    float _fmax_72 = fmaxf(kbt[2], kbt[6]);
    float hi_171 = _fmax_72;
    float _min_72 = fminf(kbt[2], kbt[6]);
    float lo_172 = _min_72;
    kbt[2] = hi_171;
    kbt[6] = lo_172;
    float _fmax_73 = fmaxf(kbt[5], kbt[8]);
    float hi_173 = _fmax_73;
    float _min_73 = fminf(kbt[5], kbt[8]);
    float lo_174 = _min_73;
    kbt[5] = hi_173;
    kbt[8] = lo_174;
    float _fmax_74 = fmaxf(kbt[7], kbt[10]);
    float hi_175 = _fmax_74;
    float _min_74 = fminf(kbt[7], kbt[10]);
    float lo_176 = _min_74;
    kbt[7] = hi_175;
    kbt[10] = lo_176;
    float _fmax_75 = fmaxf(kbt[9], kbt[13]);
    float hi_177 = _fmax_75;
    float _min_75 = fminf(kbt[9], kbt[13]);
    float lo_178 = _min_75;
    kbt[9] = hi_177;
    kbt[13] = lo_178;
    float _fmax_76 = fmaxf(kbt[11], kbt[14]);
    float hi_179 = _fmax_76;
    float _min_76 = fminf(kbt[11], kbt[14]);
    float lo_180 = _min_76;
    kbt[11] = hi_179;
    kbt[14] = lo_180;
    float _fmax_77 = fmaxf(kbt[2], kbt[4]);
    float hi_181 = _fmax_77;
    float _min_77 = fminf(kbt[2], kbt[4]);
    float lo_182 = _min_77;
    kbt[2] = hi_181;
    kbt[4] = lo_182;
    float _fmax_78 = fmaxf(kbt[3], kbt[6]);
    float hi_183 = _fmax_78;
    float _min_78 = fminf(kbt[3], kbt[6]);
    float lo_184 = _min_78;
    kbt[3] = hi_183;
    kbt[6] = lo_184;
    float _fmax_79 = fmaxf(kbt[9], kbt[12]);
    float hi_185 = _fmax_79;
    float _min_79 = fminf(kbt[9], kbt[12]);
    float lo_186 = _min_79;
    kbt[9] = hi_185;
    kbt[12] = lo_186;
    float _fmax_80 = fmaxf(kbt[11], kbt[13]);
    float hi_187 = _fmax_80;
    float _min_80 = fminf(kbt[11], kbt[13]);
    float lo_188 = _min_80;
    kbt[11] = hi_187;
    kbt[13] = lo_188;
    float _fmax_81 = fmaxf(kbt[3], kbt[5]);
    float hi_189 = _fmax_81;
    float _min_81 = fminf(kbt[3], kbt[5]);
    float lo_190 = _min_81;
    kbt[3] = hi_189;
    kbt[5] = lo_190;
    float _fmax_82 = fmaxf(kbt[6], kbt[8]);
    float hi_191 = _fmax_82;
    float _min_82 = fminf(kbt[6], kbt[8]);
    float lo_192 = _min_82;
    kbt[6] = hi_191;
    kbt[8] = lo_192;
    float _fmax_83 = fmaxf(kbt[7], kbt[9]);
    float hi_193 = _fmax_83;
    float _min_83 = fminf(kbt[7], kbt[9]);
    float lo_194 = _min_83;
    kbt[7] = hi_193;
    kbt[9] = lo_194;
    float _fmax_84 = fmaxf(kbt[10], kbt[12]);
    float hi_195 = _fmax_84;
    float _min_84 = fminf(kbt[10], kbt[12]);
    float lo_196 = _min_84;
    kbt[10] = hi_195;
    kbt[12] = lo_196;
    float _fmax_85 = fmaxf(kbt[3], kbt[4]);
    float hi_197 = _fmax_85;
    float _min_85 = fminf(kbt[3], kbt[4]);
    float lo_198 = _min_85;
    kbt[3] = hi_197;
    kbt[4] = lo_198;
    float _fmax_86 = fmaxf(kbt[5], kbt[6]);
    float hi_199 = _fmax_86;
    float _min_86 = fminf(kbt[5], kbt[6]);
    float lo_200 = _min_86;
    kbt[5] = hi_199;
    kbt[6] = lo_200;
    float _fmax_87 = fmaxf(kbt[7], kbt[8]);
    float hi_201 = _fmax_87;
    float _min_87 = fminf(kbt[7], kbt[8]);
    float lo_202 = _min_87;
    kbt[7] = hi_201;
    kbt[8] = lo_202;
    float _fmax_88 = fmaxf(kbt[9], kbt[10]);
    float hi_203 = _fmax_88;
    float _min_88 = fminf(kbt[9], kbt[10]);
    float lo_204 = _min_88;
    kbt[9] = hi_203;
    kbt[10] = lo_204;
    float _fmax_89 = fmaxf(kbt[11], kbt[12]);
    float hi_205 = _fmax_89;
    float _min_89 = fminf(kbt[11], kbt[12]);
    float lo_206 = _min_89;
    kbt[11] = hi_205;
    kbt[12] = lo_206;
    float _fmax_90 = fmaxf(kbt[6], kbt[7]);
    float hi_207 = _fmax_90;
    float _min_90 = fminf(kbt[6], kbt[7]);
    float lo_208 = _min_90;
    kbt[6] = hi_207;
    kbt[7] = lo_208;
    float _fmax_91 = fmaxf(kbt[8], kbt[9]);
    float hi_209 = _fmax_91;
    float _min_91 = fminf(kbt[8], kbt[9]);
    float lo_210 = _min_91;
    kbt[8] = hi_209;
    kbt[9] = lo_210;
    float _fmax_92 = fmaxf(kbt[16], kbt[29]);
    float hi_211 = _fmax_92;
    float _min_92 = fminf(kbt[16], kbt[29]);
    float lo_212 = _min_92;
    kbt[16] = hi_211;
    kbt[29] = lo_212;
    float _fmax_93 = fmaxf(kbt[17], kbt[28]);
    float hi_213 = _fmax_93;
    float _min_93 = fminf(kbt[17], kbt[28]);
    float lo_214 = _min_93;
    kbt[17] = hi_213;
    kbt[28] = lo_214;
    float _fmax_94 = fmaxf(kbt[18], kbt[31]);
    float hi_215 = _fmax_94;
    float _min_94 = fminf(kbt[18], kbt[31]);
    float lo_216 = _min_94;
    kbt[18] = hi_215;
    kbt[31] = lo_216;
    float _fmax_95 = fmaxf(kbt[19], kbt[30]);
    float hi_217 = _fmax_95;
    float _min_95 = fminf(kbt[19], kbt[30]);
    float lo_218 = _min_95;
    kbt[19] = hi_217;
    kbt[30] = lo_218;
    float _fmax_96 = fmaxf(kbt[20], kbt[24]);
    float hi_219 = _fmax_96;
    float _min_96 = fminf(kbt[20], kbt[24]);
    float lo_220 = _min_96;
    kbt[20] = hi_219;
    kbt[24] = lo_220;
    float _fmax_97 = fmaxf(kbt[21], kbt[22]);
    float hi_221 = _fmax_97;
    float _min_97 = fminf(kbt[21], kbt[22]);
    float lo_222 = _min_97;
    kbt[21] = hi_221;
    kbt[22] = lo_222;
    float _fmax_98 = fmaxf(kbt[23], kbt[27]);
    float hi_223 = _fmax_98;
    float _min_98 = fminf(kbt[23], kbt[27]);
    float lo_224 = _min_98;
    kbt[23] = hi_223;
    kbt[27] = lo_224;
    float _fmax_99 = fmaxf(kbt[25], kbt[26]);
    float hi_225 = _fmax_99;
    float _min_99 = fminf(kbt[25], kbt[26]);
    float lo_226 = _min_99;
    kbt[25] = hi_225;
    kbt[26] = lo_226;
    float _fmax_100 = fmaxf(kbt[16], kbt[21]);
    float hi_227 = _fmax_100;
    float _min_100 = fminf(kbt[16], kbt[21]);
    float lo_228 = _min_100;
    kbt[16] = hi_227;
    kbt[21] = lo_228;
    float _fmax_101 = fmaxf(kbt[17], kbt[23]);
    float hi_229 = _fmax_101;
    float _min_101 = fminf(kbt[17], kbt[23]);
    float lo_230 = _min_101;
    kbt[17] = hi_229;
    kbt[23] = lo_230;
    float _fmax_102 = fmaxf(kbt[18], kbt[25]);
    float hi_231 = _fmax_102;
    float _min_102 = fminf(kbt[18], kbt[25]);
    float lo_232 = _min_102;
    kbt[18] = hi_231;
    kbt[25] = lo_232;
    float _fmax_103 = fmaxf(kbt[19], kbt[20]);
    float hi_233 = _fmax_103;
    float _min_103 = fminf(kbt[19], kbt[20]);
    float lo_234 = _min_103;
    kbt[19] = hi_233;
    kbt[20] = lo_234;
    float _fmax_104 = fmaxf(kbt[22], kbt[29]);
    float hi_235 = _fmax_104;
    float _min_104 = fminf(kbt[22], kbt[29]);
    float lo_236 = _min_104;
    kbt[22] = hi_235;
    kbt[29] = lo_236;
    float _fmax_105 = fmaxf(kbt[24], kbt[30]);
    float hi_237 = _fmax_105;
    float _min_105 = fminf(kbt[24], kbt[30]);
    float lo_238 = _min_105;
    kbt[24] = hi_237;
    kbt[30] = lo_238;
    float _fmax_106 = fmaxf(kbt[26], kbt[31]);
    float hi_239 = _fmax_106;
    float _min_106 = fminf(kbt[26], kbt[31]);
    float lo_240 = _min_106;
    kbt[26] = hi_239;
    kbt[31] = lo_240;
    float _fmax_107 = fmaxf(kbt[27], kbt[28]);
    float hi_241 = _fmax_107;
    float _min_107 = fminf(kbt[27], kbt[28]);
    float lo_242 = _min_107;
    kbt[27] = hi_241;
    kbt[28] = lo_242;
    float _fmax_108 = fmaxf(kbt[16], kbt[17]);
    float hi_243 = _fmax_108;
    float _min_108 = fminf(kbt[16], kbt[17]);
    float lo_244 = _min_108;
    kbt[16] = hi_243;
    kbt[17] = lo_244;
    float _fmax_109 = fmaxf(kbt[18], kbt[19]);
    float hi_245 = _fmax_109;
    float _min_109 = fminf(kbt[18], kbt[19]);
    float lo_246 = _min_109;
    kbt[18] = hi_245;
    kbt[19] = lo_246;
    float _fmax_110 = fmaxf(kbt[20], kbt[21]);
    float hi_247 = _fmax_110;
    float _min_110 = fminf(kbt[20], kbt[21]);
    float lo_248 = _min_110;
    kbt[20] = hi_247;
    kbt[21] = lo_248;
    float _fmax_111 = fmaxf(kbt[22], kbt[24]);
    float hi_249 = _fmax_111;
    float _min_111 = fminf(kbt[22], kbt[24]);
    float lo_250 = _min_111;
    kbt[22] = hi_249;
    kbt[24] = lo_250;
    float _fmax_112 = fmaxf(kbt[23], kbt[25]);
    float hi_251 = _fmax_112;
    float _min_112 = fminf(kbt[23], kbt[25]);
    float lo_252 = _min_112;
    kbt[23] = hi_251;
    kbt[25] = lo_252;
    float _fmax_113 = fmaxf(kbt[26], kbt[27]);
    float hi_253 = _fmax_113;
    float _min_113 = fminf(kbt[26], kbt[27]);
    float lo_254 = _min_113;
    kbt[26] = hi_253;
    kbt[27] = lo_254;
    float _fmax_114 = fmaxf(kbt[28], kbt[29]);
    float hi_255 = _fmax_114;
    float _min_114 = fminf(kbt[28], kbt[29]);
    float lo_256 = _min_114;
    kbt[28] = hi_255;
    kbt[29] = lo_256;
    float _fmax_115 = fmaxf(kbt[30], kbt[31]);
    float hi_257 = _fmax_115;
    float _min_115 = fminf(kbt[30], kbt[31]);
    float lo_258 = _min_115;
    kbt[30] = hi_257;
    kbt[31] = lo_258;
    float _fmax_116 = fmaxf(kbt[16], kbt[18]);
    float hi_259 = _fmax_116;
    float _min_116 = fminf(kbt[16], kbt[18]);
    float lo_260 = _min_116;
    kbt[16] = hi_259;
    kbt[18] = lo_260;
    float _fmax_117 = fmaxf(kbt[17], kbt[19]);
    float hi_261 = _fmax_117;
    float _min_117 = fminf(kbt[17], kbt[19]);
    float lo_262 = _min_117;
    kbt[17] = hi_261;
    kbt[19] = lo_262;
    float _fmax_118 = fmaxf(kbt[20], kbt[26]);
    float hi_263 = _fmax_118;
    float _min_118 = fminf(kbt[20], kbt[26]);
    float lo_264 = _min_118;
    kbt[20] = hi_263;
    kbt[26] = lo_264;
    float _fmax_119 = fmaxf(kbt[21], kbt[27]);
    float hi_265 = _fmax_119;
    float _min_119 = fminf(kbt[21], kbt[27]);
    float lo_266 = _min_119;
    kbt[21] = hi_265;
    kbt[27] = lo_266;
    float _fmax_120 = fmaxf(kbt[22], kbt[23]);
    float hi_267 = _fmax_120;
    float _min_120 = fminf(kbt[22], kbt[23]);
    float lo_268 = _min_120;
    kbt[22] = hi_267;
    kbt[23] = lo_268;
    float _fmax_121 = fmaxf(kbt[24], kbt[25]);
    float hi_269 = _fmax_121;
    float _min_121 = fminf(kbt[24], kbt[25]);
    float lo_270 = _min_121;
    kbt[24] = hi_269;
    kbt[25] = lo_270;
    float _fmax_122 = fmaxf(kbt[28], kbt[30]);
    float hi_271 = _fmax_122;
    float _min_122 = fminf(kbt[28], kbt[30]);
    float lo_272 = _min_122;
    kbt[28] = hi_271;
    kbt[30] = lo_272;
    float _fmax_123 = fmaxf(kbt[29], kbt[31]);
    float hi_273 = _fmax_123;
    float _min_123 = fminf(kbt[29], kbt[31]);
    float lo_274 = _min_123;
    kbt[29] = hi_273;
    kbt[31] = lo_274;
    float _fmax_124 = fmaxf(kbt[17], kbt[18]);
    float hi_275 = _fmax_124;
    float _min_124 = fminf(kbt[17], kbt[18]);
    float lo_276 = _min_124;
    kbt[17] = hi_275;
    kbt[18] = lo_276;
    float _fmax_125 = fmaxf(kbt[19], kbt[28]);
    float hi_277 = _fmax_125;
    float _min_125 = fminf(kbt[19], kbt[28]);
    float lo_278 = _min_125;
    kbt[19] = hi_277;
    kbt[28] = lo_278;
    float _fmax_126 = fmaxf(kbt[20], kbt[22]);
    float hi_279 = _fmax_126;
    float _min_126 = fminf(kbt[20], kbt[22]);
    float lo_280 = _min_126;
    kbt[20] = hi_279;
    kbt[22] = lo_280;
    float _fmax_127 = fmaxf(kbt[21], kbt[23]);
    float hi_281 = _fmax_127;
    float _min_127 = fminf(kbt[21], kbt[23]);
    float lo_282 = _min_127;
    kbt[21] = hi_281;
    kbt[23] = lo_282;
    float _fmax_128 = fmaxf(kbt[24], kbt[26]);
    float hi_283 = _fmax_128;
    float _min_128 = fminf(kbt[24], kbt[26]);
    float lo_284 = _min_128;
    kbt[24] = hi_283;
    kbt[26] = lo_284;
    float _fmax_129 = fmaxf(kbt[25], kbt[27]);
    float hi_285 = _fmax_129;
    float _min_129 = fminf(kbt[25], kbt[27]);
    float lo_286 = _min_129;
    kbt[25] = hi_285;
    kbt[27] = lo_286;
    float _fmax_130 = fmaxf(kbt[29], kbt[30]);
    float hi_287 = _fmax_130;
    float _min_130 = fminf(kbt[29], kbt[30]);
    float lo_288 = _min_130;
    kbt[29] = hi_287;
    kbt[30] = lo_288;
    float _fmax_131 = fmaxf(kbt[17], kbt[20]);
    float hi_289 = _fmax_131;
    float _min_131 = fminf(kbt[17], kbt[20]);
    float lo_290 = _min_131;
    kbt[17] = hi_289;
    kbt[20] = lo_290;
    float _fmax_132 = fmaxf(kbt[18], kbt[22]);
    float hi_291 = _fmax_132;
    float _min_132 = fminf(kbt[18], kbt[22]);
    float lo_292 = _min_132;
    kbt[18] = hi_291;
    kbt[22] = lo_292;
    float _fmax_133 = fmaxf(kbt[21], kbt[24]);
    float hi_293 = _fmax_133;
    float _min_133 = fminf(kbt[21], kbt[24]);
    float lo_294 = _min_133;
    kbt[21] = hi_293;
    kbt[24] = lo_294;
    float _fmax_134 = fmaxf(kbt[23], kbt[26]);
    float hi_295 = _fmax_134;
    float _min_134 = fminf(kbt[23], kbt[26]);
    float lo_296 = _min_134;
    kbt[23] = hi_295;
    kbt[26] = lo_296;
    float _fmax_135 = fmaxf(kbt[25], kbt[29]);
    float hi_297 = _fmax_135;
    float _min_135 = fminf(kbt[25], kbt[29]);
    float lo_298 = _min_135;
    kbt[25] = hi_297;
    kbt[29] = lo_298;
    float _fmax_136 = fmaxf(kbt[27], kbt[30]);
    float hi_299 = _fmax_136;
    float _min_136 = fminf(kbt[27], kbt[30]);
    float lo_300 = _min_136;
    kbt[27] = hi_299;
    kbt[30] = lo_300;
    float _fmax_137 = fmaxf(kbt[18], kbt[20]);
    float hi_301 = _fmax_137;
    float _min_137 = fminf(kbt[18], kbt[20]);
    float lo_302 = _min_137;
    kbt[18] = hi_301;
    kbt[20] = lo_302;
    float _fmax_138 = fmaxf(kbt[19], kbt[22]);
    float hi_303 = _fmax_138;
    float _min_138 = fminf(kbt[19], kbt[22]);
    float lo_304 = _min_138;
    kbt[19] = hi_303;
    kbt[22] = lo_304;
    float _fmax_139 = fmaxf(kbt[25], kbt[28]);
    float hi_305 = _fmax_139;
    float _min_139 = fminf(kbt[25], kbt[28]);
    float lo_306 = _min_139;
    kbt[25] = hi_305;
    kbt[28] = lo_306;
    float _fmax_140 = fmaxf(kbt[27], kbt[29]);
    float hi_307 = _fmax_140;
    float _min_140 = fminf(kbt[27], kbt[29]);
    float lo_308 = _min_140;
    kbt[27] = hi_307;
    kbt[29] = lo_308;
    float _fmax_141 = fmaxf(kbt[19], kbt[21]);
    float hi_309 = _fmax_141;
    float _min_141 = fminf(kbt[19], kbt[21]);
    float lo_310 = _min_141;
    kbt[19] = hi_309;
    kbt[21] = lo_310;
    float _fmax_142 = fmaxf(kbt[22], kbt[24]);
    float hi_311 = _fmax_142;
    float _min_142 = fminf(kbt[22], kbt[24]);
    float lo_312 = _min_142;
    kbt[22] = hi_311;
    kbt[24] = lo_312;
    float _fmax_143 = fmaxf(kbt[23], kbt[25]);
    float hi_313 = _fmax_143;
    float _min_143 = fminf(kbt[23], kbt[25]);
    float lo_314 = _min_143;
    kbt[23] = hi_313;
    kbt[25] = lo_314;
    float _fmax_144 = fmaxf(kbt[26], kbt[28]);
    float hi_315 = _fmax_144;
    float _min_144 = fminf(kbt[26], kbt[28]);
    float lo_316 = _min_144;
    kbt[26] = hi_315;
    kbt[28] = lo_316;
    float _fmax_145 = fmaxf(kbt[19], kbt[20]);
    float hi_317 = _fmax_145;
    float _min_145 = fminf(kbt[19], kbt[20]);
    float lo_318 = _min_145;
    kbt[19] = hi_317;
    kbt[20] = lo_318;
    float _fmax_146 = fmaxf(kbt[21], kbt[22]);
    float hi_319 = _fmax_146;
    float _min_146 = fminf(kbt[21], kbt[22]);
    float lo_320 = _min_146;
    kbt[21] = hi_319;
    kbt[22] = lo_320;
    float _fmax_147 = fmaxf(kbt[23], kbt[24]);
    float hi_321 = _fmax_147;
    float _min_147 = fminf(kbt[23], kbt[24]);
    float lo_322 = _min_147;
    kbt[23] = hi_321;
    kbt[24] = lo_322;
    float _fmax_148 = fmaxf(kbt[25], kbt[26]);
    float hi_323 = _fmax_148;
    float _min_148 = fminf(kbt[25], kbt[26]);
    float lo_324 = _min_148;
    kbt[25] = hi_323;
    kbt[26] = lo_324;
    float _fmax_149 = fmaxf(kbt[27], kbt[28]);
    float hi_325 = _fmax_149;
    float _min_149 = fminf(kbt[27], kbt[28]);
    float lo_326 = _min_149;
    kbt[27] = hi_325;
    kbt[28] = lo_326;
    float _fmax_150 = fmaxf(kbt[22], kbt[23]);
    float hi_327 = _fmax_150;
    float _min_150 = fminf(kbt[22], kbt[23]);
    float lo_328 = _min_150;
    kbt[22] = hi_327;
    kbt[23] = lo_328;
    float _fmax_151 = fmaxf(kbt[24], kbt[25]);
    float hi_329 = _fmax_151;
    float _min_151 = fminf(kbt[24], kbt[25]);
    float lo_330 = _min_151;
    kbt[24] = hi_329;
    kbt[25] = lo_330;
    float _fmax_152 = fmaxf(kbt[0], kbt[31]);
    float hi_331 = _fmax_152;
    float _min_152 = fminf(kbt[0], kbt[31]);
    float lo_332 = _min_152;
    kbt[0] = hi_331;
    float _fmax_153 = fmaxf(rr, lo_332);
    rr = _fmax_153;
    float _fmax_154 = fmaxf(kbt[1], kbt[30]);
    float hi_333 = _fmax_154;
    float _min_153 = fminf(kbt[1], kbt[30]);
    float lo_334 = _min_153;
    kbt[1] = hi_333;
    float _fmax_155 = fmaxf(rr, lo_334);
    rr = _fmax_155;
    float _fmax_156 = fmaxf(kbt[2], kbt[29]);
    float hi_335 = _fmax_156;
    float _min_154 = fminf(kbt[2], kbt[29]);
    float lo_336 = _min_154;
    kbt[2] = hi_335;
    float _fmax_157 = fmaxf(rr, lo_336);
    rr = _fmax_157;
    float _fmax_158 = fmaxf(kbt[3], kbt[28]);
    float hi_337 = _fmax_158;
    float _min_155 = fminf(kbt[3], kbt[28]);
    float lo_338 = _min_155;
    kbt[3] = hi_337;
    float _fmax_159 = fmaxf(rr, lo_338);
    rr = _fmax_159;
    float _fmax_160 = fmaxf(kbt[4], kbt[27]);
    float hi_339 = _fmax_160;
    float _min_156 = fminf(kbt[4], kbt[27]);
    float lo_340 = _min_156;
    kbt[4] = hi_339;
    float _fmax_161 = fmaxf(rr, lo_340);
    rr = _fmax_161;
    float _fmax_162 = fmaxf(kbt[5], kbt[26]);
    float hi_341 = _fmax_162;
    float _min_157 = fminf(kbt[5], kbt[26]);
    float lo_342 = _min_157;
    kbt[5] = hi_341;
    float _fmax_163 = fmaxf(rr, lo_342);
    rr = _fmax_163;
    float _fmax_164 = fmaxf(kbt[6], kbt[25]);
    float hi_343 = _fmax_164;
    float _min_158 = fminf(kbt[6], kbt[25]);
    float lo_344 = _min_158;
    kbt[6] = hi_343;
    float _fmax_165 = fmaxf(rr, lo_344);
    rr = _fmax_165;
    float _fmax_166 = fmaxf(kbt[7], kbt[24]);
    float hi_345 = _fmax_166;
    float _min_159 = fminf(kbt[7], kbt[24]);
    float lo_346 = _min_159;
    kbt[7] = hi_345;
    float _fmax_167 = fmaxf(rr, lo_346);
    rr = _fmax_167;
    float _fmax_168 = fmaxf(kbt[8], kbt[23]);
    float hi_347 = _fmax_168;
    float _min_160 = fminf(kbt[8], kbt[23]);
    float lo_348 = _min_160;
    kbt[8] = hi_347;
    float _fmax_169 = fmaxf(rr, lo_348);
    rr = _fmax_169;
    float _fmax_170 = fmaxf(kbt[9], kbt[22]);
    float hi_349 = _fmax_170;
    float _min_161 = fminf(kbt[9], kbt[22]);
    float lo_350 = _min_161;
    kbt[9] = hi_349;
    float _fmax_171 = fmaxf(rr, lo_350);
    rr = _fmax_171;
    float _fmax_172 = fmaxf(kbt[10], kbt[21]);
    float hi_351 = _fmax_172;
    float _min_162 = fminf(kbt[10], kbt[21]);
    float lo_352 = _min_162;
    kbt[10] = hi_351;
    float _fmax_173 = fmaxf(rr, lo_352);
    rr = _fmax_173;
    float _fmax_174 = fmaxf(kbt[11], kbt[20]);
    float hi_353 = _fmax_174;
    float _min_163 = fminf(kbt[11], kbt[20]);
    float lo_354 = _min_163;
    kbt[11] = hi_353;
    float _fmax_175 = fmaxf(rr, lo_354);
    rr = _fmax_175;
    float _fmax_176 = fmaxf(kbt[12], kbt[19]);
    float hi_355 = _fmax_176;
    float _min_164 = fminf(kbt[12], kbt[19]);
    float lo_356 = _min_164;
    kbt[12] = hi_355;
    float _fmax_177 = fmaxf(rr, lo_356);
    rr = _fmax_177;
    float _fmax_178 = fmaxf(kbt[13], kbt[18]);
    float hi_357 = _fmax_178;
    float _min_165 = fminf(kbt[13], kbt[18]);
    float lo_358 = _min_165;
    kbt[13] = hi_357;
    float _fmax_179 = fmaxf(rr, lo_358);
    rr = _fmax_179;
    float _fmax_180 = fmaxf(kbt[14], kbt[17]);
    float hi_359 = _fmax_180;
    float _min_166 = fminf(kbt[14], kbt[17]);
    float lo_360 = _min_166;
    kbt[14] = hi_359;
    float _fmax_181 = fmaxf(rr, lo_360);
    rr = _fmax_181;
    float _fmax_182 = fmaxf(kbt[15], kbt[16]);
    float hi_361 = _fmax_182;
    float _min_167 = fminf(kbt[15], kbt[16]);
    float lo_362 = _min_167;
    kbt[15] = hi_361;
    float _fmax_183 = fmaxf(rr, lo_362);
    rr = _fmax_183;
    float _fmax_184 = fmaxf(kbt[0], kbt[8]);
    float hi_363 = _fmax_184;
    float _min_168 = fminf(kbt[0], kbt[8]);
    float lo_364 = _min_168;
    kbt[0] = hi_363;
    kbt[8] = lo_364;
    float _fmax_185 = fmaxf(kbt[1], kbt[9]);
    float hi_365 = _fmax_185;
    float _min_169 = fminf(kbt[1], kbt[9]);
    float lo_366 = _min_169;
    kbt[1] = hi_365;
    kbt[9] = lo_366;
    float _fmax_186 = fmaxf(kbt[2], kbt[10]);
    float hi_367 = _fmax_186;
    float _min_170 = fminf(kbt[2], kbt[10]);
    float lo_368 = _min_170;
    kbt[2] = hi_367;
    kbt[10] = lo_368;
    float _fmax_187 = fmaxf(kbt[3], kbt[11]);
    float hi_369 = _fmax_187;
    float _min_171 = fminf(kbt[3], kbt[11]);
    float lo_370 = _min_171;
    kbt[3] = hi_369;
    kbt[11] = lo_370;
    float _fmax_188 = fmaxf(kbt[4], kbt[12]);
    float hi_371 = _fmax_188;
    float _min_172 = fminf(kbt[4], kbt[12]);
    float lo_372 = _min_172;
    kbt[4] = hi_371;
    kbt[12] = lo_372;
    float _fmax_189 = fmaxf(kbt[5], kbt[13]);
    float hi_373 = _fmax_189;
    float _min_173 = fminf(kbt[5], kbt[13]);
    float lo_374 = _min_173;
    kbt[5] = hi_373;
    kbt[13] = lo_374;
    float _fmax_190 = fmaxf(kbt[6], kbt[14]);
    float hi_375 = _fmax_190;
    float _min_174 = fminf(kbt[6], kbt[14]);
    float lo_376 = _min_174;
    kbt[6] = hi_375;
    kbt[14] = lo_376;
    float _fmax_191 = fmaxf(kbt[7], kbt[15]);
    float hi_377 = _fmax_191;
    float _min_175 = fminf(kbt[7], kbt[15]);
    float lo_378 = _min_175;
    kbt[7] = hi_377;
    kbt[15] = lo_378;
    float _fmax_192 = fmaxf(kbt[0], kbt[4]);
    float hi_379 = _fmax_192;
    float _min_176 = fminf(kbt[0], kbt[4]);
    float lo_380 = _min_176;
    kbt[0] = hi_379;
    kbt[4] = lo_380;
    float _fmax_193 = fmaxf(kbt[1], kbt[5]);
    float hi_381 = _fmax_193;
    float _min_177 = fminf(kbt[1], kbt[5]);
    float lo_382 = _min_177;
    kbt[1] = hi_381;
    kbt[5] = lo_382;
    float _fmax_194 = fmaxf(kbt[2], kbt[6]);
    float hi_383 = _fmax_194;
    float _min_178 = fminf(kbt[2], kbt[6]);
    float lo_384 = _min_178;
    kbt[2] = hi_383;
    kbt[6] = lo_384;
    float _fmax_195 = fmaxf(kbt[3], kbt[7]);
    float hi_385 = _fmax_195;
    float _min_179 = fminf(kbt[3], kbt[7]);
    float lo_386 = _min_179;
    kbt[3] = hi_385;
    kbt[7] = lo_386;
    float _fmax_196 = fmaxf(kbt[8], kbt[12]);
    float hi_387 = _fmax_196;
    float _min_180 = fminf(kbt[8], kbt[12]);
    float lo_388 = _min_180;
    kbt[8] = hi_387;
    kbt[12] = lo_388;
    float _fmax_197 = fmaxf(kbt[9], kbt[13]);
    float hi_389 = _fmax_197;
    float _min_181 = fminf(kbt[9], kbt[13]);
    float lo_390 = _min_181;
    kbt[9] = hi_389;
    kbt[13] = lo_390;
    float _fmax_198 = fmaxf(kbt[10], kbt[14]);
    float hi_391 = _fmax_198;
    float _min_182 = fminf(kbt[10], kbt[14]);
    float lo_392 = _min_182;
    kbt[10] = hi_391;
    kbt[14] = lo_392;
    float _fmax_199 = fmaxf(kbt[11], kbt[15]);
    float hi_393 = _fmax_199;
    float _min_183 = fminf(kbt[11], kbt[15]);
    float lo_394 = _min_183;
    kbt[11] = hi_393;
    kbt[15] = lo_394;
    float _fmax_200 = fmaxf(kbt[0], kbt[2]);
    float hi_395 = _fmax_200;
    float _min_184 = fminf(kbt[0], kbt[2]);
    float lo_396 = _min_184;
    kbt[0] = hi_395;
    kbt[2] = lo_396;
    float _fmax_201 = fmaxf(kbt[1], kbt[3]);
    float hi_397 = _fmax_201;
    float _min_185 = fminf(kbt[1], kbt[3]);
    float lo_398 = _min_185;
    kbt[1] = hi_397;
    kbt[3] = lo_398;
    float _fmax_202 = fmaxf(kbt[4], kbt[6]);
    float hi_399 = _fmax_202;
    float _min_186 = fminf(kbt[4], kbt[6]);
    float lo_400 = _min_186;
    kbt[4] = hi_399;
    kbt[6] = lo_400;
    float _fmax_203 = fmaxf(kbt[5], kbt[7]);
    float hi_401 = _fmax_203;
    float _min_187 = fminf(kbt[5], kbt[7]);
    float lo_402 = _min_187;
    kbt[5] = hi_401;
    kbt[7] = lo_402;
    float _fmax_204 = fmaxf(kbt[8], kbt[10]);
    float hi_403 = _fmax_204;
    float _min_188 = fminf(kbt[8], kbt[10]);
    float lo_404 = _min_188;
    kbt[8] = hi_403;
    kbt[10] = lo_404;
    float _fmax_205 = fmaxf(kbt[9], kbt[11]);
    float hi_405 = _fmax_205;
    float _min_189 = fminf(kbt[9], kbt[11]);
    float lo_406 = _min_189;
    kbt[9] = hi_405;
    kbt[11] = lo_406;
    float _fmax_206 = fmaxf(kbt[12], kbt[14]);
    float hi_407 = _fmax_206;
    float _min_190 = fminf(kbt[12], kbt[14]);
    float lo_408 = _min_190;
    kbt[12] = hi_407;
    kbt[14] = lo_408;
    float _fmax_207 = fmaxf(kbt[13], kbt[15]);
    float hi_409 = _fmax_207;
    float _min_191 = fminf(kbt[13], kbt[15]);
    float lo_410 = _min_191;
    kbt[13] = hi_409;
    kbt[15] = lo_410;
    float _fmax_208 = fmaxf(kbt[0], kbt[1]);
    float hi_411 = _fmax_208;
    float _min_192 = fminf(kbt[0], kbt[1]);
    float lo_412 = _min_192;
    kbt[0] = hi_411;
    kbt[1] = lo_412;
    float _fmax_209 = fmaxf(kbt[2], kbt[3]);
    float hi_413 = _fmax_209;
    float _min_193 = fminf(kbt[2], kbt[3]);
    float lo_414 = _min_193;
    kbt[2] = hi_413;
    kbt[3] = lo_414;
    float _fmax_210 = fmaxf(kbt[4], kbt[5]);
    float hi_415 = _fmax_210;
    float _min_194 = fminf(kbt[4], kbt[5]);
    float lo_416 = _min_194;
    kbt[4] = hi_415;
    kbt[5] = lo_416;
    float _fmax_211 = fmaxf(kbt[6], kbt[7]);
    float hi_417 = _fmax_211;
    float _min_195 = fminf(kbt[6], kbt[7]);
    float lo_418 = _min_195;
    kbt[6] = hi_417;
    kbt[7] = lo_418;
    float _fmax_212 = fmaxf(kbt[8], kbt[9]);
    float hi_419 = _fmax_212;
    float _min_196 = fminf(kbt[8], kbt[9]);
    float lo_420 = _min_196;
    kbt[8] = hi_419;
    kbt[9] = lo_420;
    float _fmax_213 = fmaxf(kbt[10], kbt[11]);
    float hi_421 = _fmax_213;
    float _min_197 = fminf(kbt[10], kbt[11]);
    float lo_422 = _min_197;
    kbt[10] = hi_421;
    kbt[11] = lo_422;
    float _fmax_214 = fmaxf(kbt[12], kbt[13]);
    float hi_423 = _fmax_214;
    float _min_198 = fminf(kbt[12], kbt[13]);
    float lo_424 = _min_198;
    kbt[12] = hi_423;
    kbt[13] = lo_424;
    float _fmax_215 = fmaxf(kbt[14], kbt[15]);
    float hi_425 = _fmax_215;
    float _min_199 = fminf(kbt[14], kbt[15]);
    float lo_426 = _min_199;
    kbt[14] = hi_425;
    kbt[15] = lo_426;
    rej = rr;
    a[0] = kbt[0];
    a[1] = kbt[1];
    a[2] = kbt[2];
    a[3] = kbt[3];
    a[4] = kbt[4];
    a[5] = kbt[5];
    a[6] = kbt[6];
    a[7] = kbt[7];
    a[8] = kbt[8];
    a[9] = kbt[9];
    a[10] = kbt[10];
    a[11] = kbt[11];
    a[12] = kbt[12];
    a[13] = kbt[13];
    a[14] = kbt[14];
    a[15] = kbt[15];
    int ln = tid_1 & 15;
    int g = tid_1 >> 4;
    int cg = g & 15;
    int sg = g >> 4;
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
    int s0 = (sg * 16 * 16 + cg) * 17;
    int s1 = ((sg * 16 + 8) * 16 + cg) * 17;
    float x0 = pub[s0 + ln];
    float y0 = pub[s1 + lnr];
    float _min_200 = fminf(x0, y0);
    float lo0 = _min_200;
    float _fmax_216 = fmaxf(r, lo0);
    r = _fmax_216;
    float _fmax_217 = fmaxf(x0, y0);
    float hi0 = _fmax_217;
    float cur = hi0;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
    float pv = _shfl_xor_0;
    float _fmax_218 = fmaxf(cur, pv);
    float hi_427 = _fmax_218;
    float _min_201 = fminf(cur, pv);
    float lo_428 = _min_201;
    cur = ((up[0] != 0) ? hi_427 : lo_428);
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
    float pv_429 = _shfl_xor_1;
    float _fmax_219 = fmaxf(cur, pv_429);
    float hi_430 = _fmax_219;
    float _min_202 = fminf(cur, pv_429);
    float lo_431 = _min_202;
    cur = ((up[1] != 0) ? hi_430 : lo_431);
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
    float pv_432 = _shfl_xor_2;
    float _fmax_220 = fmaxf(cur, pv_432);
    float hi_433 = _fmax_220;
    float _min_203 = fminf(cur, pv_432);
    float lo_434 = _min_203;
    cur = ((up[2] != 0) ? hi_433 : lo_434);
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
    float pv_435 = _shfl_xor_3;
    float _fmax_221 = fmaxf(cur, pv_435);
    float hi_436 = _fmax_221;
    float _min_204 = fminf(cur, pv_435);
    float lo_437 = _min_204;
    cur = ((up[3] != 0) ? hi_436 : lo_437);
    V[0] = cur;
    int s0_438 = ((sg * 16 + 1) * 16 + cg) * 17;
    int s1_439 = ((sg * 16 + 1 + 8) * 16 + cg) * 17;
    float x0_440 = pub[s0_438 + ln];
    float y0_441 = pub[s1_439 + lnr];
    float _min_205 = fminf(x0_440, y0_441);
    float lo0_442 = _min_205;
    float _fmax_222 = fmaxf(r, lo0_442);
    r = _fmax_222;
    float _fmax_223 = fmaxf(x0_440, y0_441);
    float hi0_443 = _fmax_223;
    float cur_444 = hi0_443;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, cur_444, 8);
    float pv_445 = _shfl_xor_4;
    float _fmax_224 = fmaxf(cur_444, pv_445);
    float hi_446 = _fmax_224;
    float _min_206 = fminf(cur_444, pv_445);
    float lo_447 = _min_206;
    cur_444 = ((up[0] != 0) ? hi_446 : lo_447);
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, cur_444, 4);
    float pv_448 = _shfl_xor_5;
    float _fmax_225 = fmaxf(cur_444, pv_448);
    float hi_449 = _fmax_225;
    float _min_207 = fminf(cur_444, pv_448);
    float lo_450 = _min_207;
    cur_444 = ((up[1] != 0) ? hi_449 : lo_450);
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, cur_444, 2);
    float pv_451 = _shfl_xor_6;
    float _fmax_226 = fmaxf(cur_444, pv_451);
    float hi_452 = _fmax_226;
    float _min_208 = fminf(cur_444, pv_451);
    float lo_453 = _min_208;
    cur_444 = ((up[2] != 0) ? hi_452 : lo_453);
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, cur_444, 1);
    float pv_454 = _shfl_xor_7;
    float _fmax_227 = fmaxf(cur_444, pv_454);
    float hi_455 = _fmax_227;
    float _min_209 = fminf(cur_444, pv_454);
    float lo_456 = _min_209;
    cur_444 = ((up[3] != 0) ? hi_455 : lo_456);
    V[1] = cur_444;
    int s0_457 = ((sg * 16 + 2) * 16 + cg) * 17;
    int s1_458 = ((sg * 16 + 2 + 8) * 16 + cg) * 17;
    float x0_459 = pub[s0_457 + ln];
    float y0_460 = pub[s1_458 + lnr];
    float _min_210 = fminf(x0_459, y0_460);
    float lo0_461 = _min_210;
    float _fmax_228 = fmaxf(r, lo0_461);
    r = _fmax_228;
    float _fmax_229 = fmaxf(x0_459, y0_460);
    float hi0_462 = _fmax_229;
    float cur_463 = hi0_462;
    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, cur_463, 8);
    float pv_464 = _shfl_xor_8;
    float _fmax_230 = fmaxf(cur_463, pv_464);
    float hi_465 = _fmax_230;
    float _min_211 = fminf(cur_463, pv_464);
    float lo_466 = _min_211;
    cur_463 = ((up[0] != 0) ? hi_465 : lo_466);
    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, cur_463, 4);
    float pv_467 = _shfl_xor_9;
    float _fmax_231 = fmaxf(cur_463, pv_467);
    float hi_468 = _fmax_231;
    float _min_212 = fminf(cur_463, pv_467);
    float lo_469 = _min_212;
    cur_463 = ((up[1] != 0) ? hi_468 : lo_469);
    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, cur_463, 2);
    float pv_470 = _shfl_xor_10;
    float _fmax_232 = fmaxf(cur_463, pv_470);
    float hi_471 = _fmax_232;
    float _min_213 = fminf(cur_463, pv_470);
    float lo_472 = _min_213;
    cur_463 = ((up[2] != 0) ? hi_471 : lo_472);
    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, cur_463, 1);
    float pv_473 = _shfl_xor_11;
    float _fmax_233 = fmaxf(cur_463, pv_473);
    float hi_474 = _fmax_233;
    float _min_214 = fminf(cur_463, pv_473);
    float lo_475 = _min_214;
    cur_463 = ((up[3] != 0) ? hi_474 : lo_475);
    V[2] = cur_463;
    int s0_476 = ((sg * 16 + 3) * 16 + cg) * 17;
    int s1_477 = ((sg * 16 + 3 + 8) * 16 + cg) * 17;
    float x0_478 = pub[s0_476 + ln];
    float y0_479 = pub[s1_477 + lnr];
    float _min_215 = fminf(x0_478, y0_479);
    float lo0_480 = _min_215;
    float _fmax_234 = fmaxf(r, lo0_480);
    r = _fmax_234;
    float _fmax_235 = fmaxf(x0_478, y0_479);
    float hi0_481 = _fmax_235;
    float cur_482 = hi0_481;
    float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, cur_482, 8);
    float pv_483 = _shfl_xor_12;
    float _fmax_236 = fmaxf(cur_482, pv_483);
    float hi_484 = _fmax_236;
    float _min_216 = fminf(cur_482, pv_483);
    float lo_485 = _min_216;
    cur_482 = ((up[0] != 0) ? hi_484 : lo_485);
    float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, cur_482, 4);
    float pv_486 = _shfl_xor_13;
    float _fmax_237 = fmaxf(cur_482, pv_486);
    float hi_487 = _fmax_237;
    float _min_217 = fminf(cur_482, pv_486);
    float lo_488 = _min_217;
    cur_482 = ((up[1] != 0) ? hi_487 : lo_488);
    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, cur_482, 2);
    float pv_489 = _shfl_xor_14;
    float _fmax_238 = fmaxf(cur_482, pv_489);
    float hi_490 = _fmax_238;
    float _min_218 = fminf(cur_482, pv_489);
    float lo_491 = _min_218;
    cur_482 = ((up[2] != 0) ? hi_490 : lo_491);
    float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, cur_482, 1);
    float pv_492 = _shfl_xor_15;
    float _fmax_239 = fmaxf(cur_482, pv_492);
    float hi_493 = _fmax_239;
    float _min_219 = fminf(cur_482, pv_492);
    float lo_494 = _min_219;
    cur_482 = ((up[3] != 0) ? hi_493 : lo_494);
    V[3] = cur_482;
    int s0_495 = ((sg * 16 + 4) * 16 + cg) * 17;
    int s1_496 = ((sg * 16 + 4 + 8) * 16 + cg) * 17;
    float x0_497 = pub[s0_495 + ln];
    float y0_498 = pub[s1_496 + lnr];
    float _min_220 = fminf(x0_497, y0_498);
    float lo0_499 = _min_220;
    float _fmax_240 = fmaxf(r, lo0_499);
    r = _fmax_240;
    float _fmax_241 = fmaxf(x0_497, y0_498);
    float hi0_500 = _fmax_241;
    float cur_501 = hi0_500;
    float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, cur_501, 8);
    float pv_502 = _shfl_xor_16;
    float _fmax_242 = fmaxf(cur_501, pv_502);
    float hi_503 = _fmax_242;
    float _min_221 = fminf(cur_501, pv_502);
    float lo_504 = _min_221;
    cur_501 = ((up[0] != 0) ? hi_503 : lo_504);
    float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, cur_501, 4);
    float pv_505 = _shfl_xor_17;
    float _fmax_243 = fmaxf(cur_501, pv_505);
    float hi_506 = _fmax_243;
    float _min_222 = fminf(cur_501, pv_505);
    float lo_507 = _min_222;
    cur_501 = ((up[1] != 0) ? hi_506 : lo_507);
    float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, cur_501, 2);
    float pv_508 = _shfl_xor_18;
    float _fmax_244 = fmaxf(cur_501, pv_508);
    float hi_509 = _fmax_244;
    float _min_223 = fminf(cur_501, pv_508);
    float lo_510 = _min_223;
    cur_501 = ((up[2] != 0) ? hi_509 : lo_510);
    float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, cur_501, 1);
    float pv_511 = _shfl_xor_19;
    float _fmax_245 = fmaxf(cur_501, pv_511);
    float hi_512 = _fmax_245;
    float _min_224 = fminf(cur_501, pv_511);
    float lo_513 = _min_224;
    cur_501 = ((up[3] != 0) ? hi_512 : lo_513);
    V[4] = cur_501;
    int s0_514 = ((sg * 16 + 5) * 16 + cg) * 17;
    int s1_515 = ((sg * 16 + 5 + 8) * 16 + cg) * 17;
    float x0_516 = pub[s0_514 + ln];
    float y0_517 = pub[s1_515 + lnr];
    float _min_225 = fminf(x0_516, y0_517);
    float lo0_518 = _min_225;
    float _fmax_246 = fmaxf(r, lo0_518);
    r = _fmax_246;
    float _fmax_247 = fmaxf(x0_516, y0_517);
    float hi0_519 = _fmax_247;
    float cur_520 = hi0_519;
    float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, cur_520, 8);
    float pv_521 = _shfl_xor_20;
    float _fmax_248 = fmaxf(cur_520, pv_521);
    float hi_522 = _fmax_248;
    float _min_226 = fminf(cur_520, pv_521);
    float lo_523 = _min_226;
    cur_520 = ((up[0] != 0) ? hi_522 : lo_523);
    float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, cur_520, 4);
    float pv_524 = _shfl_xor_21;
    float _fmax_249 = fmaxf(cur_520, pv_524);
    float hi_525 = _fmax_249;
    float _min_227 = fminf(cur_520, pv_524);
    float lo_526 = _min_227;
    cur_520 = ((up[1] != 0) ? hi_525 : lo_526);
    float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, cur_520, 2);
    float pv_527 = _shfl_xor_22;
    float _fmax_250 = fmaxf(cur_520, pv_527);
    float hi_528 = _fmax_250;
    float _min_228 = fminf(cur_520, pv_527);
    float lo_529 = _min_228;
    cur_520 = ((up[2] != 0) ? hi_528 : lo_529);
    float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, cur_520, 1);
    float pv_530 = _shfl_xor_23;
    float _fmax_251 = fmaxf(cur_520, pv_530);
    float hi_531 = _fmax_251;
    float _min_229 = fminf(cur_520, pv_530);
    float lo_532 = _min_229;
    cur_520 = ((up[3] != 0) ? hi_531 : lo_532);
    V[5] = cur_520;
    int s0_533 = ((sg * 16 + 6) * 16 + cg) * 17;
    int s1_534 = ((sg * 16 + 6 + 8) * 16 + cg) * 17;
    float x0_535 = pub[s0_533 + ln];
    float y0_536 = pub[s1_534 + lnr];
    float _min_230 = fminf(x0_535, y0_536);
    float lo0_537 = _min_230;
    float _fmax_252 = fmaxf(r, lo0_537);
    r = _fmax_252;
    float _fmax_253 = fmaxf(x0_535, y0_536);
    float hi0_538 = _fmax_253;
    float cur_539 = hi0_538;
    float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, cur_539, 8);
    float pv_540 = _shfl_xor_24;
    float _fmax_254 = fmaxf(cur_539, pv_540);
    float hi_541 = _fmax_254;
    float _min_231 = fminf(cur_539, pv_540);
    float lo_542 = _min_231;
    cur_539 = ((up[0] != 0) ? hi_541 : lo_542);
    float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, cur_539, 4);
    float pv_543 = _shfl_xor_25;
    float _fmax_255 = fmaxf(cur_539, pv_543);
    float hi_544 = _fmax_255;
    float _min_232 = fminf(cur_539, pv_543);
    float lo_545 = _min_232;
    cur_539 = ((up[1] != 0) ? hi_544 : lo_545);
    float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, cur_539, 2);
    float pv_546 = _shfl_xor_26;
    float _fmax_256 = fmaxf(cur_539, pv_546);
    float hi_547 = _fmax_256;
    float _min_233 = fminf(cur_539, pv_546);
    float lo_548 = _min_233;
    cur_539 = ((up[2] != 0) ? hi_547 : lo_548);
    float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, cur_539, 1);
    float pv_549 = _shfl_xor_27;
    float _fmax_257 = fmaxf(cur_539, pv_549);
    float hi_550 = _fmax_257;
    float _min_234 = fminf(cur_539, pv_549);
    float lo_551 = _min_234;
    cur_539 = ((up[3] != 0) ? hi_550 : lo_551);
    V[6] = cur_539;
    int s0_552 = ((sg * 16 + 7) * 16 + cg) * 17;
    int s1_553 = ((sg * 16 + 7 + 8) * 16 + cg) * 17;
    float x0_554 = pub[s0_552 + ln];
    float y0_555 = pub[s1_553 + lnr];
    float _min_235 = fminf(x0_554, y0_555);
    float lo0_556 = _min_235;
    float _fmax_258 = fmaxf(r, lo0_556);
    r = _fmax_258;
    float _fmax_259 = fmaxf(x0_554, y0_555);
    float hi0_557 = _fmax_259;
    float cur_558 = hi0_557;
    float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, cur_558, 8);
    float pv_559 = _shfl_xor_28;
    float _fmax_260 = fmaxf(cur_558, pv_559);
    float hi_560 = _fmax_260;
    float _min_236 = fminf(cur_558, pv_559);
    float lo_561 = _min_236;
    cur_558 = ((up[0] != 0) ? hi_560 : lo_561);
    float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, cur_558, 4);
    float pv_562 = _shfl_xor_29;
    float _fmax_261 = fmaxf(cur_558, pv_562);
    float hi_563 = _fmax_261;
    float _min_237 = fminf(cur_558, pv_562);
    float lo_564 = _min_237;
    cur_558 = ((up[1] != 0) ? hi_563 : lo_564);
    float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, cur_558, 2);
    float pv_565 = _shfl_xor_30;
    float _fmax_262 = fmaxf(cur_558, pv_565);
    float hi_566 = _fmax_262;
    float _min_238 = fminf(cur_558, pv_565);
    float lo_567 = _min_238;
    cur_558 = ((up[2] != 0) ? hi_566 : lo_567);
    float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, cur_558, 1);
    float pv_568 = _shfl_xor_31;
    float _fmax_263 = fmaxf(cur_558, pv_568);
    float hi_569 = _fmax_263;
    float _min_239 = fminf(cur_558, pv_568);
    float lo_570 = _min_239;
    cur_558 = ((up[3] != 0) ? hi_569 : lo_570);
    V[7] = cur_558;
    float rs = pub[((sg * 16 + ln) * 16 + cg) * 17 + 16];
    float _fmax_264 = fmaxf(r, rs);
    r = _fmax_264;
    float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
    float y1 = _shfl_xor_32;
    float _min_240 = fminf(V[0], y1);
    float lo1 = _min_240;
    float _fmax_265 = fmaxf(r, lo1);
    r = _fmax_265;
    float _fmax_266 = fmaxf(V[0], y1);
    float hi1 = _fmax_266;
    float cur_571 = hi1;
    float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, cur_571, 8);
    float pv_572 = _shfl_xor_33;
    float _fmax_267 = fmaxf(cur_571, pv_572);
    float hi_573 = _fmax_267;
    float _min_241 = fminf(cur_571, pv_572);
    float lo_574 = _min_241;
    cur_571 = ((up[0] != 0) ? hi_573 : lo_574);
    float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, cur_571, 4);
    float pv_575 = _shfl_xor_34;
    float _fmax_268 = fmaxf(cur_571, pv_575);
    float hi_576 = _fmax_268;
    float _min_242 = fminf(cur_571, pv_575);
    float lo_577 = _min_242;
    cur_571 = ((up[1] != 0) ? hi_576 : lo_577);
    float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, cur_571, 2);
    float pv_578 = _shfl_xor_35;
    float _fmax_269 = fmaxf(cur_571, pv_578);
    float hi_579 = _fmax_269;
    float _min_243 = fminf(cur_571, pv_578);
    float lo_580 = _min_243;
    cur_571 = ((up[2] != 0) ? hi_579 : lo_580);
    float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, cur_571, 1);
    float pv_581 = _shfl_xor_36;
    float _fmax_270 = fmaxf(cur_571, pv_581);
    float hi_582 = _fmax_270;
    float _min_244 = fminf(cur_571, pv_581);
    float lo_583 = _min_244;
    cur_571 = ((up[3] != 0) ? hi_582 : lo_583);
    V[0] = cur_571;
    float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
    float y1_584 = _shfl_xor_37;
    float _min_245 = fminf(V[1], y1_584);
    float lo1_585 = _min_245;
    float _fmax_271 = fmaxf(r, lo1_585);
    r = _fmax_271;
    float _fmax_272 = fmaxf(V[1], y1_584);
    float hi1_586 = _fmax_272;
    float cur_587 = hi1_586;
    float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, cur_587, 8);
    float pv_588 = _shfl_xor_38;
    float _fmax_273 = fmaxf(cur_587, pv_588);
    float hi_589 = _fmax_273;
    float _min_246 = fminf(cur_587, pv_588);
    float lo_590 = _min_246;
    cur_587 = ((up[0] != 0) ? hi_589 : lo_590);
    float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, cur_587, 4);
    float pv_591 = _shfl_xor_39;
    float _fmax_274 = fmaxf(cur_587, pv_591);
    float hi_592 = _fmax_274;
    float _min_247 = fminf(cur_587, pv_591);
    float lo_593 = _min_247;
    cur_587 = ((up[1] != 0) ? hi_592 : lo_593);
    float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, cur_587, 2);
    float pv_594 = _shfl_xor_40;
    float _fmax_275 = fmaxf(cur_587, pv_594);
    float hi_595 = _fmax_275;
    float _min_248 = fminf(cur_587, pv_594);
    float lo_596 = _min_248;
    cur_587 = ((up[2] != 0) ? hi_595 : lo_596);
    float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, cur_587, 1);
    float pv_597 = _shfl_xor_41;
    float _fmax_276 = fmaxf(cur_587, pv_597);
    float hi_598 = _fmax_276;
    float _min_249 = fminf(cur_587, pv_597);
    float lo_599 = _min_249;
    cur_587 = ((up[3] != 0) ? hi_598 : lo_599);
    V[1] = cur_587;
    float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
    float y1_600 = _shfl_xor_42;
    float _min_250 = fminf(V[2], y1_600);
    float lo1_601 = _min_250;
    float _fmax_277 = fmaxf(r, lo1_601);
    r = _fmax_277;
    float _fmax_278 = fmaxf(V[2], y1_600);
    float hi1_602 = _fmax_278;
    float cur_603 = hi1_602;
    float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, cur_603, 8);
    float pv_604 = _shfl_xor_43;
    float _fmax_279 = fmaxf(cur_603, pv_604);
    float hi_605 = _fmax_279;
    float _min_251 = fminf(cur_603, pv_604);
    float lo_606 = _min_251;
    cur_603 = ((up[0] != 0) ? hi_605 : lo_606);
    float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, cur_603, 4);
    float pv_607 = _shfl_xor_44;
    float _fmax_280 = fmaxf(cur_603, pv_607);
    float hi_608 = _fmax_280;
    float _min_252 = fminf(cur_603, pv_607);
    float lo_609 = _min_252;
    cur_603 = ((up[1] != 0) ? hi_608 : lo_609);
    float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, cur_603, 2);
    float pv_610 = _shfl_xor_45;
    float _fmax_281 = fmaxf(cur_603, pv_610);
    float hi_611 = _fmax_281;
    float _min_253 = fminf(cur_603, pv_610);
    float lo_612 = _min_253;
    cur_603 = ((up[2] != 0) ? hi_611 : lo_612);
    float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, cur_603, 1);
    float pv_613 = _shfl_xor_46;
    float _fmax_282 = fmaxf(cur_603, pv_613);
    float hi_614 = _fmax_282;
    float _min_254 = fminf(cur_603, pv_613);
    float lo_615 = _min_254;
    cur_603 = ((up[3] != 0) ? hi_614 : lo_615);
    V[2] = cur_603;
    float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
    float y1_616 = _shfl_xor_47;
    float _min_255 = fminf(V[3], y1_616);
    float lo1_617 = _min_255;
    float _fmax_283 = fmaxf(r, lo1_617);
    r = _fmax_283;
    float _fmax_284 = fmaxf(V[3], y1_616);
    float hi1_618 = _fmax_284;
    float cur_619 = hi1_618;
    float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, cur_619, 8);
    float pv_620 = _shfl_xor_48;
    float _fmax_285 = fmaxf(cur_619, pv_620);
    float hi_621 = _fmax_285;
    float _min_256 = fminf(cur_619, pv_620);
    float lo_622 = _min_256;
    cur_619 = ((up[0] != 0) ? hi_621 : lo_622);
    float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, cur_619, 4);
    float pv_623 = _shfl_xor_49;
    float _fmax_286 = fmaxf(cur_619, pv_623);
    float hi_624 = _fmax_286;
    float _min_257 = fminf(cur_619, pv_623);
    float lo_625 = _min_257;
    cur_619 = ((up[1] != 0) ? hi_624 : lo_625);
    float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, cur_619, 2);
    float pv_626 = _shfl_xor_50;
    float _fmax_287 = fmaxf(cur_619, pv_626);
    float hi_627 = _fmax_287;
    float _min_258 = fminf(cur_619, pv_626);
    float lo_628 = _min_258;
    cur_619 = ((up[2] != 0) ? hi_627 : lo_628);
    float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, cur_619, 1);
    float pv_629 = _shfl_xor_51;
    float _fmax_288 = fmaxf(cur_619, pv_629);
    float hi_630 = _fmax_288;
    float _min_259 = fminf(cur_619, pv_629);
    float lo_631 = _min_259;
    cur_619 = ((up[3] != 0) ? hi_630 : lo_631);
    V[3] = cur_619;
    float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
    float y1_632 = _shfl_xor_52;
    float _min_260 = fminf(V[0], y1_632);
    float lo1_633 = _min_260;
    float _fmax_289 = fmaxf(r, lo1_633);
    r = _fmax_289;
    float _fmax_290 = fmaxf(V[0], y1_632);
    float hi1_634 = _fmax_290;
    float cur_635 = hi1_634;
    float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, cur_635, 8);
    float pv_636 = _shfl_xor_53;
    float _fmax_291 = fmaxf(cur_635, pv_636);
    float hi_637 = _fmax_291;
    float _min_261 = fminf(cur_635, pv_636);
    float lo_638 = _min_261;
    cur_635 = ((up[0] != 0) ? hi_637 : lo_638);
    float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, cur_635, 4);
    float pv_639 = _shfl_xor_54;
    float _fmax_292 = fmaxf(cur_635, pv_639);
    float hi_640 = _fmax_292;
    float _min_262 = fminf(cur_635, pv_639);
    float lo_641 = _min_262;
    cur_635 = ((up[1] != 0) ? hi_640 : lo_641);
    float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, cur_635, 2);
    float pv_642 = _shfl_xor_55;
    float _fmax_293 = fmaxf(cur_635, pv_642);
    float hi_643 = _fmax_293;
    float _min_263 = fminf(cur_635, pv_642);
    float lo_644 = _min_263;
    cur_635 = ((up[2] != 0) ? hi_643 : lo_644);
    float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, cur_635, 1);
    float pv_645 = _shfl_xor_56;
    float _fmax_294 = fmaxf(cur_635, pv_645);
    float hi_646 = _fmax_294;
    float _min_264 = fminf(cur_635, pv_645);
    float lo_647 = _min_264;
    cur_635 = ((up[3] != 0) ? hi_646 : lo_647);
    V[0] = cur_635;
    float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
    float y1_648 = _shfl_xor_57;
    float _min_265 = fminf(V[1], y1_648);
    float lo1_649 = _min_265;
    float _fmax_295 = fmaxf(r, lo1_649);
    r = _fmax_295;
    float _fmax_296 = fmaxf(V[1], y1_648);
    float hi1_650 = _fmax_296;
    float cur_651 = hi1_650;
    float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, cur_651, 8);
    float pv_652 = _shfl_xor_58;
    float _fmax_297 = fmaxf(cur_651, pv_652);
    float hi_653 = _fmax_297;
    float _min_266 = fminf(cur_651, pv_652);
    float lo_654 = _min_266;
    cur_651 = ((up[0] != 0) ? hi_653 : lo_654);
    float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, cur_651, 4);
    float pv_655 = _shfl_xor_59;
    float _fmax_298 = fmaxf(cur_651, pv_655);
    float hi_656 = _fmax_298;
    float _min_267 = fminf(cur_651, pv_655);
    float lo_657 = _min_267;
    cur_651 = ((up[1] != 0) ? hi_656 : lo_657);
    float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, cur_651, 2);
    float pv_658 = _shfl_xor_60;
    float _fmax_299 = fmaxf(cur_651, pv_658);
    float hi_659 = _fmax_299;
    float _min_268 = fminf(cur_651, pv_658);
    float lo_660 = _min_268;
    cur_651 = ((up[2] != 0) ? hi_659 : lo_660);
    float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, cur_651, 1);
    float pv_661 = _shfl_xor_61;
    float _fmax_300 = fmaxf(cur_651, pv_661);
    float hi_662 = _fmax_300;
    float _min_269 = fminf(cur_651, pv_661);
    float lo_663 = _min_269;
    cur_651 = ((up[3] != 0) ? hi_662 : lo_663);
    V[1] = cur_651;
    float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
    float yl = _shfl_xor_62;
    float _min_270 = fminf(V[0], yl);
    float lol = _min_270;
    float _fmax_301 = fmaxf(r, lol);
    r = _fmax_301;
    float _fmax_302 = fmaxf(V[0], yl);
    float hil = _fmax_302;
    V[0] = hil;
    float K = V[0];
    rr1[0] = r;
    int ucol = blockIdx.x * 16 + cg;
    int commit = 0;
    if (g < 16 && ucol < total_q) {
        commit = 1;
    }
    if (tid_1 < 256) {
        float u16 = K;
        float cr = rr1[0];
        float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, u16, 8);
        float _min_271 = fminf(u16, _shfl_xor_63);
        u16 = _min_271;
        float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, cr, 8);
        float _fmax_303 = fmaxf(cr, _shfl_xor_64);
        cr = _fmax_303;
        float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, u16, 4);
        float _min_272 = fminf(u16, _shfl_xor_65);
        u16 = _min_272;
        float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, cr, 4);
        float _fmax_304 = fmaxf(cr, _shfl_xor_66);
        cr = _fmax_304;
        float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, u16, 2);
        float _min_273 = fminf(u16, _shfl_xor_67);
        u16 = _min_273;
        float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, cr, 2);
        float _fmax_305 = fmaxf(cr, _shfl_xor_68);
        cr = _fmax_305;
        float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, u16, 1);
        float _min_274 = fminf(u16, _shfl_xor_69);
        u16 = _min_274;
        float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, cr, 1);
        float _fmax_306 = fmaxf(cr, _shfl_xor_70);
        cr = _fmax_306;
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
        if (ln == 0 && g < 16) {
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
        if (flagged != 0) {
            unsigned int mm = 0;
            float sc_1 = __uint_as_float(cbt[0]);
            float _fmax_307 = fmaxf(sc_1, -1.7014118346046923e+38f);
            sc_1 = _fmax_307;
            float _min_275 = fminf(sc_1, 1.7014118346046923e+38f);
            sc_1 = _min_275;
            sc_1 = sc_1;
            unsigned int cls9 = __as_u32(sc_1) & 4294966784u;
            if (cls9 == qc) {
                mm = mm | 1;
            }
            float sc_2 = __uint_as_float(cbt[1]);
            float _fmax_308 = fmaxf(sc_2, -1.7014118346046923e+38f);
            sc_2 = _fmax_308;
            float _min_276 = fminf(sc_2, 1.7014118346046923e+38f);
            sc_2 = _min_276;
            sc_2 = sc_2;
            unsigned int cls9_3 = __as_u32(sc_2) & 4294966784u;
            if (cls9_3 == qc) {
                mm = mm | 2;
            }
            float sc_4 = __uint_as_float(cbt[2]);
            float _fmax_309 = fmaxf(sc_4, -1.7014118346046923e+38f);
            sc_4 = _fmax_309;
            float _min_277 = fminf(sc_4, 1.7014118346046923e+38f);
            sc_4 = _min_277;
            sc_4 = sc_4;
            unsigned int cls9_5 = __as_u32(sc_4) & 4294966784u;
            if (cls9_5 == qc) {
                mm = mm | 4;
            }
            float sc_7 = __uint_as_float(cbt[3]);
            float _fmax_310 = fmaxf(sc_7, -1.7014118346046923e+38f);
            sc_7 = _fmax_310;
            float _min_278 = fminf(sc_7, 1.7014118346046923e+38f);
            sc_7 = _min_278;
            sc_7 = sc_7;
            unsigned int cls9_8 = __as_u32(sc_7) & 4294966784u;
            if (cls9_8 == qc) {
                mm = mm | 8;
            }
            float sc_10 = __uint_as_float(cbt[4]);
            float _fmax_311 = fmaxf(sc_10, -1.7014118346046923e+38f);
            sc_10 = _fmax_311;
            float _min_279 = fminf(sc_10, 1.7014118346046923e+38f);
            sc_10 = _min_279;
            sc_10 = sc_10;
            unsigned int cls9_11 = __as_u32(sc_10) & 4294966784u;
            if (cls9_11 == qc) {
                mm = mm | 16;
            }
            float sc_13 = __uint_as_float(cbt[5]);
            float _fmax_312 = fmaxf(sc_13, -1.7014118346046923e+38f);
            sc_13 = _fmax_312;
            float _min_280 = fminf(sc_13, 1.7014118346046923e+38f);
            sc_13 = _min_280;
            sc_13 = sc_13;
            unsigned int cls9_14 = __as_u32(sc_13) & 4294966784u;
            if (cls9_14 == qc) {
                mm = mm | 32;
            }
            float sc_16 = __uint_as_float(cbt[6]);
            float _fmax_313 = fmaxf(sc_16, -1.7014118346046923e+38f);
            sc_16 = _fmax_313;
            float _min_281 = fminf(sc_16, 1.7014118346046923e+38f);
            sc_16 = _min_281;
            sc_16 = sc_16;
            unsigned int cls9_17 = __as_u32(sc_16) & 4294966784u;
            if (cls9_17 == qc) {
                mm = mm | 64;
            }
            float sc_19 = __uint_as_float(cbt[7]);
            float _fmax_314 = fmaxf(sc_19, -1.7014118346046923e+38f);
            sc_19 = _fmax_314;
            float _min_282 = fminf(sc_19, 1.7014118346046923e+38f);
            sc_19 = _min_282;
            sc_19 = sc_19;
            unsigned int cls9_20 = __as_u32(sc_19) & 4294966784u;
            if (cls9_20 == qc) {
                mm = mm | 128;
            }
            float sc_22 = __uint_as_float(cbt[8]);
            float _fmax_315 = fmaxf(sc_22, -1.7014118346046923e+38f);
            sc_22 = _fmax_315;
            float _min_283 = fminf(sc_22, 1.7014118346046923e+38f);
            sc_22 = _min_283;
            sc_22 = sc_22;
            unsigned int cls9_23 = __as_u32(sc_22) & 4294966784u;
            if (cls9_23 == qc) {
                mm = mm | 256;
            }
            float sc_25 = __uint_as_float(cbt[9]);
            float _fmax_316 = fmaxf(sc_25, -1.7014118346046923e+38f);
            sc_25 = _fmax_316;
            float _min_284 = fminf(sc_25, 1.7014118346046923e+38f);
            sc_25 = _min_284;
            sc_25 = sc_25;
            unsigned int cls9_26 = __as_u32(sc_25) & 4294966784u;
            if (cls9_26 == qc) {
                mm = mm | 512;
            }
            float sc_28 = __uint_as_float(cbt[10]);
            float _fmax_317 = fmaxf(sc_28, -1.7014118346046923e+38f);
            sc_28 = _fmax_317;
            float _min_285 = fminf(sc_28, 1.7014118346046923e+38f);
            sc_28 = _min_285;
            sc_28 = sc_28;
            unsigned int cls9_29 = __as_u32(sc_28) & 4294966784u;
            if (cls9_29 == qc) {
                mm = mm | 1024;
            }
            float sc_31 = __uint_as_float(cbt[11]);
            float _fmax_318 = fmaxf(sc_31, -1.7014118346046923e+38f);
            sc_31 = _fmax_318;
            float _min_286 = fminf(sc_31, 1.7014118346046923e+38f);
            sc_31 = _min_286;
            sc_31 = sc_31;
            unsigned int cls9_32 = __as_u32(sc_31) & 4294966784u;
            if (cls9_32 == qc) {
                mm = mm | 2048;
            }
            float sc_34 = __uint_as_float(cbt[12]);
            float _fmax_319 = fmaxf(sc_34, -1.7014118346046923e+38f);
            sc_34 = _fmax_319;
            float _min_287 = fminf(sc_34, 1.7014118346046923e+38f);
            sc_34 = _min_287;
            sc_34 = sc_34;
            unsigned int cls9_35 = __as_u32(sc_34) & 4294966784u;
            if (cls9_35 == qc) {
                mm = mm | 4096;
            }
            float sc_37 = __uint_as_float(cbt[13]);
            float _fmax_320 = fmaxf(sc_37, -1.7014118346046923e+38f);
            sc_37 = _fmax_320;
            float _min_288 = fminf(sc_37, 1.7014118346046923e+38f);
            sc_37 = _min_288;
            sc_37 = sc_37;
            unsigned int cls9_38 = __as_u32(sc_37) & 4294966784u;
            if (cls9_38 == qc) {
                mm = mm | 8192;
            }
            float sc_40 = __uint_as_float(cbt[14]);
            float _fmax_321 = fmaxf(sc_40, -1.7014118346046923e+38f);
            sc_40 = _fmax_321;
            float _min_289 = fminf(sc_40, 1.7014118346046923e+38f);
            sc_40 = _min_289;
            sc_40 = sc_40;
            unsigned int cls9_41 = __as_u32(sc_40) & 4294966784u;
            if (cls9_41 == qc) {
                mm = mm | 16384;
            }
            float sc_43 = __uint_as_float(cbt[15]);
            float _fmax_322 = fmaxf(sc_43, -1.7014118346046923e+38f);
            sc_43 = _fmax_322;
            float _min_290 = fminf(sc_43, 1.7014118346046923e+38f);
            sc_43 = _min_290;
            sc_43 = sc_43;
            unsigned int cls9_44 = __as_u32(sc_43) & 4294966784u;
            if (cls9_44 == qc) {
                mm = mm | 32768;
            }
            float sc_46 = __uint_as_float(cbt[16]);
            float _fmax_323 = fmaxf(sc_46, -1.7014118346046923e+38f);
            sc_46 = _fmax_323;
            float _min_291 = fminf(sc_46, 1.7014118346046923e+38f);
            sc_46 = _min_291;
            sc_46 = sc_46;
            unsigned int cls9_47 = __as_u32(sc_46) & 4294966784u;
            if (cls9_47 == qc) {
                mm = mm | 65536;
            }
            float sc_49 = __uint_as_float(cbt[17]);
            float _fmax_324 = fmaxf(sc_49, -1.7014118346046923e+38f);
            sc_49 = _fmax_324;
            float _min_292 = fminf(sc_49, 1.7014118346046923e+38f);
            sc_49 = _min_292;
            sc_49 = sc_49;
            unsigned int cls9_50 = __as_u32(sc_49) & 4294966784u;
            if (cls9_50 == qc) {
                mm = mm | 131072;
            }
            float sc_52 = __uint_as_float(cbt[18]);
            float _fmax_325 = fmaxf(sc_52, -1.7014118346046923e+38f);
            sc_52 = _fmax_325;
            float _min_293 = fminf(sc_52, 1.7014118346046923e+38f);
            sc_52 = _min_293;
            sc_52 = sc_52;
            unsigned int cls9_53 = __as_u32(sc_52) & 4294966784u;
            if (cls9_53 == qc) {
                mm = mm | 262144;
            }
            float sc_55 = __uint_as_float(cbt[19]);
            float _fmax_326 = fmaxf(sc_55, -1.7014118346046923e+38f);
            sc_55 = _fmax_326;
            float _min_294 = fminf(sc_55, 1.7014118346046923e+38f);
            sc_55 = _min_294;
            sc_55 = sc_55;
            unsigned int cls9_56 = __as_u32(sc_55) & 4294966784u;
            if (cls9_56 == qc) {
                mm = mm | 524288;
            }
            float sc_58 = __uint_as_float(cbt[20]);
            float _fmax_327 = fmaxf(sc_58, -1.7014118346046923e+38f);
            sc_58 = _fmax_327;
            float _min_295 = fminf(sc_58, 1.7014118346046923e+38f);
            sc_58 = _min_295;
            sc_58 = sc_58;
            unsigned int cls9_59 = __as_u32(sc_58) & 4294966784u;
            if (cls9_59 == qc) {
                mm = mm | 1048576;
            }
            float sc_61 = __uint_as_float(cbt[21]);
            float _fmax_328 = fmaxf(sc_61, -1.7014118346046923e+38f);
            sc_61 = _fmax_328;
            float _min_296 = fminf(sc_61, 1.7014118346046923e+38f);
            sc_61 = _min_296;
            sc_61 = sc_61;
            unsigned int cls9_62 = __as_u32(sc_61) & 4294966784u;
            if (cls9_62 == qc) {
                mm = mm | 2097152;
            }
            float sc_64 = __uint_as_float(cbt[22]);
            float _fmax_329 = fmaxf(sc_64, -1.7014118346046923e+38f);
            sc_64 = _fmax_329;
            float _min_297 = fminf(sc_64, 1.7014118346046923e+38f);
            sc_64 = _min_297;
            sc_64 = sc_64;
            unsigned int cls9_65 = __as_u32(sc_64) & 4294966784u;
            if (cls9_65 == qc) {
                mm = mm | 4194304;
            }
            float sc_67 = __uint_as_float(cbt[23]);
            float _fmax_330 = fmaxf(sc_67, -1.7014118346046923e+38f);
            sc_67 = _fmax_330;
            float _min_298 = fminf(sc_67, 1.7014118346046923e+38f);
            sc_67 = _min_298;
            sc_67 = sc_67;
            unsigned int cls9_68 = __as_u32(sc_67) & 4294966784u;
            if (cls9_68 == qc) {
                mm = mm | 8388608;
            }
            float sc_70 = __uint_as_float(cbt[24]);
            float _fmax_331 = fmaxf(sc_70, -1.7014118346046923e+38f);
            sc_70 = _fmax_331;
            float _min_299 = fminf(sc_70, 1.7014118346046923e+38f);
            sc_70 = _min_299;
            sc_70 = sc_70;
            unsigned int cls9_71 = __as_u32(sc_70) & 4294966784u;
            if (cls9_71 == qc) {
                mm = mm | 16777216;
            }
            float sc_73 = __uint_as_float(cbt[25]);
            float _fmax_332 = fmaxf(sc_73, -1.7014118346046923e+38f);
            sc_73 = _fmax_332;
            float _min_300 = fminf(sc_73, 1.7014118346046923e+38f);
            sc_73 = _min_300;
            sc_73 = sc_73;
            unsigned int cls9_74 = __as_u32(sc_73) & 4294966784u;
            if (cls9_74 == qc) {
                mm = mm | 33554432;
            }
            float sc_76 = __uint_as_float(cbt[26]);
            float _fmax_333 = fmaxf(sc_76, -1.7014118346046923e+38f);
            sc_76 = _fmax_333;
            float _min_301 = fminf(sc_76, 1.7014118346046923e+38f);
            sc_76 = _min_301;
            sc_76 = sc_76;
            unsigned int cls9_77 = __as_u32(sc_76) & 4294966784u;
            if (cls9_77 == qc) {
                mm = mm | 67108864;
            }
            float sc_79 = __uint_as_float(cbt[27]);
            float _fmax_334 = fmaxf(sc_79, -1.7014118346046923e+38f);
            sc_79 = _fmax_334;
            float _min_302 = fminf(sc_79, 1.7014118346046923e+38f);
            sc_79 = _min_302;
            sc_79 = sc_79;
            unsigned int cls9_80 = __as_u32(sc_79) & 4294966784u;
            if (cls9_80 == qc) {
                mm = mm | 134217728;
            }
            float sc_82 = __uint_as_float(cbt[28]);
            float _fmax_335 = fmaxf(sc_82, -1.7014118346046923e+38f);
            sc_82 = _fmax_335;
            float _min_303 = fminf(sc_82, 1.7014118346046923e+38f);
            sc_82 = _min_303;
            sc_82 = sc_82;
            unsigned int cls9_83 = __as_u32(sc_82) & 4294966784u;
            if (cls9_83 == qc) {
                mm = mm | 268435456;
            }
            float sc_85 = __uint_as_float(cbt[29]);
            float _fmax_336 = fmaxf(sc_85, -1.7014118346046923e+38f);
            sc_85 = _fmax_336;
            float _min_304 = fminf(sc_85, 1.7014118346046923e+38f);
            sc_85 = _min_304;
            sc_85 = sc_85;
            unsigned int cls9_86 = __as_u32(sc_85) & 4294966784u;
            if (cls9_86 == qc) {
                mm = mm | 536870912;
            }
            float sc_88 = __uint_as_float(cbt[30]);
            float _fmax_337 = fmaxf(sc_88, -1.7014118346046923e+38f);
            sc_88 = _fmax_337;
            float _min_305 = fminf(sc_88, 1.7014118346046923e+38f);
            sc_88 = _min_305;
            sc_88 = sc_88;
            unsigned int cls9_89 = __as_u32(sc_88) & 4294966784u;
            if (cls9_89 == qc) {
                mm = mm | 1073741824;
            }
            float sc_91 = __uint_as_float(cbt[31]);
            float _fmax_338 = fmaxf(sc_91, -1.7014118346046923e+38f);
            sc_91 = _fmax_338;
            float _min_306 = fminf(sc_91, 1.7014118346046923e+38f);
            sc_91 = _min_306;
            sc_91 = sc_91;
            unsigned int cls9_92 = __as_u32(sc_91) & 4294966784u;
            if (cls9_92 == qc) {
                mm = mm | 2147483648u;
            }
            int f_93 = 0;
            if ((tb < fb || tb + 32 > lim - fe) && tb < lim) {
                f_93 = 1;
            }
            int fsp = f_93;
            if (fsp != 0) {
                int f_0_1 = 0;
                if ((tb < fb || tb >= lim - fe) && lim > tb) {
                    f_0_1 = 1;
                }
                if (f_0_1 != 0) {
                    mm = mm & 4294967294u;
                }
                int f_1_1 = 0;
                if ((tb + 1 < fb || tb + 1 >= lim - fe) && lim > tb + 1) {
                    f_1_1 = 1;
                }
                if (f_1_1 != 0) {
                    mm = mm & 4294967293u;
                }
                int f_2_1 = 0;
                if ((tb + 2 < fb || tb + 2 >= lim - fe) && lim > tb + 2) {
                    f_2_1 = 1;
                }
                if (f_2_1 != 0) {
                    mm = mm & 4294967291u;
                }
                int f_3_1 = 0;
                if ((tb + 3 < fb || tb + 3 >= lim - fe) && lim > tb + 3) {
                    f_3_1 = 1;
                }
                if (f_3_1 != 0) {
                    mm = mm & 4294967287u;
                }
                int f_4_1 = 0;
                if ((tb + 4 < fb || tb + 4 >= lim - fe) && lim > tb + 4) {
                    f_4_1 = 1;
                }
                if (f_4_1 != 0) {
                    mm = mm & 4294967279u;
                }
                int f_5_1 = 0;
                if ((tb + 5 < fb || tb + 5 >= lim - fe) && lim > tb + 5) {
                    f_5_1 = 1;
                }
                if (f_5_1 != 0) {
                    mm = mm & 4294967263u;
                }
                int f_6_1 = 0;
                if ((tb + 6 < fb || tb + 6 >= lim - fe) && lim > tb + 6) {
                    f_6_1 = 1;
                }
                if (f_6_1 != 0) {
                    mm = mm & 4294967231u;
                }
                int f_7_1 = 0;
                if ((tb + 7 < fb || tb + 7 >= lim - fe) && lim > tb + 7) {
                    f_7_1 = 1;
                }
                if (f_7_1 != 0) {
                    mm = mm & 4294967167u;
                }
                int f_8_1 = 0;
                if ((tb + 8 < fb || tb + 8 >= lim - fe) && lim > tb + 8) {
                    f_8_1 = 1;
                }
                if (f_8_1 != 0) {
                    mm = mm & 4294967039u;
                }
                int f_9_1 = 0;
                if ((tb + 9 < fb || tb + 9 >= lim - fe) && lim > tb + 9) {
                    f_9_1 = 1;
                }
                if (f_9_1 != 0) {
                    mm = mm & 4294966783u;
                }
                int f_10_1 = 0;
                if ((tb + 10 < fb || tb + 10 >= lim - fe) && lim > tb + 10) {
                    f_10_1 = 1;
                }
                if (f_10_1 != 0) {
                    mm = mm & 4294966271u;
                }
                int f_11_1 = 0;
                if ((tb + 11 < fb || tb + 11 >= lim - fe) && lim > tb + 11) {
                    f_11_1 = 1;
                }
                if (f_11_1 != 0) {
                    mm = mm & 4294965247u;
                }
                int f_12_1 = 0;
                if ((tb + 12 < fb || tb + 12 >= lim - fe) && lim > tb + 12) {
                    f_12_1 = 1;
                }
                if (f_12_1 != 0) {
                    mm = mm & 4294963199u;
                }
                int f_13_1 = 0;
                if ((tb + 13 < fb || tb + 13 >= lim - fe) && lim > tb + 13) {
                    f_13_1 = 1;
                }
                if (f_13_1 != 0) {
                    mm = mm & 4294959103u;
                }
                int f_14_1 = 0;
                if ((tb + 14 < fb || tb + 14 >= lim - fe) && lim > tb + 14) {
                    f_14_1 = 1;
                }
                if (f_14_1 != 0) {
                    mm = mm & 4294950911u;
                }
                int f_15_1 = 0;
                if ((tb + 15 < fb || tb + 15 >= lim - fe) && lim > tb + 15) {
                    f_15_1 = 1;
                }
                if (f_15_1 != 0) {
                    mm = mm & 4294934527u;
                }
                int f_16_1 = 0;
                if ((tb + 16 < fb || tb + 16 >= lim - fe) && lim > tb + 16) {
                    f_16_1 = 1;
                }
                if (f_16_1 != 0) {
                    mm = mm & 4294901759u;
                }
                int f_17_1 = 0;
                if ((tb + 17 < fb || tb + 17 >= lim - fe) && lim > tb + 17) {
                    f_17_1 = 1;
                }
                if (f_17_1 != 0) {
                    mm = mm & 4294836223u;
                }
                int f_18_1 = 0;
                if ((tb + 18 < fb || tb + 18 >= lim - fe) && lim > tb + 18) {
                    f_18_1 = 1;
                }
                if (f_18_1 != 0) {
                    mm = mm & 4294705151u;
                }
                int f_19_1 = 0;
                if ((tb + 19 < fb || tb + 19 >= lim - fe) && lim > tb + 19) {
                    f_19_1 = 1;
                }
                if (f_19_1 != 0) {
                    mm = mm & 4294443007u;
                }
                int f_20_1 = 0;
                if ((tb + 20 < fb || tb + 20 >= lim - fe) && lim > tb + 20) {
                    f_20_1 = 1;
                }
                if (f_20_1 != 0) {
                    mm = mm & 4293918719u;
                }
                int f_21_1 = 0;
                if ((tb + 21 < fb || tb + 21 >= lim - fe) && lim > tb + 21) {
                    f_21_1 = 1;
                }
                if (f_21_1 != 0) {
                    mm = mm & 4292870143u;
                }
                int f_22_1 = 0;
                if ((tb + 22 < fb || tb + 22 >= lim - fe) && lim > tb + 22) {
                    f_22_1 = 1;
                }
                if (f_22_1 != 0) {
                    mm = mm & 4290772991u;
                }
                int f_23_1 = 0;
                if ((tb + 23 < fb || tb + 23 >= lim - fe) && lim > tb + 23) {
                    f_23_1 = 1;
                }
                if (f_23_1 != 0) {
                    mm = mm & 4286578687u;
                }
                int f_24_1 = 0;
                if ((tb + 24 < fb || tb + 24 >= lim - fe) && lim > tb + 24) {
                    f_24_1 = 1;
                }
                if (f_24_1 != 0) {
                    mm = mm & 4278190079u;
                }
                int f_25_1 = 0;
                if ((tb + 25 < fb || tb + 25 >= lim - fe) && lim > tb + 25) {
                    f_25_1 = 1;
                }
                if (f_25_1 != 0) {
                    mm = mm & 4261412863u;
                }
                int f_26_1 = 0;
                if ((tb + 26 < fb || tb + 26 >= lim - fe) && lim > tb + 26) {
                    f_26_1 = 1;
                }
                if (f_26_1 != 0) {
                    mm = mm & 4227858431u;
                }
                int f_27_1 = 0;
                if ((tb + 27 < fb || tb + 27 >= lim - fe) && lim > tb + 27) {
                    f_27_1 = 1;
                }
                if (f_27_1 != 0) {
                    mm = mm & 4160749567u;
                }
                int f_28_1 = 0;
                if ((tb + 28 < fb || tb + 28 >= lim - fe) && lim > tb + 28) {
                    f_28_1 = 1;
                }
                if (f_28_1 != 0) {
                    mm = mm & 4026531839u;
                }
                int f_29_1 = 0;
                if ((tb + 29 < fb || tb + 29 >= lim - fe) && lim > tb + 29) {
                    f_29_1 = 1;
                }
                if (f_29_1 != 0) {
                    mm = mm & 3758096383u;
                }
                int f_30_1 = 0;
                if ((tb + 30 < fb || tb + 30 >= lim - fe) && lim > tb + 30) {
                    f_30_1 = 1;
                }
                if (f_30_1 != 0) {
                    mm = mm & 3221225471u;
                }
                int f_31_1 = 0;
                if ((tb + 31 < fb || tb + 31 >= lim - fe) && lim > tb + 31) {
                    f_31_1 = 1;
                }
                if (f_31_1 != 0) {
                    mm = mm & 2147483647;
                }
            }
            unsigned int mm_94 = mm;
            if (mm_94 != 0) {
                int _popc_0 = __popc(mm_94);
                int n9 = _popc_0;
                unsigned int _atomic_old_0 = atomicAdd(&ccnt[c], (unsigned int)n9);
                unsigned int base9 = _atomic_old_0;
                if (base9 + (unsigned int)n9 > 16) {
                    flagw[1] = 1;
                } else {
                    int pos9 = c * 16 + (int)base9;
                    if ((mm_94 & 1) != 0) {
                        float sc_5 = __uint_as_float(cbt[0]);
                        float _fmax_339 = fmaxf(sc_5, -1.7014118346046923e+38f);
                        sc_5 = _fmax_339;
                        float _min_307 = fminf(sc_5, 1.7014118346046923e+38f);
                        sc_5 = _min_307;
                        sc_5 = sc_5;
                        unsigned int u9 = __as_u32(sc_5);
                        unsigned int lowb9 = (u9 ^ (unsigned int)((int)u9 >> 31) & 511) & 511;
                        unsigned int km9 = 536870912 | lowb9 << 9 | (unsigned int)tb;
                        cbuf[pos9] = km9;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 2) != 0) {
                        float sc_5_1 = __uint_as_float(cbt[1]);
                        float _fmax_340 = fmaxf(sc_5_1, -1.7014118346046923e+38f);
                        sc_5_1 = _fmax_340;
                        float _min_308 = fminf(sc_5_1, 1.7014118346046923e+38f);
                        sc_5_1 = _min_308;
                        sc_5_1 = sc_5_1;
                        unsigned int u9_1 = __as_u32(sc_5_1);
                        unsigned int lowb9_1 = (u9_1 ^ (unsigned int)((int)u9_1 >> 31) & 511) & 511;
                        unsigned int km9_1 = 536870912 | lowb9_1 << 9 | (unsigned int)(tb + 1);
                        cbuf[pos9] = km9_1;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 4) != 0) {
                        float sc_5_2 = __uint_as_float(cbt[2]);
                        float _fmax_341 = fmaxf(sc_5_2, -1.7014118346046923e+38f);
                        sc_5_2 = _fmax_341;
                        float _min_309 = fminf(sc_5_2, 1.7014118346046923e+38f);
                        sc_5_2 = _min_309;
                        sc_5_2 = sc_5_2;
                        unsigned int u9_2 = __as_u32(sc_5_2);
                        unsigned int lowb9_2 = (u9_2 ^ (unsigned int)((int)u9_2 >> 31) & 511) & 511;
                        unsigned int km9_2 = 536870912 | lowb9_2 << 9 | (unsigned int)(tb + 2);
                        cbuf[pos9] = km9_2;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 8) != 0) {
                        float sc_5_3 = __uint_as_float(cbt[3]);
                        float _fmax_342 = fmaxf(sc_5_3, -1.7014118346046923e+38f);
                        sc_5_3 = _fmax_342;
                        float _min_310 = fminf(sc_5_3, 1.7014118346046923e+38f);
                        sc_5_3 = _min_310;
                        sc_5_3 = sc_5_3;
                        unsigned int u9_3 = __as_u32(sc_5_3);
                        unsigned int lowb9_3 = (u9_3 ^ (unsigned int)((int)u9_3 >> 31) & 511) & 511;
                        unsigned int km9_3 = 536870912 | lowb9_3 << 9 | (unsigned int)(tb + 3);
                        cbuf[pos9] = km9_3;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 16) != 0) {
                        float sc_5_4 = __uint_as_float(cbt[4]);
                        float _fmax_343 = fmaxf(sc_5_4, -1.7014118346046923e+38f);
                        sc_5_4 = _fmax_343;
                        float _min_311 = fminf(sc_5_4, 1.7014118346046923e+38f);
                        sc_5_4 = _min_311;
                        sc_5_4 = sc_5_4;
                        unsigned int u9_4 = __as_u32(sc_5_4);
                        unsigned int lowb9_4 = (u9_4 ^ (unsigned int)((int)u9_4 >> 31) & 511) & 511;
                        unsigned int km9_4 = 536870912 | lowb9_4 << 9 | (unsigned int)(tb + 4);
                        cbuf[pos9] = km9_4;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 32) != 0) {
                        float sc_5_5 = __uint_as_float(cbt[5]);
                        float _fmax_344 = fmaxf(sc_5_5, -1.7014118346046923e+38f);
                        sc_5_5 = _fmax_344;
                        float _min_312 = fminf(sc_5_5, 1.7014118346046923e+38f);
                        sc_5_5 = _min_312;
                        sc_5_5 = sc_5_5;
                        unsigned int u9_5 = __as_u32(sc_5_5);
                        unsigned int lowb9_5 = (u9_5 ^ (unsigned int)((int)u9_5 >> 31) & 511) & 511;
                        unsigned int km9_5 = 536870912 | lowb9_5 << 9 | (unsigned int)(tb + 5);
                        cbuf[pos9] = km9_5;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 64) != 0) {
                        float sc_5_6 = __uint_as_float(cbt[6]);
                        float _fmax_345 = fmaxf(sc_5_6, -1.7014118346046923e+38f);
                        sc_5_6 = _fmax_345;
                        float _min_313 = fminf(sc_5_6, 1.7014118346046923e+38f);
                        sc_5_6 = _min_313;
                        sc_5_6 = sc_5_6;
                        unsigned int u9_6 = __as_u32(sc_5_6);
                        unsigned int lowb9_6 = (u9_6 ^ (unsigned int)((int)u9_6 >> 31) & 511) & 511;
                        unsigned int km9_6 = 536870912 | lowb9_6 << 9 | (unsigned int)(tb + 6);
                        cbuf[pos9] = km9_6;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 128) != 0) {
                        float sc_5_7 = __uint_as_float(cbt[7]);
                        float _fmax_346 = fmaxf(sc_5_7, -1.7014118346046923e+38f);
                        sc_5_7 = _fmax_346;
                        float _min_314 = fminf(sc_5_7, 1.7014118346046923e+38f);
                        sc_5_7 = _min_314;
                        sc_5_7 = sc_5_7;
                        unsigned int u9_7 = __as_u32(sc_5_7);
                        unsigned int lowb9_7 = (u9_7 ^ (unsigned int)((int)u9_7 >> 31) & 511) & 511;
                        unsigned int km9_7 = 536870912 | lowb9_7 << 9 | (unsigned int)(tb + 7);
                        cbuf[pos9] = km9_7;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 256) != 0) {
                        float sc_5_8 = __uint_as_float(cbt[8]);
                        float _fmax_347 = fmaxf(sc_5_8, -1.7014118346046923e+38f);
                        sc_5_8 = _fmax_347;
                        float _min_315 = fminf(sc_5_8, 1.7014118346046923e+38f);
                        sc_5_8 = _min_315;
                        sc_5_8 = sc_5_8;
                        unsigned int u9_8 = __as_u32(sc_5_8);
                        unsigned int lowb9_8 = (u9_8 ^ (unsigned int)((int)u9_8 >> 31) & 511) & 511;
                        unsigned int km9_8 = 536870912 | lowb9_8 << 9 | (unsigned int)(tb + 8);
                        cbuf[pos9] = km9_8;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 512) != 0) {
                        float sc_5_9 = __uint_as_float(cbt[9]);
                        float _fmax_348 = fmaxf(sc_5_9, -1.7014118346046923e+38f);
                        sc_5_9 = _fmax_348;
                        float _min_316 = fminf(sc_5_9, 1.7014118346046923e+38f);
                        sc_5_9 = _min_316;
                        sc_5_9 = sc_5_9;
                        unsigned int u9_9 = __as_u32(sc_5_9);
                        unsigned int lowb9_9 = (u9_9 ^ (unsigned int)((int)u9_9 >> 31) & 511) & 511;
                        unsigned int km9_9 = 536870912 | lowb9_9 << 9 | (unsigned int)(tb + 9);
                        cbuf[pos9] = km9_9;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 1024) != 0) {
                        float sc_5_10 = __uint_as_float(cbt[10]);
                        float _fmax_349 = fmaxf(sc_5_10, -1.7014118346046923e+38f);
                        sc_5_10 = _fmax_349;
                        float _min_317 = fminf(sc_5_10, 1.7014118346046923e+38f);
                        sc_5_10 = _min_317;
                        sc_5_10 = sc_5_10;
                        unsigned int u9_10 = __as_u32(sc_5_10);
                        unsigned int lowb9_10 = (u9_10 ^ (unsigned int)((int)u9_10 >> 31) & 511) & 511;
                        unsigned int km9_10 = 536870912 | lowb9_10 << 9 | (unsigned int)(tb + 10);
                        cbuf[pos9] = km9_10;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 2048) != 0) {
                        float sc_5_11 = __uint_as_float(cbt[11]);
                        float _fmax_350 = fmaxf(sc_5_11, -1.7014118346046923e+38f);
                        sc_5_11 = _fmax_350;
                        float _min_318 = fminf(sc_5_11, 1.7014118346046923e+38f);
                        sc_5_11 = _min_318;
                        sc_5_11 = sc_5_11;
                        unsigned int u9_11 = __as_u32(sc_5_11);
                        unsigned int lowb9_11 = (u9_11 ^ (unsigned int)((int)u9_11 >> 31) & 511) & 511;
                        unsigned int km9_11 = 536870912 | lowb9_11 << 9 | (unsigned int)(tb + 11);
                        cbuf[pos9] = km9_11;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 4096) != 0) {
                        float sc_5_12 = __uint_as_float(cbt[12]);
                        float _fmax_351 = fmaxf(sc_5_12, -1.7014118346046923e+38f);
                        sc_5_12 = _fmax_351;
                        float _min_319 = fminf(sc_5_12, 1.7014118346046923e+38f);
                        sc_5_12 = _min_319;
                        sc_5_12 = sc_5_12;
                        unsigned int u9_12 = __as_u32(sc_5_12);
                        unsigned int lowb9_12 = (u9_12 ^ (unsigned int)((int)u9_12 >> 31) & 511) & 511;
                        unsigned int km9_12 = 536870912 | lowb9_12 << 9 | (unsigned int)(tb + 12);
                        cbuf[pos9] = km9_12;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 8192) != 0) {
                        float sc_5_13 = __uint_as_float(cbt[13]);
                        float _fmax_352 = fmaxf(sc_5_13, -1.7014118346046923e+38f);
                        sc_5_13 = _fmax_352;
                        float _min_320 = fminf(sc_5_13, 1.7014118346046923e+38f);
                        sc_5_13 = _min_320;
                        sc_5_13 = sc_5_13;
                        unsigned int u9_13 = __as_u32(sc_5_13);
                        unsigned int lowb9_13 = (u9_13 ^ (unsigned int)((int)u9_13 >> 31) & 511) & 511;
                        unsigned int km9_13 = 536870912 | lowb9_13 << 9 | (unsigned int)(tb + 13);
                        cbuf[pos9] = km9_13;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 16384) != 0) {
                        float sc_5_14 = __uint_as_float(cbt[14]);
                        float _fmax_353 = fmaxf(sc_5_14, -1.7014118346046923e+38f);
                        sc_5_14 = _fmax_353;
                        float _min_321 = fminf(sc_5_14, 1.7014118346046923e+38f);
                        sc_5_14 = _min_321;
                        sc_5_14 = sc_5_14;
                        unsigned int u9_14 = __as_u32(sc_5_14);
                        unsigned int lowb9_14 = (u9_14 ^ (unsigned int)((int)u9_14 >> 31) & 511) & 511;
                        unsigned int km9_14 = 536870912 | lowb9_14 << 9 | (unsigned int)(tb + 14);
                        cbuf[pos9] = km9_14;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 32768) != 0) {
                        float sc_5_15 = __uint_as_float(cbt[15]);
                        float _fmax_354 = fmaxf(sc_5_15, -1.7014118346046923e+38f);
                        sc_5_15 = _fmax_354;
                        float _min_322 = fminf(sc_5_15, 1.7014118346046923e+38f);
                        sc_5_15 = _min_322;
                        sc_5_15 = sc_5_15;
                        unsigned int u9_15 = __as_u32(sc_5_15);
                        unsigned int lowb9_15 = (u9_15 ^ (unsigned int)((int)u9_15 >> 31) & 511) & 511;
                        unsigned int km9_15 = 536870912 | lowb9_15 << 9 | (unsigned int)(tb + 15);
                        cbuf[pos9] = km9_15;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 65536) != 0) {
                        float sc_5_16 = __uint_as_float(cbt[16]);
                        float _fmax_355 = fmaxf(sc_5_16, -1.7014118346046923e+38f);
                        sc_5_16 = _fmax_355;
                        float _min_323 = fminf(sc_5_16, 1.7014118346046923e+38f);
                        sc_5_16 = _min_323;
                        sc_5_16 = sc_5_16;
                        unsigned int u9_16 = __as_u32(sc_5_16);
                        unsigned int lowb9_16 = (u9_16 ^ (unsigned int)((int)u9_16 >> 31) & 511) & 511;
                        unsigned int km9_16 = 536870912 | lowb9_16 << 9 | (unsigned int)(tb + 16);
                        cbuf[pos9] = km9_16;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 131072) != 0) {
                        float sc_5_17 = __uint_as_float(cbt[17]);
                        float _fmax_356 = fmaxf(sc_5_17, -1.7014118346046923e+38f);
                        sc_5_17 = _fmax_356;
                        float _min_324 = fminf(sc_5_17, 1.7014118346046923e+38f);
                        sc_5_17 = _min_324;
                        sc_5_17 = sc_5_17;
                        unsigned int u9_17 = __as_u32(sc_5_17);
                        unsigned int lowb9_17 = (u9_17 ^ (unsigned int)((int)u9_17 >> 31) & 511) & 511;
                        unsigned int km9_17 = 536870912 | lowb9_17 << 9 | (unsigned int)(tb + 17);
                        cbuf[pos9] = km9_17;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 262144) != 0) {
                        float sc_5_18 = __uint_as_float(cbt[18]);
                        float _fmax_357 = fmaxf(sc_5_18, -1.7014118346046923e+38f);
                        sc_5_18 = _fmax_357;
                        float _min_325 = fminf(sc_5_18, 1.7014118346046923e+38f);
                        sc_5_18 = _min_325;
                        sc_5_18 = sc_5_18;
                        unsigned int u9_18 = __as_u32(sc_5_18);
                        unsigned int lowb9_18 = (u9_18 ^ (unsigned int)((int)u9_18 >> 31) & 511) & 511;
                        unsigned int km9_18 = 536870912 | lowb9_18 << 9 | (unsigned int)(tb + 18);
                        cbuf[pos9] = km9_18;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 524288) != 0) {
                        float sc_5_19 = __uint_as_float(cbt[19]);
                        float _fmax_358 = fmaxf(sc_5_19, -1.7014118346046923e+38f);
                        sc_5_19 = _fmax_358;
                        float _min_326 = fminf(sc_5_19, 1.7014118346046923e+38f);
                        sc_5_19 = _min_326;
                        sc_5_19 = sc_5_19;
                        unsigned int u9_19 = __as_u32(sc_5_19);
                        unsigned int lowb9_19 = (u9_19 ^ (unsigned int)((int)u9_19 >> 31) & 511) & 511;
                        unsigned int km9_19 = 536870912 | lowb9_19 << 9 | (unsigned int)(tb + 19);
                        cbuf[pos9] = km9_19;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 1048576) != 0) {
                        float sc_5_20 = __uint_as_float(cbt[20]);
                        float _fmax_359 = fmaxf(sc_5_20, -1.7014118346046923e+38f);
                        sc_5_20 = _fmax_359;
                        float _min_327 = fminf(sc_5_20, 1.7014118346046923e+38f);
                        sc_5_20 = _min_327;
                        sc_5_20 = sc_5_20;
                        unsigned int u9_20 = __as_u32(sc_5_20);
                        unsigned int lowb9_20 = (u9_20 ^ (unsigned int)((int)u9_20 >> 31) & 511) & 511;
                        unsigned int km9_20 = 536870912 | lowb9_20 << 9 | (unsigned int)(tb + 20);
                        cbuf[pos9] = km9_20;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 2097152) != 0) {
                        float sc_5_21 = __uint_as_float(cbt[21]);
                        float _fmax_360 = fmaxf(sc_5_21, -1.7014118346046923e+38f);
                        sc_5_21 = _fmax_360;
                        float _min_328 = fminf(sc_5_21, 1.7014118346046923e+38f);
                        sc_5_21 = _min_328;
                        sc_5_21 = sc_5_21;
                        unsigned int u9_21 = __as_u32(sc_5_21);
                        unsigned int lowb9_21 = (u9_21 ^ (unsigned int)((int)u9_21 >> 31) & 511) & 511;
                        unsigned int km9_21 = 536870912 | lowb9_21 << 9 | (unsigned int)(tb + 21);
                        cbuf[pos9] = km9_21;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 4194304) != 0) {
                        float sc_5_22 = __uint_as_float(cbt[22]);
                        float _fmax_361 = fmaxf(sc_5_22, -1.7014118346046923e+38f);
                        sc_5_22 = _fmax_361;
                        float _min_329 = fminf(sc_5_22, 1.7014118346046923e+38f);
                        sc_5_22 = _min_329;
                        sc_5_22 = sc_5_22;
                        unsigned int u9_22 = __as_u32(sc_5_22);
                        unsigned int lowb9_22 = (u9_22 ^ (unsigned int)((int)u9_22 >> 31) & 511) & 511;
                        unsigned int km9_22 = 536870912 | lowb9_22 << 9 | (unsigned int)(tb + 22);
                        cbuf[pos9] = km9_22;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 8388608) != 0) {
                        float sc_5_23 = __uint_as_float(cbt[23]);
                        float _fmax_362 = fmaxf(sc_5_23, -1.7014118346046923e+38f);
                        sc_5_23 = _fmax_362;
                        float _min_330 = fminf(sc_5_23, 1.7014118346046923e+38f);
                        sc_5_23 = _min_330;
                        sc_5_23 = sc_5_23;
                        unsigned int u9_23 = __as_u32(sc_5_23);
                        unsigned int lowb9_23 = (u9_23 ^ (unsigned int)((int)u9_23 >> 31) & 511) & 511;
                        unsigned int km9_23 = 536870912 | lowb9_23 << 9 | (unsigned int)(tb + 23);
                        cbuf[pos9] = km9_23;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 16777216) != 0) {
                        float sc_5_24 = __uint_as_float(cbt[24]);
                        float _fmax_363 = fmaxf(sc_5_24, -1.7014118346046923e+38f);
                        sc_5_24 = _fmax_363;
                        float _min_331 = fminf(sc_5_24, 1.7014118346046923e+38f);
                        sc_5_24 = _min_331;
                        sc_5_24 = sc_5_24;
                        unsigned int u9_24 = __as_u32(sc_5_24);
                        unsigned int lowb9_24 = (u9_24 ^ (unsigned int)((int)u9_24 >> 31) & 511) & 511;
                        unsigned int km9_24 = 536870912 | lowb9_24 << 9 | (unsigned int)(tb + 24);
                        cbuf[pos9] = km9_24;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 33554432) != 0) {
                        float sc_5_25 = __uint_as_float(cbt[25]);
                        float _fmax_364 = fmaxf(sc_5_25, -1.7014118346046923e+38f);
                        sc_5_25 = _fmax_364;
                        float _min_332 = fminf(sc_5_25, 1.7014118346046923e+38f);
                        sc_5_25 = _min_332;
                        sc_5_25 = sc_5_25;
                        unsigned int u9_25 = __as_u32(sc_5_25);
                        unsigned int lowb9_25 = (u9_25 ^ (unsigned int)((int)u9_25 >> 31) & 511) & 511;
                        unsigned int km9_25 = 536870912 | lowb9_25 << 9 | (unsigned int)(tb + 25);
                        cbuf[pos9] = km9_25;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 67108864) != 0) {
                        float sc_5_26 = __uint_as_float(cbt[26]);
                        float _fmax_365 = fmaxf(sc_5_26, -1.7014118346046923e+38f);
                        sc_5_26 = _fmax_365;
                        float _min_333 = fminf(sc_5_26, 1.7014118346046923e+38f);
                        sc_5_26 = _min_333;
                        sc_5_26 = sc_5_26;
                        unsigned int u9_26 = __as_u32(sc_5_26);
                        unsigned int lowb9_26 = (u9_26 ^ (unsigned int)((int)u9_26 >> 31) & 511) & 511;
                        unsigned int km9_26 = 536870912 | lowb9_26 << 9 | (unsigned int)(tb + 26);
                        cbuf[pos9] = km9_26;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 134217728) != 0) {
                        float sc_5_27 = __uint_as_float(cbt[27]);
                        float _fmax_366 = fmaxf(sc_5_27, -1.7014118346046923e+38f);
                        sc_5_27 = _fmax_366;
                        float _min_334 = fminf(sc_5_27, 1.7014118346046923e+38f);
                        sc_5_27 = _min_334;
                        sc_5_27 = sc_5_27;
                        unsigned int u9_27 = __as_u32(sc_5_27);
                        unsigned int lowb9_27 = (u9_27 ^ (unsigned int)((int)u9_27 >> 31) & 511) & 511;
                        unsigned int km9_27 = 536870912 | lowb9_27 << 9 | (unsigned int)(tb + 27);
                        cbuf[pos9] = km9_27;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 268435456) != 0) {
                        float sc_5_28 = __uint_as_float(cbt[28]);
                        float _fmax_367 = fmaxf(sc_5_28, -1.7014118346046923e+38f);
                        sc_5_28 = _fmax_367;
                        float _min_335 = fminf(sc_5_28, 1.7014118346046923e+38f);
                        sc_5_28 = _min_335;
                        sc_5_28 = sc_5_28;
                        unsigned int u9_28 = __as_u32(sc_5_28);
                        unsigned int lowb9_28 = (u9_28 ^ (unsigned int)((int)u9_28 >> 31) & 511) & 511;
                        unsigned int km9_28 = 536870912 | lowb9_28 << 9 | (unsigned int)(tb + 28);
                        cbuf[pos9] = km9_28;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 536870912) != 0) {
                        float sc_5_29 = __uint_as_float(cbt[29]);
                        float _fmax_368 = fmaxf(sc_5_29, -1.7014118346046923e+38f);
                        sc_5_29 = _fmax_368;
                        float _min_336 = fminf(sc_5_29, 1.7014118346046923e+38f);
                        sc_5_29 = _min_336;
                        sc_5_29 = sc_5_29;
                        unsigned int u9_29 = __as_u32(sc_5_29);
                        unsigned int lowb9_29 = (u9_29 ^ (unsigned int)((int)u9_29 >> 31) & 511) & 511;
                        unsigned int km9_29 = 536870912 | lowb9_29 << 9 | (unsigned int)(tb + 29);
                        cbuf[pos9] = km9_29;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 1073741824) != 0) {
                        float sc_5_30 = __uint_as_float(cbt[30]);
                        float _fmax_369 = fmaxf(sc_5_30, -1.7014118346046923e+38f);
                        sc_5_30 = _fmax_369;
                        float _min_337 = fminf(sc_5_30, 1.7014118346046923e+38f);
                        sc_5_30 = _min_337;
                        sc_5_30 = sc_5_30;
                        unsigned int u9_30 = __as_u32(sc_5_30);
                        unsigned int lowb9_30 = (u9_30 ^ (unsigned int)((int)u9_30 >> 31) & 511) & 511;
                        unsigned int km9_30 = 536870912 | lowb9_30 << 9 | (unsigned int)(tb + 30);
                        cbuf[pos9] = km9_30;
                        pos9 = pos9 + 1;
                    }
                    if ((mm_94 & 2147483648u) != 0) {
                        float sc_5_31 = __uint_as_float(cbt[31]);
                        float _fmax_370 = fmaxf(sc_5_31, -1.7014118346046923e+38f);
                        sc_5_31 = _fmax_370;
                        float _min_338 = fminf(sc_5_31, 1.7014118346046923e+38f);
                        sc_5_31 = _min_338;
                        sc_5_31 = sc_5_31;
                        unsigned int u9_31 = __as_u32(sc_5_31);
                        unsigned int lowb9_31 = (u9_31 ^ (unsigned int)((int)u9_31 >> 31) & 511) & 511;
                        unsigned int km9_31 = 536870912 | lowb9_31 << 9 | (unsigned int)(tb + 31);
                        cbuf[pos9] = km9_31;
                        pos9 = pos9 + 1;
                    }
                }
            }
        }
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        unsigned int need_fullx = flagw[1];
        unsigned int cntx = ccnt[c];
        int ovfx = 0;
        if (flagged != 0 && cntx > 16) {
            ovfx = 1;
        }
        if (need_fullx != 0) {
            int lim2x = 0;
            if (ovfx != 0) {
                lim2x = lim;
            }
            float a2x[16];
            a2x[0] = 0.0f;
            a2x[1] = 0.0f;
            a2x[2] = 0.0f;
            a2x[3] = 0.0f;
            a2x[4] = 0.0f;
            a2x[5] = 0.0f;
            a2x[6] = 0.0f;
            a2x[7] = 0.0f;
            a2x[8] = 0.0f;
            a2x[9] = 0.0f;
            a2x[10] = 0.0f;
            a2x[11] = 0.0f;
            a2x[12] = 0.0f;
            a2x[13] = 0.0f;
            a2x[14] = 0.0f;
            a2x[15] = 0.0f;
            if (ovfx != 0) {
                float kbt2[32];
                float r2o = neg_inf;
                float qf = __uint_as_float(qc);
                float sc_1_1 = __uint_as_float(cbt[0]);
                float _fmax_371 = fmaxf(sc_1_1, -1.7014118346046923e+38f);
                sc_1_1 = _fmax_371;
                float _min_339 = fminf(sc_1_1, 1.7014118346046923e+38f);
                sc_1_1 = _min_339;
                sc_1_1 = sc_1_1;
                float sc2 = sc_1_1;
                unsigned int u = __as_u32(sc2);
                unsigned int cls = u & 4294966784u;
                int f_2_2 = 0;
                if ((tb < fb || tb >= lim - fe) && lim > tb) {
                    f_2_2 = 1;
                }
                if (f_2_2 != 0) {
                    cls = 2139094528;
                }
                unsigned int key_3 = 0;
                if (qf < __uint_as_float(cls) && cls < 4278190080u) {
                    key_3 = 1073741824 | (unsigned int)tb;
                }
                if (cls == qc) {
                    unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 511) & 511;
                    key_3 = 536870912 | lowb << 9 | (unsigned int)tb;
                }
                kbt2[0] = __uint_as_float(key_3);
                float sc_4_1 = __uint_as_float(cbt[1]);
                float _fmax_372 = fmaxf(sc_4_1, -1.7014118346046923e+38f);
                sc_4_1 = _fmax_372;
                float _min_340 = fminf(sc_4_1, 1.7014118346046923e+38f);
                sc_4_1 = _min_340;
                sc_4_1 = sc_4_1;
                float sc2_5 = sc_4_1;
                unsigned int u_6 = __as_u32(sc2_5);
                unsigned int cls_7 = u_6 & 4294966784u;
                int f_8_2 = 0;
                if ((tb + 1 < fb || tb + 1 >= lim - fe) && lim > tb + 1) {
                    f_8_2 = 1;
                }
                if (f_8_2 != 0) {
                    cls_7 = 2139094528;
                }
                unsigned int key_9 = 0;
                if (qf < __uint_as_float(cls_7) && cls_7 < 4278190080u) {
                    key_9 = 1073741824 | (unsigned int)(tb + 1);
                }
                if (cls_7 == qc) {
                    unsigned int lowb_1 = (u_6 ^ (unsigned int)((int)u_6 >> 31) & 511) & 511;
                    key_9 = 536870912 | lowb_1 << 9 | (unsigned int)(tb + 1);
                }
                kbt2[1] = __uint_as_float(key_9);
                float sc_10_1 = __uint_as_float(cbt[2]);
                float _fmax_373 = fmaxf(sc_10_1, -1.7014118346046923e+38f);
                sc_10_1 = _fmax_373;
                float _min_341 = fminf(sc_10_1, 1.7014118346046923e+38f);
                sc_10_1 = _min_341;
                sc_10_1 = sc_10_1;
                float sc2_11 = sc_10_1;
                unsigned int u_12 = __as_u32(sc2_11);
                unsigned int cls_13 = u_12 & 4294966784u;
                int f_14_2 = 0;
                if ((tb + 2 < fb || tb + 2 >= lim - fe) && lim > tb + 2) {
                    f_14_2 = 1;
                }
                if (f_14_2 != 0) {
                    cls_13 = 2139094528;
                }
                unsigned int key_15 = 0;
                if (qf < __uint_as_float(cls_13) && cls_13 < 4278190080u) {
                    key_15 = 1073741824 | (unsigned int)(tb + 2);
                }
                if (cls_13 == qc) {
                    unsigned int lowb_2 = (u_12 ^ (unsigned int)((int)u_12 >> 31) & 511) & 511;
                    key_15 = 536870912 | lowb_2 << 9 | (unsigned int)(tb + 2);
                }
                kbt2[2] = __uint_as_float(key_15);
                float sc_16_1 = __uint_as_float(cbt[3]);
                float _fmax_374 = fmaxf(sc_16_1, -1.7014118346046923e+38f);
                sc_16_1 = _fmax_374;
                float _min_342 = fminf(sc_16_1, 1.7014118346046923e+38f);
                sc_16_1 = _min_342;
                sc_16_1 = sc_16_1;
                float sc2_17 = sc_16_1;
                unsigned int u_18 = __as_u32(sc2_17);
                unsigned int cls_19 = u_18 & 4294966784u;
                int f_20_2 = 0;
                if ((tb + 3 < fb || tb + 3 >= lim - fe) && lim > tb + 3) {
                    f_20_2 = 1;
                }
                if (f_20_2 != 0) {
                    cls_19 = 2139094528;
                }
                unsigned int key_21 = 0;
                if (qf < __uint_as_float(cls_19) && cls_19 < 4278190080u) {
                    key_21 = 1073741824 | (unsigned int)(tb + 3);
                }
                if (cls_19 == qc) {
                    unsigned int lowb_3 = (u_18 ^ (unsigned int)((int)u_18 >> 31) & 511) & 511;
                    key_21 = 536870912 | lowb_3 << 9 | (unsigned int)(tb + 3);
                }
                kbt2[3] = __uint_as_float(key_21);
                float sc_22_1 = __uint_as_float(cbt[4]);
                float _fmax_375 = fmaxf(sc_22_1, -1.7014118346046923e+38f);
                sc_22_1 = _fmax_375;
                float _min_343 = fminf(sc_22_1, 1.7014118346046923e+38f);
                sc_22_1 = _min_343;
                sc_22_1 = sc_22_1;
                float sc2_23 = sc_22_1;
                unsigned int u_24 = __as_u32(sc2_23);
                unsigned int cls_25 = u_24 & 4294966784u;
                int f_26_2 = 0;
                if ((tb + 4 < fb || tb + 4 >= lim - fe) && lim > tb + 4) {
                    f_26_2 = 1;
                }
                if (f_26_2 != 0) {
                    cls_25 = 2139094528;
                }
                unsigned int key_27 = 0;
                if (qf < __uint_as_float(cls_25) && cls_25 < 4278190080u) {
                    key_27 = 1073741824 | (unsigned int)(tb + 4);
                }
                if (cls_25 == qc) {
                    unsigned int lowb_4 = (u_24 ^ (unsigned int)((int)u_24 >> 31) & 511) & 511;
                    key_27 = 536870912 | lowb_4 << 9 | (unsigned int)(tb + 4);
                }
                kbt2[4] = __uint_as_float(key_27);
                float sc_28_1 = __uint_as_float(cbt[5]);
                float _fmax_376 = fmaxf(sc_28_1, -1.7014118346046923e+38f);
                sc_28_1 = _fmax_376;
                float _min_344 = fminf(sc_28_1, 1.7014118346046923e+38f);
                sc_28_1 = _min_344;
                sc_28_1 = sc_28_1;
                float sc2_29 = sc_28_1;
                unsigned int u_30 = __as_u32(sc2_29);
                unsigned int cls_31 = u_30 & 4294966784u;
                int f_32 = 0;
                if ((tb + 5 < fb || tb + 5 >= lim - fe) && lim > tb + 5) {
                    f_32 = 1;
                }
                if (f_32 != 0) {
                    cls_31 = 2139094528;
                }
                unsigned int key_33 = 0;
                if (qf < __uint_as_float(cls_31) && cls_31 < 4278190080u) {
                    key_33 = 1073741824 | (unsigned int)(tb + 5);
                }
                if (cls_31 == qc) {
                    unsigned int lowb_5 = (u_30 ^ (unsigned int)((int)u_30 >> 31) & 511) & 511;
                    key_33 = 536870912 | lowb_5 << 9 | (unsigned int)(tb + 5);
                }
                kbt2[5] = __uint_as_float(key_33);
                float sc_34_1 = __uint_as_float(cbt[6]);
                float _fmax_377 = fmaxf(sc_34_1, -1.7014118346046923e+38f);
                sc_34_1 = _fmax_377;
                float _min_345 = fminf(sc_34_1, 1.7014118346046923e+38f);
                sc_34_1 = _min_345;
                sc_34_1 = sc_34_1;
                float sc2_35 = sc_34_1;
                unsigned int u_36 = __as_u32(sc2_35);
                unsigned int cls_37 = u_36 & 4294966784u;
                int f_38 = 0;
                if ((tb + 6 < fb || tb + 6 >= lim - fe) && lim > tb + 6) {
                    f_38 = 1;
                }
                if (f_38 != 0) {
                    cls_37 = 2139094528;
                }
                unsigned int key_39 = 0;
                if (qf < __uint_as_float(cls_37) && cls_37 < 4278190080u) {
                    key_39 = 1073741824 | (unsigned int)(tb + 6);
                }
                if (cls_37 == qc) {
                    unsigned int lowb_6 = (u_36 ^ (unsigned int)((int)u_36 >> 31) & 511) & 511;
                    key_39 = 536870912 | lowb_6 << 9 | (unsigned int)(tb + 6);
                }
                kbt2[6] = __uint_as_float(key_39);
                float sc_40_1 = __uint_as_float(cbt[7]);
                float _fmax_378 = fmaxf(sc_40_1, -1.7014118346046923e+38f);
                sc_40_1 = _fmax_378;
                float _min_346 = fminf(sc_40_1, 1.7014118346046923e+38f);
                sc_40_1 = _min_346;
                sc_40_1 = sc_40_1;
                float sc2_41 = sc_40_1;
                unsigned int u_42 = __as_u32(sc2_41);
                unsigned int cls_43 = u_42 & 4294966784u;
                int f_44 = 0;
                if ((tb + 7 < fb || tb + 7 >= lim - fe) && lim > tb + 7) {
                    f_44 = 1;
                }
                if (f_44 != 0) {
                    cls_43 = 2139094528;
                }
                unsigned int key_45 = 0;
                if (qf < __uint_as_float(cls_43) && cls_43 < 4278190080u) {
                    key_45 = 1073741824 | (unsigned int)(tb + 7);
                }
                if (cls_43 == qc) {
                    unsigned int lowb_7 = (u_42 ^ (unsigned int)((int)u_42 >> 31) & 511) & 511;
                    key_45 = 536870912 | lowb_7 << 9 | (unsigned int)(tb + 7);
                }
                kbt2[7] = __uint_as_float(key_45);
                float sc_46_1 = __uint_as_float(cbt[8]);
                float _fmax_379 = fmaxf(sc_46_1, -1.7014118346046923e+38f);
                sc_46_1 = _fmax_379;
                float _min_347 = fminf(sc_46_1, 1.7014118346046923e+38f);
                sc_46_1 = _min_347;
                sc_46_1 = sc_46_1;
                float sc2_47 = sc_46_1;
                unsigned int u_48 = __as_u32(sc2_47);
                unsigned int cls_49 = u_48 & 4294966784u;
                int f_50 = 0;
                if ((tb + 8 < fb || tb + 8 >= lim - fe) && lim > tb + 8) {
                    f_50 = 1;
                }
                if (f_50 != 0) {
                    cls_49 = 2139094528;
                }
                unsigned int key_51 = 0;
                if (qf < __uint_as_float(cls_49) && cls_49 < 4278190080u) {
                    key_51 = 1073741824 | (unsigned int)(tb + 8);
                }
                if (cls_49 == qc) {
                    unsigned int lowb_8 = (u_48 ^ (unsigned int)((int)u_48 >> 31) & 511) & 511;
                    key_51 = 536870912 | lowb_8 << 9 | (unsigned int)(tb + 8);
                }
                kbt2[8] = __uint_as_float(key_51);
                float sc_52_1 = __uint_as_float(cbt[9]);
                float _fmax_380 = fmaxf(sc_52_1, -1.7014118346046923e+38f);
                sc_52_1 = _fmax_380;
                float _min_348 = fminf(sc_52_1, 1.7014118346046923e+38f);
                sc_52_1 = _min_348;
                sc_52_1 = sc_52_1;
                float sc2_53 = sc_52_1;
                unsigned int u_54 = __as_u32(sc2_53);
                unsigned int cls_55 = u_54 & 4294966784u;
                int f_56 = 0;
                if ((tb + 9 < fb || tb + 9 >= lim - fe) && lim > tb + 9) {
                    f_56 = 1;
                }
                if (f_56 != 0) {
                    cls_55 = 2139094528;
                }
                unsigned int key_57 = 0;
                if (qf < __uint_as_float(cls_55) && cls_55 < 4278190080u) {
                    key_57 = 1073741824 | (unsigned int)(tb + 9);
                }
                if (cls_55 == qc) {
                    unsigned int lowb_9 = (u_54 ^ (unsigned int)((int)u_54 >> 31) & 511) & 511;
                    key_57 = 536870912 | lowb_9 << 9 | (unsigned int)(tb + 9);
                }
                kbt2[9] = __uint_as_float(key_57);
                float sc_58_1 = __uint_as_float(cbt[10]);
                float _fmax_381 = fmaxf(sc_58_1, -1.7014118346046923e+38f);
                sc_58_1 = _fmax_381;
                float _min_349 = fminf(sc_58_1, 1.7014118346046923e+38f);
                sc_58_1 = _min_349;
                sc_58_1 = sc_58_1;
                float sc2_59 = sc_58_1;
                unsigned int u_60 = __as_u32(sc2_59);
                unsigned int cls_61 = u_60 & 4294966784u;
                int f_62 = 0;
                if ((tb + 10 < fb || tb + 10 >= lim - fe) && lim > tb + 10) {
                    f_62 = 1;
                }
                if (f_62 != 0) {
                    cls_61 = 2139094528;
                }
                unsigned int key_63 = 0;
                if (qf < __uint_as_float(cls_61) && cls_61 < 4278190080u) {
                    key_63 = 1073741824 | (unsigned int)(tb + 10);
                }
                if (cls_61 == qc) {
                    unsigned int lowb_10 = (u_60 ^ (unsigned int)((int)u_60 >> 31) & 511) & 511;
                    key_63 = 536870912 | lowb_10 << 9 | (unsigned int)(tb + 10);
                }
                kbt2[10] = __uint_as_float(key_63);
                float sc_64_1 = __uint_as_float(cbt[11]);
                float _fmax_382 = fmaxf(sc_64_1, -1.7014118346046923e+38f);
                sc_64_1 = _fmax_382;
                float _min_350 = fminf(sc_64_1, 1.7014118346046923e+38f);
                sc_64_1 = _min_350;
                sc_64_1 = sc_64_1;
                float sc2_65 = sc_64_1;
                unsigned int u_66 = __as_u32(sc2_65);
                unsigned int cls_67 = u_66 & 4294966784u;
                int f_68 = 0;
                if ((tb + 11 < fb || tb + 11 >= lim - fe) && lim > tb + 11) {
                    f_68 = 1;
                }
                if (f_68 != 0) {
                    cls_67 = 2139094528;
                }
                unsigned int key_69 = 0;
                if (qf < __uint_as_float(cls_67) && cls_67 < 4278190080u) {
                    key_69 = 1073741824 | (unsigned int)(tb + 11);
                }
                if (cls_67 == qc) {
                    unsigned int lowb_11 = (u_66 ^ (unsigned int)((int)u_66 >> 31) & 511) & 511;
                    key_69 = 536870912 | lowb_11 << 9 | (unsigned int)(tb + 11);
                }
                kbt2[11] = __uint_as_float(key_69);
                float sc_70_1 = __uint_as_float(cbt[12]);
                float _fmax_383 = fmaxf(sc_70_1, -1.7014118346046923e+38f);
                sc_70_1 = _fmax_383;
                float _min_351 = fminf(sc_70_1, 1.7014118346046923e+38f);
                sc_70_1 = _min_351;
                sc_70_1 = sc_70_1;
                float sc2_71 = sc_70_1;
                unsigned int u_72 = __as_u32(sc2_71);
                unsigned int cls_73 = u_72 & 4294966784u;
                int f_74 = 0;
                if ((tb + 12 < fb || tb + 12 >= lim - fe) && lim > tb + 12) {
                    f_74 = 1;
                }
                if (f_74 != 0) {
                    cls_73 = 2139094528;
                }
                unsigned int key_75 = 0;
                if (qf < __uint_as_float(cls_73) && cls_73 < 4278190080u) {
                    key_75 = 1073741824 | (unsigned int)(tb + 12);
                }
                if (cls_73 == qc) {
                    unsigned int lowb_12 = (u_72 ^ (unsigned int)((int)u_72 >> 31) & 511) & 511;
                    key_75 = 536870912 | lowb_12 << 9 | (unsigned int)(tb + 12);
                }
                kbt2[12] = __uint_as_float(key_75);
                float sc_76_1 = __uint_as_float(cbt[13]);
                float _fmax_384 = fmaxf(sc_76_1, -1.7014118346046923e+38f);
                sc_76_1 = _fmax_384;
                float _min_352 = fminf(sc_76_1, 1.7014118346046923e+38f);
                sc_76_1 = _min_352;
                sc_76_1 = sc_76_1;
                float sc2_77 = sc_76_1;
                unsigned int u_78 = __as_u32(sc2_77);
                unsigned int cls_79 = u_78 & 4294966784u;
                int f_80 = 0;
                if ((tb + 13 < fb || tb + 13 >= lim - fe) && lim > tb + 13) {
                    f_80 = 1;
                }
                if (f_80 != 0) {
                    cls_79 = 2139094528;
                }
                unsigned int key_81 = 0;
                if (qf < __uint_as_float(cls_79) && cls_79 < 4278190080u) {
                    key_81 = 1073741824 | (unsigned int)(tb + 13);
                }
                if (cls_79 == qc) {
                    unsigned int lowb_13 = (u_78 ^ (unsigned int)((int)u_78 >> 31) & 511) & 511;
                    key_81 = 536870912 | lowb_13 << 9 | (unsigned int)(tb + 13);
                }
                kbt2[13] = __uint_as_float(key_81);
                float sc_82_1 = __uint_as_float(cbt[14]);
                float _fmax_385 = fmaxf(sc_82_1, -1.7014118346046923e+38f);
                sc_82_1 = _fmax_385;
                float _min_353 = fminf(sc_82_1, 1.7014118346046923e+38f);
                sc_82_1 = _min_353;
                sc_82_1 = sc_82_1;
                float sc2_83 = sc_82_1;
                unsigned int u_84 = __as_u32(sc2_83);
                unsigned int cls_85 = u_84 & 4294966784u;
                int f_86 = 0;
                if ((tb + 14 < fb || tb + 14 >= lim - fe) && lim > tb + 14) {
                    f_86 = 1;
                }
                if (f_86 != 0) {
                    cls_85 = 2139094528;
                }
                unsigned int key_87 = 0;
                if (qf < __uint_as_float(cls_85) && cls_85 < 4278190080u) {
                    key_87 = 1073741824 | (unsigned int)(tb + 14);
                }
                if (cls_85 == qc) {
                    unsigned int lowb_14 = (u_84 ^ (unsigned int)((int)u_84 >> 31) & 511) & 511;
                    key_87 = 536870912 | lowb_14 << 9 | (unsigned int)(tb + 14);
                }
                kbt2[14] = __uint_as_float(key_87);
                float sc_88_1 = __uint_as_float(cbt[15]);
                float _fmax_386 = fmaxf(sc_88_1, -1.7014118346046923e+38f);
                sc_88_1 = _fmax_386;
                float _min_354 = fminf(sc_88_1, 1.7014118346046923e+38f);
                sc_88_1 = _min_354;
                sc_88_1 = sc_88_1;
                float sc2_89 = sc_88_1;
                unsigned int u_90 = __as_u32(sc2_89);
                unsigned int cls_91 = u_90 & 4294966784u;
                int f_92 = 0;
                if ((tb + 15 < fb || tb + 15 >= lim - fe) && lim > tb + 15) {
                    f_92 = 1;
                }
                if (f_92 != 0) {
                    cls_91 = 2139094528;
                }
                unsigned int key_93 = 0;
                if (qf < __uint_as_float(cls_91) && cls_91 < 4278190080u) {
                    key_93 = 1073741824 | (unsigned int)(tb + 15);
                }
                if (cls_91 == qc) {
                    unsigned int lowb_15 = (u_90 ^ (unsigned int)((int)u_90 >> 31) & 511) & 511;
                    key_93 = 536870912 | lowb_15 << 9 | (unsigned int)(tb + 15);
                }
                kbt2[15] = __uint_as_float(key_93);
                float sc_94 = __uint_as_float(cbt[16]);
                float _fmax_387 = fmaxf(sc_94, -1.7014118346046923e+38f);
                sc_94 = _fmax_387;
                float _min_355 = fminf(sc_94, 1.7014118346046923e+38f);
                sc_94 = _min_355;
                sc_94 = sc_94;
                float sc2_95 = sc_94;
                unsigned int u_96 = __as_u32(sc2_95);
                unsigned int cls_97 = u_96 & 4294966784u;
                int f_98 = 0;
                if ((tb + 16 < fb || tb + 16 >= lim - fe) && lim > tb + 16) {
                    f_98 = 1;
                }
                if (f_98 != 0) {
                    cls_97 = 2139094528;
                }
                unsigned int key_99 = 0;
                if (qf < __uint_as_float(cls_97) && cls_97 < 4278190080u) {
                    key_99 = 1073741824 | (unsigned int)(tb + 16);
                }
                if (cls_97 == qc) {
                    unsigned int lowb_16 = (u_96 ^ (unsigned int)((int)u_96 >> 31) & 511) & 511;
                    key_99 = 536870912 | lowb_16 << 9 | (unsigned int)(tb + 16);
                }
                kbt2[16] = __uint_as_float(key_99);
                float sc_100 = __uint_as_float(cbt[17]);
                float _fmax_388 = fmaxf(sc_100, -1.7014118346046923e+38f);
                sc_100 = _fmax_388;
                float _min_356 = fminf(sc_100, 1.7014118346046923e+38f);
                sc_100 = _min_356;
                sc_100 = sc_100;
                float sc2_101 = sc_100;
                unsigned int u_102 = __as_u32(sc2_101);
                unsigned int cls_103 = u_102 & 4294966784u;
                int f_104 = 0;
                if ((tb + 17 < fb || tb + 17 >= lim - fe) && lim > tb + 17) {
                    f_104 = 1;
                }
                if (f_104 != 0) {
                    cls_103 = 2139094528;
                }
                unsigned int key_105 = 0;
                if (qf < __uint_as_float(cls_103) && cls_103 < 4278190080u) {
                    key_105 = 1073741824 | (unsigned int)(tb + 17);
                }
                if (cls_103 == qc) {
                    unsigned int lowb_17 = (u_102 ^ (unsigned int)((int)u_102 >> 31) & 511) & 511;
                    key_105 = 536870912 | lowb_17 << 9 | (unsigned int)(tb + 17);
                }
                kbt2[17] = __uint_as_float(key_105);
                float sc_106 = __uint_as_float(cbt[18]);
                float _fmax_389 = fmaxf(sc_106, -1.7014118346046923e+38f);
                sc_106 = _fmax_389;
                float _min_357 = fminf(sc_106, 1.7014118346046923e+38f);
                sc_106 = _min_357;
                sc_106 = sc_106;
                float sc2_107 = sc_106;
                unsigned int u_108 = __as_u32(sc2_107);
                unsigned int cls_109 = u_108 & 4294966784u;
                int f_110 = 0;
                if ((tb + 18 < fb || tb + 18 >= lim - fe) && lim > tb + 18) {
                    f_110 = 1;
                }
                if (f_110 != 0) {
                    cls_109 = 2139094528;
                }
                unsigned int key_111 = 0;
                if (qf < __uint_as_float(cls_109) && cls_109 < 4278190080u) {
                    key_111 = 1073741824 | (unsigned int)(tb + 18);
                }
                if (cls_109 == qc) {
                    unsigned int lowb_18 = (u_108 ^ (unsigned int)((int)u_108 >> 31) & 511) & 511;
                    key_111 = 536870912 | lowb_18 << 9 | (unsigned int)(tb + 18);
                }
                kbt2[18] = __uint_as_float(key_111);
                float sc_112 = __uint_as_float(cbt[19]);
                float _fmax_390 = fmaxf(sc_112, -1.7014118346046923e+38f);
                sc_112 = _fmax_390;
                float _min_358 = fminf(sc_112, 1.7014118346046923e+38f);
                sc_112 = _min_358;
                sc_112 = sc_112;
                float sc2_113 = sc_112;
                unsigned int u_114 = __as_u32(sc2_113);
                unsigned int cls_115 = u_114 & 4294966784u;
                int f_116 = 0;
                if ((tb + 19 < fb || tb + 19 >= lim - fe) && lim > tb + 19) {
                    f_116 = 1;
                }
                if (f_116 != 0) {
                    cls_115 = 2139094528;
                }
                unsigned int key_117 = 0;
                if (qf < __uint_as_float(cls_115) && cls_115 < 4278190080u) {
                    key_117 = 1073741824 | (unsigned int)(tb + 19);
                }
                if (cls_115 == qc) {
                    unsigned int lowb_19 = (u_114 ^ (unsigned int)((int)u_114 >> 31) & 511) & 511;
                    key_117 = 536870912 | lowb_19 << 9 | (unsigned int)(tb + 19);
                }
                kbt2[19] = __uint_as_float(key_117);
                float sc_118 = __uint_as_float(cbt[20]);
                float _fmax_391 = fmaxf(sc_118, -1.7014118346046923e+38f);
                sc_118 = _fmax_391;
                float _min_359 = fminf(sc_118, 1.7014118346046923e+38f);
                sc_118 = _min_359;
                sc_118 = sc_118;
                float sc2_119 = sc_118;
                unsigned int u_120 = __as_u32(sc2_119);
                unsigned int cls_121 = u_120 & 4294966784u;
                int f_122 = 0;
                if ((tb + 20 < fb || tb + 20 >= lim - fe) && lim > tb + 20) {
                    f_122 = 1;
                }
                if (f_122 != 0) {
                    cls_121 = 2139094528;
                }
                unsigned int key_123 = 0;
                if (qf < __uint_as_float(cls_121) && cls_121 < 4278190080u) {
                    key_123 = 1073741824 | (unsigned int)(tb + 20);
                }
                if (cls_121 == qc) {
                    unsigned int lowb_20 = (u_120 ^ (unsigned int)((int)u_120 >> 31) & 511) & 511;
                    key_123 = 536870912 | lowb_20 << 9 | (unsigned int)(tb + 20);
                }
                kbt2[20] = __uint_as_float(key_123);
                float sc_124 = __uint_as_float(cbt[21]);
                float _fmax_392 = fmaxf(sc_124, -1.7014118346046923e+38f);
                sc_124 = _fmax_392;
                float _min_360 = fminf(sc_124, 1.7014118346046923e+38f);
                sc_124 = _min_360;
                sc_124 = sc_124;
                float sc2_125 = sc_124;
                unsigned int u_126 = __as_u32(sc2_125);
                unsigned int cls_127 = u_126 & 4294966784u;
                int f_128 = 0;
                if ((tb + 21 < fb || tb + 21 >= lim - fe) && lim > tb + 21) {
                    f_128 = 1;
                }
                if (f_128 != 0) {
                    cls_127 = 2139094528;
                }
                unsigned int key_129 = 0;
                if (qf < __uint_as_float(cls_127) && cls_127 < 4278190080u) {
                    key_129 = 1073741824 | (unsigned int)(tb + 21);
                }
                if (cls_127 == qc) {
                    unsigned int lowb_21 = (u_126 ^ (unsigned int)((int)u_126 >> 31) & 511) & 511;
                    key_129 = 536870912 | lowb_21 << 9 | (unsigned int)(tb + 21);
                }
                kbt2[21] = __uint_as_float(key_129);
                float sc_130 = __uint_as_float(cbt[22]);
                float _fmax_393 = fmaxf(sc_130, -1.7014118346046923e+38f);
                sc_130 = _fmax_393;
                float _min_361 = fminf(sc_130, 1.7014118346046923e+38f);
                sc_130 = _min_361;
                sc_130 = sc_130;
                float sc2_131 = sc_130;
                unsigned int u_132 = __as_u32(sc2_131);
                unsigned int cls_133 = u_132 & 4294966784u;
                int f_134 = 0;
                if ((tb + 22 < fb || tb + 22 >= lim - fe) && lim > tb + 22) {
                    f_134 = 1;
                }
                if (f_134 != 0) {
                    cls_133 = 2139094528;
                }
                unsigned int key_135 = 0;
                if (qf < __uint_as_float(cls_133) && cls_133 < 4278190080u) {
                    key_135 = 1073741824 | (unsigned int)(tb + 22);
                }
                if (cls_133 == qc) {
                    unsigned int lowb_22 = (u_132 ^ (unsigned int)((int)u_132 >> 31) & 511) & 511;
                    key_135 = 536870912 | lowb_22 << 9 | (unsigned int)(tb + 22);
                }
                kbt2[22] = __uint_as_float(key_135);
                float sc_136 = __uint_as_float(cbt[23]);
                float _fmax_394 = fmaxf(sc_136, -1.7014118346046923e+38f);
                sc_136 = _fmax_394;
                float _min_362 = fminf(sc_136, 1.7014118346046923e+38f);
                sc_136 = _min_362;
                sc_136 = sc_136;
                float sc2_137 = sc_136;
                unsigned int u_138 = __as_u32(sc2_137);
                unsigned int cls_139 = u_138 & 4294966784u;
                int f_140 = 0;
                if ((tb + 23 < fb || tb + 23 >= lim - fe) && lim > tb + 23) {
                    f_140 = 1;
                }
                if (f_140 != 0) {
                    cls_139 = 2139094528;
                }
                unsigned int key_141 = 0;
                if (qf < __uint_as_float(cls_139) && cls_139 < 4278190080u) {
                    key_141 = 1073741824 | (unsigned int)(tb + 23);
                }
                if (cls_139 == qc) {
                    unsigned int lowb_23 = (u_138 ^ (unsigned int)((int)u_138 >> 31) & 511) & 511;
                    key_141 = 536870912 | lowb_23 << 9 | (unsigned int)(tb + 23);
                }
                kbt2[23] = __uint_as_float(key_141);
                float sc_142 = __uint_as_float(cbt[24]);
                float _fmax_395 = fmaxf(sc_142, -1.7014118346046923e+38f);
                sc_142 = _fmax_395;
                float _min_363 = fminf(sc_142, 1.7014118346046923e+38f);
                sc_142 = _min_363;
                sc_142 = sc_142;
                float sc2_143 = sc_142;
                unsigned int u_144 = __as_u32(sc2_143);
                unsigned int cls_145 = u_144 & 4294966784u;
                int f_146 = 0;
                if ((tb + 24 < fb || tb + 24 >= lim - fe) && lim > tb + 24) {
                    f_146 = 1;
                }
                if (f_146 != 0) {
                    cls_145 = 2139094528;
                }
                unsigned int key_147 = 0;
                if (qf < __uint_as_float(cls_145) && cls_145 < 4278190080u) {
                    key_147 = 1073741824 | (unsigned int)(tb + 24);
                }
                if (cls_145 == qc) {
                    unsigned int lowb_24 = (u_144 ^ (unsigned int)((int)u_144 >> 31) & 511) & 511;
                    key_147 = 536870912 | lowb_24 << 9 | (unsigned int)(tb + 24);
                }
                kbt2[24] = __uint_as_float(key_147);
                float sc_148 = __uint_as_float(cbt[25]);
                float _fmax_396 = fmaxf(sc_148, -1.7014118346046923e+38f);
                sc_148 = _fmax_396;
                float _min_364 = fminf(sc_148, 1.7014118346046923e+38f);
                sc_148 = _min_364;
                sc_148 = sc_148;
                float sc2_149 = sc_148;
                unsigned int u_150 = __as_u32(sc2_149);
                unsigned int cls_151 = u_150 & 4294966784u;
                int f_152 = 0;
                if ((tb + 25 < fb || tb + 25 >= lim - fe) && lim > tb + 25) {
                    f_152 = 1;
                }
                if (f_152 != 0) {
                    cls_151 = 2139094528;
                }
                unsigned int key_153 = 0;
                if (qf < __uint_as_float(cls_151) && cls_151 < 4278190080u) {
                    key_153 = 1073741824 | (unsigned int)(tb + 25);
                }
                if (cls_151 == qc) {
                    unsigned int lowb_25 = (u_150 ^ (unsigned int)((int)u_150 >> 31) & 511) & 511;
                    key_153 = 536870912 | lowb_25 << 9 | (unsigned int)(tb + 25);
                }
                kbt2[25] = __uint_as_float(key_153);
                float sc_154 = __uint_as_float(cbt[26]);
                float _fmax_397 = fmaxf(sc_154, -1.7014118346046923e+38f);
                sc_154 = _fmax_397;
                float _min_365 = fminf(sc_154, 1.7014118346046923e+38f);
                sc_154 = _min_365;
                sc_154 = sc_154;
                float sc2_155 = sc_154;
                unsigned int u_156 = __as_u32(sc2_155);
                unsigned int cls_157 = u_156 & 4294966784u;
                int f_158 = 0;
                if ((tb + 26 < fb || tb + 26 >= lim - fe) && lim > tb + 26) {
                    f_158 = 1;
                }
                if (f_158 != 0) {
                    cls_157 = 2139094528;
                }
                unsigned int key_159 = 0;
                if (qf < __uint_as_float(cls_157) && cls_157 < 4278190080u) {
                    key_159 = 1073741824 | (unsigned int)(tb + 26);
                }
                if (cls_157 == qc) {
                    unsigned int lowb_26 = (u_156 ^ (unsigned int)((int)u_156 >> 31) & 511) & 511;
                    key_159 = 536870912 | lowb_26 << 9 | (unsigned int)(tb + 26);
                }
                kbt2[26] = __uint_as_float(key_159);
                float sc_160 = __uint_as_float(cbt[27]);
                float _fmax_398 = fmaxf(sc_160, -1.7014118346046923e+38f);
                sc_160 = _fmax_398;
                float _min_366 = fminf(sc_160, 1.7014118346046923e+38f);
                sc_160 = _min_366;
                sc_160 = sc_160;
                float sc2_161 = sc_160;
                unsigned int u_162 = __as_u32(sc2_161);
                unsigned int cls_163 = u_162 & 4294966784u;
                int f_164 = 0;
                if ((tb + 27 < fb || tb + 27 >= lim - fe) && lim > tb + 27) {
                    f_164 = 1;
                }
                if (f_164 != 0) {
                    cls_163 = 2139094528;
                }
                unsigned int key_165 = 0;
                if (qf < __uint_as_float(cls_163) && cls_163 < 4278190080u) {
                    key_165 = 1073741824 | (unsigned int)(tb + 27);
                }
                if (cls_163 == qc) {
                    unsigned int lowb_27 = (u_162 ^ (unsigned int)((int)u_162 >> 31) & 511) & 511;
                    key_165 = 536870912 | lowb_27 << 9 | (unsigned int)(tb + 27);
                }
                kbt2[27] = __uint_as_float(key_165);
                float sc_166 = __uint_as_float(cbt[28]);
                float _fmax_399 = fmaxf(sc_166, -1.7014118346046923e+38f);
                sc_166 = _fmax_399;
                float _min_367 = fminf(sc_166, 1.7014118346046923e+38f);
                sc_166 = _min_367;
                sc_166 = sc_166;
                float sc2_167 = sc_166;
                unsigned int u_168 = __as_u32(sc2_167);
                unsigned int cls_169 = u_168 & 4294966784u;
                int f_170 = 0;
                if ((tb + 28 < fb || tb + 28 >= lim - fe) && lim > tb + 28) {
                    f_170 = 1;
                }
                if (f_170 != 0) {
                    cls_169 = 2139094528;
                }
                unsigned int key_171 = 0;
                if (qf < __uint_as_float(cls_169) && cls_169 < 4278190080u) {
                    key_171 = 1073741824 | (unsigned int)(tb + 28);
                }
                if (cls_169 == qc) {
                    unsigned int lowb_28 = (u_168 ^ (unsigned int)((int)u_168 >> 31) & 511) & 511;
                    key_171 = 536870912 | lowb_28 << 9 | (unsigned int)(tb + 28);
                }
                kbt2[28] = __uint_as_float(key_171);
                float sc_172 = __uint_as_float(cbt[29]);
                float _fmax_400 = fmaxf(sc_172, -1.7014118346046923e+38f);
                sc_172 = _fmax_400;
                float _min_368 = fminf(sc_172, 1.7014118346046923e+38f);
                sc_172 = _min_368;
                sc_172 = sc_172;
                float sc2_173 = sc_172;
                unsigned int u_174 = __as_u32(sc2_173);
                unsigned int cls_175 = u_174 & 4294966784u;
                int f_176 = 0;
                if ((tb + 29 < fb || tb + 29 >= lim - fe) && lim > tb + 29) {
                    f_176 = 1;
                }
                if (f_176 != 0) {
                    cls_175 = 2139094528;
                }
                unsigned int key_177 = 0;
                if (qf < __uint_as_float(cls_175) && cls_175 < 4278190080u) {
                    key_177 = 1073741824 | (unsigned int)(tb + 29);
                }
                if (cls_175 == qc) {
                    unsigned int lowb_29 = (u_174 ^ (unsigned int)((int)u_174 >> 31) & 511) & 511;
                    key_177 = 536870912 | lowb_29 << 9 | (unsigned int)(tb + 29);
                }
                kbt2[29] = __uint_as_float(key_177);
                float sc_178 = __uint_as_float(cbt[30]);
                float _fmax_401 = fmaxf(sc_178, -1.7014118346046923e+38f);
                sc_178 = _fmax_401;
                float _min_369 = fminf(sc_178, 1.7014118346046923e+38f);
                sc_178 = _min_369;
                sc_178 = sc_178;
                float sc2_179 = sc_178;
                unsigned int u_180 = __as_u32(sc2_179);
                unsigned int cls_181 = u_180 & 4294966784u;
                int f_182 = 0;
                if ((tb + 30 < fb || tb + 30 >= lim - fe) && lim > tb + 30) {
                    f_182 = 1;
                }
                if (f_182 != 0) {
                    cls_181 = 2139094528;
                }
                unsigned int key_183 = 0;
                if (qf < __uint_as_float(cls_181) && cls_181 < 4278190080u) {
                    key_183 = 1073741824 | (unsigned int)(tb + 30);
                }
                if (cls_181 == qc) {
                    unsigned int lowb_30 = (u_180 ^ (unsigned int)((int)u_180 >> 31) & 511) & 511;
                    key_183 = 536870912 | lowb_30 << 9 | (unsigned int)(tb + 30);
                }
                kbt2[30] = __uint_as_float(key_183);
                float sc_184 = __uint_as_float(cbt[31]);
                float _fmax_402 = fmaxf(sc_184, -1.7014118346046923e+38f);
                sc_184 = _fmax_402;
                float _min_370 = fminf(sc_184, 1.7014118346046923e+38f);
                sc_184 = _min_370;
                sc_184 = sc_184;
                float sc2_185 = sc_184;
                unsigned int u_186 = __as_u32(sc2_185);
                unsigned int cls_187 = u_186 & 4294966784u;
                int f_188 = 0;
                if ((tb + 31 < fb || tb + 31 >= lim - fe) && lim > tb + 31) {
                    f_188 = 1;
                }
                if (f_188 != 0) {
                    cls_187 = 2139094528;
                }
                unsigned int key_189 = 0;
                if (qf < __uint_as_float(cls_187) && cls_187 < 4278190080u) {
                    key_189 = 1073741824 | (unsigned int)(tb + 31);
                }
                if (cls_187 == qc) {
                    unsigned int lowb_31 = (u_186 ^ (unsigned int)((int)u_186 >> 31) & 511) & 511;
                    key_189 = 536870912 | lowb_31 << 9 | (unsigned int)(tb + 31);
                }
                kbt2[31] = __uint_as_float(key_189);
                float rr_190 = r2o;
                float _fmax_403 = fmaxf(kbt2[0], kbt2[13]);
                float hi_192 = _fmax_403;
                float _min_371 = fminf(kbt2[0], kbt2[13]);
                float lo_193 = _min_371;
                kbt2[0] = hi_192;
                kbt2[13] = lo_193;
                float _fmax_404 = fmaxf(kbt2[1], kbt2[12]);
                float hi_194 = _fmax_404;
                float _min_372 = fminf(kbt2[1], kbt2[12]);
                float lo_195 = _min_372;
                kbt2[1] = hi_194;
                kbt2[12] = lo_195;
                float _fmax_405 = fmaxf(kbt2[2], kbt2[15]);
                float hi_196 = _fmax_405;
                float _min_373 = fminf(kbt2[2], kbt2[15]);
                float lo_197 = _min_373;
                kbt2[2] = hi_196;
                kbt2[15] = lo_197;
                float _fmax_406 = fmaxf(kbt2[3], kbt2[14]);
                float hi_198 = _fmax_406;
                float _min_374 = fminf(kbt2[3], kbt2[14]);
                float lo_199 = _min_374;
                kbt2[3] = hi_198;
                kbt2[14] = lo_199;
                float _fmax_407 = fmaxf(kbt2[4], kbt2[8]);
                float hi_200 = _fmax_407;
                float _min_375 = fminf(kbt2[4], kbt2[8]);
                float lo_201 = _min_375;
                kbt2[4] = hi_200;
                kbt2[8] = lo_201;
                float _fmax_408 = fmaxf(kbt2[5], kbt2[6]);
                float hi_202 = _fmax_408;
                float _min_376 = fminf(kbt2[5], kbt2[6]);
                float lo_203 = _min_376;
                kbt2[5] = hi_202;
                kbt2[6] = lo_203;
                float _fmax_409 = fmaxf(kbt2[7], kbt2[11]);
                float hi_204 = _fmax_409;
                float _min_377 = fminf(kbt2[7], kbt2[11]);
                float lo_205 = _min_377;
                kbt2[7] = hi_204;
                kbt2[11] = lo_205;
                float _fmax_410 = fmaxf(kbt2[9], kbt2[10]);
                float hi_206 = _fmax_410;
                float _min_378 = fminf(kbt2[9], kbt2[10]);
                float lo_207 = _min_378;
                kbt2[9] = hi_206;
                kbt2[10] = lo_207;
                float _fmax_411 = fmaxf(kbt2[0], kbt2[5]);
                float hi_208 = _fmax_411;
                float _min_379 = fminf(kbt2[0], kbt2[5]);
                float lo_209 = _min_379;
                kbt2[0] = hi_208;
                kbt2[5] = lo_209;
                float _fmax_412 = fmaxf(kbt2[1], kbt2[7]);
                float hi_210 = _fmax_412;
                float _min_380 = fminf(kbt2[1], kbt2[7]);
                float lo_211 = _min_380;
                kbt2[1] = hi_210;
                kbt2[7] = lo_211;
                float _fmax_413 = fmaxf(kbt2[2], kbt2[9]);
                float hi_212 = _fmax_413;
                float _min_381 = fminf(kbt2[2], kbt2[9]);
                float lo_213 = _min_381;
                kbt2[2] = hi_212;
                kbt2[9] = lo_213;
                float _fmax_414 = fmaxf(kbt2[3], kbt2[4]);
                float hi_214 = _fmax_414;
                float _min_382 = fminf(kbt2[3], kbt2[4]);
                float lo_215 = _min_382;
                kbt2[3] = hi_214;
                kbt2[4] = lo_215;
                float _fmax_415 = fmaxf(kbt2[6], kbt2[13]);
                float hi_216 = _fmax_415;
                float _min_383 = fminf(kbt2[6], kbt2[13]);
                float lo_217 = _min_383;
                kbt2[6] = hi_216;
                kbt2[13] = lo_217;
                float _fmax_416 = fmaxf(kbt2[8], kbt2[14]);
                float hi_218 = _fmax_416;
                float _min_384 = fminf(kbt2[8], kbt2[14]);
                float lo_219 = _min_384;
                kbt2[8] = hi_218;
                kbt2[14] = lo_219;
                float _fmax_417 = fmaxf(kbt2[10], kbt2[15]);
                float hi_220 = _fmax_417;
                float _min_385 = fminf(kbt2[10], kbt2[15]);
                float lo_221 = _min_385;
                kbt2[10] = hi_220;
                kbt2[15] = lo_221;
                float _fmax_418 = fmaxf(kbt2[11], kbt2[12]);
                float hi_222 = _fmax_418;
                float _min_386 = fminf(kbt2[11], kbt2[12]);
                float lo_223 = _min_386;
                kbt2[11] = hi_222;
                kbt2[12] = lo_223;
                float _fmax_419 = fmaxf(kbt2[0], kbt2[1]);
                float hi_224 = _fmax_419;
                float _min_387 = fminf(kbt2[0], kbt2[1]);
                float lo_225 = _min_387;
                kbt2[0] = hi_224;
                kbt2[1] = lo_225;
                float _fmax_420 = fmaxf(kbt2[2], kbt2[3]);
                float hi_226 = _fmax_420;
                float _min_388 = fminf(kbt2[2], kbt2[3]);
                float lo_227 = _min_388;
                kbt2[2] = hi_226;
                kbt2[3] = lo_227;
                float _fmax_421 = fmaxf(kbt2[4], kbt2[5]);
                float hi_228 = _fmax_421;
                float _min_389 = fminf(kbt2[4], kbt2[5]);
                float lo_229 = _min_389;
                kbt2[4] = hi_228;
                kbt2[5] = lo_229;
                float _fmax_422 = fmaxf(kbt2[6], kbt2[8]);
                float hi_230 = _fmax_422;
                float _min_390 = fminf(kbt2[6], kbt2[8]);
                float lo_231 = _min_390;
                kbt2[6] = hi_230;
                kbt2[8] = lo_231;
                float _fmax_423 = fmaxf(kbt2[7], kbt2[9]);
                float hi_232 = _fmax_423;
                float _min_391 = fminf(kbt2[7], kbt2[9]);
                float lo_233 = _min_391;
                kbt2[7] = hi_232;
                kbt2[9] = lo_233;
                float _fmax_424 = fmaxf(kbt2[10], kbt2[11]);
                float hi_234 = _fmax_424;
                float _min_392 = fminf(kbt2[10], kbt2[11]);
                float lo_235 = _min_392;
                kbt2[10] = hi_234;
                kbt2[11] = lo_235;
                float _fmax_425 = fmaxf(kbt2[12], kbt2[13]);
                float hi_236 = _fmax_425;
                float _min_393 = fminf(kbt2[12], kbt2[13]);
                float lo_237 = _min_393;
                kbt2[12] = hi_236;
                kbt2[13] = lo_237;
                float _fmax_426 = fmaxf(kbt2[14], kbt2[15]);
                float hi_238 = _fmax_426;
                float _min_394 = fminf(kbt2[14], kbt2[15]);
                float lo_239 = _min_394;
                kbt2[14] = hi_238;
                kbt2[15] = lo_239;
                float _fmax_427 = fmaxf(kbt2[0], kbt2[2]);
                float hi_240 = _fmax_427;
                float _min_395 = fminf(kbt2[0], kbt2[2]);
                float lo_241 = _min_395;
                kbt2[0] = hi_240;
                kbt2[2] = lo_241;
                float _fmax_428 = fmaxf(kbt2[1], kbt2[3]);
                float hi_242 = _fmax_428;
                float _min_396 = fminf(kbt2[1], kbt2[3]);
                float lo_243 = _min_396;
                kbt2[1] = hi_242;
                kbt2[3] = lo_243;
                float _fmax_429 = fmaxf(kbt2[4], kbt2[10]);
                float hi_244 = _fmax_429;
                float _min_397 = fminf(kbt2[4], kbt2[10]);
                float lo_245 = _min_397;
                kbt2[4] = hi_244;
                kbt2[10] = lo_245;
                float _fmax_430 = fmaxf(kbt2[5], kbt2[11]);
                float hi_246 = _fmax_430;
                float _min_398 = fminf(kbt2[5], kbt2[11]);
                float lo_247 = _min_398;
                kbt2[5] = hi_246;
                kbt2[11] = lo_247;
                float _fmax_431 = fmaxf(kbt2[6], kbt2[7]);
                float hi_248 = _fmax_431;
                float _min_399 = fminf(kbt2[6], kbt2[7]);
                float lo_249 = _min_399;
                kbt2[6] = hi_248;
                kbt2[7] = lo_249;
                float _fmax_432 = fmaxf(kbt2[8], kbt2[9]);
                float hi_250 = _fmax_432;
                float _min_400 = fminf(kbt2[8], kbt2[9]);
                float lo_251 = _min_400;
                kbt2[8] = hi_250;
                kbt2[9] = lo_251;
                float _fmax_433 = fmaxf(kbt2[12], kbt2[14]);
                float hi_252 = _fmax_433;
                float _min_401 = fminf(kbt2[12], kbt2[14]);
                float lo_253 = _min_401;
                kbt2[12] = hi_252;
                kbt2[14] = lo_253;
                float _fmax_434 = fmaxf(kbt2[13], kbt2[15]);
                float hi_254 = _fmax_434;
                float _min_402 = fminf(kbt2[13], kbt2[15]);
                float lo_255 = _min_402;
                kbt2[13] = hi_254;
                kbt2[15] = lo_255;
                float _fmax_435 = fmaxf(kbt2[1], kbt2[2]);
                float hi_256 = _fmax_435;
                float _min_403 = fminf(kbt2[1], kbt2[2]);
                float lo_257 = _min_403;
                kbt2[1] = hi_256;
                kbt2[2] = lo_257;
                float _fmax_436 = fmaxf(kbt2[3], kbt2[12]);
                float hi_258 = _fmax_436;
                float _min_404 = fminf(kbt2[3], kbt2[12]);
                float lo_259 = _min_404;
                kbt2[3] = hi_258;
                kbt2[12] = lo_259;
                float _fmax_437 = fmaxf(kbt2[4], kbt2[6]);
                float hi_260 = _fmax_437;
                float _min_405 = fminf(kbt2[4], kbt2[6]);
                float lo_261 = _min_405;
                kbt2[4] = hi_260;
                kbt2[6] = lo_261;
                float _fmax_438 = fmaxf(kbt2[5], kbt2[7]);
                float hi_262 = _fmax_438;
                float _min_406 = fminf(kbt2[5], kbt2[7]);
                float lo_263 = _min_406;
                kbt2[5] = hi_262;
                kbt2[7] = lo_263;
                float _fmax_439 = fmaxf(kbt2[8], kbt2[10]);
                float hi_264 = _fmax_439;
                float _min_407 = fminf(kbt2[8], kbt2[10]);
                float lo_265 = _min_407;
                kbt2[8] = hi_264;
                kbt2[10] = lo_265;
                float _fmax_440 = fmaxf(kbt2[9], kbt2[11]);
                float hi_266 = _fmax_440;
                float _min_408 = fminf(kbt2[9], kbt2[11]);
                float lo_267 = _min_408;
                kbt2[9] = hi_266;
                kbt2[11] = lo_267;
                float _fmax_441 = fmaxf(kbt2[13], kbt2[14]);
                float hi_268 = _fmax_441;
                float _min_409 = fminf(kbt2[13], kbt2[14]);
                float lo_269 = _min_409;
                kbt2[13] = hi_268;
                kbt2[14] = lo_269;
                float _fmax_442 = fmaxf(kbt2[1], kbt2[4]);
                float hi_270 = _fmax_442;
                float _min_410 = fminf(kbt2[1], kbt2[4]);
                float lo_271 = _min_410;
                kbt2[1] = hi_270;
                kbt2[4] = lo_271;
                float _fmax_443 = fmaxf(kbt2[2], kbt2[6]);
                float hi_272 = _fmax_443;
                float _min_411 = fminf(kbt2[2], kbt2[6]);
                float lo_273 = _min_411;
                kbt2[2] = hi_272;
                kbt2[6] = lo_273;
                float _fmax_444 = fmaxf(kbt2[5], kbt2[8]);
                float hi_274 = _fmax_444;
                float _min_412 = fminf(kbt2[5], kbt2[8]);
                float lo_275 = _min_412;
                kbt2[5] = hi_274;
                kbt2[8] = lo_275;
                float _fmax_445 = fmaxf(kbt2[7], kbt2[10]);
                float hi_276 = _fmax_445;
                float _min_413 = fminf(kbt2[7], kbt2[10]);
                float lo_277 = _min_413;
                kbt2[7] = hi_276;
                kbt2[10] = lo_277;
                float _fmax_446 = fmaxf(kbt2[9], kbt2[13]);
                float hi_278 = _fmax_446;
                float _min_414 = fminf(kbt2[9], kbt2[13]);
                float lo_279 = _min_414;
                kbt2[9] = hi_278;
                kbt2[13] = lo_279;
                float _fmax_447 = fmaxf(kbt2[11], kbt2[14]);
                float hi_280 = _fmax_447;
                float _min_415 = fminf(kbt2[11], kbt2[14]);
                float lo_281 = _min_415;
                kbt2[11] = hi_280;
                kbt2[14] = lo_281;
                float _fmax_448 = fmaxf(kbt2[2], kbt2[4]);
                float hi_282 = _fmax_448;
                float _min_416 = fminf(kbt2[2], kbt2[4]);
                float lo_283 = _min_416;
                kbt2[2] = hi_282;
                kbt2[4] = lo_283;
                float _fmax_449 = fmaxf(kbt2[3], kbt2[6]);
                float hi_284 = _fmax_449;
                float _min_417 = fminf(kbt2[3], kbt2[6]);
                float lo_285 = _min_417;
                kbt2[3] = hi_284;
                kbt2[6] = lo_285;
                float _fmax_450 = fmaxf(kbt2[9], kbt2[12]);
                float hi_286 = _fmax_450;
                float _min_418 = fminf(kbt2[9], kbt2[12]);
                float lo_287 = _min_418;
                kbt2[9] = hi_286;
                kbt2[12] = lo_287;
                float _fmax_451 = fmaxf(kbt2[11], kbt2[13]);
                float hi_288 = _fmax_451;
                float _min_419 = fminf(kbt2[11], kbt2[13]);
                float lo_289 = _min_419;
                kbt2[11] = hi_288;
                kbt2[13] = lo_289;
                float _fmax_452 = fmaxf(kbt2[3], kbt2[5]);
                float hi_290 = _fmax_452;
                float _min_420 = fminf(kbt2[3], kbt2[5]);
                float lo_291 = _min_420;
                kbt2[3] = hi_290;
                kbt2[5] = lo_291;
                float _fmax_453 = fmaxf(kbt2[6], kbt2[8]);
                float hi_292 = _fmax_453;
                float _min_421 = fminf(kbt2[6], kbt2[8]);
                float lo_293 = _min_421;
                kbt2[6] = hi_292;
                kbt2[8] = lo_293;
                float _fmax_454 = fmaxf(kbt2[7], kbt2[9]);
                float hi_294 = _fmax_454;
                float _min_422 = fminf(kbt2[7], kbt2[9]);
                float lo_295 = _min_422;
                kbt2[7] = hi_294;
                kbt2[9] = lo_295;
                float _fmax_455 = fmaxf(kbt2[10], kbt2[12]);
                float hi_296 = _fmax_455;
                float _min_423 = fminf(kbt2[10], kbt2[12]);
                float lo_297 = _min_423;
                kbt2[10] = hi_296;
                kbt2[12] = lo_297;
                float _fmax_456 = fmaxf(kbt2[3], kbt2[4]);
                float hi_298 = _fmax_456;
                float _min_424 = fminf(kbt2[3], kbt2[4]);
                float lo_299 = _min_424;
                kbt2[3] = hi_298;
                kbt2[4] = lo_299;
                float _fmax_457 = fmaxf(kbt2[5], kbt2[6]);
                float hi_300 = _fmax_457;
                float _min_425 = fminf(kbt2[5], kbt2[6]);
                float lo_301 = _min_425;
                kbt2[5] = hi_300;
                kbt2[6] = lo_301;
                float _fmax_458 = fmaxf(kbt2[7], kbt2[8]);
                float hi_302 = _fmax_458;
                float _min_426 = fminf(kbt2[7], kbt2[8]);
                float lo_303 = _min_426;
                kbt2[7] = hi_302;
                kbt2[8] = lo_303;
                float _fmax_459 = fmaxf(kbt2[9], kbt2[10]);
                float hi_304 = _fmax_459;
                float _min_427 = fminf(kbt2[9], kbt2[10]);
                float lo_305 = _min_427;
                kbt2[9] = hi_304;
                kbt2[10] = lo_305;
                float _fmax_460 = fmaxf(kbt2[11], kbt2[12]);
                float hi_306 = _fmax_460;
                float _min_428 = fminf(kbt2[11], kbt2[12]);
                float lo_307 = _min_428;
                kbt2[11] = hi_306;
                kbt2[12] = lo_307;
                float _fmax_461 = fmaxf(kbt2[6], kbt2[7]);
                float hi_308 = _fmax_461;
                float _min_429 = fminf(kbt2[6], kbt2[7]);
                float lo_309 = _min_429;
                kbt2[6] = hi_308;
                kbt2[7] = lo_309;
                float _fmax_462 = fmaxf(kbt2[8], kbt2[9]);
                float hi_310 = _fmax_462;
                float _min_430 = fminf(kbt2[8], kbt2[9]);
                float lo_311 = _min_430;
                kbt2[8] = hi_310;
                kbt2[9] = lo_311;
                float _fmax_463 = fmaxf(kbt2[16], kbt2[29]);
                float hi_312 = _fmax_463;
                float _min_431 = fminf(kbt2[16], kbt2[29]);
                float lo_313 = _min_431;
                kbt2[16] = hi_312;
                kbt2[29] = lo_313;
                float _fmax_464 = fmaxf(kbt2[17], kbt2[28]);
                float hi_314 = _fmax_464;
                float _min_432 = fminf(kbt2[17], kbt2[28]);
                float lo_315 = _min_432;
                kbt2[17] = hi_314;
                kbt2[28] = lo_315;
                float _fmax_465 = fmaxf(kbt2[18], kbt2[31]);
                float hi_316 = _fmax_465;
                float _min_433 = fminf(kbt2[18], kbt2[31]);
                float lo_317 = _min_433;
                kbt2[18] = hi_316;
                kbt2[31] = lo_317;
                float _fmax_466 = fmaxf(kbt2[19], kbt2[30]);
                float hi_318 = _fmax_466;
                float _min_434 = fminf(kbt2[19], kbt2[30]);
                float lo_319 = _min_434;
                kbt2[19] = hi_318;
                kbt2[30] = lo_319;
                float _fmax_467 = fmaxf(kbt2[20], kbt2[24]);
                float hi_320 = _fmax_467;
                float _min_435 = fminf(kbt2[20], kbt2[24]);
                float lo_321 = _min_435;
                kbt2[20] = hi_320;
                kbt2[24] = lo_321;
                float _fmax_468 = fmaxf(kbt2[21], kbt2[22]);
                float hi_322 = _fmax_468;
                float _min_436 = fminf(kbt2[21], kbt2[22]);
                float lo_323 = _min_436;
                kbt2[21] = hi_322;
                kbt2[22] = lo_323;
                float _fmax_469 = fmaxf(kbt2[23], kbt2[27]);
                float hi_324 = _fmax_469;
                float _min_437 = fminf(kbt2[23], kbt2[27]);
                float lo_325 = _min_437;
                kbt2[23] = hi_324;
                kbt2[27] = lo_325;
                float _fmax_470 = fmaxf(kbt2[25], kbt2[26]);
                float hi_326 = _fmax_470;
                float _min_438 = fminf(kbt2[25], kbt2[26]);
                float lo_327 = _min_438;
                kbt2[25] = hi_326;
                kbt2[26] = lo_327;
                float _fmax_471 = fmaxf(kbt2[16], kbt2[21]);
                float hi_328 = _fmax_471;
                float _min_439 = fminf(kbt2[16], kbt2[21]);
                float lo_329 = _min_439;
                kbt2[16] = hi_328;
                kbt2[21] = lo_329;
                float _fmax_472 = fmaxf(kbt2[17], kbt2[23]);
                float hi_330 = _fmax_472;
                float _min_440 = fminf(kbt2[17], kbt2[23]);
                float lo_331 = _min_440;
                kbt2[17] = hi_330;
                kbt2[23] = lo_331;
                float _fmax_473 = fmaxf(kbt2[18], kbt2[25]);
                float hi_332 = _fmax_473;
                float _min_441 = fminf(kbt2[18], kbt2[25]);
                float lo_333 = _min_441;
                kbt2[18] = hi_332;
                kbt2[25] = lo_333;
                float _fmax_474 = fmaxf(kbt2[19], kbt2[20]);
                float hi_334 = _fmax_474;
                float _min_442 = fminf(kbt2[19], kbt2[20]);
                float lo_335 = _min_442;
                kbt2[19] = hi_334;
                kbt2[20] = lo_335;
                float _fmax_475 = fmaxf(kbt2[22], kbt2[29]);
                float hi_336 = _fmax_475;
                float _min_443 = fminf(kbt2[22], kbt2[29]);
                float lo_337 = _min_443;
                kbt2[22] = hi_336;
                kbt2[29] = lo_337;
                float _fmax_476 = fmaxf(kbt2[24], kbt2[30]);
                float hi_338 = _fmax_476;
                float _min_444 = fminf(kbt2[24], kbt2[30]);
                float lo_339 = _min_444;
                kbt2[24] = hi_338;
                kbt2[30] = lo_339;
                float _fmax_477 = fmaxf(kbt2[26], kbt2[31]);
                float hi_340 = _fmax_477;
                float _min_445 = fminf(kbt2[26], kbt2[31]);
                float lo_341 = _min_445;
                kbt2[26] = hi_340;
                kbt2[31] = lo_341;
                float _fmax_478 = fmaxf(kbt2[27], kbt2[28]);
                float hi_342 = _fmax_478;
                float _min_446 = fminf(kbt2[27], kbt2[28]);
                float lo_343 = _min_446;
                kbt2[27] = hi_342;
                kbt2[28] = lo_343;
                float _fmax_479 = fmaxf(kbt2[16], kbt2[17]);
                float hi_344 = _fmax_479;
                float _min_447 = fminf(kbt2[16], kbt2[17]);
                float lo_345 = _min_447;
                kbt2[16] = hi_344;
                kbt2[17] = lo_345;
                float _fmax_480 = fmaxf(kbt2[18], kbt2[19]);
                float hi_346 = _fmax_480;
                float _min_448 = fminf(kbt2[18], kbt2[19]);
                float lo_347 = _min_448;
                kbt2[18] = hi_346;
                kbt2[19] = lo_347;
                float _fmax_481 = fmaxf(kbt2[20], kbt2[21]);
                float hi_348 = _fmax_481;
                float _min_449 = fminf(kbt2[20], kbt2[21]);
                float lo_349 = _min_449;
                kbt2[20] = hi_348;
                kbt2[21] = lo_349;
                float _fmax_482 = fmaxf(kbt2[22], kbt2[24]);
                float hi_350 = _fmax_482;
                float _min_450 = fminf(kbt2[22], kbt2[24]);
                float lo_351 = _min_450;
                kbt2[22] = hi_350;
                kbt2[24] = lo_351;
                float _fmax_483 = fmaxf(kbt2[23], kbt2[25]);
                float hi_352 = _fmax_483;
                float _min_451 = fminf(kbt2[23], kbt2[25]);
                float lo_353 = _min_451;
                kbt2[23] = hi_352;
                kbt2[25] = lo_353;
                float _fmax_484 = fmaxf(kbt2[26], kbt2[27]);
                float hi_354 = _fmax_484;
                float _min_452 = fminf(kbt2[26], kbt2[27]);
                float lo_355 = _min_452;
                kbt2[26] = hi_354;
                kbt2[27] = lo_355;
                float _fmax_485 = fmaxf(kbt2[28], kbt2[29]);
                float hi_356 = _fmax_485;
                float _min_453 = fminf(kbt2[28], kbt2[29]);
                float lo_357 = _min_453;
                kbt2[28] = hi_356;
                kbt2[29] = lo_357;
                float _fmax_486 = fmaxf(kbt2[30], kbt2[31]);
                float hi_358 = _fmax_486;
                float _min_454 = fminf(kbt2[30], kbt2[31]);
                float lo_359 = _min_454;
                kbt2[30] = hi_358;
                kbt2[31] = lo_359;
                float _fmax_487 = fmaxf(kbt2[16], kbt2[18]);
                float hi_360 = _fmax_487;
                float _min_455 = fminf(kbt2[16], kbt2[18]);
                float lo_361 = _min_455;
                kbt2[16] = hi_360;
                kbt2[18] = lo_361;
                float _fmax_488 = fmaxf(kbt2[17], kbt2[19]);
                float hi_362 = _fmax_488;
                float _min_456 = fminf(kbt2[17], kbt2[19]);
                float lo_363 = _min_456;
                kbt2[17] = hi_362;
                kbt2[19] = lo_363;
                float _fmax_489 = fmaxf(kbt2[20], kbt2[26]);
                float hi_364 = _fmax_489;
                float _min_457 = fminf(kbt2[20], kbt2[26]);
                float lo_365 = _min_457;
                kbt2[20] = hi_364;
                kbt2[26] = lo_365;
                float _fmax_490 = fmaxf(kbt2[21], kbt2[27]);
                float hi_366 = _fmax_490;
                float _min_458 = fminf(kbt2[21], kbt2[27]);
                float lo_367 = _min_458;
                kbt2[21] = hi_366;
                kbt2[27] = lo_367;
                float _fmax_491 = fmaxf(kbt2[22], kbt2[23]);
                float hi_368 = _fmax_491;
                float _min_459 = fminf(kbt2[22], kbt2[23]);
                float lo_369 = _min_459;
                kbt2[22] = hi_368;
                kbt2[23] = lo_369;
                float _fmax_492 = fmaxf(kbt2[24], kbt2[25]);
                float hi_370 = _fmax_492;
                float _min_460 = fminf(kbt2[24], kbt2[25]);
                float lo_371 = _min_460;
                kbt2[24] = hi_370;
                kbt2[25] = lo_371;
                float _fmax_493 = fmaxf(kbt2[28], kbt2[30]);
                float hi_372 = _fmax_493;
                float _min_461 = fminf(kbt2[28], kbt2[30]);
                float lo_373 = _min_461;
                kbt2[28] = hi_372;
                kbt2[30] = lo_373;
                float _fmax_494 = fmaxf(kbt2[29], kbt2[31]);
                float hi_374 = _fmax_494;
                float _min_462 = fminf(kbt2[29], kbt2[31]);
                float lo_375 = _min_462;
                kbt2[29] = hi_374;
                kbt2[31] = lo_375;
                float _fmax_495 = fmaxf(kbt2[17], kbt2[18]);
                float hi_376 = _fmax_495;
                float _min_463 = fminf(kbt2[17], kbt2[18]);
                float lo_377 = _min_463;
                kbt2[17] = hi_376;
                kbt2[18] = lo_377;
                float _fmax_496 = fmaxf(kbt2[19], kbt2[28]);
                float hi_378 = _fmax_496;
                float _min_464 = fminf(kbt2[19], kbt2[28]);
                float lo_379 = _min_464;
                kbt2[19] = hi_378;
                kbt2[28] = lo_379;
                float _fmax_497 = fmaxf(kbt2[20], kbt2[22]);
                float hi_380 = _fmax_497;
                float _min_465 = fminf(kbt2[20], kbt2[22]);
                float lo_381 = _min_465;
                kbt2[20] = hi_380;
                kbt2[22] = lo_381;
                float _fmax_498 = fmaxf(kbt2[21], kbt2[23]);
                float hi_382 = _fmax_498;
                float _min_466 = fminf(kbt2[21], kbt2[23]);
                float lo_383 = _min_466;
                kbt2[21] = hi_382;
                kbt2[23] = lo_383;
                float _fmax_499 = fmaxf(kbt2[24], kbt2[26]);
                float hi_384 = _fmax_499;
                float _min_467 = fminf(kbt2[24], kbt2[26]);
                float lo_385 = _min_467;
                kbt2[24] = hi_384;
                kbt2[26] = lo_385;
                float _fmax_500 = fmaxf(kbt2[25], kbt2[27]);
                float hi_386 = _fmax_500;
                float _min_468 = fminf(kbt2[25], kbt2[27]);
                float lo_387 = _min_468;
                kbt2[25] = hi_386;
                kbt2[27] = lo_387;
                float _fmax_501 = fmaxf(kbt2[29], kbt2[30]);
                float hi_388 = _fmax_501;
                float _min_469 = fminf(kbt2[29], kbt2[30]);
                float lo_389 = _min_469;
                kbt2[29] = hi_388;
                kbt2[30] = lo_389;
                float _fmax_502 = fmaxf(kbt2[17], kbt2[20]);
                float hi_390 = _fmax_502;
                float _min_470 = fminf(kbt2[17], kbt2[20]);
                float lo_391 = _min_470;
                kbt2[17] = hi_390;
                kbt2[20] = lo_391;
                float _fmax_503 = fmaxf(kbt2[18], kbt2[22]);
                float hi_392 = _fmax_503;
                float _min_471 = fminf(kbt2[18], kbt2[22]);
                float lo_393 = _min_471;
                kbt2[18] = hi_392;
                kbt2[22] = lo_393;
                float _fmax_504 = fmaxf(kbt2[21], kbt2[24]);
                float hi_394 = _fmax_504;
                float _min_472 = fminf(kbt2[21], kbt2[24]);
                float lo_395 = _min_472;
                kbt2[21] = hi_394;
                kbt2[24] = lo_395;
                float _fmax_505 = fmaxf(kbt2[23], kbt2[26]);
                float hi_396 = _fmax_505;
                float _min_473 = fminf(kbt2[23], kbt2[26]);
                float lo_397 = _min_473;
                kbt2[23] = hi_396;
                kbt2[26] = lo_397;
                float _fmax_506 = fmaxf(kbt2[25], kbt2[29]);
                float hi_398 = _fmax_506;
                float _min_474 = fminf(kbt2[25], kbt2[29]);
                float lo_399 = _min_474;
                kbt2[25] = hi_398;
                kbt2[29] = lo_399;
                float _fmax_507 = fmaxf(kbt2[27], kbt2[30]);
                float hi_400 = _fmax_507;
                float _min_475 = fminf(kbt2[27], kbt2[30]);
                float lo_401 = _min_475;
                kbt2[27] = hi_400;
                kbt2[30] = lo_401;
                float _fmax_508 = fmaxf(kbt2[18], kbt2[20]);
                float hi_402 = _fmax_508;
                float _min_476 = fminf(kbt2[18], kbt2[20]);
                float lo_403 = _min_476;
                kbt2[18] = hi_402;
                kbt2[20] = lo_403;
                float _fmax_509 = fmaxf(kbt2[19], kbt2[22]);
                float hi_404 = _fmax_509;
                float _min_477 = fminf(kbt2[19], kbt2[22]);
                float lo_405 = _min_477;
                kbt2[19] = hi_404;
                kbt2[22] = lo_405;
                float _fmax_510 = fmaxf(kbt2[25], kbt2[28]);
                float hi_406 = _fmax_510;
                float _min_478 = fminf(kbt2[25], kbt2[28]);
                float lo_407 = _min_478;
                kbt2[25] = hi_406;
                kbt2[28] = lo_407;
                float _fmax_511 = fmaxf(kbt2[27], kbt2[29]);
                float hi_408 = _fmax_511;
                float _min_479 = fminf(kbt2[27], kbt2[29]);
                float lo_409 = _min_479;
                kbt2[27] = hi_408;
                kbt2[29] = lo_409;
                float _fmax_512 = fmaxf(kbt2[19], kbt2[21]);
                float hi_410 = _fmax_512;
                float _min_480 = fminf(kbt2[19], kbt2[21]);
                float lo_411 = _min_480;
                kbt2[19] = hi_410;
                kbt2[21] = lo_411;
                float _fmax_513 = fmaxf(kbt2[22], kbt2[24]);
                float hi_412 = _fmax_513;
                float _min_481 = fminf(kbt2[22], kbt2[24]);
                float lo_413 = _min_481;
                kbt2[22] = hi_412;
                kbt2[24] = lo_413;
                float _fmax_514 = fmaxf(kbt2[23], kbt2[25]);
                float hi_414 = _fmax_514;
                float _min_482 = fminf(kbt2[23], kbt2[25]);
                float lo_415 = _min_482;
                kbt2[23] = hi_414;
                kbt2[25] = lo_415;
                float _fmax_515 = fmaxf(kbt2[26], kbt2[28]);
                float hi_416 = _fmax_515;
                float _min_483 = fminf(kbt2[26], kbt2[28]);
                float lo_417 = _min_483;
                kbt2[26] = hi_416;
                kbt2[28] = lo_417;
                float _fmax_516 = fmaxf(kbt2[19], kbt2[20]);
                float hi_418 = _fmax_516;
                float _min_484 = fminf(kbt2[19], kbt2[20]);
                float lo_419 = _min_484;
                kbt2[19] = hi_418;
                kbt2[20] = lo_419;
                float _fmax_517 = fmaxf(kbt2[21], kbt2[22]);
                float hi_420 = _fmax_517;
                float _min_485 = fminf(kbt2[21], kbt2[22]);
                float lo_421 = _min_485;
                kbt2[21] = hi_420;
                kbt2[22] = lo_421;
                float _fmax_518 = fmaxf(kbt2[23], kbt2[24]);
                float hi_422 = _fmax_518;
                float _min_486 = fminf(kbt2[23], kbt2[24]);
                float lo_423 = _min_486;
                kbt2[23] = hi_422;
                kbt2[24] = lo_423;
                float _fmax_519 = fmaxf(kbt2[25], kbt2[26]);
                float hi_424 = _fmax_519;
                float _min_487 = fminf(kbt2[25], kbt2[26]);
                float lo_425 = _min_487;
                kbt2[25] = hi_424;
                kbt2[26] = lo_425;
                float _fmax_520 = fmaxf(kbt2[27], kbt2[28]);
                float hi_426 = _fmax_520;
                float _min_488 = fminf(kbt2[27], kbt2[28]);
                float lo_427 = _min_488;
                kbt2[27] = hi_426;
                kbt2[28] = lo_427;
                float _fmax_521 = fmaxf(kbt2[22], kbt2[23]);
                float hi_428 = _fmax_521;
                float _min_489 = fminf(kbt2[22], kbt2[23]);
                float lo_429 = _min_489;
                kbt2[22] = hi_428;
                kbt2[23] = lo_429;
                float _fmax_522 = fmaxf(kbt2[24], kbt2[25]);
                float hi_431 = _fmax_522;
                float _min_490 = fminf(kbt2[24], kbt2[25]);
                float lo_432 = _min_490;
                kbt2[24] = hi_431;
                kbt2[25] = lo_432;
                float _fmax_523 = fmaxf(kbt2[0], kbt2[31]);
                float hi_434 = _fmax_523;
                float _min_491 = fminf(kbt2[0], kbt2[31]);
                float lo_435 = _min_491;
                kbt2[0] = hi_434;
                float _fmax_524 = fmaxf(rr_190, lo_435);
                rr_190 = _fmax_524;
                float _fmax_525 = fmaxf(kbt2[1], kbt2[30]);
                float hi_437 = _fmax_525;
                float _min_492 = fminf(kbt2[1], kbt2[30]);
                float lo_438 = _min_492;
                kbt2[1] = hi_437;
                float _fmax_526 = fmaxf(rr_190, lo_438);
                rr_190 = _fmax_526;
                float _fmax_527 = fmaxf(kbt2[2], kbt2[29]);
                float hi_439 = _fmax_527;
                float _min_493 = fminf(kbt2[2], kbt2[29]);
                float lo_440 = _min_493;
                kbt2[2] = hi_439;
                float _fmax_528 = fmaxf(rr_190, lo_440);
                rr_190 = _fmax_528;
                float _fmax_529 = fmaxf(kbt2[3], kbt2[28]);
                float hi_441 = _fmax_529;
                float _min_494 = fminf(kbt2[3], kbt2[28]);
                float lo_442 = _min_494;
                kbt2[3] = hi_441;
                float _fmax_530 = fmaxf(rr_190, lo_442);
                rr_190 = _fmax_530;
                float _fmax_531 = fmaxf(kbt2[4], kbt2[27]);
                float hi_443 = _fmax_531;
                float _min_495 = fminf(kbt2[4], kbt2[27]);
                float lo_444 = _min_495;
                kbt2[4] = hi_443;
                float _fmax_532 = fmaxf(rr_190, lo_444);
                rr_190 = _fmax_532;
                float _fmax_533 = fmaxf(kbt2[5], kbt2[26]);
                float hi_445 = _fmax_533;
                float _min_496 = fminf(kbt2[5], kbt2[26]);
                float lo_446 = _min_496;
                kbt2[5] = hi_445;
                float _fmax_534 = fmaxf(rr_190, lo_446);
                rr_190 = _fmax_534;
                float _fmax_535 = fmaxf(kbt2[6], kbt2[25]);
                float hi_447 = _fmax_535;
                float _min_497 = fminf(kbt2[6], kbt2[25]);
                float lo_448 = _min_497;
                kbt2[6] = hi_447;
                float _fmax_536 = fmaxf(rr_190, lo_448);
                rr_190 = _fmax_536;
                float _fmax_537 = fmaxf(kbt2[7], kbt2[24]);
                float hi_450 = _fmax_537;
                float _min_498 = fminf(kbt2[7], kbt2[24]);
                float lo_451 = _min_498;
                kbt2[7] = hi_450;
                float _fmax_538 = fmaxf(rr_190, lo_451);
                rr_190 = _fmax_538;
                float _fmax_539 = fmaxf(kbt2[8], kbt2[23]);
                float hi_453 = _fmax_539;
                float _min_499 = fminf(kbt2[8], kbt2[23]);
                float lo_454 = _min_499;
                kbt2[8] = hi_453;
                float _fmax_540 = fmaxf(rr_190, lo_454);
                rr_190 = _fmax_540;
                float _fmax_541 = fmaxf(kbt2[9], kbt2[22]);
                float hi_456 = _fmax_541;
                float _min_500 = fminf(kbt2[9], kbt2[22]);
                float lo_457 = _min_500;
                kbt2[9] = hi_456;
                float _fmax_542 = fmaxf(rr_190, lo_457);
                rr_190 = _fmax_542;
                float _fmax_543 = fmaxf(kbt2[10], kbt2[21]);
                float hi_458 = _fmax_543;
                float _min_501 = fminf(kbt2[10], kbt2[21]);
                float lo_459 = _min_501;
                kbt2[10] = hi_458;
                float _fmax_544 = fmaxf(rr_190, lo_459);
                rr_190 = _fmax_544;
                float _fmax_545 = fmaxf(kbt2[11], kbt2[20]);
                float hi_460 = _fmax_545;
                float _min_502 = fminf(kbt2[11], kbt2[20]);
                float lo_461 = _min_502;
                kbt2[11] = hi_460;
                float _fmax_546 = fmaxf(rr_190, lo_461);
                rr_190 = _fmax_546;
                float _fmax_547 = fmaxf(kbt2[12], kbt2[19]);
                float hi_462 = _fmax_547;
                float _min_503 = fminf(kbt2[12], kbt2[19]);
                float lo_463 = _min_503;
                kbt2[12] = hi_462;
                float _fmax_548 = fmaxf(rr_190, lo_463);
                rr_190 = _fmax_548;
                float _fmax_549 = fmaxf(kbt2[13], kbt2[18]);
                float hi_464 = _fmax_549;
                float _min_504 = fminf(kbt2[13], kbt2[18]);
                float lo_465 = _min_504;
                kbt2[13] = hi_464;
                float _fmax_550 = fmaxf(rr_190, lo_465);
                rr_190 = _fmax_550;
                float _fmax_551 = fmaxf(kbt2[14], kbt2[17]);
                float hi_466 = _fmax_551;
                float _min_505 = fminf(kbt2[14], kbt2[17]);
                float lo_467 = _min_505;
                kbt2[14] = hi_466;
                float _fmax_552 = fmaxf(rr_190, lo_467);
                rr_190 = _fmax_552;
                float _fmax_553 = fmaxf(kbt2[15], kbt2[16]);
                float hi_469 = _fmax_553;
                float _min_506 = fminf(kbt2[15], kbt2[16]);
                float lo_470 = _min_506;
                kbt2[15] = hi_469;
                float _fmax_554 = fmaxf(rr_190, lo_470);
                rr_190 = _fmax_554;
                float _fmax_555 = fmaxf(kbt2[0], kbt2[8]);
                float hi_472 = _fmax_555;
                float _min_507 = fminf(kbt2[0], kbt2[8]);
                float lo_473 = _min_507;
                kbt2[0] = hi_472;
                kbt2[8] = lo_473;
                float _fmax_556 = fmaxf(kbt2[1], kbt2[9]);
                float hi_475 = _fmax_556;
                float _min_508 = fminf(kbt2[1], kbt2[9]);
                float lo_476 = _min_508;
                kbt2[1] = hi_475;
                kbt2[9] = lo_476;
                float _fmax_557 = fmaxf(kbt2[2], kbt2[10]);
                float hi_477 = _fmax_557;
                float _min_509 = fminf(kbt2[2], kbt2[10]);
                float lo_478 = _min_509;
                kbt2[2] = hi_477;
                kbt2[10] = lo_478;
                float _fmax_558 = fmaxf(kbt2[3], kbt2[11]);
                float hi_479 = _fmax_558;
                float _min_510 = fminf(kbt2[3], kbt2[11]);
                float lo_480 = _min_510;
                kbt2[3] = hi_479;
                kbt2[11] = lo_480;
                float _fmax_559 = fmaxf(kbt2[4], kbt2[12]);
                float hi_481 = _fmax_559;
                float _min_511 = fminf(kbt2[4], kbt2[12]);
                float lo_482 = _min_511;
                kbt2[4] = hi_481;
                kbt2[12] = lo_482;
                float _fmax_560 = fmaxf(kbt2[5], kbt2[13]);
                float hi_483 = _fmax_560;
                float _min_512 = fminf(kbt2[5], kbt2[13]);
                float lo_484 = _min_512;
                kbt2[5] = hi_483;
                kbt2[13] = lo_484;
                float _fmax_561 = fmaxf(kbt2[6], kbt2[14]);
                float hi_485 = _fmax_561;
                float _min_513 = fminf(kbt2[6], kbt2[14]);
                float lo_486 = _min_513;
                kbt2[6] = hi_485;
                kbt2[14] = lo_486;
                float _fmax_562 = fmaxf(kbt2[7], kbt2[15]);
                float hi_488 = _fmax_562;
                float _min_514 = fminf(kbt2[7], kbt2[15]);
                float lo_489 = _min_514;
                kbt2[7] = hi_488;
                kbt2[15] = lo_489;
                float _fmax_563 = fmaxf(kbt2[0], kbt2[4]);
                float hi_491 = _fmax_563;
                float _min_515 = fminf(kbt2[0], kbt2[4]);
                float lo_492 = _min_515;
                kbt2[0] = hi_491;
                kbt2[4] = lo_492;
                float _fmax_564 = fmaxf(kbt2[1], kbt2[5]);
                float hi_494 = _fmax_564;
                float _min_516 = fminf(kbt2[1], kbt2[5]);
                float lo_495 = _min_516;
                kbt2[1] = hi_494;
                kbt2[5] = lo_495;
                float _fmax_565 = fmaxf(kbt2[2], kbt2[6]);
                float hi_496 = _fmax_565;
                float _min_517 = fminf(kbt2[2], kbt2[6]);
                float lo_497 = _min_517;
                kbt2[2] = hi_496;
                kbt2[6] = lo_497;
                float _fmax_566 = fmaxf(kbt2[3], kbt2[7]);
                float hi_498 = _fmax_566;
                float _min_518 = fminf(kbt2[3], kbt2[7]);
                float lo_499 = _min_518;
                kbt2[3] = hi_498;
                kbt2[7] = lo_499;
                float _fmax_567 = fmaxf(kbt2[8], kbt2[12]);
                float hi_500 = _fmax_567;
                float _min_519 = fminf(kbt2[8], kbt2[12]);
                float lo_501 = _min_519;
                kbt2[8] = hi_500;
                kbt2[12] = lo_501;
                float _fmax_568 = fmaxf(kbt2[9], kbt2[13]);
                float hi_502 = _fmax_568;
                float _min_520 = fminf(kbt2[9], kbt2[13]);
                float lo_503 = _min_520;
                kbt2[9] = hi_502;
                kbt2[13] = lo_503;
                float _fmax_569 = fmaxf(kbt2[10], kbt2[14]);
                float hi_504 = _fmax_569;
                float _min_521 = fminf(kbt2[10], kbt2[14]);
                float lo_505 = _min_521;
                kbt2[10] = hi_504;
                kbt2[14] = lo_505;
                float _fmax_570 = fmaxf(kbt2[11], kbt2[15]);
                float hi_507 = _fmax_570;
                float _min_522 = fminf(kbt2[11], kbt2[15]);
                float lo_508 = _min_522;
                kbt2[11] = hi_507;
                kbt2[15] = lo_508;
                float _fmax_571 = fmaxf(kbt2[0], kbt2[2]);
                float hi_510 = _fmax_571;
                float _min_523 = fminf(kbt2[0], kbt2[2]);
                float lo_511 = _min_523;
                kbt2[0] = hi_510;
                kbt2[2] = lo_511;
                float _fmax_572 = fmaxf(kbt2[1], kbt2[3]);
                float hi_513 = _fmax_572;
                float _min_524 = fminf(kbt2[1], kbt2[3]);
                float lo_514 = _min_524;
                kbt2[1] = hi_513;
                kbt2[3] = lo_514;
                float _fmax_573 = fmaxf(kbt2[4], kbt2[6]);
                float hi_515 = _fmax_573;
                float _min_525 = fminf(kbt2[4], kbt2[6]);
                float lo_516 = _min_525;
                kbt2[4] = hi_515;
                kbt2[6] = lo_516;
                float _fmax_574 = fmaxf(kbt2[5], kbt2[7]);
                float hi_517 = _fmax_574;
                float _min_526 = fminf(kbt2[5], kbt2[7]);
                float lo_518 = _min_526;
                kbt2[5] = hi_517;
                kbt2[7] = lo_518;
                float _fmax_575 = fmaxf(kbt2[8], kbt2[10]);
                float hi_519 = _fmax_575;
                float _min_527 = fminf(kbt2[8], kbt2[10]);
                float lo_520 = _min_527;
                kbt2[8] = hi_519;
                kbt2[10] = lo_520;
                float _fmax_576 = fmaxf(kbt2[9], kbt2[11]);
                float hi_521 = _fmax_576;
                float _min_528 = fminf(kbt2[9], kbt2[11]);
                float lo_522 = _min_528;
                kbt2[9] = hi_521;
                kbt2[11] = lo_522;
                float _fmax_577 = fmaxf(kbt2[12], kbt2[14]);
                float hi_523 = _fmax_577;
                float _min_529 = fminf(kbt2[12], kbt2[14]);
                float lo_524 = _min_529;
                kbt2[12] = hi_523;
                kbt2[14] = lo_524;
                float _fmax_578 = fmaxf(kbt2[13], kbt2[15]);
                float hi_526 = _fmax_578;
                float _min_530 = fminf(kbt2[13], kbt2[15]);
                float lo_527 = _min_530;
                kbt2[13] = hi_526;
                kbt2[15] = lo_527;
                float _fmax_579 = fmaxf(kbt2[0], kbt2[1]);
                float hi_529 = _fmax_579;
                float _min_531 = fminf(kbt2[0], kbt2[1]);
                float lo_530 = _min_531;
                kbt2[0] = hi_529;
                kbt2[1] = lo_530;
                float _fmax_580 = fmaxf(kbt2[2], kbt2[3]);
                float hi_532 = _fmax_580;
                float _min_532 = fminf(kbt2[2], kbt2[3]);
                float lo_533 = _min_532;
                kbt2[2] = hi_532;
                kbt2[3] = lo_533;
                float _fmax_581 = fmaxf(kbt2[4], kbt2[5]);
                float hi_534 = _fmax_581;
                float _min_533 = fminf(kbt2[4], kbt2[5]);
                float lo_535 = _min_533;
                kbt2[4] = hi_534;
                kbt2[5] = lo_535;
                float _fmax_582 = fmaxf(kbt2[6], kbt2[7]);
                float hi_536 = _fmax_582;
                float _min_534 = fminf(kbt2[6], kbt2[7]);
                float lo_537 = _min_534;
                kbt2[6] = hi_536;
                kbt2[7] = lo_537;
                float _fmax_583 = fmaxf(kbt2[8], kbt2[9]);
                float hi_538 = _fmax_583;
                float _min_535 = fminf(kbt2[8], kbt2[9]);
                float lo_539 = _min_535;
                kbt2[8] = hi_538;
                kbt2[9] = lo_539;
                float _fmax_584 = fmaxf(kbt2[10], kbt2[11]);
                float hi_540 = _fmax_584;
                float _min_536 = fminf(kbt2[10], kbt2[11]);
                float lo_541 = _min_536;
                kbt2[10] = hi_540;
                kbt2[11] = lo_541;
                float _fmax_585 = fmaxf(kbt2[12], kbt2[13]);
                float hi_542 = _fmax_585;
                float _min_537 = fminf(kbt2[12], kbt2[13]);
                float lo_543 = _min_537;
                kbt2[12] = hi_542;
                kbt2[13] = lo_543;
                float _fmax_586 = fmaxf(kbt2[14], kbt2[15]);
                float hi_545 = _fmax_586;
                float _min_538 = fminf(kbt2[14], kbt2[15]);
                float lo_546 = _min_538;
                kbt2[14] = hi_545;
                kbt2[15] = lo_546;
                r2o = rr_190;
                a2x[0] = kbt2[0];
                a2x[1] = kbt2[1];
                a2x[2] = kbt2[2];
                a2x[3] = kbt2[3];
                a2x[4] = kbt2[4];
                a2x[5] = kbt2[5];
                a2x[6] = kbt2[6];
                a2x[7] = kbt2[7];
                a2x[8] = kbt2[8];
                a2x[9] = kbt2[9];
                a2x[10] = kbt2[10];
                a2x[11] = kbt2[11];
                a2x[12] = kbt2[12];
                a2x[13] = kbt2[13];
                a2x[14] = kbt2[14];
                a2x[15] = kbt2[15];
            }
            float rr2x[1];
            int pb_0 = tid_1 * 17;
            pub[pb_0] = a2x[0];
            pub[pb_0 + 1] = a2x[1];
            pub[pb_0 + 2] = a2x[2];
            pub[pb_0 + 3] = a2x[3];
            pub[pb_0 + 4] = a2x[4];
            pub[pb_0 + 5] = a2x[5];
            pub[pb_0 + 6] = a2x[6];
            pub[pb_0 + 7] = a2x[7];
            pub[pb_0 + 8] = a2x[8];
            pub[pb_0 + 9] = a2x[9];
            pub[pb_0 + 10] = a2x[10];
            pub[pb_0 + 11] = a2x[11];
            pub[pb_0 + 12] = a2x[12];
            pub[pb_0 + 13] = a2x[13];
            pub[pb_0 + 14] = a2x[14];
            pub[pb_0 + 15] = a2x[15];
            pub[pb_0 + 16] = neg_inf;
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            float r_1 = neg_inf;
            float V_2[8];
            int s0_3 = (sg * 16 * 16 + cg) * 17;
            int s1_4 = ((sg * 16 + 8) * 16 + cg) * 17;
            float x0_5 = pub[s0_3 + ln];
            float y0_6 = pub[s1_4 + lnr];
            float _min_539 = fminf(x0_5, y0_6);
            float lo0_7 = _min_539;
            float _fmax_587 = fmaxf(r_1, lo0_7);
            r_1 = _fmax_587;
            float _fmax_588 = fmaxf(x0_5, y0_6);
            float hi0_8 = _fmax_588;
            float cur_9 = hi0_8;
            float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 8);
            float pv_10 = _shfl_xor_71;
            float _fmax_589 = fmaxf(cur_9, pv_10);
            float hi_11 = _fmax_589;
            float _min_540 = fminf(cur_9, pv_10);
            float lo_12 = _min_540;
            cur_9 = ((up[0] != 0) ? hi_11 : lo_12);
            float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 4);
            float pv_13 = _shfl_xor_72;
            float _fmax_590 = fmaxf(cur_9, pv_13);
            float hi_14 = _fmax_590;
            float _min_541 = fminf(cur_9, pv_13);
            float lo_15 = _min_541;
            cur_9 = ((up[1] != 0) ? hi_14 : lo_15);
            float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 2);
            float pv_16 = _shfl_xor_73;
            float _fmax_591 = fmaxf(cur_9, pv_16);
            float hi_17 = _fmax_591;
            float _min_542 = fminf(cur_9, pv_16);
            float lo_18 = _min_542;
            cur_9 = ((up[2] != 0) ? hi_17 : lo_18);
            float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 1);
            float pv_19 = _shfl_xor_74;
            float _fmax_592 = fmaxf(cur_9, pv_19);
            float hi_20 = _fmax_592;
            float _min_543 = fminf(cur_9, pv_19);
            float lo_21 = _min_543;
            cur_9 = ((up[3] != 0) ? hi_20 : lo_21);
            V_2[0] = cur_9;
            int s0_22 = ((sg * 16 + 1) * 16 + cg) * 17;
            int s1_23 = ((sg * 16 + 1 + 8) * 16 + cg) * 17;
            float x0_24 = pub[s0_22 + ln];
            float y0_25 = pub[s1_23 + lnr];
            float _min_544 = fminf(x0_24, y0_25);
            float lo0_26 = _min_544;
            float _fmax_593 = fmaxf(r_1, lo0_26);
            r_1 = _fmax_593;
            float _fmax_594 = fmaxf(x0_24, y0_25);
            float hi0_27 = _fmax_594;
            float cur_28 = hi0_27;
            float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 8);
            float pv_29 = _shfl_xor_75;
            float _fmax_595 = fmaxf(cur_28, pv_29);
            float hi_30 = _fmax_595;
            float _min_545 = fminf(cur_28, pv_29);
            float lo_31 = _min_545;
            cur_28 = ((up[0] != 0) ? hi_30 : lo_31);
            float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 4);
            float pv_32 = _shfl_xor_76;
            float _fmax_596 = fmaxf(cur_28, pv_32);
            float hi_33 = _fmax_596;
            float _min_546 = fminf(cur_28, pv_32);
            float lo_34 = _min_546;
            cur_28 = ((up[1] != 0) ? hi_33 : lo_34);
            float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 2);
            float pv_35 = _shfl_xor_77;
            float _fmax_597 = fmaxf(cur_28, pv_35);
            float hi_36 = _fmax_597;
            float _min_547 = fminf(cur_28, pv_35);
            float lo_37 = _min_547;
            cur_28 = ((up[2] != 0) ? hi_36 : lo_37);
            float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 1);
            float pv_38 = _shfl_xor_78;
            float _fmax_598 = fmaxf(cur_28, pv_38);
            float hi_39 = _fmax_598;
            float _min_548 = fminf(cur_28, pv_38);
            float lo_40 = _min_548;
            cur_28 = ((up[3] != 0) ? hi_39 : lo_40);
            V_2[1] = cur_28;
            int s0_41 = ((sg * 16 + 2) * 16 + cg) * 17;
            int s1_42 = ((sg * 16 + 2 + 8) * 16 + cg) * 17;
            float x0_43 = pub[s0_41 + ln];
            float y0_44 = pub[s1_42 + lnr];
            float _min_549 = fminf(x0_43, y0_44);
            float lo0_45 = _min_549;
            float _fmax_599 = fmaxf(r_1, lo0_45);
            r_1 = _fmax_599;
            float _fmax_600 = fmaxf(x0_43, y0_44);
            float hi0_46 = _fmax_600;
            float cur_47 = hi0_46;
            float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 8);
            float pv_48 = _shfl_xor_79;
            float _fmax_601 = fmaxf(cur_47, pv_48);
            float hi_49 = _fmax_601;
            float _min_550 = fminf(cur_47, pv_48);
            float lo_50 = _min_550;
            cur_47 = ((up[0] != 0) ? hi_49 : lo_50);
            float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 4);
            float pv_51 = _shfl_xor_80;
            float _fmax_602 = fmaxf(cur_47, pv_51);
            float hi_52 = _fmax_602;
            float _min_551 = fminf(cur_47, pv_51);
            float lo_53 = _min_551;
            cur_47 = ((up[1] != 0) ? hi_52 : lo_53);
            float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 2);
            float pv_54 = _shfl_xor_81;
            float _fmax_603 = fmaxf(cur_47, pv_54);
            float hi_55 = _fmax_603;
            float _min_552 = fminf(cur_47, pv_54);
            float lo_56 = _min_552;
            cur_47 = ((up[2] != 0) ? hi_55 : lo_56);
            float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, cur_47, 1);
            float pv_57 = _shfl_xor_82;
            float _fmax_604 = fmaxf(cur_47, pv_57);
            float hi_58 = _fmax_604;
            float _min_553 = fminf(cur_47, pv_57);
            float lo_59 = _min_553;
            cur_47 = ((up[3] != 0) ? hi_58 : lo_59);
            V_2[2] = cur_47;
            int s0_60 = ((sg * 16 + 3) * 16 + cg) * 17;
            int s1_61 = ((sg * 16 + 3 + 8) * 16 + cg) * 17;
            float x0_62 = pub[s0_60 + ln];
            float y0_63 = pub[s1_61 + lnr];
            float _min_554 = fminf(x0_62, y0_63);
            float lo0_64 = _min_554;
            float _fmax_605 = fmaxf(r_1, lo0_64);
            r_1 = _fmax_605;
            float _fmax_606 = fmaxf(x0_62, y0_63);
            float hi0_65 = _fmax_606;
            float cur_66 = hi0_65;
            float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 8);
            float pv_67 = _shfl_xor_83;
            float _fmax_607 = fmaxf(cur_66, pv_67);
            float hi_68 = _fmax_607;
            float _min_555 = fminf(cur_66, pv_67);
            float lo_69 = _min_555;
            cur_66 = ((up[0] != 0) ? hi_68 : lo_69);
            float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 4);
            float pv_70 = _shfl_xor_84;
            float _fmax_608 = fmaxf(cur_66, pv_70);
            float hi_71 = _fmax_608;
            float _min_556 = fminf(cur_66, pv_70);
            float lo_72 = _min_556;
            cur_66 = ((up[1] != 0) ? hi_71 : lo_72);
            float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 2);
            float pv_73 = _shfl_xor_85;
            float _fmax_609 = fmaxf(cur_66, pv_73);
            float hi_74 = _fmax_609;
            float _min_557 = fminf(cur_66, pv_73);
            float lo_75 = _min_557;
            cur_66 = ((up[2] != 0) ? hi_74 : lo_75);
            float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, cur_66, 1);
            float pv_76 = _shfl_xor_86;
            float _fmax_610 = fmaxf(cur_66, pv_76);
            float hi_77 = _fmax_610;
            float _min_558 = fminf(cur_66, pv_76);
            float lo_78 = _min_558;
            cur_66 = ((up[3] != 0) ? hi_77 : lo_78);
            V_2[3] = cur_66;
            int s0_79 = ((sg * 16 + 4) * 16 + cg) * 17;
            int s1_80 = ((sg * 16 + 4 + 8) * 16 + cg) * 17;
            float x0_81 = pub[s0_79 + ln];
            float y0_82 = pub[s1_80 + lnr];
            float _min_559 = fminf(x0_81, y0_82);
            float lo0_83 = _min_559;
            float _fmax_611 = fmaxf(r_1, lo0_83);
            r_1 = _fmax_611;
            float _fmax_612 = fmaxf(x0_81, y0_82);
            float hi0_84 = _fmax_612;
            float cur_85 = hi0_84;
            float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, cur_85, 8);
            float pv_86 = _shfl_xor_87;
            float _fmax_613 = fmaxf(cur_85, pv_86);
            float hi_87 = _fmax_613;
            float _min_560 = fminf(cur_85, pv_86);
            float lo_88 = _min_560;
            cur_85 = ((up[0] != 0) ? hi_87 : lo_88);
            float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, cur_85, 4);
            float pv_89 = _shfl_xor_88;
            float _fmax_614 = fmaxf(cur_85, pv_89);
            float hi_90 = _fmax_614;
            float _min_561 = fminf(cur_85, pv_89);
            float lo_91 = _min_561;
            cur_85 = ((up[1] != 0) ? hi_90 : lo_91);
            float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, cur_85, 2);
            float pv_92 = _shfl_xor_89;
            float _fmax_615 = fmaxf(cur_85, pv_92);
            float hi_94 = _fmax_615;
            float _min_562 = fminf(cur_85, pv_92);
            float lo_95 = _min_562;
            cur_85 = ((up[2] != 0) ? hi_94 : lo_95);
            float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, cur_85, 1);
            float pv_96 = _shfl_xor_90;
            float _fmax_616 = fmaxf(cur_85, pv_96);
            float hi_98 = _fmax_616;
            float _min_563 = fminf(cur_85, pv_96);
            float lo_99 = _min_563;
            cur_85 = ((up[3] != 0) ? hi_98 : lo_99);
            V_2[4] = cur_85;
            int s0_100 = ((sg * 16 + 5) * 16 + cg) * 17;
            int s1_101 = ((sg * 16 + 5 + 8) * 16 + cg) * 17;
            float x0_102 = pub[s0_100 + ln];
            float y0_103 = pub[s1_101 + lnr];
            float _min_564 = fminf(x0_102, y0_103);
            float lo0_104 = _min_564;
            float _fmax_617 = fmaxf(r_1, lo0_104);
            r_1 = _fmax_617;
            float _fmax_618 = fmaxf(x0_102, y0_103);
            float hi0_105 = _fmax_618;
            float cur_106 = hi0_105;
            float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, cur_106, 8);
            float pv_107 = _shfl_xor_91;
            float _fmax_619 = fmaxf(cur_106, pv_107);
            float hi_108 = _fmax_619;
            float _min_565 = fminf(cur_106, pv_107);
            float lo_109 = _min_565;
            cur_106 = ((up[0] != 0) ? hi_108 : lo_109);
            float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, cur_106, 4);
            float pv_110 = _shfl_xor_92;
            float _fmax_620 = fmaxf(cur_106, pv_110);
            float hi_112 = _fmax_620;
            float _min_566 = fminf(cur_106, pv_110);
            float lo_113 = _min_566;
            cur_106 = ((up[1] != 0) ? hi_112 : lo_113);
            float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, cur_106, 2);
            float pv_114 = _shfl_xor_93;
            float _fmax_621 = fmaxf(cur_106, pv_114);
            float hi_116 = _fmax_621;
            float _min_567 = fminf(cur_106, pv_114);
            float lo_117 = _min_567;
            cur_106 = ((up[2] != 0) ? hi_116 : lo_117);
            float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, cur_106, 1);
            float pv_118 = _shfl_xor_94;
            float _fmax_622 = fmaxf(cur_106, pv_118);
            float hi_120 = _fmax_622;
            float _min_568 = fminf(cur_106, pv_118);
            float lo_121 = _min_568;
            cur_106 = ((up[3] != 0) ? hi_120 : lo_121);
            V_2[5] = cur_106;
            int s0_122 = ((sg * 16 + 6) * 16 + cg) * 17;
            int s1_123 = ((sg * 16 + 6 + 8) * 16 + cg) * 17;
            float x0_124 = pub[s0_122 + ln];
            float y0_125 = pub[s1_123 + lnr];
            float _min_569 = fminf(x0_124, y0_125);
            float lo0_126 = _min_569;
            float _fmax_623 = fmaxf(r_1, lo0_126);
            r_1 = _fmax_623;
            float _fmax_624 = fmaxf(x0_124, y0_125);
            float hi0_127 = _fmax_624;
            float cur_128 = hi0_127;
            float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, cur_128, 8);
            float pv_129 = _shfl_xor_95;
            float _fmax_625 = fmaxf(cur_128, pv_129);
            float hi_130 = _fmax_625;
            float _min_570 = fminf(cur_128, pv_129);
            float lo_131 = _min_570;
            cur_128 = ((up[0] != 0) ? hi_130 : lo_131);
            float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, cur_128, 4);
            float pv_132 = _shfl_xor_96;
            float _fmax_626 = fmaxf(cur_128, pv_132);
            float hi_134 = _fmax_626;
            float _min_571 = fminf(cur_128, pv_132);
            float lo_135 = _min_571;
            cur_128 = ((up[1] != 0) ? hi_134 : lo_135);
            float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, cur_128, 2);
            float pv_136 = _shfl_xor_97;
            float _fmax_627 = fmaxf(cur_128, pv_136);
            float hi_138 = _fmax_627;
            float _min_572 = fminf(cur_128, pv_136);
            float lo_139 = _min_572;
            cur_128 = ((up[2] != 0) ? hi_138 : lo_139);
            float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, cur_128, 1);
            float pv_140 = _shfl_xor_98;
            float _fmax_628 = fmaxf(cur_128, pv_140);
            float hi_142 = _fmax_628;
            float _min_573 = fminf(cur_128, pv_140);
            float lo_143 = _min_573;
            cur_128 = ((up[3] != 0) ? hi_142 : lo_143);
            V_2[6] = cur_128;
            int s0_144 = ((sg * 16 + 7) * 16 + cg) * 17;
            int s1_145 = ((sg * 16 + 7 + 8) * 16 + cg) * 17;
            float x0_146 = pub[s0_144 + ln];
            float y0_147 = pub[s1_145 + lnr];
            float _min_574 = fminf(x0_146, y0_147);
            float lo0_148 = _min_574;
            float _fmax_629 = fmaxf(r_1, lo0_148);
            r_1 = _fmax_629;
            float _fmax_630 = fmaxf(x0_146, y0_147);
            float hi0_149 = _fmax_630;
            float cur_150 = hi0_149;
            float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, cur_150, 8);
            float pv_151 = _shfl_xor_99;
            float _fmax_631 = fmaxf(cur_150, pv_151);
            float hi_152 = _fmax_631;
            float _min_575 = fminf(cur_150, pv_151);
            float lo_153 = _min_575;
            cur_150 = ((up[0] != 0) ? hi_152 : lo_153);
            float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, cur_150, 4);
            float pv_154 = _shfl_xor_100;
            float _fmax_632 = fmaxf(cur_150, pv_154);
            float hi_156 = _fmax_632;
            float _min_576 = fminf(cur_150, pv_154);
            float lo_157 = _min_576;
            cur_150 = ((up[1] != 0) ? hi_156 : lo_157);
            float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, cur_150, 2);
            float pv_158 = _shfl_xor_101;
            float _fmax_633 = fmaxf(cur_150, pv_158);
            float hi_160 = _fmax_633;
            float _min_577 = fminf(cur_150, pv_158);
            float lo_161 = _min_577;
            cur_150 = ((up[2] != 0) ? hi_160 : lo_161);
            float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, cur_150, 1);
            float pv_162 = _shfl_xor_102;
            float _fmax_634 = fmaxf(cur_150, pv_162);
            float hi_164 = _fmax_634;
            float _min_578 = fminf(cur_150, pv_162);
            float lo_165 = _min_578;
            cur_150 = ((up[3] != 0) ? hi_164 : lo_165);
            V_2[7] = cur_150;
            float rs_166 = pub[((sg * 16 + ln) * 16 + cg) * 17 + 16];
            float _fmax_635 = fmaxf(r_1, rs_166);
            r_1 = _fmax_635;
            float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, V_2[4], 15);
            float y1_167 = _shfl_xor_103;
            float _min_579 = fminf(V_2[0], y1_167);
            float lo1_168 = _min_579;
            float _fmax_636 = fmaxf(r_1, lo1_168);
            r_1 = _fmax_636;
            float _fmax_637 = fmaxf(V_2[0], y1_167);
            float hi1_169 = _fmax_637;
            float cur_170 = hi1_169;
            float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, cur_170, 8);
            float pv_171 = _shfl_xor_104;
            float _fmax_638 = fmaxf(cur_170, pv_171);
            float hi_172 = _fmax_638;
            float _min_580 = fminf(cur_170, pv_171);
            float lo_173 = _min_580;
            cur_170 = ((up[0] != 0) ? hi_172 : lo_173);
            float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, cur_170, 4);
            float pv_174 = _shfl_xor_105;
            float _fmax_639 = fmaxf(cur_170, pv_174);
            float hi_176 = _fmax_639;
            float _min_581 = fminf(cur_170, pv_174);
            float lo_177 = _min_581;
            cur_170 = ((up[1] != 0) ? hi_176 : lo_177);
            float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, cur_170, 2);
            float pv_178 = _shfl_xor_106;
            float _fmax_640 = fmaxf(cur_170, pv_178);
            float hi_180 = _fmax_640;
            float _min_582 = fminf(cur_170, pv_178);
            float lo_181 = _min_582;
            cur_170 = ((up[2] != 0) ? hi_180 : lo_181);
            float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, cur_170, 1);
            float pv_182 = _shfl_xor_107;
            float _fmax_641 = fmaxf(cur_170, pv_182);
            float hi_184 = _fmax_641;
            float _min_583 = fminf(cur_170, pv_182);
            float lo_185 = _min_583;
            cur_170 = ((up[3] != 0) ? hi_184 : lo_185);
            V_2[0] = cur_170;
            float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, V_2[5], 15);
            float y1_186 = _shfl_xor_108;
            float _min_584 = fminf(V_2[1], y1_186);
            float lo1_187 = _min_584;
            float _fmax_642 = fmaxf(r_1, lo1_187);
            r_1 = _fmax_642;
            float _fmax_643 = fmaxf(V_2[1], y1_186);
            float hi1_188 = _fmax_643;
            float cur_189 = hi1_188;
            float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, cur_189, 8);
            float pv_190 = _shfl_xor_109;
            float _fmax_644 = fmaxf(cur_189, pv_190);
            float hi_192_1 = _fmax_644;
            float _min_585 = fminf(cur_189, pv_190);
            float lo_193_1 = _min_585;
            cur_189 = ((up[0] != 0) ? hi_192_1 : lo_193_1);
            float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, cur_189, 4);
            float pv_194 = _shfl_xor_110;
            float _fmax_645 = fmaxf(cur_189, pv_194);
            float hi_196_1 = _fmax_645;
            float _min_586 = fminf(cur_189, pv_194);
            float lo_197_1 = _min_586;
            cur_189 = ((up[1] != 0) ? hi_196_1 : lo_197_1);
            float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, cur_189, 2);
            float pv_198 = _shfl_xor_111;
            float _fmax_646 = fmaxf(cur_189, pv_198);
            float hi_200_1 = _fmax_646;
            float _min_587 = fminf(cur_189, pv_198);
            float lo_201_1 = _min_587;
            cur_189 = ((up[2] != 0) ? hi_200_1 : lo_201_1);
            float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, cur_189, 1);
            float pv_202 = _shfl_xor_112;
            float _fmax_647 = fmaxf(cur_189, pv_202);
            float hi_204_1 = _fmax_647;
            float _min_588 = fminf(cur_189, pv_202);
            float lo_205_1 = _min_588;
            cur_189 = ((up[3] != 0) ? hi_204_1 : lo_205_1);
            V_2[1] = cur_189;
            float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, V_2[6], 15);
            float y1_206 = _shfl_xor_113;
            float _min_589 = fminf(V_2[2], y1_206);
            float lo1_207 = _min_589;
            float _fmax_648 = fmaxf(r_1, lo1_207);
            r_1 = _fmax_648;
            float _fmax_649 = fmaxf(V_2[2], y1_206);
            float hi1_208 = _fmax_649;
            float cur_209 = hi1_208;
            float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, cur_209, 8);
            float pv_210 = _shfl_xor_114;
            float _fmax_650 = fmaxf(cur_209, pv_210);
            float hi_212_1 = _fmax_650;
            float _min_590 = fminf(cur_209, pv_210);
            float lo_213_1 = _min_590;
            cur_209 = ((up[0] != 0) ? hi_212_1 : lo_213_1);
            float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, cur_209, 4);
            float pv_214 = _shfl_xor_115;
            float _fmax_651 = fmaxf(cur_209, pv_214);
            float hi_216_1 = _fmax_651;
            float _min_591 = fminf(cur_209, pv_214);
            float lo_217_1 = _min_591;
            cur_209 = ((up[1] != 0) ? hi_216_1 : lo_217_1);
            float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, cur_209, 2);
            float pv_218 = _shfl_xor_116;
            float _fmax_652 = fmaxf(cur_209, pv_218);
            float hi_220_1 = _fmax_652;
            float _min_592 = fminf(cur_209, pv_218);
            float lo_221_1 = _min_592;
            cur_209 = ((up[2] != 0) ? hi_220_1 : lo_221_1);
            float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, cur_209, 1);
            float pv_222 = _shfl_xor_117;
            float _fmax_653 = fmaxf(cur_209, pv_222);
            float hi_224_1 = _fmax_653;
            float _min_593 = fminf(cur_209, pv_222);
            float lo_225_1 = _min_593;
            cur_209 = ((up[3] != 0) ? hi_224_1 : lo_225_1);
            V_2[2] = cur_209;
            float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, V_2[7], 15);
            float y1_226 = _shfl_xor_118;
            float _min_594 = fminf(V_2[3], y1_226);
            float lo1_227 = _min_594;
            float _fmax_654 = fmaxf(r_1, lo1_227);
            r_1 = _fmax_654;
            float _fmax_655 = fmaxf(V_2[3], y1_226);
            float hi1_228 = _fmax_655;
            float cur_229 = hi1_228;
            float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, cur_229, 8);
            float pv_230 = _shfl_xor_119;
            float _fmax_656 = fmaxf(cur_229, pv_230);
            float hi_232_1 = _fmax_656;
            float _min_595 = fminf(cur_229, pv_230);
            float lo_233_1 = _min_595;
            cur_229 = ((up[0] != 0) ? hi_232_1 : lo_233_1);
            float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, cur_229, 4);
            float pv_234 = _shfl_xor_120;
            float _fmax_657 = fmaxf(cur_229, pv_234);
            float hi_236_1 = _fmax_657;
            float _min_596 = fminf(cur_229, pv_234);
            float lo_237_1 = _min_596;
            cur_229 = ((up[1] != 0) ? hi_236_1 : lo_237_1);
            float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, cur_229, 2);
            float pv_238 = _shfl_xor_121;
            float _fmax_658 = fmaxf(cur_229, pv_238);
            float hi_240_1 = _fmax_658;
            float _min_597 = fminf(cur_229, pv_238);
            float lo_241_1 = _min_597;
            cur_229 = ((up[2] != 0) ? hi_240_1 : lo_241_1);
            float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, cur_229, 1);
            float pv_242 = _shfl_xor_122;
            float _fmax_659 = fmaxf(cur_229, pv_242);
            float hi_244_1 = _fmax_659;
            float _min_598 = fminf(cur_229, pv_242);
            float lo_245_1 = _min_598;
            cur_229 = ((up[3] != 0) ? hi_244_1 : lo_245_1);
            V_2[3] = cur_229;
            float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, V_2[2], 15);
            float y1_246 = _shfl_xor_123;
            float _min_599 = fminf(V_2[0], y1_246);
            float lo1_247 = _min_599;
            float _fmax_660 = fmaxf(r_1, lo1_247);
            r_1 = _fmax_660;
            float _fmax_661 = fmaxf(V_2[0], y1_246);
            float hi1_248 = _fmax_661;
            float cur_249 = hi1_248;
            float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, cur_249, 8);
            float pv_250 = _shfl_xor_124;
            float _fmax_662 = fmaxf(cur_249, pv_250);
            float hi_252_1 = _fmax_662;
            float _min_600 = fminf(cur_249, pv_250);
            float lo_253_1 = _min_600;
            cur_249 = ((up[0] != 0) ? hi_252_1 : lo_253_1);
            float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, cur_249, 4);
            float pv_254 = _shfl_xor_125;
            float _fmax_663 = fmaxf(cur_249, pv_254);
            float hi_256_1 = _fmax_663;
            float _min_601 = fminf(cur_249, pv_254);
            float lo_257_1 = _min_601;
            cur_249 = ((up[1] != 0) ? hi_256_1 : lo_257_1);
            float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, cur_249, 2);
            float pv_258 = _shfl_xor_126;
            float _fmax_664 = fmaxf(cur_249, pv_258);
            float hi_260_1 = _fmax_664;
            float _min_602 = fminf(cur_249, pv_258);
            float lo_261_1 = _min_602;
            cur_249 = ((up[2] != 0) ? hi_260_1 : lo_261_1);
            float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, cur_249, 1);
            float pv_262 = _shfl_xor_127;
            float _fmax_665 = fmaxf(cur_249, pv_262);
            float hi_264_1 = _fmax_665;
            float _min_603 = fminf(cur_249, pv_262);
            float lo_265_1 = _min_603;
            cur_249 = ((up[3] != 0) ? hi_264_1 : lo_265_1);
            V_2[0] = cur_249;
            float _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, V_2[3], 15);
            float y1_266 = _shfl_xor_128;
            float _min_604 = fminf(V_2[1], y1_266);
            float lo1_267 = _min_604;
            float _fmax_666 = fmaxf(r_1, lo1_267);
            r_1 = _fmax_666;
            float _fmax_667 = fmaxf(V_2[1], y1_266);
            float hi1_268 = _fmax_667;
            float cur_269 = hi1_268;
            float _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, cur_269, 8);
            float pv_270 = _shfl_xor_129;
            float _fmax_668 = fmaxf(cur_269, pv_270);
            float hi_272_1 = _fmax_668;
            float _min_605 = fminf(cur_269, pv_270);
            float lo_273_1 = _min_605;
            cur_269 = ((up[0] != 0) ? hi_272_1 : lo_273_1);
            float _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, cur_269, 4);
            float pv_274 = _shfl_xor_130;
            float _fmax_669 = fmaxf(cur_269, pv_274);
            float hi_276_1 = _fmax_669;
            float _min_606 = fminf(cur_269, pv_274);
            float lo_277_1 = _min_606;
            cur_269 = ((up[1] != 0) ? hi_276_1 : lo_277_1);
            float _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, cur_269, 2);
            float pv_278 = _shfl_xor_131;
            float _fmax_670 = fmaxf(cur_269, pv_278);
            float hi_280_1 = _fmax_670;
            float _min_607 = fminf(cur_269, pv_278);
            float lo_281_1 = _min_607;
            cur_269 = ((up[2] != 0) ? hi_280_1 : lo_281_1);
            float _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, cur_269, 1);
            float pv_282 = _shfl_xor_132;
            float _fmax_671 = fmaxf(cur_269, pv_282);
            float hi_284_1 = _fmax_671;
            float _min_608 = fminf(cur_269, pv_282);
            float lo_285_1 = _min_608;
            cur_269 = ((up[3] != 0) ? hi_284_1 : lo_285_1);
            V_2[1] = cur_269;
            float _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, V_2[1], 15);
            float yl_286 = _shfl_xor_133;
            float _min_609 = fminf(V_2[0], yl_286);
            float lol_287 = _min_609;
            float _fmax_672 = fmaxf(r_1, lol_287);
            r_1 = _fmax_672;
            float _fmax_673 = fmaxf(V_2[0], yl_286);
            float hil_288 = _fmax_673;
            V_2[0] = hil_288;
            float K_289 = V_2[0];
            rr2x[0] = r_1;
            K2 = K_289;
        }
        if (tid_1 < 256) {
            unsigned int cgx = ccnt[cg];
            unsigned int k1x = __as_u32(K);
            int isq = 0;
            if (gflag != 0 && (k1x & 4294966784u) == qg) {
                isq = 1;
            }
            unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, isq != 0);
            unsigned int mqx = _vote_0;
            unsigned int gmx = mqx & 65535;
            if ((tid_1 & 16) != 0) {
                gmx = mqx >> 16 & 65535;
            }
            unsigned int lowx = (unsigned int)((1 << ln) - 1);
            int _popc_1 = __popc(gmx & lowx);
            int rx = _popc_1;
            if (gflag != 0 && cgx <= 16) {
                K2 = __uint_as_float(1073741824 | k1x & 511);
                if (isq != 0) {
                    if (cgx <= 4) {
                        float x4[4];
                        x4[0] = 0.0f;
                        if (cgx > 0) {
                            x4[0] = __uint_as_float(cbuf[cg * 16]);
                        }
                        x4[1] = 0.0f;
                        if (cgx > 1) {
                            x4[1] = __uint_as_float(cbuf[cg * 16 + 1]);
                        }
                        x4[2] = 0.0f;
                        if (cgx > 2) {
                            x4[2] = __uint_as_float(cbuf[cg * 16 + 2]);
                        }
                        x4[3] = 0.0f;
                        if (cgx > 3) {
                            x4[3] = __uint_as_float(cbuf[cg * 16 + 3]);
                        }
                        int rk4 = 0;
                        if (x4[1] > x4[0]) {
                            rk4 = rk4 + 1;
                        }
                        if (x4[2] > x4[0]) {
                            rk4 = rk4 + 1;
                        }
                        if (x4[3] > x4[0]) {
                            rk4 = rk4 + 1;
                        }
                        if (rk4 == rx) {
                            K2 = x4[0];
                        }
                        int rk4_0 = 0;
                        if (x4[0] > x4[1]) {
                            rk4_0 = rk4_0 + 1;
                        }
                        if (x4[2] > x4[1]) {
                            rk4_0 = rk4_0 + 1;
                        }
                        if (x4[3] > x4[1]) {
                            rk4_0 = rk4_0 + 1;
                        }
                        if (rk4_0 == rx) {
                            K2 = x4[1];
                        }
                        int rk4_1 = 0;
                        if (x4[0] > x4[2]) {
                            rk4_1 = rk4_1 + 1;
                        }
                        if (x4[1] > x4[2]) {
                            rk4_1 = rk4_1 + 1;
                        }
                        if (x4[3] > x4[2]) {
                            rk4_1 = rk4_1 + 1;
                        }
                        if (rk4_1 == rx) {
                            K2 = x4[2];
                        }
                        int rk4_2 = 0;
                        if (x4[0] > x4[3]) {
                            rk4_2 = rk4_2 + 1;
                        }
                        if (x4[1] > x4[3]) {
                            rk4_2 = rk4_2 + 1;
                        }
                        if (x4[2] > x4[3]) {
                            rk4_2 = rk4_2 + 1;
                        }
                        if (rk4_2 == rx) {
                            K2 = x4[3];
                        }
                    } else {
                        float xsl[16];
                        xsl[0] = 0.0f;
                        if (cgx > 0) {
                            xsl[0] = __uint_as_float(cbuf[cg * 16]);
                        }
                        xsl[1] = 0.0f;
                        if (cgx > 1) {
                            xsl[1] = __uint_as_float(cbuf[cg * 16 + 1]);
                        }
                        xsl[2] = 0.0f;
                        if (cgx > 2) {
                            xsl[2] = __uint_as_float(cbuf[cg * 16 + 2]);
                        }
                        xsl[3] = 0.0f;
                        if (cgx > 3) {
                            xsl[3] = __uint_as_float(cbuf[cg * 16 + 3]);
                        }
                        xsl[4] = 0.0f;
                        if (cgx > 4) {
                            xsl[4] = __uint_as_float(cbuf[cg * 16 + 4]);
                        }
                        xsl[5] = 0.0f;
                        if (cgx > 5) {
                            xsl[5] = __uint_as_float(cbuf[cg * 16 + 5]);
                        }
                        xsl[6] = 0.0f;
                        if (cgx > 6) {
                            xsl[6] = __uint_as_float(cbuf[cg * 16 + 6]);
                        }
                        xsl[7] = 0.0f;
                        if (cgx > 7) {
                            xsl[7] = __uint_as_float(cbuf[cg * 16 + 7]);
                        }
                        xsl[8] = 0.0f;
                        if (cgx > 8) {
                            xsl[8] = __uint_as_float(cbuf[cg * 16 + 8]);
                        }
                        xsl[9] = 0.0f;
                        if (cgx > 9) {
                            xsl[9] = __uint_as_float(cbuf[cg * 16 + 9]);
                        }
                        xsl[10] = 0.0f;
                        if (cgx > 10) {
                            xsl[10] = __uint_as_float(cbuf[cg * 16 + 10]);
                        }
                        xsl[11] = 0.0f;
                        if (cgx > 11) {
                            xsl[11] = __uint_as_float(cbuf[cg * 16 + 11]);
                        }
                        xsl[12] = 0.0f;
                        if (cgx > 12) {
                            xsl[12] = __uint_as_float(cbuf[cg * 16 + 12]);
                        }
                        xsl[13] = 0.0f;
                        if (cgx > 13) {
                            xsl[13] = __uint_as_float(cbuf[cg * 16 + 13]);
                        }
                        xsl[14] = 0.0f;
                        if (cgx > 14) {
                            xsl[14] = __uint_as_float(cbuf[cg * 16 + 14]);
                        }
                        xsl[15] = 0.0f;
                        if (cgx > 15) {
                            xsl[15] = __uint_as_float(cbuf[cg * 16 + 15]);
                        }
                        float _fmax_674 = fmaxf(xsl[0], xsl[13]);
                        float hi_0 = _fmax_674;
                        float _min_610 = fminf(xsl[0], xsl[13]);
                        float lo_1 = _min_610;
                        xsl[0] = hi_0;
                        xsl[13] = lo_1;
                        float _fmax_675 = fmaxf(xsl[1], xsl[12]);
                        float hi_2 = _fmax_675;
                        float _min_611 = fminf(xsl[1], xsl[12]);
                        float lo_3 = _min_611;
                        xsl[1] = hi_2;
                        xsl[12] = lo_3;
                        float _fmax_676 = fmaxf(xsl[2], xsl[15]);
                        float hi_4 = _fmax_676;
                        float _min_612 = fminf(xsl[2], xsl[15]);
                        float lo_5 = _min_612;
                        xsl[2] = hi_4;
                        xsl[15] = lo_5;
                        float _fmax_677 = fmaxf(xsl[3], xsl[14]);
                        float hi_6 = _fmax_677;
                        float _min_613 = fminf(xsl[3], xsl[14]);
                        float lo_7 = _min_613;
                        xsl[3] = hi_6;
                        xsl[14] = lo_7;
                        float _fmax_678 = fmaxf(xsl[4], xsl[8]);
                        float hi_8 = _fmax_678;
                        float _min_614 = fminf(xsl[4], xsl[8]);
                        float lo_9 = _min_614;
                        xsl[4] = hi_8;
                        xsl[8] = lo_9;
                        float _fmax_679 = fmaxf(xsl[5], xsl[6]);
                        float hi_10 = _fmax_679;
                        float _min_615 = fminf(xsl[5], xsl[6]);
                        float lo_11 = _min_615;
                        xsl[5] = hi_10;
                        xsl[6] = lo_11;
                        float _fmax_680 = fmaxf(xsl[7], xsl[11]);
                        float hi_12 = _fmax_680;
                        float _min_616 = fminf(xsl[7], xsl[11]);
                        float lo_13 = _min_616;
                        xsl[7] = hi_12;
                        xsl[11] = lo_13;
                        float _fmax_681 = fmaxf(xsl[9], xsl[10]);
                        float hi_14_1 = _fmax_681;
                        float _min_617 = fminf(xsl[9], xsl[10]);
                        float lo_15_1 = _min_617;
                        xsl[9] = hi_14_1;
                        xsl[10] = lo_15_1;
                        float _fmax_682 = fmaxf(xsl[0], xsl[5]);
                        float hi_16 = _fmax_682;
                        float _min_618 = fminf(xsl[0], xsl[5]);
                        float lo_17 = _min_618;
                        xsl[0] = hi_16;
                        xsl[5] = lo_17;
                        float _fmax_683 = fmaxf(xsl[1], xsl[7]);
                        float hi_18 = _fmax_683;
                        float _min_619 = fminf(xsl[1], xsl[7]);
                        float lo_19 = _min_619;
                        xsl[1] = hi_18;
                        xsl[7] = lo_19;
                        float _fmax_684 = fmaxf(xsl[2], xsl[9]);
                        float hi_20_1 = _fmax_684;
                        float _min_620 = fminf(xsl[2], xsl[9]);
                        float lo_21_1 = _min_620;
                        xsl[2] = hi_20_1;
                        xsl[9] = lo_21_1;
                        float _fmax_685 = fmaxf(xsl[3], xsl[4]);
                        float hi_22 = _fmax_685;
                        float _min_621 = fminf(xsl[3], xsl[4]);
                        float lo_23 = _min_621;
                        xsl[3] = hi_22;
                        xsl[4] = lo_23;
                        float _fmax_686 = fmaxf(xsl[6], xsl[13]);
                        float hi_24 = _fmax_686;
                        float _min_622 = fminf(xsl[6], xsl[13]);
                        float lo_25 = _min_622;
                        xsl[6] = hi_24;
                        xsl[13] = lo_25;
                        float _fmax_687 = fmaxf(xsl[8], xsl[14]);
                        float hi_26 = _fmax_687;
                        float _min_623 = fminf(xsl[8], xsl[14]);
                        float lo_27 = _min_623;
                        xsl[8] = hi_26;
                        xsl[14] = lo_27;
                        float _fmax_688 = fmaxf(xsl[10], xsl[15]);
                        float hi_28 = _fmax_688;
                        float _min_624 = fminf(xsl[10], xsl[15]);
                        float lo_29 = _min_624;
                        xsl[10] = hi_28;
                        xsl[15] = lo_29;
                        float _fmax_689 = fmaxf(xsl[11], xsl[12]);
                        float hi_30_1 = _fmax_689;
                        float _min_625 = fminf(xsl[11], xsl[12]);
                        float lo_31_1 = _min_625;
                        xsl[11] = hi_30_1;
                        xsl[12] = lo_31_1;
                        float _fmax_690 = fmaxf(xsl[0], xsl[1]);
                        float hi_32 = _fmax_690;
                        float _min_626 = fminf(xsl[0], xsl[1]);
                        float lo_33 = _min_626;
                        xsl[0] = hi_32;
                        xsl[1] = lo_33;
                        float _fmax_691 = fmaxf(xsl[2], xsl[3]);
                        float hi_34 = _fmax_691;
                        float _min_627 = fminf(xsl[2], xsl[3]);
                        float lo_35 = _min_627;
                        xsl[2] = hi_34;
                        xsl[3] = lo_35;
                        float _fmax_692 = fmaxf(xsl[4], xsl[5]);
                        float hi_36_1 = _fmax_692;
                        float _min_628 = fminf(xsl[4], xsl[5]);
                        float lo_37_1 = _min_628;
                        xsl[4] = hi_36_1;
                        xsl[5] = lo_37_1;
                        float _fmax_693 = fmaxf(xsl[6], xsl[8]);
                        float hi_38 = _fmax_693;
                        float _min_629 = fminf(xsl[6], xsl[8]);
                        float lo_39 = _min_629;
                        xsl[6] = hi_38;
                        xsl[8] = lo_39;
                        float _fmax_694 = fmaxf(xsl[7], xsl[9]);
                        float hi_40 = _fmax_694;
                        float _min_630 = fminf(xsl[7], xsl[9]);
                        float lo_41 = _min_630;
                        xsl[7] = hi_40;
                        xsl[9] = lo_41;
                        float _fmax_695 = fmaxf(xsl[10], xsl[11]);
                        float hi_42 = _fmax_695;
                        float _min_631 = fminf(xsl[10], xsl[11]);
                        float lo_43 = _min_631;
                        xsl[10] = hi_42;
                        xsl[11] = lo_43;
                        float _fmax_696 = fmaxf(xsl[12], xsl[13]);
                        float hi_44 = _fmax_696;
                        float _min_632 = fminf(xsl[12], xsl[13]);
                        float lo_45 = _min_632;
                        xsl[12] = hi_44;
                        xsl[13] = lo_45;
                        float _fmax_697 = fmaxf(xsl[14], xsl[15]);
                        float hi_46 = _fmax_697;
                        float _min_633 = fminf(xsl[14], xsl[15]);
                        float lo_47 = _min_633;
                        xsl[14] = hi_46;
                        xsl[15] = lo_47;
                        float _fmax_698 = fmaxf(xsl[0], xsl[2]);
                        float hi_48 = _fmax_698;
                        float _min_634 = fminf(xsl[0], xsl[2]);
                        float lo_49 = _min_634;
                        xsl[0] = hi_48;
                        xsl[2] = lo_49;
                        float _fmax_699 = fmaxf(xsl[1], xsl[3]);
                        float hi_50 = _fmax_699;
                        float _min_635 = fminf(xsl[1], xsl[3]);
                        float lo_51 = _min_635;
                        xsl[1] = hi_50;
                        xsl[3] = lo_51;
                        float _fmax_700 = fmaxf(xsl[4], xsl[10]);
                        float hi_52_1 = _fmax_700;
                        float _min_636 = fminf(xsl[4], xsl[10]);
                        float lo_53_1 = _min_636;
                        xsl[4] = hi_52_1;
                        xsl[10] = lo_53_1;
                        float _fmax_701 = fmaxf(xsl[5], xsl[11]);
                        float hi_54 = _fmax_701;
                        float _min_637 = fminf(xsl[5], xsl[11]);
                        float lo_55 = _min_637;
                        xsl[5] = hi_54;
                        xsl[11] = lo_55;
                        float _fmax_702 = fmaxf(xsl[6], xsl[7]);
                        float hi_56 = _fmax_702;
                        float _min_638 = fminf(xsl[6], xsl[7]);
                        float lo_57 = _min_638;
                        xsl[6] = hi_56;
                        xsl[7] = lo_57;
                        float _fmax_703 = fmaxf(xsl[8], xsl[9]);
                        float hi_58_1 = _fmax_703;
                        float _min_639 = fminf(xsl[8], xsl[9]);
                        float lo_59_1 = _min_639;
                        xsl[8] = hi_58_1;
                        xsl[9] = lo_59_1;
                        float _fmax_704 = fmaxf(xsl[12], xsl[14]);
                        float hi_60 = _fmax_704;
                        float _min_640 = fminf(xsl[12], xsl[14]);
                        float lo_61 = _min_640;
                        xsl[12] = hi_60;
                        xsl[14] = lo_61;
                        float _fmax_705 = fmaxf(xsl[13], xsl[15]);
                        float hi_62 = _fmax_705;
                        float _min_641 = fminf(xsl[13], xsl[15]);
                        float lo_63 = _min_641;
                        xsl[13] = hi_62;
                        xsl[15] = lo_63;
                        float _fmax_706 = fmaxf(xsl[1], xsl[2]);
                        float hi_64 = _fmax_706;
                        float _min_642 = fminf(xsl[1], xsl[2]);
                        float lo_65 = _min_642;
                        xsl[1] = hi_64;
                        xsl[2] = lo_65;
                        float _fmax_707 = fmaxf(xsl[3], xsl[12]);
                        float hi_66 = _fmax_707;
                        float _min_643 = fminf(xsl[3], xsl[12]);
                        float lo_67 = _min_643;
                        xsl[3] = hi_66;
                        xsl[12] = lo_67;
                        float _fmax_708 = fmaxf(xsl[4], xsl[6]);
                        float hi_68_1 = _fmax_708;
                        float _min_644 = fminf(xsl[4], xsl[6]);
                        float lo_69_1 = _min_644;
                        xsl[4] = hi_68_1;
                        xsl[6] = lo_69_1;
                        float _fmax_709 = fmaxf(xsl[5], xsl[7]);
                        float hi_70 = _fmax_709;
                        float _min_645 = fminf(xsl[5], xsl[7]);
                        float lo_71 = _min_645;
                        xsl[5] = hi_70;
                        xsl[7] = lo_71;
                        float _fmax_710 = fmaxf(xsl[8], xsl[10]);
                        float hi_72 = _fmax_710;
                        float _min_646 = fminf(xsl[8], xsl[10]);
                        float lo_73 = _min_646;
                        xsl[8] = hi_72;
                        xsl[10] = lo_73;
                        float _fmax_711 = fmaxf(xsl[9], xsl[11]);
                        float hi_74_1 = _fmax_711;
                        float _min_647 = fminf(xsl[9], xsl[11]);
                        float lo_75_1 = _min_647;
                        xsl[9] = hi_74_1;
                        xsl[11] = lo_75_1;
                        float _fmax_712 = fmaxf(xsl[13], xsl[14]);
                        float hi_76 = _fmax_712;
                        float _min_648 = fminf(xsl[13], xsl[14]);
                        float lo_77 = _min_648;
                        xsl[13] = hi_76;
                        xsl[14] = lo_77;
                        float _fmax_713 = fmaxf(xsl[1], xsl[4]);
                        float hi_78 = _fmax_713;
                        float _min_649 = fminf(xsl[1], xsl[4]);
                        float lo_79 = _min_649;
                        xsl[1] = hi_78;
                        xsl[4] = lo_79;
                        float _fmax_714 = fmaxf(xsl[2], xsl[6]);
                        float hi_80 = _fmax_714;
                        float _min_650 = fminf(xsl[2], xsl[6]);
                        float lo_81 = _min_650;
                        xsl[2] = hi_80;
                        xsl[6] = lo_81;
                        float _fmax_715 = fmaxf(xsl[5], xsl[8]);
                        float hi_82 = _fmax_715;
                        float _min_651 = fminf(xsl[5], xsl[8]);
                        float lo_83 = _min_651;
                        xsl[5] = hi_82;
                        xsl[8] = lo_83;
                        float _fmax_716 = fmaxf(xsl[7], xsl[10]);
                        float hi_84 = _fmax_716;
                        float _min_652 = fminf(xsl[7], xsl[10]);
                        float lo_85 = _min_652;
                        xsl[7] = hi_84;
                        xsl[10] = lo_85;
                        float _fmax_717 = fmaxf(xsl[9], xsl[13]);
                        float hi_86 = _fmax_717;
                        float _min_653 = fminf(xsl[9], xsl[13]);
                        float lo_87 = _min_653;
                        xsl[9] = hi_86;
                        xsl[13] = lo_87;
                        float _fmax_718 = fmaxf(xsl[11], xsl[14]);
                        float hi_88 = _fmax_718;
                        float _min_654 = fminf(xsl[11], xsl[14]);
                        float lo_89 = _min_654;
                        xsl[11] = hi_88;
                        xsl[14] = lo_89;
                        float _fmax_719 = fmaxf(xsl[2], xsl[4]);
                        float hi_90_1 = _fmax_719;
                        float _min_655 = fminf(xsl[2], xsl[4]);
                        float lo_91_1 = _min_655;
                        xsl[2] = hi_90_1;
                        xsl[4] = lo_91_1;
                        float _fmax_720 = fmaxf(xsl[3], xsl[6]);
                        float hi_92 = _fmax_720;
                        float _min_656 = fminf(xsl[3], xsl[6]);
                        float lo_93 = _min_656;
                        xsl[3] = hi_92;
                        xsl[6] = lo_93;
                        float _fmax_721 = fmaxf(xsl[9], xsl[12]);
                        float hi_94_1 = _fmax_721;
                        float _min_657 = fminf(xsl[9], xsl[12]);
                        float lo_95_1 = _min_657;
                        xsl[9] = hi_94_1;
                        xsl[12] = lo_95_1;
                        float _fmax_722 = fmaxf(xsl[11], xsl[13]);
                        float hi_96 = _fmax_722;
                        float _min_658 = fminf(xsl[11], xsl[13]);
                        float lo_97 = _min_658;
                        xsl[11] = hi_96;
                        xsl[13] = lo_97;
                        float _fmax_723 = fmaxf(xsl[3], xsl[5]);
                        float hi_98_1 = _fmax_723;
                        float _min_659 = fminf(xsl[3], xsl[5]);
                        float lo_99_1 = _min_659;
                        xsl[3] = hi_98_1;
                        xsl[5] = lo_99_1;
                        float _fmax_724 = fmaxf(xsl[6], xsl[8]);
                        float hi_100 = _fmax_724;
                        float _min_660 = fminf(xsl[6], xsl[8]);
                        float lo_101 = _min_660;
                        xsl[6] = hi_100;
                        xsl[8] = lo_101;
                        float _fmax_725 = fmaxf(xsl[7], xsl[9]);
                        float hi_102 = _fmax_725;
                        float _min_661 = fminf(xsl[7], xsl[9]);
                        float lo_103 = _min_661;
                        xsl[7] = hi_102;
                        xsl[9] = lo_103;
                        float _fmax_726 = fmaxf(xsl[10], xsl[12]);
                        float hi_104 = _fmax_726;
                        float _min_662 = fminf(xsl[10], xsl[12]);
                        float lo_105 = _min_662;
                        xsl[10] = hi_104;
                        xsl[12] = lo_105;
                        float _fmax_727 = fmaxf(xsl[3], xsl[4]);
                        float hi_106 = _fmax_727;
                        float _min_663 = fminf(xsl[3], xsl[4]);
                        float lo_107 = _min_663;
                        xsl[3] = hi_106;
                        xsl[4] = lo_107;
                        float _fmax_728 = fmaxf(xsl[5], xsl[6]);
                        float hi_108_1 = _fmax_728;
                        float _min_664 = fminf(xsl[5], xsl[6]);
                        float lo_109_1 = _min_664;
                        xsl[5] = hi_108_1;
                        xsl[6] = lo_109_1;
                        float _fmax_729 = fmaxf(xsl[7], xsl[8]);
                        float hi_110 = _fmax_729;
                        float _min_665 = fminf(xsl[7], xsl[8]);
                        float lo_111 = _min_665;
                        xsl[7] = hi_110;
                        xsl[8] = lo_111;
                        float _fmax_730 = fmaxf(xsl[9], xsl[10]);
                        float hi_112_1 = _fmax_730;
                        float _min_666 = fminf(xsl[9], xsl[10]);
                        float lo_113_1 = _min_666;
                        xsl[9] = hi_112_1;
                        xsl[10] = lo_113_1;
                        float _fmax_731 = fmaxf(xsl[11], xsl[12]);
                        float hi_114 = _fmax_731;
                        float _min_667 = fminf(xsl[11], xsl[12]);
                        float lo_115 = _min_667;
                        xsl[11] = hi_114;
                        xsl[12] = lo_115;
                        float _fmax_732 = fmaxf(xsl[6], xsl[7]);
                        float hi_116_1 = _fmax_732;
                        float _min_668 = fminf(xsl[6], xsl[7]);
                        float lo_117_1 = _min_668;
                        xsl[6] = hi_116_1;
                        xsl[7] = lo_117_1;
                        float _fmax_733 = fmaxf(xsl[8], xsl[9]);
                        float hi_118 = _fmax_733;
                        float _min_669 = fminf(xsl[8], xsl[9]);
                        float lo_119 = _min_669;
                        xsl[8] = hi_118;
                        xsl[9] = lo_119;
                        K2 = xsl[0];
                        if (rx == 1) {
                            K2 = xsl[1];
                        }
                        if (rx == 2) {
                            K2 = xsl[2];
                        }
                        if (rx == 3) {
                            K2 = xsl[3];
                        }
                        if (rx == 4) {
                            K2 = xsl[4];
                        }
                        if (rx == 5) {
                            K2 = xsl[5];
                        }
                        if (rx == 6) {
                            K2 = xsl[6];
                        }
                        if (rx == 7) {
                            K2 = xsl[7];
                        }
                        if (rx == 8) {
                            K2 = xsl[8];
                        }
                        if (rx == 9) {
                            K2 = xsl[9];
                        }
                        if (rx == 10) {
                            K2 = xsl[10];
                        }
                        if (rx == 11) {
                            K2 = xsl[11];
                        }
                        if (rx == 12) {
                            K2 = xsl[12];
                        }
                        if (rx == 13) {
                            K2 = xsl[13];
                        }
                        if (rx == 14) {
                            K2 = xsl[14];
                        }
                        if (rx == 15) {
                            K2 = xsl[15];
                        }
                    }
                }
            }
        }
    }
    if (tid_1 < 256) {
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
        unsigned int _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, rkey, 1);
        unsigned int ox = _shfl_xor_134;
        if (ox < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, rkey, 2);
        unsigned int ox_0 = _shfl_xor_135;
        if (ox_0 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_136 = __shfl_xor_sync(0xFFFFFFFF, rkey, 3);
        unsigned int ox_1 = _shfl_xor_136;
        if (ox_1 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_137 = __shfl_xor_sync(0xFFFFFFFF, rkey, 4);
        unsigned int ox_2 = _shfl_xor_137;
        if (ox_2 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_138 = __shfl_xor_sync(0xFFFFFFFF, rkey, 5);
        unsigned int ox_3 = _shfl_xor_138;
        if (ox_3 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_139 = __shfl_xor_sync(0xFFFFFFFF, rkey, 6);
        unsigned int ox_4 = _shfl_xor_139;
        if (ox_4 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_140 = __shfl_xor_sync(0xFFFFFFFF, rkey, 7);
        unsigned int ox_5 = _shfl_xor_140;
        if (ox_5 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_141 = __shfl_xor_sync(0xFFFFFFFF, rkey, 8);
        unsigned int ox_6 = _shfl_xor_141;
        if (ox_6 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_142 = __shfl_xor_sync(0xFFFFFFFF, rkey, 9);
        unsigned int ox_7 = _shfl_xor_142;
        if (ox_7 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_143 = __shfl_xor_sync(0xFFFFFFFF, rkey, 10);
        unsigned int ox_8 = _shfl_xor_143;
        if (ox_8 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_144 = __shfl_xor_sync(0xFFFFFFFF, rkey, 11);
        unsigned int ox_9 = _shfl_xor_144;
        if (ox_9 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_145 = __shfl_xor_sync(0xFFFFFFFF, rkey, 12);
        unsigned int ox_10 = _shfl_xor_145;
        if (ox_10 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_146 = __shfl_xor_sync(0xFFFFFFFF, rkey, 13);
        unsigned int ox_11 = _shfl_xor_146;
        if (ox_11 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_147 = __shfl_xor_sync(0xFFFFFFFF, rkey, 14);
        unsigned int ox_12 = _shfl_xor_147;
        if (ox_12 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_148 = __shfl_xor_sync(0xFFFFFFFF, rkey, 15);
        unsigned int ox_13 = _shfl_xor_148;
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
